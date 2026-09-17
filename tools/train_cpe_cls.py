# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import logging
import os
import pprint
import time
import random
from math import cos, pi
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data._utils.collate import default_collate

from utils.fire import fire_reinit_cpe, reset_prototype_memory
from utils.ad_after_cpe import run_ad_after_cpe_task_if_needed


import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

import matplotlib

try:
    import tabulate
except ImportError:
    tabulate = None

from datasets.data_builder import build_dataloader
from utils.misc_helper import create_logger, get_current_time, set_random_seed


parser = argparse.ArgumentParser(
    description="UCA-3DAL style CPE training with epoch validation"
)
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--ckpt", default="", help="checkpoint path for --evaluate")


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss used by UCA-3DAL CPE stage.

    Args:
        features: [B, V, D]
        labels:   [B]
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise RuntimeError(
                f"SupConLoss expects features [B,V,D], got {tuple(features.shape)}"
            )

        device = features.device
        batch_size, view_num, dim = features.shape

        feats = F.normalize(features.reshape(batch_size * view_num, dim), dim=1)
        labels = labels.view(-1, 1).long()

        if labels.shape[0] != batch_size:
            raise RuntimeError(
                f"labels batch size {labels.shape[0]} != features batch size {batch_size}"
            )

        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat_interleave(view_num, dim=0).repeat_interleave(view_num, dim=1)

        logits = torch.matmul(feats, feats.T) / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(
            batch_size * view_num,
            device=device,
            dtype=mask.dtype,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1.0e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1.0e-12)
        loss = -mean_log_prob_pos
        loss = loss.view(batch_size, view_num).mean()

        return loss

class AdaptiveMultiPrototypeMemory(nn.Module):
    """
    Adaptive multi-prototype memory.

    Keeps legacy fields:
        proto:  [C, D]
        counts: [C]

    Uses multi-prototype fields:
        proto_multi:  [C, K, D]
        counts_multi: [C, K]
        active_multi: [C, K]
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        num_prototypes: int = 4,
        momentum: float = 0.9,
        adaptive: bool = True,
        new_proto_threshold: float = 0.55,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.num_prototypes = int(num_prototypes)
        self.momentum = float(momentum)
        self.adaptive = bool(adaptive)
        self.new_proto_threshold = float(new_proto_threshold)

        if self.num_prototypes <= 0:
            raise RuntimeError(
                f"num_prototypes must be positive, got {self.num_prototypes}"
            )

        # Legacy class-level prototype for compatibility.
        self.register_buffer("proto", torch.zeros(self.num_classes, self.dim))
        self.register_buffer("counts", torch.zeros(self.num_classes))

        # Real multi-prototype memory.
        self.register_buffer(
            "proto_multi",
            torch.zeros(self.num_classes, self.num_prototypes, self.dim),
        )
        self.register_buffer(
            "counts_multi",
            torch.zeros(self.num_classes, self.num_prototypes),
        )
        self.register_buffer(
            "active_multi",
            torch.zeros(self.num_classes, self.num_prototypes, dtype=torch.bool),
        )

    @torch.no_grad()
    def reset(self) -> None:
        self.proto.zero_()
        self.counts.zero_()
        self.proto_multi.zero_()
        self.counts_multi.zero_()
        self.active_multi.zero_()

    def active_mask(self) -> torch.Tensor:
        return self.active_multi & (self.counts_multi > 0)

    def class_counts(self) -> torch.Tensor:
        return self.counts_multi.sum(dim=1)

    @torch.no_grad()
    def refresh_class_proto(self) -> None:
        """
        Build legacy [C,D] class prototype from active multi-prototypes.
        """
        weights = self.counts_multi.clamp_min(0.0)
        total = weights.sum(dim=1)

        self.proto.zero_()
        self.counts.copy_(total)

        valid = total > 0
        if not bool(valid.any()):
            return

        weighted = (self.proto_multi * weights.unsqueeze(-1)).sum(dim=1)
        self.proto[valid] = F.normalize(weighted[valid], dim=1)

    @torch.no_grad()
    def bootstrap_from_legacy_proto(self) -> None:
        """
        Useful when loading an old checkpoint that only had proto/counts.
        """
        valid = self.counts > 0
        if not bool(valid.any()):
            return

        self.proto_multi[valid, 0] = F.normalize(self.proto[valid], dim=1)
        self.counts_multi[valid, 0] = self.counts[valid]
        self.active_multi[valid, 0] = True

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """
        Online EMA update.

        If a class sample is far from all currently active prototypes,
        allocate a new prototype slot until K_max is reached.
        """
        if z.numel() == 0:
            return

        z = F.normalize(z.detach(), dim=1)
        y = y.detach().view(-1).long()

        if z.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"z batch size {z.shape[0]} != y batch size {y.shape[0]}"
            )

        for cls_id_tensor in y.unique():
            cid = int(cls_id_tensor.item())

            if cid < 0 or cid >= self.num_classes:
                raise RuntimeError(
                    f"Class id {cid} out of range [0, {self.num_classes - 1}]"
                )

            z_cls = z[y == cid]

            for zz in z_cls:
                active = self.active_mask()[cid]

                if not bool(active.any()):
                    slot = 0
                else:
                    active_idx = torch.nonzero(active, as_tuple=False).view(-1)
                    cur_proto = F.normalize(
                        self.proto_multi[cid, active_idx],
                        dim=1,
                    )
                    sim = torch.matmul(cur_proto, zz.view(-1, 1)).view(-1)

                    best_rel = int(torch.argmax(sim).item())
                    best_slot = int(active_idx[best_rel].item())
                    best_sim = float(sim[best_rel].item())

                    has_free_slot = int(active_idx.numel()) < self.num_prototypes

                    if (
                        self.adaptive
                        and has_free_slot
                        and best_sim < self.new_proto_threshold
                    ):
                        inactive_idx = torch.nonzero(
                            ~active,
                            as_tuple=False,
                        ).view(-1)
                        slot = int(inactive_idx[0].item())
                    else:
                        slot = best_slot

                if (
                    not bool(self.active_multi[cid, slot].item())
                    or float(self.counts_multi[cid, slot].item()) <= 0.0
                ):
                    self.proto_multi[cid, slot] = zz
                    self.active_multi[cid, slot] = True
                else:
                    old = self.proto_multi[cid, slot]
                    new = F.normalize(
                        self.momentum * old + (1.0 - self.momentum) * zz,
                        dim=0,
                    )
                    self.proto_multi[cid, slot] = new

                self.counts_multi[cid, slot] += 1.0

        self.refresh_class_proto()

    def class_scores(
        self,
        z: torch.Tensor,
        pool_tau: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            class_scores: [B, C], unscaled cosine-like score
            sim:          [B, C, K]
        """
        z = F.normalize(z, dim=1)
        proto = F.normalize(self.proto_multi, dim=-1)

        sim = torch.einsum("bd,ckd->bck", z, proto)

        active = self.active_mask().to(device=sim.device)
        valid_class = active.any(dim=1)

        if not bool(valid_class.any()):
            raise RuntimeError(
                "No valid prototype found. "
                "Maybe prototype memory was not updated or rebuilt."
            )

        sim = sim.masked_fill(~active.view(1, self.num_classes, self.num_prototypes), -1.0e4)

        if float(pool_tau) <= 0.0:
            class_scores = sim.max(dim=2).values
        else:
            class_scores = float(pool_tau) * torch.logsumexp(
                sim / float(pool_tau),
                dim=2,
            )

        class_scores = class_scores.masked_fill(
            ~valid_class.view(1, self.num_classes),
            -1.0e4,
        )

        return class_scores, sim

    def diversity_regularization(
        self,
        stiefel_cos: float = 0.35,
        inter_cos: float = 0.20,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Soft-Stiefel + inter-class separation.

        stiefel_cos:
            Same-class prototypes are penalized only when |cos| > stiefel_cos.

        inter_cos:
            Different-class prototypes are penalized when cos > inter_cos.
        """
        device = self.proto_multi.device
        dtype = self.proto_multi.dtype

        zero = self.proto_multi.new_tensor(0.0)
        active = self.active_mask()

        intra_terms = []

        for cid in range(self.num_classes):
            idx = active[cid]
            k_eff = int(idx.sum().item())

            if k_eff <= 1:
                continue

            p = F.normalize(self.proto_multi[cid, idx], dim=1)
            g = torch.matmul(p, p.T)

            eye = torch.eye(k_eff, device=device, dtype=dtype)
            off = g - eye

            # Relaxed Stiefel: do not force exact orthogonality.
            penalty = F.relu(off.abs() - float(stiefel_cos)).pow(2)

            denom = max(k_eff * (k_eff - 1), 1)
            intra_terms.append(penalty.sum() / float(denom))

        if len(intra_terms) > 0:
            intra_loss = torch.stack(intra_terms).mean()
        else:
            intra_loss = zero

        flat_proto = []
        flat_label = []

        for cid in range(self.num_classes):
            idx = active[cid]
            if not bool(idx.any()):
                continue

            p = F.normalize(self.proto_multi[cid, idx], dim=1)
            flat_proto.append(p)
            flat_label.append(
                torch.full(
                    (p.shape[0],),
                    cid,
                    device=device,
                    dtype=torch.long,
                )
            )

        if len(flat_proto) <= 1:
            inter_loss = zero
        else:
            p_all = torch.cat(flat_proto, dim=0)
            y_all = torch.cat(flat_label, dim=0)

            if p_all.shape[0] <= 1:
                inter_loss = zero
            else:
                g = torch.matmul(p_all, p_all.T)

                eye = torch.eye(
                    p_all.shape[0],
                    device=device,
                    dtype=torch.bool,
                )
                diff_class = y_all.view(-1, 1) != y_all.view(1, -1)
                mask = diff_class & (~eye)

                if bool(mask.any()):
                    inter_loss = F.relu(g[mask] - float(inter_cos)).pow(2).mean()
                else:
                    inter_loss = zero

        return intra_loss + inter_loss, {
            "intra_stiefel": intra_loss.detach(),
            "inter_margin": inter_loss.detach(),
        }


class PrototypeMemory(nn.Module):
    """EMA-updated class prototypes."""

    def __init__(self, num_classes: int, dim: int, momentum: float = 0.9) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.momentum = float(momentum)

        self.register_buffer("proto", torch.zeros(self.num_classes, self.dim))
        self.register_buffer("counts", torch.zeros(self.num_classes))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        if z.numel() == 0:
            return

        z = F.normalize(z, dim=1)
        y = y.view(-1).long()

        for cls_id in y.unique():
            cid = int(cls_id.item())
            if cid < 0 or cid >= self.num_classes:
                raise RuntimeError(
                    f"Class id {cid} out of range [0, {self.num_classes - 1}]"
                )

            idx = y == cid
            if int(idx.sum().item()) == 0:
                continue

            z_mean = F.normalize(z[idx].mean(dim=0, keepdim=True), dim=1)
            old = self.proto[cid : cid + 1]

            if float(self.counts[cid].item()) == 0.0:
                new = z_mean
            else:
                new = F.normalize(
                    self.momentum * old + (1.0 - self.momentum) * z_mean,
                    dim=1,
                )

            self.proto[cid : cid + 1] = new
            self.counts[cid] += idx.sum().to(self.counts.dtype)


class MultiPrototypeNCELoss(nn.Module):
    """
    Multi-prototype classification loss.

    Components:
      1. CE with additive cosine margin.
      2. Sample-level max-margin hinge.
      3. Prototype diversity / separation regularization.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        pool_tau: float = 0.02,
        logit_margin: float = 0.10,
        sample_margin: float = 0.15,
        hinge_weight: float = 0.30,
        diversity_weight: float = 0.02,
        stiefel_cos: float = 0.35,
        inter_cos: float = 0.20,
    ) -> None:
        super().__init__()

        self.temperature = float(temperature)
        self.pool_tau = float(pool_tau)
        self.logit_margin = float(logit_margin)
        self.sample_margin = float(sample_margin)
        self.hinge_weight = float(hinge_weight)
        self.diversity_weight = float(diversity_weight)
        self.stiefel_cos = float(stiefel_cos)
        self.inter_cos = float(inter_cos)

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        proto_mod: AdaptiveMultiPrototypeMemory,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = F.normalize(z, dim=1)
        y = y.view(-1).long()

        class_scores, sim = proto_mod.class_scores(
            z,
            pool_tau=self.pool_tau,
        )

        batch_idx = torch.arange(y.shape[0], device=z.device)

        # Additive cosine margin on ground-truth class.
        # This forces the correct class score to be larger by margin.
        logits_score = class_scores.clone()

        if self.logit_margin > 0.0:
            logits_score[batch_idx, y] -= self.logit_margin

        logits = logits_score / self.temperature
        ce_loss = F.cross_entropy(logits, y)

        # Sample-level max-margin:
        # max positive prototype should beat hardest negative prototype.
        pos = sim[batch_idx, y, :].max(dim=1).values

        neg_sim = sim.clone()
        neg_sim[batch_idx, y, :] = -1.0e4
        neg = neg_sim.amax(dim=(1, 2))

        if self.sample_margin > 0.0 and self.hinge_weight > 0.0:
            hinge_loss = F.relu(self.sample_margin + neg - pos).mean()
        else:
            hinge_loss = z.new_tensor(0.0)

        diversity_loss, diversity_info = proto_mod.diversity_regularization(
            stiefel_cos=self.stiefel_cos,
            inter_cos=self.inter_cos,
        )

        loss = (
            ce_loss
            + self.hinge_weight * hinge_loss
            + self.diversity_weight * diversity_loss
        )

        info = {
            "proto_ce": ce_loss.detach(),
            "proto_hinge": hinge_loss.detach(),
            "proto_diversity": diversity_loss.detach(),
            "proto_intra_stiefel": diversity_info["intra_stiefel"],
            "proto_inter_margin": diversity_info["inter_margin"],
        }

        return loss, info



def cosine_lr_after_step(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch: int,
    step_epoch: int,
    total_epochs: int,
    clip: float = 1.0e-6,
) -> float:
    if epoch < step_epoch:
        lr = float(base_lr)
    else:
        lr = clip + 0.5 * (float(base_lr) - clip) * (
            1.0 + cos(pi * ((epoch - step_epoch) / max(total_epochs - step_epoch, 1)))
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]

POINT_KEY_CANDIDATES = [
    "pointcloud",
    "points",
    "xyz",
    "coord",
    "coords",
    "point_cloud",
    "pc",
]

def _resolve_loader_indices(
    cfg: EasyDict,
    key: str,
    num_loaders: int,
    default_all: bool = True,
) -> set:
    """
    Resolve loader indices from config.

    If key is missing:
      - default_all=True  -> use all loaders
      - default_all=False -> use no loaders

    If key exists and is []:
      - use no loaders
    """
    if key not in cfg:
        return set(range(num_loaders)) if default_all else set()

    value = cfg.get(key)

    if value is None:
        return set(range(num_loaders)) if default_all else set()

    if isinstance(value, str) and value.lower() == "all":
        return set(range(num_loaders))

    indices = set(int(x) for x in _as_list(value))

    for idx in indices:
        if idx < 0 or idx >= num_loaders:
            raise RuntimeError(
                f"Invalid {key}: index {idx} out of range [0, {num_loaders - 1}]"
            )

    return indices

def _get_first_dim(value: Any) -> Optional[int]:
    """
    Return value.shape[0] if value looks like a tensor / ndarray with at least one dim.
    Otherwise return None.
    """
    if torch.is_tensor(value):
        if value.dim() <= 0:
            return None
        return int(value.shape[0])

    if hasattr(value, "shape"):
        try:
            if len(value.shape) <= 0:
                return None
            return int(value.shape[0])
        except Exception:
            return None

    return None


def _as_tensor_for_collate(value: Any) -> torch.Tensor:
    """
    Convert tensor-like data to torch.Tensor.
    """
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _find_point_key(sample: Dict[str, Any], preferred_key: str = "pointcloud") -> Optional[str]:
    """
    Find the point cloud key in one dataset sample.
    """
    if preferred_key in sample and _get_first_dim(sample[preferred_key]) is not None:
        return preferred_key

    for key in POINT_KEY_CANDIDATES:
        if key in sample and _get_first_dim(sample[key]) is not None:
            return key

    return None


def _make_repeat_pad_index(
    src_num_points: int,
    target_num_points: int,
    shuffle_index: bool = True,
) -> torch.Tensor:
    """
    Build indices for repeat-padding or downsampling.

    If src_num_points < target_num_points:
        keep all original points and repeat-sample extra points.

    If src_num_points == target_num_points:
        keep all points.

    If src_num_points > target_num_points:
        randomly sample target_num_points points.
        This normally should not happen if target_num_points is global max,
        but this branch makes the code robust.
    """
    src_num_points = int(src_num_points)
    target_num_points = int(target_num_points)

    if src_num_points <= 0:
        raise RuntimeError(
            "Found empty point cloud with 0 points. "
            "Cannot repeat-sample padding from an empty point cloud."
        )

    if target_num_points <= 0:
        raise RuntimeError(f"Invalid target_num_points={target_num_points}")

    if src_num_points > target_num_points:
        index = torch.randperm(src_num_points)[:target_num_points]
        return index.long()

    base_index = torch.arange(src_num_points, dtype=torch.long)

    if src_num_points < target_num_points:
        extra_num = target_num_points - src_num_points
        extra_index = torch.randint(
            low=0,
            high=src_num_points,
            size=(extra_num,),
            dtype=torch.long,
        )
        index = torch.cat([base_index, extra_index], dim=0)
    else:
        index = base_index

    if shuffle_index and index.numel() > 1:
        perm = torch.randperm(index.numel())
        index = index[perm]

    return index.long()


class RepeatPadToGlobalMaxCollate:
    """
    Collate function for variable-length point clouds.

    It pads every sample to target_num_points by repeat-sampling existing points.

    Important:
      - It does NOT zero-pad.
      - It applies the same sampled indices to every field whose first dim
        equals the original point number, so per-point labels/features stay aligned.
      - Other fields are collated using PyTorch default_collate.
    """

    def __init__(
        self,
        target_num_points: int,
        point_key: str = "pointcloud",
        shuffle_index: bool = True,
    ) -> None:
        self.target_num_points = int(target_num_points)
        self.point_key = str(point_key)
        self.shuffle_index = bool(shuffle_index)

        if self.target_num_points <= 0:
            raise RuntimeError(
                f"target_num_points must be positive, got {self.target_num_points}"
            )

    def __call__(self, batch: List[Any]) -> Any:
        if len(batch) == 0:
            return batch

        if not isinstance(batch[0], dict):
            return default_collate(batch)

        point_key = _find_point_key(batch[0], preferred_key=self.point_key)

        if point_key is None:
            # No point key found; fallback to PyTorch default behavior.
            return default_collate(batch)

        lengths: List[int] = []
        index_list: List[torch.Tensor] = []

        for item_idx, item in enumerate(batch):
            if point_key not in item:
                raise KeyError(
                    f"point_key='{point_key}' exists in batch[0], "
                    f"but is missing from batch[{item_idx}]."
                )

            src_num_points = _get_first_dim(item[point_key])
            if src_num_points is None:
                raise RuntimeError(
                    f"Cannot get point number from key='{point_key}' "
                    f"at batch[{item_idx}]. type={type(item[point_key])}"
                )

            lengths.append(int(src_num_points))

            index = _make_repeat_pad_index(
                src_num_points=int(src_num_points),
                target_num_points=self.target_num_points,
                shuffle_index=self.shuffle_index,
            )
            index_list.append(index)

        output: Dict[str, Any] = {}

        for key in batch[0].keys():
            values = [item[key] for item in batch]

            # If this field has the same first dim as the point cloud for every sample,
            # treat it as a per-point field and apply the same repeat-padding indices.
            is_per_point_field = True
            for value, src_len in zip(values, lengths):
                first_dim = _get_first_dim(value)
                if first_dim is None or int(first_dim) != int(src_len):
                    is_per_point_field = False
                    break

            if is_per_point_field:
                padded_values = []
                for value, index in zip(values, index_list):
                    tensor_value = _as_tensor_for_collate(value)
                    index = index.to(device=tensor_value.device)
                    padded_value = tensor_value.index_select(0, index)
                    padded_values.append(padded_value)

                output[key] = default_collate(padded_values)
            else:
                try:
                    output[key] = default_collate(values)
                except Exception:
                    # For strings / metadata / variable non-point fields.
                    output[key] = values

        output[f"{point_key}_orig_lengths"] = torch.tensor(lengths, dtype=torch.long)
        output[f"{point_key}_target_num_points"] = torch.full(
            size=(len(batch),),
            fill_value=self.target_num_points,
            dtype=torch.long,
        )

        return output


def _save_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": None,
        "numpy": None,
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    if np is not None:
        state["numpy"] = np.random.get_state()

    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])

    if "torch_cpu" in state and state["torch_cpu"] is not None:
        torch.random.set_rng_state(state["torch_cpu"])

    if torch.cuda.is_available() and state.get("torch_cuda", None) is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    if np is not None and state.get("numpy", None) is not None:
        np.random.set_state(state["numpy"])


def infer_max_points_from_dataset(
    dataset: Any,
    point_key: str = "pointcloud",
    logger: Optional[logging.Logger] = None,
    dataset_name: str = "",
) -> Tuple[int, str]:
    """
    Scan one dataset and return:
      - max number of points
      - actual point key used

    This calls dataset[idx] once for every sample.
    """
    max_points = 0
    max_index = -1
    used_point_key = point_key

    dataset_len = int(len(dataset))

    rng_state = _save_rng_state()

    try:
        for idx in range(dataset_len):
            sample = dataset[idx]

            if not isinstance(sample, dict):
                continue

            cur_point_key = _find_point_key(sample, preferred_key=point_key)
            if cur_point_key is None:
                continue

            n = _get_first_dim(sample[cur_point_key])
            if n is None:
                continue

            n = int(n)
            if n > max_points:
                max_points = n
                max_index = idx
                used_point_key = cur_point_key
    finally:
        # Restore RNG state so scanning dataset does not disturb later training randomness too much.
        _restore_rng_state(rng_state)

    if max_points <= 0:
        raise RuntimeError(
            "Failed to infer max point number from dataset. "
            f"dataset_name={dataset_name}, point_key={point_key}, len={dataset_len}"
        )

    if logger is not None:
        logger.info(
            "[RepeatPadCollate] dataset={} len={} point_key={} "
            "max_points={} max_index={}".format(
                dataset_name,
                dataset_len,
                used_point_key,
                max_points,
                max_index,
            )
        )

    return int(max_points), str(used_point_key)


def infer_global_max_points_from_loaders(
    loaders: List[Any],
    point_key: str = "pointcloud",
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, str]:
    """
    Scan all datasets behind the given loaders and return the global max point count.
    """
    global_max_points = 0
    used_point_key = point_key
    seen_dataset_ids = set()

    for loader_idx, loader in enumerate(loaders):
        if loader is None:
            continue

        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            continue

        dataset_id = id(dataset)
        if dataset_id in seen_dataset_ids:
            continue
        seen_dataset_ids.add(dataset_id)

        local_max, local_key = infer_max_points_from_dataset(
            dataset=dataset,
            point_key=point_key,
            logger=logger,
            dataset_name=f"loader_{loader_idx}",
        )

        if local_max > global_max_points:
            global_max_points = int(local_max)
            used_point_key = str(local_key)

    if global_max_points <= 0:
        raise RuntimeError(
            "Failed to infer global max point number from all loaders."
        )

    if logger is not None:
        logger.info(
            "[RepeatPadCollate] GLOBAL point_key={} global_max_points={}".format(
                used_point_key,
                global_max_points,
            )
        )

    return int(global_max_points), str(used_point_key)


def install_repeat_pad_collate_to_loaders(
    loaders: List[Any],
    target_num_points: int,
    point_key: str = "pointcloud",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Attach RepeatPadToGlobalMaxCollate to existing DataLoader objects.

    PyTorch DataLoader allows changing collate_fn before iteration.
    """
    collate_fn = RepeatPadToGlobalMaxCollate(
        target_num_points=int(target_num_points),
        point_key=str(point_key),
        shuffle_index=True,
    )

    for loader_idx, loader in enumerate(loaders):
        if loader is None:
            continue

        loader.collate_fn = collate_fn

        if logger is not None:
            logger.info(
                "[RepeatPadCollate] installed to loader_{} with "
                "target_num_points={} point_key={}".format(
                    loader_idx,
                    int(target_num_points),
                    str(point_key),
                )
            )


def build_backbone_from_config(backbone_cfg: EasyDict) -> nn.Module:
    type_name = str(backbone_cfg.type)
    module_path, factory_name = type_name.rsplit(".", 1)

    module = importlib.import_module(module_path)
    if hasattr(module, factory_name):
        factory = getattr(module, factory_name)
    else:
        submodule = importlib.import_module(type_name)
        if not hasattr(submodule, factory_name):
            raise AttributeError(
                f"Cannot find factory '{factory_name}' in '{module_path}' or '{type_name}'"
            )
        factory = getattr(submodule, factory_name)

    kwargs = backbone_cfg.get("kwargs", {})
    return factory(**kwargs)


def get_optimizer(parameters: Iterable[torch.nn.Parameter], cfg: EasyDict) -> torch.optim.Optimizer:
    opt_type = str(cfg.optimizer).lower()

    if opt_type == "adam":
        return optim.Adam(
            parameters,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    if opt_type == "sgd":
        return optim.SGD(
            parameters,
            lr=float(cfg.lr),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.weight_decay),
        )

    if opt_type == "adamw":
        betas = cfg.get("betas", [0.9, 0.99])
        return optim.AdamW(
            parameters,
            lr=float(cfg.lr),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(cfg.weight_decay),
        )

    raise NotImplementedError(f"Unsupported optimizer: {cfg.optimizer}")

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _strip_module_prefix_state_dict(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        return {
            k[len("module.") :]: v
            for k, v in state_dict.items()
        }

    return state_dict


def _select_state_dict_from_ckpt(ckpt: Any) -> Any:
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "base_model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def resolve_optional_path(config: EasyDict, path: Any) -> Optional[str]:
    if path is None or path == "":
        return None

    path = str(path)
    if not os.path.isabs(path):
        path = os.path.join(config.exp_path, path)

    return path


def resolve_pretrain_path(config: EasyDict) -> Optional[str]:
    return resolve_optional_path(
        config,
        config.saver.get("pretrain_model", None),
    )


def load_pretrained_network_weights_only(
    model: nn.Module,
    path: Optional[str],
    logger: logging.Logger,
    strict: bool = False,
    load_projection: bool = True,
) -> None:
    """
    Load external checkpoint only as network initialization.

    This function intentionally does NOT load:
      - optimizer
      - prototypes
      - epoch
      - best_metric
    """

    if not path:
        return

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {path}")

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)

    state_dict = _select_state_dict_from_ckpt(ckpt)
    state_dict = _strip_module_prefix_state_dict(state_dict)

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Invalid pretrain checkpoint format: {path}. "
            "Expected state_dict or dict containing key 'model' / 'state_dict' / 'base_model'."
        )

    if not load_projection:
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not (str(k) == "proj" or str(k).startswith("proj."))
        }

    raw_model = unwrap_model(model)

    if strict:
        incompatible = raw_model.load_state_dict(state_dict, strict=True)
        logger.info(
            "[Pretrain Init] Loaded network weights from {} with strict=True | "
            "missing={} unexpected={}".format(
                path,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        )
        return

    current_state = raw_model.state_dict()
    compatible_state = {}
    skipped = []

    for k, v in state_dict.items():
        if k not in current_state:
            skipped.append((k, "not_in_current_model"))
            continue

        if not torch.is_tensor(v):
            skipped.append((k, "not_tensor"))
            continue

        if tuple(current_state[k].shape) != tuple(v.shape):
            skipped.append(
                (
                    k,
                    "shape_mismatch ckpt={} model={}".format(
                        tuple(v.shape),
                        tuple(current_state[k].shape),
                    ),
                )
            )
            continue

        compatible_state[k] = v

    incompatible = raw_model.load_state_dict(compatible_state, strict=False)

    logger.info(
        "[Pretrain Init] Loaded network weights only from {} | "
        "matched={} missing={} unexpected={} skipped={}".format(
            path,
            len(compatible_state),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            len(skipped),
        )
    )

    if len(incompatible.missing_keys) > 0:
        logger.info(
            "[Pretrain Init] missing keys sample: {}".format(
                incompatible.missing_keys[:50]
            )
        )

    if len(incompatible.unexpected_keys) > 0:
        logger.info(
            "[Pretrain Init] unexpected keys sample: {}".format(
                incompatible.unexpected_keys[:50]
            )
        )

    if len(skipped) > 0:
        logger.info(
            "[Pretrain Init] skipped keys sample: {}".format(
                skipped[:50]
            )
        )


def apply_cpe_finetune_policy(
    model: nn.Module,
    config: EasyDict,
    logger: logging.Logger,
) -> None:
    """
    If cpe.freeze_backbone=True:
      freeze everything except trainable_prefixes, default ['proj'].

    For UCA3DALMinkBackbone:
      backbone.* = MinkUNet
      proj.*     = CPE projection head
    """

    raw_model = unwrap_model(model)

    freeze_backbone = bool(config.cpe.get("freeze_backbone", False))

    if not freeze_backbone:
        for p in raw_model.parameters():
            p.requires_grad = True

        logger.info("[FinetunePolicy] Full model is trainable.")
    else:
        trainable_prefixes = config.cpe.get("trainable_prefixes", ["proj"])
        trainable_prefixes = [str(x) for x in trainable_prefixes]

        for name, p in raw_model.named_parameters():
            p.requires_grad = False

            for prefix in trainable_prefixes:
                if name == prefix or name.startswith(prefix + "."):
                    p.requires_grad = True
                    break

        logger.info(
            "[FinetunePolicy] Freeze backbone=True. Trainable prefixes={}".format(
                trainable_prefixes
            )
        )

    total_params = 0
    trainable_params = 0
    trainable_names = []

    for name, p in raw_model.named_parameters():
        n = int(p.numel())
        total_params += n

        if p.requires_grad:
            trainable_params += n
            trainable_names.append(name)

    if trainable_params <= 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check cpe.freeze_backbone and cpe.trainable_prefixes."
        )

    logger.info(
        "[FinetunePolicy] trainable_params={} / total_params={} ({:.4f}%)".format(
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )
    )

    logger.info(
        "[FinetunePolicy] trainable parameter names: {}".format(
            trainable_names
        )
    )


def set_cpe_train_mode(
    model: nn.Module,
    config: EasyDict,
) -> None:
    """
    Keep UCA3DALMinkBackbone itself in train mode so train_augment still works.

    If backbone is frozen, only put raw_model.backbone, i.e. MinkUNet, into eval mode.
    This prevents frozen backbone BN stats from drifting, while projection head stays train().
    """

    model.train()

    raw_model = unwrap_model(model)

    if bool(config.cpe.get("freeze_backbone", False)):
        if hasattr(raw_model, "backbone"):
            raw_model.backbone.eval()

        if hasattr(raw_model, "proj"):
            raw_model.proj.train()


def normalize_cls_name(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")

    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.item())
        return str(x.detach().cpu().tolist())

    return str(x)


def get_cls_targets(
    batch: Dict[str, Any],
    device: Optional[torch.device] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    Use object category label, not anomaly label.
    batch['label'] is normal/abnormal; batch['cls_label'] is category id.
    """

    for label_key in ["cls_label", "class_label", "category_id"]:
        if label_key in batch:
            value = batch[label_key]

            if torch.is_tensor(value):
                target = value.view(-1).long()
            elif isinstance(value, (list, tuple)):
                target = torch.tensor(value, dtype=torch.long)
            else:
                target = torch.tensor([value], dtype=torch.long)

            if device is not None:
                target = target.to(device=device, dtype=torch.long)
            return target

    if class_to_idx is None:
        raise KeyError(
            "Cannot build CPE target. Expected 'cls_label' in batch, "
            "or provide class_to_idx for clsname fallback."
        )

    name_key = None
    for key in ["clsname", "class_name", "category", "class"]:
        if key in batch:
            name_key = key
            break

    if name_key is None:
        raise KeyError(
            "Cannot build CPE target. Expected one of "
            "['cls_label', 'class_label', 'category_id', 'clsname']."
        )

    cls_names = batch[name_key]
    if isinstance(cls_names, (str, bytes)):
        cls_names = [cls_names]

    labels = []
    for cls_name in cls_names:
        name = normalize_cls_name(cls_name)
        if name not in class_to_idx:
            raise KeyError(
                f"Class name '{name}' not found in class_to_idx. "
                f"Available classes={list(class_to_idx.keys())}"
            )
        labels.append(int(class_to_idx[name]))

    target = torch.tensor(labels, dtype=torch.long)
    if device is not None:
        target = target.to(device=device, dtype=torch.long)
    return target


def _get_nested_dataset_attr(dataset: Any, attr: str) -> Any:
    seen = set()
    cur = dataset

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, attr):
            return getattr(cur, attr)
        cur = getattr(cur, "dataset", None)

    return None


def infer_class_names_from_loader(loader: Any, num_classes: int) -> List[str]:
    """
    No need for dataset.class_names in config.
    Prefer CustomDataset.label_dict, so the display order follows your dataset.
    """

    dataset = getattr(loader, "dataset", None)
    label_dict = _get_nested_dataset_attr(dataset, "label_dict")

    names = [None for _ in range(num_classes)]

    if isinstance(label_dict, dict) and len(label_dict) > 0:
        for name, idx in label_dict.items():
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if 0 <= idx_int < num_classes:
                names[idx_int] = str(name)

    if any(item is None for item in names):
        metas = _get_nested_dataset_attr(dataset, "metas")
        if isinstance(metas, list):
            for item in metas:
                if not isinstance(item, dict):
                    continue
                if "clsname" not in item:
                    continue
                label = item.get("cls_label", None)
                if label is not None:
                    idx_int = int(label)
                    if 0 <= idx_int < num_classes and names[idx_int] is None:
                        names[idx_int] = str(item["clsname"])

    for idx in range(num_classes):
        if names[idx] is None:
            names[idx] = f"class_{idx}"

    return [str(x) for x in names]


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        value_f = float(value)
    except Exception:
        return str(value)

    if value_f != value_f:
        return "nan"

    return f"{value_f:.{digits}f}"


def _fallback_tabulate(records: List[List[Any]], headers: List[str]) -> str:
    rows = [headers] + records
    rows = [[str(item) for item in row] for row in rows]
    widths = [max(len(row[col]) for row in rows) for col in range(len(headers))]

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(
            row[col].center(widths[col])
            for col in range(len(headers))
        ) + " |"

    sep = "|:" + ":|:".join("-" * width for width in widths) + ":|"
    output = [fmt_row(rows[0]), sep]
    output.extend(fmt_row(row) for row in rows[1:])
    return "\n".join(output)


def compute_cls_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    pred = pred.detach().cpu().long().view(-1)
    target = target.detach().cpu().long().view(-1)

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(num_classes)]

    if len(class_names) != num_classes:
        raise RuntimeError(
            f"len(class_names)={len(class_names)} but num_classes={num_classes}"
        )

    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for t, p in zip(target.tolist(), pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_mat[t, p] += 1

    tp = conf_mat.diag().double()
    support = conf_mat.sum(dim=1).double()
    pred_count = conf_mat.sum(dim=0).double()

    total_num = int(support.sum().item())
    correct_num = int(tp.sum().item())
    acc1 = correct_num / max(total_num, 1)

    valid_class = support > 0

    recall = torch.full((num_classes,), float("nan"), dtype=torch.double)
    recall[valid_class] = tp[valid_class] / support[valid_class].clamp_min(1.0)

    precision = torch.full((num_classes,), float("nan"), dtype=torch.double)
    valid_pred = pred_count > 0
    precision[valid_pred] = tp[valid_pred] / pred_count[valid_pred].clamp_min(1.0)

    f1 = torch.zeros(num_classes, dtype=torch.double)
    denom = precision + recall
    valid_f1 = torch.isfinite(denom) & (denom > 0)
    f1[valid_f1] = 2.0 * precision[valid_f1] * recall[valid_f1] / denom[valid_f1]

    if bool(valid_class.any()):
        mean_class_acc = float(recall[valid_class].mean().item())
        macro_f1 = float(f1[valid_class].mean().item())
    else:
        mean_class_acc = 0.0
        macro_f1 = 0.0

    per_class = []
    for idx in range(num_classes):
        per_class.append(
            {
                "idx": idx,
                "clsname": str(class_names[idx]),
                "correct": int(tp[idx].item()),
                "total": int(support[idx].item()),
                "pred_total": int(pred_count[idx].item()),
                "acc": float(recall[idx].item()) if bool(valid_class[idx]) else float("nan"),
                "precision": float(precision[idx].item()) if bool(valid_pred[idx]) else float("nan"),
                "f1": float(f1[idx].item()),
            }
        )

    return {
        "acc1": float(acc1),
        "overall_acc": float(acc1),
        "mean_class_acc": float(mean_class_acc),
        "macro_f1": float(macro_f1),
        "total_num": total_num,
        "correct_num": correct_num,
        "per_class": per_class,
        "conf_mat": conf_mat,
    }


def format_cls_acc_table(ret_metrics: Dict[str, Any]) -> str:
    headers = ["clsname", "cls-idx", "correct", "total", "cls-ACC"]
    records = []

    for item in ret_metrics.get("per_class", []):
        records.append(
            [
                item["clsname"],
                item["idx"],
                item["correct"],
                item["total"],
                _format_float(item["acc"]),
            ]
        )

    records.append(["mean", "-", "-", "-", _format_float(ret_metrics["mean_class_acc"])])
    records.append(
        [
            "overall",
            "-",
            ret_metrics["correct_num"],
            ret_metrics["total_num"],
            _format_float(ret_metrics["overall_acc"]),
        ]
    )

    if tabulate is not None:
        return tabulate.tabulate(
            records,
            headers,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )

    return _fallback_tabulate(records, headers)


def encode_once(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """
    One CPE augmented view.

    If UCA3DALMinkBackbone has train_augment=True and model.train(), two calls
    produce two different augmented sparse views from the same point cloud.
    """

    if hasattr(model, "forward_embed"):
        z = model.forward_embed(batch)
        return F.normalize(z, dim=1)

    if hasattr(model, "forward_features"):
        features, _ = model.forward_features(batch["pointcloud"])
        z = features.squeeze(-1)
        return F.normalize(z, dim=1)

    outputs = model(batch)
    for key in ["proto_feature", "raw_proto_feature", "xyz_features", "raw_xyz_features"]:
        if key not in outputs:
            continue
        z = outputs[key]
        if z.dim() == 3:
            z = z.mean(dim=-1)
        return F.normalize(z, dim=1)

    raise KeyError(
        "Cannot get CPE embedding. Expected model.forward_embed(), "
        "model.forward_features(), or output feature keys."
    )


def train_one_epoch(
    train_loader: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    supcon_crit: nn.Module,
    proto_mod: PrototypeMemory,
    proto_crit: nn.Module,
    cfg: EasyDict,
    epoch: int,
    logger: logging.Logger,
    tb_logger: Optional[SummaryWriter],
    global_step_base: int,
    task_id: int = 0,
) -> Dict[str, float]:
    set_cpe_train_mode(model, cfg)

    loss_sum = 0.0
    supcon_sum = 0.0
    proto_sum = 0.0
    acc_sum = 0.0
    num_sum = 0

    end = time.time()

    for i, batch in enumerate(train_loader):
        batch["task_id"] = task_id
        batch["is_train"] = True

        target_cpu = get_cls_targets(batch, device=None)
        if int(target_cpu.numel()) <= 1:
            logger.info(
                "Skip tiny train batch at epoch {} iter {} because batch size is {}. "
                "CPE projection uses BatchNorm1d in train mode.".format(
                    epoch + 1,
                    i + 1,
                    int(target_cpu.numel()),
                )
            )
            continue

        lr = cosine_lr_after_step(
            optimizer=optimizer,
            base_lr=float(cfg.cpe.lr),
            epoch=epoch,
            step_epoch=int(cfg.cpe.step_epoch),
            total_epochs=int(cfg.cpe.epochs),
            clip=1.0e-6,
        )

        z1 = encode_once(model, batch)
        z2 = encode_once(model, batch)
        
        target = target_cpu.to(device=z1.device, dtype=torch.long)

        features = torch.stack([z1, z2], dim=1)
        supcon_loss = supcon_crit(features, target)

        proto_loss = torch.tensor(0.0, device=z1.device)
        loss = supcon_loss

        if float(cfg.cpe.proto_loss_weight) > 0.0:
            z_all = torch.cat([z1, z2], dim=0)
            y_all = torch.cat([target, target], dim=0)

            with torch.no_grad():
                proto_mod.update(z_all.detach(), y_all.detach())

            proto_loss, proto_info = proto_crit(
                z=z_all,
                y=y_all,
                proto_mod=proto_mod,
            )

            loss = supcon_loss + float(cfg.cpe.proto_loss_weight) * proto_loss
        else:
            with torch.no_grad():
                z_all = torch.cat([z1, z2], dim=0)
                y_all = torch.cat([target, target], dim=0)
                proto_mod.update(z_all.detach(), y_all.detach())

        optimizer.zero_grad()
        loss.backward()

        if cfg.cpe.get("clip_max_norm", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg.cpe.clip_max_norm),
            )

        optimizer.step()

        with torch.no_grad():
            z_mean = F.normalize(0.5 * (z1 + z2), dim=1)
            mp_cfg = EasyDict(cfg.cpe.get("multi_proto", {}))

            class_scores, _ = proto_mod.class_scores(
                z_mean,
                pool_tau=float(mp_cfg.get("pool_tau", 0.02)),
            )

            logits = class_scores / float(cfg.cpe.temperature)
            pred = logits.argmax(dim=1)
            acc = (pred == target).float().mean()


        batch_size = int(target.numel())
        loss_sum += float(loss.item()) * batch_size
        supcon_sum += float(supcon_loss.item()) * batch_size
        proto_sum += float(proto_loss.item()) * batch_size
        acc_sum += float(acc.item()) * batch_size
        num_sum += batch_size

        global_step = global_step_base + i + 1

        if (i + 1) % int(cfg.cpe.print_freq_step) == 0:
            avg_loss = loss_sum / max(num_sum, 1)
            avg_supcon = supcon_sum / max(num_sum, 1)
            avg_proto = proto_sum / max(num_sum, 1)
            avg_acc = acc_sum / max(num_sum, 1)

            if tb_logger is not None:
                tb_logger.add_scalar("cpe_train/loss", avg_loss, global_step)
                tb_logger.add_scalar("cpe_train/supcon_loss", avg_supcon, global_step)
                tb_logger.add_scalar("cpe_train/proto_nce_loss", avg_proto, global_step)
                tb_logger.add_scalar("cpe_train/proto_acc", avg_acc, global_step)
                tb_logger.add_scalar("cpe_train/lr", lr, global_step)
                tb_logger.flush()

            logger.info(
                "Epoch: [{}/{}]\t"
                "Iter: [{}/{}]\t"
                "Time {:.2f}\t"
                "Loss {:.5f}\t"
                "SupCon {:.5f}\t"
                "ProtoNCE {:.5f}\t"
                "ProtoAcc {:.4f}\t"
                "LR {:.6f}".format(
                    epoch + 1,
                    int(cfg.cpe.epochs),
                    i + 1,
                    len(train_loader),
                    time.time() - end,
                    avg_loss,
                    avg_supcon,
                    avg_proto,
                    avg_acc,
                    lr,
                )
            )
            end = time.time()

    return {
        "loss": float(loss_sum / max(num_sum, 1)),
        "supcon_loss": float(supcon_sum / max(num_sum, 1)),
        "proto_nce_loss": float(proto_sum / max(num_sum, 1)),
        "proto_acc": float(acc_sum / max(num_sum, 1)),
        "total_num": float(num_sum),
    }


@torch.no_grad()
def validate(
    val_loader: Any,
    model: nn.Module,
    proto_mod: AdaptiveMultiPrototypeMemory,
    cfg: EasyDict,
    logger: logging.Logger,
    epoch: Optional[int] = None,
    task_id: int = 0,
    split_name: str = "test",
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()

    device = torch.device("cuda", torch.cuda.current_device())

    # multi-prototype 版本不要再依赖局部变量 prototypes
    num_classes = int(proto_mod.num_classes)
    # 或者也可以写成：
    # num_classes = int(proto_mod.proto.shape[0])

    class_names = infer_class_names_from_loader(val_loader, num_classes)

    local_loss_sum = 0.0
    local_num = 0
    pred_list = []
    target_list = []

    batch_time_sum = 0.0
    batch_time_count = 0
    end = time.time()

    val_print_freq = int(cfg.cpe.get("val_print_freq_step", cfg.cpe.print_freq_step))
    val_print_freq = max(val_print_freq, 1)

    for i, batch in enumerate(val_loader):
        batch["task_id"] = task_id
        batch["is_train"] = False

        z = encode_once(model, batch)
        target = get_cls_targets(batch, device=z.device)

        mp_cfg = EasyDict(cfg.cpe.get("multi_proto", {}))

        class_scores, _ = proto_mod.class_scores(
            z,
            pool_tau=float(mp_cfg.get("pool_tau", 0.02)),
        )

        logits = class_scores / float(cfg.cpe.temperature)
        loss = F.cross_entropy(logits, target, reduction="mean")
        pred = logits.argmax(dim=1)

        topk = min(int(mp_cfg.get("eval_topk", 3)), logits.shape[1])
        topk_pred = logits.topk(k=topk, dim=1).indices
        topk_correct = (topk_pred == target.view(-1, 1)).any(dim=1).float().mean()

        logger.info(
            "Top{}Acc {:.6f}".format(
                topk,
                float(topk_correct.item()),
            )
        )

        batch_size = int(target.numel())
        local_loss_sum += float(loss.item()) * batch_size
        local_num += batch_size
        pred_list.append(pred.detach().cpu())
        target_list.append(target.detach().cpu())

        elapsed = time.time() - end
        batch_time_sum += elapsed
        batch_time_count += 1
        end = time.time()

        if (i + 1) % val_print_freq == 0:
            logger.info(
                "Test: [{}/{}]\tTime {:.3f} ({:.3f})".format(
                    i + 1,
                    len(val_loader),
                    elapsed,
                    batch_time_sum / max(batch_time_count, 1),
                )
            )

    if len(pred_list) == 0:
        raise RuntimeError(f"No samples found in {split_name} loader.")

    all_pred = torch.cat(pred_list, dim=0).to(device)
    all_target = torch.cat(target_list, dim=0).to(device)
    final_loss = float(local_loss_sum / max(local_num, 1))

    logger.info("Gathering final classification results ...")

    ret_metrics = compute_cls_metrics(
        pred=all_pred,
        target=all_target,
        num_classes=num_classes,
        class_names=class_names,
    )
    ret_metrics["loss"] = final_loss

    epoch_text = "" if epoch is None else f"epoch={epoch + 1} "
    logger.info(
        " * {}{} Loss {:.5f}\ttotal_num={:.1f}\t"
        "Acc@1 {:.6f}\tMeanClassAcc {:.6f}\tMacroF1 {:.6f}".format(
            epoch_text,
            split_name,
            ret_metrics["loss"],
            float(ret_metrics["total_num"]),
            ret_metrics["acc1"],
            ret_metrics["mean_class_acc"],
            ret_metrics["macro_f1"],
        )
    )

    logger.info("\n{}".format(format_cls_acc_table(ret_metrics)))

    if was_training:
        set_cpe_train_mode(model, cfg)

    return ret_metrics


def _move_optimizer_state_to_cuda(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cuda()


def restore_cpe_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    proto_module: Optional[PrototypeMemory],
    path: Optional[str],
    logger: Optional[logging.Logger] = None,
    load_optimizer: bool = True,
) -> Tuple[int, Optional[float]]:
    if not path:
        return 0, None

    if not os.path.isfile(path):
        if logger is not None:
            logger.info(f"Skip resume because checkpoint does not exist: {path}")
        return 0, None

    ckpt = torch.load(path, map_location="cuda")

    if isinstance(ckpt, dict) and "model" in ckpt:
        model_state = ckpt["model"]
    else:
        model_state = ckpt

    incompatible = model.load_state_dict(model_state, strict=False)

    if logger is not None:
        logger.info(
            "Loaded CPE model from {} | missing={} unexpected={}".format(
                path,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        )

    if (
        load_optimizer
        and isinstance(ckpt, dict)
        and optimizer is not None
        and "optimizer" in ckpt
        and ckpt["optimizer"] is not None
    ):
        optimizer.load_state_dict(ckpt["optimizer"])
        _move_optimizer_state_to_cuda(optimizer)
        if logger is not None:
            logger.info("Loaded optimizer state.")

    if (
        isinstance(ckpt, dict)
        and proto_module is not None
        and "prototypes" in ckpt
        and ckpt["prototypes"] is not None
    ):
        proto_module.load_state_dict(ckpt["prototypes"], strict=False)
        if logger is not None:
            logger.info("Loaded prototype memory.")

    start_epoch = 0

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        start_epoch = int(ckpt["epoch"]) + 1

    best_metric = None
    if isinstance(ckpt, dict) and "best_metric" in ckpt:
        try:
            best_metric = float(ckpt["best_metric"])
        except Exception:
            best_metric = None

    return start_epoch, best_metric

def _get_tsne_cfg(cfg: EasyDict) -> EasyDict:
    return EasyDict(cfg.cpe.get("tsne", {}))


def _tsne_is_enabled(cfg: EasyDict) -> bool:
    tsne_cfg = _get_tsne_cfg(cfg)
    if not bool(tsne_cfg.get("enabled", False)):
        return False

    if not bool(tsne_cfg.get("save", True)) and not bool(tsne_cfg.get("visualize", False)):
        return False

    return True


def _resolve_tsne_save_dir(config: EasyDict, tsne_cfg: EasyDict) -> str:
    save_dir = str(tsne_cfg.get("save_dir", "fig"))

    if os.path.isabs(save_dir):
        return save_dir

    # 推荐写 save_dir: fig，此时保存到 ./log/fig
    if save_dir in ["fig", "./fig"]:
        return os.path.join(config.log_path, "fig")

    # 如果你写 ./log/fig，则相对 config.yaml 所在目录
    return os.path.join(config.exp_path, save_dir)


def _normalize_xyz_shape_for_tsne(xyz: Any) -> torch.Tensor:
    if not torch.is_tensor(xyz):
        xyz = torch.as_tensor(xyz, dtype=torch.float32)

    xyz = xyz.detach().cpu().float()

    if xyz.dim() == 2:
        if xyz.shape[-1] == 3:
            xyz = xyz.unsqueeze(0)
        elif xyz.shape[0] == 3:
            xyz = xyz.t().unsqueeze(0)
        else:
            raise RuntimeError(
                f"Cannot infer point dimension from 2D shape={tuple(xyz.shape)}."
            )

    if xyz.dim() != 3:
        raise RuntimeError(
            f"pointcloud for t-SNE must be 3D, got shape={tuple(xyz.shape)}"
        )

    if xyz.shape[-1] == 3:
        return xyz.contiguous()

    if xyz.shape[1] == 3:
        return xyz.transpose(1, 2).contiguous()

    raise RuntimeError(
        f"Cannot infer point dimension from shape={tuple(xyz.shape)}. "
        "Expected [B,N,3] or [B,3,N]."
    )


def _resolve_batch_point_key(batch: Dict[str, Any], preferred_key: str = "pointcloud") -> str:
    if preferred_key in batch:
        return preferred_key

    for key in POINT_KEY_CANDIDATES:
        if key in batch:
            return key

    raise KeyError(
        f"Cannot find point cloud key in batch. "
        f"Tried preferred_key={preferred_key}, candidates={POINT_KEY_CANDIDATES}, "
        f"batch keys={list(batch.keys())}"
    )


def _pointcloud_stats_np(xyz_b_n_3: np.ndarray) -> np.ndarray:
    """
    Build low-dimensional geometry statistics for input-side t-SNE.
    Shape:
        input:  [B, N, 3]
        output: [B, 22]
    """
    feats = []

    for b in range(xyz_b_n_3.shape[0]):
        p = np.asarray(xyz_b_n_3[b], dtype=np.float32)

        if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] <= 0:
            raise RuntimeError(f"Invalid point cloud shape for stats: {p.shape}")

        n = int(p.shape[0])
        centroid = p.mean(axis=0)
        centered = p - centroid.reshape(1, 3)

        std = centered.std(axis=0)
        xyz_min = centered.min(axis=0)
        xyz_max = centered.max(axis=0)

        radius = np.sqrt(np.sum(centered ** 2, axis=1))
        radius_mean = np.array([radius.mean()], dtype=np.float32)
        radius_std = np.array([radius.std()], dtype=np.float32)
        radius_max = np.array([radius.max()], dtype=np.float32)
        radius_q = np.quantile(radius, [0.25, 0.50, 0.75]).astype(np.float32)

        if n > 1:
            cov = np.cov(centered.T).astype(np.float32)
            eig = np.linalg.eigvalsh(cov).astype(np.float32)
            eig = np.sort(eig)[::-1]
        else:
            eig = np.zeros((3,), dtype=np.float32)

        feat = np.concatenate(
            [
                centroid.astype(np.float32),
                std.astype(np.float32),
                xyz_min.astype(np.float32),
                xyz_max.astype(np.float32),
                radius_mean,
                radius_std,
                radius_max,
                radius_q,
                eig.astype(np.float32),
                np.array([float(n)], dtype=np.float32),
            ],
            axis=0,
        )

        feats.append(feat)

    return np.stack(feats, axis=0).astype(np.float32)


def _pointcloud_raw_sample_np(
    xyz_b_n_3: np.ndarray,
    num_points: int = 256,
) -> np.ndarray:
    """
    Deterministically convert each raw point cloud to a fixed vector.

    This is for visualizing input-side distribution before the model.
    It does not use the model's augmentation or sparse voxelization.

    Shape:
        input:  [B, N, 3]
        output: [B, num_points * 3]
    """
    num_points = int(num_points)
    if num_points <= 0:
        raise RuntimeError(f"input_num_points must be positive, got {num_points}")

    output = []

    for b in range(xyz_b_n_3.shape[0]):
        p = np.asarray(xyz_b_n_3[b], dtype=np.float32)

        if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] <= 0:
            raise RuntimeError(f"Invalid point cloud shape for raw_sample: {p.shape}")

        # Center + scale normalize only for visualization.
        p = p - p.mean(axis=0, keepdims=True)
        radius = np.max(np.sqrt(np.sum(p ** 2, axis=1)))
        radius = max(float(radius), 1.0e-12)
        p = p / radius

        # Make sampling deterministic by sorting coordinates.
        # np.lexsort uses the last key as primary key.
        order = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
        p = p[order]

        n = int(p.shape[0])

        if n >= num_points:
            idx = np.linspace(0, n - 1, num_points)
            idx = np.round(idx).astype(np.int64)
            sampled = p[idx]
        else:
            base_idx = np.arange(n, dtype=np.int64)
            extra_num = num_points - n
            extra_idx = np.linspace(0, n - 1, extra_num)
            extra_idx = np.round(extra_idx).astype(np.int64)
            idx = np.concatenate([base_idx, extra_idx], axis=0)
            sampled = p[idx]

        output.append(sampled.reshape(-1).astype(np.float32))

    return np.stack(output, axis=0).astype(np.float32)


def build_input_tsne_features(
    batch: Dict[str, Any],
    cfg: EasyDict,
    tsne_cfg: EasyDict,
) -> np.ndarray:
    point_key = str(tsne_cfg.get("point_key", cfg.cpe.get("point_key", "pointcloud")))
    point_key = _resolve_batch_point_key(batch, preferred_key=point_key)

    xyz = _normalize_xyz_shape_for_tsne(batch[point_key])
    xyz_np = xyz.numpy().astype(np.float32)

    mode = str(tsne_cfg.get("input_mode", "raw_sample")).lower()

    if mode in ["raw", "raw_sample", "point", "points"]:
        num_points = int(tsne_cfg.get("input_num_points", 256))
        return _pointcloud_raw_sample_np(xyz_np, num_points=num_points)

    if mode in ["stats", "geometry", "geom"]:
        return _pointcloud_stats_np(xyz_np)

    raise ValueError(
        f"Unsupported cpe.tsne.input_mode={mode}. "
        "Use 'raw_sample' or 'stats'."
    )


@torch.no_grad()
def collect_tsne_features(
    loader: Any,
    model: nn.Module,
    proto_mod: PrototypeMemory,
    cfg: EasyDict,
    logger: logging.Logger,
    task_id: int = 0,
    split_name: str = "test",
) -> Dict[str, Any]:
    tsne_cfg = _get_tsne_cfg(cfg)

    max_samples = int(tsne_cfg.get("max_samples_per_split", 1200))
    if max_samples <= 0:
        max_samples = 10 ** 12

    was_training = model.training
    model.eval()

    prototypes = F.normalize(proto_mod.proto.detach(), dim=1)

    input_features = []
    output_features = []
    labels = []
    preds = []
    splits = []

    collected = 0

    for i, batch in enumerate(loader):
        if collected >= max_samples:
            break

        batch["task_id"] = task_id
        batch["is_train"] = False

        input_np = build_input_tsne_features(
            batch=batch,
            cfg=cfg,
            tsne_cfg=tsne_cfg,
        )

        z = encode_once(model, batch)
        target = get_cls_targets(batch, device=z.device)

        logits = torch.matmul(z, prototypes.T) / float(cfg.cpe.temperature)
        pred = logits.argmax(dim=1)

        batch_size = int(target.numel())
        take = min(batch_size, max_samples - collected)

        input_features.append(input_np[:take])
        output_features.append(z[:take].detach().cpu().float().numpy())
        labels.append(target[:take].detach().cpu().long().numpy())
        preds.append(pred[:take].detach().cpu().long().numpy())
        splits.extend([split_name for _ in range(take)])

        collected += take

    if was_training:
        set_cpe_train_mode(model, cfg)
    else:
        model.eval()

    if collected <= 0:
        raise RuntimeError(f"No samples collected for t-SNE split={split_name}")

    return {
        "input_features": np.concatenate(input_features, axis=0),
        "output_features": np.concatenate(output_features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "preds": np.concatenate(preds, axis=0),
        "splits": np.asarray(splits),
    }


def _fit_tsne_np(
    features: np.ndarray,
    tsne_cfg: EasyDict,
    logger: logging.Logger,
    name: str,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    x = np.asarray(features, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim != 2:
        raise RuntimeError(f"t-SNE features must be 2D, got shape={x.shape}")

    n_samples, feat_dim = x.shape

    if n_samples < 3:
        raise RuntimeError(
            f"Need at least 3 samples for t-SNE, got {n_samples} for {name}"
        )

    # Standardize.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / (std + 1.0e-6)

    # PCA first is faster and usually more stable for high-dimensional raw point vectors.
    pca_dim = int(tsne_cfg.get("pca_dim", 50))
    if pca_dim > 0 and feat_dim > pca_dim and n_samples > 3:
        pca_dim_eff = min(pca_dim, feat_dim, n_samples - 1)
        logger.info(
            f"[t-SNE] {name}: PCA {feat_dim} -> {pca_dim_eff} before t-SNE"
        )
        x = PCA(
            n_components=pca_dim_eff,
            random_state=int(tsne_cfg.get("random_state", 42)),
        ).fit_transform(x)

    perplexity = float(tsne_cfg.get("perplexity", 30))
    # sklearn requires perplexity < n_samples.
    perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
    # A conservative upper bound often works better.
    perplexity = min(perplexity, max(1.0, float(n_samples - 1) / 3.0))

    logger.info(
        f"[t-SNE] {name}: samples={n_samples}, dim={x.shape[1]}, perplexity={perplexity:.2f}"
    )

    common_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(tsne_cfg.get("random_state", 42)),
    )

    n_iter = int(tsne_cfg.get("n_iter", 1000))

    # sklearn old/new versions use different argument names.
    try:
        emb2d = TSNE(
            **common_kwargs,
            max_iter=n_iter,
        ).fit_transform(x)
    except TypeError:
        emb2d = TSNE(
            **common_kwargs,
            n_iter=n_iter,
        ).fit_transform(x)

    return np.asarray(emb2d, dtype=np.float32)


def _plot_tsne_2d(
    emb2d: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    splits: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str,
    tsne_cfg: EasyDict,
    logger: logging.Logger,
) -> None:
    show_fig = bool(tsne_cfg.get("visualize", False))
    save_fig = bool(tsne_cfg.get("save", True))

    import matplotlib

    if not show_fig:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    emb2d = np.asarray(emb2d)
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    splits = np.asarray(splits)

    fig_w = float(tsne_cfg.get("fig_w", 10))
    fig_h = float(tsne_cfg.get("fig_h", 8))
    point_size = float(tsne_cfg.get("point_size", 14))
    alpha = float(tsne_cfg.get("alpha", 0.75))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    unique_splits = list(dict.fromkeys(splits.tolist()))
    markers = {
        "train": "o",
        "train_eval": "o",
        "val": "^",
        "test": "^",
        "prototype": "*",
    }
    fallback_markers = ["o", "^", "s", "D", "P", "X"]

    scatter_obj = None

    num_classes = max(len(class_names), int(labels.max()) + 1 if labels.size > 0 else 1)

    for split_idx, split in enumerate(unique_splits):
        idx = splits == split
        if not np.any(idx):
            continue

        marker = markers.get(str(split), fallback_markers[split_idx % len(fallback_markers)])

        size = point_size
        if str(split) == "prototype":
            size = point_size * 8.0

        scatter_obj = ax.scatter(
            emb2d[idx, 0],
            emb2d[idx, 1],
            c=labels[idx],
            cmap="tab20",
            vmin=0,
            vmax=max(num_classes - 1, 1),
            s=size,
            alpha=1.0 if str(split) == "prototype" else alpha,
            marker=marker,
            label=str(split),
            linewidths=0.4 if str(split) == "prototype" else 0.0,
            edgecolors="black" if str(split) == "prototype" else "none",
        )

    # Mark wrong predictions.
    valid_pred = preds >= 0
    wrong = valid_pred & (preds != labels)

    if np.any(wrong):
        ax.scatter(
            emb2d[wrong, 0],
            emb2d[wrong, 1],
            facecolors="none",
            edgecolors="black",
            s=point_size * 2.2,
            linewidths=0.8,
            marker="o",
            label="wrong pred",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if scatter_obj is not None:
        cbar = fig.colorbar(scatter_obj, ax=ax)
        cbar.set_label("true class id")

        if bool(tsne_cfg.get("show_class_names_on_colorbar", True)):
            ticks = np.arange(len(class_names))
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(
                [f"{idx}: {name}" for idx, name in enumerate(class_names)]
            )

    fig.tight_layout()

    if save_fig:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=int(tsne_cfg.get("dpi", 200)))
        logger.info(f"[t-SNE] saved figure: {save_path}")

    if show_fig:
        plt.show()

    plt.close(fig)


@torch.no_grad()
def run_tsne_visualization(
    train_loader: Optional[Any],
    val_loader: Optional[Any],
    model: nn.Module,
    proto_mod: PrototypeMemory,
    cfg: EasyDict,
    logger: logging.Logger,
    epoch: Optional[int],
    task_id: int = 0,
) -> None:
    if not _tsne_is_enabled(cfg):
        return

    tsne_cfg = _get_tsne_cfg(cfg)

    if epoch is None:
        if not bool(tsne_cfg.get("run_on_evaluate", True)):
            return
    else:
        epoch_freq = int(tsne_cfg.get("epoch_freq", cfg.cpe.get("val_frequent_epoch", 10)))
        epoch_freq = max(epoch_freq, 1)
        if (epoch + 1) % epoch_freq != 0:
            return

    split_cfg = [str(x).lower() for x in _as_list(tsne_cfg.get("splits", ["train", "test"]))]

    collected = []

    if train_loader is not None and ("train" in split_cfg or "train_eval" in split_cfg):
        logger.info("[t-SNE] collecting train_eval features ...")
        collected.append(
            collect_tsne_features(
                loader=train_loader,
                model=model,
                proto_mod=proto_mod,
                cfg=cfg,
                logger=logger,
                task_id=task_id,
                split_name="train_eval",
            )
        )

    if val_loader is not None and ("test" in split_cfg or "val" in split_cfg):
        logger.info("[t-SNE] collecting test features ...")
        collected.append(
            collect_tsne_features(
                loader=val_loader,
                model=model,
                proto_mod=proto_mod,
                cfg=cfg,
                logger=logger,
                task_id=task_id,
                split_name="test",
            )
        )

    if len(collected) == 0:
        logger.info("[t-SNE] no split selected, skip.")
        return

    class_names = infer_class_names_from_loader(
        train_loader if train_loader is not None else val_loader,
        int(proto_mod.proto.shape[0]),
    )

    save_dir = _resolve_tsne_save_dir(cfg, tsne_cfg)
    os.makedirs(save_dir, exist_ok=True)

    tag = "eval" if epoch is None else f"epoch_{epoch + 1:04d}"

    # -------------------------
    # Input-side t-SNE.
    # -------------------------
    input_features = np.concatenate(
        [item["input_features"] for item in collected],
        axis=0,
    )
    input_labels = np.concatenate(
        [item["labels"] for item in collected],
        axis=0,
    )
    input_preds = np.concatenate(
        [item["preds"] for item in collected],
        axis=0,
    )
    input_splits = np.concatenate(
        [item["splits"] for item in collected],
        axis=0,
    )

    input_emb2d = _fit_tsne_np(
        input_features,
        tsne_cfg=tsne_cfg,
        logger=logger,
        name=f"task{task_id}_{tag}_input",
    )

    input_path = os.path.join(
        save_dir,
        f"tsne_task{task_id}_{tag}_input.png",
    )

    _plot_tsne_2d(
        emb2d=input_emb2d,
        labels=input_labels,
        preds=input_preds,
        splits=input_splits,
        class_names=class_names,
        save_path=input_path,
        title=f"Input point-cloud t-SNE | task={task_id} | {tag}",
        tsne_cfg=tsne_cfg,
        logger=logger,
    )

    # -------------------------
    # Output embedding t-SNE.
    # -------------------------
    output_features = np.concatenate(
        [item["output_features"] for item in collected],
        axis=0,
    )
    output_labels = np.concatenate(
        [item["labels"] for item in collected],
        axis=0,
    )
    output_preds = np.concatenate(
        [item["preds"] for item in collected],
        axis=0,
    )
    output_splits = np.concatenate(
        [item["splits"] for item in collected],
        axis=0,
    )

    if bool(tsne_cfg.get("include_prototypes", True)):
        proto = F.normalize(proto_mod.proto.detach(), dim=1).cpu().float().numpy()
        counts = proto_mod.counts.detach().cpu().numpy()
        valid = counts > 0

        if np.any(valid):
            proto_labels = np.arange(proto.shape[0], dtype=np.int64)[valid]
            output_features = np.concatenate(
                [output_features, proto[valid]],
                axis=0,
            )
            output_labels = np.concatenate(
                [output_labels, proto_labels],
                axis=0,
            )
            output_preds = np.concatenate(
                [output_preds, -np.ones_like(proto_labels)],
                axis=0,
            )
            output_splits = np.concatenate(
                [
                    output_splits,
                    np.asarray(["prototype" for _ in range(int(valid.sum()))]),
                ],
                axis=0,
            )

    output_emb2d = _fit_tsne_np(
        output_features,
        tsne_cfg=tsne_cfg,
        logger=logger,
        name=f"task{task_id}_{tag}_output",
    )

    output_path = os.path.join(
        save_dir,
        f"tsne_task{task_id}_{tag}_output.png",
    )

    _plot_tsne_2d(
        emb2d=output_emb2d,
        labels=output_labels,
        preds=output_preds,
        splits=output_splits,
        class_names=class_names,
        save_path=output_path,
        title=f"Output embedding t-SNE | task={task_id} | {tag}",
        tsne_cfg=tsne_cfg,
        logger=logger,
    )


def save_cpe_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    proto_module: Optional[PrototypeMemory],
    path: str,
    epoch: int,
    task_id: int = 0,
    best_metric: Optional[float] = None,
    train_metrics: Optional[Dict[str, Any]] = None,
    val_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "prototypes": proto_module.state_dict() if proto_module is not None else None,
            "epoch": int(epoch),
            "task_id": int(task_id),
            "best_metric": best_metric,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def resolve_train_resume_path(config: EasyDict) -> Optional[str]:
    latest_path = os.path.join(config.save_path, "latest.pth")

    if bool(config.saver.get("auto_resume", False)) and os.path.exists(latest_path):
        return latest_path

    path = config.saver.get("resume_model", None)
    if not path:
        path = config.saver.get("load_path", None)

    if not path:
        return None

    path = str(path)
    if not os.path.isabs(path):
        path = os.path.join(config.exp_path, path)

    return path


def resolve_eval_path(args: argparse.Namespace, config: EasyDict) -> str:
    candidates = []

    if args.ckpt:
        candidates.append(args.ckpt)

    for key in ["eval_model", "resume_model", "load_path"]:
        value = config.saver.get(key, None)
        if value:
            candidates.append(value)

    candidates.append(os.path.join(config.save_path, "best.pth"))
    candidates.append(os.path.join(config.save_path, "latest.pth"))

    for path in candidates:
        path = str(path)
        if not os.path.isabs(path):
            path = os.path.join(config.exp_path, path)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "No checkpoint found for evaluation. Tried: {}".format(candidates)
    )


def metric_is_better(metric_name: str, current: float, best: Optional[float]) -> bool:
    metric_name = str(metric_name).lower()

    if best is None:
        return True

    if "loss" in metric_name:
        return current < best

    return current > best

@torch.no_grad()
def _adaptive_k_from_dispersion(
    z: torch.Tensor,
    k_max: int,
    min_samples_per_proto: int = 8,
    dispersion_step: float = 0.08,
) -> int:
    """
    Choose effective prototype number for one class.

    z: [N, D], normalized.
    """
    n = int(z.shape[0])

    if n <= 1:
        return 1

    center = F.normalize(z.mean(dim=0, keepdim=True), dim=1)
    avg_cos = torch.matmul(z, center.T).mean()
    dispersion = float((1.0 - avg_cos).clamp_min(0.0).item())

    k_by_dispersion = 1 + int(dispersion / float(dispersion_step))
    k_by_count = max(1, n // max(int(min_samples_per_proto), 1))

    k_eff = min(
        int(k_max),
        n,
        max(1, min(k_by_dispersion, k_by_count)),
    )

    return max(1, int(k_eff))


@torch.no_grad()
def _spherical_kmeans(
    z: torch.Tensor,
    k: int,
    num_iters: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    z: [N, D], normalized
    returns:
        centers: [k, D]
        counts:  [k]
    """
    z = F.normalize(z, dim=1)
    n, dim = z.shape
    k = min(int(k), int(n))

    if k <= 1:
        center = F.normalize(z.mean(dim=0, keepdim=True), dim=1)
        counts = torch.tensor(
            [float(n)],
            device=z.device,
            dtype=z.dtype,
        )
        return center, counts

    centers = []

    first = F.normalize(z.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    centers.append(first)

    # Farthest-point initialization on cosine sphere.
    for _ in range(1, k):
        cur = torch.stack(centers, dim=0)
        nearest_sim = torch.matmul(z, cur.T).max(dim=1).values
        farthest_idx = torch.argmin(nearest_sim)
        centers.append(z[farthest_idx])

    centers = F.normalize(torch.stack(centers, dim=0), dim=1)

    for _ in range(int(num_iters)):
        sim = torch.matmul(z, centers.T)
        assign = sim.argmax(dim=1)

        new_centers = []

        for j in range(k):
            idx = assign == j

            if bool(idx.any()):
                new_center = F.normalize(
                    z[idx].mean(dim=0, keepdim=True),
                    dim=1,
                ).squeeze(0)
            else:
                nearest_sim = torch.matmul(z, centers.T).max(dim=1).values
                farthest_idx = torch.argmin(nearest_sim)
                new_center = z[farthest_idx]

            new_centers.append(new_center)

        centers = F.normalize(torch.stack(new_centers, dim=0), dim=1)

    sim = torch.matmul(z, centers.T)
    assign = sim.argmax(dim=1)
    counts = torch.bincount(assign, minlength=k).to(dtype=z.dtype)

    return centers, counts

@torch.no_grad()
def rebuild_eval_multi_prototypes_from_train(
    train_loader: Any,
    model: nn.Module,
    proto_mod: AdaptiveMultiPrototypeMemory,
    cfg: EasyDict,
    logger: logging.Logger,
    task_id: int = 0,
) -> None:
    was_training = model.training
    model.eval()

    mp_cfg = EasyDict(cfg.cpe.get("multi_proto", {}))

    k_max = int(mp_cfg.get("num_prototypes", proto_mod.num_prototypes))
    kmeans_iters = int(mp_cfg.get("kmeans_iters", 12))
    min_samples_per_proto = int(mp_cfg.get("min_samples_per_proto", 8))
    dispersion_step = float(mp_cfg.get("dispersion_step", 0.08))

    all_z = []
    all_y = []

    for batch in train_loader:
        batch["task_id"] = task_id
        batch["is_train"] = False

        z = encode_once(model, batch)
        y = get_cls_targets(batch, device=z.device)

        all_z.append(F.normalize(z.detach(), dim=1))
        all_y.append(y.detach().view(-1).long())

    if len(all_z) == 0:
        raise RuntimeError("No samples found when rebuilding multi-prototypes.")

    all_z = torch.cat(all_z, dim=0)
    all_y = torch.cat(all_y, dim=0)

    proto_mod.reset()

    for cid in range(proto_mod.num_classes):
        idx = all_y == cid

        if not bool(idx.any()):
            continue

        z_cls = all_z[idx]
        z_cls = F.normalize(z_cls, dim=1)

        k_eff = _adaptive_k_from_dispersion(
            z=z_cls,
            k_max=k_max,
            min_samples_per_proto=min_samples_per_proto,
            dispersion_step=dispersion_step,
        )

        centers, counts = _spherical_kmeans(
            z=z_cls,
            k=k_eff,
            num_iters=kmeans_iters,
        )

        proto_mod.proto_multi[cid, :k_eff] = centers
        proto_mod.counts_multi[cid, :k_eff] = counts
        proto_mod.active_multi[cid, :k_eff] = True

    proto_mod.refresh_class_proto()

    logger.info(
        "[EvalMultiProto] rebuilt multi-prototypes. "
        "class_counts={} active_k={}".format(
            proto_mod.class_counts().detach().cpu().long().tolist(),
            proto_mod.active_mask().sum(dim=1).detach().cpu().long().tolist(),
        )
    )

    if was_training:
        set_cpe_train_mode(model, cfg)


def get_metric_value(
    metric_name: str,
    train_metrics: Optional[Dict[str, Any]],
    val_metrics: Optional[Dict[str, Any]],
) -> float:
    metric_name = str(metric_name).lower()

    aliases = {
        "acc": "acc1",
        "accuracy": "acc1",
        "overall_acc": "acc1",
        "mean_acc": "mean_class_acc",
        "mca": "mean_class_acc",
        "f1": "macro_f1",
    }
    key = aliases.get(metric_name, metric_name)

    source = val_metrics if val_metrics is not None else train_metrics
    if source is None:
        raise RuntimeError("No metric source is available.")

    if key not in source:
        raise KeyError(
            f"best_metric='{metric_name}' resolved to key='{key}', "
            f"but available metrics are {list(source.keys())}"
        )

    return float(source[key])

def log_class_mapping_check(
    train_loaders,
    val_loaders,
    cfg,
    logger,
):
    num_classes = int(cfg.cpe.num_classes)

    for task_id in range(max(len(train_loaders), len(val_loaders))):
        train_names = None
        val_names = None

        if task_id < len(train_loaders):
            train_names = infer_class_names_from_loader(
                train_loaders[task_id],
                num_classes,
            )
            logger.info(
                "[ClassMapping] train_task{}: {}".format(
                    task_id,
                    list(enumerate(train_names)),
                )
            )

        if task_id < len(val_loaders):
            val_names = infer_class_names_from_loader(
                val_loaders[task_id],
                num_classes,
            )
            logger.info(
                "[ClassMapping] val_task{}: {}".format(
                    task_id,
                    list(enumerate(val_names)),
                )
            )

        if train_names is not None and val_names is not None:
            if train_names != val_names:
                logger.info(
                    "[ClassMapping][WARNING] train/test class name order mismatch "
                    "at task {}. This can directly cause low test accuracy.".format(task_id)
                )
                raise RuntimeError(
                    "Train/test class mapping mismatch. Fix label_dict first."
                )


def main() -> None:
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    current_time = get_current_time()
    logger = create_logger(
        "global_logger",
        os.path.join(config.log_path, f"cpe_{current_time}.log"),
    )
    tb_logger = SummaryWriter(
        os.path.join(config.log_path, "events_cpe", current_time)
    )

    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: {}".format(pprint.pformat(config)))

    if not torch.cuda.is_available():
        raise RuntimeError("CPE training/eval requires CUDA because MinkowskiEngine is used.")

    random_seed = config.get("random_seed", 42)
    set_random_seed(random_seed, reproduce=config.get("reproduce", False))

    model = build_backbone_from_config(config.net[0])
    model.cuda()

    resume_path = resolve_train_resume_path(config)

    if not resume_path:
        pretrain_path = resolve_pretrain_path(config)

        load_pretrained_network_weights_only(
            model=model,
            path=pretrain_path,
            logger=logger,
            strict=bool(config.saver.get("pretrain_strict", False)),
            load_projection=bool(config.saver.get("pretrain_load_projection", True)),
        )
    else:
        logger.info(
            "[Resume] Local checkpoint is set: {}. Skip external pretrain initialization.".format(
                resume_path
            )
        )

    # 冻结策略必须在创建 optimizer 之前执行
    apply_cpe_finetune_policy(
        model=model,
        config=config,
        logger=logger,
    )

    optimizer = get_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        config.cpe,
    )


    supcon_crit = SupConLoss(
        temperature=float(config.cpe.temperature),
    ).cuda()

    mp_cfg = EasyDict(config.cpe.get("multi_proto", {}))

    proto_mod = AdaptiveMultiPrototypeMemory(
        num_classes=int(config.cpe.num_classes),
        dim=int(config.cpe.proj_dim),
        num_prototypes=int(mp_cfg.get("num_prototypes", 4)),
        momentum=float(config.cpe.proto_m),
        adaptive=bool(mp_cfg.get("adaptive", True)),
        new_proto_threshold=float(mp_cfg.get("new_proto_threshold", 0.55)),
    ).cuda()

    proto_crit = MultiPrototypeNCELoss(
        temperature=float(config.cpe.temperature),
        pool_tau=float(mp_cfg.get("pool_tau", 0.02)),
        logit_margin=float(mp_cfg.get("logit_margin", 0.10)),
        sample_margin=float(mp_cfg.get("sample_margin", 0.15)),
        hinge_weight=float(mp_cfg.get("hinge_weight", 0.30)),
        diversity_weight=float(mp_cfg.get("diversity_weight", 0.02)),
        stiefel_cos=float(mp_cfg.get("stiefel_cos", 0.35)),
        inter_cos=float(mp_cfg.get("inter_cos", 0.20)),
    ).cuda()


    train_loaders, val_loaders = build_dataloader(
        config.dataset,
        distributed=False,
    )

    log_class_mapping_check(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        cfg=config,
        logger=logger,
    )

    train_loaders = _as_list(train_loaders)
    val_loaders = _as_list(val_loaders)

    point_key = str(config.cpe.get("point_key", "pointcloud"))

    repeat_pad_cfg = EasyDict(config.cpe.get("repeat_pad", {}))
    repeat_pad_enabled = bool(repeat_pad_cfg.get("enabled", True))

    if repeat_pad_enabled:
        train_pad_indices = _resolve_loader_indices(
            repeat_pad_cfg,
            key="train_loader_indices",
            num_loaders=len(train_loaders),
            default_all=True,
        )

        val_pad_indices = _resolve_loader_indices(
            repeat_pad_cfg,
            key="val_loader_indices",
            num_loaders=len(val_loaders),
            default_all=True,
        )

        pad_train_loaders = [
            loader for idx, loader in enumerate(train_loaders)
            if idx in train_pad_indices
        ]

        pad_val_loaders = [
            loader for idx, loader in enumerate(val_loaders)
            if idx in val_pad_indices
        ]

        pad_loaders = pad_train_loaders + pad_val_loaders

        if len(pad_loaders) > 0:
            global_max_points, actual_point_key = infer_global_max_points_from_loaders(
                loaders=pad_loaders,
                point_key=point_key,
                logger=logger,
            )

            install_repeat_pad_collate_to_loaders(
                loaders=pad_loaders,
                target_num_points=global_max_points,
                point_key=actual_point_key,
                logger=logger,
            )

            logger.info(
                "[RepeatPadCollate] enabled for train_indices={} val_indices={} "
                "target_num_points={} point_key={}".format(
                    sorted(list(train_pad_indices)),
                    sorted(list(val_pad_indices)),
                    global_max_points,
                    actual_point_key,
                )
            )
        else:
            logger.info("[RepeatPadCollate] enabled=True but no loaders selected.")
    else:
        logger.info("[RepeatPadCollate] disabled by config.")



    if args.evaluate:
        ckpt_path = resolve_eval_path(args, config)
        restore_cpe_checkpoint(
            model=model,
            optimizer=None,
            proto_module=proto_mod,
            path=ckpt_path,
            logger=logger,
            load_optimizer=False,
        )

        if len(val_loaders) == 0:
            raise RuntimeError("No validation/test loader is available.")

        for task_id, val_loader_task in enumerate(val_loaders):
            train_loader_task = train_loaders[task_id] if task_id < len(train_loaders) else None

            if train_loader_task is None:
                raise RuntimeError(
                    f"No train loader found for task_id={task_id}. "
                    "Cannot rebuild eval multi-prototypes."
                )

            logger.info(f"proto counts = {proto_mod.counts.detach().cpu().long().tolist()}")
            logger.info(f"proto norms = {torch.norm(proto_mod.proto.detach(), dim=1).cpu().tolist()}")

            rebuild_eval_multi_prototypes_from_train(
                train_loader=train_loader_task,
                model=model,
                proto_mod=proto_mod,
                cfg=config,
                logger=logger,
                task_id=task_id,
            )

            validate(
                val_loader=val_loader_task,
                model=model,
                proto_mod=proto_mod,
                cfg=config,
                logger=logger,
                epoch=None,
                task_id=task_id,
                split_name=f"test_task{task_id}",
            )

            run_tsne_visualization(
                train_loader=train_loader_task,
                val_loader=val_loader_task,
                model=model,
                proto_mod=proto_mod,
                cfg=config,
                logger=logger,
                epoch=None,
                task_id=task_id,
            )

        logger.info("CPE evaluation done.")
        return

    resume_path = resolve_train_resume_path(config)
    if resume_path:
        start_epoch, restored_best = restore_cpe_checkpoint(
            model=model,
            optimizer=optimizer,
            proto_module=proto_mod,
            path=resume_path,
            logger=logger,
            load_optimizer=bool(config.saver.get("resume_load_optimizer", True)),
        )
    else:
        start_epoch, restored_best = 0, None


    best_metric_name = str(config.cpe.get("best_metric", "mean_class_acc"))
    default_best = None if restored_best is None else restored_best
    global_step_base = 0

    val_freq = int(config.cpe.get("val_frequent_epoch", config.cpe.get("val_freq_epoch", 1)))
    val_freq = max(val_freq, 1)

    for task_id, train_loader_task in enumerate(train_loaders):
        logger.info(f"Training CPE task {task_id}")
        
        fire_cfg = EasyDict(config.cpe.get("fire", {}))
        fire_enabled = bool(fire_cfg.get("enabled", False))

        apply_before_first_task = bool(
            fire_cfg.get("apply_before_first_task", False)
        )

        apply_at_task_boundary = bool(
            fire_cfg.get("apply_at_task_boundary", True)
        )

        should_apply_fire = False

        if fire_enabled:
            if task_id == 0 and apply_before_first_task and not resume_path:
                should_apply_fire = True

            if task_id > 0 and apply_at_task_boundary:
                should_apply_fire = True

        if should_apply_fire:
            logger.info(f"[FIRE] Applying FIRE before CPE task {task_id}")

            changed = fire_reinit_cpe(
                model=model,
                num_iters=int(fire_cfg.get("iter_num", 10)),
                only_trainable=bool(fire_cfg.get("only_trainable", True)),
                include_name_keywords=fire_cfg.get("include_name_keywords", None),
                sparse_kernel_layout=str(fire_cfg.get("sparse_kernel_layout", "auto")),
                logger=logger,
            )

            if changed <= 0:
                logger.warning(
                    "[FIRE] No parameter was changed. "
                    "Check include_name_keywords, trainable_prefixes, and parameter names."
                )

            if bool(fire_cfg.get("reset_optimizer", True)):
                optimizer = get_optimizer(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    config.cpe,
                )
                logger.info("[FIRE] optimizer reset after FIRE.")

            if bool(fire_cfg.get("reset_prototypes", True)):
                reset_prototype_memory(
                    proto_mod=proto_mod,
                    logger=logger,
                )

        val_loader_task = None
        if task_id < len(val_loaders):
            val_loader_task = val_loaders[task_id]

        best_metric = default_best

        for epoch in range(start_epoch, int(config.cpe.epochs)):
            train_metrics = train_one_epoch(
                train_loader=train_loader_task,
                model=model,
                optimizer=optimizer,
                supcon_crit=supcon_crit,
                proto_mod=proto_mod,
                proto_crit=proto_crit,
                cfg=config,
                epoch=epoch,
                logger=logger,
                tb_logger=tb_logger,
                global_step_base=global_step_base,
                task_id=task_id,
            )
            global_step_base += len(train_loader_task)

            logger.info(
                "[CPE Train Summary] epoch={}/{} loss={:.6f} "
                "supcon={:.6f} proto_nce={:.6f} proto_acc={:.6f} total_num={:.1f}".format(
                    epoch + 1,
                    int(config.cpe.epochs),
                    train_metrics["loss"],
                    train_metrics["supcon_loss"],
                    train_metrics["proto_nce_loss"],
                    train_metrics["proto_acc"],
                    train_metrics["total_num"],
                )
            )

            latest_path = os.path.join(config.save_path, "latest.pth")
            save_cpe_checkpoint(
                model=model,
                optimizer=optimizer,
                proto_module=proto_mod,
                path=latest_path,
                epoch=epoch,
                task_id=task_id,
                best_metric=best_metric,
                train_metrics=train_metrics,
                val_metrics=None,
            )

            save_freq = int(config.cpe.get("save_freq", 0))
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                epoch_path = os.path.join(config.save_path, f"ckpt_cpe_{epoch + 1}.pth")
                save_cpe_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    proto_module=proto_mod,
                    path=epoch_path,
                    epoch=epoch,
                    task_id=task_id,
                    best_metric=best_metric,
                    train_metrics=train_metrics,
                    val_metrics=None,
                )

            val_metrics = None
            if val_loader_task is not None and (epoch + 1) % val_freq == 0:
                rebuild_eval_multi_prototypes_from_train(
                    train_loader=train_loader_task,
                    model=model,
                    proto_mod=proto_mod,
                    cfg=config,
                    logger=logger,
                    task_id=task_id,
                )

                logger.info(f"proto counts = {proto_mod.counts.detach().cpu().long().tolist()}")
                logger.info(f"proto norms = {torch.norm(proto_mod.proto.detach(), dim=1).cpu().tolist()}")

                train_eval_metrics = validate(
                    val_loader=train_loader_task,
                    model=model,
                    proto_mod=proto_mod,
                    cfg=config,
                    logger=logger,
                    epoch=epoch,
                    task_id=task_id,
                    split_name=f"train_eval_task{task_id}",
                )

                if tb_logger is not None:
                    tb_logger.add_scalar("cpe_val/loss", train_eval_metrics["loss"], epoch + 1)
                    tb_logger.add_scalar("cpe_val/acc1", train_eval_metrics["acc1"], epoch + 1)
                    tb_logger.add_scalar(
                        "cpe_val/mean_class_acc",
                        train_eval_metrics["mean_class_acc"],
                        epoch + 1,
                    )
                    tb_logger.add_scalar("cpe_val/macro_f1", train_eval_metrics["macro_f1"], epoch + 1)
                    tb_logger.flush()

                # current_metric = get_metric_value(
                #     best_metric_name,
                #     train_metrics=train_metrics,
                #     val_metrics=train_eval_metrics,
                # )

                # if metric_is_better(best_metric_name, current_metric, best_metric):
                #     best_metric = current_metric
                #     best_path = os.path.join(config.save_path, "best.pth")
                #     save_cpe_checkpoint(
                #         model=model,
                #         optimizer=optimizer,
                #         proto_module=proto_mod,
                #         path=best_path,
                #         epoch=epoch,
                #         task_id=task_id,
                #         best_metric=best_metric,
                #         train_metrics=train_metrics,
                #         val_metrics=train_eval_metrics,
                #     )
                #     logger.info(
                #         "[Best CPE] epoch={} {}={:.6f} path={}".format(
                #             epoch + 1,
                #             best_metric_name,
                #             best_metric,
                #             best_path,
                #         )
                #     )


                val_metrics = validate(
                    val_loader=val_loader_task,
                    model=model,
                    proto_mod=proto_mod,
                    cfg=config,
                    logger=logger,
                    epoch=epoch,
                    task_id=task_id,
                    split_name=f"test_task{task_id}",
                )

                run_tsne_visualization(
                    train_loader=train_loader_task,
                    val_loader=val_loader_task,
                    model=model,
                    proto_mod=proto_mod,
                    cfg=config,
                    logger=logger,
                    epoch=epoch,
                    task_id=task_id,
                )

                if tb_logger is not None:
                    tb_logger.add_scalar("cpe_val/loss", val_metrics["loss"], epoch + 1)
                    tb_logger.add_scalar("cpe_val/acc1", val_metrics["acc1"], epoch + 1)
                    tb_logger.add_scalar(
                        "cpe_val/mean_class_acc",
                        val_metrics["mean_class_acc"],
                        epoch + 1,
                    )
                    tb_logger.add_scalar("cpe_val/macro_f1", val_metrics["macro_f1"], epoch + 1)
                    tb_logger.flush()

                current_metric = get_metric_value(
                    best_metric_name,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )

                if metric_is_better(best_metric_name, current_metric, best_metric):
                    best_metric = current_metric
                    best_path = os.path.join(config.save_path, "best.pth")
                    save_cpe_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        proto_module=proto_mod,
                        path=best_path,
                        epoch=epoch,
                        task_id=task_id,
                        best_metric=best_metric,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                    )
                    logger.info(
                        "[Best CPE] epoch={} {}={:.6f} path={}".format(
                            epoch + 1,
                            best_metric_name,
                            best_metric,
                            best_path,
                        )
                    )

        run_ad_after_cpe_task_if_needed(
            cpe_model=model,
            cpe_cfg=config,
            cpe_train_loaders=train_loaders,
            cpe_task_id=task_id,
            proto_mod=proto_mod,
            logger=logger,
        )

    logger.info("CPE training done.")


if __name__ == "__main__":
    main()
