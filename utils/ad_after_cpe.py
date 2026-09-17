# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

from datasets.data_builder import build_dataloader
from models.model_helper import ModelHelper
from utils.criterion_helper import build_criterion
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import AverageMeter, get_current_time
from utils.optimizer_helper import get_optimizer as get_ad_optimizer

_AD_GATE_FEATURE_KEYS = [
    "feature_align",
    "features",
    "feature",
    "x",
    "tokens",
    "token",
    "token_features",
    "pointmae_feature",
    "pointmae_features",
    "xyz_features",
    "raw_xyz_features",
    "proto_feature",
    "raw_proto_feature",
    "global_feature",
    "cls_feature",
]


def _get_reconstruction_kwargs(ad_cfg: EasyDict) -> Any:
    for module_cfg in _as_list(ad_cfg.net):
        if str(_cfg_get(module_cfg, "name", "")) != "reconstruction":
            continue
        return _cfg_get(module_cfg, "kwargs", {})
    return {}


def _resolve_assignment_feature_keys(ad_cfg: EasyDict) -> List[str]:
    kwargs = _get_reconstruction_kwargs(ad_cfg)
    keys = _cfg_get(kwargs, "gate_feature_keys", None)

    ret = []
    for key in _as_list(keys) + _AD_GATE_FEATURE_KEYS:
        key = str(key)
        if key not in ret:
            ret.append(key)

    return ret


def _resolve_assignment_feature_dim(ad_cfg: EasyDict) -> Optional[int]:
    kwargs = _get_reconstruction_kwargs(ad_cfg)
    value = _cfg_get(kwargs, "gate_feature_dim", None)

    if value is None:
        return None

    value = int(value)

    if value <= 0:
        return None

    return value


def _find_tensor_by_keys_local(x: Any, keys: List[str]) -> Optional[torch.Tensor]:
    if not isinstance(x, dict):
        return None

    for key in keys:
        if key in x and torch.is_tensor(x[key]):
            return x[key]

    for value in x.values():
        if isinstance(value, dict):
            found = _find_tensor_by_keys_local(value, keys)
            if found is not None:
                return found

    return None


def _as_btn_feature_for_assignment(
    feat: torch.Tensor,
    feature_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Return [B, T, C].
    """
    if feat.dim() == 2:
        return feat.contiguous().unsqueeze(1)

    if feat.dim() != 3:
        raise RuntimeError(
            f"Assignment feature should be [B,C], [B,T,C], or [B,C,T], "
            f"got {tuple(feat.shape)}."
        )

    if feature_dim is not None and int(feature_dim) > 0:
        feature_dim = int(feature_dim)

        if int(feat.shape[-1]) == feature_dim:
            return feat.contiguous()

        if int(feat.shape[1]) == feature_dim:
            return feat.transpose(1, 2).contiguous()

        raise RuntimeError(
            f"Cannot infer assignment feature layout. "
            f"feature_dim={feature_dim}, shape={tuple(feat.shape)}."
        )

    # fallback heuristic
    if int(feat.shape[1]) > int(feat.shape[2]):
        feat = feat.transpose(1, 2)

    return feat.contiguous()


def _feature_to_assignment_vector(
    feat: torch.Tensor,
    feature_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert token feature to one vector per sample.

    feat: [B,C], [B,T,C], or [B,C,T]
    return: [B,C]
    """
    feat = _as_btn_feature_for_assignment(
        feat=feat,
        feature_dim=feature_dim,
    )

    vec = feat.float().mean(dim=1)
    vec = torch.nan_to_num(vec, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    vec = F.normalize(vec, dim=1)

    return vec


def _select_ad_assignment_loader_items(
    ad_train_loaders: Any,
    cpe_task_id: int,
    scope: str,
) -> List[Tuple[int, Any]]:
    loaders = _as_list(ad_train_loaders)
    cpe_task_id = int(cpe_task_id)
    scope = str(scope).lower()

    if len(loaders) == 0:
        raise RuntimeError("[MoE Assign] no AD train loaders.")

    if scope in ["current", "cur"]:
        if cpe_task_id >= len(loaders):
            raise RuntimeError(
                f"[MoE Assign] cpe_task_id={cpe_task_id} out of range "
                f"for AD train loaders len={len(loaders)}."
            )
        return [(cpe_task_id, loaders[cpe_task_id])]

    if scope in ["seen", "seen_classes"]:
        end = min(cpe_task_id + 1, len(loaders))
        return [(idx, loaders[idx]) for idx in range(end)]

    if scope in ["all", "full"]:
        return [(idx, loader) for idx, loader in enumerate(loaders)]

    raise ValueError(
        f"Unsupported moe.assignment_scope={scope}. "
        "Use current, seen, or all."
    )


@torch.no_grad()
def build_ad_feature_class_prototypes(
    ad_model: nn.Module,
    loader_items: List[Tuple[int, Any]],
    ad_cfg: EasyDict,
    cpe_cfg: EasyDict,
    num_classes: int,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build class prototypes in PointMAE / UniAD feature space.

    This is used only to decide class -> expert assignment.
    """
    raw_model = unwrap_model(ad_model)

    if not hasattr(raw_model, "backbone"):
        raise RuntimeError("[MoE Assign] AD model has no backbone.")

    was_training = ad_model.training
    ad_model.eval()

    device = next(ad_model.parameters()).device

    moe_class_key = _resolve_moe_class_key(
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
    )
    moe_point_key = _resolve_moe_point_key(
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
    )
    allow_raw_point_fallback = _resolve_moe_allow_raw_point_fallback(
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
    )

    feature_keys = _resolve_assignment_feature_keys(ad_cfg)
    feature_dim = _resolve_assignment_feature_dim(ad_cfg)

    sums = None
    counts = torch.zeros(
        int(num_classes),
        device=device,
        dtype=torch.float32,
    )

    for source_task_id, loader in loader_items:
        logger.info(
            "[MoE Assign] scanning AD train source_task={} for feature prototypes".format(
                source_task_id
            )
        )

        for batch in loader:
            batch = dict(batch)

            batch = inject_gt_registration_clsname(
                batch=batch,
                moe_class_key=moe_class_key,
            )
            batch["task_id"] = int(source_task_id)

            backbone_out = raw_model.backbone(batch)

            backbone_out = _copy_ad_context_to_backbone_output(
                backbone_out=backbone_out,
                batch=batch,
                class_key=moe_class_key,
                moe_point_key=moe_point_key,
                allow_raw_point_fallback=allow_raw_point_fallback,
            )

            feat = _find_tensor_by_keys_local(
                backbone_out,
                feature_keys,
            )

            # 如果 backbone_out 没有直接暴露 feature，
            # 就从 shared UniAD 的 feature_align 里取。
            # 这通常就是 UniAD 正在重构的 PointMAE feature。
            if feat is None:
                reconstruction = getattr(raw_model, "reconstruction", None)

                if reconstruction is not None and hasattr(reconstruction, "shared_expert"):
                    shared_out = reconstruction.shared_expert(backbone_out)

                    feat = _find_tensor_by_keys_local(
                        shared_out,
                        feature_keys + ["feature_align"],
                    )

            if feat is None:
                raise RuntimeError(
                    "[MoE Assign] cannot find PointMAE feature. "
                    f"Expected one of keys={feature_keys}. "
                    "Please print backbone_out.keys() and add the real key "
                    "to reconstruction.kwargs.gate_feature_keys."
                )

            vec = _feature_to_assignment_vector(
                feat=feat,
                feature_dim=feature_dim,
            ).to(device=device)

            y = get_cls_targets(
                batch=batch,
                device=device,
                class_to_idx=None,
            ).view(-1).long()

            if int(y.numel()) == 1 and int(vec.shape[0]) > 1:
                y = y.expand(int(vec.shape[0]))

            if int(y.numel()) != int(vec.shape[0]):
                raise RuntimeError(
                    "[MoE Assign] feature batch size {} != label batch size {}".format(
                        int(vec.shape[0]),
                        int(y.numel()),
                    )
                )

            valid = (y >= 0) & (y < int(num_classes))

            if not bool(valid.all()):
                bad = y[~valid].detach().cpu().tolist()
                raise RuntimeError(
                    f"[MoE Assign] labels out of range: {bad}, "
                    f"num_classes={num_classes}"
                )

            if sums is None:
                sums = torch.zeros(
                    int(num_classes),
                    int(vec.shape[1]),
                    device=device,
                    dtype=torch.float32,
                )

            sums.index_add_(0, y, vec.detach().float())

            ones = torch.ones(
                int(y.numel()),
                device=device,
                dtype=torch.float32,
            )
            counts.index_add_(0, y, ones)

    if sums is None:
        raise RuntimeError("[MoE Assign] no features collected.")

    prototypes = torch.zeros_like(sums)
    valid_class = counts > 0

    if bool(valid_class.any()):
        prototypes[valid_class] = F.normalize(
            sums[valid_class] / counts[valid_class].view(-1, 1).clamp_min(1.0),
            dim=1,
        )

    logger.info(
        "[MoE Assign] class feature counts={}".format(
            counts.detach().cpu().long().tolist()
        )
    )

    if was_training:
        ad_model.train()
    else:
        ad_model.eval()

    return prototypes.detach(), counts.detach()

def _farthest_order(x: torch.Tensor) -> List[int]:
    """
    x: [N,D], normalized.
    return local indices ordered by farthest-point traversal.
    """
    n = int(x.shape[0])

    if n == 0:
        return []

    if n == 1:
        return [0]

    sim = torch.matmul(x, x.T)
    dist = 1.0 - sim

    first = int(torch.argmax(dist.mean(dim=1)).item())

    order = [first]
    selected = torch.zeros(n, device=x.device, dtype=torch.bool)
    selected[first] = True

    while len(order) < n:
        selected_idx = torch.tensor(order, device=x.device, dtype=torch.long)
        min_dist = dist[:, selected_idx].min(dim=1).values
        min_dist[selected] = -1.0

        nxt = int(torch.argmax(min_dist).item())
        order.append(nxt)
        selected[nxt] = True

    return order


def assign_classes_similar_cluster(
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    num_experts: int,
    kmeans_iters: int = 50,
) -> torch.Tensor:
    """
    Similar scheme:
      feature-similar classes -> same expert.

    Uses cosine k-means over class prototypes.
    """
    device = prototypes.device
    num_classes = int(prototypes.shape[0])
    num_experts = int(num_experts)

    mapping = torch.full(
        (num_classes,),
        -1,
        device=device,
        dtype=torch.long,
    )

    valid_idx = torch.nonzero(
        counts > 0,
        as_tuple=False,
    ).view(-1)

    if int(valid_idx.numel()) == 0:
        return mapping.cpu()

    x = prototypes[valid_idx].float()
    x = F.normalize(x, dim=1)

    n = int(x.shape[0])
    k = min(num_experts, n)

    order = _farthest_order(x)
    centers = x[torch.tensor(order[:k], device=device, dtype=torch.long)].clone()

    labels = torch.zeros(n, device=device, dtype=torch.long)

    for _ in range(int(kmeans_iters)):
        sim = torch.matmul(x, centers.T)
        labels = sim.argmax(dim=1)

        new_centers = []

        for e in range(k):
            mask = labels == e

            if bool(mask.any()):
                center = x[mask].mean(dim=0)
                center = F.normalize(center, dim=0)
            else:
                # Reinitialize empty cluster with worst represented class.
                max_sim = sim.max(dim=1).values
                center = x[int(torch.argmin(max_sim).item())]

            new_centers.append(center)

        centers = torch.stack(new_centers, dim=0)

    mapping[valid_idx] = labels
    return mapping.cpu()


def assign_classes_diverse_group(
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Diverse scheme:
      feature-different classes -> same expert.

    Greedy balanced grouping:
      - choose farthest seeds
      - then put each remaining class into the expert whose current members
        are most dissimilar to it
    """
    device = prototypes.device
    num_classes = int(prototypes.shape[0])
    num_experts = int(num_experts)

    mapping = torch.full(
        (num_classes,),
        -1,
        device=device,
        dtype=torch.long,
    )

    valid_idx = torch.nonzero(
        counts > 0,
        as_tuple=False,
    ).view(-1)

    if int(valid_idx.numel()) == 0:
        return mapping.cpu()

    x = prototypes[valid_idx].float()
    x = F.normalize(x, dim=1)

    n = int(x.shape[0])
    k = min(num_experts, n)

    order = _farthest_order(x)

    local_labels = torch.full(
        (n,),
        -1,
        device=device,
        dtype=torch.long,
    )

    groups: List[List[int]] = [[] for _ in range(k)]
    max_cap = int((n + k - 1) // k)

    # Seed each expert with one far-apart class.
    for e, local_c in enumerate(order[:k]):
        groups[e].append(int(local_c))
        local_labels[int(local_c)] = int(e)

    # Assign remaining classes to the most dissimilar group.
    for local_c in order[k:]:
        local_c = int(local_c)

        best_e = None
        best_score = None

        for e in range(k):
            if len(groups[e]) >= max_cap:
                continue

            member_idx = torch.tensor(
                groups[e],
                device=device,
                dtype=torch.long,
            )

            sim = torch.matmul(
                x[local_c:local_c + 1],
                x[member_idx].T,
            ).mean()

            dist = 1.0 - sim

            # tiny term prefers smaller groups on ties
            score = float(dist.item()) - 1.0e-3 * float(len(groups[e]))

            if best_score is None or score > best_score:
                best_score = score
                best_e = e

        if best_e is None:
            sizes = [len(g) for g in groups]
            best_e = int(np.argmin(sizes))

        groups[best_e].append(local_c)
        local_labels[local_c] = int(best_e)

    mapping[valid_idx] = local_labels
    return mapping.cpu()


def _format_class_to_expert(
    mapping: torch.Tensor,
    class_names: List[str],
    num_experts: int,
) -> Dict[int, List[str]]:
    mapping = mapping.view(-1).cpu().long()

    groups: Dict[int, List[str]] = {
        int(e): []
        for e in range(int(num_experts))
    }

    for c, e in enumerate(mapping.tolist()):
        if e < 0:
            continue

        name = str(class_names[c]) if c < len(class_names) else f"class_{c}"
        groups[int(e)].append(name)

    return groups

def configure_moe_class_assignment_if_needed(
    ad_model: nn.Module,
    ad_train_loaders: Any,
    ad_cfg: EasyDict,
    cpe_cfg: EasyDict,
    cpe_task_id: int,
    class_names: List[str],
    logger: logging.Logger,
) -> None:
    """
    Configure class_to_expert for UniADMoE.

    ad_cfg.moe.assignment_mode:
      - none
      - similar_cluster
      - diverse_group
    """
    moe_cfg = EasyDict(ad_cfg.get("moe", {}))

    mode = str(moe_cfg.get("assignment_mode", "none")).lower()

    if mode in ["none", "off", "disable", "disabled", ""]:
        return

    raw_model = unwrap_model(ad_model)
    reconstruction = getattr(raw_model, "reconstruction", None)

    if reconstruction is None or not hasattr(reconstruction, "set_class_to_expert"):
        raise RuntimeError(
            "[MoE Assign] reconstruction does not support set_class_to_expert(). "
            "Please make sure reconstruction is UniADMoE with the patched code."
        )

    num_classes = int(len(class_names))
    num_experts = int(getattr(reconstruction, "num_experts"))

    scope = str(moe_cfg.get("assignment_scope", "seen")).lower()

    loader_items = _select_ad_assignment_loader_items(
        ad_train_loaders=ad_train_loaders,
        cpe_task_id=cpe_task_id,
        scope=scope,
    )

    prototypes, counts = build_ad_feature_class_prototypes(
        ad_model=ad_model,
        loader_items=loader_items,
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
        num_classes=num_classes,
        logger=logger,
    )

    if mode in ["similar", "similar_cluster", "cluster", "kmeans"]:
        mapping = assign_classes_similar_cluster(
            prototypes=prototypes,
            counts=counts,
            num_experts=num_experts,
        )

    elif mode in ["diverse", "diverse_group", "dissimilar"]:
        mapping = assign_classes_diverse_group(
            prototypes=prototypes,
            counts=counts,
            num_experts=num_experts,
        )

    else:
        raise ValueError(
            f"Unsupported ad_cfg.moe.assignment_mode={mode}. "
            "Use none, similar_cluster, or diverse_group."
        )

    reconstruction.set_class_to_expert(mapping)

    groups = _format_class_to_expert(
        mapping=mapping,
        class_names=class_names,
        num_experts=num_experts,
    )

    logger.info(
        "[MoE Assign] mode={} scope={} class_to_expert={}".format(
            mode,
            scope,
            mapping.view(-1).cpu().long().tolist(),
        )
    )
    logger.info(
        "[MoE Assign] expert groups={}".format(groups)
    )


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _save_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": None,
        "numpy": np.random.get_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if state.get("python", None) is not None:
        random.setstate(state["python"])

    if state.get("torch_cpu", None) is not None:
        torch.random.set_rng_state(state["torch_cpu"])

    if torch.cuda.is_available() and state.get("torch_cuda", None) is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    if state.get("numpy", None) is not None:
        np.random.set_state(state["numpy"])


def _resolve_config_path(cpe_cfg: EasyDict, path: Any) -> str:
    path = str(path)

    if os.path.isabs(path):
        return path

    base = str(getattr(cpe_cfg, "exp_path", "."))
    return os.path.normpath(os.path.join(base, path))


def _path_under_exp(exp_path: str, path: Any, *extra: str) -> str:
    path = str(path)

    if os.path.isabs(path):
        return os.path.join(path, *extra)

    return os.path.join(exp_path, path, *extra)


def normalize_cls_name(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")

    if isinstance(x, np.bytes_):
        return x.decode("utf-8")

    if isinstance(x, np.str_):
        return str(x)

    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.detach().cpu().item())
        return str(x.detach().cpu().tolist())

    return str(x)


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
    dataset = getattr(loader, "dataset", None)

    names: List[Optional[str]] = [None for _ in range(num_classes)]

    label_dict = _get_nested_dataset_attr(dataset, "label_dict")
    if isinstance(label_dict, dict):
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
                if label is None:
                    continue

                idx_int = int(label)
                if 0 <= idx_int < num_classes and names[idx_int] is None:
                    names[idx_int] = str(item["clsname"])

    for idx in range(num_classes):
        if names[idx] is None:
            names[idx] = f"class_{idx}"

    return [str(x) for x in names]


def infer_class_names_from_loaders(loaders: Any, num_classes: int) -> List[str]:
    loaders = _as_list(loaders)
    names: List[Optional[str]] = [None for _ in range(num_classes)]

    for loader in loaders:
        if loader is None:
            continue

        cur_names = infer_class_names_from_loader(loader, num_classes)

        for idx, name in enumerate(cur_names):
            if idx >= num_classes:
                continue

            name = str(name)

            if names[idx] is None:
                names[idx] = name
            elif names[idx] == f"class_{idx}" and name != f"class_{idx}":
                names[idx] = name

    for idx in range(num_classes):
        if names[idx] is None:
            names[idx] = f"class_{idx}"

    return [str(x) for x in names]


def resolve_class_names(
    cpe_cfg: EasyDict,
    cpe_train_loaders: Any,
    ad_train_loaders: Any,
    ad_val_loaders: Any,
) -> List[str]:
    num_classes = int(cpe_cfg.cpe.num_classes)

    if "class_names" in cpe_cfg.cpe and cpe_cfg.cpe.class_names is not None:
        class_names = [str(x) for x in cpe_cfg.cpe.class_names]

        if len(class_names) != num_classes:
            raise RuntimeError(
                f"len(cpe.class_names)={len(class_names)} but cpe.num_classes={num_classes}"
            )

        return class_names

    # 优先从 CPE 数据集推断，因为 CPE prototype 的 label 顺序必须和这里一致。
    class_names = infer_class_names_from_loaders(cpe_train_loaders, num_classes)

    # 如果 CPE 侧推断失败，再用 AD loader 补。
    if all(class_names[idx] == f"class_{idx}" for idx in range(num_classes)):
        class_names = infer_class_names_from_loaders(
            _as_list(ad_train_loaders) + _as_list(ad_val_loaders),
            num_classes,
        )

    return class_names


def get_cls_targets(
    batch: Dict[str, Any],
    device: Optional[torch.device] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    for label_key in ["cls_label", "class_label", "category_id"]:
        if label_key not in batch:
            continue

        value = batch[label_key]

        if torch.is_tensor(value):
            target = value.view(-1).long()
        elif isinstance(value, (list, tuple)):
            target = torch.tensor(value, dtype=torch.long).view(-1)
        else:
            target = torch.tensor([value], dtype=torch.long)

        if device is not None:
            target = target.to(device=device, dtype=torch.long)

        return target

    if class_to_idx is None:
        raise KeyError(
            "Cannot build classification target. "
            "Expected cls_label/class_label/category_id, "
            "or provide class_to_idx for clsname fallback."
        )

    name_key = None
    for key in ["clsname", "class_name", "category", "class"]:
        if key in batch:
            name_key = key
            break

    if name_key is None:
        raise KeyError(
            "Cannot build classification target. "
            "Expected one of cls_label/class_label/category_id/clsname."
        )

    names_raw = batch[name_key]

    if isinstance(names_raw, (str, bytes, np.str_, np.bytes_)):
        names_raw = [names_raw]

    labels = []
    for item in names_raw:
        name = normalize_cls_name(item)

        if name not in class_to_idx:
            raise KeyError(
                f"Class name '{name}' not found in class_to_idx. "
                f"Available={list(class_to_idx.keys())}"
            )

        labels.append(int(class_to_idx[name]))

    target = torch.tensor(labels, dtype=torch.long)

    if device is not None:
        target = target.to(device=device, dtype=torch.long)

    return target


@torch.no_grad()
def encode_cpe_once(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    raw_model = unwrap_model(model)

    if hasattr(raw_model, "forward_embed"):
        z = raw_model.forward_embed(batch)
        return F.normalize(z, dim=1)

    if hasattr(raw_model, "forward_features"):
        features, _ = raw_model.forward_features(batch["pointcloud"])
        z = features.squeeze(-1)
        return F.normalize(z, dim=1)

    outputs = raw_model(batch)

    for key in ["proto_feature", "raw_proto_feature", "xyz_features", "raw_xyz_features"]:
        if key not in outputs:
            continue

        z = outputs[key]

        if z.dim() == 3:
            z = z.mean(dim=-1)

        return F.normalize(z, dim=1)

    raise KeyError(
        "Cannot get CPE embedding. Expected forward_embed(), "
        "forward_features(), or output feature keys."
    )


def _select_cpe_proto_loader_items(
    cpe_train_loaders: Any,
    cpe_task_id: int,
    scope: str,
) -> List[Tuple[int, Any]]:
    train_loaders = _as_list(cpe_train_loaders)
    cpe_task_id = int(cpe_task_id)
    scope = str(scope).lower()

    if len(train_loaders) == 0:
        raise RuntimeError("[CPE Eval Proto] cpe_train_loaders is empty.")

    if scope in ["current", "cur"]:
        if cpe_task_id >= len(train_loaders):
            raise RuntimeError(
                f"cpe_task_id={cpe_task_id} out of range for train_loaders len={len(train_loaders)}"
            )
        return [(cpe_task_id, train_loaders[cpe_task_id])]

    if scope in ["seen", "seen_classes"]:
        end = min(cpe_task_id + 1, len(train_loaders))
        return [(idx, train_loaders[idx]) for idx in range(end)]

    if scope in ["all", "full"]:
        return [(idx, loader) for idx, loader in enumerate(train_loaders)]

    raise ValueError(
        f"Unsupported ad_after_cpe.prototype_scope={scope}. "
        "Use current, seen, or all."
    )


@torch.no_grad()
def build_cpe_eval_prototype_bank(
    cpe_model: nn.Module,
    cpe_cfg: EasyDict,
    loader_items: List[Tuple[int, Any]],
    current_cpe_task_id: int,
    proto_mod: nn.Module,
    class_names: List[str],
    logger: logging.Logger,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(loader_items) == 0:
        raise RuntimeError("[CPE Eval Proto] loader_items is empty.")

    was_training = cpe_model.training
    cpe_model.eval()

    proto_ref = proto_mod.proto
    device = proto_ref.device
    num_classes = int(proto_ref.shape[0])
    dim = int(proto_ref.shape[1])

    class_to_idx = {str(name): idx for idx, name in enumerate(class_names)}

    sums = torch.zeros(num_classes, dim, device=device, dtype=torch.float32)
    counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

    hook_cfg = EasyDict(cpe_cfg.get("ad_after_cpe", {}))
    task_id_policy = str(hook_cfg.get("prototype_task_id_policy", "source")).lower()

    for source_task_id, loader in loader_items:
        logger.info(
            "[CPE Eval Proto] scanning source_task={} current_cpe_task={}".format(
                source_task_id,
                current_cpe_task_id,
            )
        )

        for batch in loader:
            batch = dict(batch)
            batch["is_train"] = False

            if task_id_policy == "source":
                batch["task_id"] = int(source_task_id)
            else:
                batch["task_id"] = int(current_cpe_task_id)

            z = encode_cpe_once(cpe_model, batch)
            y = get_cls_targets(
                batch,
                device=z.device,
                class_to_idx=class_to_idx,
            ).view(-1).long()

            if int(z.shape[0]) != int(y.shape[0]):
                raise RuntimeError(
                    "[CPE Eval Proto] embedding batch size {} != label batch size {}".format(
                        int(z.shape[0]),
                        int(y.shape[0]),
                    )
                )

            valid = (y >= 0) & (y < num_classes)
            if not bool(valid.all()):
                bad = y[~valid].detach().cpu().tolist()
                raise RuntimeError(
                    f"[CPE Eval Proto] labels out of range: {bad}, "
                    f"num_classes={num_classes}"
                )

            z = F.normalize(z, dim=1)

            sums.index_add_(
                0,
                y,
                z.detach().to(device=device, dtype=torch.float32),
            )

            ones = torch.ones(y.shape[0], device=device, dtype=torch.float32)
            counts.index_add_(0, y, ones)

    proto_bank = torch.zeros_like(sums)
    valid_class = counts > 0

    if bool(valid_class.any()):
        proto_bank[valid_class] = F.normalize(
            sums[valid_class] / counts[valid_class].view(-1, 1).clamp_min(1.0),
            dim=1,
        )

    logger.info(
        "[CPE Eval Proto] rebuilt from tasks={} | counts={}".format(
            [int(x[0]) for x in loader_items],
            counts.detach().cpu().long().tolist(),
        )
    )

    zero_classes = torch.nonzero(
        counts <= 0,
        as_tuple=False,
    ).view(-1).detach().cpu().tolist()

    if len(zero_classes) > 0:
        logger.info(
            "[CPE Eval Proto] classes without train samples yet: {}".format(
                zero_classes
            )
        )

    if was_training:
        cpe_model.train()
    else:
        cpe_model.eval()

    return proto_bank.detach(), counts.detach()


@torch.no_grad()
def inject_cpe_pred_registration_clsname(
    batch: Dict[str, Any],
    cpe_model: nn.Module,
    cpe_cfg: EasyDict,
    current_cpe_task_id: int,
    source_task_id: int,
    proto_bank: torch.Tensor,
    proto_counts: torch.Tensor,
    class_names: List[str],
    moe_class_key: str = "moe_class_id",
) -> Dict[str, Any]:

    batch = dict(batch)

    cpe_model.eval()

    cpe_batch = dict(batch)
    cpe_batch["is_train"] = False

    hook_cfg = EasyDict(cpe_cfg.get("ad_after_cpe", {}))
    task_id_policy = str(hook_cfg.get("cpe_pred_task_id_policy", "current")).lower()

    if task_id_policy == "source":
        cpe_batch["task_id"] = int(source_task_id)
    else:
        cpe_batch["task_id"] = int(current_cpe_task_id)

    z = encode_cpe_once(cpe_model, cpe_batch)

    prototypes = F.normalize(proto_bank.to(z.device), dim=1)
    valid_proto = proto_counts.to(z.device) > 0

    if not bool(valid_proto.any()):
        raise RuntimeError("[CPE Predict] no valid prototypes.")

    logits = torch.matmul(z, prototypes.T) / float(cpe_cfg.cpe.temperature)
    logits[:, ~valid_proto] = -1.0e9

    pred_label = logits.argmax(dim=1).detach().cpu().long()
    pred_names = [str(class_names[int(idx)]) for idx in pred_label.tolist()]

    batch["pred_cls_label"] = pred_label
    batch["registration_cls_label"] = pred_label
    batch["pred_clsname"] = pred_names
    batch["registration_clsname"] = pred_names

    batch = inject_moe_class_id(
        batch=batch,
        class_id=pred_label,
        class_key=moe_class_key,
    )


    # AD test 时禁止 PointMAE 通过 GT 类别配准。
    batch["is_train"] = False
    batch["use_gt_registration"] = False
    batch["force_gt_registration"] = False

    return batch


def inject_gt_registration_clsname(
    batch: Dict[str, Any],
    moe_class_key: str = "moe_class_id",
) -> Dict[str, Any]:

    batch = dict(batch)

    has_explicit_key = any(
        key in batch
        for key in [
            "registration_clsname",
            "registration_class_name",
            "registration_cls_label",
            "pred_clsname",
            "pred_cls_label",
        ]
    )

    if not has_explicit_key:
        for name_key in ["clsname", "class_name", "category", "class"]:
            if name_key in batch:
                batch["registration_clsname"] = batch[name_key]
                break

    if "registration_clsname" not in batch and "registration_cls_label" not in batch:
        for label_key in ["cls_label", "class_label", "category_id"]:
            if label_key in batch:
                batch["registration_cls_label"] = batch[label_key]
                break

    batch["is_train"] = True
    batch["use_gt_registration"] = True
    batch["force_gt_registration"] = True

    gt_cls_label = get_cls_targets(
        batch=batch,
        device=None,
        class_to_idx=None,
    ).view(-1).long()

    batch = inject_moe_class_id(
        batch=batch,
        class_id=gt_cls_label,
        class_key=moe_class_key,
    )


    return batch


def _batch_size_from_batch(batch: Dict[str, Any]) -> int:
    for key in ["filename", "cls_label", "label", "pointcloud"]:
        if key not in batch:
            continue

        value = batch[key]

        if torch.is_tensor(value):
            if value.dim() == 0:
                return 1
            return int(value.shape[0])

        if isinstance(value, (list, tuple)):
            return len(value)

        if hasattr(value, "shape"):
            try:
                if len(value.shape) == 0:
                    return 1
                return int(value.shape[0])
            except Exception:
                pass

    return 1

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def _resolve_moe_class_key(
    ad_cfg: Optional[EasyDict] = None,
    cpe_cfg: Optional[EasyDict] = None,
) -> str:
    """
    Resolve which batch key is used for MoE routing.

    Priority:
      1. config_ad_after_cpe.yaml top-level moe.class_key
      2. config_ad_after_cpe.yaml net reconstruction kwargs class_key
      3. CPE config ad_after_cpe.moe.class_key
      4. default: moe_class_id
    """

    moe_cfg = _cfg_get(ad_cfg, "moe", None)
    if _cfg_get(moe_cfg, "class_key", None):
        return str(_cfg_get(moe_cfg, "class_key"))

    net_cfg = _cfg_get(ad_cfg, "net", [])
    for module_cfg in _as_list(net_cfg):
        if str(_cfg_get(module_cfg, "name", "")) != "reconstruction":
            continue

        kwargs = _cfg_get(module_cfg, "kwargs", {})
        if _cfg_get(kwargs, "class_key", None):
            return str(_cfg_get(kwargs, "class_key"))

    hook_cfg = _cfg_get(cpe_cfg, "ad_after_cpe", {})
    moe_cfg = _cfg_get(hook_cfg, "moe", {})
    if _cfg_get(moe_cfg, "class_key", None):
        return str(_cfg_get(moe_cfg, "class_key"))

    return "moe_class_id"

_MOE_CENTER_POINT_KEYS = [
    # PointMAE backbone 主分支输出，最推荐
    "center",
    "centers",

    # 如果你在 backbone 里显式区分注册后中心点，可以兼容这些名字
    "registered_center",
    "registered_centers",
    "registered_xyz_center",
    "registered_xyz_centers",

    "aligned_center",
    "aligned_centers",
    "transformed_center",
    "transformed_centers",

    # 采样中心点的常见别名
    "sampled_center",
    "sampled_centers",
    "group_center",
    "group_centers",
    "fps_center",
    "fps_centers",

    # raw 分支中心点。默认不建议优先用，但比原始 pointcloud 更接近 token 空间
    "raw_center",
    "raw_centers",
]

_MOE_REGISTERED_POINT_KEYS = [
    # 最推荐 backbone 输出这些 key 之一
    "registered_points",
    "registered_xyz",
    "registered_pointcloud",
    "registered_pointcloud_xyz",

    # 兼容一些常见命名
    "aligned_points",
    "aligned_xyz",
    "transformed_points",
    "transformed_xyz",
    "points_registered",
    "pointcloud_registered",
    "registration_points",
    "registration_xyz",
]

_MOE_RAW_POINT_KEYS = [
    "pointcloud",
    "points",
    "xyz",
    "coord",
    "coords",
    "raw_points",
    "raw_pointcloud",
]


def _dedup_keys(keys: List[Any]) -> List[str]:
    ret: List[str] = []

    for key in keys:
        if key is None:
            continue

        key = str(key)

        if key == "":
            continue

        if key not in ret:
            ret.append(key)

    return ret


def _get_first_point_tensor(
    source: Any,
    keys: List[str],
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if not isinstance(source, dict):
        return None, None

    for key in keys:
        if key not in source:
            continue

        value = source[key]

        if torch.is_tensor(value):
            return value, key

        if isinstance(value, np.ndarray):
            return torch.from_numpy(value), key

    return None, None


def _resolve_moe_point_key(
    ad_cfg: Optional[EasyDict] = None,
    cpe_cfg: Optional[EasyDict] = None,
) -> str:
    """
    Resolve which key is used as geometry gate point cloud.

    Priority:
      1. AD config top-level moe.gate_point_key / moe.point_key / moe.moe_point_key
      2. AD reconstruction kwargs gate_point_key / moe_point_key / point_key
      3. CPE config ad_after_cpe.moe.*
      4. default: moe_points
    """

    moe_cfg = _cfg_get(ad_cfg, "moe", None)
    for key in ["gate_point_key", "moe_point_key", "point_key"]:
        value = _cfg_get(moe_cfg, key, None)
        if value:
            return str(value)

    net_cfg = _cfg_get(ad_cfg, "net", [])
    for module_cfg in _as_list(net_cfg):
        if str(_cfg_get(module_cfg, "name", "")) != "reconstruction":
            continue

        kwargs = _cfg_get(module_cfg, "kwargs", {})
        for key in ["gate_point_key", "moe_point_key", "point_key"]:
            value = _cfg_get(kwargs, key, None)
            if value:
                return str(value)

    hook_cfg = _cfg_get(cpe_cfg, "ad_after_cpe", {})
    moe_cfg = _cfg_get(hook_cfg, "moe", {})
    for key in ["gate_point_key", "moe_point_key", "point_key"]:
        value = _cfg_get(moe_cfg, key, None)
        if value:
            return str(value)

    return "moe_points"

def _resolve_moe_allow_raw_point_fallback(
    ad_cfg: Optional[EasyDict] = None,
    cpe_cfg: Optional[EasyDict] = None,
) -> bool:
    """
    Whether geometry gate is allowed to fallback to raw pointcloud.

    推荐默认 False：
      - gate 优先看 backbone center
      - 没有 center 就报错
      - 避免无意中退回原始 pointcloud
    """

    moe_cfg = _cfg_get(ad_cfg, "moe", None)
    value = _cfg_get(moe_cfg, "allow_raw_point_fallback", None)
    if value is not None:
        return bool(value)

    net_cfg = _cfg_get(ad_cfg, "net", [])
    for module_cfg in _as_list(net_cfg):
        if str(_cfg_get(module_cfg, "name", "")) != "reconstruction":
            continue

        kwargs = _cfg_get(module_cfg, "kwargs", {})
        value = _cfg_get(kwargs, "allow_raw_point_fallback", None)
        if value is not None:
            return bool(value)

    hook_cfg = _cfg_get(cpe_cfg, "ad_after_cpe", {})
    moe_cfg = _cfg_get(hook_cfg, "moe", {})
    value = _cfg_get(moe_cfg, "allow_raw_point_fallback", None)
    if value is not None:
        return bool(value)

    return False


def _ensure_moe_points_for_gate(
    dst: Any,
    batch: Optional[Dict[str, Any]] = None,
    point_key: str = "moe_points",
    allow_raw_point_fallback: bool = False,
) -> Any:
    """
    Ensure reconstruction input has dst[point_key] and dst["moe_points"].

    新优先级：
      1. backbone_out explicit moe_points / point_key
      2. backbone_out center / registered_center / aligned_center / raw_center
      3. backbone_out registered xyz point cloud
      4. batch explicit moe_points / point_key
      5. batch center-like keys
      6. raw pointcloud fallback, only if allow_raw_point_fallback=True

    推荐：
      allow_raw_point_fallback=False
    这样可以保证 gate 真正在看 PointMAE center，而不是偷偷退回原始 pointcloud。
    """

    if not isinstance(dst, dict):
        return dst

    point_key = str(point_key or "moe_points")

    explicit_keys = _dedup_keys([
        point_key,
        "moe_points",
    ])

    # backbone_out 里最重要的是 center。
    backbone_search_keys = _dedup_keys(
        explicit_keys
        + _MOE_CENTER_POINT_KEYS
        + _MOE_REGISTERED_POINT_KEYS
    )

    batch_search_keys = _dedup_keys(
        explicit_keys
        + _MOE_CENTER_POINT_KEYS
        + _MOE_REGISTERED_POINT_KEYS
    )

    if allow_raw_point_fallback:
        backbone_search_keys = _dedup_keys(
            backbone_search_keys + _MOE_RAW_POINT_KEYS
        )
        batch_search_keys = _dedup_keys(
            batch_search_keys + _MOE_RAW_POINT_KEYS
        )

    search_items = [
        ("backbone_out", dst, backbone_search_keys),
        ("batch", batch, batch_search_keys),
    ]

    for source_name, source, keys in search_items:
        points, used_key = _get_first_point_tensor(source, keys)

        if points is None:
            continue

        points = points.detach()

        dst[point_key] = points
        dst["moe_points"] = points
        dst["moe_points_source"] = f"{source_name}.{used_key}"

        return dst

    # 不在这里强行 fallback 到原始 pointcloud。
    # 让 UniADMoE 报错，方便你发现 backbone 没有传 center。
    return dst


def inject_moe_class_id(
    batch: Dict[str, Any],
    class_id: Any,
    class_key: str = "moe_class_id",
) -> Dict[str, Any]:
    """
    Put class id into batch for MoE routing.

    注意：
    - 这里必须是物体类别 id。
    - 不能用 batch["label"]，因为 label 通常是 normal/anomaly 标签。
    """

    if not torch.is_tensor(class_id):
        class_id = torch.as_tensor(class_id, dtype=torch.long)

    class_id = class_id.detach().view(-1).long().cpu()

    batch[class_key] = class_id

    # 永远也写一份标准 key，避免 config/class_key 不一致时不好排查
    batch["moe_class_id"] = class_id

    return batch


def _copy_ad_context_to_backbone_output(
    backbone_out: Any,
    batch: Dict[str, Any],
    class_key: str = "moe_class_id",
    moe_point_key: str = "moe_points",
    allow_raw_point_fallback: bool = False,
) -> Any:

    """
    Copy routing and eval metadata from batch to backbone output.

    reconstruction / UniADMoE receives backbone_out, so it needs filename
    and routing fields here.
    """

    if not isinstance(backbone_out, dict):
        return backbone_out

    route_keys = [
        class_key,
        "moe_class_id",

        "registration_cls_label",
        "pred_cls_label",
        "cls_label",
        "class_label",
        "category_id",

        "registration_clsname",
        "pred_clsname",
        "clsname",
        "class_name",
        "category",
        "class",

        "filename",

        # 建议补上，后续 dump / merge 可能会用
        "label",
        "mask",
        "masks",
                # geometry gate candidates from backbone
        # 注意：这里不要提前从 batch 复制 moe_points。
        # 后面会优先选 backbone 的 registered points，再 fallback 到 raw pointcloud。
        "registered_points",
        "registered_xyz",
        "registered_pointcloud",
        "registered_pointcloud_xyz",
        "aligned_points",
        "aligned_xyz",
        "transformed_points",
        "transformed_xyz",
        "points_registered",
        "pointcloud_registered",
        "registration_points",
        "registration_xyz",

        # raw point cloud fallback
        "points",
        "pointcloud",
        "xyz",
        "coord",
        "coords",
        "raw_points",
        "raw_pointcloud",

        "task_id",
        "is_train",
        "use_gt_registration",
        "force_gt_registration",

        # PointMAE sampled centers for geometry gate
        "center",
        "centers",
        "registered_center",
        "registered_centers",
        "aligned_center",
        "aligned_centers",
        "transformed_center",
        "transformed_centers",
        "sampled_center",
        "sampled_centers",
        "group_center",
        "group_centers",
        "fps_center",
        "fps_centers",
        "raw_center",
        "raw_centers",

    ]

    for key in route_keys:
        if key in batch and key not in backbone_out:
            backbone_out[key] = batch[key]

    if "points" not in backbone_out and "pointcloud" in batch:
        backbone_out["points"] = batch["pointcloud"]

    if "pointcloud" not in backbone_out and "points" in batch:
        backbone_out["pointcloud"] = batch["points"]

    if "mask" not in backbone_out and "masks" in batch:
        backbone_out["mask"] = batch["masks"]

    if "masks" not in backbone_out and "mask" in batch:
        backbone_out["masks"] = batch["mask"]

    backbone_out = _ensure_moe_points_for_gate(
        dst=backbone_out,
        batch=batch,
        point_key=moe_point_key,
        allow_raw_point_fallback=allow_raw_point_fallback,
    )

    return backbone_out


def _copy_ad_eval_meta_to_outputs(
    outputs: Any,
    batch: Optional[Dict[str, Any]] = None,
    backbone_out: Optional[Dict[str, Any]] = None,
    class_key: str = "moe_class_id",
) -> Any:
    """
    Copy AD evaluation metadata to final model outputs.

    Sources:
      - batch: original dataloader metadata, e.g. filename / label / mask / cls_label
      - backbone_out: backbone-generated metadata, e.g. center_idx / center / ori_idx

    eval_helper.dump() reads these fields from outputs.
    UniAD / UniADMoE usually does not return them by default.
    """

    if not isinstance(outputs, dict):
        return outputs

    meta_keys = [
        # eval_helper.dump 必需或常用字段
        "filename",
        "label",
        "mask",
        "masks",

        # backbone 生成的点分组 / 插值索引
        "center_idx",
        "center",
        "ori_idx",

        # 有些代码可能使用 raw 分支字段
        "raw_center_idx",
        "raw_center",
        "raw_ori_idx",

        # point cloud info
        "points",
        "pointcloud",

        # class GT
        "cls_label",
        "class_label",
        "category_id",
        "clsname",
        "class_name",
        "category",
        "class",

        # CPE predicted registration / MoE routing
        class_key,
        "moe_class_id",
        "registration_cls_label",
        "registration_clsname",
        "registration_source",
        "pred_cls_label",
        "pred_clsname",

        # misc flags
        "task_id",
        "is_train",
        "use_gt_registration",
        "force_gt_registration",
        "is_registered",

        "moe_points_source",

    ]

    # 先从 backbone_out 补，因为 center_idx / center / ori_idx 在这里。
    # 再从 batch 补，因为 filename / label / mask / cls_label 通常在这里。
    for source in [backbone_out, batch]:
        if not isinstance(source, dict):
            continue

        for key in meta_keys:
            if key in source and key not in outputs:
                outputs[key] = source[key]

    # center_idx fallback
    if "center_idx" not in outputs and "raw_center_idx" in outputs:
        outputs["center_idx"] = outputs["raw_center_idx"]

    if "center" not in outputs and "raw_center" in outputs:
        outputs["center"] = outputs["raw_center"]

    if "ori_idx" not in outputs and "raw_ori_idx" in outputs:
        outputs["ori_idx"] = outputs["raw_ori_idx"]

    # mask / masks 互相兜底
    if "mask" not in outputs and "masks" in outputs:
        outputs["mask"] = outputs["masks"]

    if "masks" not in outputs and "mask" in outputs:
        outputs["masks"] = outputs["mask"]

    # points / pointcloud 互相兜底
    if "points" not in outputs and "pointcloud" in outputs:
        outputs["points"] = outputs["pointcloud"]

    if "pointcloud" not in outputs and "points" in outputs:
        outputs["pointcloud"] = outputs["points"]

    # 统一类别 GT key
    if "cls_label" not in outputs:
        for source in [outputs, backbone_out, batch]:
            if not isinstance(source, dict):
                continue

            for key in ["class_label", "category_id"]:
                if key in source:
                    outputs["cls_label"] = source[key]
                    break

            if "cls_label" in outputs:
                break

    # 不覆盖 UniAD 自己输出的 cls_pred。
    # UniAD.forward() 原本会返回 cls_pred logits。
    # 只有完全没有 cls_pred 时，才用 CPE 预测类别兜底。
    if "cls_pred" not in outputs:
        for source in [backbone_out, batch]:
            if isinstance(source, dict) and "pred_cls_label" in source:
                outputs["cls_pred"] = source["pred_cls_label"]
                break

    return outputs



def forward_ad_model_with_moe_context(
    ad_model: nn.Module,
    batch: Dict[str, Any],
    class_key: str = "moe_class_id",
    moe_point_key: str = "moe_points",
    allow_raw_point_fallback: bool = False,
) -> Dict[str, Any]:

    """
    Forward AD model while ensuring MoE routing info reaches reconstruction,
    and eval metadata reaches final outputs.
    """

    raw_model = unwrap_model(ad_model)

    if hasattr(raw_model, "backbone") and hasattr(raw_model, "reconstruction"):
        backbone_out = raw_model.backbone(batch)

        backbone_out = _copy_ad_context_to_backbone_output(
            backbone_out=backbone_out,
            batch=batch,
            class_key=class_key,
            moe_point_key=moe_point_key,
            allow_raw_point_fallback=allow_raw_point_fallback,
        )

        outputs = raw_model.reconstruction(backbone_out)

        outputs = _copy_ad_eval_meta_to_outputs(
            outputs=outputs,
            batch=batch,
            backbone_out=backbone_out,
            class_key=class_key,
        )

        return outputs

    if isinstance(batch, dict):
        batch = _ensure_moe_points_for_gate(
            dst=dict(batch),
            batch=batch,
            point_key=moe_point_key,
            allow_raw_point_fallback=allow_raw_point_fallback,
        )

    outputs = ad_model(batch)

    outputs = _copy_ad_eval_meta_to_outputs(
        outputs=outputs,
        batch=batch,
        backbone_out=None,
        class_key=class_key,
    )

    return outputs



def freeze_ad_layers(ad_model: nn.Module, frozen_layers: List[str]) -> None:
    raw_model = unwrap_model(ad_model)

    for layer in frozen_layers:
        if not hasattr(raw_model, layer):
            raise AttributeError(
                f"AD frozen layer '{layer}' not found. "
                f"Available children={[name for name, _ in raw_model.named_children()]}"
            )

        module = getattr(raw_model, layer)
        module.eval()

        for param in module.parameters():
            param.requires_grad = False


def set_ad_train_mode(ad_model: nn.Module, frozen_layers: List[str]) -> None:
    ad_model.train()
    freeze_ad_layers(ad_model, frozen_layers)


def compute_ad_loss(
    outputs: Dict[str, Any],
    criterion: Dict[str, Any],
    device: torch.device,
    ad_cfg: Optional[EasyDict] = None,
    include_moe_aux: bool = False,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)

    for _, criterion_loss in criterion.items():
        weight = float(criterion_loss.weight)
        loss = loss + weight * criterion_loss(outputs)

    if include_moe_aux and isinstance(outputs, dict):
        moe_cfg = _cfg_get(ad_cfg, "moe", {})
        auto_add_aux = bool(_cfg_get(moe_cfg, "auto_add_aux_loss", True))
        aux_weight = float(_cfg_get(moe_cfg, "aux_loss_weight", 1.0))

        if auto_add_aux and aux_weight != 0.0 and "moe_aux_loss" in outputs:
            moe_aux_loss = outputs["moe_aux_loss"]

            if torch.is_tensor(moe_aux_loss):
                moe_aux_loss = moe_aux_loss.to(device=device)

                if moe_aux_loss.dim() > 0:
                    moe_aux_loss = moe_aux_loss.mean()

                loss = loss + aux_weight * moe_aux_loss

    return loss


def train_ad_one_epoch(
    train_loader: Any,
    ad_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    tb_logger: Optional[SummaryWriter],
    criterion: Dict[str, Any],
    frozen_layers: List[str],
    ad_cfg: EasyDict,
    ad_task_id: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    set_ad_train_mode(ad_model, frozen_layers)

    moe_class_key = _resolve_moe_class_key(ad_cfg=ad_cfg)
    moe_point_key = _resolve_moe_point_key(ad_cfg=ad_cfg)
    allow_raw_point_fallback = _resolve_moe_allow_raw_point_fallback(
        ad_cfg=ad_cfg,
    )


    batch_time = AverageMeter(int(ad_cfg.trainer.print_freq_step))
    data_time = AverageMeter(int(ad_cfg.trainer.print_freq_step))
    losses = AverageMeter(int(ad_cfg.trainer.print_freq_step))
    losses_total = AverageMeter(0)

    end = time.time()

    for i, batch in enumerate(train_loader):
        batch = inject_gt_registration_clsname(
            batch=batch,
            moe_class_key=moe_class_key,
        )

        batch["task_id"] = int(ad_task_id)

        curr_step = epoch * len(train_loader) + i
        current_lr = float(optimizer.param_groups[0]["lr"])

        data_time.update(time.time() - end)

        outputs = forward_ad_model_with_moe_context(
            ad_model=ad_model,
            batch=batch,
            class_key=moe_class_key,
            moe_point_key=moe_point_key,
            allow_raw_point_fallback=allow_raw_point_fallback,
        )

        device = next(ad_model.parameters()).device
        loss = compute_ad_loss(
            outputs=outputs,
            criterion=criterion,
            device=device,
            ad_cfg=ad_cfg,
            include_moe_aux=True,
        )

        optimizer.zero_grad()
        loss.backward()

        if ad_cfg.trainer.get("clip_max_norm", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                ad_model.parameters(),
                float(ad_cfg.trainer.clip_max_norm),
            )

        optimizer.step()

        batch_size = _batch_size_from_batch(batch)
        losses.update(float(loss.item()), 1)
        losses_total.update(float(loss.item()), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % int(ad_cfg.trainer.print_freq_step) == 0:
            if tb_logger is not None:
                tb_logger.add_scalar("ad_train/loss", losses.avg, curr_step + 1)
                tb_logger.add_scalar("ad_train/lr", current_lr, curr_step + 1)

                for moe_loss_key in [
                    "moe_aux_loss",
                    "moe_load_balance_loss",
                    "moe_class_balance_loss",
                    "moe_entropy_loss",
                ]:
                    if moe_loss_key in outputs and torch.is_tensor(outputs[moe_loss_key]):
                        tb_logger.add_scalar(
                            f"ad_train/{moe_loss_key}",
                            float(outputs[moe_loss_key].detach().mean().item()),
                            curr_step + 1,
                        )

                if "moe_expert_id" in outputs and torch.is_tensor(outputs["moe_expert_id"]):
                    expert_id = outputs["moe_expert_id"].detach().view(-1).float()
                    tb_logger.add_scalar(
                        "ad_train/moe_expert_id_mean",
                        float(expert_id.mean().item()),
                        curr_step + 1,
                    )

                tb_logger.flush()

            logger.info(
                "[AD Train] Epoch: [{}/{}]\t"
                "Iter: [{}/{}]\t"
                "Time {:.2f} ({:.2f})\t"
                "Data {:.2f} ({:.2f})\t"
                "Loss {:.5f} ({:.5f})\t"
                "LR {:.6f}".format(
                    epoch + 1,
                    int(ad_cfg.trainer.max_epoch),
                    i + 1,
                    len(train_loader),
                    batch_time.val,
                    batch_time.avg,
                    data_time.val,
                    data_time.avg,
                    losses.val,
                    losses.avg,
                    current_lr,
                )
            )

    return {
        "loss": float(losses_total.avg),
        "total_num": float(losses_total.count),
    }



@torch.no_grad()
def validate_ad_seen_with_cpe_registration(
    seen_val_items: List[Tuple[int, Any]],
    ad_model: nn.Module,
    criterion: Dict[str, Any],
    ad_cfg: EasyDict,
    cpe_model: nn.Module,
    cpe_cfg: EasyDict,
    current_cpe_task_id: int,
    proto_bank: torch.Tensor,
    proto_counts: torch.Tensor,
    class_names: List[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    if len(seen_val_items) == 0:
        raise RuntimeError("[AD Seen Test] seen_val_items is empty.")

    ad_model.eval()
    cpe_model.eval()

    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    eval_dir = ad_cfg.evaluator.eval_dir

    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)

    os.makedirs(eval_dir, exist_ok=True)

    hook_cfg = EasyDict(cpe_cfg.get("ad_after_cpe", {}))
    ad_task_id_policy = str(hook_cfg.get("ad_test_task_id_policy", "current")).lower()

    moe_class_key = _resolve_moe_class_key(
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
    )
    moe_point_key = _resolve_moe_point_key(
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
    )
    allow_raw_point_fallback = _resolve_moe_allow_raw_point_fallback(
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
    )


    end = time.time()

    for source_task_id, val_loader in seen_val_items:
        logger.info(
            "[AD Seen Test] source_task={} current_cpe_task={}".format(
                source_task_id,
                current_cpe_task_id,
            )
        )

        for i, batch in enumerate(val_loader):
            batch = dict(batch)

            if ad_task_id_policy == "source":
                batch["task_id"] = int(source_task_id)
            else:
                batch["task_id"] = int(current_cpe_task_id)

            batch = inject_cpe_pred_registration_clsname(
                batch=batch,
                cpe_model=cpe_model,
                cpe_cfg=cpe_cfg,
                current_cpe_task_id=current_cpe_task_id,
                source_task_id=source_task_id,
                proto_bank=proto_bank,
                proto_counts=proto_counts,
                class_names=class_names,
                moe_class_key=moe_class_key,
            )

            outputs = forward_ad_model_with_moe_context(
                ad_model=ad_model,
                batch=batch,
                class_key=moe_class_key,
                moe_point_key=moe_point_key,
                allow_raw_point_fallback=allow_raw_point_fallback,
            )

            dump(eval_dir, outputs)

            device = next(ad_model.parameters()).device
            loss = compute_ad_loss(
                outputs=outputs,
                criterion=criterion,
                device=device,
                ad_cfg=ad_cfg,
                include_moe_aux=False,
            )

            batch_size = _batch_size_from_batch(batch)
            losses.update(float(loss.item()), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % int(ad_cfg.trainer.print_freq_step) == 0:
                logger.info(
                    "[AD Seen Test] source_task={} iter[{}/{}]\t"
                    "Time {:.3f} ({:.3f})".format(
                        source_task_id,
                        i + 1,
                        len(val_loader),
                        batch_time.val,
                        batch_time.avg,
                    )
                )

    final_loss = float(losses.avg)

    logger.info(
        "[AD Seen Test] * Loss {:.5f}\ttotal_num={}".format(
            final_loss,
            losses.count,
        )
    )

    logger.info("[AD Seen Test] Gathering final results ...")

    fileinfos, labels, image_max_score, masks, pred, cls_label, points, cls_pred = merge_together(
        eval_dir
    )

    shutil.rmtree(eval_dir)

    ret_metrics = performances(
        fileinfos,
        labels,
        image_max_score,
        masks,
        pred,
        cls_label,
        cls_pred,
    )

    ret_metrics["loss"] = final_loss

    log_metrics(ret_metrics, ad_cfg.evaluator.metrics)

    ad_model.train()

    return ret_metrics


def load_ad_cfg_base(
    ad_config_path: str,
    cpe_task_id: int,
) -> EasyDict:
    with open(ad_config_path, "r") as f:
        ad_cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    ad_cfg.exp_path = os.path.dirname(ad_config_path)
    if ad_cfg.exp_path == "":
        ad_cfg.exp_path = "."

    ad_cfg.save_path = _path_under_exp(
        ad_cfg.exp_path,
        ad_cfg.saver.save_dir,
        f"cpe_task_{int(cpe_task_id)}",
    )

    ad_cfg.log_path = _path_under_exp(
        ad_cfg.exp_path,
        ad_cfg.saver.log_dir,
        f"ad_cpe_task_{int(cpe_task_id)}",
    )

    ad_cfg.evaluator.eval_dir = _path_under_exp(
        ad_cfg.exp_path,
        ad_cfg.evaluator.save_dir,
        f"ad_cpe_task_{int(cpe_task_id)}",
    )

    os.makedirs(ad_cfg.save_path, exist_ok=True)
    os.makedirs(ad_cfg.log_path, exist_ok=True)

    return ad_cfg


def prepare_ad_pointmae_cfg(
    ad_cfg: EasyDict,
    class_names: List[str],
) -> None:
    """
    只改 AD model 构造参数，不改 AD dataloader。

    重点：
    - ad_cfg.dataset.data_dir 是 AD 数据集读取路径。
    - ad_cfg.net.backbone.kwargs.data_dir 是 PointMAE 模板目录。
    - 如果 backbone.kwargs.data_dir 没写，默认用 ad_cfg.dataset.data_dir 当模板目录。
    """

    for module_cfg in ad_cfg.net:
        if "kwargs" not in module_cfg or module_cfg.kwargs is None:
            module_cfg.kwargs = EasyDict()

        if str(module_cfg.name) != "backbone":
            continue

        module_cfg.kwargs["class_names"] = [str(x) for x in class_names]

        # 不新增 template_dir。
        # 这里的 data_dir 是 PointMAE 模板目录。
        # 如果你没单独写，就默认用 AD dataset.data_dir。
        if "data_dir" not in module_cfg.kwargs or module_cfg.kwargs["data_dir"] is None:
            module_cfg.kwargs["data_dir"] = ad_cfg.dataset.data_dir

        module_cfg.kwargs.setdefault("dual_feature", False)
        module_cfg.kwargs.setdefault("return_raw_features", False)
        module_cfg.kwargs.setdefault("return_raw_features_in_registered_pass", False)
        module_cfg.kwargs.setdefault("train_register_with_gt", True)
        module_cfg.kwargs.setdefault("eval_use_gt_registration", False)
        module_cfg.kwargs.setdefault("cls_raw_coord_preprocess", False)


def select_ad_train_and_test_loaders(
    train_loaders: Any,
    val_loaders: Any,
    cpe_task_id: int,
    hook_cfg: EasyDict,
) -> Tuple[Any, List[Tuple[int, Any]]]:
    train_loaders = _as_list(train_loaders)
    val_loaders = _as_list(val_loaders)
    cpe_task_id = int(cpe_task_id)

    if len(train_loaders) == 0:
        raise RuntimeError("[AD Loader] no train loader.")
    if len(val_loaders) == 0:
        raise RuntimeError("[AD Loader] no val/test loader.")

    train_scope = str(hook_cfg.get("train_scope", "current")).lower()
    test_scope = str(hook_cfg.get("test_scope", "seen")).lower()

    if len(train_loaders) == 1:
        ad_train_loader = train_loaders[0]
    else:
        if train_scope in ["current", "cur"]:
            if cpe_task_id >= len(train_loaders):
                raise RuntimeError(
                    f"[AD Loader] cpe_task_id={cpe_task_id} out of range "
                    f"for AD train loaders len={len(train_loaders)}. "
                    "如果要每个 CPE task 后跑 AD，建议 AD config.dataset.task_num=[3,3,3,3]。"
                )
            ad_train_loader = train_loaders[cpe_task_id]
        elif train_scope in ["seen", "all_seen"]:
            # 第一版仍然只返回当前 train loader，避免把 seen train 混在一起导致训练量暴涨。
            # 如果你后面要 seen replay，可以在这里改成 ConcatDataset。
            if cpe_task_id >= len(train_loaders):
                raise RuntimeError(
                    f"[AD Loader] cpe_task_id={cpe_task_id} out of range "
                    f"for AD train loaders len={len(train_loaders)}."
                )
            ad_train_loader = train_loaders[cpe_task_id]
        else:
            raise ValueError(
                f"Unsupported ad_after_cpe.train_scope={train_scope}. "
                "Use current."
            )

    if len(val_loaders) == 1:
        seen_val_items = [(0, val_loaders[0])]
    else:
        if test_scope in ["current", "cur"]:
            if cpe_task_id >= len(val_loaders):
                raise RuntimeError(
                    f"[AD Loader] cpe_task_id={cpe_task_id} out of range "
                    f"for AD val loaders len={len(val_loaders)}."
                )
            seen_val_items = [(cpe_task_id, val_loaders[cpe_task_id])]

        elif test_scope in ["seen", "seen_classes"]:
            end = min(cpe_task_id + 1, len(val_loaders))
            seen_val_items = [(idx, val_loaders[idx]) for idx in range(end)]

        elif test_scope in ["all", "full"]:
            seen_val_items = [(idx, loader) for idx, loader in enumerate(val_loaders)]

        else:
            raise ValueError(
                f"Unsupported ad_after_cpe.test_scope={test_scope}. "
                "Use current, seen, or all."
            )

    return ad_train_loader, seen_val_items


def _strip_module_prefix_for_load(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        return {
            str(k)[len("module.") :]: v
            for k, v in state_dict.items()
        }

    return state_dict


def load_ad_previous_task_if_needed(
    ad_model: nn.Module,
    ad_cfg: EasyDict,
    cpe_task_id: int,
    logger: logging.Logger,
) -> None:
    cpe_task_id = int(cpe_task_id)

    if cpe_task_id <= 0:
        logger.info("[AD Resume] task 0 starts from initialized AD model.")
        return

    base_save_dir = _path_under_exp(
        ad_cfg.exp_path,
        ad_cfg.saver.save_dir,
    )

    prev_dir = os.path.join(
        base_save_dir,
        f"cpe_task_{cpe_task_id - 1}",
    )

    candidates = [
        os.path.join(prev_dir, "ckpt.pth.tar"),
        os.path.join(prev_dir, "ckpt_best.pth.tar"),
        os.path.join(prev_dir, "latest.pth"),
        os.path.join(prev_dir, "best.pth"),
    ]

    prev_ckpt = None
    for path in candidates:
        if os.path.isfile(path):
            prev_ckpt = path
            break

    if prev_ckpt is None:
        logger.info(
            "[AD Resume] previous AD checkpoint not found. Tried: {}".format(
                candidates
            )
        )
        return

    ckpt = torch.load(prev_ckpt, map_location="cuda")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = _strip_module_prefix_for_load(state_dict)

    incompatible = ad_model.load_state_dict(state_dict, strict=False)

    logger.info(
        "[AD Resume] loaded previous AD checkpoint: {} | missing={} unexpected={}".format(
            prev_ckpt,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    )

    if len(incompatible.missing_keys) > 0:
        logger.info(
            "[AD Resume] missing keys sample: {}".format(
                incompatible.missing_keys[:30]
            )
        )

    if len(incompatible.unexpected_keys) > 0:
        logger.info(
            "[AD Resume] unexpected keys sample: {}".format(
                incompatible.unexpected_keys[:30]
            )
        )


def _metric_is_better(metric_name: str, current: float, best: float) -> bool:
    metric_name = str(metric_name).lower()

    if current != current:
        return False

    if "loss" in metric_name:
        return current < best

    return current > best


def _initial_best_metric(metric_name: str) -> float:
    metric_name = str(metric_name).lower()

    if "loss" in metric_name:
        return float("inf")

    return float("-inf")


def save_ad_checkpoint(
    ad_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ad_cfg: EasyDict,
    epoch: int,
    cpe_task_id: int,
    best_metric: float,
    is_best: bool,
    ret_metrics: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    proto_counts: Optional[torch.Tensor] = None,
) -> None:
    os.makedirs(ad_cfg.save_path, exist_ok=True)

    state = {
        "epoch": int(epoch),
        "cpe_task_id": int(cpe_task_id),
        "arch": ad_cfg.net,
        "state_dict": ad_model.state_dict(),
        "best_metric": float(best_metric),
        "optimizer": optimizer.state_dict(),
        "ret_metrics": ret_metrics,
        "class_names": class_names,
        "proto_counts": None if proto_counts is None else proto_counts.detach().cpu(),
    }

    latest_path = os.path.join(ad_cfg.save_path, "ckpt.pth.tar")
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(ad_cfg.save_path, "ckpt_best.pth.tar")
        torch.save(state, best_path)


def _should_run_ad_after_task(run_after_task_ids: Any, cpe_task_id: int) -> bool:
    if run_after_task_ids is None:
        return True

    if isinstance(run_after_task_ids, str):
        if run_after_task_ids.lower() == "all":
            return True
        return int(run_after_task_ids) == int(cpe_task_id)

    if isinstance(run_after_task_ids, (list, tuple)):
        return int(cpe_task_id) in [int(x) for x in run_after_task_ids]

    return int(run_after_task_ids) == int(cpe_task_id)


def run_ad_after_cpe_task(
    cpe_model: nn.Module,
    cpe_cfg: EasyDict,
    cpe_train_loaders: Any,
    cpe_task_id: int,
    proto_mod: nn.Module,
    ad_config_path: str,
    logger: logging.Logger,
) -> None:
    hook_cfg = EasyDict(cpe_cfg.get("ad_after_cpe", {}))

    ad_cfg = load_ad_cfg_base(
        ad_config_path=ad_config_path,
        cpe_task_id=cpe_task_id,
    )

    logger.info("[AD-after-CPE] loaded AD config from {}".format(ad_config_path))
    logger.info("[AD-after-CPE] AD dataset config: {}".format(ad_cfg.dataset))

    ad_train_loaders, ad_val_loaders = build_dataloader(
        ad_cfg.dataset,
        distributed=False,
    )

    ad_train_loaders = _as_list(ad_train_loaders)
    ad_val_loaders = _as_list(ad_val_loaders)

    class_names = resolve_class_names(
        cpe_cfg=cpe_cfg,
        cpe_train_loaders=cpe_train_loaders,
        ad_train_loaders=ad_train_loaders,
        ad_val_loaders=ad_val_loaders,
    )

    logger.info(
        "[AD-after-CPE] class_names={}".format(
            list(enumerate(class_names))
        )
    )

    prepare_ad_pointmae_cfg(
        ad_cfg=ad_cfg,
        class_names=class_names,
    )

    proto_scope = str(hook_cfg.get("prototype_scope", "seen")).lower()
    proto_loader_items = _select_cpe_proto_loader_items(
        cpe_train_loaders=cpe_train_loaders,
        cpe_task_id=cpe_task_id,
        scope=proto_scope,
    )

    proto_bank, proto_counts = build_cpe_eval_prototype_bank(
        cpe_model=cpe_model,
        cpe_cfg=cpe_cfg,
        loader_items=proto_loader_items,
        current_cpe_task_id=cpe_task_id,
        proto_mod=proto_mod,
        class_names=class_names,
        logger=logger,
    )

    ad_train_loader, ad_seen_val_items = select_ad_train_and_test_loaders(
        train_loaders=ad_train_loaders,
        val_loaders=ad_val_loaders,
        cpe_task_id=cpe_task_id,
        hook_cfg=hook_cfg,
    )

    ad_model = ModelHelper(ad_cfg.net)
    ad_model.cuda()

    frozen_layers = [str(x) for x in ad_cfg.get("frozen_layers", [])]

    if bool(hook_cfg.get("resume_ad_from_previous_task", True)):
        load_ad_previous_task_if_needed(
            ad_model=ad_model,
            ad_cfg=ad_cfg,
            cpe_task_id=cpe_task_id,
            logger=logger,
        )
    
    configure_moe_class_assignment_if_needed(
        ad_model=ad_model,
        ad_train_loaders=ad_train_loaders,
        ad_cfg=ad_cfg,
        cpe_cfg=cpe_cfg,
        cpe_task_id=cpe_task_id,
        class_names=class_names,
        logger=logger,
    )


    freeze_ad_layers(ad_model, frozen_layers)

    layers = [str(module_cfg["name"]) for module_cfg in ad_cfg.net]
    active_layers = [layer for layer in layers if layer not in frozen_layers]

    params = []
    for layer in active_layers:
        layer_module = getattr(ad_model, layer)
        layer_params = [
            p for p in layer_module.parameters()
            if p.requires_grad
        ]

        if len(layer_params) > 0:
            params.append({"params": layer_params})

    if len(params) == 0:
        raise RuntimeError(
            f"No trainable AD parameters. layers={layers}, frozen_layers={frozen_layers}"
        )

    optimizer = get_ad_optimizer(params, ad_cfg.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, ad_cfg.trainer.lr_scheduler)
    criterion = build_criterion(ad_cfg.criterion)

    tb_logger = SummaryWriter(
        os.path.join(ad_cfg.log_path, "events_ad", get_current_time())
    )

    key_metric = str(ad_cfg.evaluator.key_metric)
    best_metric = _initial_best_metric(key_metric)
    last_metrics = None

    max_epoch = int(ad_cfg.trainer.max_epoch)
    val_freq = int(ad_cfg.trainer.get("val_freq_epoch", 1))
    val_freq = max(val_freq, 1)

    logger.info(
        "[AD-after-CPE] start AD training | cpe_task={} max_epoch={} val_freq={}".format(
            cpe_task_id,
            max_epoch,
            val_freq,
        )
    )

    for epoch in range(max_epoch):
        train_metrics = train_ad_one_epoch(
            train_loader=ad_train_loader,
            ad_model=ad_model,
            optimizer=optimizer,
            epoch=epoch,
            tb_logger=tb_logger,
            criterion=criterion,
            frozen_layers=frozen_layers,
            ad_cfg=ad_cfg,
            ad_task_id=cpe_task_id,
            logger=logger,
        )

        lr_scheduler.step(epoch)

        logger.info(
            "[AD Train Summary] cpe_task={} epoch={}/{} loss={:.6f} total_num={:.1f}".format(
                cpe_task_id,
                epoch + 1,
                max_epoch,
                train_metrics["loss"],
                train_metrics["total_num"],
            )
        )

        save_ad_checkpoint(
            ad_model=ad_model,
            optimizer=optimizer,
            ad_cfg=ad_cfg,
            epoch=epoch + 1,
            cpe_task_id=cpe_task_id,
            best_metric=best_metric,
            is_best=False,
            ret_metrics=last_metrics,
            class_names=class_names,
            proto_counts=proto_counts,
        )

        if (epoch + 1) % val_freq != 0:
            continue

        ret_metrics = validate_ad_seen_with_cpe_registration(
            seen_val_items=ad_seen_val_items,
            ad_model=ad_model,
            criterion=criterion,
            ad_cfg=ad_cfg,
            cpe_model=cpe_model,
            cpe_cfg=cpe_cfg,
            current_cpe_task_id=cpe_task_id,
            proto_bank=proto_bank,
            proto_counts=proto_counts,
            class_names=class_names,
            logger=logger,
        )

        last_metrics = ret_metrics

        if key_metric not in ret_metrics:
            raise KeyError(
                f"AD key_metric={key_metric} not in ret_metrics. "
                f"Available keys={list(ret_metrics.keys())}"
            )

        cur_metric = float(ret_metrics[key_metric])
        is_best = _metric_is_better(key_metric, cur_metric, best_metric)

        if is_best:
            best_metric = cur_metric

        save_ad_checkpoint(
            ad_model=ad_model,
            optimizer=optimizer,
            ad_cfg=ad_cfg,
            epoch=epoch + 1,
            cpe_task_id=cpe_task_id,
            best_metric=best_metric,
            is_best=is_best,
            ret_metrics=ret_metrics,
            class_names=class_names,
            proto_counts=proto_counts,
        )

        logger.info(
            "[AD-after-CPE] cpe_task={} epoch={} {}={:.6f} best={:.6f}".format(
                cpe_task_id,
                epoch + 1,
                key_metric,
                cur_metric,
                best_metric,
            )
        )

    tb_logger.close()

    del ad_model
    del optimizer
    del lr_scheduler
    del criterion

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_ad_after_cpe_task_if_needed(
    cpe_model: nn.Module,
    cpe_cfg: EasyDict,
    cpe_train_loaders: Any,
    cpe_task_id: int,
    proto_mod: nn.Module,
    logger: logging.Logger,
) -> None:
    hook_cfg = EasyDict(cpe_cfg.get("ad_after_cpe", {}))

    if not bool(hook_cfg.get("enabled", False)):
        return

    run_after_task_ids = hook_cfg.get("run_after_task_ids", "all")

    if not _should_run_ad_after_task(run_after_task_ids, cpe_task_id):
        logger.info(
            "[AD-after-CPE] skip cpe_task={} because run_after_task_ids={}".format(
                cpe_task_id,
                run_after_task_ids,
            )
        )
        return

    ad_config_path = hook_cfg.get("config", None)
    if not ad_config_path:
        raise RuntimeError("ad_after_cpe.enabled=True but ad_after_cpe.config is empty.")

    ad_config_path = _resolve_config_path(cpe_cfg, ad_config_path)

    restore_rng = bool(hook_cfg.get("restore_rng_after_ad", True))
    rng_state = _save_rng_state() if restore_rng else None

    was_training = cpe_model.training

    logger.info(
        "[AD-after-CPE] start after CPE task {} | config={}".format(
            cpe_task_id,
            ad_config_path,
        )
    )

    try:
        run_ad_after_cpe_task(
            cpe_model=cpe_model,
            cpe_cfg=cpe_cfg,
            cpe_train_loaders=cpe_train_loaders,
            cpe_task_id=cpe_task_id,
            proto_mod=proto_mod,
            ad_config_path=ad_config_path,
            logger=logger,
        )
    finally:
        if rng_state is not None:
            _restore_rng_state(rng_state)

        if was_training:
            cpe_model.train()
        else:
            cpe_model.eval()

    logger.info(
        "[AD-after-CPE] done after CPE task {}".format(
            cpe_task_id
        )
    )
