# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from feature.pointops.functions import pointops
except Exception:
    pointops = None


try:
    from models.reconstructions import UniAD
except Exception:
    from models.reconstructions.uniad import UniAD


def _get_batch_size(x: Any) -> Optional[int]:
    if torch.is_tensor(x):
        if x.dim() > 0:
            return int(x.shape[0])
        return 1

    if isinstance(x, dict):
        for v in x.values():
            if torch.is_tensor(v) and v.dim() > 0:
                return int(v.shape[0])

        for v in x.values():
            if isinstance(v, (list, tuple)):
                return len(v)

    return None


def _get_device(x: Any) -> torch.device:
    if torch.is_tensor(x):
        return x.device

    if isinstance(x, dict):
        for v in x.values():
            if torch.is_tensor(v):
                return v.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _slice_by_mask(x: Any, mask: torch.Tensor, batch_size: int) -> Any:
    if torch.is_tensor(x):
        if x.dim() > 0 and int(x.shape[0]) == int(batch_size):
            # 关键修复：mask 必须和当前被索引 tensor 在同一设备
            local_mask = mask.to(device=x.device, dtype=torch.bool)
            return x[local_mask]
        return x

    if isinstance(x, dict):
        return {
            k: _slice_by_mask(v, mask, batch_size)
            for k, v in x.items()
        }

    if isinstance(x, list):
        if len(x) == int(batch_size):
            idx = torch.nonzero(
                mask.detach().cpu(),
                as_tuple=False
            ).view(-1).tolist()
            return [x[i] for i in idx]
        return x

    if isinstance(x, tuple):
        if len(x) == int(batch_size):
            idx = torch.nonzero(
                mask.detach().cpu(),
                as_tuple=False
            ).view(-1).tolist()
            return tuple(x[i] for i in idx)
        return x

    return x

def _find_tensor_by_keys(x: Any, keys: list[str]) -> Optional[torch.Tensor]:
    """
    Recursively find the first tensor whose key is in keys.
    """
    if not isinstance(x, dict):
        return None

    for k in keys:
        if k in x and torch.is_tensor(x[k]):
            return x[k]

    for v in x.values():
        if isinstance(v, dict):
            found = _find_tensor_by_keys(v, keys)
            if found is not None:
                return found

    return None

def _as_btn_features(
    feat: torch.Tensor,
    batch_size: int,
    feature_dim: Optional[int] = None,
    key_name: str = "feature",
) -> torch.Tensor:
    """
    Normalize PointMAE / UniAD feature to [B, T, C].

    Accept:
        [B, C]       -> [B, 1, C]
        [B, T, C]
        [B, C, T]

    Strongly recommend setting feature_dim in config.
    For PointMAE this is often 384, 768, etc.
    """
    if feat.dim() == 2:
        if int(feat.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"{key_name} batch mismatch: feature B={feat.shape[0]}, "
                f"batch_size={batch_size}."
            )
        return feat.contiguous().unsqueeze(1)

    if feat.dim() != 3:
        raise RuntimeError(
            f"{key_name} should have shape [B,C], [B,T,C], or [B,C,T], "
            f"got {tuple(feat.shape)}."
        )

    if int(feat.shape[0]) != int(batch_size):
        raise RuntimeError(
            f"{key_name} batch mismatch: feature B={feat.shape[0]}, "
            f"batch_size={batch_size}."
        )

    if feature_dim is not None and int(feature_dim) > 0:
        feature_dim = int(feature_dim)

        if int(feat.shape[-1]) == feature_dim:
            return feat.contiguous()

        if int(feat.shape[1]) == feature_dim:
            return feat.transpose(1, 2).contiguous()

        raise RuntimeError(
            f"{key_name} cannot infer feature layout with feature_dim={feature_dim}. "
            f"Got shape={tuple(feat.shape)}. "
            "Please set gate_feature_dim to the PointMAE feature channel size."
        )

    # Fallback heuristic. Better to avoid this by setting gate_feature_dim.
    if int(feat.shape[1]) > int(feat.shape[2]):
        feat = feat.transpose(1, 2)

    return feat.contiguous()


def _feature_descriptor(feat_btn: torch.Tensor, desc_mode: str) -> torch.Tensor:
    """
    feat_btn: [B, T, C]
    return: [B, D]
    """
    feat_btn = feat_btn.float()
    feat_btn = torch.nan_to_num(feat_btn, nan=0.0, posinf=1.0e4, neginf=-1.0e4)

    desc_mode = str(desc_mode).lower()

    mean = feat_btn.mean(dim=1)

    if desc_mode == "mean":
        return mean

    std = feat_btn.std(dim=1, unbiased=False)

    if desc_mode == "mean_std":
        return torch.cat([mean, std], dim=-1)

    maxv = feat_btn.amax(dim=1)

    if desc_mode == "mean_std_max":
        return torch.cat([mean, std, maxv], dim=-1)

    minv = feat_btn.amin(dim=1)

    if desc_mode == "mean_std_max_min":
        return torch.cat([mean, std, maxv, minv], dim=-1)

    raise ValueError(
        f"Unsupported gate_feature_desc={desc_mode}. "
        "Use mean, mean_std, mean_std_max, or mean_std_max_min."
    )


class FeatureGate(nn.Module):
    """
    PointMAE-feature-aware gate.

    It uses the feature before / inside UniAD instead of hand-crafted geometry.
    Recommended input feature shape:
        [B, C, T] or [B, T, C]
    """

    def __init__(
        self,
        num_experts: int,
        feature_dim: int,
        desc_mode: str = "mean_std_max_min",
        hidden_dim: int = 128,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_experts = int(num_experts)
        self.feature_dim = int(feature_dim)
        self.desc_mode = str(desc_mode).lower()
        self.temperature = float(temperature)

        if self.feature_dim <= 0:
            raise ValueError(
                "gate_feature_dim must be > 0. "
                "Set it to the PointMAE feature channel size, e.g. 384 or 768."
            )

        if self.desc_mode == "mean":
            desc_dim = self.feature_dim
        elif self.desc_mode == "mean_std":
            desc_dim = self.feature_dim * 2
        elif self.desc_mode == "mean_std_max":
            desc_dim = self.feature_dim * 3
        elif self.desc_mode == "mean_std_max_min":
            desc_dim = self.feature_dim * 4
        else:
            raise ValueError(
                f"Unsupported gate_feature_desc={self.desc_mode}."
            )

        self.net = nn.Sequential(
            nn.LayerNorm(desc_dim),
            nn.Linear(desc_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )

    def forward(self, feat_btn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        desc = _feature_descriptor(
            feat_btn=feat_btn,
            desc_mode=self.desc_mode,
        )

        logits = self.net(desc)

        temperature = max(float(self.temperature), 1.0e-6)
        probs = F.softmax(logits / temperature, dim=-1)

        return logits, probs

def _as_bnc_points(points: torch.Tensor, batch_size: int, key_name: str) -> torch.Tensor:
    """
    Normalize point tensor shape to [B, N, 3].
    Accept:
        [B, N, 3]
        [B, 3, N]
        [N, 3] when B == 1
    """
    if points.dim() == 2:
        if points.shape[-1] != 3:
            raise RuntimeError(
                f"{key_name} should have shape [N, 3], got {tuple(points.shape)}."
            )
        points = points.unsqueeze(0)

    elif points.dim() == 3:
        if points.shape[-1] == 3:
            pass
        elif points.shape[1] == 3:
            points = points.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                f"{key_name} should have shape [B, N, 3] or [B, 3, N], "
                f"got {tuple(points.shape)}."
            )
    else:
        raise RuntimeError(
            f"{key_name} should have shape [B, N, 3], [B, 3, N], or [N, 3], "
            f"got {tuple(points.shape)}."
        )

    if int(points.shape[0]) != int(batch_size):
        raise RuntimeError(
            f"{key_name} batch size mismatch: points B={points.shape[0]}, "
            f"but inferred batch_size={batch_size}."
        )

    return points.contiguous()


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return flattened KNN indices with shape [B*N, k].
    points: [B, N, 3]

    CUDA path uses pointops.
    CPU fallback uses torch.cdist, only for debugging / small data.
    """
    B, N, _ = points.shape
    device = points.device

    if N <= 1:
        raise RuntimeError("KNN requires N > 1.")

    k = min(int(k), int(N - 1))

    if device.type == "cuda":
        if pointops is None:
            raise RuntimeError(
                "pointops is required on CUDA but import failed. "
                "Please check: from feature.pointops.functions import pointops"
            )

        points_flat = points.contiguous().view(-1, 3)

        batch_offset = torch.arange(
            0,
            (B + 1) * N,
            N,
            device=device,
            dtype=torch.int32,
        )

        knn_idx, _ = pointops.knnquery(
            k + 1,
            points_flat,
            points_flat,
            batch_offset,
            batch_offset,
        )

        knn_idx = knn_idx.to(torch.int32)[:, 1:]
        return knn_idx

    # CPU fallback
    dist = torch.cdist(points.float(), points.float())
    idx = dist.topk(k + 1, dim=-1, largest=False).indices[:, :, 1:]
    offset = torch.arange(B, device=device).view(B, 1, 1) * N
    idx = idx + offset
    return idx.reshape(B * N, k).to(torch.int32)


def _triangular_repsurf8(points: torch.Tensor, k: int = 9) -> torch.Tensor:
    """
    points: [B, N, 3]
    return: [B, N, 8]

    Feature:
        centroid xyz: 3
        normal xyz:   3
        height:       1
        local trace:  1

    Total 8 dims.
    """
    B, N, _ = points.shape
    device = points.device

    if N <= 2:
        return points.new_zeros(B, N, 8)

    points = points.float().contiguous()
    k = min(int(k), int(N - 1))

    points_flat = points.view(-1, 3)
    knn_idx = _knn_indices(points, k=k)

    knn_points = points_flat[knn_idx.reshape(-1).long()].view(-1, k, 3)

    centroids = knn_points.mean(dim=1)
    centered = knn_points - centroids.unsqueeze(1)

    cov = torch.einsum("nki,nkj->nij", centered, centered) / float(k)
    identity = torch.eye(3, device=device, dtype=cov.dtype).expand_as(cov)
    cov = cov + identity * 1.0e-6

    chunk_size = 10000
    total_points = cov.shape[0]
    eigenvectors = []

    for i in range(0, total_points, chunk_size):
        cov_chunk = cov[i:i + chunk_size]
        _, eigenvectors_chunk = torch.linalg.eigh(cov_chunk)
        eigenvectors.append(eigenvectors_chunk)

    eigenvectors = torch.cat(eigenvectors, dim=0)

    normals = eigenvectors[:, :, 0]
    sign = torch.where(
        normals[:, 2:3] >= 0,
        torch.ones_like(normals[:, 2:3]),
        -torch.ones_like(normals[:, 2:3]),
    )
    normals = normals * sign

    heights = torch.sum(normals * centroids, dim=1, keepdim=True) / math.sqrt(3.0)

    trace = cov.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)

    repsurf = torch.cat(
        [
            centroids,
            normals,
            heights,
            trace,
        ],
        dim=1,
    ).view(B, N, 8)

    return repsurf


def _normalize_points_for_gate(points: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """
    Translation and scale normalization for geometry gate.
    This makes the gate focus more on shape rather than absolute position / scale.
    """
    center = points.mean(dim=1, keepdim=True)
    points = points - center

    scale = torch.sqrt(
        torch.sum(points ** 2, dim=-1).mean(dim=1, keepdim=True)
    ).view(points.shape[0], 1, 1)

    points = points / scale.clamp_min(eps)
    return points


def _geometry_descriptor(
    points: torch.Tensor,
    k: int = 9,
    max_points: int = 2048,
    normalize_points: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Build one global geometry descriptor per sample.

    Input:
        points: [B, N, 3]

    Output:
        desc: [B, 44]

    44 dims:
        repsurf mean: 8
        repsurf std:  8
        repsurf max:  8
        repsurf min:  8
        bbox extent:  3
        bbox ratio:   3
        global eig:   3
        shape index:  3
    """
    with torch.no_grad():
        points = points.float().contiguous()

        B, N, _ = points.shape

        if max_points is not None and int(max_points) > 0 and N > int(max_points):
            idx = torch.linspace(
                0,
                N - 1,
                steps=int(max_points),
                device=points.device,
            ).long()
            points = points.index_select(1, idx)

        if normalize_points:
            points = _normalize_points_for_gate(points, eps=eps)

        B, N, _ = points.shape

        repsurf = _triangular_repsurf8(points, k=k)

        surf_mean = repsurf.mean(dim=1)
        surf_std = repsurf.std(dim=1, unbiased=False)
        surf_max = repsurf.amax(dim=1)
        surf_min = repsurf.amin(dim=1)

        bbox = points.amax(dim=1) - points.amin(dim=1)

        sorted_bbox, _ = torch.sort(bbox, dim=-1, descending=True)
        bbox_ratio = sorted_bbox / sorted_bbox[:, :1].clamp_min(eps)

        centered = points - points.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered) / float(max(N - 1, 1))
        eye = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
        cov = cov + eye * eps

        eig = torch.linalg.eigvalsh(cov)
        eig = eig.clamp_min(eps)
        eig, _ = torch.sort(eig, dim=-1, descending=True)

        l1 = eig[:, 0].clamp_min(eps)
        l2 = eig[:, 1]
        l3 = eig[:, 2]

        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        scattering = l3 / l1

        shape_index = torch.stack(
            [
                linearity,
                planarity,
                scattering,
            ],
            dim=-1,
        )

        desc = torch.cat(
            [
                surf_mean,
                surf_std,
                surf_max,
                surf_min,
                bbox,
                bbox_ratio,
                eig,
                shape_index,
            ],
            dim=-1,
        )

        desc = torch.nan_to_num(
            desc,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        )

    return desc


class GeometryGate(nn.Module):
    """
    Geometry-aware gate.

    It does not use UniAD backbone features.
    It uses raw / registered xyz points to decide which private expert to activate.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        k: int = 9,
        max_points: int = 2048,
        normalize_points: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_experts = int(num_experts)
        self.k = int(k)
        self.max_points = int(max_points)
        self.normalize_points = bool(normalize_points)
        self.temperature = float(temperature)

        desc_dim = 44

        self.net = nn.Sequential(
            nn.LayerNorm(desc_dim),
            nn.Linear(desc_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        desc = _geometry_descriptor(
            points,
            k=self.k,
            max_points=self.max_points,
            normalize_points=self.normalize_points,
        )

        logits = self.net(desc)

        temperature = max(float(self.temperature), 1.0e-6)
        probs = F.softmax(logits / temperature, dim=-1)

        return logits, probs


class UniADMoE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_experts: int = 4,
        class_key: str = "moe_class_id",
        gate_point_key: str = "moe_points",
        gate_point_keys: Optional[list[str]] = None,
        allow_raw_point_fallback: bool = False,

        # 新增：gate 输入类型
        gate_input: str = "geometry",

        # 新增：PointMAE feature gate 配置
        gate_feature_keys: Optional[list[str]] = None,
        gate_feature_dim: int = 384,
        gate_feature_desc: str = "mean_std_max_min",
        gate_feature_detach: bool = True,

        route_source: str = "input",
        mix_mode: str = "fixed",
        shared_weight: float = 0.5,
        private_weight: float = 0.5,
        init_private_from_shared: bool = True,
        train_route_mode: str = "soft_all",
        eval_route_mode: str = "hard_top1",
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.0,
        gate_k: int = 9,
        gate_max_points: int = 2048,
        gate_normalize_points: bool = True,
        gate_temperature: float = 1.0,
        load_balance_loss_weight: float = 0.0,
        class_balance_loss_weight: float = 0.01,
        entropy_loss_weight: float = 0.001,

        # 新增：用 class_to_expert 监督 gate
        target_gate_loss_weight: float = 0.0,
        class_to_expert: Optional[list[int]] = None,

        class_balance_ema_momentum: float = 0.95,
        debug: bool = False,
        **uniad_kwargs,
    ):

        super().__init__()

        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)

        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0.")

        self.class_key = str(class_key)
        self.gate_point_key = str(gate_point_key)
        self.allow_raw_point_fallback = bool(allow_raw_point_fallback)
        self.route_source = str(route_source).lower()
        self.mix_mode = str(mix_mode).lower()
        self.train_route_mode = str(train_route_mode).lower()
        self.eval_route_mode = str(eval_route_mode).lower()
        self.debug = bool(debug)

        self.gate_input = str(gate_input).lower()
        self.gate_feature_dim = int(gate_feature_dim)
        self.gate_feature_desc = str(gate_feature_desc).lower()
        self.gate_feature_detach = bool(gate_feature_detach)
        self.target_gate_loss_weight = float(target_gate_loss_weight)


        if self.train_route_mode not in ["soft_all", "hard_top1"]:
            raise ValueError(
                f"Unsupported train_route_mode={train_route_mode}. "
                "Use 'soft_all' or 'hard_top1'."
            )

        if self.eval_route_mode not in ["hard_top1"]:
            raise ValueError(
                f"Unsupported eval_route_mode={eval_route_mode}. "
                "Currently only 'hard_top1' is recommended."
            )

        self.load_balance_loss_weight = float(load_balance_loss_weight)
        self.class_balance_loss_weight = float(class_balance_loss_weight)
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.class_balance_ema_momentum = float(class_balance_ema_momentum)

        # gate 优先使用 PointMAE sampled center。
        # 这样 gate 看到的几何点和 UniAD 重构 token 对应。
        center_point_keys = [
            self.gate_point_key,
            "moe_points",

            "center",
            "centers",

            "registered_center",
            "registered_centers",
            "registered_xyz_center",
            "registered_xyz_centers",

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

        registered_point_keys = [
            "registered_points",
            "registered_xyz",
            "registered_pointcloud",
            "registered_pointcloud_xyz",
            "aligned_points",
            "aligned_xyz",
            "transformed_points",
            "transformed_xyz",
        ]

        raw_point_keys = [
            "points",
            "point",
            "pointcloud",
            "xyz",
            "coord",
            "coords",
            "raw_points",
            "raw_pointcloud",
        ]

        default_feature_keys = [
            # 最推荐：UniAD 内部常用的对齐特征
            "feature_align",

            # 常见 backbone output key
            "features",
            "feature",
            "x",
            "tokens",
            "token",
            "token_features",
            "pointmae_feature",
            "pointmae_features",

            # 兼容你 CPE / PointMAE 相关命名
            "xyz_features",
            "raw_xyz_features",
            "proto_feature",
            "raw_proto_feature",
            "global_feature",
            "cls_feature",
        ]

        if gate_feature_keys is not None:
            default_feature_keys = list(gate_feature_keys) + default_feature_keys

        dedup_feature_keys = []
        for k in default_feature_keys:
            if k not in dedup_feature_keys:
                dedup_feature_keys.append(k)

        self.gate_feature_keys = dedup_feature_keys


        default_point_keys = center_point_keys + registered_point_keys

        if self.allow_raw_point_fallback:
            default_point_keys = default_point_keys + raw_point_keys

        if gate_point_keys is not None:
            default_point_keys = list(gate_point_keys) + default_point_keys

        dedup_keys = []
        for k in default_point_keys:
            if k not in dedup_keys:
                dedup_keys.append(k)
        self.gate_point_keys = dedup_keys

        self.shared_expert = UniAD(**copy.deepcopy(uniad_kwargs))

        self.private_experts = nn.ModuleList(
            [
                UniAD(**copy.deepcopy(uniad_kwargs))
                for _ in range(self.num_experts)
            ]
        )

        if init_private_from_shared:
            shared_state = self.shared_expert.state_dict()
            for expert in self.private_experts:
                expert.load_state_dict(shared_state, strict=False)

        if self.gate_input in ["geometry", "point", "points", "xyz"]:
            self.gate = GeometryGate(
                num_experts=self.num_experts,
                hidden_dim=int(gate_hidden_dim),
                dropout=float(gate_dropout),
                k=int(gate_k),
                max_points=int(gate_max_points),
                normalize_points=bool(gate_normalize_points),
                temperature=float(gate_temperature),
            )

        elif self.gate_input in ["feature", "pointmae", "pointmae_feature"]:
            self.gate = FeatureGate(
                num_experts=self.num_experts,
                feature_dim=int(gate_feature_dim),
                desc_mode=str(gate_feature_desc),
                hidden_dim=int(gate_hidden_dim),
                dropout=float(gate_dropout),
                temperature=float(gate_temperature),
            )

        else:
            raise ValueError(
                f"Unsupported gate_input={gate_input}. "
                "Use geometry or feature."
            )


        if self.mix_mode == "fixed":
            total = float(shared_weight + private_weight)
            if total <= 0:
                raise ValueError("shared_weight + private_weight must be > 0.")

            self.register_buffer(
                "shared_w",
                torch.tensor(float(shared_weight) / total),
                persistent=False,
            )
            self.register_buffer(
                "private_w",
                torch.tensor(float(private_weight) / total),
                persistent=False,
            )

        elif self.mix_mode == "learnable_scalar":
            self.private_logit = nn.Parameter(torch.zeros(1))

        else:
            raise ValueError(
                f"Unsupported mix_mode={mix_mode}. "
                "Use fixed or learnable_scalar."
            )

        # EMA memory for class-balanced expert usage.
        # 用于 batch_size=1 时仍然能做按类别的长期均衡。
        self.register_buffer(
            "class_gate_ema",
            torch.zeros(self.num_classes, self.num_experts),
            persistent=False,
        )
        self.register_buffer(
            "class_seen",
            torch.zeros(self.num_classes),
            persistent=False,
        )

        mapping = torch.full(
            (self.num_classes,),
            -1,
            dtype=torch.long,
        )

        if class_to_expert is not None:
            if len(class_to_expert) != self.num_classes:
                raise ValueError(
                    f"len(class_to_expert)={len(class_to_expert)} "
                    f"but num_classes={self.num_classes}."
                )

            for c, e in enumerate(class_to_expert):
                if e is None:
                    continue

                e_int = int(e)

                if e_int < 0:
                    continue

                if e_int >= self.num_experts:
                    raise ValueError(
                        f"class_to_expert[{c}]={e_int} out of range. "
                        f"num_experts={self.num_experts}."
                    )

                mapping[c] = e_int

        self.register_buffer(
            "class_to_expert",
            mapping,
            persistent=False,
        )

    @torch.no_grad()
    def set_class_to_expert(self, mapping: Any) -> None:
        """
        mapping[c] = expert id for class c.
        -1 means this class has no target expert.
        """
        if not torch.is_tensor(mapping):
            mapping = torch.as_tensor(mapping, dtype=torch.long)

        mapping = mapping.view(-1).long().cpu()

        if int(mapping.numel()) != self.num_classes:
            raise RuntimeError(
                f"class_to_expert length mismatch: got {mapping.numel()}, "
                f"expected {self.num_classes}."
            )

        valid = mapping >= 0

        if bool(valid.any()):
            if int(mapping[valid].max().item()) >= self.num_experts:
                raise RuntimeError(
                    f"class_to_expert max={int(mapping[valid].max().item())} "
                    f"but num_experts={self.num_experts}."
                )

        self.class_to_expert.copy_(mapping.to(self.class_to_expert.device))

    def _resolve_gate_feature(
        self,
        x: Any,
        shared_out: dict,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Prefer feature from backbone_out x.
        If not found, fallback to shared_out['feature_align'].
        """
        feat = _find_tensor_by_keys(x, self.gate_feature_keys)

        if feat is None:
            feat = _find_tensor_by_keys(
                shared_out,
                self.gate_feature_keys + ["feature_align"],
            )

        if feat is None:
            raise RuntimeError(
                "UniADMoE cannot find PointMAE feature for feature gate. "
                f"Expected one of keys: {self.gate_feature_keys}. "
                "Please print backbone_out.keys(), then add the actual feature key "
                "to gate_feature_keys. A common valid key is 'feature_align'."
            )

        feat = _as_btn_features(
            feat,
            batch_size=batch_size,
            feature_dim=self.gate_feature_dim,
            key_name="gate_feature",
        )

        if self.gate_feature_detach:
            feat = feat.detach()

        return feat

    def _target_gate_loss(
        self,
        gate_logits: torch.Tensor,
        class_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Supervise gate with class_to_expert.

        This is the key for:
          1. similar_cluster experts
          2. diverse_group experts
        """
        if class_id is None:
            return gate_logits.new_tensor(0.0)

        if self.target_gate_loss_weight == 0.0:
            return gate_logits.new_tensor(0.0)

        mapping = self.class_to_expert.to(device=class_id.device)
        target = mapping[class_id.long().view(-1)]

        valid = target >= 0

        if not bool(valid.any()):
            return gate_logits.new_tensor(0.0)

        return F.cross_entropy(
            gate_logits[valid.to(gate_logits.device)],
            target[valid].to(device=gate_logits.device, dtype=torch.long),
        )


    def _get_mix_weights(self, device: torch.device):
        if self.mix_mode == "fixed":
            return self.shared_w.to(device), self.private_w.to(device)

        private_w = torch.sigmoid(self.private_logit).to(device)
        shared_w = 1.0 - private_w
        return shared_w, private_w

    def _resolve_class_id_optional(
        self,
        x: Any,
        shared_out: dict,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        class_id = None

        if isinstance(x, dict):
            if self.class_key in x:
                class_id = x[self.class_key]
            else:
                for key in [
                    "moe_class_id",
                    "registration_cls_label",
                    "pred_cls_label",
                    "cls_label",
                    "class_label",
                    "category_id",
                ]:
                    if key in x:
                        class_id = x[key]
                        break

        if class_id is None and self.route_source == "shared_pred":
            if "cls_pred" not in shared_out:
                raise RuntimeError(
                    "route_source='shared_pred' requires shared_out['cls_pred']."
                )
            class_id = torch.argmax(shared_out["cls_pred"].detach(), dim=1)

        if class_id is None:
            return None

        return self._normalize_class_id(class_id, batch_size, device)

    def _normalize_class_id(
        self,
        class_id: Any,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(class_id):
            class_id = torch.as_tensor(class_id)

        class_id = class_id.to(device=device).long().view(-1)

        if class_id.numel() == 1 and int(batch_size) > 1:
            class_id = class_id.expand(int(batch_size))

        if int(class_id.numel()) != int(batch_size):
            raise RuntimeError(
                f"class_id length mismatch: got {class_id.numel()}, "
                f"but batch_size={batch_size}."
            )

        if torch.any(class_id < 0) or torch.any(class_id >= self.num_classes):
            raise RuntimeError(
                f"class_id out of range. Valid range=[0, {self.num_classes - 1}], "
                f"got min={int(class_id.min())}, max={int(class_id.max())}."
            )

        return class_id

    def _resolve_gate_points(self, x: Any, batch_size: int) -> torch.Tensor:
        points = _find_tensor_by_keys(x, self.gate_point_keys)

        if points is None:
            raise RuntimeError(
                "UniADMoE cannot find points for geometry gate. "
                f"Expected one of keys: {self.gate_point_keys}. "
                "For PointMAE + UniAD, the recommended solution is: "
                "set backbone_out['moe_points'] = backbone_out['center'] before reconstruction. "
                "If you really want to fallback to raw pointcloud, set "
                "allow_raw_point_fallback=True in UniADMoE config."
            )


        points = _as_bnc_points(
            points,
            batch_size=batch_size,
            key_name=self.gate_point_key,
        )

        return points

    def _class_balance_loss_with_ema(
        self,
        gate_probs: torch.Tensor,
        class_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Class-balanced expert usage loss.

        batch_size=1 时，普通 batch-level balance 基本没意义。
        所以这里维护 class -> gate probability 的 EMA，
        让已经出现过的类别在专家维度上整体更均衡。

        This loss does not force:
            class c -> expert c

        It only encourages:
            different classes should not all collapse to the same expert.
        """
        if class_id is None:
            return gate_probs.new_tensor(0.0)

        if self.num_experts <= 1:
            return gate_probs.new_tensor(0.0)

        unique_classes = torch.unique(class_id)

        old_sum = gate_probs.new_zeros(self.num_experts)
        old_count = 0

        current_sum = gate_probs.new_zeros(self.num_experts)
        current_count = 0

        current_class_set = set(int(c.item()) for c in unique_classes)

        for c in range(self.num_classes):
            if c in current_class_set:
                continue

            if float(self.class_seen[c].item()) > 0:
                old_sum = old_sum + self.class_gate_ema[c].to(gate_probs.device)
                old_count += 1

        for c in unique_classes:
            mask = class_id == c
            p = gate_probs[mask].mean(dim=0)
            current_sum = current_sum + p
            current_count += 1

        denom = max(old_count + current_count, 1)
        class_usage = (old_sum + current_sum) / float(denom)

        loss = self.num_experts * torch.sum(class_usage ** 2) - 1.0
        return loss

    @torch.no_grad()
    def _update_class_gate_ema(
        self,
        gate_probs: torch.Tensor,
        class_id: Optional[torch.Tensor],
    ) -> None:
        if class_id is None:
            return

        momentum = float(self.class_balance_ema_momentum)

        probs = gate_probs.detach()
        unique_classes = torch.unique(class_id)

        for c in unique_classes:
            c_int = int(c.item())
            mask = class_id == c
            p = probs[mask].mean(dim=0).to(self.class_gate_ema.device)

            if float(self.class_seen[c_int].item()) <= 0:
                self.class_gate_ema[c_int].copy_(p)
                self.class_seen[c_int].fill_(1.0)
            else:
                self.class_gate_ema[c_int].mul_(momentum).add_(
                    p,
                    alpha=1.0 - momentum,
                )

    def _compute_moe_aux_loss(
        self,
        gate_probs: torch.Tensor,
        class_id: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_experts <= 1:
            zero = gate_probs.new_tensor(0.0)
            return zero, zero, zero, zero

        eps = 1.0e-8

        # Sample-level load balance.
        # 注意：batch_size=1 时不建议给它较大权重，否则会把每个样本都推成均匀分配。
        importance = gate_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(importance ** 2) - 1.0

        # Class-level balance with EMA.
        class_balance_loss = self._class_balance_loss_with_ema(
            gate_probs=gate_probs,
            class_id=class_id,
        )

        # Low entropy encourages one sample to prefer one expert.
        # 这个和 balance loss 搭配使用：
        #   balance: 防止所有样本塌到同一个专家
        #   entropy: 防止每个样本都平均用所有专家
        entropy = -torch.sum(
            gate_probs * torch.log(gate_probs.clamp_min(eps)),
            dim=1,
        ).mean()

        entropy = entropy / math.log(float(self.num_experts))

        total = (
            self.load_balance_loss_weight * load_balance_loss
            + self.class_balance_loss_weight * class_balance_loss
            + self.entropy_loss_weight * entropy
        )

        return total, load_balance_loss, class_balance_loss, entropy

    def _run_private_soft_all(
        self,
        x: Any,
        gate_probs: torch.Tensor,
        shared_feature_rec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training mode.

        Run all private experts and use gate_probs to mix them.

        This is more expensive, but gate can receive gradients from reconstruction loss.
        Since num_experts is small, this is usually acceptable.
        """
        batch_size = int(gate_probs.shape[0])
        private_feature_rec = torch.zeros_like(shared_feature_rec)

        for expert_idx, expert in enumerate(self.private_experts):
            out_e = expert(x)

            if "feature_rec" not in out_e:
                raise RuntimeError("Private UniAD output must contain 'feature_rec'.")

            rec_e = out_e["feature_rec"]

            w = gate_probs[:, expert_idx].to(
                device=rec_e.device,
                dtype=rec_e.dtype,
            )

            view_shape = [batch_size] + [1] * (rec_e.dim() - 1)
            w = w.view(*view_shape)

            private_feature_rec = private_feature_rec + w * rec_e

        return private_feature_rec

    def _run_private_hard_top1(
        self,
        x: Any,
        expert_id: torch.Tensor,
        batch_size: int,
        shared_feature_rec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference mode.

        Only run activated private experts.
        """
        private_feature_rec = torch.empty_like(shared_feature_rec)

        unique_experts = torch.unique(expert_id)

        if self.debug:
            print(
                "[UniADMoE] batch_size={}, activate private experts={}".format(
                    batch_size,
                    [int(e.item()) for e in unique_experts],
                )
            )

        for e in unique_experts:
            e_int = int(e.item())
            mask = expert_id == e

            x_part = _slice_by_mask(x, mask, batch_size)
            out_part = self.private_experts[e_int](x_part)

            if "feature_rec" not in out_part:
                raise RuntimeError("Private UniAD output must contain 'feature_rec'.")

            local_mask = mask.to(device=private_feature_rec.device, dtype=torch.bool)
            private_feature_rec[local_mask] = out_part["feature_rec"]

        return private_feature_rec

    def forward(self, x: Any) -> dict:
        if not isinstance(x, dict):
            raise RuntimeError(
                "UniADMoE expects dict input, same as UniAD.forward(input)."
            )

        batch_size = _get_batch_size(x)
        if batch_size is None:
            raise RuntimeError("Cannot infer batch size in UniADMoE.")

        device = _get_device(x)

        # 1. shared expert always runs
        shared_out = self.shared_expert(x)

        if not isinstance(shared_out, dict):
            raise RuntimeError("UniADMoE expects UniAD output to be dict.")

        if "feature_rec" not in shared_out:
            raise RuntimeError("UniAD output must contain 'feature_rec'.")

        if "feature_align" not in shared_out:
            raise RuntimeError("UniAD output must contain 'feature_align'.")

        shared_feature_rec = shared_out["feature_rec"]
        feature_align = shared_out["feature_align"]

        # 2. gate
        if self.gate_input in ["feature", "pointmae", "pointmae_feature"]:
            gate_feature = self._resolve_gate_feature(
                x=x,
                shared_out=shared_out,
                batch_size=batch_size,
            )
            gate_feature = gate_feature.to(device=device)

            gate_logits, gate_probs = self.gate(gate_feature)

        else:
            gate_points = self._resolve_gate_points(x, batch_size=batch_size)
            gate_points = gate_points.to(device=device)

            gate_logits, gate_probs = self.gate(gate_points)

        gate_logits = gate_logits.to(device=device)
        gate_probs = gate_probs.to(device=device)


        expert_id = torch.argmax(gate_probs.detach(), dim=1)

        # 3. optional class id for class-balanced regularization
        class_id = self._resolve_class_id_optional(
            x=x,
            shared_out=shared_out,
            batch_size=batch_size,
            device=device,
        )

        # 4. private expert route
        if self.training and self.train_route_mode == "soft_all":
            private_feature_rec = self._run_private_soft_all(
                x=x,
                gate_probs=gate_probs,
                shared_feature_rec=shared_feature_rec,
            )
        else:
            private_feature_rec = self._run_private_hard_top1(
                x=x,
                expert_id=expert_id,
                batch_size=batch_size,
                shared_feature_rec=shared_feature_rec,
            )

        # 5. mix shared/private reconstruction
        ws, wp = self._get_mix_weights(device)

        feature_rec = ws * shared_feature_rec + wp * private_feature_rec

        pred = torch.sqrt(
            torch.sum(
                (feature_rec - feature_align) ** 2,
                dim=1,
                keepdim=True,
            )
            + 1.0e-12
        )

        # 6. MoE aux loss
        moe_aux_loss, load_balance_loss, class_balance_loss, entropy_loss = (
            self._compute_moe_aux_loss(
                gate_probs=gate_probs,
                class_id=class_id,
            )
        )

        target_gate_loss = self._target_gate_loss(
            gate_logits=gate_logits,
            class_id=class_id,
        )

        moe_aux_loss = (
            moe_aux_loss
            + self.target_gate_loss_weight * target_gate_loss
        )


        if self.training:
            self._update_class_gate_ema(
                gate_probs=gate_probs,
                class_id=class_id,
            )

        out = dict(shared_out)

        out["feature_rec"] = feature_rec
        out["feature_align"] = feature_align
        out["pred"] = pred

        # For logging / debugging
        out["moe_gate_logits"] = gate_logits.detach()
        out["moe_gate_probs"] = gate_probs.detach()
        out["moe_expert_id"] = expert_id.detach()

        if class_id is not None:
            out["moe_class_id"] = class_id.detach()

        # Important: this tensor keeps gradient.
        # 你的 trainer / criterion 需要把它加到总 loss 里才会生效。
        out["moe_aux_loss"] = moe_aux_loss

        out["moe_load_balance_loss"] = load_balance_loss.detach()
        out["moe_class_balance_loss"] = class_balance_loss.detach()
        out["moe_entropy_loss"] = entropy_loss.detach()
        out["moe_target_gate_loss"] = target_gate_loss.detach()


        # cls_pred 保留 shared expert 的输出，避免 evaluator 接口变化
        if "cls_pred" in shared_out:
            out["cls_pred"] = shared_out["cls_pred"]

        return out
