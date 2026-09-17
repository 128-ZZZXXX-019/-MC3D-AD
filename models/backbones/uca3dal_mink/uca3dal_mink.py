# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from .mink import Mink_unet


logger = logging.getLogger("global_logger")

def save_point_cloud_xyz(coord: np.ndarray, filename: str):
    """
    保存点云到 txt，每行 x y z
    coord: [N,3] numpy array
    filename: 保存文件名，例如 'cloud1.txt'
    """
    save_dir = "/data/data_lhq/point_cloud_check/"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, filename)
    
    if coord.shape[1] != 3:
        raise ValueError(f"Point cloud shape should be [N,3], got {coord.shape}")
    
    np.savetxt(save_path, coord, fmt="%.6f")
    print(f"Saved point cloud to {save_path}")

def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        return {
            k[len("module.") :]: v
            for k, v in state_dict.items()
        }

    return state_dict


def _select_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "base_model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


class UCA3DALMinkBackbone(nn.Module):
    """
    Adapter for UCA-3DAL Stage-1 CPE backbone.

    Input from your project:
        input_dict["pointcloud"]: [B, N, 3] or [B, 3, N]

    Internal CPE-style pipeline:
        CenterShift
        optional train augmentation
        NormalizeCoord
        optional partial-view crop
        MinkowskiEngine sparse_quantize
        MinkUNet34C
        GlobalAvgPooling
        optional CPE projection head

    Output for your existing classifier:
        raw_xyz_features: [B, C, 1]
        xyz_features:     [B, C, 1]
        is_registered:    False

    Notes:
        - This backbone does not do PointMAE grouping.
        - This backbone does not do your old template registration.
        - It intentionally exposes a PointMAE-compatible feature dictionary,
          so PointMAEProtoClsHead can be reused without changing train_cls.py.

    Partial-view crop:
        - Only active during training: self.training and partial_crop.
        - It samples a virtual view direction and keeps points closer to that view.
        - It does not force exactly 50% points; keep ratio is randomly sampled.
        - Crop is applied after normalization so the partial cloud remains in
          the full-object normalized coordinate frame.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        proj_dim: int = 128,
        arch: str = "MinkUNet34C",
        voxel_size: float = 0.03,
        use_cpe_projection: bool = True,
        checkpoint_path: str = "",
        checkpoint_strict: bool = False,
        load_projection: bool = True,
        feature_mode: str = "coord",

        # CPE / Real3D preprocessing.
        center_shift: bool = True,
        center_shift_apply_z: bool = True,
        normalize_coord: bool = True,
        normalize_eps: float = 1.0e-12,

        # Optional CPE-style training augmentation.
        # Original contrast_aug:
        #   CenterShift -> RandomRotate z/y/x -> RandomScale -> RandomJitter -> NormalizeCoord
        train_augment: bool = False,
        random_rotate: bool = True,
        random_rotate_angle_min: float = -1.0,
        random_rotate_angle_max: float = 1.0,
        random_scale: bool = True,
        random_scale_low: float = 0.9,
        random_scale_high: float = 1.1,
        random_scale_p: float = 0.5,
        random_jitter: bool = True,
        random_jitter_sigma: float = 0.005,
        random_jitter_clip: float = 0.02,
        random_jitter_p: float = 0.5,

        # Real3D_ADLoader-style training augmentation.
        real3d_augment: bool = False,
        real3d_fps_npoints: int = 1024,
        real3d_first_rotate: bool = True,
        real3d_second_rotate: bool = True,
        real3d_dropout: bool = True,
        real3d_dropout_max_ratio: float = 0.875,
        real3d_scale: bool = True,
        real3d_scale_low: float = 0.8,
        real3d_scale_high: float = 1.25,
        real3d_shift: bool = True,
        real3d_shift_range: float = 0.1,


        # Partial-view crop augmentation.
        # This simulates partial scans from virtual viewpoints.
        partial_crop: bool = False,
        partial_crop_p: float = 0.35,
        partial_keep_ratio_min: float = 0.65,
        partial_keep_ratio_max: float = 0.90,
        partial_min_points: int = 128,
        partial_view_jitter: float = 0.20,
        partial_soft_width: float = 0.02,
        partial_bottom_view_weight: float = 0.20,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj_dim = int(proj_dim)
        self.arch = str(arch)
        self.voxel_size = float(voxel_size)
        self.use_cpe_projection = bool(use_cpe_projection)
        self.checkpoint_path = str(checkpoint_path or "")
        self.checkpoint_strict = bool(checkpoint_strict)
        self.load_projection = bool(load_projection)
        self.feature_mode = str(feature_mode).lower()

        self.center_shift = bool(center_shift)
        self.center_shift_apply_z = bool(center_shift_apply_z)
        self.normalize_coord = bool(normalize_coord)
        self.normalize_eps = float(normalize_eps)

        self.train_augment = bool(train_augment)
        self.random_rotate = bool(random_rotate)
        self.random_rotate_angle_min = float(random_rotate_angle_min)
        self.random_rotate_angle_max = float(random_rotate_angle_max)
        self.random_scale = bool(random_scale)
        self.random_scale_low = float(random_scale_low)
        self.random_scale_high = float(random_scale_high)
        self.random_scale_p = float(random_scale_p)
        self.random_jitter = bool(random_jitter)
        self.random_jitter_sigma = float(random_jitter_sigma)
        self.random_jitter_clip = float(random_jitter_clip)
        self.random_jitter_p = float(random_jitter_p)

        self.real3d_augment = bool(real3d_augment)
        self.real3d_fps_npoints = int(real3d_fps_npoints)
        self.real3d_first_rotate = bool(real3d_first_rotate)
        self.real3d_second_rotate = bool(real3d_second_rotate)
        self.real3d_dropout = bool(real3d_dropout)
        self.real3d_dropout_max_ratio = float(real3d_dropout_max_ratio)
        self.real3d_scale = bool(real3d_scale)
        self.real3d_scale_low = float(real3d_scale_low)
        self.real3d_scale_high = float(real3d_scale_high)
        self.real3d_shift = bool(real3d_shift)
        self.real3d_shift_range = float(real3d_shift_range)


        self.partial_crop = bool(partial_crop)
        self.partial_crop_p = float(partial_crop_p)
        self.partial_keep_ratio_min = float(partial_keep_ratio_min)
        self.partial_keep_ratio_max = float(partial_keep_ratio_max)
        self.partial_min_points = int(partial_min_points)
        self.partial_view_jitter = float(partial_view_jitter)
        self.partial_soft_width = float(partial_soft_width)
        self.partial_bottom_view_weight = float(partial_bottom_view_weight)

        if self.feature_mode in ["coord", "xyz"] and self.in_channels != 3:
            raise ValueError(
                "feature_mode='coord' uses xyz as voxel feature, so in_channels must be 3. "
                f"Got in_channels={self.in_channels}."
            )

        if not (0.0 <= self.partial_crop_p <= 1.0):
            raise ValueError(
                f"partial_crop_p must be in [0, 1], got {self.partial_crop_p}."
            )

        if not (
            0.0 < self.partial_keep_ratio_min
            <= self.partial_keep_ratio_max
            <= 1.0
        ):
            raise ValueError(
                "Require 0 < partial_keep_ratio_min <= partial_keep_ratio_max <= 1. "
                f"Got min={self.partial_keep_ratio_min}, "
                f"max={self.partial_keep_ratio_max}."
            )

        if self.partial_min_points < 1:
            raise ValueError(
                f"partial_min_points must be >= 1, got {self.partial_min_points}."
            )

        if self.partial_view_jitter < 0.0:
            raise ValueError(
                f"partial_view_jitter must be >= 0, got {self.partial_view_jitter}."
            )

        if self.partial_soft_width < 0.0:
            raise ValueError(
                f"partial_soft_width must be >= 0, got {self.partial_soft_width}."
            )

        if self.partial_bottom_view_weight < 0.0:
            raise ValueError(
                "partial_bottom_view_weight must be >= 0, "
                f"got {self.partial_bottom_view_weight}."
            )

        self.backbone = Mink_unet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            D=3,
            arch=self.arch,
        )

        self.global_pool = ME.MinkowskiGlobalAvgPooling()

        # Same projection head as UCA-3DAL network/cpe.py.
        if self.use_cpe_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.out_channels, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.proj_dim, bias=True),
            )
            self.output_channels = self.proj_dim
        else:
            self.proj = nn.Identity()
            self.output_channels = self.out_channels

        if self.checkpoint_path:
            self.load_cpe_checkpoint(
                self.checkpoint_path,
                strict=self.checkpoint_strict,
                load_projection=self.load_projection,
            )

    def load_cpe_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = False,
        load_projection: bool = True,
    ) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"CPE checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _select_state_dict(ckpt)
        state_dict = _strip_module_prefix(state_dict)

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Invalid checkpoint format: {checkpoint_path}. "
                "Expected a state_dict or a dict containing key 'model'."
            )

        # If only using the raw MinkUNet feature, ignore CPE projection weights.
        if (not self.use_cpe_projection) or (not load_projection):
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }

        incompatible = self.load_state_dict(state_dict, strict=strict)

        msg = (
            f"Loaded UCA-3DAL CPE checkpoint from {checkpoint_path}. "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

        if logger.handlers:
            logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def _normalize_xyz_shape(xyz: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(xyz):
            xyz = torch.as_tensor(xyz, dtype=torch.float32)

        if xyz.dim() == 2:
            if xyz.shape[-1] == 3:
                xyz = xyz.unsqueeze(0)
            elif xyz.shape[0] == 3:
                xyz = xyz.t().unsqueeze(0)
            else:
                raise RuntimeError(
                    f"Cannot infer point dimension from 2D shape={tuple(xyz.shape)}. "
                    "Expected [N,3] or [3,N]."
                )

        if xyz.dim() != 3:
            raise RuntimeError(
                f"pointcloud must be 3D, got shape={tuple(xyz.shape)}"
            )

        if xyz.shape[-1] == 3:
            return xyz.contiguous()

        if xyz.shape[1] == 3:
            return xyz.transpose(1, 2).contiguous()

        raise RuntimeError(
            f"Cannot infer point dimension from shape={tuple(xyz.shape)}. "
            "Expected [B,N,3] or [B,3,N]."
        )

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            return torch.device("cpu")

    @staticmethod
    def _me_device_str(device: torch.device) -> str:
        if device.type == "cuda":
            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            return f"cuda:{device_index}"
        return "cpu"

    @staticmethod
    def _check_cuda_if_needed(x: torch.Tensor, name: str, model_device: torch.device) -> None:
        if model_device.type == "cuda" and not x.is_cuda:
            raise RuntimeError(
                f"{name} is on CPU, but model is on {model_device}. "
                "This would reintroduce CPU/GPU transfer. "
                "Please check your MinkowskiEngine build/version and whether "
                "ME.utils.sparse_quantize / sparse_collate supports device='cuda'."
            )

    def _center_shift_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord

        xyz_min = coord.amin(dim=0)
        xyz_max = coord.amax(dim=0)

        if self.center_shift_apply_z:
            shift = torch.stack(
                [
                    (xyz_min[0] + xyz_max[0]) * 0.5,
                    (xyz_min[1] + xyz_max[1]) * 0.5,
                    xyz_min[2],
                ],
                dim=0,
            )
        else:
            shift = torch.stack(
                [
                    (xyz_min[0] + xyz_max[0]) * 0.5,
                    (xyz_min[1] + xyz_max[1]) * 0.5,
                    coord.new_zeros(()),
                ],
                dim=0,
            )

        return coord - shift.view(1, 3)

    def _normalize_coord_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord

        coord = coord - coord.mean(dim=0, keepdim=True)
        radius = torch.linalg.norm(coord, dim=1).max()
        radius = radius.clamp_min(self.normalize_eps)

        return coord / radius

    @staticmethod
    def _bbox_center_torch(coord: torch.Tensor) -> torch.Tensor:
        xyz_min = coord.amin(dim=0)
        xyz_max = coord.amax(dim=0)
        return (xyz_min + xyz_max) * 0.5

    def _random_rotate_torch(
        self,
        coord: torch.Tensor,
        axis: str,
        center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        angle = torch.empty(
            (),
            device=coord.device,
            dtype=coord.dtype,
        ).uniform_(
            self.random_rotate_angle_min,
            self.random_rotate_angle_max,
        )
        angle = angle * float(np.pi)

        rot_cos = torch.cos(angle)
        rot_sin = torch.sin(angle)

        rot = coord.new_zeros((3, 3))

        if axis == "x":
            rot[0, 0] = 1.0
            rot[1, 1] = rot_cos
            rot[1, 2] = -rot_sin
            rot[2, 1] = rot_sin
            rot[2, 2] = rot_cos
        elif axis == "y":
            rot[0, 0] = rot_cos
            rot[0, 2] = rot_sin
            rot[1, 1] = 1.0
            rot[2, 0] = -rot_sin
            rot[2, 2] = rot_cos
        elif axis == "z":
            rot[0, 0] = rot_cos
            rot[0, 1] = -rot_sin
            rot[1, 0] = rot_sin
            rot[1, 1] = rot_cos
            rot[2, 2] = 1.0
        else:
            raise ValueError(f"Unsupported rotate axis={axis}")

        if center is None:
            center = self._bbox_center_torch(coord)
        else:
            center = center.to(device=coord.device, dtype=coord.dtype)

        coord = coord - center.view(1, 3)
        coord = coord @ rot.t()
        coord = coord + center.view(1, 3)

        return coord.contiguous()

    def _maybe_pose_augment_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.train_augment):
            return coord.contiguous()

        if self.random_rotate:
            coord = self._random_rotate_torch(
                coord,
                axis="z",
                center=coord.new_zeros(3),
            )
            coord = self._random_rotate_torch(coord, axis="y", center=None)
            coord = self._random_rotate_torch(coord, axis="x", center=None)

        if self.random_scale:
            do_scale = (
                torch.rand((), device=coord.device) < self.random_scale_p
            ).to(dtype=coord.dtype)

            scale = torch.empty(
                (),
                device=coord.device,
                dtype=coord.dtype,
            ).uniform_(
                self.random_scale_low,
                self.random_scale_high,
            )

            coord = coord * (1.0 + do_scale * (scale - 1.0))

        return coord.contiguous()

    def _maybe_jitter_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.train_augment):
            return coord.contiguous()

        if self.random_jitter:
            do_jitter = (
                torch.rand((), device=coord.device) < self.random_jitter_p
            ).to(dtype=coord.dtype)

            jitter = torch.randn_like(coord) * self.random_jitter_sigma
            jitter = torch.clamp(
                jitter,
                min=-self.random_jitter_clip,
                max=self.random_jitter_clip,
            )

            coord = coord + do_jitter * jitter

        return coord.contiguous()

    def _sample_partial_view_dir_torch(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_dirs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.5],
                [1.0, -1.0, 0.5],
                [-1.0, 1.0, 0.5],
                [-1.0, -1.0, 0.5],
                [0.0, 0.0, -1.0],
            ],
            device=device,
            dtype=dtype,
        )

        weights = torch.tensor(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                self.partial_bottom_view_weight,
            ],
            device=device,
            dtype=dtype,
        )

        weight_sum = weights.sum()
        weights = torch.where(
            weight_sum > 0.0,
            weights / weight_sum.clamp_min(1.0e-12),
            torch.full_like(weights, 1.0 / float(weights.numel())),
        )

        idx = torch.multinomial(weights, num_samples=1, replacement=True)
        direction = base_dirs.index_select(0, idx).squeeze(0)

        if self.partial_view_jitter > 0.0:
            direction = direction + torch.randn(
                3,
                device=device,
                dtype=dtype,
            ) * self.partial_view_jitter

        raw_norm = torch.linalg.norm(direction)
        fallback = torch.tensor(
            [1.0, 0.0, 0.0],
            device=device,
            dtype=dtype,
        )

        direction = torch.where(
            raw_norm < 1.0e-6,
            fallback,
            direction / raw_norm.clamp_min(1.0e-6),
        )

        return direction.contiguous()

    def _partial_view_crop_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord

        num_points = int(coord.shape[0])

        if num_points <= self.partial_min_points:
            return coord.contiguous()

        # 这里用 CPU RNG 只决定一个 Python int，不搬运点云数据，不会造成 CPU/GPU 大量传输。
        keep_ratio = float(
            np.random.uniform(
                self.partial_keep_ratio_min,
                self.partial_keep_ratio_max,
            )
        )

        target_k = int(round(float(num_points) * keep_ratio))
        target_k = max(self.partial_min_points, target_k)
        target_k = min(num_points, target_k)

        if target_k >= num_points:
            return coord.contiguous()

        view_dir = self._sample_partial_view_dir_torch(
            coord.device,
            coord.dtype,
        )

        score = coord @ view_dir

        if self.partial_soft_width > 0.0:
            coord_centered = coord - coord.mean(dim=0, keepdim=True)
            coord_radius = torch.linalg.norm(coord_centered, dim=1).max()
            coord_radius = coord_radius.clamp_min(1.0e-6)

            score = score + torch.randn_like(score) * (
                self.partial_soft_width * coord_radius
            )

        keep_idx = torch.topk(
            score,
            k=target_k,
            largest=True,
            sorted=False,
        ).indices

        keep_idx = torch.sort(keep_idx).values

        return coord.index_select(0, keep_idx).contiguous()

    def _maybe_partial_crop_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.partial_crop):
            return coord.contiguous()

        # 同样，这里只是 CPU RNG 决定是否 crop，不涉及点云 tensor 回 CPU。
        if float(np.random.rand()) >= self.partial_crop_p:
            return coord.contiguous()

        return self._partial_view_crop_torch(coord)

    def _real3d_rotate_yz_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord.contiguous()

        coord = coord.clone()

        # 1) Random rotation around y axis.
        rotation_angle = torch.rand(
            (),
            device=coord.device,
            dtype=coord.dtype,
        ) * (2.0 * float(np.pi))

        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)

        rotation_matrix_y = coord.new_zeros((3, 3))
        rotation_matrix_y[0, 0] = cosval
        rotation_matrix_y[0, 2] = sinval
        rotation_matrix_y[1, 1] = 1.0
        rotation_matrix_y[2, 0] = -sinval
        rotation_matrix_y[2, 2] = cosval

        coord = coord @ rotation_matrix_y

        # 2) Random rotation around z axis.
        rotation_angle = torch.rand(
            (),
            device=coord.device,
            dtype=coord.dtype,
        ) * (2.0 * float(np.pi))

        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)

        rotation_matrix_z = coord.new_zeros((3, 3))
        rotation_matrix_z[0, 0] = cosval
        rotation_matrix_z[0, 1] = sinval
        rotation_matrix_z[1, 0] = -sinval
        rotation_matrix_z[1, 1] = cosval
        rotation_matrix_z[2, 2] = 1.0

        coord = coord @ rotation_matrix_z

        return coord.contiguous()

    def _real3d_farthest_point_sample_torch(
        self,
        point: torch.Tensor,
        npoint: int,
    ) -> torch.Tensor:
        if npoint <= 0:
            return point.contiguous()

        point = point.contiguous()

        N = int(point.shape[0])
        if N == 0:
            return point

        xyz = point[:, :3]

        centroids = torch.empty(
            int(npoint),
            device=point.device,
            dtype=torch.long,
        )

        distance = torch.full(
            (N,),
            1.0e10,
            device=point.device,
            dtype=point.dtype,
        )

        farthest = torch.randint(
            low=0,
            high=N,
            size=(1,),
            device=point.device,
            dtype=torch.long,
        )

        # 注意：这里没有 .item()，不会每一步强制 GPU 同步到 CPU。
        for i in range(int(npoint)):
            centroids[i] = farthest[0]

            centroid = xyz.index_select(0, farthest).view(1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)

            distance = torch.minimum(distance, dist)
            farthest = torch.argmax(distance).view(1)

        return point.index_select(0, centroids).contiguous()

    def _real3d_random_point_dropout_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord.contiguous()

        dropout_ratio = torch.rand(
            (),
            device=coord.device,
            dtype=coord.dtype,
        ) * self.real3d_dropout_max_ratio

        drop_mask = torch.rand(
            (coord.shape[0],),
            device=coord.device,
            dtype=coord.dtype,
        ) <= dropout_ratio

        first_point = coord[:1, :].expand_as(coord)

        coord = torch.where(
            drop_mask.view(-1, 1),
            first_point,
            coord,
        )

        return coord.contiguous()

    def _real3d_random_scale_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord.contiguous()

        scale = torch.empty(
            (),
            device=coord.device,
            dtype=coord.dtype,
        ).uniform_(
            self.real3d_scale_low,
            self.real3d_scale_high,
        )

        return (coord * scale).contiguous()

    def _real3d_shift_torch(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.numel() == 0:
            return coord.contiguous()

        shifts = torch.empty(
            (1, 3),
            device=coord.device,
            dtype=coord.dtype,
        ).uniform_(
            -self.real3d_shift_range,
            self.real3d_shift_range,
        )

        return (coord + shifts).contiguous()

    def _real3d_preprocess_torch(self, coord: torch.Tensor) -> torch.Tensor:
        coord = coord.to(dtype=torch.float32).contiguous().clone()

        if coord.dim() != 2 or coord.shape[1] != 3:
            raise RuntimeError(
                f"Real3D preprocess expects [N,3], got {tuple(coord.shape)}."
            )

        if coord.shape[0] == 0:
            raise RuntimeError("Empty point cloud is not supported.")

        if self.training:
            coord = self._normalize_coord_torch(coord)

            if self.real3d_first_rotate:
                coord = self._real3d_rotate_yz_torch(coord)

            if self.real3d_fps_npoints > 0:
                coord = self._real3d_farthest_point_sample_torch(
                    coord,
                    self.real3d_fps_npoints,
                )

            coord = self._normalize_coord_torch(coord)

            if self.real3d_second_rotate:
                coord = self._real3d_rotate_yz_torch(coord)

            if self.real3d_dropout:
                coord = self._real3d_random_point_dropout_torch(coord)

            if self.real3d_scale:
                coord = self._real3d_random_scale_torch(coord)

            if self.real3d_shift:
                coord = self._real3d_shift_torch(coord)

        else:
            if self.real3d_fps_npoints > 0:
                coord = self._real3d_farthest_point_sample_torch(
                    coord,
                    self.real3d_fps_npoints,
                )

            coord = self._normalize_coord_torch(coord)

        return coord.to(dtype=torch.float32).contiguous()

    def _preprocess_one_torch(self, coord: torch.Tensor) -> torch.Tensor:
        coord = coord.to(dtype=torch.float32).contiguous().clone()

        if coord.dim() != 2 or coord.shape[1] != 3:
            raise RuntimeError(
                f"Each point cloud must have shape [N,3], got {tuple(coord.shape)}"
            )

        if coord.shape[0] == 0:
            raise RuntimeError("Empty point cloud is not supported.")

        if self.real3d_augment:
            return self._real3d_preprocess_torch(coord)

        if self.center_shift:
            coord = self._center_shift_torch(coord)

        coord = self._maybe_pose_augment_torch(coord)
        coord = self._maybe_partial_crop_torch(coord)

        if coord.shape[0] == 0:
            raise RuntimeError(
                "Partial-view crop produced an empty point cloud. "
                "Please increase partial_keep_ratio_min or reduce partial_crop strength."
            )

        if self.normalize_coord:
            coord = self._normalize_coord_torch(coord)

        coord = self._maybe_jitter_torch(coord)

        if coord.shape[0] == 0:
            raise RuntimeError("Preprocess produced an empty point cloud.")

        return coord.to(dtype=torch.float32).contiguous()

    def _make_sparse_batch(
        self,
        xyz_b_n_3: torch.Tensor,
        model_device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)

        if model_device is None:
            model_device = self._get_model_device()

        sparse_device = self._me_device_str(model_device)

        # 关键：只在这里做一次 CPU -> GPU。
        # 如果输入本来就在 GPU，这里不会发生 CPU 回传。
        xyz_b_n_3 = xyz_b_n_3.detach().to(
            device=model_device,
            dtype=torch.float32,
            non_blocking=True,
        ).contiguous()

        coords_list = []
        feats_list = []
        centers = []

        # 与原代码一致：原来 detach().cpu().numpy() 已经切断了输入点坐标梯度。
        # 这里 no_grad 只包住预处理和量化，不影响 backbone/proj 参数训练。
        with torch.no_grad():
            for b in range(xyz_b_n_3.shape[0]):
                coord = self._preprocess_one_torch(xyz_b_n_3[b])

                centers.append(coord.mean(dim=0))

                if self.feature_mode in ["coord", "xyz"]:
                    feat = coord
                elif self.feature_mode in ["ones", "constant"]:
                    feat = torch.ones(
                        (coord.shape[0], self.in_channels),
                        device=model_device,
                        dtype=torch.float32,
                    )
                else:
                    raise ValueError(
                        f"Unsupported feature_mode={self.feature_mode}. "
                        "Use 'coord' or 'ones'."
                    )

                # 不再 return_index / return_inverse，因为你后面没有用。
                # 这能少算 mapping，减少一点开销。
                discrete_coords = torch.floor(
                    coord.contiguous() / self.voxel_size
                ).to(dtype=torch.int32)
                unique_map, _ = ME.utils.unique_coordinate_map(discrete_coords)
                quantized_coords = discrete_coords[unique_map]
                quantized_feats = feat.contiguous()[unique_map]

                self._check_cuda_if_needed(
                    quantized_coords,
                    "quantized_coords",
                    model_device,
                )
                self._check_cuda_if_needed(
                    quantized_feats,
                    "quantized_feats",
                    model_device,
                )

                quantized_coords = quantized_coords.to(dtype=torch.int32)
                quantized_feats = quantized_feats.to(dtype=torch.float32)

                coords_list.append(quantized_coords)
                feats_list.append(quantized_feats)

            coords_batch, feats_batch = ME.utils.sparse_collate(
                coords_list,
                feats_list,
                dtype=torch.int32,
                device=sparse_device,
            )

            self._check_cuda_if_needed(
                coords_batch,
                "coords_batch after sparse_collate",
                model_device,
            )
            self._check_cuda_if_needed(
                feats_batch,
                "feats_batch after sparse_collate",
                model_device,
            )

            coords_batch = coords_batch.to(dtype=torch.int32)
            feats_batch = feats_batch.to(dtype=torch.float32)

            centers = torch.stack(centers, dim=0).to(
                device=model_device,
                dtype=torch.float32,
            )

        return coords_batch, feats_batch, centers


    def forward_features(
        self,
        xyz_b_n_3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)
        batch_size = xyz_b_n_3.shape[0]

        model_device = self._get_model_device()
        sparse_device = self._me_device_str(model_device)

        coords_batch, feats_batch, centers = self._make_sparse_batch(
            xyz_b_n_3,
            model_device=model_device,
        )

        # 关键：features 和 coordinates 已经在 GPU。
        # 不再用 CPU coords 初始化 SparseTensor。
        sparse_input = ME.SparseTensor(
            features=feats_batch,
            coordinates=coords_batch,
            device=sparse_device,
        )

        voxel_feat = self.backbone(sparse_input)
        pooled = self.global_pool(voxel_feat).F

        if pooled.shape[0] != batch_size:
            raise RuntimeError(
                "MinkowskiGlobalAvgPooling returned unexpected batch size: "
                f"{pooled.shape[0]} vs expected {batch_size}."
            )

        if self.use_cpe_projection:
            pooled = self.proj(pooled)
            pooled = F.normalize(pooled, dim=1)
        else:
            pooled = self.proj(pooled)

        features = pooled.unsqueeze(-1).contiguous()

        centers = centers.to(device=features.device, dtype=features.dtype)
        centers = centers.view(batch_size, 1, 3)

        return features, centers


    def forward_embed(self, input_data):
        if isinstance(input_data, dict):
            if "pointcloud" not in input_data:
                raise KeyError(
                    f"forward_embed needs key 'pointcloud', got keys={list(input_data.keys())}"
                )
            xyz = input_data["pointcloud"]
        else:
            xyz = input_data

        features, _ = self.forward_features(xyz)

        # features: [B, C, 1]
        z = features.squeeze(-1)

        return F.normalize(z, dim=1)

    def forward(self, input_data: Dict[str, Any] | torch.Tensor) -> Dict[str, Any]:
        if isinstance(input_data, dict):
            if "pointcloud" not in input_data:
                raise KeyError(
                    f"UCA3DALMinkBackbone needs key 'pointcloud', "
                    f"got keys={list(input_data.keys())}"
                )
            xyz = input_data["pointcloud"]
        else:
            xyz = input_data

        xyz = self._normalize_xyz_shape(xyz)
        features, centers = self.forward_features(xyz)

        batch_size = features.shape[0]
        device = features.device

        # Placeholder fields for compatibility with your old PointMAE output format.
        # Classification does not use ori_idx / center_idx, but keeping them avoids
        # downstream KeyError if a logger or helper expects these keys.
        center_idx = torch.zeros(
            batch_size,
            1,
            dtype=torch.long,
            device=device,
        )
        ori_idx = torch.zeros(
            batch_size,
            1,
            1,
            dtype=torch.long,
            device=device,
        )

        return {
            # Raw branch.
            "raw_xyz_features": features,
            "raw_center": centers,
            "raw_ori_idx": ori_idx,
            "raw_center_idx": center_idx,

            # Legacy main branch.
            # Since this backbone does not do template registration, main == raw.
            "xyz_features": features,
            "center": centers,
            "ori_idx": ori_idx,
            "center_idx": center_idx,

            "is_registered": False,
            "registration_clsname": None,
            "registration_source": "uca3dal_mink_no_registration",
        }


def uca3dal_mink(
    in_channels: int = 3,
    out_channels: int = 32,
    proj_dim: int = 128,
    arch: str = "MinkUNet34C",
    voxel_size: float = 0.03,
    use_cpe_projection: bool = True,
    checkpoint_path: str = "",
    checkpoint_strict: bool = False,
    load_projection: bool = True,
    feature_mode: str = "coord",
    center_shift: bool = True,
    center_shift_apply_z: bool = True,
    normalize_coord: bool = True,
    normalize_eps: float = 1.0e-12,
    train_augment: bool = False,
    random_rotate: bool = True,
    random_rotate_angle_min: float = -1.0,
    random_rotate_angle_max: float = 1.0,
    random_scale: bool = True,
    random_scale_low: float = 0.9,
    random_scale_high: float = 1.1,
    random_scale_p: float = 0.5,
    random_jitter: bool = True,
    random_jitter_sigma: float = 0.005,
    random_jitter_clip: float = 0.02,
    random_jitter_p: float = 0.5,
    
    real3d_augment: bool = False,
    real3d_fps_npoints: int = 1024,
    real3d_first_rotate: bool = True,
    real3d_second_rotate: bool = True,
    real3d_dropout: bool = True,
    real3d_dropout_max_ratio: float = 0.875,
    real3d_scale: bool = True,
    real3d_scale_low: float = 0.8,
    real3d_scale_high: float = 1.25,
    real3d_shift: bool = True,
    real3d_shift_range: float = 0.1,


    # Partial-view crop augmentation.
    partial_crop: bool = False,
    partial_crop_p: float = 0.35,
    partial_keep_ratio_min: float = 0.65,
    partial_keep_ratio_max: float = 0.90,
    partial_min_points: int = 128,
    partial_view_jitter: float = 0.20,
    partial_soft_width: float = 0.02,
    partial_bottom_view_weight: float = 0.20,
):
    return UCA3DALMinkBackbone(
        in_channels=in_channels,
        out_channels=out_channels,
        proj_dim=proj_dim,
        arch=arch,
        voxel_size=voxel_size,
        use_cpe_projection=use_cpe_projection,
        checkpoint_path=checkpoint_path,
        checkpoint_strict=checkpoint_strict,
        load_projection=load_projection,
        feature_mode=feature_mode,
        center_shift=center_shift,
        center_shift_apply_z=center_shift_apply_z,
        normalize_coord=normalize_coord,
        normalize_eps=normalize_eps,
        train_augment=train_augment,
        random_rotate=random_rotate,
        random_rotate_angle_min=random_rotate_angle_min,
        random_rotate_angle_max=random_rotate_angle_max,
        random_scale=random_scale,
        random_scale_low=random_scale_low,
        random_scale_high=random_scale_high,
        random_scale_p=random_scale_p,
        random_jitter=random_jitter,
        random_jitter_sigma=random_jitter_sigma,
        random_jitter_clip=random_jitter_clip,
        random_jitter_p=random_jitter_p,
        partial_crop=partial_crop,
        partial_crop_p=partial_crop_p,
        partial_keep_ratio_min=partial_keep_ratio_min,
        partial_keep_ratio_max=partial_keep_ratio_max,
        partial_min_points=partial_min_points,
        partial_view_jitter=partial_view_jitter,
        partial_soft_width=partial_soft_width,
        partial_bottom_view_weight=partial_bottom_view_weight,
        real3d_augment=real3d_augment,
        real3d_fps_npoints=real3d_fps_npoints,
        real3d_first_rotate=real3d_first_rotate,
        real3d_second_rotate=real3d_second_rotate,
        real3d_dropout=real3d_dropout,
        real3d_dropout_max_ratio=real3d_dropout_max_ratio,
        real3d_scale=real3d_scale,
        real3d_scale_low=real3d_scale_low,
        real3d_scale_high=real3d_scale_high,
        real3d_shift=real3d_shift,
        real3d_shift_range=real3d_shift_range,

    )
