# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from timm.models.layers import DropPath


def square_distance(src, dst):
    # src: [B, N, C], dst: [B, M, C], return [B, N, M]
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2.0 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points, idx):
    # points: [B, N, C], idx: [B, ...], return [B, ..., C]
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


class KNN(nn.Module):
    # GPU KNN replacement. ref: [B,N,3], query: [B,G,3]
    def __init__(self, k, transpose_mode=False, chunk_size=256, return_sqrt_dist=False):
        super().__init__()
        self.k = int(k)
        self._t = bool(transpose_mode)
        self.chunk_size = int(chunk_size)
        self.return_sqrt_dist = bool(return_sqrt_dist)

    def _maybe_transpose(self, x):
        if self._t and x.dim() == 3 and x.shape[1] == 3 and x.shape[-1] != 3:
            x = x.transpose(1, 2).contiguous()
        return x

    def forward(self, ref, query):
        ref = self._maybe_transpose(ref)
        query = self._maybe_transpose(query)

        assert ref.size(0) == query.size(0), f"ref.shape={ref.shape} != query.shape={query.shape}"
        assert ref.dim() == 3 and query.dim() == 3, "KNN expects [B,N,C] and [B,G,C]"
        assert ref.size(-1) == query.size(-1), "ref/query last dim mismatch"
        assert self.k <= ref.size(1), f"k={self.k} is larger than N={ref.size(1)}"

        with torch.no_grad():
            ref = ref.contiguous().float()
            query = query.contiguous().float()
            dist_chunks = []
            idx_chunks = []

            for q in query.split(self.chunk_size, dim=1):
                dist2 = square_distance(q, ref)
                d2, idx = torch.topk(
                    dist2,
                    k=self.k,
                    dim=-1,
                    largest=False,
                    sorted=False,
                )
                if self.return_sqrt_dist:
                    d = torch.sqrt(torch.clamp(d2, min=0.0))
                else:
                    d = d2
                dist_chunks.append(d)
                idx_chunks.append(idx)

            distances = torch.cat(dist_chunks, dim=1)
            indices = torch.cat(idx_chunks, dim=1)

        return distances, indices


def fps(data, number):
    # data: [B,N,3]
    fps_idx = pointnet2_utils.furthest_point_sample(data, int(number))
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data, fps_idx


_REG_TRANS_INIT = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def get_first_file_in_subfolders(directory):
    # Return {class_name: first_train_file_full_path}
    subfolders = [
        f for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ]
    subfolder_file_dict = {}

    for subfolder in sorted(subfolders):
        train_dir = os.path.join(directory, subfolder, "train")
        if not os.path.isdir(train_dir):
            continue

        files = [
            f for f in os.listdir(train_dir)
            if os.path.isfile(os.path.join(train_dir, f))
        ]
        files = sorted(files)

        if files:
            subfolder_file_dict[subfolder] = os.path.join(train_dir, files[0])

    return subfolder_file_dict


def make_o3d_pcd(points_np):
    points_np = np.asarray(points_np, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(float(voxel_size))

    if len(pcd_down.points) == 0:
        raise RuntimeError(
            f"Downsampled point cloud is empty. voxel_size={voxel_size} may be too large."
        )

    radius_normal = float(voxel_size) * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = float(voxel_size) * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return pcd_down, pcd_fpfh


def build_template_registration_cache(template_points, voxel_size):
    template_points = np.asarray(template_points, dtype=np.float32)
    target_pcd = make_o3d_pcd(template_points)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    return {
        "points": template_points,
        "target_pcd": target_pcd,
        "target_down": target_down,
        "target_fpfh": target_fpfh,
    }


def prepare_source_for_registration(source_data, voxel_size):
    source_pcd = make_o3d_pcd(source_data)
    source_pcd.transform(_REG_TRANS_INIT.copy())
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    return source_pcd, source_down, source_fpfh


def execute_global_registration_ransac(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size,
    distance_multiplier=1.5,
    ransac_n=3,
    max_iteration=20000,
    confidence=0.999,
):
    distance_threshold = float(voxel_size) * float(distance_multiplier)
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        int(ransac_n),
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(max_iteration),
            float(confidence),
        ),
    )


def execute_global_registration_fgr(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size,
    distance_multiplier=0.5,
    iteration_number=64,
    maximum_tuple_count=1000,
):
    distance_threshold = float(voxel_size) * float(distance_multiplier)

    try:
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            iteration_number=int(iteration_number),
            maximum_tuple_count=int(maximum_tuple_count),
        )
    except TypeError:
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )

    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        option,
    )


def registration_result_is_valid(result, min_fitness=1e-6):
    if result is None:
        return False
    T = np.asarray(result.transformation)
    if T.shape != (4, 4):
        return False
    if not np.isfinite(T).all():
        return False
    fitness = float(getattr(result, "fitness", 0.0))
    return fitness > float(min_fitness)


def open3d_cuda_available():
    try:
        return bool(o3d.core.cuda.is_available())
    except Exception:
        return False


def make_tensor_pcd_from_legacy_pcd(pcd_legacy, device):
    points = np.asarray(pcd_legacy.points, dtype=np.float32)

    try:
        pcd_t = o3d.t.geometry.PointCloud(device)
    except TypeError:
        pcd_t = o3d.t.geometry.PointCloud()

    points_t = o3d.core.Tensor(points, o3d.core.float32, device)

    try:
        pcd_t.point["positions"] = points_t
    except Exception:
        pcd_t.point.positions = points_t

    return pcd_t


def refine_registration_cuda_icp(
    source_down,
    target_down,
    init_transform,
    voxel_size,
    icp_distance_multiplier=0.4,
    max_iteration=20,
):
    if not open3d_cuda_available():
        return None

    try:
        device = o3d.core.Device("CUDA:0")
        source_t = make_tensor_pcd_from_legacy_pcd(source_down, device)
        target_t = make_tensor_pcd_from_legacy_pcd(target_down, device)
        init_t = o3d.core.Tensor(
            np.asarray(init_transform, dtype=np.float64),
            o3d.core.float64,
            device,
        )
        max_corr = float(voxel_size) * float(icp_distance_multiplier)

        try:
            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=int(max_iteration)
            )
        except TypeError:
            criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
                1e-6,
                1e-6,
                int(max_iteration),
            )

        result = o3d.t.pipelines.registration.icp(
            source_t,
            target_t,
            max_corr,
            init_t,
            o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )

        T = np.asarray(result.transformation.cpu().numpy(), dtype=np.float64)
        if T.shape == (4, 4) and np.isfinite(T).all():
            return T
        return None

    except Exception:
        return None


def refine_registration_legacy_icp(
    source_down,
    target_down,
    init_transform,
    voxel_size,
    icp_distance_multiplier=0.4,
    max_iteration=30,
    point_to_plane=True,
):
    max_corr = float(voxel_size) * float(icp_distance_multiplier)

    can_use_point_to_plane = (
        point_to_plane
        and hasattr(source_down, "has_normals")
        and hasattr(target_down, "has_normals")
        and source_down.has_normals()
        and target_down.has_normals()
    )

    if can_use_point_to_plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_corr,
        init_transform,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iteration)),
    )

    return np.asarray(result.transformation, dtype=np.float64)


def refine_registration_icp_auto(
    source_down,
    target_down,
    init_transform,
    voxel_size,
    icp_backend="auto",
    icp_distance_multiplier=0.4,
    icp_max_iteration=30,
):
    icp_backend = str(icp_backend).lower()

    if icp_backend == "none":
        return np.asarray(init_transform, dtype=np.float64)

    if icp_backend in ["auto", "cuda"]:
        T_cuda = refine_registration_cuda_icp(
            source_down=source_down,
            target_down=target_down,
            init_transform=init_transform,
            voxel_size=voxel_size,
            icp_distance_multiplier=icp_distance_multiplier,
            max_iteration=icp_max_iteration,
        )
        if T_cuda is not None:
            return T_cuda
        if icp_backend == "cuda":
            return np.asarray(init_transform, dtype=np.float64)

    return refine_registration_legacy_icp(
        source_down=source_down,
        target_down=target_down,
        init_transform=init_transform,
        voxel_size=voxel_size,
        icp_distance_multiplier=icp_distance_multiplier,
        max_iteration=icp_max_iteration,
        point_to_plane=True,
    )


def get_registration_np_cached(
    source_data,
    template_cache,
    voxel_size=0.5,
    registration_method="fgr",
    fallback_to_ransac=True,
    min_global_fitness=1e-6,
    refine_icp=True,
    icp_backend="auto",
    ransac_max_iteration=20000,
    ransac_confidence=0.999,
    fgr_iteration_number=64,
):
    source_data = np.asarray(source_data, dtype=np.float32)

    source_pcd, source_down, source_fpfh = prepare_source_for_registration(
        source_data,
        voxel_size,
    )

    target_down = template_cache["target_down"]
    target_fpfh = template_cache["target_fpfh"]

    method = str(registration_method).lower()
    global_result = None

    if method in ["fgr", "fast", "fast_global", "fast_global_registration"]:
        try:
            global_result = execute_global_registration_fgr(
                source_down=source_down,
                target_down=target_down,
                source_fpfh=source_fpfh,
                target_fpfh=target_fpfh,
                voxel_size=voxel_size,
                iteration_number=fgr_iteration_number,
            )
        except Exception:
            global_result = None

        if fallback_to_ransac and not registration_result_is_valid(
            global_result,
            min_fitness=min_global_fitness,
        ):
            global_result = execute_global_registration_ransac(
                source_down=source_down,
                target_down=target_down,
                source_fpfh=source_fpfh,
                target_fpfh=target_fpfh,
                voxel_size=voxel_size,
                max_iteration=ransac_max_iteration,
                confidence=ransac_confidence,
            )

    elif method in ["ransac", "global_ransac"]:
        global_result = execute_global_registration_ransac(
            source_down=source_down,
            target_down=target_down,
            source_fpfh=source_fpfh,
            target_fpfh=target_fpfh,
            voxel_size=voxel_size,
            max_iteration=ransac_max_iteration,
            confidence=ransac_confidence,
        )
    else:
        raise ValueError(f"Unknown registration_method={registration_method}. Use 'fgr' or 'ransac'.")

    if registration_result_is_valid(global_result, min_fitness=min_global_fitness):
        T = np.asarray(global_result.transformation, dtype=np.float64)
    else:
        T = np.eye(4, dtype=np.float64)

    if refine_icp:
        T = refine_registration_icp_auto(
            source_down=source_down,
            target_down=target_down,
            init_transform=T,
            voxel_size=voxel_size,
            icp_backend=icp_backend,
            icp_distance_multiplier=0.4,
            icp_max_iteration=30,
        )

    source_pcd.transform(T)
    return np.asarray(source_pcd.points, dtype=np.float32)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        # xyz: [B,N,3]
        batch_size, num_points, _ = xyz.shape
        center, center_idx = fps(xyz.contiguous().float(), self.num_group)
        _, idx = self.knn(xyz, center)
        idx = idx.to(device=xyz.device)

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size

        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(
            batch_size,
            self.num_group,
            self.group_size,
            3,
        ).contiguous()

        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        # point_groups: [B,G,N,3]
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    # attn_type: standard / performer / cosformer
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_type="standard",
        performer_nb_features=256,
        performer_redraw_projection=False,
        performer_redraw_interval=0,
        cosformer_act="relu",
        eps=1e-6,
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.attn_type = str(attn_type).lower()
        assert self.attn_type in ["standard", "softmax", "performer", "cosformer"], (
            f"Unknown attn_type={attn_type}"
        )

        self.performer_nb_features = int(performer_nb_features)
        self.performer_redraw_projection = bool(performer_redraw_projection)
        self.performer_redraw_interval = int(performer_redraw_interval)
        self.cosformer_act = str(cosformer_act).lower()
        self.eps = float(eps)

        assert self.cosformer_act in ["relu", "elu"], "cosformer_act must be 'relu' or 'elu'"

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

        if self.attn_type == "performer":
            projection = self._create_projection_matrix(
                self.performer_nb_features,
                self.head_dim,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.register_buffer("projection_matrix", projection, persistent=False)
            self.register_buffer("calls_since_redraw", torch.zeros((), dtype=torch.long), persistent=False)
        else:
            self.register_buffer("projection_matrix", torch.empty(0), persistent=False)
            self.register_buffer("calls_since_redraw", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def _create_projection_matrix(nb_features, dim, device, dtype=torch.float32, scaling=0):
        nb_features = int(nb_features)
        dim = int(dim)
        nb_full_blocks = nb_features // dim
        remaining_rows = nb_features - nb_full_blocks * dim

        block_list = []
        for _ in range(nb_full_blocks):
            unstructured_block = torch.randn((dim, dim), device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(unstructured_block, mode="reduced")
            block_list.append(q.t())

        if remaining_rows > 0:
            unstructured_block = torch.randn((dim, dim), device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(unstructured_block, mode="reduced")
            block_list.append(q.t()[:remaining_rows])

        final_matrix = torch.cat(block_list, dim=0)

        if scaling == 0:
            multiplier = torch.randn((nb_features, dim), device=device, dtype=torch.float32).norm(dim=1)
        elif scaling == 1:
            multiplier = math.sqrt(float(dim)) * torch.ones((nb_features,), device=device, dtype=torch.float32)
        else:
            raise ValueError("scaling must be 0 or 1")

        final_matrix = final_matrix * multiplier.unsqueeze(1)
        return final_matrix.to(dtype=dtype)

    def _maybe_redraw_projection(self, device):
        if self.attn_type != "performer":
            return

        if self.projection_matrix.device != device:
            self.projection_matrix = self.projection_matrix.to(device)

        if not self.training:
            return
        if not self.performer_redraw_projection:
            return
        if self.performer_redraw_interval <= 0:
            return

        self.calls_since_redraw += 1
        if int(self.calls_since_redraw.item()) >= self.performer_redraw_interval:
            with torch.no_grad():
                self.projection_matrix = self._create_projection_matrix(
                    self.performer_nb_features,
                    self.head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                self.calls_since_redraw.zero_()

    def _softmax_attention(self, q, k, v):
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

    def _performer_softmax_kernel(self, data, is_query):
        orig_dtype = data.dtype
        data = data.float()

        projection = self.projection_matrix
        if projection.device != data.device:
            projection = projection.to(data.device)
        projection = projection.float()

        data_normalizer = self.head_dim ** -0.25
        data = data * data_normalizer

        data_dash = torch.einsum("bhnd,md->bhnm", data, projection)
        diag_data = (data ** 2).sum(dim=-1, keepdim=True) / 2.0

        if is_query:
            max_val = data_dash.max(dim=-1, keepdim=True).values.detach()
        else:
            max_val = data_dash.max(dim=-1, keepdim=True).values
            max_val = max_val.max(dim=-2, keepdim=True).values.detach()

        features = torch.exp(data_dash - diag_data - max_val)
        features = features * (self.performer_nb_features ** -0.5)
        features = features + self.eps
        return features.to(orig_dtype)

    def _performer_attention(self, q, k, v):
        self._maybe_redraw_projection(q.device)

        orig_dtype = q.dtype
        q_prime = self._performer_softmax_kernel(q, is_query=True).float()
        k_prime = self._performer_softmax_kernel(k, is_query=False).float()
        v_float = v.float()

        kv = torch.einsum("bhnm,bhnd->bhmd", k_prime, v_float)
        k_sum = k_prime.sum(dim=2)
        denom = torch.einsum("bhnm,bhm->bhn", q_prime, k_sum)
        denom = torch.clamp(denom, min=self.eps)

        out = torch.einsum("bhnm,bhmd,bhn->bhnd", q_prime, kv, 1.0 / denom)
        out = out.to(orig_dtype)
        out = self.attn_drop(out)
        return out

    def _cosformer_feature_activation(self, x):
        if self.cosformer_act == "relu":
            return F.relu(x) + self.eps
        return F.elu(x) + 1.0 + self.eps

    def _cosformer_attention(self, q, k, v):
        orig_dtype = q.dtype
        q = self._cosformer_feature_activation(q).float()
        k = self._cosformer_feature_activation(k).float()
        v = v.float()

        B, H, N, D = q.shape
        index = torch.arange(1, N + 1, device=q.device, dtype=torch.float32).view(1, 1, N, 1)
        angle = (math.pi / 2.0) * index / float(N)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        q_ = torch.cat([q * sin, q * cos], dim=-1)
        k_ = torch.cat([k * sin, k * cos], dim=-1)

        kv = torch.einsum("bhnd,bhne->bhde", k_, v)
        k_sum = k_.sum(dim=2)
        denom = torch.einsum("bhnd,bhd->bhn", q_, k_sum)
        denom = torch.clamp(denom, min=self.eps)

        out = torch.einsum("bhnd,bhde,bhn->bhne", q_, kv, 1.0 / denom)
        out = out.to(orig_dtype)
        out = self.attn_drop(out)
        return out

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type in ["standard", "softmax"]:
            out = self._softmax_attention(q, k, v)
        elif self.attn_type == "performer":
            out = self._performer_attention(q, k, v)
        elif self.attn_type == "cosformer":
            out = self._cosformer_attention(q, k, v)
        else:
            raise RuntimeError(f"Unsupported attn_type={self.attn_type}")

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_type="standard",
        performer_nb_features=256,
        performer_redraw_projection=False,
        performer_redraw_interval=0,
        cosformer_act="relu",
        linear_attn_eps=1e-6,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_type=attn_type,
            performer_nb_features=performer_nb_features,
            performer_redraw_projection=performer_redraw_projection,
            performer_redraw_interval=performer_redraw_interval,
            cosformer_act=cosformer_act,
            eps=linear_attn_eps,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_type="standard",
        performer_nb_features=256,
        performer_redraw_projection=False,
        performer_redraw_interval=0,
        cosformer_act="relu",
        linear_attn_eps=1e-6,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                attn_type=attn_type,
                performer_nb_features=performer_nb_features,
                performer_redraw_projection=performer_redraw_projection,
                performer_redraw_interval=performer_redraw_interval,
                cosformer_act=cosformer_act,
                linear_attn_eps=linear_attn_eps,
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]

        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)

        return feature_list


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = int(in_channel)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, int(out_channel), 1))
            self.mlp_bns.append(nn.BatchNorm1d(int(out_channel)))
            last_channel = int(out_channel)

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1),
                dim=2,
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointTransformer(nn.Module):
    def __init__(
        self,
        group_size=128,
        num_group=1024,
        encoder_dims=384,
        attn_type="performer",
        performer_nb_features=256,
        performer_redraw_projection=False,
        performer_redraw_interval=0,
        cosformer_act="relu",
        linear_attn_eps=1e-6,
    ):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6

        self.group_size = int(group_size)
        self.num_group = int(num_group)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder_dims = int(encoder_dims)

        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            attn_type=attn_type,
            performer_nb_features=performer_nb_features,
            performer_redraw_projection=performer_redraw_projection,
            performer_redraw_interval=performer_redraw_interval,
            cosformer_act=cosformer_act,
            linear_attn_eps=linear_attn_eps,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder."):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model."):]] = base_ckpt[k]
                    del base_ckpt[k]

            self.load_state_dict(base_ckpt, strict=False)

    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}

        for k in list(base_ckpt.keys()):
            if k.startswith("transformer_q") and not k.startswith("transformer_q.cls_head"):
                base_ckpt[k[len("transformer_q."):]] = base_ckpt[k]
            elif k.startswith("model_state_dict"):
                base_ckpt[k[len("model_state_dict."):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print("missing_keys")
            print(incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("unexpected_keys")
            print(incompatible.unexpected_keys)

        print(f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}")

    def forward(self, pts):
        # pts: [B,3,N]
        if self.encoder_dims != self.trans_dim:
            B, C, N = pts.shape
            pts = pts.transpose(-1, -2)
            neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
            group_input_tokens = self.encoder(neighborhood)
            group_input_tokens = self.reduce_dim(group_input_tokens)

            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

            pos = self.pos_embed(center)
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            feature_list = self.blocks(x, pos)
            feature_list = [
                self.norm(item)[:, 1:].transpose(-1, -2).contiguous()
                for item in feature_list
            ]
            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)
            return x, center, ori_idx, center_idx

        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)
        neighborhood, center, ori_idx, center_idx = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)
        x = group_input_tokens

        feature_list = self.blocks(x, pos)
        feature_list = [
            self.norm(item).transpose(-1, -2).contiguous()
            for item in feature_list
        ]

        x = feature_list[0]
        return x, center, ori_idx, center_idx


def _bool_from_value(x, default=False):
    if x is None:
        return bool(default)
    if torch.is_tensor(x):
        if x.numel() == 0:
            return bool(default)
        return bool(x.detach().cpu().view(-1)[0].item())
    if isinstance(x, str):
        return x.lower() in ["1", "true", "yes", "y"]
    return bool(x)


class Model1(torch.nn.Module):
    # Main PointMAE backbone wrapper.
    #
    # 重要逻辑：
    # - raw_* 分支：用于分类第一阶段；可选使用 CenterShift + NormalizeCoord 后的点云。
    # - 主分支 xyz_features：
    #   - 如果已配准：来自“原始点云 -> 模板配准 -> backbone”。
    #   - 如果未配准：来自 raw_* 分支，供第一阶段分类使用。
    #
    # 新增行为：
    # - cls_raw_coord_preprocess=True 时，只复制一份 xyz 给 raw 分支做中心化/归一化。
    # - 原始 xyz 本体不被覆盖，因此模板配准仍然拿到原始点云。
    def __init__(
        self,
        data_dir,
        device,
        out_indices=None,
        checkpoint_path="",
        xyz_backbone_name="Point_MAE",
        group_size=128,
        num_group=1024,
        voxel_size=0.5,
        registration_method="fgr",
        fallback_to_ransac=True,
        refine_icp=True,
        icp_backend="auto",
        ransac_max_iteration=20000,
        ransac_confidence=0.999,
        fgr_iteration_number=64,
        min_global_fitness=1e-6,
        dual_feature=True,
        train_register_with_gt=True,
        eval_use_gt_registration=False,
        return_raw_features=True,
        return_raw_features_in_registered_pass=False,
        class_names=None,
        backbone_attn_type="cosformer",
        performer_nb_features=256,
        performer_redraw_projection=False,
        performer_redraw_interval=0,
        cosformer_act="elu",
        linear_attn_eps=1e-6,

        # 新增：只作用于 raw 分类分支，不作用于模板配准分支。
        cls_raw_coord_preprocess=False,
        cls_raw_center_shift=True,
        cls_raw_center_shift_apply_z=True,
        cls_raw_normalize_coord=True,
        cls_raw_preprocess_order=("center_shift", "normalize_coord"),
        cls_raw_normalize_eps=1e-12,
    ):
        super().__init__()

        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.data_path = data_dir

        self.voxel_size = float(voxel_size)
        self.registration_method = registration_method
        self.fallback_to_ransac = bool(fallback_to_ransac)
        self.refine_icp = bool(refine_icp)
        self.icp_backend = icp_backend
        self.ransac_max_iteration = int(ransac_max_iteration)
        self.ransac_confidence = float(ransac_confidence)
        self.fgr_iteration_number = int(fgr_iteration_number)
        self.min_global_fitness = float(min_global_fitness)

        self.dual_feature = bool(dual_feature)
        self.train_register_with_gt = bool(train_register_with_gt)
        self.eval_use_gt_registration = bool(eval_use_gt_registration)
        self.return_raw_features = bool(return_raw_features)
        self.return_raw_features_in_registered_pass = bool(return_raw_features_in_registered_pass)
        self.class_names = [str(x) for x in class_names] if class_names is not None else None

        # 新增：raw 分类分支的坐标预处理配置。
        self.cls_raw_coord_preprocess = bool(cls_raw_coord_preprocess)
        self.cls_raw_center_shift = bool(cls_raw_center_shift)
        self.cls_raw_center_shift_apply_z = bool(cls_raw_center_shift_apply_z)
        self.cls_raw_normalize_coord = bool(cls_raw_normalize_coord)
        self.cls_raw_normalize_eps = float(cls_raw_normalize_eps)

        if isinstance(cls_raw_preprocess_order, str):
            cls_raw_preprocess_order = [
                item.strip()
                for item in cls_raw_preprocess_order.split(",")
                if item.strip()
            ]

        self.cls_raw_preprocess_order = [
            str(item).lower()
            for item in cls_raw_preprocess_order
        ]

        kwargs = {"features_only": True if out_indices else False}
        if out_indices:
            kwargs.update({"out_indices": out_indices})

        self.template = get_first_file_in_subfolders(self.data_path)

        self.template_cache = {}
        for cls_name, template_path in self.template.items():
            resolved_path = self._resolve_template_path(template_path)

            template_pcd = o3d.io.read_point_cloud(resolved_path)
            template_points = np.asarray(template_pcd.points, dtype=np.float32)

            if template_points.ndim != 2 or template_points.shape[1] != 3:
                raise RuntimeError(
                    f"Invalid template point cloud for class {cls_name}: "
                    f"{resolved_path}, shape={template_points.shape}"
                )

            template_points = self.norm_pcd(template_points).astype(np.float32)
            self.template_cache[cls_name] = build_template_registration_cache(
                template_points=template_points,
                voxel_size=self.voxel_size,
            )

        if xyz_backbone_name == "Point_MAE":
            self.xyz_backbone = PointTransformer(
                group_size=group_size,
                num_group=num_group,
                attn_type=backbone_attn_type,
                performer_nb_features=performer_nb_features,
                performer_redraw_projection=performer_redraw_projection,
                performer_redraw_interval=performer_redraw_interval,
                cosformer_act=cosformer_act,
                linear_attn_eps=linear_attn_eps,
            )

            if checkpoint_path:
                self.xyz_backbone.load_model_from_ckpt(checkpoint_path)

            self.xyz_backbone.to(self.device)
        else:
            raise ValueError(f"Unsupported xyz_backbone_name={xyz_backbone_name}")

    def _resolve_template_path(self, template_path):
        if os.path.isabs(template_path):
            return template_path
        if os.path.exists(template_path):
            return template_path
        return os.path.join(self.data_path, template_path)

    def get_outstrides(self):
        return getattr(self, "outstrides", None)

    def norm_pcd(self, point_cloud):
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        center = np.mean(point_cloud, axis=0, keepdims=True)
        return point_cloud - center

    def _normalize_cls_name(self, cls_name):
        if isinstance(cls_name, bytes):
            return cls_name.decode("utf-8")
        if isinstance(cls_name, np.bytes_):
            return cls_name.decode("utf-8")
        if isinstance(cls_name, np.str_):
            return str(cls_name)
        if isinstance(cls_name, torch.Tensor):
            if cls_name.numel() == 1:
                value = cls_name.item()
                if isinstance(value, (int, np.integer)) and self.class_names is not None:
                    idx = int(value)
                    if 0 <= idx < len(self.class_names):
                        return self.class_names[idx]
                return str(value)
            return str(cls_name.detach().cpu().tolist())
        if isinstance(cls_name, (int, np.integer)) and self.class_names is not None:
            idx = int(cls_name)
            if 0 <= idx < len(self.class_names):
                return self.class_names[idx]
        return str(cls_name)

    def _names_to_list(self, value, batch_size=None):
        if value is None:
            return None

        if torch.is_tensor(value):
            if value.dim() == 0:
                raw_items = [value.item()]
            else:
                raw_items = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            raw_items = value.tolist()
        elif isinstance(value, (str, bytes, np.str_, np.bytes_)):
            raw_items = [value]
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            raw_items = [value]

        names = []
        for item in raw_items:
            while isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            names.append(self._normalize_cls_name(item))

        if batch_size is not None:
            if len(names) == 1 and batch_size > 1:
                names = names * batch_size
            if len(names) != batch_size:
                raise RuntimeError(
                    f"Expected {batch_size} class names, got {len(names)}: {names}"
                )

        return names

    def _normalize_xyz_shape(self, xyz):
        if not torch.is_tensor(xyz):
            xyz = torch.as_tensor(xyz, dtype=torch.float32)

        if xyz.dim() != 3:
            raise RuntimeError(f"pointcloud must be 3D, got shape={tuple(xyz.shape)}")

        if xyz.shape[-1] == 3:
            return xyz.contiguous()

        if xyz.shape[1] == 3:
            return xyz.transpose(1, 2).contiguous()

        raise RuntimeError(
            f"Cannot infer point dimension from shape={tuple(xyz.shape)}. "
            "Expected [B,N,3] or [B,3,N]."
        )

    def _center_shift_xyz(self, xyz_b_n_3):
        # 等价于你给出的 CenterShift，但支持 batch tensor。
        # xyz_b_n_3: [B, N, 3]
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)

        xyz_min = xyz_b_n_3.amin(dim=1)
        xyz_max = xyz_b_n_3.amax(dim=1)

        shift_x = (xyz_min[:, 0] + xyz_max[:, 0]) / 2.0
        shift_y = (xyz_min[:, 1] + xyz_max[:, 1]) / 2.0

        if self.cls_raw_center_shift_apply_z:
            shift_z = xyz_min[:, 2]
        else:
            shift_z = torch.zeros_like(shift_x)

        shift = torch.stack([shift_x, shift_y, shift_z], dim=1).view(-1, 1, 3)
        return xyz_b_n_3 - shift

    def _normalize_coord_xyz(self, xyz_b_n_3):
        # 等价于你给出的 NormalizeCoord，但支持 batch tensor。
        # xyz_b_n_3: [B, N, 3]
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)

        centroid = xyz_b_n_3.mean(dim=1, keepdim=True)
        xyz_b_n_3 = xyz_b_n_3 - centroid

        radius = torch.sqrt(torch.sum(xyz_b_n_3 ** 2, dim=-1)).amax(dim=1)
        radius = torch.clamp(radius, min=self.cls_raw_normalize_eps).view(-1, 1, 1)

        return xyz_b_n_3 / radius

    def _preprocess_cls_raw_xyz(self, xyz_b_n_3):
        # 只给 raw 分类分支使用。
        # 注意：这里 clone()，绝不覆盖原始 xyz，因此不会影响模板配准。
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)

        if not self.cls_raw_coord_preprocess:
            return xyz_b_n_3

        out = xyz_b_n_3.clone()

        for op in self.cls_raw_preprocess_order:
            if op in ["center_shift", "centershift", "center"]:
                if self.cls_raw_center_shift:
                    out = self._center_shift_xyz(out)

            elif op in ["normalize_coord", "normalize", "norm"]:
                if self.cls_raw_normalize_coord:
                    out = self._normalize_coord_xyz(out)

            elif op in ["none", "identity", "skip"]:
                continue

            else:
                raise ValueError(
                    f"Unsupported cls_raw_preprocess_order item={op}. "
                    "Supported: center_shift, normalize_coord."
                )

        return out.contiguous()

    def _extract_feature_dict(self, xyz_b_n_3, prefix=""):
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)
        xyz_input = (
            xyz_b_n_3.to(self.device, dtype=torch.float32)
            .permute(0, 2, 1)
            .contiguous()
        )

        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz_input)

        return {
            f"{prefix}xyz_features": xyz_features,
            f"{prefix}center": center,
            f"{prefix}ori_idx": ori_idx,
            f"{prefix}center_idx": center_idx,
        }

    def _extract_raw_feature_dict(self, xyz_b_n_3):
        # raw 分支专用入口。
        # 开关打开时：raw 特征来自 CenterShift + NormalizeCoord 后的点云。
        # 开关关闭时：保持旧逻辑，raw 特征来自原始未配准点云。
        raw_xyz = self._preprocess_cls_raw_xyz(xyz_b_n_3)
        return self._extract_feature_dict(raw_xyz, prefix="raw_")

    def _copy_raw_to_main(self, out):
        # 无配准第一阶段：把 raw 分支暴露成 legacy main branch。
        # 如果 cls_raw_coord_preprocess=True，则这里的 xyz_features 也来自归一化 raw 点云，
        # 这正是第一阶段分类需要的输入。
        out["xyz_features"] = out["raw_xyz_features"]
        out["center"] = out["raw_center"]
        out["ori_idx"] = out["raw_ori_idx"]
        out["center_idx"] = out["raw_center_idx"]
        return out

    def _select_registration_names(self, pointcloud, batch_size):
        if _bool_from_value(pointcloud.get("force_no_registration", False), default=False):
            return None, None

        # 显式注入的类别优先级最高。
        # AD train:
        #   registration_clsname / registration_cls_label 来自 GT
        #
        # AD test:
        #   registration_clsname / registration_cls_label 来自 CPE 预测
        explicit_keys = [
            "registration_clsname",
            "registration_class_name",
            "registration_cls_label",

            "pred_clsname",
            "pred_class_name",
            "pred_category",
            "pred_cls_label",
        ]

        for key in explicit_keys:
            if key in pointcloud and pointcloud[key] is not None:
                return self._names_to_list(pointcloud[key], batch_size=batch_size), key

        # freeze_selected_layers / ModelHelper.train() 可能让 backbone.eval()，
        # 所以 AD train 阶段需要外部显式写入 is_train/use_gt_registration。
        train_like = (
            _bool_from_value(pointcloud.get("is_train", False), default=False)
            or _bool_from_value(pointcloud.get("use_gt_registration", False), default=False)
            or _bool_from_value(pointcloud.get("force_gt_registration", False), default=False)
        )

        # 训练阶段可用 GT 类别做配准。
        if (train_like or self.training) and self.train_register_with_gt:
            for key in ["clsname", "class_name", "category", "class"]:
                if key in pointcloud and pointcloud[key] is not None:
                    return self._names_to_list(pointcloud[key], batch_size=batch_size), key

            for key in ["cls_label", "class_label", "category_id"]:
                if key in pointcloud and pointcloud[key] is not None:
                    return self._names_to_list(pointcloud[key], batch_size=batch_size), key

        # 测试阶段默认禁止用 GT 类别。
        # 只有 eval_use_gt_registration=True 时，才允许 oracle ablation。
        if (not train_like) and (not self.training) and self.eval_use_gt_registration:
            for key in ["clsname", "class_name", "category", "class"]:
                if key in pointcloud and pointcloud[key] is not None:
                    return self._names_to_list(pointcloud[key], batch_size=batch_size), key

            for key in ["cls_label", "class_label", "category_id"]:
                if key in pointcloud and pointcloud[key] is not None:
                    return self._names_to_list(pointcloud[key], batch_size=batch_size), key

        return None, None


    def _register_batch_by_names(self, xyz_b_n_3, cls_names):
        # 注意：这里接收的必须是原始 xyz，不是 raw 分支归一化后的 xyz。
        xyz_b_n_3 = self._normalize_xyz_shape(xyz_b_n_3)
        B = xyz_b_n_3.shape[0]

        if len(cls_names) != B:
            raise RuntimeError(f"cls_names length {len(cls_names)} != batch size {B}")

        reg_list = []

        with torch.no_grad():
            for idx in range(B):
                cls_name = self._normalize_cls_name(cls_names[idx])

                if cls_name not in self.template_cache:
                    raise KeyError(
                        f"Class name {cls_name} not found in template cache. "
                        f"Available classes: {list(self.template_cache.keys())[:20]}"
                    )

                source_np = (
                    xyz_b_n_3[idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                reg_data = get_registration_np_cached(
                    source_data=source_np,
                    template_cache=self.template_cache[cls_name],
                    voxel_size=self.voxel_size,
                    registration_method=self.registration_method,
                    fallback_to_ransac=self.fallback_to_ransac,
                    min_global_fitness=self.min_global_fitness,
                    refine_icp=self.refine_icp,
                    icp_backend=self.icp_backend,
                    ransac_max_iteration=self.ransac_max_iteration,
                    ransac_confidence=self.ransac_confidence,
                    fgr_iteration_number=self.fgr_iteration_number,
                )

                reg_list.append(reg_data)

        arr = np.asarray(reg_list, dtype=np.float32)
        return torch.from_numpy(arr)

    def forward(self, pointcloud):
        if "pointcloud" not in pointcloud:
            raise KeyError(f"Model1 needs key 'pointcloud', got keys={list(pointcloud.keys())}")

        # 这里的 xyz 始终代表原始点云。
        # 后续配准必须使用这个 xyz。
        xyz = self._normalize_xyz_shape(pointcloud["pointcloud"])
        B = xyz.shape[0]

        registration_names, registration_source = self._select_registration_names(pointcloud, B)
        do_register = registration_names is not None

        train_like = (
            _bool_from_value(pointcloud.get("is_train", False), default=False)
            or _bool_from_value(pointcloud.get("use_gt_registration", False), default=False)
            or _bool_from_value(pointcloud.get("force_gt_registration", False), default=False)
        )

        if train_like or self.training:
            compute_raw = self.dual_feature or self.return_raw_features or (not do_register)
        else:
            if do_register:
                compute_raw = self.return_raw_features_in_registered_pass
            else:
                compute_raw = True

        out = {}

        if compute_raw:
            # 这里是本次修改的关键：
            # raw 分支不再直接使用 xyz，而是走 _extract_raw_feature_dict。
            # _extract_raw_feature_dict 内部会根据 cls_raw_coord_preprocess 决定是否复制并归一化。
            out.update(self._extract_raw_feature_dict(xyz))

        if do_register:
            # 这里必须继续使用原始 xyz。
            # 不要改成 raw_xyz，也不要在 dataset transform 里覆盖 pointcloud。
            registered_xyz = self._register_batch_by_names(xyz, registration_names)
            out.update(self._extract_feature_dict(registered_xyz, prefix=""))

            out["is_registered"] = True
            out["registration_clsname"] = registration_names
            out["registration_source"] = registration_source

            return out

        # No registration:
        # 第一阶段分类时没有类别，不能配准。
        # 此时 raw 分支作为 legacy main branch 暴露给分类头。
        if "raw_xyz_features" not in out:
            out.update(self._extract_raw_feature_dict(xyz))

        out = self._copy_raw_to_main(out)
        out["is_registered"] = False
        out["registration_clsname"] = None
        out["registration_source"] = "none"
        return out
