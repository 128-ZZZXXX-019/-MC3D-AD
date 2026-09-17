# -*- coding: utf-8 -*-
import os

import numpy as np
import open3d as o3d


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

    transform = np.asarray(result.transformation)

    if transform.shape != (4, 4):
        return False

    if not np.isfinite(transform).all():
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

        transform = np.asarray(result.transformation.cpu().numpy(), dtype=np.float64)

        if transform.shape == (4, 4) and np.isfinite(transform).all():
            return transform

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
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iteration)
        ),
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
        transform_cuda = refine_registration_cuda_icp(
            source_down=source_down,
            target_down=target_down,
            init_transform=init_transform,
            voxel_size=voxel_size,
            icp_distance_multiplier=icp_distance_multiplier,
            max_iteration=icp_max_iteration,
        )

        if transform_cuda is not None:
            return transform_cuda

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
        raise ValueError(
            f"Unknown registration_method={registration_method}. "
            "Use 'fgr' or 'ransac'."
        )

    if registration_result_is_valid(global_result, min_fitness=min_global_fitness):
        transform = np.asarray(global_result.transformation, dtype=np.float64)
    else:
        transform = np.eye(4, dtype=np.float64)

    if refine_icp:
        transform = refine_registration_icp_auto(
            source_down=source_down,
            target_down=target_down,
            init_transform=transform,
            voxel_size=voxel_size,
            icp_backend=icp_backend,
            icp_distance_multiplier=0.4,
            icp_max_iteration=30,
        )

    source_pcd.transform(transform)

    return np.asarray(source_pcd.points, dtype=np.float32)
