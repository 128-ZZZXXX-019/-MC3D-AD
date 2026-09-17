# -*- coding: utf-8 -*-
import torch

from .pointmae import Model1

__version__ = "0.7.1"

__all__ = [
    "pointmae",
]


def _resolve_device(device=None):
    if device is not None:
        return device

    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"

    return "cpu"


def pointmae(
    pretrained=None,
    outlayers=None,
    checkpoint_path="",
    group_size=128,
    num_group=1024,
    data_dir=None,
    device=None,
    out_indices=None,
    **kwargs,
):

    if data_dir is None:
        raise ValueError(
            "pointmae() needs data_dir. "
            "Here data_dir is used as PointMAE registration template directory."
        )

    # 兼容旧配置: outlayers -> out_indices
    if out_indices is None:
        out_indices = outlayers

    default_kwargs = dict(
        voxel_size=0.5,
        registration_method="ransac",
        fallback_to_ransac=True,
        refine_icp=False,
        icp_backend="auto",
        ransac_max_iteration=100000,
        ransac_confidence=0.999,
        fgr_iteration_number=64,
    )

    default_kwargs.update(kwargs)

    model = Model1(
        data_dir=data_dir,
        device=_resolve_device(device),
        out_indices=out_indices,
        checkpoint_path=checkpoint_path,
        group_size=group_size,
        num_group=num_group,
        **default_kwargs,
    )

    return model
