# interventions/fire_cpe.py

import math
from typing import Iterable, Optional

import torch
from torch import nn


def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _name_allowed(name: str, include_name_keywords: Optional[Iterable[str]]) -> bool:
    keywords = list(include_name_keywords or [])
    if len(keywords) == 0:
        return True
    return any(k in name for k in keywords)


def _newton_schulz_2d(
    matrix: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """
    FIRE core: approximate polar orthogonalization with Newton-Schulz.

    Input:
        matrix: 2D tensor.

    Return:
        semi-orthogonalized matrix with the same shape.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={tuple(matrix.shape)}")

    orig_dtype = matrix.dtype
    orig_device = matrix.device

    X = matrix.detach().to(dtype=torch.float32)

    # Work on tall matrix when possible.
    # If out_dim < in_dim, transpose first, then transpose back.
    do_transpose = X.shape[0] < X.shape[1]
    if do_transpose:
        X = X.t()

    norm = X.norm()
    if not torch.isfinite(norm) or norm.item() < eps:
        return matrix.detach().clone()

    X = X / (norm + eps)

    for _ in range(num_iters):
        A = X.t() @ X
        X = 1.5 * X - 0.5 * (X @ A)

    if do_transpose:
        X = X.t()

    return X.to(device=orig_device, dtype=orig_dtype)


def _fire_linear_weight(weight: torch.Tensor, num_iters: int) -> torch.Tensor:
    """
    Linear weight shape: [out_dim, in_dim].
    Scale follows FIRE appendix:
        sqrt(out_dim / in_dim)
    """
    out_dim, in_dim = weight.shape[0], weight.shape[1]
    scale = math.sqrt(float(out_dim) / float(max(in_dim, 1)))
    return _newton_schulz_2d(weight, num_iters=num_iters) * scale


def _fire_dense_conv_weight(weight: torch.Tensor, num_iters: int) -> torch.Tensor:
    """
    PyTorch ConvNd weight shape:
        [out_channels, in_channels, *kernel_shape]

    FIRE applies orthogonalization slice-by-slice over spatial kernel positions.
    Scale:
        sqrt(out_channels / in_channels) / kernel_volume
    """
    out_c, in_c = weight.shape[0], weight.shape[1]
    kernel_volume = 1
    for s in weight.shape[2:]:
        kernel_volume *= int(s)

    flat = weight.detach().clone().reshape(out_c, in_c, kernel_volume)
    new_flat = torch.empty_like(flat)

    scale = math.sqrt(float(out_c) / float(max(in_c, 1))) / float(max(kernel_volume, 1))

    for k in range(kernel_volume):
        new_flat[:, :, k] = _newton_schulz_2d(
            flat[:, :, k],
            num_iters=num_iters,
        ) * scale

    return new_flat.reshape_as(weight)


def _infer_sparse_kernel_layout(module: nn.Module, weight: torch.Tensor, layout: str) -> str:
    """
    For MinkowskiEngine-like kernels.

    Common layouts:
        KIO: [kernel_volume, in_channels, out_channels]
        KOI: [kernel_volume, out_channels, in_channels]
    """
    if layout != "auto":
        return layout

    in_c = getattr(module, "in_channels", None)
    out_c = getattr(module, "out_channels", None)

    if in_c is not None and out_c is not None:
        in_c = int(in_c)
        out_c = int(out_c)

        if weight.ndim == 3:
            if weight.shape[1] == in_c and weight.shape[2] == out_c:
                return "KIO"
            if weight.shape[1] == out_c and weight.shape[2] == in_c:
                return "KOI"

    # Default for many MinkowskiEngine kernels.
    return "KIO"


def _fire_sparse_kernel_weight(
    module: nn.Module,
    weight: torch.Tensor,
    num_iters: int,
    sparse_kernel_layout: str = "auto",
) -> torch.Tensor:
    """
    Sparse convolution kernel, usually:
        [kernel_volume, in_channels, out_channels]  -> KIO
    or:
        [kernel_volume, out_channels, in_channels]  -> KOI

    We apply FIRE per sparse kernel offset.
    """
    if weight.ndim != 3:
        raise ValueError(f"Expected 3D sparse kernel, got shape={tuple(weight.shape)}")

    layout = _infer_sparse_kernel_layout(module, weight, sparse_kernel_layout)

    K = int(weight.shape[0])
    new_weight = torch.empty_like(weight)

    if layout == "KIO":
        # weight[k]: [in_c, out_c], transpose to [out_c, in_c]
        in_c = int(weight.shape[1])
        out_c = int(weight.shape[2])
        scale = math.sqrt(float(out_c) / float(max(in_c, 1))) / float(max(K, 1))

        for k in range(K):
            mat_out_in = weight[k].t()
            new_weight[k] = (
                _newton_schulz_2d(mat_out_in, num_iters=num_iters) * scale
            ).t()

    elif layout == "KOI":
        # weight[k]: [out_c, in_c]
        out_c = int(weight.shape[1])
        in_c = int(weight.shape[2])
        scale = math.sqrt(float(out_c) / float(max(in_c, 1))) / float(max(K, 1))

        for k in range(K):
            new_weight[k] = _newton_schulz_2d(
                weight[k],
                num_iters=num_iters,
            ) * scale

    else:
        raise ValueError(
            f"Unknown sparse_kernel_layout={layout}. Use auto, KIO, or KOI."
        )

    return new_weight


@torch.no_grad()
def fire_reinit_cpe(
    model: nn.Module,
    num_iters: int = 10,
    only_trainable: bool = True,
    include_name_keywords: Optional[Iterable[str]] = None,
    sparse_kernel_layout: str = "auto",
    logger=None,
) -> int:
    """
    FIRE for your CPE/Minkowski setting.

    Returns:
        number of parameters changed.
    """
    changed = 0
    skipped = 0
    seen_param_ids = set()

    dense_conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    linear_types = (nn.Linear,)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))

            if only_trainable and not param.requires_grad:
                skipped += 1
                continue

            if not _name_allowed(full_name, include_name_keywords):
                skipped += 1
                continue

            # FIRE only applies to matrix/kernel weights, not bias/norm scalars.
            if param.ndim < 2:
                skipped += 1
                continue

            if not ("weight" in param_name or "kernel" in param_name):
                skipped += 1
                continue

            old_shape = tuple(param.shape)

            try:
                if isinstance(module, linear_types) and param.ndim == 2:
                    new_param = _fire_linear_weight(param, num_iters=num_iters)

                elif isinstance(module, dense_conv_types) and param.ndim >= 3:
                    new_param = _fire_dense_conv_weight(param, num_iters=num_iters)

                elif param.ndim == 2:
                    # Generic 2D matrix, e.g. projection weights not implemented as nn.Linear.
                    new_param = _fire_linear_weight(param, num_iters=num_iters)

                elif param.ndim == 3:
                    # Generic sparse convolution kernel, e.g. MinkowskiEngine-style kernel.
                    new_param = _fire_sparse_kernel_weight(
                        module=module,
                        weight=param,
                        num_iters=num_iters,
                        sparse_kernel_layout=sparse_kernel_layout,
                    )

                else:
                    skipped += 1
                    continue

                if new_param.shape != param.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {full_name}: "
                        f"old={old_shape}, new={tuple(new_param.shape)}"
                    )

                param.copy_(new_param)
                changed += 1

                _log(
                    logger,
                    f"[FIRE] changed {full_name}, shape={old_shape}",
                )

            except Exception as e:
                skipped += 1
                _log(
                    logger,
                    f"[FIRE] skipped {full_name}, shape={old_shape}, reason={repr(e)}",
                )

    _log(
        logger,
        f"[FIRE] done. changed={changed}, skipped={skipped}, "
        f"num_iters={num_iters}, only_trainable={only_trainable}, "
        f"include_name_keywords={list(include_name_keywords or [])}, "
        f"sparse_kernel_layout={sparse_kernel_layout}",
    )

    return changed


@torch.no_grad()
def reset_prototype_memory(proto_mod: nn.Module, logger=None) -> None:
    """
    FIRE changes the feature/projection space.
    Prototype memory should be reset or rebuilt after FIRE.
    """
    if hasattr(proto_mod, "reset") and callable(getattr(proto_mod, "reset")):
        proto_mod.reset()
        _log(logger, "[FIRE] prototype memory reset by proto_mod.reset().")
        return

    if hasattr(proto_mod, "proto"):
        proto_mod.proto.zero_()

    if hasattr(proto_mod, "counts"):
        proto_mod.counts.zero_()

    _log(logger, "[FIRE] prototype memory zeroed: proto/counts.")
