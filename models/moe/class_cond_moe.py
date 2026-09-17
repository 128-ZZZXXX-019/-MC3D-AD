import copy
import importlib
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def _import_object(path: str):
    """
    支持两种写法：

    1. module path:
       models.reconstructions.uniad

    2. class path:
       models.reconstructions.uniad.UniAD
    """
    try:
        module = importlib.import_module(path)
        return module
    except ImportError:
        module_path, obj_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, obj_name)


def _build_module(type_path: str, kwargs: Dict[str, Any]) -> nn.Module:
    """
    尽量兼容你项目里不同的 model 写法。

    会依次尝试：
    - 模块里是否有 build_model
    - 模块里是否有 get_model
    - 模块里是否有 UniAD
    - 模块里是否有 Model
    - type_path 本身是否就是 class
    """
    obj = _import_object(type_path)

    if isinstance(obj, type):
        return obj(**kwargs)

    for name in ["build_model", "get_model", "UniAD", "Model"]:
        if hasattr(obj, name):
            cls_or_fn = getattr(obj, name)
            return cls_or_fn(**kwargs)

    raise RuntimeError(
        f"Cannot build expert from {type_path}. "
        f"Please expose one of: build_model, get_model, UniAD, Model, "
        f"or pass a full class path."
    )


def _get_batch_size(x: Any) -> Optional[int]:
    if torch.is_tensor(x):
        return x.shape[0]

    if isinstance(x, dict):
        for v in x.values():
            if torch.is_tensor(v) and v.ndim > 0:
                return v.shape[0]

    return None


def _slice_batch(x: Any, mask: torch.Tensor, batch_size: int) -> Any:
    """
    只切第一维等于 batch_size 的 Tensor。
    其他内容原样保留。
    """
    if torch.is_tensor(x):
        if x.ndim > 0 and x.shape[0] == batch_size:
            return x[mask]
        return x

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[k] = _slice_batch(v, mask, batch_size)
        return out

    if isinstance(x, list):
        return [_slice_batch(v, mask, batch_size) for v in x]

    if isinstance(x, tuple):
        return tuple(_slice_batch(v, mask, batch_size) for v in x)

    return x


def _scatter_tensor(
    full: Optional[torch.Tensor],
    part: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    将某个类别 expert 的输出放回原 batch 位置。
    """
    if part.ndim == 0:
        # 标量 loss 类输出不适合 scatter，这里返回原标量。
        return part

    if part.shape[0] != int(mask.sum().item()):
        return part

    if full is None:
        shape = (batch_size,) + tuple(part.shape[1:])
        full = part.new_empty(shape)

    full[mask] = part
    return full


def _scatter_output(
    full: Any,
    part: Any,
    mask: torch.Tensor,
    batch_size: int,
) -> Any:
    """
    递归 scatter Tensor / dict / list / tuple 输出。
    """
    if torch.is_tensor(part):
        if torch.is_tensor(full):
            return _scatter_tensor(full, part, mask, batch_size)
        return _scatter_tensor(None, part, mask, batch_size)

    if isinstance(part, dict):
        if full is None:
            full = {}
        for k, v in part.items():
            full[k] = _scatter_output(full.get(k, None), v, mask, batch_size)
        return full

    if isinstance(part, list):
        if full is None:
            full = [None for _ in part]
        return [
            _scatter_output(full[i], part[i], mask, batch_size)
            for i in range(len(part))
        ]

    if isinstance(part, tuple):
        if full is None:
            full = tuple(None for _ in part)
        return tuple(
            _scatter_output(full[i], part[i], mask, batch_size)
            for i in range(len(part))
        )

    return part


def _mix_outputs(
    shared: Any,
    private: Any,
    shared_weight: torch.Tensor,
    private_weight: torch.Tensor,
    mix_keys: Optional[Tuple[str, ...]] = None,
) -> Any:
    """
    混合 shared expert 和 private expert 的输出。

    如果输出是 Tensor:
        out = ws * shared + wp * private

    如果输出是 dict:
        默认混合所有 shape 一致的 Tensor。
        如果 mix_keys 不为 None，则只混合指定 key。
    """
    if torch.is_tensor(shared) and torch.is_tensor(private):
        if shared.shape == private.shape:
            return shared_weight * shared + private_weight * private
        return private

    if isinstance(shared, dict) and isinstance(private, dict):
        out = {}
        keys = set(shared.keys()) | set(private.keys())

        for k in keys:
            if k not in shared:
                out[k] = private[k]
                continue

            if k not in private:
                out[k] = shared[k]
                continue

            if mix_keys is not None and k not in mix_keys:
                out[k] = private[k]
                continue

            out[k] = _mix_outputs(
                shared[k],
                private[k],
                shared_weight,
                private_weight,
                mix_keys=None,
            )

        return out

    if isinstance(shared, list) and isinstance(private, list):
        return [
            _mix_outputs(s, p, shared_weight, private_weight, mix_keys)
            for s, p in zip(shared, private)
        ]

    if isinstance(shared, tuple) and isinstance(private, tuple):
        return tuple(
            _mix_outputs(s, p, shared_weight, private_weight, mix_keys)
            for s, p in zip(shared, private)
        )

    return private


class ClassConditionalMoE(nn.Module):
    """
    Class-conditioned MoE wrapper.

    结构：
        shared_expert
        private_experts[class_id]

    forward 时：
        shared_expert 一定激活
        private_experts 只激活 batch 中出现的类别

    测试 batch_size=1 时：
        只激活 shared_expert 和对应 class 的 private_expert
    """

    def __init__(
        self,
        num_classes: int,
        expert_type: str,
        expert_kwargs: Optional[Dict[str, Any]] = None,
        class_key: str = "moe_class_id",
        mix_mode: str = "learnable_scalar",
        shared_weight: float = 0.5,
        private_weight: float = 0.5,
        mix_keys: Optional[list] = None,
        init_private_from_shared: bool = True,
        fallback_to_shared: bool = True,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.expert_type = expert_type
        self.expert_kwargs = expert_kwargs or {}
        self.class_key = class_key
        self.mix_mode = mix_mode
        self.fallback_to_shared = fallback_to_shared
        self.mix_keys = tuple(mix_keys) if mix_keys is not None else None

        self.shared_expert = _build_module(
            self.expert_type,
            copy.deepcopy(self.expert_kwargs),
        )

        self.private_experts = nn.ModuleList([
            _build_module(
                self.expert_type,
                copy.deepcopy(self.expert_kwargs),
            )
            for _ in range(self.num_classes)
        ])

        if init_private_from_shared:
            shared_state = self.shared_expert.state_dict()
            for expert in self.private_experts:
                expert.load_state_dict(shared_state, strict=False)

        if mix_mode == "learnable_scalar":
            # sigmoid(0)=0.5，初始 shared/private 各占一半
            self.private_logit = nn.Parameter(torch.zeros(1))
        elif mix_mode == "fixed":
            total = float(shared_weight + private_weight)
            if total <= 0:
                raise ValueError("shared_weight + private_weight must be > 0.")
            self.register_buffer(
                "_shared_weight",
                torch.tensor(float(shared_weight) / total),
                persistent=False,
            )
            self.register_buffer(
                "_private_weight",
                torch.tensor(float(private_weight) / total),
                persistent=False,
            )
        else:
            raise ValueError(
                f"Unsupported mix_mode: {mix_mode}. "
                f"Use 'learnable_scalar' or 'fixed'."
            )

    def _resolve_class_id(
        self,
        x: Any,
        class_id: Optional[torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        if class_id is not None:
            return class_id

        if self.class_key in kwargs:
            return kwargs[self.class_key]

        for key in ["class_id", "cls_id", "pred_class", "label", "labels", "y"]:
            if key in kwargs:
                return kwargs[key]

        if isinstance(x, dict):
            if self.class_key in x:
                return x[self.class_key]

            for key in ["class_id", "cls_id", "pred_class", "label", "labels", "y"]:
                if key in x:
                    return x[key]

        return None

    def _weights(self, device: torch.device):
        if self.mix_mode == "learnable_scalar":
            private_weight = torch.sigmoid(self.private_logit).to(device)
            shared_weight = 1.0 - private_weight
            return shared_weight, private_weight

        return self._shared_weight.to(device), self._private_weight.to(device)

    def forward(self, x: Any, *args, class_id: Optional[torch.Tensor] = None, **kwargs):
        batch_size = _get_batch_size(x)

        cls = self._resolve_class_id(x, class_id, kwargs)

        shared_out = self.shared_expert(x, *args, **kwargs)

        if cls is None:
            if self.fallback_to_shared:
                return shared_out
            raise RuntimeError(
                "ClassConditionalMoE needs class_id / moe_class_id / label / pred_class, "
                "but none was provided."
            )

        if batch_size is None:
            raise RuntimeError("Cannot infer batch size for ClassConditionalMoE routing.")

        if not torch.is_tensor(cls):
            cls = torch.as_tensor(cls)

        device = None
        if torch.is_tensor(x):
            device = x.device
        elif isinstance(x, dict):
            for v in x.values():
                if torch.is_tensor(v):
                    device = v.device
                    break

        if device is None:
            device = cls.device

        cls = cls.to(device=device).long().view(-1)

        if cls.numel() == 1 and batch_size > 1:
            cls = cls.expand(batch_size)

        if cls.numel() != batch_size:
            raise RuntimeError(
                f"class_id length mismatch. Got {cls.numel()}, "
                f"but batch size is {batch_size}."
            )

        if torch.any(cls < 0) or torch.any(cls >= self.num_classes):
            raise RuntimeError(
                f"class_id out of range. Valid range is [0, {self.num_classes - 1}], "
                f"got min={int(cls.min())}, max={int(cls.max())}."
            )

        private_out = None

        # 只激活当前 batch 里出现的类别专家
        for c in torch.unique(cls):
            c_int = int(c.item())
            mask = cls == c

            x_part = _slice_batch(x, mask, batch_size)
            args_part = tuple(_slice_batch(a, mask, batch_size) for a in args)
            kwargs_part = {
                k: _slice_batch(v, mask, batch_size)
                for k, v in kwargs.items()
            }

            out_part = self.private_experts[c_int](
                x_part,
                *args_part,
                **kwargs_part,
            )

            private_out = _scatter_output(
                private_out,
                out_part,
                mask,
                batch_size,
            )

        ws, wp = self._weights(device)

        return _mix_outputs(
            shared_out,
            private_out,
            ws,
            wp,
            mix_keys=self.mix_keys,
        )
