# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type, dim):
    norm_type = str(norm_type).lower()

    if norm_type == "bn":
        return nn.BatchNorm1d(dim)

    if norm_type == "ln":
        return nn.LayerNorm(dim)

    if norm_type in ["none", "identity"]:
        return nn.Identity()

    raise ValueError(f"Unsupported norm_type={norm_type}")


def _as_bool(x, default=False):
    if x is None:
        return bool(default)
    if torch.is_tensor(x):
        if x.numel() == 0:
            return bool(default)
        return bool(x.detach().cpu().view(-1)[0].item())
    if isinstance(x, str):
        return x.lower() in ["1", "true", "yes", "y"]
    return bool(x)


class PointMAEProtoClsHead(nn.Module):
    # Prototype classification head for PointMAE features.
    #
    # Supported input keys:
    # - xyz_features: legacy main features. In this modified pipeline it means:
    #   registered features when is_registered=True, otherwise raw features.
    # - raw_xyz_features: unregistered feature branch.
    #
    # Outputs:
    # - cls_logits: selected/fused logits for compatibility.
    # - raw_cls_logits/raw_proto_feature/raw_cls_feature.
    # - reg_cls_logits/reg_proto_feature/reg_cls_feature when registered features exist.
    def __init__(
        self,
        inplanes=384,
        cls_num=12,
        hidden_dim=128,
        proj_dim=128,
        dropout=0.0,
        pooling="avgmax",
        norm_type="ln",
        temperature=0.07,
        prototype_mode="learnable",
        proto_momentum=0.9,
        feature_mode="auto",
        logit_fusion="avg",
        raw_logit_weight=1.0,
        reg_logit_weight=1.0,
    ):
        super().__init__()

        self.inplanes = int(inplanes)
        self.cls_num = int(cls_num)
        self.hidden_dim = int(hidden_dim)
        self.proj_dim = int(proj_dim)
        self.pooling = str(pooling).lower()
        self.temperature = float(temperature)
        self.prototype_mode = str(prototype_mode).lower()
        self.proto_momentum = float(proto_momentum)

        self.feature_mode = str(feature_mode).lower()
        self.logit_fusion = str(logit_fusion).lower()
        self.raw_logit_weight = float(raw_logit_weight)
        self.reg_logit_weight = float(reg_logit_weight)

        if self.pooling == "avgmax":
            head_inplanes = self.inplanes * 2
        elif self.pooling in ["avg", "max"]:
            head_inplanes = self.inplanes
        else:
            raise ValueError(f"Unsupported pooling={pooling}. Use 'avg', 'max', or 'avgmax'.")

        self.projector = nn.Sequential(
            nn.Linear(head_inplanes, hidden_dim, bias=False),
            _make_norm(norm_type, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, proj_dim, bias=True),
        )

        if self.prototype_mode == "learnable":
            self.prototypes = nn.Parameter(torch.empty(self.cls_num, proj_dim))
            nn.init.trunc_normal_(self.prototypes, std=0.02)

        elif self.prototype_mode == "ema":
            self.register_buffer("prototypes", torch.zeros(self.cls_num, proj_dim))
            self.register_buffer("counts", torch.zeros(self.cls_num))

        else:
            raise ValueError(
                f"Unsupported prototype_mode={prototype_mode}. Use 'learnable' or 'ema'."
            )

    def _format_feature(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"PointMAEProtoClsHead expects 3D feature, got shape={x.shape}")

        if x.shape[1] == self.inplanes:
            return x

        if x.shape[-1] == self.inplanes:
            return x.transpose(1, 2).contiguous()

        raise RuntimeError(
            f"Cannot infer channel dimension from shape={x.shape}, expected channel dim={self.inplanes}."
        )

    def _pool_feature(self, x):
        # x: [B,C,G]
        if self.pooling == "avg":
            return x.mean(dim=-1)

        if self.pooling == "max":
            return x.max(dim=-1).values

        if self.pooling == "avgmax":
            avg_feat = x.mean(dim=-1)
            max_feat = x.max(dim=-1).values
            return torch.cat([avg_feat, max_feat], dim=1)

        raise RuntimeError(f"Unexpected pooling={self.pooling}")

    def _compute_branch(self, x):
        x = self._format_feature(x)
        pooled = self._pool_feature(x)
        z = self.projector(pooled)
        z = F.normalize(z, dim=1)

        proto = F.normalize(self.prototypes, dim=1)
        logits = torch.matmul(z, proto.t()) / self.temperature

        return pooled, z, logits

    @torch.no_grad()
    def update_prototypes(self, z, labels):
        # EMA prototype update. z must be [B,proj_dim] and already normalized.
        if self.prototype_mode != "ema":
            return

        labels = labels.view(-1).long()

        for cls_id in labels.unique():
            cid = int(cls_id.item())
            idx = labels == cid

            if idx.sum() == 0:
                continue

            z_mean = F.normalize(z[idx].mean(dim=0, keepdim=True), dim=1)
            old = self.prototypes[cid: cid + 1]

            if self.counts[cid].item() == 0:
                new = z_mean
            else:
                new = F.normalize(
                    self.proto_momentum * old + (1.0 - self.proto_momentum) * z_mean,
                    dim=1,
                )

            self.prototypes[cid: cid + 1] = new
            self.counts[cid] += idx.sum()

    def fuse_logits(self, outputs):
        raw_logits = outputs.get("raw_cls_logits", None)
        reg_logits = outputs.get("reg_cls_logits", None)

        if reg_logits is None:
            reg_logits = outputs.get("registered_cls_logits", None)

        if raw_logits is not None and reg_logits is not None:
            if self.logit_fusion in ["avg", "mean", "weighted_avg", "weighted"]:
                w_raw = max(self.raw_logit_weight, 0.0)
                w_reg = max(self.reg_logit_weight, 0.0)
                denom = max(w_raw + w_reg, 1e-12)
                return (w_raw * raw_logits + w_reg * reg_logits) / denom

            if self.logit_fusion in ["sum", "add"]:
                return self.raw_logit_weight * raw_logits + self.reg_logit_weight * reg_logits

            if self.logit_fusion in ["raw", "unregistered"]:
                return raw_logits

            if self.logit_fusion in ["reg", "registered", "aligned"]:
                return reg_logits

            raise ValueError(
                f"Unsupported logit_fusion={self.logit_fusion}. "
                "Use avg, sum, raw, or registered."
            )

        if raw_logits is not None:
            return raw_logits

        if reg_logits is not None:
            return reg_logits

        if "cls_logits" in outputs:
            return outputs["cls_logits"]

        raise KeyError(f"No logits found in outputs keys={list(outputs.keys())}")

    def _write_branch_outputs(self, out, branch_name, pooled, z, logits):
        if branch_name == "raw":
            out["raw_cls_feature"] = pooled
            out["raw_proto_feature"] = z
            out["raw_cls_logits"] = logits
            return

        if branch_name in ["reg", "registered"]:
            out["reg_cls_feature"] = pooled
            out["reg_proto_feature"] = z
            out["reg_cls_logits"] = logits

            # Backward-compatible aliases.
            out["registered_cls_feature"] = pooled
            out["registered_proto_feature"] = z
            out["registered_cls_logits"] = logits
            return

        out[f"{branch_name}_cls_feature"] = pooled
        out[f"{branch_name}_proto_feature"] = z
        out[f"{branch_name}_cls_logits"] = logits

    def _legacy_branch_for_cls_feature(self, out):
        # Prefer registered feature as legacy cls_feature when available.
        if "reg_proto_feature" in out:
            out["cls_feature"] = out["reg_cls_feature"]
            out["proto_feature"] = out["reg_proto_feature"]
        elif "raw_proto_feature" in out:
            out["cls_feature"] = out["raw_cls_feature"]
            out["proto_feature"] = out["raw_proto_feature"]
        return out

    def forward(self, inputs, labels=None, update_proto=False):
        if isinstance(inputs, dict):
            out = dict(inputs)
            raw_x = inputs.get("raw_xyz_features", None)
            main_x = inputs.get("xyz_features", None)
            is_registered = _as_bool(inputs.get("is_registered", False), default=False)
        else:
            out = {}
            raw_x = None
            main_x = inputs
            is_registered = False

        branches = []

        if self.feature_mode in ["raw", "unregistered"]:
            if raw_x is not None:
                branches.append(("raw", raw_x))
            elif main_x is not None:
                branches.append(("raw", main_x))
            else:
                raise KeyError("feature_mode='raw' but neither raw_xyz_features nor xyz_features exists.")

        elif self.feature_mode in ["reg", "registered", "aligned"]:
            if main_x is None:
                raise KeyError("feature_mode='registered' but xyz_features does not exist.")
            branches.append(("reg", main_x))

        elif self.feature_mode in ["dual", "both", "auto"]:
            if raw_x is not None:
                branches.append(("raw", raw_x))

            if main_x is not None:
                if is_registered:
                    branches.append(("reg", main_x))
                elif raw_x is None:
                    branches.append(("raw", main_x))

            if len(branches) == 0:
                raise KeyError(
                    "PointMAEProtoClsHead cannot find features. "
                    "Expected xyz_features or raw_xyz_features."
                )

        else:
            raise ValueError(
                f"Unsupported feature_mode={self.feature_mode}. "
                "Use auto, raw, registered, or dual."
            )

        # Avoid accidental duplicate if raw_xyz_features and xyz_features are the same object in no-registration pass.
        unique_branches = []
        seen = set()
        for name, tensor in branches:
            marker = (name, id(tensor))
            if marker not in seen:
                unique_branches.append((name, tensor))
                seen.add(marker)

        for branch_name, x in unique_branches:
            pooled, z, logits = self._compute_branch(x)
            self._write_branch_outputs(out, branch_name, pooled, z, logits)

        out["cls_logits"] = self.fuse_logits(out)
        out = self._legacy_branch_for_cls_feature(out)

        # Optional in-forward EMA update is kept for direct use, but train_cls.py updates EMA manually
        # to make DDP prototypes consistent and to recompute logits after the update.
        if (
            self.training
            and self.prototype_mode == "ema"
            and update_proto
            and labels is not None
        ):
            z_list = []
            y_list = []
            if "raw_proto_feature" in out:
                z_list.append(out["raw_proto_feature"].detach())
                y_list.append(labels.view(-1).long())
            if "reg_proto_feature" in out:
                z_list.append(out["reg_proto_feature"].detach())
                y_list.append(labels.view(-1).long())
            if z_list:
                self.update_prototypes(torch.cat(z_list, dim=0), torch.cat(y_list, dim=0))

        return out


def pointmae_proto_cls_head(
    inplanes=384,
    cls_num=12,
    hidden_dim=128,
    proj_dim=128,
    dropout=0.0,
    pooling="avgmax",
    norm_type="ln",
    temperature=0.07,
    prototype_mode="learnable",
    proto_momentum=0.9,
    feature_mode="auto",
    logit_fusion="avg",
    raw_logit_weight=1.0,
    reg_logit_weight=1.0,
):
    return PointMAEProtoClsHead(
        inplanes=inplanes,
        cls_num=cls_num,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        dropout=dropout,
        pooling=pooling,
        norm_type=norm_type,
        temperature=temperature,
        prototype_mode=prototype_mode,
        proto_momentum=proto_momentum,
        feature_mode=feature_mode,
        logit_fusion=logit_fusion,
        raw_logit_weight=raw_logit_weight,
        reg_logit_weight=reg_logit_weight,
    )
