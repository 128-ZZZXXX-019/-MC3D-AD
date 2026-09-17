# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Same logic as UCA-3DAL network/cpe.py:
      features: [B, V, D]
      labels:   [B]
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device

        if features.dim() != 3:
            raise RuntimeError(
                f"SupConLoss expects features [B,V,D], got {tuple(features.shape)}"
            )

        batch_size, view_num, dim = features.shape

        feats = F.normalize(
            features.reshape(batch_size * view_num, dim),
            dim=1,
        )

        labels = labels.view(-1, 1)

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
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1.0e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1.0e-12)

        loss = -mean_log_prob_pos
        loss = loss.view(batch_size, view_num).mean()

        return loss


class PrototypeMemory(nn.Module):
    """
    EMA-updated class prototypes.

    Same role as UCA-3DAL PrototypeMemory.
    """

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

            if idx.sum() == 0:
                continue

            z_mean = F.normalize(z[idx].mean(dim=0, keepdim=True), dim=1)
            old = self.proto[cid: cid + 1]

            if self.counts[cid].item() == 0:
                new = z_mean
            else:
                new = F.normalize(
                    self.momentum * old + (1.0 - self.momentum) * z_mean,
                    dim=1,
                )

            self.proto[cid: cid + 1] = new
            self.counts[cid] += idx.sum()


class PrototypeNCELoss(nn.Module):
    """
    Prototype NCE loss.

    Same role as UCA-3DAL PrototypeNCELoss.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        logits = torch.matmul(z, prototypes.T) / self.temperature
        loss = F.cross_entropy(logits, y.view(-1).long())

        return loss
