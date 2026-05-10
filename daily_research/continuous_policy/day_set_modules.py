from __future__ import annotations

import math

import torch
from torch import nn


class PortfolioSlotAttention(nn.Module):
    def __init__(self, *, input_dim: int, slot_count: int = 32, slot_dim: int = 128) -> None:
        super().__init__()
        self.slot_count = max(1, int(slot_count))
        self.slot_dim = max(1, int(slot_dim))
        self.slot_queries = nn.Parameter(torch.randn(self.slot_count, self.slot_dim) * 0.02)
        self.key_projection = nn.Linear(int(input_dim), self.slot_dim)
        self.value_projection = nn.Linear(int(input_dim), int(input_dim))
        self.output_projection = nn.Linear(int(input_dim), int(input_dim))

    def forward(self, row_features: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        mask = sample_mask.to(device=row_features.device, dtype=torch.bool)
        keys = self.key_projection(row_features)
        values = self.value_projection(row_features)
        logits = torch.einsum("bnd,kd->bkn", keys, self.slot_queries) / math.sqrt(float(self.slot_dim))
        logits = logits.masked_fill(~mask[:, None, :], -1.0e9)
        attention = torch.softmax(logits, dim=-1)
        attention = torch.where(mask[:, None, :], attention, torch.zeros_like(attention))
        normalizer = torch.clamp(attention.sum(dim=-1, keepdim=True), min=1.0e-8)
        attention = attention / normalizer
        slot_context = torch.einsum("bkn,bnh->bkh", attention, values)
        pooled = slot_context.mean(dim=1)
        return self.output_projection(pooled)
