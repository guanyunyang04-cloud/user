from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class DaySetTensorDataset(Dataset):
    """Dataset that keeps every trading date as a complete stock set."""

    def __init__(
        self,
        *,
        date_codes: torch.Tensor | np.ndarray,
        static_x: torch.Tensor | np.ndarray,
        sequence_x: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
        duration: torch.Tensor | np.ndarray,
        action_soft: torch.Tensor | np.ndarray,
        sample_targets: dict[str, torch.Tensor | np.ndarray],
        daily_x: torch.Tensor | np.ndarray,
        daily_targets: dict[str, torch.Tensor | np.ndarray] | None = None,
        allowed_date_codes: torch.Tensor | np.ndarray | list[int] | None = None,
    ) -> None:
        self.date_codes = torch.as_tensor(date_codes, dtype=torch.long).flatten()
        self.static_x = torch.as_tensor(static_x, dtype=torch.float32)
        self.sequence_x = torch.as_tensor(sequence_x, dtype=torch.float32)
        self.action = torch.as_tensor(action, dtype=torch.long).flatten()
        self.duration = torch.as_tensor(duration, dtype=torch.long).flatten()
        self.action_soft = torch.as_tensor(action_soft, dtype=torch.float32)
        self.sample_targets = {
            str(name): torch.as_tensor(values, dtype=torch.float32).flatten()
            for name, values in dict(sample_targets or {}).items()
        }
        self.daily_x = torch.as_tensor(daily_x, dtype=torch.float32)
        self.daily_targets = {
            str(name): torch.as_tensor(values, dtype=torch.float32).flatten()
            for name, values in dict(daily_targets or {}).items()
        }
        if self.static_x.shape[0] != self.date_codes.numel() or self.sequence_x.shape[0] != self.date_codes.numel():
            raise ValueError("DaySetTensorDataset requires aligned date/static/sequence row counts.")
        all_codes = torch.unique(self.date_codes).tolist()
        if allowed_date_codes is None:
            selected_codes = [int(code) for code in all_codes]
        else:
            allowed = {int(code) for code in torch.as_tensor(allowed_date_codes, dtype=torch.long).flatten().tolist()}
            selected_codes = [int(code) for code in all_codes if int(code) in allowed]
        self.date_code_values = sorted(selected_codes)
        self.row_indices_by_date = [
            torch.nonzero(self.date_codes == int(code), as_tuple=False).flatten()
            for code in self.date_code_values
        ]

    def __len__(self) -> int:
        return len(self.date_code_values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        date_code = int(self.date_code_values[index])
        row_indices = self.row_indices_by_date[index]
        daily_index = min(max(date_code, 0), max(0, int(self.daily_x.shape[0]) - 1))
        return {
            "date_code": self.date_codes[row_indices],
            "row_indices": row_indices,
            "static_x": self.static_x[row_indices],
            "sequence_x": self.sequence_x[row_indices],
            "action": self.action[row_indices],
            "duration": self.duration[row_indices],
            "action_soft": self.action_soft[row_indices],
            "sample_targets": {
                name: values[row_indices]
                for name, values in self.sample_targets.items()
            },
            "daily_x": self.daily_x[daily_index],
            "daily_targets": {
                name: values[daily_index]
                for name, values in self.daily_targets.items()
                if values.numel() > daily_index
            },
        }


def _collate_day_set_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("_collate_day_set_batch requires at least one day item.")
    batch_size = len(items)
    max_rows = max(int(item["static_x"].shape[0]) for item in items)
    static_dim = int(items[0]["static_x"].shape[-1])
    sequence_steps = int(items[0]["sequence_x"].shape[-2])
    sequence_dim = int(items[0]["sequence_x"].shape[-1])
    action_classes = int(items[0]["action_soft"].shape[-1])
    daily_dim = int(items[0]["daily_x"].shape[-1])

    static_x = torch.zeros(batch_size, max_rows, static_dim, dtype=torch.float32)
    sequence_x = torch.zeros(batch_size, max_rows, sequence_steps, sequence_dim, dtype=torch.float32)
    action = torch.zeros(batch_size, max_rows, dtype=torch.long)
    duration = torch.zeros(batch_size, max_rows, dtype=torch.long)
    action_soft = torch.zeros(batch_size, max_rows, action_classes, dtype=torch.float32)
    sample_mask = torch.zeros(batch_size, max_rows, dtype=torch.bool)
    date_code = torch.zeros(batch_size, max_rows, dtype=torch.long)
    row_indices = torch.full((batch_size, max_rows), -1, dtype=torch.long)
    daily_x = torch.zeros(batch_size, daily_dim, dtype=torch.float32)

    sample_target_names = sorted({name for item in items for name in item["sample_targets"].keys()})
    daily_target_names = sorted({name for item in items for name in item["daily_targets"].keys()})
    sample_targets = {
        name: torch.zeros(batch_size, max_rows, dtype=torch.float32)
        for name in sample_target_names
    }
    daily_targets = {
        name: torch.zeros(batch_size, dtype=torch.float32)
        for name in daily_target_names
    }
    for batch_index, item in enumerate(items):
        row_count = int(item["static_x"].shape[0])
        static_x[batch_index, :row_count] = item["static_x"].float()
        sequence_x[batch_index, :row_count] = item["sequence_x"].float()
        action[batch_index, :row_count] = item["action"].long()
        duration[batch_index, :row_count] = item["duration"].long()
        action_soft[batch_index, :row_count] = item["action_soft"].float()
        sample_mask[batch_index, :row_count] = True
        date_code[batch_index, :row_count] = item["date_code"].long()
        row_indices[batch_index, :row_count] = item["row_indices"].long()
        daily_x[batch_index] = item["daily_x"].float()
        for name in sample_target_names:
            if name in item["sample_targets"]:
                sample_targets[name][batch_index, :row_count] = item["sample_targets"][name].float()
        for name in daily_target_names:
            if name in item["daily_targets"]:
                daily_targets[name][batch_index] = item["daily_targets"][name].float()

    return {
        "static_x": static_x,
        "sequence_x": sequence_x,
        "action": action,
        "duration": duration,
        "action_soft": action_soft,
        "sample_mask": sample_mask,
        "date_code": date_code,
        "row_indices": row_indices,
        "sample_targets": sample_targets,
        "daily_x": daily_x,
        "daily_targets": daily_targets,
    }
