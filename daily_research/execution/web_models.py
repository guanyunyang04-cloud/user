from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskRunRequest(BaseModel):
    task_name: str = Field(min_length=1)
    job_label: str = ""
    python_executable: str = ""
    form_payload: dict[str, Any] = Field(default_factory=dict)
    raw_args_text: str = ""
    force_unlock: bool = False
    background: bool = True


class ResumeRequest(BaseModel):
    job_id: str = ""
    job_label: str = ""
    force_unlock: bool = False
    background: bool = True


class UnlockRequest(BaseModel):
    force: bool = False


class AccountPositionInput(BaseModel):
    stock: str = ""
    shares: int | float | str = ""
    cost_price: float | int | str = 0.0


class AccountSnapshotRequest(BaseModel):
    available_cash: float | int | str | None = None
    positions: list[AccountPositionInput] = Field(default_factory=list)
