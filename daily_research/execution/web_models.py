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


class ModelTrainRequest(BaseModel):
    model_id: str = ""
    dataset_mode: str = "latest"
    dataset_id: str = ""
    start_date: str = ""
    end_date: str = ""
    job_label: str = ""
    force_unlock: bool = False
    background: bool = True
    advanced_args: str = ""


class DataRefreshRequest(BaseModel):
    as_of_date: str = ""
    start_date: str = ""
    universe: str = "all_a"
    domains: list[str] = Field(default_factory=lambda: ["market_daily"])
    job_label: str = ""
    force_unlock: bool = False
    background: bool = True
    advanced_args: str = ""


class TradePlanGenerateRequest(BaseModel):
    candidate_profile: str = ""
    positions_file: str = ""
    cash: float | int | str | None = None
    lot_size: int | str | None = None
    target_weight_top_k: int | str | None = None
    target_weight_min_weight: float | int | str | None = None
    raw_args_text: str = ""
    job_label: str = ""
    force_unlock: bool = False
    background: bool = True
