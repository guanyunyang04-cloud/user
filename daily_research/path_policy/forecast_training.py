from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import inspect
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from daily_research.continuous_policy.runtime import write_json
from daily_research.path_policy.forecast_dataset import ForecastMemmapDataset, ForecastSequenceDataset
from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS, PATH20_HORIZON
from daily_research.path_policy.models import (
    DLinearPath20Forecaster,
    GRUPath20Forecaster,
    LinearPath20Forecaster,
    PatchTransformerPath20Forecaster,
    PATH20_DECISION_AUX_DIM,
    PATH20_FORECAST_AUX_DIM,
    Path20ForecasterMLP,
    SectorSlotMixerPath20Forecaster,
    StaticContextPath20Forecaster,
    StockMixerPath20Forecaster,
    pairwise_rank_loss,
    pinball_loss,
)


FORECAST_MODEL_FAMILIES = (
    "linear_last_day",
    "dlinear_sequence",
    "mlp_last_day",
    "gru_sequence",
    "patch_transformer",
    "gru_sequence_static_context",
    "patch_transformer_static_context",
    "stock_mixer_sequence",
    "sector_slot_mixer_sequence",
)
FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES = ("stock_mixer_sequence", "sector_slot_mixer_sequence")
FORECAST_OUTPUT_PROFILES = ("forecast_path_v1", "decision_utility_v1")
FORECAST_SELECTION_PROFILES = ("multiscale", "trend20", "short_burst", "decision_utility")
FORECAST_LOSS_PROFILES = ("default", "rank_aux", "multitask_v1", "decision_utility_v1")
FORECAST_RANKING_BASELINES = ("none", "lightgbm", "xgboost")
FORECAST_RISK_AUX_NAMES = ("downside_floor_20d", "worst_1d_20d", "upside_20d")
FORECAST_RANK_LOSS_WEIGHTS = {1: 0.0025, 3: 0.0050, 5: 0.0075, 10: 0.0075, 20: 0.0100}
FORECAST_PROFILE_HORIZON_WEIGHTS: dict[str, dict[int, float]] = {
    "multiscale": {1: 0.05, 3: 0.15, 5: 0.20, 10: 0.25, 20: 0.25},
    "trend20": {10: 0.35, 20: 0.65},
    "short_burst": {1: 0.05, 3: 0.35, 5: 0.30, 10: 0.15, 20: 0.05},
}


class _EagerTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ForecastSequenceDataset, indices: np.ndarray, *, target_scale: float) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_scale = float(target_scale)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        row_idx = int(self.indices[int(item)])
        y_risk = np.asarray(
            [
                self.dataset.y_max_drawdown_20d[row_idx],
                self.dataset.y_worst_1d_20d[row_idx],
                self.dataset.y_upside_20d[row_idx],
            ],
            dtype=np.float32,
        )
        items: tuple[torch.Tensor, ...] = (
            torch.as_tensor(self.dataset.x[row_idx], dtype=torch.float32),
            torch.as_tensor(self.dataset.y_daily_excess[row_idx] * self.target_scale, dtype=torch.float32),
            torch.as_tensor(self.dataset.y_cum_excess[row_idx] * self.target_scale, dtype=torch.float32),
            torch.as_tensor(y_risk * self.target_scale, dtype=torch.float32),
            torch.as_tensor(row_idx, dtype=torch.long),
        )
        if self.dataset.static_context_ids is not None:
            items = (*items, torch.as_tensor(self.dataset.static_context_ids[row_idx], dtype=torch.long))
        return items


class _ForecastDatasetView:
    def __init__(self, dataset: ForecastSequenceDataset | ForecastMemmapDataset) -> None:
        self.dataset = dataset
        self.manifest = dict(dataset.manifest)
        self.normalization_manifest = dict(dataset.normalization_manifest)
        self.feature_columns = list(dataset.feature_columns)
        self.dataset_mode = str(self.manifest.get("dataset_mode", "eager"))
        self.static_context_schema = dict(self.manifest.get("static_context_schema", {}) or {"enabled": False})
        self.static_context_vocab_sizes = dict(self.static_context_schema.get("vocab_sizes", {}) or {})
        self.symbol_vocab_fingerprint = str(self.manifest.get("symbol_vocab_fingerprint", "") or "")
        self.industry_vocab_fingerprint = str(self.manifest.get("industry_vocab_fingerprint", "") or "")
        self.board_vocab_fingerprint = str(self.manifest.get("board_vocab_fingerprint", "") or "")
        if isinstance(dataset, ForecastMemmapDataset):
            self.row_count = int(dataset.row_count)
            self.input_dim = int(dataset.input_dim)
            self.lookback_days = int(dataset.lookback_days)
        else:
            self.row_count = int(dataset.x.shape[0])
            self.input_dim = int(dataset.x.shape[-1]) if dataset.x.ndim == 3 else 0
            self.lookback_days = int(dataset.x.shape[1]) if dataset.x.ndim == 3 else int(self.manifest.get("lookback_days", 0))

    def role_indices(self, role: str) -> np.ndarray:
        if isinstance(self.dataset, ForecastMemmapDataset):
            return self.dataset.role_indices(role)
        return np.flatnonzero(self.dataset.role == role)

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float) -> torch.utils.data.Dataset:
        if isinstance(self.dataset, ForecastMemmapDataset):
            return self.dataset.torch_dataset(indices, target_scale=target_scale)
        return _EagerTorchDataset(self.dataset, indices, target_scale=target_scale)

    def date_batch_torch_dataset(self, indices: np.ndarray, *, target_scale: float) -> torch.utils.data.Dataset:
        if not isinstance(self.dataset, ForecastMemmapDataset):
            raise ValueError("date-level forecast batches require a memmap dataset.")
        return self.dataset.date_batch_torch_dataset(indices, target_scale=target_scale)

    @property
    def supports_static_context(self) -> bool:
        return bool(self.static_context_schema.get("enabled", False))


class LinearLastDayPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int = PATH20_HORIZON,
        output_profile: str = "forecast_path_v1",
    ) -> None:
        super().__init__()
        self.output_profile = str(output_profile or "forecast_path_v1").strip().lower()
        self.base = LinearPath20Forecaster(input_dim=int(input_dim), horizon=int(horizon), output_profile=self.output_profile)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.base(x)


class MLPLastDayPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        dropout: float = 0.15,
        horizon: int = PATH20_HORIZON,
        output_profile: str = "forecast_path_v1",
    ) -> None:
        super().__init__()
        self.output_profile = str(output_profile or "forecast_path_v1").strip().lower()
        self.base = Path20ForecasterMLP(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            horizon=int(horizon),
            output_profile=self.output_profile,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.base(x)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_forecast_model(
    family: str,
    *,
    input_dim: int,
    hidden_dim: int = 192,
    horizon: int = PATH20_HORIZON,
    dropout: float = 0.15,
    gru_layers: int = 2,
    transformer_layers: int = 4,
    transformer_heads: int = 6,
    patch_sizes: tuple[int, ...] | list[int] = (4, 20),
    static_context_vocab_sizes: dict[str, int] | None = None,
    static_context_embedding_dims: dict[str, int] | None = None,
    static_context_fields: tuple[str, ...] | list[str] | None = None,
    static_context_dropout: float = 0.20,
    slot_count: int = 8,
    output_profile: str = "forecast_path_v1",
) -> nn.Module:
    family = str(family).strip()
    output_profile = str(output_profile or "forecast_path_v1").strip().lower()
    if output_profile not in FORECAST_OUTPUT_PROFILES:
        raise ValueError(f"Unsupported forecast output profile: {output_profile}")
    if family == "linear_last_day":
        return LinearLastDayPath20Forecaster(input_dim=input_dim, horizon=horizon, output_profile=output_profile)
    if family == "dlinear_sequence":
        return DLinearPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            output_profile=output_profile,
        )
    if family == "mlp_last_day":
        return MLPLastDayPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            output_profile=output_profile,
        )
    if family == "gru_sequence":
        return GRUPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
            output_profile=output_profile,
        )
    if family == "patch_transformer":
        requested_heads = max(int(transformer_heads), 1)
        num_heads = requested_heads if int(hidden_dim) % requested_heads == 0 else 1
        return PatchTransformerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_profile=output_profile,
        )
    if family == "gru_sequence_static_context":
        temporal = GRUPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
        )
        return StaticContextPath20Forecaster(
            temporal_encoder=temporal,
            temporal_dim=int(hidden_dim) * 2,
            hidden_dim=hidden_dim,
            horizon=horizon,
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            static_fields=static_context_fields,
            static_dropout=static_context_dropout,
            dropout=dropout,
            output_profile=output_profile,
        )
    if family == "patch_transformer_static_context":
        requested_heads = max(int(transformer_heads), 1)
        num_heads = requested_heads if int(hidden_dim) % requested_heads == 0 else 1
        temporal = PatchTransformerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        return StaticContextPath20Forecaster(
            temporal_encoder=temporal,
            temporal_dim=int(hidden_dim),
            hidden_dim=hidden_dim,
            horizon=horizon,
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            static_fields=static_context_fields,
            static_dropout=static_context_dropout,
            dropout=dropout,
            output_profile=output_profile,
        )
    if family == "stock_mixer_sequence":
        return StockMixerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            output_profile=output_profile,
        )
    if family == "sector_slot_mixer_sequence":
        return SectorSlotMixerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            slot_count=slot_count,
            output_profile=output_profile,
        )
    raise ValueError(f"Unsupported forecast model family: {family}")


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _now_iso_seconds() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _forecast_resume_contract(
    *,
    model_family: str,
    seed: int,
    feature_columns: list[str],
    feature_profile: str,
    lookback_days: int,
    target_scale: float,
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    selection_profile: str,
    loss_profile: str = "default",
    output_profile: str = "forecast_path_v1",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    static_context_schema: dict[str, Any] | None = None,
    symbol_vocab_fingerprint: str = "",
    industry_vocab_fingerprint: str = "",
    board_vocab_fingerprint: str = "",
) -> dict[str, Any]:
    return {
        "model_family": str(model_family),
        "seed": int(seed),
        "feature_columns": list(feature_columns),
        "feature_profile": str(feature_profile),
        "lookback_days": int(lookback_days),
        "horizon": int(PATH20_HORIZON),
        "target_scale": float(target_scale),
        "model_config": _json_ready(model_config),
        "optimizer_config": _json_ready(optimizer_config),
        "selection_profile": str(selection_profile),
        "loss_profile": str(loss_profile or "default"),
        "output_profile": str(output_profile or "forecast_path_v1"),
        "decision_cost_bps": float(decision_cost_bps),
        "decision_hit_threshold_bps": float(decision_hit_threshold_bps),
        "decision_drawdown_penalty": float(decision_drawdown_penalty),
        "static_context_schema": _json_ready(static_context_schema or {"enabled": False}),
        "symbol_vocab_fingerprint": str(symbol_vocab_fingerprint or ""),
        "industry_vocab_fingerprint": str(industry_vocab_fingerprint or ""),
        "board_vocab_fingerprint": str(board_vocab_fingerprint or ""),
    }


def _forecast_model_state_dict(model: nn.Module) -> dict[str, Any]:
    return {
        str(key): value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in model.state_dict().items()
    }


def _forecast_checkpoint_payload(
    *,
    checkpoint_kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    model_family: str,
    seed: int,
    epoch: int,
    best_epoch: int,
    best_score: float,
    best_validation_loss: float,
    best_validation_metrics: dict[str, Any],
    best_train_loss: float,
    last_train_loss: float,
    patience_used: int,
    feature_columns: list[str],
    feature_profile: str,
    feature_manifest: dict[str, Any],
    normalization_manifest: dict[str, Any],
    target_scale: float,
    lookback_days: int,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    selection_profile: str,
    learning_rows: list[dict[str, Any]],
    loss_profile: str = "default",
    output_profile: str = "forecast_path_v1",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    static_context_schema: dict[str, Any] | None = None,
    symbol_vocab_fingerprint: str = "",
    industry_vocab_fingerprint: str = "",
    board_vocab_fingerprint: str = "",
    best_checkpoint_pt: Path | None = None,
    best_checkpoint_payload: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_kind": str(checkpoint_kind),
        "checkpoint_created_at": _now_iso_seconds(),
        "model_family": str(model_family),
        "seed": int(seed),
        "epoch": int(epoch),
        "state_dict": _forecast_model_state_dict(model),
        "feature_columns": list(feature_columns),
        "feature_profile": str(feature_profile),
        "feature_manifest": _json_ready(feature_manifest),
        "normalization": _json_ready(normalization_manifest),
        "target_scale": float(target_scale),
        "lookback_days": int(lookback_days),
        "horizon": int(PATH20_HORIZON),
        "model_config": _json_ready(model_config),
        "training_config": _json_ready(training_config),
        "optimizer_config": _json_ready(optimizer_config),
        "resume_contract": _forecast_resume_contract(
            model_family=model_family,
            seed=int(seed),
            feature_columns=list(feature_columns),
            feature_profile=feature_profile,
            lookback_days=int(lookback_days),
            target_scale=float(target_scale),
            model_config=model_config,
            optimizer_config=optimizer_config,
            selection_profile=selection_profile,
            loss_profile=loss_profile,
            output_profile=output_profile,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
            static_context_schema=static_context_schema,
            symbol_vocab_fingerprint=symbol_vocab_fingerprint,
            industry_vocab_fingerprint=industry_vocab_fingerprint,
            board_vocab_fingerprint=board_vocab_fingerprint,
        ),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_metrics": _json_ready(best_validation_metrics),
        "best_train_loss": float(best_train_loss),
        "last_train_loss": float(last_train_loss),
        "patience_used": int(patience_used),
        "learning_rows": _json_ready(learning_rows),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if best_checkpoint_pt is not None:
        payload["best_checkpoint_pt"] = str(best_checkpoint_pt.resolve())
    if best_checkpoint_payload is not None:
        payload["best_checkpoint_payload"] = best_checkpoint_payload
    if rng_state is not None:
        payload["rng_state"] = rng_state
    return payload


def _save_forecast_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)
    return str(path.resolve())


def _load_forecast_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"forecast resume checkpoint does not exist: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("forecast resume checkpoint payload must be a dictionary.")
    if payload.get("checkpoint_kind") != "last":
        raise ValueError("forecast resume checkpoint must be a last checkpoint.")
    for key in ("state_dict", "optimizer_state_dict", "scaler_state_dict", "epoch", "resume_contract"):
        if key not in payload:
            raise ValueError(f"forecast resume checkpoint is missing {key}.")
    return payload


def _validate_forecast_resume_checkpoint(
    payload: dict[str, Any],
    *,
    model_family: str,
    seed: int,
    dataset_view: "_ForecastDatasetView",
    target_scale: float,
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    selection_profile: str,
    loss_profile: str = "default",
    output_profile: str = "forecast_path_v1",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> None:
    expected = _forecast_resume_contract(
        model_family=model_family,
        seed=int(seed),
        feature_columns=list(dataset_view.feature_columns),
        feature_profile=str(dataset_view.manifest.get("feature_profile", "")),
        lookback_days=int(dataset_view.lookback_days),
        target_scale=float(target_scale),
        model_config=model_config,
        optimizer_config=optimizer_config,
        selection_profile=selection_profile,
        loss_profile=loss_profile,
        output_profile=output_profile,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
        static_context_schema=dataset_view.static_context_schema,
        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
    )
    actual = dict(payload.get("resume_contract", {}) or {})
    if not actual:
        raise ValueError("forecast resume checkpoint is missing resume_contract.")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"forecast resume checkpoint {key} mismatch: "
                f"expected {expected_value!r}, got {actual.get(key)!r}"
            )


def _forecast_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "data_loader_generator_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_forecast_rng_state(payload: dict[str, Any], generator: torch.Generator) -> None:
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        return
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "data_loader_generator_state" in state:
        generator.set_state(state["data_loader_generator_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _write_learning_curve_incremental(path: Path, learning_rows: list[dict[str, Any]]) -> str:
    return _write_frame(path, pd.DataFrame(_json_ready(learning_rows)))


def _write_forecast_progress(
    path: Path,
    *,
    status: str,
    model_family: str,
    seed: int,
    current_epoch: int,
    max_epochs: int,
    min_epochs: int,
    patience_limit: int,
    patience_used: int,
    best_epoch: int,
    best_score: float,
    best_validation_loss: float,
    best_validation_metrics: dict[str, Any],
    last_train_loss: float,
    epoch_seconds: float,
    run_started_at: str,
    elapsed_seconds: float,
    learning_rows: list[dict[str, Any]],
    last_checkpoint_pt: Path | None,
    best_checkpoint_pt: Path | None,
) -> str:
    current_rows = [
        row
        for row in learning_rows
        if str(row.get("model_family")) == str(model_family) and int(row.get("seed", -1)) == int(seed)
    ]
    epoch_durations = [float(row.get("epoch_seconds", 0.0) or 0.0) for row in current_rows if row.get("epoch_seconds") is not None]
    avg_epoch_seconds = float(np.mean(epoch_durations)) if epoch_durations else float(epoch_seconds)
    remaining_epochs = max(int(max_epochs) - int(current_epoch), 0) if str(status) == "running" else 0
    estimated_remaining_seconds = float(avg_epoch_seconds * remaining_epochs)
    eta_at = (
        datetime.now().astimezone() + timedelta(seconds=estimated_remaining_seconds)
    ).isoformat(timespec="seconds")
    payload = {
        "status": str(status),
        "updated_at": _now_iso_seconds(),
        "run_started_at": str(run_started_at),
        "model_family": str(model_family),
        "seed": int(seed),
        "current_epoch": int(current_epoch),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "patience_limit": int(patience_limit),
        "patience_used": int(patience_used),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_metrics": _json_ready(best_validation_metrics),
        "last_train_loss": float(last_train_loss),
        "epoch_seconds": float(epoch_seconds),
        "avg_epoch_seconds": float(avg_epoch_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_remaining_seconds": float(estimated_remaining_seconds),
        "eta_at": eta_at,
        "last_checkpoint_pt": str(last_checkpoint_pt.resolve()) if last_checkpoint_pt is not None else "",
        "best_checkpoint_pt": str(best_checkpoint_pt.resolve()) if best_checkpoint_pt is not None else "",
    }
    write_json(path, _json_ready(payload))
    return str(path.resolve())


def _data_loader_kwargs(*, pin_memory: bool, dataloader_num_workers: int, prefetch_factor: int) -> dict[str, Any]:
    workers = max(int(dataloader_num_workers), 0)
    kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": bool(pin_memory)}
    if workers > 0:
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return kwargs


def _forecast_loss(
    prediction: dict[str, torch.Tensor],
    y_daily_scaled: torch.Tensor,
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    *,
    loss_profile: str = "default",
    target_scale: float = 100.0,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> torch.Tensor:
    profile = str(loss_profile or "default").strip().lower()
    if profile not in FORECAST_LOSS_PROFILES:
        raise ValueError(f"Unsupported forecast loss profile: {profile}")
    daily_weights = torch.ones((y_daily_scaled.shape[1],), device=y_daily_scaled.device, dtype=y_daily_scaled.dtype)
    daily_weights[:3] = 1.15
    daily_weights[3:5] = 1.05
    daily_loss = F.huber_loss(prediction["mu"], y_daily_scaled, reduction="none")
    loss = (daily_loss * daily_weights.reshape(1, -1)).mean()
    loss = loss + 0.20 * pinball_loss(prediction["q10"], y_daily_scaled, 0.10)
    loss = loss + 0.20 * pinball_loss(prediction["q50"], y_daily_scaled, 0.50)
    loss = loss + 0.20 * pinball_loss(prediction["q90"], y_daily_scaled, 0.90)
    cum_count = len(PATH20_CUMULATIVE_HORIZONS)
    loss = loss + 0.25 * F.huber_loss(prediction["aux"][:, :cum_count], y_cum_scaled)
    loss = loss + 0.05 * F.huber_loss(prediction["aux"][:, cum_count : cum_count + 3], y_risk_scaled)
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        weight = float(FORECAST_RANK_LOSS_WEIGHTS.get(int(horizon), 0.0))
        if profile == "rank_aux":
            weight *= 1.75
        elif profile == "multitask_v1":
            weight *= 2.25
        if weight <= 0.0:
            continue
        score = prediction["mu"][:, : int(horizon)].sum(dim=1)
        target = y_cum_scaled[:, pos]
        loss = loss + weight * pairwise_rank_loss(score, target)
    loss = loss + 0.005 * pairwise_rank_loss(prediction["aux"][:, cum_count + 2], y_risk_scaled[:, 2])
    if profile == "multitask_v1":
        direction_target = (y_cum_scaled[:, -1] > 0).to(dtype=prediction["aux"].dtype)
        direction_logit = prediction["mu"].sum(dim=1)
        loss = loss + 0.025 * F.binary_cross_entropy_with_logits(direction_logit, direction_target)
        downside_target = y_risk_scaled[:, 0]
        downside_score = prediction["aux"][:, cum_count]
        loss = loss + 0.010 * pairwise_rank_loss(-downside_score, -downside_target)
    if profile == "decision_utility_v1":
        if "decision_aux" not in prediction:
            raise ValueError("decision_utility_v1 loss requires decision_aux outputs.")
        decision_aux = prediction["decision_aux"]
        if int(decision_aux.shape[1]) != PATH20_DECISION_AUX_DIM:
            raise ValueError(f"decision_aux must have width {PATH20_DECISION_AUX_DIM}.")
        targets = _decision_utility_targets(
            y_cum_scaled,
            y_risk_scaled,
            target_scale=float(target_scale),
            cost_bps=float(decision_cost_bps),
            hit_threshold_bps=float(decision_hit_threshold_bps),
            drawdown_penalty=float(decision_drawdown_penalty),
        )
        utility_pred = decision_aux[:, :cum_count]
        hit_logits = decision_aux[:, cum_count : cum_count * 2]
        horizon_logits = decision_aux[:, cum_count * 2 : cum_count * 3]
        utility_target = targets["utility_scaled"].to(dtype=utility_pred.dtype)
        hit_target = targets["hit_label"].to(dtype=hit_logits.dtype)
        best_horizon_index = targets["best_horizon_index"].to(device=horizon_logits.device, dtype=torch.long)
        pred_decision_score = utility_pred.max(dim=1).values
        future_decision_score = targets["decision_score_scaled"].to(dtype=pred_decision_score.dtype)
        loss = loss + 0.20 * F.huber_loss(utility_pred, utility_target)
        loss = loss + 0.05 * F.binary_cross_entropy_with_logits(hit_logits, hit_target)
        loss = loss + 0.05 * F.cross_entropy(horizon_logits, best_horizon_index)
        loss = loss + 0.05 * pairwise_rank_loss(pred_decision_score, future_decision_score)
    return loss


def _autocast_context(device: torch.device, amp_enabled: bool) -> Any:
    if device.type == "cuda" and bool(amp_enabled):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    value = str(device or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("forecast training requested CUDA, but torch.cuda.is_available() is False.")
    if value not in {"cpu", "cuda"}:
        raise ValueError("--forecast-device must be auto, cpu, or cuda.")
    return torch.device(value)


def _unpack_forecast_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 6:
        batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx, static_context_ids = batch
        return batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx, static_context_ids
    if len(batch) == 5:
        batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx = batch
        return batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx, None
    raise ValueError(f"forecast batch must contain 5 or 6 tensors, got {len(batch)}.")


def _decision_utility_targets(
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    *,
    target_scale: float,
    cost_bps: float,
    hit_threshold_bps: float,
    drawdown_penalty: float,
) -> dict[str, torch.Tensor]:
    scale = max(float(target_scale), 1.0e-8)
    y_cum = y_cum_scaled / scale
    max_drawdown = y_risk_scaled[:, 0] / scale
    horizons = torch.as_tensor(PATH20_CUMULATIVE_HORIZONS, dtype=y_cum.dtype, device=y_cum.device).reshape(1, -1)
    horizon_scale = torch.sqrt(horizons / float(PATH20_HORIZON))
    downside = torch.clamp(-max_drawdown, min=0.0).reshape(-1, 1)
    utility = y_cum - float(cost_bps) / 10000.0 - float(drawdown_penalty) * downside * horizon_scale
    hit_label = utility > (float(hit_threshold_bps) / 10000.0)
    best_horizon_index = torch.argmax(utility, dim=1)
    decision_score = utility.max(dim=1).values
    return {
        "utility": utility,
        "utility_scaled": utility * scale,
        "hit_label": hit_label,
        "best_horizon_index": best_horizon_index,
        "decision_score": decision_score,
        "decision_score_scaled": decision_score * scale,
    }


def _decision_utility_targets_np(
    y_cum: np.ndarray,
    max_drawdown: np.ndarray,
    *,
    cost_bps: float,
    hit_threshold_bps: float,
    drawdown_penalty: float,
) -> dict[str, np.ndarray]:
    y_cum_arr = np.asarray(y_cum, dtype=np.float64)
    max_dd = np.asarray(max_drawdown, dtype=np.float64).reshape(-1, 1)
    horizons = np.asarray(PATH20_CUMULATIVE_HORIZONS, dtype=np.float64).reshape(1, -1)
    utility = (
        y_cum_arr
        - float(cost_bps) / 10000.0
        - float(drawdown_penalty) * np.maximum(0.0, -max_dd) * np.sqrt(horizons / float(PATH20_HORIZON))
    )
    return {
        "utility": utility,
        "hit_label": utility > (float(hit_threshold_bps) / 10000.0),
        "best_horizon_index": np.argmax(utility, axis=1),
        "decision_score": np.max(utility, axis=1),
    }


def _collate_forecast_date_batches(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    max_stocks = max(int(item[0].shape[0]) for item in batch) if batch else 0
    if max_stocks <= 0:
        raise ValueError("date-level forecast batch is empty.")
    x_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    y_daily_rows: list[torch.Tensor] = []
    y_cum_rows: list[torch.Tensor] = []
    y_risk_rows: list[torch.Tensor] = []
    row_index_rows: list[torch.Tensor] = []
    static_rows: list[torch.Tensor] = []
    has_static = len(batch[0]) == 7
    for item in batch:
        x, mask, y_daily, y_cum, y_risk, row_indices, *maybe_static = item
        pad = max_stocks - int(x.shape[0])
        x_rows.append(F.pad(x, (0, 0, 0, 0, 0, pad)))
        mask_rows.append(F.pad(mask.to(dtype=torch.bool), (0, pad), value=False))
        y_daily_rows.append(F.pad(y_daily, (0, 0, 0, pad)))
        y_cum_rows.append(F.pad(y_cum, (0, 0, 0, pad)))
        y_risk_rows.append(F.pad(y_risk, (0, 0, 0, pad)))
        row_index_rows.append(F.pad(row_indices.to(dtype=torch.long), (0, pad), value=-1))
        if has_static:
            static_rows.append(F.pad(maybe_static[0].to(dtype=torch.long), (0, 0, 0, pad), value=0))
    static_context = torch.stack(static_rows, dim=0) if has_static else None
    return (
        torch.stack(x_rows, dim=0),
        torch.stack(mask_rows, dim=0),
        torch.stack(y_daily_rows, dim=0),
        torch.stack(y_cum_rows, dim=0),
        torch.stack(y_risk_rows, dim=0),
        torch.stack(row_index_rows, dim=0),
        static_context,
    )


def _forecast_model_accepts_static_context(model: nn.Module) -> bool:
    try:
        return "static_context_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _forecast_model_forward(
    model: nn.Module,
    batch_x: torch.Tensor,
    static_context_ids: torch.Tensor | None = None,
    stock_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if static_context_ids is not None and _forecast_model_accepts_static_context(model):
        return model(batch_x, static_context_ids=static_context_ids)
    if stock_mask is not None and _forecast_model_accepts_stock_mask(model):
        return model(batch_x, stock_mask=stock_mask)
    return model(batch_x)


def _forecast_model_accepts_stock_mask(model: nn.Module) -> bool:
    try:
        return "stock_mask" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _predict_all(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | None = None,
    amp_enabled: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    resolved_device = device or next(model.parameters()).device
    chunks: dict[str, list[np.ndarray]] = {"mu": [], "q10": [], "q50": [], "q90": [], "aux": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], max(int(batch_size), 1)):
            batch = x[start : start + max(int(batch_size), 1)].to(resolved_device, non_blocking=resolved_device.type == "cuda")
            with _autocast_context(resolved_device, amp_enabled):
                pred = _forecast_model_forward(model, batch)
            if "decision_aux" in pred and "decision_aux" not in chunks:
                chunks["decision_aux"] = []
            for key in chunks:
                if key in pred:
                    chunks[key].append(pred[key].detach().cpu().numpy())
    empty_shapes = {
        "mu": (0, PATH20_HORIZON),
        "q10": (0, PATH20_HORIZON),
        "q50": (0, PATH20_HORIZON),
        "q90": (0, PATH20_HORIZON),
        "aux": (0, PATH20_FORECAST_AUX_DIM),
        "decision_aux": (0, PATH20_DECISION_AUX_DIM),
    }
    return {key: np.concatenate(values, axis=0) if values else np.empty(empty_shapes[key]) for key, values in chunks.items()}


def _predict_indices(
    model: nn.Module,
    dataset_view_or_x: _ForecastDatasetView | torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    target_scale: float = 100.0,
) -> dict[str, np.ndarray]:
    if len(indices) == 0:
        return {
            "mu": np.empty((0, PATH20_HORIZON)),
            "q10": np.empty((0, PATH20_HORIZON)),
            "q50": np.empty((0, PATH20_HORIZON)),
            "q90": np.empty((0, PATH20_HORIZON)),
            "aux": np.empty((0, PATH20_FORECAST_AUX_DIM)),
        }
    if isinstance(dataset_view_or_x, _ForecastDatasetView):
        model.eval()
        if _forecast_model_accepts_stock_mask(model) and dataset_view_or_x.dataset_mode == "memmap":
            ordered_indices = np.asarray(indices, dtype=np.int64)
            index_pos = {int(row_idx): pos for pos, row_idx in enumerate(ordered_indices.tolist())}
            outputs = {
                "mu": np.empty((len(ordered_indices), PATH20_HORIZON), dtype=np.float32),
                "q10": np.empty((len(ordered_indices), PATH20_HORIZON), dtype=np.float32),
                "q50": np.empty((len(ordered_indices), PATH20_HORIZON), dtype=np.float32),
                "q90": np.empty((len(ordered_indices), PATH20_HORIZON), dtype=np.float32),
                "aux": np.empty((len(ordered_indices), PATH20_FORECAST_AUX_DIM), dtype=np.float32),
            }
            if str(getattr(model, "output_profile", "") or "") == "decision_utility_v1":
                outputs["decision_aux"] = np.empty((len(ordered_indices), PATH20_DECISION_AUX_DIM), dtype=np.float32)
            loader = DataLoader(
                dataset_view_or_x.date_batch_torch_dataset(ordered_indices, target_scale=target_scale),
                batch_size=1,
                shuffle=False,
                pin_memory=device.type == "cuda",
                collate_fn=_collate_forecast_date_batches,
            )
            with torch.no_grad():
                for batch_x_cpu, stock_mask_cpu, _, _, _, row_indices_cpu, _ in loader:
                    batch_x = batch_x_cpu.to(device, non_blocking=device.type == "cuda")
                    stock_mask = stock_mask_cpu.to(device, non_blocking=device.type == "cuda")
                    with _autocast_context(device, amp_enabled):
                        pred = _forecast_model_forward(model, batch_x, stock_mask=stock_mask)
                    if "decision_aux" in pred and "decision_aux" not in outputs:
                        outputs["decision_aux"] = np.empty((len(ordered_indices), PATH20_DECISION_AUX_DIM), dtype=np.float32)
                    valid = stock_mask.reshape(-1).detach().cpu().numpy().astype(bool)
                    row_ids = row_indices_cpu.reshape(-1).detach().cpu().numpy().astype(int)[valid]
                    for key in outputs:
                        values = pred[key][torch.as_tensor(valid, dtype=torch.bool, device=pred[key].device)].detach().cpu().numpy()
                        for row_id, value in zip(row_ids.tolist(), values, strict=False):
                            outputs[key][index_pos[int(row_id)]] = value
            return outputs
        chunks: dict[str, list[np.ndarray]] = {"mu": [], "q10": [], "q50": [], "q90": [], "aux": []}
        loader = DataLoader(
            dataset_view_or_x.torch_dataset(indices, target_scale=target_scale),
            batch_size=max(int(batch_size), 1),
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        with torch.no_grad():
            for raw_batch in loader:
                batch_x, _, _, _, _, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                batch = batch_x.to(device, non_blocking=device.type == "cuda")
                static_context_ids = (
                    static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                    if static_context_ids_cpu is not None
                    else None
                )
                with _autocast_context(device, amp_enabled):
                    pred = _forecast_model_forward(model, batch, static_context_ids)
                if "decision_aux" in pred and "decision_aux" not in chunks:
                    chunks["decision_aux"] = []
                for key in chunks:
                    if key in pred:
                        chunks[key].append(pred[key].detach().cpu().numpy())
        empty_shapes = {
            "mu": (0, PATH20_HORIZON),
            "q10": (0, PATH20_HORIZON),
            "q50": (0, PATH20_HORIZON),
            "q90": (0, PATH20_HORIZON),
            "aux": (0, PATH20_FORECAST_AUX_DIM),
            "decision_aux": (0, PATH20_DECISION_AUX_DIM),
        }
        return {key: np.concatenate(values, axis=0) if values else np.empty(empty_shapes[key]) for key, values in chunks.items()}
    x = dataset_view_or_x
    return _predict_all(
        model,
        x[torch.tensor(indices, dtype=torch.long)],
        batch_size=batch_size,
        device=device,
        amp_enabled=amp_enabled,
    )


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        score = pd.to_numeric(group[score_column], errors="coerce").astype("float64")
        target = pd.to_numeric(group[target_column], errors="coerce").astype("float64")
        valid = score.notna() & target.notna()
        if int(valid.sum()) < 2:
            continue
        corr = score.loc[valid].corr(target.loc[valid], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str, frac: float = 0.20) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        work = group[[score_column, target_column]].copy()
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce").astype("float64")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce").astype("float64")
        work = work.dropna()
        if len(work) < 2:
            continue
        k = max(int(len(work) * float(frac)), 1)
        top = work.nlargest(k, score_column)[target_column].mean()
        bottom = work.nsmallest(k, score_column)[target_column].mean()
        if pd.notna(top) and pd.notna(bottom):
            values.append(float(top - bottom))
    return float(np.mean(values)) if values else 0.0


def _add_decision_utility_columns(
    columns: dict[str, Any],
    *,
    predictions: dict[str, np.ndarray],
    y_cum: np.ndarray,
    max_drawdown_20d: np.ndarray,
    target_scale: float,
    decision_cost_bps: float,
    decision_hit_threshold_bps: float,
    decision_drawdown_penalty: float,
) -> None:
    if "decision_aux" not in predictions:
        return
    decision_aux = np.asarray(predictions["decision_aux"], dtype=np.float64)
    if decision_aux.size == 0:
        return
    cum_count = len(PATH20_CUMULATIVE_HORIZONS)
    pred_utility = decision_aux[:, :cum_count] / max(float(target_scale), 1.0e-8)
    hit_logits = decision_aux[:, cum_count : cum_count * 2]
    horizon_logits = decision_aux[:, cum_count * 2 : cum_count * 3]
    future = _decision_utility_targets_np(
        y_cum,
        max_drawdown_20d,
        cost_bps=float(decision_cost_bps),
        hit_threshold_bps=float(decision_hit_threshold_bps),
        drawdown_penalty=float(decision_drawdown_penalty),
    )
    pred_hit_prob = 1.0 / (1.0 + np.exp(-np.clip(hit_logits, -60.0, 60.0)))
    pred_best_idx = np.argmax(horizon_logits, axis=1)
    pred_score = np.max(pred_utility, axis=1)
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        columns[f"pred_decision_utility_{horizon}d"] = pred_utility[:, pos]
        columns[f"future_decision_utility_{horizon}d"] = future["utility"][:, pos]
        columns[f"pred_hit_prob_{horizon}d"] = pred_hit_prob[:, pos]
        columns[f"future_hit_label_{horizon}d"] = future["hit_label"][:, pos].astype(int)
    horizons = np.asarray(PATH20_CUMULATIVE_HORIZONS, dtype=int)
    columns["pred_best_horizon"] = horizons[pred_best_idx]
    columns["future_best_horizon"] = horizons[future["best_horizon_index"].astype(int)]
    columns["pred_decision_score"] = pred_score
    columns["future_decision_score"] = future["decision_score"]


def _prediction_frame(
    dataset: ForecastSequenceDataset,
    *,
    role: str,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> pd.DataFrame:
    mask = dataset.role == role
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return pd.DataFrame()
    scale = float(target_scale)
    mu = predictions["mu"][idx] / scale
    q10 = predictions["q10"][idx] / scale
    q50 = predictions["q50"][idx] / scale
    q90 = predictions["q90"][idx] / scale
    aux = predictions["aux"][idx] / scale
    y_daily = dataset.y_daily_excess[idx]
    y_cum = dataset.y_cum_excess[idx]
    columns: dict[str, Any] = {
        "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
        "stock": dataset.stock[idx].astype(str),
        "role": dataset.role[idx].astype(str),
        "model_family": str(family),
        "future_rank_20d": dataset.y_rank_20d[idx],
        "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
        "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
        "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
    }
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        columns[f"future_rank_{horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        columns[f"future_cum_excess_return_{horizon}d"] = y_cum[:, pos]
        columns[f"pred_cum_mu_{horizon}d"] = mu[:, :horizon].sum(axis=1)
        columns[f"pred_aux_cum_{horizon}d"] = aux[:, pos]
    _add_decision_utility_columns(
        columns,
        predictions=predictions,
        y_cum=y_cum,
        max_drawdown_20d=dataset.y_max_drawdown_20d[idx],
        target_scale=target_scale,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
    )
    risk_start = len(PATH20_CUMULATIVE_HORIZONS)
    columns["pred_aux_downside_floor_20d"] = aux[:, risk_start]
    columns["pred_aux_worst_1d_20d"] = aux[:, risk_start + 1]
    columns["pred_aux_upside_20d"] = aux[:, risk_start + 2]
    for step in range(1, PATH20_HORIZON + 1):
        offset = step - 1
        columns[f"future_excess_return_{step}d"] = y_daily[:, offset]
        columns[f"target_excess_{step}d"] = y_daily[:, offset]
        columns[f"pred_mu_{step}d"] = mu[:, offset]
        columns[f"pred_q10_{step}d"] = q10[:, offset]
        columns[f"pred_q50_{step}d"] = q50[:, offset]
        columns[f"pred_q90_{step}d"] = q90[:, offset]
    return pd.DataFrame(columns)


def _prediction_frame_for_indices(
    dataset: ForecastSequenceDataset,
    *,
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> pd.DataFrame:
    if len(indices) == 0:
        return pd.DataFrame()
    scale = float(target_scale)
    idx = np.asarray(indices, dtype=int)
    mu = predictions["mu"] / scale
    q10 = predictions["q10"] / scale
    q50 = predictions["q50"] / scale
    q90 = predictions["q90"] / scale
    aux = predictions["aux"] / scale
    y_daily = dataset.y_daily_excess[idx]
    y_cum = dataset.y_cum_excess[idx]
    columns: dict[str, Any] = {
        "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
        "stock": dataset.stock[idx].astype(str),
        "role": dataset.role[idx].astype(str),
        "model_family": str(family),
        "future_rank_20d": dataset.y_rank_20d[idx],
        "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
        "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
        "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
    }
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        columns[f"future_rank_{horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        columns[f"future_cum_excess_return_{horizon}d"] = y_cum[:, pos]
        columns[f"pred_cum_mu_{horizon}d"] = mu[:, :horizon].sum(axis=1)
        columns[f"pred_aux_cum_{horizon}d"] = aux[:, pos]
    _add_decision_utility_columns(
        columns,
        predictions=predictions,
        y_cum=y_cum,
        max_drawdown_20d=dataset.y_max_drawdown_20d[idx],
        target_scale=target_scale,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
    )
    risk_start = len(PATH20_CUMULATIVE_HORIZONS)
    columns["pred_aux_downside_floor_20d"] = aux[:, risk_start]
    columns["pred_aux_worst_1d_20d"] = aux[:, risk_start + 1]
    columns["pred_aux_upside_20d"] = aux[:, risk_start + 2]
    for step in range(1, PATH20_HORIZON + 1):
        offset = step - 1
        columns[f"future_excess_return_{step}d"] = y_daily[:, offset]
        columns[f"target_excess_{step}d"] = y_daily[:, offset]
        columns[f"pred_mu_{step}d"] = mu[:, offset]
        columns[f"pred_q10_{step}d"] = q10[:, offset]
        columns[f"pred_q50_{step}d"] = q50[:, offset]
        columns[f"pred_q90_{step}d"] = q90[:, offset]
    return pd.DataFrame(columns)


def _prediction_frame_for_dataset_indices(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    *,
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> pd.DataFrame:
    if isinstance(dataset, ForecastMemmapDataset):
        if len(indices) == 0:
            return pd.DataFrame()
        scale = float(target_scale)
        idx = np.asarray(indices, dtype=int)
        rows = dataset.sample_index.iloc[idx].reset_index(drop=True)
        mu = predictions["mu"] / scale
        q10 = predictions["q10"] / scale
        q50 = predictions["q50"] / scale
        q90 = predictions["q90"] / scale
        aux = predictions["aux"] / scale
        y_daily = dataset.y_daily_excess[idx]
        y_cum = dataset.y_cum_excess[idx]
        columns: dict[str, Any] = {
            "date": rows["date"].astype(str).tolist(),
            "stock": rows["stock"].astype(str).to_numpy(),
            "role": rows["role"].astype(str).to_numpy(),
            "model_family": str(family),
            "stock_seen_in_train": rows.get("stock_seen_in_train", pd.Series(False, index=rows.index)).astype(bool).to_numpy(),
            "history_bucket": rows.get("history_bucket", pd.Series("", index=rows.index)).astype(str).to_numpy(),
            "history_valid_ratio": pd.to_numeric(rows.get("history_valid_ratio", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(),
            "future_rank_20d": dataset.y_rank_20d[idx],
            "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
            "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
            "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
        }
        for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
            columns[f"future_rank_{horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
        for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
            columns[f"future_cum_excess_return_{horizon}d"] = y_cum[:, pos]
            columns[f"pred_cum_mu_{horizon}d"] = mu[:, :horizon].sum(axis=1)
            columns[f"pred_aux_cum_{horizon}d"] = aux[:, pos]
        _add_decision_utility_columns(
            columns,
            predictions=predictions,
            y_cum=y_cum,
            max_drawdown_20d=dataset.y_max_drawdown_20d[idx],
            target_scale=target_scale,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
        )
        risk_start = len(PATH20_CUMULATIVE_HORIZONS)
        columns["pred_aux_downside_floor_20d"] = aux[:, risk_start]
        columns["pred_aux_worst_1d_20d"] = aux[:, risk_start + 1]
        columns["pred_aux_upside_20d"] = aux[:, risk_start + 2]
        for step in range(1, PATH20_HORIZON + 1):
            offset = step - 1
            columns[f"future_excess_return_{step}d"] = y_daily[:, offset]
            columns[f"target_excess_{step}d"] = y_daily[:, offset]
            columns[f"pred_mu_{step}d"] = mu[:, offset]
            columns[f"pred_q10_{step}d"] = q10[:, offset]
            columns[f"pred_q50_{step}d"] = q50[:, offset]
            columns[f"pred_q90_{step}d"] = q90[:, offset]
        return pd.DataFrame(columns)
    return _prediction_frame_for_indices(
        dataset,
        indices=indices,
        predictions=predictions,
        family=family,
        target_scale=target_scale,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
    )


def forecast_prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"status": "insufficient_or_incomplete", "reason": "empty_predictions"}
    q10_coverages: list[float] = []
    q90_coverages: list[float] = []
    for step in range(1, PATH20_HORIZON + 1):
        target = pd.to_numeric(frame[f"target_excess_{step}d"], errors="coerce")
        q10 = pd.to_numeric(frame[f"pred_q10_{step}d"], errors="coerce")
        q90 = pd.to_numeric(frame[f"pred_q90_{step}d"], errors="coerce")
        valid_q10 = target.notna() & q10.notna()
        valid_q90 = target.notna() & q90.notna()
        if bool(valid_q10.any()):
            q10_coverages.append(float((target.loc[valid_q10] >= q10.loc[valid_q10]).mean()))
        if bool(valid_q90.any()):
            q90_coverages.append(float((target.loc[valid_q90] <= q90.loc[valid_q90]).mean()))
    metrics: dict[str, Any] = {
        "status": "completed",
        "row_count": int(len(frame)),
        "date_count": int(frame["date"].nunique()),
        "q10_coverage_mean": float(np.mean(q10_coverages)) if q10_coverages else 0.0,
        "q90_coverage_mean": float(np.mean(q90_coverages)) if q90_coverages else 0.0,
    }
    for horizon in PATH20_CUMULATIVE_HORIZONS:
        pred = pd.to_numeric(frame[f"pred_cum_mu_{horizon}d"], errors="coerce")
        target = pd.to_numeric(frame[f"future_cum_excess_return_{horizon}d"], errors="coerce")
        valid = pred.notna() & target.notna()
        metrics[f"rank_ic_{horizon}d"] = _rank_ic_by_date(
            frame,
            f"pred_cum_mu_{horizon}d",
            f"future_cum_excess_return_{horizon}d",
        )
        metrics[f"top_bottom_spread_{horizon}d"] = _top_bottom_spread_by_date(
            frame,
            f"pred_cum_mu_{horizon}d",
            f"future_cum_excess_return_{horizon}d",
        )
        metrics[f"direction_accuracy_{horizon}d"] = (
            float((np.sign(pred.loc[valid]) == np.sign(target.loc[valid])).mean()) if bool(valid.any()) else 0.0
        )
    if {"pred_aux_upside_20d", "future_path_upside_capture_20d"}.issubset(frame.columns):
        metrics["rank_ic_upside_20d"] = _rank_ic_by_date(
            frame,
            "pred_aux_upside_20d",
            "future_path_upside_capture_20d",
        )
        metrics["top_bottom_spread_upside_20d"] = _top_bottom_spread_by_date(
            frame,
            "pred_aux_upside_20d",
            "future_path_upside_capture_20d",
        )
    else:
        metrics["rank_ic_upside_20d"] = 0.0
        metrics["top_bottom_spread_upside_20d"] = 0.0
    if {"pred_decision_score", "future_decision_score", "pred_best_horizon", "future_best_horizon"}.issubset(frame.columns):
        metrics["decision_score_rank_ic"] = _rank_ic_by_date(frame, "pred_decision_score", "future_decision_score")
        metrics["decision_score_top_bottom_spread"] = _top_bottom_spread_by_date(
            frame,
            "pred_decision_score",
            "future_decision_score",
        )
        hit_cols = [f"future_hit_label_{int(horizon)}d" for horizon in PATH20_CUMULATIVE_HORIZONS]
        if set(hit_cols).issubset(frame.columns):
            hit_any = frame[hit_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            lifts: list[float] = []
            for _, group in frame.assign(_future_decision_hit_any=hit_any).groupby("date", sort=True):
                work = group[["pred_decision_score", "_future_decision_hit_any"]].copy()
                work["pred_decision_score"] = pd.to_numeric(work["pred_decision_score"], errors="coerce")
                work["_future_decision_hit_any"] = pd.to_numeric(work["_future_decision_hit_any"], errors="coerce")
                work = work.dropna()
                if len(work) < 2:
                    continue
                k = max(int(len(work) * 0.20), 1)
                top_hit = float(work.nlargest(k, "pred_decision_score")["_future_decision_hit_any"].mean())
                all_hit = float(work["_future_decision_hit_any"].mean())
                lifts.append(top_hit - all_hit)
            metrics["decision_hit_lift_top20_mean"] = float(np.mean(lifts)) if lifts else 0.0
        else:
            metrics["decision_hit_lift_top20_mean"] = 0.0
        pred_horizon = pd.to_numeric(frame["pred_best_horizon"], errors="coerce")
        future_horizon = pd.to_numeric(frame["future_best_horizon"], errors="coerce")
        valid_horizon = pred_horizon.notna() & future_horizon.notna()
        metrics["decision_best_horizon_accuracy"] = (
            float((pred_horizon.loc[valid_horizon] == future_horizon.loc[valid_horizon]).mean())
            if bool(valid_horizon.any())
            else 0.0
        )
        decision_passed = (
            float(metrics.get("decision_score_rank_ic", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0) > 0.0
        )
        metrics["decision_utility_profile_status"] = "passed" if decision_passed else "failed"
    else:
        metrics["decision_score_rank_ic"] = 0.0
        metrics["decision_score_top_bottom_spread"] = 0.0
        metrics["decision_hit_lift_top20_mean"] = 0.0
        metrics["decision_best_horizon_accuracy"] = 0.0
        metrics["decision_utility_profile_status"] = "not_available"
    metrics["selected_signal_profile"] = forecast_signal_profile(metrics)
    return metrics


def stratified_forecast_prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    out: dict[str, Any] = {}
    if "stock_seen_in_train" in frame.columns:
        for value, group in frame.groupby("stock_seen_in_train", dropna=False):
            key = f"seen_in_train={str(bool(value)).lower()}" if pd.notna(value) else "seen_in_train=unknown"
            out[key] = forecast_prediction_metrics(group)
    if "history_bucket" in frame.columns:
        for value, group in frame.groupby("history_bucket", dropna=False):
            key = f"history_bucket={str(value)}"
            out[key] = forecast_prediction_metrics(group)
    return out


def _coverage_pass(metrics: dict[str, Any], coverage_range: tuple[float, float]) -> bool:
    q10 = float(metrics.get("q10_coverage_mean", 0.0) or 0.0)
    q90 = float(metrics.get("q90_coverage_mean", 0.0) or 0.0)
    low, high = coverage_range
    return bool(low <= q10 <= high and low <= q90 <= high)


def _horizon_gate(metrics: dict[str, Any], horizon: int) -> bool:
    return bool(
        float(metrics.get(f"rank_ic_{int(horizon)}d", 0.0) or 0.0) > 0.0
        and float(metrics.get(f"top_bottom_spread_{int(horizon)}d", 0.0) or 0.0) > 0.0
    )


def _short_burst_gate(metrics: dict[str, Any]) -> bool:
    gates = [
        _horizon_gate(metrics, 3),
        _horizon_gate(metrics, 5),
        bool(
            float(metrics.get("rank_ic_upside_20d", 0.0) or 0.0) > 0.0
            and float(metrics.get("top_bottom_spread_upside_20d", 0.0) or 0.0) > 0.0
        ),
    ]
    return sum(1 for item in gates if item) >= 2


def forecast_signal_profile(metrics: dict[str, Any]) -> str:
    if metrics.get("status") != "completed":
        return "failed"
    trend = _horizon_gate(metrics, 20)
    short_burst = _short_burst_gate(metrics)
    if trend and short_burst:
        return "multiscale"
    if trend:
        return "trend_20d"
    if short_burst:
        return "short_burst"
    return "failed"


def _profile_pass(metrics: dict[str, Any], selection_profile: str) -> bool:
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "decision_utility":
        return bool(
            float(metrics.get("decision_score_rank_ic", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0) > 0.0
        )
    if profile == "trend20":
        return _horizon_gate(metrics, 20)
    if profile == "short_burst":
        return _short_burst_gate(metrics)
    return _horizon_gate(metrics, 20) or _short_burst_gate(metrics)


def _profile_score(metrics: dict[str, Any], validation_loss: float, coverage_range: tuple[float, float], selection_profile: str) -> float:
    if metrics.get("status") != "completed":
        return -float(validation_loss)
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "decision_utility":
        gate_bonus = 1_000.0 if _profile_pass(metrics, profile) and _coverage_pass(metrics, coverage_range) else 0.0
        return float(
            gate_bonus
            + float(metrics.get("decision_score_rank_ic", 0.0) or 0.0) * 10.0
            + float(metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0)
            + float(metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0)
            - max(float(validation_loss), 0.0) * 1.0e-3
        )
    weights = FORECAST_PROFILE_HORIZON_WEIGHTS.get(profile, FORECAST_PROFILE_HORIZON_WEIGHTS["multiscale"])
    rank_score = sum(
        float(weight) * float(metrics.get(f"rank_ic_{int(horizon)}d", 0.0) or 0.0)
        for horizon, weight in weights.items()
    )
    spread_score = sum(
        float(weight) * float(metrics.get(f"top_bottom_spread_{int(horizon)}d", 0.0) or 0.0)
        for horizon, weight in weights.items()
    )
    upside_weight = 0.10 if profile in {"multiscale", "short_burst"} else 0.03
    rank_score += upside_weight * float(metrics.get("rank_ic_upside_20d", 0.0) or 0.0)
    spread_score += upside_weight * float(metrics.get("top_bottom_spread_upside_20d", 0.0) or 0.0)
    gate_bonus = 1_000.0 if _profile_pass(metrics, profile) and _coverage_pass(metrics, coverage_range) else 0.0
    return float(gate_bonus + rank_score * 10.0 + spread_score - max(float(validation_loss), 0.0) * 1.0e-3)


def forecast_evidence_verdict(
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    coverage_range: tuple[float, float] = (0.65, 0.95),
    selection_profile: str = "multiscale",
) -> str:
    if validation_metrics.get("status") != "completed":
        return "insufficient_or_incomplete"
    validation_promising = _profile_pass(validation_metrics, selection_profile) and _coverage_pass(validation_metrics, coverage_range)
    if not validation_promising:
        return "forecast_failed"
    if not test_metrics or test_metrics.get("status") != "completed":
        return "forecast_promising"
    if _profile_pass(test_metrics, selection_profile):
        return "forecast_test_confirmed"
    return "forecast_promising"


def _ranking_baseline_feature_frame(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return pd.DataFrame(), np.empty((0,), dtype=np.float32)
    if isinstance(dataset, ForecastMemmapDataset):
        store = dataset.open_feature_store()
        rows: list[np.ndarray] = []
        for row_idx in idx:
            window = dataset.input_window(int(row_idx), store=store)
            rows.append(
                np.concatenate(
                    [
                        window[-1],
                        np.nanmean(window, axis=0),
                        np.nanstd(window, axis=0),
                    ]
                ).astype(np.float32)
            )
        meta = dataset.sample_index.iloc[idx].reset_index(drop=True)
        x = np.vstack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)
        y = np.asarray(dataset.y_rank_20d[idx], dtype=np.float32)
    else:
        x = np.concatenate(
            [
                dataset.x[idx, -1, :],
                np.nanmean(dataset.x[idx], axis=1),
                np.nanstd(dataset.x[idx], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        y = np.asarray(dataset.y_rank_20d[idx], dtype=np.float32)
        meta = pd.DataFrame(
            {
                "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
                "stock": dataset.stock[idx].astype(str),
                "role": dataset.role[idx].astype(str),
            }
        )
    columns = [f"ranker_feature_{pos}" for pos in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    for column in ("date", "stock", "role"):
        frame[column] = meta[column].astype(str).to_numpy() if column in meta.columns else ""
    return frame, y


def ranking_relevance_labels(
    feature_frame: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    relevance_levels: int = 5,
) -> np.ndarray:
    levels = max(int(relevance_levels), 2)
    work = pd.DataFrame(
        {
            "date": feature_frame["date"].astype(str).to_numpy() if "date" in feature_frame.columns else "",
            "target": pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float),
        }
    )
    labels = np.zeros(len(work), dtype=np.int32)
    for _, group in work.dropna(subset=["target"]).groupby("date", sort=False):
        if len(group) <= 1:
            labels[group.index.to_numpy(dtype=int)] = levels - 1
            continue
        order = group["target"].rank(method="first").to_numpy(dtype=float) - 1.0
        group_labels = np.rint(order * float(levels - 1) / float(len(group) - 1)).astype(np.int32)
        labels[group.index.to_numpy(dtype=int)] = np.clip(group_labels, 0, levels - 1)
    return labels.astype(np.int32)


def _ranking_prediction_frame(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    family: str,
) -> pd.DataFrame:
    idx = np.asarray(indices, dtype=int)
    predictions = {
        "mu": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), PATH20_HORIZON, axis=1),
        "q10": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), PATH20_HORIZON, axis=1),
        "q50": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), PATH20_HORIZON, axis=1),
        "q90": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), PATH20_HORIZON, axis=1),
        "aux": np.zeros((len(idx), PATH20_FORECAST_AUX_DIM), dtype=np.float32),
    }
    return _prediction_frame_for_dataset_indices(
        dataset,
        indices=idx,
        predictions=predictions,
        family=family,
        target_scale=1.0,
    )


def run_forecast_ranking_baseline(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    *,
    study_root: Path,
    baseline: str = "none",
) -> dict[str, Any]:
    baseline = str(baseline or "none").strip().lower()
    if baseline == "none":
        return {"status": "skipped", "baseline": "none"}
    if baseline not in FORECAST_RANKING_BASELINES:
        return {
            "status": "dependency_missing",
            "baseline": baseline,
            "reason": "unsupported_ranking_baseline",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    study_root.mkdir(parents=True, exist_ok=True)
    view = _ForecastDatasetView(dataset)
    train_indices = view.role_indices("train")
    validation_indices = view.role_indices("validation")
    test_indices = view.role_indices("test")
    if len(train_indices) < 2 or len(validation_indices) < 1:
        summary = {
            "status": "insufficient_or_incomplete",
            "baseline": baseline,
            "reason": "insufficient_role_samples",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
        return summary
    try:
        if baseline == "lightgbm":
            from lightgbm import LGBMRanker  # type: ignore

            ranker: Any = LGBMRanker(n_estimators=80, learning_rate=0.05, random_state=7)
        else:
            from xgboost import XGBRanker  # type: ignore

            ranker = XGBRanker(n_estimators=80, learning_rate=0.05, random_state=7, objective="rank:pairwise")
    except Exception as exc:
        summary = {
            "status": "dependency_missing",
            "baseline": baseline,
            "reason": f"{baseline}_import_failed",
            "error": str(exc),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
        return summary
    train_x, train_y = _ranking_baseline_feature_frame(dataset, train_indices)
    validation_x, _ = _ranking_baseline_feature_frame(dataset, validation_indices)
    test_x, _ = _ranking_baseline_feature_frame(dataset, test_indices)
    feature_columns = [column for column in train_x.columns if column.startswith("ranker_feature_")]
    train_groups = train_x.groupby("date", sort=True).size().to_numpy(dtype=int)
    try:
        train_relevance = ranking_relevance_labels(train_x, train_y)
        ranker.fit(train_x[feature_columns].to_numpy(dtype=np.float32), train_relevance, group=train_groups)
        validation_scores = np.asarray(ranker.predict(validation_x[feature_columns].to_numpy(dtype=np.float32)), dtype=np.float32)
        test_scores = (
            np.asarray(ranker.predict(test_x[feature_columns].to_numpy(dtype=np.float32)), dtype=np.float32)
            if len(test_indices)
            else np.empty((0,), dtype=np.float32)
        )
    except Exception as exc:
        summary = {
            "status": "failed",
            "baseline": baseline,
            "reason": "ranker_fit_or_predict_failed",
            "error": str(exc),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
        return summary
    validation_frame = _ranking_prediction_frame(dataset, indices=validation_indices, scores=validation_scores, family=f"{baseline}_ranker")
    test_frame = _ranking_prediction_frame(dataset, indices=test_indices, scores=test_scores, family=f"{baseline}_ranker")
    validation_csv = _write_frame(study_root / f"forecast_ranking_baseline_{baseline}_validation.csv", validation_frame)
    test_csv = _write_frame(study_root / f"forecast_ranking_baseline_{baseline}_test.csv", test_frame)
    validation_metrics = forecast_prediction_metrics(validation_frame)
    test_metrics = forecast_prediction_metrics(test_frame) if not test_frame.empty else {"status": "insufficient_or_incomplete"}
    summary = {
        "status": "completed",
        "baseline": baseline,
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(len(test_indices)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_csv": validation_csv,
        "test_csv": test_csv,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
    return _json_ready(summary)


def _write_slot_diagnostics(
    path: Path,
    *,
    model: nn.Module,
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    indices: np.ndarray,
    seed: int,
) -> str:
    slot_tensor = getattr(model, "slots", None)
    slot_count = int(slot_tensor.shape[0]) if isinstance(slot_tensor, torch.Tensor) else 0
    rows = pd.DataFrame()
    if isinstance(dataset, ForecastMemmapDataset) and len(indices):
        rows = dataset.sample_index.iloc[np.asarray(indices, dtype=int)].copy()
    industry_available = "industry_id" in rows.columns and pd.to_numeric(rows["industry_id"], errors="coerce").fillna(0).ne(0).any()
    board_available = "board_id" in rows.columns and pd.to_numeric(rows["board_id"], errors="coerce").fillna(0).ne(0).any()
    payload = {
        "status": "completed",
        "slot_semantics": "dynamic_theme_factor_not_static_board",
        "seed": int(seed),
        "slot_count": int(slot_count),
        "row_count": int(len(rows)),
        "date_count": int(rows["date"].nunique()) if "date" in rows.columns else 0,
        "industry_available": bool(industry_available),
        "board_available": bool(board_available),
        "industry_id_top_counts": {
            str(key): int(value)
            for key, value in (
                rows["industry_id"].value_counts().head(10).items() if "industry_id" in rows.columns else []
            )
        },
        "board_id_top_counts": {
            str(key): int(value)
            for key, value in (
                rows["board_id"].value_counts().head(10).items() if "board_id" in rows.columns else []
            )
        },
    }
    write_json(path, _json_ready(payload))
    return str(path.resolve())


def _normalize_seeds(seeds: tuple[int, ...] | list[int] | str | None, *, default_seed: int) -> tuple[int, ...]:
    if seeds is None:
        return (int(default_seed),)
    if isinstance(seeds, str):
        parsed = tuple(int(item.strip()) for item in seeds.split(",") if item.strip())
        return parsed or (int(default_seed),)
    parsed = tuple(int(item) for item in seeds)
    return parsed or (int(default_seed),)


def _static_context_model_options(dataset_view: _ForecastDatasetView) -> dict[str, Any]:
    return {
        "static_context_vocab_sizes": dict(dataset_view.static_context_vocab_sizes),
        "static_context_embedding_dims": dict(
            dataset_view.static_context_schema.get("embedding_defaults", {}) or {}
        ),
        "static_context_fields": tuple(str(item) for item in dataset_view.static_context_schema.get("fields", []) or []),
        "static_context_dropout": float(
            dict(dataset_view.static_context_schema.get("embedding_defaults", {}) or {}).get("dropout", 0.20)
        ),
        "slot_count": 8,
    }


def _model_config_for_training(
    *,
    family: str,
    hidden_dim: int,
    dropout: float,
    gru_layers: int,
    transformer_layers: int,
    transformer_heads: int,
    patch_sizes: tuple[int, ...] | list[int],
    dataset_view: _ForecastDatasetView,
    static_model_options: dict[str, Any],
    output_profile: str = "forecast_path_v1",
) -> dict[str, Any]:
    return {
        "model_family": str(family),
        "output_profile": str(output_profile or "forecast_path_v1"),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "gru_layers": int(gru_layers),
        "transformer_layers": int(transformer_layers),
        "transformer_heads": int(transformer_heads),
        "patch_sizes": [int(item) for item in patch_sizes],
        "static_context_vocab_sizes": dict(static_model_options.get("static_context_vocab_sizes", {}) or {}),
        "static_context_embedding_dims": dict(static_model_options.get("static_context_embedding_dims", {}) or {}),
        "static_context_fields": list(static_model_options.get("static_context_fields", ()) or ()),
        "static_context_dropout": float(static_model_options.get("static_context_dropout", 0.20)),
        "slot_count": int(static_model_options.get("slot_count", 8)),
        "static_context_schema": dict(dataset_view.static_context_schema),
        "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
        "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
        "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
    }


def _validation_score(
    metrics: dict[str, Any],
    validation_loss: float,
    coverage_range: tuple[float, float],
    selection_profile: str,
) -> float:
    return _profile_score(metrics, validation_loss, coverage_range, selection_profile)


def _evaluate_loss(
    model: nn.Module,
    dataset_view_or_x: _ForecastDatasetView | torch.Tensor,
    y_daily: torch.Tensor | None,
    y_cum: torch.Tensor | None,
    y_risk: torch.Tensor | None,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    target_scale: float = 100.0,
    loss_profile: str = "default",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> float:
    if len(indices) == 0:
        return float("inf")
    model.eval()
    values: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        if isinstance(dataset_view_or_x, _ForecastDatasetView):
            if _forecast_model_accepts_stock_mask(model) and dataset_view_or_x.dataset_mode == "memmap":
                loader = DataLoader(
                    dataset_view_or_x.date_batch_torch_dataset(indices, target_scale=target_scale),
                    batch_size=1,
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                    collate_fn=_collate_forecast_date_batches,
                )
                for batch_x_cpu, stock_mask_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, _ in loader:
                    batch_x = batch_x_cpu.to(device, non_blocking=device.type == "cuda")
                    stock_mask = stock_mask_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_daily = batch_y_daily_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_cum = batch_y_cum_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_risk = batch_y_risk_cpu.to(device, non_blocking=device.type == "cuda")
                    flat_mask = stock_mask.reshape(-1)
                    with _autocast_context(device, amp_enabled):
                        pred = _forecast_model_forward(model, batch_x, stock_mask=stock_mask)
                        loss = _forecast_loss(
                            {key: value[flat_mask] for key, value in pred.items()},
                            batch_y_daily.reshape(-1, batch_y_daily.shape[-1])[flat_mask],
                            batch_y_cum.reshape(-1, batch_y_cum.shape[-1])[flat_mask],
                            batch_y_risk.reshape(-1, batch_y_risk.shape[-1])[flat_mask],
                            loss_profile=loss_profile,
                            target_scale=target_scale,
                            decision_cost_bps=decision_cost_bps,
                            decision_hit_threshold_bps=decision_hit_threshold_bps,
                            decision_drawdown_penalty=decision_drawdown_penalty,
                        )
                    values.append(float(loss.detach().cpu()))
                    counts.append(int(flat_mask.sum().detach().cpu()))
                total = sum(counts)
                return float(np.average(values, weights=counts)) if total > 0 else float("inf")
            loader = DataLoader(
                dataset_view_or_x.torch_dataset(indices, target_scale=target_scale),
                batch_size=max(int(batch_size), 1),
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            for raw_batch in loader:
                batch_x, batch_y_daily, batch_y_cum, batch_y_risk, _, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
                batch_y_daily = batch_y_daily.to(device, non_blocking=device.type == "cuda")
                batch_y_cum = batch_y_cum.to(device, non_blocking=device.type == "cuda")
                batch_y_risk = batch_y_risk.to(device, non_blocking=device.type == "cuda")
                static_context_ids = (
                    static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                    if static_context_ids_cpu is not None
                    else None
                )
                with _autocast_context(device, amp_enabled):
                    pred = _forecast_model_forward(model, batch_x, static_context_ids)
                    loss = _forecast_loss(
                        pred,
                        batch_y_daily,
                        batch_y_cum,
                        batch_y_risk,
                        loss_profile=loss_profile,
                        target_scale=target_scale,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                    )
                values.append(float(loss.detach().cpu()))
                counts.append(int(batch_x.shape[0]))
            total = sum(counts)
            return float(np.average(values, weights=counts)) if total > 0 else float("inf")
        x = dataset_view_or_x
        assert y_daily is not None and y_cum is not None and y_risk is not None
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = torch.tensor(indices[start : start + max(int(batch_size), 1)], dtype=torch.long)
            batch_x = x[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_daily = y_daily[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_cum = y_cum[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_risk = y_risk[batch_idx].to(device, non_blocking=device.type == "cuda")
            with _autocast_context(device, amp_enabled):
                pred = _forecast_model_forward(model, batch_x)
                loss = _forecast_loss(
                    pred,
                    batch_y_daily,
                    batch_y_cum,
                    batch_y_risk,
                    loss_profile=loss_profile,
                    target_scale=target_scale,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                )
            values.append(float(loss.detach().cpu()))
            counts.append(int(batch_idx.numel()))
    total = sum(counts)
    return float(np.average(values, weights=counts)) if total > 0 else float("inf")


def _family_summary(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation_metrics = [dict(item.get("validation_metrics", {})) for item in seed_summaries.values()]
    completed_metrics = [metrics for metrics in validation_metrics if metrics.get("status") == "completed"]
    q10_values = [
        float(metrics.get("q10_coverage_mean", 0.0) or 0.0)
        for metrics in completed_metrics
    ]
    q90_values = [
        float(metrics.get("q90_coverage_mean", 0.0) or 0.0)
        for metrics in completed_metrics
    ]
    multiscale_scores = [
        float(item.get("validation_multiscale_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    trend_scores = [
        float(item.get("validation_trend20_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    short_scores = [
        float(item.get("validation_short_burst_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    decision_scores = [
        float(item.get("validation_decision_utility_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    count = max(len(seed_summaries), 1)
    summary: dict[str, Any] = {
        "seed_count": int(len(seed_summaries)),
        "validation_q10_coverage_mean": float(np.mean(q10_values)) if q10_values else 0.0,
        "validation_q90_coverage_mean": float(np.mean(q90_values)) if q90_values else 0.0,
        "validation_multiscale_score_mean": float(np.mean(multiscale_scores)) if multiscale_scores else 0.0,
        "validation_multiscale_score_std": float(np.std(multiscale_scores, ddof=0)) if multiscale_scores else 0.0,
        "validation_trend20_score_mean": float(np.mean(trend_scores)) if trend_scores else 0.0,
        "validation_short_burst_score_mean": float(np.mean(short_scores)) if short_scores else 0.0,
        "validation_decision_utility_score_mean": float(np.mean(decision_scores)) if decision_scores else 0.0,
    }
    profile_counts: dict[str, int] = {"trend_20d": 0, "short_burst": 0, "multiscale": 0, "failed": 0}
    for metrics in completed_metrics:
        profile_counts[forecast_signal_profile(metrics)] = profile_counts.get(forecast_signal_profile(metrics), 0) + 1
    summary["validation_signal_profile_counts"] = profile_counts
    for horizon in PATH20_CUMULATIVE_HORIZONS:
        rank_values = [
            float(metrics.get(f"rank_ic_{int(horizon)}d", 0.0) or 0.0)
            for metrics in completed_metrics
        ]
        spread_values = [
            float(metrics.get(f"top_bottom_spread_{int(horizon)}d", 0.0) or 0.0)
            for metrics in completed_metrics
        ]
        summary[f"validation_rank_ic_{int(horizon)}d_mean"] = float(np.mean(rank_values)) if rank_values else 0.0
        summary[f"validation_rank_ic_{int(horizon)}d_std"] = float(np.std(rank_values, ddof=0)) if rank_values else 0.0
        summary[f"validation_top_bottom_spread_{int(horizon)}d_mean"] = (
            float(np.mean(spread_values)) if spread_values else 0.0
        )
        summary[f"validation_top_bottom_spread_{int(horizon)}d_std"] = (
            float(np.std(spread_values, ddof=0)) if spread_values else 0.0
        )
        summary[f"validation_rank_ic_{int(horizon)}d_positive_seed_rate"] = (
            float(sum(1 for value in rank_values if value > 0.0) / count)
        )
        summary[f"validation_top_bottom_spread_{int(horizon)}d_positive_seed_rate"] = (
            float(sum(1 for value in spread_values if value > 0.0) / count)
        )
    summary["validation_rank_ic_positive_seed_rate"] = summary.get("validation_rank_ic_20d_positive_seed_rate", 0.0)
    summary["validation_top_bottom_spread_positive_seed_rate"] = summary.get(
        "validation_top_bottom_spread_20d_positive_seed_rate",
        0.0,
    )
    return summary


def _selection_score_key(selection_profile: str) -> str:
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "decision_utility":
        return "validation_decision_utility_score"
    if profile == "trend20":
        return "validation_trend20_score"
    if profile == "short_burst":
        return "validation_short_burst_score"
    return "validation_multiscale_score"


def _select_family_seed(model_summaries: dict[str, dict[str, Any]], *, selection_profile: str) -> tuple[str, int]:
    score_key = _selection_score_key(selection_profile)

    def family_score(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float]:
        _, summary = item
        metrics = dict(summary.get("family_summary", {}))
        score = float(metrics.get(f"{score_key}_mean", 0.0) or 0.0)
        rank = float(metrics.get("validation_rank_ic_20d_mean", 0.0) or 0.0)
        spread = float(metrics.get("validation_top_bottom_spread_20d_mean", 0.0) or 0.0)
        q10 = float(metrics.get("validation_q10_coverage_mean", 0.0) or 0.0)
        q90 = float(metrics.get("validation_q90_coverage_mean", 0.0) or 0.0)
        passed = 1 if score > 1_000.0 and 0.65 <= q10 <= 0.95 and 0.65 <= q90 <= 0.95 else 0
        return (passed, score, rank + spread)

    if not model_summaries:
        return "", 0
    selected_family = max(model_summaries.items(), key=family_score)[0]
    seed_summaries = dict(model_summaries[selected_family].get("seed_summaries", {}))
    if not seed_summaries:
        return selected_family, 0

    def seed_score(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float]:
        _, summary = item
        validation = dict(summary.get("validation_metrics", {}))
        verdict = forecast_evidence_verdict(validation_metrics=validation, test_metrics=None, selection_profile=selection_profile)
        passed = 1 if verdict == "forecast_promising" else 0
        return (
            passed,
            float(summary.get(score_key, 0.0) or 0.0),
            float(validation.get("rank_ic_20d", 0.0) or 0.0),
        )

    selected_seed_text = max(seed_summaries.items(), key=seed_score)[0]
    return selected_family, int(selected_seed_text)


def train_forecast_models(
    dataset: ForecastSequenceDataset,
    *,
    study_root: Path,
    model_families: tuple[str, ...] | list[str] = FORECAST_MODEL_FAMILIES,
    epochs: int = 2,
    min_epochs: int = 1,
    early_stop_patience: int = 12,
    early_stop_min_delta: float = 1.0e-4,
    batch_size: int = 512,
    lr: float = 3.0e-4,
    hidden_dim: int = 192,
    dropout: float = 0.15,
    gru_layers: int = 2,
    transformer_layers: int = 4,
    transformer_heads: int = 6,
    patch_sizes: tuple[int, ...] | list[int] = (4, 20),
    device: str | torch.device = "auto",
    amp: bool = True,
    seeds: tuple[int, ...] | list[int] | str | None = None,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    weight_decay: float = 1.0e-4,
    write_all_predictions: bool = False,
    selection_profile: str = "multiscale",
    target_scale: float = 100.0,
    seed: int = 7,
    dataloader_num_workers: int = 0,
    prefetch_factor: int = 2,
    resume_from: str | Path | None = None,
    save_last_checkpoint: bool = True,
    checkpoint_every_n_epochs: int = 0,
    progress_json_name: str = "forecast_progress.json",
    output_profile: str = "forecast_path_v1",
    loss_profile: str = "default",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    ranking_baseline: str = "none",
    slot_diagnostics: bool = False,
) -> dict[str, Any]:
    study_root.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    amp_enabled = bool(amp) and resolved_device.type == "cuda"
    seed_values = _normalize_seeds(seeds, default_seed=int(seed))
    families = tuple(str(item).strip() for item in model_families if str(item).strip())
    invalid = sorted(set(families) - set(FORECAST_MODEL_FAMILIES))
    if invalid:
        raise ValueError(f"Unsupported forecast model families: {', '.join(invalid)}")
    resume_path = Path(resume_from) if resume_from is not None and str(resume_from).strip() else None
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        if len(families) != 1 or len(seed_values) != 1:
            raise ValueError("forecast resume requires exactly one model family and one seed.")
        resume_payload = _load_forecast_resume_checkpoint(resume_path)
    selection_profile = str(selection_profile or "multiscale").strip().lower()
    if selection_profile not in FORECAST_SELECTION_PROFILES:
        raise ValueError(f"Unsupported forecast selection profile: {selection_profile}")
    output_profile = str(output_profile or "forecast_path_v1").strip().lower()
    if output_profile not in FORECAST_OUTPUT_PROFILES:
        raise ValueError(f"Unsupported forecast output profile: {output_profile}")
    loss_profile = str(loss_profile or "default").strip().lower()
    if loss_profile not in FORECAST_LOSS_PROFILES:
        raise ValueError(f"Unsupported forecast loss profile: {loss_profile}")
    if loss_profile == "decision_utility_v1" and output_profile != "decision_utility_v1":
        raise ValueError("decision_utility_v1 loss requires output_profile=decision_utility_v1.")
    if selection_profile == "decision_utility" and output_profile != "decision_utility_v1":
        raise ValueError("decision_utility selection requires output_profile=decision_utility_v1.")
    decision_config = {
        "cost_bps": float(decision_cost_bps),
        "hit_threshold_bps": float(decision_hit_threshold_bps),
        "drawdown_penalty": float(decision_drawdown_penalty),
        "horizons": [int(item) for item in PATH20_CUMULATIVE_HORIZONS],
    }
    ranking_baseline = str(ranking_baseline or "none").strip().lower()
    if ranking_baseline not in FORECAST_RANKING_BASELINES:
        ranking_baseline_summary = {
            "status": "dependency_missing",
            "baseline": ranking_baseline,
            "reason": "unsupported_ranking_baseline",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    else:
        ranking_baseline_summary = {"status": "skipped", "baseline": "none"} if ranking_baseline == "none" else None
    dataset_view = _ForecastDatasetView(dataset)
    if ranking_baseline_summary is None:
        ranking_baseline_summary = run_forecast_ranking_baseline(dataset, study_root=study_root, baseline=ranking_baseline)
    cross_section_requested = any(family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES for family in families)
    if cross_section_requested and dataset_view.dataset_mode != "memmap":
        raise ValueError("stock_mixer_sequence and sector_slot_mixer_sequence require forecast memmap date-level batching.")
    feature_profile = str(dataset_view.manifest.get("feature_profile", ""))
    feature_manifest = dict(dataset_view.manifest.get("feature_manifest", {}))
    if dataset_view.row_count == 0:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "empty_forecast_dataset",
            "models": {},
            "family_summary": {},
            "dataset_mode": dataset_view.dataset_mode,
            "feature_profile": feature_profile,
            "feature_manifest": feature_manifest,
            "selected_seed": 0,
            "selected_model_family": "",
            "selected_signal_profile": "failed",
            "validation_multiscale_score": 0.0,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / "forecast_training_summary.json", _json_ready(summary))
        return summary

    x: torch.Tensor | None = None
    y_daily: torch.Tensor | None = None
    y_cum: torch.Tensor | None = None
    y_risk: torch.Tensor | None = None
    if dataset_view.dataset_mode != "memmap":
        eager_dataset = dataset
        assert isinstance(eager_dataset, ForecastSequenceDataset)
        x = torch.as_tensor(eager_dataset.x, dtype=torch.float32)
        y_daily = torch.as_tensor(eager_dataset.y_daily_excess * float(target_scale), dtype=torch.float32)
        y_cum = torch.as_tensor(eager_dataset.y_cum_excess * float(target_scale), dtype=torch.float32)
        y_risk_np = np.stack(
            [eager_dataset.y_max_drawdown_20d, eager_dataset.y_worst_1d_20d, eager_dataset.y_upside_20d],
            axis=1,
        )
        y_risk = torch.as_tensor(y_risk_np * float(target_scale), dtype=torch.float32)
    train_indices = dataset_view.role_indices("train")
    validation_indices = dataset_view.role_indices("validation")
    test_indices = dataset_view.role_indices("test")
    if len(train_indices) < 2 or len(validation_indices) < 1:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "insufficient_role_samples",
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "models": {},
            "family_summary": {},
            "dataset_mode": dataset_view.dataset_mode,
            "feature_profile": feature_profile,
            "feature_manifest": feature_manifest,
            "selected_seed": 0,
            "selected_model_family": "",
            "selected_signal_profile": "failed",
            "validation_multiscale_score": 0.0,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / "forecast_training_summary.json", _json_ready(summary))
        return summary

    model_summaries: dict[str, dict[str, Any]] = {}
    learning_rows: list[dict[str, Any]] = []
    if resume_payload is not None and isinstance(resume_payload.get("learning_rows"), list):
        learning_rows = [dict(row) for row in resume_payload.get("learning_rows", []) if isinstance(row, dict)]
    run_started_at = _now_iso_seconds()
    run_started_monotonic = time.monotonic()
    progress_path = study_root / str(progress_json_name or "forecast_progress.json")
    batch_size = max(int(batch_size), 1)
    max_epochs = max(int(epochs), 1)
    min_epochs = max(int(min_epochs), 1)
    patience_limit = max(int(early_stop_patience), 1)
    checkpoint_interval = max(int(checkpoint_every_n_epochs), 0)
    accum_steps = max(int(grad_accum_steps), 1)
    pin_memory = resolved_device.type == "cuda"
    loader_kwargs = _data_loader_kwargs(
        pin_memory=pin_memory,
        dataloader_num_workers=int(dataloader_num_workers),
        prefetch_factor=int(prefetch_factor),
    )
    static_model_options = _static_context_model_options(dataset_view)

    for family in families:
        seed_summaries: dict[str, dict[str, Any]] = {}
        for current_seed in seed_values:
            torch.manual_seed(int(current_seed))
            np.random.seed(int(current_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(current_seed))
            model = make_forecast_model(
                family,
                input_dim=int(dataset_view.input_dim),
                hidden_dim=int(hidden_dim),
                horizon=PATH20_HORIZON,
                dropout=float(dropout),
                gru_layers=int(gru_layers),
                transformer_layers=int(transformer_layers),
                transformer_heads=int(transformer_heads),
                patch_sizes=tuple(int(item) for item in patch_sizes),
                output_profile=output_profile,
                **static_model_options,
            ).to(resolved_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            generator = torch.Generator()
            generator.manual_seed(int(current_seed))
            cross_sectional_batching = bool(
                family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES and dataset_view.dataset_mode == "memmap"
            )
            train_loader = DataLoader(
                dataset_view.date_batch_torch_dataset(train_indices, target_scale=target_scale)
                if cross_sectional_batching
                else dataset_view.torch_dataset(train_indices, target_scale=target_scale),
                batch_size=1 if cross_sectional_batching else batch_size,
                shuffle=True,
                generator=generator,
                collate_fn=_collate_forecast_date_batches if cross_sectional_batching else None,
                **loader_kwargs,
            )
            best_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_best.pt"
            last_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_last.pt"
            best_score = -float("inf")
            best_epoch = 0
            best_validation_loss = float("inf")
            best_validation_metrics: dict[str, Any] = {"status": "not_run"}
            best_train_loss = float("inf")
            stopped_reason = "max_epochs_reached"
            patience_used = 0
            last_train_loss = float("inf")
            resume_from_checkpoint_pt = ""
            resume_start_epoch = 1
            best_checkpoint_payload: dict[str, Any] | None = None
            model_config = _model_config_for_training(
                family=family,
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                gru_layers=int(gru_layers),
                transformer_layers=int(transformer_layers),
                transformer_heads=int(transformer_heads),
                patch_sizes=tuple(int(item) for item in patch_sizes),
                dataset_view=dataset_view,
                static_model_options=static_model_options,
                output_profile=output_profile,
            )
            optimizer_config = {
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": float(grad_clip),
                "grad_accum_steps": int(accum_steps),
                "loss_profile": str(loss_profile),
            }
            training_config = {
                "epochs": int(max_epochs),
                "min_epochs": int(min_epochs),
                "early_stop_patience": int(patience_limit),
                "early_stop_min_delta": float(early_stop_min_delta),
                "batch_size": int(batch_size),
                "effective_batch_size": int(1 if cross_sectional_batching else batch_size),
                "cross_section_batching_enabled": bool(cross_sectional_batching),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": float(grad_clip),
                "grad_accum_steps": int(accum_steps),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "gru_layers": int(gru_layers),
                "transformer_layers": int(transformer_layers),
                "transformer_heads": int(transformer_heads),
                "patch_sizes": [int(item) for item in patch_sizes],
                "feature_profile": feature_profile,
                "feature_count": int(dataset_view.input_dim),
                "selection_profile": selection_profile,
                "output_profile": str(output_profile),
                "loss_profile": str(loss_profile),
                "decision_utility": dict(decision_config),
                "ranking_baseline": str(ranking_baseline),
                "slot_diagnostics": bool(slot_diagnostics),
                "static_context_schema": dict(dataset_view.static_context_schema),
                "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
                "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
                "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
                "cross_section_batching_enabled": bool(cross_sectional_batching),
            }
            if resume_payload is not None:
                _validate_forecast_resume_checkpoint(
                    resume_payload,
                    model_family=family,
                    seed=int(current_seed),
                    dataset_view=dataset_view,
                    target_scale=float(target_scale),
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    selection_profile=selection_profile,
                    loss_profile=loss_profile,
                    output_profile=output_profile,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                )
                model.load_state_dict(resume_payload["state_dict"])
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                scaler.load_state_dict(resume_payload["scaler_state_dict"])
                _restore_forecast_rng_state(resume_payload, generator)
                resume_from_checkpoint_pt = str(Path(resume_from).resolve()) if resume_from is not None else ""
                resume_start_epoch = int(resume_payload.get("epoch", 0)) + 1
                if resume_start_epoch > max_epochs:
                    raise ValueError("--forecast-resume-from epoch must be lower than --forecast-epochs.")
                best_score = float(resume_payload.get("best_score", -float("inf")) or -float("inf"))
                best_epoch = int(resume_payload.get("best_epoch", 0) or 0)
                best_validation_loss = float(resume_payload.get("best_validation_loss", float("inf")) or float("inf"))
                best_validation_metrics = dict(resume_payload.get("best_validation_metrics", {}) or {})
                best_train_loss = float(resume_payload.get("best_train_loss", float("inf")) or float("inf"))
                last_train_loss = float(resume_payload.get("last_train_loss", float("inf")) or float("inf"))
                patience_used = int(resume_payload.get("patience_used", 0) or 0)
                if isinstance(resume_payload.get("best_checkpoint_payload"), dict):
                    best_checkpoint_payload = dict(resume_payload["best_checkpoint_payload"])
                    _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
                else:
                    prior_best_path = Path(str(resume_payload.get("best_checkpoint_pt", "") or ""))
                    if prior_best_path.exists():
                        prior_best_payload = torch.load(prior_best_path, map_location="cpu", weights_only=False)
                        if isinstance(prior_best_payload, dict):
                            best_checkpoint_payload = prior_best_payload
                            _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
            for epoch in range(resume_start_epoch, max_epochs + 1):
                epoch_started_monotonic = time.monotonic()
                model.train()
                epoch_losses: list[float] = []
                epoch_counts: list[int] = []
                optimizer.zero_grad(set_to_none=True)
                for step, raw_batch in enumerate(train_loader, start=1):
                    if cross_sectional_batching:
                        batch_x_cpu, stock_mask_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, _ = raw_batch
                        batch_x = batch_x_cpu.to(resolved_device, non_blocking=pin_memory)
                        stock_mask = stock_mask_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_daily = batch_y_daily_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_cum = batch_y_cum_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_risk = batch_y_risk_cpu.to(resolved_device, non_blocking=pin_memory)
                        flat_mask = stock_mask.reshape(-1)
                        with _autocast_context(resolved_device, amp_enabled):
                            pred = _forecast_model_forward(model, batch_x, stock_mask=stock_mask)
                            loss = _forecast_loss(
                                {key: value[flat_mask] for key, value in pred.items()},
                                batch_y_daily.reshape(-1, batch_y_daily.shape[-1])[flat_mask],
                                batch_y_cum.reshape(-1, batch_y_cum.shape[-1])[flat_mask],
                                batch_y_risk.reshape(-1, batch_y_risk.shape[-1])[flat_mask],
                                loss_profile=loss_profile,
                                target_scale=target_scale,
                                decision_cost_bps=decision_cost_bps,
                                decision_hit_threshold_bps=decision_hit_threshold_bps,
                                decision_drawdown_penalty=decision_drawdown_penalty,
                            )
                        batch_count = int(flat_mask.sum().detach().cpu())
                    else:
                        batch_x_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                        batch_x = batch_x_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_daily = batch_y_daily_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_cum = batch_y_cum_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_risk = batch_y_risk_cpu.to(resolved_device, non_blocking=pin_memory)
                        static_context_ids = (
                            static_context_ids_cpu.to(resolved_device, non_blocking=pin_memory)
                            if static_context_ids_cpu is not None
                            else None
                        )
                        with _autocast_context(resolved_device, amp_enabled):
                            pred = _forecast_model_forward(model, batch_x, static_context_ids)
                            loss = _forecast_loss(
                                pred,
                                batch_y_daily,
                                batch_y_cum,
                                batch_y_risk,
                                loss_profile=loss_profile,
                                target_scale=target_scale,
                                decision_cost_bps=decision_cost_bps,
                                decision_hit_threshold_bps=decision_hit_threshold_bps,
                                decision_drawdown_penalty=decision_drawdown_penalty,
                            )
                        batch_count = int(batch_x.shape[0])
                    epoch_losses.append(float(loss.detach().cpu()))
                    epoch_counts.append(batch_count)
                    scaler.scale(loss / float(accum_steps)).backward()
                    if step % accum_steps == 0 or step == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                last_train_loss = (
                    float(np.average(epoch_losses, weights=epoch_counts)) if epoch_losses and epoch_counts else float("inf")
                )
                validation_loss = _evaluate_loss(
                    model,
                    dataset_view if dataset_view.dataset_mode == "memmap" else x,
                    y_daily,
                    y_cum,
                    y_risk,
                    validation_indices,
                    batch_size=batch_size,
                    device=resolved_device,
                    amp_enabled=amp_enabled,
                    target_scale=target_scale,
                    loss_profile=loss_profile,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                )
                validation_predictions = _predict_indices(
                    model,
                    dataset_view if dataset_view.dataset_mode == "memmap" else x,
                    validation_indices,
                    batch_size=batch_size,
                    device=resolved_device,
                    amp_enabled=amp_enabled,
                    target_scale=target_scale,
                )
                validation_frame = _prediction_frame_for_dataset_indices(
                    dataset,
                    indices=validation_indices,
                    predictions=validation_predictions,
                    family=family,
                    target_scale=target_scale,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                )
                validation_metrics = forecast_prediction_metrics(validation_frame)
                score = _validation_score(validation_metrics, validation_loss, (0.65, 0.95), selection_profile)
                multiscale_score = _profile_score(validation_metrics, validation_loss, (0.65, 0.95), "multiscale")
                improved = score > best_score + float(early_stop_min_delta)
                if improved:
                    best_score = score
                    best_epoch = int(epoch)
                    best_validation_loss = float(validation_loss)
                    best_validation_metrics = dict(validation_metrics)
                    best_train_loss = float(last_train_loss)
                    patience_used = 0
                    best_checkpoint_payload = _forecast_checkpoint_payload(
                        checkpoint_kind="best",
                        model=model,
                        optimizer=None,
                        scaler=None,
                        model_family=family,
                        seed=int(current_seed),
                        epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_score=float(best_score),
                        best_validation_loss=float(best_validation_loss),
                        best_validation_metrics=best_validation_metrics,
                        best_train_loss=float(best_train_loss),
                        last_train_loss=float(last_train_loss),
                        patience_used=int(patience_used),
                        feature_columns=list(dataset_view.feature_columns),
                        feature_profile=feature_profile,
                        feature_manifest=feature_manifest,
                        normalization_manifest=dataset_view.normalization_manifest,
                        target_scale=float(target_scale),
                        lookback_days=int(dataset_view.lookback_days),
                        model_config=model_config,
                        training_config=training_config,
                        optimizer_config=optimizer_config,
                        selection_profile=selection_profile,
                        loss_profile=loss_profile,
                        output_profile=output_profile,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        learning_rows=learning_rows,
                        static_context_schema=dataset_view.static_context_schema,
                        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                    )
                    _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
                else:
                    patience_used += 1
                epoch_seconds = float(time.monotonic() - epoch_started_monotonic)
                learning_rows.append(
                    {
                        "model_family": family,
                        "seed": int(current_seed),
                        "epoch": int(epoch),
                        "train_loss": float(last_train_loss),
                        "validation_loss": float(validation_loss),
                        "validation_multiscale_score": float(multiscale_score),
                        "validation_selection_score": float(score),
                        "selected_signal_profile": forecast_signal_profile(validation_metrics),
                        "rank_ic_20d": float(validation_metrics.get("rank_ic_20d", 0.0) or 0.0),
                        "top_bottom_spread_20d": float(validation_metrics.get("top_bottom_spread_20d", 0.0) or 0.0),
                        "rank_ic_5d": float(validation_metrics.get("rank_ic_5d", 0.0) or 0.0),
                        "top_bottom_spread_5d": float(validation_metrics.get("top_bottom_spread_5d", 0.0) or 0.0),
                        "rank_ic_3d": float(validation_metrics.get("rank_ic_3d", 0.0) or 0.0),
                        "top_bottom_spread_3d": float(validation_metrics.get("top_bottom_spread_3d", 0.0) or 0.0),
                        "rank_ic_upside_20d": float(validation_metrics.get("rank_ic_upside_20d", 0.0) or 0.0),
                        "top_bottom_spread_upside_20d": float(
                            validation_metrics.get("top_bottom_spread_upside_20d", 0.0) or 0.0
                        ),
                        "decision_score_rank_ic": float(validation_metrics.get("decision_score_rank_ic", 0.0) or 0.0),
                        "decision_score_top_bottom_spread": float(
                            validation_metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0
                        ),
                        "decision_hit_lift_top20_mean": float(
                            validation_metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0
                        ),
                        "q10_coverage_mean": float(validation_metrics.get("q10_coverage_mean", 0.0) or 0.0),
                        "q90_coverage_mean": float(validation_metrics.get("q90_coverage_mean", 0.0) or 0.0),
                        "direction_accuracy_20d": float(validation_metrics.get("direction_accuracy_20d", 0.0) or 0.0),
                        "is_best": bool(improved),
                        "patience_used": int(patience_used),
                        "epoch_seconds": float(epoch_seconds),
                    }
                )
                _write_learning_curve_incremental(study_root / "forecast_learning_curve.csv", learning_rows)
                if bool(save_last_checkpoint):
                    last_payload = _forecast_checkpoint_payload(
                        checkpoint_kind="last",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        model_family=family,
                        seed=int(current_seed),
                        epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_score=float(best_score),
                        best_validation_loss=float(best_validation_loss),
                        best_validation_metrics=best_validation_metrics,
                        best_train_loss=float(best_train_loss),
                        last_train_loss=float(last_train_loss),
                        patience_used=int(patience_used),
                        feature_columns=list(dataset_view.feature_columns),
                        feature_profile=feature_profile,
                        feature_manifest=feature_manifest,
                        normalization_manifest=dataset_view.normalization_manifest,
                        target_scale=float(target_scale),
                        lookback_days=int(dataset_view.lookback_days),
                        model_config=model_config,
                        training_config=training_config,
                        optimizer_config=optimizer_config,
                        selection_profile=selection_profile,
                        loss_profile=loss_profile,
                        output_profile=output_profile,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        learning_rows=learning_rows,
                        static_context_schema=dataset_view.static_context_schema,
                        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                        best_checkpoint_pt=best_checkpoint_path,
                        best_checkpoint_payload=best_checkpoint_payload,
                        rng_state=_forecast_rng_state(generator),
                    )
                    _save_forecast_checkpoint_atomic(last_checkpoint_path, last_payload)
                if checkpoint_interval > 0 and int(epoch) % checkpoint_interval == 0:
                    epoch_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_epoch{int(epoch)}.pt"
                    epoch_payload = _forecast_checkpoint_payload(
                        checkpoint_kind="epoch",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        model_family=family,
                        seed=int(current_seed),
                        epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_score=float(best_score),
                        best_validation_loss=float(best_validation_loss),
                        best_validation_metrics=best_validation_metrics,
                        best_train_loss=float(best_train_loss),
                        last_train_loss=float(last_train_loss),
                        patience_used=int(patience_used),
                        feature_columns=list(dataset_view.feature_columns),
                        feature_profile=feature_profile,
                        feature_manifest=feature_manifest,
                        normalization_manifest=dataset_view.normalization_manifest,
                        target_scale=float(target_scale),
                        lookback_days=int(dataset_view.lookback_days),
                        model_config=model_config,
                        training_config=training_config,
                        optimizer_config=optimizer_config,
                        selection_profile=selection_profile,
                        loss_profile=loss_profile,
                        output_profile=output_profile,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        learning_rows=learning_rows,
                        static_context_schema=dataset_view.static_context_schema,
                        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                        best_checkpoint_pt=best_checkpoint_path,
                        best_checkpoint_payload=best_checkpoint_payload,
                        rng_state=_forecast_rng_state(generator),
                    )
                    _save_forecast_checkpoint_atomic(epoch_checkpoint_path, epoch_payload)
                _write_forecast_progress(
                    progress_path,
                    status="running",
                    model_family=family,
                    seed=int(current_seed),
                    current_epoch=int(epoch),
                    max_epochs=int(max_epochs),
                    min_epochs=int(min_epochs),
                    patience_limit=int(patience_limit),
                    patience_used=int(patience_used),
                    best_epoch=int(best_epoch),
                    best_score=float(best_score),
                    best_validation_loss=float(best_validation_loss),
                    best_validation_metrics=best_validation_metrics,
                    last_train_loss=float(last_train_loss),
                    epoch_seconds=float(epoch_seconds),
                    run_started_at=run_started_at,
                    elapsed_seconds=float(time.monotonic() - run_started_monotonic),
                    learning_rows=learning_rows,
                    last_checkpoint_pt=last_checkpoint_path if bool(save_last_checkpoint) else None,
                    best_checkpoint_pt=best_checkpoint_path,
                )
                if epoch >= min_epochs and patience_used >= patience_limit:
                    stopped_reason = "early_stopping_patience_exhausted"
                    break
            if not best_checkpoint_path.exists():
                best_checkpoint_payload = _forecast_checkpoint_payload(
                    checkpoint_kind="best",
                    model=model,
                    optimizer=None,
                    scaler=None,
                    model_family=family,
                    seed=int(current_seed),
                    epoch=int(max_epochs),
                    best_epoch=int(best_epoch or max_epochs),
                    best_score=float(best_score),
                    best_validation_loss=float(best_validation_loss if np.isfinite(best_validation_loss) else last_train_loss),
                    best_validation_metrics=best_validation_metrics,
                    best_train_loss=float(best_train_loss if np.isfinite(best_train_loss) else last_train_loss),
                    last_train_loss=float(last_train_loss),
                    patience_used=int(patience_used),
                    feature_columns=list(dataset_view.feature_columns),
                    feature_profile=feature_profile,
                    feature_manifest=feature_manifest,
                    normalization_manifest=dataset_view.normalization_manifest,
                    target_scale=float(target_scale),
                    lookback_days=int(dataset_view.lookback_days),
                    model_config=model_config,
                    training_config=training_config,
                    optimizer_config=optimizer_config,
                    selection_profile=selection_profile,
                    loss_profile=loss_profile,
                    output_profile=output_profile,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                    learning_rows=learning_rows,
                    static_context_schema=dataset_view.static_context_schema,
                    symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                    industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                    board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                )
                _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
            checkpoint = torch.load(best_checkpoint_path, map_location=resolved_device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            validation_predictions = _predict_indices(
                model,
                dataset_view if dataset_view.dataset_mode == "memmap" else x,
                validation_indices,
                batch_size=batch_size,
                device=resolved_device,
                amp_enabled=amp_enabled,
                target_scale=target_scale,
            )
            test_predictions = _predict_indices(
                model,
                dataset_view if dataset_view.dataset_mode == "memmap" else x,
                test_indices,
                batch_size=batch_size,
                device=resolved_device,
                amp_enabled=amp_enabled,
                target_scale=target_scale,
            )
            validation_frame = _prediction_frame_for_dataset_indices(
                dataset,
                indices=validation_indices,
                predictions=validation_predictions,
                family=family,
                target_scale=target_scale,
                decision_cost_bps=decision_cost_bps,
                decision_hit_threshold_bps=decision_hit_threshold_bps,
                decision_drawdown_penalty=decision_drawdown_penalty,
            )
            test_frame = _prediction_frame_for_dataset_indices(
                dataset,
                indices=test_indices,
                predictions=test_predictions,
                family=family,
                target_scale=target_scale,
                decision_cost_bps=decision_cost_bps,
                decision_hit_threshold_bps=decision_hit_threshold_bps,
                decision_drawdown_penalty=decision_drawdown_penalty,
            )
            final_validation_metrics = forecast_prediction_metrics(validation_frame)
            final_test_metrics = forecast_prediction_metrics(test_frame)
            slot_diagnostics_path = ""
            if bool(slot_diagnostics) and family == "sector_slot_mixer_sequence":
                slot_diagnostics_path = _write_slot_diagnostics(
                    study_root / f"forecast_slot_diagnostics_{family}_seed{int(current_seed)}.json",
                    model=model,
                    dataset=dataset,
                    indices=validation_indices,
                    seed=int(current_seed),
                )
            final_multiscale_score = _profile_score(final_validation_metrics, best_validation_loss, (0.65, 0.95), "multiscale")
            final_trend20_score = _profile_score(final_validation_metrics, best_validation_loss, (0.65, 0.95), "trend20")
            final_short_burst_score = _profile_score(
                final_validation_metrics,
                best_validation_loss,
                (0.65, 0.95),
                "short_burst",
            )
            final_decision_utility_score = _profile_score(
                final_validation_metrics,
                best_validation_loss,
                (0.65, 0.95),
                "decision_utility",
            )
            if bool(write_all_predictions):
                _write_frame(
                    study_root / f"forecast_predictions_validation_{family}_seed{int(current_seed)}.csv",
                    validation_frame,
                )
                _write_frame(
                    study_root / f"forecast_predictions_test_{family}_seed{int(current_seed)}.csv",
                    test_frame,
                )
            seed_summaries[str(int(current_seed))] = {
                "status": "completed",
                "model_family": family,
                "seed": int(current_seed),
                "epochs_ran": int(
                    max(row["epoch"] for row in learning_rows if row["model_family"] == family and row["seed"] == int(current_seed))
                ),
                "best_epoch": int(best_epoch),
                "stopped_reason": stopped_reason,
                "final_train_loss": float(last_train_loss),
                "best_train_loss": float(best_train_loss),
                "best_validation_loss": float(best_validation_loss),
                "best_score": float(best_score),
                "best_validation_metrics": best_validation_metrics,
                "validation_metrics": final_validation_metrics,
                "test_metrics": final_test_metrics,
                "validation_signal_profile": forecast_signal_profile(final_validation_metrics),
                "validation_multiscale_score": float(final_multiscale_score),
                "validation_trend20_score": float(final_trend20_score),
                "validation_short_burst_score": float(final_short_burst_score),
                "validation_decision_utility_score": float(final_decision_utility_score),
                "validation_selection_score": float(
                    {
                        "multiscale": final_multiscale_score,
                        "trend20": final_trend20_score,
                        "short_burst": final_short_burst_score,
                        "decision_utility": final_decision_utility_score,
                    }[selection_profile]
                ),
                "checkpoint_pt": str(best_checkpoint_path.resolve()),
                "last_checkpoint_pt": str(last_checkpoint_path.resolve()) if last_checkpoint_path.exists() else "",
                "resume_from_checkpoint_pt": resume_from_checkpoint_pt,
                "resume_start_epoch": int(resume_start_epoch),
                "slot_diagnostics_path": slot_diagnostics_path,
            }
        model_summaries[family] = {
            "status": "completed",
            "model_family": family,
            "epochs": int(max_epochs),
            "min_epochs": int(min_epochs),
            "early_stop_patience": int(patience_limit),
            "batch_size": int(batch_size),
            "effective_batch_size": int(1 if family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES else batch_size),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "feature_count": int(dataset_view.input_dim),
            "selection_profile": selection_profile,
            "output_profile": str(output_profile),
            "loss_profile": str(loss_profile),
            "decision_utility": dict(decision_config),
            "slot_diagnostics": bool(slot_diagnostics),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "static_context_schema": dict(dataset_view.static_context_schema),
            "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
            "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
            "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
            "cross_section_batching_enabled": bool(family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES),
            "seed_summaries": seed_summaries,
            "family_summary": _family_summary(seed_summaries),
        }

    learning_curve_path = study_root / "forecast_learning_curve.csv"
    _write_frame(learning_curve_path, pd.DataFrame(learning_rows))
    selected_family, selected_seed = _select_family_seed(model_summaries, selection_profile=selection_profile)
    selected_seed_summary = dict(
        model_summaries.get(selected_family, {}).get("seed_summaries", {}).get(str(int(selected_seed)), {})
    )
    selected_checkpoint_path = str(selected_seed_summary.get("checkpoint_pt", "") or "")
    selected_predictions_validation: pd.DataFrame
    selected_predictions_test: pd.DataFrame
    if selected_family and selected_seed and selected_checkpoint_path:
        selected_model = make_forecast_model(
            selected_family,
            input_dim=int(dataset_view.input_dim),
            hidden_dim=int(hidden_dim),
            horizon=PATH20_HORIZON,
            dropout=float(dropout),
            gru_layers=int(gru_layers),
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            patch_sizes=tuple(int(item) for item in patch_sizes),
            output_profile=output_profile,
            **static_model_options,
        ).to(resolved_device)
        checkpoint = torch.load(selected_checkpoint_path, map_location=resolved_device, weights_only=False)
        selected_model.load_state_dict(checkpoint["state_dict"])
        validation_predictions_np = _predict_indices(
            selected_model,
            dataset_view if dataset_view.dataset_mode == "memmap" else x,
            validation_indices,
            batch_size=batch_size,
            device=resolved_device,
            amp_enabled=amp_enabled,
            target_scale=target_scale,
        )
        test_predictions_np = _predict_indices(
            selected_model,
            dataset_view if dataset_view.dataset_mode == "memmap" else x,
            test_indices,
            batch_size=batch_size,
            device=resolved_device,
            amp_enabled=amp_enabled,
            target_scale=target_scale,
        )
        selected_predictions_validation = _prediction_frame_for_dataset_indices(
            dataset,
            indices=validation_indices,
            predictions=validation_predictions_np,
            family=selected_family,
            target_scale=target_scale,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
        )
        selected_predictions_test = _prediction_frame_for_dataset_indices(
            dataset,
            indices=test_indices,
            predictions=test_predictions_np,
            family=selected_family,
            target_scale=target_scale,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
        )
    else:
        selected_predictions_validation = pd.DataFrame()
        selected_predictions_test = pd.DataFrame()
    validation_predictions = (
        selected_predictions_validation if selected_family else pd.DataFrame()
    )
    test_predictions = (
        selected_predictions_test if selected_family else pd.DataFrame()
    )
    validation_csv = _write_frame(study_root / "forecast_predictions_validation.csv", validation_predictions)
    test_csv = _write_frame(study_root / "forecast_predictions_test.csv", test_predictions)
    validation_metrics = dict(selected_seed_summary.get("validation_metrics", {}))
    test_metrics = dict(selected_seed_summary.get("test_metrics", {}))
    verdict = forecast_evidence_verdict(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selection_profile=selection_profile,
    )
    selected_signal_profile = forecast_signal_profile(validation_metrics)
    summary = {
        "status": "completed",
        "stage": "forecast_train",
        "target_scale": float(target_scale),
        "device": str(resolved_device),
        "amp_enabled": bool(amp_enabled),
        "seeds": [int(item) for item in seed_values],
        "feature_profile": feature_profile,
        "feature_manifest": feature_manifest,
        "training_config": {
            "epochs": int(max_epochs),
            "min_epochs": int(min_epochs),
            "early_stop_patience": int(patience_limit),
            "early_stop_min_delta": float(early_stop_min_delta),
            "batch_size": int(batch_size),
            "effective_batch_size": int(1 if selected_family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES else batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "grad_accum_steps": int(accum_steps),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "gru_layers": int(gru_layers),
            "transformer_layers": int(transformer_layers),
            "transformer_heads": int(transformer_heads),
            "patch_sizes": [int(item) for item in patch_sizes],
            "feature_profile": feature_profile,
            "feature_count": int(dataset_view.input_dim),
            "selection_profile": selection_profile,
            "output_profile": str(output_profile),
            "loss_profile": str(loss_profile),
            "decision_utility": dict(decision_config),
            "ranking_baseline": str(ranking_baseline),
            "slot_diagnostics": bool(slot_diagnostics),
            "save_last_checkpoint": bool(save_last_checkpoint),
            "checkpoint_every_n_epochs": int(checkpoint_interval),
            "resume_from_checkpoint_pt": str(Path(resume_from).resolve()) if resume_from is not None and str(resume_from).strip() else "",
            "static_context_schema": dict(dataset_view.static_context_schema),
            "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
            "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
            "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
            "cross_section_batching_enabled": bool(
                selected_family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES if selected_family else False
            ),
        },
        "models": model_summaries,
        "ranking_baseline_summary": ranking_baseline_summary,
        "family_summary": {family: dict(summary.get("family_summary", {})) for family, summary in model_summaries.items()},
        "selected_model_family": selected_family,
        "selected_seed": int(selected_seed),
        "selected_checkpoint_pt": selected_checkpoint_path,
        "selection_rule": f"{selection_profile}_validation_score_after_profile_and_coverage_gates_then_seed_score",
        "selected_signal_profile": selected_signal_profile,
        "validation_multiscale_score": float(selected_seed_summary.get("validation_multiscale_score", 0.0) or 0.0),
        "validation_decision_utility_score": float(
            selected_seed_summary.get("validation_decision_utility_score", 0.0) or 0.0
        ),
        "validation_selection_score": float(selected_seed_summary.get("validation_selection_score", 0.0) or 0.0),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_interpretable": verdict in {"forecast_test_confirmed", "forecast_promising"}
        and validation_metrics.get("status") == "completed"
        and _profile_pass(validation_metrics, selection_profile),
        "evidence_verdict": verdict,
        "forecast_predictions_validation_csv": validation_csv,
        "forecast_predictions_test_csv": test_csv,
        "forecast_learning_curve_csv": str(learning_curve_path.resolve()),
        "progress_json": str(progress_path.resolve()),
        "resume_from_checkpoint_pt": str(Path(resume_from).resolve()) if resume_from is not None and str(resume_from).strip() else "",
        "dataset_mode": dataset_view.dataset_mode,
        "validation_stratified_metrics": stratified_forecast_prediction_metrics(validation_predictions),
        "test_stratified_metrics": stratified_forecast_prediction_metrics(test_predictions),
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    selected_rows = [
        row
        for row in learning_rows
        if str(row.get("model_family")) == str(selected_family) and int(row.get("seed", -1)) == int(selected_seed)
    ]
    if selected_rows:
        latest_selected_row = max(selected_rows, key=lambda row: int(row.get("epoch", 0) or 0))
        selected_last_path_text = str(selected_seed_summary.get("last_checkpoint_pt", "") or "")
        _write_forecast_progress(
            progress_path,
            status="completed",
            model_family=str(selected_family),
            seed=int(selected_seed),
            current_epoch=int(selected_seed_summary.get("epochs_ran", latest_selected_row.get("epoch", 0)) or 0),
            max_epochs=int(max_epochs),
            min_epochs=int(min_epochs),
            patience_limit=int(patience_limit),
            patience_used=int(latest_selected_row.get("patience_used", 0) or 0),
            best_epoch=int(selected_seed_summary.get("best_epoch", 0) or 0),
            best_score=float(selected_seed_summary.get("best_score", 0.0) or 0.0),
            best_validation_loss=float(selected_seed_summary.get("best_validation_loss", 0.0) or 0.0),
            best_validation_metrics=dict(selected_seed_summary.get("best_validation_metrics", {}) or {}),
            last_train_loss=float(selected_seed_summary.get("final_train_loss", 0.0) or 0.0),
            epoch_seconds=float(latest_selected_row.get("epoch_seconds", 0.0) or 0.0),
            run_started_at=run_started_at,
            elapsed_seconds=float(time.monotonic() - run_started_monotonic),
            learning_rows=learning_rows,
            last_checkpoint_pt=Path(selected_last_path_text) if selected_last_path_text else None,
            best_checkpoint_pt=Path(selected_checkpoint_path) if selected_checkpoint_path else None,
        )
    summary = _json_ready(summary)
    write_json(study_root / "forecast_training_summary.json", summary)
    return summary
