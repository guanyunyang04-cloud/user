from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from daily_research.continuous_policy.runtime import write_json
from daily_research.path_policy.forecast_dataset import ForecastSequenceDataset
from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS, PATH20_HORIZON
from daily_research.path_policy.models import (
    GRUPath20Forecaster,
    LinearPath20Forecaster,
    PatchTransformerPath20Forecaster,
    PATH20_FORECAST_AUX_DIM,
    Path20ForecasterMLP,
    pairwise_rank_loss,
    pinball_loss,
)


FORECAST_MODEL_FAMILIES = ("linear_last_day", "mlp_last_day", "gru_sequence", "patch_transformer")
FORECAST_SELECTION_PROFILES = ("multiscale", "trend20", "short_burst")
FORECAST_RISK_AUX_NAMES = ("downside_floor_20d", "worst_1d_20d", "upside_20d")
FORECAST_RANK_LOSS_WEIGHTS = {1: 0.0025, 3: 0.0050, 5: 0.0075, 10: 0.0075, 20: 0.0100}
FORECAST_PROFILE_HORIZON_WEIGHTS: dict[str, dict[int, float]] = {
    "multiscale": {1: 0.05, 3: 0.15, 5: 0.20, 10: 0.25, 20: 0.25},
    "trend20": {10: 0.35, 20: 0.65},
    "short_burst": {1: 0.05, 3: 0.35, 5: 0.30, 10: 0.15, 20: 0.05},
}


class LinearLastDayPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int = PATH20_HORIZON) -> None:
        super().__init__()
        self.base = LinearPath20Forecaster(input_dim=int(input_dim), horizon=int(horizon))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.base(x)


class MLPLastDayPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.15, horizon: int = PATH20_HORIZON) -> None:
        super().__init__()
        self.base = Path20ForecasterMLP(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            horizon=int(horizon),
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
) -> nn.Module:
    family = str(family).strip()
    if family == "linear_last_day":
        return LinearLastDayPath20Forecaster(input_dim=input_dim, horizon=horizon)
    if family == "mlp_last_day":
        return MLPLastDayPath20Forecaster(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, horizon=horizon)
    if family == "gru_sequence":
        return GRUPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
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
        )
    raise ValueError(f"Unsupported forecast model family: {family}")


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _forecast_loss(
    prediction: dict[str, torch.Tensor],
    y_daily_scaled: torch.Tensor,
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
) -> torch.Tensor:
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
        if weight <= 0.0:
            continue
        score = prediction["mu"][:, : int(horizon)].sum(dim=1)
        target = y_cum_scaled[:, pos]
        loss = loss + weight * pairwise_rank_loss(score, target)
    loss = loss + 0.005 * pairwise_rank_loss(prediction["aux"][:, cum_count + 2], y_risk_scaled[:, 2])
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
                pred = model(batch)
            for key in chunks:
                chunks[key].append(pred[key].detach().cpu().numpy())
    empty_shapes = {
        "mu": (0, PATH20_HORIZON),
        "q10": (0, PATH20_HORIZON),
        "q50": (0, PATH20_HORIZON),
        "q90": (0, PATH20_HORIZON),
        "aux": (0, PATH20_FORECAST_AUX_DIM),
    }
    return {key: np.concatenate(values, axis=0) if values else np.empty(empty_shapes[key]) for key, values in chunks.items()}


def _predict_indices(
    model: nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, np.ndarray]:
    if len(indices) == 0:
        return {
            "mu": np.empty((0, PATH20_HORIZON)),
            "q10": np.empty((0, PATH20_HORIZON)),
            "q50": np.empty((0, PATH20_HORIZON)),
            "q90": np.empty((0, PATH20_HORIZON)),
            "aux": np.empty((0, PATH20_FORECAST_AUX_DIM)),
        }
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


def _prediction_frame(
    dataset: ForecastSequenceDataset,
    *,
    role: str,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
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
    metrics["selected_signal_profile"] = forecast_signal_profile(metrics)
    return metrics


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
    if profile == "trend20":
        return _horizon_gate(metrics, 20)
    if profile == "short_burst":
        return _short_burst_gate(metrics)
    return _horizon_gate(metrics, 20) or _short_burst_gate(metrics)


def _profile_score(metrics: dict[str, Any], validation_loss: float, coverage_range: tuple[float, float], selection_profile: str) -> float:
    if metrics.get("status") != "completed":
        return -float(validation_loss)
    profile = str(selection_profile or "multiscale").strip().lower()
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


def _normalize_seeds(seeds: tuple[int, ...] | list[int] | str | None, *, default_seed: int) -> tuple[int, ...]:
    if seeds is None:
        return (int(default_seed),)
    if isinstance(seeds, str):
        parsed = tuple(int(item.strip()) for item in seeds.split(",") if item.strip())
        return parsed or (int(default_seed),)
    parsed = tuple(int(item) for item in seeds)
    return parsed or (int(default_seed),)


def _validation_score(
    metrics: dict[str, Any],
    validation_loss: float,
    coverage_range: tuple[float, float],
    selection_profile: str,
) -> float:
    return _profile_score(metrics, validation_loss, coverage_range, selection_profile)


def _evaluate_loss(
    model: nn.Module,
    x: torch.Tensor,
    y_daily: torch.Tensor,
    y_cum: torch.Tensor,
    y_risk: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    if len(indices) == 0:
        return float("inf")
    model.eval()
    values: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = torch.tensor(indices[start : start + max(int(batch_size), 1)], dtype=torch.long)
            batch_x = x[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_daily = y_daily[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_cum = y_cum[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_risk = y_risk[batch_idx].to(device, non_blocking=device.type == "cuda")
            with _autocast_context(device, amp_enabled):
                pred = model(batch_x)
                loss = _forecast_loss(pred, batch_y_daily, batch_y_cum, batch_y_risk)
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
    count = max(len(seed_summaries), 1)
    summary: dict[str, Any] = {
        "seed_count": int(len(seed_summaries)),
        "validation_q10_coverage_mean": float(np.mean(q10_values)) if q10_values else 0.0,
        "validation_q90_coverage_mean": float(np.mean(q90_values)) if q90_values else 0.0,
        "validation_multiscale_score_mean": float(np.mean(multiscale_scores)) if multiscale_scores else 0.0,
        "validation_multiscale_score_std": float(np.std(multiscale_scores, ddof=0)) if multiscale_scores else 0.0,
        "validation_trend20_score_mean": float(np.mean(trend_scores)) if trend_scores else 0.0,
        "validation_short_burst_score_mean": float(np.mean(short_scores)) if short_scores else 0.0,
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
) -> dict[str, Any]:
    study_root.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    amp_enabled = bool(amp) and resolved_device.type == "cuda"
    seed_values = _normalize_seeds(seeds, default_seed=int(seed))
    families = tuple(str(item).strip() for item in model_families if str(item).strip())
    invalid = sorted(set(families) - set(FORECAST_MODEL_FAMILIES))
    if invalid:
        raise ValueError(f"Unsupported forecast model families: {', '.join(invalid)}")
    selection_profile = str(selection_profile or "multiscale").strip().lower()
    if selection_profile not in FORECAST_SELECTION_PROFILES:
        raise ValueError(f"Unsupported forecast selection profile: {selection_profile}")
    feature_profile = str(dataset.manifest.get("feature_profile", ""))
    feature_manifest = dict(dataset.manifest.get("feature_manifest", {}))
    if dataset.x.shape[0] == 0:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "empty_forecast_dataset",
            "models": {},
            "family_summary": {},
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

    x = torch.as_tensor(dataset.x, dtype=torch.float32)
    y_daily = torch.as_tensor(dataset.y_daily_excess * float(target_scale), dtype=torch.float32)
    y_cum = torch.as_tensor(dataset.y_cum_excess * float(target_scale), dtype=torch.float32)
    y_risk_np = np.stack(
        [dataset.y_max_drawdown_20d, dataset.y_worst_1d_20d, dataset.y_upside_20d],
        axis=1,
    )
    y_risk = torch.as_tensor(y_risk_np * float(target_scale), dtype=torch.float32)
    train_indices = np.flatnonzero(dataset.role == "train")
    validation_indices = np.flatnonzero(dataset.role == "validation")
    test_indices = np.flatnonzero(dataset.role == "test")
    if len(train_indices) < 2 or len(validation_indices) < 1:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "insufficient_role_samples",
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "models": {},
            "family_summary": {},
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
    batch_size = max(int(batch_size), 1)
    max_epochs = max(int(epochs), 1)
    min_epochs = max(int(min_epochs), 1)
    patience_limit = max(int(early_stop_patience), 1)
    accum_steps = max(int(grad_accum_steps), 1)
    pin_memory = resolved_device.type == "cuda"

    for family in families:
        seed_summaries: dict[str, dict[str, Any]] = {}
        for current_seed in seed_values:
            torch.manual_seed(int(current_seed))
            np.random.seed(int(current_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(current_seed))
            model = make_forecast_model(
                family,
                input_dim=int(dataset.x.shape[-1]),
                hidden_dim=int(hidden_dim),
                horizon=PATH20_HORIZON,
                dropout=float(dropout),
                gru_layers=int(gru_layers),
                transformer_layers=int(transformer_layers),
                transformer_heads=int(transformer_heads),
                patch_sizes=tuple(int(item) for item in patch_sizes),
            ).to(resolved_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            generator = torch.Generator()
            generator.manual_seed(int(current_seed))
            train_loader = DataLoader(
                TensorDataset(torch.as_tensor(train_indices, dtype=torch.long)),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                pin_memory=pin_memory,
            )
            best_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_best.pt"
            best_score = -float("inf")
            best_epoch = 0
            best_validation_loss = float("inf")
            best_validation_metrics: dict[str, Any] = {"status": "not_run"}
            best_train_loss = float("inf")
            stopped_reason = "max_epochs_reached"
            patience_used = 0
            last_train_loss = float("inf")
            model_config = {
                "model_family": family,
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "gru_layers": int(gru_layers),
                "transformer_layers": int(transformer_layers),
                "transformer_heads": int(transformer_heads),
                "patch_sizes": [int(item) for item in patch_sizes],
            }
            for epoch in range(1, max_epochs + 1):
                model.train()
                epoch_losses: list[float] = []
                epoch_counts: list[int] = []
                optimizer.zero_grad(set_to_none=True)
                for step, (batch_idx_cpu,) in enumerate(train_loader, start=1):
                    batch_idx = batch_idx_cpu.to(torch.long)
                    batch_x = x[batch_idx].to(resolved_device, non_blocking=pin_memory)
                    batch_y_daily = y_daily[batch_idx].to(resolved_device, non_blocking=pin_memory)
                    batch_y_cum = y_cum[batch_idx].to(resolved_device, non_blocking=pin_memory)
                    batch_y_risk = y_risk[batch_idx].to(resolved_device, non_blocking=pin_memory)
                    with _autocast_context(resolved_device, amp_enabled):
                        pred = model(batch_x)
                        loss = _forecast_loss(pred, batch_y_daily, batch_y_cum, batch_y_risk)
                    epoch_losses.append(float(loss.detach().cpu()))
                    epoch_counts.append(int(batch_idx.numel()))
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
                    x,
                    y_daily,
                    y_cum,
                    y_risk,
                    validation_indices,
                    batch_size=batch_size,
                    device=resolved_device,
                    amp_enabled=amp_enabled,
                )
                validation_predictions = _predict_indices(
                    model,
                    x,
                    validation_indices,
                    batch_size=batch_size,
                    device=resolved_device,
                    amp_enabled=amp_enabled,
                )
                validation_frame = _prediction_frame_for_indices(
                    dataset,
                    indices=validation_indices,
                    predictions=validation_predictions,
                    family=family,
                    target_scale=target_scale,
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
                    torch.save(
                        {
                            "model_family": family,
                            "seed": int(current_seed),
                            "state_dict": model.state_dict(),
                            "feature_columns": list(dataset.feature_columns),
                            "feature_profile": feature_profile,
                            "feature_manifest": feature_manifest,
                            "normalization": dataset.normalization_manifest,
                            "target_scale": float(target_scale),
                            "lookback_days": int(dataset.x.shape[1]),
                            "horizon": PATH20_HORIZON,
                            "model_config": model_config,
                            "best_epoch": int(best_epoch),
                            "best_validation_loss": float(best_validation_loss),
                            "best_validation_metrics": _json_ready(best_validation_metrics),
                        },
                        best_checkpoint_path,
                    )
                else:
                    patience_used += 1
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
                        "q10_coverage_mean": float(validation_metrics.get("q10_coverage_mean", 0.0) or 0.0),
                        "q90_coverage_mean": float(validation_metrics.get("q90_coverage_mean", 0.0) or 0.0),
                        "direction_accuracy_20d": float(validation_metrics.get("direction_accuracy_20d", 0.0) or 0.0),
                        "is_best": bool(improved),
                        "patience_used": int(patience_used),
                    }
                )
                if epoch >= min_epochs and patience_used >= patience_limit:
                    stopped_reason = "early_stopping_patience_exhausted"
                    break
            if not best_checkpoint_path.exists():
                torch.save(
                    {
                        "model_family": family,
                        "seed": int(current_seed),
                        "state_dict": model.state_dict(),
                        "feature_columns": list(dataset.feature_columns),
                        "feature_profile": feature_profile,
                        "feature_manifest": feature_manifest,
                        "normalization": dataset.normalization_manifest,
                        "target_scale": float(target_scale),
                        "lookback_days": int(dataset.x.shape[1]),
                        "horizon": PATH20_HORIZON,
                        "model_config": model_config,
                        "best_epoch": int(max_epochs),
                        "best_validation_loss": float(last_train_loss),
                        "best_validation_metrics": _json_ready(best_validation_metrics),
                    },
                    best_checkpoint_path,
                )
            checkpoint = torch.load(best_checkpoint_path, map_location=resolved_device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            validation_predictions = _predict_indices(
                model,
                x,
                validation_indices,
                batch_size=batch_size,
                device=resolved_device,
                amp_enabled=amp_enabled,
            )
            test_predictions = _predict_indices(
                model,
                x,
                test_indices,
                batch_size=batch_size,
                device=resolved_device,
                amp_enabled=amp_enabled,
            )
            validation_frame = _prediction_frame_for_indices(
                dataset,
                indices=validation_indices,
                predictions=validation_predictions,
                family=family,
                target_scale=target_scale,
            )
            test_frame = _prediction_frame_for_indices(
                dataset,
                indices=test_indices,
                predictions=test_predictions,
                family=family,
                target_scale=target_scale,
            )
            final_validation_metrics = forecast_prediction_metrics(validation_frame)
            final_test_metrics = forecast_prediction_metrics(test_frame)
            final_multiscale_score = _profile_score(final_validation_metrics, best_validation_loss, (0.65, 0.95), "multiscale")
            final_trend20_score = _profile_score(final_validation_metrics, best_validation_loss, (0.65, 0.95), "trend20")
            final_short_burst_score = _profile_score(
                final_validation_metrics,
                best_validation_loss,
                (0.65, 0.95),
                "short_burst",
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
                "best_validation_metrics": best_validation_metrics,
                "validation_metrics": final_validation_metrics,
                "test_metrics": final_test_metrics,
                "validation_signal_profile": forecast_signal_profile(final_validation_metrics),
                "validation_multiscale_score": float(final_multiscale_score),
                "validation_trend20_score": float(final_trend20_score),
                "validation_short_burst_score": float(final_short_burst_score),
                "validation_selection_score": float(
                    {
                        "multiscale": final_multiscale_score,
                        "trend20": final_trend20_score,
                        "short_burst": final_short_burst_score,
                    }[selection_profile]
                ),
                "checkpoint_pt": str(best_checkpoint_path.resolve()),
            }
        model_summaries[family] = {
            "status": "completed",
            "model_family": family,
            "epochs": int(max_epochs),
            "min_epochs": int(min_epochs),
            "early_stop_patience": int(patience_limit),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "feature_count": int(dataset.x.shape[-1]),
            "selection_profile": selection_profile,
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
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
            input_dim=int(dataset.x.shape[-1]),
            hidden_dim=int(hidden_dim),
            horizon=PATH20_HORIZON,
            dropout=float(dropout),
            gru_layers=int(gru_layers),
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            patch_sizes=tuple(int(item) for item in patch_sizes),
        ).to(resolved_device)
        checkpoint = torch.load(selected_checkpoint_path, map_location=resolved_device, weights_only=False)
        selected_model.load_state_dict(checkpoint["state_dict"])
        validation_predictions_np = _predict_indices(
            selected_model,
            x,
            validation_indices,
            batch_size=batch_size,
            device=resolved_device,
            amp_enabled=amp_enabled,
        )
        test_predictions_np = _predict_indices(
            selected_model,
            x,
            test_indices,
            batch_size=batch_size,
            device=resolved_device,
            amp_enabled=amp_enabled,
        )
        selected_predictions_validation = _prediction_frame_for_indices(
            dataset,
            indices=validation_indices,
            predictions=validation_predictions_np,
            family=selected_family,
            target_scale=target_scale,
        )
        selected_predictions_test = _prediction_frame_for_indices(
            dataset,
            indices=test_indices,
            predictions=test_predictions_np,
            family=selected_family,
            target_scale=target_scale,
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
            "feature_count": int(dataset.x.shape[-1]),
            "selection_profile": selection_profile,
        },
        "models": model_summaries,
        "family_summary": {family: dict(summary.get("family_summary", {})) for family, summary in model_summaries.items()},
        "selected_model_family": selected_family,
        "selected_seed": int(selected_seed),
        "selected_checkpoint_pt": selected_checkpoint_path,
        "selection_rule": f"{selection_profile}_validation_score_after_profile_and_coverage_gates_then_seed_score",
        "selected_signal_profile": selected_signal_profile,
        "validation_multiscale_score": float(selected_seed_summary.get("validation_multiscale_score", 0.0) or 0.0),
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
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    summary = _json_ready(summary)
    write_json(study_root / "forecast_training_summary.json", summary)
    return summary
