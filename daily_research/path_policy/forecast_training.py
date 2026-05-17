from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from daily_research.continuous_policy.runtime import write_json
from daily_research.path_policy.forecast_dataset import ForecastSequenceDataset
from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS, PATH20_HORIZON
from daily_research.path_policy.models import (
    GRUPath20Forecaster,
    LinearPath20Forecaster,
    PatchTransformerPath20Forecaster,
    pairwise_rank_loss,
    pinball_loss,
)


FORECAST_MODEL_FAMILIES = ("linear_last_day", "gru_sequence", "patch_transformer")


class LinearLastDayPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int = PATH20_HORIZON) -> None:
        super().__init__()
        self.base = LinearPath20Forecaster(input_dim=int(input_dim), horizon=int(horizon))

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
    hidden_dim: int = 96,
    horizon: int = PATH20_HORIZON,
    dropout: float = 0.10,
) -> nn.Module:
    family = str(family).strip()
    if family == "linear_last_day":
        return LinearLastDayPath20Forecaster(input_dim=input_dim, horizon=horizon)
    if family == "gru_sequence":
        return GRUPath20Forecaster(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, horizon=horizon)
    if family == "patch_transformer":
        num_heads = 4 if int(hidden_dim) % 4 == 0 else 2 if int(hidden_dim) % 2 == 0 else 1
        return PatchTransformerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            patch_size=4,
            num_layers=2,
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
    loss = F.huber_loss(prediction["mu"], y_daily_scaled)
    loss = loss + 0.20 * pinball_loss(prediction["q10"], y_daily_scaled, 0.10)
    loss = loss + 0.20 * pinball_loss(prediction["q50"], y_daily_scaled, 0.50)
    loss = loss + 0.20 * pinball_loss(prediction["q90"], y_daily_scaled, 0.90)
    loss = loss + 0.25 * F.huber_loss(prediction["aux"][:, :3], y_cum_scaled)
    loss = loss + 0.05 * F.huber_loss(prediction["aux"][:, 3:6], y_risk_scaled)
    loss = loss + 0.02 * pairwise_rank_loss(prediction["mu"].sum(dim=1), y_daily_scaled.sum(dim=1))
    return loss


def _predict_all(model: nn.Module, x: torch.Tensor, *, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {"mu": [], "q10": [], "q50": [], "q90": [], "aux": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], max(int(batch_size), 1)):
            batch = x[start : start + max(int(batch_size), 1)]
            pred = model(batch)
            for key in chunks:
                chunks[key].append(pred[key].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) if values else np.empty((0, PATH20_HORIZON)) for key, values in chunks.items()}


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        score = pd.to_numeric(group[score_column], errors="coerce")
        target = pd.to_numeric(group[target_column], errors="coerce")
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
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
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
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
            "stock": dataset.stock[idx].astype(str),
            "role": dataset.role[idx].astype(str),
            "model_family": str(family),
            "future_rank_20d": dataset.y_rank_20d[idx],
            "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
            "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
            "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
        }
    )
    for pos, horizon in enumerate(PATH20_CUMULATIVE_HORIZONS):
        frame[f"future_cum_excess_return_{horizon}d"] = y_cum[:, pos]
        frame[f"pred_cum_mu_{horizon}d"] = mu[:, :horizon].sum(axis=1)
        frame[f"pred_aux_cum_{horizon}d"] = aux[:, pos]
    for step in range(1, PATH20_HORIZON + 1):
        offset = step - 1
        frame[f"future_excess_return_{step}d"] = y_daily[:, offset]
        frame[f"target_excess_{step}d"] = y_daily[:, offset]
        frame[f"pred_mu_{step}d"] = mu[:, offset]
        frame[f"pred_q10_{step}d"] = q10[:, offset]
        frame[f"pred_q50_{step}d"] = q50[:, offset]
        frame[f"pred_q90_{step}d"] = q90[:, offset]
    return frame


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
    pred20 = pd.to_numeric(frame["pred_cum_mu_20d"], errors="coerce")
    target20 = pd.to_numeric(frame["future_cum_excess_return_20d"], errors="coerce")
    valid20 = pred20.notna() & target20.notna()
    return {
        "status": "completed",
        "row_count": int(len(frame)),
        "date_count": int(frame["date"].nunique()),
        "rank_ic_20d": _rank_ic_by_date(frame, "pred_cum_mu_20d", "future_cum_excess_return_20d"),
        "top_bottom_spread_20d": _top_bottom_spread_by_date(
            frame,
            "pred_cum_mu_20d",
            "future_cum_excess_return_20d",
        ),
        "q10_coverage_mean": float(np.mean(q10_coverages)) if q10_coverages else 0.0,
        "q90_coverage_mean": float(np.mean(q90_coverages)) if q90_coverages else 0.0,
        "direction_accuracy_20d": float((np.sign(pred20.loc[valid20]) == np.sign(target20.loc[valid20])).mean())
        if bool(valid20.any())
        else 0.0,
    }


def forecast_evidence_verdict(
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    coverage_range: tuple[float, float] = (0.65, 0.95),
) -> str:
    if validation_metrics.get("status") != "completed":
        return "insufficient_or_incomplete"
    rank_ic = float(validation_metrics.get("rank_ic_20d", 0.0) or 0.0)
    spread = float(validation_metrics.get("top_bottom_spread_20d", 0.0) or 0.0)
    q10 = float(validation_metrics.get("q10_coverage_mean", 0.0) or 0.0)
    q90 = float(validation_metrics.get("q90_coverage_mean", 0.0) or 0.0)
    low, high = coverage_range
    validation_promising = rank_ic > 0.0 and spread > 0.0 and low <= q10 <= high and low <= q90 <= high
    if not validation_promising:
        return "forecast_failed"
    if not test_metrics or test_metrics.get("status") != "completed":
        return "forecast_promising"
    test_rank_ic = float(test_metrics.get("rank_ic_20d", 0.0) or 0.0)
    test_spread = float(test_metrics.get("top_bottom_spread_20d", 0.0) or 0.0)
    if test_rank_ic > 0.0 and test_spread > 0.0:
        return "forecast_test_confirmed"
    return "forecast_promising"


def _select_model_family(model_summaries: dict[str, dict[str, Any]]) -> str:
    def score(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float]:
        _, summary = item
        validation = dict(summary.get("validation_metrics", {}))
        verdict = forecast_evidence_verdict(validation_metrics=validation, test_metrics=None)
        promising = 1 if verdict == "forecast_promising" else 0
        return (
            promising,
            float(validation.get("rank_ic_20d", 0.0) or 0.0),
            float(validation.get("top_bottom_spread_20d", 0.0) or 0.0),
        )

    if not model_summaries:
        return ""
    return max(model_summaries.items(), key=score)[0]


def train_forecast_models(
    dataset: ForecastSequenceDataset,
    *,
    study_root: Path,
    model_families: tuple[str, ...] | list[str] = FORECAST_MODEL_FAMILIES,
    epochs: int = 2,
    batch_size: int = 512,
    lr: float = 3.0e-4,
    hidden_dim: int = 96,
    dropout: float = 0.10,
    target_scale: float = 100.0,
    seed: int = 7,
) -> dict[str, Any]:
    study_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    families = tuple(str(item).strip() for item in model_families if str(item).strip())
    invalid = sorted(set(families) - set(FORECAST_MODEL_FAMILIES))
    if invalid:
        raise ValueError(f"Unsupported forecast model families: {', '.join(invalid)}")
    if dataset.x.shape[0] == 0:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "empty_forecast_dataset",
            "models": {},
            "selected_model_family": "",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / "forecast_training_summary.json", _json_ready(summary))
        return summary

    x = torch.tensor(dataset.x, dtype=torch.float32)
    y_daily = torch.tensor(dataset.y_daily_excess * float(target_scale), dtype=torch.float32)
    y_cum = torch.tensor(dataset.y_cum_excess * float(target_scale), dtype=torch.float32)
    y_risk_np = np.stack(
        [dataset.y_max_drawdown_20d, dataset.y_worst_1d_20d, dataset.y_upside_20d],
        axis=1,
    )
    y_risk = torch.tensor(y_risk_np * float(target_scale), dtype=torch.float32)
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
            "selected_model_family": "",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / "forecast_training_summary.json", _json_ready(summary))
        return summary

    model_summaries: dict[str, dict[str, Any]] = {}
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    batch_size = max(int(batch_size), 1)
    train_tensor_indices = torch.tensor(train_indices, dtype=torch.long)

    for family in families:
        model = make_forecast_model(
            family,
            input_dim=int(dataset.x.shape[-1]),
            hidden_dim=int(hidden_dim),
            horizon=PATH20_HORIZON,
            dropout=float(dropout),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
        losses: list[float] = []
        for _ in range(max(int(epochs), 1)):
            model.train()
            perm = train_tensor_indices[torch.randperm(train_tensor_indices.numel())]
            for start in range(0, perm.numel(), batch_size):
                batch_idx = perm[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = model(x[batch_idx])
                loss = _forecast_loss(pred, y_daily[batch_idx], y_cum[batch_idx], y_risk[batch_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        predictions = _predict_all(model, x, batch_size=batch_size)
        all_predictions[family] = predictions
        validation_frame = _prediction_frame(
            dataset,
            role="validation",
            predictions=predictions,
            family=family,
            target_scale=target_scale,
        )
        test_frame = _prediction_frame(
            dataset,
            role="test",
            predictions=predictions,
            family=family,
            target_scale=target_scale,
        )
        checkpoint_path = study_root / f"forecast_model_{family}.pt"
        torch.save(
            {
                "model_family": family,
                "state_dict": model.state_dict(),
                "feature_columns": list(dataset.feature_columns),
                "normalization": dataset.normalization_manifest,
                "target_scale": float(target_scale),
                "lookback_days": int(dataset.x.shape[1]),
                "horizon": PATH20_HORIZON,
            },
            checkpoint_path,
        )
        model_summaries[family] = {
            "status": "completed",
            "model_family": family,
            "epochs": int(max(int(epochs), 1)),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "feature_count": int(dataset.x.shape[-1]),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "final_train_loss": float(losses[-1]) if losses else 0.0,
            "validation_metrics": forecast_prediction_metrics(validation_frame),
            "test_metrics": forecast_prediction_metrics(test_frame),
            "checkpoint_pt": str(checkpoint_path.resolve()),
        }

    selected_family = _select_model_family(model_summaries)
    selected_predictions = all_predictions[selected_family] if selected_family else {}
    validation_predictions = (
        _prediction_frame(
            dataset,
            role="validation",
            predictions=selected_predictions,
            family=selected_family,
            target_scale=target_scale,
        )
        if selected_family
        else pd.DataFrame()
    )
    test_predictions = (
        _prediction_frame(
            dataset,
            role="test",
            predictions=selected_predictions,
            family=selected_family,
            target_scale=target_scale,
        )
        if selected_family
        else pd.DataFrame()
    )
    validation_csv = _write_frame(study_root / "forecast_predictions_validation.csv", validation_predictions)
    test_csv = _write_frame(study_root / "forecast_predictions_test.csv", test_predictions)
    selected_summary = model_summaries.get(selected_family, {})
    validation_metrics = dict(selected_summary.get("validation_metrics", {}))
    test_metrics = dict(selected_summary.get("test_metrics", {}))
    verdict = forecast_evidence_verdict(validation_metrics=validation_metrics, test_metrics=test_metrics)
    summary = {
        "status": "completed",
        "stage": "forecast_train",
        "target_scale": float(target_scale),
        "models": model_summaries,
        "selected_model_family": selected_family,
        "selection_rule": "validation_positive_rank_ic_and_top_bottom_spread_then_max_rank_ic",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_interpretable": verdict in {"forecast_test_confirmed", "forecast_promising"}
        and validation_metrics.get("status") == "completed"
        and float(validation_metrics.get("rank_ic_20d", 0.0) or 0.0) > 0.0
        and float(validation_metrics.get("top_bottom_spread_20d", 0.0) or 0.0) > 0.0,
        "evidence_verdict": verdict,
        "forecast_predictions_validation_csv": validation_csv,
        "forecast_predictions_test_csv": test_csv,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    summary = _json_ready(summary)
    write_json(study_root / "forecast_training_summary.json", summary)
    return summary
