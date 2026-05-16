from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.pipeline_utils import select_feature_columns
from daily_research.continuous_policy.runtime import now_iso, safe_print_json, write_json
from daily_research.continuous_policy.state_builder import DEFAULT_ALPHA_PRIOR_SOURCE, prepare_policy_inputs
from daily_research.data_lake.policy_input_loader import DEFAULT_POLICY_INPUT_LAKE_DATASET_ID
from daily_research.path_policy import (
    ALPHA_PATH20_POLICY_PROFILE,
    ALPHA_PATH20_POLICY_VERSION,
    ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
    ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
)
from daily_research.path_policy.adapter import build_path_policy_frame
from daily_research.path_policy.labels import PATH20_HORIZON, build_path20_dataset_frame
from daily_research.path_policy.models import (
    LinearPath20Forecaster,
    NeuralTargetWeightPolicy,
    PathPolicyModelConfig,
    pairwise_rank_loss,
    path_feature_tensor_from_prediction,
    pinball_loss,
    portfolio_utility_loss,
)
from daily_research.path_policy.oracle import run_oracle_path20_rollout
from daily_research.path_policy.rl_dataset import (
    DEFAULT_FULL_YEAR_WINDOWS,
    DEFAULT_RL_REWARD_PROFILE,
    Path20TrajectoryDataset,
    build_path20_sequence_trajectory_dataset,
    build_sequence_tensors,
    full_year_window,
)
from daily_research.path_policy.rl_episode import (
    EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS,
    Path20MarketEpisode,
    build_path20_market_episode,
    episode_policy_rollout_loss,
    episode_to_arrays,
    fit_episode_normalization,
    predict_episode_targets,
)
from daily_research.path_policy.rl_models import (
    DecisionTransformerTargetWeightPolicy,
    SequencePolicyConfig,
    SequenceTargetWeightPolicy,
    sequence_policy_utility_loss,
)
from daily_research.path_policy.rl_replay import run_sequence_policy_replay


PATH_POLICY_OUTPUT_ROOT = Path("daily_research/output/path_policy")
PATH_POLICY_STUDIES_ROOT = PATH_POLICY_OUTPUT_ROOT / "studies"
PATH_POLICY_DATASETS_ROOT = PATH_POLICY_OUTPUT_ROOT / "datasets"
PATH_POLICY_SEQUENCE_DATASETS_ROOT = PATH_POLICY_OUTPUT_ROOT / "sequence_datasets"
PATH_POLICY_EPISODE_DATASETS_ROOT = PATH_POLICY_OUTPUT_ROOT / "episode_datasets"
LEGACY_NEURAL_STAGES = frozenset({"dataset-smoke", "oracle-smoke", "tiny-smoke"})
SEQUENCE_RL_SMOKE_STAGES = frozenset({"rl-dataset-smoke", "rl-train-smoke", "rl-replay-smoke", "rl-multiyear-smoke"})
SEQUENCE_RL_EPISODE_STAGES = frozenset({"rl-episode-dataset", "rl-train-episode", "rl-replay-episode", "rl-walkforward-study"})
SEQUENCE_RL_STAGES = SEQUENCE_RL_SMOKE_STAGES | SEQUENCE_RL_EPISODE_STAGES
WALKFORWARD_TRAIN_YEARS = (2019, 2020)
WALKFORWARD_VALIDATION_YEARS = (2022,)
WALKFORWARD_TEST_YEARS = (2024,)

NON_PATH20_FEATURE_COLUMNS = {
    "date",
    "stock",
    "in_universe",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}__{digest}"


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _series_frame(series: pd.Series, *, index_name: str, value_name: str) -> pd.DataFrame:
    frame = series.rename(value_name).reset_index()
    if len(frame.columns) >= 2:
        frame = frame.rename(columns={frame.columns[0]: index_name, frame.columns[1]: value_name})
    return frame


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _extend_end_date_for_labels(end_date: str, *, days: int = 60) -> str:
    if not str(end_date or "").strip():
        return ""
    return (pd.Timestamp(end_date).normalize() + pd.Timedelta(days=int(days))).strftime("%Y%m%d")


def _finite_mean(values: list[float]) -> float:
    finite = [float(item) for item in values if np.isfinite(float(item))]
    return float(np.mean(finite)) if finite else 0.0


def _finite_sum(values: list[float]) -> float:
    finite = [float(item) for item in values if np.isfinite(float(item))]
    return float(np.sum(finite)) if finite else 0.0


def _safe_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _is_loose_lake_dataset_id(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"latest", "default"} or normalized.startswith("latest_") or normalized.startswith("latest-")


def _multiyear_aggregate(yearly: dict[str, Any]) -> dict[str, Any]:
    completed_years: list[str] = []
    incomplete_years: dict[str, dict[str, Any]] = {}
    total_returns: list[float] = []
    annual_returns: list[float] = []
    sharpes: list[float] = []
    max_drawdowns: list[float] = []
    monthly_win_rates: list[float] = []
    turnovers: list[float] = []
    gross_exposures: list[float] = []
    projection_distances: list[float] = []
    for year, result in yearly.items():
        dataset_summary = dict(result.get("dataset_summary", {}) or {})
        replay_summary = dict(result.get("replay_summary", {}) or {})
        metrics = dict(replay_summary.get("metrics", {}) or {})
        dataset_status = str(dataset_summary.get("status", "") or "")
        replay_status = str(replay_summary.get("status", "") or "")
        if dataset_status == "completed" and replay_status == "completed":
            completed_years.append(str(year))
            for values, key in (
                (total_returns, "total_return"),
                (annual_returns, "annual_return"),
                (sharpes, "sharpe"),
                (max_drawdowns, "max_drawdown"),
                (monthly_win_rates, "monthly_win_rate"),
                (turnovers, "avg_turnover"),
                (gross_exposures, "avg_projected_gross_exposure"),
                (projection_distances, "avg_projection_l1_distance"),
            ):
                metric = _safe_metric(metrics, key)
                if metric is not None:
                    values.append(metric)
            continue
        reasons: list[str] = []
        if dataset_status != "completed":
            reasons.append(str(dataset_summary.get("incomplete_reason") or f"dataset_status={dataset_status or 'unknown'}"))
        if replay_status != "completed":
            reasons.append(str(replay_summary.get("reason") or f"replay_status={replay_status or 'unknown'}"))
        incomplete_years[str(year)] = {
            "dataset_status": dataset_status or "unknown",
            "replay_status": replay_status or "unknown",
            "reason": "; ".join(reason for reason in reasons if reason) or "not_completed",
            "trading_day_count": int(dataset_summary.get("trading_day_count", 0) or 0),
        }
    return {
        "completed_year_count": int(len(completed_years)),
        "completed_years": completed_years,
        "incomplete_year_count": int(len(incomplete_years)),
        "incomplete_years": incomplete_years,
        "total_return_sum": _finite_sum(total_returns),
        "mean_total_return": _finite_mean(total_returns),
        "mean_annual_return": _finite_mean(annual_returns),
        "mean_sharpe": _finite_mean(sharpes),
        "worst_max_drawdown": float(np.min(max_drawdowns)) if max_drawdowns else 0.0,
        "mean_monthly_win_rate": _finite_mean(monthly_win_rates),
        "mean_avg_turnover": _finite_mean(turnovers),
        "mean_avg_projected_gross_exposure": _finite_mean(gross_exposures),
        "mean_avg_projection_l1_distance": _finite_mean(projection_distances),
    }


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 5:
            continue
        score = pd.to_numeric(group[score_column], errors="coerce")
        target = pd.to_numeric(group[target_column], errors="coerce")
        valid = score.notna() & target.notna()
        if int(valid.sum()) < 5:
            continue
        corr = score.loc[valid].corr(target.loc[valid], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return _finite_mean(values)


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str, frac: float = 0.20) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        work = group[[score_column, target_column]].copy()
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
        work = work.dropna()
        if len(work) < 10:
            continue
        k = max(int(len(work) * float(frac)), 1)
        top = work.nlargest(k, score_column)[target_column].mean()
        bottom = work.nsmallest(k, score_column)[target_column].mean()
        if pd.notna(top) and pd.notna(bottom):
            values.append(float(top - bottom))
    return _finite_mean(values)


def _build_dataset_artifact(
    *,
    prepared: Any,
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset, manifest = build_path20_dataset_frame(
        prepared,
        start_date=args.start_date,
        end_date=args.end_date,
        execution_mode=args.execution_mode,
        include_state_features=True,
    )
    dataset_id = _stable_id(
        "alpha_path20_dataset",
        {
            "version": ALPHA_PATH20_POLICY_VERSION,
            "tag": tag,
            "data_source": args.data_source,
            "lake_dataset_id": args.lake_dataset_id,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "execution_mode": args.execution_mode,
            "row_count": int(len(dataset)),
        },
    )
    dataset_root = PATH_POLICY_DATASETS_ROOT / dataset_id
    dataset_path = dataset_root / "path20_dataset.csv"
    manifest_path = dataset_root / "dataset_manifest.json"
    _write_frame(dataset_path, dataset)
    manifest = {
        **manifest,
        "dataset_id": dataset_id,
        "policy_version": ALPHA_PATH20_POLICY_VERSION,
        "policy_profile": ALPHA_PATH20_POLICY_PROFILE,
        "data_source": args.data_source,
        "lake_dataset_id": args.lake_dataset_id,
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "prepared_start_date": prepared.start_date,
        "prepared_end_date": prepared.end_date,
        "dataset_csv": str(dataset_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "loose_latest_allowed": False,
        "stage": "path20_dataset",
    }
    write_json(manifest_path, manifest)
    write_json(study_root / "dataset_manifest.json", manifest)
    return {"dataset": dataset, "manifest": manifest}


def _path20_target_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{step}d" for step in range(1, PATH20_HORIZON + 1)]


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    label_prefixes = (
        "future_",
        "path_mu_",
        "path_q10_",
        "path_q50_",
        "path_q90_",
    )
    candidates = [
        column
        for column in select_feature_columns(dataset)
        if column not in NON_PATH20_FEATURE_COLUMNS and not str(column).startswith(label_prefixes)
    ]
    return candidates[:96]


def _run_forecaster_smoke(dataset: pd.DataFrame, *, study_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if dataset.empty:
        return {"status": "skipped", "reason": "empty_dataset"}
    feature_columns = _feature_columns(dataset)
    target_columns = _path20_target_columns("future_excess_return")
    missing_targets = [column for column in target_columns if column not in dataset.columns]
    if not feature_columns or missing_targets:
        return {
            "status": "failed",
            "reason": "missing_features_or_targets",
            "feature_count": int(len(feature_columns)),
            "missing_targets": missing_targets,
        }
    work = dataset[["date", "stock", *feature_columns, *target_columns, "future_cum_excess_return_20d"]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=target_columns)
    if work.empty:
        return {"status": "skipped", "reason": "no_finite_targets"}
    if len(work) > int(args.smoke_max_rows):
        work = work.sort_values(["date", "stock"]).head(int(args.smoke_max_rows)).copy()
    x_frame = work[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_mean = x_frame.mean(axis=0)
    x_std = x_frame.std(axis=0, ddof=0).replace(0.0, 1.0)
    x = torch.tensor(((x_frame - x_mean) / x_std).to_numpy(dtype=np.float32), dtype=torch.float32)
    y_mu = torch.tensor(work[target_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    dates = pd.to_datetime(work["date"])
    unique_dates = sorted(dates.dropna().unique())
    split_idx = max(int(len(unique_dates) * 0.80), 1)
    train_dates = set(unique_dates[:split_idx])
    train_mask_np = dates.map(lambda value: value in train_dates).to_numpy(dtype=bool)
    if int(train_mask_np.sum()) < 8 or int((~train_mask_np).sum()) < 4:
        train_mask_np = np.arange(len(work)) < max(int(len(work) * 0.80), 1)
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    valid_mask = ~train_mask
    model = LinearPath20Forecaster(input_dim=len(feature_columns), horizon=PATH20_HORIZON)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.smoke_lr), weight_decay=1.0e-4)
    losses: list[float] = []
    for _ in range(max(int(args.smoke_epochs), 1)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x[train_mask])
        loss = F.huber_loss(pred["mu"], y_mu[train_mask])
        loss = loss + 0.25 * pinball_loss(pred["q10"], y_mu[train_mask], 0.10)
        loss = loss + 0.25 * pinball_loss(pred["q50"], y_mu[train_mask], 0.50)
        loss = loss + 0.25 * pinball_loss(pred["q90"], y_mu[train_mask], 0.90)
        cum20_score = pred["mu"].sum(dim=1)
        cum20_target = y_mu[train_mask].sum(dim=1)
        loss = loss + 0.02 * pairwise_rank_loss(cum20_score, cum20_target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        pred_all = model(x)
    prediction_frame = work[["date", "stock", "future_cum_excess_return_20d"]].copy()
    mu_np = pred_all["mu"].detach().cpu().numpy()
    q10_np = pred_all["q10"].detach().cpu().numpy()
    q90_np = pred_all["q90"].detach().cpu().numpy()
    for horizon in (5, 10, 20):
        prediction_frame[f"pred_cum_mu_{horizon}d"] = mu_np[:, :horizon].sum(axis=1)
    for step in range(1, PATH20_HORIZON + 1):
        prediction_frame[f"pred_q10_{step}d"] = q10_np[:, step - 1]
        prediction_frame[f"pred_q90_{step}d"] = q90_np[:, step - 1]
        prediction_frame[f"target_excess_{step}d"] = y_mu[:, step - 1].detach().cpu().numpy()
    metrics: dict[str, Any] = {
        "status": "completed",
        "model_family": "linear_path20_smoke",
        "epochs": int(max(int(args.smoke_epochs), 1)),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(train_mask.sum().item()),
        "validation_rows": int(valid_mask.sum().item()),
        "final_train_loss": float(losses[-1]) if losses else 0.0,
        "rank_ic_20d": _rank_ic_by_date(prediction_frame, "pred_cum_mu_20d", "future_cum_excess_return_20d"),
        "top_bottom_spread_20d": _top_bottom_spread_by_date(
            prediction_frame,
            "pred_cum_mu_20d",
            "future_cum_excess_return_20d",
        ),
        "q10_coverage_mean": float(
            np.mean(
                [
                    (
                        pd.to_numeric(prediction_frame[f"target_excess_{step}d"], errors="coerce")
                        >= pd.to_numeric(prediction_frame[f"pred_q10_{step}d"], errors="coerce")
                    ).mean()
                    for step in range(1, PATH20_HORIZON + 1)
                ]
            )
        ),
        "q90_coverage_mean": float(
            np.mean(
                [
                    (
                        pd.to_numeric(prediction_frame[f"target_excess_{step}d"], errors="coerce")
                        <= pd.to_numeric(prediction_frame[f"pred_q90_{step}d"], errors="coerce")
                    ).mean()
                    for step in range(1, PATH20_HORIZON + 1)
                ]
            )
        ),
        "direction_accuracy_20d": float(
            (
                np.sign(prediction_frame["pred_cum_mu_20d"].to_numpy(dtype=float))
                == np.sign(prediction_frame["future_cum_excess_return_20d"].to_numpy(dtype=float))
            ).mean()
        ),
    }
    prediction_path = study_root / "predicted_path_quality_smoke.csv"
    _write_frame(prediction_path, prediction_frame)
    metrics["prediction_csv"] = str(prediction_path.resolve())
    write_json(study_root / "predicted_path_quality_summary.json", metrics)
    return metrics


def _run_allocator_smoke(dataset: pd.DataFrame, *, study_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if dataset.empty:
        return {"status": "skipped", "reason": "empty_dataset"}
    feature_columns = _feature_columns(dataset)[:32]
    path_columns = [
        *[f"path_mu_{step}d" for step in range(1, PATH20_HORIZON + 1)],
        *[f"path_q10_{step}d" for step in range(1, PATH20_HORIZON + 1)],
        *[f"path_q50_{step}d" for step in range(1, PATH20_HORIZON + 1)],
        *[f"path_q90_{step}d" for step in range(1, PATH20_HORIZON + 1)],
        "future_cum_excess_return_5d",
        "future_cum_excess_return_10d",
        "future_cum_excess_return_20d",
        "future_path_max_drawdown_20d",
        "future_path_worst_1d_20d",
        "future_path_upside_capture_20d",
    ]
    missing = [column for column in [*feature_columns, *path_columns, "future_cum_excess_return_20d"] if column not in dataset.columns]
    if not feature_columns or missing:
        return {"status": "failed", "reason": "missing_allocator_columns", "missing_columns": missing}
    first_date = str(sorted(dataset["date"].astype(str).unique())[0])
    daily = dataset.loc[dataset["date"].astype(str) == first_date].copy()
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_cum_excess_return_20d"])
    if len(daily) > int(args.allocator_max_names):
        daily = daily.sort_values("future_cum_excess_return_20d", ascending=False).head(int(args.allocator_max_names)).copy()
    if len(daily) < 4:
        return {"status": "skipped", "reason": "insufficient_daily_rows", "date": first_date, "rows": int(len(daily))}
    stock_features = daily[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stock_features = (stock_features - stock_features.mean(axis=0)) / stock_features.std(axis=0, ddof=0).replace(0.0, 1.0)
    path_features = daily[path_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    future_return = torch.tensor(daily["future_cum_excess_return_20d"].to_numpy(dtype=np.float32), dtype=torch.float32)
    policy = NeuralTargetWeightPolicy(
        PathPolicyModelConfig(
            stock_feature_dim=len(feature_columns),
            path_feature_dim=len(path_columns),
            portfolio_feature_dim=8,
            hidden_dim=48,
            dropout=0.0,
            max_position_weight=float(args.max_position_weight),
        )
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.smoke_lr), weight_decay=1.0e-4)
    stock_tensor = torch.tensor(stock_features.to_numpy(dtype=np.float32), dtype=torch.float32)
    path_tensor = torch.tensor(path_features.to_numpy(dtype=np.float32), dtype=torch.float32)
    portfolio_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    tradable_mask = torch.ones(len(daily), dtype=torch.bool)
    losses: list[float] = []
    for _ in range(max(int(args.smoke_epochs), 1)):
        optimizer.zero_grad(set_to_none=True)
        out = policy(stock_tensor, path_tensor, portfolio_tensor, tradable_mask=tradable_mask)
        loss = portfolio_utility_loss(
            out["target_weight"],
            future_return,
            transaction_cost_rate=(args.transaction_cost_bps + args.slippage_bps) / 10_000.0,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        out = policy(stock_tensor, path_tensor, portfolio_tensor, tradable_mask=tradable_mask)
    raw_target = pd.Series(out["target_weight"].detach().cpu().numpy(), index=daily["stock"].astype(str), dtype=float)
    base_frame = daily[["date", "stock"]].copy()
    base_frame["current_weight"] = 0.0
    policy_frame, global_targets = build_path_policy_frame(
        base_frame,
        raw_target_weight=raw_target,
        current_weight=pd.Series(0.0, index=raw_target.index, dtype=float),
        tradable_mask=pd.Series(True, index=raw_target.index, dtype=bool),
        max_position_weight=float(args.max_position_weight),
        max_gross_exposure=float(args.max_gross_exposure),
        max_positions=int(args.max_positions),
        turnover_budget=float(args.turnover_budget),
        source_label="allocator_oracle_path_smoke",
    )
    realized_utility = float(
        policy_frame.set_index("stock")["portfolio_daily_target_weight"]
        .reindex(daily["stock"].astype(str))
        .fillna(0.0)
        .to_numpy(dtype=float)
        .dot(daily["future_cum_excess_return_20d"].to_numpy(dtype=float))
    )
    action_path = study_root / "allocator_smoke_action_panel.csv"
    _write_frame(action_path, policy_frame.reset_index(drop=True))
    summary = {
        "status": "completed",
        "date": first_date,
        "rows": int(len(daily)),
        "epochs": int(max(int(args.smoke_epochs), 1)),
        "final_train_loss": float(losses[-1]) if losses else 0.0,
        "target_weight_sum": float(policy_frame["portfolio_daily_target_weight"].sum()),
        "target_count": int((policy_frame["portfolio_daily_target_weight"] > 1.0e-8).sum()),
        "cash_weight": float(global_targets.get("path_policy_cash_weight", 0.0)),
        "oracle_path_realized_utility_proxy": realized_utility,
        "action_panel_csv": str(action_path.resolve()),
        "global_targets": global_targets,
    }
    write_json(study_root / "allocator_smoke_summary.json", summary)
    return summary


def _run_oracle_smoke(
    *,
    prepared: Any,
    study_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rollout = run_oracle_path20_rollout(
        prepared=prepared,
        start_date=args.start_date,
        end_date=args.end_date,
        execution_mode=args.execution_mode,
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
        sell_tax_bps=float(args.sell_tax_bps),
        max_position_weight=float(args.max_position_weight),
        max_gross_exposure=float(args.max_gross_exposure),
        max_positions=int(args.max_positions),
        turnover_budget=float(args.turnover_budget),
    )
    paths = {
        "action_panel_csv": _write_frame(study_root / "oracle_action_panel.csv", rollout["action_panel"]),
        "turnover_csv": _write_frame(study_root / "oracle_turnover.csv", rollout["turnover_frame"]),
        "position_history_csv": _write_frame(study_root / "oracle_position_history.csv", rollout["position_history"]),
        "returns_csv": _write_frame(study_root / "oracle_returns.csv", _series_frame(rollout["returns"], index_name="date", value_name="net_return")),
        "oracle_daily_csv": _write_frame(study_root / "oracle_daily.csv", rollout["oracle_daily"]),
    }
    summary = {
        "status": "completed",
        "stage": "oracle_path_upper_bound",
        "policy_version": ALPHA_PATH20_POLICY_VERSION,
        "execution_mode": args.execution_mode,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "date_count": int(len(rollout["dates"])),
        "return_count": int(len(rollout["returns"])),
        "metrics": rollout["metrics"],
        "artifacts": paths,
        "research_status": "research / shadow-only / oracle-path smoke evidence",
    }
    write_json(study_root / "oracle_path_upper_bound_summary.json", summary)
    return summary


def _execution_price_frame(prepared: Any, execution_mode: str) -> pd.DataFrame:
    mode = str(execution_mode or "next_open").strip().lower()
    if mode == "next_open":
        return prepared.open_.shift(-1)
    if mode == "close":
        return prepared.close
    raise ValueError(f"Unsupported execution_mode: {execution_mode!r}")


def _build_sequence_dataset_artifact(
    *,
    prepared: Any,
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
    year: int | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    resolved_start = str(start_date or args.start_date)
    resolved_end = str(end_date or args.end_date)
    trajectory = build_path20_sequence_trajectory_dataset(
        prepared,
        start_date=resolved_start,
        end_date=resolved_end,
        lake_dataset_id=str(args.lake_dataset_id or ""),
        year=year,
        sequence_length=int(args.sequence_length),
        execution_mode=args.execution_mode,
        reward_profile=args.reward_profile,
        max_feature_columns=int(args.rl_max_feature_columns),
        min_trading_days=int(args.min_year_trading_days) if year is not None else 2,
    )
    dataset_id = str(trajectory.manifest["dataset_id"])
    dataset_root = PATH_POLICY_SEQUENCE_DATASETS_ROOT / dataset_id
    dataset_path = dataset_root / "sequence_trajectory.csv"
    manifest_path = dataset_root / "dataset_manifest.json"
    long_frame = trajectory.to_long_frame()
    _write_frame(dataset_path, long_frame)
    manifest = {
        **trajectory.manifest,
        "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
        "data_source": args.data_source,
        "dataset_csv": str(dataset_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "row_count": int(len(long_frame)),
        "run_tag": tag,
    }
    write_json(manifest_path, manifest)
    write_json(study_root / "sequence_dataset_manifest.json", manifest)
    return {"trajectory": trajectory, "dataset_frame": long_frame, "manifest": manifest}


def _make_sequence_policy(args: argparse.Namespace, *, stock_feature_dim: int, portfolio_feature_dim: int):
    config = SequencePolicyConfig(
        stock_feature_dim=int(stock_feature_dim),
        portfolio_feature_dim=int(portfolio_feature_dim),
        hidden_dim=int(args.rl_hidden_dim),
        dropout=float(args.rl_dropout),
        max_position_weight=float(args.max_position_weight),
    )
    family = str(args.model_family or "sequence_gru").strip().lower()
    if family == "sequence_gru":
        return SequenceTargetWeightPolicy(config)
    if family == "decision_transformer":
        return DecisionTransformerTargetWeightPolicy(config)
    raise ValueError(f"Unsupported path20 sequence model family: {args.model_family!r}")


def _build_episode_dataset_artifact(
    *,
    prepared: Any,
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
    year: int | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    resolved_start = str(start_date or args.start_date)
    resolved_end = str(end_date or args.end_date)
    episode = build_path20_market_episode(
        prepared,
        start_date=resolved_start,
        end_date=resolved_end,
        lake_dataset_id=str(args.lake_dataset_id or ""),
        year=year,
        sequence_length=int(args.sequence_length),
        execution_mode=args.execution_mode,
        reward_profile=args.reward_profile,
        max_feature_columns=int(args.rl_max_feature_columns),
        min_trading_days=int(args.min_year_trading_days) if year is not None else 2,
    )
    dataset_id = str(episode.manifest["dataset_id"])
    dataset_root = PATH_POLICY_EPISODE_DATASETS_ROOT / dataset_id
    dataset_path = dataset_root / "market_episode.csv"
    manifest_path = dataset_root / "dataset_manifest.json"
    long_frame = episode.to_long_frame()
    _write_frame(dataset_path, long_frame)
    manifest = {
        **episode.manifest,
        "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
        "data_source": args.data_source,
        "dataset_csv": str(dataset_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "row_count": int(len(long_frame)),
        "run_tag": tag,
    }
    write_json(manifest_path, manifest)
    write_json(study_root / "episode_dataset_manifest.json", manifest)
    return {"episode": episode, "dataset_frame": long_frame, "manifest": manifest}


def _episode_train_model(
    *,
    episodes: list[Path20MarketEpisode],
    validation_episodes: list[Path20MarketEpisode],
    study_root: Path,
    args: argparse.Namespace,
    train_years: list[int] | None = None,
    validation_years: list[int] | None = None,
) -> dict[str, Any]:
    import torch

    completed_train = [episode for episode in episodes if str(episode.manifest.get("status", "")) == "completed"]
    if not completed_train:
        return {"summary": {"status": "skipped", "reason": "no_completed_train_episodes"}}
    normalization = fit_episode_normalization(completed_train)
    base_feature_count = int(len(completed_train[0].feature_columns))
    stock_feature_dim = base_feature_count + len(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS)
    model = _make_sequence_policy(args, stock_feature_dim=stock_feature_dim, portfolio_feature_dim=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.smoke_lr), weight_decay=1.0e-4)
    losses: list[float] = []
    projection_summaries: list[dict[str, float]] = []
    for _ in range(max(int(args.smoke_epochs), 1)):
        epoch_losses: list[float] = []
        for episode in completed_train:
            optimizer.zero_grad(set_to_none=True)
            result = episode_policy_rollout_loss(
                model,
                episode,
                sequence_length=int(args.sequence_length),
                normalization=normalization,
                model_family=str(args.model_family),
                transaction_cost_bps=float(args.transaction_cost_bps),
                slippage_bps=float(args.slippage_bps),
                sell_tax_bps=float(args.sell_tax_bps),
                max_position_weight=float(args.max_position_weight),
                max_gross_exposure=float(args.max_gross_exposure),
                max_positions=int(args.max_positions),
                turnover_budget=float(args.turnover_budget),
            )
            if result.get("status") != "completed":
                continue
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            projection_summaries.append(dict(result.get("diagnostics", {}) or {}))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    validation_rollout: dict[str, Any] = {}
    completed_validation = [episode for episode in validation_episodes if str(episode.manifest.get("status", "")) == "completed"]
    if completed_validation:
        with torch.no_grad():
            validation_result = episode_policy_rollout_loss(
                model,
                completed_validation[0],
                sequence_length=int(args.sequence_length),
                normalization=normalization,
                model_family=str(args.model_family),
                transaction_cost_bps=float(args.transaction_cost_bps),
                slippage_bps=float(args.slippage_bps),
                sell_tax_bps=float(args.sell_tax_bps),
                max_position_weight=float(args.max_position_weight),
                max_gross_exposure=float(args.max_gross_exposure),
                max_positions=int(args.max_positions),
                turnover_budget=float(args.turnover_budget),
            )
        validation_rollout = {
            "status": str(validation_result.get("status", "unknown")),
            "loss": float(validation_result["loss"].detach().cpu()) if "loss" in validation_result else 0.0,
            "diagnostics": validation_result.get("diagnostics", {}),
            "used_projected_weights_for_loss": bool(validation_result.get("used_projected_weights_for_loss", False)),
        }
    artifact_path = study_root / "sequence_policy_episode.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": str(args.model_family),
            "state_dict": model.state_dict(),
            "feature_columns": completed_train[0].feature_columns,
            "dynamic_stock_feature_columns": list(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS),
            "sequence_length": int(args.sequence_length),
            "reward_profile": str(args.reward_profile),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "normalization": normalization,
            "shadow_only": True,
        },
        artifact_path,
    )
    projection_summary = _summarize_projection_dicts(projection_summaries)
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": str(args.model_family),
        "reward_profile": str(args.reward_profile),
        "train_years": list(train_years or []),
        "validation_years": list(validation_years or []),
        "epoch_count": int(max(int(args.smoke_epochs), 1)),
        "episode_count": int(len(completed_train)),
        "train_loss_curve": losses,
        "final_train_loss": float(losses[-1]) if losses else 0.0,
        "validation_replay_metrics": validation_rollout,
        "projection_diagnostics_summary": projection_summary,
        "normalization": normalization,
        "artifact_pt": str(artifact_path.resolve()),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
    }
    write_json(study_root / "sequence_train_episode_summary.json", _json_ready(summary))
    return {"summary": summary, "model": model, "normalization": normalization}


def _summarize_projection_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    return {
        str(column): float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())
        for column in frame.columns
    }


def _run_episode_replay(
    *,
    prepared: Any,
    episode: Path20MarketEpisode,
    model: Any,
    normalization: dict[str, Any],
    study_root: Path,
    args: argparse.Namespace,
    prefix: str = "episode",
) -> dict[str, Any]:
    target_by_date, training_projection_summary = predict_episode_targets(
        model,
        episode,
        sequence_length=int(args.sequence_length),
        normalization=normalization,
        model_family=str(args.model_family),
        max_position_weight=float(args.max_position_weight),
        max_gross_exposure=float(args.max_gross_exposure),
        max_positions=int(args.max_positions),
        turnover_budget=float(args.turnover_budget),
    )
    rollout = run_sequence_policy_replay(
        trajectory=episode.to_trajectory(),
        target_weight_by_date=target_by_date,
        execution_price_frame=_execution_price_frame(prepared, args.execution_mode),
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
        sell_tax_bps=float(args.sell_tax_bps),
        max_position_weight=float(args.max_position_weight),
        max_gross_exposure=float(args.max_gross_exposure),
        max_positions=int(args.max_positions),
        turnover_budget=float(args.turnover_budget),
    )
    paths = {
        "action_panel_csv": _write_frame(study_root / f"{prefix}_rl_action_panel.csv", rollout["action_panel"]),
        "turnover_csv": _write_frame(study_root / f"{prefix}_rl_turnover.csv", rollout["turnover_frame"]),
        "position_history_csv": _write_frame(study_root / f"{prefix}_rl_position_history.csv", rollout["position_history"]),
        "projection_diagnostics_csv": _write_frame(study_root / f"{prefix}_rl_projection_diagnostics.csv", rollout["projection_diagnostics"]),
        "returns_csv": _write_frame(study_root / f"{prefix}_rl_returns.csv", _series_frame(rollout["returns"], index_name="date", value_name="net_return")),
    }
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "stage": "rl_replay_episode",
        "model_family": str(args.model_family),
        "date_count": int(len(rollout["dates"])),
        "return_count": int(len(rollout["returns"])),
        "metrics": rollout["metrics"],
        "training_projection_diagnostics_summary": training_projection_summary,
        "artifacts": paths,
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / f"{prefix}_rl_replay_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_sequence_train_smoke(
    trajectory: Path20TrajectoryDataset,
    *,
    study_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    tensors = build_sequence_tensors(trajectory, sequence_length=int(args.sequence_length))
    if int(tensors["state"].shape[0]) <= 0:
        return {"status": "skipped", "reason": "empty_sequence_tensor"}
    state = torch.tensor(tensors["state"], dtype=torch.float32)
    portfolio = torch.tensor(tensors["portfolio"], dtype=torch.float32)
    mask = torch.tensor(tensors["mask"], dtype=torch.bool)
    current_weight = torch.tensor(tensors["current_weight"], dtype=torch.float32)
    future_return = torch.tensor(tensors["future_return"], dtype=torch.float32)
    model = _make_sequence_policy(
        args,
        stock_feature_dim=int(state.shape[-1]),
        portfolio_feature_dim=int(portfolio.shape[-1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.smoke_lr), weight_decay=1.0e-4)
    losses: list[float] = []
    max_rows = min(int(args.smoke_max_rows), int(state.shape[0]))
    train_slice = slice(0, max_rows)
    for _ in range(max(int(args.smoke_epochs), 1)):
        optimizer.zero_grad(set_to_none=True)
        if str(args.model_family).strip().lower() == "decision_transformer":
            pred = model(state[train_slice], portfolio[train_slice], tradable_mask=mask[train_slice])
        else:
            pred = model(state[train_slice], portfolio[train_slice], tradable_mask=mask[train_slice])
        loss = sequence_policy_utility_loss(
            pred["raw_target_weight"],
            future_return[train_slice],
            current_weight[train_slice],
            transaction_cost_rate=(float(args.transaction_cost_bps) + float(args.slippage_bps)) / 10_000.0,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    artifact_path = study_root / "sequence_policy_smoke.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": str(args.model_family),
            "state_dict": model.state_dict(),
            "feature_columns": trajectory.feature_columns,
            "sequence_length": int(args.sequence_length),
            "reward_profile": str(args.reward_profile),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "shadow_only": True,
        },
        artifact_path,
    )
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": str(args.model_family),
        "epochs": int(max(int(args.smoke_epochs), 1)),
        "train_sequences": int(max_rows),
        "stock_feature_dim": int(state.shape[-1]),
        "portfolio_feature_dim": int(portfolio.shape[-1]),
        "final_train_loss": float(losses[-1]) if losses else 0.0,
        "artifact_pt": str(artifact_path.resolve()),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
    }
    write_json(study_root / "sequence_train_smoke_summary.json", summary)
    return {"summary": summary, "model": model, "tensors": tensors}


def _predict_sequence_targets(
    *,
    model: Any,
    tensors: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, pd.Series]:
    import torch

    if int(tensors["state"].shape[0]) <= 0:
        return {}
    state = torch.tensor(tensors["state"], dtype=torch.float32)
    portfolio = torch.tensor(tensors["portfolio"], dtype=torch.float32)
    mask = torch.tensor(tensors["mask"], dtype=torch.bool)
    with torch.no_grad():
        if str(args.model_family).strip().lower() == "decision_transformer":
            pred = model(state, portfolio, tradable_mask=mask)
        else:
            pred = model(state, portfolio, tradable_mask=mask)
    raw = pred["raw_target_weight"].detach().cpu().numpy()
    dates = [str(item) for item in tensors["dates"].tolist()]
    stocks = [str(item) for item in tensors["stocks"].tolist()]
    return {date: pd.Series(raw[idx], index=stocks, dtype=float) for idx, date in enumerate(dates)}


def _run_sequence_replay_smoke(
    *,
    prepared: Any,
    trajectory: Path20TrajectoryDataset,
    model: Any,
    tensors: dict[str, Any],
    study_root: Path,
    args: argparse.Namespace,
    prefix: str = "sequence",
) -> dict[str, Any]:
    target_by_date = _predict_sequence_targets(model=model, tensors=tensors, args=args)
    rollout = run_sequence_policy_replay(
        trajectory=trajectory,
        target_weight_by_date=target_by_date,
        execution_price_frame=_execution_price_frame(prepared, args.execution_mode),
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
        sell_tax_bps=float(args.sell_tax_bps),
        max_position_weight=float(args.max_position_weight),
        max_gross_exposure=float(args.max_gross_exposure),
        max_positions=int(args.max_positions),
        turnover_budget=float(args.turnover_budget),
    )
    paths = {
        "action_panel_csv": _write_frame(study_root / f"{prefix}_rl_action_panel.csv", rollout["action_panel"]),
        "turnover_csv": _write_frame(study_root / f"{prefix}_rl_turnover.csv", rollout["turnover_frame"]),
        "position_history_csv": _write_frame(study_root / f"{prefix}_rl_position_history.csv", rollout["position_history"]),
        "projection_diagnostics_csv": _write_frame(study_root / f"{prefix}_rl_projection_diagnostics.csv", rollout["projection_diagnostics"]),
        "returns_csv": _write_frame(study_root / f"{prefix}_rl_returns.csv", _series_frame(rollout["returns"], index_name="date", value_name="net_return")),
    }
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "stage": "rl_replay",
        "model_family": str(args.model_family),
        "date_count": int(len(rollout["dates"])),
        "return_count": int(len(rollout["returns"])),
        "metrics": rollout["metrics"],
        "artifacts": paths,
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / f"{prefix}_rl_replay_summary.json", summary)
    return summary


def _run_sequence_pipeline_for_prepared(
    *,
    prepared: Any,
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
    year: int | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    prefix: str = "sequence",
) -> dict[str, Any]:
    dataset_payload = _build_sequence_dataset_artifact(
        prepared=prepared,
        study_root=study_root,
        tag=tag,
        args=args,
        year=year,
        start_date=start_date,
        end_date=end_date,
    )
    dataset_summary = dataset_payload["manifest"]
    if str(dataset_summary.get("status", "") or "") != "completed":
        skip_summary = {
            "status": "skipped",
            "reason": "incomplete_dataset",
            "dataset_status": str(dataset_summary.get("status", "") or "unknown"),
            "incomplete_reason": str(dataset_summary.get("incomplete_reason", "") or ""),
            "trading_day_count": int(dataset_summary.get("trading_day_count", 0) or 0),
            "min_year_trading_days": int(args.min_year_trading_days),
            "shadow_only": True,
            "promotion_allowed": False,
        }
        return {
            "dataset_summary": dataset_summary,
            "train_summary": skip_summary,
            "replay_summary": skip_summary,
        }
    train_payload = _run_sequence_train_smoke(dataset_payload["trajectory"], study_root=study_root, args=args)
    replay_summary: dict[str, Any] = {"status": "not_run"}
    if train_payload.get("summary", {}).get("status") == "completed":
        replay_summary = _run_sequence_replay_smoke(
            prepared=prepared,
            trajectory=dataset_payload["trajectory"],
            model=train_payload["model"],
            tensors=train_payload["tensors"],
            study_root=study_root,
            args=args,
            prefix=prefix,
        )
    return {
        "dataset_summary": dataset_summary,
        "train_summary": train_payload.get("summary", train_payload),
        "replay_summary": replay_summary,
    }


def _prepare_for_window(args: argparse.Namespace, *, start_date: str, end_date: str, tag: str) -> Any:
    return prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=start_date,
        end_date=end_date,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        lake_dataset_id=str(args.lake_dataset_id),
        data_lake_root=args.data_lake_root,
        lake_min_trading_days=int(args.lake_min_trading_days),
        max_universe_size=int(args.max_universe_size),
        alpha_prior_source=DEFAULT_ALPHA_PRIOR_SOURCE,
        progress_desc=f"alpha_path20 sequence prepare {tag}",
    )


def _run_sequence_multiyear_smoke(*, study_root: Path, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    yearly: dict[str, Any] = {}
    for year in DEFAULT_FULL_YEAR_WINDOWS:
        start_date, end_date = full_year_window(year)
        year_root = study_root / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        prepared = _prepare_for_window(args, start_date=start_date, end_date=end_date, tag=f"{tag}_{year}")
        result = _run_sequence_pipeline_for_prepared(
            prepared=prepared,
            study_root=year_root,
            tag=f"{tag}_{year}",
            args=args,
            year=year,
            start_date=start_date,
            end_date=end_date,
            prefix=f"sequence_{year}",
        )
        yearly[str(year)] = result
    aggregate = _multiyear_aggregate(yearly)
    summary = {
        "status": "completed",
        "stage": "rl_multiyear_smoke",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "fixed_years": list(DEFAULT_FULL_YEAR_WINDOWS.keys()),
        "yearly": yearly,
        "aggregate": aggregate,
        "leakage_guard_passed": all(
            dict(result.get("dataset_summary", {}) or {}).get("leakage_guard", {}).get("status") == "passed"
            for result in yearly.values()
        ),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / "sequence_multiyear_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_walkforward_study(*, study_root: Path, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    years = [*WALKFORWARD_TRAIN_YEARS, *WALKFORWARD_VALIDATION_YEARS, *WALKFORWARD_TEST_YEARS]
    yearly: dict[str, Any] = {}
    train_episodes: list[Path20MarketEpisode] = []
    validation_episodes: list[Path20MarketEpisode] = []
    for year in years:
        start_date, end_date = full_year_window(year)
        year_root = study_root / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        prepared = _prepare_for_window(args, start_date=start_date, end_date=end_date, tag=f"{tag}_{year}")
        dataset_payload = _build_episode_dataset_artifact(
            prepared=prepared,
            study_root=year_root,
            tag=f"{tag}_{year}",
            args=args,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )
        episode = dataset_payload["episode"]
        yearly[str(year)] = {
            "role": _walkforward_role(year),
            "dataset_summary": dataset_payload["manifest"],
            "prepared_summary": prepared.to_summary(),
            "train_summary": {"status": "pending"},
            "replay_summary": {"status": "pending"},
        }
        if str(episode.manifest.get("status", "")) != "completed":
            skip = {
                "status": "skipped",
                "reason": "incomplete_dataset",
                "incomplete_reason": str(episode.manifest.get("incomplete_reason", "")),
                "trading_day_count": int(episode.manifest.get("trading_day_count", 0) or 0),
            }
            yearly[str(year)]["train_summary"] = skip
            yearly[str(year)]["replay_summary"] = skip
            continue
        if year in WALKFORWARD_TRAIN_YEARS:
            train_episodes.append(episode)
        elif year in WALKFORWARD_VALIDATION_YEARS:
            validation_episodes.append(episode)
    train_payload = _episode_train_model(
        episodes=train_episodes,
        validation_episodes=validation_episodes,
        study_root=study_root,
        args=args,
        train_years=list(WALKFORWARD_TRAIN_YEARS),
        validation_years=list(WALKFORWARD_VALIDATION_YEARS),
    )
    train_summary = dict(train_payload.get("summary", train_payload))
    if train_summary.get("status") == "completed":
        for year in years:
            year_text = str(year)
            if yearly[year_text]["dataset_summary"].get("status") != "completed":
                continue
            start_date, end_date = full_year_window(year)
            year_root = study_root / year_text
            prepared = _prepare_for_window(args, start_date=start_date, end_date=end_date, tag=f"{tag}_{year}_replay")
            episode = build_path20_market_episode(
                prepared,
                start_date=start_date,
                end_date=end_date,
                lake_dataset_id=str(args.lake_dataset_id or ""),
                year=year,
                sequence_length=int(args.sequence_length),
                execution_mode=args.execution_mode,
                reward_profile=args.reward_profile,
                max_feature_columns=int(args.rl_max_feature_columns),
                min_trading_days=int(args.min_year_trading_days),
            )
            replay_summary = _run_episode_replay(
                prepared=prepared,
                episode=episode,
                model=train_payload["model"],
                normalization=train_payload["normalization"],
                study_root=year_root,
                args=args,
                prefix=f"episode_{year}",
            )
            yearly[year_text]["train_summary"] = train_summary if year in WALKFORWARD_TRAIN_YEARS else {"status": "not_train_year"}
            yearly[year_text]["replay_summary"] = replay_summary
    aggregate = _walkforward_aggregate(yearly)
    summary = {
        "status": "completed" if train_summary.get("status") == "completed" else "skipped",
        "stage": "rl_walkforward_study",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "train_years": list(WALKFORWARD_TRAIN_YEARS),
        "validation_years": list(WALKFORWARD_VALIDATION_YEARS),
        "test_years": list(WALKFORWARD_TEST_YEARS),
        "yearly": yearly,
        "train_summary": train_summary,
        "aggregate": aggregate,
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / "sequence_walkforward_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _walkforward_role(year: int) -> str:
    if year in WALKFORWARD_TRAIN_YEARS:
        return "train"
    if year in WALKFORWARD_VALIDATION_YEARS:
        return "validation"
    if year in WALKFORWARD_TEST_YEARS:
        return "test"
    return "unused"


def _walkforward_aggregate(yearly: dict[str, Any]) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for role in ("train", "validation", "test"):
        role_yearly = {
            year: result
            for year, result in yearly.items()
            if str(result.get("role", "")) == role
        }
        by_role[role] = _multiyear_aggregate(role_yearly)
    return by_role


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run path20 sequence/RL mainline protocol. "
            "Legacy alpha_path20_neural_policy_v1 stages are diagnostic-only and require an explicit allow flag."
        )
    )
    parser.add_argument(
        "--stage",
        choices=tuple(sorted(LEGACY_NEURAL_STAGES | SEQUENCE_RL_STAGES)),
        default="rl-episode-dataset",
    )
    parser.add_argument("--tag", required=True, help="Explicit protocol/study tag. Loose latest is forbidden.")
    parser.add_argument("--policy-version", default=ALPHA_PATH20_SEQUENCE_POLICY_VERSION)
    parser.add_argument(
        "--allow-legacy-neural-policy",
        action="store_true",
        help="Allow diagnostic-only alpha_path20_neural_policy_v1 stages: dataset-smoke, oracle-smoke, tiny-smoke.",
    )
    parser.add_argument("--data-source", default="lake", choices=("lake", "csv", "tq"))
    parser.add_argument("--lake-dataset-id", default="", help="Required for --data-source lake.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-name", default="learned_all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--start-date", default="20240102")
    parser.add_argument("--end-date", default="20240329")
    parser.add_argument("--max-universe-size", type=int, default=80)
    parser.add_argument("--lake-min-trading-days", type=int, default=2)
    parser.add_argument("--execution-mode", default="next_open", choices=("next_open", "close"))
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--max-position-weight", type=float, default=0.10)
    parser.add_argument("--max-gross-exposure", type=float, default=0.95)
    parser.add_argument("--max-positions", type=int, default=30)
    parser.add_argument("--turnover-budget", type=float, default=1.00)
    parser.add_argument("--smoke-max-rows", type=int, default=4096)
    parser.add_argument("--allocator-max-names", type=int, default=80)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-lr", type=float, default=1.0e-3)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--reward-profile", default=DEFAULT_RL_REWARD_PROFILE)
    parser.add_argument("--model-family", default="sequence_gru", choices=("sequence_gru", "decision_transformer"))
    parser.add_argument("--no-oracle-input", action="store_true", default=True)
    parser.add_argument("--rl-hidden-dim", type=int, default=48)
    parser.add_argument("--rl-dropout", type=float, default=0.0)
    parser.add_argument("--rl-max-feature-columns", type=int, default=96)
    parser.add_argument("--min-year-trading-days", type=int, default=180)
    return parser


def _validate_protocol_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.data_source == "lake" and not str(args.lake_dataset_id or "").strip():
        parser.error("--data-source lake requires explicit --lake-dataset-id; do not rely on loose latest/default.")
    if _is_loose_lake_dataset_id(str(args.lake_dataset_id or "")):
        parser.error("--lake-dataset-id must be a fixed dataset id, not latest/default/latest_*.")
    if args.stage in LEGACY_NEURAL_STAGES and not bool(args.allow_legacy_neural_policy):
        parser.error(
            "dataset-smoke/oracle-smoke/tiny-smoke are legacy alpha_path20_neural_policy_v1 diagnostic stages. "
            "Pass --allow-legacy-neural-policy to run them, or use rl-* stages for the path20 mainline."
        )
    if args.stage in SEQUENCE_RL_STAGES and str(args.policy_version or "").strip() not in {
        "",
        ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
    }:
        parser.error(f"RL stages require --policy-version {ALPHA_PATH20_SEQUENCE_POLICY_VERSION}.")
    if args.stage in SEQUENCE_RL_STAGES and not bool(args.no_oracle_input):
        parser.error("RL stages require --no-oracle-input; oracle inputs are not allowed.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_protocol_args(parser, args)
    tag = str(args.tag or "").strip()
    study_root = PATH_POLICY_STUDIES_ROOT / tag
    study_root.mkdir(parents=True, exist_ok=True)
    effective_lake_dataset_id = str(args.lake_dataset_id or DEFAULT_POLICY_INPUT_LAKE_DATASET_ID).strip()
    args.lake_dataset_id = effective_lake_dataset_id
    if args.stage == "rl-walkforward-study":
        summary = {
            "status": "completed",
            "run_tag": tag,
            "study_tag": tag,
            "created_at": now_iso(),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
            "research_status": "research / shadow-only / alpha_path20_sequence_policy_v1 walk-forward evidence",
            "stage": args.stage,
            "data_source": args.data_source,
            "lake_dataset_id": effective_lake_dataset_id,
            "execution_mode": args.execution_mode,
            "sequence_policy": _run_walkforward_study(study_root=study_root, tag=tag, args=args),
            "facts": [
                "alpha_path20_sequence_policy_v1 is the path20 main research line.",
                "Walk-forward evidence separates train, validation, and final shadow test years.",
                "Exact replay remains PortfolioState.step after deterministic safety projection.",
            ],
            "boundaries": [
                "shadow_only=true",
                "no live/default promotion",
                "no active_execution_strategy.json modification",
                "no loose latest references",
                "oracle inputs are forbidden for sequence policy training and replay",
            ],
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        summary = _json_ready(summary)
        write_json(study_root / "study_summary.json", summary)
        safe_print_json(summary)
        return summary
    if args.stage == "rl-multiyear-smoke":
        summary = {
            "status": "completed",
            "run_tag": tag,
            "study_tag": tag,
            "created_at": now_iso(),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
            "research_status": "research / shadow-only / alpha_path20_sequence_policy_v1 evidence",
            "stage": args.stage,
            "data_source": args.data_source,
            "lake_dataset_id": effective_lake_dataset_id,
            "execution_mode": args.execution_mode,
            "sequence_policy": _run_sequence_multiyear_smoke(study_root=study_root, tag=tag, args=args),
            "facts": [
                "alpha_path20_sequence_policy_v1 is a pure neural sequence/offline-RL research line.",
                "Oracle and future path labels are forbidden as sequence-policy inputs.",
                "portfolio_daily_target_weight remains the simulator execution truth after safety projection.",
            ],
            "boundaries": [
                "shadow_only=true",
                "no live/default promotion",
                "no active_execution_strategy.json modification",
                "no loose latest references",
            ],
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        summary = _json_ready(summary)
        write_json(study_root / "study_summary.json", summary)
        safe_print_json(summary)
        return summary

    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=args.end_date if args.stage in SEQUENCE_RL_STAGES else _extend_end_date_for_labels(args.end_date),
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        lake_dataset_id=effective_lake_dataset_id,
        data_lake_root=args.data_lake_root,
        lake_min_trading_days=int(args.lake_min_trading_days),
        max_universe_size=int(args.max_universe_size),
        alpha_prior_source=DEFAULT_ALPHA_PRIOR_SOURCE,
        progress_desc=f"alpha_path20 prepare {tag}",
    )
    if args.stage in SEQUENCE_RL_STAGES:
        if args.stage in SEQUENCE_RL_EPISODE_STAGES:
            dataset_payload = _build_episode_dataset_artifact(prepared=prepared, study_root=study_root, tag=tag, args=args)
            train_summary = {"status": "not_run"}
            replay_summary = {"status": "not_run"}
            if args.stage in {"rl-train-episode", "rl-replay-episode"}:
                train_payload = _episode_train_model(
                    episodes=[dataset_payload["episode"]],
                    validation_episodes=[],
                    study_root=study_root,
                    args=args,
                    train_years=[],
                    validation_years=[],
                )
                train_summary = dict(train_payload.get("summary", train_payload))
                if args.stage == "rl-replay-episode" and train_summary.get("status") == "completed":
                    replay_summary = _run_episode_replay(
                        prepared=prepared,
                        episode=dataset_payload["episode"],
                        model=train_payload["model"],
                        normalization=train_payload["normalization"],
                        study_root=study_root,
                        args=args,
                        prefix="episode",
                    )
            summary = {
                "status": "completed",
                "run_tag": tag,
                "study_tag": tag,
                "created_at": now_iso(),
                "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
                "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
                "research_status": "research / shadow-only / alpha_path20_sequence_policy_v1 episode evidence",
                "stage": args.stage,
                "data_source": args.data_source,
                "lake_dataset_id": effective_lake_dataset_id,
                "execution_mode": args.execution_mode,
                "prepared_summary": prepared.to_summary(),
                "episode_dataset_summary": dataset_payload["manifest"],
                "episode_train_summary": train_summary,
                "episode_replay_summary": replay_summary,
                "facts": [
                    "Market episode artifacts keep reward columns out of model input features.",
                    "Episode training rolls current weights forward and trains on projected target weights.",
                    "Exact replay uses PortfolioState.step after deterministic safety projection.",
                ],
                "boundaries": [
                    "shadow_only=true",
                    "no live/default promotion",
                    "no active_execution_strategy.json modification",
                    "no loose latest references",
                    "oracle inputs are forbidden for sequence policy training and replay",
                ],
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
            summary = _json_ready(summary)
            write_json(study_root / "study_summary.json", summary)
            safe_print_json(summary)
            return summary
        dataset_payload = _build_sequence_dataset_artifact(prepared=prepared, study_root=study_root, tag=tag, args=args)
        train_summary: dict[str, Any] = {"status": "not_run"}
        replay_summary: dict[str, Any] = {"status": "not_run"}
        if args.stage in {"rl-train-smoke", "rl-replay-smoke"}:
            train_payload = _run_sequence_train_smoke(dataset_payload["trajectory"], study_root=study_root, args=args)
            train_summary = dict(train_payload.get("summary", train_payload))
            if args.stage == "rl-replay-smoke" and train_summary.get("status") == "completed":
                replay_summary = _run_sequence_replay_smoke(
                    prepared=prepared,
                    trajectory=dataset_payload["trajectory"],
                    model=train_payload["model"],
                    tensors=train_payload["tensors"],
                    study_root=study_root,
                    args=args,
                )
        summary = {
            "status": "completed",
            "run_tag": tag,
            "study_tag": tag,
            "created_at": now_iso(),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
            "research_status": "research / shadow-only / alpha_path20_sequence_policy_v1 evidence",
            "stage": args.stage,
            "data_source": args.data_source,
            "lake_dataset_id": effective_lake_dataset_id,
            "execution_mode": args.execution_mode,
            "prepared_summary": prepared.to_summary(),
            "sequence_dataset_summary": dataset_payload["manifest"],
            "sequence_train_summary": train_summary,
            "sequence_replay_summary": replay_summary,
            "facts": [
                "alpha_path20_sequence_policy_v1 is isolated under daily_research/path_policy.",
                "The sequence policy writes explicit dataset manifests and forbids oracle/future input columns.",
                "target_weight is projected through the deterministic safety layer before simulator replay.",
            ],
            "inferences": [
                "This smoke validates the sequence-policy data, model, projection, and replay plumbing.",
                "It is not sufficient promotion or live/default evidence.",
            ],
            "assumptions": [
                "Lake policy bundle inputs contain current market panels and membership for the requested window.",
                "Deterministic projection is a safety layer, not an oracle teacher.",
            ],
            "boundaries": [
                "shadow_only=true",
                "no live/default promotion",
                "no active_execution_strategy.json modification",
                "no loose latest references",
                "oracle inputs are forbidden for sequence policy training and replay",
            ],
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        summary = _json_ready(summary)
        write_json(study_root / "study_summary.json", summary)
        safe_print_json(summary)
        return summary

    dataset_payload: dict[str, Any] | None = None
    dataset_summary: dict[str, Any] = {"status": "not_run"}
    forecaster_summary: dict[str, Any] = {"status": "not_run"}
    allocator_summary: dict[str, Any] = {"status": "not_run"}
    oracle_summary: dict[str, Any] = {"status": "not_run"}
    if args.stage in {"dataset-smoke", "tiny-smoke"}:
        dataset_payload = _build_dataset_artifact(prepared=prepared, study_root=study_root, tag=tag, args=args)
        dataset_summary = dict(dataset_payload["manifest"])
    if args.stage == "tiny-smoke":
        assert dataset_payload is not None
        forecaster_summary = _run_forecaster_smoke(dataset_payload["dataset"], study_root=study_root, args=args)
        allocator_summary = _run_allocator_smoke(dataset_payload["dataset"], study_root=study_root, args=args)
    if args.stage in {"oracle-smoke", "tiny-smoke"}:
        oracle_summary = _run_oracle_smoke(prepared=prepared, study_root=study_root, args=args)
    summary = {
        "status": "completed",
        "run_tag": tag,
        "study_tag": tag,
        "created_at": now_iso(),
        "policy_version": ALPHA_PATH20_POLICY_VERSION,
        "policy_profile": ALPHA_PATH20_POLICY_PROFILE,
        "research_status": "legacy / diagnostic-only / alpha_path20_neural_policy_v1 evidence",
        "legacy_diagnostic_only": True,
        "no_new_mainline_budget": True,
        "stage": args.stage,
        "data_source": args.data_source,
        "lake_dataset_id": effective_lake_dataset_id,
        "execution_mode": args.execution_mode,
        "prepared_summary": prepared.to_summary(),
        "dataset_summary": dataset_summary,
        "predicted_path_quality": forecaster_summary,
        "allocator_smoke": allocator_summary,
        "oracle_path_upper_bound": oracle_summary,
        "facts": [
            "alpha_path20_neural_policy_v1 is legacy diagnostic-only evidence under daily_research/path_policy.",
            "The protocol writes path-policy artifacts under daily_research/output/path_policy with explicit tag and dataset id.",
            "target_weight is the only execution truth; source/receiver fields are derived diagnostics.",
        ],
        "inferences": [
            "This smoke can validate schema, next-open labels, target-weight projection, and oracle-path replay plumbing.",
            "It is not sequence/RL mainline evidence and must not justify new mainline budget.",
        ],
        "assumptions": [
            "Lake policy bundle inputs already contain adjusted open/close panels and universe membership.",
            "The first allocator smoke may use oracle path labels for tiny optimization without claiming deployable prediction quality.",
        ],
        "boundaries": [
            "shadow_only=true",
            "legacy_diagnostic_only=true",
            "no live/default promotion",
            "no active_execution_strategy.json modification",
            "no loose latest references",
            "no long training started by this smoke protocol",
        ],
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    summary = _json_ready(summary)
    write_json(study_root / "study_summary.json", summary)
    safe_print_json(summary)
    return summary


if __name__ == "__main__":
    main()
