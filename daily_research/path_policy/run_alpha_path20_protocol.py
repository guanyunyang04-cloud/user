from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from daily_research.path_policy.forecast_dataset import (
    build_forecast_memmap_dataset,
    build_forecast_sequence_dataset,
    load_forecast_memmap_dataset,
    normalize_static_context_fields,
    save_forecast_sequence_dataset,
)
from daily_research.path_policy.forecast_features import (
    DEFAULT_FORECAST_FEATURE_PROFILE,
    DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    FORECAST_FEATURE_PROFILES,
)
from daily_research.path_policy.forecast_training import (
    FORECAST_LOSS_PROFILES,
    FORECAST_MODEL_FAMILIES,
    FORECAST_OUTPUT_PROFILES,
    FORECAST_SELECTION_PROFILES,
    forecast_loss_profile_contract,
    train_forecast_models,
)
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
    EPISODE_PORTFOLIO_FEATURE_COLUMNS,
    Path20MarketEpisode,
    build_path20_market_episode,
    episode_array_manifest,
    episode_policy_rollout_loss,
    episode_to_arrays,
    fit_episode_normalization,
    load_episode_arrays,
    market_episode_from_long_frame,
    predict_episode_targets,
    projection_parity_diagnostics,
    save_episode_arrays,
    stable_episode_manifest_hash,
)
from daily_research.path_policy.rl_models import (
    DecisionTransformerTargetWeightPolicy,
    PortfolioDecisionTransformerPolicy,
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
FORECAST_MAINLINE_STAGES = frozenset({"forecast-dataset", "forecast-train", "forecast-walkforward-study"})
FORECAST_TRAINING_STAGES = frozenset({"forecast-train", "forecast-walkforward-study"})
NEURAL_MAINLINE_STAGES = frozenset({"dataset-smoke", "oracle-smoke", "tiny-smoke"}) | FORECAST_MAINLINE_STAGES
LEGACY_NEURAL_STAGES = frozenset()
SEQUENCE_RL_SMOKE_STAGES = frozenset({"rl-dataset-smoke", "rl-train-smoke", "rl-replay-smoke", "rl-multiyear-smoke"})
SEQUENCE_RL_EPISODE_STAGES = frozenset(
    {
        "rl-episode-dataset",
        "rl-train-episode",
        "rl-replay-episode",
        "rl-walkforward-study",
        "rl-walkforward-matrix",
        "rl-v4-validation-repair-study",
        "rl-v5-dt-validation-study",
    }
)
SEQUENCE_RL_STAGES = SEQUENCE_RL_SMOKE_STAGES | SEQUENCE_RL_EPISODE_STAGES
WALKFORWARD_TRAIN_YEARS = (2019, 2020)
WALKFORWARD_VALIDATION_YEARS = (2022,)
WALKFORWARD_TEST_YEARS = (2024,)
V4_DEFAULT_SEEDS = (7, 11, 19)
V4_PROJECTION_PENALTY_GRID = (0.05, 0.20, 0.50)
V4_BASELINES = (
    "cash_no_trade",
    "liquidity_equal_top30",
    "score_blend_top30",
    "alpha_prior_target_weight",
    "v3_final_checkpoint",
)
V5_DEFAULT_CONTEXT_GRID = (20, 60)
V5_DEFAULT_HIDDEN_DIM = 64
V5_DEFAULT_DROPOUT = 0.10
V5_DEFAULT_LR = 3.0e-4
V5_BASELINES = (
    *V4_BASELINES,
    "v4_gru_selected_checkpoint",
)

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


def _read_frame_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def _apply_stage_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not str(getattr(args, "policy_version", "") or "").strip():
        args.policy_version = (
            ALPHA_PATH20_SEQUENCE_POLICY_VERSION
            if args.stage in SEQUENCE_RL_STAGES
            else ALPHA_PATH20_POLICY_VERSION
        )
    if getattr(args, "smoke_lr", None) is None:
        args.smoke_lr = V5_DEFAULT_LR if args.stage == "rl-v5-dt-validation-study" else 1.0e-3
    if getattr(args, "rl_hidden_dim", None) is None:
        args.rl_hidden_dim = V5_DEFAULT_HIDDEN_DIM if args.stage == "rl-v5-dt-validation-study" else 48
    if getattr(args, "rl_dropout", None) is None:
        args.rl_dropout = V5_DEFAULT_DROPOUT if args.stage == "rl-v5-dt-validation-study" else 0.0
    return args


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


def _parse_int_list(value: str | None, *, default: tuple[int, ...]) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return [int(item) for item in default]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(value: str | None, *, default: tuple[float, ...]) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return [float(item) for item in default]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _set_research_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


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

    _apply_stage_defaults(args)
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


def _forecast_model_families(value: str | None) -> tuple[str, ...]:
    families = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    return families or tuple(FORECAST_MODEL_FAMILIES)


def _forecast_int_tuple(value: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return tuple(int(item) for item in default)
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _resolve_forecast_horizon_args(args: argparse.Namespace) -> tuple[int, tuple[int, ...]]:
    raw_horizons = _forecast_int_tuple(
        getattr(args, "forecast_cumulative_horizons", ""),
        default=(1, 3, 5, 10, 20),
    )
    cumulative_horizons = tuple(sorted(dict.fromkeys(int(item) for item in raw_horizons)))
    if not cumulative_horizons:
        raise ValueError("--forecast-cumulative-horizons must contain at least one horizon.")
    forecast_horizon = int(getattr(args, "forecast_horizon", 0) or 0)
    if forecast_horizon <= 0:
        forecast_horizon = int(max(cumulative_horizons))
    if int(max(cumulative_horizons)) > forecast_horizon:
        if int(getattr(args, "forecast_horizon", 0) or 0) > 0:
            raise ValueError("--forecast-horizon must be >= max(--forecast-cumulative-horizons).")
        forecast_horizon = int(max(cumulative_horizons))
    explicit_horizon = int(getattr(args, "forecast_horizon", 0) or 0) > 0
    if int(max(cumulative_horizons)) > 30 and not explicit_horizon:
        raise ValueError(
            "horizon grids above 30d require explicit --forecast-horizon 45 "
            "to make label window and purge semantics auditable."
        )
    invalid = [item for item in cumulative_horizons if int(item) <= 0 or int(item) > forecast_horizon]
    if invalid:
        raise ValueError(f"--forecast-cumulative-horizons values must be in [1, {forecast_horizon}], got {invalid}.")
    return forecast_horizon, cumulative_horizons


def _forecast_manifest_prepared_summary(manifest: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    role_years = dict(manifest.get("role_years", {}) or {})
    feature_manifest = dict(manifest.get("feature_manifest", {}) or {})
    feature_shape = list(manifest.get("feature_store_shape", []) or [])
    return {
        "source": "forecast_memmap_manifest",
        "artifact_reused": True,
        "universe_size": int(feature_shape[1]) if len(feature_shape) >= 2 else 0,
        "pool_name": args.pool_name,
        "benchmark": args.benchmark,
        "data_source": args.data_source,
        "csv_folder": args.csv_folder,
        "start_date": f"{int(role_years.get('train_start_year', args.forecast_train_start_year)) - 1}0101"
        if role_years.get("train_start_year", args.forecast_train_start_year)
        else "",
        "end_date": f"{int(role_years.get('test_year', args.forecast_test_year))}1231"
        if role_years.get("test_year", args.forecast_test_year)
        else "",
        "requested_start_date": "",
        "lake_dataset_id": args.lake_dataset_id,
        "manifest_json": str(manifest.get("manifest_json", "") or ""),
        "dataset_mode": str(manifest.get("dataset_mode", "")),
        "sample_count": int(manifest.get("sample_count", 0) or 0),
        "sample_count_by_role": dict(manifest.get("sample_count_by_role", {}) or {}),
        "feature_profile": str(manifest.get("feature_profile", feature_manifest.get("feature_profile", ""))),
        "feature_store_shape": feature_shape,
        "history_valid_ratio_summary": dict(manifest.get("history_valid_ratio_summary", {}) or {}),
        "role_years": role_years,
        "raw_cache_meta": {},
        "prepared_cache_meta": {"source": "forecast_memmap_manifest", "artifact_reused": True},
        "rolling_pool_summary": {},
        "alpha_prior_summary": {},
        "metadata_summary": {
            "sector_board_view": {
                "dataset_id": str(manifest.get("source_sector_board_view_id", "") or ""),
                "view_kind": str(manifest.get("source_sector_board_view_kind", "") or ""),
                "snapshot_semantics": str(manifest.get("source_sector_board_snapshot_semantics", "") or ""),
            }
        },
    }


def _run_forecast_walkforward_study(
    *,
    prepared: Any | None,
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    families = _forecast_model_families(getattr(args, "forecast_model_families", ""))
    forecast_horizon, cumulative_horizons = _resolve_forecast_horizon_args(args)
    dataset_mode = str(getattr(args, "forecast_dataset_mode", "eager") or "eager").strip().lower()
    if dataset_mode == "memmap":
        manifest_path = str(getattr(args, "forecast_memmap_manifest", "") or "").strip()
        if manifest_path:
            dataset = load_forecast_memmap_dataset(manifest_path)
        else:
            dataset = build_forecast_memmap_dataset(
                prepared,
                root=study_root,
                train_start_year=int(args.forecast_train_start_year),
                train_end_year=int(args.forecast_train_end_year),
                validation_year=int(args.forecast_validation_year),
                test_year=int(args.forecast_test_year),
                lookback_days=int(args.forecast_lookback_days),
                horizon=forecast_horizon,
                cumulative_horizons=cumulative_horizons,
                execution_mode=args.execution_mode,
                feature_profile=str(args.forecast_feature_profile),
                max_feature_columns=int(args.forecast_max_feature_columns),
                max_samples_per_role=int(args.forecast_max_samples_per_role),
                max_samples_per_date_per_role=int(getattr(args, "forecast_max_samples_per_date_per_role", 0)),
                min_lookback_valid_ratio=float(args.forecast_min_lookback_valid_ratio),
                include_static_context=bool(args.forecast_include_static_context),
                static_context_fields=normalize_static_context_fields(str(args.forecast_static_fields)),
            )
        dataset_manifest = dict(dataset.manifest)
        if not dataset_manifest.get("manifest_json"):
            dataset_manifest["manifest_json"] = str((study_root / "forecast_dataset_manifest.json").resolve())
    else:
        dataset = build_forecast_sequence_dataset(
            prepared,
            train_start_year=int(args.forecast_train_start_year),
            train_end_year=int(args.forecast_train_end_year),
            validation_year=int(args.forecast_validation_year),
            test_year=int(args.forecast_test_year),
            lookback_days=int(args.forecast_lookback_days),
            horizon=forecast_horizon,
            cumulative_horizons=cumulative_horizons,
            execution_mode=args.execution_mode,
            feature_profile=str(args.forecast_feature_profile),
            max_feature_columns=int(args.forecast_max_feature_columns),
            max_samples_per_role=int(args.forecast_max_samples_per_role),
            max_samples_per_date_per_role=int(getattr(args, "forecast_max_samples_per_date_per_role", 0)),
        )
        dataset_manifest = save_forecast_sequence_dataset(dataset, study_root)
    training_summary: dict[str, Any] = {"status": "not_run"}
    if args.stage in {"forecast-train", "forecast-walkforward-study"}:
        training_summary = train_forecast_models(
            dataset,
            study_root=study_root,
            model_families=families,
            epochs=int(args.forecast_epochs),
            min_epochs=int(args.forecast_min_epochs),
            early_stop_patience=int(args.forecast_early_stop_patience),
            early_stop_min_delta=float(args.forecast_early_stop_min_delta),
            batch_size=int(args.forecast_batch_size),
            lr=float(args.forecast_lr),
            hidden_dim=int(args.forecast_hidden_dim),
            dropout=float(args.forecast_dropout),
            gru_layers=int(args.forecast_gru_layers),
            transformer_layers=int(args.forecast_transformer_layers),
            transformer_heads=int(args.forecast_transformer_heads),
            patch_sizes=_forecast_int_tuple(args.forecast_patch_sizes, default=(4, 20)),
            device=args.forecast_device,
            amp=bool(args.forecast_amp),
            seeds=args.forecast_seeds,
            grad_clip=float(args.forecast_grad_clip),
            grad_accum_steps=int(args.forecast_grad_accum_steps),
            weight_decay=float(args.forecast_weight_decay),
            write_all_predictions=bool(args.forecast_write_all_predictions),
            selection_profile=str(args.forecast_selection_profile),
            dataloader_num_workers=int(args.forecast_dataloader_num_workers),
            prefetch_factor=int(args.forecast_prefetch_factor),
            resume_from=str(args.forecast_resume_from or ""),
            save_last_checkpoint=bool(args.forecast_save_last),
            checkpoint_every_n_epochs=int(args.forecast_checkpoint_every_n_epochs),
            output_profile=str(args.forecast_output_profile),
            loss_profile=str(args.forecast_loss_profile),
            decision_cost_bps=float(args.forecast_decision_cost_bps),
            decision_hit_threshold_bps=float(args.forecast_decision_hit_threshold_bps),
            decision_drawdown_penalty=float(args.forecast_decision_drawdown_penalty),
            ranking_baseline=str(args.forecast_ranking_baseline),
            slot_diagnostics=bool(args.forecast_slot_diagnostics),
        )
    status = "completed"
    if dataset_manifest.get("status") != "completed" or training_summary.get("status") in {
        "insufficient_or_incomplete",
        "failed",
    }:
        status = "insufficient_or_incomplete"
    summary = {
        "status": status,
        "run_tag": tag,
        "study_tag": tag,
        "created_at": now_iso(),
        "policy_version": ALPHA_PATH20_POLICY_VERSION,
        "policy_profile": ALPHA_PATH20_POLICY_PROFILE,
        "research_status": "research / shadow-only / alpha_path20_neural_policy_v1 supervised forecaster stage1",
        "path20_research_mainline": True,
        "stage": str(args.stage).replace("-", "_"),
        "data_source": args.data_source,
        "lake_dataset_id": args.lake_dataset_id,
        "execution_mode": args.execution_mode,
        "prepared_summary": (
            prepared.to_summary()
            if prepared is not None
            else _forecast_manifest_prepared_summary(dataset_manifest, args=args)
        ),
        "dataset_manifest": dataset_manifest,
        "training_summary": training_summary,
        "evidence_verdict": training_summary.get("evidence_verdict", "insufficient_or_incomplete"),
        "forecast_model_families": list(families),
        "facts": [
            "alpha_path20_neural_policy_v1 is the current Path20 research mainline pointer.",
            f"Stage 1 is strict supervised forecasting: past stock state sequence to future {forecast_horizon}d excess-return path.",
            f"Stage 1 now evaluates dynamic horizons: {','.join(str(item) for item in cumulative_horizons)}.",
            "Validation evidence is interpreted before test evidence.",
            "Model outputs are trained with target_scale=100 and persisted predictions are restored to return units.",
        ],
        "inferences": [
            "Positive validation multi-horizon rank/spread metrics indicate whether the forecast task is trend-led, short-burst-led, or failed.",
            "This stage does not prove allocator, oracle, replay, or live strategy quality.",
        ],
        "assumptions": [
            "Past one-year input means the signal date and previous lookback-1 trading days.",
            f"Future {forecast_horizon}d labels follow next_open semantics unless --execution-mode close is explicitly selected.",
        ],
        "boundaries": [
            "shadow_only=true",
            "promotion_allowed=false",
            "active_execution_strategy_expected_diff=none",
            "no active_execution_strategy.json modification",
            "no allocator/replay/live execution in stage1",
        ],
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    summary = _json_ready(summary)
    write_json(study_root / "forecast_walkforward_summary.json", summary)
    write_json(study_root / "study_summary.json", summary)
    return summary


def _run_forecast_manifest_reuse_fast_path(*, study_root: Path, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "stage", "")) not in FORECAST_TRAINING_STAGES:
        raise ValueError("--forecast-memmap-manifest fast path is only valid for forecast training stages.")
    summary = _run_forecast_walkforward_study(prepared=None, study_root=study_root, tag=tag, args=args)
    summary["forecast_memmap_manifest_fast_path"] = True
    summary = _json_ready(summary)
    write_json(study_root / "forecast_walkforward_summary.json", summary)
    write_json(study_root / "study_summary.json", summary)
    return summary


def _run_allocator_smoke(dataset: pd.DataFrame, *, study_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _apply_stage_defaults(args)
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
    if family == "decision_transformer_v2":
        return PortfolioDecisionTransformerPolicy(
            config,
            temporal_layers=int(getattr(args, "dt_temporal_layers", 2)),
            cross_layers=int(getattr(args, "dt_cross_layers", 1)),
            num_heads=int(getattr(args, "dt_num_heads", 4)),
        )
    raise ValueError(f"Unsupported path20 sequence model family: {args.model_family!r}")


def _load_sequence_policy_checkpoint(
    checkpoint_path: str | Path,
    args: argparse.Namespace,
    episode: Path20MarketEpisode,
) -> dict[str, Any]:
    import torch

    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"v3 checkpoint model path does not exist: {resolved_path}")
    payload = torch.load(str(resolved_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("v3 checkpoint payload must be a dictionary.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("v3 checkpoint payload is missing a non-empty state_dict.")
    model_family = str(payload.get("model_family", "") or "").strip().lower()
    if model_family not in {"sequence_gru", "decision_transformer", "decision_transformer_v2"}:
        raise ValueError(f"Unsupported v3 checkpoint model_family: {model_family!r}")
    checkpoint_features = [str(item) for item in payload.get("feature_columns", [])]
    if not checkpoint_features:
        checkpoint_features = [str(item) for item in dict(payload.get("normalization", {}) or {}).get("feature_columns", [])]
    if checkpoint_features != list(episode.feature_columns):
        raise ValueError(
            "v3 checkpoint feature_columns do not match the current episode artifact; "
            "refusing to run a mislabeled baseline."
        )
    dynamic_columns = [str(item) for item in payload.get("dynamic_stock_feature_columns", EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS)]
    if dynamic_columns != list(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS):
        raise ValueError("v3 checkpoint dynamic stock features are incompatible with the v4 episode contract.")
    normalization = dict(payload.get("normalization", {}) or {})
    if not normalization:
        raise ValueError("v3 checkpoint payload is missing normalization stats.")
    hidden_dim = 0
    num_layers = 1
    if model_family == "sequence_gru":
        weight = state_dict.get("encoder.weight_hh_l0")
        if weight is None:
            raise ValueError("sequence_gru checkpoint is missing encoder.weight_hh_l0.")
        hidden_dim = int(weight.shape[1])
    elif model_family == "decision_transformer":
        weight = state_dict.get("input_proj.weight")
        if weight is None:
            raise ValueError("decision_transformer checkpoint is missing input_proj.weight.")
        hidden_dim = int(weight.shape[0])
        layer_ids: set[int] = set()
        for key in state_dict:
            parts = str(key).split(".")
            if len(parts) > 2 and parts[0] == "encoder" and parts[1] == "layers":
                try:
                    layer_ids.add(int(parts[2]))
                except ValueError:
                    pass
        num_layers = max(layer_ids) + 1 if layer_ids else 1
    else:
        weight = state_dict.get("input_proj.weight")
        if weight is None:
            raise ValueError("decision_transformer_v2 checkpoint is missing input_proj.weight.")
        hidden_dim = int(weight.shape[0])
        temporal_layer_ids: set[int] = set()
        cross_layer_ids: set[int] = set()
        for key in state_dict:
            parts = str(key).split(".")
            if len(parts) > 2 and parts[0] == "temporal_encoder" and parts[1] == "layers":
                try:
                    temporal_layer_ids.add(int(parts[2]))
                except ValueError:
                    pass
            if len(parts) > 2 and parts[0] == "cross_encoder" and parts[1] == "layers":
                try:
                    cross_layer_ids.add(int(parts[2]))
                except ValueError:
                    pass
        num_layers = max(temporal_layer_ids) + 1 if temporal_layer_ids else 2
        cross_layers = max(cross_layer_ids) + 1 if cross_layer_ids else 1
    config = SequencePolicyConfig(
        stock_feature_dim=int(len(checkpoint_features) + len(dynamic_columns)),
        portfolio_feature_dim=len(EPISODE_PORTFOLIO_FEATURE_COLUMNS),
        hidden_dim=int(hidden_dim),
        dropout=0.0,
        max_position_weight=float(args.max_position_weight),
    )
    if model_family == "sequence_gru":
        model = SequenceTargetWeightPolicy(config)
    elif model_family == "decision_transformer":
        num_heads = 4 if int(hidden_dim) % 4 == 0 else 1
        model = DecisionTransformerTargetWeightPolicy(config, num_layers=int(num_layers), num_heads=int(num_heads))
    else:
        num_heads = 4 if int(hidden_dim) % 4 == 0 else 1
        model = PortfolioDecisionTransformerPolicy(config, temporal_layers=int(num_layers), cross_layers=int(cross_layers), num_heads=int(num_heads))
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "model_family": model_family,
        "normalization": normalization,
        "sequence_length": int(payload.get("sequence_length", args.sequence_length)),
        "checkpoint_model_pt": str(resolved_path.resolve()),
        "feature_columns": checkpoint_features,
        "dynamic_stock_feature_columns": dynamic_columns,
        "policy_version": str(payload.get("policy_version", "")),
        "shadow_only": bool(payload.get("shadow_only", True)),
    }


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
    role = _walkforward_role(int(year)) if year is not None and str(year).strip() else "single"
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
    arrays_path = dataset_root / "market_episode_arrays.npz"
    array_manifest_path = dataset_root / "array_manifest.json"
    base_manifest = {
        **episode.manifest,
        "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
        "data_source": args.data_source,
        "role": role,
        "dataset_csv": str(dataset_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "arrays_npz": str(arrays_path.resolve()),
        "array_manifest_json": str(array_manifest_path.resolve()),
        "run_tag": tag,
        "row_count": int(sum(len(frame) for frame in episode.daily_frames)),
        "normalization_scope": "train_years_only",
        "artifact_reuse_enabled": bool(getattr(args, "reuse_episode_artifacts", True)),
    }
    base_manifest["manifest_hash"] = stable_episode_manifest_hash(base_manifest)
    if bool(getattr(args, "reuse_episode_artifacts", True)) and any(
        path.exists() for path in (manifest_path, dataset_path, arrays_path, array_manifest_path)
    ):
        missing_paths = [
            str(path.resolve())
            for path in (manifest_path, dataset_path, arrays_path, array_manifest_path)
            if not path.exists()
        ]
        if missing_paths:
            raise ValueError(f"Existing episode artifact is incomplete for {dataset_id}; missing: {missing_paths}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_hash = str(existing_manifest.get("manifest_hash", "") or "")
        if existing_hash != str(base_manifest["manifest_hash"]):
            raise ValueError(
                f"Existing episode artifact hash mismatch for {dataset_id}: "
                f"{existing_hash or 'missing'} != {base_manifest['manifest_hash']}"
            )
        long_frame = _read_frame_allow_empty(dataset_path)
        reused_episode = market_episode_from_long_frame(long_frame, existing_manifest)
        write_json(study_root / "episode_dataset_manifest.json", {**existing_manifest, "artifact_reused": True})
        return {
            "episode": reused_episode,
            "dataset_frame": long_frame,
            "manifest": {**existing_manifest, "artifact_reused": True},
            "arrays": load_episode_arrays(arrays_path),
            "array_manifest": json.loads(array_manifest_path.read_text(encoding="utf-8")) if array_manifest_path.exists() else {},
        }
    long_frame = episode.to_long_frame()
    _write_frame(dataset_path, long_frame)
    arrays = episode_to_arrays(episode)
    arrays_npz = save_episode_arrays(arrays_path, arrays)
    manifest = {**base_manifest, "row_count": int(len(long_frame))}
    manifest["manifest_hash"] = stable_episode_manifest_hash(manifest)
    array_manifest = episode_array_manifest(arrays, manifest=manifest, arrays_npz_path=arrays_npz)
    write_json(array_manifest_path, array_manifest)
    manifest["array_hash"] = array_manifest["array_hash"]
    manifest["manifest_hash"] = stable_episode_manifest_hash(manifest)
    write_json(manifest_path, manifest)
    write_json(study_root / "episode_dataset_manifest.json", manifest)
    return {"episode": episode, "dataset_frame": long_frame, "manifest": manifest, "arrays": arrays, "array_manifest": array_manifest}


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

    _apply_stage_defaults(args)
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
    context_summaries: list[dict[str, Any]] = []
    raw_intent_summaries: list[dict[str, float]] = []
    epoch_projection_summaries: list[dict[str, float]] = []
    for _ in range(max(int(args.smoke_epochs), 1)):
        epoch_losses: list[float] = []
        epoch_projection_rows: list[dict[str, float]] = []
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
                turnover_penalty=float(getattr(args, "turnover_penalty_weight", 0.20)),
                projection_penalty=float(getattr(args, "projection_penalty_weight", 0.05)),
                tail_mass_penalty=float(getattr(args, "tail_mass_penalty_weight", 0.0)),
                rollout_grad_mode=str(getattr(args, "rollout_grad_mode", "detached")),
                rollout_chunk_days=int(getattr(args, "rollout_chunk_days", 20)),
            )
            if result.get("status") != "completed":
                continue
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            projection_diag = dict(result.get("diagnostics", {}) or {})
            projection_summaries.append(projection_diag)
            epoch_projection_rows.append(projection_diag)
            context_summaries.append(dict(result.get("context_coverage", {}) or {}))
            raw_intent_summaries.append(dict(result.get("raw_intent_diagnostics", {}) or {}))
        epoch_projection_summaries.append(_summarize_projection_dicts(epoch_projection_rows))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    validation_surrogate: dict[str, Any] = {}
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
                turnover_penalty=float(getattr(args, "turnover_penalty_weight", 0.20)),
                projection_penalty=float(getattr(args, "projection_penalty_weight", 0.05)),
                tail_mass_penalty=float(getattr(args, "tail_mass_penalty_weight", 0.0)),
                rollout_grad_mode=str(getattr(args, "rollout_grad_mode", "detached")),
                rollout_chunk_days=int(getattr(args, "rollout_chunk_days", 20)),
            )
        validation_surrogate = {
            "status": str(validation_result.get("status", "unknown")),
            "loss": float(validation_result["loss"].detach().cpu()) if "loss" in validation_result else 0.0,
            "diagnostics": validation_result.get("diagnostics", {}),
            "raw_intent_diagnostics": validation_result.get("raw_intent_diagnostics", {}),
            "context_coverage": validation_result.get("context_coverage", {}),
            "used_projected_weights_for_loss": bool(validation_result.get("used_projected_weights_for_loss", False)),
        }
    normalization_manifest = {
        "artifact_type": "normalization_manifest",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": str(args.model_family),
        "train_years": list(train_years or []),
        "validation_years": list(validation_years or []),
        "lake_dataset_id": str(getattr(args, "lake_dataset_id", "") or ""),
        "normalization_scope": "train_years_only",
        "feature_columns": normalization.get("feature_columns", []),
        "mean": normalization.get("mean", []),
        "std": normalization.get("std", []),
        "shadow_only": True,
        "promotion_allowed": False,
    }
    normalization_manifest_path = study_root / "normalization_manifest.json"
    write_json(normalization_manifest_path, _json_ready(normalization_manifest))
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
    model_manifest = {
        "artifact_type": "sequence_policy_model_manifest",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": str(args.model_family),
        "train_years": list(train_years or []),
        "validation_years": list(validation_years or []),
        "lake_dataset_id": str(getattr(args, "lake_dataset_id", "") or ""),
        "model_pt": str(artifact_path.resolve()),
        "normalization_manifest_json": str(normalization_manifest_path.resolve()),
        "feature_columns": completed_train[0].feature_columns,
        "dynamic_stock_feature_columns": list(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS),
        "sequence_length": int(args.sequence_length),
        "reward_profile": str(args.reward_profile),
        "rollout_grad_mode": str(getattr(args, "rollout_grad_mode", "detached")),
        "rollout_chunk_days": int(getattr(args, "rollout_chunk_days", 20)),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
    }
    model_manifest_path = study_root / "model_manifest.json"
    write_json(model_manifest_path, _json_ready(model_manifest))
    projection_summary = _summarize_projection_dicts(projection_summaries)
    raw_intent_summary = _summarize_projection_dicts(raw_intent_summaries)
    projected_learning_health = _projected_learning_health(epoch_projection_summaries)
    context_coverage = _summarize_context_coverage(context_summaries)
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": str(args.model_family),
        "reward_profile": str(args.reward_profile),
        "turnover_penalty_weight": float(getattr(args, "turnover_penalty_weight", 0.20)),
        "projection_penalty_weight": float(getattr(args, "projection_penalty_weight", 0.05)),
        "tail_mass_penalty_weight": float(getattr(args, "tail_mass_penalty_weight", 0.0)),
        "train_years": list(train_years or []),
        "validation_years": list(validation_years or []),
        "epoch_count": int(max(int(args.smoke_epochs), 1)),
        "episode_count": int(len(completed_train)),
        "train_loss_curve": losses,
        "final_train_loss": float(losses[-1]) if losses else 0.0,
        "validation_surrogate_metrics": validation_surrogate,
        "projection_diagnostics_summary": projection_summary,
        "raw_intent_diagnostics_summary": raw_intent_summary,
        "projected_learning_health": projected_learning_health,
        "rollout_grad_mode": str(getattr(args, "rollout_grad_mode", "detached")),
        "rollout_chunk_days": int(getattr(args, "rollout_chunk_days", 20)),
        "context_coverage": context_coverage,
        "normalization": normalization,
        "normalization_manifest_json": str(normalization_manifest_path.resolve()),
        "artifact_pt": str(artifact_path.resolve()),
        "model_manifest_json": str(model_manifest_path.resolve()),
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


def _summarize_context_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "previous_weight_nonzero_rate": 0.0,
            "previous_reward_nonzero_rate": 0.0,
            "context_underused_warning": False,
        }
    weight_rates = [float(row.get("previous_weight_nonzero_rate", 0.0) or 0.0) for row in rows]
    reward_rates = [float(row.get("previous_reward_nonzero_rate", 0.0) or 0.0) for row in rows]
    return {
        "previous_weight_nonzero_rate": _finite_mean(weight_rates),
        "previous_reward_nonzero_rate": _finite_mean(reward_rates),
        "context_underused_warning": any(bool(row.get("context_underused_warning", False)) for row in rows),
    }


def _projected_learning_health(epoch_rows: list[dict[str, float]]) -> dict[str, Any]:
    rows = [dict(row) for row in epoch_rows if row]
    if not rows:
        return {"status": "not_available", "epoch_count": 0}
    def values_for(key: str) -> list[float]:
        return [float(row.get(key, 0.0) or 0.0) for row in rows]

    result: dict[str, Any] = {"status": "completed", "epoch_count": int(len(rows)), "by_epoch": rows}
    for key in ("avg_projection_l1_distance", "avg_projected_gross_exposure", "avg_projected_turnover"):
        values = values_for(key)
        result[f"{key}_first"] = float(values[0])
        result[f"{key}_last"] = float(values[-1])
        result[f"{key}_delta"] = float(values[-1] - values[0])
        result[f"{key}_decreased"] = bool(values[-1] < values[0]) if len(values) > 1 else False
    return result


def _episode_projection_parity_summary(
    episode: Path20MarketEpisode,
    *,
    args: argparse.Namespace,
    max_samples: int = 3,
) -> dict[str, Any]:
    arrays = episode_to_arrays(episode)
    if arrays["mask"].shape[0] <= 0:
        return {"status": "skipped", "reason": "empty_episode", "projection_mismatch_warning": False}
    stocks = [str(item) for item in arrays["stocks"].tolist()]
    sample_count = min(int(max_samples), int(arrays["mask"].shape[0]))
    rows: list[dict[str, Any]] = []
    current = np.zeros(int(arrays["mask"].shape[1]), dtype=np.float32)
    for idx in range(sample_count):
        mask = np.asarray(arrays["mask"][idx], dtype=bool)
        raw = np.where(mask, np.linspace(0.01, float(args.max_position_weight) * 1.5, len(stocks)), 0.0).astype(np.float32)
        row = projection_parity_diagnostics(
            raw,
            current,
            mask,
            stocks=stocks,
            max_position_weight=float(args.max_position_weight),
            max_gross_exposure=float(args.max_gross_exposure),
            max_positions=int(args.max_positions),
            turnover_budget=float(args.turnover_budget),
            max_l1_gap=float(getattr(args, "projection_parity_max_l1", 0.02)),
        )
        row["date"] = str(arrays["dates"][idx])
        rows.append(row)
    gaps = [float(row.get("target_l1_gap", 0.0) or 0.0) for row in rows]
    warning = any(bool(row.get("projection_mismatch_warning", False)) for row in rows)
    return {
        "status": "warning" if warning else "passed",
        "sample_count": int(len(rows)),
        "max_l1_gap": float(getattr(args, "projection_parity_max_l1", 0.02)),
        "max_target_l1_gap": float(max(gaps) if gaps else 0.0),
        "mean_target_l1_gap": _finite_mean(gaps),
        "projection_mismatch_warning": bool(warning),
        "samples": rows,
    }


def _aggregate_projection_parity_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(summary) for summary in summaries if isinstance(summary, dict) and summary]
    if not rows:
        return {"status": "not_run", "projection_mismatch_warning": False, "summary_count": 0}
    warning = any(bool(row.get("projection_mismatch_warning", False)) for row in rows)
    gaps = [float(row.get("max_target_l1_gap", 0.0) or row.get("target_l1_gap", 0.0) or 0.0) for row in rows]
    return {
        "status": "warning" if warning else "passed",
        "projection_mismatch_warning": bool(warning),
        "summary_count": int(len(rows)),
        "max_target_l1_gap": float(max(gaps) if gaps else 0.0),
        "mean_target_l1_gap": _finite_mean(gaps),
        "summaries": rows,
    }


def _summarize_replay_attribution(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"status": "not_available", "row_count": 0}
    numeric = frame.copy()
    for column in numeric.columns:
        if column not in {"signal_date", "return_date"}:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    summary = {
        "status": "completed",
        "row_count": int(len(numeric)),
        "gross_return_sum": float(numeric.get("gross_return", pd.Series(dtype=float)).sum()),
        "estimated_cost_sum": float(numeric.get("estimated_cost", pd.Series(dtype=float)).sum()),
        "net_return_sum": float(numeric.get("net_return", pd.Series(dtype=float)).sum()),
        "benchmark_return_sum": float(numeric.get("benchmark_return", pd.Series(dtype=float)).sum()),
        "excess_return_sum": float(numeric.get("excess_return", pd.Series(dtype=float)).sum()),
        "avg_cash_weight": float(numeric.get("cash_weight", pd.Series(dtype=float)).mean()) if "cash_weight" in numeric else 0.0,
        "avg_projected_gross_exposure": float(numeric.get("projected_gross_exposure", pd.Series(dtype=float)).mean()) if "projected_gross_exposure" in numeric else 0.0,
        "avg_projected_turnover": float(numeric.get("projected_turnover", pd.Series(dtype=float)).mean()) if "projected_turnover" in numeric else 0.0,
        "avg_projection_l1_distance": float(numeric.get("projection_l1_distance", pd.Series(dtype=float)).mean()) if "projection_l1_distance" in numeric else 0.0,
        "avg_target_count": float(numeric.get("target_count", pd.Series(dtype=float)).mean()) if "target_count" in numeric else 0.0,
    }
    return {key: (0.0 if isinstance(value, float) and not np.isfinite(value) else value) for key, value in summary.items()}


def _run_episode_replay(
    *,
    prepared: Any,
    episode: Path20MarketEpisode,
    model: Any,
    normalization: dict[str, Any],
    study_root: Path,
    args: argparse.Namespace,
    prefix: str = "episode",
    year: int | str | None = None,
    role: str | None = None,
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
        "attribution_csv": _write_frame(study_root / f"{prefix}_rl_attribution.csv", rollout["attribution_frame"]),
        "returns_csv": _write_frame(study_root / f"{prefix}_rl_returns.csv", _series_frame(rollout["returns"], index_name="date", value_name="net_return")),
    }
    attribution_summary = _summarize_replay_attribution(rollout["attribution_frame"])
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "stage": "rl_replay_episode",
        "model_family": str(args.model_family),
        "year": int(year) if year is not None and str(year).strip() else episode.manifest.get("year", ""),
        "role": str(role or episode.manifest.get("role", "")),
        "run_tag": str(episode.manifest.get("run_tag", "")),
        "lake_dataset_id": str(args.lake_dataset_id or ""),
        "date_count": int(len(rollout["dates"])),
        "return_count": int(len(rollout["returns"])),
        "metrics": rollout["metrics"],
        "attribution_summary": attribution_summary,
        "training_projection_diagnostics_summary": training_projection_summary,
        "projection_parity_summary": _episode_projection_parity_summary(episode, args=args),
        "artifacts": paths,
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    replay_manifest_path = study_root / f"{prefix}_rl_replay_manifest.json"
    summary["replay_manifest_json"] = str(replay_manifest_path.resolve())
    write_json(replay_manifest_path, _json_ready(summary))
    write_json(study_root / f"{prefix}_rl_replay_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_target_replay(
    *,
    prepared: Any,
    episode: Path20MarketEpisode,
    target_by_date: dict[str, pd.Series],
    study_root: Path,
    args: argparse.Namespace,
    prefix: str,
    year: int | str,
    role: str,
    source_label: str,
) -> dict[str, Any]:
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
        source_label=source_label,
    )
    paths = {
        "action_panel_csv": _write_frame(study_root / f"{prefix}_rl_action_panel.csv", rollout["action_panel"]),
        "turnover_csv": _write_frame(study_root / f"{prefix}_rl_turnover.csv", rollout["turnover_frame"]),
        "position_history_csv": _write_frame(study_root / f"{prefix}_rl_position_history.csv", rollout["position_history"]),
        "projection_diagnostics_csv": _write_frame(study_root / f"{prefix}_rl_projection_diagnostics.csv", rollout["projection_diagnostics"]),
        "attribution_csv": _write_frame(study_root / f"{prefix}_rl_attribution.csv", rollout["attribution_frame"]),
        "returns_csv": _write_frame(study_root / f"{prefix}_rl_returns.csv", _series_frame(rollout["returns"], index_name="date", value_name="net_return")),
    }
    summary = {
        "status": "completed",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "stage": "rl_target_replay",
        "source_label": str(source_label),
        "year": int(year),
        "role": str(role),
        "date_count": int(len(rollout["dates"])),
        "return_count": int(len(rollout["returns"])),
        "metrics": rollout["metrics"],
        "attribution_summary": _summarize_replay_attribution(rollout["attribution_frame"]),
        "artifacts": paths,
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / f"{prefix}_rl_replay_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _baseline_targets_for_episode(episode: Path20MarketEpisode, baseline_name: str, *, max_positions: int) -> dict[str, pd.Series]:
    targets: dict[str, pd.Series] = {}
    name = str(baseline_name)
    if name in {"v3_final_checkpoint", "v4_gru_selected_checkpoint"}:
        raise ValueError(f"{name} baseline must be generated by loading an explicit checkpoint model.")
    for daily in episode.daily_frames:
        date_text = pd.Timestamp(daily["date"].iloc[0]).strftime("%Y-%m-%d")
        stocks = daily["stock"].astype(str)
        target = pd.Series(0.0, index=stocks, dtype=float)
        tradable = pd.to_numeric(daily.get("tradable_mask", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
        if name == "cash_no_trade":
            targets[date_text] = target
            continue
        if name == "alpha_prior_target_weight" and "alpha_prior_target_weight" in daily.columns:
            target = pd.to_numeric(daily["alpha_prior_target_weight"], errors="coerce").fillna(0.0).astype(float)
            target.index = stocks
            targets[date_text] = target.clip(lower=0.0)
            continue
        score_column = "score_blend" if name == "score_blend_top30" and "score_blend" in daily.columns else ""
        if name == "liquidity_equal_top30" and "adv20_rank" in daily.columns:
            score = pd.to_numeric(daily["adv20_rank"], errors="coerce").fillna(0.0).astype(float)
        elif score_column:
            score = pd.to_numeric(daily[score_column], errors="coerce").fillna(0.0).astype(float)
        else:
            score = pd.Series(np.arange(len(daily), 0, -1), index=daily.index, dtype=float)
        score.index = stocks
        score = score.where(pd.Series(tradable, index=stocks), -np.inf)
        selected = score.sort_values(ascending=False).head(max(int(max_positions), 1))
        selected = selected[np.isfinite(selected.to_numpy(dtype=float))]
        if len(selected):
            target.loc[selected.index] = 1.0 / float(len(selected))
        targets[date_text] = target
    return targets


def _run_baseline_suite(
    *,
    prepared_by_year: dict[int, Any],
    episode_by_year: dict[int, Path20MarketEpisode],
    study_root: Path,
    args: argparse.Namespace,
    baseline_names: tuple[str, ...] = V4_BASELINES,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for baseline in baseline_names:
        checkpoint_bundle: dict[str, Any] | None = None
        if baseline == "v3_final_checkpoint" and not str(getattr(args, "v3_checkpoint_model_pt", "") or "").strip():
            results[baseline] = {
                "status": "skipped",
                "baseline_name": baseline,
                "reason": "missing_explicit_v3_checkpoint_model_pt",
                "oracle_used": False,
                "shadow_only": True,
                "promotion_allowed": False,
            }
            continue
        if baseline == "v4_gru_selected_checkpoint" and not str(getattr(args, "v4_gru_checkpoint_model_pt", "") or "").strip():
            results[baseline] = {
                "status": "skipped",
                "baseline_name": baseline,
                "reason": "missing_explicit_v4_gru_checkpoint_model_pt",
                "oracle_used": False,
                "shadow_only": True,
                "promotion_allowed": False,
            }
            continue
        if baseline == "v3_final_checkpoint":
            checkpoint_bundle = _load_sequence_policy_checkpoint(
                str(getattr(args, "v3_checkpoint_model_pt", "") or "").strip(),
                args,
                episode_by_year[int(WALKFORWARD_VALIDATION_YEARS[0])],
            )
        if baseline == "v4_gru_selected_checkpoint":
            checkpoint_bundle = _load_sequence_policy_checkpoint(
                str(getattr(args, "v4_gru_checkpoint_model_pt", "") or "").strip(),
                args,
                episode_by_year[int(WALKFORWARD_VALIDATION_YEARS[0])],
            )
        baseline_root = study_root / "baselines" / baseline
        yearly: dict[str, Any] = {}
        for year, role in ((WALKFORWARD_VALIDATION_YEARS[0], "validation"), (WALKFORWARD_TEST_YEARS[0], "test")):
            year_root = baseline_root / str(year)
            year_root.mkdir(parents=True, exist_ok=True)
            episode = episode_by_year[int(year)]
            checkpoint_projection_summary: dict[str, Any] = {}
            if checkpoint_bundle is not None:
                if list(checkpoint_bundle.get("feature_columns", [])) != list(episode.feature_columns):
                    raise ValueError(
                        f"v3 checkpoint feature_columns do not match {year} episode artifact; "
                        "refusing to run a mislabeled baseline."
                    )
                target_by_date, checkpoint_projection_summary = predict_episode_targets(
                    checkpoint_bundle["model"],
                    episode,
                    sequence_length=int(checkpoint_bundle["sequence_length"]),
                    normalization=dict(checkpoint_bundle["normalization"]),
                    model_family=str(checkpoint_bundle["model_family"]),
                    max_position_weight=float(args.max_position_weight),
                    max_gross_exposure=float(args.max_gross_exposure),
                    max_positions=int(args.max_positions),
                    turnover_budget=float(args.turnover_budget),
                )
            else:
                target_by_date = _baseline_targets_for_episode(episode, baseline, max_positions=int(args.max_positions))
            replay_summary = _run_target_replay(
                prepared=prepared_by_year[int(year)],
                episode=episode,
                target_by_date=target_by_date,
                study_root=year_root,
                args=args,
                prefix=f"{baseline}_{year}",
                year=year,
                role=role,
                source_label=f"path20_v4_baseline_{baseline}",
            )
            yearly[str(year)] = {
                "role": role,
                "dataset_summary": episode.manifest,
                "replay_summary": replay_summary,
                "checkpoint_projection_summary": checkpoint_projection_summary,
            }
        results[baseline] = {
            "status": "completed",
            "baseline_name": baseline,
            "checkpoint_model_pt": str(checkpoint_bundle.get("checkpoint_model_pt", "")) if checkpoint_bundle is not None else "",
            "checkpoint_model_family": str(checkpoint_bundle.get("model_family", "")) if checkpoint_bundle is not None else "",
            "yearly": yearly,
            "validation": _role_exact_metrics(yearly, "validation"),
            "test": _role_exact_metrics(yearly, "test"),
            "oracle_used": False,
            "shadow_only": True,
            "promotion_allowed": False,
        }
    write_json(study_root / "baseline_comparison.json", _json_ready(results))
    return _json_ready(results)


def _baseline_validation_return(baseline_result: dict[str, Any]) -> float:
    aggregate = dict(dict(baseline_result.get("validation", {}) or {}).get("aggregate", {}) or {})
    return float(aggregate.get("mean_total_return", 0.0) or 0.0)


def _checkpoint_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, float]:
    metrics = dict(row.get("validation_metrics", {}) or {})
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    projection_l1 = float(metrics.get("avg_projection_l1_distance", 0.0) or 0.0)
    turnover = float(metrics.get("avg_turnover", 0.0) or 0.0)
    return (
        1 if total_return > 0.0 else 0,
        1 if sharpe > 0.0 else 0,
        total_return,
        -projection_l1,
        -turnover,
    )


def _select_v4_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_available", "selection_reason": "no_checkpoints"}
    selected = max(rows, key=_checkpoint_sort_key)
    metrics = dict(selected.get("validation_metrics", {}) or {})
    reason = (
        "positive_validation_return_and_sharpe"
        if float(metrics.get("total_return", 0.0) or 0.0) > 0.0 and float(metrics.get("sharpe", 0.0) or 0.0) > 0.0
        else "best_available_validation_exact_replay"
    )
    return {**selected, "status": "selected", "selection_reason": reason, "test_metrics_used_for_selection": False}


def _v4_multi_seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_available", "seed_count": 0}
    validation_returns = [float(row.get("selected_checkpoint", {}).get("validation_metrics", {}).get("total_return", 0.0) or 0.0) for row in rows]
    validation_sharpes = [float(row.get("selected_checkpoint", {}).get("validation_metrics", {}).get("sharpe", 0.0) or 0.0) for row in rows]
    return {
        "status": "completed",
        "seed_count": int(len(rows)),
        "validation_return_mean": _finite_mean(validation_returns),
        "validation_return_median": float(np.median(validation_returns)) if validation_returns else 0.0,
        "validation_return_std": float(np.std(validation_returns)) if validation_returns else 0.0,
        "validation_sharpe_mean": _finite_mean(validation_sharpes),
        "positive_validation_seed_count": int(sum(1 for value in validation_returns if value > 0.0)),
    }


def _v4_verdict(
    *,
    selected_checkpoint: dict[str, Any],
    baseline_comparison: dict[str, Any],
    projection_parity_summary: dict[str, Any],
) -> str:
    if bool(projection_parity_summary.get("projection_mismatch_warning", False)):
        return "insufficient_or_incomplete"
    metrics = dict(selected_checkpoint.get("validation_metrics", {}) or {})
    validation_return = float(metrics.get("total_return", 0.0) or 0.0)
    validation_sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    best_baseline_return = max((_baseline_validation_return(result) for result in baseline_comparison.values()), default=0.0)
    if validation_return > 0.0 and validation_sharpe > 0.0 and validation_return > best_baseline_return:
        return "validation_promising"
    if selected_checkpoint.get("status") == "selected":
        return "diagnosed_failure"
    return "contract_passed"


def _v5_architecture_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_family": "decision_transformer_v2",
        "architecture": "factorized_causal_decision_transformer",
        "temporal_encoder": "per-stock causal Transformer over market state, previous weights, previous rewards, and portfolio context",
        "cross_sectional_encoder": "Transformer over latest per-stock temporal latents",
        "time_positional_embedding": "learned",
        "stock_slot_embedding": "learned",
        "training_semantics": "pure_policy_utility",
        "action_imitation": False,
        "oracle_used": False,
        "hidden_dim": int(args.rl_hidden_dim),
        "dropout": float(args.rl_dropout),
        "temporal_layers": int(getattr(args, "dt_temporal_layers", 2)),
        "cross_layers": int(getattr(args, "dt_cross_layers", 1)),
        "num_heads": int(getattr(args, "dt_num_heads", 4)),
    }


def _v5_temporal_context_diagnostics(
    rows: list[dict[str, Any]],
    *,
    primary_context_length: int = 60,
    default_primary_context_length: int = 60,
) -> dict[str, Any]:
    completed = [dict(row) for row in rows if row.get("status") == "completed"]
    if not completed:
        return {
            "status": "not_available",
            "context_count": 0,
            "primary_context_length": int(primary_context_length),
            "default_primary_context_length": int(default_primary_context_length),
            "long_context_primary": str(int(primary_context_length)),
            "primary_context_available": False,
            "long_context_available": False,
            "default_60_context_available": False,
        }
    by_context: dict[str, Any] = {}
    for context_length in sorted({int(row.get("context_length", 0) or 0) for row in completed}):
        subset = [row for row in completed if int(row.get("context_length", 0) or 0) == context_length]
        returns = [
            float(row.get("selected_checkpoint", {}).get("validation_metrics", {}).get("total_return", 0.0) or 0.0)
            for row in subset
        ]
        by_context[str(context_length)] = {
            "run_count": int(len(subset)),
            "validation_return_mean": _finite_mean(returns),
            "validation_return_median": float(np.median(returns)) if returns else 0.0,
            "positive_validation_count": int(sum(1 for value in returns if value > 0.0)),
        }
    return {
        "status": "completed",
        "context_count": int(len(by_context)),
        "by_context": by_context,
        "primary_context_length": int(primary_context_length),
        "default_primary_context_length": int(default_primary_context_length),
        "long_context_primary": str(int(primary_context_length)),
        "primary_context_available": str(int(primary_context_length)) in by_context,
        "long_context_available": str(int(primary_context_length)) in by_context,
        "default_60_context_available": "60" in by_context,
    }


def _run_v4_seed_penalty_experiment(
    *,
    prepared_by_year: dict[int, Any],
    episode_by_year: dict[int, Path20MarketEpisode],
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
    seed: int,
    projection_penalty: float,
) -> dict[str, Any]:
    import torch

    _apply_stage_defaults(args)
    _set_research_seed(int(seed))
    experiment_root = study_root / f"seed_{int(seed)}" / f"projection_penalty_{str(float(projection_penalty)).replace('.', '_')}"
    experiment_root.mkdir(parents=True, exist_ok=True)
    train_episodes = [episode_by_year[int(year)] for year in WALKFORWARD_TRAIN_YEARS]
    validation_episode = episode_by_year[int(WALKFORWARD_VALIDATION_YEARS[0])]
    test_episode = episode_by_year[int(WALKFORWARD_TEST_YEARS[0])]
    completed_train = [episode for episode in train_episodes if str(episode.manifest.get("status", "")) == "completed"]
    if not completed_train:
        return {"status": "skipped", "reason": "no_completed_train_episodes"}
    normalization = fit_episode_normalization(completed_train)
    base_feature_count = int(len(completed_train[0].feature_columns))
    stock_feature_dim = base_feature_count + len(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS)
    experiment_args = copy.copy(args)
    experiment_args.model_family = "sequence_gru"
    experiment_args.projection_penalty_weight = float(projection_penalty)
    model = _make_sequence_policy(experiment_args, stock_feature_dim=stock_feature_dim, portfolio_feature_dim=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.smoke_lr), weight_decay=1.0e-4)
    checkpoint_rows: list[dict[str, Any]] = []
    losses: list[float] = []
    raw_intent_rows: list[dict[str, float]] = []
    projection_rows: list[dict[str, float]] = []
    for epoch in range(1, max(int(args.smoke_epochs), 1) + 1):
        epoch_losses: list[float] = []
        for episode in completed_train:
            optimizer.zero_grad(set_to_none=True)
            result = episode_policy_rollout_loss(
                model,
                episode,
                sequence_length=int(args.sequence_length),
                normalization=normalization,
                model_family="sequence_gru",
                transaction_cost_bps=float(args.transaction_cost_bps),
                slippage_bps=float(args.slippage_bps),
                sell_tax_bps=float(args.sell_tax_bps),
                max_position_weight=float(args.max_position_weight),
                max_gross_exposure=float(args.max_gross_exposure),
                max_positions=int(args.max_positions),
                turnover_budget=float(args.turnover_budget),
                turnover_penalty=float(getattr(args, "turnover_penalty_weight", 0.20)),
                projection_penalty=float(projection_penalty),
                tail_mass_penalty=float(getattr(args, "tail_mass_penalty_weight", 0.0)),
                rollout_grad_mode=str(getattr(args, "rollout_grad_mode", "detached")),
                rollout_chunk_days=int(getattr(args, "rollout_chunk_days", 20)),
            )
            if result.get("status") != "completed":
                continue
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            raw_intent_rows.append(dict(result.get("raw_intent_diagnostics", {}) or {}))
            projection_rows.append(dict(result.get("diagnostics", {}) or {}))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        checkpoint_path = experiment_root / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "model_family": "sequence_gru",
                "state_dict": model.state_dict(),
                "seed": int(seed),
                "epoch": int(epoch),
                "projection_penalty_weight": float(projection_penalty),
                "normalization": normalization,
                "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
                "shadow_only": True,
                "promotion_allowed": False,
            },
            checkpoint_path,
        )
        validation_root = experiment_root / "validation_checkpoints" / f"epoch_{epoch}"
        validation_root.mkdir(parents=True, exist_ok=True)
        validation_replay = _run_episode_replay(
            prepared=prepared_by_year[int(WALKFORWARD_VALIDATION_YEARS[0])],
            episode=validation_episode,
            model=model,
            normalization=normalization,
            study_root=validation_root,
            args=experiment_args,
            prefix=f"checkpoint_epoch_{epoch}_validation",
            year=WALKFORWARD_VALIDATION_YEARS[0],
            role="validation",
        )
        checkpoint_rows.append(
            {
                "epoch": int(epoch),
                "checkpoint_pt": str(checkpoint_path.resolve()),
                "train_loss": float(losses[-1]),
                "validation_metrics": validation_replay.get("metrics", {}),
                "validation_replay_summary_json": validation_replay.get("replay_manifest_json", ""),
                "validation_projection_parity_summary": validation_replay.get("projection_parity_summary", {}),
                "projection_tail_mass_summary": _summarize_projection_dicts(raw_intent_rows),
                "test_metrics_used_for_selection": False,
            }
        )
    selected = _select_v4_checkpoint(checkpoint_rows)
    selected_epoch = int(selected.get("epoch", 0) or 0)
    selected_state = (
        torch.load(str(selected.get("checkpoint_pt")), map_location="cpu", weights_only=False)
        if selected.get("checkpoint_pt")
        else None
    )
    if selected_state:
        model.load_state_dict(selected_state["state_dict"])
    selected_validation_passed = (
        float(dict(selected.get("validation_metrics", {}) or {}).get("total_return", 0.0) or 0.0) > 0.0
        and float(dict(selected.get("validation_metrics", {}) or {}).get("sharpe", 0.0) or 0.0) > 0.0
    )
    test_root = experiment_root / "selected_test_replay"
    test_root.mkdir(parents=True, exist_ok=True)
    test_replay = _run_episode_replay(
        prepared=prepared_by_year[int(WALKFORWARD_TEST_YEARS[0])],
        episode=test_episode,
        model=model,
        normalization=normalization,
        study_root=test_root,
        args=experiment_args,
        prefix=f"selected_epoch_{selected_epoch}_test",
        year=WALKFORWARD_TEST_YEARS[0],
        role="test",
    )
    summary = {
        "status": "completed",
        "stage": "rl_v4_seed_projection_penalty_experiment",
        "tag": str(tag),
        "seed": int(seed),
        "model_family": "sequence_gru",
        "projection_penalty_weight": float(projection_penalty),
        "turnover_penalty_weight": float(getattr(args, "turnover_penalty_weight", 0.20)),
        "tail_mass_penalty_weight": float(getattr(args, "tail_mass_penalty_weight", 0.0)),
        "train_years": list(WALKFORWARD_TRAIN_YEARS),
        "validation_years": list(WALKFORWARD_VALIDATION_YEARS),
        "test_years": list(WALKFORWARD_TEST_YEARS),
        "train_loss_curve": losses,
        "checkpoint_exact_replay": checkpoint_rows,
        "selected_checkpoint": selected,
        "selection_reason": str(selected.get("selection_reason", "")),
        "validation_passed": bool(selected_validation_passed),
        "test_interpretable": bool(selected_validation_passed),
        "test_exact_replay_metrics": test_replay.get("metrics", {}),
        "test_replay_summary": test_replay,
        "projection_tail_mass_summary": _summarize_projection_dicts(raw_intent_rows),
        "projection_diagnostics_summary": _summarize_projection_dicts(projection_rows),
        "projection_parity_summary": _aggregate_projection_parity_summaries(
            [
                dict(row.get("validation_projection_parity_summary", {}) or {})
                for row in checkpoint_rows
            ]
            + [dict(test_replay.get("projection_parity_summary", {}) or {})]
        ),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(experiment_root / "v4_seed_projection_penalty_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_v4_validation_repair_study(*, study_root: Path, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    _apply_stage_defaults(args)
    years = [*WALKFORWARD_TRAIN_YEARS, *WALKFORWARD_VALIDATION_YEARS, *WALKFORWARD_TEST_YEARS]
    prepared_by_year: dict[int, Any] = {}
    episode_by_year: dict[int, Path20MarketEpisode] = {}
    yearly: dict[str, Any] = {}
    for year in years:
        start_date, end_date = full_year_window(year)
        year_root = study_root / "episodes" / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        prepared = _prepare_for_window(args, start_date=start_date, end_date=end_date, tag=f"{tag}_{year}")
        prepared_by_year[int(year)] = prepared
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
        episode_by_year[int(year)] = episode
        yearly[str(year)] = {
            "role": _walkforward_role(year),
            "dataset_summary": dataset_payload["manifest"],
            "array_manifest": dataset_payload.get("array_manifest", {}),
            "prepared_summary": prepared.to_summary(),
        }
    baseline_comparison = _run_baseline_suite(
        prepared_by_year=prepared_by_year,
        episode_by_year=episode_by_year,
        study_root=study_root,
        args=args,
    )
    seeds = _parse_int_list(str(getattr(args, "rl_seed_matrix", "")), default=V4_DEFAULT_SEEDS)
    projection_grid = _parse_float_list(
        str(getattr(args, "projection_penalty_grid", "")),
        default=V4_PROJECTION_PENALTY_GRID,
    )
    experiment_results: list[dict[str, Any]] = []
    for seed in seeds:
        for projection_penalty in projection_grid:
            experiment_results.append(
                _run_v4_seed_penalty_experiment(
                    prepared_by_year=prepared_by_year,
                    episode_by_year=episode_by_year,
                    study_root=study_root,
                    tag=tag,
                    args=args,
                    seed=int(seed),
                    projection_penalty=float(projection_penalty),
                )
            )
    best_experiment = max(
        experiment_results,
        key=lambda row: _checkpoint_sort_key(dict(row.get("selected_checkpoint", {}) or {})),
        default={},
    )
    best_selected = dict(best_experiment.get("selected_checkpoint", {}) or {})
    baseline_returns = {
        name: _baseline_validation_return(dict(result))
        for name, result in baseline_comparison.items()
    }
    best_baseline_return = max(baseline_returns.values(), default=0.0)
    validation_metrics = dict(best_selected.get("validation_metrics", {}) or {})
    validation_passed = (
        float(validation_metrics.get("total_return", 0.0) or 0.0) > 0.0
        and float(validation_metrics.get("sharpe", 0.0) or 0.0) > 0.0
        and float(validation_metrics.get("total_return", 0.0) or 0.0) > float(best_baseline_return)
    )
    projection_parity_summary = _aggregate_projection_parity_summaries(
        [dict(result.get("projection_parity_summary", {}) or {}) for result in experiment_results]
    )
    summary = {
        "status": "completed",
        "stage": "rl_v4_validation_repair_study",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": "sequence_gru",
        "train_years": list(WALKFORWARD_TRAIN_YEARS),
        "validation_years": list(WALKFORWARD_VALIDATION_YEARS),
        "test_years": list(WALKFORWARD_TEST_YEARS),
        "seeds": seeds,
        "projection_penalty_grid": projection_grid,
        "yearly": yearly,
        "baseline_comparison": baseline_comparison,
        "checkpoint_exact_replay": experiment_results,
        "selected_checkpoint": best_selected,
        "selection_reason": str(best_selected.get("selection_reason", "")),
        "best_experiment": best_experiment,
        "multi_seed_summary": _v4_multi_seed_summary(experiment_results),
        "projection_tail_mass_summary": _summarize_projection_dicts(
            [
                dict(result.get("projection_tail_mass_summary", {}) or {})
                for result in experiment_results
                if result.get("status") == "completed"
            ]
        ),
        "validation_passed": bool(validation_passed),
        "test_interpretable": bool(validation_passed),
        "baseline_validation_returns": baseline_returns,
        "evidence_verdict": _v4_verdict(
            selected_checkpoint=best_selected,
            baseline_comparison=baseline_comparison,
            projection_parity_summary=projection_parity_summary,
        ),
        "projection_parity_summary": projection_parity_summary,
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / "v4_validation_repair_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_v5_dt_context_experiment(
    *,
    prepared_by_year: dict[int, Any],
    episode_by_year: dict[int, Path20MarketEpisode],
    study_root: Path,
    tag: str,
    args: argparse.Namespace,
    seed: int,
    context_length: int,
) -> dict[str, Any]:
    import torch

    _apply_stage_defaults(args)
    _set_research_seed(int(seed))
    experiment_root = study_root / f"context_{int(context_length)}" / f"seed_{int(seed)}"
    experiment_root.mkdir(parents=True, exist_ok=True)
    train_episodes = [episode_by_year[int(year)] for year in WALKFORWARD_TRAIN_YEARS]
    validation_episode = episode_by_year[int(WALKFORWARD_VALIDATION_YEARS[0])]
    test_episode = episode_by_year[int(WALKFORWARD_TEST_YEARS[0])]
    completed_train = [episode for episode in train_episodes if str(episode.manifest.get("status", "")) == "completed"]
    if not completed_train:
        return {"status": "skipped", "reason": "no_completed_train_episodes", "context_length": int(context_length), "seed": int(seed)}
    min_required = int(context_length)
    incomplete_years = [
        int(episode.manifest.get("year", 0) or 0)
        for episode in [*completed_train, validation_episode, test_episode]
        if int(episode.manifest.get("trading_day_count", 0) or 0) < min_required
    ]
    if incomplete_years:
        return {
            "status": "incomplete",
            "reason": "trading_day_count_below_context_length",
            "context_length": int(context_length),
            "seed": int(seed),
            "incomplete_years": incomplete_years,
            "shadow_only": True,
            "promotion_allowed": False,
        }
    normalization = fit_episode_normalization(completed_train)
    stock_feature_dim = int(len(completed_train[0].feature_columns) + len(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS))
    experiment_args = copy.copy(args)
    experiment_args.model_family = "decision_transformer_v2"
    experiment_args.sequence_length = int(context_length)
    model = _make_sequence_policy(experiment_args, stock_feature_dim=stock_feature_dim, portfolio_feature_dim=len(EPISODE_PORTFOLIO_FEATURE_COLUMNS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.smoke_lr), weight_decay=1.0e-4)
    checkpoint_rows: list[dict[str, Any]] = []
    losses: list[float] = []
    context_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, float]] = []
    projection_rows: list[dict[str, float]] = []
    for epoch in range(1, max(int(args.smoke_epochs), 1) + 1):
        epoch_losses: list[float] = []
        for episode in completed_train:
            optimizer.zero_grad(set_to_none=True)
            result = episode_policy_rollout_loss(
                model,
                episode,
                sequence_length=int(context_length),
                normalization=normalization,
                model_family="decision_transformer_v2",
                transaction_cost_bps=float(args.transaction_cost_bps),
                slippage_bps=float(args.slippage_bps),
                sell_tax_bps=float(args.sell_tax_bps),
                max_position_weight=float(args.max_position_weight),
                max_gross_exposure=float(args.max_gross_exposure),
                max_positions=int(args.max_positions),
                turnover_budget=float(args.turnover_budget),
                turnover_penalty=float(getattr(args, "turnover_penalty_weight", 0.20)),
                projection_penalty=float(getattr(args, "projection_penalty_weight", 0.05)),
                tail_mass_penalty=float(getattr(args, "tail_mass_penalty_weight", 0.0)),
                rollout_grad_mode=str(getattr(args, "rollout_grad_mode", "detached")),
                rollout_chunk_days=int(getattr(args, "rollout_chunk_days", 20)),
            )
            if result.get("status") != "completed":
                continue
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(args, "dt_grad_clip", 1.0)))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            context_rows.append(dict(result.get("context_coverage", {}) or {}))
            raw_rows.append(dict(result.get("raw_intent_diagnostics", {}) or {}))
            projection_rows.append(dict(result.get("diagnostics", {}) or {}))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        checkpoint_path = experiment_root / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "model_family": "decision_transformer_v2",
                "state_dict": model.state_dict(),
                "feature_columns": completed_train[0].feature_columns,
                "dynamic_stock_feature_columns": list(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS),
                "sequence_length": int(context_length),
                "normalization": normalization,
                "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
                "architecture_contract": _v5_architecture_contract(experiment_args),
                "shadow_only": True,
                "promotion_allowed": False,
            },
            checkpoint_path,
        )
        validation_root = experiment_root / "validation_checkpoints" / f"epoch_{epoch}"
        validation_root.mkdir(parents=True, exist_ok=True)
        validation_replay = _run_episode_replay(
            prepared=prepared_by_year[int(WALKFORWARD_VALIDATION_YEARS[0])],
            episode=validation_episode,
            model=model,
            normalization=normalization,
            study_root=validation_root,
            args=experiment_args,
            prefix=f"checkpoint_epoch_{epoch}_validation",
            year=WALKFORWARD_VALIDATION_YEARS[0],
            role="validation",
        )
        checkpoint_rows.append(
            {
                "epoch": int(epoch),
                "checkpoint_pt": str(checkpoint_path.resolve()),
                "train_loss": float(losses[-1]),
                "validation_metrics": validation_replay.get("metrics", {}),
                "validation_replay_summary_json": validation_replay.get("replay_manifest_json", ""),
                "validation_projection_parity_summary": validation_replay.get("projection_parity_summary", {}),
                "test_metrics_used_for_selection": False,
            }
        )
    selected = _select_v4_checkpoint(checkpoint_rows)
    selected_state = (
        torch.load(str(selected.get("checkpoint_pt")), map_location="cpu", weights_only=False)
        if selected.get("checkpoint_pt")
        else None
    )
    if selected_state:
        model.load_state_dict(selected_state["state_dict"])
    selected_validation_passed = (
        float(dict(selected.get("validation_metrics", {}) or {}).get("total_return", 0.0) or 0.0) > 0.0
        and float(dict(selected.get("validation_metrics", {}) or {}).get("sharpe", 0.0) or 0.0) > 0.0
    )
    test_root = experiment_root / "selected_test_replay"
    test_root.mkdir(parents=True, exist_ok=True)
    test_replay = _run_episode_replay(
        prepared=prepared_by_year[int(WALKFORWARD_TEST_YEARS[0])],
        episode=test_episode,
        model=model,
        normalization=normalization,
        study_root=test_root,
        args=experiment_args,
        prefix=f"selected_epoch_{int(selected.get('epoch', 0) or 0)}_test",
        year=WALKFORWARD_TEST_YEARS[0],
        role="test",
    )
    summary = {
        "status": "completed",
        "stage": "rl_v5_dt_context_experiment",
        "tag": str(tag),
        "seed": int(seed),
        "context_length": int(context_length),
        "model_family": "decision_transformer_v2",
        "architecture_contract": _v5_architecture_contract(experiment_args),
        "train_loss_curve": losses,
        "checkpoint_exact_validation_replay": checkpoint_rows,
        "selected_checkpoint": selected,
        "selection_reason": str(selected.get("selection_reason", "")),
        "validation_passed": bool(selected_validation_passed),
        "test_interpretable": bool(selected_validation_passed),
        "test_exact_replay_metrics": test_replay.get("metrics", {}),
        "test_replay_summary": test_replay,
        "context_coverage": _summarize_context_coverage(context_rows),
        "raw_intent_diagnostics_summary": _summarize_projection_dicts(raw_rows),
        "projection_diagnostics_summary": _summarize_projection_dicts(projection_rows),
        "projection_parity_summary": _aggregate_projection_parity_summaries(
            [
                dict(row.get("validation_projection_parity_summary", {}) or {})
                for row in checkpoint_rows
            ]
            + [dict(test_replay.get("projection_parity_summary", {}) or {})]
        ),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(experiment_root / "v5_dt_context_experiment_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_v5_dt_validation_study(*, study_root: Path, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    _apply_stage_defaults(args)
    years = [*WALKFORWARD_TRAIN_YEARS, *WALKFORWARD_VALIDATION_YEARS, *WALKFORWARD_TEST_YEARS]
    prepared_by_year: dict[int, Any] = {}
    episode_by_year: dict[int, Path20MarketEpisode] = {}
    yearly: dict[str, Any] = {}
    context_grid = _parse_int_list(str(getattr(args, "dt_context_grid", "")), default=V5_DEFAULT_CONTEXT_GRID)
    max_context = max(context_grid) if context_grid else 60
    primary_context_length = 60 if 60 in set(context_grid) else int(max_context)
    for year in years:
        start_date, end_date = full_year_window(year)
        year_root = study_root / "episodes" / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        prepared = _prepare_for_window(args, start_date=start_date, end_date=end_date, tag=f"{tag}_{year}")
        prepared_by_year[int(year)] = prepared
        dataset_args = copy.copy(args)
        dataset_args.sequence_length = int(max_context)
        dataset_payload = _build_episode_dataset_artifact(
            prepared=prepared,
            study_root=year_root,
            tag=f"{tag}_{year}",
            args=dataset_args,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )
        episode = dataset_payload["episode"]
        episode_by_year[int(year)] = episode
        yearly[str(year)] = {
            "role": _walkforward_role(year),
            "dataset_summary": dataset_payload["manifest"],
            "array_manifest": dataset_payload.get("array_manifest", {}),
            "prepared_summary": prepared.to_summary(),
        }
    baseline_comparison = _run_baseline_suite(
        prepared_by_year=prepared_by_year,
        episode_by_year=episode_by_year,
        study_root=study_root,
        args=args,
        baseline_names=V5_BASELINES,
    )
    seeds = _parse_int_list(str(getattr(args, "rl_seed_matrix", "")), default=V4_DEFAULT_SEEDS)
    experiment_results: list[dict[str, Any]] = []
    for context_length in context_grid:
        for seed in seeds:
            experiment_results.append(
                _run_v5_dt_context_experiment(
                    prepared_by_year=prepared_by_year,
                    episode_by_year=episode_by_year,
                    study_root=study_root,
                    tag=tag,
                    args=args,
                    seed=int(seed),
                    context_length=int(context_length),
                )
            )
    completed_results = [row for row in experiment_results if row.get("status") == "completed"]
    primary_results = [row for row in completed_results if int(row.get("context_length", 0) or 0) == int(primary_context_length)]
    selection_pool = primary_results or completed_results
    best_experiment = max(
        selection_pool,
        key=lambda row: _checkpoint_sort_key(dict(row.get("selected_checkpoint", {}) or {})),
        default={},
    )
    best_selected = dict(best_experiment.get("selected_checkpoint", {}) or {})
    baseline_returns = {name: _baseline_validation_return(dict(result)) for name, result in baseline_comparison.items()}
    best_baseline_return = max(baseline_returns.values(), default=0.0)
    validation_metrics = dict(best_selected.get("validation_metrics", {}) or {})
    validation_passed = (
        float(validation_metrics.get("total_return", 0.0) or 0.0) > 0.0
        and float(validation_metrics.get("sharpe", 0.0) or 0.0) > 0.0
        and float(validation_metrics.get("total_return", 0.0) or 0.0) > float(best_baseline_return)
    )
    projection_parity_summary = _aggregate_projection_parity_summaries(
        [dict(result.get("projection_parity_summary", {}) or {}) for result in completed_results]
    )
    context_diagnostics = _v5_temporal_context_diagnostics(
        experiment_results,
        primary_context_length=int(primary_context_length),
        default_primary_context_length=60,
    )
    summary = {
        "status": "completed",
        "stage": "rl_v5_dt_validation_study",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family": "decision_transformer_v2",
        "architecture_contract": _v5_architecture_contract(args),
        "train_years": list(WALKFORWARD_TRAIN_YEARS),
        "validation_years": list(WALKFORWARD_VALIDATION_YEARS),
        "test_years": list(WALKFORWARD_TEST_YEARS),
        "seeds": seeds,
        "context_grid": context_grid,
        "primary_context_length": int(primary_context_length),
        "default_primary_context_length": 60,
        "yearly": yearly,
        "baseline_comparison": baseline_comparison,
        "context_length_comparison": experiment_results,
        "checkpoint_exact_validation_replay": experiment_results,
        "selected_checkpoint": best_selected,
        "selection_reason": str(best_selected.get("selection_reason", "")),
        "best_experiment": best_experiment,
        "multi_seed_summary": _v4_multi_seed_summary(completed_results),
        "temporal_context_diagnostics": context_diagnostics,
        "validation_passed": bool(validation_passed),
        "test_interpretable": bool(validation_passed),
        "baseline_validation_returns": baseline_returns,
        "projection_parity_summary": projection_parity_summary,
        "evidence_verdict": _v4_verdict(
            selected_checkpoint=best_selected,
            baseline_comparison=baseline_comparison,
            projection_parity_summary=projection_parity_summary,
        ),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    if not bool(context_diagnostics.get("primary_context_available", False)):
        summary["evidence_verdict"] = "insufficient_or_incomplete"
        summary["long_context_incomplete_warning"] = True
    write_json(study_root / "v5_dt_validation_summary.json", _json_ready(summary))
    return _json_ready(summary)


def _run_sequence_train_smoke(
    trajectory: Path20TrajectoryDataset,
    *,
    study_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    _apply_stage_defaults(args)
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
        "attribution_csv": _write_frame(study_root / f"{prefix}_rl_attribution.csv", rollout["attribution_frame"]),
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
        "attribution_summary": _summarize_replay_attribution(rollout["attribution_frame"]),
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
        pool_view_id=str(getattr(args, "pool_view_id", "") or ""),
        pool_view_spec=_pool_view_spec_from_args(args, start_date=start_date, end_date=end_date),
        sector_board_view_id=str(getattr(args, "sector_board_view_id", "") or ""),
        sector_board_view_spec=_sector_board_view_spec_from_args(args),
        max_universe_size=int(args.max_universe_size),
        alpha_prior_source=DEFAULT_ALPHA_PRIOR_SOURCE,
        require_lake_benchmark_open=str(args.execution_mode or "next_open").strip().lower() == "next_open",
        progress_desc=f"alpha_path20 sequence prepare {tag}",
    )


def _pool_view_spec_from_args(args: argparse.Namespace, *, start_date: str, end_date: str) -> dict[str, Any] | None:
    if str(getattr(args, "pool_view_id", "") or "").strip():
        return None
    view_kind = str(getattr(args, "pool_view_kind", "") or "").strip().lower()
    if not view_kind:
        return None
    view_name = str(getattr(args, "pool_view_name", "") or "").strip().lower() or view_kind
    spec: dict[str, Any] = {
        "source_market_dataset_id": str(getattr(args, "lake_dataset_id", "") or ""),
        "view_kind": view_kind,
        "view_name": view_name,
        "start_date": str(start_date or ""),
        "end_date": str(end_date or ""),
    }
    if view_kind == "rolling_liquidity":
        pool_name = view_name.removeprefix("rolling_")
        spec.update({"pool_name": pool_name or str(getattr(args, "pool_name", "") or "")})
    if view_kind == "exchange":
        suffix = view_name.removeprefix("exchange_")
        spec["exchange_suffix"] = f".{suffix.upper()}" if suffix else ""
    exclude_symbol_prefixes = tuple(
        item.strip().upper()
        for item in str(getattr(args, "pool_view_exclude_symbol_prefixes", "") or "").split(",")
        if item.strip()
    )
    if exclude_symbol_prefixes:
        spec["exclude_symbol_prefixes"] = exclude_symbol_prefixes
    return spec


def _sector_board_view_spec_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if str(getattr(args, "sector_board_view_id", "") or "").strip():
        return None
    view_kind = str(getattr(args, "sector_board_view_kind", "") or "").strip().lower()
    as_of_date = str(getattr(args, "sector_board_as_of_date", "") or "").strip()
    industry_source_path = str(getattr(args, "sector_board_industry_source_path", "") or "").strip()
    board_source_path = str(getattr(args, "sector_board_source_path", "") or "").strip()
    if not any([view_kind, as_of_date, industry_source_path, board_source_path]):
        return None
    return {
        "source_market_dataset_id": str(getattr(args, "lake_dataset_id", "") or ""),
        "view_kind": view_kind or "latest_static_snapshot",
        "view_name": "sector_board_latest_static",
        "snapshot_semantics": "latest_static_snapshot",
        "as_of_date": as_of_date,
        **({"industry_source_path": industry_source_path} if industry_source_path else {}),
        **({"board_source_path": board_source_path} if board_source_path else {}),
    }


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
    _apply_stage_defaults(args)
    years = [*WALKFORWARD_TRAIN_YEARS, *WALKFORWARD_VALIDATION_YEARS, *WALKFORWARD_TEST_YEARS]
    yearly: dict[str, Any] = {}
    train_episodes: list[Path20MarketEpisode] = []
    validation_episodes: list[Path20MarketEpisode] = []
    prepared_by_year: dict[int, Any] = {}
    episode_by_year: dict[int, Path20MarketEpisode] = {}
    for year in years:
        start_date, end_date = full_year_window(year)
        year_root = study_root / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        prepared = _prepare_for_window(args, start_date=start_date, end_date=end_date, tag=f"{tag}_{year}")
        prepared_by_year[int(year)] = prepared
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
        episode_by_year[int(year)] = episode
        yearly[str(year)] = {
            "role": _walkforward_role(year),
            "dataset_summary": dataset_payload["manifest"],
            "array_manifest": dataset_payload.get("array_manifest", {}),
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
            year_root = study_root / year_text
            replay_summary = _run_episode_replay(
                prepared=prepared_by_year[int(year)],
                episode=episode_by_year[int(year)],
                model=train_payload["model"],
                normalization=train_payload["normalization"],
                study_root=year_root,
                args=args,
                prefix=f"episode_{year}",
                year=year,
                role=_walkforward_role(year),
            )
            yearly[year_text]["train_summary"] = train_summary if year in WALKFORWARD_TRAIN_YEARS else {"status": "not_train_year"}
            yearly[year_text]["replay_summary"] = replay_summary
    aggregate = _walkforward_aggregate(yearly)
    projection_parity_summary = _walkforward_projection_parity(yearly)
    surrogate_exact_gap = _surrogate_exact_gap(train_summary, yearly)
    exact_validation_metrics = _role_exact_metrics(yearly, "validation")
    exact_test_metrics = _role_exact_metrics(yearly, "test")
    evidence_diagnostics = _walkforward_evidence_diagnostics(
        yearly=yearly,
        train_summary=train_summary,
        aggregate=aggregate,
        projection_parity_summary=projection_parity_summary,
        surrogate_exact_gap=surrogate_exact_gap,
    )
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
        "projection_parity_summary": projection_parity_summary,
        "validation_surrogate_metrics": train_summary.get("validation_surrogate_metrics", {}),
        "validation_exact_replay_metrics": exact_validation_metrics,
        "test_exact_replay_metrics": exact_test_metrics,
        "surrogate_exact_gap": surrogate_exact_gap,
        "evidence_diagnostics": evidence_diagnostics,
        "rollout_grad_mode": str(getattr(args, "rollout_grad_mode", "detached")),
        "rollout_chunk_days": int(getattr(args, "rollout_chunk_days", 20)),
        "evidence_verdict": _walkforward_evidence_verdict(
            aggregate=aggregate,
            projection_parity_summary=projection_parity_summary,
            validation_exact_replay_metrics=exact_validation_metrics,
            test_exact_replay_metrics=exact_test_metrics,
            evidence_diagnostics=evidence_diagnostics,
        ),
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


def _walkforward_projection_parity(yearly: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for year, result in yearly.items():
        replay_summary = dict(result.get("replay_summary", {}) or {})
        parity = dict(replay_summary.get("projection_parity_summary", {}) or {})
        if not parity:
            continue
        rows.append({"year": str(year), "role": str(result.get("role", "")), **parity})
    warning = any(bool(row.get("projection_mismatch_warning", False)) for row in rows)
    gaps = [float(row.get("max_target_l1_gap", 0.0) or 0.0) for row in rows]
    return {
        "status": "warning" if warning else ("passed" if rows else "not_run"),
        "projection_mismatch_warning": bool(warning),
        "year_count": int(len(rows)),
        "max_target_l1_gap": float(max(gaps) if gaps else 0.0),
        "mean_max_target_l1_gap": _finite_mean(gaps),
        "by_year": rows,
    }


def _role_exact_metrics(yearly: dict[str, Any], role: str) -> dict[str, Any]:
    role_years = {
        year: result
        for year, result in yearly.items()
        if str(result.get("role", "")) == str(role)
        and str(dict(result.get("dataset_summary", {}) or {}).get("status", "")) == "completed"
        and str(dict(result.get("replay_summary", {}) or {}).get("status", "")) == "completed"
    }
    if not role_years:
        return {"status": "not_available", "role": str(role), "year_count": 0}
    aggregate = _multiyear_aggregate(role_years)
    by_year = {
        str(year): dict(dict(result.get("replay_summary", {}) or {}).get("metrics", {}) or {})
        for year, result in role_years.items()
    }
    return {"status": "completed", "role": str(role), "year_count": int(len(role_years)), "aggregate": aggregate, "by_year": by_year}


def _surrogate_exact_gap(train_summary: dict[str, Any], yearly: dict[str, Any]) -> dict[str, Any]:
    surrogate = dict(train_summary.get("validation_surrogate_metrics", {}) or {})
    exact = _role_exact_metrics(yearly, "validation")
    if surrogate.get("status") != "completed" or exact.get("status") != "completed":
        return {"status": "not_available", "reason": "missing_surrogate_or_exact_validation"}
    surrogate_diag = dict(surrogate.get("diagnostics", {}) or {})
    exact_aggregate = dict(exact.get("aggregate", {}) or {})
    return {
        "status": "completed",
        "surrogate_loss": float(surrogate.get("loss", 0.0) or 0.0),
        "surrogate_avg_projection_l1_distance": float(surrogate_diag.get("avg_projection_l1_distance", 0.0) or 0.0),
        "exact_mean_total_return": float(exact_aggregate.get("mean_total_return", 0.0) or 0.0),
        "exact_mean_avg_projection_l1_distance": float(exact_aggregate.get("mean_avg_projection_l1_distance", 0.0) or 0.0),
        "projection_l1_gap": float(
            (surrogate_diag.get("avg_projection_l1_distance", 0.0) or 0.0)
            - (exact_aggregate.get("mean_avg_projection_l1_distance", 0.0) or 0.0)
        ),
    }


def _role_attribution_summary(yearly: dict[str, Any], role: str) -> dict[str, Any]:
    rows = [
        dict(dict(result.get("replay_summary", {}) or {}).get("attribution_summary", {}) or {})
        for result in yearly.values()
        if str(result.get("role", "")) == str(role)
        and str(dict(result.get("replay_summary", {}) or {}).get("status", "")) == "completed"
    ]
    rows = [row for row in rows if row.get("status") == "completed"]
    if not rows:
        return {"status": "not_available", "role": str(role), "year_count": 0}
    keys = [
        "gross_return_sum",
        "estimated_cost_sum",
        "net_return_sum",
        "benchmark_return_sum",
        "excess_return_sum",
        "avg_cash_weight",
        "avg_projected_gross_exposure",
        "avg_projected_turnover",
        "avg_projection_l1_distance",
        "avg_target_count",
    ]
    result: dict[str, Any] = {"status": "completed", "role": str(role), "year_count": int(len(rows))}
    for key in keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        if key.endswith("_sum"):
            result[key] = _finite_sum(values)
        else:
            result[key] = _finite_mean(values)
    return result


def _walkforward_evidence_diagnostics(
    *,
    yearly: dict[str, Any],
    train_summary: dict[str, Any],
    aggregate: dict[str, Any],
    projection_parity_summary: dict[str, Any],
    surrogate_exact_gap: dict[str, Any],
) -> dict[str, Any]:
    train_agg = dict(aggregate.get("train", {}) or {})
    validation_agg = dict(aggregate.get("validation", {}) or {})
    test_agg = dict(aggregate.get("test", {}) or {})
    validation_return = _safe_metric(validation_agg, "mean_total_return")
    train_return = _safe_metric(train_agg, "mean_total_return")
    test_return = _safe_metric(test_agg, "mean_total_return")
    validation_sharpe = _safe_metric(validation_agg, "mean_sharpe")
    validation_gross = _safe_metric(validation_agg, "mean_avg_projected_gross_exposure")
    validation_projection_l1 = _safe_metric(validation_agg, "mean_avg_projection_l1_distance")
    validation_turnover = _safe_metric(validation_agg, "mean_avg_turnover")
    raw_diag = dict(train_summary.get("raw_intent_diagnostics_summary", {}) or {})
    train_projection_diag = dict(train_summary.get("projection_diagnostics_summary", {}) or {})
    surrogate_gap = float(surrogate_exact_gap.get("projection_l1_gap", 0.0) or 0.0) if surrogate_exact_gap.get("status") == "completed" else 0.0
    train_validation_gap = (
        float(validation_return - train_return)
        if validation_return is not None and train_return is not None
        else None
    )
    validation_test_gap = (
        float(test_return - validation_return)
        if validation_return is not None and test_return is not None
        else None
    )
    attribution_by_role = {
        role: _role_attribution_summary(yearly, role)
        for role in ("train", "validation", "test")
    }
    warnings = {
        "under_exposure_warning": bool(validation_gross is not None and validation_gross < 0.10),
        "projection_overcorrection_warning": bool(
            validation_projection_l1 is not None
            and validation_gross is not None
            and validation_projection_l1 > max(validation_gross, 1.0e-8)
        ),
        "surrogate_exact_gap_warning": bool(abs(surrogate_gap) > 0.02),
        "validation_generalization_failure": bool(validation_return is not None and validation_return <= 0.0),
        "projection_mismatch_warning": bool(projection_parity_summary.get("projection_mismatch_warning", False)),
    }
    diagnosed = any(
        warnings[key]
        for key in (
            "under_exposure_warning",
            "projection_overcorrection_warning",
            "surrogate_exact_gap_warning",
            "validation_generalization_failure",
        )
    )
    return {
        "status": "diagnosed" if diagnosed else "clean",
        "warnings": warnings,
        "train_validation_return_gap": train_validation_gap,
        "validation_test_return_gap": validation_test_gap,
        "validation_return": validation_return,
        "validation_sharpe": validation_sharpe,
        "validation_avg_turnover": validation_turnover,
        "validation_avg_projected_gross_exposure": validation_gross,
        "validation_avg_projection_l1_distance": validation_projection_l1,
        "raw_intent_summary": raw_diag,
        "projected_training_summary": train_projection_diag,
        "attribution_by_role": attribution_by_role,
        "next_research_actions": _diagnostic_next_actions(warnings),
    }


def _diagnostic_next_actions(warnings: dict[str, bool]) -> list[str]:
    actions: list[str] = []
    if warnings.get("under_exposure_warning"):
        actions.append("Investigate cash/gross regularization and policy output scale; validation exposure is below research-useful range.")
    if warnings.get("projection_overcorrection_warning"):
        actions.append("Reduce raw/projected mismatch before model changes; projection is dominating neural intent.")
    if warnings.get("surrogate_exact_gap_warning"):
        actions.append("Improve surrogate-exact alignment or train selection on exact replay diagnostics.")
    if warnings.get("validation_generalization_failure"):
        actions.append("Do not interpret positive test return without validation repair; inspect 2022 regime and train-validation feature drift.")
    if not actions:
        actions.append("No blocking diagnostic warning; continue with controlled training/reward ablation.")
    return actions


def _walkforward_evidence_verdict(
    *,
    aggregate: dict[str, Any],
    projection_parity_summary: dict[str, Any],
    validation_exact_replay_metrics: dict[str, Any],
    test_exact_replay_metrics: dict[str, Any],
    evidence_diagnostics: dict[str, Any] | None = None,
) -> str:
    validation_aggregate = dict(aggregate.get("validation", {}) or {})
    test_aggregate = dict(aggregate.get("test", {}) or {})
    if bool(projection_parity_summary.get("projection_mismatch_warning", False)):
        return "insufficient_or_incomplete"
    if int(validation_aggregate.get("completed_year_count", 0) or 0) < len(WALKFORWARD_VALIDATION_YEARS):
        return "insufficient_or_incomplete"
    validation_return = _safe_metric(dict(validation_aggregate), "mean_total_return")
    validation_drawdown = _safe_metric(dict(validation_aggregate), "worst_max_drawdown")
    validation_sharpe = _safe_metric(dict(validation_aggregate), "mean_sharpe")
    validation_promising = (
        validation_return is not None
        and validation_return > 0.0
        and validation_sharpe is not None
        and validation_sharpe > 0.0
        and validation_drawdown is not None
    )
    if not validation_promising:
        diagnostics = dict(evidence_diagnostics or {})
        warnings = dict(diagnostics.get("warnings", {}) or {})
        if any(
            bool(warnings.get(key, False))
            for key in (
                "under_exposure_warning",
                "projection_overcorrection_warning",
                "surrogate_exact_gap_warning",
                "validation_generalization_failure",
            )
        ):
            return "diagnosed_failure"
        return "contract_passed"
    if int(test_aggregate.get("completed_year_count", 0) or 0) < len(WALKFORWARD_TEST_YEARS):
        return "validation_promising"
    test_return = _safe_metric(dict(test_aggregate), "mean_total_return")
    if test_return is not None and test_return > 0.0 and validation_exact_replay_metrics.get("status") == "completed" and test_exact_replay_metrics.get("status") == "completed":
        return "test_promising"
    return "validation_promising"


def _matrix_evidence_verdict(model_results: dict[str, Any]) -> str:
    if not model_results:
        return "insufficient_or_incomplete"
    verdicts = [
        str(dict(result.get("walkforward_summary", {}) or {}).get("evidence_verdict", "insufficient_or_incomplete"))
        for result in model_results.values()
    ]
    if any(verdict == "test_promising" for verdict in verdicts):
        return "test_promising"
    if any(verdict == "validation_promising" for verdict in verdicts):
        return "validation_promising"
    if any(verdict == "diagnosed_failure" for verdict in verdicts):
        return "diagnosed_failure"
    if any(verdict == "contract_passed" for verdict in verdicts):
        return "contract_passed"
    return "insufficient_or_incomplete"


def _matrix_evidence_diagnostics(model_results: dict[str, Any]) -> dict[str, Any]:
    if not model_results:
        return {"status": "not_available", "model_count": 0}
    warnings_by_model = {
        family: dict(dict(result.get("evidence_diagnostics", {}) or {}).get("warnings", {}) or {})
        for family, result in model_results.items()
    }
    active_warnings = sorted(
        {
            key
            for warnings in warnings_by_model.values()
            for key, value in warnings.items()
            if bool(value)
        }
    )
    validation_returns: dict[str, float | None] = {}
    for family, result in model_results.items():
        walk_summary = dict(result.get("walkforward_summary", {}) or {})
        validation_agg = dict(dict(walk_summary.get("aggregate", {}) or {}).get("validation", {}) or {})
        validation_returns[family] = _safe_metric(validation_agg, "mean_total_return")
    best_family = max(
        (family for family, value in validation_returns.items() if value is not None),
        key=lambda family: float(validation_returns[family]),
        default="",
    )
    return {
        "status": "diagnosed" if active_warnings else "clean",
        "model_count": int(len(model_results)),
        "warnings_by_model": warnings_by_model,
        "active_warnings": active_warnings,
        "validation_returns": validation_returns,
        "best_validation_family": best_family,
    }


def _clone_args_with_model_family(args: argparse.Namespace, family: str) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["model_family"] = str(family)
    return argparse.Namespace(**payload)


def _run_walkforward_matrix(*, study_root: Path, tag: str, args: argparse.Namespace) -> dict[str, Any]:
    _apply_stage_defaults(args)
    families = [
        item.strip()
        for item in str(getattr(args, "model_family_matrix", "sequence_gru,decision_transformer")).split(",")
        if item.strip()
    ]
    allowed = {"sequence_gru", "decision_transformer"}
    invalid = sorted(set(families) - allowed)
    if invalid:
        raise ValueError(f"Unsupported --model-family-matrix entries: {invalid}")
    model_results: dict[str, Any] = {}
    for family in families:
        family_root = study_root / family
        family_root.mkdir(parents=True, exist_ok=True)
        family_args = _clone_args_with_model_family(args, family)
        walk_summary = _run_walkforward_study(study_root=family_root, tag=f"{tag}_{family}", args=family_args)
        model_results[family] = {
            "model_family": family,
            "artifact_root": str(family_root.resolve()),
            "walkforward_summary": walk_summary,
            "validation_exact_replay_metrics": walk_summary.get("validation_exact_replay_metrics", {}),
            "test_exact_replay_metrics": walk_summary.get("test_exact_replay_metrics", {}),
            "projection_parity_summary": walk_summary.get("projection_parity_summary", {}),
            "surrogate_exact_gap": walk_summary.get("surrogate_exact_gap", {}),
            "evidence_diagnostics": walk_summary.get("evidence_diagnostics", {}),
            "evidence_verdict": walk_summary.get("evidence_verdict", "insufficient_or_incomplete"),
        }
    matrix_diagnostics = _matrix_evidence_diagnostics(model_results)
    summary = {
        "status": "completed",
        "stage": "rl_walkforward_matrix",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "model_family_matrix": families,
        "train_years": list(WALKFORWARD_TRAIN_YEARS),
        "validation_years": list(WALKFORWARD_VALIDATION_YEARS),
        "test_years": list(WALKFORWARD_TEST_YEARS),
        "rollout_grad_mode": str(getattr(args, "rollout_grad_mode", "detached")),
        "rollout_chunk_days": int(getattr(args, "rollout_chunk_days", 20)),
        "projection_parity_max_l1": float(getattr(args, "projection_parity_max_l1", 0.02)),
        "models": model_results,
        "evidence_diagnostics": matrix_diagnostics,
        "evidence_verdict": _matrix_evidence_verdict(model_results),
        "oracle_used": False,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / "sequence_walkforward_matrix_summary.json", _json_ready(summary))
    return _json_ready(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the path20 research protocol. "
            "alpha_path20_neural_policy_v1 is the current Path20 research mainline pointer; "
            "sequence/RL stages remain available as shadow comparison routes."
        )
    )
    parser.add_argument(
        "--stage",
        choices=tuple(sorted(NEURAL_MAINLINE_STAGES | SEQUENCE_RL_STAGES)),
        default="dataset-smoke",
    )
    parser.add_argument("--tag", required=True, help="Explicit protocol/study tag. Loose latest is forbidden.")
    parser.add_argument("--policy-version", default="", help="Optional explicit policy version; defaults from the selected stage.")
    parser.add_argument(
        "--allow-legacy-neural-policy",
        action="store_true",
        help="Compatibility no-op retained for old scripts; neural policy stages are now the current Path20 research mainline.",
    )
    parser.add_argument("--data-source", default="lake", choices=("lake",))
    parser.add_argument("--lake-dataset-id", default="", help="Required for --data-source lake.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-name", default="learned_all_a")
    parser.add_argument("--pool-view-id", default="", help="Optional data-lake policy_pool_view dataset id.")
    parser.add_argument("--pool-view-kind", default="", choices=("", "learned_all_a", "rolling_liquidity", "exchange", "static_symbols"))
    parser.add_argument("--pool-view-name", default="")
    parser.add_argument(
        "--pool-view-exclude-symbol-prefixes",
        default="",
        help="Comma-separated stock-code prefixes excluded before building a data-lake pool view, e.g. 300,301,688,689.",
    )
    parser.add_argument("--sector-board-view-id", default="", help="Optional data-lake policy_sector_board_view dataset id.")
    parser.add_argument("--sector-board-view-kind", default="", choices=("", "latest_static_snapshot"))
    parser.add_argument("--sector-board-as-of-date", default="")
    parser.add_argument("--sector-board-industry-source-path", default="")
    parser.add_argument("--sector-board-source-path", default="")
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
    parser.add_argument("--smoke-lr", type=float, default=None)
    parser.add_argument("--forecast-lookback-days", type=int, default=252)
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=0,
        help="Maximum forecast horizon. Defaults to max(--forecast-cumulative-horizons).",
    )
    parser.add_argument(
        "--forecast-cumulative-horizons",
        default="1,3,5,10,20",
        help="Comma-separated cumulative/decision horizons, e.g. 1,2,3,5,8,10,15,20,30.",
    )
    parser.add_argument("--forecast-train-start-year", type=int, default=2019)
    parser.add_argument("--forecast-train-end-year", type=int, default=2022)
    parser.add_argument("--forecast-validation-year", type=int, default=2023)
    parser.add_argument("--forecast-test-year", type=int, default=2024)
    parser.add_argument("--forecast-model-families", default="linear_last_day,mlp_last_day,gru_sequence,patch_transformer")
    parser.add_argument("--forecast-epochs", type=int, default=2)
    parser.add_argument("--forecast-batch-size", type=int, default=512)
    parser.add_argument("--forecast-lr", type=float, default=3.0e-4)
    parser.add_argument("--forecast-hidden-dim", type=int, default=192)
    parser.add_argument("--forecast-dropout", type=float, default=0.15)
    parser.add_argument("--forecast-gru-layers", type=int, default=2)
    parser.add_argument("--forecast-transformer-layers", type=int, default=4)
    parser.add_argument("--forecast-transformer-heads", type=int, default=6)
    parser.add_argument("--forecast-patch-sizes", default="4,20")
    parser.add_argument("--forecast-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--forecast-amp", dest="forecast_amp", action="store_true", default=True)
    parser.add_argument("--no-forecast-amp", dest="forecast_amp", action="store_false")
    parser.add_argument("--forecast-seeds", default="7")
    parser.add_argument("--forecast-min-epochs", type=int, default=20)
    parser.add_argument("--forecast-early-stop-patience", type=int, default=12)
    parser.add_argument("--forecast-early-stop-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--forecast-grad-clip", type=float, default=1.0)
    parser.add_argument("--forecast-grad-accum-steps", type=int, default=1)
    parser.add_argument("--forecast-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--forecast-write-all-predictions", action="store_true")
    parser.add_argument("--forecast-resume-from", default="")
    parser.add_argument("--forecast-save-last", dest="forecast_save_last", action="store_true", default=True)
    parser.add_argument("--no-forecast-save-last", dest="forecast_save_last", action="store_false")
    parser.add_argument("--forecast-checkpoint-every-n-epochs", type=int, default=0)
    parser.add_argument(
        "--forecast-include-static-context",
        action="store_true",
        help="Write static symbol/exchange/industry ids into newly built memmap forecast datasets.",
    )
    parser.add_argument(
        "--forecast-static-fields",
        default="symbol,exchange,industry,liquidity_bucket,price_bucket",
        help="Comma-separated static context fields: symbol,exchange,industry,board,liquidity_bucket,price_bucket.",
    )
    parser.add_argument("--forecast-ranking-baseline", default="none", choices=("none", "lightgbm", "xgboost"))
    parser.add_argument("--forecast-output-profile", default="forecast_path_v1", choices=FORECAST_OUTPUT_PROFILES)
    parser.add_argument("--forecast-loss-profile", default="default", choices=FORECAST_LOSS_PROFILES)
    parser.add_argument("--forecast-decision-cost-bps", type=float, default=20.0)
    parser.add_argument("--forecast-decision-hit-threshold-bps", type=float, default=20.0)
    parser.add_argument("--forecast-decision-drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--forecast-slot-diagnostics", action="store_true")
    parser.add_argument("--forecast-selection-profile", default="multiscale", choices=FORECAST_SELECTION_PROFILES)
    parser.add_argument("--forecast-feature-profile", default=DEFAULT_FORECAST_FEATURE_PROFILE, choices=FORECAST_FEATURE_PROFILES)
    parser.add_argument("--forecast-max-feature-columns", type=int, default=DEFAULT_FORECAST_MAX_FEATURE_COLUMNS)
    parser.add_argument("--forecast-max-samples-per-role", type=int, default=0)
    parser.add_argument("--forecast-max-samples-per-date-per-role", type=int, default=0)
    parser.add_argument("--forecast-dataset-mode", default="eager", choices=("eager", "memmap"))
    parser.add_argument("--forecast-memmap-manifest", default="", help="Reuse an existing forecast memmap dataset manifest.")
    parser.add_argument("--forecast-min-lookback-valid-ratio", type=float, default=0.80)
    parser.add_argument("--forecast-dataloader-num-workers", type=int, default=0)
    parser.add_argument("--forecast-prefetch-factor", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--reward-profile", default=DEFAULT_RL_REWARD_PROFILE)
    parser.add_argument("--model-family", default="sequence_gru", choices=("sequence_gru", "decision_transformer", "decision_transformer_v2"))
    parser.add_argument("--no-oracle-input", action="store_true", default=True)
    parser.add_argument("--rl-hidden-dim", type=int, default=None)
    parser.add_argument("--rl-dropout", type=float, default=None)
    parser.add_argument("--rl-max-feature-columns", type=int, default=96)
    parser.add_argument("--min-year-trading-days", type=int, default=180)
    parser.add_argument("--rollout-grad-mode", default="detached", choices=("detached", "truncated"))
    parser.add_argument("--rollout-chunk-days", type=int, default=20)
    parser.add_argument("--projection-parity-max-l1", type=float, default=0.02)
    parser.add_argument("--reuse-episode-artifacts", dest="reuse_episode_artifacts", action="store_true", default=True)
    parser.add_argument("--no-reuse-episode-artifacts", dest="reuse_episode_artifacts", action="store_false")
    parser.add_argument("--model-family-matrix", default="sequence_gru,decision_transformer")
    parser.add_argument("--rl-seed-matrix", default="7,11,19")
    parser.add_argument("--projection-penalty-grid", default="0.05,0.20,0.50")
    parser.add_argument("--projection-penalty-weight", type=float, default=0.05)
    parser.add_argument("--tail-mass-penalty-weight", type=float, default=0.0)
    parser.add_argument("--turnover-penalty-weight", type=float, default=0.20)
    parser.add_argument("--v3-checkpoint-model-pt", default="")
    parser.add_argument("--v4-gru-checkpoint-model-pt", default="")
    parser.add_argument("--dt-context-grid", default="20,60")
    parser.add_argument("--dt-temporal-layers", type=int, default=2)
    parser.add_argument("--dt-cross-layers", type=int, default=1)
    parser.add_argument("--dt-num-heads", type=int, default=4)
    parser.add_argument("--dt-grad-clip", type=float, default=1.0)
    return parser


def _validate_protocol_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _apply_stage_defaults(args)
    if args.data_source == "lake" and not str(args.lake_dataset_id or "").strip():
        parser.error("--data-source lake requires explicit --lake-dataset-id; do not rely on loose latest/default.")
    if _is_loose_lake_dataset_id(str(args.lake_dataset_id or "")):
        parser.error("--lake-dataset-id must be a fixed dataset id, not latest/default/latest_*.")
    if args.stage in NEURAL_MAINLINE_STAGES and str(args.policy_version or "").strip() not in {
        "",
        ALPHA_PATH20_POLICY_VERSION,
    }:
        parser.error(f"Neural Path20 stages require --policy-version {ALPHA_PATH20_POLICY_VERSION}.")
    if args.stage in SEQUENCE_RL_STAGES and str(args.policy_version or "").strip() not in {
        "",
        ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
    }:
        parser.error(f"RL stages require --policy-version {ALPHA_PATH20_SEQUENCE_POLICY_VERSION}.")
    if args.stage in SEQUENCE_RL_STAGES and not bool(args.no_oracle_input):
        parser.error("RL stages require --no-oracle-input; oracle inputs are not allowed.")
    if args.stage in FORECAST_MAINLINE_STAGES:
        if int(getattr(args, "forecast_lookback_days", 252)) <= 0:
            parser.error("--forecast-lookback-days must be positive.")
        try:
            forecast_horizon, cumulative_horizons = _resolve_forecast_horizon_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        args.forecast_horizon = int(forecast_horizon)
        args.forecast_cumulative_horizons = ",".join(str(item) for item in cumulative_horizons)
        if int(getattr(args, "forecast_epochs", 2)) <= 0:
            parser.error("--forecast-epochs must be positive.")
        if int(getattr(args, "forecast_batch_size", 512)) <= 0:
            parser.error("--forecast-batch-size must be positive.")
        if float(getattr(args, "forecast_lr", 3.0e-4)) <= 0.0:
            parser.error("--forecast-lr must be positive.")
        if int(getattr(args, "forecast_hidden_dim", 192)) <= 0:
            parser.error("--forecast-hidden-dim must be positive.")
        if not (0.0 <= float(getattr(args, "forecast_dropout", 0.15)) < 1.0):
            parser.error("--forecast-dropout must be in [0, 1).")
        if int(getattr(args, "forecast_gru_layers", 2)) <= 0:
            parser.error("--forecast-gru-layers must be positive.")
        if int(getattr(args, "forecast_transformer_layers", 4)) <= 0:
            parser.error("--forecast-transformer-layers must be positive.")
        if int(getattr(args, "forecast_transformer_heads", 6)) <= 0:
            parser.error("--forecast-transformer-heads must be positive.")
        try:
            patch_sizes = _forecast_int_tuple(getattr(args, "forecast_patch_sizes", ""), default=(4, 20))
        except ValueError as exc:
            parser.error(str(exc))
        if not patch_sizes or any(int(item) <= 0 for item in patch_sizes):
            parser.error("--forecast-patch-sizes must contain positive integers.")
        if int(getattr(args, "forecast_min_epochs", 20)) <= 0:
            parser.error("--forecast-min-epochs must be positive.")
        if int(getattr(args, "forecast_early_stop_patience", 12)) <= 0:
            parser.error("--forecast-early-stop-patience must be positive.")
        if float(getattr(args, "forecast_early_stop_min_delta", 1.0e-4)) < 0.0:
            parser.error("--forecast-early-stop-min-delta must be non-negative.")
        if float(getattr(args, "forecast_grad_clip", 1.0)) <= 0.0:
            parser.error("--forecast-grad-clip must be positive.")
        if int(getattr(args, "forecast_grad_accum_steps", 1)) <= 0:
            parser.error("--forecast-grad-accum-steps must be positive.")
        if float(getattr(args, "forecast_weight_decay", 1.0e-4)) < 0.0:
            parser.error("--forecast-weight-decay must be non-negative.")
        if int(getattr(args, "forecast_checkpoint_every_n_epochs", 0)) < 0:
            parser.error("--forecast-checkpoint-every-n-epochs must be non-negative.")
        try:
            normalize_static_context_fields(str(getattr(args, "forecast_static_fields", "")))
        except ValueError as exc:
            parser.error(str(exc))
        try:
            loss_contract = forecast_loss_profile_contract(
                getattr(args, "forecast_loss_profile", "default"),
                cumulative_horizons=cumulative_horizons,
                forecast_horizon=forecast_horizon,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if (
            str(loss_contract.get("required_output_profile", "")) == "decision_utility_v1"
            or str(getattr(args, "forecast_selection_profile", "multiscale")) == "decision_utility"
        ):
            args.forecast_output_profile = "decision_utility_v1"
        if str(getattr(args, "forecast_output_profile", "forecast_path_v1")) == "decision_utility_v1" and str(
            getattr(args, "forecast_loss_profile", "default")
        ) == "default":
            args.forecast_loss_profile = "decision_utility_v1"
        if float(getattr(args, "forecast_decision_cost_bps", 20.0)) < 0.0:
            parser.error("--forecast-decision-cost-bps must be non-negative.")
        if float(getattr(args, "forecast_decision_hit_threshold_bps", 20.0)) < 0.0:
            parser.error("--forecast-decision-hit-threshold-bps must be non-negative.")
        if float(getattr(args, "forecast_decision_drawdown_penalty", 0.25)) < 0.0:
            parser.error("--forecast-decision-drawdown-penalty must be non-negative.")
        resume_from = str(getattr(args, "forecast_resume_from", "") or "").strip()
        if resume_from and not Path(resume_from).exists():
            parser.error("--forecast-resume-from must point to an existing checkpoint.")
        try:
            seeds = _forecast_int_tuple(getattr(args, "forecast_seeds", ""), default=(7,))
        except ValueError as exc:
            parser.error(str(exc))
        if not seeds or any(int(item) < 0 for item in seeds):
            parser.error("--forecast-seeds must contain non-negative integer seeds.")
        if int(getattr(args, "forecast_max_samples_per_role", 0)) < 0:
            parser.error("--forecast-max-samples-per-role must be >= 0.")
        if int(getattr(args, "forecast_max_samples_per_date_per_role", 0)) < 0:
            parser.error("--forecast-max-samples-per-date-per-role must be >= 0.")
        if int(getattr(args, "forecast_max_feature_columns", DEFAULT_FORECAST_MAX_FEATURE_COLUMNS)) <= 0:
            parser.error("--forecast-max-feature-columns must be positive.")
        if not (0.0 <= float(getattr(args, "forecast_min_lookback_valid_ratio", 0.80)) <= 1.0):
            parser.error("--forecast-min-lookback-valid-ratio must be in [0, 1].")
        if int(getattr(args, "forecast_dataloader_num_workers", 0)) < 0:
            parser.error("--forecast-dataloader-num-workers must be >= 0.")
        if int(getattr(args, "forecast_prefetch_factor", 2)) <= 0:
            parser.error("--forecast-prefetch-factor must be positive.")
        if int(getattr(args, "max_universe_size", 80)) == 0 and str(getattr(args, "forecast_dataset_mode", "eager")) == "eager":
            parser.error("full_universe_requires_memmap_dataset_mode: use --forecast-dataset-mode memmap when --max-universe-size 0.")
        if str(getattr(args, "forecast_memmap_manifest", "") or "").strip() and str(getattr(args, "forecast_dataset_mode", "eager")) != "memmap":
            parser.error("--forecast-memmap-manifest requires --forecast-dataset-mode memmap.")
        if not (
            int(args.forecast_train_start_year)
            <= int(args.forecast_train_end_year)
            < int(args.forecast_validation_year)
            < int(args.forecast_test_year)
        ):
            parser.error(
                "Forecast years require train_start <= train_end < validation_year < test_year."
            )
        families = _forecast_model_families(getattr(args, "forecast_model_families", ""))
        invalid = sorted(set(families) - set(FORECAST_MODEL_FAMILIES))
        if invalid:
            parser.error(f"Unsupported --forecast-model-families: {', '.join(invalid)}.")
    if int(getattr(args, "rollout_chunk_days", 20)) <= 0:
        parser.error("--rollout-chunk-days must be positive.")
    if float(getattr(args, "projection_parity_max_l1", 0.02)) < 0.0:
        parser.error("--projection-parity-max-l1 must be non-negative.")
    if args.stage == "rl-walkforward-matrix":
        families = [
            item.strip()
            for item in str(getattr(args, "model_family_matrix", "")).split(",")
            if item.strip()
        ]
        invalid = sorted(set(families) - {"sequence_gru", "decision_transformer"})
        if not families or invalid:
            parser.error("--model-family-matrix must contain sequence_gru and/or decision_transformer.")
    if args.stage == "rl-v4-validation-repair-study":
        try:
            _parse_int_list(str(getattr(args, "rl_seed_matrix", "")), default=V4_DEFAULT_SEEDS)
            _parse_float_list(str(getattr(args, "projection_penalty_grid", "")), default=V4_PROJECTION_PENALTY_GRID)
        except ValueError as exc:
            parser.error(str(exc))
        if float(getattr(args, "projection_penalty_weight", 0.05)) < 0.0:
            parser.error("--projection-penalty-weight must be non-negative.")
        if float(getattr(args, "tail_mass_penalty_weight", 0.0)) < 0.0:
            parser.error("--tail-mass-penalty-weight must be non-negative.")
        if float(getattr(args, "turnover_penalty_weight", 0.20)) < 0.0:
            parser.error("--turnover-penalty-weight must be non-negative.")
    if args.stage == "rl-v5-dt-validation-study":
        try:
            context_grid = _parse_int_list(str(getattr(args, "dt_context_grid", "")), default=V5_DEFAULT_CONTEXT_GRID)
            _parse_int_list(str(getattr(args, "rl_seed_matrix", "")), default=V4_DEFAULT_SEEDS)
        except ValueError as exc:
            parser.error(str(exc))
        if not context_grid:
            parser.error("--dt-context-grid must contain at least one positive context length.")
        if any(int(item) <= 0 for item in context_grid):
            parser.error("--dt-context-grid values must be positive.")
        if int(getattr(args, "dt_temporal_layers", 2)) <= 0:
            parser.error("--dt-temporal-layers must be positive.")
        if int(getattr(args, "dt_cross_layers", 1)) <= 0:
            parser.error("--dt-cross-layers must be positive.")
        if int(getattr(args, "dt_num_heads", 4)) <= 0:
            parser.error("--dt-num-heads must be positive.")
        if float(getattr(args, "dt_grad_clip", 1.0)) <= 0.0:
            parser.error("--dt-grad-clip must be positive.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_protocol_args(parser, args)
    tag = str(args.tag or "").strip()
    study_root = PATH_POLICY_STUDIES_ROOT / tag
    study_root.mkdir(parents=True, exist_ok=True)
    effective_lake_dataset_id = str(args.lake_dataset_id or DEFAULT_POLICY_INPUT_LAKE_DATASET_ID).strip()
    args.lake_dataset_id = effective_lake_dataset_id
    if args.stage == "rl-v5-dt-validation-study":
        sequence_policy_summary = _run_v5_dt_validation_study(study_root=study_root, tag=tag, args=args)
        summary = {
            "status": "completed",
            "run_tag": tag,
            "study_tag": tag,
            "created_at": now_iso(),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
            "research_status": "research / shadow-only / alpha_path20_sequence_policy_v1 v5 paper-grade DT evidence contract",
            "stage": args.stage,
            "data_source": args.data_source,
            "lake_dataset_id": effective_lake_dataset_id,
            "execution_mode": args.execution_mode,
            "sequence_policy": sequence_policy_summary,
            "facts": [
                "v5 uses decision_transformer_v2 with learned time and stock-slot embeddings.",
                "v5 selects checkpoints only with 2022 exact validation replay.",
                "2024 test replay is interpretable only after validation passes.",
                "v5 training uses pure policy utility, not action imitation.",
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
    if args.stage == "rl-v4-validation-repair-study":
        sequence_policy_summary = _run_v4_validation_repair_study(study_root=study_root, tag=tag, args=args)
        summary = {
            "status": "completed",
            "run_tag": tag,
            "study_tag": tag,
            "created_at": now_iso(),
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "policy_profile": ALPHA_PATH20_SEQUENCE_POLICY_PROFILE,
            "research_status": "research / shadow-only / alpha_path20_sequence_policy_v1 v4 validation-first repair evidence",
            "stage": args.stage,
            "data_source": args.data_source,
            "lake_dataset_id": effective_lake_dataset_id,
            "execution_mode": args.execution_mode,
            "sequence_policy": sequence_policy_summary,
            "facts": [
                "v4 selects checkpoints only with 2022 exact validation replay.",
                "2024 test replay is interpretable only after validation passes.",
                "Baseline comparisons use the same exact replay, costs, and projection constraints.",
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
    if args.stage in {"rl-walkforward-study", "rl-walkforward-matrix"}:
        sequence_policy_summary = (
            _run_walkforward_matrix(study_root=study_root, tag=tag, args=args)
            if args.stage == "rl-walkforward-matrix"
            else _run_walkforward_study(study_root=study_root, tag=tag, args=args)
        )
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
            "sequence_policy": sequence_policy_summary,
            "facts": [
                "alpha_path20_sequence_policy_v1 is the path20 main research line.",
                "Walk-forward evidence separates train, validation, and final shadow test years.",
                "Exact replay remains PortfolioState.step after deterministic safety projection.",
                "v3 matrix evidence compares fixed GRU and Decision Transformer families when rl-walkforward-matrix is selected.",
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

    if args.stage in FORECAST_TRAINING_STAGES and str(getattr(args, "forecast_memmap_manifest", "") or "").strip():
        summary = _run_forecast_manifest_reuse_fast_path(study_root=study_root, tag=tag, args=args)
        safe_print_json(summary)
        return summary

    prepare_start_date = args.start_date
    prepare_end_date = args.end_date
    if args.stage in FORECAST_MAINLINE_STAGES:
        prepare_start_date = f"{int(args.forecast_train_start_year) - 1}0101"
        prepare_end_date = f"{int(args.forecast_test_year)}1231"
    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=prepare_start_date,
        end_date=prepare_end_date
        if args.stage in (SEQUENCE_RL_STAGES | FORECAST_MAINLINE_STAGES)
        else _extend_end_date_for_labels(prepare_end_date),
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        lake_dataset_id=effective_lake_dataset_id,
        data_lake_root=args.data_lake_root,
        lake_min_trading_days=int(args.lake_min_trading_days),
        pool_view_id=str(getattr(args, "pool_view_id", "") or ""),
        pool_view_spec=_pool_view_spec_from_args(args, start_date=prepare_start_date, end_date=prepare_end_date),
        sector_board_view_id=str(getattr(args, "sector_board_view_id", "") or ""),
        sector_board_view_spec=_sector_board_view_spec_from_args(args),
        max_universe_size=int(args.max_universe_size),
        alpha_prior_source=DEFAULT_ALPHA_PRIOR_SOURCE,
        require_lake_benchmark_open=str(args.execution_mode or "next_open").strip().lower() == "next_open",
        progress_desc=f"alpha_path20 prepare {tag}",
    )
    if args.stage in FORECAST_MAINLINE_STAGES:
        summary = _run_forecast_walkforward_study(prepared=prepared, study_root=study_root, tag=tag, args=args)
        safe_print_json(summary)
        return summary
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
        "research_status": "current path20 research mainline pointer / research / shadow-only / alpha_path20_neural_policy_v1 evidence",
        "path20_research_mainline": True,
        "legacy_diagnostic_only": False,
        "no_new_mainline_budget": False,
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
            "alpha_path20_neural_policy_v1 is the current Path20 research mainline pointer under daily_research/path_policy.",
            "The protocol writes path-policy artifacts under daily_research/output/path_policy with explicit tag and dataset id.",
            "target_weight is the only execution truth; source/receiver fields are derived diagnostics.",
        ],
        "inferences": [
            "This smoke can validate schema, next-open labels, target-weight projection, and oracle-path replay plumbing.",
            "It is current Path20 neural-policy mainline evidence, but it is not live/default or promotion evidence.",
        ],
        "assumptions": [
            "Lake policy bundle inputs already contain adjusted open/close panels and universe membership.",
            "The first allocator smoke may use oracle path labels for tiny optimization without claiming deployable prediction quality.",
        ],
        "boundaries": [
            "shadow_only=true",
            "path20_research_mainline=true",
            "mainline pointer is switchable by explicit future decision",
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
