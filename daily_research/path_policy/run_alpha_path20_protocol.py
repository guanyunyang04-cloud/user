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
from daily_research.path_policy import ALPHA_PATH20_POLICY_PROFILE, ALPHA_PATH20_POLICY_VERSION
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


PATH_POLICY_OUTPUT_ROOT = Path("daily_research/output/path_policy")
PATH_POLICY_STUDIES_ROOT = PATH_POLICY_OUTPUT_ROOT / "studies"
PATH_POLICY_DATASETS_ROOT = PATH_POLICY_OUTPUT_ROOT / "datasets"

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated alpha_path20_neural_policy_v1 research protocol.")
    parser.add_argument("--stage", choices=("dataset-smoke", "oracle-smoke", "tiny-smoke"), default="tiny-smoke")
    parser.add_argument("--tag", required=True, help="Explicit protocol/study tag. Loose latest is forbidden.")
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
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.data_source == "lake" and not str(args.lake_dataset_id or "").strip():
        parser.error("--data-source lake requires explicit --lake-dataset-id; do not rely on loose latest/default.")
    if str(args.lake_dataset_id or "").strip().lower() in {"latest", "default"}:
        parser.error("--lake-dataset-id must be a fixed dataset id, not latest/default.")
    tag = str(args.tag or "").strip()
    study_root = PATH_POLICY_STUDIES_ROOT / tag
    study_root.mkdir(parents=True, exist_ok=True)
    effective_lake_dataset_id = str(args.lake_dataset_id or DEFAULT_POLICY_INPUT_LAKE_DATASET_ID).strip()
    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=_extend_end_date_for_labels(args.end_date),
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
        "research_status": "research / shadow-only / alpha_path20_neural_policy_v1 smoke evidence",
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
            "alpha_path20_neural_policy_v1 is isolated under daily_research/path_policy.",
            "The protocol writes path-policy artifacts under daily_research/output/path_policy with explicit tag and dataset id.",
            "target_weight is the only execution truth; source/receiver fields are derived diagnostics.",
        ],
        "inferences": [
            "This smoke can validate schema, next-open labels, target-weight projection, and oracle-path replay plumbing.",
            "It is not sufficient promotion or longrun evidence; full multi-window forecaster and allocator training remain pending.",
        ],
        "assumptions": [
            "Lake policy bundle inputs already contain adjusted open/close panels and universe membership.",
            "The first allocator smoke may use oracle path labels for tiny optimization without claiming deployable prediction quality.",
        ],
        "boundaries": [
            "shadow_only=true",
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
