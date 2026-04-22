from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from daily_research.baseline.backtest import backtest, summarize_backtest_by_month, summarize_monthly_diagnostics
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    get_latest_completed_trading_date,
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.portfolio import (
    build_research_raw_target_weights,
    build_target_weights,
)
from daily_research.baseline.regime import compute_market_regime_state
from daily_research.deep_alpha.cache_utils import cache_key, frame_signature, get_cache_root, load_pickle, save_pickle, series_signature
from daily_research.deep_alpha.config import DeepAlphaConfig
from daily_research.deep_alpha.execution_alignment import (
    default_auto_profile_argument,
    evaluate_profile as evaluate_execution_alignment_profile,
    fit_execution_alignment,
    get_profile as get_execution_alignment_profile,
    list_profile_lines as list_execution_alignment_profile_lines,
    parse_profile_name_list as parse_execution_alignment_profile_names,
    resolve_profile as resolve_execution_alignment_profile,
)
from daily_research.deep_alpha.experiment_guardrails import (
    DEFAULT_FOREGROUND_TIMEOUT_HOURS,
    DEFAULT_RESUME_DISCIPLINE,
    compare_expected_fields,
    contract_fingerprint,
    runtime_metadata,
)
from daily_research.deep_alpha.family_epoch_budget import (
    DEFAULT_MIN_START_EPOCH_BUDGET,
    default_min_epochs_for_budget,
)
from daily_research.deep_alpha.models import MultiTaskRanker
from daily_research.deep_alpha.pipeline_utils import (
    RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    RESEARCH_TIME_UNIT_TRADING_DAYS,
    build_score_horizon_weights,
    build_target_loss_weights,
    load_cached_or_build_liquidity_buckets,
    load_cached_or_build_rolling_pool,
    load_cached_or_fit_market_state,
    normalize_research_time_unit,
    load_raw_market_data,
    load_stocks_from_file,
    parse_float_list,
    parse_horizons,
    parse_name_list,
    parse_stocks,
    resolve_recent_window_start,
    resolve_split_window,
    resolve_stocks_file,
    subset_df_dict_to_stocks,
)
from daily_research.deep_alpha.risk_gate import apply_state_risk_gate, fit_state_risk_gate
from daily_research.deep_alpha.runtime_profile import configure_torch_runtime, resolve_runtime_profile
from daily_research.deep_alpha.research_objective import (
    CHECKPOINT_SELECTION_OBJECTIVES,
    DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    DEFAULT_EXECUTION_ALIGNMENT_MODE,
    DEFAULT_EXECUTION_ALIGNMENT_OBJECTIVE,
    DEFAULT_EXECUTION_ALIGNMENT_SELL_TAX_BPS,
    DEFAULT_EXECUTION_ALIGNMENT_SLIPPAGE_BPS,
    DEFAULT_EXECUTION_ALIGNMENT_TRANSACTION_COST_BPS,
    DEFAULT_RESEARCH_OBJECTIVE_MODE,
    resolve_checkpoint_metric_name,
    resolve_checkpoint_metric_value,
    resolve_primary_backtest,
    resolve_primary_backtest_label,
    resolve_primary_monthly_diagnostics_label,
    resolve_primary_monthly_summary_label,
    resolve_primary_panel_mode,
    summarize_primary_monthly_objectives,
)
from daily_research.deep_alpha.score_head import (
    apply_score_head,
    build_policy_decision_score_frame,
    build_policy_target_weight_frame,
    fit_score_head,
    is_policy_score_head_method,
)
from daily_research.deep_alpha.sequence_dataset import (
    DateGroupedBatchSampler,
    StockSequenceDataset,
    STRUCTURE_LABELS,
    build_structure_label_frame,
    build_sequence_corpus,
    build_sequence_features,
    build_targets,
    transform_return_target_frames,
)
from daily_research.deep_alpha.trainer import (
    collate_batch,
    compute_rankic,
    compute_rankic_timeseries,
    infer_dataset,
    train_multitask_model,
)
from daily_research.progress import StageProgress, progress_write


def parse_args():
    parser = argparse.ArgumentParser(description="Deep alpha research branch: market state + sequence encoder + cross-sectional ranking")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--force-raw-cache-path", default="")
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Path to txt/csv file containing stock codes.")
    parser.add_argument("--liquidity-pool", choices=["liquid300", "liquid500", "liquid800"], default="", help="Use the latest daily-updated high-liquidity pool from daily_research/execution/universe/.")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="",
        help="Use a historical rolling high-liquidity pool for formal research validation instead of a fixed latest pool.",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21, help="Trading-day frequency for rebuilding the historical rolling liquidity pool.")
    parser.add_argument("--pool-adv-window", type=int, default=20, help="ADV lookback window used when rebuilding the historical rolling liquidity pool.")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--lookback-window", type=int, default=120)
    parser.add_argument("--prediction-horizons", default="5,10,20")
    parser.add_argument("--return-loss-mode", choices=["regression", "top_rest_bce", "top_bottom_bce"], default="top_bottom_bce")
    parser.add_argument("--return-target-transform", choices=["raw", "cs_rank", "cs_zscore"], default="raw")
    parser.add_argument("--return-top-frac", type=float, default=0.2)
    parser.add_argument("--return-bottom-frac", type=float, default=0.2)
    parser.add_argument("--liquidity-conditioning-mode", choices=["none", "top_vs_other", "smooth_bucket"], default="none")
    parser.add_argument("--top-liquidity-return-loss-weight", type=float, default=1.0)
    parser.add_argument("--other-liquidity-return-loss-weight", type=float, default=1.0)
    parser.add_argument("--top-liquidity-rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--other-liquidity-rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--top-liquidity-sample-weight", type=float, default=1.0)
    parser.add_argument("--other-liquidity-sample-weight", type=float, default=1.0)
    parser.add_argument("--structure-conditioning-mode", choices=["none", "targeted_liquidity_rank", "state_targeted_rank"], default="none")
    parser.add_argument("--top-attack-structures", default="trend_breakout,high_vol_expansion")
    parser.add_argument("--other-protect-structures", default="neutral_mixed,pullback_rebound,low_vol_trend")
    parser.add_argument("--top-attack-rank-weight", type=float, default=1.0)
    parser.add_argument("--other-protect-rank-weight", type=float, default=1.0)
    parser.add_argument("--target-state-names", default="trend_up_low_vol")
    parser.add_argument("--target-state-attack-structures", default="neutral_mixed,pullback_rebound,trend_breakout")
    parser.add_argument("--target-state-protect-structures", default="low_vol_trend")
    parser.add_argument("--target-state-rank-weight", type=float, default=1.20)
    parser.add_argument("--target-state-protect-rank-weight", type=float, default=1.05)
    parser.add_argument("--task-loss-weights", default="", help="Named weights like 5:0.15,10:0.30,20:0.55,downside:0.35")
    parser.add_argument("--score-horizon-weights", default="", help="Named score weights like 5:0.1,10:0.35,20:0.55")
    parser.add_argument("--score-rank-blend", type=float, default=0.35)
    parser.add_argument("--score-downside-penalty", type=float, default=0.25)
    parser.add_argument("--score-risk-mode", choices=["subtract", "gate", "state_gate", "state_liquidity_gate"], default="subtract")
    parser.add_argument("--score-risk-gate-threshold", type=float, default=0.35)
    parser.add_argument("--score-risk-state-thresholds", default="0.0,0.2,0.35,0.5,0.65")
    parser.add_argument(
        "--score-head-method",
        choices=["manual", "ridge", "lgbm", "short_expert", "policy_v1", "policy_v2", "policy_v2a", "policy_v2b", "policy_v2c", "policy_v4a", "policy_v4b", "policy_v5a", "policy_v5b", "policy_v5c", "policy_v5d", "policy_v5e", "policy_v3"],
        default="manual",
    )
    parser.add_argument("--adaptive-task-weights", action="store_true", help="Learn task importance from train-period RankIC instead of using only fixed manual weights.")
    parser.add_argument(
        "--research-time-unit",
        choices=[RESEARCH_TIME_UNIT_TRADING_DAYS, RESEARCH_TIME_UNIT_CALENDAR_MONTHS],
        default=RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
        help="Window protocol used for train/valid splits and monthly research outputs.",
    )
    parser.add_argument("--adaptive-task-window-days", type=int, default=126, help="Recent train-window length for adaptive task weights.")
    parser.add_argument("--adaptive-task-window-months", type=int, default=6, help="Recent train-window length for adaptive task weights in calendar_months mode.")
    parser.add_argument("--train-end-date", default="")
    parser.add_argument("--valid-start-date", default="")
    parser.add_argument("--valid-days", type=int, default=252)
    parser.add_argument("--valid-months", type=int, default=12, help="Formal holdout size in calendar months when research-time-unit=calendar_months.")
    parser.add_argument("--train-eval-window-days", type=int, default=126, help="Only use the most recent N train-side trading days to fit score_head and state_gate. Use 0 to evaluate on the full train span.")
    parser.add_argument("--train-eval-window-months", type=int, default=6, help="Only use the most recent N train-side calendar months to fit score_head and state_gate when research-time-unit=calendar_months.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-family", choices=["gru", "transformer", "patch_transformer", "mamba", "ssm"], default="gru")
    parser.add_argument("--patch-len", type=int, default=5)
    parser.add_argument("--pretrained-encoder-path", default="", help="Optional path to a masked-pretrained patch encoder artifact for ranking fine-tuning.")
    parser.add_argument("--resume-run-dir", default="", help="Existing deep_alpha run directory containing deep_alpha_model.pt and train_history.csv.")
    parser.add_argument("--resume-model-path", default="", help="Direct path to a deep_alpha_model.pt artifact. Overrides --resume-run-dir when provided.")
    parser.add_argument(
        "--resume-mode",
        choices=["off", "strict", "warm_start"],
        default="off",
        help="strict restores last-epoch optimizer/scheduler/scaler state; warm_start only loads the selected model weights.",
    )
    parser.add_argument("--return-head-mode", choices=["shared", "liquidity_switch"], default="shared")
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--state-context", action="store_true", help="Use learned market-state embeddings as lightweight context.")
    parser.add_argument("--liquidity-context", action="store_true", help="Use learned liquidity-bucket embeddings as lightweight context.")
    parser.add_argument("--structure-context", action="store_true", help="Use learned structure-label embeddings as lightweight context.")
    parser.add_argument("--aux-structure-task", action="store_true", help="Add a light auxiliary structure-classification task to improve first-layer representations.")
    parser.add_argument("--aux-structure-loss-weight", type=float, default=0.10)
    parser.add_argument("--aux-structure-label-smoothing", type=float, default=0.05)
    parser.add_argument("--structure-prototype-task", action="store_true", help="Use learned structure prototypes to regularize first-layer embeddings without directly changing ranking loss weights.")
    parser.add_argument("--structure-prototype-loss-weight", type=float, default=0.05)
    parser.add_argument("--structure-prototype-temperature", type=float, default=0.20)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MIN_START_EPOCH_BUDGET)
    parser.add_argument("--min-epochs", type=int, default=default_min_epochs_for_budget(DEFAULT_MIN_START_EPOCH_BUDGET))
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--lr-plateau-patience", type=int, default=1)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision when CUDA is available.")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--safe-runtime-profile", action="store_true", help="Auto-cap batch size and workers for this local machine.")
    parser.add_argument("--no-safe-runtime-profile", dest="safe_runtime_profile", action="store_false")
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=0.35)
    parser.add_argument("--max-rank-pairs-per-group", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--market-state-count", type=int, default=4)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--relation-layer", action="store_true", help="Enable lightweight relation features such as industry-relative ranking and style strength.")
    parser.add_argument("--liquidity-layer", action="store_true", help="Enable liquidity-stratification features and bucket-aware relation features.")
    parser.add_argument("--liquidity-bucket-count", type=int, default=5)
    parser.add_argument("--dynamic-graph-layer", action="store_true", help="Enable a daily-updated top-k peer graph feature layer built from cross-sectional similarity.")
    parser.add_argument("--dynamic-graph-top-k", type=int, default=8)
    parser.add_argument("--dynamic-graph-temperature", type=float, default=0.35)
    parser.add_argument("--dynamic-graph-industry-boost", type=float, default=0.15)
    parser.add_argument("--dynamic-graph-style-boost", type=float, default=0.05)
    parser.add_argument("--short-alpha-features", action="store_true", help="Add short-line breakout/compression/energy candidate inputs.")
    parser.add_argument("--breakout-event-horizon", type=int, default=5, help="Forward event horizon used by short-line breakout event targets.")
    parser.add_argument("--breakout-event-threshold", type=float, default=0.08, help="Required forward max gain for breakout event labels.")
    parser.add_argument("--breakout-event-pullback-limit", type=float, default=0.03, help="Maximum tolerated pullback before breakout for clean breakout labels.")
    parser.add_argument("--breakout-event-loss-weight", type=float, default=0.0, help="Auxiliary loss weight for breakout event labels.")
    parser.add_argument("--clean-breakout-event-loss-weight", type=float, default=0.0, help="Auxiliary loss weight for clean breakout event labels.")
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument(
        "--research-objective-mode",
        choices=["execution_first", "raw_holdout"],
        default=DEFAULT_RESEARCH_OBJECTIVE_MODE,
        help="Primary research winner objective. execution_first promotes after-cost executed profit over raw holdout metrics.",
    )
    parser.add_argument(
        "--checkpoint-selection-objective",
        choices=list(CHECKPOINT_SELECTION_OBJECTIVES),
        default=DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
        help="Criterion used to keep the best training checkpoint. primary_* objectives are evaluated on the primary research backtest.",
    )
    parser.add_argument("--checkpoint-selection-min-improvement", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-eval-interval",
        type=int,
        default=1,
        help="Evaluate expensive primary_* checkpoint objectives every N epochs. Final epoch is always evaluated.",
    )
    parser.add_argument(
        "--checkpoint-eval-start-epoch",
        type=int,
        default=1,
        help="First epoch that may run an expensive primary_* checkpoint evaluation.",
    )
    parser.add_argument(
        "--execution-alignment-transaction-cost-bps",
        type=float,
        default=DEFAULT_EXECUTION_ALIGNMENT_TRANSACTION_COST_BPS,
    )
    parser.add_argument(
        "--execution-alignment-slippage-bps",
        type=float,
        default=DEFAULT_EXECUTION_ALIGNMENT_SLIPPAGE_BPS,
    )
    parser.add_argument(
        "--execution-alignment-sell-tax-bps",
        type=float,
        default=DEFAULT_EXECUTION_ALIGNMENT_SELL_TAX_BPS,
    )
    parser.add_argument(
        "--execution-alignment-mode",
        choices=["off", "profile", "train_eval_auto"],
        default=DEFAULT_EXECUTION_ALIGNMENT_MODE,
        help="Optional train-side execution-objective alignment applied after raw target-weight construction.",
    )
    parser.add_argument(
        "--execution-alignment-profile",
        default="regoff_k1_5d_ensemble_native_anchor",
        help="Profile used when --execution-alignment-mode=profile.",
    )
    parser.add_argument(
        "--execution-alignment-objective",
        choices=["excess_annual_return", "excess_sharpe", "robust_composite"],
        default=DEFAULT_EXECUTION_ALIGNMENT_OBJECTIVE,
        help="Train-side selection objective used by --execution-alignment-mode=train_eval_auto.",
    )
    parser.add_argument(
        "--execution-alignment-candidate-profiles",
        default="",
        help=(
            "Optional comma-separated override for train_eval_auto execution alignment profiles or profile sets. "
            f"Empty defaults to `{default_auto_profile_argument()}`."
        ),
    )
    parser.add_argument(
        "--execution-alignment-shortlist-size",
        type=int,
        default=8,
        help="When train_eval_auto is used, keep only the top-N screen-stage profiles for full train-eval backtesting. Use 0 to disable shortlist screening.",
    )
    parser.add_argument(
        "--execution-alignment-screen-window-days",
        type=int,
        default=63,
        help="Trading-day lookback used by the fast screen stage before full train-eval execution alignment.",
    )
    parser.add_argument(
        "--list-execution-alignment-profiles",
        action="store_true",
        help="List available execution alignment profiles and exit.",
    )
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable deep_alpha raw/feature cache.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing cache files and rebuild.")
    parser.set_defaults(pin_memory=False, use_amp=True, safe_runtime_profile=True)
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()
    if "--min-epochs" not in sys.argv:
        args.min_epochs = max(int(args.min_epochs), default_min_epochs_for_budget(int(args.epochs)))
    if args.list_execution_alignment_profiles:
        for line in list_execution_alignment_profile_lines():
            print(line)
        raise SystemExit(0)
    return args


def _load_cached_rolling_pool_union(pool_name: str, start_date: str, end_date: str) -> list[str]:
    rolling_root = get_cache_root() / "rolling_pools"
    if not rolling_root.exists():
        return []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    for path in sorted(rolling_root.glob("*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True):
        artifact = load_pickle(path)
        if artifact is None or str(getattr(artifact, "pool_name", "")) != str(pool_name):
            continue
        membership = getattr(artifact, "membership_frame", None)
        if membership is None or membership.empty:
            continue
        membership = membership.copy()
        membership.index = pd.to_datetime(membership.index)
        if membership.index.max() < end_ts:
            continue
        effective_start = max(start_ts, membership.index.min())
        window_membership = membership.loc[(membership.index >= effective_start) & (membership.index <= end_ts)]
        if window_membership.empty:
            continue
        union = window_membership.columns[window_membership.any(axis=0)].tolist()
        if union:
            return sorted({str(stock) for stock in union})
    return []


def _should_run_checkpoint_objective_eval(
    *,
    epoch: int,
    total_epochs: int,
    start_epoch: int,
    interval: int,
) -> bool:
    normalized_start = max(int(start_epoch), 1)
    normalized_interval = max(int(interval), 1)
    if int(epoch) == int(total_epochs):
        return True
    if int(epoch) < normalized_start:
        return False
    return (int(epoch) - normalized_start) % normalized_interval == 0


def _cross_sectional_signal(pivot: pd.DataFrame, rank_blend: float) -> pd.DataFrame:
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1).replace(0, np.nan), axis=0)
    rank = pivot.rank(axis=1, pct=True)
    rank_centered = (rank - 0.5) * 2.0
    return z.fillna(0.0) * (1.0 - rank_blend) + rank_centered.fillna(0.0) * rank_blend


def _build_score_frame(
    pred_df: pd.DataFrame,
    target_names: list[str],
    horizon_weights: dict[int, float],
    score_rank_blend: float,
    score_downside_penalty: float,
    score_risk_mode: str,
    score_risk_gate_threshold: float,
    all_dates: pd.Index,
    all_stocks: list[str],
) -> pd.DataFrame:
    frames = []
    for name in target_names:
        horizon = int(name.replace("fwd_excess_", "")) if name.startswith("fwd_excess_") else None
        if horizon is None:
            continue
        weight = float(horizon_weights.get(horizon, 0.0))
        if weight <= 0:
            continue
        pivot = pred_df.pivot(index="date", columns="stock", values=f"pred_{name}")
        signal = _cross_sectional_signal(pivot, score_rank_blend)
        frames.append(signal * weight)
    if not frames:
        return pd.DataFrame(index=all_dates, columns=all_stocks, dtype=float)
    score = sum(frame.reindex(index=all_dates, columns=all_stocks).fillna(0.0) for frame in frames)
    downside_col = "pred_risk_downside_20"
    if downside_col in pred_df.columns and score_downside_penalty > 0:
        downside_pivot = pred_df.pivot(index="date", columns="stock", values=downside_col)
        risk_signal = _cross_sectional_signal(-downside_pivot, score_rank_blend).reindex(index=all_dates, columns=all_stocks).fillna(0.0)
        if score_risk_mode == "subtract":
            score = score - float(score_downside_penalty) * risk_signal
        elif score_risk_mode == "gate":
            safe_rank = downside_pivot.rank(axis=1, pct=True).reindex(index=all_dates, columns=all_stocks).fillna(0.5)
            score = score.where(safe_rank >= float(score_risk_gate_threshold))
        elif score_risk_mode in {"state_gate", "state_liquidity_gate"}:
            pass
        else:
            raise ValueError(f"Unsupported score_risk_mode: {score_risk_mode}")
    valid = pd.concat([frame.reindex(index=all_dates, columns=all_stocks).notna() for frame in frames]).groupby(level=0).max()
    return score.where(valid)


def _panel_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock", value_name])
    return (
        frame.copy()
        .rename_axis(index="date", columns="stock")
        .stack()
        .rename(value_name)
        .reset_index()
    )


def _select_month_end_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work.index = pd.DatetimeIndex(pd.to_datetime(work.index))
    last_in_month = work.index.to_series().groupby(work.index.to_period("M")).transform("max")
    return work.loc[work.index == pd.DatetimeIndex(last_in_month)]


def _summarize_rankic_by_month(rankic_ts: pd.DataFrame) -> pd.DataFrame:
    columns = ["month", "target", "rankic_mean", "rankic_std", "rankic_ir", "observation_count"]
    if rankic_ts.empty:
        return pd.DataFrame(columns=columns)
    work = rankic_ts.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["month"] = work["date"].dt.to_period("M").astype(str)
    summary = (
        work.groupby(["month", "target"], sort=True)["rankic"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "rankic_mean", "std": "rankic_std", "count": "observation_count"})
    )
    summary["rankic_ir"] = summary["rankic_mean"] / summary["rankic_std"].replace(0, np.nan)
    return summary[columns]


def _run_write_tasks(write_tasks: list[tuple[str, Any]]) -> None:
    if not write_tasks:
        return
    total = len(write_tasks)
    for index, (label, writer) in enumerate(write_tasks, start=1):
        progress_write(f"Write artifact {index}/{total}: {label}")
        writer()


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _write_resume_artifacts(
    *,
    run_dir: Path,
    model_artifact: dict[str, Any],
    history_rows: list[dict[str, Any]],
    status_payload: dict[str, Any],
) -> None:
    _atomic_write_csv(pd.DataFrame(history_rows), run_dir / "train_history.csv")
    _atomic_torch_save(model_artifact, run_dir / "deep_alpha_model.pt")
    _atomic_write_text(
        run_dir / "resume_status.json",
        json.dumps(status_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_resume_invariant_contract(
    *,
    args: argparse.Namespace,
    cfg: DeepAlphaConfig,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    feature_names: list[str],
    target_names: list[str],
    stocks_file: str | None,
) -> dict[str, Any]:
    return {
        "framework": "deep_alpha_research",
        "data_source": str(args.data_source),
        "start_date": str(cfg.start_date),
        "end_date": str(cfg.end_date),
        "benchmark": str(cfg.benchmark),
        "universe_scope": str(cfg.universe_scope),
        "stocks_file": str(stocks_file or ""),
        "liquidity_pool": str(args.liquidity_pool or ""),
        "rolling_liquidity_pool": str(args.rolling_liquidity_pool or ""),
        "pool_rebalance_days": int(args.pool_rebalance_days),
        "pool_adv_window": int(args.pool_adv_window),
        "research_time_unit": str(cfg.research_time_unit),
        "train_end": str(train_end.date()),
        "valid_start": str(valid_start.date()),
        "valid_end": str(valid_end.date()),
        "valid_days": int(cfg.valid_days),
        "valid_months": int(cfg.valid_months),
        "train_eval_window_days": int(cfg.train_eval_window_days),
        "train_eval_window_months": int(cfg.train_eval_window_months),
        "lookback_window": int(cfg.lookback_window),
        "prediction_horizons": [int(item) for item in cfg.prediction_horizons],
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "hidden_dim": int(cfg.hidden_dim),
        "encoder_family": str(cfg.encoder_family),
        "patch_len": int(cfg.patch_len),
        "return_head_mode": str(cfg.return_head_mode),
        "context_dim": int(cfg.context_dim),
        "state_context": bool(cfg.state_context),
        "liquidity_context": bool(cfg.liquidity_context),
        "structure_context": bool(cfg.structure_context),
        "aux_structure_task": bool(cfg.aux_structure_task),
        "aux_structure_loss_weight": float(cfg.aux_structure_loss_weight),
        "aux_structure_label_smoothing": float(cfg.aux_structure_label_smoothing),
        "structure_prototype_task": bool(cfg.structure_prototype_task),
        "structure_prototype_loss_weight": float(cfg.structure_prototype_loss_weight),
        "structure_prototype_temperature": float(cfg.structure_prototype_temperature),
        "transformer_heads": int(cfg.transformer_heads),
        "transformer_layers": int(cfg.transformer_layers),
        "dropout": float(cfg.dropout),
        "learning_rate": float(cfg.learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "return_loss_mode": str(cfg.return_loss_mode),
        "return_target_transform": str(cfg.return_target_transform),
        "return_top_frac": float(cfg.return_top_frac),
        "return_bottom_frac": float(cfg.return_bottom_frac),
        "target_loss_weights": dict(cfg.target_loss_weights),
        "score_horizon_weights": dict(cfg.score_horizon_weights),
        "score_rank_blend": float(cfg.score_rank_blend),
        "score_downside_penalty": float(cfg.score_downside_penalty),
        "score_risk_mode": str(cfg.score_risk_mode),
        "score_risk_gate_threshold": float(cfg.score_risk_gate_threshold),
        "score_risk_state_thresholds": parse_float_list(args.score_risk_state_thresholds),
        "score_head_method": str(args.score_head_method),
        "adaptive_task_weights": bool(args.adaptive_task_weights),
        "adaptive_task_window_days": int(args.adaptive_task_window_days),
        "adaptive_task_window_months": int(args.adaptive_task_window_months),
        "ranking_loss_weight": float(cfg.ranking_loss_weight),
        "listwise_loss_weight": float(cfg.listwise_loss_weight),
        "listwise_temperature": float(cfg.listwise_temperature),
        "max_rank_pairs_per_group": int(cfg.max_rank_pairs_per_group),
        "liquidity_conditioning_mode": str(cfg.liquidity_conditioning_mode),
        "top_liquidity_return_loss_weight": float(cfg.top_liquidity_return_loss_weight),
        "other_liquidity_return_loss_weight": float(cfg.other_liquidity_return_loss_weight),
        "top_liquidity_rank_loss_weight": float(cfg.top_liquidity_rank_loss_weight),
        "other_liquidity_rank_loss_weight": float(cfg.other_liquidity_rank_loss_weight),
        "top_liquidity_sample_weight": float(cfg.top_liquidity_sample_weight),
        "other_liquidity_sample_weight": float(cfg.other_liquidity_sample_weight),
        "structure_conditioning_mode": str(cfg.structure_conditioning_mode),
        "top_attack_structure_names": list(cfg.top_attack_structure_names),
        "other_protect_structure_names": list(cfg.other_protect_structure_names),
        "top_attack_rank_weight": float(cfg.top_attack_rank_weight),
        "other_protect_rank_weight": float(cfg.other_protect_rank_weight),
        "target_state_names": list(cfg.target_state_names),
        "target_state_attack_structure_names": list(cfg.target_state_attack_structure_names),
        "target_state_protect_structure_names": list(cfg.target_state_protect_structure_names),
        "target_state_rank_weight": float(cfg.target_state_rank_weight),
        "target_state_protect_rank_weight": float(cfg.target_state_protect_rank_weight),
        "market_state_count": int(cfg.market_state_count),
        "holding_count": int(cfg.holding_count),
        "rebalance_freq": str(cfg.rebalance_freq),
        "max_weight": float(cfg.max_weight),
        "min_adv20": float(cfg.min_adv20),
        "min_price": float(cfg.min_price),
        "max_price": float(cfg.max_price),
        "liquidity_layer": bool(cfg.liquidity_layer),
        "liquidity_bucket_count": int(cfg.liquidity_bucket_count),
        "dynamic_graph_layer": bool(cfg.dynamic_graph_layer),
        "dynamic_graph_top_k": int(cfg.dynamic_graph_top_k),
        "dynamic_graph_temperature": float(cfg.dynamic_graph_temperature),
        "dynamic_graph_industry_boost": float(cfg.dynamic_graph_industry_boost),
        "dynamic_graph_style_boost": float(cfg.dynamic_graph_style_boost),
        "short_alpha_features": bool(cfg.short_alpha_features),
        "breakout_event_horizon": int(cfg.breakout_event_horizon),
        "breakout_event_threshold": float(cfg.breakout_event_threshold),
        "breakout_event_pullback_limit": float(cfg.breakout_event_pullback_limit),
        "breakout_event_loss_weight": float(cfg.breakout_event_loss_weight),
        "clean_breakout_event_loss_weight": float(cfg.clean_breakout_event_loss_weight),
        "research_objective_mode": str(args.research_objective_mode),
        "checkpoint_selection_objective": str(args.checkpoint_selection_objective),
        "checkpoint_selection_min_improvement": float(args.checkpoint_selection_min_improvement),
        "checkpoint_eval_interval": int(args.checkpoint_eval_interval),
        "checkpoint_eval_start_epoch": int(args.checkpoint_eval_start_epoch),
        "execution_alignment_mode": str(args.execution_alignment_mode),
        "execution_alignment_objective": str(args.execution_alignment_objective),
        "execution_alignment_candidate_profiles": parse_execution_alignment_profile_names(
            args.execution_alignment_candidate_profiles
        ),
        "execution_alignment_shortlist_size": int(args.execution_alignment_shortlist_size),
        "execution_alignment_screen_window_days": int(args.execution_alignment_screen_window_days),
        "execution_alignment_transaction_cost_bps": float(args.execution_alignment_transaction_cost_bps),
        "execution_alignment_slippage_bps": float(args.execution_alignment_slippage_bps),
        "execution_alignment_sell_tax_bps": float(args.execution_alignment_sell_tax_bps),
        "random_seed": int(cfg.random_seed),
        "pretrained_encoder_path": str(cfg.pretrained_encoder_path or ""),
    }


def _build_run_contract(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    resume_invariant_contract: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "contract_schema_version": 1,
        "experiment_tag": str(args.experiment_tag or "").strip(),
        "run_dir": str(run_dir.resolve()),
        "epochs": int(args.epochs),
        "min_epochs": int(args.min_epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "lr_plateau_patience": int(args.lr_plateau_patience),
        "lr_plateau_factor": float(args.lr_plateau_factor),
        "min_improvement": float(args.min_improvement),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "use_amp": bool(args.use_amp),
        "safe_runtime_profile": bool(args.safe_runtime_profile),
        **runtime_metadata(),
        **resume_invariant_contract,
    }
    payload["resume_invariant_contract_fingerprint"] = contract_fingerprint(resume_invariant_contract)
    payload["run_contract_fingerprint"] = contract_fingerprint(payload)
    return payload


def _build_legacy_resume_contract(
    *,
    artifact: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    config = dict(artifact.get("config") or {})
    observed: dict[str, Any] = {}
    field_sources = (
        metrics,
        config,
        artifact,
    )
    keys = [
        "framework",
        "data_source",
        "start_date",
        "end_date",
        "benchmark",
        "universe_scope",
        "stocks_file",
        "liquidity_pool",
        "rolling_liquidity_pool",
        "pool_rebalance_days",
        "pool_adv_window",
        "research_time_unit",
        "train_end",
        "valid_start",
        "valid_end",
        "valid_days",
        "valid_months",
        "train_eval_window_days",
        "train_eval_window_months",
        "lookback_window",
        "prediction_horizons",
        "feature_names",
        "target_names",
        "hidden_dim",
        "encoder_family",
        "patch_len",
        "return_head_mode",
        "context_dim",
        "state_context",
        "liquidity_context",
        "structure_context",
        "aux_structure_task",
        "aux_structure_loss_weight",
        "aux_structure_label_smoothing",
        "structure_prototype_task",
        "structure_prototype_loss_weight",
        "structure_prototype_temperature",
        "transformer_heads",
        "transformer_layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "return_loss_mode",
        "return_target_transform",
        "return_top_frac",
        "return_bottom_frac",
        "target_loss_weights",
        "score_horizon_weights",
        "score_rank_blend",
        "score_downside_penalty",
        "score_risk_mode",
        "score_risk_gate_threshold",
        "score_risk_state_thresholds",
        "score_head_method",
        "adaptive_task_weights",
        "adaptive_task_window_days",
        "adaptive_task_window_months",
        "ranking_loss_weight",
        "listwise_loss_weight",
        "listwise_temperature",
        "max_rank_pairs_per_group",
        "liquidity_conditioning_mode",
        "top_liquidity_return_loss_weight",
        "other_liquidity_return_loss_weight",
        "top_liquidity_rank_loss_weight",
        "other_liquidity_rank_loss_weight",
        "top_liquidity_sample_weight",
        "other_liquidity_sample_weight",
        "structure_conditioning_mode",
        "top_attack_structure_names",
        "other_protect_structure_names",
        "top_attack_rank_weight",
        "other_protect_rank_weight",
        "target_state_names",
        "target_state_attack_structure_names",
        "target_state_protect_structure_names",
        "target_state_rank_weight",
        "target_state_protect_rank_weight",
        "market_state_count",
        "holding_count",
        "rebalance_freq",
        "max_weight",
        "min_adv20",
        "min_price",
        "max_price",
        "liquidity_layer",
        "liquidity_bucket_count",
        "dynamic_graph_layer",
        "dynamic_graph_top_k",
        "dynamic_graph_temperature",
        "dynamic_graph_industry_boost",
        "dynamic_graph_style_boost",
        "short_alpha_features",
        "breakout_event_horizon",
        "breakout_event_threshold",
        "breakout_event_pullback_limit",
        "breakout_event_loss_weight",
        "clean_breakout_event_loss_weight",
        "research_objective_mode",
        "checkpoint_selection_objective",
        "checkpoint_selection_min_improvement",
        "checkpoint_eval_interval",
        "checkpoint_eval_start_epoch",
        "execution_alignment_mode",
        "execution_alignment_objective",
        "execution_alignment_candidate_profiles",
        "execution_alignment_shortlist_size",
        "execution_alignment_screen_window_days",
        "execution_alignment_transaction_cost_bps",
        "execution_alignment_slippage_bps",
        "execution_alignment_sell_tax_bps",
        "random_seed",
        "pretrained_encoder_path",
    ]
    for key in keys:
        for source in field_sources:
            if key not in source:
                continue
            value = source.get(key)
            if value in (None, "", {}):
                continue
            observed[key] = value
            break
    if "framework" not in observed:
        observed["framework"] = "deep_alpha_research"
    return observed


def _resolve_resume_context(args: argparse.Namespace) -> dict[str, Any] | None:
    resume_mode = str(getattr(args, "resume_mode", "off") or "off").strip().lower()
    resume_model_path = str(getattr(args, "resume_model_path", "") or "").strip()
    resume_run_dir = str(getattr(args, "resume_run_dir", "") or "").strip()
    if resume_mode == "off":
        if resume_model_path or resume_run_dir:
            raise ValueError("Set --resume-mode strict|warm_start when providing --resume-model-path or --resume-run-dir.")
        return None

    if resume_model_path:
        artifact_path = Path(resume_model_path).expanduser().resolve()
        run_dir = artifact_path.parent
    elif resume_run_dir:
        run_dir = Path(resume_run_dir).expanduser().resolve()
        artifact_path = run_dir / "deep_alpha_model.pt"
    else:
        raise ValueError("--resume-mode requires --resume-model-path or --resume-run-dir.")

    if not artifact_path.exists():
        raise FileNotFoundError(f"Resume artifact not found: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"Resume artifact is not a dict: {artifact_path}")

    history_rows: list[dict[str, Any]] = []
    history_path = run_dir / "train_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        history_rows = history_df.to_dict(orient="records")

    metrics_path = run_dir / "metrics.json"
    metrics_payload: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            loaded_metrics = json.load(f)
        if isinstance(loaded_metrics, dict):
            metrics_payload = loaded_metrics

    training_resume_state = artifact.get("training_resume_state") or artifact.get("training_state") or {}
    if resume_mode == "strict":
        if not artifact.get("last_model_state_dict"):
            raise ValueError(
                "Strict resume requires an artifact with last_model_state_dict. "
                "This source run was saved before strict-resume support was added; use --resume-mode warm_start instead."
            )
        if not training_resume_state:
            raise ValueError(
                "Strict resume requires training_resume_state in the source artifact. "
                "This source run does not have resumable optimizer/scheduler state."
            )

    return {
        "resume_mode": resume_mode,
        "artifact_path": artifact_path,
        "run_dir": run_dir,
        "artifact": artifact,
        "history_rows": history_rows,
        "metrics": metrics_payload,
        "training_resume_state": dict(training_resume_state),
    }


def _resolve_manual_score_config(
    *,
    base_horizon_weights: dict[int, float],
    base_downside_penalty: float,
    task_weights: dict[str, float],
) -> tuple[dict[int, float], float]:
    if not task_weights:
        return dict(base_horizon_weights), float(base_downside_penalty)

    derived_returns: dict[int, float] = {}
    for horizon in sorted(base_horizon_weights):
        weight = float(task_weights.get(f"fwd_excess_{int(horizon)}", 0.0))
        if weight > 0:
            derived_returns[int(horizon)] = weight

    if not derived_returns:
        return dict(base_horizon_weights), float(base_downside_penalty)

    total_return_weight = sum(derived_returns.values())
    horizon_weights = {
        int(horizon): float(weight / total_return_weight)
        for horizon, weight in derived_returns.items()
    }

    downside_penalty = float(base_downside_penalty)
    risk_weight = float(task_weights.get("risk_downside_20", 0.0))
    if risk_weight > 0 and total_return_weight > 0:
        downside_penalty = risk_weight / total_return_weight
    return horizon_weights, downside_penalty


def _build_live_inference_outputs(
    *,
    cfg: DeepAlphaConfig,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    benchmark_open: pd.Series,
    open_df: pd.DataFrame,
    amount_frame: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
    train_target_frames: dict[str, pd.DataFrame],
    target_frames: dict[str, pd.DataFrame],
    state_frame: pd.DataFrame,
    liquidity_bucket_frame: pd.DataFrame | None,
    structure_label_frame: pd.DataFrame | None,
    rolling_membership_frame: pd.DataFrame | None,
    live_start: pd.Timestamp,
    model: MultiTaskRanker,
    loader_kwargs: dict[str, Any],
    device: torch.device,
    target_names: list[str],
    score_head_method: str,
    score_head_artifact: Any,
    applied_score_horizon_weights: dict[int, float],
    applied_score_downside_penalty: float,
    risk_gate_artifact: Any,
    use_amp: bool,
) -> dict[str, Any]:
    latest_market_date = pd.Timestamp(close.index.max())
    live_dates = list(close.index[(close.index >= live_start) & (close.index <= latest_market_date)])
    empty_panel = pd.DataFrame(index=pd.Index(live_dates), columns=close.columns, dtype=float)
    empty_pred = pd.DataFrame(columns=["date", "stock"])
    empty_emb = pd.DataFrame(columns=["date", "stock"])
    if not live_dates:
        return {
            "live_dates": pd.Index([]),
            "pred_df": empty_pred,
            "emb_df": empty_emb,
            "score_frame": empty_panel.copy(),
            "target_weights": empty_panel.copy(),
        }

    live_corpus = build_sequence_corpus(
        feature_frames=feature_frames,
        train_target_frames=train_target_frames,
        raw_target_frames=target_frames,
        lookback_window=cfg.lookback_window,
        sample_dates=live_dates,
        min_adv20=cfg.min_adv20,
        amount_frame=amount_frame,
        close_frame=close,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
        state_frame=state_frame,
        liquidity_bucket_frame=liquidity_bucket_frame,
        structure_label_frame=structure_label_frame,
        universe_membership_frame=rolling_membership_frame,
        require_targets=False,
    )
    live_indices = live_corpus.build_index(start_date=live_start, end_date=latest_market_date)
    if not live_indices:
        return {
            "live_dates": pd.Index(live_dates),
            "pred_df": empty_pred,
            "emb_df": empty_emb,
            "score_frame": empty_panel.copy(),
            "target_weights": empty_panel.copy(),
        }

    live_ds = StockSequenceDataset(corpus=live_corpus, indices=live_indices)
    live_loader = DataLoader(
        live_ds,
        batch_sampler=DateGroupedBatchSampler(
            live_ds.meta,
            batch_size=cfg.batch_size,
            shuffle_dates=False,
            random_seed=cfg.random_seed,
        ),
        **loader_kwargs,
    )
    live_pred_df, live_emb_df = infer_dataset(
        model,
        live_loader,
        device,
        target_names,
        use_amp=bool(use_amp),
        task_label="Inference live",
    )
    live_index = pd.Index(live_dates)
    live_score_outputs = pd.DataFrame()
    if score_head_method == "manual":
        live_score_frame = _build_score_frame(
            pred_df=live_pred_df,
            target_names=target_names,
            horizon_weights=applied_score_horizon_weights,
            score_rank_blend=cfg.score_rank_blend,
            score_downside_penalty=applied_score_downside_penalty,
            score_risk_mode=cfg.score_risk_mode,
            score_risk_gate_threshold=cfg.score_risk_gate_threshold,
            all_dates=live_index,
            all_stocks=list(close.columns),
        )
    else:
        live_score_outputs = apply_score_head(score_head_artifact, live_pred_df, target_names)
        if is_policy_score_head_method(score_head_method):
            live_score_frame = build_policy_decision_score_frame(
                live_score_outputs,
                all_dates=live_index,
                all_stocks=list(close.columns),
                holding_count=cfg.holding_count,
                artifact=score_head_artifact,
            )
        else:
            live_score_frame = (
                live_score_outputs
                .pivot(index="date", columns="stock", values="learned_score")
                .reindex(index=live_index, columns=close.columns)
            )
    if rolling_membership_frame is not None:
        membership = rolling_membership_frame.reindex(index=live_score_frame.index, columns=live_score_frame.columns).fillna(False)
        live_score_frame = live_score_frame.where(membership)
    if risk_gate_artifact is not None:
        live_score_frame = apply_state_risk_gate(
            score_frame=live_score_frame,
            pred_df=live_pred_df,
            state_frame=state_frame,
            artifact=risk_gate_artifact,
            liquidity_bucket_frame=liquidity_bucket_frame if cfg.score_risk_mode == "state_liquidity_gate" else None,
        )
    live_cfg = ResearchConfig(
        start_date=str(pd.Timestamp(live_start).date()).replace("-", ""),
        end_date=str(latest_market_date.date()).replace("-", ""),
        benchmark=cfg.benchmark,
        execution_mode="next_open",
        holding_count=cfg.holding_count,
        weighting_method="score",
        rebalance_freq=cfg.rebalance_freq,
        max_weight=cfg.max_weight,
        min_adv20=cfg.min_adv20,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
    )
    portfolio_live_target_weights = build_target_weights(live_score_frame, live_cfg)
    if is_policy_score_head_method(score_head_method):
        live_target_weights = build_policy_target_weight_frame(
            live_score_outputs,
            all_dates=live_index,
            all_stocks=list(close.columns),
            holding_count=cfg.holding_count,
            artifact=score_head_artifact,
        )
    else:
        live_target_weights = build_research_raw_target_weights(live_score_frame, live_cfg)
    return {
        "live_dates": live_index,
        "pred_df": live_pred_df,
        "emb_df": live_emb_df,
        "score_frame": live_score_frame,
        "target_weights": live_target_weights,
        "portfolio_capped_target_weights": portfolio_live_target_weights,
    }


def _build_live_execution_aligned_outputs(
    *,
    execution_alignment_artifact: Any,
    raw_live_score_frame: pd.DataFrame,
    raw_live_target_weights: pd.DataFrame,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    cfg: DeepAlphaConfig,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> dict[str, Any] | None:
    if execution_alignment_artifact is None:
        return None
    profile = resolve_execution_alignment_profile(
        name=str(getattr(execution_alignment_artifact, "selected_profile", "") or ""),
        spec=getattr(execution_alignment_artifact, "selected_profile_spec", None),
    )
    metrics, aligned_scores, aligned_target_weights, meta = evaluate_execution_alignment_profile(
        raw_target_weights=raw_live_target_weights.reindex(raw_live_score_frame.index).fillna(0.0),
        raw_score_frame=raw_live_score_frame.fillna(0.0),
        close=close,
        benchmark_close=benchmark_close,
        open_df=open_df,
        benchmark_open=benchmark_open,
        benchmark=cfg.benchmark,
        holding_count=cfg.holding_count,
        max_weight=cfg.max_weight,
        min_adv20=cfg.min_adv20,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        profile=profile,
    )
    return {
        "score_frame": aligned_scores,
        "target_weights": aligned_target_weights,
        "metrics": metrics,
        "meta": meta,
    }


def _summarize_structure_outputs(pred_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    structure_aux_summary: dict[str, Any] = {}
    if {"structure_id", "pred_structure_id"}.issubset(pred_df.columns):
        valid_structure = pred_df["structure_id"].notna() & pred_df["pred_structure_id"].notna()
        if bool(valid_structure.any()):
            structure_aux_summary = {
                "structure_aux_accuracy": float(
                    (
                        pred_df.loc[valid_structure, "structure_id"].astype(int)
                        == pred_df.loc[valid_structure, "pred_structure_id"].astype(int)
                    ).mean()
                ),
                "structure_aux_samples": int(valid_structure.sum()),
            }
    structure_prototype_summary: dict[str, Any] = {}
    if {"structure_id", "pred_prototype_structure_id"}.issubset(pred_df.columns):
        valid_proto = pred_df["structure_id"].notna() & pred_df["pred_prototype_structure_id"].notna()
        if bool(valid_proto.any()):
            structure_prototype_summary = {
                "structure_prototype_accuracy": float(
                    (
                        pred_df.loc[valid_proto, "structure_id"].astype(int)
                        == pred_df.loc[valid_proto, "pred_prototype_structure_id"].astype(int)
                    ).mean()
                ),
                "structure_prototype_samples": int(valid_proto.sum()),
            }
    return structure_aux_summary, structure_prototype_summary


def _evaluate_research_outputs(
    *,
    model: MultiTaskRanker,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
    cfg: DeepAlphaConfig,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    train_ds: StockSequenceDataset,
    train_eval_loader: DataLoader,
    valid_loader: DataLoader,
    train_eval_dates: list[pd.Timestamp],
    valid_dates: list[pd.Timestamp],
    state_frame: pd.DataFrame,
    liquidity_bucket_frame: pd.DataFrame | None,
    rolling_membership_frame: pd.DataFrame | None,
) -> dict[str, Any]:
    train_pred_df, _ = infer_dataset(
        model,
        train_eval_loader,
        device,
        train_ds.target_names,
        use_amp=bool(use_amp),
        task_label="Inference train_eval",
    )
    pred_df, emb_df = infer_dataset(
        model,
        valid_loader,
        device,
        train_ds.target_names,
        use_amp=bool(use_amp),
        task_label="Inference holdout",
    )
    rankic_timeseries = compute_rankic_timeseries(pred_df, train_ds.target_names)
    rankic_summary = compute_rankic(pred_df, train_ds.target_names)
    monthly_rankic_summary = _summarize_rankic_by_month(rankic_timeseries)
    structure_aux_summary, structure_prototype_summary = _summarize_structure_outputs(pred_df)

    score_head_artifact = fit_score_head(
        train_pred_df=train_pred_df,
        target_names=train_ds.target_names,
        method=args.score_head_method,
        adaptive_task_weights=bool(args.adaptive_task_weights),
        adaptive_window_days=args.adaptive_task_window_days,
        research_time_unit=cfg.research_time_unit,
        adaptive_window_months=cfg.adaptive_task_window_months,
        holding_count=cfg.holding_count,
    )
    applied_score_horizon_weights = dict(cfg.score_horizon_weights)
    applied_score_downside_penalty = float(cfg.score_downside_penalty)
    train_score_outputs = pd.DataFrame()
    score_outputs = pd.DataFrame()
    if args.score_head_method == "manual":
        applied_score_horizon_weights, applied_score_downside_penalty = _resolve_manual_score_config(
            base_horizon_weights=cfg.score_horizon_weights,
            base_downside_penalty=cfg.score_downside_penalty,
            task_weights=score_head_artifact.task_weights,
        )
        train_score_frame = _build_score_frame(
            pred_df=train_pred_df,
            target_names=train_ds.target_names,
            horizon_weights=applied_score_horizon_weights,
            score_rank_blend=cfg.score_rank_blend,
            score_downside_penalty=applied_score_downside_penalty,
            score_risk_mode=cfg.score_risk_mode,
            score_risk_gate_threshold=cfg.score_risk_gate_threshold,
            all_dates=pd.Index(train_eval_dates),
            all_stocks=list(close.columns),
        )
        score_frame = _build_score_frame(
            pred_df=pred_df,
            target_names=train_ds.target_names,
            horizon_weights=applied_score_horizon_weights,
            score_rank_blend=cfg.score_rank_blend,
            score_downside_penalty=applied_score_downside_penalty,
            score_risk_mode=cfg.score_risk_mode,
            score_risk_gate_threshold=cfg.score_risk_gate_threshold,
            all_dates=pd.Index(valid_dates),
            all_stocks=list(close.columns),
        )
    else:
        train_score_outputs = apply_score_head(score_head_artifact, train_pred_df, train_ds.target_names)
        score_outputs = apply_score_head(score_head_artifact, pred_df, train_ds.target_names)
        if is_policy_score_head_method(args.score_head_method):
            train_score_frame = build_policy_decision_score_frame(
                train_score_outputs,
                all_dates=pd.Index(train_eval_dates),
                all_stocks=list(close.columns),
                holding_count=cfg.holding_count,
                artifact=score_head_artifact,
            )
            score_frame = build_policy_decision_score_frame(
                score_outputs,
                all_dates=pd.Index(valid_dates),
                all_stocks=list(close.columns),
                holding_count=cfg.holding_count,
                artifact=score_head_artifact,
            )
        else:
            train_score_frame = (
                train_score_outputs
                .pivot(index="date", columns="stock", values="learned_score")
                .reindex(index=train_eval_dates, columns=close.columns)
            )
            score_frame = (
                score_outputs
                .pivot(index="date", columns="stock", values="learned_score")
                .reindex(index=valid_dates, columns=close.columns)
            )
    if rolling_membership_frame is not None:
        train_score_frame = train_score_frame.where(
            rolling_membership_frame.reindex(index=train_score_frame.index, columns=train_score_frame.columns).fillna(False)
        )
        score_frame = score_frame.where(
            rolling_membership_frame.reindex(index=score_frame.index, columns=score_frame.columns).fillna(False)
        )

    risk_gate_artifact = None
    if args.score_risk_mode in {"state_gate", "state_liquidity_gate"}:
        risk_gate_artifact = fit_state_risk_gate(
            train_pred_df=train_pred_df,
            state_frame=state_frame,
            target_names=train_ds.target_names,
            horizon_weights=cfg.score_horizon_weights,
            score_rank_blend=cfg.score_rank_blend,
            holding_count=cfg.holding_count,
            candidate_thresholds=parse_float_list(args.score_risk_state_thresholds),
            liquidity_bucket_frame=liquidity_bucket_frame if args.score_risk_mode == "state_liquidity_gate" else None,
        )
        train_score_frame = apply_state_risk_gate(
            score_frame=train_score_frame,
            pred_df=train_pred_df,
            state_frame=state_frame,
            artifact=risk_gate_artifact,
            liquidity_bucket_frame=liquidity_bucket_frame if args.score_risk_mode == "state_liquidity_gate" else None,
        )
        score_frame = apply_state_risk_gate(
            score_frame=score_frame,
            pred_df=pred_df,
            state_frame=state_frame,
            artifact=risk_gate_artifact,
            liquidity_bucket_frame=liquidity_bucket_frame if args.score_risk_mode == "state_liquidity_gate" else None,
        )

    train_eval_cfg = ResearchConfig(
        start_date="" if len(train_eval_dates) == 0 else str(pd.Timestamp(train_eval_dates[0]).date()).replace("-", ""),
        end_date="" if len(train_eval_dates) == 0 else str(pd.Timestamp(train_eval_dates[-1]).date()).replace("-", ""),
        benchmark=cfg.benchmark,
        execution_mode="next_open",
        holding_count=cfg.holding_count,
        weighting_method="score",
        rebalance_freq=cfg.rebalance_freq,
        max_weight=cfg.max_weight,
        min_adv20=cfg.min_adv20,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
    )
    research_cfg = ResearchConfig(
        start_date="" if len(valid_dates) == 0 else str(pd.Timestamp(valid_dates[0]).date()).replace("-", ""),
        end_date="" if len(valid_dates) == 0 else str(pd.Timestamp(valid_dates[-1]).date()).replace("-", ""),
        benchmark=cfg.benchmark,
        execution_mode="next_open",
        holding_count=cfg.holding_count,
        weighting_method="score",
        rebalance_freq=cfg.rebalance_freq,
        max_weight=cfg.max_weight,
        min_adv20=cfg.min_adv20,
        min_price=cfg.min_price,
        max_price=cfg.max_price,
    )
    if is_policy_score_head_method(args.score_head_method):
        train_eval_target_weights = build_policy_target_weight_frame(
            train_score_outputs,
            all_dates=pd.Index(train_eval_dates),
            all_stocks=list(close.columns),
            holding_count=cfg.holding_count,
            artifact=score_head_artifact,
        )
        target_weights = build_policy_target_weight_frame(
            score_outputs,
            all_dates=pd.Index(valid_dates),
            all_stocks=list(close.columns),
            holding_count=cfg.holding_count,
            artifact=score_head_artifact,
        )
    else:
        train_eval_target_weights = build_target_weights(train_score_frame, train_eval_cfg)
        target_weights = build_target_weights(score_frame, research_cfg)
    equity_df, action_df, holdout_metrics = backtest(
        close=close.reindex(valid_dates),
        benchmark_close=benchmark_close.reindex(valid_dates),
        target_weights=target_weights.reindex(valid_dates),
        target_scores=score_frame.reindex(valid_dates).fillna(0.0),
        config=research_cfg,
        regime_on=pd.Series(True, index=pd.Index(valid_dates)),
        open_df=open_df.reindex(valid_dates),
        benchmark_open=benchmark_open.reindex(valid_dates),
    )
    monthly_backtest_summary = summarize_backtest_by_month(equity_df, action_df)
    monthly_backtest_diagnostics = summarize_monthly_diagnostics(
        monthly_backtest_summary,
        return_column="excess_return",
    )

    execution_alignment_artifact = None
    execution_aligned_equity_df = None
    execution_aligned_action_df = None
    execution_aligned_metrics = None
    execution_aligned_score_frame = None
    execution_aligned_target_weights = None
    execution_aligned_monthly_backtest_summary = None
    execution_aligned_monthly_backtest_diagnostics = None
    if args.execution_alignment_mode != "off":
        if args.execution_alignment_mode == "profile":
            candidate_profiles = [get_execution_alignment_profile(args.execution_alignment_profile).name]
        else:
            candidate_profiles = parse_execution_alignment_profile_names(args.execution_alignment_candidate_profiles)
        execution_alignment_artifact = fit_execution_alignment(
            mode=args.execution_alignment_mode,
            objective_metric=args.execution_alignment_objective,
            candidate_profiles=candidate_profiles,
            shortlist_size=int(args.execution_alignment_shortlist_size),
            screen_window_days=int(args.execution_alignment_screen_window_days),
            train_raw_target_weights=train_eval_target_weights.reindex(train_eval_dates).fillna(0.0),
            train_raw_score_frame=train_score_frame.reindex(train_eval_dates).fillna(0.0),
            valid_raw_target_weights=target_weights.reindex(valid_dates).fillna(0.0),
            valid_raw_score_frame=score_frame.reindex(valid_dates).fillna(0.0),
            close=close,
            benchmark_close=benchmark_close,
            open_df=open_df,
            benchmark_open=benchmark_open,
            benchmark=cfg.benchmark,
            holding_count=cfg.holding_count,
            max_weight=cfg.max_weight,
            min_adv20=cfg.min_adv20,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            transaction_cost_bps=args.execution_alignment_transaction_cost_bps,
            slippage_bps=args.execution_alignment_slippage_bps,
            sell_tax_bps=args.execution_alignment_sell_tax_bps,
        )
        execution_aligned_score_frame = execution_alignment_artifact.valid_score_frame.reindex(valid_dates).fillna(0.0)
        execution_aligned_target_weights = execution_alignment_artifact.valid_target_weights.reindex(valid_dates).fillna(0.0)
        execution_aligned_metrics = dict(execution_alignment_artifact.valid_metrics)
        execution_alignment_backtest_cfg = ResearchConfig(
            start_date="" if len(valid_dates) == 0 else str(pd.Timestamp(valid_dates[0]).date()).replace("-", ""),
            end_date="" if len(valid_dates) == 0 else str(pd.Timestamp(valid_dates[-1]).date()).replace("-", ""),
            benchmark=cfg.benchmark,
            execution_mode="next_open",
            holding_count=cfg.holding_count,
            weighting_method="score",
            rebalance_freq=str(execution_aligned_metrics.get("rebalance_freq", cfg.rebalance_freq)),
            max_weight=cfg.max_weight,
            min_adv20=cfg.min_adv20,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            transaction_cost_bps=args.execution_alignment_transaction_cost_bps,
            slippage_bps=args.execution_alignment_slippage_bps,
            sell_tax_bps=args.execution_alignment_sell_tax_bps,
            enable_market_regime_filter=bool(execution_aligned_metrics.get("market_regime_filter", False)),
            regime_ma_window=50,
            regime_vol_window=20,
            regime_max_annual_vol=0.32,
            regime_trend_flat_band=0.01,
            regime_vol_transition_band=0.10,
            regime_allowed_quadrants=["trend_up_low_vol", "trend_up_high_vol"],
        )
        execution_alignment_regime_on = compute_market_regime_state(
            benchmark_close,
            execution_alignment_backtest_cfg,
        )["regime_on"].reindex(valid_dates)
        execution_aligned_equity_df, execution_aligned_action_df, execution_aligned_metrics = backtest(
            close=close.reindex(valid_dates),
            benchmark_close=benchmark_close.reindex(valid_dates),
            target_weights=execution_aligned_target_weights,
            target_scores=execution_aligned_score_frame,
            config=execution_alignment_backtest_cfg,
            regime_on=execution_alignment_regime_on,
            open_df=open_df.reindex(valid_dates),
            benchmark_open=benchmark_open.reindex(valid_dates),
        )
        execution_aligned_monthly_backtest_summary = summarize_backtest_by_month(
            execution_aligned_equity_df,
            execution_aligned_action_df,
        )
        execution_aligned_monthly_backtest_diagnostics = summarize_monthly_diagnostics(
            execution_aligned_monthly_backtest_summary,
            return_column="excess_return",
        )

    primary_metrics_payload = {
        "holdout_backtest": dict(holdout_metrics),
        "execution_aligned_holdout_backtest": {}
        if execution_aligned_metrics is None
        else dict(execution_aligned_metrics),
        "research_objective_mode": str(args.research_objective_mode),
    }
    primary_backtest_label, primary_backtest = resolve_primary_backtest(
        primary_metrics_payload,
        research_objective_mode=args.research_objective_mode,
    )
    primary_monthly_summary_label = resolve_primary_monthly_summary_label(
        primary_metrics_payload,
        research_objective_mode=args.research_objective_mode,
    )
    primary_monthly_diagnostics_label = resolve_primary_monthly_diagnostics_label(
        primary_metrics_payload,
        research_objective_mode=args.research_objective_mode,
    )
    if primary_monthly_summary_label == "execution_aligned_monthly_backtest_summary" and execution_aligned_monthly_backtest_summary is not None:
        primary_monthly_summary = execution_aligned_monthly_backtest_summary
        primary_monthly_diagnostics = (
            execution_aligned_monthly_backtest_diagnostics
            if isinstance(execution_aligned_monthly_backtest_diagnostics, dict)
            else summarize_monthly_diagnostics(execution_aligned_monthly_backtest_summary, return_column="excess_return")
        )
    else:
        primary_monthly_summary = monthly_backtest_summary
        primary_monthly_diagnostics = monthly_backtest_diagnostics
    checkpoint_metric_name, checkpoint_metric_value = resolve_checkpoint_metric_value(
        primary_backtest,
        args.checkpoint_selection_objective,
        primary_monthly_diagnostics=primary_monthly_diagnostics,
    )
    primary_monthly_objectives = summarize_primary_monthly_objectives(primary_monthly_diagnostics)
    return {
        "train_pred_df": train_pred_df,
        "pred_df": pred_df,
        "emb_df": emb_df,
        "rankic_timeseries": rankic_timeseries,
        "rankic_summary": rankic_summary,
        "monthly_rankic_summary": monthly_rankic_summary,
        "structure_aux_summary": structure_aux_summary,
        "structure_prototype_summary": structure_prototype_summary,
        "score_head_artifact": score_head_artifact,
        "applied_score_horizon_weights": applied_score_horizon_weights,
        "applied_score_downside_penalty": applied_score_downside_penalty,
        "train_score_frame": train_score_frame,
        "score_frame": score_frame,
        "risk_gate_artifact": risk_gate_artifact,
        "train_eval_target_weights": train_eval_target_weights,
        "target_weights": target_weights,
        "equity_df": equity_df,
        "action_df": action_df,
        "monthly_backtest_summary": monthly_backtest_summary,
        "monthly_backtest_diagnostics": dict(monthly_backtest_diagnostics),
        "holdout_metrics": dict(holdout_metrics),
        "execution_alignment_artifact": execution_alignment_artifact,
        "execution_aligned_equity_df": execution_aligned_equity_df,
        "execution_aligned_action_df": execution_aligned_action_df,
        "execution_aligned_metrics": None if execution_aligned_metrics is None else dict(execution_aligned_metrics),
        "execution_aligned_score_frame": execution_aligned_score_frame,
        "execution_aligned_target_weights": execution_aligned_target_weights,
        "execution_aligned_monthly_backtest_summary": execution_aligned_monthly_backtest_summary,
        "execution_aligned_monthly_backtest_diagnostics": None
        if execution_aligned_monthly_backtest_diagnostics is None
        else dict(execution_aligned_monthly_backtest_diagnostics),
        "primary_backtest_label": primary_backtest_label,
        "primary_backtest": dict(primary_backtest),
        "primary_monthly_summary_label": str(primary_monthly_summary_label),
        "primary_monthly_summary": primary_monthly_summary,
        "primary_monthly_diagnostics_label": str(primary_monthly_diagnostics_label),
        "primary_monthly_diagnostics": dict(primary_monthly_diagnostics),
        "primary_monthly_objectives": dict(primary_monthly_objectives),
        "checkpoint_metric_name": checkpoint_metric_name,
        "checkpoint_metric_value": checkpoint_metric_value,
    }


def main():
    args = parse_args()
    args.research_time_unit = normalize_research_time_unit(args.research_time_unit)
    output_root = Path("daily_research/output")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or f"deep_alpha_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    explicit_resume_requested = bool(str(args.resume_run_dir or "").strip() or str(args.resume_model_path or "").strip())
    if not explicit_resume_requested and str(args.resume_mode or "off").strip().lower() == "off":
        resume_artifact_path = run_dir / "deep_alpha_model.pt"
        if resume_artifact_path.exists():
            args.resume_run_dir = str(run_dir.resolve())
            args.resume_mode = "strict"
            progress_write(f"Auto strict resume from existing run_dir: {run_dir.resolve()}")
    resume_context = _resolve_resume_context(args)
    if args.liquidity_pool and args.rolling_liquidity_pool:
        raise ValueError("Use either --liquidity-pool or --rolling-liquidity-pool, not both.")
    if args.rolling_liquidity_pool and (args.stocks or args.stocks_file):
        raise ValueError("Use either a fixed --stocks/--stocks-file universe or --rolling-liquidity-pool, not both.")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    horizons = parse_horizons(args.prediction_horizons)

    cfg = DeepAlphaConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        universe_scope=args.universe_scope,
        lookback_window=args.lookback_window,
        prediction_horizons=horizons,
        research_time_unit=args.research_time_unit,
        train_end_date=args.train_end_date,
        valid_start_date=args.valid_start_date,
        valid_days=args.valid_days,
        valid_months=args.valid_months,
        train_eval_window_days=args.train_eval_window_days,
        train_eval_window_months=args.train_eval_window_months,
        adaptive_task_window_months=args.adaptive_task_window_months,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        hidden_dim=args.hidden_dim,
        encoder_family=args.encoder_family,
        patch_len=args.patch_len,
        pretrained_encoder_path=args.pretrained_encoder_path,
        return_head_mode=args.return_head_mode,
        context_dim=args.context_dim,
        state_context=args.state_context,
        liquidity_context=args.liquidity_context,
        structure_context=args.structure_context,
        aux_structure_task=args.aux_structure_task,
        aux_structure_loss_weight=args.aux_structure_loss_weight,
        aux_structure_label_smoothing=args.aux_structure_label_smoothing,
        structure_prototype_task=args.structure_prototype_task,
        structure_prototype_loss_weight=args.structure_prototype_loss_weight,
        structure_prototype_temperature=args.structure_prototype_temperature,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        early_stop_patience=args.early_stop_patience,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_improvement=args.min_improvement,
        use_amp=args.use_amp,
        safe_runtime_profile=args.safe_runtime_profile,
        ranking_loss_weight=args.ranking_loss_weight,
        listwise_loss_weight=args.listwise_loss_weight,
        listwise_temperature=args.listwise_temperature,
        max_rank_pairs_per_group=args.max_rank_pairs_per_group,
        return_loss_mode=args.return_loss_mode,
        return_target_transform=args.return_target_transform,
        return_top_frac=args.return_top_frac,
        return_bottom_frac=args.return_bottom_frac,
        liquidity_conditioning_mode=args.liquidity_conditioning_mode,
        top_liquidity_return_loss_weight=args.top_liquidity_return_loss_weight,
        other_liquidity_return_loss_weight=args.other_liquidity_return_loss_weight,
        top_liquidity_rank_loss_weight=args.top_liquidity_rank_loss_weight,
        other_liquidity_rank_loss_weight=args.other_liquidity_rank_loss_weight,
        top_liquidity_sample_weight=args.top_liquidity_sample_weight,
        other_liquidity_sample_weight=args.other_liquidity_sample_weight,
        structure_conditioning_mode=args.structure_conditioning_mode,
        top_attack_structure_names=tuple(parse_name_list(args.top_attack_structures)),
        other_protect_structure_names=tuple(parse_name_list(args.other_protect_structures)),
        top_attack_rank_weight=args.top_attack_rank_weight,
        other_protect_rank_weight=args.other_protect_rank_weight,
        target_state_names=tuple(parse_name_list(args.target_state_names)),
        target_state_attack_structure_names=tuple(parse_name_list(args.target_state_attack_structures)),
        target_state_protect_structure_names=tuple(parse_name_list(args.target_state_protect_structures)),
        target_state_rank_weight=args.target_state_rank_weight,
        target_state_protect_rank_weight=args.target_state_protect_rank_weight,
        target_loss_weights=build_target_loss_weights(horizons, args.task_loss_weights),
        score_horizon_weights=build_score_horizon_weights(horizons, args.score_horizon_weights),
        score_rank_blend=args.score_rank_blend,
        score_downside_penalty=args.score_downside_penalty,
        score_risk_mode=args.score_risk_mode,
        score_risk_gate_threshold=args.score_risk_gate_threshold,
        liquidity_layer=args.liquidity_layer,
        liquidity_bucket_count=args.liquidity_bucket_count,
        dynamic_graph_layer=args.dynamic_graph_layer,
        dynamic_graph_top_k=args.dynamic_graph_top_k,
        dynamic_graph_temperature=args.dynamic_graph_temperature,
        dynamic_graph_industry_boost=args.dynamic_graph_industry_boost,
        dynamic_graph_style_boost=args.dynamic_graph_style_boost,
        short_alpha_features=args.short_alpha_features,
        breakout_event_horizon=args.breakout_event_horizon,
        breakout_event_threshold=args.breakout_event_threshold,
        breakout_event_pullback_limit=args.breakout_event_pullback_limit,
        breakout_event_loss_weight=args.breakout_event_loss_weight,
        clean_breakout_event_loss_weight=args.clean_breakout_event_loss_weight,
        random_seed=args.random_seed,
        market_state_count=args.market_state_count,
        holding_count=args.holding_count,
        rebalance_freq=args.rebalance_freq,
        max_weight=args.max_weight,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    if cfg.encoder_family == "ssm":
        cfg.encoder_family = "mamba"
    if not cfg.end_date:
        cfg.end_date = pd.Timestamp(get_latest_completed_trading_date()).strftime("%Y%m%d")
    if float(cfg.breakout_event_loss_weight) > 0:
        cfg.target_loss_weights[f"event_breakout_{int(cfg.breakout_event_horizon)}"] = float(cfg.breakout_event_loss_weight)
    if float(cfg.clean_breakout_event_loss_weight) > 0:
        cfg.target_loss_weights[f"event_clean_breakout_{int(cfg.breakout_event_horizon)}"] = float(cfg.clean_breakout_event_loss_weight)

    stage_progress = StageProgress(total=8, label="DeepAlpha")
    stage_progress.__enter__()
    stage_progress.start_stage(1, "Load universe and market data")

    stocks_file = resolve_stocks_file(args)
    universe = load_stocks_from_file(stocks_file) or parse_stocks(args.stocks)
    if args.data_source == "tq":
        if args.rolling_liquidity_pool:
            progress_write(f"Load rolling {args.rolling_liquidity_pool} base universe for research")
            try:
                universe = load_universe_from_tq(cfg.universe_scope)
            except Exception as exc:
                fallback_universe = _load_cached_rolling_pool_union(
                    pool_name=args.rolling_liquidity_pool,
                    start_date=cfg.start_date,
                    end_date=cfg.end_date,
                )
                if not fallback_universe:
                    raise
                universe = fallback_universe
                progress_write(
                    f"TQ universe unavailable, fall back to cached rolling union ({len(universe)} names): {exc}"
                )
        elif not universe and cfg.universe_scope == "all_a":
            progress_write("Load all-A universe from TQ")
            universe = load_universe_from_tq(cfg.universe_scope)
        elif not universe:
            raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
    raw_df_dict, raw_key = load_raw_market_data(cfg, args, universe)

    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    rolling_pool_artifact = None
    rolling_membership_frame = None
    rolling_pool_key = ""
    if args.rolling_liquidity_pool:
        rolling_pool_artifact, rolling_pool_key = load_cached_or_build_rolling_pool(
            args=args,
            cfg=cfg,
            df_dict=df_dict,
            raw_key=raw_key,
        )
        rolling_membership_frame = rolling_pool_artifact.membership_frame
        rolling_union = rolling_membership_frame.columns[rolling_membership_frame.any(axis=0)].tolist()
        if not rolling_union:
            raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} research pool is empty for the requested window.")
        df_dict = subset_df_dict_to_stocks(df_dict, rolling_union)
        rolling_membership_frame = rolling_membership_frame.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns).fillna(False)
    close = df_dict["Close"]
    train_end, valid_start, valid_end = resolve_split_window(
        close.index,
        cfg.train_end_date,
        cfg.valid_start_date,
        cfg.valid_days,
        research_time_unit=cfg.research_time_unit,
        valid_months=cfg.valid_months,
    )

    stage_progress.complete_stage(1)
    stage_progress.start_stage(2, "Build market state and liquidity")
    state_frame, state_name_map, state_key = load_cached_or_fit_market_state(
        args=args,
        cfg=cfg,
        benchmark_close=benchmark_close,
        raw_key=raw_key,
    )
    liquidity_bucket_frame, liquidity_bucket_key = load_cached_or_build_liquidity_buckets(
        args=args,
        cfg=cfg,
        amount_frame=df_dict["Amount"],
        raw_key=raw_key,
    )

    stage_progress.complete_stage(2)
    stage_progress.start_stage(3, "Build features and targets")
    industry_map = None
    style_map = None
    if (args.relation_layer or cfg.dynamic_graph_layer) and args.data_source == "tq":
        progress_write("Load relation priors: industry/style")
        try:
            industry_map = load_industry_map_from_tq(list(close.columns))
        except Exception as exc:
            progress_write(f"Industry mapping unavailable: {exc}")
        try:
            style_map = load_style_map_from_tq(list(close.columns))
        except Exception as exc:
            progress_write(f"Style mapping unavailable: {exc}")
    feature_meta = {
        "version": 7,
        "raw_key": raw_key,
        "relation_layer": bool(args.relation_layer),
        "liquidity_layer": bool(cfg.liquidity_layer),
        "liquidity_bucket_count": int(cfg.liquidity_bucket_count),
        "dynamic_graph_layer": bool(cfg.dynamic_graph_layer),
        "dynamic_graph_top_k": int(cfg.dynamic_graph_top_k),
        "dynamic_graph_temperature": float(cfg.dynamic_graph_temperature),
        "dynamic_graph_industry_boost": float(cfg.dynamic_graph_industry_boost),
        "dynamic_graph_style_boost": float(cfg.dynamic_graph_style_boost),
        "short_alpha_features": bool(cfg.short_alpha_features),
        "prediction_horizons": list(cfg.prediction_horizons),
        "breakout_event_horizon": int(cfg.breakout_event_horizon),
        "breakout_event_threshold": float(cfg.breakout_event_threshold),
        "breakout_event_pullback_limit": float(cfg.breakout_event_pullback_limit),
        "breakout_event_task": bool(float(cfg.breakout_event_loss_weight) > 0),
        "clean_breakout_event_task": bool(float(cfg.clean_breakout_event_loss_weight) > 0),
        "market_state_count": cfg.market_state_count,
        "state_key": state_key,
        "liquidity_bucket_key": liquidity_bucket_key,
        "industry_signature": series_signature(industry_map),
        "style_signature": frame_signature(style_map),
        "rolling_pool_key": rolling_pool_key,
    }
    feature_key = cache_key(feature_meta)
    feature_path = get_cache_root() / "features" / f"{feature_key}.pkl"
    feature_cached = load_pickle(feature_path) if args.use_cache and not args.refresh_cache else None
    if feature_cached is not None:
        progress_write(f"Load feature cache: {feature_path.name}")
        feature_frames = feature_cached["feature_frames"]
        target_frames = feature_cached["target_frames"]
        structure_label_frame = feature_cached.get("structure_label_frame")
    else:
        feature_frames = build_sequence_features(
            df_dict,
            benchmark_close,
            state_frame,
            industry_map=industry_map,
            style_map=style_map,
            liquidity_layer=cfg.liquidity_layer,
            liquidity_bucket_count=cfg.liquidity_bucket_count,
            liquidity_bucket_frame=liquidity_bucket_frame,
            dynamic_graph_layer=cfg.dynamic_graph_layer,
            dynamic_graph_top_k=cfg.dynamic_graph_top_k,
            dynamic_graph_temperature=cfg.dynamic_graph_temperature,
            dynamic_graph_industry_boost=cfg.dynamic_graph_industry_boost,
            dynamic_graph_style_boost=cfg.dynamic_graph_style_boost,
            short_alpha_features=cfg.short_alpha_features,
        )
        target_frames = build_targets(
            close,
            benchmark_close,
            cfg.prediction_horizons,
            open_df=df_dict["Open"],
            benchmark_open=benchmark_open,
            execution_mode="next_open",
            breakout_event_horizon=cfg.breakout_event_horizon,
            breakout_event_threshold=cfg.breakout_event_threshold,
            breakout_event_pullback_limit=cfg.breakout_event_pullback_limit,
            breakout_event_task=bool(float(cfg.breakout_event_loss_weight) > 0),
            clean_breakout_event_task=bool(float(cfg.clean_breakout_event_loss_weight) > 0),
        )
        structure_label_frame = build_structure_label_frame(
            close=df_dict["Close"].astype(float),
            open_df=df_dict["Open"].astype(float),
            high=df_dict["High"].astype(float),
            low=df_dict["Low"].astype(float),
            amount=df_dict["Amount"].astype(float),
            benchmark_close=benchmark_close.astype(float).reindex(df_dict["Close"].index),
        )
        if args.use_cache:
            save_pickle(
                feature_path,
                {
                    "feature_frames": feature_frames,
                    "target_frames": target_frames,
                    "structure_label_frame": structure_label_frame,
                },
            )
            progress_write(f"Write feature cache: {feature_path.name}")
    if structure_label_frame is None:
        structure_label_frame = build_structure_label_frame(
            close=df_dict["Close"].astype(float),
            open_df=df_dict["Open"].astype(float),
            high=df_dict["High"].astype(float),
            low=df_dict["Low"].astype(float),
            amount=df_dict["Amount"].astype(float),
            benchmark_close=benchmark_close.astype(float).reindex(df_dict["Close"].index),
        )
    train_target_frames = transform_return_target_frames(target_frames, cfg.return_target_transform)
    target_names = list(target_frames.keys())

    sample_dates = list(
        close.index[
            (close.index <= train_end)
            | ((close.index >= valid_start) & (close.index <= valid_end))
        ]
    )
    train_dates = list(close.index[close.index <= train_end])
    valid_dates = list(close.index[(close.index >= valid_start) & (close.index <= valid_end)])

    corpus_meta = {
        "version": 1,
        "feature_key": feature_key,
        "return_target_transform": cfg.return_target_transform,
        "lookback_window": int(cfg.lookback_window),
        "min_adv20": float(cfg.min_adv20),
        "min_price": float(cfg.min_price),
        "max_price": float(cfg.max_price),
        "train_end": str(pd.Timestamp(train_end).date()),
        "valid_start": str(pd.Timestamp(valid_start).date()),
        "valid_end": str(pd.Timestamp(valid_end).date()),
        "sample_date_count": int(len(sample_dates)),
        "rolling_pool_key": rolling_pool_key,
    }
    corpus_key = cache_key(corpus_meta)
    corpus_path = get_cache_root() / "corpus" / f"{corpus_key}.pkl"
    corpus = load_pickle(corpus_path) if args.use_cache and not args.refresh_cache else None
    stage_progress.complete_stage(3)
    stage_progress.start_stage(4, "Build corpus and samples")
    if corpus is not None:
        progress_write(f"Load corpus cache: {corpus_path.name}")
    else:
        progress_write("Build sequence corpus and split train/valid views")
        corpus = build_sequence_corpus(
            feature_frames=feature_frames,
            train_target_frames=train_target_frames,
            raw_target_frames=target_frames,
            lookback_window=cfg.lookback_window,
            sample_dates=sample_dates,
            min_adv20=cfg.min_adv20,
            amount_frame=df_dict["Amount"],
            close_frame=close,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            state_frame=state_frame,
            liquidity_bucket_frame=liquidity_bucket_frame,
            structure_label_frame=structure_label_frame,
            universe_membership_frame=rolling_membership_frame,
        )
        if args.use_cache:
            save_pickle(corpus_path, corpus)
            progress_write(f"Write corpus cache: {corpus_path.name}")
    train_ds = StockSequenceDataset(corpus=corpus, indices=corpus.build_index(end_date=train_end))
    valid_ds = StockSequenceDataset(corpus=corpus, indices=corpus.build_index(start_date=valid_start, end_date=valid_end))
    train_eval_start = resolve_recent_window_start(
        close.index,
        train_end,
        cfg.train_eval_window_days,
        research_time_unit=cfg.research_time_unit,
        window_months=cfg.train_eval_window_months,
    )
    if train_eval_start is None:
        train_eval_indices = corpus.build_index(end_date=train_end)
        train_eval_dates = list(close.index[close.index <= train_end])
    else:
        train_eval_indices = corpus.build_index(start_date=train_eval_start, end_date=train_end)
        train_eval_dates = list(close.index[(close.index >= train_eval_start) & (close.index <= train_end)])
    train_eval_ds = StockSequenceDataset(corpus=corpus, indices=train_eval_indices)
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise RuntimeError("Deep alpha dataset is empty. Try a longer history or smaller lookback window.")
    if len(train_eval_ds) == 0:
        raise RuntimeError(
            "Deep alpha train-eval dataset is empty. "
            "Increase --train-eval-window-days/--train-eval-window-months or history length."
        )

    stage_progress.complete_stage(4)
    stage_progress.start_stage(5, "Prepare runtime and model")
    progress_write(
        f"Dataset summary train={len(train_ds)} valid={len(valid_ds)} train_eval={len(train_eval_ds)}"
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training is required for deep_alpha research. "
            "No CUDA device is available, so the run must stop instead of silently falling back to CPU."
        )
    device = torch.device("cuda")
    runtime_profile = resolve_runtime_profile(
        stage="finetune",
        encoder_family=cfg.encoder_family,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        use_amp=cfg.use_amp,
        safe_profile=bool(cfg.safe_runtime_profile),
    )
    cfg.batch_size = runtime_profile.batch_size
    cfg.num_workers = runtime_profile.num_workers
    cfg.pin_memory = runtime_profile.pin_memory
    cfg.use_amp = runtime_profile.use_amp
    configure_torch_runtime(device, use_amp=bool(cfg.use_amp))
    for note in runtime_profile.applied_notes:
        progress_write(f"Runtime adjustment: {note}")
    pin_memory = bool(cfg.pin_memory and torch.cuda.is_available())
    loader_kwargs = {
        "num_workers": int(cfg.num_workers),
        "collate_fn": collate_batch,
        "pin_memory": pin_memory,
    }
    if int(cfg.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        if runtime_profile.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(runtime_profile.prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=DateGroupedBatchSampler(train_ds.meta, batch_size=cfg.batch_size, shuffle_dates=True, random_seed=cfg.random_seed),
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_sampler=DateGroupedBatchSampler(valid_ds.meta, batch_size=cfg.batch_size, shuffle_dates=False, random_seed=cfg.random_seed),
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_sampler=DateGroupedBatchSampler(train_eval_ds.meta, batch_size=cfg.batch_size, shuffle_dates=False, random_seed=cfg.random_seed),
        **loader_kwargs,
    )

    model = MultiTaskRanker(
        input_dim=len(train_ds.feature_names),
        hidden_dim=cfg.hidden_dim,
        output_dim=len(train_ds.target_names),
        return_output_dim=sum(1 for name in train_ds.target_names if name.startswith("fwd_excess_")),
        risk_output_dim=sum(1 for name in train_ds.target_names if not name.startswith("fwd_excess_")),
        dropout=cfg.dropout,
        encoder_family=cfg.encoder_family,
        patch_len=cfg.patch_len,
        return_head_mode=cfg.return_head_mode,
        context_dim=cfg.context_dim,
        state_context=cfg.state_context,
        liquidity_context=cfg.liquidity_context,
        structure_context=cfg.structure_context,
        aux_structure_task=cfg.aux_structure_task,
        structure_prototype_task=cfg.structure_prototype_task,
        state_vocab_size=cfg.market_state_count,
        liquidity_bucket_count=cfg.liquidity_bucket_count,
        structure_vocab_size=len(STRUCTURE_LABELS),
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
    )
    resume_invariant_contract = _build_resume_invariant_contract(
        args=args,
        cfg=cfg,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        feature_names=list(train_ds.feature_names),
        target_names=list(train_ds.target_names),
        stocks_file=stocks_file,
    )
    run_contract = _build_run_contract(
        args=args,
        run_dir=run_dir,
        resume_invariant_contract=resume_invariant_contract,
    )
    _atomic_write_text(
        run_dir / "run_contract.json",
        json.dumps(run_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    training_resume_payload: dict[str, Any] | None = None
    if cfg.pretrained_encoder_path:
        artifact_path = Path(cfg.pretrained_encoder_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Pretrained encoder artifact not found: {artifact_path}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        encoder_state = artifact.get("encoder_state_dict") or artifact.get("model_state_dict")
        if not encoder_state:
            raise ValueError(f"Pretrained encoder artifact missing encoder_state_dict: {artifact_path}")
        if cfg.encoder_family != "patch_transformer":
            raise ValueError("Pretrained encoder loading is currently only supported with --encoder-family patch_transformer.")
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        progress_write(f"Loaded pretrained encoder: {artifact_path.name}")
        if missing:
            progress_write(f"Missing pretrained encoder keys: {missing}")
        if unexpected:
            progress_write(f"Unexpected pretrained encoder keys: {unexpected}")
    if resume_context is not None:
        resume_artifact = dict(resume_context["artifact"])
        resume_feature_names = list(resume_artifact.get("feature_names", []) or [])
        resume_target_names = list(resume_artifact.get("target_names", []) or [])
        if resume_feature_names and list(train_ds.feature_names) != resume_feature_names:
            raise ValueError("Resume artifact feature_names do not match the current dataset; strict continuation would be invalid.")
        if resume_target_names and list(train_ds.target_names) != resume_target_names:
            raise ValueError("Resume artifact target_names do not match the current dataset; strict continuation would be invalid.")
        resume_mode = str(resume_context["resume_mode"])
        if resume_mode == "strict" and Path(resume_context["run_dir"]).resolve() != run_dir.resolve():
            raise ValueError(
                "Strict resume is only allowed back into the same run_dir / experiment-tag. "
                "Use the original run_dir or switch to --resume-mode warm_start."
            )
        observed_resume_contract = resume_artifact.get("resume_invariant_contract")
        if not isinstance(observed_resume_contract, dict) or not observed_resume_contract:
            observed_resume_contract = _build_legacy_resume_contract(
                artifact=resume_artifact,
                metrics=dict(resume_context.get("metrics", {})),
            )
            if observed_resume_contract:
                progress_write(
                    "Resume source is legacy and missing resume_invariant_contract; "
                    "fall back to partial contract validation."
                )
        if observed_resume_contract:
            expected_subset = {
                key: resume_invariant_contract[key]
                for key in observed_resume_contract.keys()
                if key in resume_invariant_contract
            }
            contract_mismatches = compare_expected_fields(observed_resume_contract, expected_subset)
            if contract_mismatches:
                mismatch_preview = "; ".join(contract_mismatches[:8])
                raise ValueError(
                    "Resume artifact protocol mismatch. Strict continuation would mix experiments. "
                    f"Mismatches: {mismatch_preview}"
                )
        selected_model_state = resume_artifact.get("selected_model_state_dict") or resume_artifact.get("model_state_dict")
        if selected_model_state is None:
            raise ValueError(f"Resume artifact missing model weights: {resume_context['artifact_path']}")
        if resume_mode == "warm_start":
            model.load_state_dict(selected_model_state)
            progress_write(f"Warm-start model loaded from {resume_context['artifact_path'].name}")
        else:
            training_resume_payload = {
                "history": list(resume_context["history_rows"]),
                "selected_model_state_dict": selected_model_state,
                "last_model_state_dict": resume_artifact.get("last_model_state_dict"),
                "optimizer_state_dict": resume_artifact.get("optimizer_state_dict"),
                "scheduler_state_dict": resume_artifact.get("scheduler_state_dict"),
                "scaler_state_dict": resume_artifact.get("scaler_state_dict"),
                "training_state": dict(resume_context["training_resume_state"]),
            }
            progress_write(
                f"Strict resume prepared from {resume_context['artifact_path'].name} "
                f"(epochs_completed={int(training_resume_payload['training_state'].get('epochs_completed', 0) or 0)})"
            )

    def _build_resume_model_artifact_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_state_dict": snapshot["selected_model_state_dict"],
            "selected_model_state_dict": snapshot["selected_model_state_dict"],
            "last_model_state_dict": snapshot["last_model_state_dict"],
            "optimizer_state_dict": snapshot["optimizer_state_dict"],
            "scheduler_state_dict": snapshot["scheduler_state_dict"],
            "scaler_state_dict": snapshot["scaler_state_dict"],
            "training_resume_state": snapshot["training_state"],
            "feature_names": train_ds.feature_names,
            "target_names": train_ds.target_names,
            "config": vars(cfg),
            "train_end": str(train_end.date()),
            "valid_start": str(valid_start.date()),
            "run_contract": run_contract,
            "resume_invariant_contract": resume_invariant_contract,
        }

    def _persist_training_resume_snapshot(snapshot: dict[str, Any]) -> None:
        history_rows = list(snapshot.get("history", []))
        training_state = dict(snapshot.get("training_state", {}))
        status_payload = {
            "run_dir": str(run_dir.resolve()),
            "last_saved_at": datetime.now().isoformat(timespec="seconds"),
            "resume_ready": True,
            "epochs_completed": int(training_state.get("epochs_completed", len(history_rows)) or len(history_rows)),
            "selected_epoch": int(training_state.get("selected_epoch", 0) or 0),
            "selected_metric_name": str(training_state.get("selected_metric_name", "") or ""),
            "selected_metric_value": float(training_state.get("selected_metric_value", float("nan"))),
            "checkpoint_selection_mode": str(training_state.get("checkpoint_selection_mode", "") or ""),
            "stopped_early": bool(training_state.get("stopped_early", False)),
            "resume_invariant_contract_fingerprint": str(run_contract.get("resume_invariant_contract_fingerprint", "")),
            "run_contract_fingerprint": str(run_contract.get("run_contract_fingerprint", "")),
            **runtime_metadata(),
        }
        _write_resume_artifacts(
            run_dir=run_dir,
            model_artifact=_build_resume_model_artifact_from_snapshot(snapshot),
            history_rows=history_rows,
            status_payload=status_payload,
        )
        progress_write(
            f"Saved resumable checkpoint at epoch {status_payload['epochs_completed']} "
            f"under {run_dir.name}"
        )

    target_state_ids = sorted(
        {
            int(state_id)
            for state_id, state_name in state_name_map.items()
            if state_name in set(cfg.target_state_names)
        }
    )

    stage_progress.complete_stage(5)
    stage_progress.start_stage(6, "Train model")
    progress_write("Start deep alpha training")
    checkpoint_selection_mode = resolve_checkpoint_metric_name(args.checkpoint_selection_objective)
    checkpoint_selection_callback = None
    if checkpoint_selection_mode != "valid_loss":
        checkpoint_eval_interval = max(int(args.checkpoint_eval_interval), 1)
        checkpoint_eval_start_epoch = max(int(args.checkpoint_eval_start_epoch), 1)
        def _checkpoint_selection_callback(model_for_eval: MultiTaskRanker, epoch: int) -> dict[str, Any]:
            if not _should_run_checkpoint_objective_eval(
                epoch=int(epoch),
                total_epochs=int(cfg.epochs),
                start_epoch=checkpoint_eval_start_epoch,
                interval=checkpoint_eval_interval,
            ):
                progress_write(
                    f"epoch {epoch}: skip expensive primary research objective "
                    f"(start={checkpoint_eval_start_epoch}, interval={checkpoint_eval_interval})"
                )
                return {
                    "metric_name": str(args.checkpoint_selection_objective),
                    "metric_value": float("nan"),
                    "skip_selection_update": True,
                }
            progress_write(f"epoch {epoch}: evaluate primary research objective")
            evaluation = _evaluate_research_outputs(
                model=model_for_eval,
                device=device,
                use_amp=bool(cfg.use_amp),
                args=args,
                cfg=cfg,
                close=close,
                benchmark_close=benchmark_close,
                open_df=df_dict["Open"],
                benchmark_open=benchmark_open,
                train_ds=train_ds,
                train_eval_loader=train_eval_loader,
                valid_loader=valid_loader,
                train_eval_dates=train_eval_dates,
                valid_dates=valid_dates,
                state_frame=state_frame,
                liquidity_bucket_frame=liquidity_bucket_frame,
                rolling_membership_frame=rolling_membership_frame,
            )
            progress_write(
                f"epoch {epoch}: {evaluation['checkpoint_metric_name']}="
                f"{float(evaluation['checkpoint_metric_value']):.6f} "
                f"via {evaluation['primary_backtest_label']}"
            )
            return {
                "metric_name": str(evaluation["checkpoint_metric_name"]),
                "metric_value": float(evaluation["checkpoint_metric_value"]),
                "skip_selection_update": False,
            }

        checkpoint_selection_callback = _checkpoint_selection_callback
    train_result = train_multitask_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        target_names=train_ds.target_names,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        ranking_loss_weight=cfg.ranking_loss_weight,
        listwise_loss_weight=cfg.listwise_loss_weight,
        listwise_temperature=cfg.listwise_temperature,
        max_rank_pairs_per_group=cfg.max_rank_pairs_per_group,
        target_loss_weights=cfg.target_loss_weights,
        return_loss_mode=cfg.return_loss_mode,
        return_top_frac=cfg.return_top_frac,
        return_bottom_frac=cfg.return_bottom_frac,
        liquidity_conditioning_mode=cfg.liquidity_conditioning_mode,
        liquidity_bucket_count=cfg.liquidity_bucket_count,
        top_liquidity_return_loss_weight=cfg.top_liquidity_return_loss_weight,
        other_liquidity_return_loss_weight=cfg.other_liquidity_return_loss_weight,
        top_liquidity_rank_loss_weight=cfg.top_liquidity_rank_loss_weight,
        other_liquidity_rank_loss_weight=cfg.other_liquidity_rank_loss_weight,
        top_liquidity_sample_weight=cfg.top_liquidity_sample_weight,
        other_liquidity_sample_weight=cfg.other_liquidity_sample_weight,
        structure_conditioning_mode=cfg.structure_conditioning_mode,
        top_attack_structure_names=list(cfg.top_attack_structure_names),
        other_protect_structure_names=list(cfg.other_protect_structure_names),
        top_attack_rank_weight=cfg.top_attack_rank_weight,
        other_protect_rank_weight=cfg.other_protect_rank_weight,
        target_state_ids=target_state_ids,
        target_state_attack_structure_names=list(cfg.target_state_attack_structure_names),
        target_state_protect_structure_names=list(cfg.target_state_protect_structure_names),
        target_state_rank_weight=cfg.target_state_rank_weight,
        target_state_protect_rank_weight=cfg.target_state_protect_rank_weight,
        aux_structure_task=cfg.aux_structure_task,
        aux_structure_loss_weight=cfg.aux_structure_loss_weight,
        aux_structure_label_smoothing=cfg.aux_structure_label_smoothing,
        structure_prototype_task=cfg.structure_prototype_task,
        structure_prototype_loss_weight=cfg.structure_prototype_loss_weight,
        structure_prototype_temperature=cfg.structure_prototype_temperature,
        device=device,
        use_amp=bool(cfg.use_amp),
        min_epochs=cfg.min_epochs,
        early_stop_patience=cfg.early_stop_patience,
        lr_plateau_patience=cfg.lr_plateau_patience,
        lr_plateau_factor=cfg.lr_plateau_factor,
        min_improvement=cfg.min_improvement,
        checkpoint_selection_mode=checkpoint_selection_mode,
        checkpoint_selection_callback=checkpoint_selection_callback,
        checkpoint_selection_min_improvement=args.checkpoint_selection_min_improvement,
        resume_payload=training_resume_payload,
        epoch_end_callback=_persist_training_resume_snapshot,
    )
    history = train_result.history
    training_diagnostics = train_result.diagnostics
    progress_write(
        f"Training diagnostics status={training_diagnostics.status}, "
        f"selected_epoch={training_diagnostics.selected_epoch}/{training_diagnostics.epochs_completed}, "
        f"{training_diagnostics.selected_metric_name}={training_diagnostics.selected_metric_value:.6f}, "
        f"best_valid_loss={training_diagnostics.best_valid_loss:.6f}, "
        f"budget_pressure={training_diagnostics.objective_aligned_budget_pressure}"
    )

    stage_progress.complete_stage(6)
    stage_progress.start_stage(7, "Validate inference and backtest")
    progress_write("Run validation inference and holdout backtest")
    evaluation = _evaluate_research_outputs(
        model=model,
        device=device,
        use_amp=bool(cfg.use_amp),
        args=args,
        cfg=cfg,
        close=close,
        benchmark_close=benchmark_close,
        open_df=df_dict["Open"],
        benchmark_open=benchmark_open,
        train_ds=train_ds,
        train_eval_loader=train_eval_loader,
        valid_loader=valid_loader,
        train_eval_dates=train_eval_dates,
        valid_dates=valid_dates,
        state_frame=state_frame,
        liquidity_bucket_frame=liquidity_bucket_frame,
        rolling_membership_frame=rolling_membership_frame,
    )
    train_pred_df = evaluation["train_pred_df"]
    pred_df = evaluation["pred_df"]
    emb_df = evaluation["emb_df"]
    rankic_timeseries = evaluation["rankic_timeseries"]
    rankic_summary = evaluation["rankic_summary"]
    monthly_rankic_summary = evaluation["monthly_rankic_summary"]
    structure_aux_summary = evaluation["structure_aux_summary"]
    structure_prototype_summary = evaluation["structure_prototype_summary"]
    score_head_artifact = evaluation["score_head_artifact"]
    applied_score_horizon_weights = evaluation["applied_score_horizon_weights"]
    applied_score_downside_penalty = evaluation["applied_score_downside_penalty"]
    train_score_frame = evaluation["train_score_frame"]
    score_frame = evaluation["score_frame"]
    risk_gate_artifact = evaluation["risk_gate_artifact"]
    train_eval_target_weights = evaluation["train_eval_target_weights"]
    target_weights = evaluation["target_weights"]
    equity_df = evaluation["equity_df"]
    action_df = evaluation["action_df"]
    monthly_backtest_summary = evaluation["monthly_backtest_summary"]
    monthly_backtest_diagnostics = dict(evaluation["monthly_backtest_diagnostics"])
    metrics = evaluation["holdout_metrics"]
    execution_alignment_artifact = evaluation["execution_alignment_artifact"]
    execution_aligned_equity_df = evaluation["execution_aligned_equity_df"]
    execution_aligned_action_df = evaluation["execution_aligned_action_df"]
    execution_aligned_metrics = evaluation["execution_aligned_metrics"]
    execution_aligned_score_frame = evaluation["execution_aligned_score_frame"]
    execution_aligned_target_weights = evaluation["execution_aligned_target_weights"]
    execution_aligned_monthly_backtest_summary = evaluation["execution_aligned_monthly_backtest_summary"]
    execution_aligned_monthly_backtest_diagnostics = (
        None
        if evaluation["execution_aligned_monthly_backtest_diagnostics"] is None
        else dict(evaluation["execution_aligned_monthly_backtest_diagnostics"])
    )
    primary_backtest_label = str(evaluation["primary_backtest_label"])
    primary_backtest = dict(evaluation["primary_backtest"])
    primary_monthly_summary_label = str(evaluation["primary_monthly_summary_label"])
    primary_monthly_summary = evaluation["primary_monthly_summary"]
    primary_monthly_diagnostics_label = str(evaluation["primary_monthly_diagnostics_label"])
    primary_monthly_diagnostics = dict(evaluation["primary_monthly_diagnostics"])
    primary_monthly_objectives = dict(evaluation["primary_monthly_objectives"])

    live_outputs = _build_live_inference_outputs(
        cfg=cfg,
        close=close,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        open_df=df_dict["Open"],
        amount_frame=df_dict["Amount"],
        feature_frames=feature_frames,
        train_target_frames=train_target_frames,
        target_frames=target_frames,
        state_frame=state_frame,
        liquidity_bucket_frame=liquidity_bucket_frame,
        structure_label_frame=structure_label_frame,
        rolling_membership_frame=rolling_membership_frame,
        live_start=valid_start,
        model=model,
        loader_kwargs=loader_kwargs,
        device=device,
        target_names=train_ds.target_names,
        score_head_method=args.score_head_method,
        score_head_artifact=score_head_artifact,
        applied_score_horizon_weights=applied_score_horizon_weights,
        applied_score_downside_penalty=applied_score_downside_penalty,
        risk_gate_artifact=risk_gate_artifact,
        use_amp=bool(cfg.use_amp),
    )
    live_score_frame = live_outputs["score_frame"]
    live_target_weights = live_outputs["target_weights"]
    portfolio_capped_live_target_weights = live_outputs["portfolio_capped_target_weights"]
    live_latest_scores = None
    live_daily_score_panel = _panel_to_long(live_score_frame, "score")
    live_daily_target_weight_panel = _panel_to_long(live_target_weights, "target_weight")
    portfolio_capped_live_daily_target_weight_panel = _panel_to_long(
        portfolio_capped_live_target_weights,
        "target_weight",
    )
    if not live_score_frame.dropna(how="all").empty:
        live_latest_scores = live_score_frame.loc[[live_score_frame.dropna(how="all").index.max()]].T.reset_index()
        live_latest_scores.columns = ["stock", "latest_score"]
        live_latest_scores = live_latest_scores.sort_values("latest_score", ascending=False, na_position="last")

    execution_aligned_live_outputs = _build_live_execution_aligned_outputs(
        execution_alignment_artifact=execution_alignment_artifact,
        raw_live_score_frame=live_score_frame,
        raw_live_target_weights=live_target_weights,
        close=close,
        benchmark_close=benchmark_close,
        open_df=df_dict["Open"],
        benchmark_open=benchmark_open,
        cfg=cfg,
        transaction_cost_bps=args.execution_alignment_transaction_cost_bps,
        slippage_bps=args.execution_alignment_slippage_bps,
        sell_tax_bps=args.execution_alignment_sell_tax_bps,
    )
    execution_aligned_live_score_panel = None
    execution_aligned_live_target_weight_panel = None
    execution_aligned_live_latest_scores = None
    if execution_aligned_live_outputs is not None:
        execution_aligned_live_score_frame = execution_aligned_live_outputs["score_frame"]
        execution_aligned_live_target_weights = execution_aligned_live_outputs["target_weights"]
        execution_aligned_live_score_panel = _panel_to_long(execution_aligned_live_score_frame, "score")
        execution_aligned_live_target_weight_panel = _panel_to_long(execution_aligned_live_target_weights, "target_weight")
        if not execution_aligned_live_score_frame.dropna(how="all").empty:
            execution_aligned_live_latest_scores = execution_aligned_live_score_frame.loc[
                [execution_aligned_live_score_frame.dropna(how="all").index.max()]
            ].T.reset_index()
            execution_aligned_live_latest_scores.columns = ["stock", "latest_score"]
            execution_aligned_live_latest_scores = execution_aligned_live_latest_scores.sort_values(
                "latest_score",
                ascending=False,
                na_position="last",
            )

    stage_progress.complete_stage(7)
    stage_progress.start_stage(8, "Write artifacts")
    progress_write("Prepare output directory and write files")

    history_df = pd.DataFrame([record.__dict__ for record in history])
    state_df = state_frame.copy()
    latest_scores = score_frame.loc[[score_frame.dropna(how="all").index.max()]].T.reset_index()
    latest_scores.columns = ["stock", "latest_score"]
    latest_scores = latest_scores.sort_values("latest_score", ascending=False, na_position="last")
    daily_score_panel = _panel_to_long(score_frame.reindex(valid_dates), "score")
    daily_target_weight_panel = _panel_to_long(target_weights.reindex(valid_dates), "target_weight")
    monthly_score_panel = _panel_to_long(_select_month_end_panel(score_frame.reindex(valid_dates)), "score")
    monthly_target_weight_panel = _panel_to_long(_select_month_end_panel(target_weights.reindex(valid_dates)), "target_weight")
    execution_aligned_latest_scores = None
    execution_aligned_daily_score_panel = None
    execution_aligned_daily_target_weight_panel = None
    execution_aligned_monthly_score_panel = None
    execution_aligned_monthly_target_weight_panel = None
    if execution_aligned_score_frame is not None and execution_aligned_target_weights is not None:
        execution_aligned_latest_scores = execution_aligned_score_frame.loc[[execution_aligned_score_frame.dropna(how="all").index.max()]].T.reset_index()
        execution_aligned_latest_scores.columns = ["stock", "latest_score"]
        execution_aligned_latest_scores = execution_aligned_latest_scores.sort_values("latest_score", ascending=False, na_position="last")
        execution_aligned_daily_score_panel = _panel_to_long(execution_aligned_score_frame.reindex(valid_dates), "score")
        execution_aligned_daily_target_weight_panel = _panel_to_long(execution_aligned_target_weights.reindex(valid_dates), "target_weight")
        execution_aligned_monthly_score_panel = _panel_to_long(
            _select_month_end_panel(execution_aligned_score_frame.reindex(valid_dates)),
            "score",
        )
        execution_aligned_monthly_target_weight_panel = _panel_to_long(
            _select_month_end_panel(execution_aligned_target_weights.reindex(valid_dates)),
            "target_weight",
        )

    equity_export = equity_df.reset_index().rename(columns={equity_df.index.name or "index": "date"})
    if execution_aligned_equity_df is not None and execution_aligned_action_df is not None:
        execution_aligned_equity_export = execution_aligned_equity_df.reset_index().rename(columns={execution_aligned_equity_df.index.name or "index": "date"})
    metrics_payload = {
        "framework": "deep_alpha_research",
        "contract_schema_version": 1,
        "research_time_unit": str(cfg.research_time_unit),
        "train_end": str(train_end.date()),
        "valid_start": str(valid_start.date()),
        "valid_end": str(valid_end.date()),
        "train_samples": len(train_ds),
        "train_eval_samples": len(train_eval_ds),
        "valid_samples": len(valid_ds),
        "valid_days": int(cfg.valid_days),
        "valid_months": int(cfg.valid_months),
        "train_eval_window_days": cfg.train_eval_window_days,
        "train_eval_window_months": int(cfg.train_eval_window_months),
        "train_eval_start": "" if train_eval_start is None else str(train_eval_start.date()),
        "device": str(device),
        "market_state_count": cfg.market_state_count,
        "feature_count": len(train_ds.feature_names),
        "target_names": train_ds.target_names,
        "hidden_dim": int(cfg.hidden_dim),
        "encoder_family": cfg.encoder_family,
        "patch_len": cfg.patch_len,
        "transformer_heads": int(cfg.transformer_heads),
        "transformer_layers": int(cfg.transformer_layers),
        "pretrained_encoder_path": cfg.pretrained_encoder_path,
        "resume_mode": "" if resume_context is None else str(resume_context["resume_mode"]),
        "resume_source_run_dir": "" if resume_context is None else str(Path(resume_context["run_dir"]).resolve()),
        "resume_source_model_path": "" if resume_context is None else str(Path(resume_context["artifact_path"]).resolve()),
        "resume_source_epochs_completed": (
            0
            if resume_context is None
            else int(resume_context["training_resume_state"].get("epochs_completed", 0) or 0)
        ),
        "resume_invariant_contract_fingerprint": str(run_contract.get("resume_invariant_contract_fingerprint", "")),
        "run_contract_fingerprint": str(run_contract.get("run_contract_fingerprint", "")),
        "epochs": cfg.epochs,
        "min_epochs": cfg.min_epochs,
        "early_stop_patience": cfg.early_stop_patience,
        "lr_plateau_patience": cfg.lr_plateau_patience,
        "lr_plateau_factor": cfg.lr_plateau_factor,
        "min_improvement": cfg.min_improvement,
        "return_head_mode": cfg.return_head_mode,
        "context_dim": cfg.context_dim,
        "state_context": cfg.state_context,
        "liquidity_context": cfg.liquidity_context,
        "structure_context": cfg.structure_context,
        "aux_structure_task": cfg.aux_structure_task,
        "aux_structure_loss_weight": cfg.aux_structure_loss_weight,
        "aux_structure_label_smoothing": cfg.aux_structure_label_smoothing,
        "structure_prototype_task": cfg.structure_prototype_task,
        "structure_prototype_loss_weight": cfg.structure_prototype_loss_weight,
        "structure_prototype_temperature": cfg.structure_prototype_temperature,
        "ranking_loss_weight": cfg.ranking_loss_weight,
        "listwise_loss_weight": cfg.listwise_loss_weight,
        "listwise_temperature": cfg.listwise_temperature,
        "max_rank_pairs_per_group": cfg.max_rank_pairs_per_group,
        "return_loss_mode": cfg.return_loss_mode,
        "return_target_transform": cfg.return_target_transform,
        "return_top_frac": cfg.return_top_frac,
        "return_bottom_frac": cfg.return_bottom_frac,
        "liquidity_conditioning_mode": cfg.liquidity_conditioning_mode,
        "top_liquidity_return_loss_weight": cfg.top_liquidity_return_loss_weight,
        "other_liquidity_return_loss_weight": cfg.other_liquidity_return_loss_weight,
        "top_liquidity_rank_loss_weight": cfg.top_liquidity_rank_loss_weight,
        "other_liquidity_rank_loss_weight": cfg.other_liquidity_rank_loss_weight,
        "top_liquidity_sample_weight": cfg.top_liquidity_sample_weight,
        "other_liquidity_sample_weight": cfg.other_liquidity_sample_weight,
        "structure_conditioning_mode": cfg.structure_conditioning_mode,
        "top_attack_structure_names": list(cfg.top_attack_structure_names),
        "other_protect_structure_names": list(cfg.other_protect_structure_names),
        "top_attack_rank_weight": cfg.top_attack_rank_weight,
        "other_protect_rank_weight": cfg.other_protect_rank_weight,
        "target_state_names": list(cfg.target_state_names),
        "target_state_ids": list(target_state_ids),
        "target_state_attack_structure_names": list(cfg.target_state_attack_structure_names),
        "target_state_protect_structure_names": list(cfg.target_state_protect_structure_names),
        "target_state_rank_weight": cfg.target_state_rank_weight,
        "target_state_protect_rank_weight": cfg.target_state_protect_rank_weight,
        "target_loss_weights": cfg.target_loss_weights,
        "score_horizon_weights": cfg.score_horizon_weights,
        "applied_score_horizon_weights": applied_score_horizon_weights,
        "score_rank_blend": cfg.score_rank_blend,
        "score_downside_penalty": cfg.score_downside_penalty,
        "applied_score_downside_penalty": applied_score_downside_penalty,
        "score_risk_mode": cfg.score_risk_mode,
        "score_risk_gate_threshold": cfg.score_risk_gate_threshold,
        "score_risk_state_thresholds": parse_float_list(args.score_risk_state_thresholds),
        "liquidity_layer": bool(cfg.liquidity_layer),
        "liquidity_bucket_count": int(cfg.liquidity_bucket_count),
        "dynamic_graph_layer": bool(cfg.dynamic_graph_layer),
        "dynamic_graph_top_k": int(cfg.dynamic_graph_top_k),
        "dynamic_graph_temperature": float(cfg.dynamic_graph_temperature),
        "dynamic_graph_industry_boost": float(cfg.dynamic_graph_industry_boost),
        "dynamic_graph_style_boost": float(cfg.dynamic_graph_style_boost),
        "short_alpha_features": bool(cfg.short_alpha_features),
        "breakout_event_horizon": int(cfg.breakout_event_horizon),
        "breakout_event_threshold": float(cfg.breakout_event_threshold),
        "breakout_event_pullback_limit": float(cfg.breakout_event_pullback_limit),
        "breakout_event_loss_weight": float(cfg.breakout_event_loss_weight),
        "clean_breakout_event_loss_weight": float(cfg.clean_breakout_event_loss_weight),
        "safe_runtime_profile": bool(cfg.safe_runtime_profile),
        "runtime_profile": runtime_profile.__dict__,
        **runtime_metadata(),
        "training_diagnostics": training_diagnostics.__dict__,
        "research_objective_mode": str(args.research_objective_mode),
        "checkpoint_selection_objective": str(args.checkpoint_selection_objective),
        "checkpoint_selection_min_improvement": float(args.checkpoint_selection_min_improvement),
        "score_head_method": args.score_head_method,
        "adaptive_task_weights": bool(args.adaptive_task_weights),
        "adaptive_task_window_days": args.adaptive_task_window_days,
        "adaptive_task_window_months": int(cfg.adaptive_task_window_months),
        "score_head_task_weights": score_head_artifact.task_weights,
        "score_head_extra": score_head_artifact.extra,
        "structure_aux_summary": structure_aux_summary,
        "structure_prototype_summary": structure_prototype_summary,
        "risk_gate_global_threshold": None if risk_gate_artifact is None else risk_gate_artifact.global_threshold,
        "risk_gate_state_thresholds": {} if risk_gate_artifact is None else risk_gate_artifact.state_thresholds,
        "risk_gate_group_thresholds": {} if risk_gate_artifact is None else risk_gate_artifact.group_thresholds,
        "execution_alignment_mode": str(args.execution_alignment_mode),
        "execution_alignment_objective": str(args.execution_alignment_objective),
        "execution_alignment_shortlist_size": int(args.execution_alignment_shortlist_size),
        "execution_alignment_screen_window_days": int(args.execution_alignment_screen_window_days),
        "execution_alignment_transaction_cost_bps": float(args.execution_alignment_transaction_cost_bps),
        "execution_alignment_slippage_bps": float(args.execution_alignment_slippage_bps),
        "execution_alignment_sell_tax_bps": float(args.execution_alignment_sell_tax_bps),
        "execution_alignment_profile": "" if execution_alignment_artifact is None else execution_alignment_artifact.selected_profile,
        "execution_alignment_profile_description": "" if execution_alignment_artifact is None else execution_alignment_artifact.selected_profile_description,
        "execution_alignment_selected_profile_spec": {} if execution_alignment_artifact is None else execution_alignment_artifact.selected_profile_spec,
        "execution_alignment_candidate_profiles": [] if execution_alignment_artifact is None else execution_alignment_artifact.candidate_profiles,
        "execution_alignment_selected_bridge_meta": {} if execution_alignment_artifact is None else execution_alignment_artifact.selected_bridge_meta,
        "execution_alignment_selected_train_metrics": {} if execution_alignment_artifact is None else execution_alignment_artifact.selected_train_metrics,
        "relation_layer": bool(args.relation_layer),
        "rankic_summary": rankic_summary.to_dict(orient="records"),
        "holdout_backtest": metrics,
        "monthly_backtest_diagnostics": monthly_backtest_diagnostics,
        "execution_aligned_holdout_backtest": {} if execution_aligned_metrics is None else execution_aligned_metrics,
        "execution_aligned_monthly_backtest_diagnostics": (
            {} if execution_aligned_monthly_backtest_diagnostics is None else execution_aligned_monthly_backtest_diagnostics
        ),
        "primary_research_backtest_label": str(primary_backtest_label),
        "primary_research_backtest": primary_backtest,
        "primary_research_monthly_summary_label": str(primary_monthly_summary_label),
        "primary_research_monthly_diagnostics_label": str(primary_monthly_diagnostics_label),
        "primary_research_monthly_diagnostics": primary_monthly_diagnostics,
        "primary_research_monthly_objectives": primary_monthly_objectives,
        "primary_execution_panel_mode": resolve_primary_panel_mode(
            {
                "research_objective_mode": str(args.research_objective_mode),
                "holdout_backtest": metrics,
                "execution_aligned_holdout_backtest": {} if execution_aligned_metrics is None else execution_aligned_metrics,
                "execution_alignment_profile": "" if execution_alignment_artifact is None else execution_alignment_artifact.selected_profile,
                "primary_research_backtest_label": str(primary_backtest_label),
            },
            research_objective_mode=args.research_objective_mode,
        ),
        "live_signal_date": (
            ""
            if live_score_frame.dropna(how="all").empty
            else str(pd.Timestamp(live_score_frame.dropna(how="all").index.max()).date())
        ),
        "execution_aligned_live_signal_date": (
            ""
            if execution_aligned_live_outputs is None
            else str(pd.Timestamp(execution_aligned_live_outputs["score_frame"].dropna(how="all").index.max()).date())
            if not execution_aligned_live_outputs["score_frame"].dropna(how="all").empty
            else ""
        ),
        "execution_mode": "next_open",
        "rebalance_freq": cfg.rebalance_freq,
        "liquidity_pool": args.liquidity_pool or "",
        "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
        "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
        "rolling_pool_adv_window": int(args.pool_adv_window),
        "rolling_pool_union_size": 0 if rolling_pool_artifact is None else int(rolling_membership_frame.columns[rolling_membership_frame.any(axis=0)].size),
        "rolling_pool_rebalance_count": 0 if rolling_pool_artifact is None else int(len(rolling_pool_artifact.schedule_df)),
        "stocks_file": stocks_file or "",
        "raw_cache_key": raw_key,
        "rolling_pool_cache_key": rolling_pool_key,
        "state_cache_key": state_key,
        "liquidity_bucket_cache_key": liquidity_bucket_key,
        "feature_cache_key": feature_key,
        "corpus_cache_key": corpus_key,
    }
    model_artifact = {
        "model_state_dict": train_result.selected_model_state_dict,
        "selected_model_state_dict": train_result.selected_model_state_dict,
        "last_model_state_dict": train_result.last_model_state_dict,
        "optimizer_state_dict": train_result.optimizer_state_dict,
        "scheduler_state_dict": train_result.scheduler_state_dict,
        "scaler_state_dict": train_result.scaler_state_dict,
        "training_resume_state": train_result.training_state,
        "feature_names": train_ds.feature_names,
        "target_names": train_ds.target_names,
        "config": vars(cfg),
        "train_end": str(train_end.date()),
        "valid_start": str(valid_start.date()),
        "run_contract": run_contract,
        "resume_invariant_contract": resume_invariant_contract,
    }

    def _write_metrics_json() -> None:
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    write_tasks: list[tuple[str, Any]] = [
        ("train_history.csv", lambda: history_df.to_csv(run_dir / "train_history.csv", index=False, encoding="utf-8-sig")),
        ("validation_predictions.csv", lambda: pred_df.to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")),
        ("validation_embeddings.csv", lambda: emb_df.to_csv(run_dir / "validation_embeddings.csv", index=False, encoding="utf-8-sig")),
        ("validation_rankic_timeseries.csv", lambda: rankic_timeseries.to_csv(run_dir / "validation_rankic_timeseries.csv", index=False, encoding="utf-8-sig")),
        ("validation_rankic_summary.csv", lambda: rankic_summary.to_csv(run_dir / "validation_rankic_summary.csv", index=False, encoding="utf-8-sig")),
        ("validation_rankic_monthly_summary.csv", lambda: monthly_rankic_summary.to_csv(run_dir / "validation_rankic_monthly_summary.csv", index=False, encoding="utf-8-sig")),
        ("market_state_frame.csv", lambda: state_df.to_csv(run_dir / "market_state_frame.csv", encoding="utf-8-sig")),
        ("latest_scores.csv", lambda: latest_scores.to_csv(run_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")),
        ("daily_score_panel.csv", lambda: daily_score_panel.to_csv(run_dir / "daily_score_panel.csv", index=False, encoding="utf-8-sig")),
        ("daily_target_weight_panel.csv", lambda: daily_target_weight_panel.to_csv(run_dir / "daily_target_weight_panel.csv", index=False, encoding="utf-8-sig")),
        ("monthly_score_panel.csv", lambda: monthly_score_panel.to_csv(run_dir / "monthly_score_panel.csv", index=False, encoding="utf-8-sig")),
        ("monthly_target_weight_panel.csv", lambda: monthly_target_weight_panel.to_csv(run_dir / "monthly_target_weight_panel.csv", index=False, encoding="utf-8-sig")),
        ("daily_live_score_panel.csv", lambda: live_daily_score_panel.to_csv(run_dir / "daily_live_score_panel.csv", index=False, encoding="utf-8-sig")),
        ("daily_live_target_weight_panel.csv", lambda: live_daily_target_weight_panel.to_csv(run_dir / "daily_live_target_weight_panel.csv", index=False, encoding="utf-8-sig")),
        (
            "portfolio_capped_daily_live_target_weight_panel.csv",
            lambda: portfolio_capped_live_daily_target_weight_panel.to_csv(
                run_dir / "portfolio_capped_daily_live_target_weight_panel.csv",
                index=False,
                encoding="utf-8-sig",
            ),
        ),
        ("equity_curve.csv", lambda: equity_export.to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")),
        ("monthly_backtest_summary.csv", lambda: monthly_backtest_summary.to_csv(run_dir / "monthly_backtest_summary.csv", index=False, encoding="utf-8-sig")),
        ("monthly_backtest_diagnostics.json", lambda: _write_json_payload(run_dir / "monthly_backtest_diagnostics.json", monthly_backtest_diagnostics)),
        ("primary_research_monthly_summary.csv", lambda: primary_monthly_summary.to_csv(run_dir / "primary_research_monthly_summary.csv", index=False, encoding="utf-8-sig")),
        ("primary_research_monthly_diagnostics.json", lambda: _write_json_payload(run_dir / "primary_research_monthly_diagnostics.json", primary_monthly_diagnostics)),
        ("primary_research_monthly_objectives.json", lambda: _write_json_payload(run_dir / "primary_research_monthly_objectives.json", primary_monthly_objectives)),
        ("actions.csv", lambda: action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")),
        ("deep_alpha_model.pt", lambda: torch.save(model_artifact, run_dir / "deep_alpha_model.pt")),
        ("run_contract.json", lambda: _write_json_payload(run_dir / "run_contract.json", run_contract)),
        ("score_head_artifact.pkl", lambda: save_pickle(run_dir / "score_head_artifact.pkl", score_head_artifact)),
        ("metrics.json", _write_metrics_json),
    ]
    if live_latest_scores is not None:
        write_tasks.append(("live_latest_scores.csv", lambda: live_latest_scores.to_csv(run_dir / "live_latest_scores.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_latest_scores is not None:
        write_tasks.append(("execution_aligned_latest_scores.csv", lambda: execution_aligned_latest_scores.to_csv(run_dir / "execution_aligned_latest_scores.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_daily_score_panel is not None:
        write_tasks.append(("execution_aligned_daily_score_panel.csv", lambda: execution_aligned_daily_score_panel.to_csv(run_dir / "execution_aligned_daily_score_panel.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_daily_target_weight_panel is not None:
        write_tasks.append(("execution_aligned_daily_target_weight_panel.csv", lambda: execution_aligned_daily_target_weight_panel.to_csv(run_dir / "execution_aligned_daily_target_weight_panel.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_monthly_score_panel is not None:
        write_tasks.append(("execution_aligned_monthly_score_panel.csv", lambda: execution_aligned_monthly_score_panel.to_csv(run_dir / "execution_aligned_monthly_score_panel.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_monthly_target_weight_panel is not None:
        write_tasks.append(("execution_aligned_monthly_target_weight_panel.csv", lambda: execution_aligned_monthly_target_weight_panel.to_csv(run_dir / "execution_aligned_monthly_target_weight_panel.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_live_latest_scores is not None:
        write_tasks.append(("execution_aligned_live_latest_scores.csv", lambda: execution_aligned_live_latest_scores.to_csv(run_dir / "execution_aligned_live_latest_scores.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_live_score_panel is not None:
        write_tasks.append(("execution_aligned_daily_live_score_panel.csv", lambda: execution_aligned_live_score_panel.to_csv(run_dir / "execution_aligned_daily_live_score_panel.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_live_target_weight_panel is not None:
        write_tasks.append(("execution_aligned_daily_live_target_weight_panel.csv", lambda: execution_aligned_live_target_weight_panel.to_csv(run_dir / "execution_aligned_daily_live_target_weight_panel.csv", index=False, encoding="utf-8-sig")))
    if execution_aligned_equity_df is not None and execution_aligned_action_df is not None:
        write_tasks.append(("execution_aligned_equity_curve.csv", lambda: execution_aligned_equity_export.to_csv(run_dir / "execution_aligned_equity_curve.csv", index=False, encoding="utf-8-sig")))
        write_tasks.append(("execution_aligned_monthly_backtest_summary.csv", lambda: execution_aligned_monthly_backtest_summary.to_csv(run_dir / "execution_aligned_monthly_backtest_summary.csv", index=False, encoding="utf-8-sig")))
        write_tasks.append(
            (
                "execution_aligned_monthly_backtest_diagnostics.json",
                lambda: _write_json_payload(
                    run_dir / "execution_aligned_monthly_backtest_diagnostics.json",
                    {} if execution_aligned_monthly_backtest_diagnostics is None else execution_aligned_monthly_backtest_diagnostics,
                ),
            )
        )
        write_tasks.append(("execution_aligned_actions.csv", lambda: execution_aligned_action_df.to_csv(run_dir / "execution_aligned_actions.csv", index=False, encoding="utf-8-sig")))
    if rolling_pool_artifact is not None:
        write_tasks.append(("rolling_liquidity_schedule.csv", lambda: rolling_pool_artifact.schedule_df.to_csv(run_dir / "rolling_liquidity_schedule.csv", index=False, encoding="utf-8-sig")))
        write_tasks.append(("rolling_liquidity_summary.csv", lambda: rolling_pool_artifact.summary_df.to_csv(run_dir / "rolling_liquidity_summary.csv", index=False, encoding="utf-8-sig")))
    if risk_gate_artifact is not None and risk_gate_artifact.objective_rows:
        risk_gate_rows = pd.DataFrame(risk_gate_artifact.objective_rows)
        write_tasks.append(("risk_gate_objective_rows.csv", lambda: risk_gate_rows.to_csv(run_dir / "risk_gate_objective_rows.csv", index=False, encoding="utf-8-sig")))
    if execution_alignment_artifact is not None and execution_alignment_artifact.objective_rows:
        execution_alignment_rows = pd.DataFrame(execution_alignment_artifact.objective_rows)
        write_tasks.append(("execution_alignment_objective_rows.csv", lambda: execution_alignment_rows.to_csv(run_dir / "execution_alignment_objective_rows.csv", index=False, encoding="utf-8-sig")))
    if risk_gate_artifact is not None:
        write_tasks.append(("risk_gate_artifact.pkl", lambda: save_pickle(run_dir / "risk_gate_artifact.pkl", risk_gate_artifact)))

    _run_write_tasks(write_tasks)
    stage_progress.complete_stage(8)
    stage_progress.complete()
    stage_progress.close()
    print(f"Output: {run_dir}")
    summary = {
        "holdout_backtest": metrics,
        "primary_research_backtest_label": str(primary_backtest_label),
        "primary_research_backtest": primary_backtest,
        "primary_research_monthly_summary_label": str(primary_monthly_summary_label),
        "primary_research_monthly_diagnostics": primary_monthly_diagnostics,
        "primary_research_monthly_objectives": primary_monthly_objectives,
        "live_signal_date": (
            ""
            if live_score_frame.dropna(how="all").empty
            else str(pd.Timestamp(live_score_frame.dropna(how="all").index.max()).date())
        ),
    }
    if execution_aligned_metrics is not None and execution_alignment_artifact is not None:
        summary["execution_aligned_holdout_backtest"] = execution_aligned_metrics
        summary["execution_alignment_profile"] = execution_alignment_artifact.selected_profile
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
