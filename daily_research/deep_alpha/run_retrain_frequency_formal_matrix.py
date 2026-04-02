from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.backtest import _annualized_return, _annualized_vol, _max_drawdown
from daily_research.execution.update_default_candidate_production import (
    _append_arg,
    _append_flag,
    _format_prediction_horizons,
    _format_score_horizon_weights,
    _format_state_thresholds,
    _format_task_loss_weights,
    _load_source_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
DEFAULT_SOURCE_RUN = PROJECT_ROOT / "daily_research" / "output" / "deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1"


@dataclass(frozen=True)
class FrequencySpec:
    name: str
    label: str
    mode: str
    trading_days: int = 0


@dataclass(frozen=True)
class BlockPlan:
    spec_name: str
    spec_label: str
    block_index: int
    execution_start: str
    execution_end: str
    signal_start: str
    train_end: str
    valid_start: str
    valid_days: int
    execution_days: int
    uses_source_slice: bool
    experiment_tag: str

    @property
    def block_slug(self) -> str:
        return f"block_{self.block_index:02d}_{self.execution_start.replace('-', '')}_{self.execution_end.replace('-', '')}"


DEFAULT_SPECS: tuple[FrequencySpec, ...] = (
    FrequencySpec(name="annual_freeze", label="Freeze 1Y", mode="source_full"),
    FrequencySpec(name="quarterly_63d", label="Retrain 63D", mode="fixed_days", trading_days=63),
    FrequencySpec(name="monthly_calendar", label="Retrain Monthly", mode="calendar_month"),
    FrequencySpec(name="every_21d", label="Retrain 21D", mode="fixed_days", trading_days=21),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a formal deep_alpha retrain-frequency matrix on top of a source formal winner. "
            "The matrix compares a frozen full-window model with blockwise retraining cadences."
        )
    )
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--root-tag", default="deep_alpha_retrain_frequency_formal_20260402_r1")
    parser.add_argument(
        "--frequencies",
        default=",".join(spec.name for spec in DEFAULT_SPECS),
        help="Comma-separated frequency names: annual_freeze, quarterly_63d, monthly_calendar, every_21d",
    )
    parser.add_argument(
        "--python-executable",
        default=r"C:\Users\ASUS\miniconda3\envs\yolos\python.exe",
        help="Python executable used to run child deep_alpha jobs.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _normalize_spec_names(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _resolve_specs(raw: str) -> list[FrequencySpec]:
    selected = _normalize_spec_names(raw)
    lookup = {spec.name: spec for spec in DEFAULT_SPECS}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(f"Unknown frequencies: {', '.join(missing)}")
    return [lookup[name] for name in selected]


def _load_source_equity(source_run_dir: Path) -> pd.DataFrame:
    equity_path = source_run_dir / "equity_curve.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Source equity curve not found: {equity_path}")
    equity = pd.read_csv(
        equity_path,
        encoding="utf-8-sig",
        parse_dates=["date", "signal_date", "execution_date"],
    )
    if equity.empty:
        raise RuntimeError(f"Source equity curve is empty: {equity_path}")
    return equity.sort_values("date").reset_index(drop=True)


def _load_source_actions(source_run_dir: Path) -> pd.DataFrame:
    actions_path = source_run_dir / "actions.csv"
    if not actions_path.exists():
        return pd.DataFrame()
    return pd.read_csv(actions_path, encoding="utf-8-sig", parse_dates=["signal_date", "execution_date"])


def _fixed_day_groups(execution_dates: pd.Index, trading_days: int) -> list[pd.Index]:
    groups: list[pd.Index] = []
    for start in range(0, len(execution_dates), int(trading_days)):
        groups.append(execution_dates[start : start + int(trading_days)])
    return groups


def _calendar_month_groups(execution_dates: pd.Index) -> list[pd.Index]:
    work = pd.DataFrame({"date": pd.to_datetime(execution_dates)})
    groups: list[pd.Index] = []
    for _period, frame in work.groupby(work["date"].dt.to_period("M"), sort=True):
        groups.append(pd.Index(frame["date"].tolist()))
    return groups


def _build_groups(spec: FrequencySpec, execution_dates: pd.Index) -> list[pd.Index]:
    if spec.mode == "source_full":
        return [execution_dates]
    if spec.mode == "fixed_days":
        return _fixed_day_groups(execution_dates, trading_days=spec.trading_days)
    if spec.mode == "calendar_month":
        return _calendar_month_groups(execution_dates)
    raise ValueError(f"Unsupported frequency mode: {spec.mode}")


def _build_block_plans(
    *,
    specs: list[FrequencySpec],
    execution_dates: pd.Index,
    source_train_end: pd.Timestamp,
    source_valid_start: pd.Timestamp,
    root_tag: str,
) -> list[BlockPlan]:
    plans: list[BlockPlan] = []
    execution_dates = pd.DatetimeIndex(pd.to_datetime(execution_dates))
    if execution_dates.empty:
        raise RuntimeError("No execution dates available to build retrain blocks.")

    for spec in specs:
        groups = _build_groups(spec, execution_dates)
        for idx, group in enumerate(groups, start=1):
            group = pd.DatetimeIndex(pd.to_datetime(group))
            exec_start = pd.Timestamp(group[0])
            exec_end = pd.Timestamp(group[-1])
            uses_source_slice = idx == 1
            if uses_source_slice:
                train_end = pd.Timestamp(source_train_end)
                valid_start = pd.Timestamp(source_valid_start)
                valid_days = int(len(group))
                signal_start = valid_start
            else:
                start_pos = execution_dates.get_loc(exec_start)
                signal_start = pd.Timestamp(execution_dates[start_pos - 1])
                train_end = pd.Timestamp(source_train_end) if start_pos < 2 else pd.Timestamp(execution_dates[start_pos - 2])
                valid_start = signal_start
                valid_days = int(len(group) + 1)
            experiment_tag = f"{root_tag}/runs/{spec.name}/block_{idx:02d}_{exec_start.strftime('%Y%m%d')}_{exec_end.strftime('%Y%m%d')}"
            plans.append(
                BlockPlan(
                    spec_name=spec.name,
                    spec_label=spec.label,
                    block_index=idx,
                    execution_start=str(exec_start.date()),
                    execution_end=str(exec_end.date()),
                    signal_start=str(pd.Timestamp(signal_start).date()),
                    train_end=str(pd.Timestamp(train_end).date()),
                    valid_start=str(pd.Timestamp(valid_start).date()),
                    valid_days=int(valid_days),
                    execution_days=int(len(group)),
                    uses_source_slice=uses_source_slice,
                    experiment_tag=experiment_tag,
                )
            )
    return plans


def _build_block_command(
    *,
    source_run_dir: Path,
    python_executable: str,
    block: BlockPlan,
    source_valid_end: str,
) -> list[str]:
    metrics, cfg = _load_source_config(source_run_dir)
    cmd: list[str] = [python_executable, str(RUN_SCRIPT)]

    _append_arg(cmd, "--data-source", "tq")
    _append_arg(cmd, "--start-date", cfg.get("start_date", "20210101"))
    _append_arg(cmd, "--end-date", source_valid_end.replace("-", ""))
    _append_arg(cmd, "--benchmark", cfg.get("benchmark", "000300.SH"))

    rolling_pool = str(metrics.get("rolling_liquidity_pool", "") or "").strip()
    liquidity_pool = str(metrics.get("liquidity_pool", "") or "").strip()
    stocks_file = str(metrics.get("stocks_file", "") or "").strip()
    if rolling_pool:
        _append_arg(cmd, "--rolling-liquidity-pool", rolling_pool)
        _append_arg(cmd, "--pool-rebalance-days", int(metrics.get("rolling_pool_rebalance_days", 21) or 21))
        _append_arg(cmd, "--pool-adv-window", int(metrics.get("rolling_pool_adv_window", 20) or 20))
    elif liquidity_pool:
        _append_arg(cmd, "--liquidity-pool", liquidity_pool)
    elif stocks_file:
        _append_arg(cmd, "--stocks-file", stocks_file)

    _append_arg(cmd, "--lookback-window", cfg.get("lookback_window", 120))
    _append_arg(cmd, "--prediction-horizons", _format_prediction_horizons(cfg.get("prediction_horizons")))
    _append_arg(cmd, "--return-loss-mode", cfg.get("return_loss_mode", "top_bottom_bce"))
    _append_arg(cmd, "--return-target-transform", cfg.get("return_target_transform", "raw"))
    _append_arg(cmd, "--return-top-frac", cfg.get("return_top_frac", 0.2))
    _append_arg(cmd, "--return-bottom-frac", cfg.get("return_bottom_frac", 0.2))
    _append_arg(cmd, "--task-loss-weights", _format_task_loss_weights(cfg.get("target_loss_weights")))
    _append_arg(cmd, "--score-horizon-weights", _format_score_horizon_weights(cfg.get("score_horizon_weights")))
    _append_arg(cmd, "--score-rank-blend", cfg.get("score_rank_blend", 0.35))
    _append_arg(cmd, "--score-downside-penalty", cfg.get("score_downside_penalty", 0.25))
    _append_arg(cmd, "--score-risk-mode", cfg.get("score_risk_mode", "subtract"))
    _append_arg(cmd, "--score-risk-gate-threshold", cfg.get("score_risk_gate_threshold", 0.35))
    _append_arg(cmd, "--score-risk-state-thresholds", _format_state_thresholds(metrics.get("score_risk_state_thresholds")))
    _append_arg(cmd, "--score-head-method", metrics.get("score_head_method", "manual"))
    _append_flag(cmd, "--adaptive-task-weights", bool(metrics.get("adaptive_task_weights", False)))
    _append_arg(cmd, "--adaptive-task-window-days", metrics.get("adaptive_task_window_days", cfg.get("train_eval_window_days", 126)))

    _append_arg(cmd, "--train-end-date", block.train_end)
    _append_arg(cmd, "--valid-start-date", block.valid_start)
    _append_arg(cmd, "--valid-days", int(block.valid_days))
    _append_arg(cmd, "--train-eval-window-days", cfg.get("train_eval_window_days", 126))

    _append_arg(cmd, "--batch-size", cfg.get("batch_size", 256))
    _append_arg(cmd, "--num-workers", cfg.get("num_workers", 0))
    _append_flag(cmd, "--pin-memory", bool(cfg.get("pin_memory", True)), negative_flag="--no-pin-memory")
    _append_arg(cmd, "--hidden-dim", cfg.get("hidden_dim", 96))
    _append_arg(cmd, "--encoder-family", cfg.get("encoder_family", "patch_transformer"))
    _append_arg(cmd, "--patch-len", cfg.get("patch_len", 5))
    _append_arg(cmd, "--pretrained-encoder-path", cfg.get("pretrained_encoder_path", ""))
    _append_arg(cmd, "--return-head-mode", cfg.get("return_head_mode", "shared"))
    _append_arg(cmd, "--context-dim", cfg.get("context_dim", 16))
    _append_flag(cmd, "--state-context", bool(cfg.get("state_context", False)))
    _append_flag(cmd, "--liquidity-context", bool(cfg.get("liquidity_context", False)))
    _append_flag(cmd, "--structure-context", bool(cfg.get("structure_context", False)))
    _append_flag(cmd, "--aux-structure-task", bool(cfg.get("aux_structure_task", False)))
    _append_arg(cmd, "--aux-structure-loss-weight", cfg.get("aux_structure_loss_weight", 0.1))
    _append_arg(cmd, "--aux-structure-label-smoothing", cfg.get("aux_structure_label_smoothing", 0.05))
    _append_flag(cmd, "--structure-prototype-task", bool(cfg.get("structure_prototype_task", False)))
    _append_arg(cmd, "--structure-prototype-loss-weight", cfg.get("structure_prototype_loss_weight", 0.05))
    _append_arg(cmd, "--structure-prototype-temperature", cfg.get("structure_prototype_temperature", 0.2))
    _append_arg(cmd, "--transformer-heads", cfg.get("transformer_heads", 4))
    _append_arg(cmd, "--transformer-layers", cfg.get("transformer_layers", 2))
    _append_arg(cmd, "--dropout", cfg.get("dropout", 0.10))
    _append_arg(cmd, "--learning-rate", cfg.get("learning_rate", 1e-3))
    _append_arg(cmd, "--weight-decay", cfg.get("weight_decay", 1e-4))
    _append_arg(cmd, "--epochs", cfg.get("epochs", 8))
    _append_arg(cmd, "--min-epochs", cfg.get("min_epochs", 4))
    _append_arg(cmd, "--early-stop-patience", cfg.get("early_stop_patience", 2))
    _append_arg(cmd, "--lr-plateau-patience", cfg.get("lr_plateau_patience", 1))
    _append_arg(cmd, "--lr-plateau-factor", cfg.get("lr_plateau_factor", 0.5))
    _append_arg(cmd, "--min-improvement", cfg.get("min_improvement", 1e-4))
    _append_flag(cmd, "--use-amp", bool(cfg.get("use_amp", True)), negative_flag="--no-amp")
    _append_flag(cmd, "--safe-runtime-profile", bool(cfg.get("safe_runtime_profile", True)), negative_flag="--no-safe-runtime-profile")
    _append_arg(cmd, "--ranking-loss-weight", cfg.get("ranking_loss_weight", 0.0))
    _append_arg(cmd, "--listwise-loss-weight", cfg.get("listwise_loss_weight", 0.0))
    _append_arg(cmd, "--listwise-temperature", cfg.get("listwise_temperature", 0.35))
    _append_arg(cmd, "--max-rank-pairs-per-group", cfg.get("max_rank_pairs_per_group", 2048))
    _append_arg(cmd, "--random-seed", cfg.get("random_seed", 7))
    _append_arg(cmd, "--market-state-count", cfg.get("market_state_count", 4))
    _append_arg(cmd, "--holding-count", cfg.get("holding_count", 5))
    _append_arg(cmd, "--rebalance-freq", cfg.get("rebalance_freq", "1d"))
    _append_arg(cmd, "--max-weight", cfg.get("max_weight", 0.25))
    _append_flag(cmd, "--relation-layer", bool(metrics.get("relation_layer", False)))
    _append_flag(cmd, "--liquidity-layer", bool(cfg.get("liquidity_layer", False)))
    _append_arg(cmd, "--liquidity-bucket-count", cfg.get("liquidity_bucket_count", 5))
    _append_flag(cmd, "--dynamic-graph-layer", bool(cfg.get("dynamic_graph_layer", False)))
    _append_arg(cmd, "--dynamic-graph-top-k", cfg.get("dynamic_graph_top_k", 8))
    _append_arg(cmd, "--dynamic-graph-temperature", cfg.get("dynamic_graph_temperature", 0.35))
    _append_arg(cmd, "--dynamic-graph-industry-boost", cfg.get("dynamic_graph_industry_boost", 0.15))
    _append_arg(cmd, "--dynamic-graph-style-boost", cfg.get("dynamic_graph_style_boost", 0.05))
    _append_flag(cmd, "--short-alpha-features", bool(cfg.get("short_alpha_features", False)))
    _append_arg(cmd, "--breakout-event-horizon", cfg.get("breakout_event_horizon", 5))
    _append_arg(cmd, "--breakout-event-threshold", cfg.get("breakout_event_threshold", 0.08))
    _append_arg(cmd, "--breakout-event-pullback-limit", cfg.get("breakout_event_pullback_limit", 0.03))
    _append_arg(cmd, "--breakout-event-loss-weight", cfg.get("breakout_event_loss_weight", 0.0))
    _append_arg(cmd, "--clean-breakout-event-loss-weight", cfg.get("clean_breakout_event_loss_weight", 0.0))
    _append_arg(cmd, "--min-adv20", cfg.get("min_adv20", 50_000.0))
    _append_arg(cmd, "--min-price", cfg.get("min_price", 2.0))
    _append_arg(cmd, "--max-price", cfg.get("max_price", 300.0))

    execution_alignment_mode = str(metrics.get("execution_alignment_mode", "off") or "off")
    _append_arg(cmd, "--execution-alignment-mode", execution_alignment_mode)
    _append_arg(cmd, "--execution-alignment-profile", metrics.get("execution_alignment_profile", "regoff_k2_10d_ensemble_native_anchor"))
    _append_arg(cmd, "--execution-alignment-objective", metrics.get("execution_alignment_objective", "robust_composite"))
    candidate_profiles = metrics.get("execution_alignment_candidate_profiles", [])
    if isinstance(candidate_profiles, list) and candidate_profiles:
        _append_arg(cmd, "--execution-alignment-candidate-profiles", ",".join(str(item) for item in candidate_profiles))
    _append_arg(cmd, "--execution-alignment-transaction-cost-bps", metrics.get("execution_alignment_transaction_cost_bps", 0.0))
    _append_arg(cmd, "--execution-alignment-slippage-bps", metrics.get("execution_alignment_slippage_bps", 0.0))
    _append_arg(cmd, "--execution-alignment-sell-tax-bps", metrics.get("execution_alignment_sell_tax_bps", 0.0))

    _append_arg(cmd, "--experiment-tag", block.experiment_tag)
    return cmd


def _block_metrics_path(root_tag: str, block: BlockPlan) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / block.spec_name / block.block_slug / "metrics.json"


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_equity(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "equity_curve.csv"
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date", "signal_date", "execution_date"])


def _load_run_actions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "actions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["signal_date", "execution_date"])


def _compute_equity_metrics(frame: pd.DataFrame) -> dict[str, float]:
    work = frame.sort_values("date").reset_index(drop=True).copy()
    if work.empty:
        raise ValueError("Cannot score an empty stitched equity frame.")

    portfolio_equity = pd.Series((1.0 + work["portfolio_return"].fillna(0.0)).cumprod().to_numpy(), index=work["date"])
    benchmark_equity = pd.Series((1.0 + work["benchmark_return"].fillna(0.0)).cumprod().to_numpy(), index=work["date"])
    excess_equity = (portfolio_equity / benchmark_equity).replace([np.inf, -np.inf], np.nan).ffill().dropna()

    portfolio_ann_ret = _annualized_return(portfolio_equity)
    benchmark_ann_ret = _annualized_return(benchmark_equity)
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else 0.0
    portfolio_ann_vol = _annualized_vol(work["portfolio_return"].dropna())
    excess_ann_vol = _annualized_vol(work["excess_return"].dropna())

    out = {
        "total_return": float(portfolio_equity.iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "annual_vol": float(portfolio_ann_vol),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else 0.0,
        "max_drawdown": float(_max_drawdown(portfolio_equity)),
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else 0.0,
        "excess_annual_return": float(excess_ann_ret),
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 else 0.0,
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else 0.0,
        "avg_holding_count": float(work["holding_count"].fillna(0.0).mean()),
        "avg_turnover": float(work["turnover"].fillna(0.0).mean()),
        "hit_rate": float(work["portfolio_return"].fillna(0.0).gt(work["benchmark_return"].fillna(0.0)).mean()),
        "regime_active_ratio": float(work["regime_on"].fillna(False).astype(bool).mean()) if "regime_on" in work.columns else 1.0,
    }
    if "gross_portfolio_return" in work.columns:
        gross_equity = pd.Series((1.0 + work["gross_portfolio_return"].fillna(0.0)).cumprod().to_numpy(), index=work["date"])
        gross_ann_ret = _annualized_return(gross_equity)
        gross_ann_vol = _annualized_vol(work["gross_portfolio_return"].dropna())
        out.update(
            {
                "gross_total_return": float(gross_equity.iloc[-1] - 1.0),
                "gross_annual_return": float(gross_ann_ret),
                "gross_annual_vol": float(gross_ann_vol),
                "gross_sharpe": float(gross_ann_ret / gross_ann_vol) if gross_ann_vol > 0 else 0.0,
                "gross_max_drawdown": float(_max_drawdown(gross_equity)),
            }
        )
    if "buy_turnover" in work.columns:
        out["avg_buy_turnover"] = float(work["buy_turnover"].fillna(0.0).mean())
    if "sell_turnover" in work.columns:
        out["avg_sell_turnover"] = float(work["sell_turnover"].fillna(0.0).mean())
    if "trading_cost_return" in work.columns:
        out["avg_trading_cost_return"] = float(work["trading_cost_return"].fillna(0.0).mean())
        out["total_trading_cost_return"] = float(work["trading_cost_return"].fillna(0.0).sum())
    return out


def _slice_source_block(
    *,
    source_equity: pd.DataFrame,
    source_actions: pd.DataFrame,
    block: BlockPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = pd.Timestamp(block.execution_start)
    end_ts = pd.Timestamp(block.execution_end)
    equity = source_equity.loc[(source_equity["date"] >= start_ts) & (source_equity["date"] <= end_ts)].copy()
    actions = source_actions.loc[
        (source_actions["execution_date"] >= start_ts) & (source_actions["execution_date"] <= end_ts)
    ].copy() if not source_actions.empty else pd.DataFrame()
    return equity, actions


def _slice_run_block(
    *,
    run_dir: Path,
    block: BlockPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    execution_start = pd.Timestamp(block.execution_start)
    execution_end = pd.Timestamp(block.execution_end)
    equity = _load_run_equity(run_dir)
    actions = _load_run_actions(run_dir)
    equity = equity.loc[(equity["date"] >= execution_start) & (equity["date"] <= execution_end)].copy()
    if not actions.empty:
        actions = actions.loc[
            (actions["execution_date"] >= execution_start) & (actions["execution_date"] <= execution_end)
        ].copy()
    return equity, actions


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{pd.Timestamp(start).date()} -> {pd.Timestamp(end).date()}"


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve()
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source run directory not found: {source_run_dir}")

    specs = _resolve_specs(args.frequencies)
    source_metrics, _source_cfg = _load_source_config(source_run_dir)
    source_equity = _load_source_equity(source_run_dir)
    source_actions = _load_source_actions(source_run_dir)

    execution_dates = pd.DatetimeIndex(pd.to_datetime(source_equity["date"]))
    source_train_end = pd.Timestamp(source_metrics["train_end"])
    source_valid_start = pd.Timestamp(source_metrics["valid_start"])
    source_valid_end = str(source_metrics["valid_end"])
    root_tag = str(args.root_tag).strip()

    plans = _build_block_plans(
        specs=specs,
        execution_dates=execution_dates,
        source_train_end=source_train_end,
        source_valid_start=source_valid_start,
        root_tag=root_tag,
    )

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([plan.__dict__ for plan in plans]).to_csv(output_dir / "block_plan.csv", index=False, encoding="utf-8-sig")

    block_detail_rows: list[dict[str, Any]] = []
    stitched_summary_rows: list[dict[str, Any]] = []
    stitched_equity_map: dict[str, pd.DataFrame] = {}
    source_map: dict[str, list[dict[str, Any]]] = {}

    for spec in specs:
        spec_plans = [plan for plan in plans if plan.spec_name == spec.name]
        stitched_equity_parts: list[pd.DataFrame] = []
        stitched_action_parts: list[pd.DataFrame] = []
        source_map[spec.name] = []

        for block in spec_plans:
            if block.uses_source_slice:
                metrics_path = source_run_dir / "metrics.json"
                run_dir = source_run_dir
                reused_existing = True
                equity_slice, actions_slice = _slice_source_block(
                    source_equity=source_equity,
                    source_actions=source_actions,
                    block=block,
                )
            else:
                metrics_path = _block_metrics_path(root_tag, block)
                run_dir = metrics_path.parent
                reused_existing = False
                if not metrics_path.exists() or args.force_rerun:
                    command = _build_block_command(
                        source_run_dir=source_run_dir,
                        python_executable=args.python_executable,
                        block=block,
                        source_valid_end=source_valid_end,
                    )
                    _run_command(command)
                else:
                    reused_existing = True
                equity_slice, actions_slice = _slice_run_block(run_dir=run_dir, block=block)

            if equity_slice.empty:
                raise RuntimeError(
                    f"Block {block.spec_name}:{block.block_slug} produced no equity rows for "
                    f"{block.execution_start} -> {block.execution_end}."
                )

            block_metrics = _compute_equity_metrics(equity_slice)
            block_equity = equity_slice.copy()
            block_equity["frequency_name"] = block.spec_name
            block_equity["frequency_label"] = block.spec_label
            block_equity["block_index"] = block.block_index
            block_equity["block_slug"] = block.block_slug
            stitched_equity_parts.append(block_equity)

            if not actions_slice.empty:
                block_actions = actions_slice.copy()
                block_actions["frequency_name"] = block.spec_name
                block_actions["frequency_label"] = block.spec_label
                block_actions["block_index"] = block.block_index
                block_actions["block_slug"] = block.block_slug
                stitched_action_parts.append(block_actions)

            block_detail_rows.append(
                {
                    "frequency_name": block.spec_name,
                    "frequency_label": block.spec_label,
                    "block_index": block.block_index,
                    "block_slug": block.block_slug,
                    "train_end": block.train_end,
                    "valid_start": block.valid_start,
                    "signal_start": block.signal_start,
                    "execution_start": block.execution_start,
                    "execution_end": block.execution_end,
                    "valid_days": block.valid_days,
                    "execution_days": block.execution_days,
                    "uses_source_slice": block.uses_source_slice,
                    "reused_existing": reused_existing,
                    "metrics_path": str(metrics_path.resolve()),
                    "run_dir": str(run_dir.resolve()),
                    **block_metrics,
                }
            )
            source_map[spec.name].append(
                {
                    "block_index": block.block_index,
                    "block_slug": block.block_slug,
                    "metrics_path": str(metrics_path.resolve()),
                    "run_dir": str(run_dir.resolve()),
                    "uses_source_slice": bool(block.uses_source_slice),
                }
            )

        stitched_equity = pd.concat(stitched_equity_parts, ignore_index=True).sort_values("date").reset_index(drop=True)
        if stitched_equity["date"].duplicated().any():
            dupes = stitched_equity.loc[stitched_equity["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
            raise RuntimeError(f"Duplicate stitched dates detected for {spec.name}: {dupes[:5]}")
        stitched_actions = (
            pd.concat(stitched_action_parts, ignore_index=True).sort_values("execution_date").reset_index(drop=True)
            if stitched_action_parts
            else pd.DataFrame()
        )

        stitched_metrics = _compute_equity_metrics(stitched_equity)
        stitched_metrics.update(
            {
                "frequency_name": spec.name,
                "frequency_label": spec.label,
                "block_count": int(len(spec_plans)),
                "retrained_block_count": int(sum(0 if plan.uses_source_slice else 1 for plan in spec_plans)),
                "execution_start": str(pd.Timestamp(stitched_equity["date"].min()).date()),
                "execution_end": str(pd.Timestamp(stitched_equity["date"].max()).date()),
                "source_run_dir": str(source_run_dir.resolve()),
            }
        )
        stitched_summary_rows.append(stitched_metrics)
        stitched_equity_map[spec.name] = stitched_equity.copy()

        stitched_equity.to_csv(output_dir / f"{spec.name}_stitched_equity_curve.csv", index=False, encoding="utf-8-sig")
        if not stitched_actions.empty:
            stitched_actions.to_csv(output_dir / f"{spec.name}_stitched_actions.csv", index=False, encoding="utf-8-sig")
        (output_dir / f"{spec.name}_stitched_metrics.json").write_text(
            json.dumps(stitched_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    detail_df = pd.DataFrame(block_detail_rows).sort_values(["frequency_name", "block_index"]).reset_index(drop=True)
    summary_df = pd.DataFrame(stitched_summary_rows).sort_values(
        ["excess_annual_return", "excess_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    common_start = max(pd.Timestamp(frame["date"].min()) for frame in stitched_equity_map.values())
    common_end = min(pd.Timestamp(frame["date"].max()) for frame in stitched_equity_map.values())
    if common_start > common_end:
        raise RuntimeError(f"Computed an invalid common comparison window: {_window_label(common_start, common_end)}")

    common_summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        stitched_equity = stitched_equity_map[spec.name]
        common_equity = stitched_equity.loc[
            (stitched_equity["date"] >= common_start) & (stitched_equity["date"] <= common_end)
        ].copy()
        if common_equity.empty:
            raise RuntimeError(
                f"Common comparison window {_window_label(common_start, common_end)} "
                f"produced no rows for {spec.name}."
            )
        row = _compute_equity_metrics(common_equity)
        row.update(
            {
                "frequency_name": spec.name,
                "frequency_label": spec.label,
                "block_count": int(len([plan for plan in plans if plan.spec_name == spec.name])),
                "retrained_block_count": int(sum(1 for plan in plans if plan.spec_name == spec.name and not plan.uses_source_slice)),
                "execution_start": str(pd.Timestamp(common_equity["date"].min()).date()),
                "execution_end": str(pd.Timestamp(common_equity["date"].max()).date()),
                "comparison_window_start": str(common_start.date()),
                "comparison_window_end": str(common_end.date()),
                "raw_execution_end": str(pd.Timestamp(stitched_equity["date"].max()).date()),
                "source_run_dir": str(source_run_dir.resolve()),
            }
        )
        common_summary_rows.append(row)
    common_summary_df = pd.DataFrame(common_summary_rows).sort_values(
        ["excess_annual_return", "excess_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    detail_df.to_csv(output_dir / "block_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "frequency_summary.csv", index=False, encoding="utf-8-sig")
    common_summary_df.to_csv(output_dir / "frequency_summary_common_window.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8")

    best_by_excess_annual = common_summary_df.sort_values("excess_annual_return", ascending=False).iloc[0]
    best_by_excess_sharpe = common_summary_df.sort_values("excess_sharpe", ascending=False).iloc[0]

    lines = [
        "# Deep Alpha Retrain Frequency Formal Matrix",
        "",
        "## Protocol",
        f"- source_run: `{source_run_dir.as_posix()}`",
        f"- formal_window: `{source_metrics['valid_start']} -> {source_metrics['valid_end']}`",
        f"- source_train_end: `{source_metrics['train_end']}`",
        "- execution_mode: `next_open`",
        "- comparison rule: same formal winner config, then vary only retrain cadence",
        "- block stitching rule: block 1 reuses the source formal run; later blocks retrain before the block and keep returns only on the target execution range",
        f"- common_comparison_window: `{_window_label(common_start, common_end)}`",
        "",
        "## Raw Summary",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"### {row['frequency_label']}",
                f"- block_count: `{int(row['block_count'])}`",
                f"- retrained_block_count: `{int(row['retrained_block_count'])}`",
                f"- annual_return: `{_format_pct(float(row['annual_return']))}`",
                f"- excess_annual_return: `{_format_pct(float(row['excess_annual_return']))}`",
                f"- excess_sharpe: `{_format_float(float(row['excess_sharpe']))}`",
                f"- excess_max_drawdown: `{_format_pct(float(row['excess_max_drawdown']))}`",
                f"- avg_turnover: `{_format_float(float(row['avg_turnover']))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Common Window Summary",
        ]
    )
    for _, row in common_summary_df.iterrows():
        lines.extend(
            [
                f"### {row['frequency_label']}",
                f"- comparison_window: `{row['execution_start']} -> {row['execution_end']}`",
                f"- raw_execution_end: `{row['raw_execution_end']}`",
                f"- annual_return: `{_format_pct(float(row['annual_return']))}`",
                f"- excess_annual_return: `{_format_pct(float(row['excess_annual_return']))}`",
                f"- excess_sharpe: `{_format_float(float(row['excess_sharpe']))}`",
                f"- excess_max_drawdown: `{_format_pct(float(row['excess_max_drawdown']))}`",
                f"- avg_turnover: `{_format_float(float(row['avg_turnover']))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Leaderboard",
            "- leaderboard_basis: `common comparison window`",
            f"- best_by_excess_annual_return: `{best_by_excess_annual['frequency_label']}` = `{_format_pct(float(best_by_excess_annual['excess_annual_return']))}`",
            f"- best_by_excess_sharpe: `{best_by_excess_sharpe['frequency_label']}` = `{_format_float(float(best_by_excess_sharpe['excess_sharpe']))}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "best_by_excess_annual_return": {
                    "frequency_name": str(best_by_excess_annual["frequency_name"]),
                    "frequency_label": str(best_by_excess_annual["frequency_label"]),
                    "value": float(best_by_excess_annual["excess_annual_return"]),
                },
                "best_by_excess_sharpe": {
                    "frequency_name": str(best_by_excess_sharpe["frequency_name"]),
                    "frequency_label": str(best_by_excess_sharpe["frequency_label"]),
                    "value": float(best_by_excess_sharpe["excess_sharpe"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
