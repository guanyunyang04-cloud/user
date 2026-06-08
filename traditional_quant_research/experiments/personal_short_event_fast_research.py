"""Fast personal-capital diagnostics for short-event ML opportunities."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.all_limitup_followup_research import (
    DEFAULT_ML_RUN_ROOT,
    LABEL_WINDOWS,
    apply_stress_adjustments,
    enrich_predictions,
    latest_run_dir,
    read_followup_event_panel,
    read_predictions,
    read_source_summary,
    source_event_path,
    _json_ready,
    _markdown_table,
    _truthy_col,
)
from traditional_quant_research.experiments.all_limitup_ml_strategy_research import (
    DEFAULT_EVAL_YEARS,
    DEFAULT_FEE_BPS,
    DEFAULT_PROFILE,
    DEFAULT_TARGET_WINDOW,
    find_latest_event_panel,
    summarize_portfolio,
    summarize_portfolio_yearly,
)
from traditional_quant_research.experiments.all_limitup_risk_penalized_strategy_research import (
    PENALTY_SPECS,
    RISK_FLAG_COLUMNS,
    build_penalty_scores,
    build_risk_flags,
)
from traditional_quant_research.experiments.short_open_known_factor_rebuild import (
    audit_buy_feature_columns,
    feature_columns_for_profile,
)
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    _assert_memory_available,
    available_memory_gb,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/personal_short_event_fast_research")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-07_personal_short_event_fast_research.md")
DEFAULT_GENERALIZED_RUN_ROOT = Path("traditional_quant_research/output/experiments/generalized_strong_event_pool_research")
DEFAULT_SCORE_FLOORS = (-np.inf, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_SCORE_QUANTILES = (np.nan, 0.80, 0.90, 0.95)
DEFAULT_MAX_POSITIONS = (1, 2)
DEFAULT_MIN_PERIODS_FOR_BEST = 20


@dataclass(frozen=True)
class GateSpec:
    name: str
    max_risk_flags: int | None = None
    block_weak_market: bool = False
    block_weak_industry: bool = False
    block_high_open_gap: bool = False
    block_overextended: bool = False
    require_kama_atr: bool = False
    first_board_only: bool = False


ALL_LIMITUP_GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec("score_only"),
    GateSpec("risk_le2", max_risk_flags=2),
    GateSpec("risk_le1", max_risk_flags=1),
    GateSpec("no_weak_no_gap", block_weak_market=True, block_weak_industry=True, block_high_open_gap=True),
    GateSpec("confirm_kama_atr", require_kama_atr=True),
    GateSpec("first_board_risk_le2", max_risk_flags=2, first_board_only=True),
)
GENERALIZED_EVENT_POOLS = (
    "gen_exec_all",
    "gen_non_limit",
    "gen_big_up",
    "gen_kama_breakout",
    "gen_new_high_breakout",
    "gen_big_up_kama_new_high",
)


def run_personal_short_event_fast_research(
    *,
    ml_run_dir: str | Path | None = None,
    event_file: str | Path | None = None,
    generalized_run_dir: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    target_window: int = DEFAULT_TARGET_WINDOW,
    fee_bps: float = DEFAULT_FEE_BPS,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    score_floors: Sequence[float] = DEFAULT_SCORE_FLOORS,
    score_quantiles: Sequence[float] = DEFAULT_SCORE_QUANTILES,
    max_positions_values: Sequence[int] = DEFAULT_MAX_POSITIONS,
    min_periods_for_best: int = DEFAULT_MIN_PERIODS_FOR_BEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
) -> dict[str, Any]:
    run_id = f"personal_short_event_fast_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_eval_years = tuple(int(year) for year in eval_years)

    _assert_memory_available(min_available_memory_gb, context="before personal fast inputs")
    all_limitup_scored, source_meta = load_all_limitup_scored_frame(
        ml_run_dir=ml_run_dir,
        event_file=event_file,
        profile=profile,
        fee_bps=fee_bps,
        eval_years=selected_eval_years,
    )
    _assert_memory_available(min_available_memory_gb, context="after all-limit-up personal scoring")
    generalized_predictions, generalized_meta = load_generalized_predictions(generalized_run_dir)
    generalized_predictions = generalized_predictions.loc[generalized_predictions["eval_year"].isin(selected_eval_years)].copy()

    all_limitup_summary, all_limitup_yearly, all_limitup_stress, all_limitup_selected = evaluate_all_limitup_fast_grid(
        all_limitup_scored,
        score_floors=score_floors,
        score_quantiles=score_quantiles,
        max_positions_values=max_positions_values,
    )
    generalized_summary, generalized_yearly, generalized_stress, generalized_selected = evaluate_generalized_fast_grid(
        generalized_predictions,
        score_floors=score_floors,
        score_quantiles=score_quantiles,
        max_positions_values=max_positions_values,
    )
    strategy_summary = pd.concat([all_limitup_summary, generalized_summary], ignore_index=True)
    strategy_yearly = pd.concat([all_limitup_yearly, generalized_yearly], ignore_index=True)
    stress_summary = pd.concat([all_limitup_stress, generalized_stress], ignore_index=True)
    selected_trades = pd.concat([all_limitup_selected, generalized_selected], ignore_index=True)
    strategy_summary = strategy_summary.sort_values(
        ["mean_period_net_ret_pct", "positive_year_rate", "min_year_period_ret_pct", "period_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    stress_summary = stress_summary.sort_values(
        ["scenario", "mean_period_net_ret_pct", "positive_year_rate", "period_count"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    best_rows = select_fast_watchlist(strategy_summary, stress_summary, min_periods=int(min_periods_for_best))
    summary = build_summary(
        run_id=run_id,
        run_dir=run_dir,
        source_meta=source_meta,
        generalized_meta=generalized_meta,
        all_limitup_scored=all_limitup_scored,
        generalized_predictions=generalized_predictions,
        strategy_summary=strategy_summary,
        stress_summary=stress_summary,
        best_rows=best_rows,
        profile=profile,
        target_window=target_window,
        fee_bps=fee_bps,
        min_periods_for_best=int(min_periods_for_best),
        min_available_memory_gb=min_available_memory_gb,
    )
    best_strategy_id = str(summary.get("best_fast_watchlist", {}).get("strategy_id", ""))
    best_trades = selected_trades.loc[selected_trades["strategy_id"].eq(best_strategy_id)].copy() if best_strategy_id else pd.DataFrame()
    markdown = render_markdown(
        summary,
        strategy_summary=strategy_summary,
        stress_summary=stress_summary,
        strategy_yearly=strategy_yearly,
        best_rows=best_rows,
        best_trades=best_trades,
    )

    strategy_summary.to_csv(run_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    strategy_yearly.to_csv(run_dir / "strategy_yearly.csv", index=False, encoding="utf-8-sig")
    stress_summary.to_csv(run_dir / "stress_summary.csv", index=False, encoding="utf-8-sig")
    selected_trades.to_csv(run_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    best_rows.to_csv(run_dir / "fast_watchlist.csv", index=False, encoding="utf-8-sig")
    best_trades.to_csv(run_dir / "best_strategy_trades.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    del all_limitup_scored, generalized_predictions
    gc.collect()
    _assert_memory_available(min_available_memory_gb, context="after personal fast outputs")
    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path else None}


def load_all_limitup_scored_frame(
    *,
    ml_run_dir: str | Path | None,
    event_file: str | Path | None,
    profile: str,
    fee_bps: float,
    eval_years: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_ml_run_dir = Path(ml_run_dir) if ml_run_dir is not None else latest_run_dir(DEFAULT_ML_RUN_ROOT)
    source_summary = read_source_summary(source_ml_run_dir)
    source_event_file = Path(event_file) if event_file is not None else source_event_path(source_summary)
    if not source_event_file.exists():
        source_event_file = find_latest_event_panel()
    feature_columns = tuple(feature_columns_for_profile(profile))
    audit_buy_feature_columns(feature_columns)
    predictions = read_predictions(source_ml_run_dir / "ml_predictions.csv")
    event_panel = read_followup_event_panel(
        source_event_file,
        feature_columns=feature_columns,
        label_windows=LABEL_WINDOWS,
        fee_bps=fee_bps,
    )
    enriched = enrich_predictions(predictions, event_panel, feature_columns=feature_columns)
    enriched = enriched.loc[enriched["eval_year"].isin(tuple(int(year) for year in eval_years))].copy()
    enriched = build_risk_flags(enriched, event_panel, eval_years=eval_years)
    scored = build_penalty_scores(enriched, PENALTY_SPECS)
    return scored, {
        "source_ml_run_dir": str(source_ml_run_dir),
        "source_event_file": str(source_event_file),
        "source_run_id": source_summary.get("run_id"),
    }


def load_generalized_predictions(generalized_run_dir: str | Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(generalized_run_dir) if generalized_run_dir is not None else latest_run_dir(DEFAULT_GENERALIZED_RUN_ROOT)
    path = run_dir / "ml_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    required = {
        "date",
        "year",
        "eval_year",
        "code",
        "name_on_date",
        "industry",
        "entry_date",
        "entry_open",
        "entry_open_near_limit",
        "executable_entry",
        "exit_date",
        "target_net_ret_pct",
        "predicted_ret_pct",
        "predicted_big_loss_prob",
        "ml_score",
        "primary_event_type",
        "event_strength_score",
        "event_flag_count",
        "event_limit_up_core",
        "event_near_limit",
        "event_big_up",
        "event_volume_atr_breakout",
        "event_new_high_breakout",
        "event_kama_breakout",
        "event_trend_accel",
    }
    frame = pd.read_csv(path, usecols=lambda column: column in required, low_memory=False)
    for column in ("date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame = reduce_memory(frame)
    summary_path = run_dir / "summary.json"
    meta = {"generalized_run_dir": str(run_dir)}
    if summary_path.exists():
        meta.update(json.loads(summary_path.read_text(encoding="utf-8")))
    return frame, meta


def evaluate_all_limitup_fast_grid(
    frame: pd.DataFrame,
    *,
    score_floors: Sequence[float],
    score_quantiles: Sequence[float],
    max_positions_values: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = frame.loc[_truthy_col(frame, "executable_entry") & ~_truthy_col(frame, "entry_open_near_limit")].copy()
    pool_masks = build_all_limitup_pool_masks(base)
    score_cols = ("ml_score", "risk_score_balanced_soft", "risk_score_balanced_strict")
    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    stress_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for pool_name, pool_mask in pool_masks.items():
        pool = base.loc[pool_mask].copy()
        eligible_days = int(pd.to_datetime(pool["entry_date"]).nunique())
        for gate in ALL_LIMITUP_GATE_SPECS:
            gated = pool.loc[build_gate_mask(pool, gate)].copy()
            for score_col in score_cols:
                for score_floor in score_floors:
                    for score_quantile in score_quantiles:
                        filtered = apply_score_floor_and_prior_quantile(
                            gated,
                            score_col=score_col,
                            score_floor=float(score_floor),
                            score_quantile=float(score_quantile),
                        )
                        for max_positions in max_positions_values:
                            selected = select_positions(filtered, score_col=score_col, max_positions=int(max_positions))
                            strategy_id = fast_strategy_id(
                                family="limitup",
                                pool_name=pool_name,
                                gate_name=gate.name,
                                score_col=score_col,
                                score_floor=float(score_floor),
                                score_quantile=float(score_quantile),
                                max_positions=int(max_positions),
                            )
                            rows.append(
                                summarize_fast_strategy(
                                    selected,
                                    strategy_id=strategy_id,
                                    family="limitup",
                                    pool_name=pool_name,
                                    gate_name=gate.name,
                                    score_col=score_col,
                                    score_floor=float(score_floor),
                                    score_quantile=float(score_quantile),
                                    eligible_days=eligible_days,
                                    max_positions=int(max_positions),
                                )
                            )
                            if not selected.empty:
                                selected = selected.assign(
                                    strategy_id=strategy_id,
                                    family="limitup",
                                    pool_name=pool_name,
                                    gate_name=gate.name,
                                    score_col=score_col,
                                    score_floor=float(score_floor),
                                    score_quantile=float(score_quantile),
                                )
                                selected_frames.append(selected)
                                yearly = summarize_portfolio_yearly(selected, strategy_id=strategy_id)
                                yearly["family"] = "limitup"
                                yearly["pool_name"] = pool_name
                                yearly_frames.append(yearly)
                            stress_rows.extend(stress_strategy_rows(selected, strategy_id=strategy_id, family="limitup", pool_name=pool_name, max_positions=int(max_positions)))
    return pack_results(rows, yearly_frames, stress_rows, selected_frames)


def evaluate_generalized_fast_grid(
    frame: pd.DataFrame,
    *,
    score_floors: Sequence[float],
    score_quantiles: Sequence[float],
    max_positions_values: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = frame.loc[_truthy_col(frame, "executable_entry") & ~_truthy_col(frame, "entry_open_near_limit")].copy()
    pool_masks = build_generalized_pool_masks(base)
    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    stress_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for pool_name, pool_mask in pool_masks.items():
        pool = base.loc[pool_mask].copy()
        eligible_days = int(pd.to_datetime(pool["entry_date"]).nunique())
        for score_floor in score_floors:
            for score_quantile in score_quantiles:
                filtered = apply_score_floor_and_prior_quantile(
                    pool,
                    score_col="ml_score",
                    score_floor=float(score_floor),
                    score_quantile=float(score_quantile),
                )
                for max_positions in max_positions_values:
                    selected = select_positions(filtered, score_col="ml_score", max_positions=int(max_positions))
                    strategy_id = fast_strategy_id(
                        family="generalized",
                        pool_name=pool_name,
                        gate_name="score_only",
                        score_col="ml_score",
                        score_floor=float(score_floor),
                        score_quantile=float(score_quantile),
                        max_positions=int(max_positions),
                    )
                    rows.append(
                        summarize_fast_strategy(
                            selected,
                            strategy_id=strategy_id,
                            family="generalized",
                            pool_name=pool_name,
                            gate_name="score_only",
                            score_col="ml_score",
                            score_floor=float(score_floor),
                            score_quantile=float(score_quantile),
                            eligible_days=eligible_days,
                            max_positions=int(max_positions),
                        )
                    )
                    if not selected.empty:
                        selected = selected.assign(
                            strategy_id=strategy_id,
                            family="generalized",
                            pool_name=pool_name,
                            gate_name="score_only",
                            score_col="ml_score",
                            score_floor=float(score_floor),
                            score_quantile=float(score_quantile),
                        )
                        selected_frames.append(selected)
                        yearly = summarize_portfolio_yearly(selected, strategy_id=strategy_id)
                        yearly["family"] = "generalized"
                        yearly["pool_name"] = pool_name
                        yearly_frames.append(yearly)
                    stress_rows.extend(stress_strategy_rows(selected, strategy_id=strategy_id, family="generalized", pool_name=pool_name, max_positions=int(max_positions)))
    return pack_results(rows, yearly_frames, stress_rows, selected_frames)


def build_all_limitup_pool_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    index = frame.index
    run = pd.to_numeric(frame.get("limit_up_run_ending_today"), errors="coerce").fillna(0)
    kama = _truthy_col(frame, "open_below_kama_break_limitup")
    atr = _truthy_col(frame, "close_cross_atr_upper")
    market_breadth = pd.to_numeric(frame.get("market_breadth_5d"), errors="coerce")
    market_heat = pd.to_numeric(frame.get("market_limitup_rate"), errors="coerce")
    return {
        "exec_all": pd.Series(True, index=index),
        "first_board": run.le(1),
        "advanced_board": run.ge(2),
        "kama_atr": kama & atr,
        "kama_or_atr": kama | atr,
        "hot_market": market_breadth.ge(0.50) & market_heat.ge(0.02),
        "calm_market": market_breadth.lt(0.50) | market_heat.lt(0.02),
    }


def build_generalized_pool_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    index = frame.index
    limit = _truthy_col(frame, "event_limit_up_core")
    big = frame.get("primary_event_type", pd.Series("", index=index)).astype(str).eq("big_up") | _truthy_col(frame, "event_big_up")
    kama = frame.get("primary_event_type", pd.Series("", index=index)).astype(str).eq("kama_breakout") | _truthy_col(frame, "event_kama_breakout")
    high = frame.get("primary_event_type", pd.Series("", index=index)).astype(str).eq("new_high_breakout") | _truthy_col(frame, "event_new_high_breakout")
    return {
        "gen_exec_all": pd.Series(True, index=index),
        "gen_non_limit": ~limit,
        "gen_big_up": big & ~limit,
        "gen_kama_breakout": kama & ~limit,
        "gen_new_high_breakout": high & ~limit,
        "gen_big_up_kama_new_high": (big | kama | high) & ~limit,
    }


def build_gate_mask(frame: pd.DataFrame, gate: GateSpec) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if gate.max_risk_flags is not None:
        mask &= pd.to_numeric(frame.get("risk_flag_count"), errors="coerce").le(int(gate.max_risk_flags))
    if gate.block_weak_market:
        mask &= ~_truthy_col(frame, "risk_weak_market")
    if gate.block_weak_industry:
        mask &= ~_truthy_col(frame, "risk_weak_industry")
    if gate.block_high_open_gap:
        mask &= ~_truthy_col(frame, "risk_high_open_gap")
    if gate.block_overextended:
        mask &= ~_truthy_col(frame, "risk_overextended")
    if gate.require_kama_atr:
        mask &= _truthy_col(frame, "open_below_kama_break_limitup") & _truthy_col(frame, "close_cross_atr_upper")
    if gate.first_board_only:
        mask &= pd.to_numeric(frame.get("limit_up_run_ending_today"), errors="coerce").fillna(0).le(1)
    return mask.fillna(False)


def apply_score_floor_and_prior_quantile(
    frame: pd.DataFrame,
    *,
    score_col: str,
    score_floor: float,
    score_quantile: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    score = pd.to_numeric(output[score_col], errors="coerce")
    mask = score.notna()
    if np.isfinite(float(score_floor)):
        mask &= score.ge(float(score_floor))
    if np.isfinite(float(score_quantile)):
        threshold = prior_year_score_thresholds(output, score_col=score_col, score_quantile=float(score_quantile))
        mask &= score.ge(threshold)
    return output.loc[mask].copy()


def prior_year_score_thresholds(frame: pd.DataFrame, *, score_col: str, score_quantile: float) -> pd.Series:
    thresholds = pd.Series(-np.inf, index=frame.index, dtype=float)
    eval_years = sorted(pd.to_numeric(frame["eval_year"], errors="coerce").dropna().astype(int).unique().tolist())
    years = pd.to_numeric(frame["eval_year"], errors="coerce")
    scores = pd.to_numeric(frame[score_col], errors="coerce")
    for year in eval_years:
        history = scores.loc[years.lt(year)].dropna()
        threshold = float(history.quantile(float(score_quantile))) if not history.empty else -np.inf
        thresholds.loc[years.eq(year)] = threshold
    return thresholds


def select_positions(frame: pd.DataFrame, *, score_col: str, max_positions: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    clean = frame.dropna(subset=["entry_date", "exit_date", score_col, "target_net_ret_pct"]).copy()
    if clean.empty:
        return clean
    clean["entry_date"] = pd.to_datetime(clean["entry_date"])
    clean["exit_date"] = pd.to_datetime(clean["exit_date"])
    clean = clean.sort_values(["entry_date", score_col, "ml_score"], ascending=[True, False, False])
    open_exit_dates: list[pd.Timestamp] = []
    selected_indices: list[Any] = []
    for entry_date, group in clean.groupby("entry_date", sort=True):
        entry_ts = pd.Timestamp(entry_date)
        open_exit_dates = [date for date in open_exit_dates if date >= entry_ts]
        slots = int(max_positions) - len(open_exit_dates)
        if slots <= 0:
            continue
        take = group.head(slots)
        selected_indices.extend(take.index.tolist())
        open_exit_dates.extend(pd.to_datetime(take["exit_date"]).tolist())
    return clean.loc[selected_indices].copy().reset_index(drop=True)


def summarize_fast_strategy(
    selected: pd.DataFrame,
    *,
    strategy_id: str,
    family: str,
    pool_name: str,
    gate_name: str,
    score_col: str,
    score_floor: float,
    score_quantile: float,
    eligible_days: int,
    max_positions: int,
) -> dict[str, Any]:
    summary = summarize_portfolio(selected, pool_name=pool_name, max_positions=int(max_positions))
    summary.update(
        {
            "strategy_id": strategy_id,
            "family": family,
            "pool_name": pool_name,
            "gate_name": gate_name,
            "score_col": score_col,
            "score_floor": float(score_floor),
            "score_quantile": None if not np.isfinite(float(score_quantile)) else float(score_quantile),
            "eligible_entry_days": int(eligible_days),
            "selection_rate": float(summary["period_count"] / int(eligible_days)) if int(eligible_days) else np.nan,
            "mean_risk_flag_count": _mean(pd.to_numeric(selected.get("risk_flag_count"), errors="coerce")) if not selected.empty and "risk_flag_count" in selected.columns else np.nan,
            "mean_predicted_ret_pct": _mean(pd.to_numeric(selected.get("predicted_ret_pct"), errors="coerce")) if not selected.empty else np.nan,
            "mean_big_loss_prob": _mean(pd.to_numeric(selected.get("predicted_big_loss_prob"), errors="coerce")) if not selected.empty else np.nan,
        }
    )
    return summary


def stress_strategy_rows(selected: pd.DataFrame, *, strategy_id: str, family: str, pool_name: str, max_positions: int) -> list[dict[str, Any]]:
    scenarios = (
        {"scenario": "reported_30bps"},
        {"scenario": "fee_100bps", "extra_fee_bps": 70.0},
        {"scenario": "tail_x1_25", "tail_quantile": 0.10, "tail_loss_multiplier": 1.25},
        {"scenario": "combined_light", "extra_fee_bps": 70.0, "tail_quantile": 0.10, "tail_loss_multiplier": 1.25},
    )
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        stressed = apply_simple_stress(selected, scenario)
        summary = summarize_portfolio(stressed, pool_name=pool_name, max_positions=int(max_positions))
        summary.update({"strategy_id": strategy_id, "family": family, "pool_name": pool_name, "scenario": scenario["scenario"]})
        rows.append(summary)
    return rows


def apply_simple_stress(selected: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    if str(scenario.get("scenario")) == "reported_30bps":
        return selected.copy()
    if "next_open_gap_pct" in selected.columns:
        try:
            stressed, _ = apply_stress_adjustments(selected, scenario)
            return stressed
        except Exception:
            pass
    output = selected.copy()
    returns = pd.to_numeric(output["target_net_ret_pct"], errors="coerce").astype(float)
    returns = returns - float(scenario.get("extra_fee_bps", 0.0)) / 100.0
    if "tail_quantile" in scenario and "tail_loss_multiplier" in scenario and returns.notna().any():
        cutoff = returns.quantile(float(scenario["tail_quantile"]))
        tail = returns.le(cutoff)
        returns.loc[tail] = returns.loc[tail] * float(scenario["tail_loss_multiplier"])
    output["target_net_ret_pct"] = returns
    return output


def pack_results(
    rows: list[dict[str, Any]],
    yearly_frames: list[pd.DataFrame],
    stress_rows: list[dict[str, Any]],
    selected_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(rows)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    stress = pd.DataFrame(stress_rows)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return summary, yearly, stress, selected


def select_fast_watchlist(strategy_summary: pd.DataFrame, stress_summary: pd.DataFrame, *, min_periods: int) -> pd.DataFrame:
    if strategy_summary.empty:
        return pd.DataFrame()
    reported = strategy_summary.loc[strategy_summary["period_count"].ge(int(min_periods))].copy()
    harsh = stress_summary.loc[stress_summary["scenario"].eq("combined_light")].copy()
    harsh = harsh.loc[:, ["strategy_id", "mean_period_net_ret_pct", "positive_year_rate", "min_year_period_ret_pct"]].rename(
        columns={
            "mean_period_net_ret_pct": "combined_light_mean_period_net_ret_pct",
            "positive_year_rate": "combined_light_positive_year_rate",
            "min_year_period_ret_pct": "combined_light_min_year_period_ret_pct",
        }
    )
    merged = reported.merge(harsh, on="strategy_id", how="left")
    watch = merged.loc[
        merged["mean_period_net_ret_pct"].gt(0.5)
        & merged["combined_light_mean_period_net_ret_pct"].gt(0.0)
        & merged["positive_year_rate"].ge(0.50)
    ].copy()
    return watch.sort_values(
        ["combined_light_mean_period_net_ret_pct", "mean_period_net_ret_pct", "period_count"],
        ascending=[False, False, False],
        na_position="last",
    ).head(30)


def build_summary(
    *,
    run_id: str,
    run_dir: Path,
    source_meta: Mapping[str, Any],
    generalized_meta: Mapping[str, Any],
    all_limitup_scored: pd.DataFrame,
    generalized_predictions: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    best_rows: pd.DataFrame,
    profile: str,
    target_window: int,
    fee_bps: float,
    min_periods_for_best: int,
    min_available_memory_gb: float,
) -> dict[str, Any]:
    best = best_rows.iloc[0].to_dict() if not best_rows.empty else {}
    best_reported = strategy_summary.iloc[0].to_dict() if not strategy_summary.empty else {}
    practical_limitup = strategy_summary.loc[strategy_summary["family"].eq("limitup")].head(1)
    practical_gen = strategy_summary.loc[strategy_summary["family"].eq("generalized")].head(1)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "profile": profile,
        "target_window": int(target_window),
        "fee_bps": float(fee_bps),
        "all_limitup_prediction_count": int(len(all_limitup_scored)),
        "generalized_prediction_count": int(len(generalized_predictions)),
        "source_ml_run_dir": source_meta.get("source_ml_run_dir"),
        "source_event_file": source_meta.get("source_event_file"),
        "generalized_run_dir": generalized_meta.get("generalized_run_dir") or generalized_meta.get("run_dir"),
        "strategy_count": int(len(strategy_summary)),
        "fast_watchlist_count": int(len(best_rows)),
        "min_periods_for_best": int(min_periods_for_best),
        "best_fast_watchlist": best,
        "best_reported_strategy": best_reported,
        "best_limitup_strategy": practical_limitup.iloc[0].to_dict() if not practical_limitup.empty else {},
        "best_generalized_strategy": practical_gen.iloc[0].to_dict() if not practical_gen.empty else {},
        "decision": "personal_short_event_fast_diagnostic",
        "strategy_candidate_count": 0,
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
    }


def render_markdown(
    summary: Mapping[str, Any],
    *,
    strategy_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    strategy_yearly: pd.DataFrame,
    best_rows: pd.DataFrame,
    best_trades: pd.DataFrame,
) -> str:
    best_id = str(summary.get("best_fast_watchlist", {}).get("strategy_id", ""))
    best_yearly = strategy_yearly.loc[strategy_yearly["strategy_id"].eq(best_id)].copy() if not strategy_yearly.empty and best_id else pd.DataFrame()
    best_stress = stress_summary.loc[stress_summary["strategy_id"].eq(best_id)].copy() if not stress_summary.empty and best_id else pd.DataFrame()
    lines = [
        "# Personal Short-Event Fast Research",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- all_limitup_prediction_count: `{summary.get('all_limitup_prediction_count')}`",
        f"- generalized_prediction_count: `{summary.get('generalized_prediction_count')}`",
        f"- strategy_count: `{summary.get('strategy_count')}`",
        f"- fast_watchlist_count: `{summary.get('fast_watchlist_count')}`",
        f"- best_fast_watchlist_id: `{best_id}`",
        f"- decision: `{summary.get('decision')}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count')}`",
        "",
        "## Fast Watchlist",
        "",
        _markdown_table(
            best_rows,
            [
                "strategy_id",
                "family",
                "pool_name",
                "gate_name",
                "score_col",
                "period_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "selection_rate",
                "combined_light_mean_period_net_ret_pct",
                "combined_light_min_year_period_ret_pct",
            ],
        ),
        "",
        "## Top Reported Strategies",
        "",
        _markdown_table(
            strategy_summary.head(40),
            [
                "strategy_id",
                "family",
                "pool_name",
                "gate_name",
                "score_col",
                "score_floor",
                "score_quantile",
                "trade_count",
                "period_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "selection_rate",
                "mean_risk_flag_count",
                "mean_big_loss_prob",
            ],
        ),
        "",
        "## Best Stress",
        "",
        _markdown_table(best_stress),
        "",
        "## Best Yearly",
        "",
        _markdown_table(best_yearly),
        "",
        "## Best Trades Sample",
        "",
        _markdown_table(
            best_trades.head(40),
            [
                "date",
                "entry_date",
                "code",
                "name_on_date",
                "family",
                "pool_name",
                "gate_name",
                "score_col",
                "ml_score",
                "risk_score_balanced_soft",
                "predicted_ret_pct",
                "predicted_big_loss_prob",
                "risk_flag_count",
                "primary_event_type",
                "target_net_ret_pct",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a fast personal-capital diagnostic over existing predictions, not a formal strategy promotion gate.",
        "- Refusal trading means no trade is taken when the score/risk/event gate has no candidate.",
        "- Prior score-quantile gates use only earlier prediction years; the first eval year has no quantile threshold.",
        "- Generalized non-limit branches reuse the existing generalized strong-event model; they are not yet separately retrained branch models.",
    ]
    return "\n".join(lines) + "\n"


def fast_strategy_id(
    *,
    family: str,
    pool_name: str,
    gate_name: str,
    score_col: str,
    score_floor: float,
    score_quantile: float,
    max_positions: int,
) -> str:
    floor = "none" if not np.isfinite(float(score_floor)) else str(float(score_floor)).replace("-", "m").replace(".", "p")
    quantile = "none" if not np.isfinite(float(score_quantile)) else f"q{int(round(float(score_quantile) * 100))}"
    score = score_col.replace("risk_score_", "rs_").replace("_", "")
    return f"{family}__{pool_name}__{gate_name}__{score}__floor_{floor}__{quantile}__pos{int(max_positions)}"


def reduce_memory(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column in {"date", "entry_date", "exit_date"}:
            continue
        if pd.api.types.is_bool_dtype(output[column]):
            output[column] = output[column].astype(bool)
        elif pd.api.types.is_integer_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], downcast="integer")
        elif pd.api.types.is_float_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], downcast="float")
    return output


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def parse_float_values(values: Sequence[float] | str) -> tuple[float, ...]:
    if isinstance(values, str):
        parsed: list[float] = []
        for value in values.split(","):
            value = value.strip()
            if not value:
                continue
            parsed.append(-np.inf if value.lower() in {"none", "-inf"} else float(value))
        return tuple(parsed)
    return tuple(float(value) for value in values)


def parse_int_values(values: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        parsed = tuple(int(value.strip()) for value in values.split(",") if value.strip())
    else:
        parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError("integer values must not be empty")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-run-dir", default=None)
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--generalized-run-dir", default=None)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--target-window", type=int, default=DEFAULT_TARGET_WINDOW)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--eval-years", default=",".join(str(year) for year in DEFAULT_EVAL_YEARS))
    parser.add_argument("--score-floors", default=",".join("none" if not np.isfinite(value) else str(value) for value in DEFAULT_SCORE_FLOORS))
    parser.add_argument("--score-quantiles", default=",".join("none" if not np.isfinite(value) else str(value) for value in DEFAULT_SCORE_QUANTILES))
    parser.add_argument("--max-positions", default=",".join(str(value) for value in DEFAULT_MAX_POSITIONS))
    parser.add_argument("--min-periods-for-best", type=int, default=DEFAULT_MIN_PERIODS_FOR_BEST)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_personal_short_event_fast_research(
        ml_run_dir=args.ml_run_dir,
        event_file=args.event_file,
        generalized_run_dir=args.generalized_run_dir,
        profile=args.profile,
        target_window=args.target_window,
        fee_bps=args.fee_bps,
        eval_years=parse_int_values(args.eval_years),
        score_floors=parse_float_values(args.score_floors),
        score_quantiles=parse_float_values(args.score_quantiles),
        max_positions_values=parse_int_values(args.max_positions),
        min_periods_for_best=args.min_periods_for_best,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
        min_available_memory_gb=args.min_available_memory_gb,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
