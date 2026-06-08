"""Risk-penalized strategy diagnostics for the all-limit-up ML ranker."""

from __future__ import annotations

import argparse
import gc
import json
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
from traditional_quant_research.experiments.short_open_known_factor_rebuild import (
    audit_buy_feature_columns,
    feature_columns_for_profile,
)
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    _assert_memory_available,
    available_memory_gb,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/all_limitup_risk_penalized_strategy_research")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-07_all_limitup_risk_penalized_strategy_research.md")
DEFAULT_MIN_SCORE_VALUES = (-np.inf, -0.5, 0.0, 0.5, 1.0)
DEFAULT_MAX_ORIGINAL_RANK_VALUES: tuple[int | None, ...] = (1, 3, None)
DEFAULT_MAX_POSITIONS = (1, 2)
RISK_FLAG_COLUMNS = (
    "risk_weak_market",
    "risk_weak_industry",
    "risk_overextended",
    "risk_high_volatility",
    "risk_advanced_board",
    "risk_no_kama_atr_confirmation",
    "risk_high_open_gap",
    "risk_low_liquidity",
)
PENALTY_SPECS: tuple[dict[str, Any], ...] = (
    {"penalty_name": "base_original", "weights": {}, "big_loss_prob_weight": 0.0},
    {
        "penalty_name": "confirmation_soft",
        "weights": {"risk_no_kama_atr_confirmation": 0.20, "risk_high_open_gap": 0.35},
        "big_loss_prob_weight": 0.50,
    },
    {
        "penalty_name": "regime_soft",
        "weights": {"risk_weak_market": 0.45, "risk_weak_industry": 0.45, "risk_no_kama_atr_confirmation": 0.15},
        "big_loss_prob_weight": 0.75,
    },
    {
        "penalty_name": "tail_soft",
        "weights": {
            "risk_overextended": 0.45,
            "risk_high_volatility": 0.45,
            "risk_advanced_board": 0.35,
            "risk_no_kama_atr_confirmation": 0.15,
        },
        "big_loss_prob_weight": 1.50,
    },
    {
        "penalty_name": "balanced_soft",
        "weights": {
            "risk_weak_market": 0.35,
            "risk_weak_industry": 0.35,
            "risk_overextended": 0.35,
            "risk_high_volatility": 0.35,
            "risk_advanced_board": 0.25,
            "risk_no_kama_atr_confirmation": 0.20,
            "risk_high_open_gap": 0.45,
            "risk_low_liquidity": 0.25,
        },
        "big_loss_prob_weight": 1.50,
    },
    {
        "penalty_name": "balanced_strict",
        "weights": {
            "risk_weak_market": 0.70,
            "risk_weak_industry": 0.60,
            "risk_overextended": 0.60,
            "risk_high_volatility": 0.60,
            "risk_advanced_board": 0.50,
            "risk_no_kama_atr_confirmation": 0.35,
            "risk_high_open_gap": 0.70,
            "risk_low_liquidity": 0.40,
        },
        "big_loss_prob_weight": 3.00,
    },
)
STRESS_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"scenario": "reported_30bps"},
    {"scenario": "fee_100bps", "extra_fee_bps": 70.0},
    {"scenario": "tail_x1_5", "tail_quantile": 0.10, "tail_loss_multiplier": 1.50},
    {
        "scenario": "combined_harsh",
        "extra_fee_bps": 70.0,
        "gap_haircuts": ((7.0, 100.0), (5.0, 50.0)),
        "low_liquidity_haircut_bps": 75.0,
        "liquidity_quantile": 0.20,
        "tail_quantile": 0.10,
        "tail_loss_multiplier": 1.25,
    },
)


def run_all_limitup_risk_penalized_strategy_research(
    *,
    ml_run_dir: str | Path | None = None,
    event_file: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    target_window: int = DEFAULT_TARGET_WINDOW,
    fee_bps: float = DEFAULT_FEE_BPS,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    min_score_values: Sequence[float] = DEFAULT_MIN_SCORE_VALUES,
    max_original_rank_values: Sequence[int | None] = DEFAULT_MAX_ORIGINAL_RANK_VALUES,
    max_positions_values: Sequence[int] = DEFAULT_MAX_POSITIONS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
) -> dict[str, Any]:
    """Run risk-penalty, threshold, and fallback diagnostics over saved ML predictions."""

    run_id = f"all_limitup_risk_penalized_strategy_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_ml_run_dir = Path(ml_run_dir) if ml_run_dir is not None else latest_run_dir(DEFAULT_ML_RUN_ROOT)
    source_summary = read_source_summary(source_ml_run_dir)
    source_event_file = Path(event_file) if event_file is not None else source_event_path(source_summary)
    if not source_event_file.exists():
        source_event_file = find_latest_event_panel()

    feature_columns = tuple(feature_columns_for_profile(profile))
    audit_buy_feature_columns(feature_columns)
    selected_eval_years = tuple(int(year) for year in eval_years)
    _assert_memory_available(min_available_memory_gb, context="before reading risk-penalty inputs")
    predictions = read_predictions(source_ml_run_dir / "ml_predictions.csv")
    event_panel = read_followup_event_panel(
        source_event_file,
        feature_columns=feature_columns,
        label_windows=LABEL_WINDOWS,
        fee_bps=fee_bps,
    )
    enriched = enrich_predictions(predictions, event_panel, feature_columns=feature_columns)
    enriched = enriched.loc[enriched["eval_year"].isin(selected_eval_years)].copy()
    enriched = build_risk_flags(enriched, event_panel, eval_years=selected_eval_years)
    _assert_memory_available(min_available_memory_gb, context="after building risk flags")

    scored = build_penalty_scores(enriched, PENALTY_SPECS)
    risk_flag_summary = summarize_risk_flags(scored)
    strategy_summary, strategy_yearly, strategy_stress, selected_trades = evaluate_risk_penalty_grid(
        scored,
        penalty_specs=PENALTY_SPECS,
        min_score_values=min_score_values,
        max_original_rank_values=max_original_rank_values,
        max_positions_values=max_positions_values,
    )
    summary = build_summary(
        run_id=run_id,
        run_dir=run_dir,
        source_ml_run_dir=source_ml_run_dir,
        source_event_file=source_event_file,
        source_summary=source_summary,
        scored=scored,
        strategy_summary=strategy_summary,
        strategy_stress=strategy_stress,
        profile=profile,
        fee_bps=fee_bps,
        target_window=target_window,
        min_available_memory_gb=min_available_memory_gb,
    )
    best_strategy_id = str(summary.get("best_reported_strategy", {}).get("strategy_id", ""))
    best_trades = selected_trades.loc[selected_trades["strategy_id"].eq(best_strategy_id)].copy() if best_strategy_id else pd.DataFrame()
    markdown = render_markdown(
        summary,
        risk_flag_summary=risk_flag_summary,
        strategy_summary=strategy_summary,
        strategy_stress=strategy_stress,
        strategy_yearly=strategy_yearly,
        best_trades=best_trades,
    )

    risk_flag_summary.to_csv(run_dir / "risk_flag_summary.csv", index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(run_dir / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    strategy_yearly.to_csv(run_dir / "strategy_yearly.csv", index=False, encoding="utf-8-sig")
    strategy_stress.to_csv(run_dir / "strategy_stress_summary.csv", index=False, encoding="utf-8-sig")
    selected_trades.to_csv(run_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    best_trades.to_csv(run_dir / "best_strategy_trades.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    del event_panel, enriched, scored
    gc.collect()
    _assert_memory_available(min_available_memory_gb, context="after writing risk-penalty outputs")
    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path else None}


def build_risk_flags(frame: pd.DataFrame, reference: pd.DataFrame, *, eval_years: Sequence[int]) -> pd.DataFrame:
    """Attach risk flags using prior-year reference quantiles for each eval year."""

    output = frame.copy()
    thresholds = build_prior_year_thresholds(reference, eval_years=eval_years)
    for column in RISK_FLAG_COLUMNS:
        output[column] = False
    output["risk_flag_count"] = 0
    for eval_year, subset_index in output.groupby("eval_year").groups.items():
        year = int(eval_year)
        row = thresholds.get(year, {})
        idx = list(subset_index)
        part = output.loc[idx]
        weak_market = pd.to_numeric(part.get("market_breadth_5d"), errors="coerce").le(row.get("market_breadth_5d_q25", np.nan)) | pd.to_numeric(
            part.get("market_limitup_rate"), errors="coerce"
        ).le(row.get("market_limitup_rate_q25", np.nan))
        weak_industry = pd.to_numeric(part.get("industry_ret5_mean"), errors="coerce").le(row.get("industry_ret5_mean_q25", np.nan)) | pd.to_numeric(
            part.get("industry_limitup_rate"), errors="coerce"
        ).le(row.get("industry_limitup_rate_q25", np.nan))
        overextended = pd.to_numeric(part.get("signal_ret20_before_pct"), errors="coerce").ge(row.get("signal_ret20_before_pct_q75", np.nan)) | pd.to_numeric(
            part.get("price_position_60d"), errors="coerce"
        ).ge(row.get("price_position_60d_q75", np.nan))
        high_vol = pd.to_numeric(part.get("volatility_20_pct"), errors="coerce").ge(row.get("volatility_20_pct_q75", np.nan))
        advanced_board = pd.to_numeric(part.get("limit_up_run_ending_today"), errors="coerce").ge(3.0)
        no_confirmation = ~(_truthy_col(part, "open_below_kama_break_limitup") & _truthy_col(part, "close_cross_atr_upper"))
        high_open_gap = pd.to_numeric(part.get("next_open_gap_pct"), errors="coerce").ge(5.0)
        amount = pd.to_numeric(part.get("signal_amount_log10"), errors="coerce")
        low_liquidity = amount.le(row.get("signal_amount_log10_q25", np.nan))
        if "liquid_amount_ok" in part.columns:
            low_liquidity |= ~_truthy_col(part, "liquid_amount_ok")
        assignments = {
            "risk_weak_market": weak_market,
            "risk_weak_industry": weak_industry,
            "risk_overextended": overextended,
            "risk_high_volatility": high_vol,
            "risk_advanced_board": advanced_board,
            "risk_no_kama_atr_confirmation": no_confirmation,
            "risk_high_open_gap": high_open_gap,
            "risk_low_liquidity": low_liquidity,
        }
        for column, values in assignments.items():
            output.loc[idx, column] = values.fillna(False).to_numpy(dtype=bool)
    output["risk_flag_count"] = output.loc[:, list(RISK_FLAG_COLUMNS)].sum(axis=1).astype("int16")
    return output


def build_prior_year_thresholds(reference: pd.DataFrame, *, eval_years: Sequence[int]) -> dict[int, dict[str, float]]:
    thresholds: dict[int, dict[str, float]] = {}
    year_col = pd.to_numeric(reference["year"], errors="coerce")
    for eval_year in sorted(int(year) for year in eval_years):
        history = reference.loc[year_col.lt(eval_year)].copy()
        if history.empty:
            history = reference.copy()
        thresholds[eval_year] = {}
        for column in (
            "market_breadth_5d",
            "market_limitup_rate",
            "industry_ret5_mean",
            "industry_limitup_rate",
            "signal_ret20_before_pct",
            "price_position_60d",
            "volatility_20_pct",
            "signal_amount_log10",
        ):
            values = pd.to_numeric(history.get(column), errors="coerce")
            thresholds[eval_year][f"{column}_q25"] = float(values.quantile(0.25)) if values.notna().any() else np.nan
            thresholds[eval_year][f"{column}_q75"] = float(values.quantile(0.75)) if values.notna().any() else np.nan
    return thresholds


def build_penalty_scores(frame: pd.DataFrame, penalty_specs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    output = frame.copy()
    base_score = pd.to_numeric(output["ml_score"], errors="coerce").astype(float)
    big_loss_prob = pd.to_numeric(output.get("predicted_big_loss_prob"), errors="coerce").fillna(0.0).astype(float)
    for spec in penalty_specs:
        penalty = pd.Series(0.0, index=output.index)
        for flag, weight in dict(spec.get("weights", {})).items():
            penalty += _truthy_col(output, flag).astype(float) * float(weight)
        penalty += big_loss_prob * float(spec.get("big_loss_prob_weight", 0.0))
        name = str(spec["penalty_name"])
        output[f"risk_penalty_{name}"] = penalty
        output[f"risk_score_{name}"] = base_score - penalty
    return output


def summarize_risk_flags(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flag in RISK_FLAG_COLUMNS:
        mask = _truthy_col(frame, flag)
        subset = frame.loc[mask].copy()
        returns = pd.to_numeric(subset.get("target_net_ret_pct"), errors="coerce").dropna()
        rows.append(
            {
                "risk_flag": flag,
                "flagged_count": int(mask.sum()),
                "flagged_rate": float(mask.mean()) if len(mask) else np.nan,
                "mean_net_ret_pct": _mean(returns),
                "median_net_ret_pct": _median(returns),
                "win_rate": _rate(returns > 0),
                "big_loss_rate": _rate(returns <= -5.0),
                "mean_ml_score": _mean(pd.to_numeric(subset.get("ml_score"), errors="coerce")),
            }
        )
    count = pd.to_numeric(frame["risk_flag_count"], errors="coerce")
    for bucket, mask in {
        "risk_count_0_1": count.le(1),
        "risk_count_2_3": count.between(2, 3, inclusive="both"),
        "risk_count_4plus": count.ge(4),
    }.items():
        subset = frame.loc[mask].copy()
        returns = pd.to_numeric(subset.get("target_net_ret_pct"), errors="coerce").dropna()
        rows.append(
            {
                "risk_flag": bucket,
                "flagged_count": int(mask.sum()),
                "flagged_rate": float(mask.mean()) if len(mask) else np.nan,
                "mean_net_ret_pct": _mean(returns),
                "median_net_ret_pct": _median(returns),
                "win_rate": _rate(returns > 0),
                "big_loss_rate": _rate(returns <= -5.0),
                "mean_ml_score": _mean(pd.to_numeric(subset.get("ml_score"), errors="coerce")),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_net_ret_pct", "flagged_count"], ascending=[True, False], na_position="last").reset_index(drop=True)


def evaluate_risk_penalty_grid(
    frame: pd.DataFrame,
    *,
    penalty_specs: Sequence[Mapping[str, Any]],
    min_score_values: Sequence[float],
    max_original_rank_values: Sequence[int | None],
    max_positions_values: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = frame.loc[_truthy_col(frame, "executable_entry") & ~_truthy_col(frame, "entry_open_near_limit")].copy()
    eligible_days = int(pd.to_datetime(base["entry_date"]).nunique())
    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    stress_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for spec in penalty_specs:
        penalty_name = str(spec["penalty_name"])
        score_col = f"risk_score_{penalty_name}"
        for min_score in min_score_values:
            for max_original_rank in max_original_rank_values:
                for max_positions in max_positions_values:
                    selected = select_risk_adjusted_positions(
                        base,
                        score_col=score_col,
                        min_score=float(min_score),
                        max_original_rank=max_original_rank,
                        max_positions=int(max_positions),
                    )
                    strategy_id = strategy_id_for(
                        penalty_name=penalty_name,
                        min_score=float(min_score),
                        max_original_rank=max_original_rank,
                        max_positions=int(max_positions),
                    )
                    summary = summarize_portfolio(selected, pool_name="executable_only", max_positions=int(max_positions))
                    summary.update(
                        {
                            "strategy_id": strategy_id,
                            "penalty_name": penalty_name,
                            "min_score": float(min_score),
                            "max_original_rank": "all" if max_original_rank is None else int(max_original_rank),
                            "eligible_entry_days": eligible_days,
                            "selection_rate": float(summary["period_count"] / eligible_days) if eligible_days else np.nan,
                            "mean_risk_flag_count": _mean(pd.to_numeric(selected.get("risk_flag_count"), errors="coerce")) if not selected.empty else np.nan,
                            "mean_selected_risk_score": _mean(pd.to_numeric(selected.get(score_col), errors="coerce")) if not selected.empty else np.nan,
                        }
                    )
                    summary_rows.append(summary)
                    if not selected.empty:
                        selected = selected.assign(
                            strategy_id=strategy_id,
                            penalty_name=penalty_name,
                            min_score=float(min_score),
                            max_original_rank="all" if max_original_rank is None else int(max_original_rank),
                        )
                        selected_frames.append(selected)
                        yearly = summarize_portfolio_yearly(selected, strategy_id=strategy_id)
                        yearly["penalty_name"] = penalty_name
                        yearly["min_score"] = float(min_score)
                        yearly["max_original_rank"] = "all" if max_original_rank is None else int(max_original_rank)
                        yearly_frames.append(yearly)
                    for scenario in STRESS_SCENARIOS:
                        stressed = stress_selected(selected, scenario)
                        stress = summarize_portfolio(stressed, pool_name="executable_only", max_positions=int(max_positions))
                        stress.update(
                            {
                                "strategy_id": strategy_id,
                                "scenario": str(scenario["scenario"]),
                                "penalty_name": penalty_name,
                                "min_score": float(min_score),
                                "max_original_rank": "all" if max_original_rank is None else int(max_original_rank),
                                "eligible_entry_days": eligible_days,
                                "selection_rate": float(stress["period_count"] / eligible_days) if eligible_days else np.nan,
                            }
                        )
                        stress_rows.append(stress)
    strategy = pd.DataFrame(summary_rows).sort_values(
        ["mean_period_net_ret_pct", "positive_year_rate", "min_year_period_ret_pct", "selection_rate"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    stress = pd.DataFrame(stress_rows).sort_values(
        ["scenario", "mean_period_net_ret_pct", "positive_year_rate", "selection_rate"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return strategy, yearly, stress, selected_all


def select_risk_adjusted_positions(
    frame: pd.DataFrame,
    *,
    score_col: str,
    min_score: float,
    max_original_rank: int | None,
    max_positions: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    clean = frame.dropna(subset=["entry_date", "exit_date", "ml_score", score_col, "target_net_ret_pct"]).copy()
    if clean.empty:
        return clean
    clean["entry_date"] = pd.to_datetime(clean["entry_date"])
    clean["exit_date"] = pd.to_datetime(clean["exit_date"])
    clean = clean.sort_values(["entry_date", "ml_score"], ascending=[True, False])
    clean["original_rank"] = clean.groupby("entry_date")["ml_score"].rank(method="first", ascending=False)
    if max_original_rank is not None:
        clean = clean.loc[clean["original_rank"].le(int(max_original_rank))].copy()
    if np.isfinite(float(min_score)):
        clean = clean.loc[pd.to_numeric(clean[score_col], errors="coerce").ge(float(min_score))].copy()
    if clean.empty:
        return clean
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
    selected = clean.loc[selected_indices].copy()
    selected["risk_score"] = pd.to_numeric(selected[score_col], errors="coerce")
    selected["pool_name"] = "executable_only"
    selected["max_positions"] = int(max_positions)
    return selected.reset_index(drop=True)


def stress_selected(selected: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    stressed, _ = apply_stress_adjustments(selected, scenario)
    return stressed


def strategy_id_for(*, penalty_name: str, min_score: float, max_original_rank: int | None, max_positions: int) -> str:
    threshold = "none" if not np.isfinite(float(min_score)) else str(float(min_score)).replace("-", "m").replace(".", "p")
    rank = "all" if max_original_rank is None else f"scan{int(max_original_rank)}"
    return f"{penalty_name}__{rank}__score_{threshold}__pos{int(max_positions)}"


def build_summary(
    *,
    run_id: str,
    run_dir: Path,
    source_ml_run_dir: Path,
    source_event_file: Path,
    source_summary: Mapping[str, Any],
    scored: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    strategy_stress: pd.DataFrame,
    profile: str,
    fee_bps: float,
    target_window: int,
    min_available_memory_gb: float,
) -> dict[str, Any]:
    reported = strategy_stress.loc[strategy_stress["scenario"].eq("reported_30bps")].copy()
    harsh = strategy_stress.loc[strategy_stress["scenario"].eq("combined_harsh")].copy()
    eligible_reported = reported.loc[reported["period_count"].ge(30)].copy()
    eligible_harsh = harsh.loc[harsh["period_count"].ge(30)].copy()
    best_reported = eligible_reported.iloc[0].to_dict() if not eligible_reported.empty else {}
    best_harsh = eligible_harsh.iloc[0].to_dict() if not eligible_harsh.empty else {}
    baseline_mask = (
        strategy_summary["penalty_name"].eq("base_original")
        & strategy_summary["max_positions"].eq(1)
        & strategy_summary["max_original_rank"].astype(str).eq("all")
        & np.isneginf(pd.to_numeric(strategy_summary["min_score"], errors="coerce"))
    )
    baseline = strategy_summary.loc[baseline_mask].iloc[0].to_dict() if baseline_mask.any() else {}
    candidate_count = int(
        (
            eligible_harsh["mean_period_net_ret_pct"].gt(0)
            & eligible_harsh["positive_year_rate"].ge(0.75)
            & eligible_harsh["selection_rate"].ge(0.25)
        ).sum()
    ) if not eligible_harsh.empty else 0
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_ml_run_dir": str(source_ml_run_dir),
        "source_event_file": str(source_event_file),
        "source_ml_run_id": source_summary.get("run_id"),
        "profile": profile,
        "fee_bps": float(fee_bps),
        "target_window": int(target_window),
        "prediction_count": int(len(scored)),
        "eligible_executable_count": int((_truthy_col(scored, "executable_entry") & ~_truthy_col(scored, "entry_open_near_limit")).sum()),
        "baseline_original_strategy": baseline,
        "best_reported_strategy": best_reported,
        "best_combined_harsh_strategy": best_harsh,
        "decision": "all_limitup_risk_penalized_diagnostic",
        "strategy_candidate_count": candidate_count,
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
    }


def render_markdown(
    summary: Mapping[str, Any],
    *,
    risk_flag_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    strategy_stress: pd.DataFrame,
    strategy_yearly: pd.DataFrame,
    best_trades: pd.DataFrame,
) -> str:
    baseline = summary.get("baseline_original_strategy", {}) or {}
    best = summary.get("best_reported_strategy", {}) or {}
    harsh = summary.get("best_combined_harsh_strategy", {}) or {}
    best_id = str(best.get("strategy_id", ""))
    best_yearly = strategy_yearly.loc[strategy_yearly["strategy_id"].eq(best_id)].copy() if not strategy_yearly.empty and best_id else pd.DataFrame()
    reported = strategy_stress.loc[strategy_stress["scenario"].eq("reported_30bps")].head(20).copy()
    harsh_table = strategy_stress.loc[strategy_stress["scenario"].eq("combined_harsh")].head(20).copy()
    lines = [
        "# All-Limit-Up Risk-Penalized Strategy Research",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- source_ml_run_dir: `{summary.get('source_ml_run_dir')}`",
        f"- source_event_file: `{summary.get('source_event_file')}`",
        f"- prediction_count: `{summary.get('prediction_count')}`",
        f"- eligible_executable_count: `{summary.get('eligible_executable_count')}`",
        f"- baseline_mean_pct: `{baseline.get('mean_period_net_ret_pct')}`",
        f"- best_reported_strategy_id: `{best.get('strategy_id')}`",
        f"- best_reported_mean_pct: `{best.get('mean_period_net_ret_pct')}`",
        f"- best_harsh_strategy_id: `{harsh.get('strategy_id')}`",
        f"- best_harsh_mean_pct: `{harsh.get('mean_period_net_ret_pct')}`",
        f"- decision: `{summary.get('decision')}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count')}`",
        "",
        "## Risk Flag Summary",
        "",
        _markdown_table(risk_flag_summary),
        "",
        "## Reported Ranking",
        "",
        _markdown_table(
            reported,
            [
                "strategy_id",
                "trade_count",
                "period_count",
                "selection_rate",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "mean_risk_flag_count",
                "mean_selected_risk_score",
            ],
        ),
        "",
        "## Combined Harsh Ranking",
        "",
        _markdown_table(
            harsh_table,
            [
                "strategy_id",
                "trade_count",
                "period_count",
                "selection_rate",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
            ],
        ),
        "",
        "## Base Strategy Grid",
        "",
        _markdown_table(
            strategy_summary.head(20),
            [
                "strategy_id",
                "trade_count",
                "period_count",
                "selection_rate",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "mean_risk_flag_count",
            ],
        ),
        "",
        "## Best Reported Yearly",
        "",
        _markdown_table(best_yearly),
        "",
        "## Best Trades Sample",
        "",
        _markdown_table(
            best_trades.head(30),
            [
                "date",
                "entry_date",
                "code",
                "name_on_date",
                "target_net_ret_pct",
                "ml_score",
                "risk_score",
                "risk_flag_count",
                "predicted_ret_pct",
                "predicted_big_loss_prob",
                "next_open_gap_pct",
                "market_breadth_5d",
                "industry_ret5_mean",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a diagnostic grid over saved open-known ML predictions, not an execution recommendation.",
        "- Risk thresholds use only prior-year event-panel quantiles for each eval year.",
        "- The current profile remains valid only after the next open print is known.",
        "- Five-minute data is still needed for real fill, slippage, and intraday confirmation modeling.",
    ]
    return "\n".join(lines) + "\n"


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _rate(values: pd.Series) -> float:
    clean = pd.Series(values).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def parse_float_values(values: Sequence[float] | str) -> tuple[float, ...]:
    if isinstance(values, str):
        parsed: list[float] = []
        for item in values.split(","):
            raw = item.strip()
            if not raw:
                continue
            parsed.append(-np.inf if raw.lower() in {"none", "-inf", "all"} else float(raw))
        return tuple(parsed)
    return tuple(float(item) for item in values)


def parse_rank_values(values: Sequence[int | None] | str) -> tuple[int | None, ...]:
    if isinstance(values, str):
        parsed: list[int | None] = []
        for item in values.split(","):
            raw = item.strip().lower()
            if not raw:
                continue
            parsed.append(None if raw in {"all", "none"} else int(raw))
        return tuple(parsed)
    return tuple(None if item is None else int(item) for item in values)


def parse_int_values(values: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        return tuple(int(item.strip()) for item in values.split(",") if item.strip())
    return tuple(int(item) for item in values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-run-dir", default=None)
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--target-window", type=int, default=DEFAULT_TARGET_WINDOW)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--eval-years", default=",".join(str(year) for year in DEFAULT_EVAL_YEARS))
    parser.add_argument("--min-score-values", default="-inf,-0.5,0,0.5,1")
    parser.add_argument("--max-original-rank-values", default="1,3,all")
    parser.add_argument("--max-positions-values", default="1,2")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_all_limitup_risk_penalized_strategy_research(
        ml_run_dir=args.ml_run_dir,
        event_file=args.event_file,
        profile=args.profile,
        target_window=args.target_window,
        fee_bps=args.fee_bps,
        eval_years=parse_int_values(args.eval_years),
        min_score_values=parse_float_values(args.min_score_values),
        max_original_rank_values=parse_rank_values(args.max_original_rank_values),
        max_positions_values=parse_int_values(args.max_positions_values),
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
        min_available_memory_gb=args.min_available_memory_gb,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
