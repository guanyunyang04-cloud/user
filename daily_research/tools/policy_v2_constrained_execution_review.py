from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.advanced_ml_runtime import HistoryWindow, load_raw_data_with_cache
from daily_research.baseline.backtest import backtest, summarize_backtest_by_month, summarize_monthly_diagnostics
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import split_benchmark_from_universe
from daily_research.baseline.external_target_weight_bridge import apply_rebalance_schedule, build_target_weight_bridge
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.deep_alpha import execution_alignment as exec_align


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_ROOT_TAG = "short_alpha_policy_v2_constrained_execution_review_20260410_r1"
DEFAULT_POLICY_V2_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_policy_v2_review_20260410_r1" / "runs" / "short_expert_policy_v2"
)
DEFAULT_CURRENT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    profile_name: str
    candidate_cap: int = 0
    gross_floor: float = 0.0
    gross_cap: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate policy_v2 raw learned target weights under a constrained set of execution "
            "profiles and pre-bridge sparsity/gross transforms on the latest formal window."
        )
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_POLICY_V2_RUN_DIR))
    parser.add_argument("--current-formal-run-dir", default=str(DEFAULT_CURRENT_FORMAL_RUN_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_long_panel(path: Path, value_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Empty panel: {path}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["stock"] = raw["stock"].astype(str).str.upper().str.strip()
    raw[value_name] = pd.to_numeric(raw[value_name], errors="coerce")
    raw = raw.dropna(subset=["date", "stock", value_name])
    return (
        raw.sort_values(["date", "stock"])
        .drop_duplicates(subset=["date", "stock"], keep="last")
        .pivot(index="date", columns="stock", values=value_name)
        .sort_index()
        .fillna(0.0)
    )


def _monthly_robust_score(monthly_diag: dict[str, Any]) -> float:
    positive_ratio = float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0)
    median_monthly_return = float(monthly_diag.get("median_monthly_return", 0.0) or 0.0)
    mean_monthly_return = float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0)
    worst_monthly_return = float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0)
    top3_positive_share = float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0)
    longest_negative_streak = int(monthly_diag.get("longest_negative_streak", 0) or 0)
    downside_penalty = max(-worst_monthly_return, 0.0)
    concentration_penalty = max(top3_positive_share - 0.60, 0.0)
    streak_penalty = max(longest_negative_streak - 2, 0)
    return float(
        mean_monthly_return
        + median_monthly_return
        + 0.05 * (positive_ratio - 0.50)
        - 0.35 * downside_penalty
        - 0.05 * concentration_penalty
        - 0.01 * float(streak_penalty)
    )


def _cap_candidate_count(frame: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    if int(max_candidates or 0) <= 0:
        return frame.copy()
    rows: list[pd.Series] = []
    for dt, row in frame.iterrows():
        clean = row.astype(float).clip(lower=0.0)
        gross = float(clean.sum())
        positive = clean[clean > 0.0].sort_values(ascending=False).head(int(max_candidates))
        aligned = pd.Series(0.0, index=frame.columns, name=dt, dtype=float)
        if gross > 0.0 and not positive.empty and float(positive.sum()) > 0.0:
            aligned.loc[positive.index] = positive.values / float(positive.sum()) * gross
        rows.append(aligned)
    return pd.DataFrame(rows, index=frame.index, columns=frame.columns).fillna(0.0)


def _clip_gross_band(frame: pd.DataFrame, gross_floor: float, gross_cap: float) -> pd.DataFrame:
    if gross_floor <= 0.0 and gross_cap <= 0.0:
        return frame.copy()
    rows: list[pd.Series] = []
    for dt, row in frame.iterrows():
        clean = row.astype(float).clip(lower=0.0)
        gross = float(clean.sum())
        aligned = pd.Series(0.0, index=frame.columns, name=dt, dtype=float)
        if gross <= 0.0:
            rows.append(aligned)
            continue
        target_gross = gross
        if gross_floor > 0.0 and target_gross < gross_floor:
            target_gross = float(gross_floor)
        if gross_cap > 0.0 and target_gross > gross_cap:
            target_gross = float(gross_cap)
        aligned.loc[clean.index] = clean.values / gross * target_gross
        rows.append(aligned)
    return pd.DataFrame(rows, index=frame.index, columns=frame.columns).fillna(0.0)


def _panel_stats(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    gross = clean.sum(axis=1)
    counts = (clean > 1e-12).sum(axis=1)
    top1 = clean.max(axis=1)
    top2_share_rows = []
    hhi_rows = []
    for _, row in clean.iterrows():
        total = float(row.sum())
        if total <= 0.0:
            top2_share_rows.append(0.0)
            hhi_rows.append(0.0)
            continue
        weights = row[row > 0.0].sort_values(ascending=False)
        top2_share_rows.append(float(weights.head(2).sum() / total))
        normalized = weights / total
        hhi_rows.append(float((normalized.pow(2)).sum()))
    return {
        "avg_gross_exposure": float(gross.mean()),
        "median_gross_exposure": float(gross.median()),
        "avg_positive_count": float(counts.mean()),
        "median_positive_count": float(counts.median()),
        "avg_top1_weight": float(top1.mean()),
        "avg_top2_share": float(pd.Series(top2_share_rows).mean()),
        "avg_hhi": float(pd.Series(hhi_rows).mean()),
    }


def _build_exec_cfg(
    *,
    benchmark: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    holding_count: int,
    max_weight: float,
    min_adv20: float,
    min_price: float,
    max_price: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    profile: exec_align.ExecutionAlignmentProfile,
) -> ResearchConfig:
    return ResearchConfig(
        start_date=str(period_start.date()).replace("-", ""),
        end_date=str(period_end.date()).replace("-", ""),
        benchmark=benchmark,
        execution_mode="next_open",
        holding_count=holding_count,
        weighting_method="score",
        rebalance_freq=profile.rebalance_freq,
        max_weight=max_weight,
        min_adv20=min_adv20,
        min_price=min_price,
        max_price=max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        enable_market_regime_filter=bool(profile.use_market_regime_filter),
        regime_ma_window=exec_align.EXECUTION_ALIGNMENT_REGIME_MA_WINDOW,
        regime_vol_window=exec_align.EXECUTION_ALIGNMENT_REGIME_VOL_WINDOW,
        regime_max_annual_vol=exec_align.EXECUTION_ALIGNMENT_REGIME_MAX_ANNUAL_VOL,
        regime_trend_flat_band=exec_align.EXECUTION_ALIGNMENT_REGIME_TREND_FLAT_BAND,
        regime_vol_transition_band=exec_align.EXECUTION_ALIGNMENT_REGIME_VOL_TRANSITION_BAND,
        regime_allowed_quadrants=list(exec_align.EXECUTION_ALIGNMENT_ALLOWED_QUADRANTS),
    )


def _evaluate_variant(
    *,
    raw_target_weights: pd.DataFrame,
    raw_score_frame: pd.DataFrame,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    benchmark: str,
    holding_count: int,
    max_weight: float,
    min_adv20: float,
    min_price: float,
    max_price: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    profile: exec_align.ExecutionAlignmentProfile,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    resolved_anchor_date = exec_align._resolve_window_anchor_date(raw_target_weights.index, profile.rebalance_anchor_date)
    aligned_target_weights, bridge_meta = build_target_weight_bridge(
        raw_target_weights,
        rebalance_freq=profile.rebalance_freq,
        rebalance_offset=0,
        rebalance_offset_mode=profile.rebalance_offset_mode,
        rebalance_anchor_date=resolved_anchor_date,
        top_k=profile.target_weight_top_k,
        min_weight=profile.target_weight_min_weight,
        power=profile.target_weight_power,
        full_invest=bool(profile.target_weight_full_invest),
    )
    aligned_scores, score_meta = apply_rebalance_schedule(
        raw_score_frame.fillna(0.0),
        rebalance_freq=profile.rebalance_freq,
        rebalance_offset=0,
        rebalance_offset_mode=profile.rebalance_offset_mode,
        rebalance_anchor_date=resolved_anchor_date,
    )
    period_index = aligned_target_weights.index
    cfg = _build_exec_cfg(
        benchmark=benchmark,
        period_start=pd.Timestamp(period_index.min()),
        period_end=pd.Timestamp(period_index.max()),
        holding_count=holding_count,
        max_weight=max_weight,
        min_adv20=min_adv20,
        min_price=min_price,
        max_price=max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        profile=profile,
    )
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    if cfg.enable_market_regime_filter:
        aligned_target_weights, aligned_scores = apply_market_regime_filter(
            target_weights=aligned_target_weights,
            target_scores=aligned_scores,
            regime_state=regime_state,
        )
    equity_df, action_df, metrics = backtest(
        close=close.reindex(period_index),
        benchmark_close=benchmark_close.reindex(period_index),
        target_weights=aligned_target_weights.reindex(period_index).fillna(0.0),
        target_scores=aligned_scores.reindex(period_index).fillna(0.0),
        config=cfg,
        regime_on=regime_state["regime_on"].reindex(period_index),
        open_df=open_df.reindex(period_index),
        benchmark_open=benchmark_open.reindex(period_index),
    )
    monthly_summary = summarize_backtest_by_month(equity_df, action_df)
    monthly_diag = summarize_monthly_diagnostics(monthly_summary, return_column="excess_return")
    metrics = dict(metrics)
    metrics.update(
        {
            "profile_name": profile.name,
            "profile_description": profile.description,
            "rebalance_freq": str(profile.rebalance_freq),
            "rebalance_offset_mode": str(bridge_meta.get("rebalance_offset_mode", profile.rebalance_offset_mode)),
            "rebalance_anchor_date": str(bridge_meta.get("rebalance_anchor_date", profile.rebalance_anchor_date)),
            "target_weight_top_k": int(bridge_meta.get("target_weight_top_k", profile.target_weight_top_k)),
            "target_weight_min_weight": float(bridge_meta.get("target_weight_min_weight", profile.target_weight_min_weight)),
            "target_weight_power": float(bridge_meta.get("target_weight_power", profile.target_weight_power)),
            "target_weight_full_invest": bool(bridge_meta.get("target_weight_full_invest", profile.target_weight_full_invest)),
            "market_regime_filter": bool(profile.use_market_regime_filter),
            "rebalance_sleeve_count": int(bridge_meta.get("rebalance_sleeve_count", 1)),
            "score_rebalance_offset_mode": str(score_meta.get("rebalance_offset_mode", profile.rebalance_offset_mode)),
        }
    )
    return metrics, monthly_summary, monthly_diag, _panel_stats(aligned_target_weights)


def _load_formal_market_data(
    *,
    stocks: list[str],
    start_date: str,
    end_date: str,
    benchmark: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    history_window = HistoryWindow(
        mode="infer",
        requested_start_date=start_date,
        effective_start_date=start_date,
        end_date=end_date,
        required_trading_days=0,
    )
    raw_df_dict, _ = load_raw_data_with_cache(
        data_source="tq",
        csv_folder=None,
        universe=stocks,
        benchmark=benchmark,
        history_window=history_window,
        use_cache=True,
        refresh_cache=False,
        progress_desc="Load formal replay market data",
    )
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, benchmark)
    benchmark_open = raw_df_dict["Open"][benchmark].copy()
    return df_dict["Close"], df_dict["Open"], benchmark_close, benchmark_open


def _reference_row(current_formal_run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(current_formal_run_dir / "metrics.json")
    monthly_diag = _load_json(current_formal_run_dir / "primary_research_monthly_diagnostics.json")
    return {
        "variant_name": "current_mainline_reference",
        "description": "short_expert_monthly_v1 current formal reference",
        "profile_name": str(metrics.get("execution_alignment_profile", "")),
        "annual_return": float(metrics.get("primary_research_backtest", {}).get("annual_return", 0.0) or 0.0),
        "excess_annual_return": float(metrics.get("primary_research_backtest", {}).get("excess_annual_return", 0.0) or 0.0),
        "excess_sharpe": float(metrics.get("primary_research_backtest", {}).get("excess_sharpe", 0.0) or 0.0),
        "avg_turnover": float(metrics.get("primary_research_backtest", {}).get("avg_turnover", 0.0) or 0.0),
        "positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
        "median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
        "worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
        "top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
        "longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
        "monthly_robust_score": _monthly_robust_score(monthly_diag),
        "avg_gross_exposure": float("nan"),
        "avg_positive_count": float("nan"),
        "avg_top1_weight": float("nan"),
        "avg_top2_share": float("nan"),
        "avg_hhi": float("nan"),
    }


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _format_num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    current_formal_run_dir = Path(args.current_formal_run_dir).resolve()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(run_dir / "metrics.json")
    if not metrics:
        raise FileNotFoundError(f"Missing metrics.json under {run_dir}")
    raw_target_weights = _load_long_panel(run_dir / "daily_target_weight_panel.csv", "target_weight")
    raw_score_frame = _load_long_panel(run_dir / "daily_score_panel.csv", "score")
    raw_target_weights = raw_target_weights.reindex(columns=sorted(set(raw_target_weights.columns) | set(raw_score_frame.columns))).fillna(0.0)
    raw_score_frame = raw_score_frame.reindex(index=raw_target_weights.index, columns=raw_target_weights.columns).fillna(0.0)

    benchmark = str(metrics.get("benchmark", "000300.SH") or "000300.SH")
    valid_start = str(metrics.get("valid_start", "") or "").replace("-", "")
    valid_end = str(metrics.get("valid_end", "") or "").replace("-", "")
    if not valid_start or not valid_end:
        raise ValueError(f"Run {run_dir} is missing valid_start/valid_end metadata.")

    close, open_df, benchmark_close, benchmark_open = _load_formal_market_data(
        stocks=list(raw_target_weights.columns),
        start_date=valid_start,
        end_date=valid_end,
        benchmark=benchmark,
    )

    holding_count = int(metrics.get("score_head_extra", {}).get("holding_count", metrics.get("holding_count", 5)) or 5)
    max_weight = float(metrics.get("max_weight", 0.25) or 0.25)
    min_adv20 = float(metrics.get("min_adv20", 50000.0) or 50000.0)
    min_price = float(metrics.get("min_price", 2.0) or 2.0)
    max_price = float(metrics.get("max_price", 300.0) or 300.0)
    transaction_cost_bps = float(metrics.get("execution_alignment_transaction_cost_bps", 3.0) or 3.0)
    slippage_bps = float(metrics.get("execution_alignment_slippage_bps", 7.0) or 7.0)
    sell_tax_bps = float(metrics.get("execution_alignment_sell_tax_bps", 10.0) or 10.0)

    variants = [
        VariantSpec("policy_v2_raw_1d", "Original learned target weights with no execution bridge.", "raw_1d"),
        VariantSpec("policy_v2_current_k2_3d", "Original learned target weights locked to k2 3d ensemble.", "regoff_k2_3d_ensemble_native_anchor"),
        VariantSpec("policy_v2_current_k2_5d", "Original learned target weights locked to k2 5d ensemble.", "regoff_k2_5d_ensemble_native_anchor"),
        VariantSpec("policy_v2_current_k2_10d", "Original learned target weights locked to k2 10d ensemble.", "regoff_k2_10d_ensemble_native_anchor"),
        VariantSpec("policy_v2_current_k2_20d", "Current selected policy_v2 execution profile.", "regoff_k2_20d_ensemble_native_anchor"),
        VariantSpec("policy_v2_cap6_k2_5d", "Cap learned candidates at 6 names before k2 5d bridge.", "regoff_k2_5d_ensemble_native_anchor", candidate_cap=6),
        VariantSpec("policy_v2_cap4_k2_5d", "Cap learned candidates at 4 names before k2 5d bridge.", "regoff_k2_5d_ensemble_native_anchor", candidate_cap=4),
        VariantSpec("policy_v2_cap6_g090_096_k2_5d", "Cap to 6 names and clip gross to 0.90-0.96 before k2 5d bridge.", "regoff_k2_5d_ensemble_native_anchor", candidate_cap=6, gross_floor=0.90, gross_cap=0.96),
        VariantSpec("policy_v2_cap4_g092_098_k2_5d", "Cap to 4 names and clip gross to 0.92-0.98 before k2 5d bridge.", "regoff_k2_5d_ensemble_native_anchor", candidate_cap=4, gross_floor=0.92, gross_cap=0.98),
    ]

    rows: list[dict[str, Any]] = []
    variant_payloads: dict[str, dict[str, Any]] = {}
    raw_stats = _panel_stats(raw_target_weights)
    for spec in variants:
        prepared_target_weights = raw_target_weights.copy()
        if spec.candidate_cap > 0:
            prepared_target_weights = _cap_candidate_count(prepared_target_weights, spec.candidate_cap)
        if spec.gross_floor > 0.0 or spec.gross_cap > 0.0:
            prepared_target_weights = _clip_gross_band(prepared_target_weights, spec.gross_floor, spec.gross_cap)
        prepared_stats = _panel_stats(prepared_target_weights)
        profile = exec_align.get_profile(spec.profile_name)
        variant_metrics, monthly_summary, monthly_diag, aligned_stats = _evaluate_variant(
            raw_target_weights=prepared_target_weights.reindex(index=raw_score_frame.index).fillna(0.0),
            raw_score_frame=raw_score_frame,
            close=close,
            benchmark_close=benchmark_close,
            open_df=open_df,
            benchmark_open=benchmark_open,
            benchmark=benchmark,
            holding_count=holding_count,
            max_weight=max_weight,
            min_adv20=min_adv20,
            min_price=min_price,
            max_price=max_price,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            profile=profile,
        )
        variant_root = output_dir / "variants" / spec.name
        variant_root.mkdir(parents=True, exist_ok=True)
        monthly_summary.to_csv(variant_root / "monthly_backtest_summary.csv", index=False, encoding="utf-8-sig")
        (variant_root / "monthly_backtest_diagnostics.json").write_text(
            json.dumps(monthly_diag, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        row = {
            "variant_name": spec.name,
            "description": spec.description,
            "profile_name": profile.name,
            "candidate_cap": int(spec.candidate_cap or 0),
            "gross_floor": float(spec.gross_floor or 0.0),
            "gross_cap": float(spec.gross_cap or 0.0),
            "annual_return": float(variant_metrics.get("annual_return", 0.0) or 0.0),
            "excess_annual_return": float(variant_metrics.get("excess_annual_return", 0.0) or 0.0),
            "excess_sharpe": float(variant_metrics.get("excess_sharpe", 0.0) or 0.0),
            "avg_turnover": float(variant_metrics.get("avg_turnover", 0.0) or 0.0),
            "positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
            "median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
            "mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
            "worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
            "top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
            "longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
            "monthly_robust_score": _monthly_robust_score(monthly_diag),
            "prepared_avg_gross_exposure": float(prepared_stats["avg_gross_exposure"]),
            "prepared_avg_positive_count": float(prepared_stats["avg_positive_count"]),
            "aligned_avg_gross_exposure": float(aligned_stats["avg_gross_exposure"]),
            "aligned_avg_positive_count": float(aligned_stats["avg_positive_count"]),
            "aligned_avg_top1_weight": float(aligned_stats["avg_top1_weight"]),
            "aligned_avg_top2_share": float(aligned_stats["avg_top2_share"]),
            "aligned_avg_hhi": float(aligned_stats["avg_hhi"]),
        }
        rows.append(row)
        variant_payloads[spec.name] = {
            "metrics": variant_metrics,
            "monthly_diagnostics": monthly_diag,
            "prepared_panel_stats": prepared_stats,
            "aligned_panel_stats": aligned_stats,
            "raw_panel_stats": raw_stats,
        }

    scoreboard = pd.DataFrame(rows).sort_values(
        [
            "monthly_robust_score",
            "positive_month_ratio",
            "median_monthly_return",
            "excess_annual_return",
            "excess_sharpe",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    scoreboard.to_csv(output_dir / "variant_scoreboard.csv", index=False, encoding="utf-8-sig")

    best_row = scoreboard.iloc[0].to_dict()
    selected_row = scoreboard.loc[scoreboard["variant_name"] == "policy_v2_current_k2_20d"].iloc[0].to_dict()
    reference_row = _reference_row(current_formal_run_dir)
    summary_payload = {
        "run_dir": str(run_dir),
        "current_formal_run_dir": str(current_formal_run_dir),
        "best_variant": best_row,
        "selected_policy_v2_variant": selected_row,
        "current_mainline_reference": reference_row,
        "variant_payloads": variant_payloads,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Policy V2 Constrained Execution Review",
        "",
        "## Scope",
        f"- run_dir: `{run_dir}`",
        f"- formal window: `{valid_start} -> {valid_end}`",
        f"- raw target-weight avg gross / names: `{_format_pct(raw_stats['avg_gross_exposure'])}` / `{_format_num(raw_stats['avg_positive_count'])}`",
        "",
        "## Direct Answer",
        f"- best constrained variant: `{best_row['variant_name']}`",
        f"- best monthly robust: `{_format_num(best_row['monthly_robust_score'])}`",
        f"- current selected `k2_20d` monthly robust: `{_format_num(selected_row['monthly_robust_score'])}`",
        f"- delta vs current selected `k2_20d`: `{_format_num(float(best_row['monthly_robust_score']) - float(selected_row['monthly_robust_score']))}`",
        f"- current mainline formal monthly robust: `{_format_num(reference_row['monthly_robust_score'])}`",
        f"- delta vs current mainline formal monthly robust: `{_format_num(float(best_row['monthly_robust_score']) - float(reference_row['monthly_robust_score']))}`",
        "",
        "## Scoreboard",
    ]
    for _, row in scoreboard.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: profile `{row['profile_name']}`, excess annual `{_format_pct(row['excess_annual_return'])}`, "
            f"excess Sharpe `{_format_num(row['excess_sharpe'])}`, positive-month `{_format_pct(row['positive_month_ratio'])}`, "
            f"median monthly `{_format_pct(row['median_monthly_return'])}`, worst month `{_format_pct(row['worst_monthly_return'])}`, "
            f"monthly robust `{_format_num(row['monthly_robust_score'])}`, aligned names `{_format_num(row['aligned_avg_positive_count'])}`"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "- `raw_1d` remains a bad answer, so the issue is not that execution bridge should disappear immediately.",
            "- If constrained `k2_3d / k2_5d / k2_10d` beat the current selected `k2_20d`, the formal loss is more likely coming from bridge drift that is too slow, not from the learned-control direction itself.",
            "- If capped candidate-count variants improve together with the faster bridge, `policy_v3` should internalize narrower candidate counts and a tighter gross band instead of widening the raw panel further.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
