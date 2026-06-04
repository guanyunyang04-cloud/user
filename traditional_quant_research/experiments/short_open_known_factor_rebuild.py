"""Build open-known short-line limit-up factors and prior-fit strategy evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.experiments.short_limitup_strategy_search import conservative_stop_target_return


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/short_open_known_factor_rebuild")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-04_short_open_known_factor_rebuild.md")
DEFAULT_YEARS = tuple(range(2017, 2027))
DEFAULT_SELL_WINDOWS = (1, 3, 5, 10, 20)
DEFAULT_STOP_LOSSES = (5.0, 7.0, 10.0)
DEFAULT_TARGETS = (10.0, 15.0, 20.0, 30.0)
DEFAULT_TOP_K_VALUES = (1, 3, 5, 10)
DEFAULT_FEE_BPS = 30.0
DEFAULT_MIN_TRADES = 500
DEFAULT_MIN_TRAIN_TRADES = 100
DEFAULT_MIN_SIGNAL_AMOUNT = 20_000_000.0
PROFILES = ("preopen_submit", "open_print_filter", "executable_only")
EXIT_RULES = ("window_close", "kama_break", "ma5_break", "ma10_break", "upper_shadow", "weak_close")

OPEN_BUCKET_ORDER = {
    "near_limit_up_open": 0,
    "gap_ge_6": 1,
    "gap_3_to_6": 2,
    "gap_0_to_3": 3,
    "flat": 4,
    "gap_minus3_to_0": 5,
    "gap_le_minus3": 6,
}

FORBIDDEN_BUY_FEATURES = {
    "entry_limit_up",
    "entry_one_word_limit",
    "entry_close",
    "entry_high",
    "entry_low",
    "entry_day_close_ret_pct",
    "entry_day_high_ret_pct",
    "entry_day_low_ret_pct",
    "first_sell_open_ret_pct",
    "first_sell_close_ret_pct",
}
FORBIDDEN_BUY_PREFIXES = ("sell", "managed_", "future_", "label_")

PREOPEN_BUY_FEATURE_COLUMNS = (
    "ma5_bias_pct",
    "ma10_bias_pct",
    "ma20_bias_pct",
    "ma30_bias_pct",
    "ma60_bias_pct",
    "ma5_slope_pct",
    "ma10_slope_pct",
    "ma20_slope_pct",
    "ma30_slope_pct",
    "ma60_slope_pct",
    "ma_stack_bullish",
    "ma_compression_5_60_pct",
    "ma_compression_tight",
    "kama_slope_pct",
    "bias_kama_pct",
    "open_below_kama_break_limitup",
    "kama_ma20_above",
    "kama_ma60_above",
    "kama_slope_turn_positive",
    "kama_bias_0_3",
    "kama_bias_3_8",
    "kama_bias_8_15",
    "kama_bias_gt15",
    "kama_slope_positive",
    "close_cross_atr_upper",
    "high_cross_atr_upper",
    "ma60_trend_positive",
    "limit_up_run_ending_today",
    "signal_open_gap_pct",
    "signal_day_ret_from_open_pct",
    "signal_range_pct",
    "signal_close_position",
    "signal_vrat5",
    "signal_turn_x20",
    "signal_amount_x20",
    "turn_x60",
    "signal_ret5_before_pct",
    "signal_ret3_before_pct",
    "signal_ret10_before_pct",
    "signal_ret20_before_pct",
    "signal_ret60_before_pct",
    "max_drawdown_20_before_pct",
    "volatility_20_pct",
    "volatility_compression_20_60",
    "new_high_20",
    "new_high_60",
    "recent_limitup_count_20",
    "recent_limitup_count_60",
    "signal_amount_log10",
    "turn",
    "price",
    "price_low_bucket",
    "price_high_bucket",
    "liquid_amount_ok",
    "market_limitup_count",
    "market_limitup_rate",
    "market_limitup_count_ma5",
    "market_limitup_rate_ma20",
    "market_breadth_5d",
    "market_breadth_20d",
    "industry_limitup_count",
    "industry_limitup_rate",
    "industry_limitup_count_ma5",
    "industry_ret5_mean",
    "price_position_60d",
)
OPEN_KNOWN_FEATURE_COLUMNS = PREOPEN_BUY_FEATURE_COLUMNS + (
    "next_open_gap_pct",
    "next_gap_bucket",
    "entry_open_near_limit",
    "entry_open_normal",
    "entry_open_above_kama",
    "entry_open_above_ma5",
    "entry_open_above_ma10",
    "entry_open_low_flat_small_high",
)
EXECUTABLE_BUY_FEATURE_COLUMNS = OPEN_KNOWN_FEATURE_COLUMNS + ("executable_entry",)


@dataclass(frozen=True)
class RuleSpec:
    name: str
    profile: str
    diagnostic_only: bool = False
    requires_open_known: bool = False
    executable_only: bool = False


def run_short_open_known_factor_rebuild(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    profile: str = "preopen_submit",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sell_windows: Sequence[int] = DEFAULT_SELL_WINDOWS,
    stop_losses: Sequence[float] = DEFAULT_STOP_LOSSES,
    targets: Sequence[float] = DEFAULT_TARGETS,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    fee_bps: float = DEFAULT_FEE_BPS,
    min_trades: int = DEFAULT_MIN_TRADES,
    min_train_trades: int = DEFAULT_MIN_TRAIN_TRADES,
    min_signal_amount: float = DEFAULT_MIN_SIGNAL_AMOUNT,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    selected_years = _parse_int_values(years, "years")
    if not selected_years:
        raise ValueError("years must not be empty")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    feature_columns = feature_columns_for_profile(profile)
    audit_buy_feature_columns(feature_columns)

    run_id = f"short_open_known_factor_rebuild_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    panel_start = f"{min(selected_years) - 1}-01-01"
    panel_end = f"{max(selected_years)}-12-31"
    raw_panel = load_tradeable_panel(root, start_date=panel_start, end_date=panel_end, include_metrics=True, include_industry=True)
    event_panel = build_event_feature_panel(raw_panel, sell_windows=sell_windows, min_signal_amount=min_signal_amount)
    event_panel = event_panel.loc[event_panel["year"].between(min(selected_years) - 1, max(selected_years))].reset_index(drop=True)

    factor_diagnostics = build_factor_diagnostics(event_panel, profile=profile)
    factor_interactions = build_factor_interaction_diagnostics(event_panel, profile=profile)
    rule_grid = build_rule_grid(
        event_panel.loc[event_panel["year"].isin(selected_years)],
        profile=profile,
        sell_windows=sell_windows,
        stop_losses=stop_losses,
        targets=targets,
        fee_bps=fee_bps,
    )
    prior_plan, oos_trades = build_prior_fit_oos(
        event_panel,
        years=selected_years,
        profile=profile,
        sell_windows=sell_windows,
        stop_losses=stop_losses,
        targets=targets,
        fee_bps=fee_bps,
        min_train_trades=min_train_trades,
    )
    sell_diagnostics = build_sell_rule_diagnostics(rule_grid)
    yearly = summarize_oos_yearly(oos_trades)
    portfolio = build_portfolio_topk_summary(oos_trades, top_k_values=top_k_values)
    candidates = build_candidate_strategy_summary(
        oos_trades,
        yearly,
        prior_plan,
        profile=profile,
        expected_years=selected_years,
        min_trades=min_trades,
    )
    summary = summarize_run(
        event_panel,
        rule_grid,
        prior_plan,
        oos_trades,
        yearly,
        portfolio,
        candidates,
        run_id=run_id,
        run_dir=run_dir,
        profile=profile,
        years=selected_years,
        feature_columns=feature_columns,
        manifest=manifest,
        quality=quality,
        fee_bps=fee_bps,
        min_signal_amount=min_signal_amount,
    )
    markdown = render_markdown(summary, candidates, yearly, portfolio, factor_diagnostics, factor_interactions, sell_diagnostics)

    event_panel.to_csv(run_dir / "event_feature_panel_v2.csv", index=False, encoding="utf-8-sig")
    factor_diagnostics.to_csv(run_dir / "factor_bucket_diagnostics_v2.csv", index=False, encoding="utf-8-sig")
    factor_interactions.to_csv(run_dir / "factor_interaction_diagnostics_v2.csv", index=False, encoding="utf-8-sig")
    rule_grid.to_csv(run_dir / "executable_rule_grid_v2.csv", index=False, encoding="utf-8-sig")
    prior_plan.to_csv(run_dir / "prior_fit_rule_plan_v2.csv", index=False, encoding="utf-8-sig")
    oos_trades.to_csv(run_dir / "prior_fit_oos_trades_v2.csv", index=False, encoding="utf-8-sig")
    sell_diagnostics.to_csv(run_dir / "sell_rule_diagnostics_v2.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(run_dir / "oos_yearly_summary.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(run_dir / "portfolio_topk_summary_v2.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(run_dir / "candidate_shortline_summary_v2.csv", index=False, encoding="utf-8-sig")
    event_panel.to_csv(run_dir / "event_feature_panel.csv", index=False, encoding="utf-8-sig")
    factor_diagnostics.to_csv(run_dir / "factor_diagnostics.csv", index=False, encoding="utf-8-sig")
    rule_grid.to_csv(run_dir / "rule_grid.csv", index=False, encoding="utf-8-sig")
    prior_plan.to_csv(run_dir / "prior_fit_rule_plan.csv", index=False, encoding="utf-8-sig")
    oos_trades.to_csv(run_dir / "prior_fit_oos_trades.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(run_dir / "portfolio_topk_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(run_dir / "candidate_strategy_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        log_path = Path(research_log_path)
        if log_path == DEFAULT_RESEARCH_LOG:
            log_path = log_path.with_name(f"2026-06-04_short_open_known_factor_rebuild_{profile}.md")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")
    else:
        log_path = None

    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path is not None else None}


def build_event_feature_panel(
    raw_panel: pd.DataFrame,
    *,
    sell_windows: Sequence[int] = DEFAULT_SELL_WINDOWS,
    min_signal_amount: float = DEFAULT_MIN_SIGNAL_AMOUNT,
) -> pd.DataFrame:
    if raw_panel.empty:
        return pd.DataFrame()
    panel = prepare_short_factor_panel(raw_panel)
    rows: list[dict[str, Any]] = []
    for _, group in panel.groupby("code", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        date_idx_to_pos = {int(value): pos for pos, value in enumerate(group["date_index"].to_numpy())}
        limit_positions = np.flatnonzero(group["limit_up_like"].to_numpy(dtype=bool))
        for pos in limit_positions:
            event_row = group.iloc[pos]
            event_idx = int(event_row["date_index"])
            entry_pos = date_idx_to_pos.get(event_idx + 1)
            if entry_pos is None:
                continue
            entry = group.iloc[entry_pos]
            record = build_event_record(event_row, entry, min_signal_amount=min_signal_amount)
            for window in _parse_int_values(sell_windows, "sell_windows"):
                start_idx = event_idx + 2
                end_idx = event_idx + 1 + int(window)
                window_frame = group.loc[group["date_index"].between(start_idx, end_idx)]
                exit_pos = date_idx_to_pos.get(end_idx)
                if window_frame.empty or exit_pos is None:
                    record[f"sell{window}_close_ret_pct"] = np.nan
                    record[f"sell{window}_max_high_pct"] = np.nan
                    record[f"sell{window}_min_low_pct"] = np.nan
                    continue
                exit_row = group.iloc[exit_pos]
                entry_open = float(entry["open"])
                record[f"sell{window}_close_ret_pct"] = (float(exit_row["close"]) / entry_open - 1.0) * 100.0
                record[f"sell{window}_max_high_pct"] = (float(window_frame["high"].max()) / entry_open - 1.0) * 100.0
                record[f"sell{window}_min_low_pct"] = (float(window_frame["low"].min()) / entry_open - 1.0) * 100.0
                for exit_rule in EXIT_RULES:
                    metrics = managed_exit_metrics(window_frame, entry_open=entry_open, exit_rule=exit_rule, fallback_exit=exit_row)
                    prefix = f"managed_{exit_rule}_sell{window}"
                    record[f"{prefix}_close_ret_pct"] = metrics["close_ret_pct"]
                    record[f"{prefix}_max_high_pct"] = metrics["max_high_pct"]
                    record[f"{prefix}_min_low_pct"] = metrics["min_low_pct"]
                    record[f"{prefix}_holding_days"] = metrics["holding_days"]
            if any(pd.notna(record.get(f"sell{window}_close_ret_pct")) for window in _parse_int_values(sell_windows, "sell_windows")):
                rows.append(record)
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values(["date", "code"]).reset_index(drop=True)


def prepare_short_factor_panel(raw_panel: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "code", "open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
    missing = [column for column in required if column not in raw_panel.columns]
    if missing:
        raise ValueError(f"raw_panel missing columns: {missing}")
    panel = raw_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if "industry" not in panel.columns:
        panel["industry"] = "__unknown__"
    panel["industry"] = panel["industry"].fillna("__unknown__").astype(str)
    calendar = pd.Index(sorted(panel["date"].dropna().unique()))
    date_index = {date: idx for idx, date in enumerate(calendar)}
    panel["date_index"] = panel["date"].map(date_index).astype(int)
    panel["year"] = panel["date"].dt.year.astype(int)
    panel["prev_close"] = panel.groupby("code", sort=False)["close"].shift(1)
    panel["signal_open_gap_pct"] = (panel["open"] / panel["prev_close"] - 1.0) * 100.0
    panel["signal_day_ret_from_open_pct"] = (panel["close"] / panel["open"] - 1.0) * 100.0
    panel["signal_range_pct"] = (panel["high"] / panel["low"] - 1.0) * 100.0
    panel["signal_close_position"] = (panel["close"] - panel["low"]) / (panel["high"] - panel["low"]).replace(0, np.nan)
    for window in (3, 5, 10, 20, 60):
        panel[f"signal_ret{window}_before_pct"] = (panel["close"] / panel.groupby("code", sort=False)["close"].shift(window) - 1.0) * 100.0
    panel["volume_ma5_prior"] = _rolling_by_code(panel, "volume", 5, min_periods=2).groupby(panel["code"], sort=False).shift(1)
    panel["turn_ma20"] = _rolling_by_code(panel, "turn", 20, min_periods=5)
    panel["turn_ma60"] = _rolling_by_code(panel, "turn", 60, min_periods=20)
    panel["amount_ma20"] = _rolling_by_code(panel, "amount", 20, min_periods=5)
    panel["signal_vrat5"] = panel["volume"] / panel["volume_ma5_prior"].replace(0, np.nan)
    panel["signal_turn_x20"] = panel["turn"] / panel["turn_ma20"].replace(0, np.nan)
    panel["turn_x60"] = panel["turn"] / panel["turn_ma60"].replace(0, np.nan)
    panel["signal_amount_x20"] = panel["amount"] / panel["amount_ma20"].replace(0, np.nan)
    panel["signal_amount_log10"] = np.log10(panel["amount"].clip(lower=1))
    panel["price"] = panel["close"]
    panel["price_low_bucket"] = panel["close"] < 8.0
    panel["price_high_bucket"] = panel["close"] > 50.0
    panel = add_kama_and_atr_features(panel)
    panel = add_limitup_features(panel)
    panel = add_market_and_industry_features(panel)
    return panel


def add_market_and_industry_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    market = panel.groupby("date", as_index=False).agg(
        market_limitup_count=("limit_up_like", "sum"),
        market_tradeable_count=("code", "count"),
        market_breadth=("pctChg", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
    )
    market["market_limitup_rate"] = market["market_limitup_count"] / market["market_tradeable_count"].replace(0, np.nan)
    market = market.sort_values("date")
    market["market_limitup_count_ma5"] = market["market_limitup_count"].rolling(5, min_periods=2).mean()
    market["market_limitup_rate_ma20"] = market["market_limitup_rate"].rolling(20, min_periods=5).mean()
    market["market_breadth_5d"] = market["market_breadth"].rolling(5, min_periods=2).mean()
    market["market_breadth_20d"] = market["market_breadth"].rolling(20, min_periods=5).mean()
    frame = frame.merge(
        market[
            [
                "date",
                "market_limitup_count",
                "market_limitup_rate",
                "market_limitup_count_ma5",
                "market_limitup_rate_ma20",
                "market_breadth_5d",
                "market_breadth_20d",
            ]
        ],
        on="date",
        how="left",
    )
    industry = frame.groupby(["industry", "date"], as_index=False).agg(
        industry_limitup_count=("limit_up_like", "sum"),
        industry_tradeable_count=("code", "count"),
        industry_ret_mean=("pctChg", "mean"),
    )
    industry["industry_limitup_rate"] = industry["industry_limitup_count"] / industry["industry_tradeable_count"].replace(0, np.nan)
    industry = industry.sort_values(["industry", "date"])
    industry["industry_limitup_count_ma5"] = (
        industry.groupby("industry", sort=False)["industry_limitup_count"]
        .rolling(5, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    industry["industry_ret5_mean"] = (
        industry.groupby("industry", sort=False)["industry_ret_mean"]
        .rolling(5, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return frame.merge(
        industry[
            [
                "industry",
                "date",
                "industry_limitup_count",
                "industry_limitup_rate",
                "industry_limitup_count_ma5",
                "industry_ret5_mean",
            ]
        ],
        on=["industry", "date"],
        how="left",
    )


def add_kama_and_atr_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    kama_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("code", sort=False):
        kama_parts.append(compute_kama_frame(group["close"], index=group.index))
    kama = pd.concat(kama_parts).sort_index()
    frame = pd.concat([frame, kama], axis=1)
    tr = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["prev_close"] - frame["high"]).abs(),
            (frame["prev_close"] - frame["low"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["tr"] = tr
    frame["atr14"] = _rolling_by_code(frame, "tr", 14, min_periods=7)
    frame["ema20"] = frame.groupby("code", sort=False)["close"].transform(lambda series: series.ewm(span=20, adjust=False).mean())
    frame["atr_upper"] = frame["ema20"] + 2.0 * frame["atr14"]
    for window, min_periods in ((5, 3), (10, 5), (20, 10), (30, 15), (60, 20)):
        ma_col = f"ma{window}"
        frame[ma_col] = _rolling_by_code(frame, "close", window, min_periods=min_periods)
        frame[f"ma{window}_bias_pct"] = (frame["close"] / frame[ma_col] - 1.0) * 100.0
        frame[f"ma{window}_slope_pct"] = (frame[ma_col] / frame.groupby("code", sort=False)[ma_col].shift(1) - 1.0) * 100.0
    frame["ma_stack_bullish"] = (frame["ma5"] > frame["ma10"]) & (frame["ma10"] > frame["ma20"]) & (frame["ma20"] > frame["ma60"])
    ma_stack_max = frame[["ma5", "ma10", "ma20", "ma30", "ma60"]].max(axis=1)
    ma_stack_min = frame[["ma5", "ma10", "ma20", "ma30", "ma60"]].min(axis=1)
    frame["ma_compression_5_60_pct"] = (ma_stack_max / ma_stack_min.replace(0, np.nan) - 1.0) * 100.0
    frame["ma_compression_tight"] = frame["ma_compression_5_60_pct"].le(8.0)
    low60 = _rolling_min_by_code(frame, "low", 60, min_periods=20)
    high60 = _rolling_max_by_code(frame, "high", 60, min_periods=20)
    high20 = _rolling_max_by_code(frame, "high", 20, min_periods=10)
    high_close20 = _rolling_max_by_code(frame, "close", 20, min_periods=10)
    frame["price_position_60d"] = (frame["close"] - low60) / (high60 - low60).replace(0, np.nan)
    frame["new_high_20"] = frame["high"] >= high20 * 0.999
    frame["new_high_60"] = frame["high"] >= high60 * 0.999
    frame["max_drawdown_20_before_pct"] = (frame["close"] / high_close20.replace(0, np.nan) - 1.0) * 100.0
    frame["volatility_20_pct"] = frame.groupby("code", sort=False)["pctChg"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True).sort_index()
    frame["volatility_60_pct"] = frame.groupby("code", sort=False)["pctChg"].rolling(60, min_periods=20).std().reset_index(level=0, drop=True).sort_index()
    frame["volatility_compression_20_60"] = frame["volatility_20_pct"] / frame["volatility_60_pct"].replace(0, np.nan)
    frame["kama_slope_positive"] = frame["kama_slope_pct"] > 0
    frame["kama_slope_turn_positive"] = frame["kama_slope_positive"] & ~frame.groupby("code", sort=False)["kama_slope_positive"].shift(1).fillna(False)
    frame["open_below_kama_break_limitup"] = (frame["open"] < frame["kama"]) & (frame["close"] > frame["kama"])
    frame["kama_ma20_above"] = frame["kama"] > frame["ma20"]
    frame["kama_ma60_above"] = frame["kama"] > frame["ma60"]
    frame["kama_bias_0_3"] = frame["bias_kama_pct"].between(0, 3, inclusive="both")
    frame["kama_bias_3_8"] = frame["bias_kama_pct"].gt(3) & frame["bias_kama_pct"].le(8)
    frame["kama_bias_8_15"] = frame["bias_kama_pct"].gt(8) & frame["bias_kama_pct"].le(15)
    frame["kama_bias_gt15"] = frame["bias_kama_pct"] > 15
    frame["close_cross_atr_upper"] = frame["close"] > frame["atr_upper"]
    frame["high_cross_atr_upper"] = frame["high"] > frame["atr_upper"]
    frame["ma60_trend_positive"] = (frame["close"] > frame["ma60"]) & (frame["ma60_slope_pct"] > 0)
    return frame


def compute_kama_frame(close: pd.Series, *, index: pd.Index | None = None) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce").astype(float)
    d1 = (close - close.shift(10)).abs()
    v1 = (close - close.shift(1)).abs().rolling(10, min_periods=10).sum()
    er = (d1 / v1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cs = er * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0
    cq = (cs * cs).clip(lower=0.0, upper=1.0)
    kbas_values: list[float] = []
    previous = np.nan
    for price, alpha in zip(close.to_numpy(), cq.to_numpy()):
        if np.isnan(previous):
            previous = price
        elif np.isfinite(price) and np.isfinite(alpha):
            previous = alpha * price + (1.0 - alpha) * previous
        kbas_values.append(previous)
    kbas = pd.Series(kbas_values, index=close.index)
    kama = kbas.ewm(span=2, adjust=False).mean()
    output = pd.DataFrame(
        {
            "kbas": kbas,
            "kama": kama,
            "kama_slope_pct": (kama / kama.shift(1) - 1.0) * 100.0,
            "bias_kama_pct": (close / kama - 1.0) * 100.0,
        },
        index=close.index if index is None else index,
    )
    return output


def add_limitup_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["limit_up_like"] = frame["pctChg"].ge(9.5) & frame["close"].ge(frame["high"] * 0.999)
    frame["one_word_limit_like"] = frame["limit_up_like"] & frame["open"].ge(frame["high"] * 0.999) & frame["low"].ge(frame["close"] * 0.999)
    frame["near_one_word_limit_like"] = frame["limit_up_like"] & frame["signal_range_pct"].le(1.0)
    streak = pd.Series(0, index=frame.index, dtype=int)
    for _, group in frame.groupby("code", sort=False):
        current = 0
        values = []
        for is_limit in group["limit_up_like"].to_numpy(dtype=bool):
            current = current + 1 if is_limit else 0
            values.append(current)
        streak.loc[group.index] = values
    frame["limit_up_run_ending_today"] = streak
    frame["_prior_limit_up_like"] = frame.groupby("code", sort=False)["limit_up_like"].shift(1).fillna(False).astype(float)
    frame["recent_limitup_count_20"] = _rolling_by_code(frame, "_prior_limit_up_like", 20, min_periods=5)
    frame["recent_limitup_count_60"] = _rolling_by_code(frame, "_prior_limit_up_like", 60, min_periods=20)
    frame["board_stage"] = np.select(
        [
            frame["limit_up_run_ending_today"].eq(1),
            frame["limit_up_run_ending_today"].eq(2),
            frame["limit_up_run_ending_today"].eq(3),
            frame["limit_up_run_ending_today"].ge(4),
        ],
        ["first_board", "second_board", "third_board", "fourth_plus_board"],
        default="none",
    )
    return frame.drop(columns=["_prior_limit_up_like"])


def build_event_record(event: pd.Series, entry: pd.Series, *, min_signal_amount: float = DEFAULT_MIN_SIGNAL_AMOUNT) -> dict[str, Any]:
    next_open_gap = (float(entry["open"]) / float(event["close"]) - 1.0) * 100.0
    entry_open_near_limit = next_open_gap >= 9.5
    entry_open_normal = next_open_gap < 9.5
    entry_low_flat_small_high = -3.0 <= next_open_gap < 3.0
    liquid_amount_ok = float(event.get("amount", 0.0) or 0.0) >= float(min_signal_amount)
    executable_entry = (
        (not bool(event["one_word_limit_like"]))
        and (not bool(event["near_one_word_limit_like"]))
        and (not bool(entry_open_near_limit))
        and liquid_amount_ok
    )
    return {
        "date": event["date"],
        "year": int(event["year"]),
        "code": event["code"],
        "name_on_date": event.get("name_on_date", ""),
        "industry": event.get("industry", "__unknown__"),
        "open": event["open"],
        "high": event["high"],
        "low": event["low"],
        "close": event["close"],
        "pctChg": event["pctChg"],
        "turn": event["turn"],
        "amount": event["amount"],
        "limit_up_like": bool(event["limit_up_like"]),
        "one_word_limit_like": bool(event["one_word_limit_like"]),
        "near_one_word_limit_like": bool(event["near_one_word_limit_like"]),
        "limit_up_run_ending_today": int(event["limit_up_run_ending_today"]),
        "board_stage": event["board_stage"],
        "ma5": event["ma5"],
        "ma10": event["ma10"],
        "ma20": event["ma20"],
        "ma30": event["ma30"],
        "ma60": event["ma60"],
        "ma5_bias_pct": event["ma5_bias_pct"],
        "ma10_bias_pct": event["ma10_bias_pct"],
        "ma20_bias_pct": event["ma20_bias_pct"],
        "ma30_bias_pct": event["ma30_bias_pct"],
        "ma60_bias_pct": event["ma60_bias_pct"],
        "ma5_slope_pct": event["ma5_slope_pct"],
        "ma10_slope_pct": event["ma10_slope_pct"],
        "ma20_slope_pct": event["ma20_slope_pct"],
        "ma30_slope_pct": event["ma30_slope_pct"],
        "ma60_slope_pct": event["ma60_slope_pct"],
        "ma_stack_bullish": bool(event["ma_stack_bullish"]),
        "ma_compression_5_60_pct": event["ma_compression_5_60_pct"],
        "ma_compression_tight": bool(event["ma_compression_tight"]),
        "kama": event["kama"],
        "kama_slope_pct": event["kama_slope_pct"],
        "bias_kama_pct": event["bias_kama_pct"],
        "open_below_kama_break_limitup": bool(event["open_below_kama_break_limitup"]),
        "kama_ma20_above": bool(event["kama_ma20_above"]),
        "kama_ma60_above": bool(event["kama_ma60_above"]),
        "kama_slope_turn_positive": bool(event["kama_slope_turn_positive"]),
        "kama_bias_0_3": bool(event["kama_bias_0_3"]),
        "kama_bias_3_8": bool(event["kama_bias_3_8"]),
        "kama_bias_8_15": bool(event["kama_bias_8_15"]),
        "kama_bias_gt15": bool(event["kama_bias_gt15"]),
        "kama_slope_positive": bool(event["kama_slope_positive"]),
        "atr14": event["atr14"],
        "atr_upper": event["atr_upper"],
        "close_cross_atr_upper": bool(event["close_cross_atr_upper"]),
        "high_cross_atr_upper": bool(event["high_cross_atr_upper"]),
        "ma60_trend_positive": bool(event["ma60_trend_positive"]),
        "price_position_60d": event["price_position_60d"],
        "signal_open_gap_pct": event["signal_open_gap_pct"],
        "signal_day_ret_from_open_pct": event["signal_day_ret_from_open_pct"],
        "signal_range_pct": event["signal_range_pct"],
        "signal_close_position": event["signal_close_position"],
        "signal_vrat5": event["signal_vrat5"],
        "signal_turn_x20": event["signal_turn_x20"],
        "signal_amount_x20": event["signal_amount_x20"],
        "turn_x60": event["turn_x60"],
        "signal_ret3_before_pct": event["signal_ret3_before_pct"],
        "signal_ret5_before_pct": event["signal_ret5_before_pct"],
        "signal_ret10_before_pct": event["signal_ret10_before_pct"],
        "signal_ret20_before_pct": event["signal_ret20_before_pct"],
        "signal_ret60_before_pct": event["signal_ret60_before_pct"],
        "max_drawdown_20_before_pct": event["max_drawdown_20_before_pct"],
        "volatility_20_pct": event["volatility_20_pct"],
        "volatility_compression_20_60": event["volatility_compression_20_60"],
        "new_high_20": bool(event["new_high_20"]),
        "new_high_60": bool(event["new_high_60"]),
        "recent_limitup_count_20": event["recent_limitup_count_20"],
        "recent_limitup_count_60": event["recent_limitup_count_60"],
        "signal_amount_log10": event["signal_amount_log10"],
        "price": event["price"],
        "price_low_bucket": bool(event["price_low_bucket"]),
        "price_high_bucket": bool(event["price_high_bucket"]),
        "liquid_amount_ok": bool(liquid_amount_ok),
        "market_limitup_count": event["market_limitup_count"],
        "market_limitup_rate": event["market_limitup_rate"],
        "market_limitup_count_ma5": event["market_limitup_count_ma5"],
        "market_limitup_rate_ma20": event["market_limitup_rate_ma20"],
        "market_breadth_5d": event["market_breadth_5d"],
        "market_breadth_20d": event["market_breadth_20d"],
        "industry_limitup_count": event["industry_limitup_count"],
        "industry_limitup_rate": event["industry_limitup_rate"],
        "industry_limitup_count_ma5": event["industry_limitup_count_ma5"],
        "industry_ret5_mean": event["industry_ret5_mean"],
        "entry_date": entry["date"],
        "entry_open": entry["open"],
        "entry_high": entry["high"],
        "entry_low": entry["low"],
        "entry_close": entry["close"],
        "next_open_gap_pct": next_open_gap,
        "next_gap_bucket": bucket_next_open_gap(next_open_gap),
        "entry_open_near_limit": bool(entry_open_near_limit),
        "entry_open_normal": bool(entry_open_normal),
        "entry_open_low_flat_small_high": bool(entry_low_flat_small_high),
        "entry_open_above_kama": bool(float(entry["open"]) > float(event["kama"])) if pd.notna(event["kama"]) else False,
        "entry_open_above_ma5": bool(float(entry["open"]) > float(event["ma5"])) if pd.notna(event["ma5"]) else False,
        "entry_open_above_ma10": bool(float(entry["open"]) > float(event["ma10"])) if pd.notna(event["ma10"]) else False,
        "executable_entry": bool(executable_entry),
        "entry_limit_up": bool(entry["limit_up_like"]),
        "entry_one_word_limit": bool(entry["one_word_limit_like"]),
        "entry_day_close_ret_pct": (float(entry["close"]) / float(entry["open"]) - 1.0) * 100.0,
    }


def managed_exit_metrics(
    window_frame: pd.DataFrame,
    *,
    entry_open: float,
    exit_rule: str,
    fallback_exit: pd.Series,
) -> dict[str, float]:
    if window_frame.empty:
        return {"close_ret_pct": np.nan, "max_high_pct": np.nan, "min_low_pct": np.nan, "holding_days": np.nan}
    frame = window_frame.sort_values("date_index").copy()
    if exit_rule == "window_close":
        exit_row = fallback_exit
    elif exit_rule == "kama_break":
        exit_row = _first_trigger_or_fallback(frame, frame["close"] < frame["kama"], fallback_exit)
    elif exit_rule == "ma5_break":
        exit_row = _first_trigger_or_fallback(frame, frame["close"] < frame["ma5"], fallback_exit)
    elif exit_rule == "ma10_break":
        exit_row = _first_trigger_or_fallback(frame, frame["close"] < frame["ma10"], fallback_exit)
    elif exit_rule == "upper_shadow":
        rng = (frame["high"] - frame["low"]).replace(0, np.nan)
        upper = (frame["high"] - frame[["open", "close"]].max(axis=1)) / rng
        exit_row = _first_trigger_or_fallback(frame, (upper > 0.4) & (frame["close"] <= frame["open"]), fallback_exit)
    elif exit_rule == "weak_close":
        close_position = (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]).replace(0, np.nan)
        weak = (frame["close"] < frame["open"]) & close_position.lt(0.35)
        exit_row = _first_trigger_or_fallback(frame, weak, fallback_exit)
    else:
        raise ValueError(f"unknown exit_rule: {exit_rule}")
    exit_date_index = int(exit_row["date_index"])
    used_window = frame.loc[frame["date_index"].le(exit_date_index)]
    return {
        "close_ret_pct": (float(exit_row["close"]) / float(entry_open) - 1.0) * 100.0,
        "max_high_pct": (float(used_window["high"].max()) / float(entry_open) - 1.0) * 100.0,
        "min_low_pct": (float(used_window["low"].min()) / float(entry_open) - 1.0) * 100.0,
        "holding_days": float(len(used_window)),
    }


def _first_trigger_or_fallback(frame: pd.DataFrame, trigger: pd.Series, fallback_exit: pd.Series) -> pd.Series:
    hits = frame.loc[trigger.fillna(False).to_numpy()]
    if hits.empty:
        return fallback_exit
    return hits.iloc[0]


def bucket_next_open_gap(gap: float) -> str:
    if gap >= 9.5:
        return "near_limit_up_open"
    if gap >= 6.0:
        return "gap_ge_6"
    if gap >= 3.0:
        return "gap_3_to_6"
    if gap > 0.0:
        return "gap_0_to_3"
    if abs(gap) < 1e-12:
        return "flat"
    if gap > -3.0:
        return "gap_minus3_to_0"
    return "gap_le_minus3"


def build_factor_diagnostics(events: pd.DataFrame, *, profile: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    bool_features = [
        "open_below_kama_break_limitup",
        "kama_bias_0_3",
        "kama_bias_3_8",
        "kama_bias_8_15",
        "kama_bias_gt15",
        "kama_slope_positive",
        "close_cross_atr_upper",
        "high_cross_atr_upper",
        "ma60_trend_positive",
        "one_word_limit_like",
        "near_one_word_limit_like",
        "entry_open_near_limit",
    ]
    for feature in bool_features:
        if feature not in events.columns:
            continue
        for value, subset in events.groupby(events[feature].fillna(False), dropna=False):
            rows.append(summarize_bucket(subset, feature=feature, bucket=str(bool(value))))
    numeric_features = [
        "bias_kama_pct",
        "kama_slope_pct",
        "signal_vrat5",
        "signal_turn_x20",
        "signal_ret5_before_pct",
        "signal_ret20_before_pct",
        "signal_ret60_before_pct",
        "signal_open_gap_pct",
        "signal_day_ret_from_open_pct",
        "signal_range_pct",
        "next_open_gap_pct",
    ]
    if profile == "preopen_submit":
        numeric_features = [feature for feature in numeric_features if feature != "next_open_gap_pct"]
    for feature in numeric_features:
        values = pd.to_numeric(events[feature], errors="coerce")
        q20 = values.quantile(0.2)
        q80 = values.quantile(0.8)
        rows.append(summarize_bucket(events.loc[values <= q20], feature=feature, bucket="low20", cut=q20))
        rows.append(summarize_bucket(events.loc[values >= q80], feature=feature, bucket="high20", cut=q80))
    return pd.DataFrame(rows)


def summarize_bucket(subset: pd.DataFrame, *, feature: str, bucket: str, cut: float | None = None) -> dict[str, Any]:
    return {
        "feature": feature,
        "bucket": bucket,
        "cut": cut,
        "n": int(len(subset)),
        "entry_limit_up_rate": float(subset["entry_limit_up"].mean()) if not subset.empty else np.nan,
        "sell1_close_mean": float(subset["sell1_close_ret_pct"].mean()) if "sell1_close_ret_pct" in subset else np.nan,
        "sell3_close_mean": float(subset["sell3_close_ret_pct"].mean()) if "sell3_close_ret_pct" in subset else np.nan,
        "sell20_close_mean": float(subset["sell20_close_ret_pct"].mean()) if "sell20_close_ret_pct" in subset else np.nan,
        "sell20_max_high_mean": float(subset["sell20_max_high_pct"].mean()) if "sell20_max_high_pct" in subset else np.nan,
    }


def build_rule_grid(
    events: pd.DataFrame,
    *,
    profile: str,
    sell_windows: Sequence[int],
    stop_losses: Sequence[float],
    targets: Sequence[float],
    fee_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules = rules_for_profile(profile)
    for rule in rules:
        mask = evaluate_rule(events, rule.name)
        subset = events.loc[mask].copy()
        if subset.empty:
            continue
        for sell_window in _parse_int_values(sell_windows, "sell_windows"):
            close_col = f"sell{sell_window}_close_ret_pct"
            high_col = f"sell{sell_window}_max_high_pct"
            low_col = f"sell{sell_window}_min_low_pct"
            if close_col not in subset.columns or high_col not in subset.columns or low_col not in subset.columns:
                continue
            base = subset.dropna(subset=[close_col, high_col, low_col])
            if base.empty:
                continue
            for stop_loss in _parse_float_values(stop_losses, "stop_losses"):
                for target in _parse_float_values(targets, "targets"):
                    gross = conservative_stop_target_return(base[close_col], base[high_col], base[low_col], stop_loss=stop_loss, target=target)
                    net = gross - fee_bps / 100.0
                    rows.append(
                        {
                            **summarize_return_series(net),
                            "rule_name": rule.name,
                            "profile": profile,
                            "sell_window": int(sell_window),
                            "stop_loss": float(stop_loss),
                            "target": float(target),
                            "fee_bps": float(fee_bps),
                            "diagnostic_only": bool(rule.diagnostic_only),
                            "requires_open_known": bool(rule.requires_open_known),
                            "source_rows": int(len(base)),
                        }
                    )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return pd.DataFrame(columns=rule_grid_columns())
    return grid.reindex(columns=rule_grid_columns()).sort_values(["mean_net_ret_pct", "payoff", "n"], ascending=False).reset_index(drop=True)


def build_prior_fit_oos(
    events: pd.DataFrame,
    *,
    years: Sequence[int],
    profile: str,
    sell_windows: Sequence[int],
    stop_losses: Sequence[float],
    targets: Sequence[float],
    fee_bps: float,
    min_train_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    rules = rules_for_profile(profile)
    for eval_year in _parse_int_values(years, "years"):
        train = events.loc[events["year"] < eval_year].copy()
        eval_events = events.loc[events["year"].eq(eval_year)].copy()
        train_grid = build_rule_grid(
            train,
            profile=profile,
            sell_windows=sell_windows,
            stop_losses=stop_losses,
            targets=targets,
            fee_bps=fee_bps,
        )
        train_grid = train_grid[pd.to_numeric(train_grid["n"], errors="coerce").fillna(0) >= int(min_train_trades)].copy()
        selection_grid = train_grid.loc[~train_grid["diagnostic_only"].astype(bool)].copy()
        if selection_grid.empty:
            plan_rows.append(
                {
                    "eval_year": eval_year,
                    "fit_years": ",".join(str(year) for year in sorted(train["year"].dropna().unique())),
                    "fit_uses_eval_year": False,
                    "selected_rule_name": "",
                    "sell_window": np.nan,
                    "stop_loss": np.nan,
                    "target": np.nan,
                    "train_n": 0,
                    "train_mean_net_ret_pct": np.nan,
                    "train_payoff": np.nan,
                    "diagnostic_only": False,
                    "requires_open_known": profile == "open_print_filter",
                    "selection_status": "no_executable_train_rule",
                }
            )
            continue
        selection_grid["rank_score"] = (
            selection_grid["mean_net_ret_pct"].astype(float)
            + 0.25 * selection_grid["payoff"].fillna(0).astype(float)
            - 2.0 * selection_grid["stop_hit_rate"].fillna(1).astype(float)
        )
        selected = selection_grid.sort_values(["rank_score", "mean_net_ret_pct", "payoff", "n"], ascending=False).iloc[0].to_dict()
        selected_rule = next(rule for rule in rules if rule.name == selected["rule_name"])
        fit_years = sorted(train["year"].dropna().astype(int).unique().tolist())
        plan_rows.append(
            {
                "eval_year": eval_year,
                "fit_years": ",".join(str(year) for year in fit_years),
                "fit_uses_eval_year": False,
                "selected_rule_name": selected["rule_name"],
                "sell_window": int(selected["sell_window"]),
                "stop_loss": float(selected["stop_loss"]),
                "target": float(selected["target"]),
                "train_n": int(selected["n"]),
                "train_mean_net_ret_pct": float(selected["mean_net_ret_pct"]),
                "train_payoff": float(selected["payoff"]) if pd.notna(selected["payoff"]) else np.nan,
                "diagnostic_only": bool(selected_rule.diagnostic_only),
                "requires_open_known": bool(selected_rule.requires_open_known),
                "selection_status": "selected",
            }
        )
        trades = materialize_rule_trades(
            eval_events,
            rule_name=str(selected["rule_name"]),
            sell_window=int(selected["sell_window"]),
            stop_loss=float(selected["stop_loss"]),
            target=float(selected["target"]),
            fee_bps=fee_bps,
            diagnostic_only=bool(selected_rule.diagnostic_only),
            requires_open_known=bool(selected_rule.requires_open_known),
        )
        if not trades.empty:
            trades["eval_year"] = eval_year
            trade_frames.append(trades)
    plan = pd.DataFrame(plan_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=oos_trade_columns())
    return plan, trades.reindex(columns=oos_trade_columns())


def materialize_rule_trades(
    events: pd.DataFrame,
    *,
    rule_name: str,
    sell_window: int,
    stop_loss: float,
    target: float,
    fee_bps: float,
    diagnostic_only: bool,
    requires_open_known: bool,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=oos_trade_columns())
    mask = evaluate_rule(events, rule_name)
    subset = events.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=oos_trade_columns())
    close_col = f"sell{sell_window}_close_ret_pct"
    high_col = f"sell{sell_window}_max_high_pct"
    low_col = f"sell{sell_window}_min_low_pct"
    subset = subset.dropna(subset=[close_col, high_col, low_col]).copy()
    if subset.empty:
        return pd.DataFrame(columns=oos_trade_columns())
    subset["gross_ret_pct"] = conservative_stop_target_return(
        subset[close_col],
        subset[high_col],
        subset[low_col],
        stop_loss=stop_loss,
        target=target,
    )
    subset["net_ret_pct"] = subset["gross_ret_pct"] - fee_bps / 100.0
    subset["rule_name"] = rule_name
    subset["sell_window"] = int(sell_window)
    subset["stop_loss"] = float(stop_loss)
    subset["target"] = float(target)
    subset["fee_bps"] = float(fee_bps)
    subset["diagnostic_only"] = bool(diagnostic_only)
    subset["requires_open_known"] = bool(requires_open_known)
    subset["fit_uses_eval_year"] = False
    subset["trade_score"] = score_trades(subset)
    return subset


def summarize_oos_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["eval_year", "n", "mean_net_ret_pct", "median_net_ret_pct", "win_rate", "payoff", "diagnostic_trade_rate"])
    rows = []
    for year, group in trades.groupby("eval_year", sort=True):
        stats = summarize_return_series(group["net_ret_pct"])
        rows.append(
            {
                "eval_year": int(year),
                "n": int(len(group)),
                "mean_net_ret_pct": stats["mean_net_ret_pct"],
                "median_net_ret_pct": stats["median_net_ret_pct"],
                "win_rate": stats["win_rate"],
                "payoff": stats["payoff"],
                "diagnostic_trade_rate": float(group["diagnostic_only"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_portfolio_topk_summary(trades: pd.DataFrame, *, top_k_values: Sequence[int]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["top_k", "periods", "mean_period_net_ret_pct", "median_period_net_ret_pct", "win_rate", "mean_trade_count"])
    rows = []
    for top_k in _parse_int_values(top_k_values, "top_k_values"):
        selected = (
            trades.sort_values(["date", "trade_score", "code"], ascending=[True, False, True])
            .groupby("date", group_keys=False)
            .head(int(top_k))
        )
        period = selected.groupby("date", as_index=False).agg(period_net_ret_pct=("net_ret_pct", "mean"), trade_count=("code", "count"))
        rows.append(
            {
                "top_k": int(top_k),
                "periods": int(len(period)),
                "mean_period_net_ret_pct": float(period["period_net_ret_pct"].mean()) if not period.empty else np.nan,
                "median_period_net_ret_pct": float(period["period_net_ret_pct"].median()) if not period.empty else np.nan,
                "win_rate": float((period["period_net_ret_pct"] > 0).mean()) if not period.empty else np.nan,
                "mean_trade_count": float(period["trade_count"].mean()) if not period.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_strategy_summary(
    trades: pd.DataFrame,
    yearly: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    profile: str,
    expected_years: Sequence[int],
    min_trades: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "profile": profile,
                    "n": 0,
                    "mean_net_ret_pct": np.nan,
                    "positive_year_rate": 0.0,
                    "min_year_mean_net_ret_pct": np.nan,
                    "payoff": np.nan,
                    "evidence_grade": "shortline_no_candidate",
                    "failed_gates": "no_oos_trades",
                }
            ]
        )
    stats = summarize_return_series(trades["net_ret_pct"])
    years = set(yearly["eval_year"].astype(int).tolist()) if not yearly.empty else set()
    expected_year_set = set(_parse_int_values(expected_years, "expected_years"))
    positive_year_rate = float((yearly["mean_net_ret_pct"] > 0).mean()) if not yearly.empty else 0.0
    min_year = float(yearly["mean_net_ret_pct"].min()) if not yearly.empty else np.nan
    diagnostic = bool(trades["diagnostic_only"].any())
    requires_open = bool(trades["requires_open_known"].any())
    near_limit_trade_rate = (
        float(trades["entry_open_near_limit"].fillna(False).astype(bool).mean())
        if "entry_open_near_limit" in trades.columns
        else 0.0
    )
    failures = []
    if bool(plan["fit_uses_eval_year"].any()) if "fit_uses_eval_year" in plan else False:
        failures.append("fit_uses_eval_year")
    if years != expected_year_set or expected_year_set != set(DEFAULT_YEARS):
        failures.append("incomplete_oos_years")
    if int(stats["n"]) < int(min_trades):
        failures.append("insufficient_trades")
    if not (stats["mean_net_ret_pct"] > 0):
        failures.append("nonpositive_mean")
    if positive_year_rate < 0.8:
        failures.append("positive_year_rate_lt_0.8")
    if not (min_year >= 0):
        failures.append("negative_worst_year")
    if not (stats["payoff"] >= 2.0):
        failures.append("payoff_lt_2")
    if diagnostic:
        failures.append("diagnostic_or_unfilled_path")
    if near_limit_trade_rate > 0:
        failures.append("contains_unfilled_near_limit_entries")
    if requires_open or profile == "open_print_filter":
        failures.append("open_known_diagnostic")
    if failures:
        grade = "open_known_diagnostic" if "open_known_diagnostic" in failures else "shortline_diagnostic_only"
    else:
        grade = "shortline_backtest_candidate"
    return pd.DataFrame(
        [
            {
                "profile": profile,
                "n": int(stats["n"]),
                "mean_net_ret_pct": stats["mean_net_ret_pct"],
                "median_net_ret_pct": stats["median_net_ret_pct"],
                "win_rate": stats["win_rate"],
                "payoff": stats["payoff"],
                "positive_year_rate": positive_year_rate,
                "min_year_mean_net_ret_pct": min_year,
                "diagnostic_trade_rate": float(trades["diagnostic_only"].mean()),
                "open_known_trade_rate": float(trades["requires_open_known"].mean()),
                "near_limit_trade_rate": near_limit_trade_rate,
                "evidence_grade": grade,
                "failed_gates": ",".join(failures),
            }
        ]
    )


def rules_for_profile(profile: str) -> list[RuleSpec]:
    base = [
        RuleSpec("all_limitup", profile),
        RuleSpec("kama_bias_0_8", profile),
        RuleSpec("kama_break_from_below", profile),
        RuleSpec("kama_break_atr_upper", profile),
        RuleSpec("kama_slope_positive_not_hot", profile),
        RuleSpec("atr_upper_not_hot", profile),
        RuleSpec("first_board_kama_break", profile),
        RuleSpec("second_plus_not_hot", profile),
        RuleSpec("third_plus_not_hot", profile),
        RuleSpec("low_prior_return_kama", profile),
        RuleSpec("moderate_volume_kama", profile),
        RuleSpec("signal_one_word", profile, diagnostic_only=True),
        RuleSpec("near_one_word", profile, diagnostic_only=True),
    ]
    if profile == "open_print_filter":
        base.extend(
            [
                RuleSpec("open_gap_0_3", profile, requires_open_known=True),
                RuleSpec("open_gap_m3_to_3", profile, requires_open_known=True),
                RuleSpec("kama_bias_0_8_open_0_3", profile, requires_open_known=True),
                RuleSpec("kama_break_open_0_3", profile, requires_open_known=True),
                RuleSpec("not_hot_open_0_3", profile, requires_open_known=True),
                RuleSpec("board2p_open_0_3", profile, requires_open_known=True),
                RuleSpec("avoid_near_limit_kama", profile, requires_open_known=True),
                RuleSpec("near_limit_open", profile, diagnostic_only=True, requires_open_known=True),
            ]
        )
    return base


def evaluate_rule(events: pd.DataFrame, rule_name: str) -> pd.Series:
    idx = events.index
    true = pd.Series(True, index=idx)
    not_hot = events["bias_kama_pct"].le(15).fillna(False) & events["signal_ret20_before_pct"].le(40).fillna(False)
    kama_0_8 = (events["kama_bias_0_3"].fillna(False) | events["kama_bias_3_8"].fillna(False))
    board2p = events["board_stage"].isin(["second_board", "third_board", "fourth_plus_board"])
    board3p = events["board_stage"].isin(["third_board", "fourth_plus_board"])
    open_0_3 = events.get("next_gap_bucket", pd.Series("", index=idx)).eq("gap_0_to_3")
    open_m3_3 = events.get("next_gap_bucket", pd.Series("", index=idx)).isin(["gap_minus3_to_0", "flat", "gap_0_to_3"])
    rules: dict[str, pd.Series] = {
        "all_limitup": true,
        "kama_bias_0_8": kama_0_8,
        "kama_break_from_below": events["open_below_kama_break_limitup"].fillna(False) & not_hot,
        "kama_break_atr_upper": events["open_below_kama_break_limitup"].fillna(False) & events["close_cross_atr_upper"].fillna(False) & not_hot,
        "kama_slope_positive_not_hot": events["kama_slope_positive"].fillna(False) & not_hot,
        "atr_upper_not_hot": events["close_cross_atr_upper"].fillna(False) & not_hot,
        "first_board_kama_break": events["board_stage"].eq("first_board") & events["open_below_kama_break_limitup"].fillna(False) & not_hot,
        "second_plus_not_hot": board2p & not_hot,
        "third_plus_not_hot": board3p & not_hot,
        "low_prior_return_kama": events["signal_ret20_before_pct"].le(15).fillna(False) & kama_0_8,
        "moderate_volume_kama": events["signal_vrat5"].between(1.0, 3.0).fillna(False) & kama_0_8,
        "signal_one_word": events["one_word_limit_like"].fillna(False),
        "near_one_word": (events["one_word_limit_like"].fillna(False) | events["near_one_word_limit_like"].fillna(False)),
        "open_gap_0_3": open_0_3,
        "open_gap_m3_to_3": open_m3_3,
        "kama_bias_0_8_open_0_3": kama_0_8 & open_0_3,
        "kama_break_open_0_3": events["open_below_kama_break_limitup"].fillna(False) & not_hot & open_0_3,
        "not_hot_open_0_3": not_hot & open_0_3,
        "board2p_open_0_3": board2p & not_hot & open_0_3,
        "avoid_near_limit_kama": kama_0_8 & ~events.get("entry_open_near_limit", pd.Series(False, index=idx)).fillna(False),
        "near_limit_open": events.get("entry_open_near_limit", pd.Series(False, index=idx)).fillna(False),
    }
    if rule_name not in rules:
        raise ValueError(f"unknown rule_name: {rule_name}")
    return rules[rule_name].fillna(False)


def score_trades(trades: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=trades.index)
    score += pd.to_numeric(trades.get("kama_slope_pct", 0), errors="coerce").fillna(0).clip(-5, 5)
    score += pd.to_numeric(trades.get("signal_close_position", 0), errors="coerce").fillna(0) * 2.0
    score -= pd.to_numeric(trades.get("bias_kama_pct", 0), errors="coerce").fillna(0).clip(lower=0) * 0.05
    score += pd.to_numeric(trades.get("limit_up_run_ending_today", 0), errors="coerce").fillna(0).clip(upper=4)
    score -= pd.to_numeric(trades.get("next_open_gap_pct", 0), errors="coerce").fillna(0).clip(lower=0) * 0.02
    return score


def summarize_return_series(values: pd.Series) -> dict[str, Any]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    wins = series[series > 0]
    losses = series[series < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    if avg_loss < 0:
        payoff = float(avg_win / abs(avg_loss))
    elif avg_win > 0:
        payoff = float("inf")
    else:
        payoff = np.nan
    return {
        "n": int(len(series)),
        "mean_net_ret_pct": float(series.mean()) if not series.empty else np.nan,
        "median_net_ret_pct": float(series.median()) if not series.empty else np.nan,
        "p10_net_ret_pct": float(series.quantile(0.1)) if not series.empty else np.nan,
        "p90_net_ret_pct": float(series.quantile(0.9)) if not series.empty else np.nan,
        "win_rate": float((series > 0).mean()) if not series.empty else np.nan,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "payoff": payoff,
        "stop_hit_rate": float((series <= -5.0).mean()) if not series.empty else np.nan,
    }


def feature_columns_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "preopen_submit":
        return PREOPEN_BUY_FEATURE_COLUMNS
    if profile == "open_print_filter":
        return OPEN_KNOWN_FEATURE_COLUMNS
    raise ValueError(f"unknown profile: {profile}")


def audit_buy_feature_columns(feature_columns: Sequence[str]) -> None:
    bad = []
    for column in feature_columns:
        if column in FORBIDDEN_BUY_FEATURES or column.startswith(FORBIDDEN_BUY_PREFIXES):
            bad.append(column)
    if bad:
        raise ValueError(f"buy feature columns include leakage fields: {bad}")


def summarize_run(
    events: pd.DataFrame,
    rule_grid: pd.DataFrame,
    plan: pd.DataFrame,
    trades: pd.DataFrame,
    yearly: pd.DataFrame,
    portfolio: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    profile: str,
    years: Sequence[int],
    feature_columns: Sequence[str],
    manifest: Mapping[str, Any],
    quality: Mapping[str, Any],
    fee_bps: float,
) -> dict[str, Any]:
    candidate = candidates.iloc[0].to_dict() if not candidates.empty else {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "profile": profile,
        "years": list(_parse_int_values(years, "years")),
        "snapshot_id": manifest.get("snapshot_id") or quality.get("snapshot_id"),
        "event_rows": int(len(events)),
        "rule_grid_rows": int(len(rule_grid)),
        "prior_fit_plan_rows": int(len(plan)),
        "oos_trade_rows": int(len(trades)),
        "feature_column_count": int(len(feature_columns)),
        "feature_leakage_audit": "passed",
        "fee_bps": float(fee_bps),
        "candidate": _json_ready(candidate),
        "decision": candidate.get("evidence_grade", "shortline_no_candidate"),
    }


def render_markdown(
    summary: Mapping[str, Any],
    candidates: pd.DataFrame,
    yearly: pd.DataFrame,
    portfolio: pd.DataFrame,
    factor_diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# Short Open-Known Factor Rebuild",
        "",
        "## Summary",
        "",
        f"- Profile: `{summary.get('profile')}`",
        f"- Years: `{summary.get('years')}`",
        f"- Event rows: {summary.get('event_rows')}",
        f"- OOS trades: {summary.get('oos_trade_rows')}",
        f"- Decision: `{summary.get('decision')}`",
        "- T+1 limit-up is a label only; buy features passed leakage audit.",
        "",
        "## Candidate",
        "",
        _markdown_table(candidates, ["profile", "n", "mean_net_ret_pct", "win_rate", "payoff", "positive_year_rate", "min_year_mean_net_ret_pct", "evidence_grade", "failed_gates"]),
        "",
        "## OOS Yearly",
        "",
        _markdown_table(yearly, ["eval_year", "n", "mean_net_ret_pct", "median_net_ret_pct", "win_rate", "payoff"]),
        "",
        "## Portfolio Top-K",
        "",
        _markdown_table(portfolio, ["top_k", "periods", "mean_period_net_ret_pct", "median_period_net_ret_pct", "win_rate", "mean_trade_count"]),
        "",
        "## Factor Diagnostics Snapshot",
        "",
        _markdown_table(
            factor_diagnostics.sort_values("sell3_close_mean", ascending=False).head(12) if not factor_diagnostics.empty else factor_diagnostics,
            ["feature", "bucket", "n", "entry_limit_up_rate", "sell1_close_mean", "sell3_close_mean", "sell20_close_mean", "sell20_max_high_mean"],
        ),
        "",
        "## Notes",
        "",
        "- One-word and near-limit paths remain diagnostic unless the candidate summary proves executable gates.",
        "- `open_print_filter` may be useful operationally, but is reported as open-known diagnostic rather than full pre-open evidence.",
    ]
    return "\n".join(lines)


def _rolling_by_code(frame: pd.DataFrame, column: str, window: int, *, min_periods: int) -> pd.Series:
    return (
        frame.groupby("code", sort=False)[column]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _rolling_min_by_code(frame: pd.DataFrame, column: str, window: int, *, min_periods: int) -> pd.Series:
    return (
        frame.groupby("code", sort=False)[column]
        .rolling(window=window, min_periods=min_periods)
        .min()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _rolling_max_by_code(frame: pd.DataFrame, column: str, window: int, *, min_periods: int) -> pd.Series:
    return (
        frame.groupby("code", sort=False)[column]
        .rolling(window=window, min_periods=min_periods)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[column for column in columns if column in frame.columns]].copy()
    for column in view.columns:
        if column.endswith("_pct"):
            view[column] = view[column].map(_fmt_pct)
        elif column.endswith("_rate"):
            view[column] = view[column].map(_fmt_rate)
        elif column == "payoff":
            view[column] = view[column].map(_fmt_num)
    return view.to_markdown(index=False)


def _fmt_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_rate(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_num(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _parse_int_values(values: Sequence[int] | str, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        parsed = tuple(int(value.strip()) for value in values.split(",") if value.strip())
    else:
        parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _parse_float_values(values: Sequence[float] | str, name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    else:
        parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value


def rule_grid_columns() -> list[str]:
    return [
        "rule_name",
        "profile",
        "sell_window",
        "stop_loss",
        "target",
        "fee_bps",
        "n",
        "source_rows",
        "mean_net_ret_pct",
        "median_net_ret_pct",
        "p10_net_ret_pct",
        "p90_net_ret_pct",
        "win_rate",
        "avg_win_pct",
        "avg_loss_pct",
        "payoff",
        "stop_hit_rate",
        "diagnostic_only",
        "requires_open_known",
    ]


def oos_trade_columns() -> list[str]:
    return [
        "eval_year",
        "date",
        "entry_date",
        "code",
        "rule_name",
        "sell_window",
        "stop_loss",
        "target",
        "fee_bps",
        "gross_ret_pct",
        "net_ret_pct",
        "trade_score",
        "fit_uses_eval_year",
        "diagnostic_only",
        "requires_open_known",
        "next_open_gap_pct",
        "next_gap_bucket",
        "entry_open_near_limit",
        "entry_limit_up",
        "one_word_limit_like",
        "near_one_word_limit_like",
        "board_stage",
        "bias_kama_pct",
        "kama_slope_pct",
        "open_below_kama_break_limitup",
        "close_cross_atr_upper",
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--profile", default="preopen_submit", choices=PROFILES)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sell-windows", default=",".join(str(value) for value in DEFAULT_SELL_WINDOWS))
    parser.add_argument("--stop-losses", default=",".join(str(value) for value in DEFAULT_STOP_LOSSES))
    parser.add_argument("--targets", default=",".join(str(value) for value in DEFAULT_TARGETS))
    parser.add_argument("--top-k-values", default=",".join(str(value) for value in DEFAULT_TOP_K_VALUES))
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_short_open_known_factor_rebuild(
        root=args.root,
        years=_parse_int_values(args.years, "years"),
        profile=args.profile,
        output_dir=args.output_dir,
        sell_windows=_parse_int_values(args.sell_windows, "sell_windows"),
        stop_losses=_parse_float_values(args.stop_losses, "stop_losses"),
        targets=_parse_float_values(args.targets, "targets"),
        top_k_values=_parse_int_values(args.top_k_values, "top_k_values"),
        fee_bps=args.fee_bps,
        min_trades=args.min_trades,
        min_train_trades=args.min_train_trades,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
