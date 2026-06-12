"""Research 1-2 position strategies around two-day KAMA/ATR limit-up setups."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.experiments.short_open_known_factor_rebuild import prepare_short_factor_panel
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_DATA_START_YEAR,
    DEFAULT_END_DATE,
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    DEFAULT_START_DATE,
    _assert_memory_available,
    _is_finite,
    available_memory_gb,
    bucket_atr_break_strength,
    bucket_gap,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/two_day_kama_atr_strategy_research")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-07_two_day_kama_atr_strategy_research.md")
DEFAULT_FEE_BPS = 30.0
DEFAULT_MAX_POSITIONS = (1, 2)
DEFAULT_WARMUP_YEARS = 1
DEFAULT_MAX_HOLD_OFFSET = 20
ENTRY_FILTERS = (
    "open_lt6",
    "open_0_6",
    "open_m3_6",
    "open_m3_3",
    "open_3_6",
    "fillable_open",
    "gap_le0",
    "gap_m5_m3",
    "gap_le0_atr_below_m10",
    "gap_m5_0_atr_below_m10",
    "gap_le0_second_atr_below_m5",
    "gap_m5_0_second",
)
EXIT_POLICIES = ("confirm_day3_close_nonconfirm_day3_open", "confirm_near_extend_nonconfirm_day3_open", "all_day3_close")
SCORE_COLUMNS = ("day1_strength_score", "day2_open_gap_pct", "day1_amount_log10")


@dataclass(frozen=True)
class StrategySpec:
    entry_filter: str
    exit_policy: str
    score_column: str
    max_positions: int

    @property
    def strategy_id(self) -> str:
        return f"{self.entry_filter}__{self.exit_policy}__{self.score_column}__pos{self.max_positions}"


def run_two_day_kama_atr_strategy_research(
    *,
    root: str | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fee_bps: float = DEFAULT_FEE_BPS,
    warmup_years: int = DEFAULT_WARMUP_YEARS,
    max_hold_offset: int = DEFAULT_MAX_HOLD_OFFSET,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    run_id = f"two_day_kama_atr_strategy_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    candidates, path, chunk_summary = build_day1_candidate_trades_chunked(
        root=root,
        start_date=start_date,
        end_date=end_date,
        warmup_years=warmup_years,
        max_hold_offset=max_hold_offset,
        min_available_memory_gb=min_available_memory_gb,
    )
    specs = [
        StrategySpec(entry_filter=entry_filter, exit_policy=exit_policy, score_column=score_column, max_positions=max_positions)
        for entry_filter in ENTRY_FILTERS
        for exit_policy in EXIT_POLICIES
        for score_column in SCORE_COLUMNS
        for max_positions in DEFAULT_MAX_POSITIONS
    ]
    grid_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for spec in specs:
        trades = materialize_strategy_trades(candidates, spec=spec, fee_bps=fee_bps)
        summary = summarize_strategy_trades(trades, spec=spec)
        grid_rows.append(summary)
        yearly = summarize_strategy_yearly(trades, spec=spec)
        if not yearly.empty:
            yearly_frames.append(yearly)
        if not trades.empty:
            trade_frames.append(trades.assign(strategy_id=spec.strategy_id))
    grid = pd.DataFrame(grid_rows)
    yearly_all = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    all_strategy_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    ranked = rank_strategy_grid(grid)
    best_strategy_id = str(ranked.iloc[0]["strategy_id"]) if not ranked.empty else ""
    best_trades = all_strategy_trades.loc[all_strategy_trades["strategy_id"].eq(best_strategy_id)].copy() if best_strategy_id else pd.DataFrame()

    summary = build_summary(
        candidates,
        grid,
        ranked,
        best_trades,
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        quality=quality,
        start_date=start_date,
        end_date=end_date,
        fee_bps=fee_bps,
        warmup_years=warmup_years,
        max_hold_offset=max_hold_offset,
        min_available_memory_gb=min_available_memory_gb,
        chunk_summary=chunk_summary,
    )
    markdown = render_markdown(summary, ranked, yearly_all, best_trades)

    candidates.to_csv(run_dir / "day1_candidate_trades.csv", index=False, encoding="utf-8-sig")
    path.to_csv(run_dir / "day1_candidate_future_path.csv", index=False, encoding="utf-8-sig")
    chunk_summary.to_csv(run_dir / "chunk_summary.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(run_dir / "strategy_grid.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(run_dir / "strategy_grid_ranked.csv", index=False, encoding="utf-8-sig")
    yearly_all.to_csv(run_dir / "strategy_yearly.csv", index=False, encoding="utf-8-sig")
    best_trades.to_csv(run_dir / "best_strategy_trades.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path is not None else None}


def build_day1_candidate_trades_chunked(
    *,
    root: str | None,
    start_date: str,
    end_date: str,
    warmup_years: int,
    max_hold_offset: int,
    min_available_memory_gb: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")
    candidate_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    chunk_rows: list[dict[str, Any]] = []
    for year in range(int(start.year), int(end.year) + 1):
        signal_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        signal_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if signal_end < signal_start:
            continue
        chunk_start = pd.Timestamp(year=max(DEFAULT_DATA_START_YEAR, year - max(0, int(warmup_years))), month=1, day=1)
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=max(90, int(max_hold_offset) * 3)))
        _assert_memory_available(min_available_memory_gb, context=f"before loading strategy year {year}")
        raw_panel = load_tradeable_panel(
            root,
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            include_industry=True,
            include_metrics=True,
        )
        feature_panel = prepare_short_factor_panel(raw_panel)
        del raw_panel
        _assert_memory_available(min_available_memory_gb, context=f"after feature build strategy year {year}")
        candidates, path = build_day1_candidate_trades_from_feature_panel(
            feature_panel,
            signal_start_date=signal_start.strftime("%Y-%m-%d"),
            signal_end_date=signal_end.strftime("%Y-%m-%d"),
            max_hold_offset=max_hold_offset,
        )
        chunk_rows.append(
            {
                "year": int(year),
                "chunk_start": chunk_start.strftime("%Y-%m-%d"),
                "chunk_end": chunk_end.strftime("%Y-%m-%d"),
                "signal_start": signal_start.strftime("%Y-%m-%d"),
                "signal_end": signal_end.strftime("%Y-%m-%d"),
                "feature_rows": int(len(feature_panel)),
                "candidate_count": int(len(candidates)),
                "confirmed_count": int(candidates["confirmed_two_day_breakout"].sum()) if not candidates.empty else 0,
                "future_path_rows": int(len(path)),
                "available_memory_gb_after_chunk": available_memory_gb(),
            }
        )
        if not candidates.empty:
            candidate_frames.append(candidates)
        if not path.empty:
            path_frames.append(path)
        del feature_panel, candidates, path
        gc.collect()
        _assert_memory_available(min_available_memory_gb, context=f"after releasing strategy year {year}")
    all_candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    all_path = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    if not all_candidates.empty:
        all_candidates = all_candidates.sort_values(["day1_date", "code"]).reset_index(drop=True)
    if not all_path.empty:
        all_path = all_path.sort_values(["candidate_id", "future_offset"]).reset_index(drop=True)
    return all_candidates, all_path, pd.DataFrame(chunk_rows)


def build_day1_candidate_trades_from_feature_panel(
    panel: pd.DataFrame,
    *,
    signal_start_date: str,
    signal_end_date: str,
    max_hold_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["code", "date_index", "date"]).reset_index(drop=True)
    for column in ["open", "high", "low", "close", "pctChg", "amount", "turn", "kama", "atr_upper"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    start = pd.Timestamp(signal_start_date)
    end = pd.Timestamp(signal_end_date)
    for code, group in frame.groupby("code", sort=False):
        group = group.sort_values("date_index").reset_index(drop=True)
        date_to_pos = {int(value): pos for pos, value in enumerate(group["date_index"].to_numpy(dtype=int))}
        prev_close = group["close"].shift(1)
        prev_kama = group["kama"].shift(1)
        day1_mask = (
            group["date"].between(start, end, inclusive="both")
            & group["limit_up_like"].fillna(False).astype(bool)
            & prev_close.lt(prev_kama)
            & group["close"].gt(group["kama"])
        )
        for day1_pos in np.flatnonzero(day1_mask.to_numpy(dtype=bool)):
            day1 = group.iloc[int(day1_pos)]
            day2_pos = date_to_pos.get(int(day1["date_index"]) + 1)
            if day2_pos is None:
                continue
            day2 = group.iloc[int(day2_pos)]
            day3_pos = date_to_pos.get(int(day1["date_index"]) + 2)
            day3 = group.iloc[int(day3_pos)] if day3_pos is not None else None
            candidate_id = f"{pd.Timestamp(day1['date']).strftime('%Y%m%d')}_{str(code)}"
            row = build_candidate_record(candidate_id, str(code), day1, day2, day3, prev_close.iloc[int(day1_pos)], prev_kama.iloc[int(day1_pos)])
            add_candidate_future_metrics(row, path_rows, candidate_id=candidate_id, group=group, date_to_pos=date_to_pos, day2=day2, max_hold_offset=max_hold_offset)
            rows.append(row)
    candidates = pd.DataFrame(rows)
    path = pd.DataFrame(path_rows)
    return candidates, path


def build_candidate_record(candidate_id: str, code: str, day1: pd.Series, day2: pd.Series, day3: pd.Series | None, prev_close: Any, prev_kama: Any) -> dict[str, Any]:
    day1_close = float(day1["close"])
    day2_open = float(day2["open"])
    day2_close = float(day2["close"])
    day2_open_gap_pct = _ret_pct(day2_open, day1_close)
    confirmed = bool(
        day2.get("limit_up_like", False)
        and _finite_gt(day2.get("close"), day2.get("atr_upper"))
        and _finite_le(day1.get("close"), day1.get("atr_upper"))
    )
    day3_open = float(day3["open"]) if day3 is not None and _is_finite(day3.get("open")) else np.nan
    day3_gap_pct = _ret_pct(day3_open, day2_close)
    amount_log = float(np.log10(max(float(day1.get("amount", 0.0) or 0.0), 1.0)))
    day1_strength = (
        _safe_float(day1.get("pctChg")) * 0.20
        + _ret_pct(day1.get("close"), day1.get("kama")) * 0.35
        + _safe_float(day1.get("signal_close_position")) * 2.0
        + amount_log * 0.05
    )
    return {
        "candidate_id": candidate_id,
        "code": code,
        "name_on_date": day2.get("name_on_date", day1.get("name_on_date", "")),
        "industry": day2.get("industry", day1.get("industry", "__unknown__")),
        "day1_date": day1["date"],
        "day1_year": int(pd.Timestamp(day1["date"]).year),
        "day1_close": day1["close"],
        "day1_pctChg": day1["pctChg"],
        "day1_amount": day1["amount"],
        "day1_amount_log10": amount_log,
        "day1_prev_close": prev_close,
        "day1_prev_kama": prev_kama,
        "day1_kama": day1["kama"],
        "day1_atr_upper": day1["atr_upper"],
        "day1_close_vs_kama_pct": _ret_pct(day1.get("close"), day1.get("kama")),
        "day1_close_vs_atr_upper_pct": _ret_pct(day1.get("close"), day1.get("atr_upper")),
        "day1_strength_score": day1_strength,
        "day1_board_stage": day1.get("board_stage", ""),
        "day2_date": day2["date"],
        "day2_year": int(pd.Timestamp(day2["date"]).year),
        "day2_open": day2["open"],
        "day2_close": day2["close"],
        "day2_pctChg": day2["pctChg"],
        "day2_open_gap_pct": day2_open_gap_pct,
        "day2_gap_bucket": bucket_gap(day2_open_gap_pct),
        "day2_limit_up_like": bool(day2.get("limit_up_like", False)),
        "day2_one_word_limit_like": bool(day2.get("one_word_limit_like", False)),
        "day2_near_one_word_limit_like": bool(day2.get("near_one_word_limit_like", False)),
        "day2_board_stage": day2.get("board_stage", ""),
        "day2_close_vs_atr_upper_pct": _ret_pct(day2.get("close"), day2.get("atr_upper")),
        "day2_atr_break_strength_bucket": bucket_atr_break_strength(_ret_pct(day2.get("close"), day2.get("atr_upper"))),
        "confirmed_two_day_breakout": confirmed,
        "day3_date": day3["date"] if day3 is not None else pd.NaT,
        "day3_open": day3_open,
        "day3_gap_pct": day3_gap_pct,
        "day3_gap_bucket": bucket_gap(day3_gap_pct),
        "day3_open_near_limit": bool(day3_gap_pct >= 9.5) if _is_finite(day3_gap_pct) else False,
        "day3_limit_up_like": bool(day3.get("limit_up_like", False)) if day3 is not None else False,
    }


def add_candidate_future_metrics(
    row: dict[str, Any],
    path_rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    group: pd.DataFrame,
    date_to_pos: dict[int, int],
    day2: pd.Series,
    max_hold_offset: int,
) -> None:
    day2_idx = int(day2["date_index"])
    entry_open = _safe_float(day2["open"])
    for offset in range(1, int(max_hold_offset) + 1):
        pos = date_to_pos.get(day2_idx + offset)
        if pos is None:
            continue
        future = group.iloc[int(pos)]
        path_rows.append(
            {
                "candidate_id": candidate_id,
                "future_offset": int(offset),
                "date": future["date"],
                "open": future["open"],
                "high": future["high"],
                "low": future["low"],
                "close": future["close"],
                "pctChg": future["pctChg"],
                "limit_up_like": bool(future.get("limit_up_like", False)),
                "close_ret_from_day2_open_pct": _ret_pct(future.get("close"), entry_open),
                "open_ret_from_day2_open_pct": _ret_pct(future.get("open"), entry_open),
                "high_ret_from_day2_open_pct": _ret_pct(future.get("high"), entry_open),
                "low_ret_from_day2_open_pct": _ret_pct(future.get("low"), entry_open),
            }
        )
        row[f"off{offset}_date"] = future["date"]
        row[f"off{offset}_open_ret_pct"] = _ret_pct(future.get("open"), entry_open)
        row[f"off{offset}_close_ret_pct"] = _ret_pct(future.get("close"), entry_open)
        row[f"off{offset}_high_ret_pct"] = _ret_pct(future.get("high"), entry_open)
        row[f"off{offset}_low_ret_pct"] = _ret_pct(future.get("low"), entry_open)


def materialize_strategy_trades(candidates: pd.DataFrame, *, spec: StrategySpec, fee_bps: float) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    frame = candidates.loc[entry_filter_mask(candidates, spec.entry_filter)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["entry_date"] = pd.to_datetime(frame["day2_date"])
    frame["exit_date"] = pd.NaT
    frame["gross_ret_pct"] = np.nan
    frame["exit_reason"] = ""
    fee_pct = float(fee_bps) / 100.0
    for idx, row in frame.iterrows():
        exit_date, gross_ret, reason = exit_for_policy(row, spec.exit_policy)
        frame.at[idx, "exit_date"] = exit_date
        frame.at[idx, "gross_ret_pct"] = gross_ret
        frame.at[idx, "exit_reason"] = reason
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame = frame.dropna(subset=["entry_date", "exit_date", "gross_ret_pct"])
    frame["net_ret_pct"] = frame["gross_ret_pct"] - fee_pct
    frame = frame.sort_values(["entry_date", spec.score_column], ascending=[True, False])
    selected = schedule_max_positions(frame, score_column=spec.score_column, max_positions=spec.max_positions)
    if selected.empty:
        return selected
    selected["strategy_id"] = spec.strategy_id
    selected["entry_filter"] = spec.entry_filter
    selected["exit_policy"] = spec.exit_policy
    selected["score_column"] = spec.score_column
    selected["max_positions"] = int(spec.max_positions)
    selected["fee_bps"] = float(fee_bps)
    return selected.reset_index(drop=True)


def entry_filter_mask(candidates: pd.DataFrame, name: str) -> pd.Series:
    gap = pd.to_numeric(candidates["day2_open_gap_pct"], errors="coerce")
    day1_atr_distance = pd.to_numeric(candidates["day1_close_vs_atr_upper_pct"], errors="coerce")
    day1_board = candidates["day1_board_stage"].astype(str)
    if name == "open_lt6":
        return gap.lt(6.0)
    if name == "open_0_6":
        return gap.between(0.0, 6.0, inclusive="both")
    if name == "open_m3_6":
        return gap.between(-3.0, 6.0, inclusive="both")
    if name == "open_m3_3":
        return gap.between(-3.0, 3.0, inclusive="both")
    if name == "open_3_6":
        return gap.between(3.0, 6.0, inclusive="both")
    if name == "fillable_open":
        return gap.lt(9.5)
    if name == "gap_le0":
        return gap.le(0.0)
    if name == "gap_m5_m3":
        return gap.between(-5.0, -3.0, inclusive="both")
    if name == "gap_le0_atr_below_m10":
        return gap.le(0.0) & day1_atr_distance.le(-10.0)
    if name == "gap_m5_0_atr_below_m10":
        return gap.between(-5.0, 0.0, inclusive="both") & day1_atr_distance.le(-10.0)
    if name == "gap_le0_second_atr_below_m5":
        return gap.le(0.0) & day1_board.eq("second_board") & day1_atr_distance.le(-5.0)
    if name == "gap_m5_0_second":
        return gap.between(-5.0, 0.0, inclusive="both") & day1_board.eq("second_board")
    raise ValueError(f"unknown entry filter: {name}")


def exit_for_policy(row: pd.Series, policy: str) -> tuple[Any, float, str]:
    confirmed = bool(row.get("confirmed_two_day_breakout", False))
    day3_near = bool(row.get("day3_open_near_limit", False))
    if policy == "confirm_day3_close_nonconfirm_day3_open":
        if confirmed:
            return row.get("off1_date"), _safe_float(row.get("off1_close_ret_pct")), "confirmed_day3_close"
        return row.get("off1_date"), _safe_float(row.get("off1_open_ret_pct")), "nonconfirm_day3_open"
    if policy == "confirm_near_extend_nonconfirm_day3_open":
        if confirmed and day3_near and _is_finite(row.get("off2_close_ret_pct")):
            return row.get("off2_date"), _safe_float(row.get("off2_close_ret_pct")), "confirmed_near_day4_close"
        if confirmed:
            return row.get("off1_date"), _safe_float(row.get("off1_close_ret_pct")), "confirmed_day3_close"
        return row.get("off1_date"), _safe_float(row.get("off1_open_ret_pct")), "nonconfirm_day3_open"
    if policy == "all_day3_close":
        return row.get("off1_date"), _safe_float(row.get("off1_close_ret_pct")), "day3_close"
    raise ValueError(f"unknown exit policy: {policy}")


def schedule_max_positions(frame: pd.DataFrame, *, score_column: str, max_positions: int) -> pd.DataFrame:
    open_exit_dates: list[pd.Timestamp] = []
    rows: list[pd.Series] = []
    for entry_date, group in frame.groupby("entry_date", sort=True):
        open_exit_dates = [date for date in open_exit_dates if date >= entry_date]
        slots = int(max_positions) - len(open_exit_dates)
        if slots <= 0:
            continue
        ranked = group.sort_values(score_column, ascending=False)
        for _, candidate in ranked.head(slots).iterrows():
            rows.append(candidate)
            open_exit_dates.append(pd.Timestamp(candidate["exit_date"]))
            if len(open_exit_dates) >= int(max_positions):
                break
    return pd.DataFrame(rows)


def summarize_strategy_trades(trades: pd.DataFrame, *, spec: StrategySpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "strategy_id": spec.strategy_id,
        "entry_filter": spec.entry_filter,
        "exit_policy": spec.exit_policy,
        "score_column": spec.score_column,
        "max_positions": int(spec.max_positions),
    }
    if trades.empty:
        return {
            **row,
            "trade_count": 0,
            "confirmed_trade_rate": np.nan,
            "mean_net_ret_pct": np.nan,
            "median_net_ret_pct": np.nan,
            "win_rate": np.nan,
            "p10_net_ret_pct": np.nan,
            "p90_net_ret_pct": np.nan,
            "year_count": 0,
            "positive_year_rate": np.nan,
            "min_year_mean_net_ret_pct": np.nan,
            "worst_trade_pct": np.nan,
            "score": -np.inf,
        }
    returns = pd.to_numeric(trades["net_ret_pct"], errors="coerce").dropna()
    yearly = trades.groupby(trades["entry_date"].dt.year)["net_ret_pct"].mean()
    positive_year_rate = float((yearly > 0).mean()) if not yearly.empty else np.nan
    min_year = float(yearly.min()) if not yearly.empty else np.nan
    score = float(returns.mean() + 0.25 * positive_year_rate * 100.0 + min(0.0, min_year)) if not returns.empty and np.isfinite(min_year) else -np.inf
    return {
        **row,
        "trade_count": int(len(trades)),
        "confirmed_trade_rate": float(trades["confirmed_two_day_breakout"].astype(bool).mean()) if "confirmed_two_day_breakout" in trades else np.nan,
        "mean_net_ret_pct": float(returns.mean()) if not returns.empty else np.nan,
        "median_net_ret_pct": float(returns.median()) if not returns.empty else np.nan,
        "win_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
        "p10_net_ret_pct": float(returns.quantile(0.10)) if not returns.empty else np.nan,
        "p90_net_ret_pct": float(returns.quantile(0.90)) if not returns.empty else np.nan,
        "year_count": int(yearly.index.nunique()),
        "positive_year_rate": positive_year_rate,
        "min_year_mean_net_ret_pct": min_year,
        "worst_trade_pct": float(returns.min()) if not returns.empty else np.nan,
        "score": score,
    }


def summarize_strategy_yearly(trades: pd.DataFrame, *, spec: StrategySpec) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for year, subset in trades.groupby(trades["entry_date"].dt.year):
        returns = pd.to_numeric(subset["net_ret_pct"], errors="coerce").dropna()
        rows.append(
            {
                "strategy_id": spec.strategy_id,
                "year": int(year),
                "trade_count": int(len(subset)),
                "confirmed_trade_rate": float(subset["confirmed_two_day_breakout"].astype(bool).mean()),
                "mean_net_ret_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_net_ret_pct": float(returns.median()) if not returns.empty else np.nan,
                "win_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
                "min_trade_pct": float(returns.min()) if not returns.empty else np.nan,
                "max_trade_pct": float(returns.max()) if not returns.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_strategy_grid(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    frame = grid.copy()
    frame = frame.loc[frame["trade_count"].ge(50) & frame["year_count"].ge(8)].copy()
    if frame.empty:
        return grid.sort_values("score", ascending=False).reset_index(drop=True)
    return frame.sort_values(
        ["mean_net_ret_pct", "positive_year_rate", "min_year_mean_net_ret_pct", "trade_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_summary(
    candidates: pd.DataFrame,
    grid: pd.DataFrame,
    ranked: pd.DataFrame,
    best_trades: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    manifest: dict[str, Any],
    quality: dict[str, Any],
    start_date: str,
    end_date: str,
    fee_bps: float,
    warmup_years: int,
    max_hold_offset: int,
    min_available_memory_gb: float,
    chunk_summary: pd.DataFrame,
) -> dict[str, Any]:
    best = ranked.iloc[0].to_dict() if not ranked.empty else {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "snapshot_id": manifest.get("snapshot_id") or quality.get("snapshot_id"),
        "start_date": start_date,
        "end_date": end_date,
        "fee_bps": float(fee_bps),
        "warmup_years": int(warmup_years),
        "max_hold_offset": int(max_hold_offset),
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
        "candidate_count": int(len(candidates)),
        "confirmed_candidate_count": int(candidates["confirmed_two_day_breakout"].sum()) if not candidates.empty else 0,
        "confirmed_candidate_rate": float(candidates["confirmed_two_day_breakout"].astype(bool).mean()) if not candidates.empty else np.nan,
        "strategy_count": int(len(grid)),
        "best_strategy": best,
        "best_trade_count": int(len(best_trades)),
        "best_year_count": int(best_trades["entry_date"].dt.year.nunique()) if not best_trades.empty else 0,
        "chunk_summary": chunk_summary.to_dict(orient="records") if not chunk_summary.empty else [],
        "decision": "strategy_research_candidate_diagnostic",
        "strategy_candidate_count": 0,
        "boundary": "Day2-open entry is an open-known/pre-confirmation strategy; confirmation is only known at Day2 close.",
    }


def render_markdown(summary: dict[str, Any], ranked: pd.DataFrame, yearly: pd.DataFrame, best_trades: pd.DataFrame) -> str:
    best_id = summary.get("best_strategy", {}).get("strategy_id", "")
    best_yearly = yearly.loc[yearly["strategy_id"].eq(best_id)].copy() if not yearly.empty and best_id else pd.DataFrame()
    lines = [
        "# Two-Day KAMA + ATR Strategy Research",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- snapshot_id: `{summary.get('snapshot_id')}`",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        f"- confirmed_candidate_count: `{summary.get('confirmed_candidate_count')}`",
        f"- confirmed_candidate_rate: `{_fmt(summary.get('confirmed_candidate_rate'))}`",
        f"- fee_bps: `{summary.get('fee_bps')}`",
        f"- best_strategy_id: `{best_id}`",
        f"- decision: `{summary.get('decision')}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count')}`",
        "",
        "## Strategy Template",
        "",
        "- Detect Day1 after close: limit-up, previous close below previous KAMA, and Day1 close above KAMA.",
        "- Try entry on Day2 open only when the open gap passes the selected filter.",
        "- Since A-share T+1 prevents same-day selling, non-confirmed trades are exited at Day3 open in the stricter policies.",
        "- Confirmed trades require Day2 limit-up plus close above ATR upper; confirmed trades are held to Day3 close, with one variant extending near-limit Day3 opens to Day4 close.",
        "- Portfolio scheduler enforces max concurrent holdings of 1 or 2 stocks.",
        "",
        "## Ranked Strategies",
        "",
        _markdown_table(
            ranked.head(20),
            [
                "strategy_id",
                "trade_count",
                "confirmed_trade_rate",
                "mean_net_ret_pct",
                "median_net_ret_pct",
                "win_rate",
                "p10_net_ret_pct",
                "p90_net_ret_pct",
                "positive_year_rate",
                "min_year_mean_net_ret_pct",
                "worst_trade_pct",
            ],
        ),
        "",
        "## Best Strategy Yearly",
        "",
        _markdown_table(
            best_yearly,
            ["year", "trade_count", "confirmed_trade_rate", "mean_net_ret_pct", "median_net_ret_pct", "win_rate", "min_trade_pct", "max_trade_pct"],
        ),
        "",
        "## Best Strategy Trades Sample",
        "",
        _markdown_table(
            best_trades.head(30),
            [
                "candidate_id",
                "code",
                "name_on_date",
                "entry_date",
                "exit_date",
                "entry_filter",
                "exit_reason",
                "confirmed_two_day_breakout",
                "day2_open_gap_pct",
                "net_ret_pct",
                "industry",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a strategy-research diagnostic, not an execution recommendation.",
        "- The highest-return profile depends on buying Day2 open before Day2 confirmation is known.",
        "- A real executable version must include order-fill constraints, non-confirmation loss control, position sizing, and live slippage checks.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    output = frame.copy()
    if columns is not None:
        output = output[[column for column in columns if column in output.columns]]
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    return output.to_markdown(index=False)


def _ret_pct(value: Any, base: Any) -> float:
    value = _safe_float(value)
    base = _safe_float(base)
    if not _is_finite(value) or not _is_finite(base) or base <= 0:
        return np.nan
    return (value / base - 1.0) * 100.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _finite_gt(left: Any, right: Any) -> bool:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    return _is_finite(left_value) and _is_finite(right_value) and left_value > right_value


def _finite_le(left: Any, right: Any) -> bool:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    return _is_finite(left_value) and _is_finite(right_value) and left_value <= right_value


def _fmt(value: Any) -> str:
    value = _safe_float(value)
    return "" if not _is_finite(value) else f"{value:.6g}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--warmup-years", type=int, default=DEFAULT_WARMUP_YEARS)
    parser.add_argument("--max-hold-offset", type=int, default=DEFAULT_MAX_HOLD_OFFSET)
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_two_day_kama_atr_strategy_research(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        fee_bps=args.fee_bps,
        warmup_years=args.warmup_years,
        max_hold_offset=args.max_hold_offset,
        min_available_memory_gb=args.min_available_memory_gb,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
