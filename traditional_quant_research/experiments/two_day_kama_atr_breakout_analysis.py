"""Analyze two-day limit-up KAMA and ATR-upper breakout follow-through."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.experiments.short_open_known_factor_rebuild import prepare_short_factor_panel


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/two_day_kama_atr_breakout_analysis")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-07_two_day_kama_atr_breakout_analysis.md")
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2026-06-01"
DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 20, 30, 60)
SUMMARY_HORIZONS = (1, 3, 5, 10, 20, 30, 60)
DEFAULT_WARMUP_YEARS = 1
DEFAULT_MIN_AVAILABLE_MEMORY_GB = 1.0
DEFAULT_DATA_START_YEAR = 2016


def run_two_day_kama_atr_breakout_analysis(
    *,
    root: str | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    require_day2_cross_from_below: bool = True,
    warmup_years: int = DEFAULT_WARMUP_YEARS,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
) -> dict[str, Any]:
    horizon_values = _parse_int_values(horizons, "horizons")
    if not horizon_values:
        raise ValueError("horizons must not be empty")
    run_id = f"two_day_kama_atr_breakout_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    events, future_path, chunk_summary = build_two_day_breakout_events_chunked(
        root=root,
        start_date=start_date,
        end_date=end_date,
        horizons=horizon_values,
        require_day2_cross_from_below=require_day2_cross_from_below,
        warmup_years=warmup_years,
        min_available_memory_gb=min_available_memory_gb,
    )
    horizon_summary = summarize_horizons(events, horizons=horizon_values)
    yearly_summary = summarize_by_group(events, ["day2_year"], horizons=SUMMARY_HORIZONS)
    monthly_summary = summarize_by_group(events, ["day2_year_month"], horizons=SUMMARY_HORIZONS)
    industry_summary = summarize_by_group(events, ["industry"], horizons=SUMMARY_HORIZONS).sort_values(
        ["event_count", "close_ret_20d_mean_pct"], ascending=[False, False], na_position="last"
    )
    board_summary = summarize_by_group(events, ["day1_board_stage", "day2_board_stage"], horizons=SUMMARY_HORIZONS)
    day3_gap_summary = summarize_bucket(events, "day3_gap_bucket", horizons=SUMMARY_HORIZONS)
    day2_atr_distance_summary = summarize_bucket(events, "day2_atr_break_strength_bucket", horizons=SUMMARY_HORIZONS)
    continuation_summary = summarize_continuation(events)

    summary = build_summary(
        events,
        future_path,
        horizon_summary,
        yearly_summary,
        industry_summary,
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        quality=quality,
        start_date=start_date,
        end_date=end_date,
        horizons=horizon_values,
        require_day2_cross_from_below=require_day2_cross_from_below,
        warmup_years=warmup_years,
        min_available_memory_gb=min_available_memory_gb,
        chunk_summary=chunk_summary,
    )
    markdown = render_markdown(
        summary,
        horizon_summary,
        yearly_summary,
        industry_summary,
        board_summary,
        day3_gap_summary,
        day2_atr_distance_summary,
        continuation_summary,
        events,
    )

    events.to_csv(run_dir / "two_day_breakout_events.csv", index=False, encoding="utf-8-sig")
    future_path.to_csv(run_dir / "two_day_breakout_future_path.csv", index=False, encoding="utf-8-sig")
    horizon_summary.to_csv(run_dir / "horizon_summary.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(run_dir / "yearly_summary.csv", index=False, encoding="utf-8-sig")
    monthly_summary.to_csv(run_dir / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    industry_summary.to_csv(run_dir / "industry_summary.csv", index=False, encoding="utf-8-sig")
    board_summary.to_csv(run_dir / "board_stage_summary.csv", index=False, encoding="utf-8-sig")
    day3_gap_summary.to_csv(run_dir / "day3_gap_summary.csv", index=False, encoding="utf-8-sig")
    day2_atr_distance_summary.to_csv(run_dir / "day2_atr_distance_summary.csv", index=False, encoding="utf-8-sig")
    continuation_summary.to_csv(run_dir / "continuation_summary.csv", index=False, encoding="utf-8-sig")
    chunk_summary.to_csv(run_dir / "chunk_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path is not None else None}


def build_two_day_breakout_events_from_feature_panel(
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    require_day2_cross_from_below: bool = True,
    day1_start_date: str | None = None,
    day1_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {
        "date",
        "code",
        "date_index",
        "open",
        "high",
        "low",
        "close",
        "pctChg",
        "amount",
        "turn",
        "kama",
        "atr_upper",
        "limit_up_like",
        "one_word_limit_like",
        "near_one_word_limit_like",
        "limit_up_run_ending_today",
        "board_stage",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"feature panel missing columns: {missing}")
    horizon_values = _parse_int_values(horizons, "horizons")
    max_horizon = max(horizon_values)
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["code", "date_index", "date"]).reset_index(drop=True)
    for column in ["open", "high", "low", "close", "pctChg", "amount", "turn", "kama", "atr_upper"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["limit_up_like", "one_word_limit_like", "near_one_word_limit_like"]:
        frame[column] = frame[column].fillna(False).astype(bool)

    records: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    for code, group in frame.groupby("code", sort=False):
        group = group.sort_values("date_index").reset_index(drop=True)
        date_to_pos = {int(value): pos for pos, value in enumerate(group["date_index"].to_numpy(dtype=int))}
        prev_close = group["close"].shift(1)
        prev_kama = group["kama"].shift(1)
        day1_mask = (
            group["limit_up_like"].astype(bool)
            & prev_close.lt(prev_kama)
            & group["close"].gt(group["kama"])
        )
        if day1_start_date is not None:
            day1_mask &= group["date"].ge(pd.Timestamp(day1_start_date))
        if day1_end_date is not None:
            day1_mask &= group["date"].le(pd.Timestamp(day1_end_date))
        for day1_pos in np.flatnonzero(day1_mask.to_numpy(dtype=bool)):
            day1 = group.iloc[int(day1_pos)]
            day2_pos = date_to_pos.get(int(day1["date_index"]) + 1)
            if day2_pos is None:
                continue
            day2 = group.iloc[int(day2_pos)]
            if not bool(day2["limit_up_like"]):
                continue
            if not _finite_gt(day2["close"], day2["atr_upper"]):
                continue
            if require_day2_cross_from_below and not _finite_le(day1["close"], day1["atr_upper"]):
                continue
            event_id = f"{pd.Timestamp(day1['date']).strftime('%Y%m%d')}_{str(code)}"
            record = _base_event_record(event_id, code, day1, day2, prev_close.iloc[int(day1_pos)], prev_kama.iloc[int(day1_pos)])
            day3_pos = date_to_pos.get(int(day2["date_index"]) + 1)
            day3 = group.iloc[int(day3_pos)] if day3_pos is not None else None
            _add_day3_fields(record, day2, day3)
            _add_future_metrics(
                record,
                path_records,
                event_id=event_id,
                group=group,
                date_to_pos=date_to_pos,
                day2=day2,
                day3=day3,
                horizons=horizon_values,
                max_horizon=max_horizon,
            )
            records.append(record)

    events = pd.DataFrame(records)
    future_path = pd.DataFrame(path_records)
    if not events.empty:
        events = events.sort_values(["day1_date", "code"]).reset_index(drop=True)
    if not future_path.empty:
        future_path = future_path.sort_values(["event_id", "future_offset"]).reset_index(drop=True)
    return events, future_path


def build_two_day_breakout_events_chunked(
    *,
    root: str | None,
    start_date: str,
    end_date: str,
    horizons: Sequence[int],
    require_day2_cross_from_below: bool,
    warmup_years: int,
    min_available_memory_gb: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")
    horizon_values = _parse_int_values(horizons, "horizons")
    max_horizon = max(horizon_values)
    event_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    chunk_rows: list[dict[str, Any]] = []
    for year in range(int(start.year), int(end.year) + 1):
        day1_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        day1_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if day1_end < day1_start:
            continue
        chunk_start_year = max(DEFAULT_DATA_START_YEAR, year - max(0, int(warmup_years)))
        chunk_start = pd.Timestamp(year=chunk_start_year, month=1, day=1)
        future_calendar_days = max(120, int(max_horizon) * 3)
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=future_calendar_days))
        _assert_memory_available(min_available_memory_gb, context=f"before loading year {year}")
        raw_panel = load_tradeable_panel(
            root,
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            include_industry=True,
            include_metrics=True,
        )
        _assert_memory_available(min_available_memory_gb, context=f"after loading year {year}")
        feature_panel = prepare_short_factor_panel(raw_panel)
        del raw_panel
        _assert_memory_available(min_available_memory_gb, context=f"after feature build year {year}")
        events, future_path = build_two_day_breakout_events_from_feature_panel(
            feature_panel,
            horizons=horizon_values,
            require_day2_cross_from_below=require_day2_cross_from_below,
            day1_start_date=day1_start.strftime("%Y-%m-%d"),
            day1_end_date=day1_end.strftime("%Y-%m-%d"),
        )
        chunk_rows.append(
            {
                "year": int(year),
                "chunk_start": chunk_start.strftime("%Y-%m-%d"),
                "chunk_end": chunk_end.strftime("%Y-%m-%d"),
                "day1_start": day1_start.strftime("%Y-%m-%d"),
                "day1_end": day1_end.strftime("%Y-%m-%d"),
                "feature_rows": int(len(feature_panel)),
                "event_count": int(len(events)),
                "future_path_rows": int(len(future_path)),
                "available_memory_gb_after_chunk": available_memory_gb(),
            }
        )
        if not events.empty:
            event_frames.append(events)
        if not future_path.empty:
            path_frames.append(future_path)
        del feature_panel, events, future_path
        gc.collect()
        _assert_memory_available(min_available_memory_gb, context=f"after releasing year {year}")
    all_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    all_path = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    if not all_events.empty:
        all_events = all_events.sort_values(["day1_date", "code"]).reset_index(drop=True)
    if not all_path.empty:
        all_path = all_path.sort_values(["event_id", "future_offset"]).reset_index(drop=True)
    return all_events, all_path, pd.DataFrame(chunk_rows)


def _base_event_record(event_id: str, code: str, day1: pd.Series, day2: pd.Series, prev_close: Any, prev_kama: Any) -> dict[str, Any]:
    day1_close = float(day1["close"])
    day2_close = float(day2["close"])
    day1_atr_upper = _as_float(day1.get("atr_upper"))
    day2_atr_upper = _as_float(day2.get("atr_upper"))
    return {
        "event_id": event_id,
        "code": code,
        "name_on_date": day2.get("name_on_date", day1.get("name_on_date", "")),
        "industry": day2.get("industry", day1.get("industry", "__unknown__")),
        "day1_date": day1["date"],
        "day1_year": int(pd.Timestamp(day1["date"]).year),
        "day1_open": day1["open"],
        "day1_high": day1["high"],
        "day1_low": day1["low"],
        "day1_close": day1["close"],
        "day1_pctChg": day1["pctChg"],
        "day1_turn": day1["turn"],
        "day1_amount": day1["amount"],
        "day1_prev_close": prev_close,
        "day1_prev_kama": prev_kama,
        "day1_kama": day1["kama"],
        "day1_atr_upper": day1["atr_upper"],
        "day1_close_vs_kama_pct": (day1_close / float(day1["kama"]) - 1.0) * 100.0 if _is_finite(day1["kama"]) else np.nan,
        "day1_close_vs_atr_upper_pct": (day1_close / day1_atr_upper - 1.0) * 100.0 if _is_finite(day1_atr_upper) else np.nan,
        "day1_limit_up_run_ending_today": int(day1["limit_up_run_ending_today"]),
        "day1_board_stage": day1.get("board_stage", ""),
        "day1_one_word_limit_like": bool(day1["one_word_limit_like"]),
        "day1_near_one_word_limit_like": bool(day1["near_one_word_limit_like"]),
        "day2_date": day2["date"],
        "day2_year": int(pd.Timestamp(day2["date"]).year),
        "day2_year_month": pd.Timestamp(day2["date"]).strftime("%Y-%m"),
        "day2_open": day2["open"],
        "day2_high": day2["high"],
        "day2_low": day2["low"],
        "day2_close": day2["close"],
        "day2_pctChg": day2["pctChg"],
        "day2_turn": day2["turn"],
        "day2_amount": day2["amount"],
        "day2_kama": day2["kama"],
        "day2_atr_upper": day2["atr_upper"],
        "day2_open_gap_pct": (float(day2["open"]) / day1_close - 1.0) * 100.0 if _is_finite(day1_close) else np.nan,
        "day2_close_ret_from_day1_close_pct": (day2_close / day1_close - 1.0) * 100.0 if _is_finite(day1_close) else np.nan,
        "day2_close_vs_atr_upper_pct": (day2_close / day2_atr_upper - 1.0) * 100.0 if _is_finite(day2_atr_upper) else np.nan,
        "day2_close_vs_kama_pct": (day2_close / float(day2["kama"]) - 1.0) * 100.0 if _is_finite(day2["kama"]) else np.nan,
        "day2_limit_up_run_ending_today": int(day2["limit_up_run_ending_today"]),
        "day2_board_stage": day2.get("board_stage", ""),
        "day2_one_word_limit_like": bool(day2["one_word_limit_like"]),
        "day2_near_one_word_limit_like": bool(day2["near_one_word_limit_like"]),
        "day2_atr_break_from_below": bool(_finite_le(day1["close"], day1["atr_upper"]) and _finite_gt(day2["close"], day2["atr_upper"])),
        "day2_atr_break_strength_bucket": bucket_atr_break_strength((day2_close / day2_atr_upper - 1.0) * 100.0 if _is_finite(day2_atr_upper) else np.nan),
    }


def _add_day3_fields(record: dict[str, Any], day2: pd.Series, day3: pd.Series | None) -> None:
    if day3 is None:
        record.update(
            {
                "day3_date": pd.NaT,
                "day3_open": np.nan,
                "day3_close": np.nan,
                "day3_open_gap_pct": np.nan,
                "day3_intraday_close_ret_pct": np.nan,
                "day3_limit_up_like": False,
                "day3_one_word_limit_like": False,
                "day3_near_one_word_limit_like": False,
                "day3_open_near_limit": False,
                "day3_gap_bucket": "missing",
            }
        )
        return
    day2_close = float(day2["close"])
    day3_open = float(day3["open"])
    day3_open_gap_pct = (day3_open / day2_close - 1.0) * 100.0 if _is_finite(day2_close) else np.nan
    record.update(
        {
            "day3_date": day3["date"],
            "day3_open": day3["open"],
            "day3_close": day3["close"],
            "day3_open_gap_pct": day3_open_gap_pct,
            "day3_intraday_close_ret_pct": (float(day3["close"]) / day3_open - 1.0) * 100.0 if _is_finite(day3_open) else np.nan,
            "day3_limit_up_like": bool(day3["limit_up_like"]),
            "day3_one_word_limit_like": bool(day3["one_word_limit_like"]),
            "day3_near_one_word_limit_like": bool(day3["near_one_word_limit_like"]),
            "day3_open_near_limit": bool(day3_open_gap_pct >= 9.5) if _is_finite(day3_open_gap_pct) else False,
            "day3_gap_bucket": bucket_gap(day3_open_gap_pct),
        }
    )


def _add_future_metrics(
    record: dict[str, Any],
    path_records: list[dict[str, Any]],
    *,
    event_id: str,
    group: pd.DataFrame,
    date_to_pos: dict[int, int],
    day2: pd.Series,
    day3: pd.Series | None,
    horizons: Sequence[int],
    max_horizon: int,
) -> None:
    day2_idx = int(day2["date_index"])
    day2_close = float(day2["close"])
    day3_open = float(day3["open"]) if day3 is not None and _is_finite(day3["open"]) else np.nan
    continuation = 0
    for offset in range(1, max_horizon + 1):
        pos = date_to_pos.get(day2_idx + offset)
        if pos is None:
            continue
        row = group.iloc[int(pos)]
        if offset == continuation + 1 and bool(row["limit_up_like"]):
            continuation += 1
        path_records.append(
            {
                "event_id": event_id,
                "future_offset": int(offset),
                "date": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "pctChg": row["pctChg"],
                "limit_up_like": bool(row["limit_up_like"]),
                "one_word_limit_like": bool(row["one_word_limit_like"]),
                "near_one_word_limit_like": bool(row["near_one_word_limit_like"]),
                "close_ret_from_day2_close_pct": _ret_pct(row["close"], day2_close),
                "high_ret_from_day2_close_pct": _ret_pct(row["high"], day2_close),
                "low_ret_from_day2_close_pct": _ret_pct(row["low"], day2_close),
                "close_ret_from_day3_open_pct": _ret_pct(row["close"], day3_open),
                "high_ret_from_day3_open_pct": _ret_pct(row["high"], day3_open),
                "low_ret_from_day3_open_pct": _ret_pct(row["low"], day3_open),
            }
        )
    record["continuation_limitup_days_after_day2"] = int(continuation)
    for horizon in horizons:
        future = _future_slice(group, date_to_pos, start_idx=day2_idx + 1, end_idx=day2_idx + int(horizon))
        close_pos = date_to_pos.get(day2_idx + int(horizon))
        if future.empty or close_pos is None:
            _set_horizon_nan(record, int(horizon))
            continue
        close_row = group.iloc[int(close_pos)]
        record[f"valid_future_rows_{int(horizon)}d"] = int(len(future))
        record[f"close_ret_{int(horizon)}d_from_day2_close_pct"] = _ret_pct(close_row["close"], day2_close)
        record[f"max_high_ret_{int(horizon)}d_from_day2_close_pct"] = _ret_pct(future["high"].max(), day2_close)
        record[f"min_low_ret_{int(horizon)}d_from_day2_close_pct"] = _ret_pct(future["low"].min(), day2_close)
        record[f"close_ret_{int(horizon)}d_from_day3_open_pct"] = _ret_pct(close_row["close"], day3_open)
        record[f"max_high_ret_{int(horizon)}d_from_day3_open_pct"] = _ret_pct(future["high"].max(), day3_open)
        record[f"min_low_ret_{int(horizon)}d_from_day3_open_pct"] = _ret_pct(future["low"].min(), day3_open)
        record[f"limitup_count_next_{int(horizon)}d"] = int(future["limit_up_like"].fillna(False).astype(bool).sum())
        record[f"first_limitup_offset_next_{int(horizon)}d"] = _first_offset(future, day2_idx, "limit_up_like")
        record[f"first_negative_close_offset_next_{int(horizon)}d"] = _first_negative_close_offset(future, day2_close, day2_idx)


def _future_slice(group: pd.DataFrame, date_to_pos: dict[int, int], *, start_idx: int, end_idx: int) -> pd.DataFrame:
    positions = [date_to_pos[index] for index in range(int(start_idx), int(end_idx) + 1) if index in date_to_pos]
    if not positions:
        return pd.DataFrame(columns=group.columns)
    return group.iloc[positions].copy()


def _set_horizon_nan(record: dict[str, Any], horizon: int) -> None:
    for prefix in [
        "close_ret",
        "max_high_ret",
        "min_low_ret",
    ]:
        record[f"{prefix}_{horizon}d_from_day2_close_pct"] = np.nan
        record[f"{prefix}_{horizon}d_from_day3_open_pct"] = np.nan
    record[f"valid_future_rows_{horizon}d"] = 0
    record[f"limitup_count_next_{horizon}d"] = np.nan
    record[f"first_limitup_offset_next_{horizon}d"] = np.nan
    record[f"first_negative_close_offset_next_{horizon}d"] = np.nan


def summarize_horizons(events: pd.DataFrame, *, horizons: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in _parse_int_values(horizons, "horizons"):
        close_col = f"close_ret_{horizon}d_from_day2_close_pct"
        high_col = f"max_high_ret_{horizon}d_from_day2_close_pct"
        low_col = f"min_low_ret_{horizon}d_from_day2_close_pct"
        entry_close_col = f"close_ret_{horizon}d_from_day3_open_pct"
        entry_high_col = f"max_high_ret_{horizon}d_from_day3_open_pct"
        entry_low_col = f"min_low_ret_{horizon}d_from_day3_open_pct"
        rows.append(
            {
                "horizon": horizon,
                "event_count": int(len(events)),
                **_series_stats(events.get(close_col), prefix="close_ret"),
                **_series_stats(events.get(high_col), prefix="max_high_ret"),
                **_series_stats(events.get(low_col), prefix="min_low_ret"),
                **_series_stats(events.get(entry_close_col), prefix="entry_close_ret"),
                **_series_stats(events.get(entry_high_col), prefix="entry_max_high_ret"),
                **_series_stats(events.get(entry_low_col), prefix="entry_min_low_ret"),
                "close_win_rate": _rate(events.get(close_col), lambda s: s > 0),
                "close_ge_5_rate": _rate(events.get(close_col), lambda s: s >= 5),
                "close_ge_10_rate": _rate(events.get(close_col), lambda s: s >= 10),
                "close_le_minus5_rate": _rate(events.get(close_col), lambda s: s <= -5),
                "close_le_minus10_rate": _rate(events.get(close_col), lambda s: s <= -10),
                "max_high_ge_10_rate": _rate(events.get(high_col), lambda s: s >= 10),
                "max_high_ge_20_rate": _rate(events.get(high_col), lambda s: s >= 20),
                "min_low_le_minus5_rate": _rate(events.get(low_col), lambda s: s <= -5),
                "min_low_le_minus10_rate": _rate(events.get(low_col), lambda s: s <= -10),
                "next_limitup_rate": _rate(events.get(f"limitup_count_next_{horizon}d"), lambda s: s >= 1),
                "mean_limitup_count": _mean(events.get(f"limitup_count_next_{horizon}d")),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_group(events: pd.DataFrame, group_cols: Sequence[str], *, horizons: Sequence[int]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, subset in events.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row["event_count"] = int(len(subset))
        row["day3_open_gap_mean_pct"] = _mean(subset.get("day3_open_gap_pct"))
        row["day3_open_near_limit_rate"] = _rate(subset.get("day3_open_near_limit"), lambda s: s.astype(bool))
        row["continuation_limitup_mean"] = _mean(subset.get("continuation_limitup_days_after_day2"))
        for horizon in _parse_int_values(horizons, "horizons"):
            close_col = f"close_ret_{horizon}d_from_day2_close_pct"
            high_col = f"max_high_ret_{horizon}d_from_day2_close_pct"
            low_col = f"min_low_ret_{horizon}d_from_day2_close_pct"
            row[f"close_ret_{horizon}d_mean_pct"] = _mean(subset.get(close_col))
            row[f"close_ret_{horizon}d_median_pct"] = _quantile(subset.get(close_col), 0.50)
            row[f"close_ret_{horizon}d_win_rate"] = _rate(subset.get(close_col), lambda s: s > 0)
            row[f"max_high_ret_{horizon}d_mean_pct"] = _mean(subset.get(high_col))
            row[f"min_low_ret_{horizon}d_mean_pct"] = _mean(subset.get(low_col))
            row[f"next_limitup_{horizon}d_rate"] = _rate(subset.get(f"limitup_count_next_{horizon}d"), lambda s: s >= 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def summarize_bucket(events: pd.DataFrame, bucket_col: str, *, horizons: Sequence[int]) -> pd.DataFrame:
    if bucket_col not in events.columns:
        return pd.DataFrame()
    return summarize_by_group(events, [bucket_col], horizons=horizons).sort_values("event_count", ascending=False)


def summarize_continuation(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    streak = pd.to_numeric(events["continuation_limitup_days_after_day2"], errors="coerce").fillna(0).astype(int)
    rows = []
    for threshold in [0, 1, 2, 3, 4, 5]:
        if threshold == 0:
            mask = streak.eq(0)
            label = "no_more_limitup_after_day2"
        else:
            mask = streak.ge(threshold)
            label = f"at_least_{threshold}_more_limitup"
        subset = events.loc[mask]
        rows.append(
            {
                "continuation_bucket": label,
                "event_count": int(len(subset)),
                "event_rate": float(len(subset) / len(events)) if len(events) else np.nan,
                "close_ret_5d_mean_pct": _mean(subset.get("close_ret_5d_from_day2_close_pct")),
                "close_ret_20d_mean_pct": _mean(subset.get("close_ret_20d_from_day2_close_pct")),
                "max_high_ret_20d_mean_pct": _mean(subset.get("max_high_ret_20d_from_day2_close_pct")),
                "min_low_ret_20d_mean_pct": _mean(subset.get("min_low_ret_20d_from_day2_close_pct")),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    events: pd.DataFrame,
    future_path: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    manifest: dict[str, Any],
    quality: dict[str, Any],
    start_date: str,
    end_date: str,
    horizons: Sequence[int],
    require_day2_cross_from_below: bool,
    warmup_years: int,
    min_available_memory_gb: float,
    chunk_summary: pd.DataFrame,
) -> dict[str, Any]:
    best_20 = pd.DataFrame()
    worst_20 = pd.DataFrame()
    if not events.empty and "close_ret_20d_from_day2_close_pct" in events.columns:
        best_20 = events.nlargest(10, "close_ret_20d_from_day2_close_pct")[
            ["event_id", "code", "name_on_date", "day2_date", "industry", "close_ret_20d_from_day2_close_pct"]
        ]
        worst_20 = events.nsmallest(10, "close_ret_20d_from_day2_close_pct")[
            ["event_id", "code", "name_on_date", "day2_date", "industry", "close_ret_20d_from_day2_close_pct"]
        ]
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "snapshot_id": manifest.get("snapshot_id") or quality.get("snapshot_id"),
        "start_date": start_date,
        "end_date": end_date,
        "horizons": list(horizons),
        "warmup_years": int(warmup_years),
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
        "event_definition": {
            "day1": "limit_up_like and previous close < previous KAMA and day1 close > day1 KAMA",
            "day2": "next market trading day for the same stock is limit_up_like and day2 close > day2 ATR upper",
            "day2_cross_from_below_required": bool(require_day2_cross_from_below),
            "atr_upper": "EMA20(close) + 2 * ATR14, reused from short_open_known_factor_rebuild.py",
            "future_return_anchor": "day2 close; entry-style columns also use day3 open",
        },
        "event_count": int(len(events)),
        "future_path_rows": int(len(future_path)),
        "date_min": str(pd.to_datetime(events["day1_date"]).min().date()) if not events.empty else "",
        "date_max": str(pd.to_datetime(events["day1_date"]).max().date()) if not events.empty else "",
        "code_count": int(events["code"].nunique()) if not events.empty else 0,
        "industry_count": int(events["industry"].nunique()) if not events.empty and "industry" in events.columns else 0,
        "day3_open_near_limit_rate": _rate(events.get("day3_open_near_limit"), lambda s: s.astype(bool)),
        "day3_one_word_rate": _rate(events.get("day3_one_word_limit_like"), lambda s: s.astype(bool)),
        "day2_one_word_rate": _rate(events.get("day2_one_word_limit_like"), lambda s: s.astype(bool)),
        "day2_near_one_word_rate": _rate(events.get("day2_near_one_word_limit_like"), lambda s: s.astype(bool)),
        "continuation_limitup_mean": _mean(events.get("continuation_limitup_days_after_day2")),
        "horizon_summary": horizon_summary.to_dict(orient="records"),
        "chunk_summary": chunk_summary.to_dict(orient="records") if not chunk_summary.empty else [],
        "yearly_event_count": yearly_summary[["day2_year", "event_count"]].to_dict(orient="records")
        if not yearly_summary.empty and "day2_year" in yearly_summary.columns
        else [],
        "top_industries_by_count": industry_summary.head(15).to_dict(orient="records") if not industry_summary.empty else [],
        "best_20d_events": best_20.to_dict(orient="records"),
        "worst_20d_events": worst_20.to_dict(orient="records"),
        "decision": "diagnostic_follow_through_analysis",
        "candidate_count": 0,
    }


def render_markdown(
    summary: dict[str, Any],
    horizon_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    board_summary: pd.DataFrame,
    day3_gap_summary: pd.DataFrame,
    day2_atr_distance_summary: pd.DataFrame,
    continuation_summary: pd.DataFrame,
    events: pd.DataFrame,
) -> str:
    lines = [
        "# Two-Day KAMA + ATR-Upper Limit-Up Follow-Through Analysis",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- snapshot_id: `{summary.get('snapshot_id')}`",
        f"- sample_range: `{summary.get('start_date')}` to `{summary.get('end_date')}`",
        f"- event_count: `{summary.get('event_count')}`",
        f"- code_count: `{summary.get('code_count')}`",
        f"- industry_count: `{summary.get('industry_count')}`",
        f"- decision: `{summary.get('decision')}`",
        f"- candidate_count: `{summary.get('candidate_count')}`",
        "",
        "## Definition",
        "",
        "- Day1: limit-up, previous close below previous KAMA, and Day1 close above Day1 KAMA.",
        "- Day2: the next market trading day for the same stock is again limit-up and closes above ATR upper.",
        f"- Day2 ATR cross from below required: `{summary['event_definition']['day2_cross_from_below_required']}`.",
        "- ATR upper: EMA20(close) + 2 * ATR14.",
        "- Follow-through returns use Day2 close as the primary anchor; Day3-open anchored columns are included for post-signal entry diagnostics.",
        "",
        "## Horizon Summary",
        "",
        _markdown_table(
            horizon_summary,
            [
                "horizon",
                "close_ret_valid_n",
                "close_ret_mean",
                "close_ret_median",
                "close_win_rate",
                "max_high_ret_mean",
                "min_low_ret_mean",
                "max_high_ge_10_rate",
                "min_low_le_minus10_rate",
                "next_limitup_rate",
            ],
        ),
        "",
        "## Yearly Summary",
        "",
        _markdown_table(
            yearly_summary,
            [
                "day2_year",
                "event_count",
                "day3_open_gap_mean_pct",
                "close_ret_1d_mean_pct",
                "close_ret_5d_mean_pct",
                "close_ret_20d_mean_pct",
                "close_ret_20d_win_rate",
                "max_high_ret_20d_mean_pct",
                "min_low_ret_20d_mean_pct",
            ],
        ),
        "",
        "## Board Stage Summary",
        "",
        _markdown_table(
            board_summary,
            [
                "day1_board_stage",
                "day2_board_stage",
                "event_count",
                "close_ret_5d_mean_pct",
                "close_ret_20d_mean_pct",
                "close_ret_20d_win_rate",
                "max_high_ret_20d_mean_pct",
                "min_low_ret_20d_mean_pct",
            ],
        ),
        "",
        "## Day3 Gap Summary",
        "",
        _markdown_table(
            day3_gap_summary,
            [
                "day3_gap_bucket",
                "event_count",
                "day3_open_gap_mean_pct",
                "close_ret_1d_mean_pct",
                "close_ret_5d_mean_pct",
                "close_ret_20d_mean_pct",
                "max_high_ret_20d_mean_pct",
                "min_low_ret_20d_mean_pct",
            ],
        ),
        "",
        "## ATR Break Strength Summary",
        "",
        _markdown_table(
            day2_atr_distance_summary,
            [
                "day2_atr_break_strength_bucket",
                "event_count",
                "close_ret_1d_mean_pct",
                "close_ret_5d_mean_pct",
                "close_ret_20d_mean_pct",
                "max_high_ret_20d_mean_pct",
                "min_low_ret_20d_mean_pct",
            ],
        ),
        "",
        "## Continuation Summary",
        "",
        _markdown_table(continuation_summary),
        "",
        "## Top Industries By Count",
        "",
        _markdown_table(
            industry_summary.head(20),
            [
                "industry",
                "event_count",
                "close_ret_5d_mean_pct",
                "close_ret_20d_mean_pct",
                "close_ret_20d_win_rate",
                "max_high_ret_20d_mean_pct",
                "min_low_ret_20d_mean_pct",
            ],
        ),
        "",
        "## Best And Worst 20D Close Follow-Through",
        "",
        "Best 20D close returns:",
        "",
        _markdown_table(
            events.nlargest(10, "close_ret_20d_from_day2_close_pct")[
                ["event_id", "code", "name_on_date", "day2_date", "industry", "close_ret_20d_from_day2_close_pct"]
            ]
            if not events.empty and "close_ret_20d_from_day2_close_pct" in events.columns
            else pd.DataFrame()
        ),
        "",
        "Worst 20D close returns:",
        "",
        _markdown_table(
            events.nsmallest(10, "close_ret_20d_from_day2_close_pct")[
                ["event_id", "code", "name_on_date", "day2_date", "industry", "close_ret_20d_from_day2_close_pct"]
            ]
            if not events.empty and "close_ret_20d_from_day2_close_pct" in events.columns
            else pd.DataFrame()
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a diagnostic event-study artifact, not a strategy candidate or execution recommendation.",
        "- The sample includes only tradeable mainboard A-share rows available in the local PIT snapshot.",
        "- One-word, near-one-word, and Day3 near-limit openings are retained in the event set and should be reviewed before any executable interpretation.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None, *, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    output = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in output.columns]
        output = output[available]
    output = output.head(max_rows)
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    return output.to_markdown(index=False)


def bucket_gap(value: Any) -> str:
    value = _as_float(value)
    if not _is_finite(value):
        return "missing"
    if value >= 9.5:
        return "near_limit_up_open"
    if value >= 6:
        return "gap_ge_6"
    if value >= 3:
        return "gap_3_to_6"
    if value > 0:
        return "gap_0_to_3"
    if value >= -3:
        return "gap_minus3_to_0"
    return "gap_le_minus3"


def bucket_atr_break_strength(value: Any) -> str:
    value = _as_float(value)
    if not _is_finite(value):
        return "missing"
    if value < 0:
        return "below_upper"
    if value <= 1:
        return "0_to_1_pct"
    if value <= 3:
        return "1_to_3_pct"
    if value <= 6:
        return "3_to_6_pct"
    return "gt_6_pct"


def _series_stats(series: Any, *, prefix: str) -> dict[str, float | int]:
    values = _numeric_series(series)
    return {
        f"{prefix}_valid_n": int(values.notna().sum()),
        f"{prefix}_mean": _mean(values),
        f"{prefix}_median": _quantile(values, 0.50),
        f"{prefix}_p10": _quantile(values, 0.10),
        f"{prefix}_p25": _quantile(values, 0.25),
        f"{prefix}_p75": _quantile(values, 0.75),
        f"{prefix}_p90": _quantile(values, 0.90),
    }


def _numeric_series(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _mean(series: Any) -> float:
    values = _numeric_series(series).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _quantile(series: Any, q: float) -> float:
    values = _numeric_series(series).dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def _rate(series: Any, predicate: Any) -> float:
    if series is None:
        return np.nan
    values = pd.Series(series).dropna()
    if values.empty:
        return np.nan
    mask = predicate(values)
    return float(pd.Series(mask).fillna(False).astype(bool).mean())


def _ret_pct(value: Any, base: Any) -> float:
    value = _as_float(value)
    base = _as_float(base)
    if not _is_finite(value) or not _is_finite(base) or base <= 0:
        return np.nan
    return (value / base - 1.0) * 100.0


def _first_offset(frame: pd.DataFrame, base_idx: int, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    mask = frame[column].fillna(False).astype(bool)
    if not mask.any():
        return np.nan
    first = frame.loc[mask].iloc[0]
    return float(int(first["date_index"]) - int(base_idx))


def _first_negative_close_offset(frame: pd.DataFrame, base_close: float, base_idx: int) -> float:
    if frame.empty or not _is_finite(base_close):
        return np.nan
    close = pd.to_numeric(frame["close"], errors="coerce")
    mask = close < float(base_close)
    if not mask.any():
        return np.nan
    first = frame.loc[mask].iloc[0]
    return float(int(first["date_index"]) - int(base_idx))


def _finite_gt(left: Any, right: Any) -> bool:
    left_value = _as_float(left)
    right_value = _as_float(right)
    return _is_finite(left_value) and _is_finite(right_value) and left_value > right_value


def _finite_le(left: Any, right: Any) -> bool:
    left_value = _as_float(left)
    right_value = _as_float(right)
    return _is_finite(left_value) and _is_finite(right_value) and left_value <= right_value


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _parse_int_values(values: Sequence[int] | str, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
    else:
        parts = list(values)
    parsed = tuple(int(part) for part in parts)
    if any(value <= 0 for value in parsed):
        raise ValueError(f"{name} must contain positive integers")
    return parsed


def available_memory_gb() -> float:
    available = _available_memory_bytes()
    return float(available / (1024**3)) if available is not None else np.nan


def _assert_memory_available(min_available_memory_gb: float, *, context: str) -> None:
    if min_available_memory_gb <= 0:
        return
    available = available_memory_gb()
    if np.isfinite(available) and available < float(min_available_memory_gb):
        raise RuntimeError(
            f"available memory {available:.2f} GB is below safety line "
            f"{float(min_available_memory_gb):.2f} GB at {context}"
        )


def _available_memory_bytes() -> int | None:
    if hasattr(ctypes, "windll"):
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    try:
        import os

        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


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
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    parser.add_argument("--loose-day2-upper", action="store_true", help="Do not require Day1 close to be at or below Day1 ATR upper.")
    parser.add_argument("--warmup-years", type=int, default=DEFAULT_WARMUP_YEARS)
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_two_day_kama_atr_breakout_analysis(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=_parse_int_values(args.horizons, "horizons"),
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
        require_day2_cross_from_below=not args.loose_day2_upper,
        warmup_years=args.warmup_years,
        min_available_memory_gb=args.min_available_memory_gb,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
