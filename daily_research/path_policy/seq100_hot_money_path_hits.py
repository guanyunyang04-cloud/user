from __future__ import annotations

"""Compute first-hit continuation versus adverse-path labels from a frozen event study."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = WORKSPACE_ROOT / "daily_research/research_records/seq100/seq100_hot_money_event_study_v1_20260806"
EVENT_COLUMNS = (
    "participation_shock",
    "positive_participation_shock",
    "range_expansion",
    "breakout20",
    "breakout60",
    "breakout_confirmed",
    "retest20",
    "rebreakout20",
    "climax_fade",
)
# ``entry_open_next`` is path day 1.  A-share T+1 makes path day 2 the first
# day on which a newly purchased long can be sold, so first-hit targets and
# adverse exits use days 2..20.  Entry-day lows remain a separate exposure
# diagnostic in the event study rather than an executable stop.
LEGAL_PATH_START_DAY = 2


def _hac_se(values: pd.Series, lag: int = 10) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    x -= x.mean()
    lag = min(int(lag), len(x) - 1)
    var = float(np.dot(x, x) / len(x))
    for j in range(1, lag + 1):
        var += 2.0 * (1.0 - j / (lag + 1.0)) * float(np.dot(x[j:], x[:-j]) / len(x))
    return math.sqrt(max(var, 0.0) / len(x))


def _first_hit(mask: np.ndarray, valid: np.ndarray, day_numbers: np.ndarray | None = None) -> np.ndarray:
    hit = np.asarray(mask, dtype=bool) & np.asarray(valid, dtype=bool)
    result = np.full(hit.shape[0], np.nan, dtype=float)
    any_hit = hit.any(axis=1)
    if day_numbers is None:
        day_numbers = np.arange(1, hit.shape[1] + 1, dtype=np.float64)
    day_numbers = np.asarray(day_numbers, dtype=np.float64).reshape(-1)
    if day_numbers.shape[0] != hit.shape[1]:
        raise ValueError("day_numbers must match hit path width")
    result[any_hit] = day_numbers[hit[any_hit].argmax(axis=1)]
    return result


def _period(year: int) -> str:
    if year <= 2020:
        return "development_2012_2020"
    if year <= 2022:
        return "validation_2021_2022"
    if year <= 2025:
        return "oos_2023_2025"
    return "outside"


def _year_path_hits(root: Path, year: int) -> pd.DataFrame:
    event_path = root / f"daily_events_{year}.parquet"
    base_path = root / "daily_base.parquet"
    if not event_path.is_file() or not base_path.is_file():
        raise FileNotFoundError(f"event_study_artifact_missing:{year}")
    event_columns = ["symbol", "trade_date", "entry_open_next", *EVENT_COLUMNS]
    events = pd.read_parquet(event_path, columns=event_columns)
    end = f"{year + 1}-03-31" if year < 2025 else "2025-12-31"
    bars = pd.read_parquet(
        base_path,
        columns=["symbol", "trade_date", "high", "low"],
        filters=[("trade_date", ">=", f"{year}-01-01"), ("trade_date", "<=", end)],
    )
    events["trade_date"] = events["trade_date"].astype(str)
    bars["trade_date"] = bars["trade_date"].astype(str)
    bars = bars.sort_values(["symbol", "trade_date"], kind="mergesort")
    grouped_bars = {symbol: group.reset_index(drop=True) for symbol, group in bars.groupby("symbol", sort=False)}
    parts: list[pd.DataFrame] = []
    for symbol, group in events.groupby("symbol", sort=False):
        stock_bars = grouped_bars.get(symbol)
        if stock_bars is None or stock_bars.empty:
            continue
        dates = stock_bars["trade_date"].to_numpy()
        highs = stock_bars["high"].to_numpy(dtype=float)
        lows = stock_bars["low"].to_numpy(dtype=float)
        positions = np.searchsorted(dates, group["trade_date"].to_numpy())
        safe_positions = np.minimum(positions, len(dates) - 1)
        valid = (positions < len(dates)) & (dates[safe_positions] == group["trade_date"].to_numpy())
        if not valid.any():
            continue
        offsets = np.arange(LEGAL_PATH_START_DAY, 21, dtype=np.int64)[None, :]
        indices = positions[valid, None] + offsets
        valid_path = indices < len(dates)
        safe_indices = np.minimum(indices, len(dates) - 1)
        entry = group["entry_open_next"].to_numpy(dtype=float)[valid, None]
        high_return = highs[safe_indices] / entry - 1.0
        low_return = lows[safe_indices] / entry - 1.0
        path_days = np.arange(LEGAL_PATH_START_DAY, 21, dtype=np.float64)
        up5_day = _first_hit(high_return >= 0.05, valid_path, path_days)
        up10_day = _first_hit(high_return >= 0.10, valid_path, path_days)
        down3_day = _first_hit(low_return <= -0.03, valid_path, path_days)
        down5_day = _first_hit(low_return <= -0.05, valid_path, path_days)
        out = group.loc[valid, ["symbol", "trade_date", *EVENT_COLUMNS]].copy()
        out["up5_day"] = up5_day
        out["up10_day"] = up10_day
        out["down3_day"] = down3_day
        out["down5_day"] = down5_day
        out["up5_before_down3"] = np.isfinite(up5_day) & (~np.isfinite(down3_day) | (up5_day < down3_day))
        out["up10_before_down5"] = np.isfinite(up10_day) & (~np.isfinite(down5_day) | (up10_day < down5_day))
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    result["year"] = int(year)
    result["period"] = _period(year)
    return result


def _summary_for_part(part: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in EVENT_COLUMNS:
        selected = part.loc[part[event].astype(bool)].copy()
        if selected.empty:
            continue
        for name, column in [("up5_before_down3", "up5_before_down3"), ("up10_before_down5", "up10_before_down5")]:
            date_mean = selected.groupby("trade_date", sort=True)[column].mean()
            rows.append({
                "event": event,
                "year": int(selected["year"].iloc[0]),
                "period": str(selected["period"].iloc[0]),
                "path_target": name,
                "rows": int(len(selected)),
                "dates": int(len(date_mean)),
                "hit_rate": float(selected[column].mean()),
                "date_equal_hit_rate": float(date_mean.mean()),
                "hac_se_date_equal": _hac_se(date_mean),
                "hac_lcb_date_equal": float(date_mean.mean() - 1.96 * _hac_se(date_mean)),
                "median_up5_day": float(selected["up5_day"].median()),
                "median_up10_day": float(selected["up10_day"].median()),
            })
    return pd.DataFrame(rows)


def run(*, root: Path = DEFAULT_ROOT, start_year: int = 2012, end_year: int = 2025, force: bool = False) -> dict[str, Any]:
    root = root.resolve()
    output = root / "path_hits"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "path_hit_summary.csv"
    if summary_path.is_file() and not force:
        return {"status": "completed", "summary_csv": str(summary_path.resolve())}
    summaries: list[pd.DataFrame] = []
    for year in range(int(start_year), int(end_year) + 1):
        part = _year_path_hits(root, year)
        if part.empty:
            continue
        part_path = output / f"path_hits_{year}.parquet"
        part.to_parquet(part_path, index=False, compression="zstd")
        summaries.append(_summary_for_part(part))
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    oos = summary[(summary["period"] == "oos_2023_2025") & (summary["path_target"] == "up10_before_down5")].copy() if not summary.empty else pd.DataFrame()
    lines = [
        "# First-Hit Path Audit",
        "",
        "For each event at close t, entry is the next observed open (path day 1). Under A-share T+1, first-hit targets and adverse exits are inspected only from path day 2 through day 20; entry-day adversity is not treated as an executable stop. A positive path label requires the upside threshold to be reached before the adverse threshold; this is stricter than looking at an unordered MFE/MAE envelope.",
        "",
        "Thresholds: +5% before -3%, and +10% before -5%. No future state is used for event selection. 2026 is excluded.",
        "",
    ]
    if not oos.empty:
        lines.append(oos[["event", "year", "rows", "hit_rate", "date_equal_hit_rate", "hac_lcb_date_equal", "median_up10_day"]].to_markdown(index=False, floatfmt=".4f"))
    lines.extend([
        "",
        "The path probabilities measure opportunity conditional on a filled entry, not guaranteed executable profit: a strategy must still choose an exit before the adverse threshold and pay costs.",
    ])
    record_path = output / "path_hit_record.md"
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "completed", "summary_csv": str(summary_path.resolve()), "record_md": str(record_path.resolve())}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Audit first-hit continuation after hot-money events.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run(root=Path(args.root), start_year=args.start_year, end_year=args.end_year, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
