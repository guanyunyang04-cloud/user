"""Causal open-position episodes and pre-registered exit baselines.

This study keeps the already screened K-line entries fixed and asks a narrower
question: once an entry has happened, can an observable position state tell us
when selling is better than waiting?  It is deliberately a trade-level and
episode-level diagnostic.  It does not replay a portfolio or select a live
policy.

Signals are taken from the persisted causal pattern probe.  Every episode row
uses information available at that close.  A trigger requests a sale only
after the close and is filled at the next legal close, preserving T+1,
suspensions, limits, costs, and terminal liquidation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from daily_research.path_policy import seq100_causal_pattern_strategy_probe as probe
from daily_research.path_policy import seq100_dynamic_oracle as dynamic_oracle
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import (
    _buy_order,
    _sell_order,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_causal_exit_baselines_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_causal_exit_baselines_v1"
)
FORBIDDEN_YEAR = 2026
ENTRY_PATTERNS = ("retest", "nested_reacceleration", "breakout")
POLICY_NAMES = (
    "fixed_h2",
    "fixed_h5",
    "fixed_h10",
    "fixed_h20",
    "fixed_h40",
    "volatility_1_2",
    "volatility_1_3",
    "trailing_3pct",
    "giveback_half",
    "structure_small_reversal",
)
COST_SCENARIOS = replay.COST_SCENARIOS
VOLATILITY_SCALE_FLOOR = 0.01
VOLATILITY_SCALE_CAP = 0.08
FIXED_HORIZONS = {
    "fixed_h2": 2,
    "fixed_h5": 5,
    "fixed_h10": 10,
    "fixed_h20": 20,
    "fixed_h40": 40,
}
PANEL_COLUMNS = (
    "episode_id",
    "entry_pattern",
    "symbol_idx",
    "symbol",
    "signal_date_idx",
    "entry_date_idx",
    "signal_date",
    "entry_date",
    "signal_year",
    "session",
    "date_idx",
    "date",
    "close_raw",
    "gross_return_close",
    "peak_gain_close",
    "drawdown_from_peak",
    "peak_giveback_fraction",
    "mae_close",
    "risk_scale_20d",
    "amount_log",
    "breakout_active",
    "small_event",
    "medium_event",
    "up_exhaustion",
    "at_liquidation_boundary",
)
RESULT_COLUMNS = (
    "episode_id",
    "policy",
    "entry_pattern",
    "symbol_idx",
    "symbol",
    "signal_date_idx",
    "entry_date_idx",
    "signal_date",
    "entry_date",
    "signal_year",
    "request_date_idx",
    "exit_date_idx",
    "request_session",
    "holding_sessions",
    "mfe_close",
    "mae_close",
    "giveback_at_exit",
    "terminal_unresolved",
    "net_return_base",
    "gross_return_base",
    "finite_order_filled_base",
    "net_return_double_slippage",
    "gross_return_double_slippage",
    "finite_order_filled_double_slippage",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected_json_object:{path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"json_value_not_serializable:{type(value).__name__}")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(path: str | Path, frame: pd.DataFrame) -> dict[str, Any]:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd", row_group_size=100_000)
    os.replace(temporary, target)
    return {
        "path": str(target.resolve()),
        "rows": len(frame),
        "sha256": _sha256(target),
    }


def _flush_parquet_rows(
    *,
    writer: pq.ParquetWriter | None,
    path: Path,
    rows: list[dict[str, Any]],
    columns: Sequence[str],
) -> pq.ParquetWriter | None:
    if not rows:
        return writer
    frame = pd.DataFrame.from_records(rows, columns=columns)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def _hac(values: np.ndarray, lag: int = 20) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, math.nan
    centered = x - mean
    maximum_lag = min(int(lag), len(x) - 1)
    long_run = float(np.dot(centered, centered) / len(x))
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / len(x))
        long_run += 2.0 * (1.0 - offset / (maximum_lag + 1.0)) * covariance
    return mean, math.sqrt(max(long_run, 0.0) / len(x))


def _normal_one_sided_p(mean: float, se: float) -> float:
    if not math.isfinite(mean) or not math.isfinite(se) or se <= 0.0:
        return math.nan
    return 0.5 * math.erfc(float(mean / se) / math.sqrt(2.0))


def _bh(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    result = np.full(p.shape, np.nan, dtype=np.float64)
    positions = np.flatnonzero(np.isfinite(p))
    if len(positions) == 0:
        return result
    order = positions[np.argsort(p[positions], kind="mergesort")]
    adjusted = np.empty(len(order), dtype=np.float64)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        value = p[order[rank]] * len(order) / float(rank + 1)
        running = min(running, value)
        adjusted[rank] = min(running, 1.0)
    result[order] = adjusted
    return result


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != "seq100_causal_exit_baselines_v1":
        raise ValueError("exit_baselines_study_id_mismatch")
    source = dict(study.get("source", {}))
    if source.get("quality_pool_name") != "quality_liquidity_pit":
        raise ValueError("exit_baselines_quality_pool_contract")
    if int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR:
        raise ValueError("exit_baselines_forbidden_year_contract")
    if tuple(dict(study.get("execution", {})).get("cost_scenarios", ())) != COST_SCENARIOS:
        raise ValueError("exit_baselines_cost_contract")
    if set(study.get("policies", {})) != set(POLICY_NAMES):
        raise ValueError("exit_baselines_policy_contract")
    max_sessions = int(dict(study.get("episodes", {})).get("maximum_sessions", 0))
    if max_sessions < max(FIXED_HORIZONS.values()):
        raise ValueError("exit_baselines_episode_tail_too_short")
    return study


def _load_structure(study: Mapping[str, Any], market: replay.ReplayMarket) -> probe.StructureArrays:
    probe_study = probe.load_study(_resolve(study["source"]["pattern_strategy_study"]))
    return probe.load_structure_arrays(study=probe_study, market=market)


def _entry_records(
    *,
    market: replay.ReplayMarket,
    structure: probe.StructureArrays,
    liquidation_date: str,
) -> pd.DataFrame:
    positions = np.flatnonzero(market.date_values.astype(str) == str(liquidation_date))
    if len(positions) != 1:
        raise ValueError("exit_baselines_entry_liquidation_date")
    liquidation_abs = int(positions[0])
    masks = {
        "retest": structure.retest,
        "nested_reacceleration": structure.nested_reacceleration,
        "breakout": structure.breakout,
    }
    records: list[dict[str, Any]] = []
    last_signal_local = liquidation_abs - int(market.start_idx) - 2
    for pattern in ENTRY_PATTERNS:
        signal_mask = masks[pattern]
        for local_day in range(max(last_signal_local + 1, 0)):
            absolute = int(market.start_idx + local_day)
            symbols = np.flatnonzero(
                signal_mask[local_day] & market.quality_mask[local_day]
            )
            if len(symbols) == 0:
                continue
            filled = np.asarray(market.entry_filled[absolute, symbols], dtype=bool)
            entry_prices = np.asarray(
                market.entry_open_raw[absolute + 1, symbols], dtype=np.float64
            )
            valid = filled & np.isfinite(entry_prices) & (entry_prices > 0.0)
            for symbol, entry_price in zip(
                symbols[valid], entry_prices[valid], strict=True
            ):
                signal_date = str(market.date_values[absolute])
                entry_date = str(market.date_values[absolute + 1])
                records.append(
                    {
                        "entry_pattern": pattern,
                        "symbol_idx": int(symbol),
                        "symbol": str(market.symbol_values[int(symbol)]),
                        "signal_date_idx": absolute,
                        "entry_date_idx": absolute + 1,
                        "entry_price_raw": float(entry_price),
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "signal_year": int(signal_date[:4]),
                    }
                )
    entries = pd.DataFrame.from_records(records)
    if entries.empty:
        return entries
    entries = entries.sort_values(
        ["entry_pattern", "signal_date_idx", "symbol_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if bool(
        entries.duplicated(
            ["entry_pattern", "signal_date_idx", "symbol_idx"]
        ).any()
    ):
        raise ValueError("exit_baselines_duplicate_entries")
    return entries


def _risk_scale(close: np.ndarray, signal_abs: int, symbol: int) -> float:
    start = max(0, int(signal_abs) - 20)
    values = np.asarray(
        close[start : int(signal_abs) + 1, int(symbol)], dtype=np.float64
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < 6:
        return math.nan
    returns = np.diff(np.log(values))
    returns = returns[np.isfinite(returns)]
    return float(np.std(returns, ddof=1)) if len(returns) >= 5 else math.nan


def _path_arrays(
    *,
    market: replay.ReplayMarket,
    structure: probe.StructureArrays,
    row: Mapping[str, Any],
    liquidation_abs: int,
    maximum_sessions: int,
) -> dict[str, np.ndarray]:
    signal_abs = int(row["signal_date_idx"])
    entry_abs = int(row["entry_date_idx"])
    symbol = int(row["symbol_idx"])
    if entry_abs != signal_abs + 1:
        raise ValueError("exit_baselines_entry_t_plus_one")
    # Include one extra observed close so a request at the maximum session can
    # still be filled at the next legal close.
    end_abs = min(int(market.end_idx), entry_abs + int(maximum_sessions))
    days = np.arange(entry_abs, end_abs + 1, dtype=np.int32)
    prices = np.asarray(market.exit_close_raw[days, symbol], dtype=np.float64)
    valid = np.isfinite(prices) & (prices > 0.0)
    entry_price = float(row["entry_price_raw"])
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        raise ValueError("exit_baselines_invalid_entry_price")
    filled_prices = prices.copy()
    last = entry_price
    for pos, value in enumerate(filled_prices):
        if math.isfinite(float(value)) and float(value) > 0.0:
            last = float(value)
        else:
            filled_prices[pos] = last
    peak = np.maximum.accumulate(filled_prices)
    trough = np.minimum.accumulate(filled_prices)
    peak_gain = peak / entry_price - 1.0
    drawdown = filled_prices / peak - 1.0
    denominator = np.maximum(peak - entry_price, entry_price * 1.0e-12)
    giveback = np.where(peak_gain > 0.0, (peak - filled_prices) / denominator, 0.0)
    local = days - int(market.start_idx)
    amount = np.asarray(market.amount_panel[days, symbol], dtype=np.float64)
    amount_log = np.full(len(amount), np.nan, dtype=np.float64)
    positive_amount = np.isfinite(amount) & (amount > 0.0)
    amount_log[positive_amount] = np.log(amount[positive_amount])
    return {
        "days": days,
        "prices": prices,
        "valid": valid,
        "filled_prices": filled_prices,
        "return_close": filled_prices / entry_price - 1.0,
        "peak_gain": peak_gain,
        "drawdown": drawdown,
        "giveback": giveback,
        "mae": trough / entry_price - 1.0,
        "risk_scale": np.full(
            len(days),
            _risk_scale(market.exit_close_raw, signal_abs, symbol),
        ),
        "amount_log": amount_log,
        "breakout_active": np.asarray(structure.breakout_active[local, symbol], dtype=np.int8),
        "small_event": np.asarray(structure.small_event[local, symbol], dtype=np.int8),
        "medium_event": np.asarray(structure.medium_event[local, symbol], dtype=np.int8),
        "exhaustion": np.asarray(structure.exhaustion[local, symbol], dtype=bool),
        "at_liquidation": days >= int(liquidation_abs),
    }


def _trigger_index(
    policy: str,
    path: Mapping[str, np.ndarray],
    *,
    maximum_sessions: int,
) -> int:
    days = np.asarray(path["days"], dtype=np.int32)
    sessions = np.arange(1, len(days) + 1, dtype=np.int32)
    trigger = np.zeros(len(days), dtype=bool)
    if policy in FIXED_HORIZONS:
        # A planned N-session close is requested at the preceding close.
        target = int(FIXED_HORIZONS[policy]) - 1
        trigger |= sessions == target
    elif policy == "volatility_1_2":
        scale = float(path["risk_scale"][0])
        scale = float(np.clip(scale, VOLATILITY_SCALE_FLOOR, VOLATILITY_SCALE_CAP))
        trigger |= sessions >= 1
        if math.isfinite(scale) and scale > 0.0:
            trigger &= (path["return_close"] <= -scale) | (
                path["return_close"] >= 2.0 * scale
            )
        else:
            trigger[:] = False
    elif policy == "volatility_1_3":
        scale = float(path["risk_scale"][0])
        scale = float(np.clip(scale, VOLATILITY_SCALE_FLOOR, VOLATILITY_SCALE_CAP))
        trigger |= sessions >= 1
        if math.isfinite(scale) and scale > 0.0:
            trigger &= (path["return_close"] <= -scale) | (
                path["return_close"] >= 3.0 * scale
            )
        else:
            trigger[:] = False
    elif policy == "trailing_3pct":
        trigger = (sessions >= 1) & (path["drawdown"] <= -0.03)
    elif policy == "giveback_half":
        trigger = (sessions >= 1) & (path["peak_gain"] >= 0.03) & (
            path["giveback"] >= 0.50
        )
    elif policy == "structure_small_reversal":
        trigger = (sessions >= 1) & (
            (path["breakout_active"] != 1) | (path["small_event"] == -1)
        )
    else:
        raise KeyError(policy)
    # Terminal boundary is a known study boundary, not a future-derived label.
    trigger |= np.asarray(path["at_liquidation"], dtype=bool)
    candidates = np.flatnonzero(trigger & (sessions <= int(maximum_sessions)))
    if len(candidates):
        return int(candidates[0])
    return min(int(maximum_sessions) - 1, len(days) - 2)


def _resolve_exit(
    *,
    market: replay.ReplayMarket,
    path: Mapping[str, np.ndarray],
    symbol: int,
    trigger_index: int,
) -> int:
    days = np.asarray(path["days"], dtype=np.int32)
    start = min(int(trigger_index) + 1, len(days))
    if start >= len(days):
        return -1
    legal = np.asarray(market.exit_sellable[days[start:], symbol], dtype=bool)
    prices = np.asarray(market.exit_close_raw[days[start:], symbol], dtype=np.float64)
    valid = legal & np.isfinite(prices) & (prices > 0.0)
    positions = np.flatnonzero(valid)
    return int(days[start + int(positions[0])]) if len(positions) else -1


def _trade_returns(
    *,
    market: replay.ReplayMarket,
    row: Mapping[str, Any],
    exit_abs: int,
    starting_cash: float,
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    entry_price = float(row["entry_price_raw"])
    for cost in COST_SCENARIOS:
        if exit_abs < 0:
            result[f"net_return_{cost}"] = -1.0
            result[f"gross_return_{cost}"] = -1.0
            result[f"finite_order_filled_{cost}"] = True
            continue
        multiplier = 1.0 if cost == "base" else float(market.costs.stress_slippage_multiplier)
        shares, buy_cash, _, _ = _buy_order(
            available_cash=float(starting_cash),
            allocated_cash=float(starting_cash),
            entry_price=entry_price,
            contract=market.costs,
            slippage_multiplier=multiplier,
        )
        exit_price = float(market.exit_close_raw[exit_abs, int(row["symbol_idx"])])
        if shares <= 0 or buy_cash <= 0.0:
            result[f"net_return_{cost}"] = 0.0
            result[f"gross_return_{cost}"] = exit_price / entry_price - 1.0
            result[f"finite_order_filled_{cost}"] = False
            continue
        proceeds, _, _ = _sell_order(
            shares=shares,
            exit_price=exit_price,
            exit_date_idx=exit_abs,
            date_values=market.date_values,
            contract=market.costs,
            slippage_multiplier=multiplier,
        )
        result[f"net_return_{cost}"] = (float(starting_cash) - buy_cash + proceeds) / float(starting_cash) - 1.0
        result[f"gross_return_{cost}"] = exit_price / entry_price - 1.0
        result[f"finite_order_filled_{cost}"] = True
    return result


def _panel_rows(
    *,
    episode_id: int,
    row: Mapping[str, Any],
    path: Mapping[str, np.ndarray],
    market: replay.ReplayMarket,
) -> list[dict[str, Any]]:
    dates = np.asarray(path["days"], dtype=np.int32)
    symbol = int(row["symbol_idx"])
    signal_date = str(row["signal_date"])
    entry_date = str(row["entry_date"])
    risk = float(path["risk_scale"][0])
    output: list[dict[str, Any]] = []
    for pos, absolute in enumerate(dates):
        output.append(
            {
                "episode_id": int(episode_id),
                "entry_pattern": str(row["entry_pattern"]),
                "symbol_idx": symbol,
                "symbol": str(row["symbol"]),
                "signal_date_idx": int(row["signal_date_idx"]),
                "entry_date_idx": int(row["entry_date_idx"]),
                "signal_date": signal_date,
                "entry_date": entry_date,
                "signal_year": int(row["signal_year"]),
                "session": int(pos + 1),
                "date_idx": int(absolute),
                "date": str(market.date_values[int(absolute)]),
                "close_raw": float(path["prices"][pos]) if bool(path["valid"][pos]) else math.nan,
                "gross_return_close": float(path["filled_prices"][pos] / float(row["entry_price_raw"]) - 1.0),
                "peak_gain_close": float(path["peak_gain"][pos]),
                "drawdown_from_peak": float(path["drawdown"][pos]),
                "peak_giveback_fraction": float(path["giveback"][pos]),
                "mae_close": float(path["mae"][pos]),
                "risk_scale_20d": risk,
                "amount_log": float(path["amount_log"][pos]),
                "breakout_active": int(path["breakout_active"][pos]),
                "small_event": int(path["small_event"][pos]),
                "medium_event": int(path["medium_event"][pos]),
                "up_exhaustion": bool(path["exhaustion"][pos]),
                "at_liquidation_boundary": bool(path["at_liquidation"][pos]),
            }
        )
    return output


def _evaluate(
    *,
    market: replay.ReplayMarket,
    structure: probe.StructureArrays,
    entries: pd.DataFrame,
    study: Mapping[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    execution = dict(study["execution"])
    starting_cash = float(execution["starting_cash_cny_per_entry"])
    maximum_sessions = int(dict(study["episodes"])["maximum_sessions"])
    liquidation_date = str(dict(study["period"])["terminal_liquidation_start"])
    positions = np.flatnonzero(market.date_values.astype(str) == liquidation_date)
    if len(positions) != 1:
        raise ValueError("exit_baselines_liquidation_date")
    liquidation_abs = int(positions[0])
    panel_root = output_root / "episode_panel"
    panel_root.mkdir(parents=True, exist_ok=True)
    panel_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    episode_id = 0
    minimum_available = int(psutil.virtual_memory().available / 1024**2)
    for pattern in ENTRY_PATTERNS:
        subset = entries.loc[entries["entry_pattern"].eq(pattern)].copy()
        if subset.empty:
            continue
        for year, year_frame in subset.groupby("signal_year", sort=True):
            writer: pq.ParquetWriter | None = None
            result_writer: pq.ParquetWriter | None = None
            panel_rows: list[dict[str, Any]] = []
            result_rows: list[dict[str, Any]] = []
            panel_path = (
                panel_root
                / f"entry_pattern={pattern}"
                / f"signal_year={int(year)}"
                / "part-0000.parquet"
            )
            result_path = (
                output_root
                / "policy_results"
                / f"entry_pattern={pattern}"
                / f"signal_year={int(year)}"
                / "part-0000.parquet"
            )

            for row in year_frame.to_dict("records"):
                signal_abs = int(row["signal_date_idx"])
                if signal_abs + 1 >= liquidation_abs:
                    continue
                path = _path_arrays(
                    market=market,
                    structure=structure,
                    row=row,
                    liquidation_abs=liquidation_abs,
                    maximum_sessions=maximum_sessions,
                )
                panel_rows.extend(_panel_rows(episode_id=episode_id, row=row, path=path, market=market))
                for policy in POLICY_NAMES:
                    trigger = _trigger_index(policy, path, maximum_sessions=maximum_sessions)
                    exit_abs = _resolve_exit(
                        market=market,
                        path=path,
                        symbol=int(row["symbol_idx"]),
                        trigger_index=trigger,
                    )
                    returns = _trade_returns(
                        market=market,
                        row=row,
                        exit_abs=exit_abs,
                        starting_cash=starting_cash,
                    )
                    exit_session = int(exit_abs - int(row["entry_date_idx"]) + 1) if exit_abs >= 0 else -1
                    exit_pos = int(np.searchsorted(path["days"], exit_abs)) if exit_abs >= 0 else -1
                    result_rows.append(
                        {
                            "episode_id": int(episode_id),
                            "policy": policy,
                            "entry_pattern": pattern,
                            "symbol_idx": int(row["symbol_idx"]),
                            "symbol": str(row["symbol"]),
                            "signal_date_idx": signal_abs,
                            "entry_date_idx": int(row["entry_date_idx"]),
                            "signal_date": str(row["signal_date"]),
                            "entry_date": str(row["entry_date"]),
                            "signal_year": int(row["signal_year"]),
                            "request_date_idx": int(path["days"][trigger]),
                            "exit_date_idx": int(exit_abs),
                            "request_session": int(trigger + 1),
                            "holding_sessions": exit_session,
                            "mfe_close": float(path["peak_gain"][: exit_pos + 1].max()) if exit_pos >= 0 else float(path["peak_gain"].max()),
                            "mae_close": float(path["mae"][: exit_pos + 1].min()) if exit_pos >= 0 else float(path["mae"].min()),
                            "giveback_at_exit": float(path["giveback"][exit_pos]) if exit_pos >= 0 and exit_pos < len(path["giveback"]) else math.nan,
                            "terminal_unresolved": bool(exit_abs < 0),
                            **returns,
                        }
                    )
                episode_id += 1
                if len(panel_rows) >= 50_000:
                    writer = _flush_parquet_rows(
                        writer=writer,
                        path=panel_path,
                        rows=panel_rows,
                        columns=PANEL_COLUMNS,
                    )
                if len(result_rows) >= 50_000:
                    result_writer = _flush_parquet_rows(
                        writer=result_writer,
                        path=result_path,
                        rows=result_rows,
                        columns=RESULT_COLUMNS,
                    )
            writer = _flush_parquet_rows(
                writer=writer,
                path=panel_path,
                rows=panel_rows,
                columns=PANEL_COLUMNS,
            )
            result_writer = _flush_parquet_rows(
                writer=result_writer,
                path=result_path,
                rows=result_rows,
                columns=RESULT_COLUMNS,
            )
            if writer is not None:
                writer.close()
                panel_records.append(
                    {
                        "entry_pattern": pattern,
                        "signal_year": int(year),
                        "path": str(panel_path.resolve()),
                        "rows": int(pq.ParquetFile(panel_path).metadata.num_rows),
                        "sha256": _sha256(panel_path),
                    }
                )
            if result_writer is not None:
                result_writer.close()
                result_records.append(
                    {
                        "entry_pattern": pattern,
                        "signal_year": int(year),
                        "path": str(result_path.resolve()),
                        "rows": int(pq.ParquetFile(result_path).metadata.num_rows),
                        "sha256": _sha256(result_path),
                    }
                )
            minimum_available = min(minimum_available, int(psutil.virtual_memory().available / 1024**2))
            print(json.dumps({"entry_pattern": pattern, "signal_year": int(year), "episodes": int(episode_id), "available_memory_mb": int(psutil.virtual_memory().available / 1024**2)}, ensure_ascii=False), flush=True)
    return panel_records, result_records, {
        "minimum_available_memory_mb": minimum_available,
        "episodes": episode_id,
    }


def _summaries(
    *, output_root: Path, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    primary_years = [int(value) for value in dict(study["period"])["primary_years"]]
    lag = int(dict(study["evaluation"])["hac_lag"])
    result_glob = (
        output_root / "policy_results" / "**" / "*.parquet"
    ).resolve().as_posix()
    available_mb = int(psutil.virtual_memory().available / 1024**2)
    reserve_mb = int(dict(study["resources"])["reserve_memory_mb"])
    memory_mb = max(min(int(available_mb * 0.65), available_mb - reserve_mb), 512)
    logical = int(psutil.cpu_count() or 1)
    threads = max(
        1,
        min(
            int(dict(study["resources"])["maximum_threads"]),
            max(logical - 2, 1),
            max(memory_mb // 512, 1),
        ),
    )
    years_sql = ",".join(str(value) for value in primary_years)
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit='{memory_mb}MB'")
    daily = connection.execute(
        f"""
        SELECT entry_pattern, policy, signal_date, signal_year,
               avg(net_return_base) AS net_return_base,
               avg(net_return_double_slippage) AS net_return_double_slippage,
               count(*) AS trades
        FROM read_parquet('{result_glob}', hive_partitioning=false)
        WHERE signal_year IN ({years_sql})
        GROUP BY entry_pattern, policy, signal_date, signal_year
        ORDER BY entry_pattern, policy, signal_date
        """
    ).fetchdf()
    aggregate_parts: list[pd.DataFrame] = []
    for cost in COST_SCENARIOS:
        part = connection.execute(
            f"""
            SELECT entry_pattern, policy,
                   count(*) AS trades,
                   count(DISTINCT signal_date) AS dates,
                   count(DISTINCT symbol_idx) AS symbols,
                   avg(net_return_{cost}) AS trade_weighted_net_return,
                   avg(gross_return_{cost}) AS mean_gross_return,
                   avg(CASE WHEN net_return_{cost} > 0 THEN 1.0 ELSE 0.0 END)
                       AS positive_trade_fraction,
                   median(CASE WHEN holding_sessions >= 0 THEN holding_sessions END)
                       AS median_holding_sessions,
                   avg(mfe_close) AS mean_mfe_close,
                   avg(mae_close) AS mean_mae_close,
                   avg(giveback_at_exit) AS mean_giveback_at_exit,
                   sum(CASE WHEN terminal_unresolved THEN 1 ELSE 0 END)
                       AS terminal_unresolved
            FROM read_parquet('{result_glob}', hive_partitioning=false)
            WHERE signal_year IN ({years_sql})
            GROUP BY entry_pattern, policy
            """
        ).fetchdf()
        part["cost_scenario"] = cost
        aggregate_parts.append(part)
    connection.close()
    aggregates = pd.concat(aggregate_parts, ignore_index=True)
    daily_parts: list[pd.DataFrame] = []
    for cost in COST_SCENARIOS:
        part = daily[
            ["entry_pattern", "policy", "signal_date", "signal_year", "trades"]
        ].copy()
        part["cost_scenario"] = cost
        part["mean_net_return"] = daily[f"net_return_{cost}"].to_numpy(float)
        daily_parts.append(part)
    daily_long = pd.concat(daily_parts, ignore_index=True)
    annual = (
        daily_long.groupby(
            ["entry_pattern", "policy", "cost_scenario", "signal_year"],
            sort=True,
            as_index=False,
        )
        .agg(
            mean_net_return=("mean_net_return", "mean"),
            dates=("signal_date", "nunique"),
            trades=("trades", "sum"),
        )
    )
    statistic_rows: list[dict[str, Any]] = []
    for key, group in daily_long.groupby(
        ["entry_pattern", "policy", "cost_scenario"], sort=True
    ):
        mean, se = _hac(group["mean_net_return"].to_numpy(float), lag)
        pattern, policy, cost = key
        years = annual[
            annual["entry_pattern"].eq(pattern)
            & annual["policy"].eq(policy)
            & annual["cost_scenario"].eq(cost)
        ]
        statistic_rows.append(
            {
                "entry_pattern": pattern,
                "policy": policy,
                "cost_scenario": cost,
                "mean_net_return": mean,
                "hac_se": se,
                "hac_lcb95": mean - 1.96 * se
                if math.isfinite(se)
                else math.nan,
                "one_sided_p": _normal_one_sided_p(mean, se),
                "positive_year_count": int((years["mean_net_return"] > 0.0).sum()),
            }
        )
    summary = pd.DataFrame(statistic_rows).merge(
        aggregates,
        on=["entry_pattern", "policy", "cost_scenario"],
        how="left",
        validate="one_to_one",
    )
    summary["bh_q"] = np.nan
    for cost, idx in summary.groupby("cost_scenario").groups.items():
        positions = np.asarray(list(idx), dtype=np.int64)
        summary.loc[positions, "bh_q"] = _bh(summary.loc[positions, "one_sided_p"].to_numpy(float))
    evaluation = dict(study["evaluation"])
    majority = int(evaluation["majority_positive_years"])
    summary["cost_pass"] = (
        summary["hac_lcb95"].gt(0.0)
        & summary["positive_year_count"].ge(majority)
        & summary["bh_q"].le(float(evaluation["maximum_bh_q"]))
    )
    candidate_pass = (
        summary.groupby(["entry_pattern", "policy"], sort=False)["cost_pass"]
        .agg(["all", "count"])
        .reset_index()
    )
    candidate_pass["promotion_pass"] = candidate_pass["all"] & candidate_pass[
        "count"
    ].eq(len(COST_SCENARIOS))
    summary = summary.merge(
        candidate_pass[["entry_pattern", "policy", "promotion_pass"]],
        on=["entry_pattern", "policy"],
        how="left",
        validate="many_to_one",
    )
    return (
        summary.sort_values(
            ["cost_scenario", "entry_pattern", "policy"], kind="mergesort"
        ),
        annual,
        {"duckdb_memory_limit_mb": memory_mb, "duckdb_threads": threads},
    )


def run_study(*, study_path: str | Path = DEFAULT_STUDY_PATH, output_root: str | Path | None = None, force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        return _read_json(manifest_path)
    probe_manifest_path = _resolve(study["source"]["pattern_strategy_manifest"])
    if _sha256(probe_manifest_path) != str(study["source"]["expected_pattern_strategy_manifest_sha256"]):
        raise ValueError("exit_baselines_pattern_strategy_manifest_hash")
    oracle_study = dynamic_oracle.load_study(_resolve(study["source"]["dynamic_oracle_study"]))
    market, market_audit = replay.load_market(oracle_study)
    structure = _load_structure(study, market)
    entries = _entry_records(
        market=market,
        structure=structure,
        liquidation_date=str(study["period"]["terminal_liquidation_start"]),
    )
    panel_records, result_records, runtime = _evaluate(
        market=market,
        structure=structure,
        entries=entries,
        study=study,
        output_root=root,
    )
    summary, annual, summary_runtime = _summaries(output_root=root, study=study)
    runtime.update(summary_runtime)
    results_manifest_path = root / "policy_results_manifest.json"
    _write_json(
        results_manifest_path,
        {
            "schema": "seq100_causal_exit_policy_results/1",
            "records": result_records,
            "rows": int(sum(int(record["rows"]) for record in result_records)),
        },
    )
    outputs = {
        "policy_results": {
            "path": str(results_manifest_path.resolve()),
            "rows": int(sum(int(record["rows"]) for record in result_records)),
            "sha256": _sha256(results_manifest_path),
        },
        "summary": _write_frame(root / "summary.parquet", summary),
        "annual_summary": _write_frame(root / "annual_summary.parquet", annual),
        "entry_records": _write_frame(root / "entry_records.parquet", entries),
    }
    gate = {
        "status": "passed"
        if bool(summary.get("promotion_pass", pd.Series(dtype=bool)).any())
        else "failed",
        "account_replay_allowed": False,
        "production_policy_allowed": False,
        "passed_candidates": summary.loc[
            summary["promotion_pass"], ["entry_pattern", "policy"]
        ]
        .drop_duplicates()
        .to_dict("records")
        if not summary.empty
        else [],
        "reason": "mechanical exit baseline only; no account replay is authorized by this study",
    }
    gate_path = root / "promotion_gate.json"
    _write_json(gate_path, gate)
    outputs["promotion_gate"] = {"path": str(gate_path.resolve()), "sha256": _sha256(gate_path)}
    runtime.update({"elapsed_seconds": float(time.perf_counter() - started), "available_memory_mb_at_end": int(psutil.virtual_memory().available / 1024**2)})
    manifest = {
        "schema": "seq100_causal_exit_baselines_manifest/1",
        "status": "completed",
        "study_id": study["study_id"],
        "study": {"path": str(study_file), "sha256": _sha256(study_file)},
        "market_audit": market_audit,
        "entries": len(entries),
        "episodes": int(runtime["episodes"]),
        "panel_records": panel_records,
        "policy_result_records": result_records,
        "outputs": outputs,
        "runtime": runtime,
        "boundaries": dict(study["boundaries"]),
        "promotion_gate": gate,
        "account_execution_performed": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
    }
    _write_json(manifest_path, manifest)
    del structure, market
    gc.collect()
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run causal open-position exit baselines.")
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_study(study_path=args.study, output_root=args.output_root, force=bool(args.force))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
