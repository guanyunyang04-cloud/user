"""Shared full-market material for legal daily action replay.

The module exposes only point-in-time quality-pool membership and the audited
execution arrays.  Future prices and sellability are available to explicitly
named hindsight studies, but are never silently converted into causal inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
    _stamp_tax_bps_by_date_idx,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
COST_SCENARIOS = ("base", "double_slippage")


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayMarket:
    date_values: np.ndarray
    symbol_values: np.ndarray
    start_idx: int
    end_idx: int
    entry_open_raw: np.ndarray
    exit_close_raw: np.ndarray
    exit_sellable: np.ndarray
    entry_filled: np.ndarray
    amount_panel: np.ndarray
    quality_mask: np.ndarray
    mark_close: np.ndarray
    costs: Any
    quality_rows: int

    @property
    def day_count(self) -> int:
        return int(self.end_idx - self.start_idx + 1)

    @property
    def symbol_count(self) -> int:
        return len(self.symbol_values)

    def local_idx(self, absolute_idx: int) -> int:
        local = int(absolute_idx) - int(self.start_idx)
        if local < 0 or local >= self.day_count:
            raise IndexError(absolute_idx)
        return local

    def mark_row(self, absolute_idx: int) -> np.ndarray:
        return self.mark_close[self.local_idx(absolute_idx)]


def _exact_date_index(date_values: np.ndarray, value: str) -> int:
    index = int(np.searchsorted(date_values.astype(str), str(value)))
    if index >= len(date_values) or str(date_values[index]) != str(value):
        raise ValueError(f"market replay date is absent: {value}")
    return index


def _forward_filled_marks(
    close_panel: np.ndarray, *, start_idx: int, end_idx: int
) -> np.ndarray:
    symbol_count = int(close_panel.shape[1])
    marks = np.full(
        (int(end_idx) - int(start_idx) + 1, symbol_count),
        np.nan,
        dtype=np.float32,
    )
    last = np.full(symbol_count, np.nan, dtype=np.float32)
    for absolute_idx in range(int(end_idx) + 1):
        row = np.asarray(close_panel[absolute_idx], dtype=np.float32)
        valid = np.isfinite(row) & (row > 0.0)
        last[valid] = row[valid]
        if absolute_idx >= int(start_idx):
            marks[absolute_idx - int(start_idx)] = last
    return marks


def load_market(study: Mapping[str, Any]) -> tuple[ReplayMarket, dict[str, Any]]:
    source = dict(study["source"])
    pack_path = resolve_path(source["corrected_pack_manifest"])
    quality_path = resolve_path(source["quality_training_manifest"])
    if sha256(pack_path) != str(source["expected_corrected_pack_sha256"]):
        raise ValueError("market_replay_pack_hash_mismatch")
    if sha256(quality_path) != str(source["expected_quality_training_sha256"]):
        raise ValueError("market_replay_quality_hash_mismatch")

    pack = CandidateCompleteAuditPack(pack_path)
    quality = read_json(quality_path)
    if quality.get("status") != "completed":
        raise ValueError("market_replay_quality_source_incomplete")
    if str(source["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("market_replay_quality_pool_mismatch")

    date_values = np.asarray(pack.date_values, dtype=object)
    symbol_values = np.asarray(pack.symbol_values, dtype=object)
    period = dict(study["period"])
    start_idx = _exact_date_index(date_values, str(period["first_decision_date"]))
    end_idx = _exact_date_index(date_values, str(period["terminal_cash_date"]))
    if start_idx >= end_idx:
        raise ValueError("market_replay_period_invalid")
    if int(str(date_values[end_idx])[:4]) >= int(source["forbidden_year"]):
        raise ValueError("market_replay_forbidden_period")

    day_count = end_idx - start_idx + 1
    quality_mask = np.zeros((day_count, len(symbol_values)), dtype=bool)
    duplicate_rows = 0
    quality_rows = 0
    spine_records: list[dict[str, Any]] = []
    formal_years = tuple(int(value) for value in period["formal_years"])
    spines = dict(quality["daily_quality_row_spine"])
    for year in formal_years:
        record = dict(spines[str(year)])
        path = resolve_path(record["path"])
        if sha256(path) != str(record["sha256"]):
            raise ValueError(f"market_replay_spine_hash_mismatch:{year}")
        frame = pd.read_parquet(path, columns=["date_idx", "symbol_idx"])
        dates = frame["date_idx"].to_numpy(np.int64)
        symbols = frame["symbol_idx"].to_numpy(np.int64)
        valid = (
            (dates >= start_idx)
            & (dates <= end_idx)
            & (symbols >= 0)
            & (symbols < len(symbol_values))
        )
        if not bool(valid.all()):
            raise ValueError(f"market_replay_spine_domain_invalid:{year}")
        local_dates = dates - start_idx
        duplicate_rows += int(quality_mask[local_dates, symbols].sum())
        quality_mask[local_dates, symbols] = True
        quality_rows += len(frame)
        spine_records.append(
            {
                "year": year,
                "path": str(path.resolve()),
                "sha256": str(record["sha256"]),
                "rows": len(frame),
            }
        )
    if duplicate_rows:
        raise ValueError(f"market_replay_duplicate_quality_rows:{duplicate_rows}")
    expected_rows = int(source["expected_quality_pool_rows"])
    if quality_rows != expected_rows or int(quality_mask.sum()) != expected_rows:
        raise ValueError(
            f"market_replay_quality_row_count:{quality_rows}:{quality_mask.sum()}"
        )

    pack_payload = read_json(pack_path)
    daily_raw = dict(pack_payload["feature_channels"]["daily_raw"])
    raw_shape = tuple(int(value) for value in daily_raw["shape"])
    raw_panel = np.memmap(
        resolve_path(daily_raw["path"]), dtype="float32", mode="r", shape=raw_shape
    )
    amount_index = list(daily_raw["columns"]).index("amount")
    amount_panel = raw_panel[:, :, amount_index]
    marks = _forward_filled_marks(
        pack.exit_close_raw, start_idx=start_idx, end_idx=end_idx
    )
    market = ReplayMarket(
        date_values=date_values,
        symbol_values=symbol_values,
        start_idx=start_idx,
        end_idx=end_idx,
        entry_open_raw=pack.entry_open_raw,
        exit_close_raw=pack.exit_close_raw,
        exit_sellable=pack.exit_sellable,
        entry_filled=pack.entry_filled,
        amount_panel=amount_panel,
        quality_mask=quality_mask,
        mark_close=marks,
        costs=pack.costs,
        quality_rows=quality_rows,
    )
    audit = {
        "pack_manifest": str(pack_path.resolve()),
        "pack_sha256": sha256(pack_path),
        "quality_manifest": str(quality_path.resolve()),
        "quality_sha256": sha256(quality_path),
        "quality_spines": spine_records,
        "quality_rows": quality_rows,
        "duplicate_quality_rows": duplicate_rows,
        "first_date": str(date_values[start_idx]),
        "last_date": str(date_values[end_idx]),
        "date_count": day_count,
        "symbol_count": len(symbol_values),
        "forbidden_2026_rows": 0,
    }
    return market, audit


def proportional_cost_multipliers(
    market: ReplayMarket, cost_scenario: str
) -> tuple[float, np.ndarray]:
    if str(cost_scenario) not in COST_SCENARIOS:
        raise ValueError(f"unknown market replay cost scenario: {cost_scenario}")
    multiplier = (
        1.0
        if str(cost_scenario) == "base"
        else float(market.costs.stress_slippage_multiplier)
    )
    slip = float(market.costs.slippage_bps) * multiplier / 10_000.0
    commission = float(market.costs.commission_bps) / 10_000.0
    transfer = float(market.costs.transfer_fee_bps) / 10_000.0
    buy_multiplier = (1.0 + slip) * (1.0 + commission + transfer)
    absolute_dates = np.arange(market.start_idx, market.end_idx + 1, dtype=np.int64)
    stamp = _stamp_tax_bps_by_date_idx(
        absolute_dates,
        date_values=market.date_values,
        contract=market.costs,
    ) / 10_000.0
    sell_multiplier = (1.0 - slip) * (1.0 - commission - transfer - stamp)
    if not math.isfinite(buy_multiplier) or buy_multiplier <= 0.0:
        raise ValueError("market_replay_invalid_buy_multiplier")
    if bool((~np.isfinite(sell_multiplier) | (sell_multiplier <= 0.0)).any()):
        raise ValueError("market_replay_invalid_sell_multiplier")
    return float(buy_multiplier), sell_multiplier.astype(np.float64)
