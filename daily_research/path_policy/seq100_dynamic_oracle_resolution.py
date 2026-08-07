"""Policy coalescence and prefix sensitivity for dynamic oracle values.

The final-terminal oracle induces a deterministic successor from every cash or
holding state.  Two action branches can therefore be followed until they first
reach a common cash state, where their remaining oracle value cancels exactly.
That algebraic cancellation is not automatically a causal label timestamp:
the final-terminal policy prefix may itself depend on prices after the meeting
date.  A separate truncated-prefix recomputation measures that dependence.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_market_replay as replay

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / (
    "daily_research/studies/seq100_dynamic_oracle_resolution_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_dynamic_oracle_resolution_v1"
)
STUDY_ID = "seq100_dynamic_oracle_resolution_v1"
SCHEMA_VERSION = 1

LABEL_COLUMNS = (
    "trade_date",
    "date_idx",
    "symbol_idx",
    "buy_action_valid",
    "oracle_cash_buy_selected",
    "buy_terminal_log_value",
    "cash_stay_terminal_log_value",
    "buy_advantage_vs_cash",
)


def _resolve(path: str | Path) -> Path:
    return replay.resolve_path(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return replay.read_json(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(
    path: str | Path, frame: pd.DataFrame, *, row_group_size: int
) -> dict[str, Any]:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
        row_group_size=int(row_group_size),
    )
    os.replace(temporary, target)
    return {
        "path": str(target.resolve()),
        "sha256": replay.sha256(target),
        "size": int(target.stat().st_size),
        "rows": len(frame),
    }


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("dynamic_oracle_resolution_study_id_mismatch")
    source = dict(study["source"])
    if str(source["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("dynamic_oracle_resolution_pool_mismatch")
    if int(source["forbidden_year"]) != 2026:
        raise ValueError("dynamic_oracle_resolution_forbidden_year")
    resolution = dict(study["resolution"])
    if tuple(resolution["cost_scenarios"]) != replay.COST_SCENARIOS:
        raise ValueError("dynamic_oracle_resolution_cost_scenarios")
    if bool(resolution["fixed_holding_horizon_used"]):
        raise ValueError("dynamic_oracle_resolution_fixed_horizon_forbidden")
    if bool(resolution["binary_good_stock_label_used"]):
        raise ValueError("dynamic_oracle_resolution_binary_label_forbidden")
    if bool(resolution["coalescence_is_causal_label_claim_allowed"]):
        raise ValueError("dynamic_oracle_resolution_causal_claim_forbidden")
    return study


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = dict(study["source"])
    oracle_path = _resolve(source["oracle_manifest"])
    market_path = _resolve(source["market_manifest"])
    oracle_study_path = _resolve(source["oracle_study"])
    expected = (
        (oracle_path, str(source["expected_oracle_manifest_sha256"])),
        (market_path, str(source["expected_market_manifest_sha256"])),
        (oracle_study_path, str(source["expected_oracle_study_sha256"])),
    )
    for path, digest in expected:
        if replay.sha256(path) != digest:
            raise ValueError(f"dynamic_oracle_resolution_source_hash:{path.name}")
    oracle_manifest = _read_json(oracle_path)
    market_manifest = _read_json(market_path)
    oracle_study = oracle.load_study(oracle_study_path)
    if oracle_manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_resolution_oracle_incomplete")
    if market_manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_resolution_market_incomplete")
    if bool(oracle_manifest.get("training_performed")):
        raise ValueError("dynamic_oracle_resolution_source_trained")
    records = {
        str(record["cost_scenario"]): dict(record)
        for record in oracle_manifest["relaxed_oracles"]
    }
    if tuple(records) != replay.COST_SCENARIOS:
        raise ValueError("dynamic_oracle_resolution_oracle_cost_order")
    contract = {
        "oracle_manifest": {
            "path": str(oracle_path.resolve()),
            "sha256": replay.sha256(oracle_path),
        },
        "market_manifest": {
            "path": str(market_path.resolve()),
            "sha256": replay.sha256(market_path),
        },
        "oracle_study": {
            "path": str(oracle_study_path.resolve()),
            "sha256": replay.sha256(oracle_study_path),
        },
    }
    return contract, oracle_manifest, oracle_study


def _record_by_year(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(record["year"]): dict(record) for record in records}
    if len(result) != len(records):
        raise ValueError("dynamic_oracle_resolution_duplicate_year")
    return result


def _read_labels(record: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(str(record["path"]))
    if replay.sha256(path) != str(record["sha256"]):
        raise ValueError("dynamic_oracle_resolution_label_hash")
    frame = pd.read_parquet(path, columns=list(LABEL_COLUMNS))
    if len(frame) != int(record["rows"]):
        raise ValueError("dynamic_oracle_resolution_label_rows")
    keys = (
        frame["date_idx"].to_numpy(np.int64) << np.int64(32)
    ) | frame["symbol_idx"].to_numpy(np.int64)
    if bool((keys[1:] <= keys[:-1]).any()):
        raise ValueError("dynamic_oracle_resolution_label_order")
    return frame


@dataclass(frozen=True)
class CashPolicyGraph:
    holding_first_cash: np.ndarray
    cash_parent: np.ndarray
    cash_depth: np.ndarray
    cash_up: tuple[np.ndarray, ...]
    cash_log: np.ndarray

    def first_common_cash(
        self, first: np.ndarray, second: np.ndarray
    ) -> np.ndarray:
        left = np.asarray(first, dtype=np.int32).copy()
        right = np.asarray(second, dtype=np.int32).copy()
        if left.shape != right.shape:
            raise ValueError("dynamic_oracle_resolution_lca_shape")
        if bool(
            (left < 0).any()
            or (right < 0).any()
            or (left >= len(self.cash_parent)).any()
            or (right >= len(self.cash_parent)).any()
        ):
            raise ValueError("dynamic_oracle_resolution_lca_domain")
        left_depth = self.cash_depth[left].astype(np.int32)
        right_depth = self.cash_depth[right].astype(np.int32)
        swap = left_depth < right_depth
        if bool(swap.any()):
            left_copy = left[swap].copy()
            left[swap] = right[swap]
            right[swap] = left_copy
            left_depth_copy = left_depth[swap].copy()
            left_depth[swap] = right_depth[swap]
            right_depth[swap] = left_depth_copy
        difference = left_depth - right_depth
        for level, ancestors in enumerate(self.cash_up):
            lift = (difference & (1 << level)) != 0
            if bool(lift.any()):
                left[lift] = ancestors[left[lift]]
        unequal = left != right
        for ancestors in reversed(self.cash_up):
            left_ancestor = ancestors[left]
            right_ancestor = ancestors[right]
            lift = unequal & (left_ancestor != right_ancestor)
            if bool(lift.any()):
                left[lift] = left_ancestor[lift]
                right[lift] = right_ancestor[lift]
        result = left.copy()
        remaining = left != right
        result[remaining] = self.cash_parent[left[remaining]]
        return result


def _load_array(record: Mapping[str, Any], *, mmap: bool = False) -> np.ndarray:
    path = Path(str(record["path"]))
    if replay.sha256(path) != str(record["sha256"]):
        raise ValueError("dynamic_oracle_resolution_array_hash")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def build_cash_policy_graph(
    *, market: replay.ReplayMarket, oracle_record: Mapping[str, Any]
) -> CashPolicyGraph:
    cash_action = _load_array(oracle_record["cash_action_symbol"])
    cash_log = _load_array(oracle_record["cash_log_value"])
    hold_exit = _load_array(oracle_record["holding_exit_policy"], mmap=True)
    return build_cash_policy_graph_from_arrays(
        market=market,
        cash_action_symbol=cash_action,
        cash_log_value=cash_log,
        holding_exit_policy=hold_exit,
    )


def build_cash_policy_graph_from_arrays(
    *,
    market: replay.ReplayMarket,
    cash_action_symbol: np.ndarray,
    cash_log_value: np.ndarray,
    holding_exit_policy: np.ndarray,
) -> CashPolicyGraph:
    day_count = market.day_count
    symbol_count = market.symbol_count
    cash_action = np.asarray(cash_action_symbol)
    cash_log = np.asarray(cash_log_value)
    hold_exit = np.asarray(holding_exit_policy)
    if cash_action.shape != (day_count,) or cash_log.shape != (day_count,):
        raise ValueError("dynamic_oracle_resolution_cash_array_shape")
    if hold_exit.shape != (day_count, symbol_count):
        raise ValueError("dynamic_oracle_resolution_holding_policy_shape")

    first_cash = np.full((day_count, symbol_count), -1, dtype=np.int32)
    next_sell = np.full(symbol_count, -1, dtype=np.int32)
    terminal_absolute = int(market.end_idx)
    terminal_close = np.asarray(
        market.exit_close_raw[terminal_absolute], dtype=np.float64
    )
    terminal_sellable = (
        np.asarray(market.exit_sellable[terminal_absolute], dtype=bool)
        & np.isfinite(terminal_close)
        & (terminal_close > 0.0)
    )
    next_sell[terminal_sellable] = day_count - 1
    for local in range(day_count - 2, -1, -1):
        first_cash[local] = first_cash[local + 1]
        exits = np.asarray(hold_exit[local], dtype=bool) & (next_sell >= 0)
        first_cash[local, exits] = next_sell[exits]
        absolute = int(market.start_idx + local)
        close = np.asarray(market.exit_close_raw[absolute], dtype=np.float64)
        sellable = (
            np.asarray(market.exit_sellable[absolute], dtype=bool)
            & np.isfinite(close)
            & (close > 0.0)
        )
        next_sell[sellable] = local

    cash_parent = np.full(day_count, day_count - 1, dtype=np.int32)
    cash_parent[-1] = day_count - 1
    for local in range(day_count - 2, -1, -1):
        selected = int(cash_action[local])
        parent = local + 1 if selected < 0 else int(first_cash[local + 1, selected])
        if parent <= local or parent >= day_count:
            raise ValueError(
                f"dynamic_oracle_resolution_cash_parent:{local}:{selected}:{parent}"
            )
        cash_parent[local] = parent
    cash_depth = np.zeros(day_count, dtype=np.uint16)
    for local in range(day_count - 2, -1, -1):
        depth = int(cash_depth[cash_parent[local]]) + 1
        if depth > np.iinfo(np.uint16).max:
            raise ValueError("dynamic_oracle_resolution_depth_overflow")
        cash_depth[local] = depth
    levels = max(1, math.ceil(math.log2(int(cash_depth.max()) + 1)))
    cash_up: list[np.ndarray] = [cash_parent]
    for _ in range(1, levels):
        previous = cash_up[-1]
        cash_up.append(previous[previous])
    return CashPolicyGraph(
        holding_first_cash=first_cash,
        cash_parent=cash_parent,
        cash_depth=cash_depth,
        cash_up=tuple(cash_up),
        cash_log=np.asarray(cash_log, dtype=np.float64),
    )


class _ResolutionSummary:
    def __init__(self) -> None:
        self.rows = 0
        self.selected_rows = 0
        self.delays: list[np.ndarray] = []
        self.first_cash_delays: list[np.ndarray] = []
        self.identity_errors: list[np.ndarray] = []

    def update(self, frame: pd.DataFrame) -> None:
        self.rows += len(frame)
        self.selected_rows += int(frame["oracle_cash_buy_selected"].sum())
        self.delays.append(frame["cash_resolution_delay"].to_numpy(np.int32))
        self.first_cash_delays.append(
            frame["buy_first_cash_delay"].to_numpy(np.int32)
        )
        self.identity_errors.append(
            frame["stored_value_identity_error"].to_numpy(np.float64)
        )

    def finish(self) -> dict[str, Any]:
        delays = np.concatenate(self.delays).astype(np.float64, copy=False)
        first = np.concatenate(self.first_cash_delays).astype(np.float64, copy=False)
        errors = np.concatenate(self.identity_errors).astype(np.float64, copy=False)
        quantiles = (0.5, 0.9, 0.99, 0.999)
        result: dict[str, Any] = {
            "valid_action_rows": self.rows,
            "oracle_selected_rows": self.selected_rows,
            "mean_cash_resolution_delay": float(delays.mean()),
            "maximum_cash_resolution_delay": int(delays.max()),
            "mean_buy_first_cash_delay": float(first.mean()),
            "maximum_buy_first_cash_delay": int(first.max()),
            "maximum_stored_value_identity_error": float(np.abs(errors).max()),
        }
        for value in quantiles:
            label = str(value).replace(".", "p")
            result[f"cash_resolution_delay_q{label}"] = float(
                np.quantile(delays, value)
            )
            result[f"buy_first_cash_delay_q{label}"] = float(
                np.quantile(first, value)
            )
        for sessions in (2, 5, 10, 20, 60, 252):
            result[f"cash_resolution_within_{sessions}_fraction"] = float(
                np.mean(delays <= sessions)
            )
        return result


def _resolve_label_frame(
    *,
    market: replay.ReplayMarket,
    graph: CashPolicyGraph,
    labels: pd.DataFrame,
    cost_scenario: str,
) -> pd.DataFrame:
    valid = labels["buy_action_valid"].astype(bool).to_numpy()
    frame = labels.loc[valid].reset_index(drop=True)
    signal_absolute = frame["date_idx"].to_numpy(np.int64)
    signal_local = signal_absolute - int(market.start_idx)
    symbols = frame["symbol_idx"].to_numpy(np.int64)
    if bool(
        (signal_local < 0).any()
        or (signal_local + 1 >= market.day_count).any()
        or (symbols < 0).any()
        or (symbols >= market.symbol_count).any()
    ):
        raise ValueError("dynamic_oracle_resolution_valid_action_domain")
    first_cash = graph.holding_first_cash[signal_local + 1, symbols]
    if bool((first_cash <= signal_local).any()):
        raise ValueError("dynamic_oracle_resolution_missing_first_cash")
    wait_cash = (signal_local + 1).astype(np.int32)
    resolution = graph.first_common_cash(first_cash, wait_cash)
    if bool((resolution < first_cash).any() | (resolution < wait_cash).any()):
        raise ValueError("dynamic_oracle_resolution_nonfuture_coalescence")
    cash_at_resolution = graph.cash_log[resolution]
    buy_terminal = frame["buy_terminal_log_value"].to_numpy(np.float64)
    cash_stay = frame["cash_stay_terminal_log_value"].to_numpy(np.float64)
    advantage = frame["buy_advantage_vs_cash"].to_numpy(np.float64)
    buy_prefix = buy_terminal - cash_at_resolution
    wait_prefix = cash_stay - cash_at_resolution
    decomposed = buy_prefix - wait_prefix
    return pd.DataFrame(
        {
            "cost_scenario": np.full(len(frame), str(cost_scenario)),
            "trade_date": frame["trade_date"].astype(str).to_numpy(),
            "date_idx": signal_absolute.astype(np.int32),
            "symbol_idx": symbols.astype(np.int32),
            "oracle_cash_buy_selected": frame[
                "oracle_cash_buy_selected"
            ].to_numpy(bool),
            "buy_advantage_vs_cash": advantage.astype(np.float32),
            "buy_first_cash_date_idx": (
                first_cash + int(market.start_idx)
            ).astype(np.int32),
            "cash_resolution_date_idx": (
                resolution + int(market.start_idx)
            ).astype(np.int32),
            "buy_first_cash_delay": (first_cash - signal_local).astype(np.int32),
            "cash_resolution_delay": (resolution - signal_local).astype(np.int32),
            "cash_resolution_is_terminal": resolution == market.day_count - 1,
            "buy_prefix_log_value_to_resolution": buy_prefix.astype(np.float32),
            "wait_prefix_log_value_to_resolution": wait_prefix.astype(np.float32),
            "decomposed_buy_advantage": decomposed.astype(np.float32),
            "stored_value_identity_error": (decomposed - advantage).astype(
                np.float32
            ),
        }
    )


def _prefix_terminal_action_advantage(
    *,
    market: replay.ReplayMarket,
    cost_scenario: str,
    signal_date_idx: int,
    symbol_idx: int,
    terminal_date_idx: int,
) -> float:
    signal_local = int(signal_date_idx) - int(market.start_idx)
    terminal_local = int(terminal_date_idx) - int(market.start_idx)
    if signal_local < 0 or terminal_local >= market.day_count:
        raise ValueError("dynamic_oracle_resolution_prefix_domain")
    if terminal_local < signal_local + 2:
        return math.nan
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )
    symbol_count = market.symbol_count
    cash_log = np.full(terminal_local + 1, -np.inf, dtype=np.float64)
    cash_log[terminal_local] = 0.0
    future_hold = np.full(symbol_count, -np.inf, dtype=np.float64)
    next_sell = np.full(symbol_count, -1, dtype=np.int32)
    terminal_absolute = int(market.start_idx + terminal_local)
    terminal_close = np.asarray(
        market.exit_close_raw[terminal_absolute], dtype=np.float64
    )
    terminal_sellable = (
        np.asarray(market.exit_sellable[terminal_absolute], dtype=bool)
        & np.isfinite(terminal_close)
        & (terminal_close > 0.0)
    )
    next_sell[terminal_sellable] = terminal_absolute

    for local in range(terminal_local - 1, signal_local - 1, -1):
        absolute = int(market.start_idx + local)
        mark_current = market.mark_close[local].astype(np.float64, copy=False)
        mark_next = market.mark_close[local + 1].astype(np.float64, copy=False)
        stay_log = float(cash_log[local + 1])
        quality = np.flatnonzero(market.quality_mask[local]).astype(np.int32)
        entry_open = np.asarray(
            market.entry_open_raw[absolute + 1, quality], dtype=np.float64
        )
        entry_valid = (
            np.asarray(market.entry_filled[absolute, quality], dtype=bool)
            & np.isfinite(entry_open)
            & (entry_open > 0.0)
            & np.isfinite(mark_next[quality])
            & (mark_next[quality] > 0.0)
            & np.isfinite(future_hold[quality])
        )
        buy_values = np.full(len(quality), -np.inf, dtype=np.float64)
        buy_values[entry_valid] = (
            np.log(mark_next[quality][entry_valid])
            - np.log(entry_open[entry_valid] * buy_multiplier)
            + future_hold[quality][entry_valid]
        )
        best_buy = (
            float(np.max(buy_values[entry_valid]))
            if bool(entry_valid.any())
            else -np.inf
        )
        cash_log[local] = max(stay_log, best_buy)
        if local == signal_local:
            position = np.searchsorted(quality, int(symbol_idx))
            if position >= len(quality) or int(quality[position]) != int(symbol_idx):
                return math.nan
            if not bool(entry_valid[position]):
                return math.nan
            return float(buy_values[position] - stay_log)

        continue_values = np.full(symbol_count, -np.inf, dtype=np.float64)
        continue_valid = (
            np.isfinite(mark_current)
            & (mark_current > 0.0)
            & np.isfinite(mark_next)
            & (mark_next > 0.0)
            & np.isfinite(future_hold)
        )
        continue_values[continue_valid] = (
            np.log(mark_next[continue_valid] / mark_current[continue_valid])
            + future_hold[continue_valid]
        )
        exit_values = np.full(symbol_count, -np.inf, dtype=np.float64)
        exit_symbols = np.flatnonzero(
            (next_sell >= 0) & np.isfinite(mark_current) & (mark_current > 0.0)
        )
        if len(exit_symbols):
            sale_dates = next_sell[exit_symbols]
            sale_prices = np.asarray(
                market.exit_close_raw[sale_dates, exit_symbols], dtype=np.float64
            )
            sale_local = sale_dates - int(market.start_idx)
            exit_values[exit_symbols] = (
                np.log(
                    sale_prices
                    * sell_multiplier[sale_local]
                    / mark_current[exit_symbols]
                )
                + cash_log[sale_local]
            )
        future_hold = np.maximum(continue_values, exit_values)
        current_close = np.asarray(
            market.exit_close_raw[absolute], dtype=np.float64
        )
        current_sellable = (
            np.asarray(market.exit_sellable[absolute], dtype=bool)
            & np.isfinite(current_close)
            & (current_close > 0.0)
        )
        next_sell[current_sellable] = absolute
    return math.nan


def _sample_prefix_queries(
    *,
    resolution_frame: pd.DataFrame,
    year: int,
    prefix_contract: Mapping[str, Any],
) -> pd.DataFrame:
    selected = resolution_frame["oracle_cash_buy_selected"].astype(bool)
    frames = [resolution_frame.loc[selected].assign(sample_role="oracle_selected")]
    seed = int(prefix_contract["sample_seed"]) + int(year) * 1009
    positive = resolution_frame.loc[
        (~selected) & (resolution_frame["buy_advantage_vs_cash"].astype(float) > 0.0)
    ]
    nonpositive = resolution_frame.loc[
        (~selected) & (resolution_frame["buy_advantage_vs_cash"].astype(float) <= 0.0)
    ]
    requests = (
        (
            positive,
            int(prefix_contract["random_positive_rows_per_year"]),
            "random_positive",
            seed,
        ),
        (
            nonpositive,
            int(prefix_contract["random_nonpositive_rows_per_year"]),
            "random_nonpositive",
            seed + 1,
        ),
    )
    for source, count, role, random_state in requests:
        if len(source):
            frames.append(
                source.sample(
                    n=min(count, len(source)),
                    random_state=random_state,
                    replace=False,
                ).assign(sample_role=role)
            )
    return pd.concat(frames, ignore_index=True)


def _run_prefix_audit(
    *,
    market: replay.ReplayMarket,
    queries: pd.DataFrame,
    prefix_contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cost_scenario = str(prefix_contract["cost_scenario"])
    tolerance = float(prefix_contract["prefix_stability_tolerance"])
    rows: list[dict[str, Any]] = []
    for row in queries.itertuples(index=False):
        prefix_value = _prefix_terminal_action_advantage(
            market=market,
            cost_scenario=cost_scenario,
            signal_date_idx=int(row.date_idx),
            symbol_idx=int(row.symbol_idx),
            terminal_date_idx=int(row.cash_resolution_date_idx),
        )
        final_value = float(row.buy_advantage_vs_cash)
        difference = prefix_value - final_value
        rows.append(
            {
                "sample_role": str(row.sample_role),
                "trade_date": str(row.trade_date),
                "year": int(str(row.trade_date)[:4]),
                "date_idx": int(row.date_idx),
                "symbol_idx": int(row.symbol_idx),
                "cash_resolution_date_idx": int(row.cash_resolution_date_idx),
                "cash_resolution_delay": int(row.cash_resolution_delay),
                "final_terminal_advantage": final_value,
                "prefix_terminal_advantage": prefix_value,
                "prefix_minus_final": difference,
                "prefix_value_finite": math.isfinite(prefix_value),
                "prefix_stable_within_tolerance": math.isfinite(prefix_value)
                and abs(difference) <= tolerance,
                "sign_agrees": math.isfinite(prefix_value)
                and ((prefix_value > 0.0) == (final_value > 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for role, group in [("all", frame), *list(frame.groupby("sample_role"))]:
        finite = group["prefix_value_finite"].astype(bool)
        difference = group.loc[finite, "prefix_minus_final"].to_numpy(np.float64)
        summaries.append(
            {
                "sample_role": str(role),
                "rows": len(group),
                "finite_rows": int(finite.sum()),
                "prefix_stable_fraction": float(
                    group["prefix_stable_within_tolerance"].mean()
                ),
                "sign_agreement_fraction": float(group["sign_agrees"].mean()),
                "mean_absolute_difference": (
                    float(np.abs(difference).mean()) if len(difference) else math.nan
                ),
                "median_absolute_difference": (
                    float(np.median(np.abs(difference)))
                    if len(difference)
                    else math.nan
                ),
                "q90_absolute_difference": (
                    float(np.quantile(np.abs(difference), 0.9))
                    if len(difference)
                    else math.nan
                ),
                "maximum_absolute_difference": (
                    float(np.abs(difference).max()) if len(difference) else math.nan
                ),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    return frame, {
        "cost_scenario": cost_scenario,
        "tolerance": tolerance,
        "groups": summary_frame.to_dict(orient="records"),
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_start = int(psutil.virtual_memory().available / 1024**2)
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    formal_years = tuple(int(value) for value in dict(study["period"])["formal_years"])
    source_contract, oracle_manifest, oracle_study = _source_contract(study)
    market, market_audit = replay.load_market(oracle_study)
    oracle_records = {
        str(record["cost_scenario"]): dict(record)
        for record in oracle_manifest["relaxed_oracles"]
    }
    output_records: list[dict[str, Any]] = []
    scenario_summaries: list[dict[str, Any]] = []
    prefix_queries: list[pd.DataFrame] = []
    minimum_available = memory_start

    for cost_scenario in replay.COST_SCENARIOS:
        oracle_record = oracle_records[cost_scenario]
        graph = build_cash_policy_graph(
            market=market, oracle_record=oracle_record
        )
        labels_by_year = _record_by_year(oracle_record["action_labels"])
        summary = _ResolutionSummary()
        for year in formal_years:
            labels = _read_labels(labels_by_year[year])
            resolved = _resolve_label_frame(
                market=market,
                graph=graph,
                labels=labels,
                cost_scenario=cost_scenario,
            )
            summary.update(resolved)
            path = (
                root
                / "action_resolution"
                / f"cost={cost_scenario}"
                / f"year={year}"
                / "part-0000.parquet"
            )
            record = _write_frame(path, resolved, row_group_size=row_group_size)
            record.update({"cost_scenario": cost_scenario, "year": year})
            output_records.append(record)
            if cost_scenario == str(dict(study["prefix_audit"])["cost_scenario"]):
                prefix_queries.append(
                    _sample_prefix_queries(
                        resolution_frame=resolved,
                        year=year,
                        prefix_contract=dict(study["prefix_audit"]),
                    )
                )
            minimum_available = min(
                minimum_available,
                int(psutil.virtual_memory().available / 1024**2),
            )
        scenario_summaries.append(
            {"cost_scenario": cost_scenario, **summary.finish()}
        )
        del graph
        gc.collect()

    prefix_query_frame = pd.concat(prefix_queries, ignore_index=True)
    if prefix_query_frame.duplicated(["date_idx", "symbol_idx"]).any():
        raise ValueError("dynamic_oracle_resolution_prefix_sample_duplicate")
    prefix_frame, prefix_summary = _run_prefix_audit(
        market=market,
        queries=prefix_query_frame,
        prefix_contract=dict(study["prefix_audit"]),
    )
    prefix_output = _write_frame(
        root / "prefix_sensitivity.parquet",
        prefix_frame,
        row_group_size=row_group_size,
    )
    summary_frame = pd.DataFrame(scenario_summaries)
    summary_output = _write_frame(
        root / "resolution_summary.parquet",
        summary_frame,
        row_group_size=row_group_size,
    )
    tolerance = float(dict(study["resolution"])["value_identity_tolerance"])
    maximum_identity_error = float(
        summary_frame["maximum_stored_value_identity_error"].max()
    )
    if maximum_identity_error > tolerance:
        raise ValueError(
            "dynamic_oracle_resolution_value_identity:"
            f"{maximum_identity_error}:{tolerance}"
        )
    memory_end = int(psutil.virtual_memory().available / 1024**2)
    manifest = {
        "schema": f"seq100_dynamic_oracle_resolution/{SCHEMA_VERSION}",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": {
            "path": str(study_file.resolve()),
            "sha256": replay.sha256(study_file),
        },
        "source_contract": source_contract,
        "market_audit": market_audit,
        "resolution_partitions": output_records,
        "scenario_summaries": scenario_summaries,
        "prefix_audit": prefix_summary,
        "outputs": {
            "resolution_summary": summary_output,
            "prefix_sensitivity": prefix_output,
        },
        "audit": {
            "expected_quality_rows_per_scenario": int(market.quality_rows),
            "valid_action_rows": int(
                sum(item["valid_action_rows"] for item in scenario_summaries)
            ),
            "partition_rows": int(sum(item["rows"] for item in output_records)),
            "maximum_stored_value_identity_error": maximum_identity_error,
            "prefix_sample_rows": len(prefix_frame),
            "forbidden_2026_rows": 0,
        },
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "available_memory_mb_at_start": memory_start,
            "available_memory_mb_at_end": memory_end,
            "minimum_available_memory_mb_observed": minimum_available,
            "reserve_memory_mb": int(dict(study["resources"])["reserve_memory_mb"]),
            "logical_cpu_count": int(psutil.cpu_count() or 1),
            "adaptive_memory": bool(dict(study["resources"])["adaptive_memory"]),
        },
        "fixed_holding_horizon_used": False,
        "future_path_used": True,
        "policy_conditional_coalescence_computed": True,
        "coalescence_claimed_as_causal_label_time": False,
        "training_performed": False,
        "causal_policy_evaluated": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve dynamic oracle policy branches and audit prefix sensitivity."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "valid_action_rows": result["audit"]["valid_action_rows"],
                "prefix_sample_rows": result["audit"]["prefix_sample_rows"],
                "prefix_audit": result["prefix_audit"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
