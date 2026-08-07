"""Annual as-of experience ledger from prefix-only dynamic hindsight oracles.

Each prefix oracle is solved with a terminal date that was already observed at
that historical cutoff.  A stock-day becomes learner-eligible only when its buy
and wait branches meet at a common cash state strictly before that terminal.
The following year supplies a timestamped revision, while final-2025 labels are
used only for aggregate stability diagnostics and never enter causal row data.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy import seq100_market_replay as replay

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / (
    "daily_research/studies/seq100_dynamic_oracle_causal_prefix_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_dynamic_oracle_causal_prefix_v1"
)
STUDY_ID = "seq100_dynamic_oracle_causal_prefix_v1"
SCHEMA_VERSION = 1

KEY_COLUMNS = ("date_idx", "symbol_idx")
VERSION_VALUE_COLUMNS = (
    "prefix_oracle_cash_buy_selected",
    "buy_advantage_vs_cash",
    "buy_first_cash_date_idx",
    "cash_resolution_date_idx",
    "cash_resolution_delay",
    "cash_resolution_is_terminal",
    "sessions_after_resolution",
    "naturally_resolved",
    "buy_trade_log_return_to_first_cash",
    "cash_value_difference_first_cash_vs_wait",
)
VERSION_COLUMNS = (
    "cost_scenario",
    "version_role",
    "signal_year",
    "as_of_year",
    "as_of_date",
    "as_of_date_idx",
    "trade_date",
    "date_idx",
    "symbol_idx",
    *VERSION_VALUE_COLUMNS,
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


def _file_record(path: str | Path) -> dict[str, Any]:
    target = _resolve(path)
    return {
        "path": str(target.resolve()),
        "sha256": replay.sha256(target),
        "size": int(target.stat().st_size),
    }


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
    return {**_file_record(target), "rows": len(frame)}


def _write_array(path: str | Path, values: np.ndarray) -> dict[str, Any]:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, target)
    return {**_file_record(target), "shape": list(values.shape), "dtype": str(values.dtype)}


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("dynamic_oracle_causal_prefix_study_id")
    source = dict(study["source"])
    if str(source["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("dynamic_oracle_causal_prefix_pool")
    if int(source["forbidden_year"]) != 2026:
        raise ValueError("dynamic_oracle_causal_prefix_forbidden_year")
    experience = dict(study["experience"])
    if tuple(experience["cost_scenarios"]) != replay.COST_SCENARIOS:
        raise ValueError("dynamic_oracle_causal_prefix_cost_scenarios")
    if bool(experience["fixed_holding_horizon_used"]):
        raise ValueError("dynamic_oracle_causal_prefix_fixed_horizon")
    if bool(experience["binary_good_stock_label_used"]):
        raise ValueError("dynamic_oracle_causal_prefix_binary_label")
    if bool(experience["forced_terminal_resolution_is_learnable"]):
        raise ValueError("dynamic_oracle_causal_prefix_terminal_leak")
    if bool(experience["final_oracle_row_labels_allowed_in_causal_outputs"]):
        raise ValueError("dynamic_oracle_causal_prefix_final_row_leak")
    if bool(dict(study["period"])["daily_first_availability_claimed"]):
        raise ValueError("dynamic_oracle_causal_prefix_daily_claim")
    return study


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = dict(study["source"])
    requests = (
        (
            "oracle_manifest",
            "expected_oracle_manifest_sha256",
        ),
        (
            "resolution_manifest",
            "expected_resolution_manifest_sha256",
        ),
        (
            "resolution_validation",
            "expected_resolution_validation_sha256",
        ),
        (
            "oracle_study",
            "expected_oracle_study_sha256",
        ),
    )
    contract: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for path_key, hash_key in requests:
        path = _resolve(source[path_key])
        digest = replay.sha256(path)
        if digest != str(source[hash_key]):
            raise ValueError(f"dynamic_oracle_causal_prefix_source_hash:{path_key}")
        contract[path_key] = {"path": str(path.resolve()), "sha256": digest}
        payloads[path_key] = _read_json(path)
    oracle_manifest = payloads["oracle_manifest"]
    resolution_manifest = payloads["resolution_manifest"]
    validation = payloads["resolution_validation"]
    oracle_study = oracle.load_study(source["oracle_study"])
    if oracle_manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_causal_prefix_oracle_incomplete")
    if resolution_manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_causal_prefix_resolution_incomplete")
    if validation.get("status") != "passed":
        raise ValueError("dynamic_oracle_causal_prefix_validation_incomplete")
    if bool(resolution_manifest.get("coalescence_claimed_as_causal_label_time")):
        raise ValueError("dynamic_oracle_causal_prefix_bad_source_claim")
    return contract, oracle_manifest, resolution_manifest, oracle_study


def truncate_market(
    market: replay.ReplayMarket, *, terminal_date_idx: int
) -> replay.ReplayMarket:
    terminal = int(terminal_date_idx)
    if terminal <= int(market.start_idx) or terminal > int(market.end_idx):
        raise ValueError("dynamic_oracle_causal_prefix_terminal_domain")
    count = terminal - int(market.start_idx) + 1
    quality = np.asarray(market.quality_mask[:count], dtype=bool)
    marks = np.asarray(market.mark_close[:count], dtype=np.float32)
    return replay.ReplayMarket(
        date_values=market.date_values,
        symbol_values=market.symbol_values,
        start_idx=int(market.start_idx),
        end_idx=terminal,
        entry_open_raw=market.entry_open_raw,
        exit_close_raw=market.exit_close_raw,
        exit_sellable=market.exit_sellable,
        entry_filled=market.entry_filled,
        amount_panel=market.amount_panel,
        quality_mask=quality,
        mark_close=marks,
        costs=market.costs,
        quality_rows=int(quality.sum()),
    )


def annual_cutoffs(
    market: replay.ReplayMarket, formal_years: Sequence[int]
) -> dict[int, int]:
    absolute = np.arange(int(market.start_idx), int(market.end_idx) + 1)
    years = np.asarray(
        [int(str(market.date_values[index])[:4]) for index in absolute],
        dtype=np.int32,
    )
    result: dict[int, int] = {}
    for year in formal_years:
        matches = absolute[years == int(year)]
        if not len(matches):
            raise ValueError(f"dynamic_oracle_causal_prefix_year_absent:{year}")
        result[int(year)] = int(matches[-1])
    return result


def _policy_record(
    *,
    root: Path,
    solution: oracle.RelaxedSolution,
    market: replay.ReplayMarket,
    as_of_year: int,
) -> dict[str, Any]:
    policy_root = (
        root
        / "prefix_policies"
        / f"cost={solution.cost_scenario}"
        / f"as_of_year={int(as_of_year)}"
    )
    return {
        "cost_scenario": str(solution.cost_scenario),
        "as_of_year": int(as_of_year),
        "as_of_date": str(market.date_values[market.end_idx]),
        "as_of_date_idx": int(market.end_idx),
        "date_count": int(market.day_count),
        "quality_rows": int(market.quality_rows),
        "terminal_log_wealth": float(solution.terminal_log_wealth),
        "selected_trade_rows": len(solution.selected_trades),
        "cash_log_value": _write_array(
            policy_root / "cash_log_value.npy", solution.cash_log
        ),
        "cash_action_symbol": _write_array(
            policy_root / "cash_action_symbol.npy", solution.cash_action_symbol
        ),
        "holding_exit_policy": _write_array(
            policy_root / "holding_exit_policy.npy", solution.hold_exit_policy
        ),
    }


def extract_prefix_version(
    *,
    market: replay.ReplayMarket,
    solution: oracle.RelaxedSolution,
    graph: resolution.CashPolicyGraph,
    signal_year: int,
    as_of_year: int,
    version_role: str,
) -> pd.DataFrame:
    if version_role not in {"signal_year_end", "next_year_end"}:
        raise ValueError("dynamic_oracle_causal_prefix_version_role")
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, solution.cost_scenario
    )
    rows: list[pd.DataFrame] = []
    for absolute_idx in range(int(market.start_idx), int(market.end_idx)):
        if int(str(market.date_values[absolute_idx])[:4]) != int(signal_year):
            continue
        local = absolute_idx - int(market.start_idx)
        if local + 1 >= market.day_count:
            continue
        quality = np.flatnonzero(market.quality_mask[local]).astype(np.int32)
        if not len(quality):
            continue
        first_cash = graph.holding_first_cash[local + 1, quality]
        entry = np.asarray(
            market.entry_open_raw[absolute_idx + 1, quality], dtype=np.float64
        )
        valid = (
            np.asarray(market.entry_filled[absolute_idx, quality], dtype=bool)
            & np.isfinite(entry)
            & (entry > 0.0)
            & (first_cash > local)
            & (first_cash < market.day_count)
        )
        if not bool(valid.any()):
            continue
        symbols = quality[valid]
        first = first_cash[valid].astype(np.int32)
        entries = entry[valid]
        sale_absolute = first + int(market.start_idx)
        sale_price = np.asarray(
            market.exit_close_raw[sale_absolute, symbols], dtype=np.float64
        )
        legal_sale = (
            np.asarray(market.exit_sellable[sale_absolute, symbols], dtype=bool)
            & np.isfinite(sale_price)
            & (sale_price > 0.0)
        )
        if not bool(legal_sale.all()):
            raise ValueError("dynamic_oracle_causal_prefix_first_cash_sale")
        wait_cash = np.full(len(symbols), local + 1, dtype=np.int32)
        meeting = graph.first_common_cash(first, wait_cash)
        trade_log = np.log(
            sale_price
            * sell_multiplier[first]
            / (entries * float(buy_multiplier))
        )
        cash_difference = graph.cash_log[first] - graph.cash_log[wait_cash]
        advantage = trade_log + cash_difference
        if bool((~np.isfinite(advantage)).any()):
            raise ValueError("dynamic_oracle_causal_prefix_nonfinite_value")
        rows.append(
            pd.DataFrame(
                {
                    "cost_scenario": np.full(
                        len(symbols), str(solution.cost_scenario)
                    ),
                    "version_role": np.full(len(symbols), str(version_role)),
                    "signal_year": np.full(
                        len(symbols), int(signal_year), dtype=np.int16
                    ),
                    "as_of_year": np.full(
                        len(symbols), int(as_of_year), dtype=np.int16
                    ),
                    "as_of_date": np.full(
                        len(symbols), str(market.date_values[market.end_idx])
                    ),
                    "as_of_date_idx": np.full(
                        len(symbols), int(market.end_idx), dtype=np.int32
                    ),
                    "trade_date": np.full(
                        len(symbols), str(market.date_values[absolute_idx])
                    ),
                    "date_idx": np.full(
                        len(symbols), int(absolute_idx), dtype=np.int32
                    ),
                    "symbol_idx": symbols.astype(np.int32),
                    "prefix_oracle_cash_buy_selected": (
                        symbols == int(solution.cash_action_symbol[local])
                    ),
                    "buy_advantage_vs_cash": advantage.astype(np.float64),
                    "buy_first_cash_date_idx": sale_absolute.astype(np.int32),
                    "cash_resolution_date_idx": (
                        meeting + int(market.start_idx)
                    ).astype(np.int32),
                    "cash_resolution_delay": (meeting - local).astype(np.int32),
                    "cash_resolution_is_terminal": meeting == market.day_count - 1,
                    "sessions_after_resolution": (
                        market.day_count - 1 - meeting
                    ).astype(np.int32),
                    "naturally_resolved": meeting < market.day_count - 1,
                    "buy_trade_log_return_to_first_cash": trade_log.astype(
                        np.float64
                    ),
                    "cash_value_difference_first_cash_vs_wait": (
                        cash_difference.astype(np.float64)
                    ),
                }
            )
        )
    if not rows:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in VERSION_COLUMNS})
    frame = pd.concat(rows, ignore_index=True).sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("dynamic_oracle_causal_prefix_version_duplicates")
    return frame.reset_index(drop=True)


def _version_path(
    root: Path, *, cost_scenario: str, as_of_year: int, signal_year: int
) -> Path:
    return (
        root
        / "experience_versions"
        / f"cost={cost_scenario}"
        / f"as_of_year={int(as_of_year)}"
        / f"signal_year={int(signal_year)}"
        / "part-0000.parquet"
    )


def _natural(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.loc[frame["naturally_resolved"].astype(bool)].copy()


def build_first_experience_and_updates(
    *,
    signal_year_end: pd.DataFrame,
    next_year_end: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_version = _natural(signal_year_end)
    next_version = _natural(next_year_end)
    first_keys = set(
        zip(
            first_version["date_idx"].astype(int),
            first_version["symbol_idx"].astype(int),
            strict=True,
        )
    )
    new_next = next_version.loc[
        [
            (int(date_idx), int(symbol_idx)) not in first_keys
            for date_idx, symbol_idx in zip(
                next_version["date_idx"],
                next_version["symbol_idx"],
                strict=True,
            )
        ]
    ]
    first_parts = [first_version]
    if not new_next.empty:
        first_parts.append(new_next)
    first = pd.concat(first_parts, ignore_index=True).sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    )
    if first.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("dynamic_oracle_causal_prefix_first_duplicates")
    first = first.rename(
        columns={
            "version_role": "first_observed_version_role",
            "as_of_year": "first_observed_as_of_year",
            "as_of_date": "first_observed_as_of_date",
            "as_of_date_idx": "first_observed_as_of_date_idx",
        }
    )

    if first_version.empty or next_version.empty:
        updates = pd.DataFrame()
    else:
        left = first_version[
            [
                *KEY_COLUMNS,
                "as_of_year",
                "as_of_date",
                "as_of_date_idx",
                "buy_advantage_vs_cash",
                "cash_resolution_date_idx",
                "cash_resolution_delay",
            ]
        ].rename(
            columns={
                "as_of_year": "previous_as_of_year",
                "as_of_date": "previous_as_of_date",
                "as_of_date_idx": "previous_as_of_date_idx",
                "buy_advantage_vs_cash": "previous_buy_advantage_vs_cash",
                "cash_resolution_date_idx": "previous_resolution_date_idx",
                "cash_resolution_delay": "previous_resolution_delay",
            }
        )
        right = next_version[
            [
                "cost_scenario",
                "signal_year",
                "trade_date",
                *KEY_COLUMNS,
                "as_of_year",
                "as_of_date",
                "as_of_date_idx",
                "buy_advantage_vs_cash",
                "cash_resolution_date_idx",
                "cash_resolution_delay",
            ]
        ].rename(
            columns={
                "as_of_year": "update_as_of_year",
                "as_of_date": "update_as_of_date",
                "as_of_date_idx": "update_as_of_date_idx",
                "buy_advantage_vs_cash": "updated_buy_advantage_vs_cash",
                "cash_resolution_date_idx": "updated_resolution_date_idx",
                "cash_resolution_delay": "updated_resolution_delay",
            }
        )
        updates = right.merge(left, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
        delta = (
            updates["updated_buy_advantage_vs_cash"].to_numpy(np.float64)
            - updates["previous_buy_advantage_vs_cash"].to_numpy(np.float64)
        )
        updates["update_minus_previous"] = delta.astype(np.float32)
        updates["sign_agrees"] = (
            updates["updated_buy_advantage_vs_cash"].to_numpy(np.float64) > 0.0
        ) == (
            updates["previous_buy_advantage_vs_cash"].to_numpy(np.float64) > 0.0
        )
        updates["stable_within_tolerance"] = np.abs(delta) <= float(tolerance)
        updates = updates.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
            drop=True
        )
    return first.reset_index(drop=True), updates


def _comparison(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_value: str,
    right_value: str,
    prefix: str,
    tolerance: float,
) -> dict[str, Any]:
    merged = left[[*KEY_COLUMNS, left_value]].merge(
        right[[*KEY_COLUMNS, right_value]],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if left_value == right_value:
        left_column = f"{left_value}_left"
        right_column = f"{right_value}_right"
    else:
        left_column = left_value
        right_column = right_value
    if merged.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_mean_absolute_revision": None,
            f"{prefix}_q90_absolute_revision": None,
            f"{prefix}_maximum_absolute_revision": None,
            f"{prefix}_sign_agreement_fraction": None,
            f"{prefix}_stable_fraction": None,
            f"{prefix}_positive_to_nonpositive": 0,
            f"{prefix}_nonpositive_to_positive": 0,
        }
    old = merged[left_column].to_numpy(np.float64)
    new = merged[right_column].to_numpy(np.float64)
    difference = new - old
    old_positive = old > 0.0
    new_positive = new > 0.0
    absolute = np.abs(difference)
    return {
        f"{prefix}_rows": len(merged),
        f"{prefix}_mean_absolute_revision": float(absolute.mean()),
        f"{prefix}_q90_absolute_revision": float(np.quantile(absolute, 0.9)),
        f"{prefix}_maximum_absolute_revision": float(absolute.max()),
        f"{prefix}_sign_agreement_fraction": float((old_positive == new_positive).mean()),
        f"{prefix}_stable_fraction": float((absolute <= float(tolerance)).mean()),
        f"{prefix}_positive_to_nonpositive": int((old_positive & ~new_positive).sum()),
        f"{prefix}_nonpositive_to_positive": int((~old_positive & new_positive).sum()),
    }


def _read_final_resolution(
    record: Mapping[str, Any], *, cost_scenario: str, signal_year: int
) -> pd.DataFrame:
    path = Path(str(record["path"]))
    if replay.sha256(path) != str(record["sha256"]):
        raise ValueError("dynamic_oracle_causal_prefix_final_hash")
    frame = pd.read_parquet(
        path,
        columns=[
            "date_idx",
            "symbol_idx",
            "buy_advantage_vs_cash",
            "cash_resolution_date_idx",
            "cash_resolution_is_terminal",
        ],
    )
    if len(frame) != int(record["rows"]):
        raise ValueError("dynamic_oracle_causal_prefix_final_rows")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("dynamic_oracle_causal_prefix_final_duplicates")
    frame.insert(0, "cost_scenario", str(cost_scenario))
    frame.insert(1, "signal_year", int(signal_year))
    return frame


def _revision_summary(
    *,
    cost_scenario: str,
    signal_year: int,
    signal_year_end: pd.DataFrame,
    next_year_end: pd.DataFrame,
    first_experience: pd.DataFrame,
    final_reference: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    v0 = _natural(signal_year_end)
    v1 = _natural(next_year_end)
    if v1.empty:
        latest = v0
    else:
        v1_keys = set(
            zip(v1["date_idx"].astype(int), v1["symbol_idx"].astype(int), strict=True)
        )
        remaining_v0 = v0.loc[
            [
                (int(date_idx), int(symbol_idx)) not in v1_keys
                for date_idx, symbol_idx in zip(
                    v0["date_idx"], v0["symbol_idx"], strict=True
                )
            ]
        ]
        latest = pd.concat([v1, remaining_v0], ignore_index=True)
    result: dict[str, Any] = {
        "cost_scenario": str(cost_scenario),
        "signal_year": int(signal_year),
        "final_valid_rows": len(final_reference),
        "signal_year_end_valid_rows": len(signal_year_end),
        "signal_year_end_naturally_resolved_rows": len(v0),
        "signal_year_end_natural_coverage_of_final": float(
            len(v0) / len(final_reference)
        ),
        "next_year_end_available": not next_year_end.empty,
        "next_year_end_valid_rows": len(next_year_end),
        "next_year_end_naturally_resolved_rows": len(v1),
        "next_year_end_natural_coverage_of_final": (
            float(len(v1) / len(final_reference)) if not next_year_end.empty else None
        ),
        "first_experience_rows": len(first_experience),
        "first_experience_coverage_of_final": float(
            len(first_experience) / len(final_reference)
        ),
        "first_experience_positive_fraction": (
            float((first_experience["buy_advantage_vs_cash"] > 0.0).mean())
            if len(first_experience)
            else None
        ),
    }
    result.update(
        _comparison(
            left=v0,
            right=v1,
            left_value="buy_advantage_vs_cash",
            right_value="buy_advantage_vs_cash",
            prefix="year_end_to_next_year",
            tolerance=tolerance,
        )
    )
    result.update(
        _comparison(
            left=latest,
            right=final_reference,
            left_value="buy_advantage_vs_cash",
            right_value="buy_advantage_vs_cash",
            prefix="latest_causal_to_final_diagnostic",
            tolerance=tolerance,
        )
    )
    return result


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_start = int(psutil.virtual_memory().available / 1024**2)
    minimum_available = memory_start
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    reserve_memory = int(dict(study["resources"])["reserve_memory_mb"])
    tolerance = float(dict(study["experience"])["revision_stability_tolerance"])
    formal_years = tuple(int(value) for value in dict(study["period"])["formal_years"])
    source_contract, _oracle_manifest, resolution_manifest, oracle_study = (
        _source_contract(study)
    )
    full_market, market_audit = replay.load_market(oracle_study)
    cutoffs = annual_cutoffs(full_market, formal_years)
    final_records = {
        (str(record["cost_scenario"]), int(record["year"])): dict(record)
        for record in resolution_manifest["resolution_partitions"]
    }

    policy_records: list[dict[str, Any]] = []
    version_records: list[dict[str, Any]] = []
    first_records: list[dict[str, Any]] = []
    update_records: list[dict[str, Any]] = []
    revision_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []

    for cost_scenario in replay.COST_SCENARIOS:
        previous_v0 = pd.DataFrame()
        previous_year: int | None = None
        for as_of_year in formal_years:
            available = int(psutil.virtual_memory().available / 1024**2)
            if available <= reserve_memory:
                raise MemoryError(
                    f"dynamic_oracle_causal_prefix_memory_reserve:{available}"
                )
            prefix_market = truncate_market(
                full_market, terminal_date_idx=cutoffs[as_of_year]
            )
            cutoff_started = time.perf_counter()
            solution = oracle.solve_relaxed_oracle(
                market=prefix_market,
                study=oracle_study,
                cost_scenario=cost_scenario,
                output_root=root,
                write_outputs=False,
            )
            graph = resolution.build_cash_policy_graph_from_arrays(
                market=prefix_market,
                cash_action_symbol=solution.cash_action_symbol,
                cash_log_value=solution.cash_log,
                holding_exit_policy=solution.hold_exit_policy,
            )
            policy_records.append(
                _policy_record(
                    root=root,
                    solution=solution,
                    market=prefix_market,
                    as_of_year=as_of_year,
                )
            )
            current_v0 = extract_prefix_version(
                market=prefix_market,
                solution=solution,
                graph=graph,
                signal_year=as_of_year,
                as_of_year=as_of_year,
                version_role="signal_year_end",
            )
            v0_path = _version_path(
                root,
                cost_scenario=cost_scenario,
                as_of_year=as_of_year,
                signal_year=as_of_year,
            )
            v0_record = _write_frame(
                v0_path, current_v0, row_group_size=row_group_size
            )
            v0_record.update(
                {
                    "cost_scenario": cost_scenario,
                    "as_of_year": as_of_year,
                    "signal_year": as_of_year,
                    "version_role": "signal_year_end",
                }
            )
            version_records.append(v0_record)

            if previous_year is not None:
                current_v1 = extract_prefix_version(
                    market=prefix_market,
                    solution=solution,
                    graph=graph,
                    signal_year=previous_year,
                    as_of_year=as_of_year,
                    version_role="next_year_end",
                )
                v1_path = _version_path(
                    root,
                    cost_scenario=cost_scenario,
                    as_of_year=as_of_year,
                    signal_year=previous_year,
                )
                v1_record = _write_frame(
                    v1_path, current_v1, row_group_size=row_group_size
                )
                v1_record.update(
                    {
                        "cost_scenario": cost_scenario,
                        "as_of_year": as_of_year,
                        "signal_year": previous_year,
                        "version_role": "next_year_end",
                    }
                )
                version_records.append(v1_record)
                first, updates = build_first_experience_and_updates(
                    signal_year_end=previous_v0,
                    next_year_end=current_v1,
                    tolerance=tolerance,
                )
                first_path = (
                    root
                    / "first_learnable_experience"
                    / f"cost={cost_scenario}"
                    / f"signal_year={previous_year}"
                    / "part-0000.parquet"
                )
                first_record = _write_frame(
                    first_path, first, row_group_size=row_group_size
                )
                first_record.update(
                    {"cost_scenario": cost_scenario, "signal_year": previous_year}
                )
                first_records.append(first_record)
                if not updates.empty:
                    update_path = (
                        root
                        / "annual_experience_updates"
                        / f"cost={cost_scenario}"
                        / f"signal_year={previous_year}"
                        / "part-0000.parquet"
                    )
                    update_record = _write_frame(
                        update_path, updates, row_group_size=row_group_size
                    )
                    update_record.update(
                        {
                            "cost_scenario": cost_scenario,
                            "signal_year": previous_year,
                        }
                    )
                    update_records.append(update_record)
                final_reference = _read_final_resolution(
                    final_records[(cost_scenario, previous_year)],
                    cost_scenario=cost_scenario,
                    signal_year=previous_year,
                )
                revision_rows.append(
                    _revision_summary(
                        cost_scenario=cost_scenario,
                        signal_year=previous_year,
                        signal_year_end=previous_v0,
                        next_year_end=current_v1,
                        first_experience=first,
                        final_reference=final_reference,
                        tolerance=tolerance,
                    )
                )

            cutoff_elapsed = float(time.perf_counter() - cutoff_started)
            cutoff_rows.append(
                {
                    "cost_scenario": cost_scenario,
                    "as_of_year": as_of_year,
                    "as_of_date": str(prefix_market.date_values[prefix_market.end_idx]),
                    "as_of_date_idx": int(prefix_market.end_idx),
                    "date_count": int(prefix_market.day_count),
                    "quality_rows": int(prefix_market.quality_rows),
                    "terminal_log_wealth": float(solution.terminal_log_wealth),
                    "selected_trade_rows": len(solution.selected_trades),
                    "signal_year_version_rows": len(current_v0),
                    "previous_year_version_rows": (
                        len(current_v1) if previous_year is not None else 0
                    ),
                    "elapsed_seconds": cutoff_elapsed,
                }
            )
            print(
                json.dumps(
                    {
                        "cost_scenario": cost_scenario,
                        "as_of_year": as_of_year,
                        "signal_year_rows": len(current_v0),
                        "elapsed_seconds": cutoff_elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            previous_v0 = current_v0
            previous_year = as_of_year
            minimum_available = min(
                minimum_available,
                int(psutil.virtual_memory().available / 1024**2),
            )
            del graph, solution, prefix_market
            gc.collect()

        if previous_year is None:
            raise AssertionError("dynamic_oracle_causal_prefix_no_years")
        empty_next = pd.DataFrame(columns=previous_v0.columns)
        first, _ = build_first_experience_and_updates(
            signal_year_end=previous_v0,
            next_year_end=empty_next,
            tolerance=tolerance,
        )
        first_path = (
            root
            / "first_learnable_experience"
            / f"cost={cost_scenario}"
            / f"signal_year={previous_year}"
            / "part-0000.parquet"
        )
        first_record = _write_frame(
            first_path, first, row_group_size=row_group_size
        )
        first_record.update(
            {"cost_scenario": cost_scenario, "signal_year": previous_year}
        )
        first_records.append(first_record)
        final_reference = _read_final_resolution(
            final_records[(cost_scenario, previous_year)],
            cost_scenario=cost_scenario,
            signal_year=previous_year,
        )
        revision_rows.append(
            _revision_summary(
                cost_scenario=cost_scenario,
                signal_year=previous_year,
                signal_year_end=previous_v0,
                next_year_end=empty_next,
                first_experience=first,
                final_reference=final_reference,
                tolerance=tolerance,
            )
        )

    cutoff_frame = pd.DataFrame(cutoff_rows).sort_values(
        ["cost_scenario", "as_of_year"], kind="mergesort"
    )
    revision_frame = pd.DataFrame(revision_rows).sort_values(
        ["cost_scenario", "signal_year"], kind="mergesort"
    )
    cutoff_output = _write_frame(
        root / "annual_prefix_summary.parquet",
        cutoff_frame,
        row_group_size=row_group_size,
    )
    revision_output = _write_frame(
        root / "revision_diagnostics.parquet",
        revision_frame,
        row_group_size=row_group_size,
    )
    memory_end = int(psutil.virtual_memory().available / 1024**2)
    manifest = {
        "schema": f"seq100_dynamic_oracle_causal_prefix/{SCHEMA_VERSION}",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": {
            "path": str(study_file.resolve()),
            "sha256": replay.sha256(study_file),
        },
        "source_contract": source_contract,
        "market_audit": market_audit,
        "annual_cutoffs": [
            {
                "year": year,
                "date_idx": cutoffs[year],
                "date": str(full_market.date_values[cutoffs[year]]),
            }
            for year in formal_years
        ],
        "prefix_policies": policy_records,
        "experience_versions": version_records,
        "first_learnable_partitions": first_records,
        "annual_update_partitions": update_records,
        "outputs": {
            "annual_prefix_summary": cutoff_output,
            "revision_diagnostics": revision_output,
        },
        "audit": {
            "prefix_policy_count": len(policy_records),
            "experience_version_partition_count": len(version_records),
            "first_learnable_partition_count": len(first_records),
            "annual_update_partition_count": len(update_records),
            "experience_version_rows": int(
                sum(int(record["rows"]) for record in version_records)
            ),
            "first_learnable_rows": int(
                sum(int(record["rows"]) for record in first_records)
            ),
            "annual_update_rows": int(
                sum(int(record["rows"]) for record in update_records)
            ),
            "forbidden_2026_rows": 0,
            "final_oracle_row_labels_in_causal_outputs": False,
        },
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "available_memory_mb_at_start": memory_start,
            "available_memory_mb_at_end": memory_end,
            "minimum_available_memory_mb_observed": minimum_available,
            "reserve_memory_mb": reserve_memory,
            "logical_cpu_count": int(psutil.cpu_count() or 1),
            "adaptive_memory": bool(dict(study["resources"])["adaptive_memory"]),
        },
        "annual_as_of_experience_built": True,
        "daily_first_availability_claimed": False,
        "fixed_holding_horizon_used": False,
        "binary_good_stock_label_used": False,
        "final_oracle_used_only_for_aggregate_diagnostics": True,
        "training_performed": False,
        "prediction_performed": False,
        "causal_policy_evaluated": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build annual as-of dynamic-oracle experience versions."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "prefix_policy_count": result["audit"]["prefix_policy_count"],
                "experience_version_rows": result["audit"][
                    "experience_version_rows"
                ],
                "first_learnable_rows": result["audit"]["first_learnable_rows"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
