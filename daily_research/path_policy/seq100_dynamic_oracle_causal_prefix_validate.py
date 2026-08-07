"""Independent validation for the annual causal-prefix experience ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle_causal_prefix as prefix
from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy import (
    seq100_dynamic_oracle_resolution_validate as resolution_validate,
)
from daily_research.path_policy import seq100_market_replay as replay

VALIDATION_SCHEMA = "seq100_dynamic_oracle_causal_prefix_validation/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_frame(
    record: Mapping[str, Any], *, blocking: defaultdict[str, int], name: str
) -> pd.DataFrame:
    path = Path(str(record["path"]))
    if not path.is_file() or replay.sha256(path) != str(record["sha256"]):
        blocking[f"{name}_hash"] += 1
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if len(frame) != int(record["rows"]):
        blocking[f"{name}_rows"] += 1
    return frame


def _load_array_record(
    record: Mapping[str, Any], *, blocking: defaultdict[str, int], name: str
) -> np.ndarray:
    path = Path(str(record["path"]))
    if not path.is_file() or replay.sha256(path) != str(record["sha256"]):
        blocking[f"{name}_hash"] += 1
        return np.empty(0)
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != list(record["shape"]):
        blocking[f"{name}_shape_record"] += 1
    if str(values.dtype) != str(record["dtype"]):
        blocking[f"{name}_dtype_record"] += 1
    return values


def _version_expected_keys(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["date_idx"].to_numpy(np.int64) << np.int64(32)
    ) | frame["symbol_idx"].to_numpy(np.int64)


def _validate_version_frame(
    *,
    frame: pd.DataFrame,
    record: Mapping[str, Any],
    market: replay.ReplayMarket,
    cash_log: np.ndarray,
    cash_action: np.ndarray,
    hold_exit: np.ndarray,
    trace_rows: int,
    blocking: defaultdict[str, int],
) -> int:
    if frame.empty:
        blocking["empty_version_partition"] += 1
        return 0
    required = set(prefix.VERSION_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        blocking["version_columns"] += len(missing)
        return 0
    if any(str(column).startswith("final_") for column in frame.columns):
        blocking["final_row_label_in_version"] += 1
    if frame.duplicated(list(prefix.KEY_COLUMNS)).any():
        blocking["version_duplicates"] += 1
    keys = _version_expected_keys(frame)
    if bool((keys[1:] <= keys[:-1]).any()):
        blocking["version_order"] += 1
    as_of_year = int(record["as_of_year"])
    signal_year = int(record["signal_year"])
    if not frame["as_of_year"].astype(int).eq(as_of_year).all():
        blocking["version_as_of_year"] += 1
    if not frame["signal_year"].astype(int).eq(signal_year).all():
        blocking["version_signal_year"] += 1
    if not frame["as_of_date_idx"].astype(int).eq(int(market.end_idx)).all():
        blocking["version_as_of_idx"] += 1
    if frame["trade_date"].astype(str).str[:4].astype(int).ne(signal_year).any():
        blocking["version_trade_year"] += 1
    if frame["as_of_date"].astype(str).str[:4].astype(int).ge(2026).any():
        blocking["version_forbidden_year"] += 1
    numeric = frame.select_dtypes(include=["number"]).to_numpy(np.float64)
    if np.isinf(numeric).any():
        blocking["version_infinity"] += 1
    natural = frame["naturally_resolved"].astype(bool).to_numpy()
    terminal = frame["cash_resolution_is_terminal"].astype(bool).to_numpy()
    sessions_after = frame["sessions_after_resolution"].to_numpy(np.int64)
    if not np.array_equal(natural, ~terminal):
        blocking["version_natural_identity"] += 1
    if bool((natural & (sessions_after <= 0)).any()):
        blocking["version_natural_gap"] += 1
    if bool((terminal & (sessions_after != 0)).any()):
        blocking["version_terminal_gap"] += 1
    decomposition = (
        frame["buy_trade_log_return_to_first_cash"].to_numpy(np.float64)
        + frame["cash_value_difference_first_cash_vs_wait"].to_numpy(np.float64)
    )
    if not np.allclose(
        decomposition,
        frame["buy_advantage_vs_cash"].to_numpy(np.float64),
        atol=1.0e-7,
        rtol=0.0,
    ):
        blocking["version_decomposition"] += 1

    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, str(record["cost_scenario"])
    )
    sample = frame.sample(
        n=min(int(trace_rows), len(frame)),
        random_state=20260807 + 1009 * signal_year + 17 * as_of_year,
    )
    traced = 0
    for row in sample.itertuples(index=False):
        signal_local = int(row.date_idx) - int(market.start_idx)
        first_local, meeting_local = resolution_validate._trace_resolution(
            market=market,
            cash_action_symbol=cash_action,
            hold_exit_policy=hold_exit,
            signal_local=signal_local,
            symbol_idx=int(row.symbol_idx),
        )
        if first_local + int(market.start_idx) != int(row.buy_first_cash_date_idx):
            blocking["trace_first_cash"] += 1
        if meeting_local + int(market.start_idx) != int(row.cash_resolution_date_idx):
            blocking["trace_resolution"] += 1
        if first_local < 0 or meeting_local < 0:
            blocking["trace_unresolved"] += 1
            continue
        entry = float(
            market.entry_open_raw[int(row.date_idx) + 1, int(row.symbol_idx)]
        )
        sale_absolute = int(market.start_idx) + first_local
        sale = float(market.exit_close_raw[sale_absolute, int(row.symbol_idx)])
        direct = (
            math.log(
                sale
                * float(sell_multiplier[first_local])
                / (entry * float(buy_multiplier))
            )
            + float(cash_log[first_local])
            - float(cash_log[signal_local + 1])
        )
        if abs(direct - float(row.buy_advantage_vs_cash)) > 2.0e-6:
            blocking["trace_action_value"] += 1
        selected = int(cash_action[signal_local]) == int(row.symbol_idx)
        if selected != bool(row.prefix_oracle_cash_buy_selected):
            blocking["trace_cash_selection"] += 1
        traced += 1
    return traced


def _expected_first(
    signal_year_end: pd.DataFrame, next_year_end: pd.DataFrame
) -> pd.DataFrame:
    v0 = signal_year_end.loc[
        signal_year_end["naturally_resolved"].astype(bool)
    ].copy()
    v1 = next_year_end.loc[next_year_end["naturally_resolved"].astype(bool)].copy()
    keys = set(
        zip(v0["date_idx"].astype(int), v0["symbol_idx"].astype(int), strict=True)
    )
    add = v1.loc[
        [
            (int(date_idx), int(symbol_idx)) not in keys
            for date_idx, symbol_idx in zip(
                v1["date_idx"], v1["symbol_idx"], strict=True
            )
        ]
    ]
    parts = [v0]
    if not add.empty:
        parts.append(add)
    expected = pd.concat(parts, ignore_index=True).sort_values(
        list(prefix.KEY_COLUMNS), kind="mergesort"
    )
    return expected.reset_index(drop=True)


def validate(
    *,
    study_path: str | Path = prefix.DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
    trace_rows_per_partition: int | None = None,
) -> dict[str, Any]:
    study = prefix.load_study(study_path)
    root = prefix._resolve(output_root or study["output_root"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = prefix._read_json(manifest_path)
    blocking: defaultdict[str, int] = defaultdict(int)
    if manifest.get("status") != "completed":
        blocking["manifest_status"] += 1
    if manifest.get("study_id") != prefix.STUDY_ID:
        blocking["study_id"] += 1
    if not bool(manifest.get("final_oracle_used_only_for_aggregate_diagnostics")):
        blocking["final_oracle_boundary"] += 1
    if bool(manifest.get("daily_first_availability_claimed")):
        blocking["daily_availability_claim"] += 1
    if bool(manifest.get("training_performed")):
        blocking["training_performed"] += 1

    _, oracle_manifest, _resolution_manifest, oracle_study = prefix._source_contract(
        study
    )
    full_market, _ = replay.load_market(oracle_study)
    formal_years = tuple(int(value) for value in dict(study["period"])["formal_years"])
    cutoffs = prefix.annual_cutoffs(full_market, formal_years)
    trace_rows = int(
        trace_rows_per_partition
        if trace_rows_per_partition is not None
        else dict(study["resources"])["validator_trace_rows_per_partition"]
    )
    policy_records = {
        (str(record["cost_scenario"]), int(record["as_of_year"])): dict(record)
        for record in manifest["prefix_policies"]
    }
    expected_policies = {
        (cost, year) for cost in replay.COST_SCENARIOS for year in formal_years
    }
    if set(policy_records) != expected_policies:
        blocking["policy_coverage"] += len(
            expected_policies.symmetric_difference(policy_records)
        )
    version_records = {
        (
            str(record["cost_scenario"]),
            int(record["as_of_year"]),
            int(record["signal_year"]),
        ): dict(record)
        for record in manifest["experience_versions"]
    }
    expected_versions = {
        (cost, year, signal_year)
        for cost in replay.COST_SCENARIOS
        for year in formal_years
        for signal_year in ((year,) if year == formal_years[0] else (year - 1, year))
    }
    if set(version_records) != expected_versions:
        blocking["version_coverage"] += len(
            expected_versions.symmetric_difference(version_records)
        )

    final_oracles = {
        str(record["cost_scenario"]): dict(record)
        for record in oracle_manifest["relaxed_oracles"]
    }
    traced_rows = 0
    version_frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    for cost_scenario, as_of_year in sorted(expected_policies):
        record = policy_records.get((cost_scenario, as_of_year))
        if record is None:
            continue
        prefix_market = prefix.truncate_market(
            full_market, terminal_date_idx=cutoffs[as_of_year]
        )
        cash_log = _load_array_record(
            record["cash_log_value"], blocking=blocking, name="policy_cash_log"
        )
        cash_action = _load_array_record(
            record["cash_action_symbol"], blocking=blocking, name="policy_cash_action"
        )
        hold_exit = _load_array_record(
            record["holding_exit_policy"], blocking=blocking, name="policy_hold_exit"
        )
        if cash_log.shape != (prefix_market.day_count,):
            blocking["policy_cash_log_shape"] += 1
        if cash_action.shape != (prefix_market.day_count,):
            blocking["policy_cash_action_shape"] += 1
        if hold_exit.shape != (prefix_market.day_count, prefix_market.symbol_count):
            blocking["policy_hold_exit_shape"] += 1
        if int(record["as_of_date_idx"]) != int(prefix_market.end_idx):
            blocking["policy_as_of_idx"] += 1
        if int(str(record["as_of_date"])[:4]) >= 2026:
            blocking["policy_forbidden_year"] += 1

        signal_years = (
            (as_of_year,)
            if as_of_year == formal_years[0]
            else (as_of_year - 1, as_of_year)
        )
        for signal_year in signal_years:
            version_record = version_records.get(
                (cost_scenario, as_of_year, signal_year)
            )
            if version_record is None:
                continue
            frame = _read_frame(
                version_record, blocking=blocking, name="version_partition"
            )
            version_frames[(cost_scenario, as_of_year, signal_year)] = frame
            traced_rows += _validate_version_frame(
                frame=frame,
                record=version_record,
                market=prefix_market,
                cash_log=cash_log,
                cash_action=cash_action,
                hold_exit=hold_exit,
                trace_rows=trace_rows,
                blocking=blocking,
            )

        if as_of_year == formal_years[-1]:
            final_record = final_oracles[cost_scenario]
            final_cash_log = resolution._load_array(final_record["cash_log_value"])
            final_cash_action = resolution._load_array(
                final_record["cash_action_symbol"]
            )
            final_hold_exit = resolution._load_array(
                final_record["holding_exit_policy"]
            )
            if not np.allclose(cash_log, final_cash_log, atol=1.0e-12, rtol=0.0):
                blocking["final_cash_log_mismatch"] += 1
            if not np.array_equal(cash_action, final_cash_action):
                blocking["final_cash_action_mismatch"] += 1
            if not np.array_equal(hold_exit, final_hold_exit):
                blocking["final_hold_exit_mismatch"] += 1
        del cash_log, cash_action, hold_exit, prefix_market

    first_records = {
        (str(record["cost_scenario"]), int(record["signal_year"])): dict(record)
        for record in manifest["first_learnable_partitions"]
    }
    expected_first = {
        (cost, year) for cost in replay.COST_SCENARIOS for year in formal_years
    }
    if set(first_records) != expected_first:
        blocking["first_coverage"] += len(expected_first.symmetric_difference(first_records))
    first_rows = 0
    for cost_scenario, signal_year in sorted(expected_first):
        record = first_records.get((cost_scenario, signal_year))
        if record is None:
            continue
        frame = _read_frame(record, blocking=blocking, name="first_partition")
        first_rows += len(frame)
        if any(str(column).startswith("final_") for column in frame.columns):
            blocking["final_row_label_in_first"] += 1
        if frame.duplicated(list(prefix.KEY_COLUMNS)).any():
            blocking["first_duplicates"] += 1
        if not frame["naturally_resolved"].astype(bool).all():
            blocking["first_forced_terminal"] += 1
        v0 = version_frames[(cost_scenario, signal_year, signal_year)]
        v1 = (
            version_frames[(cost_scenario, signal_year + 1, signal_year)]
            if signal_year < formal_years[-1]
            else pd.DataFrame(columns=v0.columns)
        )
        expected = _expected_first(v0, v1)
        if not np.array_equal(_version_expected_keys(frame), _version_expected_keys(expected)):
            blocking["first_key_alignment"] += 1
        if len(frame) == len(expected) and not np.allclose(
            frame["buy_advantage_vs_cash"],
            expected["buy_advantage_vs_cash"],
            atol=1.0e-7,
            rtol=0.0,
        ):
            blocking["first_value_alignment"] += 1

    update_records = {
        (str(record["cost_scenario"]), int(record["signal_year"])): dict(record)
        for record in manifest["annual_update_partitions"]
    }
    expected_updates = {
        (cost, year)
        for cost in replay.COST_SCENARIOS
        for year in formal_years[:-1]
    }
    if set(update_records) != expected_updates:
        blocking["update_coverage"] += len(
            expected_updates.symmetric_difference(update_records)
        )
    update_rows = 0
    for cost_scenario, signal_year in sorted(expected_updates):
        record = update_records.get((cost_scenario, signal_year))
        if record is None:
            continue
        frame = _read_frame(record, blocking=blocking, name="update_partition")
        update_rows += len(frame)
        if frame.duplicated(list(prefix.KEY_COLUMNS)).any():
            blocking["update_duplicates"] += 1
        if any(str(column).startswith("final_") for column in frame.columns):
            blocking["final_row_label_in_update"] += 1
        delta = (
            frame["updated_buy_advantage_vs_cash"].to_numpy(np.float64)
            - frame["previous_buy_advantage_vs_cash"].to_numpy(np.float64)
        )
        if not np.allclose(
            delta,
            frame["update_minus_previous"].to_numpy(np.float64),
            atol=1.0e-7,
            rtol=0.0,
        ):
            blocking["update_delta_identity"] += 1
        v0 = version_frames[(cost_scenario, signal_year, signal_year)]
        v1 = version_frames[(cost_scenario, signal_year + 1, signal_year)]
        common = prefix._natural(v0)[list(prefix.KEY_COLUMNS)].merge(
            prefix._natural(v1)[list(prefix.KEY_COLUMNS)],
            on=list(prefix.KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if not np.array_equal(_version_expected_keys(frame), _version_expected_keys(common)):
            blocking["update_key_alignment"] += 1

    outputs = dict(manifest["outputs"])
    revision = _read_frame(
        outputs["revision_diagnostics"], blocking=blocking, name="revision_summary"
    )
    if len(revision) != len(replay.COST_SCENARIOS) * len(formal_years):
        blocking["revision_summary_rows"] += 1
    if revision.duplicated(["cost_scenario", "signal_year"]).any():
        blocking["revision_summary_duplicates"] += 1
    if np.isinf(revision.select_dtypes(include=["number"]).to_numpy(np.float64)).any():
        blocking["revision_summary_infinity"] += 1
    if int(dict(manifest["audit"])["forbidden_2026_rows"]) != 0:
        blocking["manifest_forbidden_2026"] += 1
    if bool(dict(manifest["audit"])["final_oracle_row_labels_in_causal_outputs"]):
        blocking["manifest_final_row_leak"] += 1
    if int(dict(manifest["audit"])["first_learnable_rows"]) != first_rows:
        blocking["manifest_first_rows"] += 1
    if int(dict(manifest["audit"])["annual_update_rows"]) != update_rows:
        blocking["manifest_update_rows"] += 1

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": prefix.STUDY_ID,
        "blocking": dict(blocking),
        "prefix_policies": len(policy_records),
        "experience_version_partitions": len(version_records),
        "first_learnable_rows": first_rows,
        "annual_update_rows": update_rows,
        "independently_traced_version_rows": traced_rows,
        "final_2025_policy_identity_checked": True,
    }
    _write_json(root / "validation.json", result)
    if blocking:
        raise ValueError(
            f"dynamic_oracle_causal_prefix_validation_failed:{dict(blocking)}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate annual as-of dynamic-oracle experience outputs."
    )
    parser.add_argument("--study", default=str(prefix.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--trace-rows-per-partition", type=int, default=None)
    args = parser.parse_args(argv)
    result = validate(
        study_path=args.study,
        output_root=args.output_root,
        trace_rows_per_partition=args.trace_rows_per_partition,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
