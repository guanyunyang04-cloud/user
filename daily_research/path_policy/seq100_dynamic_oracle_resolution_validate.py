"""Independent validation for dynamic oracle cash-branch resolution."""

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

from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy import seq100_market_replay as replay

VALIDATION_SCHEMA = "seq100_dynamic_oracle_resolution_validation/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _next_legal_sale_local(
    *, market: replay.ReplayMarket, local: int, symbol_idx: int
) -> int:
    for candidate in range(int(local) + 1, market.day_count):
        absolute = int(market.start_idx + candidate)
        price = float(market.exit_close_raw[absolute, int(symbol_idx)])
        if (
            bool(market.exit_sellable[absolute, int(symbol_idx)])
            and math.isfinite(price)
            and price > 0.0
        ):
            return candidate
    return -1


def _trace_holding_to_cash(
    *,
    market: replay.ReplayMarket,
    hold_exit_policy: np.ndarray,
    local: int,
    symbol_idx: int,
) -> int:
    current = int(local)
    while current < market.day_count - 1:
        if bool(hold_exit_policy[current, int(symbol_idx)]):
            return _next_legal_sale_local(
                market=market, local=current, symbol_idx=int(symbol_idx)
            )
        current += 1
    return -1


def _trace_cash_parent(
    *,
    market: replay.ReplayMarket,
    cash_action_symbol: np.ndarray,
    hold_exit_policy: np.ndarray,
    local: int,
) -> int:
    selected = int(cash_action_symbol[int(local)])
    if selected < 0:
        return int(local) + 1
    return _trace_holding_to_cash(
        market=market,
        hold_exit_policy=hold_exit_policy,
        local=int(local) + 1,
        symbol_idx=selected,
    )


def _trace_resolution(
    *,
    market: replay.ReplayMarket,
    cash_action_symbol: np.ndarray,
    hold_exit_policy: np.ndarray,
    signal_local: int,
    symbol_idx: int,
) -> tuple[int, int]:
    buy_first_cash = _trace_holding_to_cash(
        market=market,
        hold_exit_policy=hold_exit_policy,
        local=int(signal_local) + 1,
        symbol_idx=int(symbol_idx),
    )
    if buy_first_cash < 0:
        return -1, -1
    left = int(buy_first_cash)
    right = int(signal_local) + 1
    steps = 0
    while left != right:
        if left < right:
            left = _trace_cash_parent(
                market=market,
                cash_action_symbol=cash_action_symbol,
                hold_exit_policy=hold_exit_policy,
                local=left,
            )
        else:
            right = _trace_cash_parent(
                market=market,
                cash_action_symbol=cash_action_symbol,
                hold_exit_policy=hold_exit_policy,
                local=right,
            )
        steps += 1
        if left < 0 or right < 0 or steps > 2 * market.day_count:
            return buy_first_cash, -1
    return buy_first_cash, left


def validate(
    *,
    study_path: str | Path = resolution.DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
    trace_sample_per_partition: int = 64,
) -> dict[str, Any]:
    study = resolution.load_study(study_path)
    root = resolution._resolve(output_root or study["output_root"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = resolution._read_json(manifest_path)
    blocking: defaultdict[str, int] = defaultdict(int)
    if manifest.get("status") != "completed":
        blocking["manifest_status"] += 1
    if manifest.get("study_id") != resolution.STUDY_ID:
        blocking["study_id"] += 1
    if bool(manifest.get("coalescence_claimed_as_causal_label_time")):
        blocking["causal_claim"] += 1

    _, oracle_manifest, oracle_study = resolution._source_contract(study)
    market, _ = replay.load_market(oracle_study)
    oracle_records = {
        str(record["cost_scenario"]): dict(record)
        for record in oracle_manifest["relaxed_oracles"]
    }
    source_labels = {
        cost: resolution._record_by_year(record["action_labels"])
        for cost, record in oracle_records.items()
    }
    output_records = {
        (str(record["cost_scenario"]), int(record["year"])): dict(record)
        for record in manifest["resolution_partitions"]
    }
    expected_pairs = {
        (cost, year)
        for cost in replay.COST_SCENARIOS
        for year in (int(value) for value in dict(study["period"])["formal_years"])
    }
    if set(output_records) != expected_pairs:
        blocking["partition_coverage"] += len(
            expected_pairs.symmetric_difference(output_records)
        )

    traced_rows = 0
    total_rows = 0
    maximum_identity_error = 0.0
    longest_rows: list[dict[str, Any]] = []
    for cost_scenario in replay.COST_SCENARIOS:
        oracle_record = oracle_records[cost_scenario]
        cash_action = resolution._load_array(oracle_record["cash_action_symbol"])
        hold_exit = resolution._load_array(
            oracle_record["holding_exit_policy"], mmap=True
        )
        for year in sorted(year for cost, year in expected_pairs if cost == cost_scenario):
            output_record = output_records.get((cost_scenario, year))
            if output_record is None:
                continue
            path = Path(str(output_record["path"]))
            if not path.is_file() or replay.sha256(path) != str(
                output_record["sha256"]
            ):
                blocking["partition_hash"] += 1
                continue
            frame = pd.read_parquet(path)
            total_rows += len(frame)
            if len(frame) != int(output_record["rows"]):
                blocking["partition_rows"] += 1
            if frame.duplicated(["date_idx", "symbol_idx"]).any():
                blocking["partition_duplicates"] += 1
            if frame["trade_date"].astype(str).str[:4].astype(int).ge(2026).any():
                blocking["forbidden_2026_rows"] += 1
            source = resolution._read_labels(source_labels[cost_scenario][year])
            source_valid = source.loc[source["buy_action_valid"].astype(bool)]
            source_keys = (
                source_valid["date_idx"].to_numpy(np.int64) << np.int64(32)
            ) | source_valid["symbol_idx"].to_numpy(np.int64)
            output_keys = (
                frame["date_idx"].to_numpy(np.int64) << np.int64(32)
            ) | frame["symbol_idx"].to_numpy(np.int64)
            if not np.array_equal(source_keys, output_keys):
                blocking["source_key_alignment"] += 1
            first = frame["buy_first_cash_date_idx"].to_numpy(np.int64)
            meeting = frame["cash_resolution_date_idx"].to_numpy(np.int64)
            signal = frame["date_idx"].to_numpy(np.int64)
            if bool((first < signal + 2).any() or (meeting < first).any()):
                blocking["resolution_order"] += 1
            errors = np.abs(
                frame["stored_value_identity_error"].to_numpy(np.float64)
            )
            maximum_identity_error = max(
                maximum_identity_error, float(errors.max(initial=0.0))
            )
            longest_rows.extend(
                frame.nlargest(2, "cash_resolution_delay")[
                    [
                        "cost_scenario",
                        "trade_date",
                        "date_idx",
                        "symbol_idx",
                        "buy_first_cash_date_idx",
                        "cash_resolution_date_idx",
                        "cash_resolution_delay",
                    ]
                ].to_dict(orient="records")
            )
            sample = frame.sample(
                n=min(int(trace_sample_per_partition), len(frame)),
                random_state=20260807 + year + (0 if cost_scenario == "base" else 10_000),
            )
            for row in sample.itertuples(index=False):
                first_local, meeting_local = _trace_resolution(
                    market=market,
                    cash_action_symbol=cash_action,
                    hold_exit_policy=hold_exit,
                    signal_local=int(row.date_idx) - int(market.start_idx),
                    symbol_idx=int(row.symbol_idx),
                )
                expected_first = first_local + int(market.start_idx)
                expected_meeting = meeting_local + int(market.start_idx)
                if expected_first != int(row.buy_first_cash_date_idx):
                    blocking["trace_first_cash"] += 1
                if expected_meeting != int(row.cash_resolution_date_idx):
                    blocking["trace_meeting_cash"] += 1
                traced_rows += 1
        mapping = getattr(hold_exit, "_mmap", None)
        if mapping is not None:
            mapping.close()

    tolerance = float(dict(study["resolution"])["value_identity_tolerance"])
    if maximum_identity_error > tolerance:
        blocking["value_identity"] += 1
    prefix_record = dict(dict(manifest["outputs"])["prefix_sensitivity"])
    prefix_path = Path(str(prefix_record["path"]))
    if not prefix_path.is_file() or replay.sha256(prefix_path) != str(
        prefix_record["sha256"]
    ):
        blocking["prefix_hash"] += 1
        prefix = pd.DataFrame()
    else:
        prefix = pd.read_parquet(prefix_path)
        if len(prefix) != int(prefix_record["rows"]):
            blocking["prefix_rows"] += 1
        if prefix.duplicated(["date_idx", "symbol_idx"]).any():
            blocking["prefix_duplicates"] += 1
        finite = prefix["prefix_value_finite"].astype(bool)
        recomputed_difference = (
            prefix["prefix_terminal_advantage"].astype(float)
            - prefix["final_terminal_advantage"].astype(float)
        )
        if not np.allclose(
            recomputed_difference[finite],
            prefix.loc[finite, "prefix_minus_final"].astype(float),
            atol=1.0e-12,
            rtol=0.0,
        ):
            blocking["prefix_difference_identity"] += 1

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": resolution.STUDY_ID,
        "blocking": dict(blocking),
        "partition_rows": total_rows,
        "independently_traced_rows": traced_rows,
        "maximum_stored_value_identity_error": maximum_identity_error,
        "prefix_rows": len(prefix),
        "longest_partition_examples": sorted(
            longest_rows,
            key=lambda row: int(row["cash_resolution_delay"]),
            reverse=True,
        )[:10],
    }
    _write_json(root / "validation.json", result)
    if blocking:
        raise ValueError(
            f"dynamic_oracle_resolution_validation_failed:{dict(blocking)}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate dynamic oracle branch-resolution outputs."
    )
    parser.add_argument("--study", default=str(resolution.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--trace-sample-per-partition", type=int, default=64)
    args = parser.parse_args(argv)
    result = validate(
        study_path=args.study,
        output_root=args.output_root,
        trace_sample_per_partition=args.trace_sample_per_partition,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
