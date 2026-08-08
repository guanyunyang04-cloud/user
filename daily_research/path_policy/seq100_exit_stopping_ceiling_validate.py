"""Independent checks for the future-informed stopping ceiling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle as dynamic_oracle
from daily_research.path_policy import seq100_exit_stopping_ceiling as ceiling_module
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import _buy_order, _sell_order


def _resolve(value: str | Path) -> Path:
    return ceiling_module._resolve(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return ceiling_module._read_json(path)


def _brute_force(
    *,
    row: pd.Series,
    market: replay.ReplayMarket,
    liquidation_abs: int,
    maximum_sessions: int,
    starting_cash: float,
    cost: str,
) -> tuple[float, int, int]:
    entry_abs = int(row["entry_date_idx"])
    symbol = int(row["symbol_idx"])
    entry_price = float(market.entry_open_raw[entry_abs, symbol])
    end_abs = min(int(market.end_idx), entry_abs + int(maximum_sessions))
    multiplier = (
        1.0
        if cost == "base"
        else float(market.costs.stress_slippage_multiplier)
    )
    shares, buy_cash, _, _ = _buy_order(
        available_cash=starting_cash,
        allocated_cash=starting_cash,
        entry_price=entry_price,
        contract=market.costs,
        slippage_multiplier=multiplier,
    )
    last_request = min(entry_abs + maximum_sessions - 1, end_abs - 1)
    last_request = min(last_request, liquidation_abs)
    best_return = -math.inf
    best_request = -1
    best_exit = -1
    for request_abs in range(entry_abs, last_request + 1):
        exit_abs = -1
        for candidate in range(request_abs + 1, end_abs + 1):
            price = float(market.exit_close_raw[candidate, symbol])
            if (
                bool(market.exit_sellable[candidate, symbol])
                and math.isfinite(price)
                and price > 0.0
            ):
                exit_abs = candidate
                break
        if exit_abs < 0:
            continue
        if shares <= 0:
            net_return = 0.0
        else:
            proceeds, _, _ = _sell_order(
                shares=shares,
                exit_price=float(market.exit_close_raw[exit_abs, symbol]),
                exit_date_idx=exit_abs,
                date_values=market.date_values,
                contract=market.costs,
                slippage_multiplier=multiplier,
            )
            net_return = (
                starting_cash - buy_cash + proceeds
            ) / starting_cash - 1.0
        if net_return > best_return:
            best_return = float(net_return)
            best_request = request_abs
            best_exit = exit_abs
    if not math.isfinite(best_return):
        return -1.0, -1, -1
    return best_return, best_request, best_exit


def validate(
    *,
    study_path: str | Path = ceiling_module.DEFAULT_STUDY_PATH,
    output_root: str | Path = ceiling_module.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = ceiling_module.load_study(study_path)
    root = _resolve(output_root)
    manifest = _read_json(root / "manifest.json")
    source = dict(study["source"])
    for path_key, hash_key in (
        ("mechanical_manifest", "expected_mechanical_manifest_sha256"),
        ("entry_records", "expected_entry_records_sha256"),
        (
            "mechanical_policy_results_manifest",
            "expected_mechanical_policy_results_manifest_sha256",
        ),
    ):
        if ceiling_module._sha256(_resolve(source[path_key])) != str(source[hash_key]):
            raise AssertionError(f"stopping_ceiling_validation_source:{path_key}")
    entries = pd.read_parquet(_resolve(source["entry_records"]))
    ceiling = pd.read_parquet(root / "stopping_ceiling.parquet")
    if len(entries) != len(ceiling) or len(ceiling) != int(manifest["entries"]):
        raise AssertionError("stopping_ceiling_validation_rows")
    key_columns = ["entry_pattern", "signal_date_idx", "symbol_idx"]
    left = entries[key_columns].copy()
    right = ceiling[key_columns].copy()
    for frame in (left, right):
        frame[["signal_date_idx", "symbol_idx"]] = frame[
            ["signal_date_idx", "symbol_idx"]
        ].astype(np.int64)
    if not np.array_equal(left.to_numpy(object), right.to_numpy(object)):
        raise AssertionError("stopping_ceiling_validation_keys")
    if bool(ceiling["signal_date"].astype(str).str.startswith("2026").any()):
        raise AssertionError("stopping_ceiling_validation_forbidden_year")
    for cost in replay.COST_SCENARIOS:
        resolved = ceiling[f"ceiling_exit_date_idx_{cost}"].ge(0)
        if bool(
            ceiling.loc[resolved, f"ceiling_exit_date_idx_{cost}"]
            .le(ceiling.loc[resolved, f"ceiling_request_date_idx_{cost}"])
            .any()
        ):
            raise AssertionError("stopping_ceiling_validation_same_close_exit")
        increment = (
            ceiling[f"ceiling_net_return_{cost}"]
            - ceiling[f"best_mechanical_{cost}"]
        )
        if not np.allclose(
            increment,
            ceiling[f"ceiling_increment_vs_mechanical_{cost}"],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError("stopping_ceiling_validation_increment")
        if bool((increment < -1.0e-12).any()):
            raise AssertionError("stopping_ceiling_validation_below_mechanical")

    oracle_study = dynamic_oracle.load_study(_resolve(source["dynamic_oracle_study"]))
    market, _ = replay.load_market(oracle_study)
    liquidation_abs = int(
        np.flatnonzero(
            market.date_values.astype(str)
            == str(study["period"]["terminal_liquidation_start"])
        )[0]
    )
    positions = np.linspace(0, len(ceiling) - 1, 1_000, dtype=np.int64)
    maximum_error = 0.0
    request_errors = 0
    exit_errors = 0
    starting_cash = float(study["execution"]["starting_cash_cny_per_entry"])
    for position in positions:
        row = ceiling.iloc[int(position)]
        for cost in replay.COST_SCENARIOS:
            expected_return, expected_request, expected_exit = _brute_force(
                row=row,
                market=market,
                liquidation_abs=liquidation_abs,
                maximum_sessions=int(study["episodes"]["maximum_sessions"]),
                starting_cash=starting_cash,
                cost=cost,
            )
            actual_return = float(row[f"ceiling_net_return_{cost}"])
            maximum_error = max(maximum_error, abs(actual_return - expected_return))
            request_errors += int(
                expected_request != int(row[f"ceiling_request_date_idx_{cost}"])
            )
            exit_errors += int(
                expected_exit != int(row[f"ceiling_exit_date_idx_{cost}"])
            )
            if not math.isclose(
                actual_return, expected_return, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise AssertionError("stopping_ceiling_validation_return")
    if request_errors or exit_errors:
        raise AssertionError("stopping_ceiling_validation_path_selection")

    summary = pd.read_parquet(root / "summary.parquet")
    if len(summary) != len(ceiling_module.COST_SCENARIOS) * len(
        ceiling_module.baselines.ENTRY_PATTERNS
    ):
        raise AssertionError("stopping_ceiling_validation_summary")
    result = {
        "status": "passed",
        "entry_rows": len(ceiling),
        "sampled_entries": len(positions),
        "sampled_cost_paths": len(positions) * len(replay.COST_SCENARIOS),
        "maximum_net_return_error": float(maximum_error),
        "request_date_errors": int(request_errors),
        "exit_date_errors": int(exit_errors),
        "forbidden_2026_rows": 0,
        "future_informed_teacher": True,
        "causal_policy_evaluated": False,
    }
    (root / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate stopping ceiling.")
    parser.add_argument("--study", default=str(ceiling_module.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(ceiling_module.DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    result = validate(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
