"""Independent checks for the causal K-line strategy probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_pattern_strategy_probe as probe
from daily_research.path_policy import seq100_dynamic_oracle as dynamic_oracle
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import _buy_order, _sell_order


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (probe.WORKSPACE_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(_resolve(path).read_text(encoding="utf-8")))


def _assert_close(
    actual: float, expected: float, name: str, tolerance: float = 1.0e-9
) -> None:
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ):
        raise AssertionError(
            f"pattern_strategy_validation_value:{name}:{actual}:{expected}"
        )


def validate(
    *,
    study_path: str | Path = probe.DEFAULT_STUDY_PATH,
    output_root: str | Path = probe.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = probe.load_study(study_path)
    root = _resolve(output_root)
    manifest = _read_json(root / "manifest.json")
    trades = pd.read_parquet(root / "trades.parquet")
    summary = pd.read_parquet(root / "summary.parquet")
    if int(manifest.get("trades", -1)) != len(trades):
        raise AssertionError("pattern_strategy_validation_trade_count")
    if set(trades["strategy"].unique()) - set(probe.STRATEGY_NAMES):
        raise AssertionError("pattern_strategy_validation_unknown_strategy")
    if trades["signal_date"].astype(str).str.startswith("2026").any():
        raise AssertionError("pattern_strategy_validation_forbidden_signal_year")
    if trades["entry_date"].astype(str).str.startswith("2026").any():
        raise AssertionError("pattern_strategy_validation_forbidden_entry_year")

    oracle_study = dynamic_oracle.load_study(
        _resolve(study["source"]["dynamic_oracle_study"])
    )
    market, _ = replay.load_market(oracle_study)
    symbols = trades["symbol_idx"].to_numpy(np.int64)
    signal = trades["signal_date_idx"].to_numpy(np.int64)
    entry = trades["entry_date_idx"].to_numpy(np.int64)
    exit_idx = trades["exit_date_idx"].to_numpy(np.int64)
    if bool((entry <= signal).any()):
        raise AssertionError("pattern_strategy_validation_t_plus_one_entry")
    resolved = exit_idx >= 0
    if bool((exit_idx[resolved] <= entry[resolved]).any()):
        raise AssertionError("pattern_strategy_validation_t_plus_one_exit")
    expected_entry = np.asarray(market.entry_open_raw[entry, symbols], dtype=float)
    actual_entry = trades["entry_price_raw"].to_numpy(float)
    if not np.allclose(
        expected_entry, actual_entry, rtol=0.0, atol=1.0e-6, equal_nan=False
    ):
        raise AssertionError("pattern_strategy_validation_entry_price")
    if bool((~np.asarray(market.entry_filled[signal, symbols], dtype=bool)).any()):
        raise AssertionError("pattern_strategy_validation_unfilled_trade")
    if bool(resolved.any()):
        expected_exit = np.asarray(
            market.exit_close_raw[exit_idx[resolved], symbols[resolved]], dtype=float
        )
        actual_exit = trades.loc[resolved, "exit_price_raw"].to_numpy(float)
        if not np.allclose(
            expected_exit, actual_exit, rtol=0.0, atol=1.0e-6, equal_nan=False
        ):
            raise AssertionError("pattern_strategy_validation_exit_price")
        sellable = np.asarray(
            market.exit_sellable[exit_idx[resolved], symbols[resolved]], dtype=bool
        )
        if not bool(sellable.all()):
            raise AssertionError("pattern_strategy_validation_unsellable_exit")

    cost_error = 0.0
    starting_cash = float(study["execution"]["starting_cash_cny_per_trade"])
    for cost in replay.COST_SCENARIOS:
        multiplier = (
            1.0 if cost == "base" else float(market.costs.stress_slippage_multiplier)
        )
        expected = np.full(len(trades), -1.0, dtype=float)
        expected_filled = np.ones(len(trades), dtype=bool)
        for position in np.flatnonzero(resolved):
            row = trades.iloc[int(position)]
            shares, buy_cash, _, _ = _buy_order(
                available_cash=starting_cash,
                allocated_cash=starting_cash,
                entry_price=float(row.entry_price_raw),
                contract=market.costs,
                slippage_multiplier=multiplier,
            )
            if shares <= 0:
                expected[int(position)] = 0.0
                expected_filled[int(position)] = False
                continue
            proceeds, _, _ = _sell_order(
                shares=shares,
                exit_price=float(row.exit_price_raw),
                exit_date_idx=int(row.exit_date_idx),
                date_values=market.date_values,
                contract=market.costs,
                slippage_multiplier=multiplier,
            )
            expected[int(position)] = (
                starting_cash - buy_cash + proceeds
            ) / starting_cash - 1.0
        actual = trades[f"net_return_{cost}"].to_numpy(float)
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-9, equal_nan=False):
            raise AssertionError(f"pattern_strategy_validation_cost:{cost}")
        if not np.array_equal(
            trades[f"finite_order_filled_{cost}"].to_numpy(bool), expected_filled
        ):
            raise AssertionError(f"pattern_strategy_validation_finite_fill:{cost}")
        cost_error = max(cost_error, float(np.max(np.abs(actual - expected))))

    overlap_violations = 0
    for (_strategy, symbol), group in trades.groupby(
        ["strategy", "symbol_idx"], sort=False
    ):
        ordered = group.sort_values("entry_date_idx", kind="mergesort")
        previous_exit = -1
        for row in ordered.itertuples(index=False):
            if int(row.entry_date_idx) <= previous_exit:
                overlap_violations += 1
            previous_exit = max(previous_exit, int(row.exit_date_idx))
    if overlap_violations:
        raise AssertionError(
            f"pattern_strategy_validation_overlap:{overlap_violations}"
        )
    if len(summary) != len(probe.STRATEGY_NAMES) * len(replay.COST_SCENARIOS):
        raise AssertionError("pattern_strategy_validation_summary_rows")
    result = {
        "status": "passed",
        "trade_rows": len(trades),
        "resolved_rows": int(resolved.sum()),
        "terminal_unresolved_rows": int((~resolved).sum()),
        "overlap_violations": int(overlap_violations),
        "maximum_cost_recomputation_error": float(cost_error),
        "forbidden_2026_rows": 0,
    }
    (root / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Validate the causal K-line strategy probe."
    )
    parser.add_argument("--study-path", default=str(probe.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(probe.DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    result = validate(study_path=args.study_path, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
