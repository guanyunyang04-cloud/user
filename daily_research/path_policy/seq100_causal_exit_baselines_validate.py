"""Independent validation for causal open-position exit baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from daily_research.path_policy import seq100_causal_exit_baselines as study_module
from daily_research.path_policy import seq100_dynamic_oracle as dynamic_oracle
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import _buy_order, _sell_order


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (study_module.WORKSPACE_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected_json_object:{path}")
    return payload


def _independent_signal_frame(
    *,
    market: replay.ReplayMarket,
    structure: Any,
    liquidation_date: str,
) -> pd.DataFrame:
    liquidation = np.flatnonzero(
        market.date_values.astype(str) == str(liquidation_date)
    )
    if len(liquidation) != 1:
        raise AssertionError("exit_baselines_validation_liquidation_date")
    liquidation_abs = int(liquidation[0])
    final_local = liquidation_abs - int(market.start_idx) - 2
    masks = {
        "retest": structure.retest,
        "nested_reacceleration": structure.nested_reacceleration,
        "breakout": structure.breakout,
    }
    frames: list[pd.DataFrame] = []
    for pattern in study_module.ENTRY_PATTERNS:
        mask = masks[pattern][: final_local + 1] & market.quality_mask[
            : final_local + 1
        ]
        local, symbols = np.nonzero(mask)
        absolute = local.astype(np.int64) + int(market.start_idx)
        filled = np.asarray(market.entry_filled[absolute, symbols], dtype=bool)
        prices = np.asarray(
            market.entry_open_raw[absolute + 1, symbols], dtype=np.float64
        )
        valid = filled & np.isfinite(prices) & (prices > 0.0)
        frame = pd.DataFrame(
            {
                "entry_pattern": pattern,
                "signal_date_idx": absolute[valid].astype(np.int32),
                "symbol_idx": symbols[valid].astype(np.int32),
            }
        )
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["entry_pattern", "signal_date_idx", "symbol_idx"], kind="mergesort"
        )
        .reset_index(drop=True)
    )


def _independent_request_and_exit(
    *,
    row: pd.Series,
    market: replay.ReplayMarket,
    structure: Any,
    liquidation_abs: int,
    maximum_sessions: int,
) -> tuple[int, int]:
    entry_abs = int(row["entry_date_idx"])
    signal_abs = int(row["signal_date_idx"])
    symbol = int(row["symbol_idx"])
    entry_price = float(market.entry_open_raw[entry_abs, symbol])
    end_abs = min(int(market.end_idx), entry_abs + int(maximum_sessions))
    days = np.arange(entry_abs, end_abs + 1, dtype=np.int32)
    prices = np.asarray(market.exit_close_raw[days, symbol], dtype=np.float64)
    filled = prices.copy()
    last = entry_price
    for position, value in enumerate(filled):
        if math.isfinite(float(value)) and float(value) > 0.0:
            last = float(value)
        else:
            filled[position] = last
    sessions = np.arange(1, len(days) + 1, dtype=np.int32)
    peak = np.maximum.accumulate(filled)
    peak_gain = peak / entry_price - 1.0
    drawdown = filled / peak - 1.0
    giveback = np.where(
        peak_gain > 0.0,
        (peak - filled) / np.maximum(peak - entry_price, entry_price * 1.0e-12),
        0.0,
    )
    local = days - int(market.start_idx)
    trigger = np.zeros(len(days), dtype=bool)
    policy = str(row["policy"])
    if policy in study_module.FIXED_HORIZONS:
        trigger |= sessions == int(study_module.FIXED_HORIZONS[policy]) - 1
    elif policy in {"volatility_1_2", "volatility_1_3"}:
        history = np.asarray(
            market.exit_close_raw[max(0, signal_abs - 20) : signal_abs + 1, symbol],
            dtype=np.float64,
        )
        history = history[np.isfinite(history) & (history > 0.0)]
        scale = (
            float(np.std(np.diff(np.log(history)), ddof=1))
            if len(history) >= 6
            else math.nan
        )
        scale = float(
            np.clip(
                scale,
                study_module.VOLATILITY_SCALE_FLOOR,
                study_module.VOLATILITY_SCALE_CAP,
            )
        )
        multiple = 2.0 if policy == "volatility_1_2" else 3.0
        current_return = filled / entry_price - 1.0
        if math.isfinite(scale) and scale > 0.0:
            trigger = (current_return <= -scale) | (
                current_return >= multiple * scale
            )
    elif policy == "trailing_3pct":
        trigger = drawdown <= -0.03
    elif policy == "giveback_half":
        trigger = (peak_gain >= 0.03) & (giveback >= 0.50)
    elif policy == "structure_small_reversal":
        trigger = (structure.breakout_active[local, symbol] != 1) | (
            structure.small_event[local, symbol] == -1
        )
    else:
        raise AssertionError(f"exit_baselines_validation_policy:{policy}")
    trigger |= days >= int(liquidation_abs)
    candidates = np.flatnonzero(trigger & (sessions <= int(maximum_sessions)))
    request_position = (
        int(candidates[0])
        if len(candidates)
        else min(int(maximum_sessions) - 1, len(days) - 2)
    )
    request_abs = int(days[request_position])
    later_days = days[request_position + 1 :]
    legal = np.asarray(market.exit_sellable[later_days, symbol], dtype=bool)
    later_prices = np.asarray(
        market.exit_close_raw[later_days, symbol], dtype=np.float64
    )
    exits = np.flatnonzero(legal & np.isfinite(later_prices) & (later_prices > 0.0))
    exit_abs = int(later_days[int(exits[0])]) if len(exits) else -1
    return request_abs, exit_abs


def validate(
    *,
    study_path: str | Path = study_module.DEFAULT_STUDY_PATH,
    output_root: str | Path = study_module.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = study_module.load_study(study_path)
    root = _resolve(output_root)
    manifest = _read_json(root / "manifest.json")
    entries = pd.read_parquet(root / "entry_records.parquet")
    if len(entries) != int(manifest["entries"]):
        raise AssertionError("exit_baselines_validation_entry_count")
    if bool(entries["signal_date"].astype(str).str.startswith("2026").any()):
        raise AssertionError("exit_baselines_validation_forbidden_entry")

    oracle_study = dynamic_oracle.load_study(
        _resolve(study["source"]["dynamic_oracle_study"])
    )
    market, _ = replay.load_market(oracle_study)
    structure = study_module._load_structure(study, market)
    expected = _independent_signal_frame(
        market=market,
        structure=structure,
        liquidation_date=str(study["period"]["terminal_liquidation_start"]),
    )
    actual = entries[
        ["entry_pattern", "signal_date_idx", "symbol_idx"]
    ].reset_index(drop=True)
    integer_columns = ["signal_date_idx", "symbol_idx"]
    actual[integer_columns] = actual[integer_columns].astype(np.int64)
    expected[integer_columns] = expected[integer_columns].astype(np.int64)
    if not np.array_equal(actual.to_numpy(object), expected.to_numpy(object)):
        raise AssertionError("exit_baselines_validation_entry_keys")
    symbols = entries["symbol_idx"].to_numpy(np.int64)
    signal = entries["signal_date_idx"].to_numpy(np.int64)
    entry = entries["entry_date_idx"].to_numpy(np.int64)
    if bool((entry != signal + 1).any()):
        raise AssertionError("exit_baselines_validation_entry_timing")
    prices = np.asarray(market.entry_open_raw[entry, symbols], dtype=np.float64)
    if not np.allclose(
        prices,
        entries["entry_price_raw"].to_numpy(float),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise AssertionError("exit_baselines_validation_entry_price")

    panel_rows = 0
    panel_sample_rows = 0
    entry_lookup = entries.set_index(
        ["entry_pattern", "signal_date_idx", "symbol_idx"], drop=False
    )
    for record in manifest["panel_records"]:
        path = _resolve(record["path"])
        if study_module._sha256(path) != str(record["sha256"]):
            raise AssertionError("exit_baselines_validation_panel_hash")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(record["rows"]):
            raise AssertionError("exit_baselines_validation_panel_rows")
        panel_rows += int(record["rows"])
        row_groups = sorted({0, max(parquet.num_row_groups - 1, 0)})
        for row_group in row_groups:
            sample = parquet.read_row_group(row_group).to_pandas().iloc[[0, -1]]
            for row in sample.itertuples(index=False):
                key = (
                    str(row.entry_pattern),
                    int(row.signal_date_idx),
                    int(row.symbol_idx),
                )
                entry_row = entry_lookup.loc[key]
                expected_close = float(
                    market.exit_close_raw[int(row.date_idx), int(row.symbol_idx)]
                )
                if (
                    math.isfinite(expected_close)
                    and expected_close > 0.0
                    and not math.isclose(
                        float(row.close_raw),
                        expected_close,
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                ):
                    raise AssertionError("exit_baselines_validation_panel_close")
                entry_price = float(entry_row["entry_price_raw"])
                history = np.asarray(
                    market.exit_close_raw[
                        int(row.entry_date_idx) : int(row.date_idx) + 1,
                        int(row.symbol_idx),
                    ],
                    dtype=np.float64,
                )
                valid_history = history[np.isfinite(history) & (history > 0.0)]
                mark = float(valid_history[-1]) if len(valid_history) else entry_price
                expected_return = mark / entry_price - 1.0
                if not math.isclose(
                    float(row.gross_return_close),
                    expected_return,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                ):
                    raise AssertionError("exit_baselines_validation_panel_return")
                panel_sample_rows += 1

    liquidation_abs = int(
        np.flatnonzero(
            market.date_values.astype(str)
            == str(study["period"]["terminal_liquidation_start"])
        )[0]
    )
    result_rows = 0
    sample_rows: list[pd.Series] = []
    for record in manifest["policy_result_records"]:
        path = _resolve(record["path"])
        if study_module._sha256(path) != str(record["sha256"]):
            raise AssertionError("exit_baselines_validation_result_hash")
        frame = pd.read_parquet(path)
        if len(frame) != int(record["rows"]):
            raise AssertionError("exit_baselines_validation_result_rows")
        result_rows += len(frame)
        if set(frame["policy"].unique()) != set(study_module.POLICY_NAMES):
            raise AssertionError("exit_baselines_validation_result_policies")
        if bool(frame.duplicated(["episode_id", "policy"]).any()):
            raise AssertionError("exit_baselines_validation_result_duplicates")
        resolved = frame["exit_date_idx"].ge(0)
        if bool(
            frame.loc[resolved, "exit_date_idx"]
            .le(frame.loc[resolved, "request_date_idx"])
            .any()
        ):
            raise AssertionError("exit_baselines_validation_same_close_fill")
        symbols = frame.loc[resolved, "symbol_idx"].to_numpy(np.int64)
        exits = frame.loc[resolved, "exit_date_idx"].to_numpy(np.int64)
        if bool((~np.asarray(market.exit_sellable[exits, symbols], dtype=bool)).any()):
            raise AssertionError("exit_baselines_validation_unsellable_exit")
        positions = np.linspace(0, len(frame) - 1, min(20, len(frame)), dtype=int)
        sample_rows.extend(frame.iloc[positions].iterrows().__iter__())

    expected_result_rows = len(entries) * len(study_module.POLICY_NAMES)
    if result_rows != expected_result_rows:
        raise AssertionError("exit_baselines_validation_total_result_rows")

    maximum_cost_error = 0.0
    maximum_request_error = 0
    starting_cash = float(study["execution"]["starting_cash_cny_per_entry"])
    normalized_samples: list[pd.Series] = []
    for sample in sample_rows:
        normalized_samples.append(sample[1] if isinstance(sample, tuple) else sample)
    for row in normalized_samples:
        expected_request, expected_exit = _independent_request_and_exit(
            row=row,
            market=market,
            structure=structure,
            liquidation_abs=liquidation_abs,
            maximum_sessions=int(study["episodes"]["maximum_sessions"]),
        )
        maximum_request_error = max(
            maximum_request_error,
            abs(expected_request - int(row["request_date_idx"])),
        )
        if expected_exit != int(row["exit_date_idx"]):
            raise AssertionError("exit_baselines_validation_exit_resolution")
        entry_price = float(
            market.entry_open_raw[int(row["entry_date_idx"]), int(row["symbol_idx"])]
        )
        for cost in replay.COST_SCENARIOS:
            multiplier = (
                1.0
                if cost == "base"
                else float(market.costs.stress_slippage_multiplier)
            )
            if expected_exit < 0:
                expected_return = -1.0
            else:
                shares, buy_cash, _, _ = _buy_order(
                    available_cash=starting_cash,
                    allocated_cash=starting_cash,
                    entry_price=entry_price,
                    contract=market.costs,
                    slippage_multiplier=multiplier,
                )
                if shares <= 0:
                    expected_return = 0.0
                else:
                    proceeds, _, _ = _sell_order(
                        shares=shares,
                        exit_price=float(
                            market.exit_close_raw[
                                expected_exit, int(row["symbol_idx"])
                            ]
                        ),
                        exit_date_idx=expected_exit,
                        date_values=market.date_values,
                        contract=market.costs,
                        slippage_multiplier=multiplier,
                    )
                    expected_return = (
                        starting_cash - buy_cash + proceeds
                    ) / starting_cash - 1.0
            actual_return = float(row[f"net_return_{cost}"])
            maximum_cost_error = max(
                maximum_cost_error, abs(actual_return - expected_return)
            )
            if not math.isclose(
                actual_return, expected_return, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise AssertionError("exit_baselines_validation_cost")
    if maximum_request_error:
        raise AssertionError("exit_baselines_validation_request_trigger")

    summary = pd.read_parquet(root / "summary.parquet")
    if len(summary) != (
        len(study_module.ENTRY_PATTERNS)
        * len(study_module.POLICY_NAMES)
        * len(replay.COST_SCENARIOS)
    ):
        raise AssertionError("exit_baselines_validation_summary_rows")
    gate = _read_json(root / "promotion_gate.json")
    result = {
        "status": "passed",
        "entry_rows": len(entries),
        "panel_rows": int(panel_rows),
        "panel_sample_rows": int(panel_sample_rows),
        "policy_result_rows": int(result_rows),
        "policy_sample_rows": len(normalized_samples),
        "maximum_request_date_error": int(maximum_request_error),
        "maximum_cost_recomputation_error": float(maximum_cost_error),
        "forbidden_2026_rows": 0,
        "promotion_status": str(gate["status"]),
    }
    (root / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate causal exit baselines.")
    parser.add_argument("--study", default=str(study_module.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(study_module.DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    result = validate(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
