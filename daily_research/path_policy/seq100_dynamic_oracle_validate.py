"""Independent validation for the full-universe dynamic oracle study."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_market_replay as replay

VALIDATION_SCHEMA = "seq100_dynamic_oracle_validation/1"


def independent_forward_ceiling(
    market: replay.ReplayMarket, cost_scenario: str
) -> np.ndarray:
    """Solve the implicit weighted-interval problem in chronological order."""
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )
    cash_log = np.full(market.day_count, -np.inf, dtype=np.float64)
    cash_log[0] = 0.0
    best_entry_score = np.full(market.symbol_count, -np.inf, dtype=np.float64)
    for local in range(1, market.day_count):
        absolute_idx = int(market.start_idx + local)
        cash_log[local] = cash_log[local - 1]
        signal_local = local - 2
        if signal_local >= 0:
            signal_idx = int(market.start_idx + signal_local)
            q = np.flatnonzero(market.quality_mask[signal_local])
            opens = np.asarray(
                market.entry_open_raw[signal_idx + 1, q], dtype=np.float64
            )
            valid = (
                np.asarray(market.entry_filled[signal_idx, q], dtype=bool)
                & np.isfinite(opens)
                & (opens > 0.0)
                & math.isfinite(float(cash_log[signal_local]))
            )
            scores = np.full(len(q), -np.inf, dtype=np.float64)
            scores[valid] = (
                float(cash_log[signal_local])
                - np.log(opens[valid] * buy_multiplier)
            )
            best_entry_score[q] = np.maximum(best_entry_score[q], scores)

        closes = np.asarray(market.exit_close_raw[absolute_idx], dtype=np.float64)
        sellable = (
            np.asarray(market.exit_sellable[absolute_idx], dtype=bool)
            & np.isfinite(closes)
            & (closes > 0.0)
            & np.isfinite(best_entry_score)
        )
        if bool(sellable.any()):
            candidates = best_entry_score[sellable] + np.log(
                closes[sellable] * sell_multiplier[local]
            )
            cash_log[local] = max(cash_log[local], float(candidates.max()))
    return cash_log


def _validate_label_files(
    *,
    market: replay.ReplayMarket,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = 0
    duplicate_keys = 0
    forbidden_rows = 0
    nonfinite_rows = 0
    nonmember_rows = 0
    buy_identity_violations = 0
    hold_identity_violations = 0
    selected_by_date: defaultdict[int, int] = defaultdict(int)
    for raw in records:
        record = dict(raw)
        path = Path(str(record["path"]))
        if replay.sha256(path) != str(record["sha256"]):
            raise ValueError("dynamic_oracle_validation_label_hash_mismatch")
        frame = pd.read_parquet(path)
        rows += len(frame)
        duplicate_keys += int(frame.duplicated(["date_idx", "symbol_idx"]).sum())
        forbidden_rows += int(frame["trade_date"].astype(str).str[:4].astype(int).ge(2026).sum())
        dates = frame["date_idx"].to_numpy(np.int64)
        symbols = frame["symbol_idx"].to_numpy(np.int64)
        local = dates - int(market.start_idx)
        valid_domain = (
            (local >= 0)
            & (local < market.day_count)
            & (symbols >= 0)
            & (symbols < market.symbol_count)
        )
        if not bool(valid_domain.all()):
            nonmember_rows += int((~valid_domain).sum())
        else:
            nonmember_rows += int((~market.quality_mask[local, symbols]).sum())
        float_columns = [
            "buy_terminal_log_value",
            "cash_stay_terminal_log_value",
            "buy_advantage_vs_cash",
            "holding_continue_terminal_log_value",
            "holding_exit_terminal_log_value",
            "exit_advantage_vs_continue",
        ]
        for column in float_columns:
            values = frame[column].to_numpy(np.float64)
            nonfinite_rows += int(np.isinf(values).sum())
        buy_valid = frame["buy_action_valid"].astype(bool).to_numpy()
        expected_buy = (
            frame["buy_terminal_log_value"].to_numpy(np.float64)
            - frame["cash_stay_terminal_log_value"].to_numpy(np.float64)
        )
        actual_buy = frame["buy_advantage_vs_cash"].to_numpy(np.float64)
        # Terminal values are intentionally stored as float32; the separately
        # computed action advantage retains useful small differences but the
        # reconstructed subtraction can lose a few ulps at large log wealth.
        buy_identity_violations += int(
            (np.abs(expected_buy[buy_valid] - actual_buy[buy_valid]) > 2.0e-4).sum()
        )
        comparable = frame["holding_actions_comparable"].astype(bool).to_numpy()
        expected_hold = (
            frame["holding_exit_terminal_log_value"].to_numpy(np.float64)
            - frame["holding_continue_terminal_log_value"].to_numpy(np.float64)
        )
        actual_hold = frame["exit_advantage_vs_continue"].to_numpy(np.float64)
        hold_identity_violations += int(
            (
                np.abs(expected_hold[comparable] - actual_hold[comparable])
                > 2.0e-4
            ).sum()
        )
        selected = frame["oracle_cash_buy_selected"].astype(bool)
        for date_idx, count in (
            frame.loc[selected]
            .groupby("date_idx", sort=False)
            .size()
            .to_dict()
            .items()
        ):
            selected_by_date[int(date_idx)] += int(count)
    selected_multiplicity = int(sum(count > 1 for count in selected_by_date.values()))
    return {
        "rows": rows,
        "duplicate_keys": duplicate_keys,
        "forbidden_rows": forbidden_rows,
        "nonfinite_infinity_rows": nonfinite_rows,
        "nonmember_rows": nonmember_rows,
        "buy_identity_violations": buy_identity_violations,
        "hold_identity_violations": hold_identity_violations,
        "selected_date_multiplicity_violations": selected_multiplicity,
    }


def _validate_selected_trades(
    *,
    market: replay.ReplayMarket,
    cost_scenario: str,
    frame: pd.DataFrame,
    expected_log_wealth: float,
) -> dict[str, Any]:
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )
    violations = defaultdict(int)
    previous_exit = int(market.start_idx)
    recomputed_logs: list[float] = []
    for row in frame.sort_values("trade_id", kind="mergesort").itertuples(index=False):
        signal_idx = int(row.signal_date_idx)
        entry_idx = int(row.entry_date_idx)
        exit_idx = int(row.exit_date_idx)
        symbol_idx = int(row.symbol_idx)
        if signal_idx < previous_exit:
            violations["overlap"] += 1
        if entry_idx != signal_idx + 1:
            violations["entry_timing"] += 1
        if exit_idx < signal_idx + 2:
            violations["t1"] += 1
        local = signal_idx - int(market.start_idx)
        if local < 0 or local >= market.day_count or not bool(
            market.quality_mask[local, symbol_idx]
        ):
            violations["quality_membership"] += 1
        if not bool(market.entry_filled[signal_idx, symbol_idx]):
            violations["entry_fill"] += 1
        if not bool(market.exit_sellable[exit_idx, symbol_idx]):
            violations["exit_sellability"] += 1
        entry = float(market.entry_open_raw[entry_idx, symbol_idx])
        exit_price = float(market.exit_close_raw[exit_idx, symbol_idx])
        factor = (
            exit_price
            * sell_multiplier[exit_idx - market.start_idx]
            / (entry * buy_multiplier)
        )
        recomputed = math.log(factor)
        recomputed_logs.append(recomputed)
        if abs(recomputed - float(row.proportional_log_return)) > 1.0e-10:
            violations["return_identity"] += 1
        previous_exit = exit_idx
    total = float(sum(recomputed_logs))
    if abs(total - float(expected_log_wealth)) > 2.0e-8:
        violations["terminal_identity"] += 1
    return {
        "rows": len(frame),
        "recomputed_terminal_log_wealth": total,
        "violations": dict(violations),
    }


def validate(
    *,
    study_path: str | Path = oracle.DEFAULT_STUDY_PATH,
    output_root: str | Path = oracle.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = oracle.load_study(study_path)
    root = oracle._resolve(output_root)
    manifest = oracle._read_json(root / "analysis_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_validation_source_incomplete")
    market, _ = replay.load_market(study)
    oracle_audits: list[dict[str, Any]] = []
    blocking = defaultdict(int)
    selected_frames: dict[str, pd.DataFrame] = {}
    for record_raw in manifest["relaxed_oracles"]:
        record = dict(record_raw)
        cost_scenario = str(record["cost_scenario"])
        forward = independent_forward_ceiling(market, cost_scenario)
        expected = float(record["terminal_log_wealth"])
        forward_difference = abs(float(forward[-1]) - expected)
        if forward_difference > 2.0e-8:
            blocking["forward_ceiling_mismatch"] += 1
        labels = _validate_label_files(
            market=market, records=record["action_labels"]
        )
        for key, value in labels.items():
            if key != "rows" and int(value):
                blocking[f"labels_{key}"] += int(value)
        trades_path = Path(str(dict(record["selected_trades"])["path"]))
        if replay.sha256(trades_path) != str(dict(record["selected_trades"])["sha256"]):
            raise ValueError("dynamic_oracle_validation_trade_hash_mismatch")
        trades = pd.read_parquet(trades_path)
        selected_frames[cost_scenario] = trades
        trade_audit = _validate_selected_trades(
            market=market,
            cost_scenario=cost_scenario,
            frame=trades,
            expected_log_wealth=expected,
        )
        for key, value in dict(trade_audit["violations"]).items():
            blocking[f"trades_{key}"] += int(value)
        oracle_audits.append(
            {
                "cost_scenario": cost_scenario,
                "independent_terminal_log_wealth": float(forward[-1]),
                "forward_difference": forward_difference,
                "labels": labels,
                "trades": trade_audit,
            }
        )

    finite_summary = pd.read_parquet(
        Path(str(dict(manifest["outputs"]["finite_account_summary"])["path"]))
    )
    finite_equity = pd.read_parquet(
        Path(str(dict(manifest["outputs"]["finite_equity"])["path"]))
    )
    finite_trades = pd.read_parquet(
        Path(str(dict(manifest["outputs"]["finite_trades"])["path"]))
    )
    maximum_cash_identity_error = 0.0
    maximum_cash_identity_relative_error = 0.0
    maximum_terminal_equity_error = 0.0
    maximum_terminal_equity_relative_error = 0.0
    negative_cash_rows = int((finite_equity["cash"].astype(float) < -1.0e-8).sum())
    position_count_violations = int(
        (finite_equity["position_count"].astype(int) > 1).sum()
    )
    for summary in finite_summary.itertuples(index=False):
        mask = (
            finite_trades["cost_scenario"].eq(str(summary.cost_scenario))
            & finite_trades["starting_cash_cny"].eq(float(summary.starting_cash_cny))
            & finite_trades["capacity_mode"].eq(str(summary.capacity_mode))
        )
        trades = finite_trades[mask]
        reconstructed_cash = float(summary.starting_cash_cny)
        if not trades.empty:
            reconstructed_cash += float(
                (
                    trades["sell_proceeds_cny"].astype(float)
                    - trades["buy_cash_cny"].astype(float)
                ).sum()
            )
        cash_error = abs(reconstructed_cash - float(summary.ending_equity_cny))
        maximum_cash_identity_error = max(maximum_cash_identity_error, cash_error)
        maximum_cash_identity_relative_error = max(
            maximum_cash_identity_relative_error,
            cash_error / max(abs(float(summary.ending_equity_cny)), 1.0),
        )
        equity_mask = (
            finite_equity["cost_scenario"].eq(str(summary.cost_scenario))
            & finite_equity["starting_cash_cny"].eq(float(summary.starting_cash_cny))
            & finite_equity["capacity_mode"].eq(str(summary.capacity_mode))
        )
        ending = float(
            finite_equity.loc[equity_mask]
            .sort_values("date_idx", kind="mergesort")
            .iloc[-1]["equity"]
        )
        terminal_error = abs(ending - float(summary.ending_equity_cny))
        maximum_terminal_equity_error = max(
            maximum_terminal_equity_error, terminal_error
        )
        maximum_terminal_equity_relative_error = max(
            maximum_terminal_equity_relative_error,
            terminal_error / max(abs(float(summary.ending_equity_cny)), 1.0),
        )
    if (
        maximum_cash_identity_error > 1.0e-5
        and maximum_cash_identity_relative_error > 1.0e-12
    ):
        blocking["finite_cash_identity"] += 1
    if (
        maximum_terminal_equity_error > 1.0e-5
        and maximum_terminal_equity_relative_error > 1.0e-12
    ):
        blocking["finite_terminal_identity"] += 1
    if negative_cash_rows:
        blocking["finite_negative_cash"] += negative_cash_rows
    if position_count_violations:
        blocking["finite_position_count"] += position_count_violations
    capped = finite_trades[finite_trades["capacity_mode"].eq("signal_amount_0p5pct")]
    participation_violations = int(
        (
            capped["signal_amount_participation"].astype(float)
            > float(dict(study["finite_replay"])["maximum_signal_day_amount_fraction"])
            + 1.0e-10
        ).sum()
    )
    if participation_violations:
        blocking["finite_participation"] += participation_violations

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": oracle.STUDY_ID,
        "oracle_audits": oracle_audits,
        "finite_account_audit": {
            "tasks": len(finite_summary),
            "equity_rows": len(finite_equity),
            "trade_rows": len(finite_trades),
            "maximum_cash_identity_error": maximum_cash_identity_error,
            "maximum_cash_identity_relative_error": (
                maximum_cash_identity_relative_error
            ),
            "maximum_terminal_equity_error": maximum_terminal_equity_error,
            "maximum_terminal_equity_relative_error": (
                maximum_terminal_equity_relative_error
            ),
            "negative_cash_rows": negative_cash_rows,
            "position_count_violations": position_count_violations,
            "participation_violations": participation_violations,
        },
        "blocking": dict(blocking),
    }
    oracle._write_json(root / "validation.json", result)
    if blocking:
        raise ValueError(f"dynamic_oracle_validation_failed:{dict(blocking)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the dynamic oracle study.")
    parser.add_argument("--study", default=str(oracle.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(oracle.DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args(argv)
    result = validate(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
