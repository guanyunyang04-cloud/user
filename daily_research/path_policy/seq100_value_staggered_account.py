"""Finite-capital staggered-sleeve diagnostic for the PIT value-only D60 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as causal
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_value_staggered_account_v1"
SUMMARY_SCHEMA = "seq100_value_staggered_account_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_value_staggered_account_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_value_staggered_account_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study["source"])
    if source.get("expected_input_fingerprint") != base.EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("input_row_count_contract_mismatch")
    if int(source.get("forbidden_year", -1)) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    account = dict(study["account"])
    breadths = tuple(int(value) for value in account["breadth_sensitivity"])
    if int(account["primary_breadth"]) not in breadths:
        raise ValueError("primary_breadth_missing_from_sensitivity")
    if int(account["sleeve_count"]) < 5:
        raise ValueError("sleeve_count_does_not_cover_retry_bound")
    if bool(account.get("valuation_target_exit_used", True)):
        raise ValueError("valuation_target_exit_must_remain_disabled")
    if bool(account.get("round_lots_and_minimum_commission_modeled", True)):
        raise ValueError("round_lot_contract_mismatch")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    hashed = (
        "causal_value_study",
        "causal_value_summary",
        "candidate_features",
        "selections",
    )
    paths: dict[str, Path] = {}
    for key in hashed:
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    paths["input_manifest"] = base._resolve_path(source["input_manifest"])
    paths["label_manifest"] = base._resolve_path(source["label_manifest"])
    parent = _read_json(paths["causal_value_summary"])
    causal.validate_summary(paths["causal_value_summary"])
    if parent.get("52_week_low_reward_used") is not False:
        raise ValueError("parent_low_price_reward_contract_failed")
    return paths


def _outcome_ledger(
    panel: base.StockPanel,
    *,
    features_path: Path,
    selections_path: Path,
    policy: str,
) -> pd.DataFrame:
    features = pd.read_parquet(
        features_path,
        columns=["row_position", "date_idx", "trade_date", "symbol"],
    )
    selections = pd.read_parquet(selections_path)
    selections = selections.loc[selections["policy"].eq(str(policy))].copy()
    if selections.empty:
        raise ValueError("policy_selection_empty")
    rows = features["row_position"].to_numpy(dtype=np.int64)
    outcome = causal._d60_timeout(panel, rows, retry_days=20)
    outcome_frame = pd.DataFrame(
        {
            "row_position": rows,
            "calendar_evaluable": np.asarray(outcome["within"], dtype=bool),
            "entry_filled": np.asarray(outcome["entry_valid"], dtype=bool),
            "outcome_valid": np.asarray(outcome["valid"], dtype=bool),
            "entry_price": np.asarray(outcome["entry_price"], dtype=float),
            "exit_offset": np.asarray(outcome["exit_offset"], dtype=np.int16),
            "simple_return": np.asarray(outcome["simple_return"], dtype=float),
            "pack_date_idx": np.asarray(outcome["date_idx"], dtype=np.int64),
            "pack_symbol_idx": np.asarray(outcome["symbol_idx"], dtype=np.int64),
        }
    )
    ledger = selections.merge(
        outcome_frame, on="row_position", how="left", validate="many_to_one"
    )
    if bool(ledger["calendar_evaluable"].isna().any()):
        raise ValueError("selection_outcome_alignment_failed")
    ledger["extended_exit"] = False
    ledger["terminal_writeoff"] = False
    trapped = (
        ledger["calendar_evaluable"] & ledger["entry_filled"] & ~ledger["outcome_valid"]
    )
    if bool(trapped.any()):
        _resolve_censored_positions(panel, ledger, trapped.to_numpy(dtype=bool))
    unresolved = (
        ledger["calendar_evaluable"] & ledger["entry_filled"] & ~ledger["outcome_valid"]
    )
    if bool(unresolved.any()):
        raise ValueError(f"unresolved_filled_position:{int(unresolved.sum())}")
    ledger["applied_cost"] = np.where(
        ledger["outcome_valid"],
        np.where(ledger["terminal_writeoff"], 0.003, 0.006),
        0.0,
    )
    if bool(ledger["trade_date"].astype(str).str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_selection")
    return ledger


def _resolve_censored_positions(
    panel: base.StockPanel, ledger: pd.DataFrame, trapped: np.ndarray
) -> None:
    """Resolve selected filled positions beyond D80 without future filtering."""

    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff_positions = np.flatnonzero(dates == base.MAXIMUM_OUTCOME_DATE)
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_outcome_date_missing")
    cutoff = int(cutoff_positions[0])
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    price_observed = base._open_pack_array(
        pack, "masks", "price_observed", dtype=np.bool_
    )
    exit_sellable = base._open_pack_array(
        pack, "masks", "exit_sellable", dtype=np.bool_
    )
    delisted = base._open_pack_array(pack, "masks", "is_delisted", dtype=np.bool_)
    for position in np.flatnonzero(np.asarray(trapped, dtype=bool)):
        row_index = ledger.index[position]
        row = ledger.iloc[position]
        signal_idx = int(row["pack_date_idx"])
        symbol_idx = int(row["pack_symbol_idx"])
        entry_price = float(row["entry_price"])
        resolved = False
        for exit_idx in range(signal_idx + 81, cutoff + 1):
            if bool(delisted[exit_idx, symbol_idx]):
                ledger.loc[row_index, "outcome_valid"] = True
                ledger.loc[row_index, "exit_offset"] = exit_idx - signal_idx
                ledger.loc[row_index, "simple_return"] = -1.0
                ledger.loc[row_index, "extended_exit"] = True
                ledger.loc[row_index, "terminal_writeoff"] = True
                resolved = True
                break
            close = float(raw[exit_idx, symbol_idx, 3])
            if (
                bool(exit_sellable[exit_idx, symbol_idx])
                and bool(price_observed[exit_idx, symbol_idx])
                and np.isfinite(close)
                and close > 0.0
            ):
                ledger.loc[row_index, "outcome_valid"] = True
                ledger.loc[row_index, "exit_offset"] = exit_idx - signal_idx
                ledger.loc[row_index, "simple_return"] = close / entry_price - 1.0
                ledger.loc[row_index, "extended_exit"] = True
                resolved = True
                break
        if not resolved:
            ledger.loc[row_index, "outcome_valid"] = True
            ledger.loc[row_index, "exit_offset"] = cutoff - signal_idx
            ledger.loc[row_index, "simple_return"] = -1.0
            ledger.loc[row_index, "extended_exit"] = True
            ledger.loc[row_index, "terminal_writeoff"] = True


def _cohort_table(
    ledger: pd.DataFrame,
    *,
    breadth: int,
    sleeve_count: int,
    cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chosen = ledger.loc[ledger["selection_rank"] <= int(breadth)].copy()
    signal_dates = (
        ledger[["date_idx", "trade_date"]]
        .drop_duplicates("date_idx")
        .sort_values("date_idx")
        .reset_index(drop=True)
    )
    signal_dates["month_ordinal"] = np.arange(len(signal_dates), dtype=np.int64)
    signal_dates["sleeve"] = signal_dates["month_ordinal"] % int(sleeve_count)
    chosen = chosen.merge(
        signal_dates[["date_idx", "month_ordinal", "sleeve"]],
        on="date_idx",
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    for (date_idx, trade_date, month_ordinal, sleeve), group in chosen.groupby(
        ["date_idx", "trade_date", "month_ordinal", "sleeve"], sort=True
    ):
        evaluable = bool(group["calendar_evaluable"].all())
        valid = group["outcome_valid"].to_numpy(dtype=bool)
        returns = group["simple_return"].to_numpy(dtype=float)
        costs = group["applied_cost"].to_numpy(dtype=float)
        if bool((costs > float(cost) + 1e-12).any()):
            raise ValueError("applied_cost_exceeds_contract")
        records.append(
            {
                "breadth": int(breadth),
                "date_idx": int(date_idx),
                "trade_date": str(trade_date),
                "month_ordinal": int(month_ordinal),
                "sleeve": int(sleeve),
                "calendar_evaluable": evaluable,
                "selected_count": len(group),
                "filled_count": int(valid.sum()) if evaluable else 0,
                "stress_net_return": (
                    (float(returns[valid].sum()) - float(costs[valid].sum()))
                    / int(breadth)
                    if evaluable
                    else np.nan
                ),
                "maximum_exit_offset": (
                    int(group.loc[valid, "exit_offset"].max())
                    if evaluable and valid.any()
                    else -1
                ),
            }
        )
    return chosen, pd.DataFrame(records)


def _simulate_sleeve(
    *,
    chosen: pd.DataFrame,
    sleeve: int,
    breadth: int,
    initial_cash: float,
    cost: float,
    raw: np.memmap,
    price_observed: np.memmap,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    wealth = np.full(cutoff + 1, float(initial_cash), dtype=np.float64)
    invested = np.zeros(cutoff + 1, dtype=np.float64)
    position_count = np.zeros(cutoff + 1, dtype=np.int16)
    current_cash = float(initial_cash)
    cursor = 0
    previous_exit = -1
    cycle_records: list[dict[str, Any]] = []
    local = chosen.loc[
        chosen["sleeve"].eq(int(sleeve)) & chosen["calendar_evaluable"]
    ].copy()
    for (date_idx, trade_date), group in local.groupby(
        ["date_idx", "trade_date"], sort=True
    ):
        signal_idx = int(date_idx)
        entry_idx = signal_idx + 1
        filled = group.loc[group["outcome_valid"]].copy()
        if len(group) != int(breadth):
            raise ValueError(f"breadth_selection_count_mismatch:{breadth}:{trade_date}")
        if filled.empty:
            continue
        exit_idx = signal_idx + filled["exit_offset"].to_numpy(dtype=np.int64)
        maximum_exit = int(exit_idx.max())
        if entry_idx <= previous_exit:
            cycle_records.append(
                {
                    "sleeve": int(sleeve),
                    "breadth": int(breadth),
                    "trade_date": str(trade_date),
                    "signal_date_idx": signal_idx,
                    "entry_date_idx": entry_idx,
                    "last_exit_date_idx": previous_exit,
                    "filled_count": 0,
                    "starting_cash": current_cash,
                    "ending_cash": current_cash,
                    "stress_net_return": 0.0,
                    "skipped_due_to_unresolved_prior": True,
                }
            )
            continue
        if maximum_exit > cutoff:
            raise ValueError("exit_after_fixed_cutoff")
        wealth[cursor:entry_idx] = current_cash
        slot = current_cash / int(breadth)
        half_cost = 0.5 * float(cost) * slot
        filled_count = len(filled)
        cash = current_cash - slot * filled_count - half_cost * filled_count
        symbols = filled["pack_symbol_idx"].to_numpy(dtype=np.int64)
        entry_price = filled["entry_price"].to_numpy(dtype=np.float64)
        exit_ratio = 1.0 + filled["simple_return"].to_numpy(dtype=np.float64)
        terminal_writeoff = filled["terminal_writeoff"].to_numpy(dtype=bool)
        last_ratio = np.ones(filled_count, dtype=np.float64)
        for date in range(entry_idx, maximum_exit + 1):
            observed = np.asarray(price_observed[date, symbols], dtype=bool)
            close = np.asarray(raw[date, symbols, 3], dtype=np.float64)
            update = observed & np.isfinite(close) & (close > 0.0)
            last_ratio[update] = close[update] / entry_price[update]
            exiting = exit_idx == int(date)
            if exiting.any():
                last_ratio[exiting] = exit_ratio[exiting]
                exit_cost = np.where(terminal_writeoff[exiting], 0.0, half_cost)
                cash += float(np.sum(slot * last_ratio[exiting] - exit_cost))
            active = exit_idx > int(date)
            active_value = float(np.sum(slot * last_ratio[active]))
            wealth[date] = cash + active_value
            invested[date] = active_value
            position_count[date] = int(active.sum())
        expected_return = float(
            (filled["simple_return"] - filled["applied_cost"]).sum()
        ) / int(breadth)
        expected_cash = current_cash * (1.0 + expected_return)
        if not np.isclose(cash, expected_cash, rtol=1e-9, atol=1e-6):
            raise ValueError(
                f"cycle_cash_reconciliation_failed:{trade_date}:{cash}:{expected_cash}"
            )
        cycle_records.append(
            {
                "sleeve": int(sleeve),
                "breadth": int(breadth),
                "trade_date": str(trade_date),
                "signal_date_idx": signal_idx,
                "entry_date_idx": entry_idx,
                "last_exit_date_idx": maximum_exit,
                "filled_count": filled_count,
                "starting_cash": current_cash,
                "ending_cash": cash,
                "stress_net_return": expected_return,
                "skipped_due_to_unresolved_prior": False,
            }
        )
        current_cash = cash
        cursor = maximum_exit + 1
        previous_exit = maximum_exit
    wealth[cursor:] = current_cash
    return wealth, invested, position_count, cycle_records


def _account_metrics(
    *,
    dates: np.ndarray,
    wealth: np.ndarray,
    invested: np.ndarray,
    position_count: np.ndarray,
    start_idx: int,
    initial_cash: float,
) -> dict[str, Any]:
    date_values = pd.to_datetime(np.asarray(dates[start_idx:], dtype=str))
    values = np.asarray(wealth[start_idx:], dtype=np.float64)
    exposure = np.divide(
        np.asarray(invested[start_idx:], dtype=np.float64),
        values,
        out=np.zeros(len(values), dtype=np.float64),
        where=values > 0.0,
    )
    peaks = np.maximum.accumulate(values)
    drawdown = values / peaks - 1.0
    elapsed_years = max((date_values[-1] - date_values[0]).days / 365.2425, 1e-9)
    annual_end = pd.DataFrame({"date": date_values, "wealth": values})
    annual_end["year"] = annual_end["date"].dt.year
    annual = annual_end.groupby("year", sort=True).tail(1).copy()
    previous = np.r_[float(initial_cash), annual["wealth"].to_numpy()[:-1]]
    annual["return"] = annual["wealth"].to_numpy() / previous - 1.0
    maximum_drawdown_position = int(np.argmin(drawdown))
    return {
        "start_date": str(date_values[0].date()),
        "end_date": str(date_values[-1].date()),
        "initial_cash": float(initial_cash),
        "ending_wealth": float(values[-1]),
        "terminal_multiple": float(values[-1] / initial_cash),
        "cagr": float((values[-1] / initial_cash) ** (1.0 / elapsed_years) - 1.0),
        "maximum_drawdown": float(drawdown[maximum_drawdown_position]),
        "maximum_drawdown_date": str(date_values[maximum_drawdown_position].date()),
        "mean_exposure": float(exposure.mean()),
        "maximum_exposure": float(exposure.max()),
        "mean_open_positions": float(np.asarray(position_count[start_idx:]).mean()),
        "maximum_open_positions": int(np.asarray(position_count[start_idx:]).max()),
        "positive_year_count": int((annual["return"] > 0.0).sum()),
        "year_count": len(annual),
        "annual": annual[["year", "wealth", "return"]].to_dict(orient="records"),
    }


def _simulate_account(
    panel: base.StockPanel,
    chosen: pd.DataFrame,
    *,
    breadth: int,
    sleeve_count: int,
    initial_cash: float,
    cost: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    pack = panel.pack_manifest
    dates = np.asarray(pack["date_values"], dtype=str)
    cutoff_positions = np.flatnonzero(dates == base.MAXIMUM_OUTCOME_DATE)
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_outcome_date_missing")
    cutoff = int(cutoff_positions[0])
    raw = base._open_pack_array(pack, "feature_channels", "daily_raw", dtype=np.float32)
    price_observed = base._open_pack_array(
        pack, "masks", "price_observed", dtype=np.bool_
    )
    sleeve_initial = float(initial_cash) / int(sleeve_count)
    wealth_parts: list[np.ndarray] = []
    invested_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    cycle_records: list[dict[str, Any]] = []
    sleeve_metrics: list[dict[str, Any]] = []
    evaluable = chosen.loc[chosen["calendar_evaluable"]]
    start_idx = int(evaluable["date_idx"].min()) + 1
    for sleeve in range(int(sleeve_count)):
        wealth, invested, count, cycles = _simulate_sleeve(
            chosen=chosen,
            sleeve=sleeve,
            breadth=breadth,
            initial_cash=sleeve_initial,
            cost=cost,
            raw=raw,
            price_observed=price_observed,
            cutoff=cutoff,
        )
        wealth_parts.append(wealth)
        invested_parts.append(invested)
        count_parts.append(count)
        cycle_records.extend(cycles)
        metric = _account_metrics(
            dates=dates[: cutoff + 1],
            wealth=wealth,
            invested=invested,
            position_count=count,
            start_idx=start_idx,
            initial_cash=sleeve_initial,
        )
        metric["sleeve"] = int(sleeve)
        metric["cycle_count"] = len(cycles)
        metric["skipped_cycle_count"] = int(
            sum(bool(row["skipped_due_to_unresolved_prior"]) for row in cycles)
        )
        sleeve_metrics.append(metric)
    total_wealth = np.sum(np.vstack(wealth_parts), axis=0)
    total_invested = np.sum(np.vstack(invested_parts), axis=0)
    total_count = np.sum(np.vstack(count_parts), axis=0)
    metrics = _account_metrics(
        dates=dates[: cutoff + 1],
        wealth=total_wealth,
        invested=total_invested,
        position_count=total_count,
        start_idx=start_idx,
        initial_cash=initial_cash,
    )
    metrics["breadth"] = int(breadth)
    metrics["sleeve_count"] = int(sleeve_count)
    metrics["sleeves"] = sleeve_metrics
    metrics["cycle_count"] = len(cycle_records)
    metrics["skipped_cycle_count"] = int(
        sum(bool(row["skipped_due_to_unresolved_prior"]) for row in cycle_records)
    )
    curve = pd.DataFrame(
        {
            "breadth": int(breadth),
            "trade_date": dates[start_idx : cutoff + 1],
            "account_value": total_wealth[start_idx : cutoff + 1],
            "invested_value": total_invested[start_idx : cutoff + 1],
            "open_position_count": total_count[start_idx : cutoff + 1],
        }
    )
    curve["exposure"] = curve["invested_value"] / curve["account_value"]
    curve["drawdown"] = curve["account_value"] / curve["account_value"].cummax() - 1.0
    return curve, metrics, pd.DataFrame(cycle_records)


def _decision(
    metrics: Sequence[Mapping[str, Any]], study: Mapping[str, Any]
) -> dict[str, Any]:
    primary_breadth = int(study["account"]["primary_breadth"])
    primary = next(row for row in metrics if int(row["breadth"]) == primary_breadth)
    sleeve_multiples = [float(row["terminal_multiple"]) for row in primary["sleeves"]]
    checks = {
        "terminal_wealth_above_initial": float(primary["terminal_multiple"]) > 1.0,
        "positive_cagr": float(primary["cagr"]) > 0.0,
        "every_sleeve_profitable": min(sleeve_multiples) > 1.0,
        "majority_calendar_years_positive": int(primary["positive_year_count"])
        >= math.ceil(int(primary["year_count"]) / 2),
        "maximum_drawdown_below_40_percent": float(primary["maximum_drawdown"]) > -0.40,
    }
    return {
        "primary_breadth": primary_breadth,
        "retrospective_account_diagnostic_passed": bool(all(checks.values())),
        "checks": checks,
        "forward_confirmation_required": True,
        "production_strategy_claim_allowed": False,
        "round_lots_and_minimum_commission_still_unmodeled": True,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources = _source_contract(study)
    root = base._resolve_path(output_root)
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "source_hashes": {
                key: base._sha256_file(path)
                for key, path in sources.items()
                if path.is_file()
            },
        }
    )
    summary_path = root / "summary.json"
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in dict(current.get("files", {})).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    if bool((panel.years == base.FORBIDDEN_YEAR).any()):
        raise ValueError("forbidden_2026_panel_row")
    account = dict(study["account"])
    ledger = _outcome_ledger(
        panel,
        features_path=sources["candidate_features"],
        selections_path=sources["selections"],
        policy=str(account["policy"]),
    )
    curve_frames: list[pd.DataFrame] = []
    cohort_frames: list[pd.DataFrame] = []
    cycle_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for breadth in (int(value) for value in account["breadth_sensitivity"]):
        chosen, cohorts = _cohort_table(
            ledger,
            breadth=breadth,
            sleeve_count=int(account["sleeve_count"]),
            cost=float(account["round_trip_cost_proxy"]),
        )
        curve, metric, cycles = _simulate_account(
            panel,
            chosen,
            breadth=breadth,
            sleeve_count=int(account["sleeve_count"]),
            initial_cash=float(account["initial_cash"]),
            cost=float(account["round_trip_cost_proxy"]),
        )
        curve_frames.append(curve)
        cohort_frames.append(cohorts)
        cycle_frames.append(cycles)
        metrics.append(metric)
    curves = pd.concat(curve_frames, ignore_index=True)
    cohorts = pd.concat(cohort_frames, ignore_index=True)
    cycles = pd.concat(cycle_frames, ignore_index=True)
    ledger_path = root / "selected_outcomes.parquet"
    curves_path = root / "daily_account_curves.parquet"
    cohorts_path = root / "cohort_returns.parquet"
    cycles_path = root / "sleeve_cycles.parquet"
    _write_parquet(ledger_path, ledger)
    _write_parquet(curves_path, curves)
    _write_parquet(cohorts_path, cohorts)
    _write_parquet(cycles_path, cycles)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "account_metrics": metrics,
        "decision": _decision(metrics, study),
        "posthoc_candidate": True,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "valuation_target_exit_used": False,
        "round_lots_and_minimum_commission_modeled": False,
        "execution_resolution": {
            "selected_row_count": len(ledger),
            "entry_filled_count": int(ledger["entry_filled"].sum()),
            "bounded_exit_count": int(
                (ledger["outcome_valid"] & ~ledger["extended_exit"]).sum()
            ),
            "extended_legal_exit_count": int(
                (ledger["extended_exit"] & ~ledger["terminal_writeoff"]).sum()
            ),
            "terminal_writeoff_count": int(ledger["terminal_writeoff"].sum()),
        },
        "production_strategy_claim_allowed": False,
        "files": {
            "selected_outcomes": base._file_record(ledger_path),
            "daily_account_curves": base._file_record(curves_path),
            "cohort_returns": base._file_record(cohorts_path),
            "sleeve_cycles": base._file_record(cycles_path),
        },
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "posthoc": summary.get("posthoc_candidate") is True,
        "no_low_price_reward": summary.get("52_week_low_reward_used") is False,
        "no_valuation_target": summary.get("valuation_target_exit_used") is False,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("production_strategy_claim_allowed") is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in dict(summary.get("files", {})).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
