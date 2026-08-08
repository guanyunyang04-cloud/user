"""Exact finite-account falsification of the natural-phase TTM-FCF rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy import seq100_ttm_value_account as ttm
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_ttm_fcf_rotation_account_v1"
SUMMARY_SCHEMA = "seq100_ttm_fcf_rotation_account_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_ttm_fcf_rotation_account_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_ttm_fcf_rotation_account_v1"
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
    selection = dict(study["selection"])
    expected_selection = {
        "source_variant": "ttm_fcf_yield_only",
        "source_rank_count": 48,
        "monthly_order_count": 12,
        "rank_partition_modulus": 4,
        "natural_phase": 0,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"selection_contract_mismatch:{key}")
    account = dict(study["account"])
    expected_account = {
        "starting_cash_cny": 1_000_000.0,
        "position_slots": 48,
        "cost_scenario": "double_slippage",
        "pyramiding": False,
        "lot_size": 100,
        "round_lots_and_minimum_commission_modeled": True,
    }
    for key, expected in expected_account.items():
        if account.get(key) != expected:
            raise ValueError(f"account_contract_mismatch:{key}")
    epistemic = dict(study["epistemic_contract"])
    if not epistemic.get(
        "natural_phase_zero_is_frozen_without_selecting_the_best_observed_phase"
    ):
        raise ValueError("natural_phase_freeze_missing")
    if bool(study["decision_boundary"].get("account_optimization_performed", True)):
        raise ValueError("account_optimization_forbidden")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    paths: dict[str, Path] = {}
    for key in ("ttm_value_summary", "variant_selections", "pack_manifest"):
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    paths["input_manifest"] = base._resolve_path(source["input_manifest"])
    paths["label_manifest"] = base._resolve_path(source["label_manifest"])
    ttm.validate_summary(paths["ttm_value_summary"])
    return paths


def _rotation_mask(
    selection_rank: pd.Series,
    month_ordinal: pd.Series,
    *,
    modulus: int,
    phase: int,
) -> pd.Series:
    """Return the frozen rank-residue partition for each signal month."""
    return ((selection_rank.astype(int) - 1) % modulus) == (
        (month_ordinal.astype(int) + phase) % modulus
    )


def _rotation_selection(
    panel: base.StockPanel,
    pack: CandidateCompleteAuditPack,
    *,
    study: Mapping[str, Any],
    selection_path: Path,
) -> tuple[finite.ForecastBook, pd.DataFrame, dict[str, Any]]:
    contract = dict(study["selection"])
    frame = pd.read_parquet(selection_path)
    frame = frame.loc[frame["variant"].eq(str(contract["source_variant"]))].copy()
    dates = (
        frame[["date_idx", "trade_date"]]
        .drop_duplicates("date_idx")
        .sort_values("date_idx")
        .reset_index(drop=True)
    )
    dates["month_ordinal"] = np.arange(len(dates), dtype=np.int64)
    frame = frame.merge(dates, on=["date_idx", "trade_date"], validate="many_to_one")
    modulus = int(contract["rank_partition_modulus"])
    phase = int(contract["natural_phase"])
    keep = _rotation_mask(
        frame["selection_rank"],
        frame["month_ordinal"],
        modulus=modulus,
        phase=phase,
    )
    frame = frame.loc[keep].copy()
    identity = panel.row_index[["candidate_id", "date_idx", "symbol_idx", "symbol"]]
    frame = frame.merge(
        identity,
        on=["candidate_id", "date_idx", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if bool(frame["symbol_idx"].isna().any()):
        raise ValueError("rotation_symbol_alignment_failed")
    cutoff_positions = np.flatnonzero(
        np.asarray(pack.date_values, dtype=str)
        == str(study["source"]["maximum_account_mark_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_account_mark_date_missing")
    cutoff = int(cutoff_positions[0])
    boundary_evaluable = (
        frame["date_idx"].astype(int) + int(pack.execution_days) <= cutoff
    )
    excluded_boundary_count = int((~boundary_evaluable).sum())
    frame = frame.loc[boundary_evaluable].copy()
    counts = frame.groupby("date_idx").size()
    expected_count = int(contract["monthly_order_count"])
    if not bool(counts.eq(expected_count).all()):
        raise ValueError("monthly_rotation_count_mismatch")
    book = finite.ForecastBook(
        "ttm_fcf_natural_rotation",
        top_k=expected_count,
        candidate_scan_k=expected_count,
    )
    for date_idx, group in frame.groupby("date_idx", sort=True):
        book.add_day(
            date_idx=int(date_idx),
            symbol_idx=group["symbol_idx"].to_numpy(dtype=np.int64),
            score=group["variant_score"].to_numpy(dtype=float),
            planned_day=np.full(len(group), 60, dtype=np.int16),
        )
    industry_max = (
        frame.groupby(["date_idx", "industry_code"]).size().groupby("date_idx").max()
    )
    audit = {
        "source_variant": str(contract["source_variant"]),
        "natural_phase": phase,
        "signal_date_count": len(book.days),
        "selected_row_count": len(frame),
        "selected_count_per_signal": expected_count,
        "first_signal_date_idx": int(min(book.days)),
        "last_signal_date_idx": int(max(book.days)),
        "first_signal_date": str(pack.date_values[min(book.days)]),
        "last_signal_date": str(pack.date_values[max(book.days)]),
        "boundary_excluded_row_count": excluded_boundary_count,
        "maximum_same_industry_names_per_signal": int(industry_max.max()),
        "future_outcomes_used": False,
    }
    return book, frame, audit


def _market(
    pack: CandidateCompleteAuditPack,
    *,
    adjust_factor: np.ndarray | None,
) -> finite.BacktestMarket:
    return finite.BacktestMarket(
        date_values=np.asarray(pack.date_values, dtype=object),
        symbol_values=np.asarray(pack.symbol_values, dtype=object),
        entry_open_raw=pack.entry_open_raw,
        exit_close_raw=pack.exit_close_raw,
        exit_sellable=pack.exit_sellable,
        entry_filled=pack.entry_filled,
        costs=pack.costs,
        terminal_recovery_fraction=float(pack.terminal_recovery_fraction),
        forward_days=int(pack.forward_days),
        execution_days=int(pack.execution_days),
        adjust_factor=adjust_factor,
    )


def _run_account(
    *,
    market: finite.BacktestMarket,
    book: finite.ForecastBook,
    study: Mapping[str, Any],
    years: tuple[int, ...],
    starting_cash: float,
    signal_amount: np.ndarray,
    terminal_recovery_date_idx: int | None,
    allow_pyramiding: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    account_contract = dict(study["account"])
    return finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=finite.PolicySpec(name="fixed_d60", kind="fixed", fixed_day=60),
        slots=int(account_contract["position_slots"]),
        cost_scenario=str(account_contract["cost_scenario"]),
        first_signal_date_idx=int(min(book.days)),
        last_signal_date_idx=int(max(book.days)),
        starting_cash=float(starting_cash),
        allow_pyramiding=bool(allow_pyramiding),
        replace_rejected_from_ranked_candidates=bool(
            account_contract["replace_duplicate_or_unavailable_ranked_names"]
        ),
        calendar_years=years,
        top_k=int(study["selection"]["monthly_order_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=float(
            account_contract["maximum_signal_day_amount_fraction"]
        ),
        terminal_recovery_date_idx=terminal_recovery_date_idx,
    )


def _gate(
    metric: Mapping[str, Any],
    annual: list[dict[str, Any]],
    *,
    minimum_positive_years: int,
    maximum_drawdown: float,
) -> dict[str, Any]:
    checks = {
        "liquidated_total_return_positive": float(metric["liquidated_total_return"])
        > 0.0,
        "signal_period_cagr_positive": float(metric["signal_period_cagr_trading_days"])
        > 0.0,
        "maximum_drawdown_within_limit": float(metric["signal_period_maximum_drawdown"])
        >= -float(maximum_drawdown),
        "minimum_positive_years": sum(float(row["net_return"]) > 0.0 for row in annual)
        >= int(minimum_positive_years),
        "not_ruined": metric["ruined"] is False,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_year_count": int(
            sum(float(row["net_return"]) > 0.0 for row in annual)
        ),
        "year_count": len(annual),
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
    pack = CandidateCompleteAuditPack(sources["pack_manifest"])
    feasibility._validate_pack_alignment(panel, pack)
    book, rotation, rotation_audit = _rotation_selection(
        panel,
        pack,
        study=study,
        selection_path=sources["variant_selections"],
    )
    adjust_factor, factor_audit = feasibility._load_adjust_factor_panel(
        pack, study=study
    )
    adjusted_market = _market(pack, adjust_factor=adjust_factor)
    raw_market = _market(pack, adjust_factor=None)
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    years = tuple(int(value) for value in study["evaluation"]["years"])
    starting_cash = float(study["account"]["starting_cash_cny"])
    formal_cutoff = int(
        np.flatnonzero(
            np.asarray(pack.date_values, dtype=str)
            == str(study["source"]["maximum_account_mark_date"])
        )[0]
    )
    metric, equity, trades, annual = _run_account(
        market=adjusted_market,
        book=book,
        study=study,
        years=years,
        starting_cash=starting_cash,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=formal_cutoff,
        allow_pyramiding=False,
    )
    raw_metric, raw_equity, raw_trades, raw_annual = _run_account(
        market=raw_market,
        book=book,
        study=study,
        years=years,
        starting_cash=starting_cash,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=formal_cutoff,
        allow_pyramiding=False,
    )
    bounded_metric, bounded_equity, bounded_trades, bounded_annual = _run_account(
        market=adjusted_market,
        book=book,
        study=study,
        years=years,
        starting_cash=starting_cash,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=None,
        allow_pyramiding=False,
    )
    pyramid_metric, pyramid_equity, pyramid_trades, pyramid_annual = _run_account(
        market=adjusted_market,
        book=book,
        study=study,
        years=years,
        starting_cash=starting_cash,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=formal_cutoff,
        allow_pyramiding=True,
    )
    confirmation_start = int(
        np.searchsorted(np.asarray(pack.date_values, dtype=str), "2023-01-01")
    )
    confirmation_book = feasibility._book_from_date(book, confirmation_start)
    (
        confirmation_metric,
        confirmation_equity,
        confirmation_trades,
        confirmation_annual,
    ) = _run_account(
        market=adjusted_market,
        book=confirmation_book,
        study=study,
        years=(2023, 2024, 2025),
        starting_cash=starting_cash,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=formal_cutoff,
        allow_pyramiding=False,
    )
    full_gate_contract = dict(study["evaluation"]["full_history_diagnostic"])
    confirmation_gate_contract = dict(
        study["evaluation"]["confirmation_restart_diagnostic"]
    )
    full_gate = _gate(
        metric,
        annual,
        minimum_positive_years=int(full_gate_contract["minimum_positive_years"]),
        maximum_drawdown=float(full_gate_contract["maximum_drawdown"]),
    )
    bounded_gate = _gate(
        bounded_metric,
        bounded_annual,
        minimum_positive_years=int(full_gate_contract["minimum_positive_years"]),
        maximum_drawdown=float(full_gate_contract["maximum_drawdown"]),
    )
    pyramid_gate = _gate(
        pyramid_metric,
        pyramid_annual,
        minimum_positive_years=int(full_gate_contract["minimum_positive_years"]),
        maximum_drawdown=float(full_gate_contract["maximum_drawdown"]),
    )
    confirmation_gate = _gate(
        confirmation_metric,
        confirmation_annual,
        minimum_positive_years=int(
            confirmation_gate_contract["minimum_positive_years"]
        ),
        maximum_drawdown=float(confirmation_gate_contract["maximum_drawdown"]),
    )
    capacity = feasibility._capacity_audit(
        panel,
        trades,
        maximum_fraction=float(study["account"]["maximum_signal_day_amount_fraction"]),
    )
    if not capacity["all_orders_within_limit"]:
        raise ValueError("account_capacity_limit_exceeded")
    rotation_path = root / "rotation_selections.parquet"
    equity_path = root / "equity.parquet"
    trades_path = root / "trades.parquet"
    annual_path = root / "annual.parquet"
    raw_equity_path = root / "raw_price_equity.parquet"
    raw_trades_path = root / "raw_price_trades.parquet"
    raw_annual_path = root / "raw_price_annual.parquet"
    bounded_equity_path = root / "bounded_d80_equity.parquet"
    bounded_trades_path = root / "bounded_d80_trades.parquet"
    bounded_annual_path = root / "bounded_d80_annual.parquet"
    pyramid_equity_path = root / "pyramiding_sensitivity_equity.parquet"
    pyramid_trades_path = root / "pyramiding_sensitivity_trades.parquet"
    pyramid_annual_path = root / "pyramiding_sensitivity_annual.parquet"
    confirmation_equity_path = root / "confirmation_restart_equity.parquet"
    confirmation_trades_path = root / "confirmation_restart_trades.parquet"
    confirmation_annual_path = root / "confirmation_restart_annual.parquet"
    _write_parquet(rotation_path, rotation)
    _write_parquet(equity_path, equity)
    _write_parquet(trades_path, trades)
    _write_parquet(annual_path, pd.DataFrame(annual))
    _write_parquet(raw_equity_path, raw_equity)
    _write_parquet(raw_trades_path, raw_trades)
    _write_parquet(raw_annual_path, pd.DataFrame(raw_annual))
    _write_parquet(bounded_equity_path, bounded_equity)
    _write_parquet(bounded_trades_path, bounded_trades)
    _write_parquet(bounded_annual_path, pd.DataFrame(bounded_annual))
    _write_parquet(pyramid_equity_path, pyramid_equity)
    _write_parquet(pyramid_trades_path, pyramid_trades)
    _write_parquet(pyramid_annual_path, pd.DataFrame(pyramid_annual))
    _write_parquet(confirmation_equity_path, confirmation_equity)
    _write_parquet(confirmation_trades_path, confirmation_trades)
    _write_parquet(confirmation_annual_path, pd.DataFrame(confirmation_annual))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_posthoc_exact_account_falsification_with_execution_semantics_repair",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "rotation_audit": rotation_audit,
        "metric": metric,
        "annual": annual,
        "full_history_gate": full_gate,
        "confirmation_restart_metric": confirmation_metric,
        "confirmation_restart_annual": confirmation_annual,
        "confirmation_restart_gate": confirmation_gate,
        "raw_price_sensitivity_metric": raw_metric,
        "raw_price_sensitivity_annual": raw_annual,
        "bounded_d80_writeoff_sensitivity_metric": bounded_metric,
        "bounded_d80_writeoff_sensitivity_annual": bounded_annual,
        "bounded_d80_writeoff_sensitivity_gate": bounded_gate,
        "pyramiding_sensitivity_metric": pyramid_metric,
        "pyramiding_sensitivity_annual": pyramid_annual,
        "pyramiding_sensitivity_gate": pyramid_gate,
        "capacity_audit": capacity,
        "adjust_factor_audit": factor_audit,
        "natural_phase_selected_without_optimization": True,
        "account_optimization_performed": False,
        "posthoc_candidate": True,
        "posthoc_execution_semantics_repair": True,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "round_lots_and_minimum_commission_modeled": True,
        "production_strategy_claim_allowed": False,
        "decision": {
            "retrospective_exact_account_diagnostic_passed": bool(
                full_gate["passed"] and confirmation_gate["passed"]
            ),
            "forward_confirmation_required": True,
            "production_strategy_claim_allowed": False,
        },
        "files": {
            "rotation_selections": base._file_record(rotation_path),
            "equity": base._file_record(equity_path),
            "trades": base._file_record(trades_path),
            "annual": base._file_record(annual_path),
            "raw_price_equity": base._file_record(raw_equity_path),
            "raw_price_trades": base._file_record(raw_trades_path),
            "raw_price_annual": base._file_record(raw_annual_path),
            "bounded_d80_equity": base._file_record(bounded_equity_path),
            "bounded_d80_trades": base._file_record(bounded_trades_path),
            "bounded_d80_annual": base._file_record(bounded_annual_path),
            "pyramiding_sensitivity_equity": base._file_record(pyramid_equity_path),
            "pyramiding_sensitivity_trades": base._file_record(pyramid_trades_path),
            "pyramiding_sensitivity_annual": base._file_record(pyramid_annual_path),
            "confirmation_restart_equity": base._file_record(confirmation_equity_path),
            "confirmation_restart_trades": base._file_record(confirmation_trades_path),
            "confirmation_restart_annual": base._file_record(confirmation_annual_path),
        },
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "natural_phase": summary.get("natural_phase_selected_without_optimization")
        is True,
        "no_optimization": summary.get("account_optimization_performed") is False,
        "posthoc": summary.get("posthoc_candidate") is True,
        "execution_semantics_repair": summary.get("posthoc_execution_semantics_repair")
        is True,
        "lots": summary.get("round_lots_and_minimum_commission_modeled") is True,
        "no_low_price_reward": summary.get("52_week_low_reward_used") is False,
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
