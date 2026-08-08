"""Exact finite-account replay of the information-gate-passing value policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as causal
from daily_research.path_policy import seq100_exact_value_growth_policy as exact
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_account_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_account_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_exact_value_growth_account_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exact_value_growth_account_v1"
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
        "source_policy": "exact_full_top10",
        "monthly_order_count": 10,
        "candidate_scan_count": 30,
        "maximum_names_per_pit_industry": 2,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"selection_contract_mismatch:{key}")
    account = dict(study["account"])
    expected_account = {
        "starting_cash_cny": 1_000_000.0,
        "position_slots": 30,
        "cost_scenario": "double_slippage",
        "pyramiding": False,
        "lot_size": 100,
        "round_lots_and_minimum_commission_modeled": True,
    }
    for key, expected in expected_account.items():
        if account.get(key) != expected:
            raise ValueError(f"account_contract_mismatch:{key}")
    risk = dict(study["risk_budget"])
    if risk != {
        "derivation_end_date": "2019-12-31",
        "target_annualized_volatility": 0.15,
        "minimum_gross_fraction": 0.25,
        "maximum_gross_fraction": 1.0,
        "parameter_grid_performed": False,
    }:
        raise ValueError("risk_budget_contract_mismatch")
    if bool(study["decision_boundary"].get("account_optimization_performed", True)):
        raise ValueError("account_optimization_forbidden")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    keys = (
        "exact_policy_summary",
        "candidate_features",
        "selections",
        "pack_manifest",
    )
    paths: dict[str, Path] = {}
    for key in keys:
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    paths["input_manifest"] = base._resolve_path(source["input_manifest"])
    paths["label_manifest"] = base._resolve_path(source["label_manifest"])
    exact.validate_summary(paths["exact_policy_summary"])
    summary = _read_json(paths["exact_policy_summary"])
    if summary["decision"]["information_gate_passed"] is not True:
        raise ValueError("source_information_gate_not_passed")
    if summary["decision"]["account_replay_authorized"] is not True:
        raise ValueError("source_account_replay_not_authorized")
    return paths


def _account_selection(
    panel: base.StockPanel,
    pack: CandidateCompleteAuditPack,
    *,
    study: Mapping[str, Any],
    feature_path: Path,
    selection_path: Path,
) -> tuple[finite.ForecastBook, pd.DataFrame, dict[str, Any]]:
    contract = dict(study["selection"])
    features = pd.read_parquet(feature_path)
    source_selected = pd.read_parquet(selection_path)
    source_selected = source_selected.loc[
        source_selected["policy"].eq(str(contract["source_policy"]))
    ].copy()
    identity = panel.row_index[["candidate_id", "date_idx", "symbol_idx", "symbol"]]
    features = features.merge(
        identity,
        on=["candidate_id", "date_idx", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if bool(features["symbol_idx"].isna().any()):
        raise ValueError("account_symbol_alignment_failed")
    cutoff_positions = np.flatnonzero(
        np.asarray(pack.date_values, dtype=str)
        == str(study["source"]["maximum_account_mark_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_account_mark_date_missing")
    cutoff = int(cutoff_positions[0])
    book = finite.ForecastBook(
        "exact_value_growth_top10",
        top_k=int(contract["monthly_order_count"]),
        candidate_scan_k=int(contract["candidate_scan_count"]),
    )
    records: list[pd.DataFrame] = []
    dates = features["date_idx"].to_numpy(dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(features)]
    source_alignment_count = 0
    excluded_boundary_count = 0
    for left, right in pairwise(boundaries):
        local = features.iloc[int(left) : int(right)]
        date_idx = int(local["date_idx"].iloc[0])
        if date_idx + int(pack.execution_days) > cutoff:
            excluded_boundary_count += 1
            continue
        score = local["exact_full_score"].to_numpy(dtype=float)
        eligible = local["eligible__exact_full_top10"].to_numpy(dtype=bool)
        chosen = causal._select_industry_capped(
            score=score,
            eligible=eligible,
            candidate_id=local["candidate_id"].to_numpy(dtype=np.int64),
            industry_code=local["industry_code"].to_numpy(dtype=np.int64),
            top_k=int(contract["candidate_scan_count"]),
            industry_cap=int(contract["maximum_names_per_pit_industry"]),
        )
        if not len(chosen):
            continue
        scan = local.iloc[chosen].copy()
        scan["candidate_scan_rank"] = np.arange(1, len(scan) + 1)
        expected = source_selected.loc[
            source_selected["date_idx"].eq(date_idx)
        ].sort_values("selection_rank")
        actual_top = scan.head(int(contract["monthly_order_count"]))
        if not np.array_equal(
            expected["candidate_id"].to_numpy(dtype=np.int64),
            actual_top["candidate_id"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"account_source_selection_mismatch:{date_idx}")
        source_alignment_count += len(expected)
        book.add_day(
            date_idx=date_idx,
            symbol_idx=scan["symbol_idx"].to_numpy(dtype=np.int64),
            score=scan["exact_full_score"].to_numpy(dtype=float),
            planned_day=np.full(len(scan), 60, dtype=np.int16),
        )
        records.append(
            scan[
                [
                    "candidate_id",
                    "date_idx",
                    "trade_date",
                    "symbol",
                    "symbol_idx",
                    "industry_code",
                    "candidate_scan_rank",
                    "exact_full_score",
                ]
            ]
        )
    if not book.days or not records:
        raise ValueError("account_forecast_book_empty")
    selected = pd.concat(records, ignore_index=True)
    top_counts = (
        selected.loc[
            selected["candidate_scan_rank"].le(int(contract["monthly_order_count"]))
        ]
        .groupby("date_idx")
        .size()
    )
    scan_counts = selected.groupby("date_idx").size()
    top_industry_max = (
        selected.loc[
            selected["candidate_scan_rank"].le(int(contract["monthly_order_count"]))
        ]
        .groupby(["date_idx", "industry_code"])
        .size()
        .groupby("date_idx")
        .max()
    )
    return (
        book,
        selected,
        {
            "signal_date_count": len(book.days),
            "first_signal_date_idx": int(min(book.days)),
            "last_signal_date_idx": int(max(book.days)),
            "first_signal_date": str(pack.date_values[min(book.days)]),
            "last_signal_date": str(pack.date_values[max(book.days)]),
            "boundary_excluded_signal_date_count": excluded_boundary_count,
            "top_count_min": int(top_counts.min()),
            "top_count_max": int(top_counts.max()),
            "top_count_mean": float(top_counts.mean()),
            "scan_count_min": int(scan_counts.min()),
            "scan_count_max": int(scan_counts.max()),
            "source_top_selection_alignment_count": source_alignment_count,
            "maximum_same_industry_top_names": int(top_industry_max.max()),
            "future_outcomes_used": False,
        },
    )


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
    signal_amount: np.ndarray,
    terminal_recovery_date_idx: int,
    target_gross_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    account = dict(study["account"])
    return finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=finite.PolicySpec(name="fixed_d60", kind="fixed", fixed_day=60),
        slots=int(account["position_slots"]),
        cost_scenario=str(account["cost_scenario"]),
        first_signal_date_idx=int(min(book.days)),
        last_signal_date_idx=int(max(book.days)),
        starting_cash=float(account["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=bool(
            account["replace_duplicate_ranked_names"]
        ),
        calendar_years=years,
        top_k=int(study["selection"]["monthly_order_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=float(
            account["maximum_signal_day_amount_fraction"]
        ),
        target_gross_fraction=float(target_gross_fraction),
        terminal_recovery_date_idx=int(terminal_recovery_date_idx),
    )


def _account_gate(
    metric: Mapping[str, Any],
    annual: list[dict[str, Any]],
    *,
    contract: Mapping[str, Any],
    risk_account: bool,
) -> dict[str, Any]:
    checks = {
        "liquidated_total_return_positive": float(metric["liquidated_total_return"])
        > 0.0,
        "signal_period_cagr_positive": float(metric["signal_period_cagr_trading_days"])
        > 0.0,
        "maximum_drawdown_within_limit": float(metric["signal_period_maximum_drawdown"])
        >= -float(contract["maximum_drawdown"]),
        "minimum_positive_years": sum(float(row["net_return"]) > 0.0 for row in annual)
        >= int(contract["minimum_positive_years"]),
        "not_ruined": metric["ruined"] is False,
    }
    if risk_account:
        checks["annualized_volatility_within_limit"] = float(
            metric["signal_period_annualized_volatility"]
        ) <= float(contract["maximum_annualized_volatility_for_risk_account"])
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_year_count": int(
            sum(float(row["net_return"]) > 0.0 for row in annual)
        ),
        "year_count": len(annual),
    }


def _write_run(
    root: Path,
    name: str,
    metric: Mapping[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    annual: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    equity_path = root / f"{name}_equity.parquet"
    trades_path = root / f"{name}_trades.parquet"
    annual_path = root / f"{name}_annual.parquet"
    _write_parquet(equity_path, equity)
    _write_parquet(trades_path, trades)
    _write_parquet(annual_path, pd.DataFrame(annual))
    return {
        f"{name}_equity": base._file_record(equity_path),
        f"{name}_trades": base._file_record(trades_path),
        f"{name}_annual": base._file_record(annual_path),
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
            for record in current.get("files", {}).values()
        ):
            return current
    panel = base.load_panel(
        input_manifest_path=sources["input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    pack = CandidateCompleteAuditPack(sources["pack_manifest"])
    feasibility._validate_pack_alignment(panel, pack)
    book, account_candidates, selection_audit = _account_selection(
        panel,
        pack,
        study=study,
        feature_path=sources["candidate_features"],
        selection_path=sources["selections"],
    )
    adjust_factor, factor_audit = feasibility._load_adjust_factor_panel(
        pack, study=study
    )
    adjusted_market = _market(pack, adjust_factor=adjust_factor)
    raw_market = _market(pack, adjust_factor=None)
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    years = tuple(int(value) for value in study["evaluation"]["years"])
    cutoff = int(
        np.flatnonzero(
            np.asarray(pack.date_values, dtype=str)
            == str(study["source"]["maximum_account_mark_date"])
        )[0]
    )
    full = _run_account(
        market=adjusted_market,
        book=book,
        study=study,
        years=years,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=1.0,
    )
    raw = _run_account(
        market=raw_market,
        book=book,
        study=study,
        years=years,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=1.0,
    )
    confirmation_start = int(
        np.searchsorted(np.asarray(pack.date_values, dtype=str), "2023-01-01")
    )
    confirmation_book = feasibility._book_from_date(book, confirmation_start)
    confirmation = _run_account(
        market=adjusted_market,
        book=confirmation_book,
        study=study,
        years=(2023, 2024, 2025),
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=1.0,
    )
    risk_budget = feasibility._derive_risk_budget(
        full[1],
        starting_cash=float(study["account"]["starting_cash_cny"]),
        config=study["risk_budget"],
    )
    gross_fraction = float(risk_budget["frozen_target_gross_fraction"])
    risk = _run_account(
        market=adjusted_market,
        book=book,
        study=study,
        years=years,
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=gross_fraction,
    )
    risk_confirmation = _run_account(
        market=adjusted_market,
        book=confirmation_book,
        study=study,
        years=(2023, 2024, 2025),
        signal_amount=signal_amount,
        terminal_recovery_date_idx=cutoff,
        target_gross_fraction=gross_fraction,
    )
    full_gate = _account_gate(
        full[0],
        full[3],
        contract=study["evaluation"]["full_history"],
        risk_account=False,
    )
    confirmation_gate = _account_gate(
        confirmation[0],
        confirmation[3],
        contract=study["evaluation"]["confirmation_restart"],
        risk_account=False,
    )
    risk_gate = _account_gate(
        risk[0],
        risk[3],
        contract=study["evaluation"]["full_history"],
        risk_account=True,
    )
    risk_confirmation_gate = _account_gate(
        risk_confirmation[0],
        risk_confirmation[3],
        contract=study["evaluation"]["confirmation_restart"],
        risk_account=True,
    )
    capacity = feasibility._capacity_audit(
        panel,
        risk[2],
        maximum_fraction=float(study["account"]["maximum_signal_day_amount_fraction"]),
    )
    if not capacity["all_orders_within_limit"]:
        raise ValueError("account_capacity_limit_exceeded")
    candidate_path = root / "account_candidates.parquet"
    _write_parquet(candidate_path, account_candidates)
    files: dict[str, dict[str, Any]] = {
        "account_candidates": base._file_record(candidate_path)
    }
    for name, result in (
        ("full", full),
        ("raw_price", raw),
        ("confirmation_restart", confirmation),
        ("risk_budget", risk),
        ("risk_confirmation_restart", risk_confirmation),
    ):
        files.update(_write_run(root, name, *result))
    decision_passed = bool(
        (full_gate["passed"] and confirmation_gate["passed"])
        or (risk_gate["passed"] and risk_confirmation_gate["passed"])
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_posthoc_exact_account_validation",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "source_information_gate_passed": True,
        "selection_audit": selection_audit,
        "full_metric": full[0],
        "full_annual": full[3],
        "full_gate": full_gate,
        "confirmation_restart_metric": confirmation[0],
        "confirmation_restart_annual": confirmation[3],
        "confirmation_restart_gate": confirmation_gate,
        "raw_price_sensitivity_metric": raw[0],
        "raw_price_sensitivity_annual": raw[3],
        "risk_budget": risk_budget,
        "risk_metric": risk[0],
        "risk_annual": risk[3],
        "risk_gate": risk_gate,
        "risk_confirmation_restart_metric": risk_confirmation[0],
        "risk_confirmation_restart_annual": risk_confirmation[3],
        "risk_confirmation_restart_gate": risk_confirmation_gate,
        "capacity_audit": capacity,
        "adjust_factor_audit": factor_audit,
        "account_optimization_performed": False,
        "round_lots_and_minimum_commission_modeled": True,
        "52_week_low_reward_used": False,
        "forbidden_2026_read_count": 0,
        "production_strategy_claim_allowed": False,
        "decision": {
            "retrospective_exact_account_diagnostic_passed": decision_passed,
            "full_exposure_passed": bool(
                full_gate["passed"] and confirmation_gate["passed"]
            ),
            "development_only_risk_budget_passed": bool(
                risk_gate["passed"] and risk_confirmation_gate["passed"]
            ),
            "forward_confirmation_required": True,
            "production_strategy_claim_allowed": False,
        },
        "files": files,
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "source_gate": summary.get("source_information_gate_passed") is True,
        "no_optimization": summary.get("account_optimization_performed") is False,
        "lots": summary.get("round_lots_and_minimum_commission_modeled") is True,
        "no_low": summary.get("52_week_low_reward_used") is False,
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("production_strategy_claim_allowed") is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
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
