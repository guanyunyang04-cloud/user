"""Non-promotional finite-account falsification for the margin residual policy."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from daily_research.path_policy import seq100_finite_capital_backtest as account
from daily_research.path_policy import seq100_hierarchical_residual_policy as hierarchy
from daily_research.path_policy import seq100_margin_residual_policy as margin
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_margin_residual_account_feasibility_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/"
    / "seq100_margin_residual_account_feasibility_v1"
)
STUDY_ID = "seq100_margin_residual_account_feasibility_v1"
SUMMARY_SCHEMA = "seq100_margin_residual_account_feasibility/1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    interpretation = dict(study["interpretation"])
    if interpretation.get("source_confirmation_gate_passed") is not False:
        raise ValueError("source_confirmation_failure_must_remain_explicit")
    if (
        interpretation.get("account_success_cannot_override_confirmation_failure")
        is not True
    ):
        raise ValueError("nonpromotional_boundary_missing")
    if bool(study["decision_boundary"].get("account_optimization_performed", True)):
        raise ValueError("account_grid_forbidden")
    account_config = dict(study["account"])
    expected = {
        "daily_ranked_candidate_count": 48,
        "position_slots": 48,
        "cost_scenario": "double_slippage",
        "pyramiding": False,
    }
    for key, value in expected.items():
        if account_config.get(key) != value:
            raise ValueError(f"account_contract_mismatch:{key}")
    risk = dict(study["risk_budget_diagnostic"])
    if risk.get("parameter_grid_performed") is not False:
        raise ValueError("risk_budget_grid_forbidden")
    if str(risk.get("derivation_end_date")) != "2022-12-31":
        raise ValueError("risk_budget_development_boundary_mismatch")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(study["source"])
    paths = {
        name: base._resolve_path(source[key])
        for name, key in (
            ("policy_study", "margin_policy_study"),
            ("development", "development_summary"),
            ("confirmation", "confirmation_summary"),
            ("pack", "pack_manifest"),
        )
    }
    for name, key in (
        ("development", "development_summary_sha256"),
        ("confirmation", "confirmation_summary_sha256"),
        ("pack", "pack_manifest_sha256"),
    ):
        if base._sha256_file(paths[name]) != source[key]:
            raise ValueError(f"source_hash_mismatch:{name}")
    development = _read_json(paths["development"])
    confirmation = _read_json(paths["confirmation"])
    if development["gate"]["passed"] is not True:
        raise ValueError("development_gate_not_passed")
    if confirmation["gate"]["passed"] is not False:
        raise ValueError("confirmation_failure_drifted")
    policy_study, policy_study_path = margin.load_study(paths["policy_study"])
    policy_source = margin._source_contract(policy_study)
    return {
        "paths": paths,
        "development": development,
        "confirmation": confirmation,
        "policy_study": policy_study,
        "policy_study_path": policy_study_path,
        "policy_source": policy_source,
    }


def _forecast_path_for_year(source: Mapping[str, Any], year: int) -> Path:
    paths = dict(source["paths"])
    if year in margin.NEW_DEVELOPMENT_YEARS:
        summary = _read_json(
            paths["development"].parent
            / "stock_forecasts"
            / f"y{year}"
            / "summary.json"
        )
        return Path(str(summary["file"]["path"]))
    if year in margin.REUSED_DEVELOPMENT_YEARS:
        return Path(source["policy_source"]["frozen_paths"][year])
    if year in margin.CONFIRMATION_YEARS:
        summary = _read_json(
            paths["confirmation"].parent
            / "stock_forecasts"
            / f"y{year}"
            / "summary.json"
        )
        return Path(str(summary["file"]["path"]))
    raise ValueError(f"forecast_year_unsupported:{year}")


def _validate_pack_alignment(
    panel: base.StockPanel, pack: CandidateCompleteAuditPack
) -> None:
    if len(pack.symbol_values) <= int(panel.symbol_idx.max()):
        raise ValueError("pack_symbol_dimension_too_small")
    symbols = panel.row_index[["symbol_idx", "symbol"]].drop_duplicates("symbol_idx")
    expected = np.asarray(pack.symbol_values, dtype=str)[
        symbols["symbol_idx"].to_numpy(dtype=np.int64)
    ]
    if not np.array_equal(expected, symbols["symbol"].astype(str).to_numpy()):
        raise ValueError("pack_symbol_alignment_failed")
    dates = panel.row_index[["date_idx", "trade_date"]].drop_duplicates("date_idx")
    expected_dates = np.asarray(pack.date_values, dtype=str)[
        dates["date_idx"].to_numpy(dtype=np.int64)
    ]
    if not np.array_equal(expected_dates, dates["trade_date"].astype(str).to_numpy()):
        raise ValueError("pack_date_alignment_failed")


def _build_forecast_book(
    panel: base.StockPanel,
    *,
    study: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[account.ForecastBook, dict[str, Any]]:
    account_config = dict(study["account"])
    top_k = int(account_config["daily_ranked_candidate_count"])
    book = account.ForecastBook(
        "margin_observed_residual",
        top_k=top_k,
        candidate_scan_k=top_k,
    )
    maximum_signal_date = str(study["source"]["maximum_signal_date"])
    audit_rows: list[dict[str, Any]] = []
    policy_study = source["policy_study"]
    policy_source = source["policy_source"]
    years = tuple(int(value) for value in study["evaluation"]["years"])
    for year in years:
        path = _forecast_path_for_year(source, year)
        residual, loss = hierarchy._load_frozen_stock_forecasts(panel, year, path)
        partition = dict(policy_source["margin_partitions"][str(year)])
        observed, margin_audit = margin._margin_observed_mask(
            panel, year=year, partition_record=partition
        )
        rows = panel.rows_for_year(year)
        dates = panel.date_idx[rows]
        identities = panel.row_index.iloc[rows]["candidate_id"].to_numpy(dtype=np.int64)
        symbols = panel.symbol_idx[rows]
        industries = panel.industry_code[rows]
        boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(rows)]
        added_dates = 0
        for left, right in pairwise(boundaries):
            left_i, right_i = int(left), int(right)
            trade_date = str(panel.trade_date[rows[left_i]])
            if trade_date > maximum_signal_date:
                continue
            local = slice(left_i, right_i)
            selected = margin._select_industry_capped(
                residual_score=residual[local],
                bad_tail_score=loss[local],
                candidate_id=identities[local],
                industry_code=industries[local],
                margin_observed=observed[local],
                top_k=int(policy_study["policy"]["top_k"]),
                veto_fraction=float(policy_study["policy"]["bad_tail_veto_fraction"]),
                industry_cap=int(
                    policy_study["policy"]["maximum_names_per_pit_industry"]
                ),
            )
            if len(selected) != top_k:
                raise ValueError(f"account_selection_not_full:{trade_date}")
            book.add_day(
                date_idx=int(dates[left_i]),
                symbol_idx=symbols[local][selected],
                score=residual[local][selected],
                planned_day=np.full(top_k, 20, dtype=np.int16),
            )
            added_dates += 1
        audit_rows.append(
            {
                **margin_audit,
                "forecast_path": str(path),
                "account_signal_date_count": added_dates,
            }
        )
    if not book.days:
        raise ValueError("account_forecast_book_empty")
    return book, {
        "years": audit_rows,
        "first_signal_date_idx": int(min(book.days)),
        "last_signal_date_idx": int(max(book.days)),
        "signal_date_count": len(book.days),
    }


def _book_from_date(
    book: account.ForecastBook, minimum_date_idx: int
) -> account.ForecastBook:
    result = account.ForecastBook(
        book.profile,
        top_k=book.top_k,
        candidate_scan_k=book.candidate_scan_k,
    )
    result.days = {
        int(date_idx): day
        for date_idx, day in book.days.items()
        if int(date_idx) >= int(minimum_date_idx)
    }
    if not result.days:
        raise ValueError("filtered_account_forecast_book_empty")
    return result


def _book_until_date(
    book: account.ForecastBook, maximum_date_idx: int
) -> account.ForecastBook:
    result = account.ForecastBook(
        book.profile,
        top_k=book.top_k,
        candidate_scan_k=book.candidate_scan_k,
    )
    result.days = {
        int(date_idx): day
        for date_idx, day in book.days.items()
        if int(date_idx) <= int(maximum_date_idx)
    }
    if not result.days:
        raise ValueError("filtered_development_forecast_book_empty")
    return result


def _capacity_audit(
    panel: base.StockPanel,
    trades: pd.DataFrame,
    *,
    maximum_fraction: float,
) -> dict[str, Any]:
    width = int(panel.symbol_idx.max()) + 1
    row_keys = panel.date_idx.astype(np.int64) * width + panel.symbol_idx.astype(
        np.int64
    )
    trade_keys = trades["signal_date_idx"].to_numpy(dtype=np.int64) * width + trades[
        "symbol_idx"
    ].to_numpy(dtype=np.int64)
    positions = np.searchsorted(row_keys, trade_keys)
    if not np.array_equal(row_keys[positions], trade_keys):
        raise ValueError("capacity_trade_alignment_failed")
    amount_position = base._feature_positions(panel, ("log_amount_1d",))
    signal_amount = np.expm1(
        base._feature_matrix(panel, positions, amount_position)[:, 0].astype(np.float64)
    )
    participation = (
        trades["buy_notional_cny"].to_numpy(dtype=np.float64) / signal_amount
    )
    finite = np.isfinite(participation) & (participation >= 0.0)
    if not bool(finite.all()):
        raise ValueError("capacity_participation_invalid")
    return {
        "trade_count": len(trades),
        "maximum_signal_amount_fraction": float(maximum_fraction),
        "maximum_observed_participation": float(np.max(participation)),
        "mean_observed_participation": float(np.mean(participation)),
        "p99_observed_participation": float(np.quantile(participation, 0.99)),
        "orders_above_limit": int(np.sum(participation > float(maximum_fraction))),
        "all_orders_within_limit": bool(
            np.all(participation <= float(maximum_fraction))
        ),
    }


def _signal_amount_panel(
    panel: base.StockPanel,
    pack: CandidateCompleteAuditPack,
) -> np.ndarray:
    """Build the causal signal-day amount matrix used by the execution cap."""

    result = np.full(
        (len(pack.date_values), len(pack.symbol_values)), np.nan, dtype=np.float32
    )
    amount_position = base._feature_positions(panel, ("log_amount_1d",))
    for left in range(0, len(panel.date_idx), 500_000):
        right = min(left + 500_000, len(panel.date_idx))
        rows = np.arange(left, right, dtype=np.int64)
        log_amount = base._feature_matrix(panel, rows, amount_position)[:, 0]
        amount = np.expm1(log_amount.astype(np.float64)).astype(np.float32)
        valid = np.isfinite(amount) & (amount > 0.0)
        result[panel.date_idx[rows[valid]], panel.symbol_idx[rows[valid]]] = amount[
            valid
        ]
    return result


def _derive_risk_budget(
    equity: pd.DataFrame,
    *,
    starting_cash: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dates = pd.to_datetime(equity["trade_date"], errors="raise")
    mask = dates.le(pd.Timestamp(str(config["derivation_end_date"]))) & equity[
        "inside_signal_period"
    ].astype(bool)
    values = equity.loc[mask, "equity"].to_numpy(dtype=np.float64)
    if len(values) < 252:
        raise ValueError("risk_budget_development_path_too_short")
    path = np.r_[float(starting_cash), values]
    returns = path[1:] / path[:-1] - 1.0
    realized = float(np.std(returns, ddof=1) * np.sqrt(252.0))
    target = float(config["target_annualized_volatility"])
    fraction = float(
        np.clip(
            target / realized,
            float(config["minimum_gross_fraction"]),
            float(config["maximum_gross_fraction"]),
        )
    )
    return {
        "derivation_end_date": str(config["derivation_end_date"]),
        "development_session_count": len(values),
        "development_realized_annualized_volatility": realized,
        "target_annualized_volatility": target,
        "minimum_gross_fraction": float(config["minimum_gross_fraction"]),
        "maximum_gross_fraction": float(config["maximum_gross_fraction"]),
        "frozen_target_gross_fraction": fraction,
        "parameter_grid_performed": False,
        "confirmation_data_used_in_derivation": False,
    }


def _risk_budget_gate(
    metric: Mapping[str, Any],
    annual: list[dict[str, Any]],
    *,
    years: tuple[int, ...],
    maximum_drawdown: float,
    maximum_volatility: float,
    minimum_positive_years: int,
) -> dict[str, Any]:
    selected = [row for row in annual if int(row["year"]) in set(years)]
    checks = {
        "all_years_present": len(selected) == len(years),
        "period_total_return_positive": float(metric["signal_period_total_return"])
        > 0.0,
        "maximum_drawdown_within_limit": float(metric["signal_period_maximum_drawdown"])
        >= -float(maximum_drawdown),
        "annualized_volatility_within_limit": float(
            metric["signal_period_annualized_volatility"]
        )
        <= float(maximum_volatility),
        "minimum_positive_years_met": sum(
            float(row["net_return"]) > 0.0 for row in selected
        )
        >= int(minimum_positive_years),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_year_count": int(
            sum(float(row["net_return"]) > 0.0 for row in selected)
        ),
        "year_count": len(years),
        "maximum_drawdown_limit": -float(maximum_drawdown),
        "maximum_annualized_volatility": float(maximum_volatility),
        "minimum_positive_years": int(minimum_positive_years),
    }


def _load_adjust_factor_panel(
    pack: CandidateCompleteAuditPack,
    *,
    study: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the pack-pinned total-return factor without reading 2026 rows."""

    expected_id = str(study["source"]["adjust_factor_dataset_id"])
    expected_sha256 = str(study["source"]["adjust_factor_manifest_sha256"])
    source = dict(pack.manifest["qdp_sources"]["adjust_factor"])
    if str(source["dataset_id"]) != expected_id:
        raise ValueError("adjust_factor_dataset_id_mismatch")
    manifest_path = Path(str(source["manifest_path"])).resolve()
    if base._sha256_file(manifest_path) != expected_sha256:
        raise ValueError("adjust_factor_manifest_hash_mismatch")
    manifest = _read_json(manifest_path)
    if str(manifest["dataset_id"]) != expected_id:
        raise ValueError("adjust_factor_manifest_dataset_id_mismatch")
    if dict(manifest.get("quality", {})).get("adjust_factor_semantics") != (
        "back_adjust_factor"
    ):
        raise ValueError("adjust_factor_semantics_mismatch")
    qdp_root = Path(str(pack.manifest["qdp_root"])).resolve()
    shard_paths = []
    for record in list(manifest.get("shards", []) or []):
        raw = Path(str(record["path"]))
        path = raw if raw.is_absolute() else qdp_root / raw
        if not path.is_file():
            raise FileNotFoundError(path)
        shard_paths.append(path)
    if not shard_paths:
        raise ValueError("adjust_factor_shards_empty")

    first_date = f"{min(int(value) for value in study['evaluation']['years'])}-01-01"
    last_date = str(study["source"]["maximum_account_mark_date"])
    dates = pd.Index(np.asarray(pack.date_values, dtype=str))
    symbols = pd.Index(np.asarray(pack.symbol_values, dtype=str))
    panel = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
    seen = np.zeros(panel.shape, dtype=bool)
    dataset = ds.dataset([str(path) for path in shard_paths], format="parquet")
    scanner = dataset.scanner(
        columns=["trade_date", "symbol", "adjust_factor"],
        filter=(ds.field("trade_date") >= first_date)
        & (ds.field("trade_date") <= last_date),
        batch_size=262_144,
    )
    loaded_rows = 0
    duplicate_rows = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        date_idx = dates.get_indexer(frame["trade_date"].astype(str))
        symbol_idx = symbols.get_indexer(frame["symbol"].astype(str))
        factor = pd.to_numeric(frame["adjust_factor"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        valid = (
            (date_idx >= 0) & (symbol_idx >= 0) & np.isfinite(factor) & (factor > 0.0)
        )
        if not bool(valid.all()):
            raise ValueError(
                f"adjust_factor_invalid_or_unaligned_rows:{int((~valid).sum())}"
            )
        duplicate_rows += int(seen[date_idx, symbol_idx].sum())
        panel[date_idx, symbol_idx] = factor.astype(np.float32)
        seen[date_idx, symbol_idx] = True
        loaded_rows += len(frame)
    if duplicate_rows:
        raise ValueError(f"adjust_factor_duplicate_rows:{duplicate_rows}")

    first_idx = int(
        np.searchsorted(np.asarray(pack.date_values, dtype=str), first_date)
    )
    last_idx = int(np.searchsorted(np.asarray(pack.date_values, dtype=str), last_date))
    if (
        last_idx >= len(pack.date_values)
        or str(pack.date_values[last_idx]) != last_date
    ):
        raise ValueError("maximum_account_mark_date_missing")
    required_count = 0
    missing_required_count = 0
    for left in range(first_idx, last_idx + 1, 128):
        right = min(left + 128, last_idx + 1)
        required = np.isfinite(
            np.asarray(pack.exit_close_raw[left:right])
        ) | np.isfinite(np.asarray(pack.entry_open_raw[left:right]))
        available = np.isfinite(panel[left:right]) & (panel[left:right] > 0.0)
        required_count += int(required.sum())
        missing_required_count += int((required & ~available).sum())
    if missing_required_count:
        raise ValueError(
            f"adjust_factor_missing_required_symbol_days:{missing_required_count}"
        )
    return panel, {
        "dataset_id": expected_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_sha256,
        "semantics": "back_adjust_factor_total_return_equivalent",
        "first_date_read": first_date,
        "last_date_read": last_date,
        "forbidden_2026_read_count": 0,
        "loaded_row_count": int(loaded_rows),
        "duplicate_row_count": int(duplicate_rows),
        "required_raw_open_or_close_count": int(required_count),
        "missing_required_count": int(missing_required_count),
    }


def run_account_feasibility(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, resolved_study_path = load_study(study_path)
    source = _source_contract(study)
    output = base._resolve_path(output_root)
    summary_path = output / "summary.json"
    study_sha256 = base._sha256_file(resolved_study_path)
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("study_sha256") != study_sha256:
            raise ValueError("existing_account_summary_study_mismatch")
        if not all(
            base._record_valid(record, verify_hash=True)
            for record in dict(current.get("files", {}) or {}).values()
        ):
            raise ValueError("existing_account_output_invalid")
        return current

    panel = base.load_panel(
        input_manifest_path=source["policy_source"]["input_manifest_path"],
        label_manifest_path=source["policy_source"]["label_manifest_path"],
    )
    pack = CandidateCompleteAuditPack(source["paths"]["pack"])
    _validate_pack_alignment(panel, pack)
    book, book_audit = _build_forecast_book(panel, study=study, source=source)
    first_signal = int(book_audit["first_signal_date_idx"])
    last_signal = int(book_audit["last_signal_date_idx"])
    final_date_idx = last_signal + int(pack.execution_days)
    maximum_mark_date = str(study["source"]["maximum_account_mark_date"])
    if final_date_idx >= len(pack.date_values):
        raise ValueError("account_execution_tail_incomplete")
    if str(pack.date_values[final_date_idx]) > maximum_mark_date:
        raise ValueError("account_would_read_beyond_maximum_mark_date")
    market_raw = account.BacktestMarket(
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
    )
    adjust_factor, factor_audit = _load_adjust_factor_panel(pack, study=study)
    market_adjusted = account.BacktestMarket(
        date_values=market_raw.date_values,
        symbol_values=market_raw.symbol_values,
        entry_open_raw=market_raw.entry_open_raw,
        exit_close_raw=market_raw.exit_close_raw,
        exit_sellable=market_raw.exit_sellable,
        entry_filled=market_raw.entry_filled,
        costs=market_raw.costs,
        terminal_recovery_fraction=market_raw.terminal_recovery_fraction,
        forward_days=market_raw.forward_days,
        execution_days=market_raw.execution_days,
        adjust_factor=adjust_factor,
    )
    config = dict(study["account"])
    signal_amount = _signal_amount_panel(panel, pack)
    capacity_fraction = float(config["maximum_signal_day_amount_fraction"])
    metric, equity, trades, annual = account.simulate_portfolio(
        market=market_adjusted,
        book=book,
        raw_top3_paths={},
        policy=account.PolicySpec(name="fixed_d20", kind="fixed", fixed_day=20),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=first_signal,
        last_signal_date_idx=last_signal,
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=True,
        calendar_years=tuple(int(value) for value in study["evaluation"]["years"]),
        top_k=int(config["daily_ranked_candidate_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=capacity_fraction,
    )
    confirmation_start = int(
        np.searchsorted(np.asarray(pack.date_values, dtype=str), "2023-01-01")
    )
    confirmation_book = _book_from_date(book, confirmation_start)
    (
        confirmation_metric,
        confirmation_equity,
        confirmation_trades,
        confirmation_annual,
    ) = account.simulate_portfolio(
        market=market_adjusted,
        book=confirmation_book,
        raw_top3_paths={},
        policy=account.PolicySpec(name="fixed_d20", kind="fixed", fixed_day=20),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=int(min(confirmation_book.days)),
        last_signal_date_idx=int(max(confirmation_book.days)),
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=True,
        calendar_years=(2023, 2024, 2025),
        top_k=int(config["daily_ranked_candidate_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=capacity_fraction,
    )
    raw_metric, raw_equity, raw_trades, raw_annual = account.simulate_portfolio(
        market=market_raw,
        book=book,
        raw_top3_paths={},
        policy=account.PolicySpec(name="fixed_d20", kind="fixed", fixed_day=20),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=first_signal,
        last_signal_date_idx=last_signal,
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=True,
        calendar_years=tuple(int(value) for value in study["evaluation"]["years"]),
        top_k=int(config["daily_ranked_candidate_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=capacity_fraction,
    )
    risk_config = dict(study["risk_budget_diagnostic"])
    risk_budget = _derive_risk_budget(
        equity,
        starting_cash=float(config["starting_cash_cny"]),
        config=risk_config,
    )
    risk_fraction = float(risk_budget["frozen_target_gross_fraction"])
    development_end = int(
        np.searchsorted(
            np.asarray(pack.date_values, dtype=str),
            str(risk_config["derivation_end_date"]),
            side="right",
        )
        - 1
    )
    development_book = _book_until_date(book, development_end)
    (
        risk_development_metric,
        risk_development_equity,
        risk_development_trades,
        risk_development_annual,
    ) = account.simulate_portfolio(
        market=market_adjusted,
        book=development_book,
        raw_top3_paths={},
        policy=account.PolicySpec(name="fixed_d20", kind="fixed", fixed_day=20),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=int(min(development_book.days)),
        last_signal_date_idx=int(max(development_book.days)),
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=True,
        calendar_years=tuple(range(2014, 2023)),
        top_k=int(config["daily_ranked_candidate_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=capacity_fraction,
        target_gross_fraction=risk_fraction,
    )
    risk_metric, risk_equity, risk_trades, risk_annual = account.simulate_portfolio(
        market=market_adjusted,
        book=book,
        raw_top3_paths={},
        policy=account.PolicySpec(name="fixed_d20", kind="fixed", fixed_day=20),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=first_signal,
        last_signal_date_idx=last_signal,
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=True,
        calendar_years=tuple(int(value) for value in study["evaluation"]["years"]),
        top_k=int(config["daily_ranked_candidate_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=capacity_fraction,
        target_gross_fraction=risk_fraction,
    )
    (
        risk_confirmation_metric,
        risk_confirmation_equity,
        risk_confirmation_trades,
        risk_confirmation_annual,
    ) = account.simulate_portfolio(
        market=market_adjusted,
        book=confirmation_book,
        raw_top3_paths={},
        policy=account.PolicySpec(name="fixed_d20", kind="fixed", fixed_day=20),
        slots=int(config["position_slots"]),
        cost_scenario=str(config["cost_scenario"]),
        first_signal_date_idx=int(min(confirmation_book.days)),
        last_signal_date_idx=int(max(confirmation_book.days)),
        starting_cash=float(config["starting_cash_cny"]),
        allow_pyramiding=False,
        replace_rejected_from_ranked_candidates=True,
        calendar_years=(2023, 2024, 2025),
        top_k=int(config["daily_ranked_candidate_count"]),
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=capacity_fraction,
        target_gross_fraction=risk_fraction,
    )
    development_gate_config = dict(risk_config["development_gate"])
    confirmation_gate_config = dict(risk_config["confirmation_gate"])
    risk_development_gate = _risk_budget_gate(
        risk_development_metric,
        risk_development_annual,
        years=tuple(range(2014, 2023)),
        maximum_drawdown=float(development_gate_config["maximum_drawdown"]),
        maximum_volatility=float(
            development_gate_config["maximum_annualized_volatility"]
        ),
        minimum_positive_years=int(development_gate_config["minimum_positive_years"]),
    )
    risk_confirmation_gate = _risk_budget_gate(
        risk_confirmation_metric,
        risk_confirmation_annual,
        years=(2023, 2024, 2025),
        maximum_drawdown=float(confirmation_gate_config["maximum_drawdown"]),
        maximum_volatility=float(
            confirmation_gate_config["maximum_annualized_volatility"]
        ),
        minimum_positive_years=int(confirmation_gate_config["minimum_positive_years"]),
    )
    capacity = _capacity_audit(
        panel,
        trades,
        maximum_fraction=capacity_fraction,
    )
    if not capacity["all_orders_within_limit"]:
        raise ValueError("account_capacity_limit_exceeded")
    equity_path = output / "equity.parquet"
    trade_path = output / "trades.parquet"
    annual_path = output / "annual.parquet"
    audit_path = output / "book_audit.json"
    confirmation_equity_path = output / "confirmation_restart_equity.parquet"
    confirmation_trade_path = output / "confirmation_restart_trades.parquet"
    confirmation_annual_path = output / "confirmation_restart_annual.parquet"
    capacity_path = output / "capacity_audit.json"
    factor_audit_path = output / "adjust_factor_audit.json"
    raw_equity_path = output / "raw_price_sensitivity_equity.parquet"
    raw_trade_path = output / "raw_price_sensitivity_trades.parquet"
    raw_annual_path = output / "raw_price_sensitivity_annual.parquet"
    risk_development_equity_path = output / "risk_budget_development_equity.parquet"
    risk_development_trade_path = output / "risk_budget_development_trades.parquet"
    risk_development_annual_path = output / "risk_budget_development_annual.parquet"
    risk_equity_path = output / "risk_budget_equity.parquet"
    risk_trade_path = output / "risk_budget_trades.parquet"
    risk_annual_path = output / "risk_budget_annual.parquet"
    risk_confirmation_equity_path = output / "risk_budget_confirmation_equity.parquet"
    risk_confirmation_trade_path = output / "risk_budget_confirmation_trades.parquet"
    risk_confirmation_annual_path = output / "risk_budget_confirmation_annual.parquet"
    risk_budget_path = output / "risk_budget.json"
    _write_parquet(equity_path, equity)
    _write_parquet(trade_path, trades)
    _write_parquet(annual_path, pd.DataFrame(annual))
    _write_json(audit_path, book_audit)
    _write_parquet(confirmation_equity_path, confirmation_equity)
    _write_parquet(confirmation_trade_path, confirmation_trades)
    _write_parquet(confirmation_annual_path, pd.DataFrame(confirmation_annual))
    _write_json(capacity_path, capacity)
    _write_json(factor_audit_path, factor_audit)
    _write_parquet(raw_equity_path, raw_equity)
    _write_parquet(raw_trade_path, raw_trades)
    _write_parquet(raw_annual_path, pd.DataFrame(raw_annual))
    _write_parquet(risk_development_equity_path, risk_development_equity)
    _write_parquet(risk_development_trade_path, risk_development_trades)
    _write_parquet(risk_development_annual_path, pd.DataFrame(risk_development_annual))
    _write_parquet(risk_equity_path, risk_equity)
    _write_parquet(risk_trade_path, risk_trades)
    _write_parquet(risk_annual_path, pd.DataFrame(risk_annual))
    _write_parquet(risk_confirmation_equity_path, risk_confirmation_equity)
    _write_parquet(risk_confirmation_trade_path, risk_confirmation_trades)
    _write_parquet(
        risk_confirmation_annual_path, pd.DataFrame(risk_confirmation_annual)
    )
    _write_json(
        risk_budget_path,
        {
            **risk_budget,
            "development_gate": risk_development_gate,
            "confirmation_gate": risk_confirmation_gate,
        },
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_nonpromotional_falsification",
        "study_id": STUDY_ID,
        "study_sha256": study_sha256,
        "source_confirmation_gate_passed": False,
        "account_success_cannot_override_confirmation_failure": True,
        "metric": metric,
        "annual": annual,
        "confirmation_restart_metric": confirmation_metric,
        "confirmation_restart_annual": confirmation_annual,
        "raw_price_sensitivity_metric": raw_metric,
        "raw_price_sensitivity_annual": raw_annual,
        "risk_budget": risk_budget,
        "risk_budget_development_metric": risk_development_metric,
        "risk_budget_development_annual": risk_development_annual,
        "risk_budget_development_gate": risk_development_gate,
        "risk_budget_metric": risk_metric,
        "risk_budget_annual": risk_annual,
        "risk_budget_confirmation_metric": risk_confirmation_metric,
        "risk_budget_confirmation_annual": risk_confirmation_annual,
        "risk_budget_confirmation_gate": risk_confirmation_gate,
        "accounting_semantics": (
            "raw execution prices and costs; holding value and sale proceeds use "
            "the pack-pinned back-adjust-factor ratio as a total-return equivalent"
        ),
        "adjust_factor_audit": factor_audit,
        "capacity_audit": capacity,
        "maximum_signal_date_read": str(pack.date_values[last_signal]),
        "maximum_account_mark_date_read": str(pack.date_values[final_date_idx]),
        "forbidden_2026_read_count": 0,
        "account_optimization_performed": False,
        "production_policy_selected": False,
        "profit_claim_allowed": False,
        "files": {
            "equity": base._file_record(equity_path),
            "trades": base._file_record(trade_path),
            "annual": base._file_record(annual_path),
            "book_audit": base._file_record(audit_path),
            "confirmation_restart_equity": base._file_record(confirmation_equity_path),
            "confirmation_restart_trades": base._file_record(confirmation_trade_path),
            "confirmation_restart_annual": base._file_record(confirmation_annual_path),
            "capacity_audit": base._file_record(capacity_path),
            "adjust_factor_audit": base._file_record(factor_audit_path),
            "raw_price_sensitivity_equity": base._file_record(raw_equity_path),
            "raw_price_sensitivity_trades": base._file_record(raw_trade_path),
            "raw_price_sensitivity_annual": base._file_record(raw_annual_path),
            "risk_budget_development_equity": base._file_record(
                risk_development_equity_path
            ),
            "risk_budget_development_trades": base._file_record(
                risk_development_trade_path
            ),
            "risk_budget_development_annual": base._file_record(
                risk_development_annual_path
            ),
            "risk_budget_equity": base._file_record(risk_equity_path),
            "risk_budget_trades": base._file_record(risk_trade_path),
            "risk_budget_annual": base._file_record(risk_annual_path),
            "risk_budget_confirmation_equity": base._file_record(
                risk_confirmation_equity_path
            ),
            "risk_budget_confirmation_trades": base._file_record(
                risk_confirmation_trade_path
            ),
            "risk_budget_confirmation_annual": base._file_record(
                risk_confirmation_annual_path
            ),
            "risk_budget": base._file_record(risk_budget_path),
        },
    }
    _write_json(summary_path, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_account_feasibility(
        study_path=args.study, output_root=args.output_root, force=args.force
    )
    print(
        json.dumps(
            {
                "full_history": summary["metric"],
                "confirmation_restart": summary["confirmation_restart_metric"],
                "raw_price_sensitivity": summary["raw_price_sensitivity_metric"],
                "risk_budget": summary["risk_budget"],
                "risk_budget_full_history": summary["risk_budget_metric"],
                "risk_budget_confirmation": summary["risk_budget_confirmation_metric"],
                "risk_budget_gates": {
                    "development": summary["risk_budget_development_gate"],
                    "confirmation": summary["risk_budget_confirmation_gate"],
                },
                "capacity": summary["capacity_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
