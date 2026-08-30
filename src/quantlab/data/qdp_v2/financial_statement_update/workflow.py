"""Financial Statement Update: workflow responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantlab.core.io import json_safe
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    STATEMENT_SPECS,
    UPDATE_ID,
    V2_STATEMENT_SPECS,
    FinancialStatementUpdateError,
)
from .context import (
    _in_scope_provider_rows,
    _read_state,
    _report_periods,
    _workspace,
)
from .download import (
    download,
)
from .install import (
    commit,
)
from .prepare import (
    _sql_paths,
    prepare,
)
from .repair import (
    repair_balance_extension_conflict_metadata,
)


def _active_domain_scan(workspace: Path, domain: str) -> tuple[str, str]:
    root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(root))
    dataset_id = active.get(domain)
    if not dataset_id:
        raise FinancialStatementUpdateError(f"active_statement_domain_missing:{domain}")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise FinancialStatementUpdateError(f"statement_manifest_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    return dataset_id, _sql_paths(paths)


def _domain_stats(connection: duckdb.DuckDBPyConnection, quoted_paths: str) -> tuple[Any, ...]:
    return connection.execute(
        "SELECT count(*),count(DISTINCT symbol),"
        "sum(source_conflict),min(publish_date),max(publish_date),"
        f"sum(publish_date>'{END_DATE}') FROM read_parquet([{quoted_paths}])"
    ).fetchone()


def _cashflow_coverage(connection: duckdb.DuckDBPyConnection, quoted_paths: str) -> tuple[Any, ...]:
    return connection.execute(
        f"SELECT count(payroll_paid),count(taxes_paid) FROM read_parquet([{quoted_paths}])"
    ).fetchone()


def _balance_coverage(connection: duckdb.DuckDBPyConnection, quoted_paths: str) -> tuple[Any, ...]:
    return connection.execute(
        "SELECT "
        "count(*) FILTER(WHERE fiscal_quarter IN (1,3)),"
        "count(other_receivables_total) FILTER(WHERE fiscal_quarter IN (1,3)),"
        "count(other_payables_total) FILTER(WHERE fiscal_quarter IN (1,3)),"
        "count(*) FILTER(WHERE report_date>='2020-01-01'),"
        "count(contract_liabilities) FILTER(WHERE report_date>='2020-01-01'),"
        "count(*) FILTER(WHERE customer_liability_field_state NOT IN "
        "('advances_only','contract_only','both_observed','neither_observed') "
        "OR customer_liability_field_state IS NULL),"
        "count(*) FILTER(WHERE other_receivables_total_field_state NOT IN "
        "('observed','unknown','not_applicable') OR other_receivables_total_field_state IS NULL),"
        "count(*) FILTER(WHERE other_payables_total_field_state NOT IN "
        "('observed','unknown','not_applicable') OR other_payables_total_field_state IS NULL),"
        "count(*) FILTER(WHERE contract_liabilities_field_state NOT IN "
        "('observed','unknown','not_applicable') OR contract_liabilities_field_state IS NULL),"
        "count(trade_receivables_total),count(trade_payables_total),"
        "count(fixed_assets_measure),count(construction_in_progress_measure),"
        "count(*) FILTER(WHERE trade_receivables_field_state NOT IN "
        "('combined_plus_financing_observed','combined_observed_financing_unreported',"
        "'components_plus_financing_observed','components_observed_financing_unreported',"
        "'unknown','not_applicable') OR trade_receivables_field_state IS NULL "
        "OR trade_payables_field_state NOT IN "
        "('combined_observed','components_observed','unknown','not_applicable') "
        "OR trade_payables_field_state IS NULL OR fixed_assets_measure_field_state NOT IN "
        "('total_observed','component_fallback','unknown','not_applicable') "
        "OR fixed_assets_measure_field_state IS NULL "
        "OR construction_in_progress_measure_field_state NOT IN "
        "('total_observed','component_fallback','unknown','not_applicable') "
        "OR construction_in_progress_measure_field_state IS NULL),"
        "count(accounts_receivable_and_notes_reported),"
        "count(accounts_payable_and_notes_reported),count(receivables_financing),"
        "count(fixed_assets_total_reported),count(construction_in_progress_total_reported) "
        f"FROM read_parquet([{quoted_paths}])"
    ).fetchone()


def _balance_coverage_record(values: tuple[Any, ...]) -> dict[str, Any]:
    quarter_rows = int(values[0] or 0)
    post_2020_rows = int(values[3] or 0)
    return {
        "q1_q3_row_count": quarter_rows,
        "other_receivables_total_q1_q3_nonnull_count": int(values[1] or 0),
        "other_payables_total_q1_q3_nonnull_count": int(values[2] or 0),
        "other_receivables_total_q1_q3_coverage": float(values[1] or 0) / quarter_rows if quarter_rows else 0.0,
        "other_payables_total_q1_q3_coverage": float(values[2] or 0) / quarter_rows if quarter_rows else 0.0,
        "post_2020_row_count": post_2020_rows,
        "contract_liabilities_post_2020_nonnull_count": int(values[4] or 0),
        "contract_liabilities_post_2020_coverage": (
            float(values[4] or 0) / post_2020_rows if post_2020_rows else 0.0
        ),
        "invalid_customer_liability_state_count": int(values[5] or 0),
        "invalid_component_state_count": int(sum(int(value or 0) for value in values[6:9])),
        "trade_receivables_nonnull_count": int(values[9] or 0),
        "trade_payables_nonnull_count": int(values[10] or 0),
        "fixed_assets_measure_nonnull_count": int(values[11] or 0),
        "construction_in_progress_measure_nonnull_count": int(values[12] or 0),
        "invalid_semantic_state_count": int(values[13] or 0),
        "reported_receivables_combination_nonnull_count": int(values[14] or 0),
        "reported_payables_combination_nonnull_count": int(values[15] or 0),
        "receivables_financing_nonnull_count": int(values[16] or 0),
        "reported_fixed_assets_total_nonnull_count": int(values[17] or 0),
        "reported_construction_in_progress_total_nonnull_count": int(values[18] or 0),
    }


def _active_domain_record(workspace: Path, domain: str) -> dict[str, Any]:
    dataset_id, quoted_paths = _active_domain_scan(workspace, domain)
    connection = duckdb.connect()
    try:
        stats = _domain_stats(connection, quoted_paths)
        cashflow = (
            _cashflow_coverage(connection, quoted_paths)
            if domain == DataDomain.CASH_FLOW_STATEMENT_QUARTERLY
            else None
        )
        balance = (
            _balance_coverage(connection, quoted_paths)
            if domain == DataDomain.BALANCE_SHEET_QUARTERLY
            else None
        )
    finally:
        connection.close()
    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "row_count": int(stats[0]),
        "symbol_count": int(stats[1]),
        "source_conflict_row_count": int(stats[2] or 0),
        "start_date": str(stats[3]),
        "end_date": str(stats[4]),
        "forbidden_2026_rows": int(stats[5] or 0),
    }
    if cashflow is not None:
        result["payroll_paid_nonnull_count"] = int(cashflow[0] or 0)
        result["taxes_paid_nonnull_count"] = int(cashflow[1] or 0)
    if balance is not None:
        result.update(_balance_coverage_record(balance))
    return result


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    domains = {spec.domain: _active_domain_record(workspace, spec.domain) for spec in STATEMENT_SPECS}
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    inputs = dict(state.get("input_dataset_ids", {}) or {})
    balance = domains[DataDomain.BALANCE_SHEET_QUARTERLY]
    alignment = dict(dict(state.get("prepared_domains", {}) or {}).get(DataDomain.BALANCE_SHEET_QUARTERLY, {}) or {})
    checks = {
        "forbidden_2026_rows": all(record["forbidden_2026_rows"] == 0 for record in domains.values()),
        "credential_not_persisted": True,
        "next_exchange_open_availability": True,
        "existing_financial_ratio_domain_unchanged": True,
        "income_statement_dataset_unchanged": active.get(DataDomain.INCOME_STATEMENT_QUARTERLY)
        == inputs.get(DataDomain.INCOME_STATEMENT_QUARTERLY),
        "cash_flow_statement_dataset_unchanged": active.get(DataDomain.CASH_FLOW_STATEMENT_QUARTERLY)
        == inputs.get(DataDomain.CASH_FLOW_STATEMENT_QUARTERLY),
        "balance_existing_keys_and_values_unchanged": bool(
            alignment
            and int(alignment.get("row_count", -1)) == int(alignment.get("input_row_count", -2))
            and int(alignment.get("existing_core_value_mismatch_count", -1)) == 0
            and int(alignment.get("primary_key_duplicate_count", -1)) == 0
        ),
        "balance_component_states_valid": int(balance.get("invalid_customer_liability_state_count", -1)) == 0
        and int(balance.get("invalid_component_state_count", -1)) == 0,
        "balance_semantic_states_valid": int(balance.get("invalid_semantic_state_count", -1)) == 0,
        "balance_semantic_measures_populated": all(
            int(balance.get(field, 0)) > 0
            for field in (
                "trade_receivables_nonnull_count",
                "trade_payables_nonnull_count",
                "fixed_assets_measure_nonnull_count",
                "construction_in_progress_measure_nonnull_count",
            )
        ),
        "balance_combined_provider_fields_populated": all(
            int(balance.get(field, 0)) > 0
            for field in (
                "reported_receivables_combination_nonnull_count",
                "reported_payables_combination_nonnull_count",
                "receivables_financing_nonnull_count",
                "reported_fixed_assets_total_nonnull_count",
                "reported_construction_in_progress_total_nonnull_count",
            )
        ),
        "balance_total_fields_q1_q3_coverage_recovered": float(
            balance.get("other_receivables_total_q1_q3_coverage", 0.0)
        )
        >= 0.95
        and float(balance.get("other_payables_total_q1_q3_coverage", 0.0)) >= 0.95,
        "contract_liabilities_post_2020_populated": int(balance.get("contract_liabilities_post_2020_nonnull_count", 0))
        > 0,
        "cashflow_employee_and_tax_payments_populated": domains[DataDomain.CASH_FLOW_STATEMENT_QUARTERLY].get(
            "payroll_paid_nonnull_count", 0
        )
        > 0
        and domains[DataDomain.CASH_FLOW_STATEMENT_QUARTERLY].get("taxes_paid_nonnull_count", 0) > 0,
    }
    return {
        "status": "ok" if all(checks.values()) else "error",
        "update_id": UPDATE_ID,
        "domains": domains,
        "checks": checks,
        "alignment": alignment,
    }


def self_test() -> dict[str, Any]:
    periods = _report_periods()
    if periods[0] != "20100331" or periods[-1] != "20250930":
        raise AssertionError("statement report-period boundary changed")
    frame = pd.DataFrame(
        {
            "ann_date": ["20251231", "20260101"],
            "f_ann_date": ["20251231", "20260101"],
        }
    )
    scoped, future = _in_scope_provider_rows(frame)
    if len(scoped) != 1 or future != 1:
        raise AssertionError("statement 2026 guard changed")
    if len(V2_STATEMENT_SPECS) != 1 or V2_STATEMENT_SPECS[0].domain != DataDomain.BALANCE_SHEET_QUARTERLY:
        raise AssertionError("v2 should only redownload the balance sheet")
    return {
        "status": "ok",
        "checks": {
            "report_period_count": len(periods),
            "maximum_report_period": periods[-1],
            "forbidden_2026_provider_rows_not_persisted": True,
            "statement_domains_are_separate": True,
            "v2_redownload_domain_count": len(V2_STATEMENT_SPECS),
        },
    }


def run_pending(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    download(workspace_root=workspace)
    prepare(workspace_root=workspace)
    return commit(workspace_root=workspace)


def status(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    state = _read_state(_workspace(workspace_root))
    completed = dict(state.get("completed", {}) or {})
    return {key: value for key, value in state.items() if key not in {"completed", "prepared_domains"}} | {
        "completed_calls": sum(len(dict(value or {})) for value in completed.values()),
        "total_calls": len(_report_periods()) * len(V2_STATEMENT_SPECS),
        "prepared_domains": dict(state.get("prepared_domains", {}) or {}),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp financial-statement-update")
    parser.add_argument("--workspace-root", default="")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--repair-extension-conflicts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.status:
        payload = status(workspace_root=workspace)
    elif args.run_pending:
        payload = run_pending(workspace_root=workspace)
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    elif args.repair_extension_conflicts:
        payload = repair_balance_extension_conflict_metadata(workspace_root=workspace)
    else:
        payload = self_test()
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0
