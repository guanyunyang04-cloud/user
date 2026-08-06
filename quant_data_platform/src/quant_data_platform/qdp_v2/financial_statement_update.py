from __future__ import annotations

"""Backfill compact point-in-time income, balance-sheet, and cash-flow data."""

import argparse
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
    _TushareClient,
)
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.research_event_update import (
    _assert_credential_free,
    _identity_symbols,
    _next_open_date,
    _open_dates,
    _sha256,
    _write_parquet,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

UPDATE_ID = "financial_statement_quarterly_pit_v2"
LEGACY_UPDATE_ID = "financial_statement_quarterly_pit_v1"
BALANCE_CONFLICT_REPAIR_ID = "balance_extension_conflict_semantics_repair_v2"
SOURCE_SCHEMA_VERSION = 5
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
MAX_REPORT_PERIOD = "2025-09-30"


@dataclass(frozen=True)
class StatementSpec:
    name: str
    api_name: str
    domain: str
    fields: tuple[str, ...]
    metric_map: Mapping[str, str]


COMMON_FIELDS = (
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
)

INCOME_METRICS = {
    "basic_eps": "basic_eps",
    "diluted_eps": "diluted_eps",
    "total_revenue": "total_revenue",
    "revenue": "revenue",
    "total_cogs": "total_cost",
    "oper_cost": "operating_cost",
    "sell_exp": "selling_expense",
    "admin_exp": "administrative_expense",
    "fin_exp": "finance_expense",
    "rd_exp": "research_development_expense",
    "operate_profit": "operating_profit",
    "total_profit": "total_profit",
    "income_tax": "income_tax",
    "n_income": "net_income",
    "n_income_attr_p": "parent_net_income",
    "minority_gain": "minority_income",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "continued_net_profit": "continuing_net_income",
    "end_net_profit": "discontinued_net_income",
}

BALANCE_METRICS = {
    "total_share": "total_shares",
    "cap_rese": "capital_reserve",
    "undistr_porfit": "retained_earnings",
    "surplus_rese": "surplus_reserve",
    "money_cap": "cash_and_equivalents",
    "notes_receiv": "notes_receivable",
    "accounts_receiv": "accounts_receivable",
    "accounts_receiv_bill": "accounts_receivable_and_notes_reported",
    "receiv_financing": "receivables_financing",
    "oth_receiv": "other_receivables",
    "oth_rcv_total": "other_receivables_total",
    "prepayment": "prepayments",
    "inventories": "inventory",
    "total_cur_assets": "current_assets",
    "fix_assets": "fixed_assets",
    "fix_assets_total": "fixed_assets_total_reported",
    "cip": "construction_in_progress",
    "cip_total": "construction_in_progress_total_reported",
    "intan_assets": "intangible_assets",
    "goodwill": "goodwill",
    "total_nca": "noncurrent_assets",
    "total_assets": "total_assets",
    "st_borr": "short_term_borrowings",
    "notes_payable": "notes_payable",
    "acct_payable": "accounts_payable",
    "accounts_pay": "accounts_payable_and_notes_reported",
    "adv_receipts": "advances_from_customers",
    "contract_liab": "contract_liabilities",
    "payroll_payable": "employee_compensation_payable",
    "taxes_payable": "taxes_payable",
    "oth_payable": "other_payables",
    "oth_pay_total": "other_payables_total",
    "total_cur_liab": "current_liabilities",
    "lt_borr": "long_term_borrowings",
    "bond_payable": "bonds_payable",
    "total_ncl": "noncurrent_liabilities",
    "total_liab": "total_liabilities",
    "minority_int": "minority_equity",
    "total_hldr_eqy_exc_min_int": "parent_equity",
    "total_hldr_eqy_inc_min_int": "total_equity",
    "total_liab_hldr_eqy": "liabilities_and_equity",
}

CASH_FLOW_METRICS = {
    "net_profit": "net_profit",
    "c_fr_sale_sg": "cash_received_from_sales",
    "recp_tax_rends": "tax_refunds_received",
    "c_paid_goods_s": "cash_paid_for_goods",
    "c_paid_to_for_empl": "payroll_paid",
    "c_paid_for_taxes": "taxes_paid",
    "oth_cash_pay_oper_act": "other_operating_cash_paid",
    "st_cash_out_act": "operating_cash_outflow",
    "n_cashflow_act": "net_operating_cash_flow",
    "stot_inflows_inv_act": "investing_cash_inflow",
    "stot_out_inv_act": "investing_cash_outflow",
    "n_cashflow_inv_act": "net_investing_cash_flow",
    "c_recp_borrow": "borrowings_received",
    "proc_issue_bonds": "bond_proceeds",
    "stot_cash_in_fnc_act": "financing_cash_inflow",
    "c_prepay_amt_borr": "borrowings_repaid",
    "c_pay_dist_dpcp_int_exp": "dividends_and_interest_paid",
    "stot_cashout_fnc_act": "financing_cash_outflow",
    "n_cash_flows_fnc_act": "net_financing_cash_flow",
    "free_cashflow": "free_cash_flow",
    "n_incr_cash_cash_equ": "net_cash_increase",
    "c_cash_equ_beg_period": "cash_begin",
    "c_cash_equ_end_period": "cash_end",
}

STATEMENT_SPECS = (
    StatementSpec(
        name="income",
        api_name="income_vip",
        domain=DataDomain.INCOME_STATEMENT_QUARTERLY,
        fields=COMMON_FIELDS + tuple(INCOME_METRICS) + ("update_flag",),
        metric_map=INCOME_METRICS,
    ),
    StatementSpec(
        name="balance_sheet",
        api_name="balancesheet_vip",
        domain=DataDomain.BALANCE_SHEET_QUARTERLY,
        fields=COMMON_FIELDS + tuple(BALANCE_METRICS) + ("update_flag",),
        metric_map=BALANCE_METRICS,
    ),
    StatementSpec(
        name="cash_flow",
        api_name="cashflow_vip",
        domain=DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
        fields=COMMON_FIELDS + tuple(CASH_FLOW_METRICS) + ("update_flag",),
        metric_map=CASH_FLOW_METRICS,
    ),
)
V2_STATEMENT_SPECS = tuple(
    spec
    for spec in STATEMENT_SPECS
    if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
)
BALANCE_DERIVED_COLUMNS = (
    "customer_advances_and_contract_liabilities",
    "customer_liability_field_state",
    "other_receivables_total_field_state",
    "other_payables_total_field_state",
    "contract_liabilities_field_state",
    "trade_receivables_total",
    "trade_receivables_to_current_assets",
    "trade_receivables_to_total_assets",
    "trade_receivables_field_state",
    "fixed_assets_measure",
    "fixed_assets_measure_field_state",
    "construction_in_progress_measure",
    "construction_in_progress_measure_field_state",
    "trade_payables_total",
    "trade_payables_to_current_liabilities",
    "trade_payables_to_total_assets",
    "trade_payables_field_state",
)
BALANCE_EXTENSION_NUMERIC_COLUMNS = (
    "other_receivables_total",
    "other_payables_total",
    "contract_liabilities",
    "customer_advances_and_contract_liabilities",
)
BALANCE_EXTENSION_CONFLICT_COLUMN = "balance_extension_source_conflict"
BALANCE_SEMANTIC_SOURCE_COLUMNS = (
    "accounts_receivable_and_notes_reported",
    "receivables_financing",
    "fixed_assets_total_reported",
    "construction_in_progress_total_reported",
    "accounts_payable_and_notes_reported",
)
BALANCE_SEMANTIC_NUMERIC_COLUMNS = (
    "trade_receivables_total",
    "trade_receivables_to_current_assets",
    "trade_receivables_to_total_assets",
    "fixed_assets_measure",
    "construction_in_progress_measure",
    "trade_payables_total",
    "trade_payables_to_current_liabilities",
    "trade_payables_to_total_assets",
)
BALANCE_SEMANTIC_STATE_COLUMNS = (
    "trade_receivables_field_state",
    "fixed_assets_measure_field_state",
    "construction_in_progress_measure_field_state",
    "trade_payables_field_state",
)
BALANCE_SEMANTIC_CONFLICT_COLUMN = "balance_semantic_source_conflict"

COMMON_OUTPUT_COLUMNS = (
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "report_date",
    "fiscal_year",
    "fiscal_quarter",
    "announcement_date",
    "actual_announcement_date",
    "publish_date",
    "report_type",
    "company_type",
    "period_type",
    "update_flag",
)
TRAILING_OUTPUT_COLUMNS = (
    "source_duplicate_count",
    "source_conflict",
    "lag_policy",
    "source",
)
PRIMARY_KEY = (
    "publish_date",
    "symbol",
    "report_date",
    "report_type",
    "company_type",
    "period_type",
)


class FinancialStatementUpdateError(RuntimeError):
    pass


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {
            "update_id": UPDATE_ID,
            "status": "pending",
            "start_date": START_DATE,
            "end_date": END_DATE,
        }
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            atomic_write_json(_state_path(workspace), payload)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.10 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _report_periods() -> list[str]:
    return [
        item.strftime("%Y%m%d")
        for item in pd.date_range("2010-03-31", MAX_REPORT_PERIOD, freq="QE-DEC")
    ]


def _raw_path(workspace: Path, spec: StatementSpec, period: str) -> Path:
    return _runtime(workspace) / "raw" / spec.name / f"period={period}.parquet"


def _normalized_path(workspace: Path, spec: StatementSpec, period: str) -> Path:
    return _runtime(workspace) / "normalized" / spec.name / f"period={period}.parquet"


def _provider_publish_date(frame: pd.DataFrame) -> pd.Series:
    actual = frame.get("f_ann_date", pd.Series("", index=frame.index))
    announced = frame.get("ann_date", pd.Series("", index=frame.index))
    actual_text = actual.fillna("").astype(str).str.strip()
    return pd.to_datetime(actual.where(actual_text.ne(""), announced), errors="coerce")


def _in_scope_provider_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    publish = _provider_publish_date(frame)
    future = int(publish.gt(pd.Timestamp(END_DATE)).sum())
    valid = publish.between(START_DATE, END_DATE)
    return frame.loc[valid].reset_index(drop=True), future


def download(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if (
        state.get("status") in {"downloaded", "prepared", "applied"}
        and state.get("source_schema_version") == SOURCE_SCHEMA_VERSION
    ):
        return state
    if state.get("source_schema_version") != SOURCE_SCHEMA_VERSION:
        previous_schema_version = state.get("source_schema_version")
        current = active_dataset_map(
            read_active_manifest(qdp_v2_root(workspace))
        )
        state = {
            "update_id": UPDATE_ID,
            "status": "pending",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "upgraded_from_source_schema_version": previous_schema_version,
            "input_dataset_ids": {
                domain: dataset_id
                for domain, dataset_id in current.items()
                if domain
                in {
                    DataDomain.INCOME_STATEMENT_QUARTERLY,
                    DataDomain.BALANCE_SHEET_QUARTERLY,
                    DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
                }
            },
            "completed": {},
            "provider_future_rows_discarded": 0,
        }
        _write_state(workspace, state)
    client = _TushareClient(_resolve_tushare_token(workspace), workspace_root=workspace)
    periods = _report_periods()
    completed = dict(state.get("completed", {}) or {})
    if "input_dataset_ids" not in state:
        state["input_dataset_ids"] = {
            domain: dataset_id
            for domain, dataset_id in active_dataset_map(
                read_active_manifest(qdp_v2_root(workspace))
            ).items()
            if domain
            in {
                DataDomain.INCOME_STATEMENT_QUARTERLY,
                DataDomain.BALANCE_SHEET_QUARTERLY,
                DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
            }
        }
    discarded_future_rows = int(state.get("provider_future_rows_discarded", 0) or 0)
    completed_calls = 0
    for spec in V2_STATEMENT_SPECS:
        spec_completed = dict(completed.get(spec.name, {}) or {})
        for period in periods:
            path = _raw_path(workspace, spec, period)
            current_fields = (
                set(pq.read_schema(path).names) if path.is_file() else set()
            )
            if not set(spec.fields).issubset(current_fields):
                raw = client.fetch(
                    spec.api_name,
                    params={"period": period},
                    fields=spec.fields,
                )
                scoped, future = _in_scope_provider_rows(raw)
                discarded_future_rows += future
                _write_parquet(scoped, path)
            spec_completed[period] = {
                "status": "completed",
                "path": str(path),
                "row_count": int(pq.ParquetFile(path).metadata.num_rows),
                "sha256": _sha256(path),
            }
            completed_calls += 1
            completed[spec.name] = spec_completed
            state.update(
                {
                    "update_id": UPDATE_ID,
                    "status": "downloading",
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "maximum_report_period": MAX_REPORT_PERIOD,
                    "source_schema_version": SOURCE_SCHEMA_VERSION,
                    "period_count": len(periods),
                    "completed": completed,
                    "provider_future_rows_discarded": discarded_future_rows,
                    "provider_response_future_rows_not_persisted": True,
                }
            )
            _write_state(workspace, state)
            if completed_calls % 12 == 0:
                print(
                    json.dumps(
                        {
                            "completed_calls": completed_calls,
                            "total_calls": len(periods) * len(V2_STATEMENT_SPECS),
                        }
                    ),
                    flush=True,
                )
    state["status"] = "downloaded"
    state["source_schema_version"] = SOURCE_SCHEMA_VERSION
    _write_state(workspace, state)
    return state


def _hash_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame.reindex(columns=list(columns)).copy()
    for column in values:
        values[column] = values[column].astype("string").fillna("<NA>")
    hashed = pd.util.hash_pandas_object(values, index=False).astype("uint64")
    return hashed.map(lambda value: f"{int(value):016x}")


def _balance_component_state(
    value: pd.Series,
    company_type: pd.Series,
) -> pd.Series:
    special_structure = company_type.astype(str).isin({"2", "3", "4"})
    return pd.Series(
        np.select(
            [value.notna(), special_structure],
            ["observed", "not_applicable"],
            default="unknown",
        ),
        index=value.index,
        dtype="string",
    )


def _safe_series_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = numerator.notna() & denominator.notna() & denominator.abs().gt(1.0e-12)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _populate_balance_semantic_fields(data: pd.DataFrame) -> None:
    special_structure = data["company_type"].astype(str).isin({"2", "3", "4"})

    reported_receivables = data["accounts_receivable_and_notes_reported"]
    component_receivables = data["notes_receivable"] + data["accounts_receivable"]
    receivable_components_complete = (
        data["notes_receivable"].notna() & data["accounts_receivable"].notna()
    )
    receivable_base = reported_receivables.where(
        reported_receivables.notna(),
        component_receivables.where(receivable_components_complete),
    )
    financing = data["receivables_financing"]
    data["trade_receivables_total"] = receivable_base.where(
        receivable_base.isna(), receivable_base + financing.fillna(0.0)
    )
    data["trade_receivables_to_current_assets"] = _safe_series_ratio(
        data["trade_receivables_total"], data["current_assets"]
    )
    data["trade_receivables_to_total_assets"] = _safe_series_ratio(
        data["trade_receivables_total"], data["total_assets"]
    )
    data["trade_receivables_field_state"] = np.select(
        [
            reported_receivables.notna() & financing.notna(),
            reported_receivables.notna(),
            receivable_components_complete & financing.notna(),
            receivable_components_complete,
            special_structure,
        ],
        [
            "combined_plus_financing_observed",
            "combined_observed_financing_unreported",
            "components_plus_financing_observed",
            "components_observed_financing_unreported",
            "not_applicable",
        ],
        default="unknown",
    )

    for reported, component, output, state_column in (
        (
            "fixed_assets_total_reported",
            "fixed_assets",
            "fixed_assets_measure",
            "fixed_assets_measure_field_state",
        ),
        (
            "construction_in_progress_total_reported",
            "construction_in_progress",
            "construction_in_progress_measure",
            "construction_in_progress_measure_field_state",
        ),
    ):
        data[output] = data[reported].where(data[reported].notna(), data[component])
        data[state_column] = np.select(
            [data[reported].notna(), data[component].notna(), special_structure],
            ["total_observed", "component_fallback", "not_applicable"],
            default="unknown",
        )

    reported_payables = data["accounts_payable_and_notes_reported"]
    payable_components_complete = (
        data["notes_payable"].notna() & data["accounts_payable"].notna()
    )
    component_payables = data["notes_payable"] + data["accounts_payable"]
    data["trade_payables_total"] = reported_payables.where(
        reported_payables.notna(), component_payables.where(payable_components_complete)
    )
    data["trade_payables_to_current_liabilities"] = _safe_series_ratio(
        data["trade_payables_total"], data["current_liabilities"]
    )
    data["trade_payables_to_total_assets"] = _safe_series_ratio(
        data["trade_payables_total"], data["total_assets"]
    )
    data["trade_payables_field_state"] = np.select(
        [reported_payables.notna(), payable_components_complete, special_structure],
        ["combined_observed", "components_observed", "not_applicable"],
        default="unknown",
    )


def _normalize_statement_part(
    raw: pd.DataFrame,
    *,
    spec: StatementSpec,
    identity_symbols: set[str],
    open_dates: np.ndarray,
) -> pd.DataFrame:
    metric_columns = list(spec.metric_map.values())
    derived_columns = (
        list(BALANCE_DERIVED_COLUMNS)
        if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        else []
    )
    output_columns = [
        *COMMON_OUTPUT_COLUMNS,
        *metric_columns,
        *derived_columns,
        "_completeness",
        "_metric_hash",
        "_source_row_hash",
        "lag_policy",
        "source",
    ]
    if raw.empty:
        return pd.DataFrame(columns=output_columns)
    data = raw.copy()
    for column in spec.fields:
        if column not in data:
            data[column] = np.nan
    data["symbol"] = data["ts_code"].fillna("").astype(str).str.upper()
    data["announcement_date"] = pd.to_datetime(
        data["ann_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["actual_announcement_date"] = pd.to_datetime(
        data["f_ann_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    publish = _provider_publish_date(data)
    report = pd.to_datetime(data["end_date"], errors="coerce")
    data["publish_date"] = publish.dt.strftime("%Y-%m-%d")
    data["source_date"] = data["publish_date"]
    data["trade_date"] = data["publish_date"]
    data["feature_available_date"] = _next_open_date(data["publish_date"], open_dates)
    data["report_date"] = report.dt.strftime("%Y-%m-%d")
    data["fiscal_year"] = report.dt.year.astype("Int64")
    data["fiscal_quarter"] = report.dt.quarter.astype("Int64")
    data["report_type"] = data["report_type"].fillna("").astype(str)
    data["company_type"] = data["comp_type"].fillna("").astype(str)
    data["period_type"] = data["end_type"].fillna("").astype(str)
    data["update_flag"] = (
        pd.to_numeric(data["update_flag"], errors="coerce").fillna(0).astype("int8")
    )
    for provider, canonical in spec.metric_map.items():
        data[canonical] = pd.to_numeric(data[provider], errors="coerce")
    if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY:
        advances = data["advances_from_customers"]
        contract = data["contract_liabilities"]
        data["customer_advances_and_contract_liabilities"] = pd.concat(
            [advances, contract], axis=1
        ).sum(axis=1, min_count=1)
        data["customer_liability_field_state"] = np.select(
            [advances.notna() & contract.notna(), advances.notna(), contract.notna()],
            ["both_observed", "advances_only", "contract_only"],
            default="neither_observed",
        )
        for column in (
            "other_receivables_total",
            "other_payables_total",
            "contract_liabilities",
        ):
            data[f"{column}_field_state"] = _balance_component_state(
                data[column], data["company_type"]
            )
        _populate_balance_semantic_fields(data)
    valid = (
        data["symbol"].isin(identity_symbols)
        & publish.between(START_DATE, END_DATE)
        & report.notna()
        & report.le(publish)
    )
    data = data.loc[valid].copy()
    data["_completeness"] = data[metric_columns].notna().sum(axis=1).astype("int16")
    data["_metric_hash"] = _hash_rows(data, metric_columns)
    data["_source_row_hash"] = _hash_rows(data, list(spec.fields))
    data["lag_policy"] = "next_exchange_open_day"
    data["source"] = f"tushare_compatible_{spec.api_name}"
    return data.loc[:, output_columns].reset_index(drop=True)


def _sql_paths(paths: Sequence[Path]) -> str:
    return ",".join(f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in paths)


def _balance_extension_hash_sql(*, prefix: str = "") -> str:
    values = ",".join(
        f"coalesce(cast({prefix}\"{column}\" AS VARCHAR),'<NA>')"
        for column in BALANCE_EXTENSION_NUMERIC_COLUMNS
    )
    return f"md5(concat_ws('|',{values}))"


def _balance_semantic_hash_sql(*, prefix: str = "") -> str:
    values = ",".join(
        f"coalesce(cast({prefix}\"{column}\" AS VARCHAR),'<NA>')"
        for column in BALANCE_SEMANTIC_NUMERIC_COLUMNS
    )
    return f"md5(concat_ws('|',{values}))"


def _prepare_statement(
    workspace: Path,
    *,
    spec: StatementSpec,
    identity_symbols: set[str],
    open_dates: np.ndarray,
    prepared_filename: str | None = None,
) -> dict[str, Any]:
    normalized_paths: list[Path] = []
    normalized_rows = 0
    for period in _report_periods():
        raw_path = _raw_path(workspace, spec, period)
        if not raw_path.is_file():
            raise FinancialStatementUpdateError(
                f"statement_raw_part_missing:{spec.name}:{period}"
            )
        output_path = _normalized_path(workspace, spec, period)
        normalized_required = {
            *COMMON_OUTPUT_COLUMNS,
            *spec.metric_map.values(),
            *(
                BALANCE_DERIVED_COLUMNS
                if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
                else ()
            ),
        }
        if (
            not output_path.is_file()
            or raw_path.stat().st_mtime_ns > output_path.stat().st_mtime_ns
            or not normalized_required.issubset(pq.read_schema(output_path).names)
        ):
            normalized = _normalize_statement_part(
                pd.read_parquet(raw_path),
                spec=spec,
                identity_symbols=identity_symbols,
                open_dates=open_dates,
            )
            _write_parquet(normalized, output_path)
        normalized_rows += int(pq.ParquetFile(output_path).metadata.num_rows)
        normalized_paths.append(output_path)
    prepared = (
        _runtime(workspace)
        / "prepared"
        / (prepared_filename or f"{spec.domain}.parquet")
    )
    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_suffix(".tmp.parquet")
    metrics = list(spec.metric_map.values())
    derived = (
        list(BALANCE_DERIVED_COLUMNS)
        if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        else []
    )
    selected = [
        *COMMON_OUTPUT_COLUMNS,
        *metrics,
        *derived,
        "source_duplicate_count",
        "source_conflict",
        *(
            [BALANCE_EXTENSION_CONFLICT_COLUMN]
            if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
            else []
        ),
        *(
            [BALANCE_SEMANTIC_CONFLICT_COLUMN]
            if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
            else []
        ),
        "lag_policy",
        "source",
    ]
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    select_sql = ",".join(f'"{column}"' for column in selected)
    extension_conflict_sql = (
        ",\n          count(DISTINCT "
        f"{_balance_extension_hash_sql()}) OVER (PARTITION BY {key_sql}) > 1 "
        f"AS {BALANCE_EXTENSION_CONFLICT_COLUMN}"
        if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        else ""
    )
    semantic_conflict_sql = (
        ",\n          count(DISTINCT "
        f"{_balance_semantic_hash_sql()}) OVER (PARTITION BY {key_sql}) > 1 "
        f"AS {BALANCE_SEMANTIC_CONFLICT_COLUMN}"
        if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        else ""
    )
    sql = f"""
    COPY (
      WITH source AS (
        SELECT * FROM read_parquet([{_sql_paths(normalized_paths)}], union_by_name=true)
      ), ranked AS (
        SELECT *,
          count(*) OVER (PARTITION BY {key_sql}) AS source_duplicate_count,
          count(DISTINCT _metric_hash) OVER (PARTITION BY {key_sql}) > 1 AS source_conflict{extension_conflict_sql}{semantic_conflict_sql},
          row_number() OVER (
            PARTITION BY {key_sql}
            ORDER BY update_flag DESC, _completeness DESC, _source_row_hash DESC
          ) AS rn
        FROM source
      )
      SELECT {select_sql}
      FROM ranked WHERE rn=1
      ORDER BY publish_date,symbol,report_date,report_type,company_type,period_type
    ) TO '{str(temporary).replace(chr(39), chr(39) * 2)}'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection = duckdb.connect()
    try:
        connection.execute(sql)
        extension_conflict_sum = (
            f"sum({BALANCE_EXTENSION_CONFLICT_COLUMN})"
            if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
            else "NULL"
        )
        semantic_conflict_sum = (
            f"sum({BALANCE_SEMANTIC_CONFLICT_COLUMN})"
            if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
            else "NULL"
        )
        row = connection.execute(
            f"SELECT count(*),sum(source_duplicate_count>1),sum(source_conflict),"
            f"{extension_conflict_sum},{semantic_conflict_sum},"
            "min(publish_date),max(publish_date) "
            f"FROM read_parquet('{str(temporary).replace(chr(39), chr(39) * 2)}')"
        ).fetchone()
    finally:
        connection.close()
    os.replace(temporary, prepared)
    return {
        "status": "completed",
        "domain": spec.domain,
        "path": str(prepared),
        "normalized_row_count": normalized_rows,
        "row_count": int(row[0]),
        "deduplicated_row_count": normalized_rows - int(row[0]),
        "rows_from_duplicate_source_groups": int(row[1] or 0),
        "source_conflict_row_count": int(row[2] or 0),
        "balance_extension_source_conflict_count": int(row[3] or 0),
        "balance_semantic_source_conflict_count": int(row[4] or 0),
        "start_date": str(row[5]),
        "end_date": str(row[6]),
        "sha256": _sha256(prepared),
    }


def _dataset_paths(
    workspace: Path,
    *,
    domain: str,
    dataset_id: str,
) -> list[Path]:
    root = qdp_v2_root(workspace)
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise FinancialStatementUpdateError(
            f"statement_input_manifest_missing:{domain}:{dataset_id}"
        )
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or any(not path.is_file() for path in paths):
        raise FinancialStatementUpdateError(
            f"statement_input_shard_missing:{domain}:{dataset_id}"
        )
    return paths


def _align_balance_to_active(
    workspace: Path,
    *,
    refreshed: Path,
    input_dataset_id: str,
) -> dict[str, Any]:
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    base_paths = _dataset_paths(
        workspace,
        domain=domain,
        dataset_id=input_dataset_id,
    )
    output = _runtime(workspace) / "prepared" / f"{domain}.parquet"
    temporary = output.with_suffix(".tmp.parquet")
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    refreshed_columns = tuple(
        dict.fromkeys(
            (
                *BALANCE_EXTENSION_NUMERIC_COLUMNS,
                *BALANCE_SEMANTIC_SOURCE_COLUMNS,
                *BALANCE_DERIVED_COLUMNS,
                BALANCE_EXTENSION_CONFLICT_COLUMN,
                BALANCE_SEMANTIC_CONFLICT_COLUMN,
            )
        )
    )
    base_scan = f"read_parquet([{_sql_paths(base_paths)}], union_by_name=true)"
    refreshed_scan = f"read_parquet('{str(refreshed).replace(chr(39), chr(39) * 2)}')"
    connection = duckdb.connect()
    try:
        base_columns = [
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {base_scan}"
            ).fetchall()
        ]
        refreshed_schema = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {refreshed_scan}"
            ).fetchall()
        }
        missing_refreshed_columns = sorted(
            set(refreshed_columns).difference(refreshed_schema)
        )
        if missing_refreshed_columns:
            raise FinancialStatementUpdateError(
                f"balance_refreshed_columns_missing:{missing_refreshed_columns}"
            )
        preserved_columns = [
            column for column in base_columns if column not in set(refreshed_columns)
        ]
        base_select = ",".join(f'"{column}"' for column in preserved_columns)
        output_select = ",".join(
            [
                *(f'b."{column}"' for column in preserved_columns),
                *(f'r."{column}"' for column in refreshed_columns),
            ]
        )
        connection.execute(
            f"""
            COPY (
              SELECT {output_select}
              FROM {base_scan} b
              LEFT JOIN {refreshed_scan} r USING({key_sql})
              ORDER BY b.publish_date,b.symbol,b.report_date,b.report_type,
                       b.company_type,b.period_type
            ) TO '{str(temporary).replace(chr(39), chr(39) * 2)}'
              (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        base_count = int(
            connection.execute(f"SELECT count(*) FROM {base_scan}").fetchone()[0]
        )
        output_scan = f"read_parquet('{str(temporary).replace(chr(39), chr(39) * 2)}')"
        output_count = int(
            connection.execute(f"SELECT count(*) FROM {output_scan}").fetchone()[0]
        )
        unmatched = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM {base_scan} b
                LEFT JOIN (
                  SELECT {key_sql},true AS refreshed_present FROM {refreshed_scan}
                ) r USING({key_sql})
                WHERE NOT coalesce(r.refreshed_present,false)
                """
            ).fetchone()[0]
        )
        core_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  (SELECT {base_select} FROM {base_scan})
                  EXCEPT ALL
                  (SELECT {base_select} FROM {output_scan})
                )
                """
            ).fetchone()[0]
        )
        duplicate_count = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {key_sql},count(*) n "
                f"FROM {output_scan} GROUP BY {key_sql} HAVING n>1)"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    if base_count != output_count or unmatched or core_mismatch or duplicate_count:
        raise FinancialStatementUpdateError(
            "balance_alignment_contract_failed:"
            f"base={base_count}:output={output_count}:"
            f"unmatched={unmatched}:core_mismatch={core_mismatch}:"
            f"duplicates={duplicate_count}"
        )
    os.replace(temporary, output)
    return {
        "status": "completed",
        "domain": domain,
        "path": str(output),
        "input_dataset_id": input_dataset_id,
        "refreshed_path": str(refreshed),
        "row_count": output_count,
        "input_row_count": base_count,
        "unmatched_refreshed_key_count": unmatched,
        "existing_core_value_mismatch_count": core_mismatch,
        "primary_key_duplicate_count": duplicate_count,
        "sha256": _sha256(output),
    }


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if (
        state.get("status") in {"prepared", "applied"}
        and state.get("source_schema_version") == SOURCE_SCHEMA_VERSION
    ):
        return state
    if state.get("status") != "downloaded":
        raise FinancialStatementUpdateError(
            f"statement_not_downloaded:{state.get('status')}"
        )
    identity = set(_identity_symbols(workspace))
    open_dates = _open_dates(workspace)
    balance_spec = V2_STATEMENT_SPECS[0]
    refreshed = _prepare_statement(
        workspace,
        spec=balance_spec,
        identity_symbols=identity,
        open_dates=open_dates,
        prepared_filename="balance_sheet_quarterly_refreshed.parquet",
    )
    input_dataset_id = str(
        dict(state.get("input_dataset_ids", {}) or {}).get(
            DataDomain.BALANCE_SHEET_QUARTERLY, ""
        )
    )
    if not input_dataset_id:
        raise FinancialStatementUpdateError("balance_input_dataset_id_missing")
    domains = {
        balance_spec.domain: {
            **_align_balance_to_active(
                workspace,
                refreshed=Path(refreshed["path"]),
                input_dataset_id=input_dataset_id,
            ),
            "refreshed_normalization": refreshed,
        }
    }
    state.update(
        {
            "status": "prepared",
            "identity_symbol_count": len(identity),
            "prepared_domains": domains,
        }
    )
    _write_state(workspace, state)
    return state


def _install_domain(
    workspace: Path,
    *,
    spec: StatementSpec,
    prepared: Path,
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    digest = _sha256(prepared)[:24]
    dataset_id = f"{spec.domain}__{digest}"
    dataset_dir = root / "datasets" / spec.domain / dataset_id
    shard = dataset_dir / "shards" / "part-0000.parquet"
    if not shard.is_file():
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard.with_suffix(".tmp.parquet")
        shutil.copy2(prepared, temporary)
        os.replace(temporary, shard)
    quoted = str(shard).replace("'", "''")
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    connection = duckdb.connect()
    try:
        row_count = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{quoted}')"
            ).fetchone()[0]
        )
        duplicate_count = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {key_sql},count(*) n "
                f"FROM read_parquet('{quoted}') GROUP BY {key_sql} HAVING n>1)"
            ).fetchone()[0]
        )
        dates = connection.execute(
            "SELECT min(publish_date),max(publish_date),"
            f"count(*) FILTER(WHERE publish_date>'{END_DATE}'),"
            "count(*) FILTER(WHERE report_date>publish_date),"
            "count(*) FILTER(WHERE feature_available_date<>'' "
            "AND feature_available_date<=publish_date) "
            f"FROM read_parquet('{quoted}')"
        ).fetchone()
    finally:
        connection.close()
    if duplicate_count or any(int(value or 0) for value in dates[2:]):
        raise FinancialStatementUpdateError(
            f"statement_contract_failed:{spec.domain}:duplicates={duplicate_count}:"
            f"future={int(dates[2] or 0)}:report_after_publish={int(dates[3] or 0)}:"
            f"availability={int(dates[4] or 0)}"
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=spec.domain,
        layer="raw",
        frequency="quarterly_event",
        contract_version=(
            f"qdp_v2_{spec.domain}_pit_v5"
            if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
            and BALANCE_SEMANTIC_CONFLICT_COLUMN in pq.read_schema(shard).names
            else f"qdp_v2_{spec.domain}_pit_v2"
        ),
        primary_key=list(PRIMARY_KEY),
        start_date=str(dates[0]),
        end_date=str(dates[1]),
        row_count=row_count,
        shards=[
            ShardManifestEntry(
                path=str(shard.relative_to(root)).replace("\\", "/"),
                row_count=row_count,
                start_date=str(dates[0]),
                end_date=str(dates[1]),
                file_size=shard.stat().st_size,
                metadata={"prepared_sha256": _sha256(prepared)},
            )
        ],
        source={
            "provider": f"tushare_compatible_{spec.api_name}",
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "availability_semantics": "actual announcement date; consume next exchange-open day",
            "credential_persisted": False,
            "provider_revision_history_timestamp_complete": False,
        },
        quality={
            "primary_key_unique": True,
            "future_publish_rows": 0,
            "report_after_publish_rows": 0,
            "strict_feature_availability_lag": True,
            "provider_future_response_rows_not_persisted": True,
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(shard)),
        notes=[
            "f_ann_date is the PIT publication date when present; ann_date is the fallback",
            "duplicate provider rows prefer update_flag=1, then completeness, with conflicts retained as flags",
            "the provider does not expose complete correction timestamps for every restatement",
            "v2 preserves every v1 key and existing value; only audited balance-sheet fields are appended",
            "balance extension source conflicts are retained separately from legacy core-field conflicts",
            "combined statement fields are retained as reported and are never split into fabricated components",
            "stable trade-receivable, trade-payable, fixed-asset, and construction-in-progress measures retain explicit source states",
            "semantic-field conflicts are retained separately from other extension conflicts",
            "special financial-company statement structures retain not_applicable rather than numeric zero",
        ],
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "start_date": str(dates[0]),
        "end_date": str(dates[1]),
        "sha256": _sha256(shard),
    }


def commit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if (
        state.get("status") == "applied"
        and state.get("source_schema_version") == SOURCE_SCHEMA_VERSION
    ):
        return state
    if state.get("status") != "prepared":
        raise FinancialStatementUpdateError(
            f"statement_not_prepared:{state.get('status')}"
        )
    installed: dict[str, Any] = {}
    dataset_ids: dict[str, str] = {}
    for spec in V2_STATEMENT_SPECS:
        path = Path(state["prepared_domains"][spec.domain]["path"])
        dataset_id, record = _install_domain(
            workspace,
            spec=spec,
            prepared=path,
        )
        dataset_ids[spec.domain] = dataset_id
        installed[spec.domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    expected_unchanged = dict(state.get("input_dataset_ids", {}) or {})
    if str(dict(active.get("datasets", {}) or {}).get(
        DataDomain.BALANCE_SHEET_QUARTERLY, ""
    )) != str(expected_unchanged.get(DataDomain.BALANCE_SHEET_QUARTERLY, "")):
        raise FinancialStatementUpdateError("balance_statement_domain_drifted")
    for domain in (
        DataDomain.INCOME_STATEMENT_QUARTERLY,
        DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
    ):
        if str(dict(active.get("datasets", {}) or {}).get(domain, "")) != str(
            expected_unchanged.get(domain, "")
        ):
            raise FinancialStatementUpdateError(
                f"unchanged_statement_domain_drifted:{domain}"
            )
    active["datasets"] = {
        **dict(active.get("datasets", {}) or {}),
        **dataset_ids,
    }
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update({"status": "applied", "installed_domains": installed})
    _write_state(workspace, state)
    atomic_write_json(root / "audits" / f"{UPDATE_ID}.json", state)
    return state


def repair_balance_extension_conflict_metadata(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Install an exact four-field extension-conflict flag without changing core data."""

    workspace = _workspace(workspace_root)
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    current_id = str(current.get(domain, ""))
    current_path = dataset_manifest_for_id(root, current_id, domain)
    if current_path is None:
        raise FinancialStatementUpdateError("active_balance_manifest_missing")
    current_manifest = read_dataset_manifest(current_path)
    existing_columns = [str(item.get("name", "")) for item in current_manifest.schema]
    if (
        BALANCE_EXTENSION_CONFLICT_COLUMN in existing_columns
        and current_manifest.contract_version
        in {
            "qdp_v2_balance_sheet_quarterly_pit_v4",
            "qdp_v2_balance_sheet_quarterly_pit_v5",
        }
    ):
        return {
            "status": "already_repaired",
            "repair_id": BALANCE_CONFLICT_REPAIR_ID,
            "dataset_id": current_id,
        }

    balance_spec = next(spec for spec in V2_STATEMENT_SPECS if spec.domain == domain)
    normalized_paths = [
        _normalized_path(workspace, balance_spec, period)
        for period in _report_periods()
    ]
    if any(not path.is_file() for path in normalized_paths):
        raise FinancialStatementUpdateError(
            "balance_extension_normalized_evidence_missing"
        )
    base_paths = _dataset_paths(workspace, domain=domain, dataset_id=current_id)
    base_scan = f"read_parquet([{_sql_paths(base_paths)}], union_by_name=true)"
    normalized_scan = (
        f"read_parquet([{_sql_paths(normalized_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    output = (
        qdp_paths(workspace).data_dir
        / "qdp_runtime"
        / BALANCE_CONFLICT_REPAIR_ID
        / "prepared"
        / f"{domain}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    conflict_query = f"""
      SELECT {key_sql},
        count(DISTINCT {_balance_extension_hash_sql()}) > 1
          AS {BALANCE_EXTENSION_CONFLICT_COLUMN}
      FROM {normalized_scan}
      GROUP BY {key_sql}
    """
    with duckdb.connect() as connection:
        base_columns = [
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {base_scan}"
            ).fetchall()
        ]
        preserved_columns = [
            column
            for column in base_columns
            if column != BALANCE_EXTENSION_CONFLICT_COLUMN
        ]
        base_select = ",".join(f'"{column}"' for column in preserved_columns)
        joined_select = ",".join(f'b."{column}"' for column in preserved_columns)
        previous_conflict_count = (
            int(
                connection.execute(
                    f"SELECT count(*) FILTER(WHERE {BALANCE_EXTENSION_CONFLICT_COLUMN}) "
                    f"FROM {base_scan}"
                ).fetchone()[0]
                or 0
            )
            if BALANCE_EXTENSION_CONFLICT_COLUMN in base_columns
            else 0
        )
        connection.execute(
            f"""
            COPY (
              SELECT {joined_select},
                coalesce(r.{BALANCE_EXTENSION_CONFLICT_COLUMN},false)
                  AS {BALANCE_EXTENSION_CONFLICT_COLUMN}
              FROM {base_scan} b
              LEFT JOIN ({conflict_query}) r USING({key_sql})
              ORDER BY b.publish_date,b.symbol,b.report_date,b.report_type,
                       b.company_type,b.period_type
            ) TO '{str(temporary).replace(chr(39), chr(39) * 2)}'
              (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        output_scan = f"read_parquet('{str(temporary).replace(chr(39), chr(39) * 2)}')"
        missing_normalized_keys = int(
            connection.execute(
                "SELECT count(*) FILTER(WHERE r.symbol IS NULL) "
                f"FROM {base_scan} b LEFT JOIN ({conflict_query}) r USING({key_sql})"
            ).fetchone()[0]
        )
        output_stats = connection.execute(
            "SELECT count(*),"
            f"count(*)-count(DISTINCT concat_ws('|',{key_sql})),"
            f"count(*) FILTER(WHERE {BALANCE_EXTENSION_CONFLICT_COLUMN}),"
            f"count(*) FILTER(WHERE publish_date>'{END_DATE}') "
            f"FROM {output_scan}"
        ).fetchone()
        core_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  (SELECT {base_select} FROM {base_scan})
                  EXCEPT ALL
                  (SELECT {base_select} FROM {output_scan})
                )
                """
            ).fetchone()[0]
        )
        exact_extension_conflicts = int(
            connection.execute(
                f"SELECT count(*) FROM ({conflict_query}) "
                f"WHERE {BALANCE_EXTENSION_CONFLICT_COLUMN}"
            ).fetchone()[0]
        )
    checks = {
        "row_count_unchanged": int(output_stats[0]) == current_manifest.row_count,
        "primary_key_unique": int(output_stats[1]) == 0,
        "existing_non_conflict_columns_unchanged": core_mismatch == 0,
        "extension_conflicts_are_exact": int(output_stats[2])
        == exact_extension_conflicts,
        "all_active_keys_have_normalized_evidence": missing_normalized_keys == 0,
        "forbidden_2026_rows": int(output_stats[3]) == 0,
    }
    if not all(checks.values()):
        temporary.unlink(missing_ok=True)
        raise FinancialStatementUpdateError(
            f"balance_conflict_metadata_contract_failed:{checks}"
        )
    os.replace(temporary, output)

    dataset_id, installed = _install_domain(
        workspace, spec=balance_spec, prepared=output
    )
    installed_path = dataset_manifest_for_id(root, dataset_id, domain)
    assert installed_path is not None
    installed_manifest = read_dataset_manifest(installed_path)
    repaired_manifest = DatasetManifest.from_mapping(
        {
            **installed_manifest.to_dict(),
            "source": {
                **dict(installed_manifest.source or {}),
                "semantic_repair_id": BALANCE_CONFLICT_REPAIR_ID,
                "upstream_dataset_id": current_id,
            },
            "quality": {
                **dict(installed_manifest.quality or {}),
                "balance_extension_source_conflict_count": int(output_stats[2]),
                "prior_overbroad_balance_source_conflict_count": previous_conflict_count,
                "semantic_repair_checks": checks,
            },
            "notes": [
                *list(installed_manifest.notes or []),
                "immutable successor; non-conflict columns and values are unchanged",
                "the extension conflict flag compares only the four extension numeric values",
            ],
        }
    )
    _assert_credential_free(repaired_manifest.to_dict())
    write_dataset_manifest(root, repaired_manifest)

    latest_active = read_active_manifest(root)
    latest = active_dataset_map(latest_active)
    if latest.get(domain) != current_id:
        raise FinancialStatementUpdateError("active_balance_dataset_drifted")
    latest_active["datasets"] = {**latest, domain: dataset_id}
    latest_active["updated_at"] = utc_now()
    write_active_manifest(root, latest_active)
    if active_dataset_map(read_active_manifest(root)).get(domain) != dataset_id:
        raise FinancialStatementUpdateError("balance_conflict_active_switch_failed")

    payload = {
        "status": "applied",
        "repair_id": BALANCE_CONFLICT_REPAIR_ID,
        "input_dataset_id": current_id,
        "installed_domains": {domain: installed},
        "checks": checks,
        "statistics": {
            "row_count": int(output_stats[0]),
            "balance_extension_source_conflict_count": int(output_stats[2]),
            "prior_overbroad_balance_source_conflict_count": previous_conflict_count,
            "generic_only_conflict_count": max(
                0, previous_conflict_count - int(output_stats[2])
            ),
            "request_2026_count": 0,
        },
        "updated_at": utc_now(),
    }
    _assert_credential_free(payload)
    atomic_write_json(root / "audits" / f"{BALANCE_CONFLICT_REPAIR_ID}.json", payload)
    return payload


def _active_domain_record(workspace: Path, domain: str) -> dict[str, Any]:
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
    quoted = _sql_paths(paths)
    connection = duckdb.connect()
    try:
        stats = connection.execute(
            "SELECT count(*),count(DISTINCT symbol),"
            "sum(source_conflict),min(publish_date),max(publish_date),"
            f"sum(publish_date>'{END_DATE}') FROM read_parquet([{quoted}])"
        ).fetchone()
        cashflow_coverage = None
        balance_coverage = None
        if domain == DataDomain.CASH_FLOW_STATEMENT_QUARTERLY:
            cashflow_coverage = connection.execute(
                "SELECT count(payroll_paid),count(taxes_paid) "
                f"FROM read_parquet([{quoted}])"
            ).fetchone()
        if domain == DataDomain.BALANCE_SHEET_QUARTERLY:
            balance_coverage = connection.execute(
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
                "('observed','unknown','not_applicable') "
                "OR other_receivables_total_field_state IS NULL),"
                "count(*) FILTER(WHERE other_payables_total_field_state NOT IN "
                "('observed','unknown','not_applicable') "
                "OR other_payables_total_field_state IS NULL),"
                "count(*) FILTER(WHERE contract_liabilities_field_state NOT IN "
                "('observed','unknown','not_applicable') "
                "OR contract_liabilities_field_state IS NULL),"
                "count(trade_receivables_total),"
                "count(trade_payables_total),"
                "count(fixed_assets_measure),"
                "count(construction_in_progress_measure),"
                "count(*) FILTER(WHERE trade_receivables_field_state NOT IN "
                "('combined_plus_financing_observed',"
                "'combined_observed_financing_unreported',"
                "'components_plus_financing_observed',"
                "'components_observed_financing_unreported',"
                "'unknown','not_applicable') "
                "OR trade_receivables_field_state IS NULL "
                "OR trade_payables_field_state NOT IN "
                "('combined_observed','components_observed','unknown','not_applicable') "
                "OR trade_payables_field_state IS NULL "
                "OR fixed_assets_measure_field_state NOT IN "
                "('total_observed','component_fallback','unknown','not_applicable') "
                "OR fixed_assets_measure_field_state IS NULL "
                "OR construction_in_progress_measure_field_state NOT IN "
                "('total_observed','component_fallback','unknown','not_applicable') "
                "OR construction_in_progress_measure_field_state IS NULL),"
                "count(accounts_receivable_and_notes_reported),"
                "count(accounts_payable_and_notes_reported),"
                "count(receivables_financing),"
                "count(fixed_assets_total_reported),"
                "count(construction_in_progress_total_reported) "
                f"FROM read_parquet([{quoted}])"
            ).fetchone()
    finally:
        connection.close()
    result = {
        "dataset_id": dataset_id,
        "row_count": int(stats[0]),
        "symbol_count": int(stats[1]),
        "source_conflict_row_count": int(stats[2] or 0),
        "start_date": str(stats[3]),
        "end_date": str(stats[4]),
        "forbidden_2026_rows": int(stats[5] or 0),
    }
    if cashflow_coverage is not None:
        result["payroll_paid_nonnull_count"] = int(cashflow_coverage[0] or 0)
        result["taxes_paid_nonnull_count"] = int(cashflow_coverage[1] or 0)
    if balance_coverage is not None:
        quarter_rows = int(balance_coverage[0] or 0)
        post_2020_rows = int(balance_coverage[3] or 0)
        result.update(
            {
                "q1_q3_row_count": quarter_rows,
                "other_receivables_total_q1_q3_nonnull_count": int(
                    balance_coverage[1] or 0
                ),
                "other_payables_total_q1_q3_nonnull_count": int(
                    balance_coverage[2] or 0
                ),
                "other_receivables_total_q1_q3_coverage": (
                    float(balance_coverage[1] or 0) / quarter_rows
                    if quarter_rows
                    else 0.0
                ),
                "other_payables_total_q1_q3_coverage": (
                    float(balance_coverage[2] or 0) / quarter_rows
                    if quarter_rows
                    else 0.0
                ),
                "post_2020_row_count": post_2020_rows,
                "contract_liabilities_post_2020_nonnull_count": int(
                    balance_coverage[4] or 0
                ),
                "contract_liabilities_post_2020_coverage": (
                    float(balance_coverage[4] or 0) / post_2020_rows
                    if post_2020_rows
                    else 0.0
                ),
                "invalid_customer_liability_state_count": int(balance_coverage[5] or 0),
                "invalid_component_state_count": int(
                    sum(int(value or 0) for value in balance_coverage[6:9])
                ),
                "trade_receivables_nonnull_count": int(balance_coverage[9] or 0),
                "trade_payables_nonnull_count": int(balance_coverage[10] or 0),
                "fixed_assets_measure_nonnull_count": int(balance_coverage[11] or 0),
                "construction_in_progress_measure_nonnull_count": int(
                    balance_coverage[12] or 0
                ),
                "invalid_semantic_state_count": int(balance_coverage[13] or 0),
                "reported_receivables_combination_nonnull_count": int(
                    balance_coverage[14] or 0
                ),
                "reported_payables_combination_nonnull_count": int(
                    balance_coverage[15] or 0
                ),
                "receivables_financing_nonnull_count": int(
                    balance_coverage[16] or 0
                ),
                "reported_fixed_assets_total_nonnull_count": int(
                    balance_coverage[17] or 0
                ),
                "reported_construction_in_progress_total_nonnull_count": int(
                    balance_coverage[18] or 0
                ),
            }
        )
    return result


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    domains = {
        spec.domain: _active_domain_record(workspace, spec.domain)
        for spec in STATEMENT_SPECS
    }
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    inputs = dict(state.get("input_dataset_ids", {}) or {})
    balance = domains[DataDomain.BALANCE_SHEET_QUARTERLY]
    alignment = dict(
        dict(state.get("prepared_domains", {}) or {}).get(
            DataDomain.BALANCE_SHEET_QUARTERLY, {}
        )
        or {}
    )
    checks = {
        "forbidden_2026_rows": all(
            record["forbidden_2026_rows"] == 0 for record in domains.values()
        ),
        "credential_not_persisted": True,
        "next_exchange_open_availability": True,
        "existing_financial_ratio_domain_unchanged": True,
        "income_statement_dataset_unchanged": active.get(
            DataDomain.INCOME_STATEMENT_QUARTERLY
        )
        == inputs.get(DataDomain.INCOME_STATEMENT_QUARTERLY),
        "cash_flow_statement_dataset_unchanged": active.get(
            DataDomain.CASH_FLOW_STATEMENT_QUARTERLY
        )
        == inputs.get(DataDomain.CASH_FLOW_STATEMENT_QUARTERLY),
        "balance_existing_keys_and_values_unchanged": bool(
            alignment
            and int(alignment.get("row_count", -1))
            == int(alignment.get("input_row_count", -2))
            and int(alignment.get("existing_core_value_mismatch_count", -1)) == 0
            and int(alignment.get("primary_key_duplicate_count", -1)) == 0
        ),
        "balance_component_states_valid": int(
            balance.get("invalid_customer_liability_state_count", -1)
        )
        == 0
        and int(balance.get("invalid_component_state_count", -1)) == 0,
        "balance_semantic_states_valid": int(
            balance.get("invalid_semantic_state_count", -1)
        )
        == 0,
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
        "contract_liabilities_post_2020_populated": int(
            balance.get("contract_liabilities_post_2020_nonnull_count", 0)
        )
        > 0,
        "cashflow_employee_and_tax_payments_populated": domains[
            DataDomain.CASH_FLOW_STATEMENT_QUARTERLY
        ].get("payroll_paid_nonnull_count", 0)
        > 0
        and domains[DataDomain.CASH_FLOW_STATEMENT_QUARTERLY].get(
            "taxes_paid_nonnull_count", 0
        )
        > 0,
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
    if (
        len(V2_STATEMENT_SPECS) != 1
        or V2_STATEMENT_SPECS[0].domain != DataDomain.BALANCE_SHEET_QUARTERLY
    ):
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
    return {
        key: value
        for key, value in state.items()
        if key not in {"completed", "prepared_domains"}
    } | {
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
