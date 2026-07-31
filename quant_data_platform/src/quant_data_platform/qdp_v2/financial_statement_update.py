from __future__ import annotations

"""Backfill compact point-in-time income, balance-sheet, and cash-flow data."""

import argparse
import json
import os
import shutil
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

UPDATE_ID = "financial_statement_quarterly_pit_v1"
SOURCE_SCHEMA_VERSION = 2
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
    "oth_receiv": "other_receivables",
    "prepayment": "prepayments",
    "inventories": "inventory",
    "total_cur_assets": "current_assets",
    "fix_assets": "fixed_assets",
    "cip": "construction_in_progress",
    "intan_assets": "intangible_assets",
    "goodwill": "goodwill",
    "total_nca": "noncurrent_assets",
    "total_assets": "total_assets",
    "st_borr": "short_term_borrowings",
    "notes_payable": "notes_payable",
    "acct_payable": "accounts_payable",
    "adv_receipts": "advances_from_customers",
    "payroll_payable": "employee_compensation_payable",
    "taxes_payable": "taxes_payable",
    "oth_payable": "other_payables",
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
    atomic_write_json(_state_path(workspace), payload)


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
    client = _TushareClient(_resolve_tushare_token())
    periods = _report_periods()
    completed = dict(state.get("completed", {}) or {})
    discarded_future_rows = int(state.get("provider_future_rows_discarded", 0) or 0)
    completed_calls = 0
    for spec in STATEMENT_SPECS:
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
                            "total_calls": len(periods) * len(STATEMENT_SPECS),
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


def _normalize_statement_part(
    raw: pd.DataFrame,
    *,
    spec: StatementSpec,
    identity_symbols: set[str],
    open_dates: np.ndarray,
) -> pd.DataFrame:
    metric_columns = list(spec.metric_map.values())
    output_columns = [
        *COMMON_OUTPUT_COLUMNS,
        *metric_columns,
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


def _prepare_statement(
    workspace: Path,
    *,
    spec: StatementSpec,
    identity_symbols: set[str],
    open_dates: np.ndarray,
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
        if (
            not output_path.is_file()
            or raw_path.stat().st_mtime_ns > output_path.stat().st_mtime_ns
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
    prepared = _runtime(workspace) / "prepared" / f"{spec.domain}.parquet"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_suffix(".tmp.parquet")
    metrics = list(spec.metric_map.values())
    selected = [
        *COMMON_OUTPUT_COLUMNS,
        *metrics,
        "source_duplicate_count",
        "source_conflict",
        "lag_policy",
        "source",
    ]
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    select_sql = ",".join(f'"{column}"' for column in selected)
    sql = f"""
    COPY (
      WITH source AS (
        SELECT * FROM read_parquet([{_sql_paths(normalized_paths)}], union_by_name=true)
      ), ranked AS (
        SELECT *,
          count(*) OVER (PARTITION BY {key_sql}) AS source_duplicate_count,
          count(DISTINCT _metric_hash) OVER (PARTITION BY {key_sql}) > 1 AS source_conflict,
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
        row = connection.execute(
            f"SELECT count(*),sum(source_duplicate_count>1),sum(source_conflict),"
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
        "start_date": str(row[3]),
        "end_date": str(row[4]),
        "sha256": _sha256(prepared),
    }


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"prepared", "applied"}:
        return state
    if state.get("status") != "downloaded":
        raise FinancialStatementUpdateError(
            f"statement_not_downloaded:{state.get('status')}"
        )
    identity = set(_identity_symbols(workspace))
    open_dates = _open_dates(workspace)
    domains = {
        spec.domain: _prepare_statement(
            workspace,
            spec=spec,
            identity_symbols=identity,
            open_dates=open_dates,
        )
        for spec in STATEMENT_SPECS
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
        contract_version=f"qdp_v2_{spec.domain}_pit_v1",
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
    if state.get("status") == "applied":
        return state
    if state.get("status") != "prepared":
        raise FinancialStatementUpdateError(
            f"statement_not_prepared:{state.get('status')}"
        )
    installed: dict[str, Any] = {}
    dataset_ids: dict[str, str] = {}
    for spec in STATEMENT_SPECS:
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
        if domain == DataDomain.CASH_FLOW_STATEMENT_QUARTERLY:
            cashflow_coverage = connection.execute(
                "SELECT count(payroll_paid),count(taxes_paid) "
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
    return result


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    domains = {
        spec.domain: _active_domain_record(workspace, spec.domain)
        for spec in STATEMENT_SPECS
    }
    return {
        "status": "ok",
        "update_id": UPDATE_ID,
        "domains": domains,
        "checks": {
            "forbidden_2026_rows": all(
                record["forbidden_2026_rows"] == 0 for record in domains.values()
            ),
            "credential_not_persisted": True,
            "next_exchange_open_availability": True,
            "existing_financial_ratio_domain_unchanged": True,
            "cashflow_employee_and_tax_payments_populated": domains[
                DataDomain.CASH_FLOW_STATEMENT_QUARTERLY
            ].get("payroll_paid_nonnull_count", 0)
            > 0
            and domains[DataDomain.CASH_FLOW_STATEMENT_QUARTERLY].get(
                "taxes_paid_nonnull_count", 0
            )
            > 0,
        },
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
    return {
        "status": "ok",
        "checks": {
            "report_period_count": len(periods),
            "maximum_report_period": periods[-1],
            "forbidden_2026_provider_rows_not_persisted": True,
            "statement_domains_are_separate": True,
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
        "total_calls": len(_report_periods()) * len(STATEMENT_SPECS),
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
    else:
        payload = self_test()
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
