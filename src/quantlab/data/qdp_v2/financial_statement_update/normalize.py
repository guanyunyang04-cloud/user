"""Financial Statement Update: normalize responsibilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.research_event_update.context import _next_open_date

from .config import (
    BALANCE_DERIVED_COLUMNS,
    COMMON_OUTPUT_COLUMNS,
    END_DATE,
    START_DATE,
    StatementSpec,
)
from .context import (
    _provider_publish_date,
)


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
    receivable_components_complete = data["notes_receivable"].notna() & data["accounts_receivable"].notna()
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
    payable_components_complete = data["notes_payable"].notna() & data["accounts_payable"].notna()
    component_payables = data["notes_payable"] + data["accounts_payable"]
    data["trade_payables_total"] = reported_payables.where(
        reported_payables.notna(), component_payables.where(payable_components_complete)
    )
    data["trade_payables_to_current_liabilities"] = _safe_series_ratio(
        data["trade_payables_total"], data["current_liabilities"]
    )
    data["trade_payables_to_total_assets"] = _safe_series_ratio(data["trade_payables_total"], data["total_assets"])
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
    derived_columns = list(BALANCE_DERIVED_COLUMNS) if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY else []
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
    data["announcement_date"] = pd.to_datetime(data["ann_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["actual_announcement_date"] = pd.to_datetime(data["f_ann_date"], errors="coerce").dt.strftime("%Y-%m-%d")
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
    data["update_flag"] = pd.to_numeric(data["update_flag"], errors="coerce").fillna(0).astype("int8")
    for provider, canonical in spec.metric_map.items():
        data[canonical] = pd.to_numeric(data[provider], errors="coerce")
    if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY:
        advances = data["advances_from_customers"]
        contract = data["contract_liabilities"]
        data["customer_advances_and_contract_liabilities"] = pd.concat([advances, contract], axis=1).sum(
            axis=1, min_count=1
        )
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
            data[f"{column}_field_state"] = _balance_component_state(data[column], data["company_type"])
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
