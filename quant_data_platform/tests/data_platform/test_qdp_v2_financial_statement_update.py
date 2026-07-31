from __future__ import annotations

import numpy as np
import pandas as pd
from quant_data_platform.qdp_v2.financial_statement_update import (
    CASH_FLOW_METRICS,
    START_DATE,
    STATEMENT_SPECS,
    _in_scope_provider_rows,
    _normalize_statement_part,
    _report_periods,
)


def test_statement_periods_stop_before_2025_annual_report() -> None:
    periods = _report_periods()

    assert periods[0] == "20100331"
    assert periods[-1] == "20250930"
    assert "20251231" not in periods


def test_provider_future_revision_rows_are_not_persisted() -> None:
    raw = pd.DataFrame(
        {
            "ann_date": ["20251231", "20260101"],
            "f_ann_date": ["20251231", "20260101"],
        }
    )

    scoped, future = _in_scope_provider_rows(raw)

    assert len(scoped) == 1
    assert future == 1


def test_statement_normalization_uses_actual_announcement_and_next_open() -> None:
    spec = STATEMENT_SPECS[0]
    raw = pd.DataFrame(
        {
            **{column: [None] for column in spec.fields},
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240101"],
            "f_ann_date": ["20240102"],
            "end_date": ["20231231"],
            "report_type": ["1"],
            "comp_type": ["1"],
            "end_type": ["4"],
            "revenue": [100.0],
            "n_income_attr_p": [10.0],
            "update_flag": ["1"],
        }
    )
    open_dates = np.asarray(
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
        dtype="datetime64[ns]",
    )

    result = _normalize_statement_part(
        raw,
        spec=spec,
        identity_symbols={"000001.SZ"},
        open_dates=open_dates,
    )

    assert len(result) == 1
    assert result.loc[0, "publish_date"] == "2024-01-02"
    assert result.loc[0, "feature_available_date"] == "2024-01-03"
    assert result.loc[0, "revenue"] == 100.0
    assert result.loc[0, "parent_net_income"] == 10.0
    assert result.loc[0, "source_date"] >= START_DATE


def test_statement_normalization_rejects_nonidentity_and_future_rows() -> None:
    spec = STATEMENT_SPECS[2]
    base = {column: [None, None, None] for column in spec.fields}
    raw = pd.DataFrame(
        {
            **base,
            "ts_code": ["000001.SZ", "999999.SZ", "000001.SZ"],
            "ann_date": ["20240102", "20240102", "20260102"],
            "f_ann_date": ["20240102", "20240102", "20260102"],
            "end_date": ["20231231", "20231231", "20251231"],
            "report_type": ["1", "1", "1"],
            "comp_type": ["1", "1", "1"],
            "end_type": ["4", "4", "4"],
            "n_cashflow_act": [1.0, 2.0, 3.0],
            "update_flag": ["1", "1", "1"],
        }
    )
    open_dates = np.asarray(
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
        dtype="datetime64[ns]",
    )

    result = _normalize_statement_part(
        raw,
        spec=spec,
        identity_symbols={"000001.SZ"},
        open_dates=open_dates,
    )

    assert result["symbol"].tolist() == ["000001.SZ"]
    assert result["net_operating_cash_flow"].tolist() == [1.0]


def test_cashflow_uses_provider_employee_and_tax_payment_fields() -> None:
    assert CASH_FLOW_METRICS["c_paid_to_for_empl"] == "payroll_paid"
    assert CASH_FLOW_METRICS["c_paid_for_taxes"] == "taxes_paid"
    assert "payroll_paid" not in CASH_FLOW_METRICS
    assert "tax_payments" not in CASH_FLOW_METRICS

    spec = STATEMENT_SPECS[2]
    raw = pd.DataFrame(
        {
            **{column: [None] for column in spec.fields},
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240102"],
            "f_ann_date": ["20240102"],
            "end_date": ["20231231"],
            "report_type": ["1"],
            "comp_type": ["1"],
            "end_type": ["4"],
            "c_paid_to_for_empl": [12.5],
            "c_paid_for_taxes": [3.5],
            "update_flag": ["1"],
        }
    )
    open_dates = np.asarray(
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
        dtype="datetime64[ns]",
    )

    result = _normalize_statement_part(
        raw,
        spec=spec,
        identity_symbols={"000001.SZ"},
        open_dates=open_dates,
    )

    assert result.loc[0, "payroll_paid"] == 12.5
    assert result.loc[0, "taxes_paid"] == 3.5
