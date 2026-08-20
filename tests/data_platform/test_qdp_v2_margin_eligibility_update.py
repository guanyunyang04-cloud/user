from __future__ import annotations

import io

import duckdb
import pandas as pd

from quantlab.data.qdp_v2 import margin_eligibility_update as update


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.content = payload

    def raise_for_status(self) -> None:
        return None


def _xlsx(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def test_szse_historical_mojibake_headers_use_verified_column_order(
    monkeypatch,
) -> None:
    raw = pd.DataFrame(
        [["000527", "name", "1,000", "2,000", "300", "400", "500", "2,500"]],
        columns=[f"broken_{index}" for index in range(8)],
    )
    monkeypatch.setattr(
        update.requests, "get", lambda *args, **kwargs: _Response(_xlsx(raw))
    )

    result = update._fetch_szse_excel("2011-01-04", endpoint="szse_detail")

    row = result.frame.iloc[0]
    assert row["symbol"] == "000527.SZ"
    assert row["rzmre"] == 1_000
    assert row["rzye"] == 2_000
    assert row["rqmcl"] == 300
    assert row["rqyl"] == 400
    assert row["rqye"] == 500
    assert row["rzrqye"] == 2_500


def test_szse_eligibility_is_not_inferred_from_detail_balance(monkeypatch) -> None:
    raw = pd.DataFrame(
        [["000001", "name", "Y", "N", "N", "N", ""]],
        columns=[f"broken_{index}" for index in range(7)],
    )
    monkeypatch.setattr(
        update.requests, "get", lambda *args, **kwargs: _Response(_xlsx(raw))
    )

    result = update._fetch_szse_excel("2011-01-04", endpoint="szse_eligibility")

    row = result.frame.iloc[0]
    assert bool(row["eligible"])
    assert bool(row["finance_eligible"])
    assert not bool(row["securities_lending_eligible"])


def test_sse_detail_absence_is_unknown_not_known_ineligible() -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH", "000001.SZ", "000002.SZ"],
            "trade_date": ["2024-01-02"] * 4,
            "exchange": ["SH", "SH", "SZ", "SZ"],
            "board": ["main"] * 4,
        }
    )
    eligibility = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": ["2024-01-02"],
            "exchange": ["SZ"],
            "eligible": [True],
            "finance_eligible": [True],
            "securities_lending_eligible": [False],
            "source": ["szse"],
        }
    )
    detail = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2024-01-02"],
            "exchange": ["SH"],
        }
    )
    next_open = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "feature_available_date": ["2024-01-03"],
        }
    )
    with duckdb.connect() as connection:
        connection.register("universe", universe)
        connection.register("official_eligibility", eligibility)
        connection.register("official_detail", detail)
        connection.register("next_open_dates", next_open)
        result = connection.execute(
            update._eligibility_query(universe_scan="universe", year=2024)
        ).fetchdf()

    result = result.set_index("symbol")
    assert result.loc["600000.SH", "eligibility_state"] == "eligible_observed"
    assert bool(result.loc["600000.SH", "eligible"])
    assert result.loc["600001.SH", "eligibility_state"] == "source_unavailable"
    assert pd.isna(result.loc["600001.SH", "eligible"])
    assert not bool(result.loc["600001.SH", "source_available"])
    assert result.loc["000001.SZ", "eligibility_state"] == "eligible_observed"
    assert result.loc["000002.SZ", "eligibility_state"] == "known_ineligible"
    assert not bool(result.loc["000002.SZ", "eligible"])
