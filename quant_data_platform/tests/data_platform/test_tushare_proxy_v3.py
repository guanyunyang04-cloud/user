from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from quant_data_platform.domains.contracts import DataDomain, HistoryPageFetchRequest
from quant_data_platform.qdp_v3.constants import EXPECTED_1M_BAR_ENDS, EXPECTED_5M_BAR_ENDS
from quant_data_platform.qdp_v3.intraday_1m import (
    aggregate_1m_to_5m,
    is_complete_1m_day,
    normalize_provider_1m,
)
from quant_data_platform.qdp_v3.intraday_build import _quality_coverage
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry
from quant_data_platform.qdp_v3.proxy_factor import (
    derive_proxy_factor_evidence,
    reconcile_proxy_factor_third_path,
)
from quant_data_platform.qdp_v3.storage import write_raw_partition
from quant_data_platform.tushare_proxy import (
    TushareProxyClient,
    TushareProxyConfig,
    TushareProxyProtocolError,
    redact_secrets,
)


@dataclass
class _FakeResponse:
    payload: object
    status_code: int = 200

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http status {self.status_code}")


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}
        self.requests: list[dict] = []

    def post(self, url, *, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def _client(payload: object) -> TushareProxyClient:
    session = _FakeSession([_FakeResponse(payload)])
    return TushareProxyClient(
        TushareProxyConfig(token="unit-test-secret", url="https://example.test/api", retries=0),
        session_factory=lambda: session,
        sleeper=lambda _: None,
    )


def _history_payload(row_count: int) -> dict:
    end = pd.Timestamp("2026-03-31 15:00:00")
    rows = []
    for offset in range(row_count):
        timestamp = end - pd.Timedelta(minutes=offset)
        rows.append(["600000.SH", timestamp.strftime("%Y-%m-%d %H:%M:%S"), 10, 10.1, 9.9, 10, 100, 1000])
    return {
        "code": 0,
        "msg": "",
        "data": {
            "fields": ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"],
            "items": rows,
        },
    }


@pytest.mark.parametrize("row_count,complete", [(7_999, True), (8_000, False)])
def test_history_page_full_page_requires_next_cursor(row_count: int, complete: bool) -> None:
    client = _client(_history_payload(row_count))
    result = client.fetch_history_page(
        HistoryPageFetchRequest(
            domain=DataDomain.MARKET_INTRADAY_1M,
            provider_symbol="600000.SH",
            frequency="1m",
            start_at="2010-01-01 00:00:00",
            end_at="2026-03-31 15:00:00",
        )
    )
    assert result.row_count == row_count
    assert result.is_complete is complete
    assert bool(result.next_end_at) is (not complete)
    assert "unit-test-secret" not in str(result.request_metadata_without_token)


def test_proxy_rejects_field_width_mismatch() -> None:
    client = _client({"code": 0, "data": {"fields": ["a", "b"], "items": [[1]]}})
    with pytest.raises(TushareProxyProtocolError, match="field_width_mismatch"):
        client.fetch_frame(api_name="daily", params={}, fields="a,b")


def test_recursive_redaction_covers_token_and_mcp_url() -> None:
    redacted = redact_secrets(
        {"token": "abc", "message": "failed abc https://example/mcp/token=abc?q=1"},
        secrets=("abc",),
    )
    serialized = str(redacted)
    assert "abc" not in serialized
    assert "<redacted>" in serialized


def _raw_241() -> pd.DataFrame:
    times = ["09:30", *EXPECTED_1M_BAR_ENDS]
    rows = []
    for index, bar_end in enumerate(times):
        rows.append(
            {
                "ts_code": "600000.SH",
                "trade_time": f"2010-01-04 {bar_end}:00",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "vol": 1.0,
                "amount": 10.0,
            }
        )
    rows[0].update({"open": 9.8, "high": 10.2, "low": 9.7, "close": 9.9, "vol": 2.0, "amount": 20.0})
    rows[1].update({"open": 10.0, "high": 10.3, "low": 9.8, "close": 10.1, "vol": 3.0, "amount": 30.0})
    return pd.DataFrame(rows)


def test_proxy_241_to_canonical_240_and_5m() -> None:
    one = normalize_provider_1m(
        _raw_241(),
        provider_symbol="600000.SH",
        source="tushare_proxy",
        merge_0930_into_0931=True,
    )
    assert is_complete_1m_day(one)
    first = one.loc[one["bar_end"].eq("09:31")].iloc[0]
    assert first["open"] == pytest.approx(9.8)
    assert first["high"] == pytest.approx(10.3)
    assert first["low"] == pytest.approx(9.7)
    assert first["close"] == pytest.approx(10.1)
    assert first["volume"] == pytest.approx(5.0)
    assert first["amount"] == pytest.approx(50.0)
    five = aggregate_1m_to_5m(one)
    assert len(five) == 48
    assert tuple(five["bar_end"]) == EXPECTED_5M_BAR_ENDS


def test_proxy_factor_comparison_is_constant_scale_invariant(tmp_path) -> None:
    registry = SecurityIdentityRegistry.from_sources(provider_symbols=["600000.SH"])
    raw = pd.DataFrame(
        {
            "ts_code": ["600000.SH"] * 3,
            "trade_date": ["2000-01-01", "2000-01-02", "2001-01-01"],
            # The proxy is scaled by ten relative to BaoStock.  Only the
            # adjacent 2x change is semantically meaningful.
            "adj_factor": [10.0, 10.0, 20.0],
        }
    )
    ref, _ = write_raw_partition(
        raw_domain="test_tushare_proxy_factor_raw",
        partition_field="provider_symbol",
        partition_value="600000.SH",
        frame=raw,
        receipt={"provider": "tushare_proxy", "quality_tier": "provisional"},
        workspace_root=tmp_path,
    )
    baselines, proxy_events, conflicts, metrics = derive_proxy_factor_evidence(
        [ref], identity_registry=registry
    )
    assert conflicts.empty
    assert metrics["factor_change_event_count"] == 1
    assert proxy_events.loc[0, "factor_ratio"] == pytest.approx(2.0)

    security_id = registry.security_id_for_provider_symbol("600000.SH")
    bao = pd.DataFrame(
        {
            "security_id": [security_id, security_id],
            "divid_operate_date": ["2000-01-01", "2001-01-01"],
            "symbol_on_date": ["600000.SH", "600000.SH"],
            "provider_symbol": ["600000.SH", "600000.SH"],
            "fore_adjust_factor": [1.0, 0.5],
            "back_adjust_factor": [1.0, 2.0],
            "adjust_factor": [1.0, 2.0],
            "query_date": ["", ""],
            "source_method": ["symbol_history", "symbol_history"],
            "verification_status": ["symbol_history_unreconciled"] * 2,
            "identity_mapping_status": ["mapped"] * 2,
            "source": ["baostock.query_adjust_factor"] * 2,
        }
    )
    admitted, disputed, findings, reconciliation = reconcile_proxy_factor_third_path(
        bao,
        proxy_baselines=baselines,
        proxy_events=proxy_events,
        comparable_start="2010-01-01",
        comparable_end="2026-12-31",
    )
    assert disputed.empty
    assert not findings
    assert set(admitted["divid_operate_date"]) == {"2000-01-01", "2001-01-01"}
    assert reconciliation["verified_ratio_count"] == 1


def test_intraday_watermark_excludes_trailing_lag_but_rejects_internal_hole() -> None:
    import duckdb

    connection = duckdb.connect(":memory:")
    try:
        connection.execute("CREATE TABLE expected (security_id VARCHAR, trade_date VARCHAR)")
        connection.execute("CREATE TABLE actual_1m (security_id VARCHAR, trade_date VARCHAR)")
        connection.execute("CREATE TABLE explained_1m (security_id VARCHAR, trade_date VARCHAR)")
        expected = [
            (security, date)
            for date in ("2026-07-09", "2026-07-10", "2026-07-13")
            for security in ("S1", "S2")
        ]
        connection.executemany("INSERT INTO expected VALUES (?, ?)", expected)
        connection.executemany(
            "INSERT INTO actual_1m VALUES (?, ?)",
            [("S1", "2026-07-09"), ("S2", "2026-07-09")],
        )
        connection.executemany(
            "INSERT INTO explained_1m VALUES (?, ?)",
            [("S1", "2026-07-09"), ("S2", "2026-07-09"), ("S1", "2026-07-13")],
        )
        coverage, _ = _quality_coverage(connection)
    finally:
        connection.close()
    assert coverage["watermark"] == "2026-07-09"
    assert coverage["expected_stock_day_count"] == 2
    assert coverage["strict_coverage_rate"] == pytest.approx(1.0)
    assert coverage["trailing_lag_trade_date_count"] == 2
    assert coverage["non_continuous_history_hole"] is True
