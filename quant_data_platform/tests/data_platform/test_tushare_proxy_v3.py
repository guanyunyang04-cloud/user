from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_data_platform.domains.contracts import HistoryPageFetchRequest, HistoryPageResult
from quant_data_platform.qdp_v3.constants import EXPECTED_5M_BAR_ENDS
from quant_data_platform.qdp_v3.historical import (
    HistoricalJobState,
    HistoricalTaskState,
    _history_worker_limit,
    ingest_tushare_proxy_intraday,
    _merge_intraday_range_frames,
    _official_restatement_aliases,
    _proxy_performance_gate_failed,
    _prove_provider_symbol_restatement,
    _run_intraday_symbol,
    _staging_root,
    _write_staged_page,
)
from quant_data_platform.qdp_v3.intraday import is_complete_5m_day, normalize_tushare_proxy_5m
from quant_data_platform.qdp_v3.intraday_build import _quality_coverage
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.proxy_compatibility import (
    REQUIRED_PROXY_APIS,
    _evaluate_5m_unit_contract,
    lock_bootstrap_cutoff,
    run_tushare_proxy_compatibility_gate,
)
from quant_data_platform.qdp_v3.proxy_factor import (
    derive_proxy_factor_evidence,
    reconcile_proxy_factor_third_path,
)
from quant_data_platform.qdp_v3.storage import (
    get_raw_partition,
    read_raw_receipt,
    write_raw_partition,
)
from quant_data_platform.tushare_proxy import (
    TushareProxyClient,
    TushareProxyConfig,
    TushareProxyProtocolError,
    TushareProxyQuotaError,
    TushareProxyRateLimiter,
    redact_secrets,
)


@dataclass
class _FakeResponse:
    payload: object
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)

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
            provider_symbol="600000.SH",
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


def test_rate_limit_response_defers_all_workers_until_retry_after() -> None:
    now = [0.0]

    def advance(seconds: float) -> None:
        now[0] += float(seconds)

    limiter = TushareProxyRateLimiter(
        rate_per_minute=135,
        burst=4,
        clock=lambda: now[0],
        sleeper=advance,
    )
    session = _FakeSession(
        [
            _FakeResponse({}, status_code=429, headers={"Retry-After": "3"}),
            _FakeResponse({"code": 0, "data": {"fields": ["a"], "items": [[1]]}}),
        ]
    )
    client = TushareProxyClient(
        TushareProxyConfig(token="unit-test-secret", url="https://example.test/api", retries=1),
        limiter=limiter,
        session_factory=lambda: session,
        sleeper=advance,
    )

    result = client.fetch_frame(api_name="daily", params={}, fields="a")

    assert result.frame.to_dict("records") == [{"a": 1}]
    assert now[0] == pytest.approx(3.0)
    assert client.metrics.rate_limit_count == 1
    assert client.metrics.success_count == 1
    assert limiter.snapshot()["rate_per_minute"] == 121
    assert limiter.snapshot()["rate_reduction_count"] == 1


def test_exhausted_http_429_retries_are_classified_as_daily_quota() -> None:
    now = [0.0]

    def advance(seconds: float) -> None:
        now[0] += float(seconds)

    session = _FakeSession(
        [
            _FakeResponse({}, status_code=429, headers={"Retry-After": "2"}),
            _FakeResponse({}, status_code=429, headers={"Retry-After": "2"}),
        ]
    )
    client = TushareProxyClient(
        TushareProxyConfig(token="unit-test-secret", url="https://example.test/api", retries=1),
        limiter=TushareProxyRateLimiter(
            rate_per_minute=96,
            burst=1,
            clock=lambda: now[0],
            sleeper=advance,
        ),
        session_factory=lambda: session,
        sleeper=advance,
    )

    with pytest.raises(TushareProxyQuotaError, match="http_429_exhausted"):
        client.fetch_frame(api_name="stk_mins", params={}, fields="ts_code,trade_time")

    assert client.metrics.rate_limit_count == 2
    assert client.metrics.success_count == 0


def test_bootstrap_update_can_reuse_matching_passed_compatibility_report(tmp_path: Path) -> None:
    paths = ensure_qdp_v3_layout(tmp_path)
    client = TushareProxyClient(
        TushareProxyConfig(token="unit-test-secret", url="https://example.test/api"),
        session_factory=lambda: (_ for _ in ()).throw(AssertionError("network call not expected")),
    )
    report_path = paths.compatibility / "tushare_proxy__passed.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "provider": "tushare_proxy",
                "endpoint": client.config.url,
                "probe_date": "2026-07-13",
                "smoke": False,
                "entitlement": {"token_sha256": client.config.token_sha256},
                "probes": [
                    {"api_name": api_name, "status": "passed"}
                    for api_name in REQUIRED_PROXY_APIS
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_tushare_proxy_compatibility_gate(
        workspace_root=tmp_path,
        as_of_date="2026-07-13",
        client=client,
        reuse_passed=True,
    )

    assert result["status"] == "passed"
    assert result["reused"] is True
    assert result["report_path"] == str(report_path.resolve())


def test_rate_limiter_reduces_at_most_once_per_cooldown_window() -> None:
    now = [0.0]
    limiter = TushareProxyRateLimiter(
        rate_per_minute=135,
        burst=4,
        clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + float(seconds)),
    )

    limiter.defer(2.0, rate_limited=True)
    limiter.defer(2.0, rate_limited=True)
    assert limiter.snapshot()["rate_per_minute"] == 121
    assert limiter.snapshot()["rate_reduction_count"] == 1

    now[0] = 61.0
    limiter.defer(2.0, rate_limited=True)
    assert limiter.snapshot()["rate_per_minute"] == 108
    assert limiter.snapshot()["rate_reduction_count"] == 2


def test_production_performance_gate_uses_attempt_error_rate_after_one_thousand_requests() -> None:
    assert not _proxy_performance_gate_failed({"request_count": 999, "error_rate": 1.0})
    assert not _proxy_performance_gate_failed({"request_count": 1_000, "error_rate": 0.02})
    assert _proxy_performance_gate_failed({"request_count": 1_000, "error_rate": 0.0201})


def test_history_worker_limit_honors_low_memory_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDP_TUSHARE_PROXY_HISTORY_WORKERS", raising=False)
    assert _history_worker_limit(4) == 3

    monkeypatch.setenv("QDP_TUSHARE_PROXY_HISTORY_WORKERS", "2")
    assert _history_worker_limit(3) == 2
    assert _history_worker_limit(1) == 1

    monkeypatch.setenv("QDP_TUSHARE_PROXY_HISTORY_WORKERS", "bad")
    with pytest.raises(ValueError, match="history_worker_env_invalid"):
        _history_worker_limit(3)


def test_proxy_rate_can_be_overridden_for_a_supervised_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDP_TUSHARE_PROXY_TOKEN", "unit-test-secret")
    monkeypatch.setenv("QDP_TUSHARE_PROXY_RATE_PER_MINUTE", "100")
    assert TushareProxyConfig.from_env().safe_rate_per_minute == 100

    monkeypatch.setenv("QDP_TUSHARE_PROXY_RATE_PER_MINUTE", "bad")
    with pytest.raises(ValueError, match="tushare_proxy_rate_env_invalid"):
        TushareProxyConfig.from_env()


def test_recursive_redaction_covers_token_and_mcp_url() -> None:
    redacted = redact_secrets(
        {"token": "abc", "message": "failed abc https://example/mcp/token=abc?q=1"},
        secrets=("abc",),
    )
    serialized = str(redacted)
    assert "abc" not in serialized
    assert "<redacted>" in serialized


def _raw_5m() -> pd.DataFrame:
    rows = []
    for bar_end in EXPECTED_5M_BAR_ENDS:
        rows.append(
            {
                "ts_code": "600000.SH",
                "trade_time": f"2010-01-04 {bar_end}:00",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "vol": 100.0,
                "amount": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_proxy_direct_5m_normalization_has_exact_bar_contract() -> None:
    five = normalize_tushare_proxy_5m(_raw_5m(), provider_symbol="600000.SH")
    assert is_complete_5m_day(five)
    assert len(five) == 48
    assert tuple(five["bar_end"]) == EXPECTED_5M_BAR_ENDS
    assert five["volume"].eq(100.0).all()


def test_proxy_5m_unit_contract_selects_unique_identity_scales() -> None:
    minute = pd.DataFrame({"vol": [600.0, 400.0], "amount": [6_000.0, 4_000.0]})
    daily = pd.DataFrame({"vol": [10.0], "amount": [10.0]})

    result = _evaluate_5m_unit_contract(minute, daily)

    assert result["raw_volume_unit"] == "share"
    assert result["raw_amount_unit"] == "CNY"
    assert result["canonical_volume_scale"] == 1.0
    assert result["canonical_amount_scale"] == 1.0


class _CutoffClient:
    def __init__(self) -> None:
        self.config = TushareProxyConfig(token="unit-test-secret", url="https://example.test/api")
        self.history_requests: list[HistoryPageFetchRequest] = []

    def fetch_frame(self, *, api_name: str, params: dict, fields: str):
        if api_name == "trade_cal":
            frame = pd.DataFrame(
                [{"exchange": "SSE", "cal_date": "20260713", "is_open": "1", "pretrade_date": "20260710"}]
            )
        elif api_name == "daily":
            frame = pd.DataFrame(
                [{"ts_code": "600000.SH", "trade_date": "20260713", "open": 10, "high": 10, "low": 10, "close": 10, "pre_close": 10, "vol": 1, "amount": 1}]
            )
        else:  # pragma: no cover - protects the fixture contract
            raise AssertionError(api_name)
        return SimpleNamespace(frame=frame, response_sha256=f"{api_name}-sha")

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        self.history_requests.append(request)
        return HistoryPageResult(
            provider="tushare_proxy",
            request=request,
            raw_data=_raw_5m().assign(trade_time=lambda frame: frame["trade_time"].str.replace("2010-01-04", "2026-07-13")),
            fields=("ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"),
            row_count=48,
            min_timestamp="2026-07-13 09:35:00",
            max_timestamp="2026-07-13 15:00:00",
            next_end_at="",
            response_sha256="5m-sha",
            is_complete=True,
            request_metadata_without_token={},
        )


def test_existing_cutoff_without_5m_proof_is_revalidated_without_relocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(tmp_path))
    paths = ensure_qdp_v3_layout(tmp_path)
    cutoff_path = paths.metadata / "tushare_proxy_bootstrap_cutoff.json"
    old_lock = {
        "status": "locked",
        "bootstrap_cutoff": "2026-07-13",
        "frequency": "1m",
        "minute_proofs": [{"provider_symbol": "600000.SH", "row_count": 241}],
    }
    cutoff_path.write_text(json.dumps(old_lock), encoding="utf-8")

    client = _CutoffClient()
    result = lock_bootstrap_cutoff(
        requested_cutoff="2026-07-13",
        workspace_root=tmp_path,
        client=client,
    )

    proof_path = paths.metadata / "tushare_proxy_bootstrap_cutoff_5m_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert result["status"] == "revalidated_existing_lock"
    assert proof["frequency"] == "5m"
    assert proof["minute_proofs"][0]["row_count"] == 48
    assert proof["minute_proofs"][0]["max_timestamp"] == "2026-07-13 15:00:00"
    assert json.loads(cutoff_path.read_text(encoding="utf-8")) == old_lock
    assert client.history_requests[0].start_at == "2026-07-13 00:00:00"
    assert client.history_requests[0].end_at == "2026-07-13 23:59:59"


def test_existing_cutoff_with_5m_proof_returns_5m_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(tmp_path))
    paths = ensure_qdp_v3_layout(tmp_path)
    cutoff_path = paths.metadata / "tushare_proxy_bootstrap_cutoff.json"
    proof_path = paths.metadata / "tushare_proxy_bootstrap_cutoff_5m_proof.json"
    old_lock = {
        "status": "locked",
        "bootstrap_cutoff": "2026-07-13",
        "frequency": "1m",
        "minute_proofs": [{"provider_symbol": "600000.SH", "row_count": 241}],
    }
    five_minute_proof = {
        "status": "revalidated_existing_lock",
        "bootstrap_cutoff": "2026-07-13",
        "frequency": "5m",
        "minute_proofs": [
            {
                "provider_symbol": "600000.SH",
                "row_count": 48,
                "max_timestamp": "2026-07-13 15:00:00",
            }
        ],
    }
    cutoff_path.write_text(json.dumps(old_lock), encoding="utf-8")
    proof_path.write_text(json.dumps(five_minute_proof), encoding="utf-8")

    result = lock_bootstrap_cutoff(
        requested_cutoff="2026-07-13",
        workspace_root=tmp_path,
        client=_CutoffClient(),
    )

    assert result["status"] == "already_locked_5m_proved"
    assert result["frequency"] == "5m"
    assert result["minute_proofs"][0]["row_count"] == 48
    assert result["minute_proofs"][0]["max_timestamp"] == "2026-07-13 15:00:00"
    assert json.loads(cutoff_path.read_text(encoding="utf-8")) == old_lock


def test_existing_cutoff_with_incomplete_5m_proof_is_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(tmp_path))
    paths = ensure_qdp_v3_layout(tmp_path)
    cutoff_path = paths.metadata / "tushare_proxy_bootstrap_cutoff.json"
    proof_path = paths.metadata / "tushare_proxy_bootstrap_cutoff_5m_proof.json"
    cutoff_path.write_text(
        json.dumps({"status": "locked", "bootstrap_cutoff": "2026-07-13"}),
        encoding="utf-8",
    )
    proof_path.write_text(
        json.dumps(
            {
                "status": "revalidated_existing_lock",
                "bootstrap_cutoff": "2026-07-13",
                "frequency": "5m",
                "minute_proofs": [
                    {
                        "provider_symbol": "600000.SH",
                        "row_count": 47,
                        "max_timestamp": "2026-07-13 15:00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    client = _CutoffClient()
    result = lock_bootstrap_cutoff(
        requested_cutoff="2026-07-13",
        workspace_root=tmp_path,
        client=client,
    )

    assert result["status"] == "revalidated_existing_lock"
    assert result["frequency"] == "5m"
    assert result["minute_proofs"][0]["row_count"] == 48
    assert len(client.history_requests) == 1


def test_official_alias_can_prove_provider_restatement_empty_history() -> None:
    workspace = Path(__file__).resolve().parents[3]
    aliases = _official_restatement_aliases(
        ["300114.SZ", "302132.SZ"],
        workspace_root=workspace,
    )
    assert aliases["300114.SZ"] == ("302132.SZ",)

    client = _CutoffClient()
    evidence = _prove_provider_symbol_restatement(
        client=client,
        provider_symbol="300114.SZ",
        aliases=aliases["300114.SZ"],
        start_at="2016-01-01 00:00:00",
        end_at="2016-12-31 23:59:59",
    )

    assert evidence["reason"] == "provider_current_code_restatement"
    assert evidence["history_provider_symbol"] == "302132.SZ"
    assert client.history_requests[0].page_size == 1


def test_two_legitimate_empty_intraday_waves_merge_without_timestamp_schema() -> None:
    prior = pd.DataFrame(columns=["provider_symbol"])
    current = pd.DataFrame(columns=["provider_symbol"])

    merged, timestamp_column = _merge_intraday_range_frames(
        prior,
        current,
        symbol="300114.SZ",
    )

    assert merged.empty
    assert list(merged.columns) == ["provider_symbol"]
    assert timestamp_column == ""


class _SuccessfulEmptyHistoryClient:
    def __init__(self) -> None:
        self.config = TushareProxyConfig(
            token="unit-test-secret",
            url="https://example.test/api",
        )

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        fields = (
            "ts_code",
            "trade_time",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        )
        return HistoryPageResult(
            provider="tushare_proxy",
            request=request,
            raw_data=pd.DataFrame(columns=list(fields)),
            fields=fields,
            row_count=0,
            min_timestamp="",
            max_timestamp="",
            next_end_at="",
            response_sha256="a" * 64,
            is_complete=True,
            request_metadata_without_token={},
        )

    def operational_metrics(self) -> dict:
        return {"request_count": 1, "success_count": 1, "error_rate": 0.0}

    def close_thread_session(self) -> None:
        return None


class _NoFetchHistoryClient:
    def __init__(self) -> None:
        self.config = TushareProxyConfig(
            token="unit-test-secret",
            url="https://example.test/api",
        )
        self.request_count = 0

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        self.request_count += 1
        raise AssertionError("completed staging must not issue another page request")

    def operational_metrics(self) -> dict:
        return {"request_count": self.request_count, "success_count": 0, "error_rate": 0.0}


def test_successful_empty_intraday_provider_gap_is_quarantined_not_failed(
    tmp_path: Path,
) -> None:
    result = ingest_tushare_proxy_intraday(
        symbols=["000005.SZ"],
        start_date="2010-01-01",
        end_date="2019-12-31",
        workspace_root=tmp_path,
        client=_SuccessfulEmptyHistoryClient(),
        lifecycle_ranges={"000005.SZ": ("1990-12-10", "2023-06-30")},
        max_workers=1,
        minimum_free_bytes=0,
    )

    assert result["status"] == "completed"
    assert result["completed_count"] == 1
    ref = get_raw_partition(
        "tushare_proxy_intraday_5m_raw",
        partition_field="provider_symbol",
        partition_value="000005.SZ",
        workspace_root=tmp_path,
    )
    assert ref is not None
    assert ref.row_count == 0
    assert ref.quality_tier == "quarantined"
    receipt = read_raw_receipt(ref)
    assert receipt["quarantined_empty"] is True
    assert (
        receipt["quarantined_empty_evidence"]["reason"]
        == "provider_successful_empty_with_lifecycle_overlap"
    )
    assert not list(
        (ensure_qdp_v3_layout(tmp_path).raw / "tushare_proxy_intraday_5m_raw").rglob(
            "*.parquet"
        )
    )


def test_completed_intraday_staging_resumes_without_repeating_the_first_page(
    tmp_path: Path,
) -> None:
    symbol = "600000.SH"
    job_id = "resume-complete-page"
    request = HistoryPageFetchRequest(
        provider_symbol=symbol,
        start_at="2010-01-04 00:00:00",
        end_at="2010-01-04 23:59:59",
    )
    result = HistoryPageResult(
        provider="tushare_proxy",
        request=request,
        raw_data=_raw_5m().iloc[::-1].reset_index(drop=True),
        fields=("ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"),
        row_count=48,
        min_timestamp="2010-01-04 09:35:00",
        max_timestamp="2010-01-04 15:00:00",
        next_end_at="",
        response_sha256="b" * 64,
        is_complete=True,
        request_metadata_without_token={},
    )
    task_root = _staging_root(job_id, workspace_root=tmp_path) / symbol
    _write_staged_page(task_root=task_root, page_number=1, result=result)
    state = HistoricalTaskState(
        task_id=symbol,
        status="running",
        cursor_end_at="",
        page_count=1,
        row_count=48,
    )
    job = HistoricalJobState(
        job_id=job_id,
        provider="tushare_proxy",
        mode="historical",
        domain="intraday",
        start_date="2010-01-04",
        end_date="2010-01-04",
        frequency="5m",
        task_ids=[symbol],
        tasks={symbol: state},
    )
    client = _NoFetchHistoryClient()

    _, outcome = _run_intraday_symbol(
        symbol=symbol,
        client=client,
        job=job,
        job_guard=threading.RLock(),
        workspace_root=tmp_path,
        lifecycle_ranges={symbol: ("1999-11-10", "")},
        restatement_aliases={},
        minimum_free_bytes=0,
    )

    assert outcome == "completed"
    assert client.request_count == 0
    assert state.status == "completed"
    assert state.row_count == 48
    assert not task_root.exists()


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
        connection.execute("CREATE TABLE actual_5m (security_id VARCHAR, trade_date VARCHAR)")
        connection.execute("CREATE TABLE explained_5m (security_id VARCHAR, trade_date VARCHAR)")
        expected = [
            (security, date)
            for date in ("2026-07-09", "2026-07-10", "2026-07-13")
            for security in ("S1", "S2")
        ]
        connection.executemany("INSERT INTO expected VALUES (?, ?)", expected)
        connection.executemany(
            "INSERT INTO actual_5m VALUES (?, ?)",
            [("S1", "2026-07-09"), ("S2", "2026-07-09")],
        )
        connection.executemany(
            "INSERT INTO explained_5m VALUES (?, ?)",
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
