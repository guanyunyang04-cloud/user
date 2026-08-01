from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quant_data_platform.qdp_v2.research_event_update as research_update
from quant_data_platform.qdp_v2.research_event_update import (
    ResearchEventUpdateError,
    _announcement_raw_source_paths,
    _eastmoney_announcement_path,
    _merge_reports,
    _normalize_cninfo_announcements,
    _normalize_eastmoney_announcements,
    _normalize_eastmoney_reports,
    _normalize_tushare_report_year,
    _parquet_safe_provider_frame,
    _report_source_key,
    _report_year_source_coverage,
)

from quant_data_platform import providers


def _open_dates() -> np.ndarray:
    return np.asarray(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        dtype="datetime64[ns]",
    )


def test_report_rc_rows_split_report_identity_from_forecast_quarters() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_date": ["20240102", "20240102"],
            "report_title": ["盈利预测：更新", "盈利预测：更新"],
            "org_name": ["某某证券股份有限公司", "某某证券股份有限公司"],
            "author_name": ["甲", "甲"],
            "quarter": ["2024Q1", "2024Q2"],
            "eps": [1.0, 1.2],
            "pe": [10.0, 9.0],
            "rating": ["买入", "买入"],
        }
    )

    reports, forecasts, statistics = _normalize_tushare_report_year(
        raw,
        open_dates=_open_dates(),
    )

    assert len(reports) == 1
    assert len(forecasts) == 2
    assert reports.loc[0, "feature_available_date"] == "2024-01-03"
    assert forecasts["forecast_quarter"].tolist() == ["2024Q1", "2024Q2"]
    assert statistics["raw_row_count"] == 2
    assert statistics["unique_report_count"] == 1


def test_eastmoney_metadata_merges_without_repeating_report_weight() -> None:
    tushare_raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_date": ["20240102"],
            "report_title": ["盈利预测：更新"],
            "org_name": ["某某证券股份有限公司"],
            "author_name": ["甲"],
            "quarter": ["2024Q1"],
            "eps": [1.0],
            "rating": ["买入"],
        }
    )
    tushare, _, _ = _normalize_tushare_report_year(
        tushare_raw,
        open_dates=_open_dates(),
    )
    eastmoney = _normalize_eastmoney_reports(
        pd.DataFrame(
            {
                "_query_symbol": ["000001.SZ"],
                "publishDate": ["2024-01-02 00:00:00"],
                "title": ["盈利预测更新"],
                "orgName": ["某某证券"],
                "researcher": ["甲,乙"],
                "infoCode": ["AP1"],
                "sRatingName": ["买入"],
                "attachSize": [100],
                "attachPages": [8],
            }
        ),
        open_dates=_open_dates(),
    )

    merged = _merge_reports(tushare, eastmoney)

    assert len(merged) == 1
    assert bool(merged.loc[0, "tushare_present"])
    assert bool(merged.loc[0, "eastmoney_present"])
    assert merged.loc[0, "analyst"] == "乙;甲"
    assert json.loads(merged.loc[0, "eastmoney_info_codes"]) == ["AP1"]
    assert merged.loc[0, "url"].endswith("H3_AP1_1.pdf")


def test_report_identity_normalizes_punctuation_and_legal_suffixes() -> None:
    assert _report_source_key(
        "000001.SZ",
        "2024-01-02",
        "盈利预测：更新",
        "某某证券股份有限公司",
    ) == _report_source_key(
        "000001.SZ",
        "2024-01-02",
        "盈利预测更新",
        "某某证券",
    )


def test_report_source_coverage_distinguishes_empty_and_partial_years() -> None:
    assert (
        _report_year_source_coverage(
            2021,
            {"raw_row_count": 0, "earliest_report_date": "", "latest_report_date": ""},
        )["tushare_source_coverage_status"]
        == "source_unavailable"
    )
    partial = _report_year_source_coverage(
        2023,
        {
            "raw_row_count": 10,
            "earliest_report_date": "2023-11-13",
            "latest_report_date": "2023-12-31",
        },
    )
    assert partial["tushare_source_coverage_status"] == "partial_year_span"
    assert partial["tushare_source_unavailable_is_not_zero_reports"]


def test_report_source_coverage_uses_closed_daily_ledger() -> None:
    result = _report_year_source_coverage(
        2021,
        {
            "raw_row_count": 12,
            "earliest_report_date": "2021-02-01",
            "latest_report_date": "2021-12-30",
        },
        {
            "requested_date_count": 365,
            "terminal_date_count": 365,
            "failed_date_count": 0,
            "pending_date_count": 0,
        },
    )

    assert result["tushare_source_coverage_status"] == "complete_daily_task_ledger"
    assert not result["tushare_source_unavailable_is_not_zero_reports"]


def test_report_daily_download_pages_confirms_empty_and_resumes_failure(
    tmp_path,
    monkeypatch,
) -> None:
    requested_dates = ("2024-01-01", "2024-01-02", "2024-01-03")
    calls: Counter[tuple[str, int]] = Counter()
    fail_once = {"enabled": True}

    class FakeClient:
        def __init__(self, token, workspace_root):
            assert token == "fixture-token"
            assert workspace_root == tmp_path

        def fetch(self, api_name, *, params, fields):
            assert api_name == "report_rc"
            assert int(params["limit"]) == 3_000
            report_date = pd.to_datetime(params["start_date"]).strftime("%Y-%m-%d")
            assert params["start_date"] == params["end_date"]
            offset = int(params["offset"])
            calls[(report_date, offset)] += 1
            if report_date == "2024-01-01":
                count = 3_000 if offset == 0 else 7
                return pd.DataFrame({"ts_code": ["000001.SZ"] * count})
            if report_date == "2024-01-02":
                return pd.DataFrame(columns=["ts_code"])
            if fail_once["enabled"]:
                raise RuntimeError("transient fixture failure")
            return pd.DataFrame({"ts_code": ["000003.SZ"]})

    monkeypatch.setattr(research_update, "_report_request_dates", lambda: requested_dates)
    monkeypatch.setattr(
        research_update, "_resolve_tushare_token", lambda workspace: "fixture-token"
    )
    monkeypatch.setattr(research_update, "_TushareClient", FakeClient)

    with pytest.raises(ResearchEventUpdateError, match="daily_tasks_incomplete:1"):
        research_update.download_tushare_reports(
            workspace_root=tmp_path,
            max_workers=1,
        )

    assert calls[("2024-01-01", 0)] == 1
    assert calls[("2024-01-01", 3_000)] == 1
    assert calls[("2024-01-02", 0)] == 2
    assert calls[("2024-01-03", 0)] == 1

    fail_once["enabled"] = False
    result = research_update.download_tushare_reports(
        workspace_root=tmp_path,
        max_workers=1,
    )

    assert result["status"] == "completed"
    assert calls[("2024-01-01", 0)] == 1
    assert calls[("2024-01-02", 0)] == 2
    assert calls[("2024-01-03", 0)] == 2
    assert result["days"]["2024-01-02"]["status"] == "confirmed_empty"
    assert all(not value.startswith("2026-") for value in requested_dates)


def test_report_state_atomic_replace_retries_transient_windows_lock(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def flaky_write(path, payload):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("fixture lock")

    monkeypatch.setattr(research_update, "atomic_write_json", flaky_write)
    monkeypatch.setattr(research_update.time, "sleep", lambda seconds: None)

    research_update._write_state(tmp_path, {"status": "downloading"})

    assert calls["count"] == 3


def test_provider_json_cache_serializes_mixed_object_columns() -> None:
    result = _parquet_safe_provider_frame(
        pd.DataFrame(
            {
                "ratingChange": [3, ""],
                "author": [["1.甲"], ""],
            }
        )
    )

    assert result["ratingChange"].tolist() == ["3", ""]
    assert result["author"].tolist() == ['["1.甲"]', ""]


def test_announcement_sources_share_key_and_next_open_availability() -> None:
    cninfo = _normalize_cninfo_announcements(
        pd.DataFrame(
            {
                "_query_symbol": ["000001.SZ"],
                "trade_date": ["2024-01-02"],
                "title": ["公司股票可能被终止上市的风险提示公告"],
                "announcement_type_codes": ["0101||0102"],
                "cninfo_announcement_id": ["1"],
            }
        ),
        open_dates=_open_dates(),
    )
    eastmoney = _normalize_eastmoney_announcements(
        pd.DataFrame(
            {
                "_query_symbol": ["000001.SZ"],
                "notice_date": ["2024-01-02"],
                "title": ["公司股票可能被终止上市的风险提示公告"],
                "column_name": ["风险提示"],
                "column_code": ["3"],
                "art_code": ["AN1"],
                "stock_code": ["000001"],
            }
        ),
        open_dates=_open_dates(),
    )

    assert cninfo.loc[0, "announcement_id"] == eastmoney.loc[0, "announcement_id"]
    assert cninfo.loc[0, "feature_available_date"] == "2024-01-03"
    assert cninfo.loc[0, "category"] == "risk_warning"


def test_cninfo_fetch_uses_org_id_and_reads_all_pages(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url, headers, data, timeout):
        calls.append(dict(data))
        page = int(data["pageNum"])
        count = 30 if page == 1 else 1
        rows = [
            {
                "secCode": "000001",
                "announcementId": f"{page}-{index}",
                "announcementTitle": f"公告 {page}-{index}",
                "announcementTime": 1674144000000,
                "announcementType": "0101||0102",
                "adjunctUrl": f"file/{page}-{index}.PDF",
                "adjunctSize": 1,
            }
            for index in range(count)
        ]
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"totalAnnouncement": 31, "announcements": rows},
        )

    class FakeSession:
        def post(self, url, headers, data, timeout):
            return fake_post(url, headers, data, timeout)

        def close(self) -> None:
            return None

    monkeypatch.setattr(providers.requests, "Session", FakeSession)

    result = providers._fetch_cninfo_announcements(
        symbol="000001.SZ",
        start_date="2023-01-01",
        end_date="2023-01-31",
        page_size=100,
        max_pages=0,
        org_id="gssz0000001",
    )

    assert len(result) == 31
    assert len(calls) == 2
    assert all(call["stock"] == "000001,gssz0000001" for call in calls)
    assert all(call["pageSize"] == 30 for call in calls)


def test_announcement_source_paths_exclude_stale_non_fallback_cache(
    tmp_path,
) -> None:
    from quant_data_platform.qdp_v2.research_event_update import (
        _cninfo_announcement_path,
    )

    cninfo = _cninfo_announcement_path(tmp_path, "000001.SZ")
    selected_fallback = _eastmoney_announcement_path(tmp_path, "000002.SZ")
    stale_fallback = _eastmoney_announcement_path(tmp_path, "000003.SZ")
    for path in (cninfo, selected_fallback, stale_fallback):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    selected = _announcement_raw_source_paths(
        tmp_path,
        fallback_symbols=["000002.SZ"],
    )

    assert selected["cninfo"] == [cninfo]
    assert selected["eastmoney"] == [selected_fallback]
    assert stale_fallback not in selected["eastmoney"]
