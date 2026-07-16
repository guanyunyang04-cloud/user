from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant_data_platform.domains.contracts import (
    HistoryPageFetchRequest,
    HistoryPageResult,
)
from quant_data_platform.qdp_v2.intraday_repair import (
    EXPECTED_5M_BAR_ENDS,
    _MemoryGuard,
)
from quant_data_platform.qdp_v2.tushare_gap_repair import (
    _normalize_gap_days,
    build_gap_tasks,
    run_tushare_gap_download,
)
from quant_data_platform.qdp_v3.intraday import normalize_tushare_proxy_5m
from quant_data_platform.tushare_proxy import TushareProxyQuotaError


def _write_inputs(
    root: Path,
    *,
    selected: list[str],
    missing: list[str],
    rejected: dict[str, int],
) -> tuple[Path, Path]:
    patch = root / "patch_manifest.json"
    patch.write_text(
        json.dumps(
            {
                "selected_symbols": selected,
                "eligible_missing_archive": missing,
            }
        ),
        encoding="utf-8",
    )
    database = root / "planner.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE symbol_receipts(symbol TEXT PRIMARY KEY, rejected_days INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO symbol_receipts(symbol,rejected_days) VALUES (?,?)",
            sorted(rejected.items()),
        )
        connection.commit()
    finally:
        connection.close()
    return patch, database


def _raw_day(symbol: str, trade_date: str) -> pd.DataFrame:
    rows = [
        {
            "ts_code": symbol,
            "trade_time": f"{trade_date} {bar_end}:00",
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "vol": 100.0,
            "amount": 1_000.0,
        }
        for bar_end in EXPECTED_5M_BAR_ENDS
    ]
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


class _SuccessfulClient:
    def __init__(self, *, allow_fetch: bool = True) -> None:
        self.config = SimpleNamespace(token="unit-test-secret")
        self.allow_fetch = allow_fetch
        self.requests: list[HistoryPageFetchRequest] = []

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        if not self.allow_fetch:
            raise AssertionError("completed symbol must not be downloaded again")
        self.requests.append(request)
        trade_date = request.end_at[:10]
        frame = _raw_day(request.provider_symbol, trade_date)
        return HistoryPageResult(
            provider="tushare_proxy",
            request=request,
            raw_data=frame,
            fields=tuple(frame.columns),
            row_count=len(frame),
            min_timestamp=f"{trade_date} 09:35:00",
            max_timestamp=f"{trade_date} 15:00:00",
            next_end_at="",
            response_sha256="a" * 64,
            is_complete=True,
            request_metadata_without_token={},
        )

    def operational_metrics(self) -> dict[str, object]:
        return {
            "request_count": len(self.requests),
            "success_count": len(self.requests),
            "error_rate": 0.0,
        }

    def close_thread_session(self) -> None:
        return None


class _QuotaClient(_SuccessfulClient):
    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        self.requests.append(request)
        raise TushareProxyQuotaError("daily quota exhausted")


def test_gap_tasks_use_full_ranges_only_for_missing_or_rejected(tmp_path: Path) -> None:
    patch, database = _write_inputs(
        tmp_path,
        selected=["600000.SH", "600001.SH", "600002.SH"],
        missing=["000001.SZ"],
        rejected={"600000.SH": 0, "600001.SH": 3, "600002.SH": 0},
    )
    tasks, inventory = build_gap_tasks(
        patch_manifest_path=patch,
        planner_database_path=database,
        workspace_root=tmp_path,
        lifecycle_ranges={
            "000001.SZ": ("1991-04-03", "2026-07-13"),
            "600000.SH": ("1999-11-10", "2026-07-13"),
            "600001.SH": ("2015-05-01", "2026-07-13"),
            "600002.SH": ("2000-01-01", "2025-12-31"),
        },
    )

    assert [item.to_dict() for item in tasks] == [
        {
            "symbol": "000001.SZ",
            "start_date": "2010-01-01",
            "end_date": "2026-07-13",
            "scope": "full_unresolved",
        },
        {
            "symbol": "600001.SH",
            "start_date": "2015-05-01",
            "end_date": "2026-07-13",
            "scope": "full_unresolved",
        },
        {
            "symbol": "600000.SH",
            "start_date": "2026-06-29",
            "end_date": "2026-07-13",
            "scope": "recent_tail",
        },
    ]
    assert inventory["outside_lifecycle_count"] == 1
    assert inventory["full_unresolved_symbol_count"] == 2


def test_gap_download_is_compact_resumable_and_secret_free(tmp_path: Path) -> None:
    patch, database = _write_inputs(
        tmp_path,
        selected=["600000.SH"],
        missing=[],
        rejected={"600000.SH": 0},
    )
    lifecycle = {"600000.SH": ("1999-11-10", "2026-07-13")}
    first_client = _SuccessfulClient()
    first = run_tushare_gap_download(
        patch_manifest_path=patch,
        planner_database_path=database,
        workspace_root=tmp_path,
        lifecycle_ranges=lifecycle,
        client=first_client,
        minimum_free_bytes=0,
    )

    assert first["status"] == "completed"
    assert first["completed"] == 1
    assert first["valid_days"] == 1
    assert first["captured_rows"] == 48
    assert len(first_client.requests) == 1
    runtime_root = Path(first["runtime_root"])
    assert sorted(path.name for path in runtime_root.iterdir()) == [
        "gap.sqlite3",
        "state.json",
        "tushare_gap_5m.zip",
    ]
    with zipfile.ZipFile(first["archive_path"]) as archive:
        assert archive.namelist() == ["sh600000.csv"]
        exported = pd.read_csv(archive.open("sh600000.csv"))
    assert len(exported) == 48
    assert exported["datetime"].iloc[0] == "2026-07-13 09:35"

    second_client = _SuccessfulClient(allow_fetch=False)
    second = run_tushare_gap_download(
        patch_manifest_path=patch,
        planner_database_path=database,
        workspace_root=tmp_path,
        lifecycle_ranges=lifecycle,
        client=second_client,
        minimum_free_bytes=0,
    )
    assert second["status"] == "completed"
    assert second_client.requests == []
    assert second["archive"]["archive_sha256"] == first["archive"]["archive_sha256"]

    for path in runtime_root.iterdir():
        assert b"unit-test-secret" not in path.read_bytes()


def test_vectorized_days_preserve_49_bar_fallback_and_reject_zero_turnover() -> None:
    auction = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_time": "2026-07-10 09:30:00",
                "open": 9.9,
                "high": 10.0,
                "low": 9.7,
                "close": 9.95,
                "vol": 50.0,
                "amount": 500.0,
            }
        ]
    )
    valid_49 = pd.concat(
        [auction, _raw_day("600000.SH", "2026-07-10")],
        ignore_index=True,
    )
    zero_turnover = _raw_day("600000.SH", "2026-07-13")
    zero_turnover.loc[:, ["vol", "amount"]] = 0.0
    normalized = normalize_tushare_proxy_5m(
        pd.concat([valid_49, zero_turnover], ignore_index=True),
        provider_symbol="600000.SH",
    ).sort_values(["trade_date", "bar_end"], kind="stable")

    frame, valid_days, rejected_days, sample = _normalize_gap_days(
        normalized,
        canonical_symbol="600000.SH",
        source_name="trusted_test_source",
        guard=_MemoryGuard(0.0, 0.0),
    )

    assert valid_days == 1
    assert rejected_days == 1
    assert len(frame) == 48
    assert frame["trade_date"].unique().tolist() == ["2026-07-10"]
    first = frame.loc[frame["bar_time"].eq("093500000")].iloc[0]
    assert first["open"] == 9.9
    assert first["volume"] == 150.0
    assert frame["source"].unique().tolist() == ["trusted_test_source"]
    assert sample == [
        {"trade_date": "2026-07-13", "reason": "zero_turnover_stock_day"}
    ]


def test_quota_leaves_symbol_pending_for_same_job_resume(tmp_path: Path) -> None:
    patch, database = _write_inputs(
        tmp_path,
        selected=["600000.SH"],
        missing=[],
        rejected={"600000.SH": 0},
    )
    result = run_tushare_gap_download(
        patch_manifest_path=patch,
        planner_database_path=database,
        workspace_root=tmp_path,
        lifecycle_ranges={"600000.SH": ("1999-11-10", "2026-07-13")},
        client=_QuotaClient(),
        minimum_free_bytes=0,
        max_workers=1,
    )

    assert result["status"] == "paused_quota"
    assert result["completed"] == 0
    assert result["pending"] == 1
    assert result["failed"] == 0
    assert not Path(result["archive_path"]).exists()


def test_full_only_scope_reuses_job_without_downloading_recent_tail(
    tmp_path: Path,
) -> None:
    patch, database = _write_inputs(
        tmp_path,
        selected=["600000.SH"],
        missing=["000001.SZ"],
        rejected={"600000.SH": 0},
    )
    client = _SuccessfulClient()
    result = run_tushare_gap_download(
        patch_manifest_path=patch,
        planner_database_path=database,
        workspace_root=tmp_path,
        lifecycle_ranges={
            "000001.SZ": ("1991-04-03", "2026-07-13"),
            "600000.SH": ("1999-11-10", "2026-07-13"),
        },
        client=client,
        minimum_free_bytes=0,
        task_scopes=("full_unresolved",),
    )

    assert result["status"] == "completed"
    assert result["selected_scopes"] == ["full_unresolved"]
    assert result["selected_task_count"] == 1
    assert result["completed"] == 1
    assert result["pending"] == 0
    assert result["global_status_counts"] == {"completed": 1, "pending": 1}
    assert [request.provider_symbol for request in client.requests] == ["000001.SZ"]
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert archive.namelist() == ["sz000001.csv"]
