from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_data_platform.qdp_v3.constants import RAW_TUSHARE_PROXY_INTRADAY_5M
from quant_data_platform.qdp_v3.historical import (
    _iter_intraday_capture_evidence,
    _write_intraday_capture_evidence,
    plan_intraday_residuals,
)
from quant_data_platform.qdp_v3.manifest import sha256_file
from quant_data_platform.qdp_v3.storage import write_raw_partition


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "brain_type": "main"}),
        encoding="utf-8",
    )
    return workspace


def _one_bar(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider_symbol": [symbol],
            "trade_date": ["2010-01-04"],
            "bar_time": ["09:35:00"],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1000.0],
        }
    )


def _old_positive_partition(workspace: Path, symbol: str):
    ref, _ = write_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        partition_field="provider_symbol",
        partition_value=symbol,
        frame=_one_bar(symbol),
        receipt={
            "provider": "tushare_proxy",
            "quality_tier": "provisional",
            "start_at": "2010-01-01 00:00:00",
            "end_at": "2010-12-31 23:59:59",
        },
        workspace_root=workspace,
    )
    return ref


def test_positive_old_plus_empty_residual_is_append_only_known_gap(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    ref = _old_positive_partition(workspace, "600000.SH")
    receipt_sha_before = sha256_file(ref.receipt_path)

    evidence_path = _write_intraday_capture_evidence(
        raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        provider_symbol="600000.SH",
        capture_start_at="2011-01-01 00:00:00",
        capture_end_at="2011-12-31 23:59:59",
        capture_kind="empty_provider_gap",
        captured_row_count=0,
        result_content_sha256=ref.content_sha256,
        network_pages=[
            {
                "page_number": 1,
                "response_sha256": "a" * 64,
                "row_count": 0,
                "min_timestamp": "",
                "max_timestamp": "",
            }
        ],
        empty_gap_evidence={
            "reason": "provider_successful_empty_with_lifecycle_overlap"
        },
        workspace_root=workspace,
    )

    assert evidence_path.exists()
    assert sha256_file(ref.receipt_path) == receipt_sha_before
    plan = plan_intraday_residuals(
        symbols=["600000.SH"],
        start_date="2010-01-01",
        end_date="2011-12-31",
        primary_raw_domains=(),
        fallback_raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        workspace_root=workspace,
        lifecycle_ranges={"600000.SH": ("1999-11-10", "")},
    )
    assert plan["download_count"] == 0
    assert plan["known_provider_gap_count"] == 1
    assert plan["known_provider_gaps"] == [
        {
            "security_id": "QDP-CN-SSE-600000",
            "provider_symbol": "600000.SH",
            "start_date": "2011-01-01",
            "end_date": "2011-12-31",
            "reason": "tushare_proxy_successful_empty_provider_gap",
        }
    ]


def test_duplicate_no_new_row_extension_records_range_and_is_idempotent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    ref = _old_positive_partition(workspace, "600004.SH")
    kwargs = {
        "raw_domain": RAW_TUSHARE_PROXY_INTRADAY_5M,
        "provider_symbol": "600004.SH",
        "capture_start_at": "2011-01-01 00:00:00",
        "capture_end_at": "2011-12-31 23:59:59",
        "capture_kind": "positive_capture",
        "captured_row_count": 48,
        "result_content_sha256": ref.content_sha256,
        "network_pages": [
            {
                "page_number": 1,
                "response_sha256": "b" * 64,
                "row_count": 48,
                "min_timestamp": "2010-01-04 09:35:00",
                "max_timestamp": "2010-01-04 15:00:00",
            }
        ],
        "workspace_root": workspace,
    }
    first = _write_intraday_capture_evidence(**kwargs)
    second = _write_intraday_capture_evidence(**kwargs)
    assert first == second
    evidence = _iter_intraday_capture_evidence(
        RAW_TUSHARE_PROXY_INTRADAY_5M,
        workspace_root=workspace,
    )
    assert len(evidence) == 1
    assert evidence[0]["capture_start_date"] == "2011-01-01"
    assert evidence[0]["capture_end_date"] == "2011-12-31"

    plan = plan_intraday_residuals(
        symbols=["600004.SH"],
        start_date="2010-01-01",
        end_date="2011-12-31",
        primary_raw_domains=(),
        fallback_raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        workspace_root=workspace,
        lifecycle_ranges={"600004.SH": ("2003-04-28", "")},
    )
    assert plan["covered_symbols"] == ["600004.SH"]
    assert plan["download_count"] == 0
    assert plan["known_provider_gap_count"] == 0
