from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform.qdp_v3.constants import (
    EXPECTED_5M_BAR_ENDS,
    RAW_EXTERNAL_QUANT_INTRADAY_5M,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_INTRADAY_5M,
)
from quant_data_platform.qdp_v3.historical import (
    plan_intraday_residuals,
    raw_partition_covers_historical_request,
)
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry
from quant_data_platform.qdp_v3.storage import (
    read_raw_receipt,
    write_empty_raw_partition,
    write_raw_partition,
)
from quant_data_platform.qdp_v3.update import _proxy_daily_expected_trade_dates, plan_update


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "brain_type": "main"}),
        encoding="utf-8",
    )
    return workspace


def _identity_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "qdp_v3_symbol_history.json"


def _one_bar(symbol: str, trade_date: str = "2010-01-04") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider_symbol": symbol,
            "trade_date": trade_date,
            "bar_time": [f"{value}:00" for value in EXPECTED_5M_BAR_ENDS],
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1000.0,
        }
    )


def test_successful_empty_raw_never_satisfies_historical_coverage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    frame = pd.DataFrame(columns=_one_bar("000005.SZ").columns)
    ref, _ = write_empty_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        partition_field="provider_symbol",
        partition_value="000005.SZ",
        frame=frame,
        receipt={
            "provider": "tushare_proxy",
            "quality_tier": "quarantined",
            "start_at": "2010-01-01 00:00:00",
            "end_at": "2019-12-31 23:59:59",
            "quarantined_empty": True,
            "quarantined_empty_evidence": {
                "reason": "provider_successful_empty_with_lifecycle_overlap"
            },
        },
        workspace_root=workspace,
    )

    assert ref.row_count == 0
    assert not raw_partition_covers_historical_request(
        ref,
        read_raw_receipt(ref),
        start_at="2010-01-01 00:00:00",
        end_at="2019-12-31 23:59:59",
    )


def test_residual_plan_uses_stable_identity_and_preserves_known_provider_gaps(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    write_raw_partition(
        raw_domain=RAW_EXTERNAL_QUANT_INTRADAY_5M,
        partition_field="provider_symbol",
        partition_value="302132.SZ",
        frame=_one_bar("302132.SZ", "2010-08-30"),
        receipt={
            "provider": "external_quant_archive",
            "quality_tier": "strict",
            "coverage_start_date": "2010-01-01",
            "coverage_end_date": "2019-12-31",
        },
        workspace_root=workspace,
    )
    empty = pd.DataFrame(columns=_one_bar("000005.SZ").columns)
    write_empty_raw_partition(
        raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        partition_field="provider_symbol",
        partition_value="000005.SZ",
        frame=empty,
        receipt={
            "provider": "tushare_proxy",
            "quality_tier": "quarantined",
            "start_at": "2010-01-01 00:00:00",
            "end_at": "2019-12-31 23:59:59",
            "quarantined_empty": True,
            "quarantined_empty_evidence": {
                "reason": "provider_successful_empty_with_lifecycle_overlap"
            },
        },
        workspace_root=workspace,
    )
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=["300114.SZ", "302132.SZ", "000005.SZ", "600000.SH"],
        config_path=_identity_config(),
    )

    plan = plan_intraday_residuals(
        symbols=["300114.SZ", "000005.SZ", "600000.SH"],
        start_date="2010-01-01",
        end_date="2019-12-31",
        primary_raw_domains=(RAW_EXTERNAL_QUANT_INTRADAY_5M,),
        fallback_raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        workspace_root=workspace,
        lifecycle_ranges={
            "300114.SZ": ("2010-08-27", ""),
            "000005.SZ": ("1990-12-10", "2023-06-30"),
            "600000.SH": ("1999-11-10", ""),
        },
        identity_registry=registry,
        trade_dates=("2010-08-30",),
    )

    assert plan["covered_symbols"] == ["300114.SZ"]
    assert plan["download_symbols"] == ["600000.SH"]
    assert plan["known_provider_gap_count"] == 1
    assert plan["known_provider_gaps"][0]["provider_symbol"] == "000005.SZ"
    assert plan["alias_coverage"] == [
        {
            "security_id": "QDP-CN-SZSE-AVICCAC-20100827",
            "requested_symbol": "300114.SZ",
            "covered_by_symbol": "302132.SZ",
            "raw_domain": RAW_EXTERNAL_QUANT_INTRADAY_5M,
        }
    ]


def test_local_archive_update_plan_is_local_first_with_residual_only(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source = tmp_path / "5minute.zip"
    plan = plan_update(
        as_of_date="2026-07-13",
        workspace_root=workspace,
        bootstrap=True,
        start_date="2010-01-01",
        historical_provider="external-quant-archive",
        source_paths=(source,),
    )

    assert plan["status"] == "planned"
    stages = {item["stage"]: item for item in plan["stages"]}
    local = stages["intraday_5m_local_archive_then_tushare_residual"]
    assert local["provider"] == "external_quant_archive_with_tushare_proxy_residual"
    assert local["source_paths"] == [str(source)]
    assert not any(
        item.get("stage") == "intraday_5m" and item.get("provider") == "tushare_proxy"
        for item in plan["stages"]
    )


def test_trusted_daily_absence_excludes_suspension_from_minute_residual(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    for trade_date, symbols in (
        ("2012-01-03", ("600000.SH",)),
        ("2012-01-04", ()),
    ):
        frame = pd.DataFrame(
            {
                "ts_code": list(symbols),
                "trade_date": [trade_date.replace("-", "")] * len(symbols),
                "open": [10.0] * len(symbols),
                "high": [10.1] * len(symbols),
                "low": [9.9] * len(symbols),
                "close": [10.0] * len(symbols),
            }
        )
        write_raw_partition(
            raw_domain=RAW_TUSHARE_PROXY_DAILY,
            partition_field="trade_date",
            partition_value=trade_date,
            frame=frame,
            receipt={"provider": "tushare_proxy", "quality_tier": "strict"},
            workspace_root=workspace,
        )
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=["600000.SH"],
        config_path=_identity_config(),
    )
    expected, evidence = _proxy_daily_expected_trade_dates(
        symbols=["600000.SH"],
        trade_dates=["2012-01-03", "2012-01-04"],
        identity_registry=registry,
        workspace_root=workspace,
    )

    assert evidence["status"] == "complete"
    assert expected["QDP-CN-SSE-600000"] == ("2012-01-03",)
    plan = plan_intraday_residuals(
        symbols=["600000.SH"],
        start_date="2012-01-03",
        end_date="2012-01-04",
        primary_raw_domains=(),
        workspace_root=workspace,
        lifecycle_ranges={"600000.SH": ("1999-11-10", "")},
        trade_dates=("2012-01-03", "2012-01-04"),
        expected_trade_dates_by_symbol=expected,
        identity_registry=registry,
    )
    assert plan["download_groups"] == [
        {
            "start_date": "2012-01-03",
            "end_date": "2012-01-03",
            "trade_dates": ["2012-01-03"],
            "symbols": ["600000.SH"],
        }
    ]


def test_residual_plan_emits_only_the_uncovered_tail_not_the_whole_wave(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    write_raw_partition(
        raw_domain=RAW_EXTERNAL_QUANT_INTRADAY_5M,
        partition_field="provider_symbol",
        partition_value="600000.SH",
        frame=_one_bar("600000.SH", "2026-03-27"),
        receipt={
            "provider": "external_quant_archive",
            "quality_tier": "strict",
            "coverage_start_date": "2010-01-01",
            "coverage_end_date": "2026-03-27",
        },
        workspace_root=workspace,
    )

    plan = plan_intraday_residuals(
        symbols=["600000.SH"],
        start_date="2020-01-01",
        end_date="2026-07-13",
        primary_raw_domains=(RAW_EXTERNAL_QUANT_INTRADAY_5M,),
        workspace_root=workspace,
        lifecycle_ranges={"600000.SH": ("1999-11-10", "")},
        trade_dates=("2026-03-27", "2026-03-30"),
    )

    assert plan["partially_covered_symbols"] == ["600000.SH"]
    assert plan["download_groups"] == [
        {
            "start_date": "2026-03-30",
            "end_date": "2026-03-30",
            "trade_dates": ["2026-03-30"],
            "symbols": ["600000.SH"],
        }
    ]


def test_internal_incomplete_archive_day_remains_an_exact_residual(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = _one_bar("600000.SH", "2012-01-03")
    malformed = _one_bar("600000.SH", "2012-01-04").iloc[:-1].copy()
    last = _one_bar("600000.SH", "2012-01-05")
    write_raw_partition(
        raw_domain=RAW_EXTERNAL_QUANT_INTRADAY_5M,
        partition_field="provider_symbol",
        partition_value="600000.SH",
        frame=pd.concat([first, malformed, last], ignore_index=True),
        receipt={
            "provider": "external_quant_archive",
            "quality_tier": "strict",
            "coverage_start_date": "2012-01-03",
            "coverage_end_date": "2012-01-05",
            "complete_stock_day_count": 2,
            "complete_trade_dates": ["2012-01-03", "2012-01-05"],
            "incomplete_trade_dates": ["2012-01-04"],
        },
        workspace_root=workspace,
    )

    plan = plan_intraday_residuals(
        symbols=["600000.SH"],
        start_date="2012-01-03",
        end_date="2012-01-05",
        primary_raw_domains=(RAW_EXTERNAL_QUANT_INTRADAY_5M,),
        workspace_root=workspace,
        lifecycle_ranges={"600000.SH": ("1999-11-10", "")},
        trade_dates=("2012-01-03", "2012-01-04", "2012-01-05"),
    )

    assert plan["partially_covered_symbols"] == ["600000.SH"]
    assert plan["download_groups"] == [
        {
            "start_date": "2012-01-04",
            "end_date": "2012-01-04",
            "trade_dates": ["2012-01-04"],
            "symbols": ["600000.SH"],
        }
    ]


def test_local_archive_update_requires_an_explicit_source_path(tmp_path: Path) -> None:
    plan = plan_update(
        as_of_date="2026-07-13",
        workspace_root=_workspace(tmp_path),
        bootstrap=True,
        historical_provider="external-quant-archive",
    )

    assert plan["status"] == "blocked"
    assert plan["blocker"] == "qdp_v3_local_archive_bootstrap_requires_source_path"


def test_bootstrap_capture_routes_exact_residual_spans_without_using_old_full_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_data_platform.qdp_v3.update as update_module

    workspace = _workspace(tmp_path)
    local_calls: list[dict[str, object]] = []
    proxy_calls: list[dict[str, object]] = []
    free_calls: list[dict[str, object]] = []
    fake_adapter = types.ModuleType("quant_data_platform.qdp_v3.external_quant_5m")

    def import_local(**kwargs: object) -> dict[str, object]:
        local_calls.append(dict(kwargs))
        return {"status": "completed", "imported_symbol_count": 2}

    fake_adapter.import_external_quant_5m = import_local  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake_adapter.__name__, fake_adapter)
    monkeypatch.setattr(
        update_module,
        "run_tushare_proxy_compatibility_gate",
        lambda **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        update_module,
        "lock_bootstrap_cutoff",
        lambda **_kwargs: {"bootstrap_cutoff": "2026-07-13"},
    )
    monkeypatch.setattr(
        update_module,
        "_proxy_reference_call",
        lambda **_kwargs: {"status": "completed"},
    )
    monkeypatch.setattr(
        update_module,
        "_calendar_dates_read_only",
        lambda **_kwargs: (["2012-01-03", "2026-03-30"], "unit"),
    )

    def inventory(*, mainboard_only: bool, **_kwargs: object):
        if mainboard_only:
            return ["000005.SZ", "600000.SH"], {
                "000005.SZ": ("1990-12-10", "2018-12-31"),
                "600000.SH": ("1999-11-10", ""),
            }
        return ["000005.SZ", "600000.SH"], {}

    monkeypatch.setattr(update_module, "proxy_symbol_inventory", inventory)
    plan_calls: dict[str, int] = {}

    def residual_plan(**kwargs: object) -> dict[str, object]:
        start = str(kwargs["start_date"])
        if start == "2026-03-28":
            return {"download_count": 0, "download_groups": []}
        plan_calls[start] = plan_calls.get(start, 0) + 1
        if plan_calls[start] > 1:
            return {"download_count": 0, "download_groups": []}
        if start < "2020-01-01":
            return {
                "download_count": 1,
                "download_groups": [
                    {"start_date": "2012-01-01", "end_date": "2012-12-31", "symbols": ["000005.SZ"]}
                ],
            }
        return {
            "download_count": 1,
            "download_groups": [
                {"start_date": "2026-03-28", "end_date": "2026-07-13", "symbols": ["600000.SH"]}
            ],
        }

    monkeypatch.setattr(update_module, "plan_intraday_residuals", residual_plan)

    def proxy_ingest(**kwargs: object) -> dict[str, object]:
        proxy_calls.append(dict(kwargs))
        return {"status": "completed"}

    def free_ingest(**kwargs: object) -> dict[str, object]:
        free_calls.append(dict(kwargs))
        return {"status": "completed"}

    monkeypatch.setattr(update_module, "ingest_tushare_proxy_intraday", proxy_ingest)
    monkeypatch.setattr(update_module, "ingest_intraday_5m", free_ingest)
    recorded: list[tuple[str, dict[str, object]]] = []
    status, blocker = update_module._run_bootstrap_capture(
        plan={
            "start_date": "2010-01-01",
            "as_of_date": "2026-07-13",
            "historical_provider": "external-quant-archive",
            "source_paths": [str(tmp_path / "5minute.zip")],
        },
        record=lambda stage, result: recorded.append((stage, result)),
        workspace_root=workspace,
    )

    assert status == "completed"
    assert blocker is None
    assert local_calls[0]["symbols"] == ["000005.SZ", "600000.SH"]
    assert local_calls[0]["workers"] == 8
    assert len(proxy_calls) == 1
    assert proxy_calls[0]["symbols"] == ("000005.SZ",)
    assert proxy_calls[0]["start_date"] == "2012-01-01"
    assert proxy_calls[0]["end_date"] == "2012-12-31"
    assert "local_residual" in str(proxy_calls[0]["job_id"])
    assert "__bootstrap_2010_2019_" not in str(proxy_calls[0]["job_id"])
    assert len(free_calls) == 1
    assert free_calls[0]["symbols"] == ("600000.SH",)
    assert free_calls[0]["trade_dates"] == ["2026-03-30"]
