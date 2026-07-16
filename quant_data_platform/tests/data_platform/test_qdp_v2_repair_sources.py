from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform.qdp_v2.repair_sources import (
    V2_ADJUST_FACTOR_COLUMNS,
    V2_MARKET_DAILY_COLUMNS,
    V2_SECURITY_STATUS_COLUMNS,
    V2_TRADING_CALENDAR_COLUMNS,
    build_mainboard_identity_tables,
    iter_adjust_factor_repairs_by_security,
    iter_market_daily_repairs_by_year,
    iter_security_status_repairs_by_year,
    tushare_daily_to_v2,
    tushare_trade_calendar_to_v2,
)
from quant_data_platform.qdp_v3.constants import (
    RAW_TUSHARE_PROXY_ADJ_FACTOR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_SUSPEND,
)
from quant_data_platform.qdp_v3.storage import RawPartitionRef, write_raw_partition


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    return workspace


def _empty_identity_config(tmp_path: Path) -> Path:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"identities": [], "symbol_history": []}), encoding="utf-8")
    return path


def _write(
    workspace: Path,
    *,
    domain: str,
    field: str,
    value: str,
    frame: pd.DataFrame,
) -> RawPartitionRef:
    ref, _ = write_raw_partition(
        raw_domain=domain,
        partition_field=field,
        partition_value=value,
        frame=frame,
        receipt={"provider": "tushare_proxy", "quality_tier": "strict"},
        workspace_root=workspace,
    )
    return ref


def _stock_basic(workspace: Path, rows: list[dict[str, object]]) -> RawPartitionRef:
    frame = pd.DataFrame(rows)
    return _write(
        workspace,
        domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
        field="as_of_date",
        value="2026-07-13",
        frame=frame,
    )


def _daily_frame(symbol: str, trade_date: str, *, close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol],
            "trade_date": [trade_date.replace("-", "")],
            "open": [close],
            "high": [close + 1.0],
            "low": [close - 1.0],
            "close": [close],
            "pre_close": [close - 0.1],
            "change": [0.1],
            "pct_chg": [1.0],
            "vol": [123.0],
            "amount": [456.0],
        }
    )


def test_trade_calendar_normalizes_trusted_rows_from_2010_and_deduplicates() -> None:
    raw = pd.DataFrame(
        {
            "exchange": ["sse", "SSE", "SSE", "SSE"],
            "cal_date": ["20100102", "20091231", "20100101", "20100101"],
            "is_open": [0, 1, 0, 0],
        }
    )

    rows = tushare_trade_calendar_to_v2(raw)

    assert tuple(rows.columns) == V2_TRADING_CALENDAR_COLUMNS
    assert rows.to_dict("records") == [
        {
            "trade_date": "2010-01-01",
            "is_open": False,
            "exchange": "SSE",
            "source": "tushare_proxy.trade_cal",
        },
        {
            "trade_date": "2010-01-02",
            "is_open": False,
            "exchange": "SSE",
            "source": "tushare_proxy.trade_cal",
        },
    ]


def test_trade_calendar_rejects_conflicting_duplicate_date() -> None:
    raw = pd.DataFrame(
        {
            "exchange": ["SSE", "SSE"],
            "cal_date": ["20100104", "20100104"],
            "is_open": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="tushare_trade_calendar_duplicate_conflict"):
        tushare_trade_calendar_to_v2(raw)


def test_mainboard_identity_tables_exclude_growth_boards(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    stock_ref = _stock_basic(
        workspace,
        [
            {
                "ts_code": "600000.SH",
                "name": "浦发银行",
                "list_date": "19991110",
                "delist_date": "",
            },
            {
                "ts_code": "300001.SZ",
                "name": "特锐德",
                "list_date": "20091030",
                "delist_date": "",
            },
            {
                "ts_code": "900901.SH",
                "name": "云赛B股",
                "list_date": "19920221",
                "delist_date": "",
            },
            {
                "ts_code": "200002.SZ",
                "name": "万科B",
                "list_date": "19920129",
                "delist_date": "",
            },
        ],
    )

    tables = build_mainboard_identity_tables(
        workspace_root=workspace,
        stock_basic_refs=[stock_ref],
        config_path=_empty_identity_config(tmp_path),
    )

    assert tables.security_identity["current_symbol"].tolist() == ["600000.SH"]
    assert tables.symbol_history["symbol"].tolist() == ["600000.SH"]
    assert tables.symbol_history.iloc[0]["effective_from"] == "1999-11-10"
    assert tables.registry.security_id_for_provider_symbol("600000.SH") == "QDP-CN-SSE-600000"


def test_code_change_history_keeps_current_symbol_as_v2_research_key(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    stock_ref = _stock_basic(
        workspace,
        [
            {
                "ts_code": "001872.SZ",
                "name": "招商港口",
                "list_date": "19930505",
                "delist_date": "",
            }
        ],
    )
    config = tmp_path / "symbol_history.json"
    config.write_text(
        json.dumps(
            {
                "identities": [
                    {
                        "security_id": "QDP-CN-SZSE-CMPORT-19930505",
                        "official_org_id": "",
                        "issuer_name": "招商港口",
                        "exchange": "SZSE",
                        "list_date": "1993-05-05",
                        "current_symbol": "001872.SZ",
                        "identity_source": "unit",
                    }
                ],
                "symbol_history": [
                    {
                        "security_id": "QDP-CN-SZSE-CMPORT-19930505",
                        "symbol": "000022.SZ",
                        "effective_from": "1993-05-05",
                        "effective_to": "2018-12-25",
                        "name_on_date": "深赤湾A",
                        "board_on_date": "MainBoard",
                        "evidence_source": "unit",
                        "official_document_hash": "",
                    },
                    {
                        "security_id": "QDP-CN-SZSE-CMPORT-19930505",
                        "symbol": "001872.SZ",
                        "effective_from": "2018-12-26",
                        "effective_to": "9999-12-31",
                        "name_on_date": "招商港口",
                        "board_on_date": "MainBoard",
                        "evidence_source": "unit",
                        "official_document_hash": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    tables = build_mainboard_identity_tables(
        stock_basic_refs=[stock_ref],
        config_path=config,
    )

    rows = tushare_daily_to_v2(
        _daily_frame("000022.SZ", "2016-01-04"),
        identity_registry=tables.registry,
    )

    assert rows["symbol"].tolist() == ["001872.SZ"]


def test_daily_repairs_stream_by_year_and_emit_only_missing_or_changed_rows(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    stock_ref = _stock_basic(
        workspace,
        [
            {
                "ts_code": "600000.SH",
                "name": "浦发银行",
                "list_date": "19991110",
                "delist_date": "",
            }
        ],
    )
    tables = build_mainboard_identity_tables(
        stock_basic_refs=[stock_ref],
        config_path=_empty_identity_config(tmp_path),
    )
    refs = [
        _write(
            workspace,
            domain=RAW_TUSHARE_PROXY_DAILY,
            field="trade_date",
            value=date,
            frame=_daily_frame("600000.SH", date, close=close),
        )
        for date, close in (
            ("2010-01-04", 10.0),
            ("2010-01-05", 11.0),
            ("2010-01-06", 12.0),
        )
    ]
    existing = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": ["2010-01-04", "2010-01-06"],
            "open": [9.0, 12.0],
            "high": [11.0, 13.0],
            "low": [9.0, 11.0],
            "close": [9.0, 12.0],
            "volume": [12_300.0, 12_300.0],
            "amount": [456_000.0, 456_000.0],
        }
    )

    batches = list(
        iter_market_daily_repairs_by_year(
            identity_registry=tables.registry,
            daily_refs=refs,
            existing_loader=lambda year: existing if year == "2010" else None,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.partition_key == "2010"
    assert batch.insert_count == 1
    assert batch.update_count == 1
    assert tuple(batch.rows.columns) == V2_MARKET_DAILY_COLUMNS
    assert batch.rows["trade_date"].tolist() == ["2010-01-04", "2010-01-05"]
    assert batch.rows["volume"].tolist() == [12_300.0, 12_300.0]
    assert batch.rows["amount"].tolist() == [456_000.0, 456_000.0]
    assert set(batch.rows["adjusted_flag"]) == {"none"}


def test_factor_repairs_normalize_first_2010_observation_and_replace_600076_pollution(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    stock_ref = _stock_basic(
        workspace,
        [
            {
                "ts_code": "600076.SH",
                "name": "康欣新材",
                "list_date": "19970526",
                "delist_date": "",
            }
        ],
    )
    tables = build_mainboard_identity_tables(
        stock_basic_refs=[stock_ref],
        config_path=_empty_identity_config(tmp_path),
    )
    factor_ref = _write(
        workspace,
        domain=RAW_TUSHARE_PROXY_ADJ_FACTOR,
        field="provider_symbol",
        value="600076.SH",
        frame=pd.DataFrame(
            {
                "ts_code": ["600076.SH"] * 4,
                "trade_date": ["20100104", "20231229", "20240102", "20241231"],
                "adj_factor": [10.0, 20.0, 20.0, 20.0],
            }
        ),
    )

    batches = list(
        iter_adjust_factor_repairs_by_security(
            identity_registry=tables.registry,
            factor_refs=[factor_ref],
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert tuple(batch.rows.columns) == V2_ADJUST_FACTOR_COLUMNS
    assert batch.rows.iloc[0]["adjust_factor"] == 1.0
    assert batch.rows.loc[batch.rows["trade_date"].eq("2023-12-29"), "adjust_factor"].iloc[0] == 2.0
    factors_2024 = batch.rows.loc[batch.rows["trade_date"].str.startswith("2024"), "adjust_factor"]
    assert factors_2024.tolist() == [2.0, 2.0]
    assert batch.insert_count == 4
    assert batch.update_count == 0


def test_security_status_combines_st_name_interval_and_explicit_suspension(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    stock_ref = _stock_basic(
        workspace,
        [
            {
                "ts_code": "600000.SH",
                "name": "浦发银行",
                "list_date": "19991110",
                "delist_date": "",
            },
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "list_date": "19910403",
                "delist_date": "",
            },
        ],
    )
    tables = build_mainboard_identity_tables(
        stock_basic_refs=[stock_ref],
        config_path=_empty_identity_config(tmp_path),
    )
    daily_ref = _write(
        workspace,
        domain=RAW_TUSHARE_PROXY_DAILY,
        field="trade_date",
        value="2010-01-04",
        frame=_daily_frame("600000.SH", "2010-01-04"),
    )
    suspend_ref = _write(
        workspace,
        domain=RAW_TUSHARE_PROXY_SUSPEND,
        field="trade_date",
        value="2010-01-04",
        frame=pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20100104"],
                "suspend_timing": ["全天"],
                "suspend_type": ["连续停牌"],
            }
        ),
    )
    name_ref = _write(
        workspace,
        domain=RAW_TUSHARE_PROXY_NAMECHANGE,
        field="provider_symbol",
        value="600000.SH",
        frame=pd.DataFrame(
            {
                "ts_code": ["600000.SH"],
                "name": ["*ST浦发"],
                "start_date": ["20100101"],
                "end_date": ["20101231"],
                "ann_date": ["20100101"],
                "change_reason": ["特别处理"],
            }
        ),
    )

    batches = list(
        iter_security_status_repairs_by_year(
            identity_registry=tables.registry,
            daily_refs=[daily_ref],
            suspend_refs=[suspend_ref],
            namechange_refs=[name_ref],
        )
    )

    assert len(batches) == 1
    rows = batches[0].rows.set_index("symbol")
    assert tuple(batches[0].rows.columns) == V2_SECURITY_STATUS_COLUMNS
    assert bool(rows.loc["600000.SH", "is_st"])
    assert rows.loc["600000.SH", "status_reason"] == "st"
    assert bool(rows.loc["000001.SZ", "is_suspended"])
    assert rows.loc["000001.SZ", "status_reason"] == "suspended"
