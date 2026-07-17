from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_data_platform.qdp_v2.current_store_repair import (
    CurrentStoreRepairError,
    EXPECTED_BAR_TIMES,
    MARKET_COLUMNS,
    _aggregate_local_1m_gap,
    _prevalidate_intraday_replacements,
    _reconcile_rejected_day,
    _resolve_tushare_token,
)


def test_tushare_proxy_token_environment_alias(monkeypatch) -> None:
    monkeypatch.delenv("QDP_TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setenv("QDP_TUSHARE_PROXY_TOKEN", "proxy-test-token")

    assert _resolve_tushare_token() == "proxy-test-token"
    assert _resolve_tushare_token("explicit-test-token") == "explicit-test-token"


def test_reconcile_rejected_day_repairs_decimal_high_and_volume_unit() -> None:
    bars = pd.DataFrame(
        {
            "symbol": "600584.SH",
            "trade_date": "2026-06-29",
            "bar_time": EXPECTED_BAR_TIMES,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.1,
            "volume": 100.0,
            "amount": 1_000.0,
            "source": "unit",
            "adjusted_flag": "none",
        }
    )
    bars.loc[0, "high"] = 102.0
    reference = {
        "open": 10.0,
        "high": 10.2,
        "low": 9.8,
        "close": 10.1,
        "volume": 48.0,
        "amount": 480.0,
    }

    repaired, audit = _reconcile_rejected_day(bars, reference)

    assert repaired.loc[:, MARKET_COLUMNS].shape == (48, len(MARKET_COLUMNS))
    assert repaired["high"].max() == 10.2
    assert repaired["low"].min() == 9.8
    assert repaired["volume"].sum() == 48.0
    assert repaired["amount"].sum() == 480.0
    assert audit["volume_scale"] == 0.01
    assert "high_decimal_shift" in audit["price_repairs"]


def test_aggregate_local_1m_gap_uses_right_closed_trading_bins() -> None:
    morning = pd.date_range("2010-08-10 09:31:00", "2010-08-10 11:30:00", freq="min")
    afternoon = pd.date_range("2010-08-10 13:01:00", "2010-08-10 15:00:00", freq="min")
    timestamps = morning.append(afternoon)
    raw = pd.DataFrame(
        {
            "\u65e5\u671f": timestamps,
            "\u5f00\u76d8": 10.0,
            "\u6700\u9ad8": 10.2,
            "\u6700\u4f4e": 9.8,
            "\u6536\u76d8": 10.1,
            "\u6210\u4ea4\u91cf(\u80a1)": 1.0,
            "\u6210\u4ea4\u989d(\u5143)": 10.0,
        }
    )

    bars = _aggregate_local_1m_gap(
        raw,
        current_symbol="001872.SZ",
        trade_date="2010-08-10",
    )

    assert bars["bar_time"].tolist() == list(EXPECTED_BAR_TIMES)
    assert bars["volume"].sum() == 240.0
    assert bars.iloc[0]["bar_time"] == "093500000"
    assert bars.iloc[-1]["bar_time"] == "150000000"


def test_intraday_partition_prevalidation_rejects_duplicate_keys(tmp_path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2020-01-02", "2020-01-02"],
            "bar_time": ["093500000", "094000000"],
        }
    ).to_parquet(first, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2021-01-04", "2021-01-04"],
            "bar_time": ["093500000", "094000000"],
        }
    ).to_parquet(second, index=False)
    context = SimpleNamespace(
        root=tmp_path,
        domain="market_intraday_5m",
        dataset_id="unit",
        shard_paths=(tmp_path / "old-first.parquet", tmp_path / "old-second.parquet"),
        manifest=SimpleNamespace(
            shards=(
                SimpleNamespace(start_date="2020-01-02", end_date="2020-12-31"),
                SimpleNamespace(start_date="2021-01-04", end_date="2021-12-31"),
            )
        ),
    )
    replacements = list(zip(context.shard_paths, (first, second), strict=True))

    mutation_id = _prevalidate_intraday_replacements(
        context,
        replacements,
        workspace=tmp_path,
    )

    assert mutation_id.startswith("shard-mutation-v1:")
    subset_mutation_id = _prevalidate_intraday_replacements(
        context,
        replacements[:1],
        workspace=tmp_path,
    )
    assert subset_mutation_id.startswith("shard-mutation-v1:")
    assert subset_mutation_id != mutation_id

    duplicate = tmp_path / "duplicate.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2020-01-02", "2020-01-02"],
            "bar_time": ["093500000", "093500000"],
        }
    ).to_parquet(duplicate, index=False)
    duplicate_context = SimpleNamespace(
        root=tmp_path,
        domain="market_intraday_5m",
        dataset_id="unit-duplicate",
        shard_paths=(tmp_path / "old-duplicate.parquet",),
        manifest=SimpleNamespace(
            shards=(SimpleNamespace(start_date="2020-01-02", end_date="2020-12-31"),)
        ),
    )

    with pytest.raises(CurrentStoreRepairError, match="primary_key_invalid"):
        _prevalidate_intraday_replacements(
            duplicate_context,
            [(duplicate_context.shard_paths[0], duplicate)],
            workspace=tmp_path,
        )
