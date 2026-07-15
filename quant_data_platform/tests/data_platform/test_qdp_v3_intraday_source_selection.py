from __future__ import annotations

import pandas as pd

from quant_data_platform.qdp_v3.constants import EXPECTED_5M_BAR_ENDS, RAW_INTRADAY_5M_SELECTED
from quant_data_platform.qdp_v3.intraday_build import (
    _select_historical_source_day,
    _selected_incremental_frame,
)
from quant_data_platform.qdp_v3.storage import write_raw_partition


class _Registry:
    def symbol_for_date(self, security_id: str, trade_date: str) -> str:
        assert security_id == "sec-sh-600000"
        assert trade_date == "2012-01-04"
        return "600000.SH"


def _day(*, source: str, price_offset: float = 0.0, volume: float = 100.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, bar_end in enumerate(EXPECTED_5M_BAR_ENDS):
        open_price = 10.0 + price_offset + index * 0.001
        close_price = open_price + 0.001
        rows.append(
            {
                "provider_symbol": "600000.SH",
                "trade_date": "2012-01-04",
                "bar_end": bar_end,
                "open": open_price,
                "high": close_price + 0.001,
                "low": open_price - 0.001,
                "close": close_price,
                "volume": volume,
                "amount": volume * close_price,
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def _daily(day: pd.DataFrame) -> dict[str, float]:
    ordered = day.sort_values("bar_end")
    return {
        "open": float(ordered.iloc[0]["open"]),
        "high": float(ordered["high"].max()),
        "low": float(ordered["low"].min()),
        "close": float(ordered.iloc[-1]["close"]),
        "volume": float(ordered["volume"].sum()),
        "amount": float(ordered["amount"].sum()),
    }


def _select(
    local: pd.DataFrame,
    proxy: pd.DataFrame,
    daily: dict[str, float] | None,
    *,
    legacy: pd.DataFrame | None = None,
    free: pd.DataFrame | None = None,
):
    return _select_historical_source_day(
        local_day=local,
        legacy_day=legacy,
        proxy_day=proxy,
        free_day=free,
        security_id="sec-sh-600000",
        trade_date="2012-01-04",
        registry=_Registry(),
        daily=daily,
    )


def test_complete_free_source_day_wins_before_tushare_residual() -> None:
    free = _day(source="baostock")
    free["quality_tier"] = "strict"
    proxy = _day(source="tushare_proxy", price_offset=0.01)

    selected, reason, evidence = _select(
        pd.DataFrame(), proxy, _daily(free), free=free
    )

    assert len(selected) == 48
    assert set(selected["source"]) == {"baostock"}
    assert reason == "baostock_complete_free_residual_structurally_valid"
    assert evidence["selected_source"] == "baostock"
    assert "proxy" not in evidence


def test_conflicting_free_source_aliases_are_rejected_before_proxy_fallback() -> None:
    free = _day(source="baostock")
    free["quality_tier"] = "strict"
    conflicting_alias = _day(source="baostock", price_offset=0.02)
    conflicting_alias["provider_symbol"] = "600001.SH"
    conflicting_alias["quality_tier"] = "strict"
    free = pd.concat([free, conflicting_alias], ignore_index=True)
    proxy = _day(source="tushare_proxy")

    selected, _, evidence = _select(pd.DataFrame(), proxy, _daily(proxy), free=free)

    assert len(selected) == 48
    assert set(selected["source"]) == {"tushare_proxy"}
    assert evidence["free"]["reason"] == "provider_code_restatement_value_conflict"


def test_complete_local_archive_day_wins_without_comparing_or_combining_proxy() -> None:
    local = _day(source="external_quant_archive")
    legacy = _day(source="legacy_v2_5m_migrated", price_offset=0.002)
    proxy = _day(source="tushare_proxy", price_offset=0.001)

    selected, reason, evidence = _select(local, proxy, _daily(local), legacy=legacy)

    assert len(selected) == 48
    assert set(selected["source"]) == {"external_quant_archive"}
    assert reason == "external_quant_archive_complete_structurally_valid"
    assert evidence["selected_source"] == "external_quant_archive"
    assert "legacy" not in evidence
    assert "proxy" not in evidence


def test_complete_legacy_v2_day_wins_before_free_and_tushare_residual() -> None:
    legacy = _day(source="legacy_v2_5m_migrated")
    free = _day(source="baostock", price_offset=0.02)
    free["quality_tier"] = "strict"
    proxy = _day(source="tushare_proxy", price_offset=0.03)

    selected, reason, evidence = _select(
        pd.DataFrame(), proxy, _daily(free), legacy=legacy, free=free
    )

    assert len(selected) == 48
    assert set(selected["source"]) == {"legacy_v2_5m_migrated"}
    assert reason == "legacy_v2_5m_migrated_complete_structurally_valid"
    assert evidence["selected_source"] == "legacy_v2_5m_migrated"
    assert evidence["legacy"]["evidence"]["conflict"] is True
    assert evidence["legacy"]["daily_reconciliation_role"] == "cross_source_diagnostic_only"
    assert "free" not in evidence
    assert "proxy" not in evidence


def test_invalid_legacy_v2_day_falls_back_to_complete_free_day() -> None:
    legacy = _day(source="legacy_v2_5m_migrated").iloc[:-1].copy()
    free = _day(source="baostock")
    free["quality_tier"] = "strict"

    selected, _, evidence = _select(
        pd.DataFrame(), pd.DataFrame(), _daily(free), legacy=legacy, free=free
    )

    assert len(selected) == 48
    assert set(selected["source"]) == {"baostock"}
    assert evidence["legacy"]["reason"] == "legacy_v2_5m_migrated_5m_structure_invalid"
    assert evidence["selected_source"] == "baostock"


def test_conflicting_legacy_v2_aliases_quarantine_the_whole_source_day() -> None:
    legacy = _day(source="legacy_v2_5m_migrated")
    conflicting_alias = _day(source="legacy_v2_5m_migrated", price_offset=0.02)
    conflicting_alias["provider_symbol"] = "600001.SH"
    legacy = pd.concat([legacy, conflicting_alias], ignore_index=True)

    selected, reason, evidence = _select(
        pd.DataFrame(), pd.DataFrame(), _daily(_day(source="reference")), legacy=legacy
    )

    assert selected.empty
    assert reason == "historical_5m_all_sources_rejected"
    assert evidence["legacy"]["reason"] == "provider_code_restatement_value_conflict"


def test_partial_bars_from_all_historical_sources_are_never_spliced() -> None:
    complete = _day(source="reference")
    local = _day(source="external_quant_archive").iloc[:12].copy()
    legacy = _day(source="legacy_v2_5m_migrated").iloc[12:24].copy()
    free = _day(source="baostock").iloc[24:36].copy()
    free["quality_tier"] = "strict"
    proxy = _day(source="tushare_proxy").iloc[36:].copy()

    selected, reason, evidence = _select(
        local, proxy, _daily(complete), legacy=legacy, free=free
    )

    assert selected.empty
    assert reason == "historical_5m_all_sources_rejected"
    assert evidence["local"]["row_count"] == 12
    assert evidence["legacy"]["row_count"] == 12
    assert evidence["free"]["row_count"] == 12
    assert evidence["proxy"]["row_count"] == 12


def test_tushare_is_whole_day_residual_when_local_day_is_incomplete() -> None:
    proxy = _day(source="tushare_proxy")
    local = _day(source="external_quant_archive").iloc[:-1].copy()

    selected, reason, evidence = _select(local, proxy, _daily(proxy))

    assert len(selected) == 48
    assert set(selected["source"]) == {"tushare_proxy"}
    assert reason == "tushare_proxy_residual_complete_and_daily_consistent"
    assert evidence["local"]["reason"] == "external_quant_archive_5m_structure_invalid"
    assert evidence["selected_source"] == "tushare_proxy"


def test_cross_source_daily_difference_does_not_reject_complete_local_day() -> None:
    proxy = _day(source="tushare_proxy")
    local = _day(source="external_quant_archive", volume=1_000.0)

    selected, reason, evidence = _select(local, proxy, _daily(proxy))

    assert len(selected) == 48
    assert set(selected["source"]) == {"external_quant_archive"}
    assert reason == "external_quant_archive_complete_structurally_valid"
    assert evidence["local"]["evidence"]["conflict"] is True
    assert evidence["local"]["daily_reconciliation_role"] == "cross_source_diagnostic_only"
    assert "proxy" not in evidence


def test_complementary_partial_sources_are_never_spliced_into_a_complete_day() -> None:
    complete = _day(source="reference")
    local = _day(source="external_quant_archive").iloc[:24].copy()
    proxy = _day(source="tushare_proxy").iloc[24:].copy()

    selected, reason, evidence = _select(local, proxy, _daily(complete))

    assert selected.empty
    assert reason == "historical_5m_all_sources_rejected"
    assert evidence["local"]["row_count"] == 24
    assert evidence["proxy"]["row_count"] == 24


def test_local_day_needs_no_cross_source_daily_proof_but_tushare_still_does() -> None:
    local = _day(source="external_quant_archive")
    proxy = _day(source="tushare_proxy")

    selected, reason, evidence = _select(local, proxy, None)

    assert len(selected) == 48
    assert reason == "external_quant_archive_complete_structurally_valid"
    assert evidence["local"]["evidence"]["reason"] == "daily_reference_missing"
    assert evidence["local"]["daily_reconciliation_role"] == "cross_source_diagnostic_only"
    assert "proxy" not in evidence


def test_tushare_day_requires_same_source_daily_proof() -> None:
    proxy = _day(source="tushare_proxy")

    selected, reason, evidence = _select(pd.DataFrame(), proxy, None)

    assert selected.empty
    assert reason == "historical_5m_all_sources_rejected"
    assert evidence["proxy"]["reason"] == "tushare_proxy_complete_daily_proof_missing"


def test_selected_free_source_can_fill_a_gap_before_nominal_proxy_cutoff(tmp_path) -> None:
    raw = _day(source="baostock")
    raw["quality_tier"] = "strict"
    raw["source_selection_reason"] = "baostock_complete_fallback"
    ref, _ = write_raw_partition(
        raw_domain=RAW_INTRADAY_5M_SELECTED,
        partition_field="symbol_month",
        partition_value="600000.SH_201201",
        frame=raw,
        receipt={"provider": "baostock", "quality_tier": "strict"},
        workspace_root=tmp_path,
    )

    selected = _selected_incremental_frame(
        [ref],
        start_date="2010-01-01",
        end_date="2026-07-13",
        bootstrap_cutoff="2026-07-13",
    )

    assert len(selected) == 48
    assert set(selected["trade_date"]) == {"2012-01-04"}
    assert set(selected["source"]) == {"baostock"}
