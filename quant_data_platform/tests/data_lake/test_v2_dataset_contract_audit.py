from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from quant_data_platform.lake import ResearchDataLake
from quant_data_platform.lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle
from quant_data_platform.lake.v2_dataset_contract_audit import (
    amount_unit_diagnostics,
    audit_v2_dataset_contract,
    main as audit_main,
    missing_fill_diagnostics,
)


def _market_frame(*, amount_multiplier: float = 1.0, missing: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    rows: list[dict[str, object]] = []
    for date in dates:
        for symbol, close in {"000001.SZ": 10.0, "600000.SH": 20.0}.items():
            open_value = close - 0.1
            high = close + 0.2
            low = close - 0.2
            volume = 1000.0
            amount = close * volume * amount_multiplier
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": open_value,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                }
            )
    frame = pd.DataFrame(rows)
    if missing:
        frame.loc[0, "open"] = np.nan
        frame.loc[1, ["open", "high", "low", "close"]] = np.nan
        frame.loc[2, "volume"] = 0.0
        frame.loc[3, "amount"] = 0.0
    return frame


def _save_bundle_and_pool(temp_dir: str, *, amount_multiplier: float = 1.0) -> tuple[ResearchDataLake, str, str]:
    dates = pd.date_range("2026-01-05", periods=30, freq="B")
    stocks = ["000001.SZ", "600000.SH"]
    close = pd.DataFrame(
        {
            "000001.SZ": [10.0 + idx * 0.01 for idx in range(len(dates))],
            "600000.SH": [20.0 + idx * 0.01 for idx in range(len(dates))],
        },
        index=dates,
    )
    volume = pd.DataFrame(1000.0, index=dates, columns=stocks)
    amount = close * volume * float(amount_multiplier)
    lake = ResearchDataLake(Path(temp_dir))
    record = lake.save_market_data_bundle(
        spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
        market_frames={
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": volume,
            "Amount": amount,
        },
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        benchmark_open=pd.Series(3990.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"adv20": amount.rolling(20, min_periods=1).mean(), "adv_ratio_5_20": amount * 0.0 + 1.0},
        source="synthetic",
    )
    pool = build_pool_view_from_policy_bundle(
        lake=lake,
        spec=PoolViewSpec(
            source_market_dataset_id=record.dataset_id,
            view_kind="rolling_liquidity",
            view_name="rolling_liquid2_mainboard",
            pool_name="liquid2",
            start_date="2026-01-05",
            end_date="2026-02-13",
            rebalance_every_days=5,
            adv_window=2,
        ),
    )
    return lake, record.dataset_id, pool.dataset_id


def _attach_sidecar_to_bundle(lake: ResearchDataLake, dataset_id: str, domain: str, sidecar_dataset_id: str) -> None:
    metadata = lake.describe_dataset(dataset_id)
    parameters = dict(metadata.get("parameters", {}) or {})
    sidecars = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    sidecars[str(domain)] = str(sidecar_dataset_id)
    parameters["sidecar_dataset_ids"] = sidecars
    lake._upsert_dataset(
        dataset_id=str(metadata["dataset_id"]),
        dataset_kind=str(metadata["dataset_kind"]),
        domain=str(metadata.get("domain", "") or "bronze_silver"),
        zone=str(metadata.get("zone", "") or "research"),
        source=str(metadata.get("source", "") or "synthetic"),
        spec=parameters,
        label_completeness_summary=dict(metadata.get("label_completeness_summary", {}) or {}),
        content_paths=dict(metadata.get("content_paths", {}) or {}),
        row_counts=dict(metadata.get("row_counts", {}) or {}),
        source_cache=dict(metadata.get("source_cache", {}) or {}),
        fingerprint=str(metadata["fingerprint"]),
        status=str(metadata.get("status", "") or "stored"),
    )


def test_amount_unit_diagnostics_flags_normal_and_scaled_units() -> None:
    normal = amount_unit_diagnostics(_market_frame(amount_multiplier=1.0))
    scaled_up = amount_unit_diagnostics(_market_frame(amount_multiplier=10000.0))
    scaled_down = amount_unit_diagnostics(_market_frame(amount_multiplier=0.0001))

    assert normal["amount_unit_policy"] == "as_is"
    assert normal["amount_unit_factor"] == pytest.approx(1.0)
    assert scaled_up["amount_unit_policy"] == "divide_by_10000"
    assert scaled_up["amount_unit_factor"] == pytest.approx(0.0001)
    assert scaled_down["amount_unit_policy"] == "multiply_by_10000"
    assert scaled_down["amount_unit_factor"] == pytest.approx(10000.0)


def test_missing_fill_diagnostics_counts_ohlc_and_activity_risks() -> None:
    report = missing_fill_diagnostics(_market_frame(missing=True))

    assert report["single_field_missing_rows"] >= 1
    assert report["all_ohlc_missing_rows"] == 1
    assert report["zero_or_missing_volume_rows"] == 1
    assert report["zero_or_missing_amount_rows"] == 1


def test_v2_contract_audit_reports_amount_sensitive_features_and_degraded_unit() -> None:
    with TemporaryDirectory() as temp_dir:
        lake, dataset_id, pool_id = _save_bundle_and_pool(temp_dir, amount_multiplier=10000.0)
        payload = audit_v2_dataset_contract(
            lake=lake,
            dataset_id=dataset_id,
            pool_view_id=pool_id,
            start_date="2026-01-05",
            end_date="2026-02-13",
        )

    assert payload["status"] == "v2_dataset_contract_degraded"
    assert payload["reports"]["amount_unit"]["amount_unit_policy"] == "divide_by_10000"
    assert "amount_unit" in payload["degraded"]
    assert payload["reports"]["feature_contract"]["amount_sensitive_feature_count"] >= 2
    assert {"adv20", "adv_ratio_5_20"}.issubset(set(payload["reports"]["feature_contract"]["amount_sensitive_features"]))


def test_v2_contract_audit_flags_active_pool_status_violations() -> None:
    with TemporaryDirectory() as temp_dir:
        lake, dataset_id, pool_id = _save_bundle_and_pool(temp_dir, amount_multiplier=1.0)
        lake.save_domain_dataset(
            domain="universe_snapshot",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-01-05", "2026-01-05"],
                    "symbol": ["000001.SZ", "600000.SH"],
                    "name": ["ok", "bad"],
                    "exchange": ["SZ", "SH"],
                    "board": ["main", "main"],
                    "list_status": ["L", "L"],
                    "list_date": ["2000-01-01", "2000-01-01"],
                    "delist_date": ["", ""],
                    "source": ["unit", "unit"],
                }
            ),
            spec={"dataset": "data_platform_universe_snapshot", "start_date": "2026-01-05", "end_date": "2026-01-05"},
            source="unit",
        )
        lake.save_domain_dataset(
            domain="security_status",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-01-05", "2026-01-05"],
                    "symbol": ["000001.SZ", "600000.SH"],
                    "is_st": [False, True],
                    "is_suspended": [False, False],
                    "is_delisted": [False, False],
                    "status_reason": ["", "st"],
                    "source": ["unit", "unit"],
                }
            ),
            spec={"dataset": "data_platform_security_status", "start_date": "2026-01-05", "end_date": "2026-01-05"},
            source="unit",
        )
        payload = audit_v2_dataset_contract(
            lake=lake,
            dataset_id=dataset_id,
            pool_view_id=pool_id,
            start_date="2026-01-05",
            end_date="2026-02-13",
        )

    status_report = payload["reports"]["pool_status_screen"]
    assert payload["status"] == "v2_dataset_contract_degraded"
    assert status_report["status"] == "degraded"
    assert status_report["active_is_st_rows"] > 0
    assert "pool_status_screen" in payload["degraded"]


def test_v2_contract_audit_prefers_explicit_v2_status_sidecar_over_global_security_status() -> None:
    with TemporaryDirectory() as temp_dir:
        lake, dataset_id, pool_id = _save_bundle_and_pool(temp_dir, amount_multiplier=1.0)
        dates = pd.date_range("2026-01-05", periods=30, freq="B")
        symbols = ["000001.SZ", "600000.SH"]
        lake.save_domain_dataset(
            domain="universe_snapshot",
            frame=pd.DataFrame(
                {
                    "trade_date": [date.strftime("%Y-%m-%d") for date in dates for _symbol in symbols],
                    "symbol": [symbol for _date in dates for symbol in symbols],
                    "name": ["ok" for _date in dates for _symbol in symbols],
                    "exchange": ["SZ" if symbol.endswith(".SZ") else "SH" for _date in dates for symbol in symbols],
                    "board": ["main" for _date in dates for _symbol in symbols],
                    "list_status": ["L" for _date in dates for _symbol in symbols],
                    "list_date": ["2000-01-01" for _date in dates for _symbol in symbols],
                    "delist_date": ["2099-12-31" for _date in dates for _symbol in symbols],
                    "source": ["unit" for _date in dates for _symbol in symbols],
                }
            ),
            spec={"dataset": "data_platform_universe_snapshot", "start_date": "2026-01-05", "end_date": "2026-02-13"},
            source="unit",
        )
        lake.save_domain_dataset(
            domain="security_status",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-01-05", "2026-01-05"],
                    "symbol": ["000001.SZ", "600000.SH"],
                    "is_st": [False, True],
                    "is_suspended": [False, False],
                    "is_delisted": [False, False],
                    "status_reason": ["", "st"],
                    "source": ["legacy_unit", "legacy_unit"],
                }
            ),
            spec={"dataset": "data_platform_security_status", "start_date": "2026-01-05", "end_date": "2026-01-05"},
            source="legacy_unit",
        )
        v2_status = lake.save_domain_dataset(
            domain="v2_status_sidecar",
            frame=pd.DataFrame(
                {
                    "trade_date": [date.strftime("%Y-%m-%d") for date in dates for _symbol in symbols],
                    "symbol": [symbol for _date in dates for symbol in symbols],
                    "is_listed_on_date": [True for _date in dates for _symbol in symbols],
                    "is_mainboard": [True for _date in dates for _symbol in symbols],
                    "is_common_a_share": [True for _date in dates for _symbol in symbols],
                    "is_st": [False for _date in dates for _symbol in symbols],
                    "is_suspended": [False for _date in dates for _symbol in symbols],
                    "is_delisted": [False for _date in dates for _symbol in symbols],
                    "is_tradeable": [True for _date in dates for _symbol in symbols],
                    "has_bar": [True for _date in dates for _symbol in symbols],
                    "reject_reason": ["" for _date in dates for _symbol in symbols],
                    "source": ["explicit_v2_unit" for _date in dates for _symbol in symbols],
                }
            ),
            spec={"dataset": "data_platform_v2_status_sidecar", "start_date": "2026-01-05", "end_date": "2026-02-13"},
            source="explicit_v2_unit",
        )
        _attach_sidecar_to_bundle(lake, dataset_id, "v2_status_sidecar", v2_status.dataset_id)

        payload = audit_v2_dataset_contract(
            lake=lake,
            dataset_id=dataset_id,
            pool_view_id=pool_id,
            start_date="2026-01-05",
            end_date="2026-02-13",
        )

    status_report = payload["reports"]["pool_status_screen"]
    pit_report = payload["reports"]["pit_status_source_contract"]
    assert status_report["status"] == "ok"
    assert status_report["status_source_kind"] == "v2_status_sidecar"
    assert status_report["active_is_st_rows"] == 0
    assert pit_report["status_source_kind"] == "v2_status_sidecar"
    assert pit_report["status_source_counts"] == {"explicit_v2_unit": 60}


def test_v2_contract_audit_flags_non_pit_status_source_contract() -> None:
    with TemporaryDirectory() as temp_dir:
        lake, dataset_id, pool_id = _save_bundle_and_pool(temp_dir, amount_multiplier=1.0)
        lake.save_domain_dataset(
            domain="universe_snapshot",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-02-13", "2026-02-13"],
                    "symbol": ["000001.SZ", "600000.SH"],
                    "name": ["ok", "ok"],
                    "exchange": ["SZ", "SH"],
                    "board": ["main", "main"],
                    "list_status": ["L", "L"],
                    "list_date": ["", ""],
                    "delist_date": ["", ""],
                    "source": ["unit"] * 2,
                }
            ),
            spec={"dataset": "data_platform_universe_snapshot", "start_date": "2026-02-13", "end_date": "2026-02-13"},
            source="unit",
        )
        lake.save_domain_dataset(
            domain="security_status",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-02-13", "2026-02-13"],
                    "symbol": ["000001.SZ", "600000.SH"],
                    "is_st": [False, False],
                    "is_suspended": [False, False],
                    "is_delisted": [False, False],
                    "status_reason": ["", ""],
                    "source": ["unit"] * 2,
                }
            ),
            spec={"dataset": "data_platform_security_status", "start_date": "2026-02-13", "end_date": "2026-02-13"},
            source="unit",
        )

        payload = audit_v2_dataset_contract(
            lake=lake,
            dataset_id=dataset_id,
            pool_view_id=pool_id,
            start_date="2026-01-05",
            end_date="2026-02-13",
        )

    report = payload["reports"]["pit_status_source_contract"]
    assert report["status"] == "degraded"
    assert "universe_snapshot_single_date_not_pit_daily" in report["risks"]
    assert "universe_missing_list_date" in report["risks"]
    assert "historical_delist_status_unavailable" in report["risks"]
    assert "pit_status_source_contract" in payload["degraded"]


def test_v2_contract_audit_cli_blocks_active_artifact_diff() -> None:
    with mock.patch("quant_data_platform.lake.v2_dataset_contract_audit._active_artifact_has_diff", return_value=True):
        with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
            audit_main(["--dataset-id", "policy_input_bundle__unit", "--pool-view-id", "policy_pool_view__unit"])
