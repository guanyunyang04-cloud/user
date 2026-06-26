from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd
import pytest

from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.build_canonical_policy_bundle import (
    BuildCanonicalPolicyBundleConfig,
    build_canonical_policy_bundle,
)
from daily_research.data_lake.import_traditional_baostock_v2_snapshot import (
    import_traditional_baostock_v2_snapshot,
    main as import_main,
)
from daily_research.data_lake.policy_input_loader import load_policy_inputs_from_lake
from daily_research.data_platform.contracts import DataDomain


def _write_snapshot(root: Path) -> Path:
    snapshot = root / "baostock_fixture"
    snapshot.mkdir(parents=True)
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    symbols = ["000001.SZ", "600000.SH"]
    rows = [{"date": date, "code": symbol} for date in dates for symbol in symbols]
    base = pd.DataFrame(rows)
    manifest = {
        "schema_version": 2,
        "snapshot_id": "baostock_fixture",
        "dataset": {
            "date_min": "2026-01-05",
            "date_max": "2026-01-07",
            "trade_date_count": 3,
            "security_count": 2,
            "daily_universe_rows": 6,
            "daily_bar_rows": 6,
            "daily_status_rows": 6,
            "stock_industry_rows": 6,
            "daily_metrics_rows": 6,
        },
        "quality": {"failure_count": 0, "tradeable_rows": 5},
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (snapshot / "quality_report.json").write_text(json.dumps({"failure_count": 0}, ensure_ascii=False), encoding="utf-8")

    bars = base.copy()
    bars["open"] = [10.0, 20.0, 10.2, 20.1, 10.4, 20.2]
    bars["high"] = bars["open"] + 0.2
    bars["low"] = bars["open"] - 0.2
    bars["close"] = bars["open"] + 0.1
    bars["volume"] = 1000.0
    bars["amount"] = bars["close"] * bars["volume"]
    bars["tradestatus"] = 1
    bars["isST"] = 0
    bars["source"] = "baostock"
    bars.to_parquet(snapshot / "daily_bars.parquet", index=False)

    universe = base.copy()
    universe["name_on_date"] = ["平安银行", "浦发银行"] * 3
    universe["ipo_date"] = "2000-01-01"
    universe["out_date"] = ""
    universe["is_listed_on_date"] = True
    universe["is_mainboard"] = True
    universe["is_common_a_share"] = True
    universe["is_st_on_date"] = False
    universe["is_suspended_on_date"] = False
    universe["has_bar"] = True
    universe["is_tradeable"] = True
    universe.loc[(universe["date"].eq(pd.Timestamp("2026-01-07"))) & (universe["code"].eq("600000.SH")), "is_tradeable"] = False
    universe["reject_reason"] = ""
    universe.loc[~universe["is_tradeable"], "reject_reason"] = "suspended_on_date"
    universe.to_parquet(snapshot / "daily_universe.parquet", index=False)

    status = base.copy()
    status["has_bar"] = True
    status["is_suspended_like"] = False
    status["is_tradeable"] = True
    status.loc[(status["date"].eq(pd.Timestamp("2026-01-07"))) & (status["code"].eq("600000.SH")), "is_suspended_like"] = True
    status.loc[(status["date"].eq(pd.Timestamp("2026-01-07"))) & (status["code"].eq("600000.SH")), "is_tradeable"] = False
    status.to_parquet(snapshot / "daily_status.parquet", index=False)

    industry = base.copy()
    industry["name_on_date"] = ["平安银行", "浦发银行"] * 3
    industry["industry"] = "银行"
    industry["industry_classification"] = "申万"
    industry["industry_update_date"] = "2026-01-01"
    industry["source"] = "baostock:month-start-ffill"
    industry.to_parquet(snapshot / "stock_industry.parquet", index=False)

    metrics = base.copy()
    metrics["turn"] = [1.0, 2.0, 1.1, 2.1, 1.2, 2.2]
    metrics["pctChg"] = [0.1, 0.2, 0.11, 0.21, 0.12, 0.22]
    metrics["peTTM"] = [8.0, 9.0, 8.1, 9.1, 8.2, 9.2]
    metrics["pbMRQ"] = [1.0, 1.2, 1.1, 1.3, 1.2, 1.4]
    metrics["psTTM"] = [2.0, 2.2, 2.1, 2.3, 2.2, 2.4]
    metrics["pcfNcfTTM"] = [3.0, 3.2, 3.1, 3.3, 3.2, 3.4]
    metrics["source"] = "baostock"
    metrics.to_parquet(snapshot / "daily_metrics.parquet", index=False)
    return snapshot


def _save_benchmark_bundle(lake: ResearchDataLake) -> str:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    close = pd.DataFrame(10.0, index=dates, columns=["000001.SZ"])
    record = lake.save_market_data_bundle(
        spec={"pool_name": "benchmark_source", "benchmark": "000300.SH", "source": "unit"},
        market_frames={
            "Open": close,
            "High": close + 0.1,
            "Low": close - 0.1,
            "Close": close,
            "Volume": close * 100.0,
            "Amount": close * 1000.0,
        },
        benchmark_close=pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH"),
        benchmark_open=pd.Series([3990.0, 4000.0, 4010.0], index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=["000001.SZ"]),
        feature_frames={},
        source="unit",
    )
    return record.dataset_id


def test_import_traditional_baostock_v2_snapshot_registers_bundle_sidecars_and_loader_inputs() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot = _write_snapshot(root / "traditional")
        lake = ResearchDataLake(root / "lake")
        benchmark_dataset_id = _save_benchmark_bundle(lake)

        payload = import_traditional_baostock_v2_snapshot(
            lake=lake,
            source_root=snapshot,
            benchmark_source_dataset_id=benchmark_dataset_id,
            start_date="2026-01-05",
            end_date="2026-01-07",
            run_tag="unit_import",
        )
        metadata = lake.describe_dataset(payload["dataset_id"])
        prepared = load_policy_inputs_from_lake(
            lake=lake,
            dataset_id=payload["dataset_id"],
            start_date="2026-01-05",
            end_date="2026-01-07",
        )

    assert payload["status"] == "ok"
    assert metadata["source"] == "traditional_baostock_v2_1_import"
    assert metadata["parameters"]["source_snapshot_id"] == "baostock_fixture"
    assert set(payload["sidecar_dataset_ids"]) == {"universe_snapshot", "v2_status_sidecar", "industry_concept", "valuation"}
    assert payload["summary"]["industry_frequency"] == "month-start-ffill"
    assert payload["summary"]["metrics_coverage_on_tradeable"]["peTTM"] == pytest.approx(1.0)
    assert prepared.raw_cache_meta["dataset_id"] == payload["dataset_id"]
    assert "turn" in prepared.derived_frames
    assert "peTTM" in prepared.derived_frames
    assert "industry_daily" in prepared.metadata_frames
    assert prepared.metadata_summary["valuation_sidecar"]["available"] is True
    assert prepared.metadata_summary["industry_sidecar"]["industry_frequency"] == "month-start-ffill"


def test_policy_loader_normalizes_legacy_valuation_metric_schema(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    dates = pd.to_datetime(["2010-01-04", "2010-01-05"])
    market_record = lake.save_domain_dataset(
        domain=DataDomain.MARKET_DAILY,
        frame=pd.DataFrame(
            {
                "trade_date": ["2010-01-04", "2010-01-04", "2010-01-04", "2010-01-05", "2010-01-05", "2010-01-05"],
                "symbol": ["000001.SZ", "600000.SH", "000300.SH", "000001.SZ", "600000.SH", "000300.SH"],
                "open": [10.0, 20.0, 3000.0, 10.1, 20.1, 3010.0],
                "high": [10.2, 20.2, 3020.0, 10.3, 20.3, 3030.0],
                "low": [9.8, 19.8, 2990.0, 10.0, 20.0, 3000.0],
                "close": [10.1, 20.1, 3010.0, 10.2, 20.2, 3020.0],
                "volume": [1000.0, 2000.0, 0.0, 1100.0, 2100.0, 0.0],
                "amount": [10100.0, 40200.0, 0.0, 11220.0, 42420.0, 0.0],
            }
        ),
        spec={"dataset": "data_platform_market_daily", "start_date": "2010-01-04", "end_date": "2010-01-05"},
        source="unit",
    )
    valuation_record = lake.save_domain_dataset(
        domain=DataDomain.VALUATION,
        frame=pd.DataFrame(
            {
                "trade_date": ["2010-01-04", "2010-01-05"],
                "symbol": ["000001.SZ", "000001.SZ"],
                "turn": [None, None],
                "turnover_rate": [1.0, 1.1],
                "peTTM": [None, None],
                "pe": [8.0, 8.1],
                "pbMRQ": [None, None],
                "pb": [1.0, 1.1],
            }
        ),
        spec={"dataset": "data_platform_valuation", "start_date": "2010-01-04", "end_date": "2010-01-05"},
        source="unit",
    )
    bundle = build_canonical_policy_bundle(
        BuildCanonicalPolicyBundleConfig(
            lake_root=tmp_path / "lake",
            market_daily_dataset_id=market_record.dataset_id,
            sidecar_dataset_ids={"valuation": valuation_record.dataset_id},
            start_date="2010-01-04",
            end_date="2010-01-05",
        )
    )
    prepared = load_policy_inputs_from_lake(
        lake=lake,
        dataset_id=bundle.dataset_id,
        start_date="2010-01-04",
        end_date="2010-01-05",
        universe=["000001.SZ", "600000.SH"],
        min_trading_days=2,
    )

    assert "turn" in prepared.derived_frames
    assert "peTTM" in prepared.derived_frames
    assert "pbMRQ" in prepared.derived_frames
    assert float(prepared.derived_frames["turn"].loc[dates[0], "000001.SZ"]) == pytest.approx(1.0)
    assert float(prepared.derived_frames["peTTM"].loc[dates[0], "000001.SZ"]) == pytest.approx(8.0)
    assert float(prepared.derived_frames["pbMRQ"].loc[dates[0], "000001.SZ"]) == pytest.approx(1.0)


def test_import_traditional_baostock_v2_snapshot_cli_blocks_active_artifact_diff() -> None:
    with mock.patch(
        "daily_research.data_lake.import_traditional_baostock_v2_snapshot._active_artifact_has_diff",
        return_value=True,
    ):
        with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
            import_main(["--source-root", "unit"])
