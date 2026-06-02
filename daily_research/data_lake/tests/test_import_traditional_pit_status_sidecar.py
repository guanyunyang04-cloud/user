from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd
import pytest

from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.import_traditional_pit_status_sidecar import (
    build_traditional_pit_status_sidecar_frame,
    import_traditional_pit_status_sidecar,
    main as import_main,
)


def _write_traditional_snapshot(root: Path) -> Path:
    snapshot = root / "snap"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "traditional_fixture",
                "dataset": {"date_min": "2026-01-05", "date_max": "2026-01-06", "trade_date_count": 2},
                "quality": {"failure_count": 0, "tradeable_rows": 2, "st_rows": 1, "suspended_like_rows": 1},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (snapshot / "quality_report.json").write_text(json.dumps({"failure_count": 0}, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "code": "600000.SH",
                "name_on_date": "ok",
                "ipo_date": "2000-01-01",
                "out_date": "",
                "is_listed_on_date": True,
                "is_mainboard": True,
                "is_common_a_share": True,
                "is_st_on_date": False,
                "is_suspended_on_date": False,
                "has_bar": True,
                "is_tradeable": True,
                "reject_reason": "",
            },
            {
                "date": "2026-01-05",
                "code": "000001.SZ",
                "name_on_date": "st",
                "ipo_date": "2000-01-01",
                "out_date": "",
                "is_listed_on_date": True,
                "is_mainboard": True,
                "is_common_a_share": True,
                "is_st_on_date": True,
                "is_suspended_on_date": False,
                "has_bar": True,
                "is_tradeable": False,
                "reject_reason": "st_on_date",
            },
            {
                "date": "2026-01-06",
                "code": "600001.SH",
                "name_on_date": "suspended",
                "ipo_date": "2000-01-01",
                "out_date": "",
                "is_listed_on_date": True,
                "is_mainboard": True,
                "is_common_a_share": True,
                "is_st_on_date": False,
                "is_suspended_on_date": True,
                "has_bar": True,
                "is_tradeable": False,
                "reject_reason": "suspended_on_date",
            },
            {
                "date": "2026-01-06",
                "code": "600002.SH",
                "name_on_date": "delisted",
                "ipo_date": "2000-01-01",
                "out_date": "2026-01-01",
                "is_listed_on_date": False,
                "is_mainboard": True,
                "is_common_a_share": True,
                "is_st_on_date": False,
                "is_suspended_on_date": False,
                "has_bar": True,
                "is_tradeable": False,
                "reject_reason": "not_listed_on_date",
            },
        ]
    ).to_parquet(snapshot / "daily_universe.parquet", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps({"snapshot_id": "traditional_fixture", "snapshot_path": str(snapshot)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _save_market_bundle(lake: ResearchDataLake) -> str:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    stocks = ["600000.SH", "000001.SZ", "600001.SH"]
    close = pd.DataFrame(10.0, index=dates, columns=stocks)
    record = lake.save_market_data_bundle(
        spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
        market_frames={
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        },
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"score_none": close * 0.0},
        source="synthetic",
    )
    return record.dataset_id


def test_traditional_pit_status_sidecar_maps_contract_and_market_intersection() -> None:
    with TemporaryDirectory() as temp_dir:
        snapshot_root = _write_traditional_snapshot(Path(temp_dir) / "traditional")
        market_keys = pd.DataFrame(
            [
                {"trade_date": "2026-01-05", "symbol": "600000.SH"},
                {"trade_date": "2026-01-05", "symbol": "000001.SZ"},
                {"trade_date": "2026-01-06", "symbol": "600001.SH"},
            ]
        )
        frame, summary = build_traditional_pit_status_sidecar_frame(
            snapshot_root=snapshot_root,
            market_keys=market_keys,
            start_date="2026-01-05",
            end_date="2026-01-06",
        )

    by_symbol = frame.set_index("symbol")
    assert list(frame.columns) == [
        "symbol",
        "trade_date",
        "is_listed_on_date",
        "is_mainboard",
        "is_common_a_share",
        "is_st",
        "is_suspended",
        "is_delisted",
        "is_tradeable",
        "has_bar",
        "reject_reason",
        "source",
    ]
    assert len(frame) == 3
    assert bool(by_symbol.loc["600000.SH", "is_tradeable"]) is True
    assert bool(by_symbol.loc["000001.SZ", "is_st"]) is True
    assert by_symbol.loc["000001.SZ", "reject_reason"] == "st_on_date"
    assert bool(by_symbol.loc["600001.SH", "is_suspended"]) is True
    assert summary["traditional_snapshot_id"] == "traditional_fixture"
    assert summary["market_key_rows"] == 3


def test_import_traditional_pit_status_sidecar_saves_source_bound_dataset() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot_root = _write_traditional_snapshot(root / "traditional")
        lake = ResearchDataLake(root / "lake")
        dataset_id = _save_market_bundle(lake)
        record, frame, summary = import_traditional_pit_status_sidecar(
            lake=lake,
            source_market_dataset_id=dataset_id,
            traditional_snapshot_root=snapshot_root,
            start_date="2026-01-05",
            end_date="2026-01-06",
        )
        metadata = lake.describe_dataset(record.dataset_id)

    assert record.dataset_kind == "data_platform_v2_status_sidecar"
    assert summary["source_market_dataset_id"] == dataset_id
    assert summary["traditional_snapshot_id"] == "traditional_fixture"
    assert int(frame["is_tradeable"].sum()) == 1
    assert metadata["parameters"]["source_market_dataset_id"] == dataset_id
    assert metadata["parameters"]["source"] == "traditional_quant_research_v2_pit"


def test_import_traditional_pit_status_sidecar_cli_blocks_active_artifact_diff() -> None:
    with mock.patch(
        "daily_research.data_lake.import_traditional_pit_status_sidecar._active_artifact_has_diff",
        return_value=True,
    ):
        with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
            import_main(["--source-market-dataset-id", "policy_input_bundle__unit"])
