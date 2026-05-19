from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle
from daily_research.data_lake.sector_board_views import SectorBoardViewSpec, build_sector_board_view_from_policy_bundle
from daily_research.deep_alpha.pipeline_utils import load_lake_market_data_with_pool_view
from daily_research.deep_alpha.run_deep_alpha_research import _pool_view_spec_from_args, parse_args


def test_deep_alpha_parser_accepts_lake_pool_view_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_deep_alpha_research.py",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--pool-view-id",
            "policy_pool_view__fixed",
            "--pool-view-kind",
            "rolling_liquidity",
        ],
    )

    args = parse_args()

    assert args.data_source == "lake"
    assert args.lake_dataset_id == "policy_input_bundle__fixed"
    assert args.pool_view_id == "policy_pool_view__fixed"
    assert args.pool_view_kind == "rolling_liquidity"


def test_deep_alpha_pool_view_kind_builds_explicit_spec(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_deep_alpha_research.py",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--pool-view-kind",
            "exchange",
            "--pool-view-name",
            "exchange_sh",
        ],
    )

    args = parse_args()
    spec = _pool_view_spec_from_args(args, start_date="2026-01-05", end_date="2026-01-07")

    assert spec["source_market_dataset_id"] == "policy_input_bundle__fixed"
    assert spec["view_kind"] == "exchange"
    assert spec["view_name"] == "exchange_sh"
    assert spec["exchange_suffix"] == ".SH"


def test_deep_alpha_lake_pool_view_bridge_returns_membership_and_market_frames() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    stocks = ["000001.SZ", "600000.SH"]
    close = pd.DataFrame([[10.0, 30.0], [10.5, 30.5], [11.0, 31.0]], index=dates, columns=stocks)
    market_frames = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
    }

    with TemporaryDirectory() as temp_dir:
        lake = ResearchDataLake(Path(temp_dir))
        market_record = lake.save_market_data_bundle(
            spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
            market_frames=market_frames,
            benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
            benchmark_open=pd.Series(3995.0, index=dates, name="000300.SH"),
            membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
            feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
            source="synthetic",
        )
        view = build_pool_view_from_policy_bundle(
            lake=lake,
            spec=PoolViewSpec(
                source_market_dataset_id=market_record.dataset_id,
                view_kind="exchange",
                view_name="exchange_sz",
                start_date="2026-01-05",
                end_date="2026-01-07",
                exchange_suffix=".SZ",
            ),
        )
        payload = load_lake_market_data_with_pool_view(
            data_lake_root=temp_dir,
            lake_dataset_id=market_record.dataset_id,
            pool_view_id=view.dataset_id,
            start_date="2026-01-05",
            end_date="2026-01-07",
            benchmark="000300.SH",
        )

    assert payload["raw_key"].startswith("lake:")
    assert payload["df_dict"]["Close"].columns.tolist() == ["000001.SZ"]
    assert payload["benchmark_open"].iloc[0] == 3995.0
    assert payload["rolling_membership_frame"].eq(True).all().all()


def test_deep_alpha_lake_bridge_returns_sector_board_metadata_for_relations() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    stocks = ["000001.SZ", "600000.SH"]
    close = pd.DataFrame([[10.0, 30.0], [10.5, 30.5], [11.0, 31.0]], index=dates, columns=stocks)
    market_frames = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
    }

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        industry_csv = root / "industry.csv"
        industry_csv.write_text("stock,industry\n000001.SZ,银行\n600000.SH,银行\n", encoding="utf-8-sig")
        block_file = root / "infoharbor_block.dat"
        block_file.write_text("#FG_高股息,2,880002,20200101,20260519,,\n0#000001,1#600000\n", encoding="gbk")
        lake = ResearchDataLake(root / "lake")
        market_record = lake.save_market_data_bundle(
            spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
            market_frames=market_frames,
            benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
            benchmark_open=pd.Series(3995.0, index=dates, name="000300.SH"),
            membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
            feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
            source="synthetic",
        )
        sector_view = build_sector_board_view_from_policy_bundle(
            lake=lake,
            spec=SectorBoardViewSpec(
                source_market_dataset_id=market_record.dataset_id,
                industry_source_path=str(industry_csv),
                board_source_path=str(block_file),
                as_of_date="2026-05-19",
            ),
        )
        payload = load_lake_market_data_with_pool_view(
            data_lake_root=str(root / "lake"),
            lake_dataset_id=market_record.dataset_id,
            sector_board_view_id=sector_view.dataset_id,
            start_date="2026-01-05",
            end_date="2026-01-07",
            benchmark="000300.SH",
        )

    assert payload["industry_map"].to_dict() == {"000001.SZ": "银行", "600000.SH": "银行"}
    assert payload["style_map"]["FG_高股息"].tolist() == [True, True]
    assert payload["sector_board_view_id"] == sector_view.dataset_id
