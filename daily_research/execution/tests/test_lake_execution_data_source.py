from __future__ import annotations

from pathlib import Path
import pandas as pd


def test_lake_raw_loader_populates_benchmark_columns_from_silver_benchmark(tmp_path: Path) -> None:
    from daily_research.baseline.advanced_ml_runtime import HistoryWindow, load_raw_data_with_cache
    from quant_data_platform.lake import ResearchDataLake

    dates = pd.to_datetime(["2026-05-20", "2026-05-21", "2026-05-22"])
    stocks = ["000001.SZ", "600000.SH"]
    close = pd.DataFrame(
        {
            "000001.SZ": [10.0, 10.2, 10.4],
            "600000.SH": [20.0, 20.1, 20.3],
        },
        index=dates,
    )
    market_frames = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
    }
    benchmark_close = pd.Series([4800.0, 4810.0, 4820.0], index=dates, name="000300.SH")
    benchmark_open = pd.Series([4795.0, 4805.0, 4815.0], index=dates, name="000300.SH")
    membership_frame = pd.DataFrame(True, index=dates, columns=stocks)
    feature_frames = {
        "score_none": pd.DataFrame(0.0, index=dates, columns=stocks),
        "score_v2": pd.DataFrame(0.1, index=dates, columns=stocks),
    }
    lake = ResearchDataLake(tmp_path)
    record = lake.save_market_data_bundle(
        spec={"pool_name": "execution_test", "benchmark": "000300.SH", "source": "synthetic"},
        market_frames=market_frames,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        membership_frame=membership_frame,
        feature_frames=feature_frames,
        source="synthetic",
    )

    raw_df_dict, _ = load_raw_data_with_cache(
        data_source="lake",
        csv_folder=None,
        universe=stocks,
        benchmark="000300.SH",
        history_window=HistoryWindow(
            mode="infer",
            requested_start_date="20260520",
            effective_start_date="20260520",
            end_date="20260522",
            required_trading_days=3,
        ),
        lake_dataset_id=record.dataset_id,
        data_lake_root=str(tmp_path),
        use_cache=False,
    )

    pd.testing.assert_series_equal(raw_df_dict["Close"]["000300.SH"], benchmark_close, check_freq=False)
    pd.testing.assert_series_equal(raw_df_dict["Open"]["000300.SH"], benchmark_open, check_freq=False)


def test_liquidity_pool_refresh_uses_lake_when_tq_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    from daily_research.execution import liquidity_universe

    called = {"tq": False, "lake": False}
    dates = pd.to_datetime(["2026-05-20", "2026-05-21", "2026-05-22"])
    close = pd.DataFrame(
        {
            "000001.SZ": [10.0, 10.2, 10.5],
            "000002.SZ": [20.0, 20.1, 20.4],
            "000300.SH": [4000.0, 4010.0, 4020.0],
        },
        index=dates,
    )
    amount = pd.DataFrame(
        {
            "000001.SZ": [100.0, 120.0, 150.0],
            "000002.SZ": [300.0, 330.0, 360.0],
            "000300.SH": [1.0, 1.0, 1.0],
        },
        index=dates,
    )

    def fake_tq(*args, **kwargs):
        called["tq"] = True
        raise AssertionError("TQ should not be used when lake data is available")

    def fake_lake(*args, **kwargs):
        called["lake"] = True
        return (
            {
                "Close": close,
                "Open": close,
                "High": close,
                "Low": close,
                "Volume": amount,
                "Amount": amount,
            },
            {"source": "data_lake", "dataset_id": "policy_input_bundle__x"},
        )

    monkeypatch.setattr(liquidity_universe, "load_universe_from_tq", fake_tq)
    monkeypatch.setattr(liquidity_universe, "load_raw_data_with_cache", fake_lake)
    monkeypatch.setattr(liquidity_universe, "get_latest_completed_trading_date", lambda: "2026-05-22")
    monkeypatch.setattr(liquidity_universe, "get_universe_dir", lambda: tmp_path)

    artifacts = liquidity_universe.update_liquidity_pool_files(pool_sizes=(1,), signal_date="2026-05-22")

    assert called == {"tq": False, "lake": True}
    assert artifacts.cache_meta["source"] == "data_lake"
    assert artifacts.latest_files[1].read_text(encoding="utf-8").strip() == "000002.SZ"


def test_trade_plan_parser_accepts_lake_dataset_options() -> None:
    from daily_research.baseline import generate_daily_trade_plan

    parser = generate_daily_trade_plan.build_arg_parser()
    args = parser.parse_args(
        [
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__x",
            "--data-lake-root",
            "daily_research/output/research_data_lake",
        ]
    )

    assert args.data_source == "lake"
    assert args.lake_dataset_id == "policy_input_bundle__x"
    assert args.data_lake_root == "daily_research/output/research_data_lake"


def test_default_pool_argument_rebuilds_when_current_pool_has_invalid_stocks(monkeypatch, tmp_path: Path) -> None:
    from daily_research.execution import entrypoint_utils

    pool_file = tmp_path / "liquid1_latest.txt"
    pool_file.write_text("688001.SH\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_ensure_default_pool_file(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("refresh_cache"):
            pool_file.write_text("600000.SH\n", encoding="utf-8")
        return pool_file

    monkeypatch.setattr(entrypoint_utils.sys, "argv", ["run_trade_plan.py"])
    monkeypatch.setattr(
        "daily_research.execution.liquidity_universe.ensure_default_pool_file",
        fake_ensure_default_pool_file,
    )
    monkeypatch.setattr(
        "daily_research.execution.liquidity_universe.get_default_pool_file",
        lambda pool_size: pool_file,
    )
    monkeypatch.setattr(
        "daily_research.execution.strategy_manifest.load_strategy_manifest",
        lambda: {},
    )
    monkeypatch.setattr(
        "daily_research.execution.liquidity_universe.DEFAULT_POOL_SIZE",
        1,
    )
    monkeypatch.setattr(
        "daily_research.baseline.data_provider.load_cached_stock_name_map",
        lambda: pd.Series(dtype=str),
    )

    entrypoint_utils.ensure_default_pool_argument(pool_size=1)

    assert [call.get("refresh_cache") for call in calls] == [False, True]
    assert entrypoint_utils.sys.argv[-2:] == ["--stocks-file", str(pool_file)]
    assert pool_file.read_text(encoding="utf-8").strip() == "600000.SH"


def test_liquidity_pool_refresh_filters_lake_columns_to_tradeable_main_board(monkeypatch, tmp_path: Path) -> None:
    from daily_research.execution import liquidity_universe

    dates = pd.to_datetime(["2026-05-20", "2026-05-21", "2026-05-22"])
    close = pd.DataFrame(
        {
            "688001.SH": [80.0, 81.0, 82.0],
            "300001.SZ": [70.0, 71.0, 72.0],
            "600000.SH": [10.0, 10.2, 10.5],
            "000001.SZ": [20.0, 20.1, 20.4],
        },
        index=dates,
    )
    amount = pd.DataFrame(
        {
            "688001.SH": [9000.0, 9100.0, 9200.0],
            "300001.SZ": [8000.0, 8100.0, 8200.0],
            "600000.SH": [700.0, 710.0, 720.0],
            "000001.SZ": [300.0, 330.0, 360.0],
        },
        index=dates,
    )

    def fake_lake(*args, **kwargs):
        return (
            {
                "Close": close,
                "Open": close,
                "High": close,
                "Low": close,
                "Volume": amount,
                "Amount": amount,
            },
            {"source": "data_lake", "dataset_id": "policy_input_bundle__x"},
        )

    monkeypatch.setattr(liquidity_universe, "load_raw_data_with_cache", fake_lake)
    monkeypatch.setattr(liquidity_universe, "get_latest_completed_trading_date", lambda: "2026-05-22")
    monkeypatch.setattr(liquidity_universe, "get_universe_dir", lambda: tmp_path)

    artifacts = liquidity_universe.update_liquidity_pool_files(pool_sizes=(1,), signal_date="2026-05-22")

    assert artifacts.latest_files[1].read_text(encoding="utf-8").strip() == "600000.SH"
