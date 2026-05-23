from __future__ import annotations

from pathlib import Path

import pandas as pd


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
