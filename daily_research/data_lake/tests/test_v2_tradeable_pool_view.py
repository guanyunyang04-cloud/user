from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle, load_pool_view


def _save_market_bundle(lake: ResearchDataLake, *, spec_extra: dict[str, object] | None = None) -> tuple[str, pd.DatetimeIndex, list[str]]:
    dates = pd.date_range("2026-01-05", periods=30, freq="B")
    stocks = ["000001.SZ", "000002.SZ", "600000.SH", "300001.SZ", "688001.SH"]
    close = pd.DataFrame(10.0, index=dates, columns=stocks)
    amount = pd.DataFrame(
        {
            "000001.SZ": [90000.0 + idx for idx in range(len(dates))],
            "000002.SZ": [80000.0 + idx for idx in range(len(dates))],
            "600000.SH": [70000.0 + idx for idx in range(len(dates))],
            "300001.SZ": [200000.0 + idx for idx in range(len(dates))],
            "688001.SH": [190000.0 + idx for idx in range(len(dates))],
        },
        index=dates,
    )
    spec = {"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"}
    spec.update(spec_extra or {})
    record = lake.save_market_data_bundle(
        spec=spec,
        market_frames={
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": amount,
        },
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"score_none": close * 0.0},
        source="synthetic",
    )
    return record.dataset_id, dates, stocks


def _save_status_sidecar(lake: ResearchDataLake, *, source_market_dataset_id: str, dates: pd.DatetimeIndex, stocks: list[str]) -> str:
    rows: list[dict[str, object]] = []
    for date in dates:
        for stock in stocks:
            rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "symbol": stock,
                    "is_listed_on_date": stock != "600000.SH",
                    "is_mainboard": stock.split(".", 1)[0].startswith(("000", "001", "002", "003", "600", "601", "603", "605")),
                    "is_common_a_share": True,
                    "is_st": stock == "000002.SZ",
                    "is_suspended": False,
                    "is_delisted": stock == "600000.SH",
                    "is_tradeable": stock == "000001.SZ",
                    "has_bar": True,
                    "reject_reason": "" if stock == "000001.SZ" else "blocked_by_unit_test",
                    "source": "unit",
                }
            )
    record = lake.save_domain_dataset(
        domain="v2_status_sidecar",
        frame=pd.DataFrame(rows),
        spec={
            "dataset": "data_platform_v2_status_sidecar",
            "source_market_dataset_id": source_market_dataset_id,
            "start_date": dates.min().strftime("%Y-%m-%d"),
            "end_date": dates.max().strftime("%Y-%m-%d"),
        },
        source="unit",
    )
    return record.dataset_id


def test_tradeable_mainboard_pool_filters_status_and_excluded_prefixes() -> None:
    with TemporaryDirectory() as temp_dir:
        lake = ResearchDataLake(Path(temp_dir))
        dataset_id, dates, stocks = _save_market_bundle(lake)
        sidecar_id = _save_status_sidecar(lake, source_market_dataset_id=dataset_id, dates=dates, stocks=stocks)
        view = build_pool_view_from_policy_bundle(
            lake=lake,
            spec=PoolViewSpec(
                source_market_dataset_id=dataset_id,
                view_kind="rolling_liquidity_tradeable_mainboard",
                view_name="rolling_liquid2_tradeable_mainboard_v2",
                pool_name="liquid2",
                start_date="2026-01-05",
                end_date="2026-02-13",
                rebalance_every_days=5,
                adv_window=2,
                exclude_symbol_prefixes=("300", "301", "688", "689"),
                status_sidecar_dataset_id=sidecar_id,
                require_tradeable=True,
            ),
        )
        loaded = load_pool_view(lake=lake, pool_view_id=view.dataset_id)

    assert set(loaded.membership_frame.columns) == {"000001.SZ", "000002.SZ", "600000.SH"}
    assert loaded.membership_frame["000001.SZ"].any()
    assert not loaded.membership_frame["000002.SZ"].any()
    assert not loaded.membership_frame["600000.SH"].any()
    assert int(loaded.membership_frame.sum(axis=1).max()) == 1
    assert loaded.metadata["source_cache"]["status_sidecar_dataset_id"] == sidecar_id
    assert loaded.metadata["parameters"]["require_tradeable"] is True
    assert loaded.metadata["parameters"]["exclude_symbol_prefixes"] == ["300", "301", "688", "689"]


def test_tradeable_mainboard_pool_requires_status_sidecar() -> None:
    with TemporaryDirectory() as temp_dir:
        lake = ResearchDataLake(Path(temp_dir))
        dataset_id, _dates, _stocks = _save_market_bundle(lake)
        with pytest.raises(ValueError, match="requires status_sidecar_dataset_id"):
            build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=dataset_id,
                    view_kind="rolling_liquidity_tradeable_mainboard",
                    view_name="rolling_liquid2_tradeable_mainboard_v2",
                    pool_name="liquid2",
                    start_date="2026-01-05",
                    end_date="2026-02-13",
                    rebalance_every_days=5,
                    adv_window=2,
                    require_tradeable=True,
                ),
            )


def test_tradeable_mainboard_pool_rejects_status_sidecar_source_mismatch() -> None:
    with TemporaryDirectory() as temp_dir:
        lake = ResearchDataLake(Path(temp_dir))
        dataset_id, dates, stocks = _save_market_bundle(lake)
        sidecar_id = _save_status_sidecar(
            lake,
            source_market_dataset_id="policy_input_bundle__other",
            dates=dates,
            stocks=stocks,
        )
        with pytest.raises(ValueError, match="status sidecar source_market_dataset_id mismatch"):
            build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=dataset_id,
                    view_kind="rolling_liquidity_tradeable_mainboard",
                    view_name="rolling_liquid2_tradeable_mainboard_v2",
                    pool_name="liquid2",
                    start_date="2026-01-05",
                    end_date="2026-02-13",
                    rebalance_every_days=5,
                    adv_window=2,
                    status_sidecar_dataset_id=sidecar_id,
                    require_tradeable=True,
                ),
            )


def test_tradeable_mainboard_pool_blocks_when_no_tradeable_members() -> None:
    with TemporaryDirectory() as temp_dir:
        lake = ResearchDataLake(Path(temp_dir))
        dataset_id, dates, stocks = _save_market_bundle(lake)
        rows = []
        for date in dates:
            for stock in stocks:
                rows.append(
                    {
                        "trade_date": date.strftime("%Y-%m-%d"),
                        "symbol": stock,
                        "is_tradeable": False,
                    }
                )
        sidecar = lake.save_domain_dataset(
            domain="v2_status_sidecar",
            frame=pd.DataFrame(rows),
            spec={
                "dataset": "data_platform_v2_status_sidecar",
                "source_market_dataset_id": dataset_id,
                "start_date": "2026-01-05",
                "end_date": "2026-02-13",
            },
            source="unit",
        )
        with pytest.raises(ValueError, match="view produced empty membership"):
            build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=dataset_id,
                    view_kind="rolling_liquidity_tradeable_mainboard",
                    view_name="rolling_liquid2_tradeable_mainboard_v2",
                    pool_name="liquid2",
                    start_date="2026-01-05",
                    end_date="2026-02-13",
                    rebalance_every_days=5,
                    adv_window=2,
                    status_sidecar_dataset_id=sidecar.dataset_id,
                    require_tradeable=True,
                ),
            )
