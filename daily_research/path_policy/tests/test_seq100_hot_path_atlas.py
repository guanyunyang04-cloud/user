from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_hot_path_atlas as atlas


def _dense_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2019-09-02", "2020-03-31")
    rows: list[dict[str, object]] = []
    for symbol in ("600001.SH", "600002.SH"):
        for date_idx, timestamp in enumerate(dates):
            price = 10.0
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": timestamp.strftime("%Y-%m-%d"),
                    "exchange": "SH",
                    "board": "main",
                    "list_date": "2010-01-01",
                    "industry": "TEST",
                    "is_st": False,
                    "is_suspended": False,
                    "is_delisted": False,
                    "raw_open": price,
                    "raw_high": price * 1.01,
                    "raw_low": price * 0.99,
                    "raw_close": price,
                    "volume": 1_000_000.0 + date_idx,
                    "amount": 10_000_000.0 + 10.0 * date_idx,
                    "adjust_factor": 1.0,
                    "date_idx": date_idx,
                    "listed_open_days": date_idx + 1,
                    "adj_open": price,
                    "adj_high": price * 1.01,
                    "adj_low": price * 0.99,
                    "adj_close": price,
                    "bar_valid": True,
                }
            )
    frame = pd.DataFrame(rows)
    signal = "2020-01-02"
    entry = "2020-01-03"
    day2 = "2020-01-06"

    suspended = (frame["symbol"] == "600001.SH") & (
        frame["trade_date"] == entry
    )
    frame.loc[suspended, "is_suspended"] = True
    frame.loc[
        suspended,
        [
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "volume",
            "amount",
            "adjust_factor",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
        ],
    ] = np.nan
    frame.loc[suspended, "bar_valid"] = False

    ambiguous = (frame["symbol"] == "600002.SH") & (
        frame["trade_date"] == day2
    )
    frame.loc[ambiguous, ["raw_high", "adj_high"]] = 11.0
    frame.loc[ambiguous, ["raw_low", "adj_low"]] = 9.5
    frame.loc[ambiguous, ["raw_close", "adj_close"]] = 10.2
    assert signal in set(frame["trade_date"])
    return frame


def test_study_contract_forbids_a_single_discovery_label() -> None:
    study = atlas.load_study()
    assert study["epistemic_contract"]["single_profit_label_is_forbidden_in_discovery"]
    assert study["price_and_path"]["market_day_offsets_not_stock_observation_offsets"]
    assert study["analysis"]["training_performed"] is False


def test_dense_query_multiplies_raw_ohlc_by_adjustment_factor() -> None:
    query = atlas._dense_base_query(
        {
            key: [Path(f"{key}.parquet")]
            for key in (
                "security_status",
                "universe_snapshot",
                "market_daily_raw",
                "adjust_factor",
                "industry_concept",
            )
        }
    )
    assert "raw_close * adjust_factor AS adj_close" in query
    assert "FROM status s" in query
    assert "LEFT JOIN daily d" in query


def test_market_day_path_does_not_skip_suspension_and_keeps_barrier_ambiguity(
    tmp_path: Path,
) -> None:
    dense_path = tmp_path / "dense.parquet"
    pool_path = tmp_path / "pool.parquet"
    _dense_fixture().to_parquet(dense_path, index=False)
    pd.DataFrame(
        {"symbol": ["600002.SH"], "trade_date": ["2020-01-02"]}
    ).to_parquet(pool_path, index=False)

    query = atlas._year_panel_query(
        dense_path,
        pool_path,
        year=2020,
        minimum_listed_days=1,
        minimum_prior_valid=1,
        cost_bps=60.0,
    )
    connection = duckdb.connect()
    result = connection.execute(query).fetchdf()
    connection.close()
    signal_rows = result[result["trade_date"] == "2020-01-02"].set_index("symbol")

    suspended = signal_rows.loc["600001.SH"]
    assert pd.isna(suspended["entry_open"])
    assert pd.isna(suspended["path_close_1"])
    assert suspended["path_close_2"] > 0
    assert not bool(suspended["entry_observed_legal"])

    ambiguous = signal_rows.loc["600002.SH"]
    assert ambiguous["up10_day"] == 2
    assert ambiguous["down5_day"] == 2
    assert bool(ambiguous["up10_down5_same_day_ambiguous"])
    assert not bool(ambiguous["up10_before_down5"])
    assert bool(ambiguous["in_formal_quality_pool"])


def test_hac_constant_series_has_zero_uncertainty() -> None:
    result = atlas._hac_mean(np.full(100, 0.01), lag=20)
    assert result["mean"] == pytest.approx(0.01)
    assert result["se"] == pytest.approx(0.0)
    assert result["lcb_95"] == pytest.approx(0.01)
