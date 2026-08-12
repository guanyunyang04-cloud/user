from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy.seq100_payoff_episode_audit import (
    assign_episodes,
    concentration_metrics,
    summarize_episodes,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": ["a", "b", "c", "d", "e"],
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ", "600000.SH"],
            "signal_date_idx": [0, 2, 6, 8, 1],
            "signal_date": [
                "2020-01-02",
                "2020-01-06",
                "2020-01-10",
                "2020-01-14",
                "2020-01-03",
            ],
            "entry_date_idx": [1, 3, 7, 9, 2],
            "entry_date": [
                "2020-01-03",
                "2020-01-07",
                "2020-01-13",
                "2020-01-15",
                "2020-01-06",
            ],
            "exit_date_idx": [4, 5, 8, 10, 3],
            "exit_date": [
                "2020-01-08",
                "2020-01-09",
                "2020-01-14",
                "2020-01-16",
                "2020-01-07",
            ],
            "pnl": [100.0, -25.0, 40.0, 10.0, -5.0],
            "trade_net_return": [0.10, -0.025, 0.04, 0.01, -0.005],
            "net_cash_outflow": [1000.0] * 5,
            "selection_rank": [1, 2, 3, 4, 1],
            "fold": [1, 1, 1, 1, 1],
            "industry_name": ["Bank"] * 5,
            "industry_code": ["B"] * 5,
            "corporate_action_count": [0, 1, 0, 0, 0],
            "factor_changed": [False, True, False, False, False],
            "adjustment_log_change_abs": [0.0, 0.1, 0.0, 0.0, 0.0],
        }
    )


def test_assign_episodes_distinguishes_overlap_and_adjacent() -> None:
    trades = _trades()
    overlap = assign_episodes(trades, gap_days=0)
    adjacent = assign_episodes(trades, gap_days=1)

    stock_overlap = overlap.loc[overlap["symbol"].eq("000001.SZ")]
    stock_adjacent = adjacent.loc[adjacent["symbol"].eq("000001.SZ")]
    assert stock_overlap["episode_id"].nunique() == 3
    assert stock_adjacent["episode_id"].nunique() == 2
    assert stock_overlap.iloc[0]["episode_id"] == stock_overlap.iloc[1]["episode_id"]
    assert stock_overlap.iloc[2]["episode_id"] != stock_overlap.iloc[3]["episode_id"]
    assert stock_adjacent.iloc[2]["episode_id"] == stock_adjacent.iloc[3]["episode_id"]


def test_summarize_episodes_preserves_pnl_and_concurrency() -> None:
    assigned = assign_episodes(_trades(), gap_days=0)
    episodes = summarize_episodes(assigned, definition="overlap")

    assert episodes["pnl"].sum() == pytest.approx(_trades()["pnl"].sum())
    repeated = episodes.loc[episodes["trade_count"].eq(2)].iloc[0]
    assert repeated["pnl"] == pytest.approx(75.0)
    assert repeated["positive_pnl"] == pytest.approx(100.0)
    assert repeated["negative_pnl"] == pytest.approx(-25.0)
    assert repeated["max_concurrent_cohorts"] == 2
    assert repeated["corporate_action_count"] == 1
    assert bool(repeated["factor_changed"])


def test_concentration_metrics_uses_positive_and_net_denominators() -> None:
    result = concentration_metrics([80.0, 20.0, -40.0, -10.0])

    assert result["net_pnl"] == pytest.approx(50.0)
    assert result["positive_pnl"] == pytest.approx(100.0)
    assert result["top_1pct_unit_count"] == 1
    assert result["top_1pct_net_pnl_share"] == pytest.approx(1.6)
    assert result["top_1pct_positive_pnl_share"] == pytest.approx(0.8)
    assert result["effective_positive_contributor_count"] == pytest.approx(
        1.0 / (0.8**2 + 0.2**2)
    )
    assert np.isfinite(result["effective_positive_contributor_count"])
