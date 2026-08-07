from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_hot_path_rules as rules


def test_contract_uses_full_path_distribution_metrics() -> None:
    study = rules.load_study()
    assert study["analysis"]["profit_claim_allowed"] is False
    assert study["analysis"]["portfolio_selection_performed"] is False
    assert len(study["feature_ranks"]) >= 10


def test_benjamini_hochberg_is_monotone_in_sorted_p_values() -> None:
    adjusted = rules._benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    order = np.argsort(np.array([0.01, 0.04, 0.03]))
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_period_summary_uses_date_equal_means() -> None:
    frame = pd.DataFrame(
        {
            "feature": ["x", "x", "x", "x"],
            "feature_bin": [0, 0, 0, 0],
            "signal_year": [2012, 2012, 2013, 2013],
            "trade_date": ["2012-01-01", "2012-01-02", "2013-01-01", "2013-01-02"],
            "signal_share": [0.2] * 4,
            "fill_rate": [0.9] * 4,
            "complete20_rate": [1.0] * 4,
            "selected_net_return_5": [0.0] * 4,
            "selected_net_return_20": [0.01, 0.03, -0.01, 0.01],
            "d20_excess": [0.02, 0.04, -0.02, 0.0],
            "selected_up10_before_down5": [0.4] * 4,
            "up10_before_down5_lift": [0.1] * 4,
            "selected_mfe20": [0.1] * 4,
            "mfe20_lift": [0.01] * 4,
            "selected_mae20": [-0.05] * 4,
            "mae20_lift": [0.01] * 4,
        }
    )
    summary, annual = rules._summarize_period(
        frame, period_name="test", years={2012, 2013}, hac_lag=1
    )
    assert summary.iloc[0]["selected_net_return_20_mean"] == pytest.approx(0.01)
    assert summary.iloc[0]["d20_excess_mean"] == pytest.approx(0.01)
    assert annual["evaluation_year"].tolist() == [2012, 2013]
