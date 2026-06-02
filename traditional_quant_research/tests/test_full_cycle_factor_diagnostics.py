from __future__ import annotations

import pandas as pd

from traditional_quant_research.diagnostics import (
    quantile_returns_by_group,
    summarize_factor_ic_by_group,
    top_n_backtest_by_group,
)
from traditional_quant_research.experiments.full_cycle_factor_diagnostics import (
    baseline_stability_rows,
    summarize_factor_stability,
)


def _group_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2025-01-02", "year": 2025, "code": "A", "score": 3.0, "fwd_ret_1d": 0.03},
            {"date": "2025-01-02", "year": 2025, "code": "B", "score": 2.0, "fwd_ret_1d": 0.02},
            {"date": "2025-01-02", "year": 2025, "code": "C", "score": 1.0, "fwd_ret_1d": 0.01},
            {"date": "2025-01-03", "year": 2025, "code": "A", "score": 1.0, "fwd_ret_1d": 0.00},
            {"date": "2025-01-03", "year": 2025, "code": "B", "score": 2.0, "fwd_ret_1d": 0.01},
            {"date": "2025-01-03", "year": 2025, "code": "C", "score": 3.0, "fwd_ret_1d": 0.02},
            {"date": "2026-01-02", "year": 2026, "code": "A", "score": 3.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-02", "year": 2026, "code": "B", "score": 2.0, "fwd_ret_1d": 0.02},
            {"date": "2026-01-02", "year": 2026, "code": "C", "score": 1.0, "fwd_ret_1d": 0.03},
        ]
    )


def test_grouped_diagnostics_return_yearly_rows() -> None:
    frame = _group_fixture()

    ic = summarize_factor_ic_by_group(frame, ["score"], "fwd_ret_1d", group_col="year")
    quantiles = quantile_returns_by_group(frame, "score", "fwd_ret_1d", group_col="year", quantiles=3)
    top_n = top_n_backtest_by_group(frame, ["score"], "fwd_ret_1d", group_col="year", top_n=1, fee_bps=10)

    assert set(ic["year"]) == {2025, 2026}
    assert set(quantiles["year"]) == {2025, 2026}
    assert set(top_n["year"]) == {2025, 2026}
    assert top_n.loc[top_n["year"] == 2025, "periods"].iloc[0] == 2


def test_factor_stability_summarizes_positive_year_rate() -> None:
    yearly_ic = pd.DataFrame(
        [
            {"horizon": 1, "year": 2025, "signal": "baseline_score", "mean_rank_ic": 0.1},
            {"horizon": 1, "year": 2026, "signal": "baseline_score", "mean_rank_ic": -0.2},
            {"horizon": 5, "year": 2025, "signal": "baseline_score", "mean_rank_ic": 0.3},
        ]
    )

    stability = summarize_factor_stability(yearly_ic)
    baseline_rows = baseline_stability_rows(stability)

    h1 = stability.loc[stability["horizon"] == 1].iloc[0]
    assert h1["positive_year_rate"] == 0.5
    assert h1["years"] == 2
    assert len(baseline_rows) == 2
