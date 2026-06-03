from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.diagnostics import assign_time_split, quantile_returns, single_factor_diagnostics, summarize_factor_ic, top_n_backtest
from traditional_quant_research.research_panel import (
    add_baseline_score,
    add_cross_sectional_zscores,
    add_cross_sectional_excess_return_labels,
    build_factor_label_panel,
    default_factor_columns,
    factor_columns_for_set,
)


def _fixture_panel() -> pd.DataFrame:
    rows = []
    closes = {
        "600000.SH": [10.0, 10.5, 10.2, 10.8, 11.0, 11.2],
        "000001.SZ": [20.0, 19.8, 20.4, 20.1, 20.8, 21.0],
        "000002.SZ": [8.0, 8.1, 8.4, 8.2, 8.5, 8.7],
    }
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    for code, code_closes in closes.items():
        for date, close in zip(dates, code_closes):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000.0 + len(rows) * 10.0,
                    "amount": close * 1000.0,
                    "turn": 1.0 + len(rows) * 0.01,
                    "pctChg": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_build_factor_label_panel_uses_next_open_for_forward_label() -> None:
    panel = _fixture_panel()

    result = build_factor_label_panel(
        panel,
        horizons=(1, 2),
        short_window=2,
        medium_window=3,
        volatility_window=2,
        min_periods=2,
    )

    first = result.loc[(result["code"] == "600000.SH") & (result["date"] == pd.Timestamp("2026-01-02"))].iloc[0]
    assert first["fwd_ret_1d"] == pytest.approx(10.5 / 10.4 - 1.0)
    assert first["fwd_ret_2d"] == pytest.approx(10.2 / 10.4 - 1.0)
    assert "reversal_2d" in result.columns
    assert "momentum_3d" in result.columns


def test_cross_sectional_zscores_and_baseline_score_are_added() -> None:
    panel = build_factor_label_panel(
        _fixture_panel(),
        horizons=(1,),
        short_window=2,
        medium_window=3,
        volatility_window=2,
        min_periods=2,
    )
    factors = default_factor_columns(short_window=2, medium_window=3, volatility_window=2)

    scored = add_cross_sectional_zscores(panel, factors)
    scored = add_baseline_score(scored, score_columns=[f"{column}_z" for column in factors], min_factors=2)

    assert "baseline_score" in scored.columns
    assert scored["baseline_score"].notna().any()


def test_factor_columns_for_set_keeps_core_default_and_adds_expanded_baostock_factors() -> None:
    core = factor_columns_for_set("core", short_window=2, medium_window=3, volatility_window=2)
    expanded = factor_columns_for_set("expanded", short_window=2, medium_window=3, volatility_window=2)

    assert core == default_factor_columns(short_window=2, medium_window=3, volatility_window=2)
    assert core == factor_columns_for_set(None, short_window=2, medium_window=3, volatility_window=2)
    assert core == factor_columns_for_set("", short_window=2, medium_window=3, volatility_window=2)
    assert set(core).issubset(set(expanded))
    assert "reversal_1d" in expanded
    assert "log_amount_mean_2d" in expanded
    assert "log_amount_change_3d" in expanded
    assert "range_position_3d" in expanded
    assert "volume_price_corr_3d" in expanded
    assert "neg_turn_mean_3d" in expanded


def test_expanded_factor_panel_uses_baostock_metrics_when_available() -> None:
    panel = _fixture_panel()

    result = build_factor_label_panel(
        panel,
        horizons=(1,),
        short_window=2,
        medium_window=3,
        volatility_window=2,
        min_periods=2,
        factor_set="expanded",
    )
    expanded = factor_columns_for_set("expanded", short_window=2, medium_window=3, volatility_window=2)

    assert set(expanded).issubset(result.columns)
    assert result["neg_turn_mean_3d"].notna().any()
    assert result["turn_change_3d"].notna().any()
    assert result["volume_price_corr_3d"].notna().any()
    assert result["range_position_3d"].dropna().between(0.0, 1.0).all()


def test_expanded_factor_panel_handles_missing_metrics_columns() -> None:
    panel = _fixture_panel().drop(columns=["turn", "pctChg"])

    result = build_factor_label_panel(
        panel,
        horizons=(1,),
        short_window=2,
        medium_window=3,
        volatility_window=2,
        min_periods=2,
        factor_set="expanded",
    )

    assert "neg_turn_mean_3d" in result.columns
    assert result["neg_turn_mean_3d"].isna().all()
    assert "pctchg_align_gap_1d" in result.columns
    assert result["pctchg_align_gap_1d"].isna().all()


def test_cross_sectional_excess_return_labels_are_demeaned_by_date() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "fwd_ret_1d": 0.03},
            {"date": "2026-01-02", "code": "B", "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "A", "fwd_ret_1d": -0.01},
            {"date": "2026-01-05", "code": "B", "fwd_ret_1d": 0.01},
        ]
    )

    result = add_cross_sectional_excess_return_labels(frame, horizons=(1,))

    assert result["xsec_excess_ret_1d"].tolist() == pytest.approx([0.01, -0.01, -0.01, 0.01])
    assert result.groupby("date")["xsec_excess_ret_1d"].mean().abs().max() == pytest.approx(0.0)


def test_factor_diagnostics_and_top_n_backtest_are_stable() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "score": 3.0, "fwd_ret_1d": 0.03},
            {"date": "2026-01-02", "code": "B", "score": 2.0, "fwd_ret_1d": 0.02},
            {"date": "2026-01-02", "code": "C", "score": 1.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "A", "score": 1.0, "fwd_ret_1d": 0.00},
            {"date": "2026-01-05", "code": "B", "score": 2.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "C", "score": 3.0, "fwd_ret_1d": 0.02},
        ]
    )

    ic = summarize_factor_ic(frame, ["score"], "fwd_ret_1d")
    quantiles = quantile_returns(frame, "score", "fwd_ret_1d", quantiles=3)
    backtest = top_n_backtest(frame, "score", "fwd_ret_1d", top_n=1, fee_bps=10)

    assert ic.loc[0, "mean_rank_ic"] == pytest.approx(1.0)
    assert quantiles.loc[quantiles["quantile"] == 3, "mean_return"].iloc[0] == pytest.approx(0.025)
    assert backtest.daily_returns["holdings"].tolist() == [1, 1]
    assert backtest.daily_returns["net_return"].iloc[0] == pytest.approx(0.03 - 0.001)


def test_single_factor_diagnostics_report_in_sample_and_out_of_sample() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "score": 3.0, "fwd_ret_1d": 0.03},
            {"date": "2026-01-02", "code": "B", "score": 2.0, "fwd_ret_1d": 0.02},
            {"date": "2026-01-02", "code": "C", "score": 1.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "A", "score": 1.0, "fwd_ret_1d": 0.00},
            {"date": "2026-01-05", "code": "B", "score": 2.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "C", "score": 3.0, "fwd_ret_1d": 0.02},
        ]
    )

    split = assign_time_split(frame, split_date="2026-01-05")
    diagnostics = single_factor_diagnostics(frame, ["score"], "fwd_ret_1d", split_date="2026-01-05", top_n=1, fee_bps=10, quantiles=3)

    assert split["sample_split"].tolist() == ["in_sample", "in_sample", "in_sample", "out_of_sample", "out_of_sample", "out_of_sample"]
    assert set(diagnostics["ic"]["sample_split"]) == {"in_sample", "out_of_sample"}
    assert set(diagnostics["quantile"]["sample_split"]) == {"in_sample", "out_of_sample"}
    assert set(diagnostics["top_n"]["sample_split"]) == {"in_sample", "out_of_sample"}
    assert diagnostics["top_n"].loc[diagnostics["top_n"]["sample_split"] == "out_of_sample", "sharpe"].notna().all()
