from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.multifactor import (
    add_equal_rank_score,
    add_ic_weighted_rank_score,
    add_rolling_ic_weighted_rank_score,
    factor_coverage,
    mean_daily_factor_correlation,
    neutralize_factors_by_date,
    rolling_ic_weights_by_date,
    select_low_correlation_factors,
    weights_from_ic_summary,
)


def _factor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "value": 10.0, "risk": 3.0, "noise": 1.0},
            {"date": "2026-01-02", "code": "B", "value": 20.0, "risk": 2.0, "noise": 2.0},
            {"date": "2026-01-02", "code": "C", "value": 30.0, "risk": 1.0, "noise": None},
            {"date": "2026-01-05", "code": "A", "value": 30.0, "risk": 3.0, "noise": 3.0},
            {"date": "2026-01-05", "code": "B", "value": 20.0, "risk": 2.0, "noise": 2.0},
            {"date": "2026-01-05", "code": "C", "value": 10.0, "risk": 1.0, "noise": 1.0},
        ]
    )


def test_add_equal_rank_score_respects_factor_directions() -> None:
    frame = _factor_frame()

    scored = add_equal_rank_score(
        frame,
        ["value", "risk"],
        directions={"value": 1, "risk": -1},
        score_col="multi_score",
        min_factors=2,
    )

    first_day = scored.loc[scored["date"] == "2026-01-02"].sort_values("code")
    assert first_day["multi_score"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert scored["multi_score"].between(0.0, 1.0).all()


def test_ic_weighted_score_uses_normalized_absolute_rank_ic_weights() -> None:
    frame = _factor_frame()
    ic_summary = pd.DataFrame(
        [
            {"signal": "value", "mean_rank_ic": 0.02},
            {"signal": "risk", "mean_rank_ic": -0.06},
        ]
    )

    weights = weights_from_ic_summary(ic_summary, ["value", "risk"])
    scored = add_ic_weighted_rank_score(
        frame,
        ["value", "risk"],
        ic_summary,
        directions={"value": 1, "risk": -1},
        score_col="ic_score",
        min_factors=2,
    )

    assert weights == pytest.approx({"value": 0.25, "risk": 0.75})
    first = scored.loc[(scored["date"] == "2026-01-02") & (scored["code"] == "A")].iloc[0]
    assert first["ic_score"] == pytest.approx((1 / 3) * 0.25 + (1 / 3) * 0.75)


def test_factor_coverage_and_mean_daily_correlation_are_reported() -> None:
    frame = _factor_frame()

    coverage = factor_coverage(frame, ["value", "noise"])
    correlation = mean_daily_factor_correlation(frame, ["value", "risk"])

    assert coverage.loc[coverage["factor"] == "value", "coverage_rate"].iloc[0] == pytest.approx(1.0)
    assert coverage.loc[coverage["factor"] == "noise", "coverage_rate"].iloc[0] == pytest.approx(5 / 6)
    assert correlation.loc["value", "risk"] == pytest.approx(0.0)
    assert correlation.loc["value", "value"] == pytest.approx(1.0)


def test_select_low_correlation_factors_keeps_stronger_uncorrelated_signals() -> None:
    correlation = pd.DataFrame(
        {
            "value": {"value": 1.0, "duplicate": 0.92, "risk": 0.20},
            "duplicate": {"value": 0.92, "duplicate": 1.0, "risk": 0.10},
            "risk": {"value": 0.20, "duplicate": 0.10, "risk": 1.0},
        }
    )
    ic_summary = pd.DataFrame(
        [
            {"signal": "value", "mean_rank_ic": 0.03},
            {"signal": "duplicate", "mean_rank_ic": 0.01},
            {"signal": "risk", "mean_rank_ic": -0.02},
        ]
    )

    selected = select_low_correlation_factors(
        correlation,
        ["value", "duplicate", "risk"],
        ic_summary=ic_summary,
        max_abs_corr=0.75,
    )

    assert selected == ["value", "risk"]


def test_neutralize_factors_by_date_removes_linear_proxy_exposure() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "factor": 2.0, "size": 1.0},
            {"date": "2026-01-02", "code": "B", "factor": 4.0, "size": 2.0},
            {"date": "2026-01-02", "code": "C", "factor": 7.0, "size": 3.0},
            {"date": "2026-01-02", "code": "D", "factor": 9.0, "size": 4.0},
            {"date": "2026-01-05", "code": "A", "factor": 3.0, "size": 1.0},
            {"date": "2026-01-05", "code": "B", "factor": 5.0, "size": 2.0},
            {"date": "2026-01-05", "code": "C", "factor": 6.0, "size": 3.0},
            {"date": "2026-01-05", "code": "D", "factor": 8.0, "size": 4.0},
        ]
    )

    result = neutralize_factors_by_date(frame, ["factor"], ["size"])

    for _, group in result.groupby("date"):
        assert group["factor_neutral"].mean() == pytest.approx(0.0)
        assert group["factor_neutral"].corr(group["size"]) == pytest.approx(0.0, abs=1e-12)


def test_rolling_ic_weights_use_only_prior_dates() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "value": 1.0, "risk": 3.0, "label": 1.0},
            {"date": "2026-01-02", "code": "B", "value": 2.0, "risk": 2.0, "label": 2.0},
            {"date": "2026-01-02", "code": "C", "value": 3.0, "risk": 1.0, "label": 3.0},
            {"date": "2026-01-05", "code": "A", "value": 1.0, "risk": 3.0, "label": 1.0},
            {"date": "2026-01-05", "code": "B", "value": 2.0, "risk": 2.0, "label": 2.0},
            {"date": "2026-01-05", "code": "C", "value": 3.0, "risk": 1.0, "label": 3.0},
            {"date": "2026-01-06", "code": "A", "value": 1.0, "risk": 3.0, "label": 3.0},
            {"date": "2026-01-06", "code": "B", "value": 2.0, "risk": 2.0, "label": 2.0},
            {"date": "2026-01-06", "code": "C", "value": 3.0, "risk": 1.0, "label": 1.0},
        ]
    )

    weights = rolling_ic_weights_by_date(frame, ["value", "risk"], "label", window=2, min_periods=2)
    third_day = weights.loc[weights["date"] == pd.Timestamp("2026-01-06")].set_index("factor")

    assert third_day.loc["value", "mean_rank_ic"] == pytest.approx(1.0)
    assert third_day.loc["risk", "mean_rank_ic"] == pytest.approx(-1.0)
    assert third_day.loc["value", "weight"] == pytest.approx(0.5)
    assert third_day.loc["risk", "weight"] == pytest.approx(0.5)
    assert third_day.loc["value", "direction"] == 1
    assert third_day.loc["risk", "direction"] == -1
    assert not bool(third_day.loc["value", "is_fallback"])


def test_rolling_ic_weighted_score_flips_negative_history_factors() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "value": 1.0, "risk": 3.0, "label": 1.0},
            {"date": "2026-01-02", "code": "B", "value": 2.0, "risk": 2.0, "label": 2.0},
            {"date": "2026-01-02", "code": "C", "value": 3.0, "risk": 1.0, "label": 3.0},
            {"date": "2026-01-05", "code": "A", "value": 1.0, "risk": 3.0, "label": 1.0},
            {"date": "2026-01-05", "code": "B", "value": 2.0, "risk": 2.0, "label": 2.0},
            {"date": "2026-01-05", "code": "C", "value": 3.0, "risk": 1.0, "label": 3.0},
            {"date": "2026-01-06", "code": "A", "value": 1.0, "risk": 3.0, "label": 0.0},
            {"date": "2026-01-06", "code": "B", "value": 2.0, "risk": 2.0, "label": 0.0},
            {"date": "2026-01-06", "code": "C", "value": 3.0, "risk": 1.0, "label": 0.0},
        ]
    )

    scored = add_rolling_ic_weighted_rank_score(
        frame,
        ["value", "risk"],
        "label",
        window=2,
        min_periods=2,
        score_col="rolling_score",
    )
    third_day = scored.loc[scored["date"] == "2026-01-06"].sort_values("code")

    assert third_day["rolling_score"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
