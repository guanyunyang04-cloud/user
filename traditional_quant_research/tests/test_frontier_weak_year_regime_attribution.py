from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import frontier_weak_year_regime_attribution as regime_attr


def _yearly_failure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"eval_year": 2024, "signal": "signal_a", "exposure_penalty_strength": 0.25, "annualized_return": 0.10, "weak_year": False},
            {"eval_year": 2025, "signal": "signal_a", "exposure_penalty_strength": 0.25, "annualized_return": -0.08, "weak_year": True},
            {"eval_year": 2024, "signal": "signal_b", "exposure_penalty_strength": 1.0, "annualized_return": 0.04, "weak_year": False},
            {"eval_year": 2025, "signal": "signal_b", "exposure_penalty_strength": 1.0, "annualized_return": -0.02, "weak_year": True},
        ]
    )


def _yearly_regime() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eval_year": 2024,
                "trade_date_count": 2,
                "security_count_mean": 100.0,
                "security_count_min": 90,
                "security_count_max": 110,
                "market_ret_1d_mean": 0.01,
                "market_ret_5d_mean": 0.03,
                "market_ret_20d_mean": 0.08,
                "breadth_1d_positive_rate": 0.55,
                "breadth_5d_positive_rate": 0.60,
                "breadth_20d_positive_rate": 0.70,
                "market_volatility_20d_mean": 0.20,
                "market_amplitude_20d_mean": 0.03,
            },
            {
                "eval_year": 2025,
                "trade_date_count": 2,
                "security_count_mean": 100.0,
                "security_count_min": 90,
                "security_count_max": 110,
                "market_ret_1d_mean": -0.02,
                "market_ret_5d_mean": -0.04,
                "market_ret_20d_mean": -0.10,
                "breadth_1d_positive_rate": 0.45,
                "breadth_5d_positive_rate": 0.40,
                "breadth_20d_positive_rate": 0.35,
                "market_volatility_20d_mean": 0.30,
                "market_amplitude_20d_mean": 0.05,
            },
        ]
    )


def _tradeable_panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=30).append(pd.bdate_range("2025-01-01", periods=30))
    for code_index, code in enumerate(["600000.SH", "000001.SZ", "000002.SZ"]):
        for date_index, date in enumerate(dates):
            year_bias = 0.05 if date.year == 2024 else -0.03
            close = 10.0 + code_index + year_bias * date_index + 0.01 * code_index * date_index
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close - 0.01,
                    "high": close + 0.10,
                    "low": close - 0.10,
                    "close": close,
                    "volume": 1000.0 + date_index,
                    "amount": close * (1000.0 + date_index),
                    "is_tradeable": True,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_weak_vs_positive_regime_compares_market_strength() -> None:
    joined = regime_attr.join_signal_year_regime(_yearly_failure(), _yearly_regime())
    summary = regime_attr.summarize_weak_vs_positive_regime(joined)

    pooled_ret = summary.loc[
        (summary["signal"] == "__pooled__") & (summary["metric"] == "market_ret_20d_mean")
    ].iloc[0]
    pooled_vol = summary.loc[
        (summary["signal"] == "__pooled__") & (summary["metric"] == "market_volatility_20d_mean")
    ].iloc[0]

    assert pooled_ret["weak_year_count"] == 2
    assert pooled_ret["positive_year_count"] == 2
    assert pooled_ret["delta_weak_minus_positive"] == pytest.approx(-0.18)
    assert pooled_ret["diagnosis"] == "weak_years_lower_market_strength"
    assert pooled_vol["delta_weak_minus_positive"] == pytest.approx(0.10)
    assert pooled_vol["diagnosis"] == "weak_years_higher_volatility_or_amplitude"


def test_weak_year_regime_profile_marks_common_weak_years() -> None:
    profile = regime_attr.build_weak_year_regime_profile(_yearly_failure(), _yearly_regime())

    year_2025 = profile.loc[profile["eval_year"] == 2025].iloc[0]
    assert year_2025["weak_signal_count"] == 2
    assert bool(year_2025["all_signals_weak"]) is True
    assert year_2025["market_ret_20d_mean"] == pytest.approx(-0.10)


def test_run_frontier_weak_year_regime_attribution_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    failure_run = tmp_path / "failure"
    failure_run.mkdir()
    _yearly_failure().to_csv(failure_run / "yearly_failure_attribution.csv", index=False)

    monkeypatch.setattr(
        regime_attr,
        "load_pit_manifest",
        lambda root=None: {
            "snapshot_id": "fixture_snapshot",
            "dataset": {"date_min": "2024-01-01", "date_max": "2025-12-31"},
        },
    )
    monkeypatch.setattr(
        regime_attr,
        "load_tradeable_panel",
        lambda root=None, start_date=None, end_date=None: _tradeable_panel(),
    )

    result = regime_attr.run_frontier_weak_year_regime_attribution(
        failure_run_dir=failure_run,
        output_dir=tmp_path / "output",
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["candidate_count"] == 0
    assert result["common_weak_years"] == ["2025"]
    assert (run_dir / "market_regime_daily.csv").exists()
    assert (run_dir / "yearly_market_regime.csv").exists()
    assert (run_dir / "weak_year_regime_profile.csv").exists()
    assert (run_dir / "signal_year_regime_attribution.csv").exists()
    assert (run_dir / "weak_vs_positive_regime_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "research_log.md").exists()
