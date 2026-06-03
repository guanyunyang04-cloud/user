from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from traditional_quant_research.experiments import v2_metrics_exposure_diagnostics as diag


def test_add_metric_exposure_fields_lags_valuation_by_code_and_zscores_by_date() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": "A",
                "turn": 1.0,
                "pctChg": -2.0,
                "log_amount_mean_20d_z": -0.5,
                "momentum_20d_z": 0.2,
                "neg_volatility_20d_z": 0.1,
                "peTTM": 10.0,
                "pbMRQ": 1.0,
                "psTTM": 2.0,
                "pcfNcfTTM": 3.0,
            },
            {
                "date": "2026-01-02",
                "code": "B",
                "turn": 3.0,
                "pctChg": 2.0,
                "log_amount_mean_20d_z": 0.5,
                "momentum_20d_z": -0.2,
                "neg_volatility_20d_z": -0.1,
                "peTTM": 30.0,
                "pbMRQ": 3.0,
                "psTTM": 6.0,
                "pcfNcfTTM": 9.0,
            },
            {
                "date": "2026-01-05",
                "code": "A",
                "turn": 2.0,
                "pctChg": 1.0,
                "log_amount_mean_20d_z": -1.0,
                "momentum_20d_z": 0.4,
                "neg_volatility_20d_z": 0.3,
                "peTTM": 20.0,
                "pbMRQ": 2.0,
                "psTTM": 4.0,
                "pcfNcfTTM": 6.0,
            },
            {
                "date": "2026-01-05",
                "code": "B",
                "turn": 4.0,
                "pctChg": 3.0,
                "log_amount_mean_20d_z": 1.0,
                "momentum_20d_z": -0.4,
                "neg_volatility_20d_z": -0.3,
                "peTTM": 40.0,
                "pbMRQ": 4.0,
                "psTTM": 8.0,
                "pcfNcfTTM": 12.0,
            },
        ]
    )

    output, exposure_columns = diag.add_metric_exposure_fields(frame)

    assert "peTTM_lag1_xsec_z" in exposure_columns
    assert "turn_xsec_z" in exposure_columns
    by_key = output.set_index(["date", "code"])
    assert by_key.loc[(pd.Timestamp("2026-01-05"), "A"), "peTTM_lag1"] == 10.0
    assert by_key.loc[(pd.Timestamp("2026-01-05"), "B"), "peTTM_lag1"] == 30.0
    assert by_key.loc[(pd.Timestamp("2026-01-05"), "A"), "peTTM_lag1_xsec_z"] == pytest.approx(-1.0)
    assert by_key.loc[(pd.Timestamp("2026-01-05"), "B"), "peTTM_lag1_xsec_z"] == pytest.approx(1.0)
    assert by_key.loc[(pd.Timestamp("2026-01-02"), "A"), "turn_xsec_z"] == pytest.approx(-1.0)
    assert by_key.loc[(pd.Timestamp("2026-01-02"), "B"), "turn_xsec_z"] == pytest.approx(1.0)


def test_daily_signal_exposure_correlation_respects_min_pairs_and_dates() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "signal": 1.0, "exposure": 1.0},
            {"date": "2026-01-02", "code": "B", "signal": 2.0, "exposure": 2.0},
            {"date": "2026-01-02", "code": "C", "signal": 3.0, "exposure": 3.0},
            {"date": "2026-01-05", "code": "A", "signal": 1.0, "exposure": 3.0},
            {"date": "2026-01-05", "code": "B", "signal": 2.0, "exposure": 2.0},
        ]
    )

    output = diag.daily_signal_exposure_correlation(
        frame,
        signal_columns=("signal",),
        exposure_columns=("exposure",),
        min_pairs=3,
    )

    assert output["date"].tolist() == ["2026-01-02"]
    assert output.iloc[0]["pairs"] == 3
    assert output.iloc[0]["pearson_corr"] == pytest.approx(1.0)


def test_summarize_basket_metric_exposure_groups_by_year_signal_factor_and_period_type() -> None:
    basket = pd.DataFrame(
        [
            {"eval_year": 2026, "signal": "s1", "factor": "turn_xsec_z", "period_type": "monthly", "period": "2026-01", "active_exposure": 0.2},
            {"eval_year": 2026, "signal": "s1", "factor": "turn_xsec_z", "period_type": "monthly", "period": "2026-02", "active_exposure": -0.4},
            {"eval_year": 2026, "signal": "s1", "factor": "peTTM_lag1_xsec_z", "period_type": "monthly", "period": "2026-01", "active_exposure": 0.5},
        ]
    )

    summary = diag.summarize_basket_metric_exposure(basket)
    turn = summary.loc[summary["factor"] == "turn_xsec_z"].iloc[0]

    assert turn["period_count"] == 2
    assert turn["mean_active_exposure"] == pytest.approx(-0.1)
    assert turn["mean_abs_active_exposure"] == pytest.approx(0.3)
    assert turn["max_abs_active_exposure"] == pytest.approx(0.4)


def test_run_metrics_exposure_diagnostics_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "code": f"{index:06d}.SH",
                "multifactor_ic_weighted_score": float(index),
                "turn_xsec_z": float(index),
            }
            for index in range(60)
        ]
    )

    def fake_year_windows(year: int, *, final_end_date: str | None = None) -> dict[str, str]:
        return {
            "history_start_date": "2025-01-01",
            "fit_start_date": "2025-01-01",
            "fit_end_date": "2025-12-31",
            "start_date": "2026-01-01",
            "end_date": final_end_date or "2026-06-01",
        }

    def fake_build_frontier_signal_panel(**kwargs):
        return {
            "evaluation_panel": panel,
            "low_corr_factor_columns": ["factor_a_z"],
            "exposure_z_columns": ["turn_xsec_z"],
        }

    def fake_backtest(*args, **kwargs):
        return SimpleNamespace(
            trades=pd.DataFrame(
                [
                    {
                        "signal_date": "2026-01-02",
                        "entry_date": "2026-01-05",
                        "exit_date": "2026-01-30",
                        "codes": "000000.SH,000001.SH",
                    }
                ]
            )
        )

    def fake_basket_exposure(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "signal": "multifactor_ic_weighted_score",
                    "period": "2026-01",
                    "period_type": "monthly",
                    "factor": "turn_xsec_z",
                    "active_exposure": 0.25,
                }
            ]
        )

    monkeypatch.setattr(diag, "load_pit_manifest", lambda root=None: {"snapshot_id": "fixture-snapshot"})
    monkeypatch.setattr(diag, "year_windows", fake_year_windows)
    monkeypatch.setattr(diag, "build_frontier_signal_panel", fake_build_frontier_signal_panel)
    monkeypatch.setattr(diag, "horizon_aligned_top_n_backtest", fake_backtest)
    monkeypatch.setattr(diag, "selected_basket_factor_exposure", fake_basket_exposure)

    result = diag.run_v2_metrics_exposure_diagnostics(
        years=(2026,),
        final_end_date="2026-06-01",
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = Path(result["output_dir"])
    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["candidate_count"] == 0
    assert result["max_mean_abs_signal_metric_corr"] == pytest.approx(1.0)
    assert result["max_mean_abs_basket_metric_active_exposure"] == pytest.approx(0.25)
    assert (run_dir / "daily_signal_metric_correlation.csv").exists()
    assert (run_dir / "yearly_signal_metric_correlation.csv").exists()
    assert (run_dir / "basket_metric_exposure_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "research_log.md").exists()
