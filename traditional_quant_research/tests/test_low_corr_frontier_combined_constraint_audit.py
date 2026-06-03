from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_frontier_combined_constraint_audit


def _frontier_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    codes = ["600000.SH", "000001.SZ", "000002.SZ", "000003.SZ"]
    signals = (
        low_corr_frontier_combined_constraint_audit.ROLLING_IC_SIGNAL,
        low_corr_frontier_combined_constraint_audit.IC_WEIGHTED_SIGNAL,
        low_corr_frontier_combined_constraint_audit.LOW_CORR_SIGNAL,
    )
    for code_index, code in enumerate(codes):
        for date_index, date in enumerate(dates):
            base = 10.0 + code_index * 2.0 + date_index * 0.1
            row = {
                "date": date,
                "code": code,
                "open": base,
                "close": base + 0.1 + code_index * 0.02,
                "amount": 100_000_000.0 + code_index * 20_000_000.0,
                "volume": 1_000_000.0 + code_index * 100_000.0,
                "is_tradeable": True,
                "log_amount_mean_20d_z": -1.0 + code_index * 0.5,
                "neg_volatility_20d_z": 1.0 - code_index * 0.25,
                "momentum_20d_z": 0.2 + code_index * 0.2,
                "turn_xsec_z": -0.8 + code_index * 0.4,
                "pctChg_xsec_z": 0.1 * code_index,
                "industry": "Finance" if code_index < 2 else "RealEstate",
            }
            for signal_offset, signal in enumerate(signals):
                row[signal] = 5.0 - code_index + date_index * 0.01 + signal_offset * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_normalize_signal_penalty_strengths_requires_complete_mapping() -> None:
    signals = (
        low_corr_frontier_combined_constraint_audit.ROLLING_IC_SIGNAL,
        low_corr_frontier_combined_constraint_audit.LOW_CORR_SIGNAL,
    )

    mapping = low_corr_frontier_combined_constraint_audit.normalize_signal_penalty_strengths(
        f"{signals[0]}=0.25,{signals[1]}=1.0",
        signals,
    )

    assert mapping[signals[0]] == pytest.approx(0.25)
    assert mapping[signals[1]] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing penalty strengths"):
        low_corr_frontier_combined_constraint_audit.normalize_signal_penalty_strengths(
            f"{signals[0]}=0.25",
            signals,
        )
    with pytest.raises(ValueError, match="unknown signals"):
        low_corr_frontier_combined_constraint_audit.normalize_signal_penalty_strengths(
            f"{signals[0]}=0.25,{signals[1]}=1.0,extra=0",
            signals,
        )


def test_summarize_combined_constraint_audit_adds_impact_and_low_corr_deltas() -> None:
    other = low_corr_frontier_combined_constraint_audit.ROLLING_IC_SIGNAL
    low = low_corr_frontier_combined_constraint_audit.LOW_CORR_SIGNAL
    summary = pd.DataFrame(
        [
            {"eval_year": 2025, "signal": low, "constraint_variant": "baseline", "exposure_penalty_cols": "style", "exposure_penalty_strength": 1.0, "group_col": "industry", "max_group_weight": 0.1, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 0.0, "annualized_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.04, "mean_turnover": 0.8, "mean_impact_cost": 0.0, "mean_total_cost": 0.003, "periods": 4, "constraint_fallback_count": 0},
            {"eval_year": 2025, "signal": low, "constraint_variant": "baseline", "exposure_penalty_cols": "style", "exposure_penalty_strength": 1.0, "group_col": "industry", "max_group_weight": 0.1, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.17, "sharpe": 0.8, "max_drawdown": -0.06, "mean_turnover": 0.8, "mean_impact_cost": 0.001, "mean_total_cost": 0.004, "periods": 4, "constraint_fallback_count": 0},
            {"eval_year": 2025, "signal": other, "constraint_variant": "baseline", "exposure_penalty_cols": "style", "exposure_penalty_strength": 0.25, "group_col": "industry", "max_group_weight": 0.1, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 0.0, "annualized_return": 0.30, "sharpe": 1.2, "max_drawdown": -0.03, "mean_turnover": 0.7, "mean_impact_cost": 0.0, "mean_total_cost": 0.003, "periods": 4, "constraint_fallback_count": 1},
            {"eval_year": 2025, "signal": other, "constraint_variant": "baseline", "exposure_penalty_cols": "style", "exposure_penalty_strength": 0.25, "group_col": "industry", "max_group_weight": 0.1, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.25, "sharpe": 1.0, "max_drawdown": -0.05, "mean_turnover": 0.7, "mean_impact_cost": 0.002, "mean_total_cost": 0.005, "periods": 4, "constraint_fallback_count": 1},
        ]
    )

    aggregate = low_corr_frontier_combined_constraint_audit.summarize_combined_constraint_audit(summary)
    other10 = aggregate.loc[(aggregate["signal"] == other) & (aggregate["impact_bps_per_1pct"] == 10.0)].iloc[0]

    assert other10["mean_annualized_return"] == pytest.approx(0.25)
    assert other10["mean_impact_drag_vs_no_impact"] == pytest.approx(-0.05)
    assert other10["mean_delta_annualized_return_vs_low_corr"] == pytest.approx(0.08)
    assert other10["exposure_penalty_strength"] == pytest.approx(0.25)
    assert other10["group_col"] == "industry"
    assert other10["constraint_variant"] == "baseline"
    assert other10["constraint_fallback_count"] == 1


def test_run_low_corr_frontier_combined_constraint_audit_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _frontier_panel()
    signals = (
        low_corr_frontier_combined_constraint_audit.ROLLING_IC_SIGNAL,
        low_corr_frontier_combined_constraint_audit.IC_WEIGHTED_SIGNAL,
        low_corr_frontier_combined_constraint_audit.LOW_CORR_SIGNAL,
    )
    captured: dict[str, object] = {}

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        captured.update(kwargs)
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "evaluation_panel": panel,
            "available_signals": list(signals),
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_frontier_combined_constraint_audit,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_frontier_combined_constraint_audit.run_low_corr_frontier_combined_constraint_audit(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=signals,
        signal_penalty_strengths={
            signals[0]: 0.25,
            signals[1]: 0.0,
            signals[2]: 1.0,
        },
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(30.0,),
        capital_amounts=(10_000_000.0,),
        impact_bps_per_1pct_values=(0.0, 5.0),
        exposure_penalty_cols=("log_amount_mean_20d_z", "turn_xsec_z"),
        exposure_columns=("log_amount_mean_20d_z", "turn_xsec_z"),
        exposure_constraint_cols=("log_amount_mean_20d_z", "turn_xsec_z"),
        max_abs_exposure=1.0,
        group_col="industry",
        max_group_weight=0.5,
        execution_constraints=False,
        include_metrics=True,
        include_industry=True,
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = tmp_path / result["run_id"]
    expected_files = [
        "combined_constraint_summary.csv",
        "combined_constraint_aggregate.csv",
        "combined_constraint_trades.csv",
        "combined_constraint_liquidity.csv",
        "combined_constraint_liquidity_summary.csv",
        "combined_constraint_basket_exposure.csv",
        "combined_constraint_basket_exposure_summary.csv",
        "combined_constraint_industry_exposure.csv",
        "combined_constraint_industry_summary.csv",
        "combined_constraint_meta.csv",
        "summary.json",
        "summary.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists()
    assert (tmp_path / "research_log.md").exists()
    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["candidate_count"] == 0
    assert result["include_metrics"] is True
    assert result["include_industry"] is True
    assert result["group_col"] == "industry"
    assert result["max_group_weight"] == pytest.approx(0.5)
    assert result["exposure_constraint_cols"] == ["log_amount_mean_20d_z", "turn_xsec_z"]
    assert result["max_abs_exposure"] == pytest.approx(1.0)
    assert result["signal_penalty_strengths"][signals[0]] == pytest.approx(0.25)
    assert captured["include_industry"] is True
    assert captured["include_metrics"] is True

    summary = pd.read_csv(run_dir / "combined_constraint_summary.csv")
    assert set(summary["signal"]) == set(signals)
    assert set(summary["constraint_variant"]) == {"baseline"}
    assert set(summary["impact_bps_per_1pct"]) == {0.0, 5.0}
    assert "constraint_fallback_count" in summary.columns
    assert summary.loc[summary["signal"] == signals[0], "exposure_penalty_strength"].dropna().eq(0.25).all()
    aggregate = pd.read_csv(run_dir / "combined_constraint_aggregate.csv")
    assert "mean_impact_drag_vs_no_impact" in aggregate.columns
    exposure_summary = pd.read_csv(run_dir / "combined_constraint_basket_exposure_summary.csv")
    assert set(exposure_summary["factor"]) == {"log_amount_mean_20d_z", "turn_xsec_z"}
    industry_summary = pd.read_csv(run_dir / "combined_constraint_industry_summary.csv")
    assert not industry_summary.empty
    liquidity = pd.read_csv(run_dir / "combined_constraint_liquidity.csv")
    assert "participation_p95_10m" in liquidity.columns
    meta = pd.read_csv(run_dir / "combined_constraint_meta.csv")
    assert set(meta["group_col"]) == {"industry"}


def test_run_combined_constraint_audit_wires_prior_fit_weak_year_variants(tmp_path: Path, monkeypatch) -> None:
    panel = _frontier_panel()
    signal = low_corr_frontier_combined_constraint_audit.ROLLING_IC_SIGNAL
    weak_run = tmp_path / "weak_rebuild"
    weak_run.mkdir()
    pd.DataFrame(
        [
            {
                "eval_year": 2026,
                "signal": signal,
                "exposure_penalty_strength": 0.25,
                "fit_years": "2017,2018,2022",
                "fit_row_count": 3,
                "metric": "breadth_20d_positive_rate",
                "threshold": 0.45,
                "eval_metric_value": 0.35,
                "eval_allowed_by_rule": False,
                "fit_uses_eval_year": False,
                "evidence_grade": "diagnostic_not_backtest",
            }
        ]
    ).to_csv(weak_run / "fit_eval_regime_candidates.csv", index=False)

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "evaluation_panel": panel,
            "available_signals": [signal],
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_frontier_combined_constraint_audit,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_frontier_combined_constraint_audit.run_low_corr_frontier_combined_constraint_audit(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=(signal,),
        signal_penalty_strengths={signal: 0.25},
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(30.0,),
        capital_amounts=(10_000_000.0,),
        impact_bps_per_1pct_values=(10.0,),
        exposure_penalty_cols=("log_amount_mean_20d_z",),
        exposure_columns=("log_amount_mean_20d_z",),
        exposure_constraint_cols=("log_amount_mean_20d_z",),
        max_abs_exposure=1.0,
        group_col="industry",
        max_group_weight=0.5,
        execution_constraints=False,
        include_metrics=True,
        include_industry=True,
        weak_year_rebuild_run_dir=weak_run,
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    summary = pd.read_csv(run_dir / "combined_constraint_summary.csv")
    assert set(summary["constraint_variant"]) == {"baseline", "regime_gated", "capital_scaled", "factor_blend"}
    gated = summary.loc[summary["constraint_variant"] == "regime_gated"].iloc[0]
    scaled = summary.loc[summary["constraint_variant"] == "capital_scaled"].iloc[0]
    assert gated["weak_year_rule_allowed"] == False
    assert gated["weak_year_rule_fit_years"] == "2017,2018,2022"
    assert gated["weak_year_rule_fit_uses_eval_year"] == False
    assert gated["evidence_grade"] == "out_of_sample_supported"
    assert scaled["weak_year_capital_scale"] == pytest.approx(0.5)

    aggregate = pd.read_csv(run_dir / "combined_constraint_aggregate.csv")
    assert set(aggregate["constraint_variant"]) == {"baseline", "regime_gated", "capital_scaled", "factor_blend"}
    meta = pd.read_csv(run_dir / "combined_constraint_meta.csv")
    assert set(meta["constraint_variants"]) == {"baseline,regime_gated,capital_scaled,factor_blend"}
    exposure_summary = pd.read_csv(run_dir / "combined_constraint_basket_exposure_summary.csv")
    assert "constraint_variant" in exposure_summary.columns


def test_run_combined_constraint_audit_reuses_panel_for_multiple_top_n(tmp_path: Path, monkeypatch) -> None:
    panel = _frontier_panel()
    signal = low_corr_frontier_combined_constraint_audit.ROLLING_IC_SIGNAL
    build_calls: list[dict[str, object]] = []

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        build_calls.append(kwargs)
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "evaluation_panel": panel,
            "available_signals": [signal],
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_frontier_combined_constraint_audit,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_frontier_combined_constraint_audit.run_low_corr_frontier_combined_constraint_audit(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=(signal,),
        signal_penalty_strengths={signal: 0.25},
        top_n=2,
        top_n_values=(1, 2),
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(30.0,),
        capital_amounts=(10_000_000.0,),
        impact_bps_per_1pct_values=(10.0,),
        exposure_penalty_cols=("log_amount_mean_20d_z",),
        exposure_columns=("log_amount_mean_20d_z",),
        group_col="industry",
        max_group_weight=0.5,
        execution_constraints=False,
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert len(build_calls) == 1
    assert result["top_n"] == 1
    assert result["top_n_values"] == [1, 2]
    summary = pd.read_csv(run_dir / "combined_constraint_summary.csv")
    aggregate = pd.read_csv(run_dir / "combined_constraint_aggregate.csv")
    exposure_summary = pd.read_csv(run_dir / "combined_constraint_basket_exposure_summary.csv")
    meta = pd.read_csv(run_dir / "combined_constraint_meta.csv")
    assert set(summary["top_n"]) == {1, 2}
    assert set(aggregate["top_n"]) == {1, 2}
    assert set(exposure_summary["top_n"]) == {1, 2}
    assert set(meta["top_n_values"]) == {"[1, 2]"}
