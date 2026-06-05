from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import frontier_personal_candidate_selection_report as mod
from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    DIAGNOSTIC_PERSONAL_GATE_SCOPE,
    FORMAL_PERSONAL_GATE_SCOPE,
    PERSONAL_BACKTEST_ONLY_LEVEL,
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
)


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_type": "personal_candidate_gate",
                "source_run_dir": "gate/formal",
                "signal": "ml_v2_regime_heads",
                "constraint_variant": "regime_weighted_capital_scaled",
                "top_n": 200,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "exposure_penalty_strength": 0.25,
                "mean_annualized_return": 0.12,
                "min_annualized_return": -0.20,
                "positive_year_rate": 0.7,
                "worst_max_drawdown": -0.12,
                "total_periods": 60,
                "eval_year_count": 10,
                "promotion_level": PERSONAL_BACKTEST_PROMOTION_LEVEL,
                "evidence_scope": FORMAL_PERSONAL_GATE_SCOPE,
                "formal_profile_gate": True,
                "failed_gates": "",
            },
            {
                "source_type": "personal_candidate_gate",
                "source_run_dir": "gate/formal",
                "signal": "ml_v2_weak_weighted",
                "constraint_variant": "baseline",
                "top_n": 200,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "exposure_penalty_strength": 0.25,
                "mean_annualized_return": 0.04,
                "min_annualized_return": -0.28,
                "positive_year_rate": 0.6,
                "worst_max_drawdown": -0.18,
                "total_periods": 60,
                "eval_year_count": 10,
                "promotion_level": PERSONAL_BACKTEST_ONLY_LEVEL,
                "evidence_scope": FORMAL_PERSONAL_GATE_SCOPE,
                "formal_profile_gate": True,
                "failed_gates": "return_gate",
            },
            {
                "source_type": "personal_protocol_grid",
                "source_run_dir": "grid/smoke",
                "signal": "ml_v2_smoke",
                "constraint_variant": "baseline",
                "top_n": 200,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "exposure_penalty_strength": 0.25,
                "mean_annualized_return": 0.30,
                "min_annualized_return": 0.10,
                "positive_year_rate": 1.0,
                "worst_max_drawdown": -0.05,
                "total_periods": 12,
                "eval_year_count": 1,
                "promotion_level": PERSONAL_BACKTEST_PROMOTION_LEVEL,
                "evidence_scope": DIAGNOSTIC_PERSONAL_GATE_SCOPE,
                "formal_profile_gate": False,
                "failed_gates": "formal_profile_gate",
            },
        ]
    )


def test_selection_report_normalizes_only_formal_promotions_to_backtest_candidate() -> None:
    selection = mod.build_candidate_selection_summary(_evidence())

    statuses = dict(zip(selection["signal"], selection["selection_status"]))
    assert statuses["ml_v2_regime_heads"] == mod.SELECTION_CANDIDATE
    assert statuses["ml_v2_weak_weighted"] == mod.SELECTION_BACKTEST_ONLY
    assert statuses["ml_v2_smoke"] == mod.SELECTION_DIAGNOSTIC
    assert set(selection["selection_status"]).issubset(
        {mod.SELECTION_CANDIDATE, mod.SELECTION_BACKTEST_ONLY, mod.SELECTION_DIAGNOSTIC}
    )
    assert selection.loc[selection["signal"].eq("ml_v2_regime_heads"), "post_selection_recommendation"].iloc[0] == "user_discretion"
    assert selection.loc[selection["signal"].eq("ml_v2_weak_weighted"), "post_selection_recommendation"].iloc[0] == "none"


def test_selection_report_summary_keeps_paper_and_trading_counts_zero() -> None:
    selection = mod.build_candidate_selection_summary(_evidence())
    rejected = mod.build_rejected_diagnostic_summary(selection)
    summary = mod.summarize_candidate_selection_report(
        selection,
        rejected,
        run_id="fixture",
        run_dir=Path("out/fixture"),
        protocol_dirs=(Path("grid/smoke"),),
        gate_dirs=(Path("gate/formal"),),
    )

    assert summary["personal_backtest_candidate_count"] == 1
    assert summary["personal_paper_candidate_count"] == 0
    assert summary["personal_trading_candidate_count"] == 0
    assert summary["post_selection_boundary"] == (
        "agent_selects_models_and_strategies_only; user_handles_risk_recording_and_live_decisions"
    )
    assert summary["allowed_selection_statuses"] == [
        mod.SELECTION_CANDIDATE,
        mod.SELECTION_BACKTEST_ONLY,
        mod.SELECTION_DIAGNOSTIC,
    ]


def test_run_selection_report_reads_latest_artifacts_and_writes_outputs(tmp_path: Path) -> None:
    protocol_run = tmp_path / "grid" / "grid_run"
    gate_run = tmp_path / "gate" / "gate_run"
    protocol_run.mkdir(parents=True)
    gate_run.mkdir(parents=True)
    evidence = _evidence()
    evidence.iloc[[2]].to_csv(protocol_run / "personal_protocol_grid_ledger.csv", index=False)
    evidence.iloc[[0, 1]].to_csv(gate_run / "personal_candidate_gate_summary.csv", index=False)

    result = mod.run_frontier_personal_candidate_selection_report(
        personal_protocol_grid_run_dirs=(protocol_run,),
        personal_gate_run_dirs=(gate_run,),
        output_dir=tmp_path / "out",
        write_research_log=True,
        research_log_path=tmp_path / "selection.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["personal_backtest_candidate_count"] == 1
    assert (run_dir / "candidate_selection_summary.csv").exists()
    assert (run_dir / "candidate_selection_rejected_diagnostics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "selection.md").exists()
    saved_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["personal_paper_candidate_count"] == 0
    saved_selection = pd.read_csv(run_dir / "candidate_selection_summary.csv")
    assert set(saved_selection["selection_status"]).issubset(
        {mod.SELECTION_CANDIDATE, mod.SELECTION_BACKTEST_ONLY, mod.SELECTION_DIAGNOSTIC}
    )
