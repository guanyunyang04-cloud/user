from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_structured_falsification_report import (
    build_failure_matrix,
    build_gate_status,
    build_next_minimum_actions,
    run_frontier_structured_falsification_report,
)


def _promotion_gate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal": "signal_a",
                "mean_annualized_return": 0.10,
                "min_annualized_return": -0.20,
                "positive_year_rate": 0.6,
                "max_monthly_mean_abs_active_exposure": 1.1,
                "size_gate": False,
                "return_gate": False,
                "year_gate": False,
                "sample_gate": True,
                "drawdown_gate": True,
                "style_exposure_gate": False,
                "promoted": False,
                "failed_gates": "size_gate,return_gate,year_gate,style_exposure_gate",
                "exposure_failures": "log_amount_mean_20d_z",
                "promotion_level": "candidate-frontier/backtest_only",
            }
        ]
    )


def test_failure_matrix_classifies_size_weak_year_and_exposure_failures() -> None:
    matrix = build_failure_matrix(
        _promotion_gate(),
        size_summary={"status": "daily_size_absent", "size_rows": 0, "current_cross_check_ok": False},
        failure_summary={"total_weak_signal_years": 4, "combined_run_dir": "combined"},
        weak_summary={"decision": "diagnostic_rebuild_rules_ready_for_backtest"},
    )

    assert set(matrix["failure_type"]) == {
        "data_size_gate",
        "weak_year_return_year_gate",
        "style_exposure_gate",
    }
    assert matrix.loc[matrix["failed_gate"].eq("size_gate"), "evidence_grade"].iloc[0] == "diagnostic"
    assert "daily_size_absent" in matrix.loc[matrix["failed_gate"].eq("size_gate"), "evidence_detail"].iloc[0]
    assert "prior-fit" in matrix.loc[
        matrix["failure_type"].eq("weak_year_return_year_gate"),
        "minimum_next_action",
    ].iloc[0]


def test_failure_matrix_classifies_optimizer_fallback_separately() -> None:
    promotion_gate = _promotion_gate().copy()
    promotion_gate["failed_gates"] = "style_exposure_gate"
    promotion_gate["exposure_failures"] = "optimizer_fallback"
    promotion_gate["constraint_fallback_count"] = 2

    matrix = build_failure_matrix(
        promotion_gate,
        size_summary={"status": "ready"},
        failure_summary={"combined_run_dir": "combined"},
        weak_summary={"decision": "diagnostic_rebuild_rules_ready_for_backtest"},
    )

    row = matrix.iloc[0]
    assert row["failure_type"] == "optimizer_fallback"
    assert "constraint_fallback_count=2" in row["evidence_detail"]
    assert "fallback" in row["minimum_next_action"]


def test_next_actions_are_prioritized_by_blocking_gate() -> None:
    matrix = build_failure_matrix(
        _promotion_gate(),
        size_summary={"status": "daily_size_absent", "size_rows": 0, "current_cross_check_ok": False},
        failure_summary={"total_weak_signal_years": 4, "combined_run_dir": "combined"},
        weak_summary={"decision": "diagnostic_rebuild_rules_ready_for_backtest"},
    )
    actions = build_next_minimum_actions(matrix, pd.DataFrame())

    assert actions.iloc[0]["failure_type"] == "data_size_gate"
    assert "daily_size" in actions.iloc[0]["success_evidence"]
    assert "weak_year_return_year_gate" in actions["failure_type"].tolist()


def test_gate_status_counts_passed_and_failed_rows() -> None:
    status = build_gate_status(_promotion_gate())

    indexed = status.set_index("gate")
    assert indexed.loc["size_gate", "failed_rows"] == 1
    assert indexed.loc["sample_gate", "passed_rows"] == 1
    assert indexed.loc["style_exposure_gate", "status"] == "failed"


def test_run_structured_falsification_report_writes_artifacts(tmp_path: Path) -> None:
    promotion = tmp_path / "promotion"
    ledger = tmp_path / "ledger"
    size = tmp_path / "size"
    failure = tmp_path / "failure"
    weak = tmp_path / "weak"
    for directory in [promotion, ledger, size, failure, weak]:
        directory.mkdir()

    _promotion_gate().to_csv(promotion / "promotion_gate_summary.csv", index=False)
    (promotion / "summary.json").write_text(
        json.dumps({"run_id": "promotion_run", "candidate_count": 0, "decision": "keep_candidate_frontier_backtest_only"}),
        encoding="utf-8",
    )
    (ledger / "summary.json").write_text(
        json.dumps({"run_id": "ledger_run", "trial_count": 1, "candidate_count": 0}),
        encoding="utf-8",
    )
    pd.DataFrame([{"experiment_family": "frontier", "trial_count": 1, "selection_bias_risk": "moderate"}]).to_csv(
        ledger / "selection_bias_report.csv",
        index=False,
    )
    pd.DataFrame([{"dimension": "strategy_candidate_readiness", "status": "blocked"}]).to_csv(
        ledger / "research_maturity_report.csv",
        index=False,
    )
    (size / "summary.json").write_text(
        json.dumps({"run_id": "size_run", "status": "daily_size_absent", "size_rows": 0, "daily_size_ready_for_research": False}),
        encoding="utf-8",
    )
    (failure / "summary.json").write_text(
        json.dumps({"run_id": "failure_run", "total_weak_signal_years": 4, "combined_run_dir": "combined"}),
        encoding="utf-8",
    )
    (weak / "summary.json").write_text(
        json.dumps({"run_id": "weak_run", "decision": "diagnostic_rebuild_rules_ready_for_backtest"}),
        encoding="utf-8",
    )
    pd.DataFrame([{"path": "portfolio_constraint_optimizer", "status": "planned", "blocking_gate": "optimizer_backtest_extension", "next_action": "build optimizer"}]).to_csv(
        weak / "rebuild_backlog.csv",
        index=False,
    )

    result = run_frontier_structured_falsification_report(
        promotion_run_dir=promotion,
        trial_ledger_run_dir=ledger,
        size_audit_run_dir=size,
        failure_run_dir=failure,
        weak_year_rebuild_run_dir=weak,
        output_dir=tmp_path / "output",
        write_research_log=True,
        research_log_path=tmp_path / "report.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "structured_falsification_keep_candidate_frontier_backtest_only"
    assert result["structured_falsification_complete"] is True
    assert "data_size_gate" in result["blocking_categories"]
    assert (run_dir / "structured_failure_matrix.csv").exists()
    assert (run_dir / "next_minimum_actions.csv").exists()
    assert (tmp_path / "report.md").exists()
