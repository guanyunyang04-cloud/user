from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_candidate_review_matrix as matrix


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_bridge(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    score_panel = root / "scores.csv"
    score_panel.write_text(
        "\n".join(
            [
                "date,stock,score",
                "2024-01-02,000001.SZ,0.30",
                "2024-01-02,600000.SH,0.10",
                "2024-01-03,000001.SZ,0.20",
                "2024-01-03,600000.SH,0.40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "v2_score_backtest_bridge_report.json",
        {
            "status": "completed",
            "bridge": {
                "backtest_score_panel_csv": str(score_panel),
                "dataset_id": "dataset",
                "pool_view_id": "pool",
            },
        },
    )
    return score_panel


def test_candidate_review_matrix_dry_run_writes_variant_commands(tmp_path: Path) -> None:
    source_bridge = tmp_path / "source_bridge"
    _write_source_bridge(source_bridge)

    report = matrix.build_candidate_review_matrix(
        run_tag="unit_matrix",
        source_bridge_root=source_bridge,
        output_root=tmp_path / "out",
        dataset_id="dataset",
        pool_view_id="pool",
        data_lake_root=tmp_path / "lake",
        holding_counts=(10, 20),
        max_weights=(0.05, 0.08),
        rebalance_freqs=("5d",),
        rebalance_offset_modes=("all",),
        transaction_cost_bps_values=(3.0,),
        slippage_bps_values=(7.0,),
        sell_tax_bps_values=(10.0,),
        run_backtests=False,
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "completed"
    assert report["summary"]["variant_count"] == 4
    assert report["summary"]["completed_backtest_count"] == 0
    assert report["boundary"]["promotion_allowed"] is False
    assert report["boundary"]["active_execution_strategy_expected_diff"] == "none"
    first_command = report["results"][0]["command"]
    assert "--data-source" in first_command
    assert first_command[first_command.index("--data-source") + 1] == "lake"
    assert first_command[first_command.index("--rebalance-offset-mode") + 1] == "all"
    assert Path(report["summary_csv"]).exists()
    summary = pd.read_csv(report["summary_csv"])
    assert len(summary) == 4


def test_candidate_review_matrix_collects_completed_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_bridge = tmp_path / "source_bridge"
    _write_source_bridge(source_bridge)

    def fake_run(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, object]:
        output_dir = Path(command[command.index("--output-dir") + 1])
        experiment_tag = command[command.index("--experiment-tag") + 1]
        run_dir = output_dir / experiment_tag
        _write_json(
            run_dir / "metrics.json",
            {
                "annual_return": 0.40,
                "excess_annual_return": 0.30,
                "excess_sharpe": 1.25,
                "max_drawdown": -0.20,
                "avg_turnover": 0.10,
                "annual_return_cost_drag": 0.05,
            },
        )
        _write_json(
            run_dir / "monthly_backtest_diagnostics.json",
            {
                "positive_month_ratio": 0.65,
                "negative_month_count": 2,
                "worst_monthly_return": -0.03,
                "issue_flags": [],
            },
        )
        return {"returncode": 0, "stdout": str(stdout_path), "stderr": str(stderr_path)}

    monkeypatch.setattr(matrix, "_run_command", fake_run)

    report = matrix.build_candidate_review_matrix(
        run_tag="unit_matrix",
        source_bridge_root=source_bridge,
        output_root=tmp_path / "out",
        dataset_id="dataset",
        pool_view_id="pool",
        data_lake_root=tmp_path / "lake",
        holding_counts=(20,),
        max_weights=(0.12,),
        rebalance_freqs=("10d",),
        transaction_cost_bps_values=(10.0,),
        slippage_bps_values=(5.0,),
        run_backtests=True,
        enforce_active_artifact_clean=False,
    )

    assert report["summary"]["completed_backtest_count"] == 1
    assert report["summary"]["promotion_review_eligible_count"] == 0
    assert report["summary"]["small_capital_gate_id"] == matrix.SMALL_CAPITAL_GATE_ID
    assert report["summary"]["small_capital_positive_transfer_count"] == 1
    assert report["summary"]["small_capital_research_grade_candidate_count"] == 1
    assert report["results"][0]["promotion_review_eligible"] is False
    assert report["results"][0]["small_capital_positive_transfer"] is True
    assert report["results"][0]["small_capital_research_grade_candidate"] is True
    assert Path(tmp_path / "out" / f"{matrix.SMALL_CAPITAL_GATE_ID}_report.json").exists()
    summary = pd.read_csv(report["summary_csv"])
    assert bool(summary.loc[0, "small_capital_research_grade_candidate"]) is True


def test_candidate_review_matrix_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(matrix, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        matrix.build_candidate_review_matrix(output_root=tmp_path / "out")
