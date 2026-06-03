from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_personal_paper_tracking_bootstrap import (
    BOOTSTRAPPED_TRACKING_STATUS,
    PERSONAL_PAPER_CANDIDATE_LEVEL,
    build_paper_tracking_candidates,
    build_paper_tracking_protocol,
    run_frontier_personal_paper_tracking_bootstrap,
)


def _gate(promoted: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "constraint_variant": "baseline",
                "signal": "multifactor_rolling_ic_weighted_score",
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "personal_capital_amount": 1_000_000.0,
                "exposure_penalty_strength": 0.25,
                "top_n": 200,
                "mean_annualized_return": 0.10,
                "min_annualized_return": -0.30,
                "positive_year_rate": 0.60,
                "worst_max_drawdown": -0.14,
                "total_periods": 60,
                "eval_year_count": 10,
                "max_proxy_mean_abs_active_exposure": 1.10,
                "promoted": promoted,
                "promotion_level": "personal_backtest_candidate" if promoted else "personal_research/backtest_only",
                "paper_tracking_recommendation": "start_paper_tracking" if promoted else "continue_research",
            }
        ]
    )


def _gate_summary() -> dict[str, object]:
    return {
        "run_id": "personal_gate_run",
        "north_star": "Baostock-only personal quant strategy research",
        "combined_run_dir": "combined",
        "snapshot_id": "baostock_v2_fixture",
        "required_year_window": "2017-2026",
    }


def _combined_summary() -> dict[str, object]:
    return {
        "run_id": "combined_run",
        "snapshot_id": "baostock_v2_fixture",
        "horizon": 20,
        "rebalance_frequency": "monthly",
        "top_n": 200,
        "buffer_multiplier": 3.0,
    }


def test_build_paper_tracking_candidates_from_personal_backtest_candidate() -> None:
    candidates = build_paper_tracking_candidates(_gate(), _gate_summary(), _combined_summary())

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["candidate_id"] == "multifactor_rolling_ic_weighted_score_baseline_penalty_0_25_top200"
    assert row["tracking_status"] == BOOTSTRAPPED_TRACKING_STATUS
    assert row["current_level"] == "personal_backtest_candidate"
    assert row["target_next_level"] == PERSONAL_PAPER_CANDIDATE_LEVEL
    assert bool(row["paper_tracking_started"]) is False
    assert bool(row["personal_paper_candidate"]) is False
    assert bool(row["strategy_candidate"]) is False


def test_build_paper_tracking_candidates_keeps_top_n_ids_distinct() -> None:
    gate = pd.concat([_gate(), _gate()], ignore_index=True)
    gate.loc[0, "top_n"] = 20
    gate.loc[1, "top_n"] = 50

    candidates = build_paper_tracking_candidates(gate, _gate_summary(), _combined_summary())

    assert set(candidates["candidate_id"]) == {
        "multifactor_rolling_ic_weighted_score_baseline_penalty_0_25_top20",
        "multifactor_rolling_ic_weighted_score_baseline_penalty_0_25_top50",
    }


def test_build_paper_tracking_protocol_keeps_personal_boundary() -> None:
    candidates = build_paper_tracking_candidates(_gate(), _gate_summary(), _combined_summary())
    protocol = build_paper_tracking_protocol(
        candidates,
        _combined_summary(),
        min_tracking_periods=6,
        min_tracking_days=120,
        max_paper_drawdown=-0.20,
        max_single_period_loss=-0.12,
        min_paper_excess_return=-0.02,
    )

    row = protocol.iloc[0]
    assert row["execution_mode"] == "paper_tracking_only"
    assert bool(row["true_size_required"]) is False
    assert bool(row["institutional_promotion_allowed"]) is False
    assert row["min_tracking_periods"] == 6
    assert row["top_n"] == 200


def test_run_bootstrap_writes_tracking_artifacts(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gate"
    combined_dir = tmp_path / "combined"
    gate_dir.mkdir()
    combined_dir.mkdir()
    gate_summary = {**_gate_summary(), "combined_run_dir": str(combined_dir)}
    _gate().to_csv(gate_dir / "personal_candidate_gate_summary.csv", index=False)
    (gate_dir / "summary.json").write_text(json.dumps(gate_summary), encoding="utf-8")
    (combined_dir / "summary.json").write_text(json.dumps(_combined_summary()), encoding="utf-8")

    result = run_frontier_personal_paper_tracking_bootstrap(
        personal_gate_run_dir=gate_dir,
        output_dir=tmp_path / "out",
        write_research_log=True,
        research_log_path=tmp_path / "paper_tracking.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "paper_tracking_bootstrap_ready"
    assert result["personal_backtest_candidate_count"] == 1
    assert result["personal_paper_candidate_count"] == 0
    assert result["strategy_candidate_count"] == 0
    assert (run_dir / "paper_tracking_candidates.csv").exists()
    assert (run_dir / "paper_tracking_protocol.csv").exists()
    assert (run_dir / "paper_tracking_log_template.csv").exists()
    assert (run_dir / "paper_tracking_review_rules.csv").exists()
    assert (tmp_path / "paper_tracking.md").exists()


def test_run_bootstrap_blocks_when_no_personal_candidate(tmp_path: Path) -> None:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    _gate(promoted=False).to_csv(gate_dir / "personal_candidate_gate_summary.csv", index=False)
    (gate_dir / "summary.json").write_text(json.dumps(_gate_summary()), encoding="utf-8")

    result = run_frontier_personal_paper_tracking_bootstrap(personal_gate_run_dir=gate_dir, output_dir=tmp_path / "out")

    assert result["decision"] == "no_personal_backtest_candidate_to_track"
    assert result["personal_backtest_candidate_count"] == 0
    assert result["personal_paper_candidate_count"] == 0
    saved = pd.read_csv(Path(result["run_dir"]) / "paper_tracking_candidates.csv")
    assert saved.empty
