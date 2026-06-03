from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import frontier_personal_protocol_grid
from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
)


def _gate_frame(top_n: int, *, promoted: bool) -> pd.DataFrame:
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
                "mean_annualized_return": 0.12 if promoted else 0.03,
                "min_annualized_return": -0.22 if promoted else -0.42,
                "positive_year_rate": 0.70 if promoted else 0.40,
                "worst_max_drawdown": -0.14,
                "total_periods": 60,
                "eval_year_count": 10,
                "max_proxy_mean_abs_active_exposure": 1.00,
                "promoted": promoted,
                "promotion_level": PERSONAL_BACKTEST_PROMOTION_LEVEL if promoted else "personal_research/backtest_only",
                "paper_tracking_recommendation": "start_paper_tracking" if promoted else "continue_research",
                "failed_gates": "" if promoted else "return_gate,weak_year_damage_gate",
            },
            {
                "constraint_variant": "baseline",
                "signal": "multifactor_low_corr_rank_score",
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "personal_capital_amount": 1_000_000.0,
                "exposure_penalty_strength": 1.0,
                "mean_annualized_return": 0.02,
                "min_annualized_return": -0.38,
                "positive_year_rate": 0.50,
                "worst_max_drawdown": -0.18,
                "total_periods": 60,
                "eval_year_count": 10,
                "max_proxy_mean_abs_active_exposure": 1.10,
                "promoted": False,
                "promotion_level": "personal_research/backtest_only",
                "paper_tracking_recommendation": "continue_research",
                "failed_gates": "return_gate",
            },
        ]
    )


def test_build_personal_protocol_ledger_ranks_promoted_protocols() -> None:
    protocol_runs = [
        {
            "top_n": 20,
            "combined_result": {
                "horizon": 20,
                "rebalance_frequency": "monthly",
                "buffer_multiplier": 3.0,
                "output_dir": "combined_top20",
            },
            "personal_gate_result": {"run_dir": "gate_top20"},
            "gate": _gate_frame(20, promoted=True),
        },
        {
            "top_n": 50,
            "combined_result": {
                "horizon": 20,
                "rebalance_frequency": "monthly",
                "buffer_multiplier": 3.0,
                "output_dir": "combined_top50",
            },
            "personal_gate_result": {"run_dir": "gate_top50"},
            "gate": _gate_frame(50, promoted=False),
        },
    ]

    ledger = frontier_personal_protocol_grid.rank_protocol_ledger(
        frontier_personal_protocol_grid.build_personal_protocol_ledger(protocol_runs)
    )
    top_n_summary = frontier_personal_protocol_grid.build_top_n_summary(ledger)

    assert len(ledger) == 4
    assert ledger.iloc[0]["top_n"] == 20
    assert ledger.iloc[0]["promotion_level"] == PERSONAL_BACKTEST_PROMOTION_LEVEL
    assert ledger.iloc[0]["protocol_id"].startswith("top20_multifactor_rolling_ic_weighted_score")
    assert bool(ledger.iloc[0]["strategy_candidate"]) is False
    assert set(top_n_summary["top_n"]) == {20, 50}
    top20 = top_n_summary.loc[top_n_summary["top_n"].eq(20)].iloc[0]
    top50 = top_n_summary.loc[top_n_summary["top_n"].eq(50)].iloc[0]
    assert top20["decision"] == "paper_tracking_ready"
    assert top50["decision"] == "continue_research"


def test_run_frontier_personal_protocol_grid_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    combined_calls: list[dict[str, object]] = []
    gate_calls: list[dict[str, object]] = []

    def fake_combined(**kwargs):
        top_n = int(kwargs["top_n"])
        combined_calls.append(kwargs)
        run_dir = Path(kwargs["output_dir"]) / f"combined_{top_n}"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "snapshot_id": "baostock_v2_fixture",
                    "horizon": kwargs["horizon"],
                    "rebalance_frequency": kwargs["rebalance_frequency"],
                    "top_n": top_n,
                    "buffer_multiplier": kwargs["buffer_multiplier"],
                    "execution_constraints": kwargs["execution_constraints"],
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": f"combined_{top_n}",
            "output_dir": str(run_dir),
            "snapshot_id": "baostock_v2_fixture",
            "horizon": kwargs["horizon"],
            "rebalance_frequency": kwargs["rebalance_frequency"],
            "top_n": top_n,
            "buffer_multiplier": kwargs["buffer_multiplier"],
            "execution_constraints": kwargs["execution_constraints"],
        }

    def fake_gate(**kwargs):
        combined_run_dir = Path(kwargs["combined_run_dir"])
        top_n = int(str(combined_run_dir.parent.parent.name).replace("top_n_", ""))
        gate_calls.append(kwargs)
        promoted = top_n == 20
        run_dir = Path(kwargs["output_dir"]) / f"gate_{top_n}"
        run_dir.mkdir(parents=True)
        _gate_frame(top_n, promoted=promoted).to_csv(run_dir / "personal_candidate_gate_summary.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": f"gate_{top_n}",
                    "combined_run_dir": str(combined_run_dir),
                    "personal_backtest_candidate_count": 1 if promoted else 0,
                    "strategy_candidate_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": f"gate_{top_n}",
            "run_dir": str(run_dir),
            "personal_backtest_candidate_count": 1 if promoted else 0,
            "strategy_candidate_count": 0,
        }

    monkeypatch.setattr(frontier_personal_protocol_grid, "run_low_corr_frontier_combined_constraint_audit", fake_combined)
    monkeypatch.setattr(frontier_personal_protocol_grid, "run_frontier_personal_candidate_gate", fake_gate)

    result = frontier_personal_protocol_grid.run_frontier_personal_protocol_grid(
        years=(2017, 2018),
        horizon=20,
        top_n_values=(20, 50),
        signals=("multifactor_rolling_ic_weighted_score", "multifactor_low_corr_rank_score"),
        signal_penalty_strengths="multifactor_rolling_ic_weighted_score=0.25,multifactor_low_corr_rank_score=1.0",
        min_eval_year_count=2,
        required_start_year=2017,
        required_end_year=2018,
        min_total_periods=20,
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "protocol_grid.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "personal_protocol_candidates_ready"
    assert result["personal_backtest_candidate_count"] == 1
    assert result["personal_paper_candidate_count"] == 0
    assert result["strategy_candidate_count"] == 0
    assert result["best_top_n"] == 20
    assert len(combined_calls) == 2
    assert len(gate_calls) == 2
    assert {call["top_n"] for call in combined_calls} == {20, 50}
    assert (run_dir / "personal_protocol_grid_ledger.csv").exists()
    assert (run_dir / "personal_protocol_grid_top_n_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "protocol_grid.md").exists()
    ledger = pd.read_csv(run_dir / "personal_protocol_grid_ledger.csv")
    assert set(ledger["top_n"]) == {20, 50}
    assert PERSONAL_BACKTEST_PROMOTION_LEVEL in set(ledger["promotion_level"])


def test_protocol_grid_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError, match="top_n_values must be positive"):
        frontier_personal_protocol_grid.run_frontier_personal_protocol_grid(top_n_values=(0,))
