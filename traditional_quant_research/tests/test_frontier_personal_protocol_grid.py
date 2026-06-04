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
    frame = pd.DataFrame(
        [
            {
                "constraint_variant": "baseline",
                "top_n": top_n,
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
                "top_n": top_n,
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
    return frame


def test_build_personal_protocol_ledger_ranks_promoted_protocols() -> None:
    protocol_runs = [
        {
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
        combined_calls.append(kwargs)
        run_dir = Path(kwargs["output_dir"]) / "combined_grid"
        run_dir.mkdir(parents=True)
        rows = []
        eval_year = int(kwargs["years"][0])
        for top_n in kwargs["top_n_values"]:
            rows.append(
                {
                    "eval_year": eval_year,
                    "top_n": int(top_n),
                    "signal": "multifactor_rolling_ic_weighted_score",
                    "constraint_variant": "baseline",
                    "evidence_grade": "backtest_only",
                    "exposure_penalty_cols": "log_amount_mean_20d_z",
                    "exposure_penalty_strength": 0.25,
                    "group_col": "industry",
                    "max_group_weight": 0.1,
                    "fee_bps": 30.0,
                    "capital_amount": 100_000_000.0,
                    "impact_bps_per_1pct": 10.0,
                    "annualized_return": 0.12 if int(top_n) == 20 else 0.03,
                    "sharpe": 1.0,
                    "max_drawdown": -0.14,
                    "mean_turnover": 0.4,
                    "mean_impact_cost": 0.001,
                    "mean_total_cost": 0.004,
                    "periods": 6,
                    "constraint_fallback_count": 0,
                    "constraint_fallback_rate": 0.0,
                }
            )
            rows.append(
                {
                    "eval_year": eval_year,
                    "top_n": int(top_n),
                    "signal": "multifactor_low_corr_rank_score",
                    "constraint_variant": "baseline",
                    "evidence_grade": "backtest_only",
                    "exposure_penalty_cols": "log_amount_mean_20d_z",
                    "exposure_penalty_strength": 1.0,
                    "group_col": "industry",
                    "max_group_weight": 0.1,
                    "fee_bps": 30.0,
                    "capital_amount": 100_000_000.0,
                    "impact_bps_per_1pct": 10.0,
                    "annualized_return": 0.02,
                    "sharpe": 0.3,
                    "max_drawdown": -0.18,
                    "mean_turnover": 0.5,
                    "mean_impact_cost": 0.001,
                    "mean_total_cost": 0.004,
                    "periods": 6,
                    "constraint_fallback_count": 0,
                    "constraint_fallback_rate": 0.0,
                }
            )
        pd.DataFrame(rows).to_csv(run_dir / "combined_constraint_summary.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "snapshot_id": "baostock_v2_fixture",
                    "horizon": kwargs["horizon"],
                    "rebalance_frequency": kwargs["rebalance_frequency"],
                    "top_n": kwargs["top_n"],
                    "top_n_values": list(kwargs["top_n_values"]),
                    "buffer_multiplier": kwargs["buffer_multiplier"],
                    "execution_constraints": kwargs["execution_constraints"],
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": "combined_grid",
            "output_dir": str(run_dir),
            "snapshot_id": "baostock_v2_fixture",
            "horizon": kwargs["horizon"],
            "rebalance_frequency": kwargs["rebalance_frequency"],
            "top_n": kwargs["top_n"],
            "top_n_values": list(kwargs["top_n_values"]),
            "buffer_multiplier": kwargs["buffer_multiplier"],
            "execution_constraints": kwargs["execution_constraints"],
        }

    def fake_gate(**kwargs):
        gate_calls.append(kwargs)
        run_dir = Path(kwargs["output_dir"]) / "gate_grid"
        run_dir.mkdir(parents=True)
        pd.concat(
            [
                _gate_frame(20, promoted=True),
                _gate_frame(50, promoted=False),
            ],
            ignore_index=True,
        ).to_csv(run_dir / "personal_candidate_gate_summary.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": "gate_grid",
                    "combined_run_dir": str(kwargs["combined_run_dir"]),
                    "personal_backtest_candidate_count": 1,
                    "strategy_candidate_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return {
            "run_id": "gate_grid",
            "run_dir": str(run_dir),
            "personal_backtest_candidate_count": 1,
            "strategy_candidate_count": 0,
        }

    monkeypatch.setattr(frontier_personal_protocol_grid, "run_low_corr_frontier_combined_constraint_audit", fake_combined)
    monkeypatch.setattr(frontier_personal_protocol_grid, "run_frontier_personal_candidate_gate", fake_gate)

    result = frontier_personal_protocol_grid.run_frontier_personal_protocol_grid(
        years=(2017, 2018),
        horizon=20,
        factor_set="expanded",
        top_n_values=(20, 50),
        signals=("multifactor_rolling_ic_weighted_score", "multifactor_low_corr_rank_score"),
        signal_penalty_strengths="multifactor_rolling_ic_weighted_score=0.25,multifactor_low_corr_rank_score=1.0",
        factor_pruning_run_dir=tmp_path / "factor_pruning",
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
    assert result["factor_set"] == "expanded"
    assert result["best_top_n"] == 20
    assert len(combined_calls) == 2
    assert len(gate_calls) == 1
    assert {tuple(call["years"]) for call in combined_calls} == {(2017,), (2018,)}
    assert {tuple(call["top_n_values"]) for call in combined_calls} == {(20, 50)}
    assert {call["factor_set"] for call in combined_calls} == {"expanded"}
    assert {call["factor_pruning_run_dir"] for call in combined_calls} == {tmp_path / "factor_pruning"}
    assert (run_dir / "personal_protocol_grid_progress.csv").exists()
    progress = pd.read_csv(run_dir / "personal_protocol_grid_progress.csv")
    assert set(progress["status"]) == {"completed"}
    assert (run_dir / "personal_protocol_grid_ledger.csv").exists()
    assert (run_dir / "personal_protocol_grid_top_n_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "protocol_grid.md").exists()
    ledger = pd.read_csv(run_dir / "personal_protocol_grid_ledger.csv")
    assert set(ledger["top_n"]) == {20, 50}
    assert PERSONAL_BACKTEST_PROMOTION_LEVEL in set(ledger["promotion_level"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["factor_set"] == "expanded"


def test_protocol_grid_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError, match="top_n_values must be positive"):
        frontier_personal_protocol_grid.run_frontier_personal_protocol_grid(top_n_values=(0,))


def test_run_yearly_combined_constraint_grid_resumes_completed_year(tmp_path: Path, monkeypatch) -> None:
    existing_run = tmp_path / "yearly" / "year_2017" / "combined_grid_2017"
    existing_run.mkdir(parents=True)
    (existing_run / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "combined_grid_2017",
                "output_dir": str(existing_run),
                "snapshot_id": "baostock_v2_fixture",
                "horizon": 20,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"eval_year": 2017, "top_n": 20}]).to_csv(
        existing_run / "combined_constraint_summary.csv",
        index=False,
    )
    pd.DataFrame([{"eval_year": 2017, "fit_end_date": "2016-12-31", "start_date": "2017-01-01"}]).to_csv(
        existing_run / "combined_constraint_meta.csv",
        index=False,
    )
    progress_path = tmp_path / "personal_protocol_grid_progress.csv"
    pd.DataFrame(
        [
            {
                "eval_year": 2017,
                "status": "completed",
                "started_at": "before",
                "finished_at": "before",
                "combined_run_dir": str(existing_run),
                "error": "",
            },
            {
                "eval_year": 2018,
                "status": "failed",
                "started_at": "before",
                "finished_at": "before",
                "combined_run_dir": "",
                "error": "interrupted",
            },
        ]
    ).to_csv(progress_path, index=False)

    combined_calls: list[dict[str, object]] = []

    def fake_combined(**kwargs):
        combined_calls.append(kwargs)
        year = int(kwargs["years"][0])
        run_dir = Path(kwargs["output_dir"]) / f"combined_grid_{year}"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": f"combined_grid_{year}",
                    "output_dir": str(run_dir),
                    "snapshot_id": "baostock_v2_fixture",
                    "horizon": kwargs["horizon"],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"eval_year": year, "top_n": int(kwargs["top_n"])}]).to_csv(
            run_dir / "combined_constraint_summary.csv",
            index=False,
        )
        pd.DataFrame([{"eval_year": year, "fit_end_date": f"{year - 1}-12-31", "start_date": f"{year}-01-01"}]).to_csv(
            run_dir / "combined_constraint_meta.csv",
            index=False,
        )
        return {
            "run_id": f"combined_grid_{year}",
            "output_dir": str(run_dir),
            "snapshot_id": "baostock_v2_fixture",
            "horizon": kwargs["horizon"],
        }

    monkeypatch.setattr(frontier_personal_protocol_grid, "run_low_corr_frontier_combined_constraint_audit", fake_combined)

    results = frontier_personal_protocol_grid.run_yearly_combined_constraint_grid(
        root=None,
        years=(2017, 2018),
        final_end_date="2026-06-01",
        horizon=20,
        label_mode="raw",
        factor_set="expanded",
        max_factor_corr=0.7,
        rolling_window=120,
        rolling_min_periods=30,
        signals=("multifactor_rolling_ic_weighted_score",),
        signal_penalty_strengths={"multifactor_rolling_ic_weighted_score": 0.25},
        top_n_values=(20, 50),
        rebalance_frequency="monthly",
        buffer_multiplier=3.0,
        fee_bps_values=(30.0,),
        capital_amounts=(100_000_000.0,),
        impact_bps_per_1pct_values=(10.0,),
        exposure_penalty_cols=("log_amount_mean_20d_z",),
        exposure_columns=("log_amount_mean_20d_z",),
        exposure_constraint_cols=(),
        max_abs_exposure=None,
        group_col="industry",
        max_group_weight=0.1,
        execution_constraints=True,
        limit_threshold=0.095,
        include_metrics=True,
        include_industry=True,
        weak_year_rebuild_run_dir=None,
        constraint_variants=None,
        output_dir=tmp_path / "yearly",
        progress_path=progress_path,
        resume=True,
    )

    assert [result["eval_year"] for result in results] == [2017, 2018]
    assert {tuple(call["years"]) for call in combined_calls} == {(2018,)}
    assert {call["factor_set"] for call in combined_calls} == {"expanded"}
    progress = pd.read_csv(progress_path)
    assert progress["status"].tolist() == ["completed", "completed"]
    assert "combined_grid_2017" in progress.loc[progress["eval_year"].eq(2017), "combined_run_dir"].iloc[0]
    assert "combined_grid_2018" in progress.loc[progress["eval_year"].eq(2018), "combined_run_dir"].iloc[0]
