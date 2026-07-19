from __future__ import annotations

import copy

import pandas as pd
import pytest

from daily_research.path_policy import seq100_checkpoint_freshness as freshness
from daily_research.path_policy.seq100_development import _parser


def _gate_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in freshness.PROFILE_ORDER:
        for vintage, offset in ((2023, 0.00), (2024, 0.01), (2025, 0.03)):
            rows.append(
                {
                    "profile": profile,
                    "checkpoint_vintage": vintage,
                    "metrics": {
                        "top1_base_alpha": offset + 0.01,
                        "top3_base_alpha": offset + 0.02,
                        "top5_base_alpha": offset + 0.03,
                        "top10_base_alpha": offset + 0.04,
                        "rank_ic": offset + 0.05,
                        "top3_opportunity_alpha": offset + 0.06,
                        "exit_regret": 0.30 - offset,
                        "path_mae": 0.20 - offset,
                    },
                }
            )
    return rows


def _topk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "top_k": top_k,
                "alpha_net_realized_plan_return_base": 0.01 * top_k,
                "alpha_net_realized_plan_return_stress": 0.009 * top_k,
                "alpha_opportunity_value": 0.02 * top_k,
                "selected_net_realized_plan_return_base": 0.015 * top_k,
                "universe_net_realized_plan_return_base": 0.005 * top_k,
                "universe_label_coverage": 0.99,
                "universe_oracle_regret": 0.12,
                "universe_oracle_regret_coverage": 0.99,
            }
            for top_k in freshness.TOP_K_VALUES
        ]
    )


def _daily_topk_frame(*, coverage: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "2025-01-02",
                "top_k": top_k,
                "selected_realized_plan_return_coverage": coverage,
                "universe_realized_plan_return_coverage": coverage,
                "universe_count": 100,
                "universe_hash": "same-universe",
                "execution_cost_contract_sha256": "same-costs",
            }
            for top_k in freshness.TOP_K_VALUES
        ]
    )


def _split_metrics() -> dict[str, float | int]:
    return {
        "row_count": 100,
        "date_count": 1,
        "candidate_complete_score_coverage": 1.0,
        "rank_ic_mean": 0.1,
        "rank_ic_positive_day_rate": 1.0,
        "path_mae": 0.2,
        "path_open_mae": 0.21,
        "path_high_mae": 0.22,
        "path_low_mae": 0.23,
        "path_close_mae": 0.24,
        "entry_fill_rate": 0.99,
    }


def test_checkpoint_normalization_comes_from_fold_contract_not_amp_scaler() -> None:
    normalization = {
        "fit_date_end_exclusive": "2024-01-02",
        "daily_raw": {"mean": [1.0], "std": [2.0]},
    }
    digest = freshness._canonical_digest(normalization)
    checkpoint = {
        "fold_training_contract": {
            "normalization_sha256": digest,
            "payload": {"normalization": normalization},
        },
        "scaler_state_dict": {"scale": 65536.0},
    }

    observed, observed_digest = freshness._normalization_from_checkpoint(checkpoint)

    assert observed == normalization
    assert observed_digest == digest
    assert "scale" not in observed


def test_freshness_gate_requires_breadth_ranking_and_path_support_for_both_profiles() -> None:
    passing = _gate_results()
    assert freshness.freshness_gate(passing)["metric_gate_pass"] is True

    failing = copy.deepcopy(passing)
    target = next(
        row
        for row in failing
        if row["profile"] == "structured_joint_turnover"
        and row["checkpoint_vintage"] == 2025
    )
    target["metrics"]["top3_base_alpha"] = -1.0  # type: ignore[index]

    decision = freshness.freshness_gate(failing)
    assert decision["metric_gate_pass"] is False
    assert (
        decision["profiles"]["structured_joint_turnover"]["primary_top3_pass"]
        is False
    )


def test_evaluation_metrics_require_complete_score_and_execution_coverage() -> None:
    metrics = freshness._evaluation_metrics(
        topk=_topk_frame(),
        daily_topk=_daily_topk_frame(),
        split_metrics=_split_metrics(),
    )
    assert metrics["candidate_score_coverage"] == 1.0
    assert metrics["execution_return_coverage"] == 1.0
    assert metrics["top3_base_alpha"] == pytest.approx(0.03)

    incomplete_score = _split_metrics()
    incomplete_score["candidate_complete_score_coverage"] = 0.999
    with pytest.raises(ValueError, match="score coverage"):
        freshness._evaluation_metrics(
            topk=_topk_frame(),
            daily_topk=_daily_topk_frame(),
            split_metrics=incomplete_score,
        )
    with pytest.raises(ValueError, match="execution coverage"):
        freshness._evaluation_metrics(
            topk=_topk_frame(),
            daily_topk=_daily_topk_frame(coverage=0.99),
            split_metrics=_split_metrics(),
        )


def test_cli_exposes_resumable_vintage_commands() -> None:
    evaluate = _parser().parse_args(
        ["evaluate-vintages", "--experiment", "structured-path-v1", "--max-tasks", "2"]
    )
    worker = _parser().parse_args(
        [
            "evaluate-vintage-task",
            "--suite-contract",
            "contract.json",
            "--profile",
            "legal_flat_baseline",
            "--vintage",
            "2023",
        ]
    )

    assert evaluate.command == "evaluate-vintages"
    assert evaluate.max_tasks == 2
    assert worker.command == "evaluate-vintage-task"
    assert worker.vintage == 2023


def test_only_older_vintages_are_registered_for_fresh_inference() -> None:
    assert [year for year in freshness.CHECKPOINT_VINTAGES if year != freshness.TARGET_YEAR] == [
        2023,
        2024,
    ]
    assert freshness.TARGET_YEAR == 2025
