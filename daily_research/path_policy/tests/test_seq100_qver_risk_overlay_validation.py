from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_qver_risk_overlay_validation as risk


def test_frozen_contract_uses_one_unblended_equal_head_challenger() -> None:
    study, _ = risk.load_study()
    assert study["candidate_band"] == {
        "source_score": "exact_full_score",
        "source_eligibility": "eligible__exact_full_top10",
        "scan_count": 30,
        "maximum_names_per_pit_industry": 2,
        "selection_count": 10,
        "definition": "the exact frozen old-account Top30 industry-capped scan list, formed before any outcome read",
    }
    weights = study["reranking"]["head_weights"]
    assert set(weights) == set(risk.HEAD_NAMES)
    assert np.isclose(sum(weights.values()), 1.0)
    assert study["reranking"]["core_score_blending"] is False
    assert study["reranking"]["hard_risk_threshold"] is None
    assert study["reranking"]["parameter_grid_performed"] is False


def test_candidate_band_respects_rank_count_and_industry_cap() -> None:
    study, _ = risk.load_study()
    rows = []
    for date_idx in (100, 101):
        for offset in range(50):
            rows.append(
                {
                    "date_idx": date_idx,
                    "candidate_id": date_idx * 1000 + offset,
                    "industry_code": offset // 2,
                    "exact_full_score": 1.0 - offset / 100.0,
                    "eligible__exact_full_top10": True,
                }
            )
    band = risk._candidate_band(pd.DataFrame(rows), study)
    assert band.groupby("date_idx").size().eq(30).all()
    assert band.groupby(["date_idx", "industry_code"]).size().le(2).all()
    assert (
        band.groupby("date_idx")["core_band_rank"]
        .apply(lambda values: values.tolist() == list(range(1, 31)))
        .all()
    )


def test_neutral_risk_preserves_old_core_order() -> None:
    local = pd.DataFrame(
        {
            "candidate_id": [30, 10, 20],
            "core_band_rank": [1, 2, 3],
        }
    )
    result = risk._rerank_local(
        local,
        head_weights={head: 1.0 / 3.0 for head in risk.HEAD_NAMES},
        all_heads_ready=False,
    )
    assert result["overlay_rank"].tolist() == [1, 2, 3]
    assert not result["reranking_active"].any()


def test_active_risk_reranking_uses_equal_probability_ranks() -> None:
    local = pd.DataFrame(
        {
            "candidate_id": [1, 2, 3],
            "core_band_rank": [1, 2, 3],
            "probability_revision_reversal": [0.9, 0.2, 0.5],
            "probability_multiple_compression": [0.8, 0.1, 0.4],
            "probability_early_spike_fade": [0.7, 0.3, 0.5],
        }
    )
    result = risk._rerank_local(
        local,
        head_weights={head: 1.0 / 3.0 for head in risk.HEAD_NAMES},
        all_heads_ready=True,
    ).set_index("candidate_id")
    assert int(result.loc[2, "overlay_rank"]) == 1
    assert int(result.loc[1, "overlay_rank"]) == 3
    assert result["reranking_active"].all()


def test_training_rows_require_strictly_prior_label_availability() -> None:
    frame = pd.DataFrame(
        {
            "label_available_date_idx": [99, 100, 101, np.nan],
            "label_revision_reversal": [0.0, 1.0, 0.0, 1.0],
        }
    )
    train = risk._training_rows(
        frame,
        signal_date_idx=100,
        label_column="label_revision_reversal",
    )
    assert train.index.tolist() == [0]


def test_acceptance_requires_every_frozen_gate() -> None:
    risk_metrics = pd.DataFrame(
        {
            "head": list(risk.HEAD_NAMES),
            "period": ["full_history"] * 3,
            "roc_auc": [0.60, 0.55, 0.52],
        }
    )
    cohorts = [
        {
            "period": "full_history",
            "horizon": 120,
            "policy": "challenger_minus_baseline",
            "paired_difference": {"lcb_95": 0.001},
        }
    ]
    selection = pd.DataFrame(
        {
            "policy": ["baseline_old_core", "challenger_risk_overlay"],
            "early_spike_fade_fraction": [0.20, 0.19],
        }
    )
    account = {
        "runs": {
            "baseline_matched": {
                "metric": {
                    "annualized_log_growth": 0.08,
                    "full_path_maximum_drawdown": -0.29,
                },
                "winner_concentration": {"top_10_positive_pnl_share": 0.10},
            },
            "challenger_matched": {
                "metric": {
                    "annualized_log_growth": 0.081,
                    "full_path_maximum_drawdown": -0.28,
                },
                "winner_concentration": {"top_10_positive_pnl_share": 0.09},
            },
        }
    }
    accepted = risk._acceptance_decision(
        risk_metrics=risk_metrics,
        cohort_summaries=cohorts,
        selection_summary=selection,
        account_summary=account,
    )
    assert accepted["passed"] is True
    risk_metrics.loc[risk_metrics["head"].eq("early_spike_fade"), "roc_auc"] = 0.49
    rejected = risk._acceptance_decision(
        risk_metrics=risk_metrics,
        cohort_summaries=cohorts,
        selection_summary=selection,
        account_summary=account,
    )
    assert rejected["passed"] is False
