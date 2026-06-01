from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daily_research.path_policy.rebuild_lineage_diff_audit import (
    build_rebuild_lineage_diff_audit,
    write_rebuild_lineage_diff_audit,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prediction_frame(*, seed: int, role: str, weak_month: bool = False) -> pd.DataFrame:
    rows = []
    for month in ("2024-01", "2024-02", "2024-03"):
        for idx in range(10):
            score = float(idx)
            future = float(idx) / 100.0
            if weak_month and month == "2024-02":
                score = 10.0 - float(idx)
            row = {
                "date": f"{month}-{idx % 5 + 1:02d}",
                "stock": f"{idx:06d}.SZ",
                "role": role,
                "pred_decision_score": score,
                "future_decision_score": future,
                "pred_best_horizon": 30 if idx % 2 == 0 else 20,
                "future_best_horizon": 20 if idx % 3 == 0 else 5,
                "history_bucket": "high",
                "stock_seen_in_train": True,
                "future_hit_label_20d": 1 if idx >= 6 else 0,
                "future_hit_label_30d": 1 if idx >= 7 else 0,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _seed_study(studies_root: Path, tag: str, *, seed: int, weak_test_month: bool = False) -> None:
    root = studies_root / tag
    root.mkdir(parents=True)
    _write_json(root / "study_summary.json", {"status": "completed", "evidence_verdict": "forecast_test_confirmed", "seed": seed})
    _write_json(root / "forecast_training_summary.json", {"status": "completed", "selected_seed": seed})
    _prediction_frame(seed=seed, role="validation").to_csv(root / "forecast_predictions_validation.csv", index=False)
    _prediction_frame(seed=seed, role="test", weak_month=weak_test_month).to_csv(root / "forecast_predictions_test.csv", index=False)


def _anchor(studies_root: Path) -> None:
    root = studies_root / "anchor"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "role": "test",
                "score_name": "pred_decision_score",
                "seed_count": 3,
                "rank_ic_mean": 0.09,
                "rank_ic_min": 0.05,
                "spread_mean": 0.03,
                "spread_min": 0.01,
                "hit_lift_mean": 0.01,
                "hit_lift_min": -0.001,
                "monthly_positive_rate_mean": 0.78,
                "monthly_positive_rate_min": 0.72,
                "negative_month_count_max": 3,
                "long_horizon_share_mean": 0.69,
                "thirty_d_concentration_mean": 0.66,
                "future_long_horizon_share_mean": 0.42,
                "pred_future_horizon_gap_mean": 14.6,
                "all_seed_rank_ic_spread_hit_positive": False,
                "stage3_weak_gate_pass": False,
            }
        ]
    ).to_csv(root / "profile_aggregate.csv", index=False)
    pd.DataFrame(
        [
            {
                "study_tag": "seed7",
                "seed": 7,
                "role": "test",
                "score_name": "pred_decision_score",
                "decision_score_rank_ic": 0.14,
                "decision_score_top_bottom_spread": 0.05,
                "decision_hit_lift_top20_mean": 0.02,
                "monthly_spread_positive_rate": 0.82,
                "negative_month_count": 2,
                "worst_month_spread": -0.01,
                "long_horizon_share": 0.8,
                "thirty_d_concentration": 0.72,
                "pred_future_horizon_gap": 15.1,
            },
            {
                "study_tag": "seed11",
                "seed": 11,
                "role": "test",
                "score_name": "pred_decision_score",
                "decision_score_rank_ic": 0.05,
                "decision_score_top_bottom_spread": 0.01,
                "decision_hit_lift_top20_mean": -0.001,
                "monthly_spread_positive_rate": 0.72,
                "negative_month_count": 3,
                "worst_month_spread": -0.07,
                "long_horizon_share": 0.7,
                "thirty_d_concentration": 0.7,
                "pred_future_horizon_gap": 15.0,
            },
        ]
    ).to_csv(root / "score_variant_comparison.csv", index=False)
    _write_json(root / "rebuild_decision_score_diagnostics.json", {"conclusion_hint": "no_selection_only_fix_found; inspect target_definition_or_input_regime_contamination"})
    _write_json(root / "rebuild_target_calibration_audit.json", {"gate_a": {"passed": True, "passed_target_count": 3}, "conclusion_hints": ["target_calibration_gate_passed"]})


def _manifest(studies_root: Path) -> None:
    root = studies_root / "seed7"
    _write_json(
        root / "forecast_dataset_manifest.json",
        {
            "source_market_dataset_id": "policy_input_bundle__new",
            "source_pool_view_id": "policy_pool_view__new",
            "source_pool_view_kind": "rolling_liquidity",
            "source_pool_view_name": "rolling_liquid500",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "feature_count_before_cap": 116,
            "feature_count_after_cap": 116,
            "feature_store_shape": [1699, 3668, 116],
            "sample_count": 638527,
            "sample_count_by_role": {"train": 434939, "validation": 100006, "test": 103582},
            "date_values": ["2018-01-02", "2018-01-03"],
            "stock_values": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "role_years": {"train_start_year": 2019, "test_year": 2024},
            "label_semantics": {"label_semantics": "next_open_entry_to_future_open"},
            "feature_group_counts": {"state": 59, "raw_kline": 15},
            "feature_columns": ["current_price", "raw_open_gap_1d"],
            "normalization": {"method": "zscore", "fit_role": "train_only", "raw_feature_nan_ratio": 0.05},
        },
    )
    _write_json(root / "rebuild_memmap_validation.json", {"status": "ok", "blockers": []})


def _lake_manifest(path: Path) -> None:
    _write_json(
        path,
        {
            "datasets": [
                {
                    "dataset_id": "policy_input_bundle__new",
                    "dataset_kind": "policy_input_bundle",
                    "start_date": "2017-01-03",
                    "end_date": "2024-12-31",
                    "row_counts": {"bronze_market_data": 99},
                    "parameters": {"provider_chain": ["research_rebuild_minimal_free"], "provider_plan": "research_rebuild_minimal_free"},
                },
                {
                    "dataset_id": "policy_pool_view__new",
                    "parameters": {"view_kind": "rolling_liquidity", "view_name": "rolling_liquid500"},
                    "source_cache": {"quality_report": {"universe_size": 3668, "average_rebalance_turnover": 0.43}},
                },
            ]
        },
    )


def test_build_rebuild_lineage_diff_audit_marks_old_payload_unavailable(tmp_path: Path) -> None:
    studies_root = tmp_path / "studies"
    _seed_study(studies_root, "seed7", seed=7)
    _seed_study(studies_root, "seed11", seed=11, weak_test_month=True)
    _seed_study(studies_root, "seed19", seed=19)
    _anchor(studies_root)
    _manifest(studies_root)
    lake_path = tmp_path / "manifest_latest.json"
    _lake_manifest(lake_path)
    stage28_ref = tmp_path / "stage28.md"
    stage28_ref.write_text(
        "\n".join(
            [
                "- Dataset: `policy_input_bundle__old`.",
                "- Full-pool manifest properties: `universe_size=2430`, `train_rows=452034`, `feature_store_shape=[1699,2430,156]`.",
                "| `target_norm_head_constraint_v1` | 3 | `0.083206` | `0.028098` | `0.010771` | `0.878788` | `2` | `0.815462` | `0.776934` | `0.436126` | `15.618355` | pass |",
            ]
        ),
        encoding="utf-8",
    )
    stage36_ref = tmp_path / "stage36.md"
    stage36_ref.write_text("- Dataset: `policy_input_bundle__old`.\n- Final decision: keep raw GRU baseline.", encoding="utf-8")

    report = build_rebuild_lineage_diff_audit(
        studies_root=studies_root,
        anchor_tag="anchor",
        new_seed_tags=["seed7", "seed11", "seed19"],
        data_lake_manifest_path=lake_path,
        stage28_reference_path=stage28_ref,
        stage36_reference_path=stage36_ref,
        old_stage28_root=tmp_path / "missing_stage28",
        old_stage36_root=tmp_path / "missing_stage36",
        short_v5b_root=tmp_path / "missing_short_v5b",
        active_manifest_path=tmp_path / "missing_active.json",
    )

    assert report["status"] == "completed"
    assert report["new_lineage"]["manifest"]["source_market_dataset_id"] == "policy_input_bundle__new"
    assert report["historical_evidence"]["stage28"]["dataset_id"] == "policy_input_bundle__old"
    assert report["historical_evidence"]["stage28"]["payload_exists"] is False
    assert report["historical_evidence"]["stage28"]["feature_store_shape"] == [1699, 2430, 156]
    assert report["payload_availability"]["short_v5b_production_root"]["exists"] is False
    assert report["root_cause_rankings"][0]["cause"] == "new_lineage_not_payload_restore"
    assert report["new_lineage"]["seed_prediction_stats"][1]["test"]["monthly_spread"]["negative_months"] == ["2024-02"]


def test_write_rebuild_lineage_diff_audit_outputs_report_tables(tmp_path: Path) -> None:
    studies_root = tmp_path / "studies"
    _seed_study(studies_root, "seed7", seed=7)
    _anchor(studies_root)
    _manifest(studies_root)
    lake_path = tmp_path / "manifest_latest.json"
    _lake_manifest(lake_path)
    stage28_ref = tmp_path / "stage28.md"
    stage28_ref.write_text("- Dataset: `policy_input_bundle__old`.", encoding="utf-8")
    stage36_ref = tmp_path / "stage36.md"
    stage36_ref.write_text("- Dataset: `policy_input_bundle__old`.", encoding="utf-8")
    report = build_rebuild_lineage_diff_audit(
        studies_root=studies_root,
        anchor_tag="anchor",
        new_seed_tags=["seed7"],
        data_lake_manifest_path=lake_path,
        stage28_reference_path=stage28_ref,
        stage36_reference_path=stage36_ref,
    )

    paths = write_rebuild_lineage_diff_audit(report, tmp_path / "out")

    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    monthly = pd.read_csv(paths["seed_month_csv"])
    features = pd.read_csv(paths["feature_columns_csv"])
    assert loaded["status"] == "completed"
    assert "Root Cause Ranking" in markdown
    assert {"tag", "seed", "role", "month", "spread", "is_negative"}.issubset(monthly.columns)
    assert features["feature"].tolist() == ["current_price", "raw_open_gap_1d"]
