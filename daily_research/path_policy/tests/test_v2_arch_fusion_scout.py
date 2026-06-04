from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_arch_fusion_scout as fusion


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_expert_study(
    studies_root: Path,
    tag: str,
    *,
    dataset_id: str = "dataset",
    pool_view_id: str = "pool",
    feature_profile: str = "feature",
    validation_score_scale: float = 1.0,
    test_reverse: bool = False,
) -> None:
    study = studies_root / tag
    _write_json(
        study / "forecast_dataset_manifest.json",
        {
            "source_market_dataset_id": dataset_id,
            "source_pool_view_id": pool_view_id,
            "feature_profile": feature_profile,
            "feature_columns": ["raw_open_gap_1d", "amount_norm"],
            "label_semantics": {"label_semantics": "next_open_entry_to_future_open"},
        },
    )
    for role in ("validation", "test"):
        rows = []
        for date in ["2024-01-02", "2024-01-03"]:
            samples = [
                ("000001.SZ", 0.30, 0.03, 1),
                ("000002.SZ", 0.20, 0.02, 1),
                ("600000.SH", 0.10, -0.01, 0),
            ]
            if role == "test" and test_reverse:
                samples = [(stock, -score, future, hit) for stock, score, future, hit in samples]
            for stock, score, future, hit in samples:
                rows.append(
                    {
                        "date": date,
                        "stock": stock,
                        "pred_decision_score": score * validation_score_scale,
                        "future_decision_score": future,
                        "future_hit_label_5d": hit,
                    }
                )
        pd.DataFrame(rows).to_csv(study / f"forecast_predictions_{role}.csv", index=False)


def _write_neural_study(studies_root: Path, tag: str, *, seed: int) -> None:
    study = studies_root / tag
    study.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in ["2024-01-02", "2024-02-02"]:
        for stock, score, future, hit in (
            ("000001.SZ", 0.30, 0.05, 1),
            ("000002.SZ", 0.20, 0.02, 1),
            ("600000.SH", 0.10, -0.02, 0),
        ):
            rows.append(
                {
                    "date": date,
                    "stock": stock,
                    "pred_decision_score": score,
                    "trade_utility_score": score,
                    "future_decision_score": future,
                    "pred_best_horizon": 30,
                    "future_best_horizon": 5,
                    "pred_decision_utility_5d": score,
                    "future_decision_utility_5d": future,
                    "pred_hit_prob_5d": score,
                    "future_hit_label_5d": hit,
                }
            )
    for role in ("validation", "test"):
        pd.DataFrame(rows).to_csv(study / f"forecast_predictions_{role}.csv", index=False)
    training = {
        "status": "completed",
        "selected_model_family": "hybrid_expert_fusion_static_context",
        "selected_seed": seed,
        "training_config": {
            "loss_profile": "horizon_30d_soft_penalty_v1",
            "output_profile": "decision_utility_v1",
            "feature_profile": "raw_kline_context_v2_tradeable_local_state_v1",
            "forecast_horizon": 30,
            "cumulative_horizons": [5],
        },
        "test_metrics": {
            "decision_score_rank_ic": 0.0,
            "decision_score_top_bottom_spread": 0.0,
            "decision_hit_lift_top20_mean": 0.0,
        },
        "evidence_verdict": "forecast_test_confirmed",
    }
    _write_json(study / "forecast_training_summary.json", training)
    _write_json(
        study / "study_summary.json",
        {
            "status": "completed",
            "evidence_verdict": "forecast_test_confirmed",
            "training_summary": training,
        },
    )


def test_late_fusion_uses_validation_weights_without_test_label_leakage(tmp_path: Path) -> None:
    studies = tmp_path / "studies"
    _write_expert_study(studies, "strong", validation_score_scale=1.0, test_reverse=True)
    _write_expert_study(studies, "weak", validation_score_scale=-1.0, test_reverse=False)

    report = fusion.run_late_fusion(
        output_root=tmp_path / "out",
        run_tag="unit_fusion",
        expert_study_tags=("strong", "weak"),
        expert_names=("strong_expert", "weak_expert"),
        studies_root=studies,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="feature",
        enforce_active_artifact_clean=False,
    )

    weights = json.loads(Path(report["fusion_weight_report_json"]).read_text(encoding="utf-8"))["weights"]
    assert report["status"] == "completed"
    assert report["boundary"]["test_label_used_for_weight_fit"] is False
    assert weights["strong_expert"] > weights["weak_expert"]
    assert Path(report["fusion_score_panel_test_csv"]).exists()


def test_arch_fusion_report_blocks_mismatched_dataset_pool_feature(tmp_path: Path) -> None:
    studies = tmp_path / "studies"
    _write_expert_study(studies, "bad", pool_view_id="wrong_pool")

    report = fusion.run_late_fusion(
        output_root=tmp_path / "out",
        run_tag="unit_fusion",
        expert_study_tags=("bad",),
        expert_names=("bad_expert",),
        studies_root=studies,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="feature",
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "blocked"
    assert "bad_expert:bad:source_pool_view_id_mismatch" in report["blockers"]
    assert Path(tmp_path / "out" / "v2_arch_fusion_scout_report.json").exists()


def test_arch_fusion_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fusion, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        fusion.run_late_fusion(output_root=tmp_path / "out")


def test_neural_fusion_task_list_uses_hybrid_family_and_research_boundary(tmp_path: Path) -> None:
    path = fusion.write_task_list(output_root=tmp_path / "out", seeds=(7,))

    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["training_tasks"][0]

    assert payload["stage"] == "neural_fusion"
    assert payload["boundary"]["promotion_allowed"] is False
    assert task["model_family"] == "hybrid_expert_fusion_static_context"
    assert task["loss_profile"] == "horizon_30d_soft_penalty_v1"
    assert "--forecast-model-families" in task["command"]
    assert task["command"][task["command"].index("--forecast-model-families") + 1] == "hybrid_expert_fusion_static_context"


def test_neural_fusion_comparison_uses_prediction_audit_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    studies = tmp_path / "studies"
    tag = fusion._study_tag(7)
    _write_neural_study(studies, tag, seed=7)
    monkeypatch.setattr(fusion, "STUDIES_ROOT", studies)

    report = fusion.run_neural_fusion_comparison(
        output_root=tmp_path / "out",
        seeds=(7,),
        run_tag="unit_arch_fusion",
    )

    aggregate = report["aggregate"]
    assert report["status"] == "completed"
    assert report["forecast_prediction_audit"]["status"] == "completed"
    assert aggregate["rank_ic_min"] > 0.0
    assert aggregate["top_bottom_spread_min"] > 0.0
    assert aggregate["hit_lift_min"] > 0.0
    assert aggregate["monthly_positive_rate_mean"] == 1.0
    assert aggregate["thirty_d_concentration_mean"] == 1.0
    assert Path(report["comparison_csv"]).exists()


def test_neural_fusion_task_list_supports_regime_routed_tier2_family(tmp_path: Path) -> None:
    path = fusion.write_task_list(
        output_root=tmp_path / "out",
        seeds=(7, 11, 19),
        model_family="regime_routed_multi_expert_horizon_v1",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["training_tasks"][0]
    command = task["command"]

    assert payload["model_family"] == "regime_routed_multi_expert_horizon_v1"
    assert payload["tier"] == "tier2_regime_routed_multi_expert"
    assert payload["fusion_version"] == "regime_routed_multi_expert_horizon_v1"
    assert payload["target_parameter_band"] == "8m_to_15m"
    assert 8_000_000 <= int(payload["estimated_parameter_count"]) <= 15_000_000
    assert payload["boundary"]["promotion_allowed"] is False
    assert payload["training_task_count"] == 3
    assert task["tag"] == "mh_v2_arch_fusion_regime_routed_multi_expert_seed7_20260603_01"
    assert task["model_family"] == "regime_routed_multi_expert_horizon_v1"
    assert 8_000_000 <= int(task["estimated_parameter_count"]) <= 15_000_000
    assert "router_entropy" in task["router_diagnostics"]
    assert command[command.index("--forecast-model-families") + 1] == "regime_routed_multi_expert_horizon_v1"
    assert command[command.index("--forecast-hidden-dim") + 1] == "256"
    assert command[command.index("--forecast-transformer-heads") + 1] == "8"
    assert command[command.index("--forecast-patch-sizes") + 1] == "4,10,20"


def test_cli_write_task_list_routes_to_neural_fusion_even_with_default_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = fusion.main(["--write-task-list", "--seeds", "7,11,19", "--output-root", str(tmp_path / "out"), "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    task_list = Path(report["actions"]["task_list"])

    assert rc == 0
    assert report["stage"] == "neural_fusion"
    assert task_list.exists()
    assert json.loads(task_list.read_text(encoding="utf-8"))["training_task_count"] == 3
