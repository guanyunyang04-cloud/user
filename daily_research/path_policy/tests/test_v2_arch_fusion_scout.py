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
