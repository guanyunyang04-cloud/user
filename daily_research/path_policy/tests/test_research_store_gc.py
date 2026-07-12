from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import research_store_gc as gc
from daily_research.path_policy import research_store_view as store_view


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(gc, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(store_view, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


def test_sequence_pack_smoke_is_safe_when_unreferenced(workspace: Path) -> None:
    pack = workspace / "daily_research/data/research_store/sequence_pack/smoke_seq100_path60"
    _write_json(pack / "manifest.json", {"feature_channels": {}, "label_arrays": {}})
    (pack / "labels").mkdir(parents=True)

    item = gc.classify_sequence_pack(pack, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "smoke_sequence_pack"
    assert item.cleanup_action == "delete_directory"
    assert item.safe_to_delete_directory is True


def test_sequence_pack_full_is_kept_until_shared_view_replaces_it(workspace: Path) -> None:
    pack = workspace / "daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_full"
    _write_json(pack / "manifest.json", {"feature_channels": {}, "label_arrays": {}})

    item = gc.classify_sequence_pack(pack, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "full_sequence_pack"
    assert item.cleanup_action == "keep"
    assert item.safe_to_delete_directory is False


def test_sequence_pack_reanchor_view_is_kept(workspace: Path) -> None:
    source = workspace / "daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_full"
    source.mkdir(parents=True)
    view = workspace / "daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_full"
    _write_json(
        view / "manifest.json",
        {
            "feature_channels": {
                "daily_raw": {"path": str(source / "panels/daily_raw.float32.dat")},
            },
            "label_arrays": {},
        },
    )

    item = gc.classify_sequence_pack(view, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "view_or_reanchor_sequence_pack"
    assert item.cleanup_action == "keep"
    assert item.safe_to_delete_directory is False


def test_referenced_smoke_pack_is_not_safe_to_delete(workspace: Path) -> None:
    pack = workspace / "daily_research/data/research_store/sequence_pack/smoke_seq100_path20"
    _write_json(pack / "manifest.json", {"feature_channels": {}, "label_arrays": {}})

    item = gc.classify_sequence_pack(
        pack,
        reference_text="keep smoke_seq100_path20 for debugging",
        large_file_threshold_bytes=1,
    )

    assert item.classification == "smoke_sequence_pack"
    assert item.referenced_by_brain is True
    assert item.safe_to_delete_directory is False


def test_cleanup_inventory_does_not_retain_its_delete_candidates(workspace: Path) -> None:
    pack_root = workspace / "daily_research/data/research_store/sequence_pack"
    smoke = pack_root / "smoke_seq100_path20"
    _write_json(smoke / "manifest.json", {"feature_channels": {}, "label_arrays": {}})
    references = workspace / "daily_research/brain/references"
    _write_json(
        references / "process_pack_cleanup_inventory.json",
        {
            "artifact_type": "seq100_process_pack_cleanup_inventory",
            "status": "pre_delete_inventory",
            "candidates": [{"path": str(smoke)}],
        },
    )

    report = gc.build_research_gc_report(
        sequence_pack_roots=[pack_root],
        artifact_roots=[],
        reference_roots=[references],
        large_file_threshold_mb=1,
    )

    assert report["totals"]["safe_delete_candidate_count"] == 1
    assert report["safe_delete_candidates"][0]["name"] == smoke.name


def test_build_report_protects_study_referenced_by_newer_registry(workspace: Path) -> None:
    studies = workspace / "daily_research/output/path_policy/studies"
    retired = studies / "generation_v1"
    _write_json(retired / "study_retirement.json", {"status": "retired"})
    successor = studies / "generation_v2"
    _write_json(successor / "study_summary.json", {"status": "completed"})
    _write_json(
        successor / "development_registry.json",
        {"additional_provenance_paths": [str(retired / "study_retirement.json")]},
    )

    report = gc.build_research_gc_report(
        sequence_pack_roots=[],
        studies_root=studies,
        reference_roots=[],
        large_file_threshold_mb=1,
    )

    retired_item = next(item for item in report["largest_artifacts"] if item["name"] == "generation_v1")
    assert retired_item["safe_to_delete_directory"] is False
    assert retired_item["referenced_by_artifact"] is True
    assert "referenced_by_registered_artifact" in retired_item["reasons"]
    assert retired_item["reference_sources"] == [
        "daily_research/output/path_policy/studies/generation_v2/development_registry.json"
    ]


def test_build_report_honors_explicit_retention_policy_paths(workspace: Path) -> None:
    pack_root = workspace / "daily_research/data/research_store/sequence_pack"
    partial = pack_root / "candidate_partial"
    _write_json(partial / "progress.json", {"status": "interrupted"})
    studies = workspace / "daily_research/output/path_policy/studies"
    retired = studies / "protected_generation_v1"
    _write_json(retired / "study_retirement.json", {"status": "retired"})
    policy_path = workspace / "daily_research/brain/research_store_retention_policy.json"
    _write_json(
        policy_path,
        {
            "artifact_type": "research_store_retention_policy",
            "protected_sequence_pack_paths": ["sequence_pack/candidate_partial"],
            "protected_evidence_chain_paths": [
                "daily_research/output/path_policy/studies/protected_generation_v1"
            ],
        },
    )

    report = gc.build_research_gc_report(
        sequence_pack_roots=[pack_root],
        research_store_root=workspace / "daily_research/data/research_store",
        studies_root=studies,
        policy_path=policy_path,
        reference_roots=[],
        large_file_threshold_mb=1,
    )

    by_key = {(item["artifact_type"], item["name"]): item for item in report["largest_artifacts"]}
    for artifact_type, name in (
        ("sequence_pack", "candidate_partial"),
        ("study", "protected_generation_v1"),
    ):
        item = by_key[(artifact_type, name)]
        assert item["safe_to_delete_directory"] is False
        assert item["referenced_by_artifact"] is True
        assert item["reference_sources"] == [
            "daily_research/brain/research_store_retention_policy.json"
        ]


def test_progress_json_interrupted_pack_is_safe_when_unreferenced(workspace: Path) -> None:
    pack = workspace / "daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_partial"
    _write_json(pack / "manifest.json", {"feature_channels": {}, "label_arrays": {}})
    _write_json(pack / "progress.json", {"status": "interrupted"})

    item = gc.classify_sequence_pack(pack, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "partial_or_interrupted_sequence_pack"
    assert item.safe_to_delete_directory is True
    assert "progress_status=interrupted" in item.reasons


def test_study_large_prediction_file_is_trim_candidate_not_directory_delete(workspace: Path) -> None:
    study = workspace / "daily_research/output/path_policy/studies/qdp_v2_seq100_path60_model"
    _write_json(study / "study_summary.json", {"run_tag": "qdp_v2_seq100_path60_model"})
    prediction = study / "forecast_predictions_test.csv"
    prediction.write_bytes(b"1234567890")

    item = gc.classify_study(study, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "study_with_large_prediction_outputs"
    assert item.cleanup_action == "trim_large_prediction_files"
    assert item.safe_to_delete_directory is False
    assert item.large_files
    assert any(file_item["kind"] == "prediction_output" for file_item in item.large_files)


def test_sequence_training_summary_counts_as_study_summary(workspace: Path) -> None:
    study = workspace / "daily_research/output/path_policy/studies/qdp_v2_seq100_path60_gru"
    _write_json(study / "sequence_path_training_summary.json", {"run_tag": "qdp_v2_seq100_path60_gru"})
    prediction = study / "predictions/test_predictions.csv"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"1234567890")

    item = gc.classify_study(study, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "study_with_large_prediction_outputs"
    assert item.summary_path.endswith("sequence_path_training_summary.json")


def test_path_value_predictability_summary_counts_as_study_summary(workspace: Path) -> None:
    study = workspace / "daily_research/output/path_policy/path_value_predictability/path_value_full"
    _write_json(study / "path_value_predictability_summary.json", {"run_tag": "path_value_full"})
    prediction = study / "predictions/prediction_rows.parquet"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"1234567890")

    item = gc.classify_study(study, reference_text="", large_file_threshold_bytes=1)

    assert item.classification == "study_with_large_prediction_outputs"
    assert item.summary_path.endswith("path_value_predictability_summary.json")


def test_build_report_groups_safe_delete_and_trim_candidates(workspace: Path) -> None:
    pack_root = workspace / "daily_research/data/research_store/sequence_pack"
    smoke = pack_root / "smoke_seq100_path20"
    _write_json(smoke / "manifest.json", {"feature_channels": {}, "label_arrays": {}})
    full = pack_root / "qdp_v2_seq100_path20_full"
    _write_json(full / "manifest.json", {"feature_channels": {}, "label_arrays": {}})

    studies = workspace / "daily_research/output/path_policy/studies"
    study = studies / "qdp_v2_seq100_path60_model"
    _write_json(study / "study_summary.json", {"run_tag": "qdp_v2_seq100_path60_model"})
    (study / "forecast_predictions_test.csv").write_bytes(b"1234567890")

    report = gc.build_research_gc_report(
        sequence_pack_roots=[pack_root],
        studies_root=studies,
        reference_roots=[],
        large_file_threshold_mb=0.000001,
    )

    assert report["status"] == "dry_run"
    assert report["totals"]["safe_delete_candidate_count"] == 1
    assert report["totals"]["prediction_trim_candidate_count"] == 1
    assert report["safe_delete_candidates"][0]["name"] == "smoke_seq100_path20"


def test_trim_prediction_outputs_dry_run_does_not_delete(workspace: Path) -> None:
    studies = workspace / "daily_research/output/path_policy/studies"
    study = studies / "qdp_v2_seq100_path60_model"
    _write_json(study / "study_summary.json", {"run_tag": "qdp_v2_seq100_path60_model"})
    prediction = study / "forecast_predictions_test.csv"
    prediction.write_bytes(b"1234567890")
    metrics = study / "topk_metrics.csv"
    metrics.write_text("top_k,value\n10,1\n", encoding="utf-8")

    result = gc.trim_prediction_outputs(
        studies_root=studies,
        reference_roots=[],
        large_file_threshold_mb=0.000001,
        delete=False,
    )

    assert result["status"] == "dry_run"
    assert result["totals"]["candidate_file_count"] == 1
    assert prediction.exists()
    assert metrics.exists()
    assert not (study / "prediction_trim_manifest.json").exists()


def test_trim_prediction_outputs_requires_confirmation(workspace: Path) -> None:
    studies = workspace / "daily_research/output/path_policy/studies"
    study = studies / "qdp_v2_seq100_path60_model"
    _write_json(study / "study_summary.json", {"run_tag": "qdp_v2_seq100_path60_model"})
    prediction = study / "forecast_predictions_test.csv"
    prediction.write_bytes(b"1234567890")

    with pytest.raises(ValueError):
        gc.trim_prediction_outputs(
            studies_root=studies,
            reference_roots=[],
            large_file_threshold_mb=0.000001,
            delete=True,
            confirm_trim="",
        )

    assert prediction.exists()


def test_trim_prediction_outputs_deletes_only_prediction_files_and_marks_summary(workspace: Path) -> None:
    studies = workspace / "daily_research/output/path_policy/studies"
    study = studies / "qdp_v2_seq100_path60_model"
    _write_json(study / "study_summary.json", {"run_tag": "qdp_v2_seq100_path60_model"})
    prediction = study / "forecast_predictions_test.csv"
    prediction.write_bytes(b"1234567890")
    metrics = study / "topk_metrics.csv"
    metrics.write_text("top_k,value\n10,1\n", encoding="utf-8")
    checkpoint = study / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = gc.trim_prediction_outputs(
        studies_root=studies,
        reference_roots=[],
        large_file_threshold_mb=0.000001,
        delete=True,
        confirm_trim=gc.TRIM_CONFIRMATION,
    )

    assert result["status"] == "trim_executed"
    assert result["totals"]["deleted_file_count"] == 1
    assert not prediction.exists()
    assert metrics.exists()
    assert checkpoint.exists()
    manifest = json.loads((study / "prediction_trim_manifest.json").read_text(encoding="utf-8"))
    assert manifest["deleted_file_count"] == 1
    assert manifest["retained_files"]
    summary = json.loads((study / "study_summary.json").read_text(encoding="utf-8"))
    assert summary["prediction_outputs_trimmed"] is True
    assert summary["prediction_trim_manifest"].endswith("prediction_trim_manifest.json")


def test_trim_prediction_outputs_scans_multiple_artifact_roots(workspace: Path) -> None:
    roots = [
        workspace / "daily_research/output/path_policy/studies",
        workspace / "daily_research/output/path_policy/sequence_path_training",
    ]
    for idx, root in enumerate(roots):
        study = root / f"run_{idx}"
        _write_json(study / "study_summary.json", {"run_tag": f"run_{idx}"})
        (study / f"prediction_rows_{idx}.csv").write_bytes(b"1234567890")

    result = gc.trim_prediction_outputs(
        artifact_roots=roots,
        reference_roots=[],
        large_file_threshold_mb=0.000001,
    )

    assert result["totals"]["candidate_study_count"] == 2
    assert result["totals"]["candidate_file_count"] == 2


def test_delete_safe_candidates_requires_confirmation(workspace: Path) -> None:
    target = workspace / "daily_research/data/research_store/sequence_pack/smoke_seq100_path20"
    target.mkdir(parents=True)
    report = {
        "safe_delete_candidates": [
            {
                "path": "daily_research/data/research_store/sequence_pack/smoke_seq100_path20",
                "size_bytes": 1,
                "safe_to_delete_directory": True,
            }
        ]
    }

    with pytest.raises(ValueError):
        gc.delete_safe_candidates(
            report,
            allowed_roots=["daily_research/data/research_store/sequence_pack"],
            confirm_delete="",
        )

    assert target.exists()


def test_delete_safe_candidates_deletes_only_allowed_safe_directories(workspace: Path) -> None:
    allowed = workspace / "daily_research/data/research_store/sequence_pack"
    safe = allowed / "smoke_seq100_path20"
    safe.mkdir(parents=True)
    outside = workspace / "daily_research/output/path_policy/studies/not_allowed"
    outside.mkdir(parents=True)
    report = {
        "safe_delete_candidates": [
            {
                "path": "daily_research/data/research_store/sequence_pack/smoke_seq100_path20",
                "size_bytes": 7,
                "safe_to_delete_directory": True,
            },
            {
                "path": "daily_research/output/path_policy/studies/not_allowed",
                "size_bytes": 11,
                "safe_to_delete_directory": True,
            },
        ]
    }

    result = gc.delete_safe_candidates(
        report,
        allowed_roots=["daily_research/data/research_store/sequence_pack"],
        confirm_delete=gc.DELETE_CONFIRMATION,
    )

    assert result["deleted_count"] == 1
    assert not safe.exists()
    assert outside.exists()
    assert result["skipped"][0]["reason"] == "outside_allowed_roots"


def test_prune_cold_store_archives_and_preserves_active_view_dependencies(workspace: Path) -> None:
    store = workspace / "daily_research/data/research_store"
    active_sample = store / "sample_index/active.parquet"
    active_sample.parent.mkdir(parents=True)
    active_sample.write_bytes(b"active")
    cold_sample = store / "sample_index/cold.parquet"
    cold_sample.write_bytes(b"cold")
    view_path = store / "views/active_view.json"
    _write_json(
        view_path,
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "artifact_view": {"view_id": "active_view", "view_type": "research_store_view"},
            "feature_channels": {},
            "label_arrays": {},
            "masks": {},
            "sample_index_path": str(active_sample.resolve()),
            "sample_count": 1,
            "sample_count_by_split": {"test": 1},
        },
    )
    policy_path = workspace / "daily_research/brain/research_store_retention_policy.json"
    _write_json(
        policy_path,
        {
            "artifact_type": "research_store_retention_policy",
            "active_view_ids": ["active_view"],
            "cold_component_paths": ["sample_index/cold.parquet"],
        },
    )
    store_view.build_research_store_index(store_root=store, policy_path=policy_path)

    result = gc.prune_cold_research_store(
        store_root=store,
        policy_path=policy_path,
        archive_root=workspace / "daily_research/brain/references",
        delete=True,
        confirm_delete=gc.COLD_STORE_CONFIRMATION,
    )

    assert result["status"] == "deleted_and_verified"
    assert result["deleted_count"] == 1
    assert active_sample.exists()
    assert view_path.exists()
    assert not cold_sample.exists()
    assert Path(result["archive_paths"]["json"]).name.startswith("research_store_cold_assets_archive_")


def test_migrate_legacy_sequence_pack_moves_dir_and_rewrites_paths(workspace: Path) -> None:
    source_root = workspace / "quant_data_platform/data/qdp_v2/research/sequence_pack"
    target_root = workspace / "daily_research/data/research_store/sequence_pack"
    source = source_root / "qdp_v2_seq100_path60_full"
    array_path = source / "panels/daily_raw.float32.dat"
    array_path.parent.mkdir(parents=True)
    array_path.write_bytes(b"1234")
    sample_index = source / "sample_index.parquet"
    sample_index.write_bytes(b"sample")
    _write_json(
        source / "manifest.json",
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "feature_channels": {"daily_raw": {"path": str(array_path), "shape": [1], "features": ["x"]}},
            "label_arrays": {},
            "masks": {},
            "sample_index_path": str(sample_index),
        },
    )
    _write_json(source / "progress.json", {"status": "completed", "manifest_json": str(source / "manifest.json")})

    result = gc.migrate_legacy_sequence_pack(source, source_root=source_root, target_root=target_root)

    assert result["status"] == "migrated"
    assert not source.exists()
    target = target_root / "qdp_v2_seq100_path60_full"
    assert target.exists()
    payload = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "qdp_v2_sequence_path_pack"
    assert payload["feature_channels"]["daily_raw"]["path"] == str(target / "panels/daily_raw.float32.dat")
    assert payload["sample_index_path"] == str(target / "sample_index.parquet")
    assert payload["artifact_migration"]["storage_policy"] == "physical"
    progress = json.loads((target / "progress.json").read_text(encoding="utf-8"))
    assert progress["manifest_json"] == str(target / "manifest.json")


def test_migrate_legacy_sequence_packs_moves_only_kept_packs_and_rewrites_cross_pack_paths(workspace: Path) -> None:
    root = workspace / "quant_data_platform/data/qdp_v2/research/sequence_pack"
    target_root = workspace / "daily_research/data/research_store/sequence_pack"
    full = root / "qdp_v2_seq100_path60_full"
    _write_json(
        full / "manifest.json",
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "feature_channels": {},
            "label_arrays": {},
            "masks": {},
            "sample_index_path": str(full / "sample_index.parquet"),
        },
    )
    view = root / "qdp_v2_seq100_path60_todayclose_full"
    _write_json(
        view / "manifest.json",
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "feature_channels": {
                "daily_raw": {"path": str(full / "panels/daily_raw.float32.dat"), "shape": [1], "features": ["x"]},
            },
            "label_arrays": {},
            "masks": {},
            "sample_index_path": str(full / "sample_index.parquet"),
            "derived_from_pack_manifest": str(full / "manifest.json"),
        },
    )
    smoke = root / "smoke_seq100_path60"
    _write_json(
        smoke / "manifest.json",
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "feature_channels": {},
            "label_arrays": {},
            "masks": {},
            "sample_index_path": str(smoke / "sample_index.parquet"),
        },
    )

    report = gc.migrate_legacy_sequence_packs(source_root=root, target_root=target_root, reference_roots=[])

    assert report["migrated_count"] == 2
    assert report["skipped_count"] == 1
    assert report["skipped"][0]["classification"] == "smoke_sequence_pack"
    todayclose = json.loads((target_root / "qdp_v2_seq100_path60_todayclose_full/manifest.json").read_text(encoding="utf-8"))
    assert todayclose["derived_from_pack_manifest"] == str(target_root / "qdp_v2_seq100_path60_full/manifest.json")
    assert todayclose["feature_channels"]["daily_raw"]["path"] == str(target_root / "qdp_v2_seq100_path60_full/panels/daily_raw.float32.dat")
