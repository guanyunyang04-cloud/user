from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import research_store_gc as gc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(gc, "WORKSPACE_ROOT", tmp_path)
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
