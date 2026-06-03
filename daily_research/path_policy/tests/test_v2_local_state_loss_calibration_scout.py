from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import v2_local_state_loss_calibration_scout as scout


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_training_tasks_reuse_local_state_manifest_and_vary_loss_profiles() -> None:
    tasks = scout.build_training_tasks(output_root=Path("anchor"), seeds=(7,), loss_profiles=("score_monthly_robust_v1",))

    assert len(tasks) == 1
    task = tasks[0]
    assert task["study_family"] == scout.STUDY_FAMILY
    assert task["feature_profile"] == scout.FEATURE_PROFILE
    assert task["loss_profile"] == "score_monthly_robust_v1"
    assert task["output_profile"] == scout.OUTPUT_PROFILE
    assert task["promotion_allowed"] is False
    assert task["active_execution_strategy_expected_diff"] == "none"
    assert "--forecast-memmap-manifest" in task["command"]
    assert task["command"][task["command"].index("--forecast-memmap-manifest") + 1].endswith("forecast_dataset_manifest.json")
    assert task["command"][task["command"].index("--forecast-loss-profile") + 1] == "score_monthly_robust_v1"


def test_build_training_tasks_reject_output_profile_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_contract(*_args, **_kwargs):
        return {"required_output_profile": "other", "loss_profile": "bad"}

    monkeypatch.setattr(scout, "forecast_loss_profile_contract", fake_contract)

    with pytest.raises(ValueError, match="loss_profile_output_mismatch"):
        scout.build_training_tasks(output_root=Path("anchor"), loss_profiles=("bad",))


def test_write_task_list_records_source_manifest_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scout, "validate_source_manifest", lambda *_args, **_kwargs: {"status": "ok", "blockers": []})

    path = scout.write_task_list(tmp_path / "out", seeds=(7,), loss_profiles=("horizon_entropy_regularized_v1",))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["research_program"] == scout.RESEARCH_PROGRAM
    assert payload["study_family"] == scout.STUDY_FAMILY
    assert payload["training_task_count"] == 1
    assert payload["loss_profiles"] == ["horizon_entropy_regularized_v1"]
    assert payload["source_manifest_validation"]["status"] == "ok"
    assert payload["boundary"]["promotion_allowed"] is False


def test_run_training_tasks_blocks_when_source_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scout, "_active_artifact_has_diff", lambda: False)
    monkeypatch.setattr(scout, "validate_source_manifest", lambda *_args, **_kwargs: {"status": "blocked", "blockers": ["missing_manifest"]})

    payload = scout.run_training_tasks(output_root=tmp_path / "out", seeds=(7,), loss_profiles=("score_monthly_robust_v1",))

    assert payload["status"] == "blocked"
    assert "invalid_or_missing_local_state_seed7_manifest" in payload["blockers"]
    assert (tmp_path / "out" / "v2_local_state_loss_calibration_training_summary.json").exists()


def test_run_training_tasks_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scout, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        scout.run_training_tasks(output_root=tmp_path / "out", seeds=(7,), loss_profiles=("score_monthly_robust_v1",))


def test_validate_source_manifest_delegates_to_local_state_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_validate(path):
        calls.append(Path(path))
        return {"status": "ok"}

    manifest = tmp_path / "forecast_dataset_manifest.json"
    _write_json(manifest, {"feature_columns": []})
    monkeypatch.setattr(scout.local_state, "validate_local_state_manifest", fake_validate)

    payload = scout.validate_source_manifest(manifest)

    assert payload["status"] == "ok"
    assert calls == [manifest]
