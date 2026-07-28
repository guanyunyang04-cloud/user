from __future__ import annotations

import json
from pathlib import Path

import torch

import daily_research.model_registry as model_registry
import daily_research.path_policy.seq100_development as development


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _registry(root: Path) -> Path:
    checkpoint = root / "daily_research/models/model/2025/checkpoint.pt"
    summary = root / "daily_research/models/model/2025/training_summary.json"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"weight": torch.ones(1)}}, checkpoint)
    _write_json(summary, {"best_epoch": 1, "rank_ic": 0.05})
    path = root / "daily_research/models/registry.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "selected_research_baseline": "model",
            "bundles": [
                {
                    "model_id": "model",
                    "year": 2025,
                    "checkpoint": checkpoint.relative_to(root).as_posix(),
                    "training_summary": summary.relative_to(root).as_posix(),
                    "metrics": {"rank_ic": 0.05},
                }
            ],
        },
    )
    return path


def test_model_registry_resolves_and_loads_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(model_registry, "WORKSPACE_ROOT", tmp_path)

    assert model_registry.verify_registry(registry)["status"] == "ok"
    resolved = model_registry.resolve_bundle("model", 2025, path=registry)
    assert Path(resolved["training_summary_path"]).is_file()
    loaded = model_registry.load_checkpoint("model", 2025, path=registry)
    torch.testing.assert_close(loaded["model_state_dict"]["weight"], torch.ones(1))


def test_model_registry_reports_missing_bundle_file(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(model_registry, "WORKSPACE_ROOT", tmp_path)
    resolved = model_registry.resolve_bundle("model", 2025, path=registry)
    Path(resolved["checkpoint_path"]).unlink()

    result = model_registry.verify_registry(registry)

    assert result["status"] == "blocked"
    assert result["errors"][0].startswith("missing:model/2025:checkpoint:")


def test_compact_status_and_verify_use_data_models_and_study_configs(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry(tmp_path)
    qdp = tmp_path / "qdp.json"
    pack = tmp_path / "pack.json"
    studies = tmp_path / "studies"
    _write_json(qdp, {"active_as_of_date": "2026-07-21"})
    _write_json(pack, {"artifact_type": "pack"})
    _write_json(studies / "study.json", {"study_id": "study", "status": "paused"})
    monkeypatch.setattr(model_registry, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(development, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(development, "QDP_ACTIVE", qdp)
    monkeypatch.setattr(development, "BASE_PACK", pack)
    monkeypatch.setattr(development, "STUDY_ROOT", studies)

    assert development.status()["registered_model_bundles"] == 1
    assert development.status()["study_configs"]["study"]["status"] == "paused"
    result = development.verify()
    assert result["status"] == "ok"
    assert result["study_config_count"] == 1
