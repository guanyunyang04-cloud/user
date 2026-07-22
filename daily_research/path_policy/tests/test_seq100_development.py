from __future__ import annotations

import hashlib
import json
from pathlib import Path

import daily_research.model_registry as model_registry
import daily_research.path_policy.seq100_development as development


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry(root: Path) -> Path:
    checkpoint = root / "daily_research/models/model/2025/checkpoint.pt"
    summary = root / "daily_research/models/model/2025/training_summary.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(summary, {"best_epoch": 1})
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
                    "checkpoint_sha256": _sha256(checkpoint),
                    "training_summary": summary.relative_to(root).as_posix(),
                    "training_summary_sha256": _sha256(summary),
                }
            ],
        },
    )
    return path


def test_model_registry_detects_bundle_drift(tmp_path: Path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(model_registry, "WORKSPACE_ROOT", tmp_path)
    assert model_registry.verify_registry(registry)["status"] == "ok"
    resolved = model_registry.resolve_bundle("model", 2025, path=registry)
    assert Path(resolved["checkpoint_path"]).read_bytes() == b"checkpoint"
    Path(resolved["checkpoint_path"]).write_bytes(b"changed")
    result = model_registry.verify_registry(registry)
    assert result["status"] == "blocked"
    assert result["errors"] == ["sha256_mismatch:model/2025:checkpoint"]


def test_compact_status_and_verify_use_only_stable_surfaces(tmp_path: Path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    qdp = tmp_path / "qdp.json"
    pack = tmp_path / "pack.json"
    record = tmp_path / "record.json"
    index = tmp_path / "index.json"
    study = tmp_path / "study.json"
    _write_json(qdp, {"active_as_of_date": "2026-07-21"})
    _write_json(pack, {"artifact_type": "pack"})
    _write_json(record, {"status": "complete"})
    _write_json(index, {"records": [{"id": "record", "path": "record.json"}]})
    _write_json(study, {"study_id": "study", "status": "paused"})
    monkeypatch.setattr(model_registry, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(development, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(development, "DEFAULT_REGISTRY", registry)
    monkeypatch.setattr(development, "QDP_ACTIVE", qdp)
    monkeypatch.setattr(development, "BASE_PACK", pack)
    monkeypatch.setattr(development, "RECORD_INDEX", index)
    monkeypatch.setattr(development, "ACTIVE_STUDIES", {"study": study})
    assert development.status()["registered_model_bundles"] == 1
    result = development.verify()
    assert result["status"] == "ok"
    assert result["research_record_count"] == 1


def test_only_batch1024_remains_paused_after_sixfold_completion() -> None:
    assert set(development.ACTIVE_STUDIES) == {"l35v2-batch1024"}
    batch = json.loads(development.ACTIVE_STUDIES["l35v2-batch1024"].read_text(encoding="utf-8"))
    assert batch["status"] == "paused_ready"
    assert "l35v2_sixfold_2020_2025" in batch["sixfold_evidence"]
    assert "gradient_accumulation_as_fake_1024" in batch["forbidden_shortcuts"]
