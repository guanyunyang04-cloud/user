from __future__ import annotations

import hashlib
import json
from pathlib import Path

import daily_research.model_registry as model_registry
from tools.brain.integrity_check import BRAIN_SCHEMA, OBJECT_SCHEMA, run_integrity_check


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _brain_manifest(
    root: Path,
    brain_id: str,
    brain_type: str,
    readme: str,
    state: str,
    references: str,
) -> dict[str, object]:
    return {
        "schema": BRAIN_SCHEMA,
        "brain_id": brain_id,
        "brain_type": brain_type,
        "project_name": brain_id,
        "parent": "brain/brain_manifest.json" if brain_type == "child" else None,
        "roles": {
            "identity": readme,
            "working_memory": state,
            "semantic_memory": readme,
            "procedural_memory": readme if brain_type == "child" else "AGENTS.md",
            "executive_control": "brain/object_registry.json",
            "episodic_memory": references,
        },
        "children": [],
    }


def _workspace(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "<!-- workspace-brain:start -->\n<!-- workspace-brain:end -->\n",
        encoding="utf-8",
    )
    _json(tmp_path / "brain/README.md", {})
    _json(tmp_path / "brain/state.md", {})
    (tmp_path / "brain/references").mkdir()
    _json(tmp_path / "brain/object_registry.json", {"schema": OBJECT_SCHEMA, "objects": []})

    children = []
    for brain_id, prefix in (
        ("daily_research", "daily_research"),
        ("quant_data_platform", "quant_data_platform"),
    ):
        readme = f"{prefix}/brain/README.md"
        state = f"{prefix}/brain/state.md"
        references = f"{prefix}/brain/references"
        _json(tmp_path / readme, {})
        _json(tmp_path / state, {})
        (tmp_path / references).mkdir()
        manifest_path = f"{prefix}/brain/brain_manifest.json"
        _json(
            tmp_path / manifest_path,
            _brain_manifest(tmp_path, brain_id, "child", readme, state, references),
        )
        children.append({"brain_id": brain_id, "manifest": manifest_path})
    root_manifest = _brain_manifest(
        tmp_path,
        "workspace",
        "project",
        "brain/README.md",
        "brain/state.md",
        "brain/references",
    )
    root_manifest["children"] = children
    root_manifest.pop("parent")
    _json(tmp_path / "brain/brain_manifest.json", root_manifest)

    _json(
        tmp_path / "quant_data_platform/data/qdp_v2/active/active.json",
        {"active_as_of_date": "2026-07-21", "datasets": {"market": "current"}},
    )
    _json(tmp_path / "quant_data_platform/data/qdp_v2/datasets/market/current/dataset.json", {})
    _json(tmp_path / "daily_research/data/research_store/pack/manifest.json", {})
    _json(
        tmp_path
        / "daily_research/data/research_store/traditional_quant_baostock_archive_v1/manifest.json",
        {},
    )
    record = tmp_path / "daily_research/research_records/result.json"
    study = tmp_path / "daily_research/studies/study.json"
    _json(record, {})
    _json(study, {"status": "paused"})
    _json(
        tmp_path / "daily_research/research_records/seq100/index.json",
        {
            "records": [{"id": "result", "path": record.relative_to(tmp_path).as_posix()}],
            "active_studies": [study.relative_to(tmp_path).as_posix()],
        },
    )
    checkpoint = tmp_path / "daily_research/models/checkpoint.pt"
    summary = tmp_path / "daily_research/models/summary.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    _json(summary, {})
    _json(
        tmp_path / "daily_research/models/registry.json",
        {
            "schema_version": 1,
            "bundles": [
                {
                    "model_id": "model",
                    "year": 2025,
                    "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
                    "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
                    "training_summary": summary.relative_to(tmp_path).as_posix(),
                    "training_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    (tmp_path / "daily_research/path_policy").mkdir(parents=True)
    monkeypatch.setattr(model_registry, "WORKSPACE_ROOT", tmp_path)


def test_compact_integrity_check_and_retired_surface_detection(
    tmp_path: Path, monkeypatch
) -> None:
    _workspace(tmp_path, monkeypatch)
    assert run_integrity_check(tmp_path)["status"] == "ok"
    (tmp_path / "t0_project").mkdir()
    result = run_integrity_check(tmp_path)
    assert result["status"] == "blocked"
    assert "retired_top_level_present:t0_project" in result["errors"]


def test_brain_integrity_is_independent_of_global_skill(
    tmp_path: Path, monkeypatch
) -> None:
    _workspace(tmp_path, monkeypatch)
    result = run_integrity_check(tmp_path)
    assert result["status"] == "ok"
    assert not (tmp_path / "brain/skills").exists()


def test_brain_manifest_rejects_legacy_schema_and_unsafe_path(
    tmp_path: Path, monkeypatch
) -> None:
    _workspace(tmp_path, monkeypatch)
    manifest_path = tmp_path / "brain/brain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "workspace-brain/v0"
    manifest["roles"]["semantic_memory"] = "../outside.md"
    _json(manifest_path, manifest)
    result = run_integrity_check(tmp_path)
    assert result["status"] == "blocked"
    assert any("brain_manifest_schema" in error for error in result["errors"])
    assert any("brain_manifest_unsafe_path" in error for error in result["errors"])


def test_repository_local_workspace_brain_skill_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    _workspace(tmp_path, monkeypatch)
    local_skill = tmp_path / "brain/skills/workspace-brain"
    local_skill.mkdir(parents=True)
    (local_skill / "SKILL.md").write_text("duplicate", encoding="utf-8")
    result = run_integrity_check(tmp_path)
    assert result["status"] == "blocked"
    assert any(
        "repository_local_workspace_brain_skill" in error
        for error in result["errors"]
    )
