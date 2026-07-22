from __future__ import annotations

import hashlib
import json
from pathlib import Path

import daily_research.model_registry as model_registry
from tools.brain.integrity_check import run_integrity_check


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compact_integrity_check_and_retired_surface_detection(tmp_path: Path, monkeypatch) -> None:
    skill = tmp_path / "brain/skills/workspace-brain/SKILL.md"
    installed = tmp_path / "installed/SKILL.md"
    skill.parent.mkdir(parents=True)
    installed.parent.mkdir(parents=True)
    skill.write_text("skill", encoding="utf-8")
    installed.write_text("skill", encoding="utf-8")
    _json(tmp_path / "brain/README.md", {})
    _json(tmp_path / "brain/state.md", {})
    _json(
        tmp_path / "brain/brain_manifest.json",
        {
            "schema_version": 2,
            "entrypoint": "brain/README.md",
            "state": "brain/state.md",
            "object_registry": "brain/object_registry.json",
            "canonical_skill": "brain/skills/workspace-brain/SKILL.md",
            "installed_skill": str(installed),
            "children": [
                {"entrypoint": "daily_research/brain/README.md"},
                {"entrypoint": "quant_data_platform/brain/README.md"},
            ],
        },
    )
    _json(tmp_path / "brain/object_registry.json", {"schema_version": 2, "objects": []})
    _json(tmp_path / "daily_research/brain/README.md", {})
    _json(tmp_path / "quant_data_platform/brain/README.md", {})
    _json(
        tmp_path / "quant_data_platform/data/qdp_v2/active/active.json",
        {"active_as_of_date": "2026-07-21", "datasets": {"market": "current"}},
    )
    _json(tmp_path / "quant_data_platform/data/qdp_v2/datasets/market/current/dataset.json", {})
    _json(tmp_path / "daily_research/data/research_store/pack/manifest.json", {})
    _json(
        tmp_path / "daily_research/data/research_store/traditional_quant_baostock_archive_v1/manifest.json",
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
    assert run_integrity_check(tmp_path)["status"] == "ok"
    (tmp_path / "t0_project").mkdir()
    result = run_integrity_check(tmp_path)
    assert result["status"] == "blocked"
    assert "retired_top_level_present:t0_project" in result["errors"]
