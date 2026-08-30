from __future__ import annotations

import json
from pathlib import Path

from tools.research_path_migration import (
    SCHEMA,
    plan_file,
    run,
    study_files,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_migration_resolves_current_paths_and_marks_deleted_inputs(tmp_path: Path) -> None:
    study = tmp_path / "research" / "studies" / "demo.json"
    current_record = tmp_path / "research" / "records" / "demo" / "result.json"
    current_record.parent.mkdir(parents=True)
    current_record.write_text("{}\n", encoding="utf-8")
    _write(
        study,
        {
            "source": {
                "study": "daily_research/research_records/demo/result.json",
                "deleted": "daily_research/output/path_policy/studies/demo/manifest.json",
            },
            "output_dir": "daily_research/output/path_policy/studies/demo",
        },
    )

    plan = plan_file(study, tmp_path)
    assert plan.changed
    assert {item.status for item in plan.changes} == {"workspace_resolved", "provenance_only", "workspace_output"}
    rewritten = json.loads(plan.rewritten_text)
    assert rewritten["source"]["study"] == "research/records/demo/result.json"
    assert rewritten["source"]["deleted"].startswith("legacy://")
    assert rewritten["output_dir"] == "runs/studies/demo"


def test_record_output_never_targets_durable_records(tmp_path: Path) -> None:
    study = tmp_path / "research" / "studies" / "demo.json"
    _write(study, {"outputs": {"record_dir": "daily_research/research_records/demo"}})
    plan = plan_file(study, tmp_path)
    assert json.loads(plan.rewritten_text)["outputs"]["record_dir"] == "runs/studies/records/demo"


def test_apply_is_idempotent_and_preserves_migration_manifest(tmp_path: Path) -> None:
    study = tmp_path / "research" / "studies" / "demo.json"
    manifest = tmp_path / "research" / "path_migration_manifest.json"
    _write(study, {"source": "daily_research/studies/other.json"})
    (tmp_path / "research" / "studies" / "other.json").write_text("{}\n", encoding="utf-8")

    first = run(tmp_path, apply=True, manifest_path=manifest)
    assert first["schema"] == SCHEMA
    assert first["summary"]["references_migrated"] == 1
    saved = manifest.read_text(encoding="utf-8")
    second = run(tmp_path, apply=True, manifest_path=manifest)
    assert second == json.loads(saved)
    assert manifest.read_text(encoding="utf-8") == saved
    assert "daily_research/" not in study.read_text(encoding="utf-8")


def test_repository_study_files_have_no_unqualified_legacy_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    assert all(not plan_file(path, root).changes for path in study_files(root))
