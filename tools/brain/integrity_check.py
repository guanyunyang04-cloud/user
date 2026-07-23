from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence

from daily_research.model_registry import verify_registry


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BRAIN_SCHEMA = "workspace-brain/v1"
OBJECT_SCHEMA = "workspace-brain/objects/v1"
REQUIRED_BRAIN_ROLES = {
    "identity",
    "working_memory",
    "semantic_memory",
    "procedural_memory",
    "executive_control",
    "episodic_memory",
}
ALLOWED_TOP_LEVEL_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".tmp",
    "brain",
    "daily_research",
    "quant_data_platform",
    "tools",
}
RETIRED_TOP_LEVEL_DIRECTORIES = {
    "canonical_data",
    "daily_stock_analysis-main",
    "t0_project",
    "traditional_quant_research",
}
RETIRED_DAILY_RESEARCH_DIRECTORIES = {
    "baseline",
    "continuous_policy",
    "deep_alpha",
    "execution",
    "tools",
}
RETIRED_QDP_SURFACES = {
    "event_packs",
    "features",
    "ingest",
    "lake",
    "memmap",
}
RETIRED_SEQ100_MODULES = {
    "seq100_qcurve.py",
    "seq100_qcurve_training.py",
    "seq100_qcurve_qonly_training.py",
    "seq100_structured_global_intraday.py",
    "seq100_structured_input_ablation.py",
    "seq100_structured_local_intraday_capital_speed.py",
    "seq100_structured_training_window.py",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _project_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return None
    if ".." in posix.parts:
        return None
    path = (root / Path(*posix.parts)).resolve()
    if path != root and root not in path.parents:
        return None
    return path


def _check_brain_manifest(
    root: Path,
    manifest_path: Path,
    errors: list[str],
    *,
    expected_type: str,
    seen: set[Path],
) -> dict[str, Any]:
    resolved = manifest_path.resolve()
    if resolved in seen:
        errors.append(f"brain_manifest_cycle_or_duplicate:{manifest_path}")
        return {}
    seen.add(resolved)
    _record_error(errors, manifest_path.is_file(), f"missing:brain_manifest:{manifest_path}")
    if not manifest_path.is_file():
        return {}
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid:brain_manifest:{manifest_path}:{exc}")
        return {}
    label = manifest_path.relative_to(root).as_posix()
    _record_error(errors, manifest.get("schema") == BRAIN_SCHEMA, f"brain_manifest_schema:{label}")
    _record_error(
        errors,
        manifest.get("brain_type") == expected_type,
        f"brain_manifest_type:{label}:{expected_type}",
    )
    _record_error(
        errors,
        isinstance(manifest.get("brain_id"), str) and bool(manifest["brain_id"].strip()),
        f"brain_manifest_id:{label}",
    )
    _record_error(
        errors,
        isinstance(manifest.get("project_name"), str),
        f"brain_manifest_project_name:{label}",
    )
    roles = manifest.get("roles")
    _record_error(errors, isinstance(roles, dict), f"brain_manifest_roles:{label}")
    if not isinstance(roles, dict):
        roles = {}
    for role in sorted(REQUIRED_BRAIN_ROLES):
        _record_error(errors, role in roles, f"brain_manifest_missing_role:{label}:{role}")
    for role, value in roles.items():
        path = _project_path(root, value)
        _record_error(errors, path is not None, f"brain_manifest_unsafe_path:{label}:{role}")
        if path is None:
            continue
        if role == "episodic_memory":
            _record_error(errors, path.is_dir(), f"missing:brain_role_dir:{label}:{role}:{path}")
        else:
            _record_error(errors, path.is_file(), f"missing:brain_role_file:{label}:{role}:{path}")
    parent = manifest.get("parent")
    if parent is not None:
        parent_path = _project_path(root, parent)
        _record_error(errors, parent_path is not None, f"brain_manifest_unsafe_parent:{label}")
        if parent_path is not None:
            _record_error(errors, parent_path.is_file(), f"missing:brain_parent:{label}:{parent_path}")
    children = manifest.get("children")
    _record_error(errors, isinstance(children, list), f"brain_manifest_children:{label}")
    if not isinstance(children, list):
        return manifest
    child_ids: set[str] = set()
    for index, child in enumerate(children):
        child_label = f"{label}:children[{index}]"
        if not isinstance(child, dict):
            errors.append(f"brain_manifest_child_object:{child_label}")
            continue
        child_id = child.get("brain_id")
        _record_error(
            errors,
            isinstance(child_id, str) and bool(child_id.strip()),
            f"brain_manifest_child_id:{child_label}",
        )
        if isinstance(child_id, str):
            _record_error(
                errors,
                child_id not in child_ids,
                f"brain_manifest_duplicate_child_id:{child_label}:{child_id}",
            )
            child_ids.add(child_id)
        child_path = _project_path(root, child.get("manifest"))
        _record_error(errors, child_path is not None, f"brain_manifest_unsafe_child:{child_label}")
        if child_path is None:
            continue
        child_manifest = _check_brain_manifest(
            root, child_path, errors, expected_type="child", seen=seen
        )
        if child_manifest and child_id:
            _record_error(
                errors,
                child_manifest.get("brain_id") == child_id,
                f"brain_manifest_child_id_mismatch:{child_label}:{child_id}",
            )
    return manifest


def _check_brain(root: Path, errors: list[str]) -> dict[str, Any]:
    manifest_path = root / "brain/brain_manifest.json"
    manifest = _check_brain_manifest(
        root, manifest_path, errors, expected_type="project", seen=set()
    )
    roles = manifest.get("roles", {}) if manifest else {}
    registry_path = _project_path(root, roles.get("executive_control"))
    _record_error(errors, registry_path is not None, "missing:object_registry_role")
    if registry_path is None or not registry_path.is_file():
        return {}
    registry = _read_json(registry_path)
    _record_error(errors, registry.get("schema") == OBJECT_SCHEMA, "object_registry_schema")
    objects = registry.get("objects")
    _record_error(errors, isinstance(objects, list), "object_registry_objects")
    if not isinstance(objects, list):
        objects = []
    object_ids: set[str] = set()
    for index, item in enumerate(objects):
        label = f"object_registry[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label}:object")
            continue
        for field in ("object_id", "owner", "type", "truth", "mutation"):
            _record_error(
                errors,
                isinstance(item.get(field), str) and bool(item[field].strip()),
                f"{label}:missing:{field}",
            )
        object_id = item.get("object_id")
        if isinstance(object_id, str):
            _record_error(errors, object_id not in object_ids, f"{label}:duplicate:{object_id}")
            object_ids.add(object_id)
        paths = item.get("paths")
        _record_error(errors, isinstance(paths, list), f"{label}:paths")
        if not isinstance(paths, list):
            continue
        for value in paths:
            path = _project_path(root, value)
            _record_error(errors, path is not None, f"{label}:unsafe_path:{value}")
            if path is not None:
                _record_error(errors, path.exists(), f"{label}:missing_path:{path}")
    local_skill = root / "brain/skills/workspace-brain"
    _record_error(errors, not local_skill.exists(), f"repository_local_workspace_brain_skill:{local_skill}")
    return {"manifest": manifest, "object_count": len(objects)}


def _check_qdp(root: Path, errors: list[str]) -> dict[str, Any]:
    qdp_root = root / "quant_data_platform/data/qdp_v2"
    active_path = qdp_root / "active/active.json"
    _record_error(errors, qdp_root.is_dir(), f"missing:qdp_root:{qdp_root}")
    _record_error(errors, active_path.is_file(), f"missing:qdp_active:{active_path}")
    if not active_path.is_file():
        return {}
    active = _read_json(active_path)
    datasets = dict(active.get("datasets", {}) or {})
    for domain, dataset_id in datasets.items():
        manifest = qdp_root / "datasets" / str(domain) / str(dataset_id) / "dataset.json"
        _record_error(errors, manifest.is_file(), f"missing:qdp_dataset_manifest:{domain}:{dataset_id}")
    return {
        "active_as_of_date": active.get("active_as_of_date"),
        "active_dataset_count": len(datasets),
    }


def _check_research(root: Path, errors: list[str]) -> dict[str, Any]:
    store = root / "daily_research/data/research_store"
    _record_error(errors, store.is_dir(), f"missing:research_store:{store}")
    manifests = list(store.rglob("manifest.json")) if store.is_dir() else []
    _record_error(errors, bool(manifests), "research_store_has_no_manifests")
    records_path = root / "daily_research/research_records/seq100/index.json"
    _record_error(errors, records_path.is_file(), f"missing:research_record_index:{records_path}")
    record_count = 0
    if records_path.is_file():
        records = _read_json(records_path)
        rows = list(records.get("records", []) or [])
        record_count = len(rows)
        for row in rows:
            artifact = root / str(row.get("path", ""))
            _record_error(errors, artifact.is_file(), f"missing:research_record:{row.get('id')}")
        for value in list(records.get("active_studies", []) or []):
            study = root / str(value)
            _record_error(errors, study.is_file(), f"missing:active_study:{study}")
    source_archive = store / "traditional_quant_baostock_archive_v1/manifest.json"
    _record_error(errors, source_archive.is_file(), f"missing:protected_source_archive:{source_archive}")
    model_check = verify_registry(root / "daily_research/models/registry.json")
    errors.extend(f"model_registry:{error}" for error in list(model_check["errors"]))
    return {
        "research_store_manifest_count": len(manifests),
        "research_record_count": record_count,
        "active_study_count": len(list(records.get("active_studies", []) or [])) if records_path.is_file() else 0,
        "model_bundle_count": model_check["bundle_count"],
    }


def _check_complexity(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    top = {path.name for path in root.iterdir() if path.is_dir()}
    for name in sorted(RETIRED_TOP_LEVEL_DIRECTORIES.intersection(top)):
        errors.append(f"retired_top_level_present:{name}")
    unexpected = sorted(top.difference(ALLOWED_TOP_LEVEL_DIRECTORIES))
    warnings.extend(f"unexpected_top_level:{name}" for name in unexpected)
    daily = root / "daily_research"
    for name in sorted(RETIRED_DAILY_RESEARCH_DIRECTORIES):
        if (daily / name).exists():
            errors.append(f"retired_daily_research_surface_present:{name}")
    policy = daily / "path_policy"
    for name in sorted(RETIRED_SEQ100_MODULES):
        if (policy / name).is_file():
            errors.append(f"retired_seq100_module_present:{name}")
    tracked_python = list((root / "daily_research/path_policy").glob("*.py"))
    if len(tracked_python) > 16:
        warnings.append(f"path_policy_module_budget_exceeded:{len(tracked_python)}>16")
    qdp_source = root / "quant_data_platform/src/quant_data_platform"
    for name in sorted(RETIRED_QDP_SURFACES):
        if (qdp_source / name).exists():
            errors.append(f"retired_qdp_surface_present:{name}")
    return {
        "top_level_directory_count": len(top),
        "path_policy_module_count": len(tracked_python),
    }


def run_integrity_check(workspace_root: Path = WORKSPACE_ROOT) -> dict[str, Any]:
    root = workspace_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    sections = {
        "brain": _check_brain(root, errors),
        "qdp": _check_qdp(root, errors),
        "research": _check_research(root, errors),
        "complexity": _check_complexity(root, errors, warnings),
    }
    return {
        "status": "ok" if not errors else "blocked",
        "workspace_root": str(root),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "sections": sections,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the compact workspace architecture.")
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_integrity_check(args.workspace_root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"status: {result['status']}")
        for error in result["errors"]:
            print(f"ERROR {error}")
        for warning in result["warnings"]:
            print(f"WARNING {warning}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
