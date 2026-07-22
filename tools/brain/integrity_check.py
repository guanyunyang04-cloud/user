from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from daily_research.model_registry import verify_registry


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _check_brain(root: Path, errors: list[str]) -> dict[str, Any]:
    manifest_path = root / "brain/brain_manifest.json"
    registry_path = root / "brain/object_registry.json"
    _record_error(errors, manifest_path.is_file(), "missing:brain_manifest")
    _record_error(errors, registry_path.is_file(), "missing:object_registry")
    if not manifest_path.is_file() or not registry_path.is_file():
        return {}
    manifest = _read_json(manifest_path)
    registry = _read_json(registry_path)
    _record_error(errors, int(manifest.get("schema_version", 0)) == 2, "brain_manifest_schema")
    _record_error(errors, int(registry.get("schema_version", 0)) == 2, "object_registry_schema")
    for key in ("entrypoint", "state", "object_registry"):
        path = root / str(manifest.get(key, ""))
        _record_error(errors, path.is_file(), f"missing:brain:{key}:{path}")
    for child in list(manifest.get("children", []) or []):
        entry = root / str(child.get("entrypoint", ""))
        _record_error(errors, entry.is_file(), f"missing:child_entry:{entry}")
    canonical = root / str(manifest.get("canonical_skill", ""))
    installed = Path(str(manifest.get("installed_skill", "")))
    _record_error(errors, canonical.is_file(), f"missing:canonical_skill:{canonical}")
    _record_error(errors, installed.is_file(), f"missing:installed_skill:{installed}")
    if canonical.is_file() and installed.is_file():
        _record_error(errors, _sha256(canonical) == _sha256(installed), "workspace_brain_skill_drift")
    for path in (root / "brain/workflows", root / "brain/skills/workspace-brain/agents", root / "brain/skills/workspace-brain/scripts"):
        _record_error(errors, not path.exists(), f"retired_brain_runtime_present:{path}")
    if installed.parent.is_dir():
        for name in ("agents", "scripts"):
            _record_error(errors, not (installed.parent / name).exists(), f"installed_skill_runtime_present:{name}")
    return {"manifest": manifest, "object_count": len(list(registry.get("objects", []) or []))}


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
