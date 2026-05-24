from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.brain.platform import build_brain_catalog, check_text_encoding, read_text as platform_read_text

MAIN_MANIFEST = Path("brain/brain_manifest.json")
BRAIN_CATALOG = Path("brain/brain_catalog.json")
SHARED_CONTRACT_KEY = "shared_regional_brain_contract"

REQUIRED_MAIN_KEYS = (
    "brain_type",
    "brain_version",
    "entrypoint",
    "read_order",
    "governance",
    "cognition_contract",
    SHARED_CONTRACT_KEY,
    "brain_contract",
    "child_brains",
    "write_routes",
    "handoff_contract",
)

OPTIONAL_MAIN_PATH_KEYS = (
    "identity_path",
    "state_path",
    "knowledge_path",
    "operations_path",
    "governance_path",
)

DEFAULT_REQUIRED_CHILD_KEYS = (
    "brain_type",
    "brain_id",
    "parent_brain",
    "attach_status",
    "entrypoint",
    "body_root",
    "shared_contract_source",
    "regional_specialization",
    "write_routes",
    "body_map",
    "modules",
    "handoff_contract",
)

OPTIONAL_CHILD_PATH_KEYS = (
    "identity_path",
    "state_path",
    "knowledge_path",
    "operations_path",
    "governance_path",
    "operating_protocol_path",
)

MOJIBAKE_MARKERS = (
    "\ufffd",
    "Ã",
    "Â",
    "â€",
    "涓昏",
    "鍒嗚",
    "韬",
    "鐘舵",
    "绠℃",
    "鈥",
    "銆",
    "乻",
    "乲",
    "乷",
    "乬",
    "",
)

COMPACT_CORE_LINE_WARNINGS = {
    "brain/identity_layer.md": 160,
    "brain/state_center.md": 160,
    "brain/knowledge_center.md": 160,
    "brain/master_brain.md": 160,
    "brain/brain_architecture.md": 160,
    "brain/operations_center.md": 160,
    "brain/governance_layer.md": 160,
    "daily_research/brain/identity_layer.md": 220,
    "daily_research/brain/brain_architecture.md": 180,
    "daily_research/brain/governance_layer.md": 220,
    "daily_research/brain/brain_operating_protocol.md": 180,
    "t0_project/brain/identity_layer.md": 180,
    "t0_project/brain/brain_architecture.md": 160,
    "t0_project/brain/governance_layer.md": 180,
    "daily_stock_analysis-main/brain/identity_layer.md": 180,
    "daily_stock_analysis-main/brain/brain_architecture.md": 160,
    "daily_stock_analysis-main/brain/governance_layer.md": 180,
}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    detail: str
    path: str | None = None


def _rel(path: Path) -> str:
    return path.as_posix()


def _workspace_path(relative_path: str | Path) -> Path:
    path = Path(str(relative_path))
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _read_text(relative_path: str | Path) -> str:
    return platform_read_text(relative_path)


def _read_json(relative_path: str | Path) -> dict[str, Any]:
    payload = json.loads(_read_text(relative_path))
    return payload if isinstance(payload, dict) else {}


def _is_relative_workspace_path(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    path = Path(text)
    return not path.is_absolute() and ".." not in path.parts


def _add_missing_key_findings(
    findings: list[Finding],
    data: dict[str, Any],
    required_keys: Iterable[str],
    path: str,
    code: str,
) -> None:
    for key in required_keys:
        if key not in data:
            findings.append(Finding("error", code, f"missing key: {key}", path))


def _add_path_finding(
    findings: list[Finding],
    value: Any,
    path: str,
    code: str,
    label: str,
    *,
    must_exist: bool = True,
) -> None:
    text = str(value or "").strip()
    if not text:
        findings.append(Finding("error", code, f"{label} is empty", path))
        return
    if not _is_relative_workspace_path(text):
        findings.append(Finding("error", code, f"{label} must be a relative workspace path: {text}", path))
        return
    if must_exist and not _workspace_path(text).exists():
        findings.append(Finding("error", code, f"{label} does not exist: {text}", path))


def _split_contract_source(source: str) -> tuple[str, str]:
    path_text, _, key = str(source).partition("#")
    return path_text, key


def _module_path_map(child_manifest: dict[str, Any]) -> dict[str, list[str]]:
    modules = child_manifest.get("modules", [])
    out: dict[str, list[str]] = {}
    if not isinstance(modules, list):
        return out
    for module in modules:
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("id", "")).strip()
        paths = module.get("paths", [])
        if not module_id or not isinstance(paths, list):
            continue
        out[module_id] = [str(path).strip() for path in paths if str(path).strip()]
    return out


def _derive_child_read_order(main_manifest: dict[str, Any], child_manifest: dict[str, Any]) -> list[str]:
    contract = main_manifest.get(SHARED_CONTRACT_KEY, {})
    module_map = _module_path_map(child_manifest)
    ordered: list[str] = []
    seen: set[str] = set()
    for module_id in contract.get("default_module_order", []):
        raw = str(module_id).strip()
        optional = raw.endswith("?")
        module_key = raw[:-1] if optional else raw
        for path in module_map.get(module_key, []):
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    modules = child_manifest.get("modules", [])
    if isinstance(modules, list):
        for module in modules:
            if not isinstance(module, dict):
                continue
            for raw_path in module.get("paths", []):
                path = str(raw_path).strip()
                if path and path not in seen:
                    seen.add(path)
                    ordered.append(path)
    return ordered


def _validate_path_list(
    findings: list[Finding],
    items: Any,
    owner_path: str,
    code: str,
    label: str,
    *,
    require_non_empty: bool = True,
) -> list[str]:
    if not isinstance(items, list) or (require_non_empty and not items):
        findings.append(Finding("error", code, f"{label} must be a non-empty list", owner_path))
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if text in seen:
            findings.append(Finding("error", code, f"{label} contains duplicate path: {text}", owner_path))
        seen.add(text)
        _add_path_finding(findings, text, owner_path, code, label)
        out.append(text)
    return out


def _validate_markdown_and_encoding(findings: list[Finding], relative_path: str) -> None:
    path = _workspace_path(relative_path)
    if not path.exists() or not path.is_file():
        return
    text = _read_text(relative_path)
    encoding = check_text_encoding(relative_path)
    markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
    if markers or not encoding.is_utf8 or encoding.has_replacement_char:
        findings.append(
            Finding(
                "error",
                "brain_text_encoding_suspicious",
                "suspicious mojibake markers: " + ", ".join(markers[:8]),
                relative_path,
            )
        )
    if path.suffix.lower() != ".md":
        return
    lines = text.splitlines()
    headings = [line for line in lines if line.startswith("# ")]
    if len(headings) != 1:
        findings.append(
            Finding(
                "warning",
                "brain_markdown_h1_count",
                f"expected exactly one H1 heading, found {len(headings)}",
                relative_path,
            )
        )
    warn_limit = COMPACT_CORE_LINE_WARNINGS.get(relative_path)
    if warn_limit is not None and len(lines) > warn_limit:
        findings.append(
            Finding(
                "warning",
                "brain_core_doc_too_long",
                f"core doc has {len(lines)} lines, compactness warning threshold is {warn_limit}",
                relative_path,
            )
        )


def _validate_main_manifest(findings: list[Finding], main_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    main_path = _rel(MAIN_MANIFEST)
    _add_missing_key_findings(findings, main_manifest, REQUIRED_MAIN_KEYS, main_path, "main_manifest_missing_key")
    if main_manifest.get("brain_type") != "main":
        findings.append(Finding("error", "main_manifest_type_mismatch", "brain_type must be main", main_path))

    entrypoint = str(main_manifest.get("entrypoint", "")).strip()
    _add_path_finding(findings, entrypoint, main_path, "main_manifest_path_invalid", "entrypoint")
    read_order = _validate_path_list(
        findings,
        main_manifest.get("read_order"),
        main_path,
        "main_manifest_read_order_invalid",
        "read_order",
    )
    if read_order and entrypoint and read_order[0] != entrypoint:
        findings.append(
            Finding(
                "error",
                "main_manifest_read_order_entrypoint_mismatch",
                f"read_order first item should equal entrypoint: {entrypoint}",
                main_path,
            )
        )

    for key in OPTIONAL_MAIN_PATH_KEYS:
        value = str(main_manifest.get(key, "")).strip()
        if value:
            _add_path_finding(findings, value, main_path, "main_manifest_optional_path_invalid", key)
            if read_order and value not in read_order:
                findings.append(
                    Finding(
                        "warning",
                        "main_optional_path_not_in_read_order",
                        f"{key} is not present in read_order: {value}",
                        main_path,
                    )
                )

    write_routes = main_manifest.get("write_routes", {})
    if not isinstance(write_routes, dict) or not write_routes:
        findings.append(Finding("error", "main_write_routes_invalid", "write_routes must be a non-empty object", main_path))
    else:
        for key, value in write_routes.items():
            _add_path_finding(findings, value, main_path, "main_write_route_invalid", f"write_routes.{key}")

    contract = main_manifest.get(SHARED_CONTRACT_KEY, {})
    if not isinstance(contract, dict) or not contract:
        findings.append(
            Finding("error", "main_shared_contract_invalid", f"{SHARED_CONTRACT_KEY} must be a non-empty object", main_path)
        )
    else:
        default_order = contract.get("default_module_order", [])
        required_modules = contract.get("required_modules", [])
        fast_modules = contract.get("fast_handoff_modules", [])
        region_bindings = contract.get("region_bindings", {})
        if not isinstance(default_order, list) or not default_order:
            findings.append(Finding("error", "main_shared_contract_invalid", "default_module_order missing", main_path))
        if not isinstance(required_modules, list) or not required_modules:
            findings.append(Finding("error", "main_shared_contract_invalid", "required_modules missing", main_path))
        else:
            normalized_order = {str(item).rstrip("?") for item in default_order}
            for module_id in required_modules:
                if str(module_id) not in normalized_order:
                    findings.append(
                        Finding(
                            "error",
                            "main_shared_contract_required_module_not_ordered",
                            f"required module is absent from default_module_order: {module_id}",
                            main_path,
                        )
                    )
        if not isinstance(fast_modules, list) or not fast_modules:
            findings.append(Finding("error", "main_shared_contract_invalid", "fast_handoff_modules missing", main_path))
        elif isinstance(required_modules, list):
            required_set = {str(item) for item in required_modules}
            for module_id in fast_modules:
                if str(module_id) not in required_set:
                    findings.append(
                        Finding(
                            "warning",
                            "main_fast_module_not_required",
                            f"fast handoff module is not in required_modules: {module_id}",
                            main_path,
                        )
                    )
        if not isinstance(region_bindings, dict) or not region_bindings:
            findings.append(Finding("error", "main_shared_contract_invalid", "region_bindings missing", main_path))
        else:
            known_modules = {str(item).rstrip("?") for item in default_order}
            for region, modules in region_bindings.items():
                if not isinstance(modules, list) or not modules:
                    findings.append(
                        Finding("error", "main_region_binding_invalid", f"region has no modules: {region}", main_path)
                    )
                    continue
                for module_id in modules:
                    if str(module_id) not in known_modules:
                        findings.append(
                            Finding(
                                "error",
                                "main_region_binding_unknown_module",
                                f"{region} references unknown module: {module_id}",
                                main_path,
                            )
                        )

    child_brains = main_manifest.get("child_brains", [])
    if not isinstance(child_brains, list) or not child_brains:
        findings.append(Finding("error", "main_child_brains_invalid", "child_brains must be a non-empty list", main_path))
        return []

    seen_child_ids: set[str] = set()
    valid_child_refs: list[dict[str, Any]] = []
    for child_ref in child_brains:
        if not isinstance(child_ref, dict):
            findings.append(Finding("error", "main_child_ref_invalid", "child_brains entries must be objects", main_path))
            continue
        child_id = str(child_ref.get("id", "")).strip()
        if not child_id:
            findings.append(Finding("error", "main_child_ref_invalid", "child id is empty", main_path))
        elif child_id in seen_child_ids:
            findings.append(Finding("error", "main_child_ref_duplicate", f"duplicate child id: {child_id}", main_path))
        seen_child_ids.add(child_id)
        for key in ("path", "body_root", "entrypoint"):
            _add_path_finding(findings, child_ref.get(key), main_path, "main_child_ref_path_invalid", f"child.{child_id}.{key}")
        if not str(child_ref.get("attach_status", "")).startswith("attached"):
            findings.append(
                Finding("error", "main_child_ref_attach_status_invalid", f"{child_id} is not attached", main_path)
            )
        valid_child_refs.append(child_ref)
    return valid_child_refs


def _validate_child_manifest(
    findings: list[Finding],
    main_manifest: dict[str, Any],
    child_ref: dict[str, Any],
    required_child_keys: tuple[str, ...],
) -> None:
    child_path = str(child_ref.get("path", "")).strip()
    if not child_path or not _workspace_path(child_path).exists():
        return
    child_manifest = _read_json(child_path)
    _add_missing_key_findings(findings, child_manifest, required_child_keys, child_path, "child_manifest_missing_key")

    child_id = str(child_ref.get("id", "")).strip()
    if child_manifest.get("brain_type") != "sub_brain":
        findings.append(Finding("error", "child_manifest_type_mismatch", "brain_type must be sub_brain", child_path))
    if child_manifest.get("brain_id") != child_id:
        findings.append(
            Finding(
                "error",
                "child_manifest_id_mismatch",
                f"brain_id does not match main child ref: {child_manifest.get('brain_id')} != {child_id}",
                child_path,
            )
        )
    if child_manifest.get("parent_brain") != _rel(MAIN_MANIFEST):
        findings.append(
            Finding("error", "child_manifest_parent_mismatch", "parent_brain must point to brain/brain_manifest.json", child_path)
        )
    if not str(child_manifest.get("attach_status", "")).startswith("attached"):
        findings.append(Finding("error", "child_manifest_attach_status_invalid", "attach_status must start with attached", child_path))

    for key in ("entrypoint", "body_root"):
        if child_manifest.get(key) != child_ref.get(key):
            findings.append(
                Finding(
                    "error",
                    "child_manifest_main_ref_mismatch",
                    f"{key} differs from main child ref: {child_manifest.get(key)} != {child_ref.get(key)}",
                    child_path,
                )
            )
        _add_path_finding(findings, child_manifest.get(key), child_path, "child_manifest_path_invalid", key)

    shared_source = str(child_manifest.get("shared_contract_source", "")).strip()
    source_path, source_key = _split_contract_source(shared_source)
    if source_path != _rel(MAIN_MANIFEST) or source_key != SHARED_CONTRACT_KEY:
        findings.append(
            Finding("error", "child_shared_contract_source_invalid", f"invalid shared contract source: {shared_source}", child_path)
        )

    module_map = _module_path_map(child_manifest)
    if not module_map:
        findings.append(Finding("error", "child_modules_invalid", "modules must be a non-empty list", child_path))
    else:
        seen_module_ids: set[str] = set()
        for module_id, paths in module_map.items():
            if module_id in seen_module_ids:
                findings.append(Finding("error", "child_module_duplicate", f"duplicate module id: {module_id}", child_path))
            seen_module_ids.add(module_id)
            if not paths:
                findings.append(Finding("error", "child_module_paths_empty", f"module has no paths: {module_id}", child_path))
            for module_path in paths:
                _add_path_finding(findings, module_path, child_path, "child_module_path_invalid", f"module.{module_id}")

    contract = main_manifest.get(SHARED_CONTRACT_KEY, {})
    required_modules = contract.get("required_modules", []) if isinstance(contract, dict) else []
    for module_id in required_modules:
        if str(module_id) not in module_map:
            findings.append(
                Finding(
                    "error",
                    "child_required_module_missing",
                    f"missing required shared module: {module_id}",
                    child_path,
                )
            )

    entrypoint = str(child_manifest.get("entrypoint", "")).strip()
    derived_read_order = _derive_child_read_order(main_manifest, child_manifest)
    if not derived_read_order:
        findings.append(Finding("error", "child_read_order_derivation_failed", "derived read order is empty", child_path))
    elif derived_read_order[0] != entrypoint:
        findings.append(
            Finding(
                "error",
                "child_read_order_entrypoint_mismatch",
                f"derived read order first item should equal entrypoint: {entrypoint}",
                child_path,
            )
        )

    for key in OPTIONAL_CHILD_PATH_KEYS:
        value = str(child_manifest.get(key, "")).strip()
        if value:
            _add_path_finding(findings, value, child_path, "child_optional_path_invalid", key)
            if derived_read_order and value not in derived_read_order:
                findings.append(
                    Finding(
                        "warning",
                        "child_optional_path_not_in_read_order",
                        f"{key} is not present in derived read order: {value}",
                        child_path,
                    )
                )

    write_routes = child_manifest.get("write_routes", {})
    if not isinstance(write_routes, dict) or not write_routes:
        findings.append(Finding("error", "child_write_routes_invalid", "write_routes must be a non-empty object", child_path))
    else:
        for key, value in write_routes.items():
            _add_path_finding(findings, value, child_path, "child_write_route_invalid", f"write_routes.{key}")

    body_map = child_manifest.get("body_map", {})
    if not isinstance(body_map, dict) or not body_map:
        findings.append(Finding("error", "child_body_map_invalid", "body_map must be a non-empty object", child_path))
    else:
        for key, values in body_map.items():
            if not isinstance(values, list) or not values:
                findings.append(Finding("error", "child_body_map_invalid", f"body_map.{key} must be a non-empty list", child_path))
                continue
            for value in values:
                _add_path_finding(findings, value, child_path, "child_body_map_path_invalid", f"body_map.{key}")

    regional = child_manifest.get("regional_specialization", {})
    if not isinstance(regional, dict) or not regional:
        findings.append(
            Finding("error", "child_regional_specialization_invalid", "regional_specialization must be non-empty", child_path)
        )
    else:
        role = str(regional.get("role", "")).strip()
        if not role:
            findings.append(Finding("error", "child_regional_role_missing", "regional role is empty", child_path))
        region_bindings = contract.get("region_bindings", {}) if isinstance(contract, dict) else {}
        for region in regional.get("priority_regions", []):
            if str(region) not in region_bindings:
                findings.append(
                    Finding("error", "child_priority_region_unknown", f"unknown priority region: {region}", child_path)
                )
        for module_id in regional.get("focus_modules", []):
            if str(module_id) not in module_map:
                findings.append(
                    Finding("error", "child_focus_module_missing", f"focus module not declared: {module_id}", child_path)
                )
        for value in regional.get("body_entry_priority", []):
            _add_path_finding(findings, value, child_path, "child_body_entry_priority_invalid", "body_entry_priority")

    handoff = child_manifest.get("handoff_contract", {})
    if not isinstance(handoff, dict) or not handoff:
        findings.append(Finding("error", "child_handoff_contract_invalid", "handoff_contract must be non-empty", child_path))
    elif not handoff.get("derive_entry_sequence_from_shared_contract") and not handoff.get("entry_sequence"):
        findings.append(
            Finding(
                "error",
                "child_handoff_contract_no_entry_sequence",
                "handoff must derive from shared contract or declare entry_sequence",
                child_path,
            )
        )

    for item in [child_path, *derived_read_order]:
        _validate_markdown_and_encoding(findings, item)


def _validate_brain_catalog(findings: list[Finding], main_manifest: dict[str, Any]) -> None:
    catalog_path = _rel(BRAIN_CATALOG)
    if not _workspace_path(BRAIN_CATALOG).exists():
        findings.append(Finding("error", "brain_catalog_missing", "brain/brain_catalog.json is missing", catalog_path))
        return
    try:
        file_catalog = _read_json(BRAIN_CATALOG)
    except Exception as exc:
        findings.append(Finding("error", "brain_catalog_invalid_json", str(exc), catalog_path))
        return
    built_catalog = build_brain_catalog()
    file_brains = file_catalog.get("brains")
    if not isinstance(file_brains, list) or not file_brains:
        findings.append(Finding("error", "brain_catalog_brains_invalid", "catalog brains must be a non-empty list", catalog_path))
        return

    records = {str(item.get("brain_id", "")): item for item in file_brains if isinstance(item, dict)}
    if "workspace_root" not in records:
        findings.append(Finding("error", "brain_catalog_workspace_root_missing", "workspace_root entry missing", catalog_path))
    elif records["workspace_root"].get("status") != "canonical_root":
        findings.append(Finding("error", "brain_catalog_workspace_root_status_invalid", "workspace_root must be canonical_root", catalog_path))

    for child in main_manifest.get("child_brains", []):
        if not isinstance(child, dict):
            continue
        child_id = str(child.get("id", "")).strip()
        record = records.get(child_id)
        if not record:
            findings.append(Finding("error", "brain_catalog_attached_child_missing", f"missing attached child: {child_id}", catalog_path))
            continue
        if record.get("status") != "attached":
            findings.append(Finding("error", "brain_catalog_attached_child_status_invalid", f"{child_id} must be attached", catalog_path))
        if record.get("manifest_path") != child.get("path"):
            findings.append(
                Finding(
                    "error",
                    "brain_catalog_manifest_mismatch",
                    f"{child_id} manifest mismatch: {record.get('manifest_path')} != {child.get('path')}",
                    catalog_path,
                )
            )

    built_records = {str(item.get("brain_id", "")): item for item in built_catalog.get("brains", []) if isinstance(item, dict)}
    for brain_id, built in built_records.items():
        if brain_id not in records:
            findings.append(Finding("warning", "brain_catalog_discovered_entry_missing", f"catalog omits discovered brain: {brain_id}", catalog_path))
    acknowledged_non_truth_statuses = {
        "cache_legacy",
        "non_truth_tooling",
        "external_or_inactive_missing_manifest",
    }
    action_needed_noncanonical_statuses = {"discovered_untracked", "missing_manifest"}
    for brain_id, record in records.items():
        status = str(record.get("status", ""))
        if status in acknowledged_non_truth_statuses | action_needed_noncanonical_statuses:
            if status in acknowledged_non_truth_statuses:
                detail = f"{brain_id} is an acknowledged non-truth source classified as {status}"
            else:
                detail = f"{brain_id} needs review: classified as {status} and is not a truth source"
            findings.append(
                Finding(
                    "warning",
                    "catalog_noncanonical_brain",
                    detail,
                    str(record.get("root", "") or catalog_path),
                )
            )


def run_checks() -> list[Finding]:
    findings: list[Finding] = []
    if not _workspace_path(MAIN_MANIFEST).exists():
        return [Finding("error", "main_manifest_missing", "brain/brain_manifest.json is missing", _rel(MAIN_MANIFEST))]

    main_manifest = _read_json(MAIN_MANIFEST)
    child_refs = _validate_main_manifest(findings, main_manifest)
    _validate_brain_catalog(findings, main_manifest)
    _validate_markdown_and_encoding(findings, _rel(MAIN_MANIFEST))
    for item in main_manifest.get("read_order", []):
        _validate_markdown_and_encoding(findings, str(item))

    contract = main_manifest.get("brain_contract", {})
    required_child_keys = DEFAULT_REQUIRED_CHILD_KEYS
    if isinstance(contract, dict):
        raw_keys = contract.get("required_sub_brain_keys")
        if isinstance(raw_keys, list) and raw_keys:
            required_child_keys = tuple(str(item) for item in raw_keys)

    for child_ref in child_refs:
        _validate_child_manifest(findings, main_manifest, child_ref, required_child_keys)
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check main/sub-brain attachment, routing, encoding, and contract integrity.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    findings = run_checks()
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    payload = {
        "status": "ok" if not errors else "failed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "findings": [finding.__dict__ for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status={payload['status']} errors={len(errors)} warnings={len(warnings)}")
        for finding in findings:
            path = f" path={finding.path}" if finding.path else ""
            print(f"[{finding.severity}] {finding.code}{path}: {finding.detail}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
