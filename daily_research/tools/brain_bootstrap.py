from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MAIN_MANIFEST = Path("brain/brain_manifest.json")
OPTIONAL_BRAIN_KEYS = (
    "identity_path",
    "state_path",
    "knowledge_path",
    "operations_path",
    "governance_path",
)
SHARED_CONTRACT_KEY = "shared_regional_brain_contract"


def _platform_runtime_fields(boot_order: list[str]) -> dict[str, Any]:
    from daily_research.tools.brain_platform import (
        check_text_encoding,
        load_workflow_registry,
        resolve_artifact_freshness,
    )

    reports = {path: check_text_encoding(Path(path)).to_dict() for path in boot_order}
    return {
        "artifact_freshness": resolve_artifact_freshness().to_dict(),
        "workflow_hints": {
            "available_workflows": sorted(load_workflow_registry()),
            "recommended_default": "brain_handoff",
        },
        "encoding_report": {
            "reports": reports,
            "has_errors": any(
                (not item["is_utf8"]) or item["has_replacement_char"] or item["suspicious_mojibake_count"] > 0
                for item in reports.values()
            ),
        },
    }


def _read_json(rel_path: Path) -> dict[str, Any]:
    path = ROOT / rel_path
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _collect_optional_brain_paths(prefix: str, manifest: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in OPTIONAL_BRAIN_KEYS:
        value = str(manifest.get(key, "")).strip()
        if value:
            out[f"{prefix}_{key}"] = value
    return out


def _ensure_exists(rel_path: Path) -> None:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing brain path: {rel_path.as_posix()}")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        normalized = path.as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(path)
    return ordered


def _validate_main_manifest(data: dict[str, Any]) -> None:
    if data.get("brain_type") != "main":
        raise ValueError("Main manifest must declare brain_type=main")
    child_brains = data.get("child_brains")
    if not isinstance(child_brains, list) or not child_brains:
        raise ValueError("Main manifest must declare child_brains")
    shared_contract = data.get(SHARED_CONTRACT_KEY)
    if not isinstance(shared_contract, dict):
        raise ValueError(f"Main manifest must declare {SHARED_CONTRACT_KEY}")


def _resolve_child(main_manifest: dict[str, Any], child_id: str) -> dict[str, Any]:
    child_brains = main_manifest.get("child_brains", [])
    for child in child_brains:
        if not isinstance(child, dict):
            continue
        if child.get("id") == child_id or child.get("body_root") == child_id:
            return child
    raise KeyError(f"Unknown child brain: {child_id}")


def _split_contract_source(source: str) -> tuple[Path, str]:
    path_text, _, key = source.partition("#")
    return Path(path_text), key


def _shared_contract(main_manifest: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    contract = main_manifest.get(SHARED_CONTRACT_KEY)
    if not isinstance(contract, dict):
        raise ValueError(f"Main manifest must declare {SHARED_CONTRACT_KEY}")
    if source:
        path, key = _split_contract_source(source)
        if path.as_posix() != MAIN_MANIFEST.as_posix() or key != SHARED_CONTRACT_KEY:
            raise ValueError(f"Unsupported shared_contract_source: {source}")
    return contract


def _parse_module_order_item(raw: Any) -> tuple[str, bool]:
    item = str(raw).strip()
    optional = item.endswith("?")
    return (item[:-1] if optional else item), optional


def _module_path_map(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for module in manifest.get("modules", []):
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("id", "")).strip()
        paths = module.get("paths")
        if not module_id or not isinstance(paths, list):
            continue
        out[module_id] = [Path(str(path)) for path in paths if str(path).strip()]
    return out


def _validate_child_link(
    main_manifest: dict[str, Any], child_ref: dict[str, Any], child_manifest_path: Path, child_manifest: dict[str, Any]
) -> None:
    if child_manifest.get("brain_type") != "sub_brain":
        raise ValueError(f"{child_manifest_path.as_posix()} must declare brain_type=sub_brain")
    if child_manifest.get("parent_brain") != MAIN_MANIFEST.as_posix():
        raise ValueError(f"{child_manifest_path.as_posix()} parent_brain mismatch")
    attach_status = str(child_manifest.get("attach_status", ""))
    if not attach_status.startswith("attached"):
        raise ValueError(f"{child_manifest_path.as_posix()} is not attached to main brain")
    if child_manifest.get("brain_id") != child_ref.get("id"):
        raise ValueError(f"{child_manifest_path.as_posix()} brain_id mismatch")
    shared_source = str(child_manifest.get("shared_contract_source", "")).strip()
    if not shared_source and "read_order" not in child_manifest:
        raise ValueError(f"{child_manifest_path.as_posix()} must declare shared_contract_source or explicit read_order")
    if shared_source:
        _shared_contract(main_manifest, shared_source)


def _build_main_boot_order(main_manifest: dict[str, Any]) -> list[Path]:
    order = [MAIN_MANIFEST]
    for item in main_manifest.get("read_order", []):
        order.append(Path(str(item)))
    return _dedupe_paths(order)


def _derive_child_read_order(main_manifest: dict[str, Any], child_manifest: dict[str, Any]) -> list[Path]:
    shared_source = str(child_manifest.get("shared_contract_source", "")).strip()
    contract = _shared_contract(main_manifest, shared_source or None)
    if not contract.get("derive_read_order_from_modules"):
        raise ValueError("Shared regional contract must enable derive_read_order_from_modules")

    module_map = _module_path_map(child_manifest)
    ordered: list[Path] = []
    seen: set[str] = set()

    for raw in contract.get("default_module_order", []):
        module_id, optional = _parse_module_order_item(raw)
        paths = module_map.get(module_id, [])
        if not paths:
            if optional:
                continue
            raise FileNotFoundError(f"Child brain missing required module for shared contract: {module_id}")
        for path in paths:
            normalized = path.as_posix()
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(path)

    for module in child_manifest.get("modules", []):
        if not isinstance(module, dict):
            continue
        for raw_path in module.get("paths", []):
            path = Path(str(raw_path))
            normalized = path.as_posix()
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(path)

    return _dedupe_paths(ordered)


def _build_child_read_order(main_manifest: dict[str, Any], child_manifest: dict[str, Any]) -> list[Path]:
    read_order = child_manifest.get("read_order")
    if isinstance(read_order, list) and read_order:
        return _dedupe_paths([Path(str(item)) for item in read_order])
    return _derive_child_read_order(main_manifest, child_manifest)


def _collect_fast_handoff_paths(main_manifest: dict[str, Any], child_manifest: dict[str, Any]) -> list[str]:
    contract = _shared_contract(main_manifest, str(child_manifest.get("shared_contract_source", "")).strip() or None)
    module_map = _module_path_map(child_manifest)
    out: list[str] = []
    for module_id in contract.get("fast_handoff_modules", []):
        for path in module_map.get(str(module_id), []):
            normalized = path.as_posix()
            if normalized not in out:
                out.append(normalized)
    return out


def _build_child_boot_order(main_manifest: dict[str, Any], child_manifest_path: Path, child_manifest: dict[str, Any]) -> list[Path]:
    handoff = child_manifest.get("handoff_contract", {})
    entry_sequence = handoff.get("entry_sequence")
    if isinstance(entry_sequence, list) and entry_sequence:
        order = [Path(str(item)) for item in entry_sequence]
    else:
        order = [child_manifest_path]
        order.extend(_build_child_read_order(main_manifest, child_manifest))
    return _dedupe_paths(order)


def build_bootstrap_payload(child_id: str | None) -> dict[str, Any]:
    main_manifest = _read_json(MAIN_MANIFEST)
    _validate_main_manifest(main_manifest)
    shared_contract = _shared_contract(main_manifest)

    main_order = _build_main_boot_order(main_manifest)
    for path in main_order:
        _ensure_exists(path)

    payload: dict[str, Any] = {
        "root": ROOT.as_posix(),
        "main_manifest": MAIN_MANIFEST.as_posix(),
        "main_entrypoint": str(main_manifest.get("entrypoint", "")),
        "main_boot_order": [path.as_posix() for path in main_order],
        "shared_module_order": list(shared_contract.get("default_module_order", [])),
        "shared_region_bindings": dict(shared_contract.get("region_bindings", {})),
        "shared_fast_handoff_modules": list(shared_contract.get("fast_handoff_modules", [])),
    }
    payload.update(_collect_optional_brain_paths("main", main_manifest))

    if child_id is None:
        payload["boot_order"] = payload["main_boot_order"]
        payload.update(_platform_runtime_fields(payload["boot_order"]))
        return payload

    child_ref = _resolve_child(main_manifest, child_id)
    child_manifest_path = Path(str(child_ref["path"]))
    _ensure_exists(child_manifest_path)
    child_manifest = _read_json(child_manifest_path)
    _validate_child_link(main_manifest, child_ref, child_manifest_path, child_manifest)

    child_order = _build_child_boot_order(main_manifest, child_manifest_path, child_manifest)
    for path in child_order:
        _ensure_exists(path)

    boot_order = _dedupe_paths(main_order + child_order)
    payload.update(
        {
            "child_brain": str(child_ref["id"]),
            "child_manifest": child_manifest_path.as_posix(),
            "child_entrypoint": str(child_manifest.get("entrypoint", "")),
            "child_attach_status": str(child_manifest.get("attach_status", "")),
            "child_regional_specialization": dict(child_manifest.get("regional_specialization", {})),
            "child_fast_handoff_paths": _collect_fast_handoff_paths(main_manifest, child_manifest),
            "child_boot_order": [path.as_posix() for path in child_order],
            "boot_order": [path.as_posix() for path in boot_order],
        }
    )
    payload.update(_collect_optional_brain_paths("child", child_manifest))
    payload.update(_platform_runtime_fields(payload["boot_order"]))
    return payload


def _print_text(payload: dict[str, Any], absolute: bool) -> None:
    child_brain = payload.get("child_brain")
    if child_brain:
        print(f"Attached main brain -> {child_brain}")
    else:
        print("Attached main brain")

    child_specialization = payload.get("child_regional_specialization")
    if isinstance(child_specialization, dict):
        role = str(child_specialization.get("role", "")).strip()
        if role:
            print(f"regional role: {role}")

    for key in ("main_identity_path", "main_state_path", "child_identity_path", "child_state_path"):
        if key not in payload:
            continue
        label = key.replace("_path", "").replace("_", " ")
        path = (ROOT / payload[key]).resolve() if absolute else Path(payload[key])
        print(f"{label}: {path}")

    for index, rel in enumerate(payload["boot_order"], start=1):
        path = (ROOT / rel).resolve() if absolute else Path(rel)
        print(f"{index:02d}. {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve the main-brain-first bootstrap order for this workspace.")
    parser.add_argument(
        "--child",
        help="Optional child brain id/body_root, e.g. daily_research, t0_project, daily_stock_analysis-main",
    )
    parser.add_argument("--json", action="store_true", help="Print the resolved bootstrap payload as JSON.")
    parser.add_argument("--absolute", action="store_true", help="Print absolute file paths in text mode.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = build_bootstrap_payload(args.child)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _print_text(payload, absolute=args.absolute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
