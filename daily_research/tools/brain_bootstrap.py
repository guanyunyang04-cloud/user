from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN_MANIFEST = Path("brain/brain_manifest.json")
OPTIONAL_BRAIN_KEYS = (
    "identity_path",
    "rule_memory_path",
    "lesson_memory_path",
    "temporal_state_path",
    "handoff_packet_path",
    "governance_path",
)


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


def _resolve_child(main_manifest: dict[str, Any], child_id: str) -> dict[str, Any]:
    child_brains = main_manifest.get("child_brains", [])
    for child in child_brains:
        if not isinstance(child, dict):
            continue
        if child.get("id") == child_id or child.get("body_root") == child_id:
            return child
    raise KeyError(f"Unknown child brain: {child_id}")


def _validate_child_link(child_ref: dict[str, Any], child_manifest_path: Path, child_manifest: dict[str, Any]) -> None:
    if child_manifest.get("brain_type") != "sub_brain":
        raise ValueError(f"{child_manifest_path.as_posix()} must declare brain_type=sub_brain")
    if child_manifest.get("parent_brain") != MAIN_MANIFEST.as_posix():
        raise ValueError(f"{child_manifest_path.as_posix()} parent_brain mismatch")
    attach_status = str(child_manifest.get("attach_status", ""))
    if not attach_status.startswith("attached"):
        raise ValueError(f"{child_manifest_path.as_posix()} is not attached to main brain")
    if child_manifest.get("brain_id") != child_ref.get("id"):
        raise ValueError(f"{child_manifest_path.as_posix()} brain_id mismatch")


def _build_main_boot_order(main_manifest: dict[str, Any]) -> list[Path]:
    order = [MAIN_MANIFEST]
    for item in main_manifest.get("read_order", []):
        order.append(Path(str(item)))
    return _dedupe_paths(order)


def _build_child_boot_order(child_manifest_path: Path, child_manifest: dict[str, Any]) -> list[Path]:
    handoff = child_manifest.get("handoff_contract", {})
    entry_sequence = handoff.get("entry_sequence")
    if isinstance(entry_sequence, list) and entry_sequence:
        order = [Path(str(item)) for item in entry_sequence]
    else:
        order = [child_manifest_path]
        for item in child_manifest.get("read_order", []):
            order.append(Path(str(item)))
    return _dedupe_paths(order)


def build_bootstrap_payload(child_id: str | None) -> dict[str, Any]:
    main_manifest = _read_json(MAIN_MANIFEST)
    _validate_main_manifest(main_manifest)

    main_order = _build_main_boot_order(main_manifest)
    for path in main_order:
        _ensure_exists(path)

    payload: dict[str, Any] = {
        "root": ROOT.as_posix(),
        "main_manifest": MAIN_MANIFEST.as_posix(),
        "main_entrypoint": str(main_manifest.get("entrypoint", "")),
        "main_boot_order": [path.as_posix() for path in main_order],
    }
    payload.update(_collect_optional_brain_paths("main", main_manifest))

    if child_id is None:
        payload["boot_order"] = payload["main_boot_order"]
        return payload

    child_ref = _resolve_child(main_manifest, child_id)
    child_manifest_path = Path(str(child_ref["path"]))
    _ensure_exists(child_manifest_path)
    child_manifest = _read_json(child_manifest_path)
    _validate_child_link(child_ref, child_manifest_path, child_manifest)

    child_order = _build_child_boot_order(child_manifest_path, child_manifest)
    for path in child_order:
        _ensure_exists(path)

    boot_order = _dedupe_paths(main_order + child_order)
    payload.update(
        {
            "child_brain": str(child_ref["id"]),
            "child_manifest": child_manifest_path.as_posix(),
            "child_entrypoint": str(child_manifest.get("entrypoint", "")),
            "child_attach_status": str(child_manifest.get("attach_status", "")),
            "child_boot_order": [path.as_posix() for path in child_order],
            "boot_order": [path.as_posix() for path in boot_order],
        }
    )
    payload.update(_collect_optional_brain_paths("child", child_manifest))
    return payload


def _print_text(payload: dict[str, Any], absolute: bool) -> None:
    child_brain = payload.get("child_brain")
    if child_brain:
        print(f"Attached main brain -> {child_brain}")
    else:
        print("Attached main brain")

    for key in ("main_identity_path", "main_handoff_packet_path", "child_identity_path", "child_handoff_packet_path"):
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
