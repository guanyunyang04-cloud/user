from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAIN_MANIFEST = Path("brain/brain_manifest.json")


@dataclass(frozen=True)
class MultiParadigmFinding:
    severity: str
    code: str
    detail: str
    path: str | None = None
    brain_id: str | None = None


def _workspace_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _read_text(path: str | Path) -> str:
    try:
        return _workspace_path(path).read_text(encoding="utf-8-sig")
    except Exception:
        return ""


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_text(path))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_exists(path: str | Path) -> bool:
    return _workspace_path(path).exists()


def _as_posix(path: str | Path) -> str:
    return Path(str(path)).as_posix()


def _module_paths(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for module in manifest.get("modules", []) or []:
        if not isinstance(module, dict):
            continue
        for raw_path in module.get("paths", []) or []:
            text = str(raw_path or "").strip().replace("\\", "/")
            if text:
                paths.append(text)
    return paths


def _declared_paths(manifest: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in (
        "identity_path",
        "state_path",
        "knowledge_path",
        "operations_path",
        "governance_path",
        "entrypoint",
    ):
        text = str(manifest.get(key, "") or "").strip().replace("\\", "/")
        if text:
            candidates.append(text)
    read_order = manifest.get("read_order")
    if isinstance(read_order, list):
        candidates.extend(str(item or "").strip().replace("\\", "/") for item in read_order)
    candidates.extend(_module_paths(manifest))
    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _current_interface_paths(manifest: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for path in _declared_paths(manifest):
        normalized = path.replace("\\", "/")
        if "/references/" in normalized or normalized.endswith("/episodic_memory.md") or normalized.endswith("episodic_memory.md"):
            continue
        if not normalized.endswith((".md", ".json")):
            continue
        if _path_exists(normalized):
            out.append(normalized)
    return out


def _has_object_interface(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "### object `",
            "## Object",
            "Object Instances",
            "Governed Objects",
            "Runtime Objects",
            "Body Map Objects",
            "对象式",
            "对象实例",
            "治理对象",
        )
    )


def _has_procedure_interface(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "### procedure `",
            "## Procedures",
            "Procedure Entries",
            "Procedure Layer",
            "过程式",
            "过程目录",
            "过程入口",
            "对象方法入口",
        )
    )


def _has_function_interface(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "## Pure Functions",
            "function `",
            "Function Layer",
            "函数式",
            "纯函数",
            "判断函数",
            "Validation Selection Function",
        )
    )


def _brain_records(main_manifest: dict[str, Any]) -> list[dict[str, str]]:
    records = [
        {
            "brain_id": "workspace_root",
            "manifest_path": MAIN_MANIFEST.as_posix(),
            "root": "brain",
            "status": "canonical_root",
        }
    ]
    for child in main_manifest.get("child_brains", []) or []:
        if not isinstance(child, dict):
            continue
        brain_id = str(child.get("id", "") or "").strip()
        manifest_path = str(child.get("path", "") or "").strip().replace("\\", "/")
        if not brain_id or not manifest_path:
            continue
        records.append(
            {
                "brain_id": brain_id,
                "manifest_path": manifest_path,
                "root": str(Path(manifest_path).parent).replace("\\", "/"),
                "status": str(child.get("attach_status", "") or "attached"),
            }
        )
    return records


def _lint_brain(record: dict[str, str]) -> dict[str, Any]:
    manifest_path = record["manifest_path"]
    manifest = _read_json(manifest_path)
    paths = _current_interface_paths(manifest)
    texts = {path: _read_text(path) for path in paths}
    object_paths = [path for path, text in texts.items() if _has_object_interface(text)]
    procedure_paths = [path for path, text in texts.items() if _has_procedure_interface(text)]
    function_paths = [path for path, text in texts.items() if _has_function_interface(text)]
    missing: list[str] = []
    if not object_paths:
        missing.append("object_interface")
    if not procedure_paths:
        missing.append("procedure_interface")
    if not function_paths:
        missing.append("function_interface")
    return {
        "brain_id": record["brain_id"],
        "status": record["status"],
        "root": record["root"],
        "manifest_path": manifest_path,
        "checked_paths": paths,
        "interfaces": {
            "object": object_paths,
            "procedure": procedure_paths,
            "function": function_paths,
        },
        "missing": missing,
        "ok": not missing,
    }


def run_multi_paradigm_lint(*, scope: str = "attached") -> dict[str, Any]:
    main_manifest = _read_json(MAIN_MANIFEST)
    records = _brain_records(main_manifest)
    if scope != "all":
        records = [record for record in records if record["status"] in {"canonical_root", "attached", "attached_to_main_brain"}]
    brains = [_lint_brain(record) for record in records]
    findings: list[MultiParadigmFinding] = []
    for brain in brains:
        for missing in brain["missing"]:
            findings.append(
                MultiParadigmFinding(
                    severity="error",
                    code="multi_paradigm_interface_missing",
                    detail=f"{brain['brain_id']} missing {missing}",
                    path=str(brain.get("manifest_path", "") or ""),
                    brain_id=str(brain.get("brain_id", "") or ""),
                )
            )
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    return {
        "status": "ok" if not errors else "failed",
        "scope": scope,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "brains": brains,
        "findings": [finding.__dict__ for finding in findings],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check brain docs for object/procedure/function interfaces.")
    parser.add_argument("--scope", choices=("attached", "all"), default="attached")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_multi_paradigm_lint(scope=str(args.scope or "attached"))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status={payload['status']} errors={payload['error_count']} warnings={payload['warning_count']}")
        for finding in payload["findings"]:
            print(f"[{finding['severity']}] {finding['code']} brain={finding.get('brain_id', '')}: {finding['detail']}")
    return 1 if payload["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
