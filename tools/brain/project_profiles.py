from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAIN_MANIFEST = Path("brain/brain_manifest.json")
PYTHON_EXECUTABLE = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
DAILY_ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"
RESOURCE_LEASE_ROOT = "brain/output/resource_leases"


def _workspace_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(_workspace_path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().replace("\\", "/") for item in value if str(item).strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip().replace("\\", "/")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _child_ref(project_id: str) -> dict[str, Any]:
    main = _load_json(MAIN_MANIFEST)
    for child in main.get("child_brains", []) if isinstance(main.get("child_brains"), list) else []:
        if isinstance(child, dict) and str(child.get("id", "") or "") == project_id:
            return child
    return {}


def _child_manifest(project_id: str) -> dict[str, Any]:
    ref = _child_ref(project_id)
    path = str(ref.get("path", "") or "").strip()
    return _load_json(path) if path else {}


def _child_manifest_paths() -> list[str]:
    main = _load_json(MAIN_MANIFEST)
    out: list[str] = []
    for child in main.get("child_brains", []) if isinstance(main.get("child_brains"), list) else []:
        if not isinstance(child, Mapping):
            continue
        path = str(child.get("path", "") or "").strip().replace("\\", "/")
        if path:
            out.append(path)
    return _dedupe(out)


def _manifest_profile(manifest: Mapping[str, Any]) -> dict[str, Any]:
    profile = manifest.get("project_profile")
    return dict(profile) if isinstance(profile, Mapping) else {}


def _body_root(project_id: str, manifest: Mapping[str, Any], child_ref: Mapping[str, Any]) -> str:
    return str(manifest.get("body_root", "") or child_ref.get("body_root", "") or project_id).strip().replace("\\", "/")


def _default_commands(project_id: str, body_root: str) -> list[str]:
    if project_id == "daily_research":
        return [
            f"git diff -- {DAILY_ACTIVE_ARTIFACT}",
            "git diff --check",
            f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check",
            f"{PYTHON_EXECUTABLE} -m tools.brain.integrity_check --json",
        ]
    return [
        "git diff --check",
        f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check",
        f"{PYTHON_EXECUTABLE} -m tools.brain.integrity_check --json",
    ]


def _default_guard_profile(project_id: str) -> dict[str, Any]:
    is_daily = project_id == "daily_research"
    return {
        "active_artifact_guard": is_daily,
        "artifact_freshness": is_daily,
        "frontier_report": is_daily,
        "project_consistency": is_daily,
        "openmp_strict": is_daily,
        "brain_rules": is_daily,
    }


def _default_verification_profile(project_id: str, body_root: str) -> dict[str, Any]:
    return {
        "always_commands": _default_commands(project_id, body_root),
        "default_test_commands": [],
        "big_artifact_paths": [
            "daily_research/output/**",
            "daily_research/cache/**",
            "daily_research/archive/**",
        ]
        if project_id == "daily_research"
        else [f"{body_root}/output/**", f"{body_root}/cache/**", f"{body_root}/archive/**"],
    }


def _default_commit_policy(project_id: str, body_root: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    write_routes = manifest.get("write_routes") if isinstance(manifest.get("write_routes"), Mapping) else {}
    route_prefixes = []
    for value in write_routes.values():
        text = str(value or "").strip().replace("\\", "/")
        if text:
            route_prefixes.append(text if text.endswith("/") else text.rsplit("/", 1)[0] + "/")
    return {
        "auto_commit": True,
        "message_prefix": project_id,
        "allowed_prefixes": _dedupe([f"{body_root}/", *route_prefixes]),
        "block_on_baseline_overlap": True,
    }


def _default_process_namespace(project_id: str, body_root: str) -> dict[str, Any]:
    root = f"{body_root}/output/agent_runs"
    return {
        "project_id": project_id,
        "root": root,
        "run_path_template": f"{root}/<run_id>",
        "pid_registry": f"{root}/<run_id>/pid.json",
        "stdout": f"{root}/<run_id>/stdout.log",
        "stderr": f"{root}/<run_id>/stderr.log",
        "progress": f"{root}/<run_id>/progress.json",
        "summary": f"{root}/<run_id>/summary.json",
    }


def _merge_dict(base: dict[str, Any], override: Any) -> dict[str, Any]:
    if not isinstance(override, Mapping):
        return base
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_dict(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_project_profile(project_id: str | None) -> dict[str, Any]:
    normalized = str(project_id or "workspace").strip() or "workspace"
    if normalized in {"workspace", "workspace_root", "workspace_governance", "workspace-brain", "workspace_brain"}:
        return {
            "schema_version": 1,
            "project_id": "workspace",
            "body_root": ".",
            "guard_profile": {
                "active_artifact_guard": False,
                "artifact_freshness": False,
                "frontier_report": False,
                "project_consistency": False,
                "openmp_strict": False,
                "brain_rules": False,
            },
            "verification_profile": {
                "always_commands": [
                    "git diff --check",
                    f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check",
                    f"{PYTHON_EXECUTABLE} -m tools.brain.integrity_check --json",
                ],
                "default_test_commands": [
                    f"{PYTHON_EXECUTABLE} -m pytest tools/brain/tests/test_selective_verification.py tools/brain/tests/test_project_commit.py tools/brain/tests/test_platform.py -q",
                ],
                "big_artifact_paths": ["brain/output/**"],
            },
            "commit_policy": {
                "auto_commit": True,
                "message_prefix": "workspace-brain",
                "allowed_prefixes": _dedupe(["brain/", "tools/brain/", *_child_manifest_paths()]),
                "block_on_baseline_overlap": True,
            },
            "process_namespace": _default_process_namespace("workspace", "brain"),
            "cross_project_policy": {"allow_by_default": False, "lease_root": RESOURCE_LEASE_ROOT},
        }

    ref = _child_ref(normalized)
    manifest = _child_manifest(normalized)
    body_root = _body_root(normalized, manifest, ref)
    profile = _manifest_profile(manifest)
    guard_profile = _merge_dict(_default_guard_profile(normalized), profile.get("guard_profile"))
    verification_profile = _merge_dict(_default_verification_profile(normalized, body_root), profile.get("verification_profile"))
    commit_policy = _merge_dict(_default_commit_policy(normalized, body_root, manifest), profile.get("commit_policy"))
    process_namespace = _merge_dict(_default_process_namespace(normalized, body_root), profile.get("process_namespace"))
    cross_project_policy = _merge_dict(
        {"allow_by_default": False, "lease_root": RESOURCE_LEASE_ROOT},
        profile.get("cross_project_policy"),
    )
    return {
        "schema_version": 1,
        "project_id": normalized,
        "body_root": body_root,
        "guard_profile": guard_profile,
        "verification_profile": verification_profile,
        "commit_policy": commit_policy,
        "process_namespace": process_namespace,
        "cross_project_policy": cross_project_policy,
    }


def infer_project_id_from_paths(paths: list[str]) -> str:
    normalized = _dedupe(paths)
    if not normalized:
        return "workspace"
    refs = _load_json(MAIN_MANIFEST).get("child_brains", [])
    matches: list[str] = []
    for child in refs if isinstance(refs, list) else []:
        if not isinstance(child, Mapping):
            continue
        project_id = str(child.get("id", "") or "")
        body_root = str(child.get("body_root", "") or project_id).replace("\\", "/").strip("/")
        if body_root and any(path == body_root or path.startswith(f"{body_root}/") for path in normalized):
            matches.append(project_id)
    return matches[0] if len(set(matches)) == 1 else ("cross_project" if matches else "workspace")
