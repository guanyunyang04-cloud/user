from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.brain.project_profiles import load_project_profile


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _normalize_path(path: str) -> str:
    text = str(path or "").strip().strip('"').strip("'").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _normalize_path(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _path_allowed(path: str, allowed_prefixes: Iterable[str]) -> bool:
    normalized = _normalize_path(path)
    for raw_prefix in allowed_prefixes:
        prefix = _normalize_path(raw_prefix)
        if not prefix:
            continue
        if prefix.endswith("/"):
            if normalized.startswith(prefix):
                return True
        elif normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    return False


def build_commit_message(*, project_id: str, task_summary: str, verified: list[str] | None = None) -> str:
    project = str(project_id or "workspace-brain").strip() or "workspace-brain"
    summary = " ".join(str(task_summary or "agent task").strip().split()) or "agent task"
    verification = "; ".join(str(item).strip() for item in (verified or []) if str(item).strip()) or "not_run"
    return "\n".join(
        [
            f"{project}: {summary}",
            "",
            f"Project: {project}",
            f"Agent-Task: {summary}",
            f"Verified: {verification}",
        ]
    )


def check_commit_scope(
    *,
    project_id: str,
    changed_paths: list[str],
    allowed_prefixes: list[str],
    baseline_dirty_paths: list[str] | None = None,
) -> dict[str, object]:
    changed = _unique(changed_paths)
    baseline = set(_unique(baseline_dirty_paths or []))
    blocked = [path for path in changed if not _path_allowed(path, allowed_prefixes)]
    overlap = [path for path in changed if path in baseline]
    status = "blocked" if blocked or overlap else "ok"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "project_id": str(project_id or ""),
        "changed_paths": changed,
        "allowed_prefixes": _unique(allowed_prefixes),
        "blocked_paths": blocked,
        "baseline_overlap_paths": overlap,
    }
    if status == "blocked":
        payload["reason"] = "project_commit_scope_conflict"
    return payload


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def changed_paths() -> list[str]:
    paths: list[str] = []
    for args in (["diff", "--name-only"], ["diff", "--cached", "--name-only"], ["ls-files", "--others", "--exclude-standard"]):
        result = _run_git(args)
        if result.returncode == 0:
            paths.extend(line.strip() for line in (result.stdout or "").splitlines() if line.strip())
    return _unique(paths)


def stage_and_commit_project(
    *,
    project_id: str,
    task_summary: str,
    verified: list[str],
    baseline_dirty_paths: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    verified_commands = [str(item).strip() for item in (verified or []) if str(item).strip()]
    if not verified_commands:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "project_commit_unverified",
            "project_id": project_id,
        }
    profile = load_project_profile(project_id)
    policy = profile.get("commit_policy", {}) if isinstance(profile.get("commit_policy"), dict) else {}
    allowed_prefixes = [str(item) for item in policy.get("allowed_prefixes", []) or []]
    paths = changed_paths()
    scope = check_commit_scope(
        project_id=project_id,
        changed_paths=paths,
        allowed_prefixes=allowed_prefixes,
        baseline_dirty_paths=baseline_dirty_paths or [],
    )
    if scope["status"] != "ok":
        return scope
    message = build_commit_message(project_id=str(policy.get("message_prefix", project_id) or project_id), task_summary=task_summary, verified=verified_commands)
    if dry_run:
        return {"schema_version": 1, "status": "dry_run", "scope": scope, "commit_message": message}
    if not paths:
        return {"schema_version": 1, "status": "blocked", "reason": "no_project_changes", "scope": scope}
    stage = _run_git(["add", "--", *paths])
    if stage.returncode != 0:
        return {"schema_version": 1, "status": "blocked", "reason": "git_stage_failed", "stderr": stage.stderr}
    commit = _run_git(["commit", "-m", message])
    if commit.returncode != 0:
        return {"schema_version": 1, "status": "blocked", "reason": "git_commit_failed", "stdout": commit.stdout, "stderr": commit.stderr}
    sha = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    return {"schema_version": 1, "status": "committed", "project_id": project_id, "commit": sha, "scope": scope}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-scoped brain commit helper.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--task-summary", default="agent task")
    parser.add_argument("--verified", nargs="*", default=[])
    parser.add_argument("--baseline-dirty-paths", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = stage_and_commit_project(
        project_id=str(args.project_id),
        task_summary=str(args.task_summary),
        verified=list(args.verified or []),
        baseline_dirty_paths=list(args.baseline_dirty_paths or []),
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)
    return 0 if payload.get("status") in {"committed", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
