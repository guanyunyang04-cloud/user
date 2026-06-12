from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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


def split_commit_paths(*, paths: list[str], allowed_prefixes: list[str]) -> tuple[list[str], list[str]]:
    project_paths: list[str] = []
    external_paths: list[str] = []
    for path in _unique(paths):
        if _path_allowed(path, allowed_prefixes):
            project_paths.append(path)
        else:
            external_paths.append(path)
    return project_paths, external_paths


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _run_git_with_pathspec(args: list[str], paths: list[str]) -> subprocess.CompletedProcess[str]:
    normalized_paths = _unique(paths)
    with tempfile.NamedTemporaryFile("wb", delete=False) as raw:
        raw.write(b"\0".join(path.encode("utf-8") for path in normalized_paths))
        raw_path = raw.name
    try:
        return _run_git([*args, f"--pathspec-from-file={raw_path}", "--pathspec-file-nul"])
    finally:
        Path(raw_path).unlink(missing_ok=True)


def _git_path_list(args: list[str]) -> list[str]:
    result = _run_git(args)
    if result.returncode != 0:
        return []
    return _unique(line.strip() for line in (result.stdout or "").splitlines() if line.strip())


def _stage_project_paths(project_paths: list[str]) -> subprocess.CompletedProcess[str]:
    project_set = set(_unique(project_paths))
    tracked_updates = [path for path in _git_path_list(["diff", "--name-only"]) if path in project_set]
    untracked_adds = [path for path in _git_path_list(["ls-files", "--others", "--exclude-standard"]) if path in project_set]
    for args, paths in ((["add", "--update"], tracked_updates), (["add"], untracked_adds)):
        if not paths:
            continue
        result = _run_git_with_pathspec(args, paths)
        if result.returncode != 0:
            return result
    return subprocess.CompletedProcess(["stage_project_paths"], 0, stdout="", stderr="")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for args in (["diff", "--name-only"], ["diff", "--cached", "--name-only"], ["ls-files", "--others", "--exclude-standard"]):
        paths.extend(_git_path_list(args))
    return _unique(paths)


def stage_and_commit_project(
    *,
    project_id: str,
    task_summary: str,
    verified: list[str],
    baseline_dirty_paths: list[str] | None = None,
    expected_paths: list[str] | None = None,
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
    project_paths, ignored_external_paths = split_commit_paths(paths=paths, allowed_prefixes=allowed_prefixes)
    expected_changed = set(_unique(expected_paths or [])).intersection(_unique(paths))
    ignored_expected_paths = [path for path in ignored_external_paths if path in expected_changed]
    scope = check_commit_scope(
        project_id=project_id,
        changed_paths=project_paths,
        allowed_prefixes=allowed_prefixes,
        baseline_dirty_paths=baseline_dirty_paths or [],
    )
    scope["ignored_external_paths"] = ignored_external_paths
    scope["expected_paths"] = _unique(expected_paths or [])
    scope["ignored_expected_paths"] = ignored_expected_paths
    if ignored_expected_paths:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "project_commit_expected_paths_out_of_scope",
            "project_id": project_id,
            "scope": scope,
        }
    if scope["status"] != "ok":
        return scope
    message = build_commit_message(project_id=str(policy.get("message_prefix", project_id) or project_id), task_summary=task_summary, verified=verified_commands)
    if not project_paths:
        return {"schema_version": 1, "status": "blocked", "reason": "no_project_changes", "scope": scope}
    if dry_run:
        return {"schema_version": 1, "status": "dry_run", "scope": scope, "commit_message": message}
    stage = _stage_project_paths(project_paths)
    if stage.returncode != 0:
        return {"schema_version": 1, "status": "blocked", "reason": "git_stage_failed", "stderr": stage.stderr}
    staged_paths = _git_path_list(["diff", "--cached", "--name-only"])
    staged_project_paths, staged_external_paths = split_commit_paths(paths=staged_paths, allowed_prefixes=allowed_prefixes)
    if staged_external_paths:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "git_staged_external_paths",
            "project_id": project_id,
            "staged_external_paths": staged_external_paths,
        }
    if not staged_project_paths:
        return {"schema_version": 1, "status": "blocked", "reason": "no_staged_project_changes", "scope": scope}
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
    parser.add_argument("--expect-paths", nargs="*", default=[])
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
        expected_paths=list(args.expect_paths or []),
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)
    return 0 if payload.get("status") in {"committed", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
