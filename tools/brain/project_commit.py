from __future__ import annotations

import argparse
import json
import re
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


def _git_output(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(
        part.strip()
        for part in (result.stdout or "", result.stderr or "")
        if part.strip()
    )
    output = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", output, flags=re.IGNORECASE)
    return re.sub(r"([?&](?:token|access_token)=)[^&\s]+", r"\1***", output, flags=re.IGNORECASE)


def inspect_push_target(*, remote: str = "origin") -> dict[str, object]:
    remote_name = str(remote or "origin").strip() or "origin"
    branch_result = _run_git(["symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch_result.returncode != 0 or not branch_result.stdout.strip():
        return {
            "status": "blocked",
            "reason": "git_detached_head",
            "remote": remote_name,
        }
    branch = branch_result.stdout.strip()
    remote_result = _run_git(["remote", "get-url", remote_name])
    if remote_result.returncode != 0:
        return {
            "status": "blocked",
            "reason": "git_remote_missing",
            "remote": remote_name,
            "branch": branch,
        }
    fetch = _run_git(["fetch", "--prune", remote_name])
    if fetch.returncode != 0:
        return {
            "status": "blocked",
            "reason": "git_fetch_failed",
            "remote": remote_name,
            "branch": branch,
            "detail": _git_output(fetch),
        }
    remote_ref = f"refs/remotes/{remote_name}/{branch}"
    remote_exists = _run_git(["show-ref", "--verify", "--quiet", remote_ref]).returncode == 0
    local_ahead = 0
    remote_ahead = 0
    if remote_exists:
        divergence = _run_git(["rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"])
        try:
            local_text, remote_text = divergence.stdout.split()
            local_ahead = int(local_text)
            remote_ahead = int(remote_text)
        except (AttributeError, TypeError, ValueError):
            return {
                "status": "blocked",
                "reason": "git_divergence_check_failed",
                "remote": remote_name,
                "branch": branch,
                "detail": _git_output(divergence),
            }
    payload: dict[str, object] = {
        "status": "ok",
        "remote": remote_name,
        "branch": branch,
        "remote_branch_exists": remote_exists,
        "local_ahead": local_ahead,
        "remote_ahead": remote_ahead,
    }
    if remote_ahead:
        payload["status"] = "blocked"
        payload["reason"] = "git_remote_branch_ahead"
    return payload


def push_current_branch(*, remote: str, branch: str) -> dict[str, object]:
    push = _run_git(["push", "--set-upstream", remote, branch])
    if push.returncode != 0:
        return {
            "status": "blocked",
            "reason": "git_push_failed",
            "remote": remote,
            "branch": branch,
            "detail": _git_output(push),
        }
    head = _run_git(["rev-parse", "HEAD"])
    remote_head = _run_git(["rev-parse", f"refs/remotes/{remote}/{branch}"])
    head_sha = head.stdout.strip()
    remote_sha = remote_head.stdout.strip()
    if head.returncode != 0 or remote_head.returncode != 0 or not head_sha or head_sha != remote_sha:
        return {
            "status": "blocked",
            "reason": "git_push_verification_failed",
            "remote": remote,
            "branch": branch,
            "head": head_sha,
            "remote_head": remote_sha,
        }
    return {
        "status": "ok",
        "remote": remote,
        "branch": branch,
        "head": head_sha,
        "remote_head": remote_sha,
    }


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


def _git_path_is_tracked(path: str) -> bool:
    return _run_git(["ls-files", "--error-unmatch", "--", _normalize_path(path)]).returncode == 0


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
    push: bool = False,
    remote: str = "origin",
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
    all_project_paths, ignored_external_paths = split_commit_paths(paths=paths, allowed_prefixes=allowed_prefixes)
    expected = _unique(expected_paths or [])
    expected_set = set(expected)
    project_paths = [path for path in all_project_paths if not expected or path in expected_set]
    ignored_project_paths = [path for path in all_project_paths if expected and path not in expected_set]
    expected_changed = expected_set.intersection(_unique(paths))
    ignored_expected_paths = [path for path in ignored_external_paths if path in expected_changed]
    missing_expected_paths = [
        path
        for path in expected
        if path not in expected_changed and not _git_path_is_tracked(path)
    ]
    scope = check_commit_scope(
        project_id=project_id,
        changed_paths=project_paths,
        allowed_prefixes=allowed_prefixes,
        baseline_dirty_paths=baseline_dirty_paths or [],
    )
    scope["ignored_external_paths"] = ignored_external_paths
    scope["ignored_project_paths"] = ignored_project_paths
    scope["expected_paths"] = expected
    scope["ignored_expected_paths"] = ignored_expected_paths
    scope["missing_expected_paths"] = missing_expected_paths
    if ignored_expected_paths:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "project_commit_expected_paths_out_of_scope",
            "project_id": project_id,
            "scope": scope,
        }
    if missing_expected_paths:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "project_commit_expected_paths_missing",
            "project_id": project_id,
            "scope": scope,
        }
    if scope["status"] != "ok":
        return scope
    message = build_commit_message(project_id=str(policy.get("message_prefix", project_id) or project_id), task_summary=task_summary, verified=verified_commands)
    if dry_run:
        if not project_paths and not push:
            return {"schema_version": 1, "status": "blocked", "reason": "no_project_changes", "scope": scope}
        return {
            "schema_version": 1,
            "status": "dry_run",
            "scope": scope,
            "commit_message": message,
            "push_requested": bool(push),
            "remote": str(remote or "origin"),
        }
    push_target: dict[str, object] | None = None
    if push:
        push_target = inspect_push_target(remote=remote)
        if push_target.get("status") != "ok":
            return {
                "schema_version": 1,
                "status": "blocked",
                "reason": push_target.get("reason", "git_push_preflight_failed"),
                "push": push_target,
                "scope": scope,
            }
    if not project_paths:
        if not push or push_target is None:
            return {"schema_version": 1, "status": "blocked", "reason": "no_project_changes", "scope": scope}
        if bool(push_target.get("remote_branch_exists")) and int(push_target.get("local_ahead", 0)) == 0:
            return {
                "schema_version": 1,
                "status": "already_published",
                "project_id": project_id,
                "push": push_target,
                "scope": scope,
            }
        publish = push_current_branch(
            remote=str(push_target["remote"]),
            branch=str(push_target["branch"]),
        )
        return {
            "schema_version": 1,
            "status": "published" if publish.get("status") == "ok" else "blocked",
            "reason": None if publish.get("status") == "ok" else publish.get("reason", "git_push_failed"),
            "project_id": project_id,
            "commit": publish.get("head", ""),
            "committed_now": False,
            "push": publish,
            "scope": scope,
        }
    staged_before = _git_path_list(["diff", "--cached", "--name-only"])
    selected_set = set(project_paths)
    unexpected_staged_paths = [path for path in staged_before if path not in selected_set]
    if unexpected_staged_paths:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "git_staged_unexpected_paths",
            "project_id": project_id,
            "staged_unexpected_paths": unexpected_staged_paths,
            "scope": scope,
        }
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
    staged_check = _run_git(["diff", "--cached", "--check"])
    if staged_check.returncode != 0:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "git_staged_diff_check_failed",
            "detail": _git_output(staged_check),
            "scope": scope,
        }
    commit = _run_git(["commit", "-m", message])
    if commit.returncode != 0:
        return {"schema_version": 1, "status": "blocked", "reason": "git_commit_failed", "stdout": commit.stdout, "stderr": commit.stderr}
    sha = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    if push and push_target is not None:
        publish = push_current_branch(
            remote=str(push_target["remote"]),
            branch=str(push_target["branch"]),
        )
        if publish.get("status") != "ok":
            return {
                "schema_version": 1,
                "status": "committed_not_pushed",
                "reason": publish.get("reason", "git_push_failed"),
                "project_id": project_id,
                "commit": sha,
                "committed_now": True,
                "push": publish,
                "scope": scope,
            }
        return {
            "schema_version": 1,
            "status": "published",
            "project_id": project_id,
            "commit": sha,
            "committed_now": True,
            "push": publish,
            "scope": scope,
        }
    return {"schema_version": 1, "status": "committed", "project_id": project_id, "commit": sha, "scope": scope}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-scoped brain commit helper.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--task-summary", default="agent task")
    parser.add_argument("--verified", nargs="*", default=[])
    parser.add_argument("--baseline-dirty-paths", nargs="*", default=[])
    parser.add_argument("--expect-paths", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--remote", default="origin")
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
        push=bool(args.push),
        remote=str(args.remote or "origin"),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)
    return 0 if payload.get("status") in {"committed", "published", "already_published", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
