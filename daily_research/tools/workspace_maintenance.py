from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_POLICY = WORKSPACE_ROOT / "daily_research" / "archive_policy.json"
REPORT_DIRS = [
    "daily_research",
    "daily_research/cache",
    "daily_research/output",
    "daily_research/archive",
    "daily_research/execution/output",
    "daily_research/execution/models",
    "t0_project",
    "t0_project/models",
    "t0_project/backtest_output",
    "t0_project/logs",
    "t0_project/ppo_tdx_tensorboard",
]

TARGET_SPECS = {
    "pycache": {
        "description": "Python bytecode caches (__pycache__ and *.pyc)",
        "include": lambda root: list(root.rglob("__pycache__")) + list(root.rglob("*.pyc")),
    },
    "tensorboard": {
        "description": "TensorBoard event logs under t0_project/ppo_tdx_tensorboard",
        "include": lambda root: [root / "t0_project" / "ppo_tdx_tensorboard"],
    },
    "daily_cache": {
        "description": "daily_research/cache generated research cache",
        "include": lambda root: [root / "daily_research" / "cache"],
    },
    "daily_output": {
        "description": "daily_research/output experiment outputs",
        "include": lambda root: [root / "daily_research" / "output"],
    },
    "execution_output": {
        "description": "daily_research/execution/output generated plans and logs",
        "include": lambda root: [root / "daily_research" / "execution" / "output"],
    },
    "t0_backtest_output": {
        "description": "t0_project/backtest_output generated backtest artifacts",
        "include": lambda root: [root / "t0_project" / "backtest_output"],
    },
    "t0_logs": {
        "description": "t0_project/logs generated runtime logs",
        "include": lambda root: [root / "t0_project" / "logs"],
    },
}


@dataclass
class PathStat:
    path: Path
    size_bytes: int
    is_dir: bool
    latest_mtime_ts: float


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(value, 0))
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.2f} {unit}"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def latest_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_mtime
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        latest = max(latest, child.stat().st_mtime)
    return latest


def iso_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_workspace_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def iter_existing_targets(root: Path, target_names: Iterable[str], older_than_days: int) -> list[PathStat]:
    cutoff = None
    if older_than_days > 0:
        cutoff = datetime.now() - timedelta(days=older_than_days)

    results: list[PathStat] = []
    seen: set[Path] = set()
    for target_name in target_names:
        spec = TARGET_SPECS[target_name]
        for path in spec["include"](root):
            if not path.exists():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if cutoff is not None and datetime.fromtimestamp(latest_mtime(path)) >= cutoff:
                continue
            seen.add(resolved)
            results.append(
                PathStat(
                    path=path,
                    size_bytes=path_size(path),
                    is_dir=path.is_dir(),
                    latest_mtime_ts=latest_mtime(path),
                )
            )
    return sorted(results, key=lambda item: (not item.is_dir, str(item.path)))


def matches_any_glob(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def iter_direct_children(source: Path, path_kind: str) -> list[PathStat]:
    if not source.exists():
        return []
    results: list[PathStat] = []
    for child in source.iterdir():
        if path_kind == "dir" and not child.is_dir():
            continue
        if path_kind == "file" and not child.is_file():
            continue
        results.append(
            PathStat(
                path=child,
                size_bytes=path_size(child),
                is_dir=child.is_dir(),
                latest_mtime_ts=latest_mtime(child),
            )
        )
    return sorted(results, key=lambda item: (item.latest_mtime_ts, item.path.name), reverse=True)


def load_archive_policy(policy_path: Path) -> dict[str, Any]:
    if not policy_path.exists():
        raise SystemExit(f"Archive policy not found: {policy_path}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if "archive_root" not in policy or "rules" not in policy:
        raise SystemExit(f"Invalid archive policy: {policy_path}")
    return policy


def serialize_path_stat(root: Path, item: PathStat) -> dict[str, Any]:
    return {
        "path": str(item.path.relative_to(root)),
        "name": item.path.name,
        "size_bytes": item.size_bytes,
        "is_dir": item.is_dir,
        "latest_mtime": iso_timestamp(item.latest_mtime_ts),
    }


def build_archive_plan(root: Path, policy_path: Path, selected_rules: Iterable[str]) -> dict[str, Any]:
    policy = load_archive_policy(policy_path)
    selected = {name for name in selected_rules if name}

    rules_payload = []
    candidate_count = 0
    candidate_size_bytes = 0
    now = datetime.now()
    for rule in policy["rules"]:
        if selected and rule["name"] not in selected:
            continue

        source = root / rule["source"]
        entries = iter_direct_children(source, rule.get("path_kind", "any"))
        keep_recent_count = int(rule.get("keep_recent_count", 0))
        keep_recent_days = int(rule.get("keep_recent_days", 0))
        protect_globs = list(rule.get("protect_globs", []))
        cutoff = now - timedelta(days=keep_recent_days) if keep_recent_days > 0 else None

        kept = []
        candidates = []
        for index, item in enumerate(entries):
            reasons = []
            if index < keep_recent_count:
                reasons.append("recent_count")
            if cutoff is not None and datetime.fromtimestamp(item.latest_mtime_ts) >= cutoff:
                reasons.append("recent_days")
            if matches_any_glob(item.path.name, protect_globs):
                reasons.append("protect_glob")

            serialized = serialize_path_stat(root, item)
            if reasons:
                serialized["keep_reasons"] = reasons
                kept.append(serialized)
            else:
                candidates.append(serialized)

        candidate_count += len(candidates)
        candidate_size_bytes += sum(item["size_bytes"] for item in candidates)
        rules_payload.append(
            {
                "name": rule["name"],
                "description": rule["description"],
                "source": rule["source"],
                "archive_subdir": rule["archive_subdir"],
                "path_kind": rule.get("path_kind", "any"),
                "keep_recent_count": keep_recent_count,
                "keep_recent_days": keep_recent_days,
                "protect_globs": protect_globs,
                "kept_count": len(kept),
                "candidate_count": len(candidates),
                "candidate_size_bytes": sum(item["size_bytes"] for item in candidates),
                "kept": kept,
                "candidates": candidates,
            }
        )

    return {
        "policy_path": str(policy_path.relative_to(root)),
        "policy_version": policy.get("version", 0),
        "archive_root": policy["archive_root"],
        "notes": list(policy.get("notes", [])),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_count": candidate_count,
        "candidate_size_bytes": candidate_size_bytes,
        "rules": rules_payload,
    }


def print_archive_plan(plan: dict[str, Any], limit: int) -> None:
    print(f"Policy: {plan['policy_path']}")
    print(f"Policy version: {plan['policy_version']}")
    print(f"Archive root: {plan['archive_root']}")
    print(f"Generated at: {plan['generated_at']}")
    print(f"Archive candidates: {plan['candidate_count']}")
    print(f"Archive size: {format_bytes(plan['candidate_size_bytes'])}")
    if plan["notes"]:
        print()
        print("Policy notes:")
        for note in plan["notes"]:
            print(f"  - {note}")

    for rule in plan["rules"]:
        print()
        print(f"[{rule['name']}] {rule['description']}")
        print(f"  source: {rule['source']}")
        print(f"  archive: {rule['archive_subdir']}")
        print(f"  keep_recent_count={rule['keep_recent_count']}, keep_recent_days={rule['keep_recent_days']}")
        print(f"  candidates: {rule['candidate_count']} ({format_bytes(rule['candidate_size_bytes'])})")
        if not rule["candidates"]:
            continue
        preview = rule["candidates"] if limit == 0 else rule["candidates"][:limit]
        for item in preview:
            kind = "dir " if item["is_dir"] else "file"
            print(f"    - [{kind}] {item['path']} ({format_bytes(item['size_bytes'])}, mtime={item['latest_mtime']})")
        hidden = len(rule["candidates"]) - len(preview)
        if hidden > 0:
            print(f"    ... {hidden} more candidate(s)")


def build_report(root: Path) -> dict:
    directory_sizes = []
    for raw in REPORT_DIRS:
        path = root / raw
        directory_sizes.append(
            {
                "path": raw,
                "exists": path.exists(),
                "size_bytes": path_size(path),
            }
        )
    directory_sizes.sort(key=lambda item: item["size_bytes"], reverse=True)

    pycache_dirs = len(list(root.rglob("__pycache__")))
    pyc_files = len(list(root.rglob("*.pyc")))
    python_files = len(list(root.rglob("*.py")))

    return {
        "workspace_root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "directory_sizes": directory_sizes,
        "pycache_dirs": pycache_dirs,
        "pyc_files": pyc_files,
        "python_files": python_files,
        "cleanup_targets": {
            name: TARGET_SPECS[name]["description"]
            for name in TARGET_SPECS
        },
    }


def print_report(report: dict) -> None:
    print(f"Workspace: {report['workspace_root']}")
    print(f"Generated at: {report['generated_at']}")
    print(f"Python files: {report['python_files']}")
    print(f"__pycache__ dirs: {report['pycache_dirs']}")
    print(f"*.pyc files: {report['pyc_files']}")
    print()
    print("Directory sizes:")
    for item in report["directory_sizes"]:
        if not item["exists"]:
            continue
        print(f"  - {item['path']}: {format_bytes(item['size_bytes'])}")
    print()
    print("Cleanup targets:")
    for name, description in report["cleanup_targets"].items():
        print(f"  - {name}: {description}")


def cmd_report(args: argparse.Namespace) -> int:
    report = build_report(WORKSPACE_ROOT)
    print_report(report)
    if args.json_out:
        out_path = Path(args.json_out)
        write_json(out_path, report)
        print()
        print(f"JSON report written to: {out_path}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    unknown = [name for name in targets if name not in TARGET_SPECS]
    if unknown:
        raise SystemExit(f"Unknown cleanup target(s): {', '.join(unknown)}")

    items = iter_existing_targets(WORKSPACE_ROOT, targets, args.older_than_days)
    print(f"Targets: {', '.join(targets)}")
    print(f"Apply mode: {args.apply}")
    print(f"Matched items: {len(items)}")
    total_bytes = sum(item.size_bytes for item in items)
    print(f"Matched size: {format_bytes(total_bytes)}")
    for item in items:
        kind = "dir " if item.is_dir else "file"
        print(f"  - [{kind}] {item.path} ({format_bytes(item.size_bytes)})")

    if not args.apply:
        print()
        print("Dry run only. Re-run with --apply to delete the matched items.")
        return 0

    for item in items:
        if item.is_dir:
            shutil.rmtree(item.path, ignore_errors=False)
        else:
            item.path.unlink(missing_ok=True)

    print()
    print("Cleanup completed.")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    policy_path = resolve_workspace_path(args.policy)
    selected_rules = [item.strip() for item in args.rules.split(",") if item.strip()]
    plan = build_archive_plan(WORKSPACE_ROOT, policy_path, selected_rules)
    print_archive_plan(plan, args.limit)

    if args.json_out:
        out_path = Path(args.json_out)
        write_json(out_path, plan)
        print()
        print(f"Archive plan JSON written to: {out_path}")

    if not args.apply:
        print()
        print("Dry run only. Re-run with --apply to move the matched items into the archive area.")
        return 0

    if plan["candidate_count"] == 0:
        print()
        print("No archive candidates matched the current policy.")
        return 0

    batch_id = args.batch_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = WORKSPACE_ROOT / plan["archive_root"]
    moved_items = []
    moved_size_bytes = 0

    for rule in plan["rules"]:
        candidates = rule["candidates"]
        if not candidates:
            continue
        destination_root = archive_root / rule["archive_subdir"] / batch_id
        destination_root.mkdir(parents=True, exist_ok=True)
        for item in candidates:
            source_path = WORKSPACE_ROOT / item["path"]
            destination_path = destination_root / source_path.name
            if destination_path.exists():
                raise SystemExit(f"Archive destination already exists: {destination_path}")
            shutil.move(str(source_path), str(destination_path))
            moved_items.append(
                {
                    "rule": rule["name"],
                    "source_path": item["path"],
                    "archived_path": str(destination_path.relative_to(WORKSPACE_ROOT)),
                    "size_bytes": item["size_bytes"],
                    "is_dir": item["is_dir"],
                    "latest_mtime": item["latest_mtime"],
                }
            )
            moved_size_bytes += int(item["size_bytes"])

    manifest = {
        "batch_id": batch_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy_path": plan["policy_path"],
        "policy_version": plan["policy_version"],
        "archive_root": plan["archive_root"],
        "selected_rules": selected_rules,
        "moved_count": len(moved_items),
        "moved_size_bytes": moved_size_bytes,
        "items": moved_items,
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else archive_root / "manifests" / f"archive_{batch_id}.json"
    write_json(manifest_path, manifest)

    print()
    print(f"Archive batch: {batch_id}")
    print(f"Moved items: {len(moved_items)}")
    print(f"Moved size: {format_bytes(moved_size_bytes)}")
    print(f"Manifest written to: {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workspace report and cleanup helper for daily_research + t0_project.")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="Print size summary and cleanup recommendations.")
    report.add_argument("--json-out", default="", help="Optional path for a UTF-8 JSON report.")
    report.set_defaults(func=cmd_report)

    clean = sub.add_parser("clean", help="Delete generated artifacts. Dry-run by default.")
    clean.add_argument("--targets", default="pycache", help="Comma-separated cleanup targets.")
    clean.add_argument("--older-than-days", type=int, default=0, help="Only delete items older than N days.")
    clean.add_argument("--apply", action="store_true", help="Actually delete matched items.")
    clean.set_defaults(func=cmd_clean)

    archive = sub.add_parser("archive", help="Plan or move aged cache/output artifacts into the cold archive area.")
    archive.add_argument(
        "--policy",
        default=str(DEFAULT_ARCHIVE_POLICY.relative_to(WORKSPACE_ROOT)),
        help="Archive policy JSON path.",
    )
    archive.add_argument("--rules", default="", help="Optional comma-separated archive rule names.")
    archive.add_argument("--limit", type=int, default=20, help="Preview at most N candidates per rule. Use 0 for all.")
    archive.add_argument("--json-out", default="", help="Optional path for a UTF-8 JSON archive plan.")
    archive.add_argument("--manifest-out", default="", help="Optional path for the archive manifest when --apply is used.")
    archive.add_argument("--batch-name", default="", help="Optional archive batch id; defaults to a timestamp.")
    archive.add_argument("--apply", action="store_true", help="Actually move matched items into the archive area.")
    archive.set_defaults(func=cmd_archive)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
