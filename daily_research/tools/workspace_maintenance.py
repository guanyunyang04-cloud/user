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
REPORT_ALERT_THRESHOLDS = {
    "daily_research/cache": 256 * 1024**2,
    "daily_research/output": 512 * 1024**2,
    "daily_research/execution/models": 128 * 1024**2,
    "daily_research/execution/output": 64 * 1024**2,
    "t0_project/ppo_tdx_tensorboard": 256 * 1024**2,
}
ARCHIVE_MONITORED_SOURCES = ("daily_research/cache", "daily_research/output")
HOTSPOT_PREVIEW_LIMIT = 5
DEFAULT_PRUNE_ARCHIVE_PREFIX = "daily_research/archive/cache/"

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


def clone_serialized_entry(item: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(item)
    if "keep_reasons" in cloned:
        cloned["keep_reasons"] = list(cloned["keep_reasons"])
    if "candidate_reasons" in cloned:
        cloned["candidate_reasons"] = list(cloned["candidate_reasons"])
    return cloned


def resolve_max_hot_size_bytes(rule: dict[str, Any]) -> int | None:
    raw = rule.get("max_hot_size_mb")
    if raw in {None, ""}:
        return None
    value = float(raw)
    if value <= 0:
        return None
    return int(value * 1024**2)


def summarize_top_entries(root: Path, entries: list[PathStat], limit: int = HOTSPOT_PREVIEW_LIMIT) -> list[dict[str, Any]]:
    ranked = sorted(entries, key=lambda item: (item.size_bytes, item.latest_mtime_ts, item.path.name), reverse=True)
    return [serialize_path_stat(root, item) for item in ranked[:limit]]


def classify_archive_entries(
    root: Path,
    entries: list[PathStat],
    *,
    keep_recent_count: int,
    cutoff: datetime | None,
    protect_globs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hard_kept: list[dict[str, Any]] = []
    soft_kept: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for index, item in enumerate(entries):
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        if index < keep_recent_count:
            hard_reasons.append("recent_count")
        if matches_any_glob(item.path.name, protect_globs):
            hard_reasons.append("protect_glob")
        if cutoff is not None and datetime.fromtimestamp(item.latest_mtime_ts) >= cutoff:
            soft_reasons.append("recent_days")

        serialized = serialize_path_stat(root, item)
        reasons = hard_reasons + soft_reasons
        if reasons:
            serialized["keep_reasons"] = reasons
        if hard_reasons:
            hard_kept.append(serialized)
        elif soft_reasons:
            soft_kept.append(serialized)
        else:
            candidates.append(serialized)

    return hard_kept, soft_kept, candidates


def apply_hot_budget(
    *,
    soft_kept: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_hot_size_bytes: int | None,
    total_source_size_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_candidate_bytes = sum(int(item["size_bytes"]) for item in candidates)
    hot_size_after_aged_candidates = max(total_source_size_bytes - base_candidate_bytes, 0)
    budget = {
        "enabled": max_hot_size_bytes is not None,
        "max_hot_size_bytes": int(max_hot_size_bytes or 0),
        "hot_size_after_aged_candidates_bytes": hot_size_after_aged_candidates,
        "hot_size_after_plan_bytes": hot_size_after_aged_candidates,
        "promoted_recent_count": 0,
        "promoted_recent_size_bytes": 0,
        "budget_unmet": False,
        "budget_shortfall_bytes": 0,
    }
    if max_hot_size_bytes is None:
        return list(soft_kept), list(candidates), budget

    if hot_size_after_aged_candidates <= max_hot_size_bytes:
        return list(soft_kept), list(candidates), budget

    promoted_paths: set[str] = set()
    promoted: list[dict[str, Any]] = []
    promoted_size = 0
    remaining_hot = hot_size_after_aged_candidates
    for item in sorted(soft_kept, key=lambda entry: (entry["latest_mtime"], entry["name"])):
        if remaining_hot <= max_hot_size_bytes:
            break
        promoted_item = clone_serialized_entry(item)
        promoted_item.pop("keep_reasons", None)
        promoted_item["candidate_reasons"] = ["budget_trimmed_recent_days"]
        promoted.append(promoted_item)
        promoted_paths.add(promoted_item["path"])
        promoted_size += int(promoted_item["size_bytes"])
        remaining_hot -= int(promoted_item["size_bytes"])

    kept_soft = [item for item in soft_kept if item["path"] not in promoted_paths]
    final_candidates = list(candidates) + promoted
    budget["hot_size_after_plan_bytes"] = max(remaining_hot, 0)
    budget["promoted_recent_count"] = len(promoted)
    budget["promoted_recent_size_bytes"] = promoted_size
    budget["budget_unmet"] = remaining_hot > max_hot_size_bytes
    budget["budget_shortfall_bytes"] = max(remaining_hot - max_hot_size_bytes, 0)
    return kept_soft, final_candidates, budget


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
        max_hot_size_bytes = resolve_max_hot_size_bytes(rule)
        cutoff = now - timedelta(days=keep_recent_days) if keep_recent_days > 0 else None

        hard_kept, soft_kept, candidates = classify_archive_entries(
            root,
            entries,
            keep_recent_count=keep_recent_count,
            cutoff=cutoff,
            protect_globs=protect_globs,
        )
        kept_soft, candidates, budget = apply_hot_budget(
            soft_kept=soft_kept,
            candidates=candidates,
            max_hot_size_bytes=max_hot_size_bytes,
            total_source_size_bytes=sum(item.size_bytes for item in entries),
        )
        kept = hard_kept + kept_soft

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
                "source_exists": source.exists(),
                "source_size_bytes": sum(item.size_bytes for item in entries),
                "source_entry_count": len(entries),
                "max_hot_size_bytes": int(max_hot_size_bytes or 0),
                "kept_count": len(kept),
                "candidate_count": len(candidates),
                "candidate_size_bytes": sum(item["size_bytes"] for item in candidates),
                "top_entries": summarize_top_entries(root, entries),
                "budget": budget,
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
        print(f"  source_size: {format_bytes(rule['source_size_bytes'])} across {rule['source_entry_count']} item(s)")
        if int(rule.get("max_hot_size_bytes", 0)) > 0:
            budget = rule["budget"]
            print(
                f"  max_hot_size: {format_bytes(rule['max_hot_size_bytes'])}, "
                f"hot_after_plan={format_bytes(budget['hot_size_after_plan_bytes'])}"
            )
            if budget["promoted_recent_count"]:
                print(
                    f"  budget_trimmed_recent_days: {budget['promoted_recent_count']} "
                    f"({format_bytes(budget['promoted_recent_size_bytes'])})"
                )
            if budget["budget_unmet"]:
                print(f"  budget_shortfall: {format_bytes(budget['budget_shortfall_bytes'])}")
        print(f"  candidates: {rule['candidate_count']} ({format_bytes(rule['candidate_size_bytes'])})")
        if rule["top_entries"]:
            print("  top_entries:")
            for item in rule["top_entries"]:
                kind = "dir " if item["is_dir"] else "file"
                print(
                    f"    - [{kind}] {item['path']} "
                    f"({format_bytes(item['size_bytes'])}, mtime={item['latest_mtime']})"
                )
        if not rule["candidates"]:
            continue
        preview = rule["candidates"] if limit == 0 else rule["candidates"][:limit]
        for item in preview:
            kind = "dir " if item["is_dir"] else "file"
            print(f"    - [{kind}] {item['path']} ({format_bytes(item['size_bytes'])}, mtime={item['latest_mtime']})")
        hidden = len(rule["candidates"]) - len(preview)
        if hidden > 0:
            print(f"    ... {hidden} more candidate(s)")


def build_archive_status(root: Path) -> dict[str, Any]:
    try:
        plan = build_archive_plan(root, DEFAULT_ARCHIVE_POLICY, selected_rules=())
    except SystemExit as exc:
        return {
            "available": False,
            "policy_path": str(DEFAULT_ARCHIVE_POLICY.relative_to(root)),
            "error": str(exc),
        }

    return {
        "available": True,
        "policy_path": plan["policy_path"],
        "candidate_count": plan["candidate_count"],
        "candidate_size_bytes": plan["candidate_size_bytes"],
        "rules": [
            {
                "name": rule["name"],
                "source": rule["source"],
                "archive_subdir": rule["archive_subdir"],
                "source_size_bytes": rule["source_size_bytes"],
                "candidate_count": rule["candidate_count"],
                "candidate_size_bytes": rule["candidate_size_bytes"],
                "max_hot_size_bytes": rule["max_hot_size_bytes"],
                "top_entries": rule["top_entries"],
                "budget": rule["budget"],
            }
            for rule in plan["rules"]
        ],
    }


def build_report_alerts(report: dict[str, Any]) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    directory_index = {
        item["path"]: item
        for item in report["directory_sizes"]
        if item["exists"]
    }
    archive_status = report["archive_status"]

    for raw_path, threshold_bytes in REPORT_ALERT_THRESHOLDS.items():
        item = directory_index.get(raw_path)
        if not item:
            continue
        effective_threshold_bytes = int(threshold_bytes)
        if archive_status.get("available"):
            budgeted_rules = [
                rule
                for rule in archive_status["rules"]
                if str(rule["source"]).startswith(raw_path) and int(rule.get("max_hot_size_bytes", 0)) > 0
            ]
            if budgeted_rules:
                effective_threshold_bytes = max(
                    effective_threshold_bytes,
                    sum(int(rule["max_hot_size_bytes"]) for rule in budgeted_rules),
                )
        if int(item["size_bytes"]) >= effective_threshold_bytes:
            alerts.append(
                {
                    "level": "warn",
                    "code": "hot_dir_size",
                    "path": raw_path,
                    "message": (
                        f"{raw_path} has grown to {format_bytes(int(item['size_bytes']))}, "
                        f"which exceeds the maintenance threshold {format_bytes(effective_threshold_bytes)}."
                    ),
                }
            )

    if report["pycache_dirs"] or report["pyc_files"]:
        alerts.append(
            {
                "level": "info",
                "code": "pycache_present",
                "path": ".",
                "message": (
                    f"Workspace currently contains {report['pycache_dirs']} __pycache__ directories "
                    f"and {report['pyc_files']} *.pyc files. Use the pycache cleanup target after validation."
                ),
            }
        )

    if archive_status.get("available"):
        for rule in archive_status["rules"]:
            budget = rule.get("budget", {})
            if budget.get("budget_unmet"):
                alerts.append(
                    {
                        "level": "warn",
                        "code": "archive_budget_unmet",
                        "path": rule["source"],
                        "message": (
                            f"{rule['source']} still exceeds its configured hot budget by "
                            f"{format_bytes(int(budget['budget_shortfall_bytes']))}. "
                            "Lower keep_recent_count or raise the budget if the latest hot set is intentionally larger."
                        ),
                    }
                )
    if archive_status.get("available") and int(archive_status["candidate_count"]) == 0:
        unresolved_sources: list[str] = []
        for raw_path in ARCHIVE_MONITORED_SOURCES:
            threshold_bytes = int(REPORT_ALERT_THRESHOLDS.get(raw_path, 0))
            if int(directory_index.get(raw_path, {}).get("size_bytes", 0)) < threshold_bytes:
                continue

            related_rules = [
                rule
                for rule in archive_status["rules"]
                if str(rule["source"]).startswith(raw_path)
            ]
            if not related_rules:
                unresolved_sources.append(raw_path)
                continue

            has_budget_unmet = any(bool(rule.get("budget", {}).get("budget_unmet")) for rule in related_rules)
            has_pending_candidates = any(int(rule.get("candidate_count", 0)) > 0 for rule in related_rules)
            if has_budget_unmet or has_pending_candidates:
                unresolved_sources.append(raw_path)

        if unresolved_sources:
            alerts.append(
                {
                    "level": "warn",
                    "code": "archive_coverage_gap",
                    "path": ",".join(unresolved_sources),
                    "message": (
                        "Archive dry-run currently matches 0 candidates while active hot areas remain in "
                        f"{', '.join(unresolved_sources)}. Review recency thresholds, hot-size budgets, or inspect the newest heavy artifacts manually."
                    ),
                }
            )

    return alerts


def build_report_hotspots(report: dict[str, Any]) -> list[dict[str, Any]]:
    archive_status = report["archive_status"]
    if not archive_status.get("available"):
        return []

    hotspots: list[dict[str, Any]] = []
    for rule in archive_status["rules"]:
        budget = rule.get("budget", {})
        source_size_bytes = int(rule.get("source_size_bytes", 0))
        if source_size_bytes <= 0:
            continue
        if source_size_bytes < 64 * 1024**2 and int(rule.get("candidate_count", 0)) == 0 and not budget.get("enabled"):
            continue
        hotspots.append(
            {
                "path": rule["source"],
                "source_size_bytes": source_size_bytes,
                "candidate_count": int(rule.get("candidate_count", 0)),
                "candidate_size_bytes": int(rule.get("candidate_size_bytes", 0)),
                "max_hot_size_bytes": int(rule.get("max_hot_size_bytes", 0)),
                "hot_size_after_plan_bytes": int(budget.get("hot_size_after_plan_bytes", 0)),
                "budget_unmet": bool(budget.get("budget_unmet")),
                "top_entries": list(rule.get("top_entries", [])),
            }
        )
    hotspots.sort(key=lambda item: item["source_size_bytes"], reverse=True)
    return hotspots[:HOTSPOT_PREVIEW_LIMIT]


def build_report(root: Path) -> dict:
    directory_sizes = []
    for raw in REPORT_DIRS:
        path = root / raw
        directory_sizes.append(
            {
                "path": raw,
                "exists": path.exists(),
                "size_bytes": path_size(path),
                "latest_mtime": iso_timestamp(latest_mtime(path)) if path.exists() else "",
            }
        )
    directory_sizes.sort(key=lambda item: item["size_bytes"], reverse=True)

    pycache_dirs = len(list(root.rglob("__pycache__")))
    pyc_files = len(list(root.rglob("*.pyc")))
    python_files = len(list(root.rglob("*.py")))
    archive_status = build_archive_status(root)

    report = {
        "workspace_root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "directory_sizes": directory_sizes,
        "pycache_dirs": pycache_dirs,
        "pyc_files": pyc_files,
        "python_files": python_files,
        "archive_status": archive_status,
        "cleanup_targets": {
            name: TARGET_SPECS[name]["description"]
            for name in TARGET_SPECS
        },
    }
    report["alerts"] = build_report_alerts(report)
    report["hotspots"] = build_report_hotspots(report)
    return report


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
        print(
            f"  - {item['path']}: {format_bytes(item['size_bytes'])} "
            f"(latest_mtime={item['latest_mtime']})"
        )
    print()
    print("Archive coverage:")
    archive_status = report["archive_status"]
    if not archive_status.get("available"):
        print(f"  - unavailable: {archive_status['error']}")
    else:
        print(
            f"  - policy={archive_status['policy_path']} "
            f"candidates={archive_status['candidate_count']} "
            f"size={format_bytes(archive_status['candidate_size_bytes'])}"
        )
        for rule in archive_status["rules"]:
            print(
                f"  - rule={rule['name']} source={rule['source']} "
                f"candidates={rule['candidate_count']} size={format_bytes(rule['candidate_size_bytes'])}"
            )
    if report["hotspots"]:
        print()
        print("Hotspots:")
        for hotspot in report["hotspots"]:
            summary = (
                f"  - {hotspot['path']}: {format_bytes(hotspot['source_size_bytes'])}, "
                f"archive_candidates={hotspot['candidate_count']} ({format_bytes(hotspot['candidate_size_bytes'])})"
            )
            if hotspot["max_hot_size_bytes"] > 0:
                summary += (
                    f", hot_budget={format_bytes(hotspot['max_hot_size_bytes'])}, "
                    f"hot_after_plan={format_bytes(hotspot['hot_size_after_plan_bytes'])}"
                )
            if hotspot["budget_unmet"]:
                summary += " [budget_unmet]"
            print(summary)
            for item in hotspot["top_entries"]:
                kind = "dir " if item["is_dir"] else "file"
                print(
                    f"    * [{kind}] {item['path']} "
                    f"({format_bytes(item['size_bytes'])}, mtime={item['latest_mtime']})"
                )
    if report["alerts"]:
        print()
        print("Alerts:")
        for alert in report["alerts"]:
            print(f"  - [{alert['level']}] {alert['message']}")
    print()
    print("Cleanup targets:")
    for name, description in report["cleanup_targets"].items():
        print(f"  - {name}: {description}")


def cmd_report(args: argparse.Namespace) -> int:
    report = build_report(WORKSPACE_ROOT)
    print_report(report)
    if args.json_out:
        out_path = resolve_workspace_path(args.json_out)
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
        out_path = resolve_workspace_path(args.json_out)
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
    manifest_path = (
        resolve_workspace_path(args.manifest_out)
        if args.manifest_out
        else archive_root / "manifests" / f"archive_{batch_id}.json"
    )
    write_json(manifest_path, manifest)

    print()
    print(f"Archive batch: {batch_id}")
    print(f"Moved items: {len(moved_items)}")
    print(f"Moved size: {format_bytes(moved_size_bytes)}")
    print(f"Manifest written to: {manifest_path}")
    return 0


def build_archive_prune_plan(root: Path, manifest_path: Path, selected_rules: Iterable[str]) -> dict[str, Any]:
    selected = {name for name in selected_rules if name}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule", "")).strip()
        if selected and rule not in selected:
            continue
        archived_path = str(item.get("archived_path", "")).replace("\\", "/").strip()
        if not archived_path.startswith(DEFAULT_PRUNE_ARCHIVE_PREFIX):
            continue
        target = root / archived_path
        if not target.exists():
            continue
        candidate = dict(item)
        candidate["archived_path"] = archived_path
        candidate["actual_size_bytes"] = path_size(target)
        candidates.append(candidate)
    return {
        "source_manifest": str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path),
        "source_batch_id": manifest.get("batch_id", ""),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prune_scope": DEFAULT_PRUNE_ARCHIVE_PREFIX,
        "selected_rules": sorted(selected),
        "candidate_count": len(candidates),
        "candidate_size_bytes": sum(int(item.get("actual_size_bytes", item.get("size_bytes", 0))) for item in candidates),
        "candidates": candidates,
    }


def _ensure_prune_target_is_safe(root: Path, target: Path) -> None:
    allowed_root = (root / DEFAULT_PRUNE_ARCHIVE_PREFIX).resolve()
    resolved = target.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise SystemExit(f"Refusing to prune outside {allowed_root}: {resolved}") from exc


def print_archive_prune_plan(plan: dict[str, Any], limit: int) -> None:
    print(f"Source manifest: {plan['source_manifest']}")
    print(f"Source batch: {plan['source_batch_id']}")
    print(f"Prune scope: {plan['prune_scope']}")
    print(f"Candidates: {plan['candidate_count']}")
    print(f"Candidate size: {format_bytes(plan['candidate_size_bytes'])}")
    preview = plan["candidates"] if limit == 0 else plan["candidates"][:limit]
    for item in preview:
        kind = "dir " if item.get("is_dir") else "file"
        size = int(item.get("actual_size_bytes", item.get("size_bytes", 0)))
        print(f"  - [{kind}] {item['archived_path']} ({format_bytes(size)})")
    hidden = len(plan["candidates"]) - len(preview)
    if hidden > 0:
        print(f"  ... {hidden} more candidate(s)")


def cmd_prune_archive(args: argparse.Namespace) -> int:
    manifest_path = resolve_workspace_path(args.manifest)
    selected_rules = [item.strip() for item in args.rules.split(",") if item.strip()]
    plan = build_archive_prune_plan(WORKSPACE_ROOT, manifest_path, selected_rules)
    print_archive_prune_plan(plan, args.limit)

    if args.json_out:
        out_path = resolve_workspace_path(args.json_out)
        write_json(out_path, plan)
        print()
        print(f"Archive prune plan JSON written to: {out_path}")

    if not args.apply:
        print()
        print("Dry run only. Re-run with --apply to delete the matched archived cache payloads.")
        return 0

    removed_items: list[dict[str, Any]] = []
    removed_size = 0
    for item in plan["candidates"]:
        target = WORKSPACE_ROOT / item["archived_path"]
        if not target.exists():
            continue
        _ensure_prune_target_is_safe(WORKSPACE_ROOT, target)
        size = path_size(target)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=False)
        else:
            target.unlink(missing_ok=True)
        removed = dict(item)
        removed["removed_size_bytes"] = size
        removed_items.append(removed)
        removed_size += size

    prune_manifest = {
        "source_manifest": plan["source_manifest"],
        "source_batch_id": plan["source_batch_id"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prune_scope": plan["prune_scope"],
        "removed_count": len(removed_items),
        "removed_size_bytes": removed_size,
        "items": removed_items,
    }
    manifest_out = (
        resolve_workspace_path(args.manifest_out)
        if args.manifest_out
        else WORKSPACE_ROOT / "daily_research" / "archive" / "manifests" / f"prune_{plan['source_batch_id'] or datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    write_json(manifest_out, prune_manifest)

    print()
    print(f"Pruned items: {len(removed_items)}")
    print(f"Pruned size: {format_bytes(removed_size)}")
    print(f"Prune manifest written to: {manifest_out}")
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

    prune = sub.add_parser("prune-archive", help="Delete archived cache payloads listed in an archive manifest. Dry-run by default.")
    prune.add_argument("--manifest", required=True, help="Archive manifest JSON path.")
    prune.add_argument("--rules", default="", help="Optional comma-separated cache archive rule names.")
    prune.add_argument("--limit", type=int, default=20, help="Preview at most N candidates. Use 0 for all.")
    prune.add_argument("--json-out", default="", help="Optional path for a UTF-8 prune plan.")
    prune.add_argument("--manifest-out", default="", help="Optional path for the prune manifest when --apply is used.")
    prune.add_argument("--apply", action="store_true", help="Actually delete matched archived cache payloads.")
    prune.set_defaults(func=cmd_prune_archive)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
