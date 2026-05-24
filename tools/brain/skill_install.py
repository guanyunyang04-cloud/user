from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SKILLS_ROOT = WORKSPACE_ROOT / "brain/skills"
DEFAULT_CODEX_SKILLS_ROOT = Path.home() / ".codex" / "skills"
IGNORED_NAMES = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _skill_dirs() -> list[Path]:
    if not PROJECT_SKILLS_ROOT.exists():
        return []
    return sorted(path for path in PROJECT_SKILLS_ROOT.iterdir() if (path / "SKILL.md").exists())


def _ignore_generated_files(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_NAMES or Path(name).suffix in IGNORED_SUFFIXES}


def _iter_hashable_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        rel_parts = item.relative_to(path).parts
        if any(part in IGNORED_NAMES for part in rel_parts):
            continue
        if item.suffix in IGNORED_SUFFIXES:
            continue
        files.append(item)
    return sorted(files)


def _hash_tree(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    for item in _iter_hashable_files(path):
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_sync_plan(target_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(target_root) if target_root else DEFAULT_CODEX_SKILLS_ROOT
    skills = []
    for source in _skill_dirs():
        target = root / source.name
        source_sha = _hash_tree(source)
        target_sha = _hash_tree(target)
        skills.append(
            {
                "skill": source.name,
                "source": str(source),
                "target": str(target),
                "target_exists": target.exists(),
                "source_sha256": source_sha,
                "target_sha256": target_sha,
                "in_sync": bool(target.exists() and source_sha and source_sha == target_sha),
            }
        )
    return {
        "status": "ok",
        "project_skills_root": str(PROJECT_SKILLS_ROOT),
        "target_root": str(root),
        "skill_count": len(skills),
        "skills": skills,
        "all_in_sync": all(bool(item["in_sync"]) for item in skills),
    }


def install_project_skills(target_root: str | Path | None = None) -> dict[str, Any]:
    plan = build_sync_plan(target_root)
    installed: list[str] = []
    for item in plan["skills"]:
        source = Path(item["source"])
        target = Path(item["target"])
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, ignore=_ignore_generated_files)
        installed.append(str(target))
    post_plan = build_sync_plan(target_root)
    return {**post_plan, "installed": installed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync repo-canonical workspace brain skills into the Codex skills directory.")
    parser.add_argument("--target-root", default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--install", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_root = str(args.target_root or "").strip() or None
    payload = build_sync_plan(target_root) if (args.dry_run or args.check) else install_project_skills(target_root)
    payload["check"] = bool(args.check)
    payload["dry_run"] = bool(args.dry_run)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
