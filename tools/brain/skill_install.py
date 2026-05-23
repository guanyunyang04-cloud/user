from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SKILLS_ROOT = WORKSPACE_ROOT / "brain/skills"
DEFAULT_CODEX_SKILLS_ROOT = Path.home() / ".codex" / "skills"


def _skill_dirs() -> list[Path]:
    if not PROJECT_SKILLS_ROOT.exists():
        return []
    return sorted(path for path in PROJECT_SKILLS_ROOT.iterdir() if (path / "SKILL.md").exists())


def build_sync_plan(target_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(target_root) if target_root else DEFAULT_CODEX_SKILLS_ROOT
    skills = []
    for source in _skill_dirs():
        target = root / source.name
        skills.append(
            {
                "skill": source.name,
                "source": str(source),
                "target": str(target),
                "target_exists": target.exists(),
            }
        )
    return {
        "status": "ok",
        "project_skills_root": str(PROJECT_SKILLS_ROOT),
        "target_root": str(root),
        "skill_count": len(skills),
        "skills": skills,
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
        shutil.copytree(source, target)
        installed.append(str(target))
    return {**plan, "installed": installed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync repo-canonical workspace brain skills into the Codex skills directory.")
    parser.add_argument("--target-root", default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--install", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_root = str(args.target_root or "").strip() or None
    payload = build_sync_plan(target_root) if args.dry_run else install_project_skills(target_root)
    payload["dry_run"] = bool(args.dry_run)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

