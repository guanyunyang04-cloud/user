"""Runtime Archive: discovery responsibilities."""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from quantlab.data.core.paths import qdp_paths

from .config import (
    DEFAULT_SELECTIONS,
    YEAR_PATTERNS,
    ArchiveUnit,
    RuntimeArchiveError,
)


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime_root(workspace: Path) -> Path:
    return (qdp_paths(workspace).data_dir / "qdp_runtime").resolve()


def _archive_root(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime_archives").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not text:
        raise RuntimeArchiveError("runtime_archive_empty_safe_name")
    return text


def _unit_directory(workspace: Path, unit: ArchiveUnit) -> Path:
    return (
        _archive_root(workspace) / _safe_name(unit.workflow) / _safe_name(unit.domain) / f"year={_safe_name(unit.year)}"
    ).resolve()


def _zstd_executable() -> str:
    candidate = shutil.which("zstd")
    if candidate:
        return candidate
    bundled = Path(sys.executable).resolve().parent / "Library" / "bin" / "zstd.exe"
    if bundled.is_file():
        return str(bundled)
    raise RuntimeArchiveError("zstd_executable_missing")


def _domain_for_file(selection: str, relative: Path) -> str:
    first = relative.parts[0] if len(relative.parts) > 1 else ""
    if selection == "symbols" or re.fullmatch(r"\d{6}_(?:SH|SZ)", first, re.IGNORECASE):
        return selection
    if first and not re.search(r"(?:year|date|task)=", first, re.IGNORECASE):
        return f"{selection}__{first}"
    return selection


def _year_for_file(relative: Path) -> str:
    for part in relative.parts:
        for pattern in YEAR_PATTERNS:
            match = pattern.search(part)
            if match:
                return match.group(1)
    return "multi"


def discover_units(
    *,
    workspace_root: str | Path | None = None,
    workflows: Sequence[str] = (),
) -> list[ArchiveUnit]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    selected = tuple(workflows) or tuple(DEFAULT_SELECTIONS)
    unknown = sorted(set(selected).difference(DEFAULT_SELECTIONS))
    if unknown:
        raise RuntimeArchiveError(f"unknown_archive_workflow:{','.join(unknown)}")
    grouped: dict[tuple[str, str, str], list[Path]] = {}
    roots: dict[str, Path] = {}
    for workflow in selected:
        workflow_root = _within(runtime / workflow, runtime)
        roots[workflow] = workflow_root
        if not workflow_root.is_dir():
            continue
        for selection in DEFAULT_SELECTIONS[workflow]:
            source_root = _within(workflow_root / selection, workflow_root)
            if not source_root.is_dir():
                continue
            for path in source_root.rglob("*"):
                if not path.is_file():
                    continue
                relative_to_selection = path.relative_to(source_root)
                domain = _domain_for_file(selection, relative_to_selection)
                year = _year_for_file(relative_to_selection)
                grouped.setdefault((workflow, domain, year), []).append(path.resolve())
    return [
        ArchiveUnit(
            workflow=workflow,
            domain=domain,
            year=year,
            workflow_root=roots[workflow],
            files=tuple(sorted(files)),
        )
        for (workflow, domain, year), files in sorted(grouped.items())
    ]
