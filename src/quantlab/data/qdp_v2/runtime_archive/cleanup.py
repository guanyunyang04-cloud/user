"""Runtime Archive: cleanup responsibilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.core.io import sha256_file as _sha256

from .config import (
    ArchiveUnit,
    RuntimeArchiveError,
)
from .discovery import (
    _within,
)


def _remove_empty_parents(paths: Sequence[Path], *, stop: Path) -> None:
    candidates: set[Path] = set()
    for path in paths:
        parent = path.parent
        while parent != stop and stop in parent.parents:
            candidates.add(parent)
            parent = parent.parent
    for directory in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _delete_exact_sources(unit: ArchiveUnit, ledger: pd.DataFrame) -> dict[str, Any]:
    deleted: list[Path] = []
    already_missing = 0
    for row in ledger.to_dict("records"):
        path = _within(unit.workflow_root / str(row["relative_path"]), unit.workflow_root)
        if not path.exists():
            already_missing += 1
            continue
        if not path.is_file():
            raise RuntimeArchiveError(f"runtime_archive_delete_target_not_file:{path}")
        if path.stat().st_size != int(row["file_size"]) or _sha256(path) != str(row["sha256"]):
            raise RuntimeArchiveError(f"runtime_archive_delete_source_changed:{path}")
        path.unlink()
        deleted.append(path)
    _remove_empty_parents(deleted, stop=unit.workflow_root)
    return {
        "deleted_file_count": len(deleted),
        "already_missing_file_count": already_missing,
        "deleted_logical_bytes": sum(
            int(row["file_size"])
            for row in ledger.to_dict("records")
            if not (unit.workflow_root / str(row["relative_path"])).exists()
        ),
    }
