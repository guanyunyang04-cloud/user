"""Fingerprint records and validation shared by minute research stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.core.io import sha256_file

_FINGERPRINT_CACHE: dict[tuple[str, int, int, int, int, int, str], bool] = {}


def artifact_record(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    metadata = pq.ParquetFile(target).metadata
    return {
        "path": str(target),
        "size": int(target.stat().st_size),
        "sha256": sha256_file(target),
        "row_count": int(metadata.num_rows),
        "row_group_count": int(metadata.num_row_groups),
    }


def artifact_matches(
    path: str | Path,
    expected: Any,
    *,
    base_directory: str | Path | None = None,
) -> bool:
    """Validate path, Parquet metadata, and content digest against a record."""

    if not isinstance(expected, dict):
        return False
    try:
        target = Path(path).resolve()
        if not target.is_file():
            return False
        expected_path = Path(str(expected.get("path", "")))
        if not expected_path.is_absolute() and base_directory is not None:
            expected_path = Path(base_directory) / expected_path
        expected_path = expected_path.resolve()
        if expected_path != target:
            return False
        stat = target.stat()
        expected_size = expected.get("size")
        expected_sha = expected.get("sha256")
        expected_rows = expected.get("row_count")
        expected_groups = expected.get("row_group_count")
        if (
            type(expected_size) is not int
            or type(expected_rows) is not int
            or type(expected_groups) is not int
            or not isinstance(expected_sha, str)
            or expected_size != int(stat.st_size)
        ):
            return False
        cache_key = (
            str(target),
            int(stat.st_size),
            int(getattr(stat, "st_mtime_ns", 0)),
            int(getattr(stat, "st_ctime_ns", 0)),
            expected_rows,
            expected_groups,
            expected_sha,
        )
        if _FINGERPRINT_CACHE.get(cache_key) is True:
            return True
        metadata = pq.ParquetFile(target).metadata
        if (
            int(metadata.num_rows) != expected_rows
            or int(metadata.num_row_groups) != expected_groups
            or sha256_file(target) != expected_sha
        ):
            return False
        _FINGERPRINT_CACHE[cache_key] = True
        return True
    except Exception:
        return False


__all__ = ["artifact_matches", "artifact_record"]
