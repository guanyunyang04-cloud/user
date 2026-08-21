"""Database audit common checks."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from quantlab.data.core.paths import qdp_paths


def _audit_temp_directory(workspace: Path):
    runtime = qdp_paths(workspace).runtime_dir
    runtime.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="audit_spill_", dir=str(runtime))


def _path_texts(paths: Iterable[Path]) -> list[str]:
    return [str(item) for item in paths]


def _q(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _finding(
    severity: str,
    category: str,
    domain: str,
    code: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "domain": domain,
        "code": code,
        "evidence": dict(evidence),
    }


def _format(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"status: {payload.get('status', '')}",
            f"dataset_count: {payload.get('dataset_count', 0)}",
            f"finding_count: {payload.get('finding_count', 0)}",
        ]
    )
