"""Recent Market Repair: state responsibilities."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
)

from .config import (
    BUCKET_COUNT,
    DEFAULT_WORKERS,
    REPAIR_VERSION,
    RecentMarketRepairError,
    _DomainInput,
)


def _load_or_initialize_state(
    runtime: Path,
    *,
    kind: str,
    start_date: str,
    end_date: str,
    task_count: int,
    input_hash: str = "",
    resume: bool,
) -> dict[str, Any]:
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "state.json"
    if state_path.is_file():
        if not resume:
            raise RecentMarketRepairError("recent_repair_existing_job_requires_resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = (kind, start_date, end_date)
        observed = (
            str(state.get("kind", "")),
            str(state.get("start_date", "")),
            str(state.get("end_date", "")),
        )
        if observed != expected:
            raise RecentMarketRepairError(f"recent_repair_state_parameters_mismatch:{observed!r}!={expected!r}")
        if input_hash and str(state.get("input_hash", "")) not in {"", input_hash}:
            raise RecentMarketRepairError("recent_repair_state_input_hash_mismatch")
        state["task_count"] = int(task_count)
        if input_hash:
            state["input_hash"] = input_hash
        return state
    state = {
        "format_version": REPAIR_VERSION,
        "kind": kind,
        "status": "pending",
        "start_date": start_date,
        "end_date": end_date,
        "task_count": int(task_count),
        "input_hash": str(input_hash),
        "bucket_count": BUCKET_COUNT,
        "workers": DEFAULT_WORKERS,
        "buckets": {},
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    atomic_write_json(state_path, state)
    return state


def _pairs_sha256(pairs: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for symbol, trade_date in pairs:
        digest.update(f"{symbol}\t{trade_date}\n".encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reused_bucket_is_valid(
    record: Mapping[str, Any],
    *,
    runtime: Path,
    wanted: Mapping[str, Sequence[str]],
) -> bool:
    if list(record.get("requested_pairs", []) or []) != _requested_pairs(wanted):
        return False
    text = str(record.get("bundle_path", ""))
    if not text:
        return int(record.get("row_count", 0)) == 0
    path = Path(text).resolve()
    if not path.is_file() or runtime.resolve() not in path.parents:
        return False
    try:
        parquet = pq.ParquetFile(path)
    except Exception:  # noqa: BLE001 - malformed parquet is a rejected bundle
        return False
    return int(parquet.metadata.num_rows) == int(record.get("row_count", 0)) and path.stat().st_size == int(
        record.get("file_size", -1)
    )


def _unresolved_mapping(
    unresolved: Mapping[tuple[str, str], str],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for symbol, trade_date in unresolved:
        result.setdefault(symbol, []).append(trade_date)
    return {symbol: tuple(sorted(set(dates))) for symbol, dates in result.items()}


def _split_symbols(symbols: Sequence[str], workers: int) -> tuple[tuple[str, ...], ...]:
    count = min(max(1, int(workers)), len(symbols))
    return tuple(tuple(symbols[index::count]) for index in range(count))


def _stable_bucket(symbol: str) -> int:
    code = str(symbol).split(".", 1)[0]
    if code.isdigit():
        return int(code) % BUCKET_COUNT
    return sum((index + 1) * ord(character) for index, character in enumerate(code)) % BUCKET_COUNT


def _requested_pairs(wanted: Mapping[str, Sequence[str]]) -> list[list[str]]:
    return [
        [str(symbol), str(trade_date)] for symbol, dates in sorted(wanted.items()) for trade_date in sorted(set(dates))
    ]


def _run_directory(kind: str, start_date: str, end_date: str) -> str:
    return f"{kind}__{start_date.replace('-', '')}_{end_date.replace('-', '')}"


def _path_texts(value: _DomainInput) -> list[str]:
    return [str(item) for item in value.paths]


def _runtime_root(workspace: Path) -> Path:
    root = (qdp_paths(workspace).data_dir / "qdp_runtime" / "recent_market_repair").resolve()
    if workspace.resolve() not in root.parents:
        raise RecentMarketRepairError(f"recent_repair_runtime_outside_workspace:{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _configure_workspace_runtime(workspace: Path) -> None:
    paths = qdp_paths(workspace)
    os.environ["QDP_DATA_ROOT"] = str(paths.data_dir)
    runtime = (paths.data_dir / "qdp_runtime").resolve()
    os.environ["QDP_RUNTIME_ROOT"] = str(runtime)
    mootdx_state = runtime / "mootdx" / "last_good_5m.json"
    os.environ.setdefault("QDP_MOOTDX_LAST_GOOD_PATH", str(mootdx_state))


def _date_text(value: str) -> str:
    return pd.Timestamp(str(value)).strftime("%Y-%m-%d")


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").floor("s").isoformat()
