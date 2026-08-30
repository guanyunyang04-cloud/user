"""Recent Market Repair: commit responsibilities."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
)
from quantlab.data.qdp_v2.repair.mutation import bulk_append_active_shards_from_parquet

from .config import (
    DAILY_COLUMNS,
    DAILY_DOMAIN,
    DAILY_SCHEMA,
    INTRADAY_COLUMNS,
    INTRADAY_SCHEMA,
)
from .state import (
    _utc_now,
)


def _commit_bundles(
    state: dict[str, Any],
    *,
    runtime: Path,
    domain: str,
    reason: str,
    workspace: Path,
) -> dict[str, Any]:
    bundle_paths = [
        Path(item["bundle_path"]) for item in state.get("buckets", {}).values() if str(item.get("bundle_path", ""))
    ]
    if bundle_paths:
        result = bulk_append_active_shards_from_parquet(
            domain,
            bundle_paths,
            reason,
            workspace_root=workspace,
        )
    else:
        result = {"status": "nothing_to_append", "row_count": 0}
    state.update(
        {
            "status": "applied",
            "commit": result,
            "updated_at": _utc_now(),
        }
    )
    atomic_write_json(runtime / "state.json", state)
    return state


def _write_final_parquet(frame: pd.DataFrame, path: Path, *, domain: str) -> None:
    columns = DAILY_COLUMNS if domain == DAILY_DOMAIN else INTRADAY_COLUMNS
    schema = DAILY_SCHEMA if domain == DAILY_DOMAIN else INTRADAY_SCHEMA
    ordered = frame.loc[:, columns].copy()
    table = pa.Table.from_pandas(ordered, schema=schema, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            row_group_size=100_000,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
