"""Pit Metadata Repair: context responsibilities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from quantlab.core.io import sha256_file
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    REPAIR_ID,
    PitMetadataRepairError,
)

_sha256 = sha256_file


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _root(workspace: Path) -> Path:
    return qdp_v2_root(workspace).resolve()


def _active_paths(root: Path, domain: str) -> tuple[Path, ...]:
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    manifest_path = dataset_manifest_for_id(root, datasets[domain], domain)
    if manifest_path is None:
        raise PitMetadataRepairError(f"active_manifest_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = tuple(resolve_manifest_path(item.path, root=root) for item in manifest.shards)
    if not paths or any(not path.is_file() for path in paths):
        raise PitMetadataRepairError(f"active_shards_missing:{domain}")
    return paths


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / REPAIR_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _schema_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _parquet_count(path: Path, predicate: str = "TRUE") -> int:
    with open_guarded_duckdb(temp_directory=path.parent / "_count_spill", threads=2) as con:
        return int(
            con.execute(
                f"SELECT count(*) FROM read_parquet(?) WHERE {predicate}",
                [str(path)],
            ).fetchone()[0]
        )


def _assert_untargeted_columns_equal(
    old_path: Path,
    new_path: Path,
    columns: Sequence[str],
    changed_columns: Sequence[str],
) -> None:
    stable = [column for column in columns if column not in set(changed_columns)]
    if not stable:
        return
    projection = ",".join(_sql_identifier(column) for column in stable)
    with open_guarded_duckdb(temp_directory=old_path.parent / "_repair_compare_spill", threads=2) as con:
        left_only = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT {projection} FROM read_parquet(?) "
                f"EXCEPT ALL SELECT {projection} FROM read_parquet(?))",
                [str(old_path), str(new_path)],
            ).fetchone()[0]
        )
        right_only = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT {projection} FROM read_parquet(?) "
                f"EXCEPT ALL SELECT {projection} FROM read_parquet(?))",
                [str(new_path), str(old_path)],
            ).fetchone()[0]
        )
    if left_only or right_only:
        raise PitMetadataRepairError(f"repair_untargeted_columns_changed:{old_path.name}:{left_only}:{right_only}")
