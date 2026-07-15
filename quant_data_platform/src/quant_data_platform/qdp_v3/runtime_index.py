from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from quant_data_platform.qdp_v3.manifest import sha256_file, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.storage import atomic_write_parquet


RUNTIME_INDEX_SCHEMA_VERSION = 3
_SECRET_KEY_FRAGMENTS = ("token", "secret", "authorization", "api_key", "apikey")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_RAW_PARTITION_COLUMNS = (
    "domain",
    "partition_key",
    "provider",
    "request_range",
    "row_count",
    "response_sha256",
    "source_content_sha256",
    "bundle_path",
    "row_group",
    "status",
    "bundle_sha256",
    "source_schema_json",
    "created_at",
)
_RAW_PARTITION_KEY = (
    "domain",
    "partition_key",
    "source_content_sha256",
    "bundle_path",
    "row_group",
)
_RAW_RECEIPT_COLUMNS = (
    "domain",
    "partition_key",
    "provider",
    "request_range",
    "row_count",
    "response_sha256",
    "source_content_sha256",
    "quality_tier",
    "receipt_json",
    "created_at",
)
_RAW_RECEIPT_KEY = ("domain", "partition_key", "source_content_sha256")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _reject_secret_keys(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
                raise ValueError(f"runtime_index_secret_key_rejected:{path}.{key}")
            _reject_secret_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secret_keys(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class RawPartitionIndexRecord:
    domain: str
    partition_key: str
    provider: str
    request_range: str
    row_count: int
    response_sha256: str
    source_content_sha256: str
    bundle_path: str
    row_group: int
    status: str
    bundle_sha256: str = ""
    source_schema_json: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class RawIndexExport:
    parquet_path: Path
    hash_path: Path
    sha256: str
    row_count: int
    receipts_path: Path
    receipts_hash_path: Path
    receipts_sha256: str
    receipt_count: int


def _redact_secret_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            is_fingerprint = normalized.endswith("_sha256") or "fingerprint" in normalized
            if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS) and not is_fingerprint:
                sanitized[str(key)] = "<redacted>"
            else:
                sanitized[str(key)] = _redact_secret_values(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_redact_secret_values(item) for item in value]
    return value


def _verified_export_frame(path: Path) -> pd.DataFrame:
    hash_path = path.with_name(f"{path.name}.sha256")
    if not path.exists() or not hash_path.exists():
        raise RuntimeError(f"runtime_index_export_pair_incomplete:{path}")
    try:
        parts = hash_path.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"runtime_index_export_hash_unreadable:{hash_path}") from exc
    if not parts or not _SHA256_PATTERN.fullmatch(parts[0]):
        raise RuntimeError(f"runtime_index_export_hash_invalid:{hash_path}")
    if len(parts) > 1 and Path(parts[-1]).name != path.name:
        raise RuntimeError(f"runtime_index_export_hash_filename_mismatch:{hash_path}")
    expected = parts[0].lower()
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"runtime_index_export_hash_mismatch:{path}")
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"runtime_index_export_parquet_invalid:{path}") from exc


@contextmanager
def _raw_index_export_lock(destination: Path) -> Iterator[None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.export.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalize_export_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    integer_columns: tuple[str, ...],
    kind: str,
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"runtime_index_export_{kind}_columns_missing:{missing}")
    normalized = frame.loc[:, list(columns)].copy()
    if normalized.isna().any().any():
        bad = sorted(column for column in columns if normalized[column].isna().any())
        raise RuntimeError(f"runtime_index_export_{kind}_null_values:{bad}")
    for column in columns:
        if column in integer_columns:
            numeric = pd.to_numeric(normalized[column], errors="coerce")
            if numeric.isna().any() or (numeric % 1).ne(0).any():
                raise RuntimeError(
                    f"runtime_index_export_{kind}_integer_invalid:{column}"
                )
            normalized[column] = numeric.astype("int64")
        else:
            normalized[column] = normalized[column].astype(str)
    return normalized


def _merge_immutable_rows(
    frames: Sequence[pd.DataFrame],
    *,
    columns: tuple[str, ...],
    key_columns: tuple[str, ...],
    kind: str,
) -> pd.DataFrame:
    rows: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for frame in frames:
        for payload in frame.loc[:, list(columns)].to_dict("records"):
            key = tuple(payload[column] for column in key_columns)
            signature = tuple(payload[column] for column in columns)
            existing = rows.get(key)
            if existing is not None and existing != signature:
                raise RuntimeError(
                    f"runtime_index_export_{kind}_conflict:{key}"
                )
            rows[key] = signature
            records[key] = payload
    ordered = [records[key] for key in sorted(records, key=lambda item: tuple(map(str, item)))]
    return pd.DataFrame(ordered, columns=list(columns))


def _validate_export_catalog(
    partitions: pd.DataFrame,
    receipts: pd.DataFrame,
) -> None:
    partition_keys = {
        (str(row.domain), str(row.partition_key), str(row.source_content_sha256))
        for row in partitions.itertuples(index=False)
    }
    receipt_keys = {
        (str(row.domain), str(row.partition_key), str(row.source_content_sha256))
        for row in receipts.itertuples(index=False)
    }
    if partition_keys != receipt_keys:
        raise RuntimeError(
            "runtime_index_export_catalog_key_mismatch:"
            f"mapping_only={sorted(partition_keys - receipt_keys)[:5]}:"
            f"receipt_only={sorted(receipt_keys - partition_keys)[:5]}"
        )
    for row in receipts.itertuples(index=False):
        content_sha = str(row.source_content_sha256)
        if not _SHA256_PATTERN.fullmatch(content_sha):
            raise RuntimeError(
                f"runtime_index_export_content_hash_invalid:{content_sha}"
            )
        try:
            receipt = json.loads(str(row.receipt_json))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"runtime_index_export_receipt_json_invalid:{row.domain}:{row.partition_key}"
            ) from exc
        if not isinstance(receipt, dict):
            raise RuntimeError(
                f"runtime_index_export_receipt_not_object:{row.domain}:{row.partition_key}"
            )
        receipt_key = (
            str(receipt.get("raw_domain", "") or ""),
            f"{receipt.get('partition_field', '')}={receipt.get('partition_value', '')}",
            str(receipt.get("content_sha256", "") or ""),
        )
        expected_key = (str(row.domain), str(row.partition_key), content_sha)
        if receipt_key != expected_key:
            raise RuntimeError(
                "runtime_index_export_receipt_identity_mismatch:"
                f"expected={expected_key}:actual={receipt_key}"
            )
        if int(receipt.get("row_count", -1)) != int(row.row_count):
            raise RuntimeError(
                f"runtime_index_export_receipt_row_count_mismatch:{expected_key}"
            )

    grouped = partitions.groupby(
        ["domain", "partition_key", "source_content_sha256"],
        sort=False,
        dropna=False,
    )
    receipt_counts = {
        (str(row.domain), str(row.partition_key), str(row.source_content_sha256)): int(row.row_count)
        for row in receipts.itertuples(index=False)
    }
    for key_values, group in grouped:
        key = tuple(map(str, key_values))
        expected_rows = receipt_counts[key]
        if int(group["row_count"].sum()) != expected_rows:
            raise RuntimeError(
                f"runtime_index_export_mapping_row_count_mismatch:{key}"
            )
        statuses = set(group["status"].astype(str))
        if expected_rows == 0:
            if len(group) != 1 or statuses != {"empty_success"}:
                raise RuntimeError(
                    f"runtime_index_export_empty_mapping_invalid:{key}"
                )
            row = group.iloc[0]
            if str(row["bundle_path"]) or int(row["row_group"]) != -1:
                raise RuntimeError(
                    f"runtime_index_export_empty_mapping_path_invalid:{key}"
                )
        else:
            if statuses != {"bundled"}:
                raise RuntimeError(
                    f"runtime_index_export_mapping_status_invalid:{key}:{statuses}"
                )
            if group["bundle_path"].astype(str).eq("").any():
                raise RuntimeError(
                    f"runtime_index_export_bundle_path_missing:{key}"
                )
            for bundle_sha in group["bundle_sha256"].astype(str):
                if not _SHA256_PATTERN.fullmatch(bundle_sha):
                    raise RuntimeError(
                        f"runtime_index_export_bundle_hash_invalid:{key}"
                    )


class RuntimeIndex:
    """Small, concurrent runtime catalog kept away from large raw bundles."""

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        database_path: str | Path | None = None,
    ) -> None:
        paths = ensure_qdp_v3_layout(workspace_root)
        self.paths = paths
        self.workspace_root = workspace_root
        self.path = Path(database_path).resolve() if database_path is not None else paths.runtime_index
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._restore_verified_exports_if_blank()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_partitions (
                    raw_partition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    request_range TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count >= 0),
                    response_sha256 TEXT NOT NULL,
                    source_content_sha256 TEXT NOT NULL,
                    bundle_path TEXT NOT NULL,
                    row_group INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    bundle_sha256 TEXT NOT NULL,
                    source_schema_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE (
                        domain,
                        partition_key,
                        source_content_sha256,
                        bundle_path,
                        row_group
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_raw_partitions_domain_partition
                    ON raw_partitions (domain, partition_key);
                CREATE INDEX IF NOT EXISTS idx_raw_partitions_bundle
                    ON raw_partitions (bundle_path, row_group);

                CREATE TABLE IF NOT EXISTS raw_receipts (
                    domain TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    request_range TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count >= 0),
                    response_sha256 TEXT NOT NULL,
                    source_content_sha256 TEXT NOT NULL,
                    quality_tier TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (domain, partition_key, source_content_sha256)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pages (
                    job_id TEXT NOT NULL,
                    task_key TEXT NOT NULL,
                    page_number INTEGER NOT NULL CHECK (page_number >= 0),
                    request_range TEXT NOT NULL,
                    row_count INTEGER NOT NULL CHECK (row_count >= 0),
                    response_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cursor_end_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, task_key, page_number)
                );

                CREATE TABLE IF NOT EXISTS corrections_applied (
                    correction_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    record_key_json TEXT NOT NULL,
                    field TEXT NOT NULL,
                    old_value_json TEXT NOT NULL,
                    new_value_json TEXT NOT NULL,
                    bundle_path TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    PRIMARY KEY (correction_id, domain, record_key_json, field)
                );
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(raw_partitions)").fetchall()
            }
            if "source_schema_json" not in columns:
                connection.execute(
                    "ALTER TABLE raw_partitions "
                    "ADD COLUMN source_schema_json TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(f"PRAGMA user_version={RUNTIME_INDEX_SCHEMA_VERSION}")
            connection.commit()

    def _verified_export_catalog(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        metadata = self.paths.metadata
        if not metadata.exists():
            return None
        index_paths = {path.parent: path for path in metadata.rglob("raw_index.parquet")}
        receipt_paths = {path.parent: path for path in metadata.rglob("raw_receipts.parquet")}
        directories = sorted(set(index_paths) | set(receipt_paths), key=str)
        if not directories:
            return None
        partition_frames: list[pd.DataFrame] = []
        receipt_frames: list[pd.DataFrame] = []
        for directory in directories:
            index_path = index_paths.get(directory)
            receipts_path = receipt_paths.get(directory)
            if index_path is None or receipts_path is None:
                raise RuntimeError(
                    f"runtime_index_export_pair_incomplete:{directory}"
                )
            with _raw_index_export_lock(directory):
                partition_frames.append(
                    _normalize_export_frame(
                        _verified_export_frame(index_path),
                        columns=_RAW_PARTITION_COLUMNS,
                        integer_columns=("row_count", "row_group"),
                        kind="partitions",
                    )
                )
                receipt_frames.append(
                    _normalize_export_frame(
                        _verified_export_frame(receipts_path),
                        columns=_RAW_RECEIPT_COLUMNS,
                        integer_columns=("row_count",),
                        kind="receipts",
                    )
                )
        partitions = _merge_immutable_rows(
            partition_frames,
            columns=_RAW_PARTITION_COLUMNS,
            key_columns=_RAW_PARTITION_KEY,
            kind="partitions",
        )
        receipts = _merge_immutable_rows(
            receipt_frames,
            columns=_RAW_RECEIPT_COLUMNS,
            key_columns=_RAW_RECEIPT_KEY,
            kind="receipts",
        )
        _validate_export_catalog(partitions, receipts)
        return partitions, receipts

    def _restore_verified_exports_if_blank(self) -> int:
        with self.connect() as connection:
            partition_count = int(
                connection.execute("SELECT COUNT(*) FROM raw_partitions").fetchone()[0]
            )
            receipt_count = int(
                connection.execute("SELECT COUNT(*) FROM raw_receipts").fetchone()[0]
            )
        if partition_count or receipt_count:
            return 0

        catalog = self._verified_export_catalog()
        if catalog is None:
            return 0
        partitions, receipts = catalog
        if partitions.empty and receipts.empty:
            return 0

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            partition_count = int(
                connection.execute("SELECT COUNT(*) FROM raw_partitions").fetchone()[0]
            )
            receipt_count = int(
                connection.execute("SELECT COUNT(*) FROM raw_receipts").fetchone()[0]
            )
            if partition_count or receipt_count:
                connection.rollback()
                return 0
            connection.executemany(
                """
                INSERT INTO raw_partitions (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, bundle_path,
                    row_group, status, bundle_sha256, source_schema_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tuple(row[column] for column in _RAW_PARTITION_COLUMNS)
                    for row in partitions.to_dict("records")
                ],
            )
            connection.executemany(
                """
                INSERT INTO raw_receipts (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, quality_tier,
                    receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tuple(row[column] for column in _RAW_RECEIPT_COLUMNS)
                    for row in receipts.to_dict("records")
                ],
            )
            connection.commit()
        return int(len(receipts))

    def restore_raw_domain_from_verified_export(self, domain: str) -> tuple[int, int]:
        """Restore one domain after a failed, not-yet-exported compaction attempt."""

        normalized_domain = str(domain).strip()
        if not normalized_domain:
            raise ValueError("runtime_index_restore_domain_missing")
        catalog = self._verified_export_catalog()
        if catalog is None:
            partitions = pd.DataFrame(columns=_RAW_PARTITION_COLUMNS)
            receipts = pd.DataFrame(columns=_RAW_RECEIPT_COLUMNS)
        else:
            all_partitions, all_receipts = catalog
            partitions = all_partitions.loc[
                all_partitions["domain"].astype(str).eq(normalized_domain)
            ].copy()
            receipts = all_receipts.loc[
                all_receipts["domain"].astype(str).eq(normalized_domain)
            ].copy()

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM raw_partitions WHERE domain = ?",
                (normalized_domain,),
            )
            connection.execute(
                "DELETE FROM raw_receipts WHERE domain = ?",
                (normalized_domain,),
            )
            connection.executemany(
                """
                INSERT INTO raw_partitions (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, bundle_path,
                    row_group, status, bundle_sha256, source_schema_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tuple(row[column] for column in _RAW_PARTITION_COLUMNS)
                    for row in partitions.to_dict("records")
                ],
            )
            connection.executemany(
                """
                INSERT INTO raw_receipts (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, quality_tier,
                    receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tuple(row[column] for column in _RAW_RECEIPT_COLUMNS)
                    for row in receipts.to_dict("records")
                ],
            )
            connection.commit()
        return int(len(partitions)), int(len(receipts))

    def record_raw_partition(self, record: RawPartitionIndexRecord) -> None:
        if int(record.row_count) < 0:
            raise ValueError("runtime_index_negative_row_count")
        created_at = record.created_at or utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_partitions (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, bundle_path,
                    row_group, status, bundle_sha256, source_schema_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    domain, partition_key, source_content_sha256, bundle_path, row_group
                ) DO UPDATE SET
                    provider=excluded.provider,
                    request_range=excluded.request_range,
                    row_count=excluded.row_count,
                    response_sha256=excluded.response_sha256,
                    status=excluded.status,
                    bundle_sha256=excluded.bundle_sha256,
                    source_schema_json=excluded.source_schema_json
                """,
                (
                    str(record.domain),
                    str(record.partition_key),
                    str(record.provider),
                    str(record.request_range),
                    int(record.row_count),
                    str(record.response_sha256),
                    str(record.source_content_sha256),
                    str(record.bundle_path),
                    int(record.row_group),
                    str(record.status),
                    str(record.bundle_sha256),
                    str(record.source_schema_json),
                    created_at,
                ),
            )
            connection.commit()

    def delete_raw_partition_mappings(
        self,
        *,
        domain: str,
        partition_key: str,
        source_content_sha256: str,
    ) -> None:
        """Retire an older bundle layout before replacing its mappings."""

        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM raw_partitions
                WHERE domain = ? AND partition_key = ? AND source_content_sha256 = ?
                """,
                (str(domain), str(partition_key), str(source_content_sha256)),
            )
            connection.commit()

    def record_empty_success(
        self,
        *,
        domain: str,
        partition_key: str,
        provider: str,
        request_range: str,
        response_sha256: str,
        source_content_sha256: str,
        source_schema_json: str = "",
    ) -> None:
        self.record_raw_partition(
            RawPartitionIndexRecord(
                domain=str(domain),
                partition_key=str(partition_key),
                provider=str(provider),
                request_range=str(request_range),
                row_count=0,
                response_sha256=str(response_sha256),
                source_content_sha256=str(source_content_sha256),
                bundle_path="",
                row_group=-1,
                status="empty_success",
                source_schema_json=str(source_schema_json),
            )
        )

    def record_index_only_empty(
        self,
        *,
        domain: str,
        partition_key: str,
        provider: str,
        request_range: str,
        response_sha256: str,
        source_content_sha256: str,
        source_schema_json: str,
        quality_tier: str,
        receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Atomically record an immutable successful-empty raw version."""

        content_sha = str(source_content_sha256)
        if not _SHA256_PATTERN.fullmatch(content_sha):
            raise ValueError(f"runtime_index_empty_content_hash_invalid:{content_sha}")
        try:
            source_schema = json.loads(str(source_schema_json))
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime_index_empty_source_schema_invalid") from exc
        if not isinstance(source_schema, Mapping):
            raise ValueError("runtime_index_empty_source_schema_not_object")
        columns = list(source_schema.get("columns", []) or [])
        dtypes = list(source_schema.get("dtypes", []) or [])
        if len(columns) != len(dtypes) or len(set(map(str, columns))) != len(columns):
            raise ValueError("runtime_index_empty_source_schema_width_mismatch")

        sanitized = _redact_secret_values(dict(receipt))
        if int(sanitized.get("row_count", -1)) != 0:
            raise ValueError("runtime_index_empty_receipt_row_count_invalid")
        if str(sanitized.get("content_sha256", "") or "") != content_sha:
            raise ValueError("runtime_index_empty_receipt_content_hash_mismatch")
        desired_revision = str(sanitized.get("revision_of", "") or "")
        receipt_json = _canonical_json(sanitized)
        created_at = str(sanitized.get("stored_at", "") or utc_now())

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt_rows = connection.execute(
                """
                SELECT * FROM raw_receipts
                WHERE domain = ? AND partition_key = ?
                ORDER BY source_content_sha256
                """,
                (str(domain), str(partition_key)),
            ).fetchall()
            mapping_rows = connection.execute(
                """
                SELECT * FROM raw_partitions
                WHERE domain = ? AND partition_key = ?
                ORDER BY source_content_sha256, bundle_path, row_group
                """,
                (str(domain), str(partition_key)),
            ).fetchall()
            receipt_hashes = {str(row["source_content_sha256"]) for row in receipt_rows}
            mapping_hashes = {str(row["source_content_sha256"]) for row in mapping_rows}
            if receipt_hashes != mapping_hashes:
                connection.rollback()
                raise RuntimeError(
                    "runtime_index_empty_existing_catalog_incomplete:"
                    f"receipt_only={sorted(receipt_hashes - mapping_hashes)}:"
                    f"mapping_only={sorted(mapping_hashes - receipt_hashes)}"
                )

            existing_receipts: dict[str, dict[str, Any]] = {}
            for row in receipt_rows:
                existing_hash = str(row["source_content_sha256"])
                try:
                    payload = json.loads(str(row["receipt_json"]))
                except (TypeError, ValueError) as exc:
                    connection.rollback()
                    raise RuntimeError(
                        f"runtime_index_empty_existing_receipt_invalid:{existing_hash}"
                    ) from exc
                if not isinstance(payload, dict) or str(payload.get("content_sha256", "") or "") != existing_hash:
                    connection.rollback()
                    raise RuntimeError(
                        f"runtime_index_empty_existing_receipt_identity_mismatch:{existing_hash}"
                    )
                existing_receipts[existing_hash] = payload

            current_hash = ""
            if existing_receipts:
                predecessors = {
                    str(payload.get("revision_of", "") or "")
                    for payload in existing_receipts.values()
                    if str(payload.get("revision_of", "") or "")
                }
                missing_predecessors = predecessors - set(existing_receipts)
                tails = set(existing_receipts) - predecessors
                roots = {
                    item
                    for item, payload in existing_receipts.items()
                    if not str(payload.get("revision_of", "") or "")
                }
                if missing_predecessors or len(tails) != 1 or len(roots) != 1:
                    connection.rollback()
                    raise RuntimeError(
                        "runtime_index_empty_existing_revision_chain_invalid:"
                        f"missing={sorted(missing_predecessors)}:tails={sorted(tails)}:roots={sorted(roots)}"
                    )
                current_hash = next(iter(tails))
                visited: set[str] = set()
                cursor = current_hash
                while cursor and cursor not in visited:
                    visited.add(cursor)
                    cursor = str(existing_receipts[cursor].get("revision_of", "") or "")
                if cursor or visited != set(existing_receipts):
                    connection.rollback()
                    raise RuntimeError("runtime_index_empty_existing_revision_chain_disconnected")

            if content_sha in existing_receipts:
                if current_hash != content_sha:
                    connection.rollback()
                    raise RuntimeError(
                        f"runtime_index_empty_content_reversion_unsupported:{content_sha}"
                    )
                matching_mappings = [
                    row for row in mapping_rows if str(row["source_content_sha256"]) == content_sha
                ]
                if len(matching_mappings) != 1:
                    connection.rollback()
                    raise RuntimeError(
                        f"runtime_index_empty_mapping_ambiguous:{content_sha}"
                    )
                mapping = matching_mappings[0]
                mapping_identity = (
                    int(mapping["row_count"]),
                    str(mapping["bundle_path"]),
                    int(mapping["row_group"]),
                    str(mapping["status"]),
                    str(mapping["source_schema_json"]),
                )
                expected_mapping = (0, "", -1, "empty_success", str(source_schema_json))
                receipt_row = next(
                    row for row in receipt_rows if str(row["source_content_sha256"]) == content_sha
                )
                receipt_identity = (
                    str(receipt_row["provider"]),
                    str(receipt_row["request_range"]),
                    int(receipt_row["row_count"]),
                    str(receipt_row["response_sha256"]),
                    str(receipt_row["quality_tier"]),
                )
                expected_receipt_identity = (
                    str(provider),
                    str(request_range),
                    0,
                    str(response_sha256),
                    str(quality_tier),
                )
                if mapping_identity != expected_mapping or receipt_identity != expected_receipt_identity:
                    connection.rollback()
                    raise RuntimeError(
                        f"runtime_index_empty_immutable_conflict:{content_sha}"
                    )
                stored = existing_receipts[content_sha]
                connection.rollback()
                return stored, False

            if desired_revision != current_hash:
                connection.rollback()
                raise RuntimeError(
                    "runtime_index_empty_revision_tail_mismatch:"
                    f"expected={current_hash}:actual={desired_revision}"
                )
            connection.execute(
                """
                INSERT INTO raw_receipts (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, quality_tier,
                    receipt_json, created_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    str(domain),
                    str(partition_key),
                    str(provider),
                    str(request_range),
                    str(response_sha256),
                    content_sha,
                    str(quality_tier),
                    receipt_json,
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO raw_partitions (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, bundle_path,
                    row_group, status, bundle_sha256, source_schema_json, created_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, '', -1, 'empty_success', '', ?, ?)
                """,
                (
                    str(domain),
                    str(partition_key),
                    str(provider),
                    str(request_range),
                    str(response_sha256),
                    content_sha,
                    str(source_schema_json),
                    created_at,
                ),
            )
            connection.commit()
        return sanitized, True

    def record_raw_receipt(
        self,
        *,
        domain: str,
        partition_key: str,
        provider: str,
        request_range: str,
        row_count: int,
        response_sha256: str,
        source_content_sha256: str,
        quality_tier: str,
        receipt: Mapping[str, Any],
    ) -> None:
        if int(row_count) < 0:
            raise ValueError("runtime_index_negative_receipt_row_count")
        sanitized = _redact_secret_values(dict(receipt))
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_receipts (
                    domain, partition_key, provider, request_range, row_count,
                    response_sha256, source_content_sha256, quality_tier,
                    receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (domain, partition_key, source_content_sha256) DO UPDATE SET
                    provider=excluded.provider,
                    request_range=excluded.request_range,
                    row_count=excluded.row_count,
                    response_sha256=excluded.response_sha256,
                    quality_tier=excluded.quality_tier,
                    receipt_json=excluded.receipt_json
                """,
                (
                    str(domain),
                    str(partition_key),
                    str(provider),
                    str(request_range),
                    int(row_count),
                    str(response_sha256),
                    str(source_content_sha256),
                    str(quality_tier),
                    _canonical_json(sanitized),
                    utc_now(),
                ),
            )
            connection.commit()

    def upsert_job(
        self,
        *,
        job_id: str,
        domain: str,
        provider: str,
        status: str,
        payload: Mapping[str, Any] | None = None,
        heartbeat_at: str = "",
    ) -> None:
        safe_payload = dict(payload or {})
        _reject_secret_keys(safe_payload)
        now = utc_now()
        heartbeat = heartbeat_at or now
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, domain, provider, status, payload_json,
                    heartbeat_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (job_id) DO UPDATE SET
                    domain=excluded.domain,
                    provider=excluded.provider,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    heartbeat_at=excluded.heartbeat_at,
                    updated_at=excluded.updated_at
                """,
                (
                    str(job_id),
                    str(domain),
                    str(provider),
                    str(status),
                    _canonical_json(safe_payload),
                    heartbeat,
                    now,
                    now,
                ),
            )
            connection.commit()

    def record_page(
        self,
        *,
        job_id: str,
        task_key: str,
        page_number: int,
        request_range: Mapping[str, Any] | str,
        row_count: int,
        response_sha256: str,
        status: str,
        cursor_end_at: str = "",
    ) -> None:
        if int(page_number) < 0 or int(row_count) < 0:
            raise ValueError("runtime_index_invalid_page_numbers")
        if isinstance(request_range, Mapping):
            _reject_secret_keys(request_range, path="request_range")
            request_text = _canonical_json(dict(request_range))
        else:
            request_text = str(request_range)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO pages (
                    job_id, task_key, page_number, request_range, row_count,
                    response_sha256, status, cursor_end_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (job_id, task_key, page_number) DO UPDATE SET
                    request_range=excluded.request_range,
                    row_count=excluded.row_count,
                    response_sha256=excluded.response_sha256,
                    status=excluded.status,
                    cursor_end_at=excluded.cursor_end_at
                """,
                (
                    str(job_id),
                    str(task_key),
                    int(page_number),
                    request_text,
                    int(row_count),
                    str(response_sha256),
                    str(status),
                    str(cursor_end_at),
                    utc_now(),
                ),
            )
            connection.commit()

    def record_correction_applied(
        self,
        *,
        correction_id: str,
        domain: str,
        record_key: Mapping[str, Any],
        field: str,
        old_value: Any,
        new_value: Any,
        bundle_path: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO corrections_applied (
                    correction_id, domain, record_key_json, field,
                    old_value_json, new_value_json, bundle_path, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (correction_id, domain, record_key_json, field) DO UPDATE SET
                    old_value_json=excluded.old_value_json,
                    new_value_json=excluded.new_value_json,
                    bundle_path=excluded.bundle_path,
                    applied_at=excluded.applied_at
                """,
                (
                    str(correction_id),
                    str(domain),
                    _canonical_json(dict(record_key)),
                    str(field),
                    _canonical_json(old_value),
                    _canonical_json(new_value),
                    str(bundle_path),
                    utc_now(),
                ),
            )
            connection.commit()

    def table_frame(self, table: str) -> pd.DataFrame:
        allowed = {"raw_partitions", "raw_receipts", "jobs", "pages", "corrections_applied"}
        if table not in allowed:
            raise ValueError(f"runtime_index_unknown_table:{table}")
        order_by = {
            "raw_partitions": "domain, partition_key, source_content_sha256, bundle_path, row_group",
            "raw_receipts": "domain, partition_key, source_content_sha256",
            "jobs": "job_id",
            "pages": "job_id, task_key, page_number",
            "corrections_applied": "correction_id, domain, record_key_json, field",
        }[table]
        with self.connect() as connection:
            return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order_by}", connection)

    def raw_catalog_frames(
        self,
        domain: str,
        *,
        partition_key: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the bundle mappings and receipts for one raw domain."""

        partition_clause = " AND partition_key = ?" if partition_key else ""
        parameters: tuple[str, ...] = (
            (str(domain), str(partition_key)) if partition_key else (str(domain),)
        )
        with self.connect() as connection:
            partitions = pd.read_sql_query(
                f"""
                SELECT * FROM raw_partitions
                WHERE domain = ?
                {partition_clause}
                ORDER BY partition_key, source_content_sha256, bundle_path, row_group
                """,
                connection,
                params=parameters,
            )
            receipts = pd.read_sql_query(
                f"""
                SELECT * FROM raw_receipts
                WHERE domain = ?
                {partition_clause}
                ORDER BY partition_key, source_content_sha256
                """,
                connection,
                params=parameters,
            )
        return partitions, receipts

    def export_raw_index(self, output_dir: str | Path) -> RawIndexExport:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with _raw_index_export_lock(destination):
            return self._export_raw_index_unlocked(destination)

    def _export_raw_index_unlocked(self, destination: Path) -> RawIndexExport:
        parquet_path = destination / "raw_index.parquet"
        with self.connect() as connection:
            connection.execute("BEGIN")
            frame = pd.read_sql_query(
                """
                SELECT * FROM raw_partitions
                ORDER BY domain, partition_key, source_content_sha256, bundle_path, row_group
                """,
                connection,
            )
            receipts = pd.read_sql_query(
                """
                SELECT * FROM raw_receipts
                ORDER BY domain, partition_key, source_content_sha256
                """,
                connection,
            )
            connection.rollback()
        atomic_write_parquet(parquet_path, frame)
        digest = sha256_file(parquet_path)
        hash_path = destination / "raw_index.parquet.sha256"
        temporary = hash_path.with_name(f".{hash_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(f"{digest}  {parquet_path.name}\n", encoding="ascii")
            temporary.replace(hash_path)
        finally:
            temporary.unlink(missing_ok=True)
        receipts_path = destination / "raw_receipts.parquet"
        atomic_write_parquet(receipts_path, receipts)
        receipts_digest = sha256_file(receipts_path)
        receipts_hash_path = destination / "raw_receipts.parquet.sha256"
        receipts_temporary = receipts_hash_path.with_name(
            f".{receipts_hash_path.name}.{os.getpid()}.tmp"
        )
        try:
            receipts_temporary.write_text(
                f"{receipts_digest}  {receipts_path.name}\n", encoding="ascii"
            )
            receipts_temporary.replace(receipts_hash_path)
        finally:
            receipts_temporary.unlink(missing_ok=True)
        verified_partitions = _normalize_export_frame(
            _verified_export_frame(parquet_path),
            columns=_RAW_PARTITION_COLUMNS,
            integer_columns=("row_count", "row_group"),
            kind="partitions",
        )
        verified_receipts = _normalize_export_frame(
            _verified_export_frame(receipts_path),
            columns=_RAW_RECEIPT_COLUMNS,
            integer_columns=("row_count",),
            kind="receipts",
        )
        _validate_export_catalog(verified_partitions, verified_receipts)
        if len(verified_partitions) != len(frame) or len(verified_receipts) != len(receipts):
            raise RuntimeError("runtime_index_export_round_trip_row_count_mismatch")
        return RawIndexExport(
            parquet_path=parquet_path,
            hash_path=hash_path,
            sha256=digest,
            row_count=int(len(frame)),
            receipts_path=receipts_path,
            receipts_hash_path=receipts_hash_path,
            receipts_sha256=receipts_digest,
            receipt_count=int(len(receipts)),
        )

    def raw_records_for_bundle(self, bundle_path: str) -> Sequence[sqlite3.Row]:
        with self.connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT * FROM raw_partitions
                    WHERE bundle_path = ?
                    ORDER BY row_group, raw_partition_id
                    """,
                    (str(bundle_path),),
                ).fetchall()
            )
