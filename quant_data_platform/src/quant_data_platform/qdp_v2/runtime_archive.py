from __future__ import annotations

"""Seal expanded QDP runtime evidence into verified, restorable archives."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.manifest import atomic_write_json, utc_now
from quant_data_platform.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)

ARCHIVE_VERSION = 1
DEFAULT_CLUSTER_BYTES = 1024 * 1024
YEAR_PATTERNS = (
    re.compile(r"^year=(20(?:0\d|1\d|2[0-6]))(?:$|[^0-9])", re.IGNORECASE),
    re.compile(r"^date=(20(?:0\d|1\d|2[0-6]))[-_]", re.IGNORECASE),
    re.compile(r"^task=(20(?:0\d|1\d|2[0-6]))(?:$|[^0-9])", re.IGNORECASE),
    re.compile(r"__(20(?:0\d|1\d|2[0-6]))\d{2}(?:\D|$)"),
)
SENSITIVE_KEYS = {"token", "api_key", "apikey", "authorization", "password", "secret"}

DEFAULT_SELECTIONS: dict[str, tuple[str, ...]] = {
    "tushare_extended_backfill_v1": ("raw",),
    "research_report_rc_backfill_v1": ("raw", "normalized"),
    "research_report_rc_backfill_v2": ("raw", "normalized"),
    "historical_intraday_5m_repair_v1": ("parts",),
    "historical_intraday_5m_external_archive_repair_v2": ("symbols",),
    "pit_history_restore": (
        "domain_parts",
        "factor_parts",
        "history_parts",
        "name_interval_parts",
        "share_event_parts",
        "supplement_parts",
    ),
    "margin_eligibility_exchange_history_v1": ("raw",),
}

LEDGER_COLUMNS = (
    "workflow",
    "domain",
    "year",
    "relative_path",
    "file_size",
    "mtime_ns",
    "sha256",
    "suffix",
    "row_count",
    "schema_sha256",
    "request_status",
    "request_offset",
    "request_params_json",
    "response_sha256",
    "error_type",
    "error_message",
)


class RuntimeArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveUnit:
    workflow: str
    domain: str
    year: str
    workflow_root: Path
    files: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.workflow}/{self.domain}/year={self.year}"


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
        _archive_root(workspace)
        / _safe_name(unit.workflow)
        / _safe_name(unit.domain)
        / f"year={_safe_name(unit.year)}"
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


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>" if str(key).lower() in SENSITIVE_KEYS else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _json_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    params = payload.get("params", payload.get("request_params", {}))
    return {
        "request_status": str(payload.get("status", "") or ""),
        "request_offset": payload.get("offset", payload.get("request_offset")),
        "request_params_json": json.dumps(
            _redact(params), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if params
        else "",
        "response_sha256": str(
            payload.get("response_sha256", payload.get("response_hash", "")) or ""
        ),
        "error_type": str(payload.get("error_type", "") or ""),
        "error_message": str(
            payload.get("last_error", payload.get("error", payload.get("message", "")))
            or ""
        )[:1000],
    }


def _ledger_record(unit: ArchiveUnit, path: Path) -> dict[str, Any]:
    stat = path.stat()
    row_count: int | None = None
    schema_hash = ""
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        row_count = int(parquet.metadata.num_rows)
        schema_hash = hashlib.sha256(
            str(parquet.schema_arrow).encode("utf-8")
        ).hexdigest()
    json_meta = _json_metadata(path) if path.suffix.lower() == ".json" else {}
    return {
        "workflow": unit.workflow,
        "domain": unit.domain,
        "year": unit.year,
        "relative_path": path.relative_to(unit.workflow_root).as_posix(),
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
        "suffix": path.suffix.lower(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "request_status": str(json_meta.get("request_status", "")),
        "request_offset": json_meta.get("request_offset"),
        "request_params_json": str(json_meta.get("request_params_json", "")),
        "response_sha256": str(json_meta.get("response_sha256", "")),
        "error_type": str(json_meta.get("error_type", "")),
        "error_message": str(json_meta.get("error_message", "")),
    }


def build_ledger(unit: ArchiveUnit) -> pd.DataFrame:
    frame = pd.DataFrame([_ledger_record(unit, path) for path in unit.files])
    if frame.empty or tuple(frame.columns) != LEDGER_COLUMNS:
        raise RuntimeArchiveError(f"runtime_archive_ledger_invalid:{unit.key}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeArchiveError(f"runtime_archive_duplicate_path:{unit.key}")
    return frame.sort_values("relative_path", kind="stable").reset_index(drop=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _create_tar_zst(unit: ArchiveUnit, ledger: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    process = subprocess.Popen(
        [_zstd_executable(), "-q", "-T0", "-1", "-f", "-o", str(partial)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        with tarfile.open(
            fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT
        ) as archive:
            for row in ledger.to_dict("records"):
                source = _within(
                    unit.workflow_root / str(row["relative_path"]),
                    unit.workflow_root,
                )
                if not source.is_file():
                    raise RuntimeArchiveError(
                        f"runtime_archive_source_disappeared:{source}"
                    )
                stat = source.stat()
                if (
                    stat.st_size != int(row["file_size"])
                    or stat.st_mtime_ns != int(row["mtime_ns"])
                    or _sha256(source) != str(row["sha256"])
                ):
                    raise RuntimeArchiveError(
                        f"runtime_archive_source_changed:{source}"
                    )
                archive.add(
                    source,
                    arcname=str(row["relative_path"]),
                    recursive=False,
                )
        process.stdin.close()
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        code = process.wait()
    except BaseException:
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        process.kill()
        process.wait()
        partial.unlink(missing_ok=True)
        raise
    if code != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeArchiveError(
            f"zstd_archive_failed:{unit.key}:{code}:{stderr[-500:]}"
        )
    os.replace(partial, target)


def _safe_member_name(value: str) -> str:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise RuntimeArchiveError(f"runtime_archive_unsafe_member:{value}")
    return posix.as_posix()


def verify_archive(archive_path: Path, ledger: pd.DataFrame) -> dict[str, Any]:
    expected = {
        str(row["relative_path"]): (int(row["file_size"]), str(row["sha256"]))
        for row in ledger.to_dict("records")
    }
    seen: dict[str, tuple[int, str]] = {}
    process = subprocess.Popen(
        [_zstd_executable(), "-q", "-d", "-c", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    raise RuntimeArchiveError(
                        f"runtime_archive_non_file_member:{member.name}"
                    )
                name = _safe_member_name(member.name)
                if name in seen or name not in expected:
                    raise RuntimeArchiveError(
                        f"runtime_archive_unexpected_or_duplicate_member:{name}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeArchiveError(
                        f"runtime_archive_member_unreadable:{name}"
                    )
                digest = hashlib.sha256()
                size = 0
                while block := handle.read(1024 * 1024):
                    size += len(block)
                    digest.update(block)
                seen[name] = (size, digest.hexdigest())
        # ``tarfile`` stops at the TAR end marker.  Drain the decompressor's
        # remaining padded output so zstd cannot block on a full stdout pipe.
        while process.stdout.read(1024 * 1024):
            pass
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        code = process.wait()
    except BaseException as exc:
        process.kill()
        process.wait()
        if isinstance(exc, RuntimeArchiveError):
            raise
        raise RuntimeArchiveError(
            f"runtime_archive_stream_invalid:{archive_path}:{type(exc).__name__}"
        ) from exc
    if code != 0:
        raise RuntimeArchiveError(
            f"zstd_archive_verify_failed:{archive_path}:{code}:{stderr[-500:]}"
        )
    if seen != expected:
        missing = sorted(set(expected).difference(seen))
        mismatched = sorted(
            name
            for name in set(expected).intersection(seen)
            if expected[name] != seen[name]
        )
        raise RuntimeArchiveError(
            f"runtime_archive_verify_mismatch:missing={missing[:3]}:mismatch={mismatched[:3]}"
        )
    return {
        "verified": True,
        "verified_file_count": len(seen),
        "verified_logical_bytes": sum(size for size, _ in seen.values()),
    }


def restore_archive(
    archive_path: str | Path,
    ledger_path: str | Path,
    *,
    target_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    archive_path = Path(archive_path).resolve()
    ledger_path = Path(ledger_path).resolve()
    target = Path(target_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(ledger_path)
    expected = {
        str(row["relative_path"]): (int(row["file_size"]), str(row["sha256"]))
        for row in ledger.to_dict("records")
    }
    restored: list[Path] = []
    process = subprocess.Popen(
        [_zstd_executable(), "-q", "-d", "-c", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    raise RuntimeArchiveError(
                        f"runtime_restore_non_file_member:{member.name}"
                    )
                name = _safe_member_name(member.name)
                if name not in expected:
                    raise RuntimeArchiveError(
                        f"runtime_restore_unexpected_member:{name}"
                    )
                destination = _within(target / Path(*PurePosixPath(name).parts), target)
                if destination.exists() and not overwrite:
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(
                    destination.suffix + ".restore.partial"
                )
                temporary.unlink(missing_ok=True)
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeArchiveError(
                        f"runtime_restore_member_unreadable:{name}"
                    )
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as output:
                    while block := handle.read(1024 * 1024):
                        output.write(block)
                        size += len(block)
                        digest.update(block)
                if (size, digest.hexdigest()) != expected[name]:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeArchiveError(f"runtime_restore_hash_mismatch:{name}")
                os.replace(temporary, destination)
                restored.append(destination)
        while process.stdout.read(1024 * 1024):
            pass
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        code = process.wait()
    except BaseException as exc:
        process.kill()
        process.wait()
        if isinstance(exc, (RuntimeArchiveError, FileExistsError)):
            raise
        raise RuntimeArchiveError(
            f"runtime_restore_stream_invalid:{archive_path}:{type(exc).__name__}"
        ) from exc
    if code != 0 or len(restored) != len(expected):
        raise RuntimeArchiveError(
            f"runtime_restore_incomplete:{code}:{len(restored)}!={len(expected)}:{stderr[-500:]}"
        )
    return {
        "status": "restored",
        "archive_path": str(archive_path),
        "target_root": str(target),
        "restored_file_count": len(restored),
        "restored_logical_bytes": sum(path.stat().st_size for path in restored),
    }


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
        path = _within(
            unit.workflow_root / str(row["relative_path"]), unit.workflow_root
        )
        if not path.exists():
            already_missing += 1
            continue
        if not path.is_file():
            raise RuntimeArchiveError(f"runtime_archive_delete_target_not_file:{path}")
        if path.stat().st_size != int(row["file_size"]) or _sha256(path) != str(
            row["sha256"]
        ):
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


def seal_unit(
    unit: ArchiveUnit,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    directory = _unit_directory(workspace, unit)
    archive_path = directory / "raw.tar.zst"
    ledger_path = directory / "ledger.parquet"
    manifest_path = directory / "manifest.json"
    ledger = build_ledger(unit)
    if manifest_path.is_file() and ledger_path.is_file() and archive_path.is_file():
        try:
            existing = dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            saved = pd.read_parquet(ledger_path)
            compare_columns = ["relative_path", "file_size", "mtime_ns", "sha256"]
            current_records = ledger.loc[:, compare_columns].to_dict("records")
            saved_records = saved.loc[:, compare_columns].to_dict("records")
            if (
                existing.get("status") == "sealed"
                and existing.get("verified") is True
                and current_records == saved_records
                and _sha256(ledger_path) == str(existing.get("ledger_sha256", ""))
                and _sha256(archive_path) == str(existing.get("archive_sha256", ""))
            ):
                return {
                    **existing,
                    "manifest_path": str(manifest_path),
                    "reused": True,
                }
        except (OSError, ValueError, KeyError):
            pass
        raise RuntimeArchiveError(
            f"runtime_archive_existing_unit_differs_restore_before_reseal:{unit.key}"
        )
    _atomic_parquet(ledger, ledger_path)
    _create_tar_zst(unit, ledger, archive_path)
    verification = verify_archive(archive_path, ledger)
    logical_bytes = int(ledger["file_size"].sum())
    manifest = {
        "archive_version": ARCHIVE_VERSION,
        "status": "sealed",
        "workflow": unit.workflow,
        "domain": unit.domain,
        "year": unit.year,
        "unit_key": unit.key,
        "workflow_root": str(unit.workflow_root),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "source_file_count": len(ledger),
        "source_logical_bytes": logical_bytes,
        "estimated_source_allocated_bytes_1mib": int(
            sum(
                math.ceil(int(size) / DEFAULT_CLUSTER_BYTES) * DEFAULT_CLUSTER_BYTES
                for size in ledger["file_size"]
            )
        ),
        "verified": bool(verification["verified"]),
        "verified_file_count": int(verification["verified_file_count"]),
        "verified_logical_bytes": int(verification["verified_logical_bytes"]),
        "restore": {
            "format": "tar+zstd",
            "member_base": "workflow_root",
            "overwrite_default": False,
        },
        "created_at": utc_now(),
    }
    _assert_credential_free(manifest)
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path), "reused": False}


def verify_unit_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    manifest = dict(json.loads(path.read_text(encoding="utf-8")))
    archive_path = Path(str(manifest["archive_path"])).resolve()
    ledger_path = Path(str(manifest["ledger_path"])).resolve()
    if _sha256(archive_path) != str(manifest.get("archive_sha256", "")) or _sha256(
        ledger_path
    ) != str(manifest.get("ledger_sha256", "")):
        raise RuntimeArchiveError(f"runtime_archive_manifest_hash_mismatch:{path}")
    ledger = pd.read_parquet(ledger_path)
    verification = verify_archive(archive_path, ledger)
    return {
        "status": "ok",
        "manifest_path": str(path),
        "unit_key": manifest.get("unit_key", ""),
        **verification,
    }


def _sample_restore(
    workspace: Path,
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not manifests:
        raise RuntimeArchiveError("runtime_archive_sample_restore_no_units")
    natural_year = [item for item in manifests if str(item.get("year", "")) != "multi"]
    choices = natural_year or list(manifests)
    seed = hashlib.sha256(
        "|".join(sorted(str(item["archive_sha256"]) for item in choices)).encode(
            "ascii"
        )
    ).digest()
    selected = random.Random(seed).choice(choices)
    check_root = _within(
        _archive_root(workspace)
        / "_restore_checks"
        / f"{_safe_name(str(selected['workflow']))}_{_safe_name(str(selected['domain']))}_{_safe_name(str(selected['year']))}",
        _archive_root(workspace),
    )
    if check_root.exists():
        existing = [path for path in check_root.rglob("*") if path.is_file()]
        for path in existing:
            _within(path, check_root).unlink()
        _remove_empty_parents(existing, stop=check_root)
        try:
            check_root.rmdir()
        except OSError:
            pass
    result = restore_archive(
        str(selected["archive_path"]),
        str(selected["ledger_path"]),
        target_root=check_root,
    )
    restored_files = [path for path in check_root.rglob("*") if path.is_file()]
    for path in restored_files:
        _within(path, check_root).unlink()
    _remove_empty_parents(restored_files, stop=check_root)
    try:
        check_root.rmdir()
    except OSError as exc:
        raise RuntimeArchiveError(
            f"runtime_archive_restore_check_cleanup:{check_root}"
        ) from exc
    return {
        **result,
        "status": "restored_and_hash_verified",
        "sample_unit_key": selected["unit_key"],
        "cleanup": "exact_files_deleted_after_verification",
    }


def seal_workflows(
    *,
    workspace_root: str | Path | None = None,
    workflows: Sequence[str] = (),
    delete_expanded: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    if delete_expanded and not yes:
        raise RuntimeArchiveError("delete_expanded_requires_yes")
    units = discover_units(workspace_root=workspace, workflows=workflows)
    if not units:
        return {
            "status": "nothing_to_seal",
            "workflows": list(workflows),
            "archive_root": str(_archive_root(workspace)),
        }
    manifests: list[dict[str, Any]] = []
    for unit in units:
        manifests.append(seal_unit(unit, workspace_root=workspace))
    sample = _sample_restore(workspace, manifests)
    deletion: list[dict[str, Any]] = []
    if delete_expanded:
        unit_map = {unit.key: unit for unit in units}
        for manifest in manifests:
            unit = unit_map[str(manifest["unit_key"])]
            ledger = pd.read_parquet(str(manifest["ledger_path"]))
            deletion.append(
                {
                    "unit_key": unit.key,
                    **_delete_exact_sources(unit, ledger),
                }
            )
    source_files = sum(int(item["source_file_count"]) for item in manifests)
    source_logical = sum(int(item["source_logical_bytes"]) for item in manifests)
    source_allocated = sum(
        int(item["estimated_source_allocated_bytes_1mib"]) for item in manifests
    )
    archive_bytes = sum(int(item["archive_bytes"]) for item in manifests)
    payload = {
        "status": "sealed_and_deleted" if delete_expanded else "sealed",
        "archive_version": ARCHIVE_VERSION,
        "archive_root": str(_archive_root(workspace)),
        "workflows": sorted({unit.workflow for unit in units}),
        "unit_count": len(units),
        "source_file_count": source_files,
        "source_logical_bytes": source_logical,
        "estimated_source_allocated_bytes_1mib": source_allocated,
        "archive_bytes": archive_bytes,
        "estimated_allocated_bytes_reclaimed_1mib": max(
            0,
            source_allocated
            - sum(
                math.ceil(int(item["archive_bytes"]) / DEFAULT_CLUSTER_BYTES)
                * DEFAULT_CLUSTER_BYTES
                for item in manifests
            ),
        )
        if delete_expanded
        else 0,
        "manifests": [str(item["manifest_path"]) for item in manifests],
        "sample_restore": sample,
        "deletion": deletion,
        "delete_expanded": delete_expanded,
        "completed_at": utc_now(),
    }
    _assert_credential_free(payload)
    receipt = (
        _archive_root(workspace)
        / "receipts"
        / f"cleanup_{utc_now().replace(':', '').replace('-', '')}.json"
    )
    atomic_write_json(receipt, payload)
    return {**payload, "cleanup_receipt": str(receipt)}


def seal_completed_workflow(
    workflow: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Seal and remove expanded evidence after a workflow reaches its terminal state.

    Verification and a deterministic sample restore complete before exact source
    files are deleted.  An existing unit with different contents is never
    overwritten implicitly.
    """

    if workflow not in DEFAULT_SELECTIONS:
        raise RuntimeArchiveError(f"unknown_archive_workflow:{workflow}")
    return seal_workflows(
        workspace_root=workspace_root,
        workflows=(workflow,),
        delete_expanded=True,
        yes=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp runtime-archive")
    parser.add_argument("--workspace-root", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--workflow", action="append", default=[])
    seal.add_argument("--delete-expanded", action="store_true")
    seal.add_argument("--yes", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--manifest", required=True)
    restore.add_argument("--target-root", required=True)
    restore.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.command == "seal":
        payload = seal_workflows(
            workspace_root=workspace,
            workflows=tuple(args.workflow),
            delete_expanded=bool(args.delete_expanded),
            yes=bool(args.yes),
        )
    elif args.command == "verify":
        payload = verify_unit_manifest(args.manifest)
    else:
        manifest = dict(
            json.loads(Path(args.manifest).resolve().read_text(encoding="utf-8"))
        )
        payload = restore_archive(
            manifest["archive_path"],
            manifest["ledger_path"],
            target_root=args.target_root,
            overwrite=bool(args.overwrite),
        )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
