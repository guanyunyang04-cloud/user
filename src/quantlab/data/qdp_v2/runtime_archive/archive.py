"""Runtime Archive: archive responsibilities."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.research_event_update import (
    _sha256,
)

from .config import (
    ArchiveUnit,
    RuntimeArchiveError,
)
from .discovery import (
    _within,
    _zstd_executable,
)


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
        with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for row in ledger.to_dict("records"):
                source = _within(
                    unit.workflow_root / str(row["relative_path"]),
                    unit.workflow_root,
                )
                if not source.is_file():
                    raise RuntimeArchiveError(f"runtime_archive_source_disappeared:{source}")
                stat = source.stat()
                if (
                    stat.st_size != int(row["file_size"])
                    or stat.st_mtime_ns != int(row["mtime_ns"])
                    or _sha256(source) != str(row["sha256"])
                ):
                    raise RuntimeArchiveError(f"runtime_archive_source_changed:{source}")
                archive.add(
                    source,
                    arcname=str(row["relative_path"]),
                    recursive=False,
                )
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
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
        raise RuntimeArchiveError(f"zstd_archive_failed:{unit.key}:{code}:{stderr[-500:]}")
    os.replace(partial, target)


def _safe_member_name(value: str) -> str:
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise RuntimeArchiveError(f"runtime_archive_unsafe_member:{value}")
    return posix.as_posix()


def verify_archive(archive_path: Path, ledger: pd.DataFrame) -> dict[str, Any]:
    expected = {
        str(row["relative_path"]): (int(row["file_size"]), str(row["sha256"])) for row in ledger.to_dict("records")
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
                    raise RuntimeArchiveError(f"runtime_archive_non_file_member:{member.name}")
                name = _safe_member_name(member.name)
                if name in seen or name not in expected:
                    raise RuntimeArchiveError(f"runtime_archive_unexpected_or_duplicate_member:{name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeArchiveError(f"runtime_archive_member_unreadable:{name}")
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
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        code = process.wait()
    except BaseException as exc:
        process.kill()
        process.wait()
        if isinstance(exc, RuntimeArchiveError):
            raise
        raise RuntimeArchiveError(f"runtime_archive_stream_invalid:{archive_path}:{type(exc).__name__}") from exc
    if code != 0:
        raise RuntimeArchiveError(f"zstd_archive_verify_failed:{archive_path}:{code}:{stderr[-500:]}")
    if seen != expected:
        missing = sorted(set(expected).difference(seen))
        mismatched = sorted(name for name in set(expected).intersection(seen) if expected[name] != seen[name])
        raise RuntimeArchiveError(f"runtime_archive_verify_mismatch:missing={missing[:3]}:mismatch={mismatched[:3]}")
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
        str(row["relative_path"]): (int(row["file_size"]), str(row["sha256"])) for row in ledger.to_dict("records")
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
                    raise RuntimeArchiveError(f"runtime_restore_non_file_member:{member.name}")
                name = _safe_member_name(member.name)
                if name not in expected:
                    raise RuntimeArchiveError(f"runtime_restore_unexpected_member:{name}")
                destination = _within(target / Path(*PurePosixPath(name).parts), target)
                if destination.exists() and not overwrite:
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".restore.partial")
                temporary.unlink(missing_ok=True)
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeArchiveError(f"runtime_restore_member_unreadable:{name}")
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
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        code = process.wait()
    except BaseException as exc:
        process.kill()
        process.wait()
        if isinstance(exc, (RuntimeArchiveError, FileExistsError)):
            raise
        raise RuntimeArchiveError(f"runtime_restore_stream_invalid:{archive_path}:{type(exc).__name__}") from exc
    if code != 0 or len(restored) != len(expected):
        raise RuntimeArchiveError(f"runtime_restore_incomplete:{code}:{len(restored)}!={len(expected)}:{stderr[-500:]}")
    return {
        "status": "restored",
        "archive_path": str(archive_path),
        "target_root": str(target),
        "restored_file_count": len(restored),
        "restored_logical_bytes": sum(path.stat().st_size for path in restored),
    }
