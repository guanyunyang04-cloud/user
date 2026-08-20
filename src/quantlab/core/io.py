from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class DataContractError(RuntimeError):
    """Raised when a persisted artifact violates its declared contract."""


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataContractError(f"missing JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def open_array(
    record: Mapping[str, Any],
    dtype: np.dtype[Any] | type | None = None,
    *,
    base: Path | None = None,
) -> np.memmap:
    shape = tuple(int(value) for value in record["shape"])
    declared_dtype = np.dtype(str(record.get("dtype", ""))) if dtype is None else np.dtype(dtype)
    path = Path(str(record["path"]))
    if not path.is_absolute():
        if base is None:
            raise DataContractError(f"relative array path has no base: {path}")
        path = base / path
    path = path.resolve()
    expected = int(np.prod(shape, dtype=np.int64)) * declared_dtype.itemsize
    if not path.is_file() or path.stat().st_size != expected:
        raise DataContractError(f"array size mismatch: {path}")
    return np.memmap(path, dtype=declared_dtype, mode="r", shape=shape)

