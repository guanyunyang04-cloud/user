from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def get_cache_root() -> Path:
    root = Path("daily_research/cache/deep_alpha")
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def series_signature(series: pd.Series | None) -> str:
    if series is None or series.empty:
        return "empty"
    normalized = series.sort_index().astype(str).to_json(force_ascii=False)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def frame_signature(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "empty"
    normalized = frame.sort_index().sort_index(axis=1).astype(str).to_json(force_ascii=False)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
