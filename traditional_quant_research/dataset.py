"""Read-only loaders for traditional quant research snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SNAPSHOT_ROOT = Path("traditional_quant_research/data/raw/tq_daily_mainboard_v1")


@dataclass(frozen=True)
class DailySnapshot:
    root: Path
    manifest: dict[str, Any]
    universe: pd.DataFrame
    daily_bars: pd.DataFrame
    daily_status: pd.DataFrame


def _resolve_snapshot_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else DEFAULT_SNAPSHOT_ROOT
    latest = base / "latest_manifest.json"
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8-sig"))
        snapshot_path = payload.get("snapshot_path")
        if snapshot_path:
            return Path(snapshot_path)
    return base


def load_manifest(root: str | Path | None = None) -> dict[str, Any]:
    snapshot_root = _resolve_snapshot_root(root)
    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8-sig"))


def load_universe(root: str | Path | None = None) -> pd.DataFrame:
    snapshot_root = _resolve_snapshot_root(root)
    parquet_path = snapshot_root / "universe.parquet"
    csv_path = snapshot_root / "universe.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"universe not found under {snapshot_root}")


def load_daily_bars(root: str | Path | None = None) -> pd.DataFrame:
    snapshot_root = _resolve_snapshot_root(root)
    path = snapshot_root / "daily_bars.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily bars not found: {path}")
    return pd.read_parquet(path)


def load_daily_status(root: str | Path | None = None) -> pd.DataFrame:
    snapshot_root = _resolve_snapshot_root(root)
    path = snapshot_root / "daily_status.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily status not found: {path}")
    return pd.read_parquet(path)


def load_daily_snapshot(root: str | Path | None = None) -> DailySnapshot:
    snapshot_root = _resolve_snapshot_root(root)
    return DailySnapshot(
        root=snapshot_root,
        manifest=load_manifest(snapshot_root),
        universe=load_universe(snapshot_root),
        daily_bars=load_daily_bars(snapshot_root),
        daily_status=load_daily_status(snapshot_root),
    )
