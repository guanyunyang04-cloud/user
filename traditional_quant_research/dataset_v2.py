"""Read-only loaders for v2 point-in-time daily research snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_V2_SNAPSHOT_ROOT = Path("traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit")


@dataclass(frozen=True)
class PitDailySnapshot:
    root: Path
    manifest: dict[str, Any]
    security_master: pd.DataFrame
    daily_universe: pd.DataFrame
    daily_bars: pd.DataFrame
    daily_status: pd.DataFrame


def _resolve_snapshot_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else DEFAULT_V2_SNAPSHOT_ROOT
    latest = base / "latest_manifest.json"
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8-sig"))
        snapshot_path = payload.get("snapshot_path")
        if snapshot_path:
            return Path(snapshot_path)
    return base


def load_pit_manifest(root: str | Path | None = None) -> dict[str, Any]:
    snapshot_root = _resolve_snapshot_root(root)
    path = snapshot_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_security_master(root: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "security_master.parquet"
    if not path.exists():
        raise FileNotFoundError(f"security master not found: {path}")
    return pd.read_parquet(path)


def load_daily_universe(root: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily universe not found: {path}")
    return pd.read_parquet(path)


def load_pit_daily_bars(root: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_bars.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily bars not found: {path}")
    return pd.read_parquet(path)


def load_pit_daily_status(root: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_status.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily status not found: {path}")
    return pd.read_parquet(path)


def load_pit_snapshot(root: str | Path | None = None) -> PitDailySnapshot:
    snapshot_root = _resolve_snapshot_root(root)
    return PitDailySnapshot(
        root=snapshot_root,
        manifest=load_pit_manifest(snapshot_root),
        security_master=load_security_master(snapshot_root),
        daily_universe=load_daily_universe(snapshot_root),
        daily_bars=load_pit_daily_bars(snapshot_root),
        daily_status=load_pit_daily_status(snapshot_root),
    )
