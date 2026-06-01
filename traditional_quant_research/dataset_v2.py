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


def load_quality_report(root: str | Path | None = None) -> dict[str, Any]:
    snapshot_root = _resolve_snapshot_root(root)
    path = snapshot_root / "quality_report.json"
    if not path.exists():
        raise FileNotFoundError(f"quality report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_security_master(root: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "security_master.parquet"
    if not path.exists():
        raise FileNotFoundError(f"security master not found: {path}")
    return pd.read_parquet(path)


def _filter_frame(
    frame: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    if "date" in output.columns and (start_date is not None or end_date is not None):
        dates = pd.to_datetime(output["date"])
        if start_date is not None:
            output = output.loc[dates >= pd.Timestamp(start_date)]
            dates = pd.to_datetime(output["date"])
        if end_date is not None:
            output = output.loc[dates <= pd.Timestamp(end_date)]
    if symbols is not None and "code" in output.columns:
        output = output.loc[output["code"].isin(set(symbols))]
    return output.reset_index(drop=True)


def load_daily_universe(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    tradeable_only: bool = False,
) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily universe not found: {path}")
    frame = _filter_frame(pd.read_parquet(path), start_date=start_date, end_date=end_date)
    if tradeable_only:
        frame = frame.loc[frame["is_tradeable"]].reset_index(drop=True)
    return frame


def load_pit_daily_bars(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_bars.parquet"
    if not path.exists():
        raise FileNotFoundError(f"daily bars not found: {path}")
    return _filter_frame(pd.read_parquet(path), start_date=start_date, end_date=end_date, symbols=symbols)


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


def load_tradeable_panel(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    universe = load_daily_universe(root, start_date=start_date, end_date=end_date, tradeable_only=True)
    if universe.empty:
        return pd.DataFrame()
    bars = load_pit_daily_bars(
        root,
        start_date=start_date,
        end_date=end_date,
        symbols=sorted(universe["code"].unique().tolist()),
    )
    panel = universe.merge(bars, on=["date", "code"], how="inner")
    return panel.loc[panel["is_tradeable"]].sort_values(["date", "code"]).reset_index(drop=True)
