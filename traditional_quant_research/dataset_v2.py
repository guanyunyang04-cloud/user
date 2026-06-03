"""Read-only loaders for v2 point-in-time daily research snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_V2_SNAPSHOT_ROOT = Path("traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit")

DAILY_SIZE_COLUMNS = [
    "date",
    "code",
    "total_market_cap",
    "float_market_cap",
    "total_share",
    "float_share",
    "free_share",
    "market_cap_unit",
    "share_unit",
    "source",
    "source_trade_date",
]


@dataclass(frozen=True)
class PitDailySnapshot:
    root: Path
    manifest: dict[str, Any]
    security_master: pd.DataFrame
    daily_universe: pd.DataFrame
    daily_bars: pd.DataFrame
    daily_status: pd.DataFrame
    daily_metrics: pd.DataFrame
    stock_industry: pd.DataFrame
    daily_size: pd.DataFrame


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


def load_pit_stock_industry(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "stock_industry.parquet"
    columns = ["date", "code", "name_on_date", "industry", "industry_classification", "industry_update_date", "source"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = _filter_frame(pd.read_parquet(path), start_date=start_date, end_date=end_date, symbols=symbols)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def load_pit_daily_metrics(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_metrics.parquet"
    columns = ["date", "code", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "source"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = _filter_frame(pd.read_parquet(path), start_date=start_date, end_date=end_date, symbols=symbols)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def load_pit_daily_size(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
) -> pd.DataFrame:
    path = _resolve_snapshot_root(root) / "daily_size.parquet"
    if not path.exists():
        return pd.DataFrame(columns=DAILY_SIZE_COLUMNS)
    frame = _filter_frame(pd.read_parquet(path), start_date=start_date, end_date=end_date, symbols=symbols)
    for column in DAILY_SIZE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[DAILY_SIZE_COLUMNS]


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
        daily_metrics=load_pit_daily_metrics(snapshot_root),
        stock_industry=load_pit_stock_industry(snapshot_root),
        daily_size=load_pit_daily_size(snapshot_root),
    )


def load_tradeable_panel(
    root: str | Path | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_industry: bool = False,
    include_metrics: bool = False,
    include_size: bool = False,
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
    panel = panel.loc[panel["is_tradeable"]].copy()
    if include_industry and not panel.empty:
        industry = load_pit_stock_industry(
            root,
            start_date=start_date,
            end_date=end_date,
            symbols=sorted(panel["code"].unique().tolist()),
        )
        if not industry.empty:
            industry = industry.rename(
                columns={
                    "name_on_date": "industry_name_on_date",
                    "source": "industry_source",
                }
            )
            panel = panel.merge(industry, on=["date", "code"], how="left")
    if include_metrics and not panel.empty:
        metrics = load_pit_daily_metrics(
            root,
            start_date=start_date,
            end_date=end_date,
            symbols=sorted(panel["code"].unique().tolist()),
        )
        if not metrics.empty:
            metrics = metrics.rename(columns={"source": "metrics_source"})
            panel = panel.merge(metrics, on=["date", "code"], how="left")
    if include_size and not panel.empty:
        size = load_pit_daily_size(
            root,
            start_date=start_date,
            end_date=end_date,
            symbols=sorted(panel["code"].unique().tolist()),
        )
        if not size.empty:
            size = size.rename(columns={"source": "size_source"})
            panel = panel.merge(size, on=["date", "code"], how="left")
    return panel.sort_values(["date", "code"]).reset_index(drop=True)
