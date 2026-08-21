"""PIT history context operations."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.core.paths import qdp_paths
from quantlab.data.core.security_status import st_status_from_name
from quantlab.data.providers import BaostockProvider
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    qdp_v2_root,
)
from quantlab.data.qdp_v2.repair import (
    resolve_active_domain,
)

from .config import (
    ARCHIVE_RECENT,
    ARCHIVE_RECOVERY,
    ARCHIVE_ROOT,
    DEFAULT_START_DATE,
    PRE_ARCHIVE_SECURITIES,
    _is_mainboard,
)


class PitHistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PitHistoryContext:
    workspace: Path
    root: Path
    runtime: Path
    start_date: str
    end_date: str


def _context(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> PitHistoryContext:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    root = qdp_v2_root(workspace).resolve()
    runtime = (qdp_paths(workspace).data_dir / "qdp_runtime" / "pit_history_restore").resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if start > end:
        raise ValueError(f"pit_history_invalid_range:{start}>{end}")
    return PitHistoryContext(workspace, root, runtime, start, end)


def _read_active_symbols(ctx: PitHistoryContext, domain: str) -> set[str]:
    current = resolve_active_domain(domain, workspace_root=ctx.workspace)
    column = "current_symbol" if domain == "security_identity" else "symbol"
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "symbol_inventory_spill",
        threads=2,
    ) as con:
        frame = con.execute(
            f"SELECT DISTINCT cast({column} AS VARCHAR) AS symbol FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in current.shard_paths]],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "symbol_inventory_spill", ignore_errors=True)
    return set(frame["symbol"].astype(str).str.upper())


def _read_symbol_history_symbols(ctx: PitHistoryContext) -> set[str]:
    current = resolve_active_domain("symbol_history", workspace_root=ctx.workspace)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "symbol_history_inventory_spill",
        threads=2,
    ) as con:
        frame = con.execute(
            "SELECT DISTINCT upper(cast(symbol AS VARCHAR)) AS symbol FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in current.shard_paths]],
        ).fetchdf()
    shutil.rmtree(
        ctx.runtime / "symbol_history_inventory_spill",
        ignore_errors=True,
    )
    return set(frame["symbol"].astype(str))


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _valid_parquet(path: Path, required: Sequence[str]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import pyarrow.parquet as pq

        return set(required).issubset(pq.read_schema(path).names)
    except Exception:
        return False


def _archive_paths(ctx: PitHistoryContext) -> dict[str, Path]:
    base = ctx.workspace / ARCHIVE_ROOT
    recent_root = base / ARCHIVE_RECENT.name
    latest_path = recent_root / "latest_manifest.json"
    if not latest_path.is_file():
        raise PitHistoryError(f"pit_history_archive_manifest_missing:{latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    snapshot_id = str(latest.get("snapshot_id", "")).strip()
    recent = recent_root / snapshot_id
    recovery = base / ARCHIVE_RECOVERY.name
    required = {
        "recent_bars": recent / "daily_bars.parquet",
        "recent_metrics": recent / "daily_metrics.parquet",
        "recent_universe": recent / "daily_universe.parquet",
        "recent_names": recent / "raw_daily_stock_lists.parquet",
        "recent_master": recent / "security_master.parquet",
        "recovery_bars": recovery / "daily_bars.parquet",
        "recovery_names": recovery / "cache" / "daily_stock_lists",
        "recovery_master": recovery / "cache" / "security_master.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise PitHistoryError(f"pit_history_archive_incomplete:{missing}")
    return required


def _workspace_relative_path(path: Path, workspace: Path) -> str:
    """Serialize provenance without binding the manifest to one machine path."""

    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        # External evidence is still useful, but keep an absolute fallback when
        # it genuinely lives outside the project boundary.
        return str(path.resolve())


def _local_stock_basic(ctx: PitHistoryContext) -> pd.DataFrame:
    paths = _archive_paths(ctx)
    rows: list[pd.DataFrame] = []
    for key in ("recovery_master", "recent_master"):
        frame = pd.read_parquet(paths[key]).copy()
        frame = frame.loc[frame.get("security_type", "1").astype(str).eq("1")]
        rows.append(
            pd.DataFrame(
                {
                    "symbol": frame["code"].astype(str).str.upper(),
                    "name": frame.get("name", "").fillna("").astype(str),
                    "list_date": frame.get("ipo_date", "").fillna("").astype(str),
                    "delist_date": frame.get("out_date", "").fillna("").astype(str),
                    "priority": 1 if key == "recovery_master" else 2,
                }
            )
        )
    try:
        current = resolve_active_domain("security_identity", workspace_root=ctx.workspace)
        with open_guarded_duckdb(
            temp_directory=ctx.runtime / "stock_basic_spill",
            threads=2,
        ) as con:
            identity = con.execute(
                "SELECT cast(current_symbol AS VARCHAR) symbol, "
                "cast(issuer_name AS VARCHAR) name, cast(list_date AS VARCHAR) list_date "
                "FROM read_parquet(?, union_by_name=true)",
                [[str(item) for item in current.shard_paths]],
            ).fetchdf()
        shutil.rmtree(ctx.runtime / "stock_basic_spill", ignore_errors=True)
        identity["delist_date"] = ""
        identity["priority"] = 3
        rows.append(identity)
    except Exception:
        shutil.rmtree(ctx.runtime / "stock_basic_spill", ignore_errors=True)
    rows.append(
        pd.DataFrame(
            PRE_ARCHIVE_SECURITIES,
            columns=["symbol", "name", "list_date", "delist_date"],
        ).assign(priority=4)
    )
    combined = pd.concat(rows, ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.loc[combined["symbol"].map(_is_mainboard)].copy()
    combined = combined.sort_values(["symbol", "priority"])

    def last_text(values: pd.Series) -> str:
        valid = [str(item).strip() for item in values if str(item).strip() not in {"", "nan", "NaT"}]
        return valid[-1] if valid else ""

    result = (
        combined.groupby("symbol", as_index=False)
        .agg(
            name=("name", last_text),
            list_date=(
                "list_date",
                lambda values: min(
                    [str(item)[:10] for item in values if str(item).strip() not in {"", "nan", "NaT"}],
                    default="",
                ),
            ),
            delist_date=("delist_date", last_text),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    result["type"] = "1"
    result["status"] = np.where(result["delist_date"].ne(""), "0", "1")
    result["list_status"] = result["status"]
    result["trade_date"] = ctx.end_date
    result["board"] = "1"
    result["is_st"] = result["name"].map(st_status_from_name).astype("boolean")
    result["is_suspended"] = False
    result["is_delisted"] = result["delist_date"].ne("")
    result["status_reason"] = ""
    return result


def _stock_basic(ctx: PitHistoryContext, provider: BaostockProvider) -> pd.DataFrame:
    path = ctx.runtime / "stock_basic.parquet"
    if _valid_parquet(path, ("symbol", "list_date", "delist_date")):
        frame = pd.read_parquet(path)
    else:
        frame = _local_stock_basic(ctx)
        if frame.empty:
            frame = provider.fetch_stock_basic_snapshot(trade_date=ctx.end_date).copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.loc[frame["symbol"].map(_is_mainboard)].copy()
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="coerce")
    frame = frame.loc[
        frame["list_date"].notna()
        & frame["list_date"].le(ctx.end_date)
        & (frame["delist_date"].isna() | frame["delist_date"].ge(ctx.start_date))
    ].copy()
    frame["list_date"] = frame["list_date"].dt.strftime("%Y-%m-%d")
    frame["delist_date"] = frame["delist_date"].dt.strftime("%Y-%m-%d").fillna("")
    frame = frame.drop_duplicates("symbol", keep="last").sort_values("symbol")
    _atomic_parquet(frame, path)
    return frame.reset_index(drop=True)


def inventory_pit_history(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
) -> dict[str, Any]:
    ctx = _context(
        start_date=start_date,
        end_date=end_date,
        workspace_root=workspace_root,
    )
    source = provider or BaostockProvider()
    owned = provider is None
    try:
        basic = _stock_basic(ctx, source)
    finally:
        if owned:
            source.close()
    active_daily = _read_active_symbols(ctx, "market_daily_raw")
    active_lifecycle = _read_active_symbols(ctx, "universe_snapshot")
    missing = sorted(set(basic["symbol"].astype(str)) - active_lifecycle)
    missing_frame = basic.loc[basic["symbol"].isin(missing)].copy()
    payload = {
        "status": "planned" if missing else "already_complete",
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "pit_mainboard_symbol_count": int(len(basic)),
        "active_daily_symbol_count": int(len(active_daily)),
        "active_lifecycle_symbol_count": int(len(active_lifecycle)),
        "missing_symbol_count": int(len(missing)),
        "missing_delisted_symbol_count": int(missing_frame["delist_date"].astype(str).ne("").sum()),
        "missing_current_or_st_symbol_count": int(missing_frame["delist_date"].astype(str).eq("").sum()),
        "missing_symbols": missing,
        "runtime": str(ctx.runtime),
    }
    atomic_write_json(ctx.runtime / "inventory.json", payload)
    return payload
