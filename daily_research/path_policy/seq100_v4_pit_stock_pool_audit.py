from __future__ import annotations

"""PIT stock-pool audit for the frozen Seq100 v4 contract.

The pool is a membership layer only.  It never retrains a model and never
uses future ST/delisting state or intraday availability to define membership.
The executable account is delegated to the already audited v4 simulator.
"""

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map
from scipy import stats

from daily_research.path_policy import seq100_v4_economic_realizability as economic

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_v4_pit_stock_pool_audit_v1"
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / "daily_research/studies/seq100_v4_pit_stock_pool_audit_v1.json"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research/output/path_policy/studies/seq100_v4_pit_stock_pool_audit_v1"
DEFAULT_RECORD_ROOT = WORKSPACE_ROOT / "daily_research/research_records/seq100/seq100_v4_pit_stock_pool_audit_v1"
YEARS = (2023, 2024, 2025)
POOL_NAMES = ("all_pit", "csi800_pit", "quality_liquidity_pit")
RANK_MODES = ("global_rank_then_filter", "filter_then_pool_rerank")
BOOK_IDS = (
    "all_pit__global_rank_then_filter",
    "all_pit__filter_then_pool_rerank",
    "csi800_pit__global_rank_then_filter",
    "csi800_pit__filter_then_pool_rerank",
    "quality_liquidity_pit__global_rank_then_filter",
    "quality_liquidity_pit__filter_then_pool_rerank",
)
NEW_BOOK_IDS = BOOK_IDS[2:]
ECONOMIC_STUDY = WORKSPACE_ROOT / "daily_research/studies/seq100_v4_economic_realizability_v1.json"
ECONOMIC_OUTPUT = WORKSPACE_ROOT / "daily_research/output/path_policy/studies/seq100_v4_economic_realizability_v1"
ORACLE_MANIFEST = WORKSPACE_ROOT / "daily_research/output/path_policy/studies/seq100_true_label_economic_ceiling_v1/oracle_book/manifest.json"
SCHEMA = "seq100_v4_pit_stock_pool_audit/v1"
BOOK_SCHEMA = "seq100_v4_pit_pool_signal_book/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (WORKSPACE_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    raise TypeError(type(value).__name__)


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": _sha256(path), **extra}


def _scan(paths: Sequence[Path]) -> str:
    literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], union_by_name=true)"


def _qdp_paths(workspace: Path, domain: str) -> tuple[Path, ...]:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    manifest_path = dataset_manifest_for_id(root, datasets[domain], domain)
    if manifest_path is None:
        raise FileNotFoundError(f"active qdp domain: {domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = tuple(resolve_manifest_path(item.path, root=root) for item in manifest.shards)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"active qdp shards: {domain}")
    return paths


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("pool study id changed")
    period = dict(payload["period"])
    if tuple(period["years"]) != YEARS or period["maximum_consumed_date"] != "2025-12-31":
        raise ValueError("pool period changed")
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 remains forbidden")
    if payload["pool_definition"]["minute_availability_is_membership_filter"]:
        raise ValueError("minute availability cannot define a pool")
    return payload


def _base_book(config: Mapping[str, Any]) -> economic.SignalBook:
    study = economic.load_study(ECONOMIC_STUDY)
    economic.prepare_signal_book(study_path=ECONOMIC_STUDY, output_root=ECONOMIC_OUTPUT)
    return economic.SignalBook(study=study, output_root=ECONOMIC_OUTPUT)


def _candidate_keys(book: economic.SignalBook, output_root: Path) -> pd.DataFrame:
    path = output_root / "pool_inputs/candidate_keys.parquet"
    if path.is_file():
        frame = pd.read_parquet(path)
        if len(frame) == int(np.sum(book.candidate_counts)):
            return frame
    rows: list[pd.DataFrame] = []
    symbols = np.asarray(book.symbol_values, dtype=str)
    for day in range(book.day_count):
        symbol_idx = np.flatnonzero(np.isfinite(np.asarray(book.rank_panel[day, :, 0], dtype=np.float64)))
        rows.append(pd.DataFrame({"day": day, "signal_date": book.date_text(day), "symbol_idx": symbol_idx.astype(np.int32), "symbol": symbols[symbol_idx]}))
    frame = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return frame


def _pool_feature_frame(book: economic.SignalBook, output_root: Path) -> pd.DataFrame:
    """Build only as-of features needed by the quality/liquidity gate."""
    cached = output_root / "pool_inputs/quality_features.parquet"
    if cached.is_file():
        frame = pd.read_parquet(cached)
        if frame["signal_date"].max() <= "2025-12-31":
            return frame
    _candidate_keys(book, output_root)
    candidate_path = output_root / "pool_inputs/candidate_keys.parquet"
    daily = _qdp_paths(WORKSPACE_ROOT, "market_daily_raw")
    status = _qdp_paths(WORKSPACE_ROOT, "security_status")
    universe = _qdp_paths(WORKSPACE_ROOT, "universe_snapshot")
    valuation = _qdp_paths(WORKSPACE_ROOT, "valuation")
    industry = _qdp_paths(WORKSPACE_ROOT, "industry_concept")
    financial = _qdp_paths(WORKSPACE_ROOT, "financial_quarterly")
    sql = f"""
    WITH c AS (SELECT * FROM read_parquet('{str(candidate_path).replace("'", "''")}')),
    d AS (
      SELECT symbol,trade_date,amount,
             count(*) FILTER (WHERE try_cast(amount AS DOUBLE)>0) OVER
               (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS valid_amount_20,
             quantile_cont(try_cast(amount AS DOUBLE),0.5) OVER
               (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amount_median_20
      FROM {_scan(daily)} WHERE trade_date BETWEEN '2022-11-01' AND '2025-12-31'
    ),
    f AS (
      SELECT c.symbol,c.signal_date,
             f.roe_avg,f.net_profit_margin,f.net_profit_yoy,f.revenue_yoy,
             f.cash_flow_ps,f.debt_to_asset,f.report_date,f.publish_date,
             row_number() OVER (PARTITION BY c.symbol,c.signal_date ORDER BY f.publish_date DESC,f.report_date DESC) AS rn
      FROM c JOIN {_scan(financial)} f
        ON f.symbol=c.symbol
       AND try_cast(f.publish_date AS DATE)<try_cast(c.signal_date AS DATE)
       AND datediff('day',try_cast(f.report_date AS DATE),try_cast(c.signal_date AS DATE)) BETWEEN 0 AND 550
    )
    SELECT c.day,c.signal_date,c.symbol_idx,c.symbol,
           u.list_status,u.list_date,
           coalesce(s.is_st,false) AS is_st,
           coalesce(s.is_suspended,false) AS is_suspended,
           coalesce(s.is_delisted,false) AS is_delisted,
           v.circ_mv,i.industry,
           d.valid_amount_20,d.amount_median_20,
           f.roe_avg,f.net_profit_margin,f.net_profit_yoy,f.revenue_yoy,
           f.cash_flow_ps,f.debt_to_asset,f.report_date,f.publish_date
    FROM c
    LEFT JOIN {_scan(universe)} u ON u.symbol=c.symbol AND u.trade_date=c.signal_date
    LEFT JOIN {_scan(status)} s ON s.symbol=c.symbol AND s.trade_date=c.signal_date
    LEFT JOIN {_scan(valuation)} v ON v.symbol=c.symbol AND v.trade_date=c.signal_date
    LEFT JOIN {_scan(industry)} i ON i.symbol=c.symbol AND i.trade_date=c.signal_date
    LEFT JOIN d ON d.symbol=c.symbol AND d.trade_date=c.signal_date
    LEFT JOIN f ON f.symbol=c.symbol AND f.signal_date=c.signal_date AND f.rn=1
    """
    con = duckdb.connect()
    try:
        frame = con.execute(sql).fetchdf()
    finally:
        con.close()
    frame["signal_date"] = frame["signal_date"].astype(str)
    cached.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached, index=False, compression="zstd")
    return frame


def _quality_mask(book: economic.SignalBook, features: pd.DataFrame, output_root: Path) -> np.ndarray:
    frame = features.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    calendar = pd.concat(
        [pd.read_parquet(path) for path in _qdp_paths(WORKSPACE_ROOT, "trading_calendar")],
        ignore_index=True,
    )
    open_dates = np.sort(pd.to_datetime(calendar.loc[calendar["is_open"].astype(bool), "trade_date"]).unique())
    list_values = pd.to_datetime(frame["list_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    signal_values = frame["signal_date"].to_numpy(dtype="datetime64[ns]")
    list_pos = np.searchsorted(open_dates, list_values, side="left")
    signal_pos = np.searchsorted(open_dates, signal_values, side="right") - 1
    frame["listed_days"] = signal_pos - list_pos + 1
    frame["amount_median_rank"] = frame.groupby("signal_date")["amount_median_20"].rank(pct=True, method="average")
    frame["circ_mv_rank"] = frame.groupby("signal_date")["circ_mv"].rank(pct=True, method="average")
    metrics = ("roe_avg", "net_profit_margin", "net_profit_yoy", "revenue_yoy", "cash_flow_ps", "debt_to_asset")
    frame["industry"] = frame["industry"].fillna("Unknown").astype(str)
    group_counts = frame.groupby(["signal_date", "industry"], dropna=False)["symbol"].transform("count")
    frame["quality_group"] = np.where((group_counts >= 20) & ~frame["industry"].str.lower().isin(("unknown", "unavailable")), frame["industry"], "__ALL__")
    ranks: list[pd.Series] = []
    for metric in metrics:
        ranked = frame.groupby(["signal_date", "quality_group"], dropna=False)[metric].rank(pct=True, method="average", ascending=metric != "debt_to_asset")
        ranks.append(ranked)
    quality_rank = pd.concat(ranks, axis=1)
    quality_rank.columns = list(metrics)
    frame["quality_nonnull"] = quality_rank.notna().sum(axis=1)
    frame["quality_score"] = quality_rank.mean(axis=1)
    valid_fin = frame["report_date"].notna() & frame["publish_date"].notna()
    keep = (
        frame["list_status"].astype(str).str.upper().eq("L")
        & ~frame["is_st"]
        & ~frame["is_suspended"]
        & ~frame["is_delisted"]
        & (frame["listed_days"] >= 250)
        & (frame["valid_amount_20"] >= 15)
        & (frame["amount_median_rank"] >= 0.30)
        & (frame["circ_mv_rank"] >= 0.20)
        & valid_fin
        & (frame["quality_nonnull"] >= 3)
        & (frame["quality_score"] >= 0.30)
    )
    output = np.zeros((book.day_count, book.symbol_count), dtype=bool)
    for row in frame.loc[keep, ["day", "symbol_idx"]].itertuples(index=False):
        output[int(row.day), int(row.symbol_idx)] = True
    diagnostics = frame.assign(quality_keep=keep)
    diagnostics_path = output_root / "pool_inputs/quality_diagnostics.parquet"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(diagnostics_path, index=False, compression="zstd")
    return output


def _csi800_mask(book: economic.SignalBook, candidate_path: Path) -> np.ndarray:
    output = np.zeros((book.day_count, book.symbol_count), dtype=bool)
    index_paths = _qdp_paths(WORKSPACE_ROOT, "index_constituents")
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""SELECT DISTINCT c.day,c.symbol_idx FROM read_parquet(?) c JOIN {_scan(index_paths)} i
                ON i.symbol=c.symbol AND i.trade_date=c.signal_date
                WHERE i.index_symbol IN ('000300.SH','000905.SH')
                  AND try_cast(i.source_snapshot_date AS DATE)<=try_cast(c.signal_date AS DATE)
                  AND c.signal_date BETWEEN '2023-01-01' AND '2025-12-31'""",
            [str(candidate_path)],
        ).fetchall()
    finally:
        con.close()
    for day, symbol_idx in rows:
        output[int(day), int(symbol_idx)] = True
    return output


def _save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)
    return _record(path, shape=list(array.shape), dtype=str(array.dtype))


def _rank01(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.any():
        ranks = stats.rankdata(values[finite], method="average")
        result[finite] = ((ranks - 1.0) / max(int(finite.sum()) - 1, 1)).astype(np.float32)
    return result


def _build_book(
    base: economic.SignalBook,
    *,
    pool_name: str,
    rank_mode: str,
    pool_mask: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    book_id = f"{pool_name}__{rank_mode}"
    root = output_root / "signal_books" / book_id
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema") == BOOK_SCHEMA and existing.get("status") == "completed":
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    candidate_mask = np.asarray(np.isfinite(base.rank_panel[:, :, 0]), dtype=bool) & np.asarray(pool_mask, dtype=bool)
    rank_panel = np.full_like(np.asarray(base.rank_panel), np.nan, dtype=np.float32)
    orders = np.full((4, base.day_count, base.symbol_count), -1, dtype=np.int32)
    counts = candidate_mask.sum(axis=1).astype(np.int32)
    for day in range(base.day_count):
        symbols = np.flatnonzero(candidate_mask[day]).astype(np.int32)
        if not len(symbols):
            continue
        values = np.asarray(base.rank_panel[day, symbols, :7], dtype=np.float32)
        if rank_mode == "filter_then_pool_rerank":
            local = np.column_stack([_rank01(values[:, column]) for column in range(7)])
            local_dual = _rank01(np.minimum(local[:, 0], local[:, 1]))
            local_full = np.column_stack([local, local_dual, _rank01(np.asarray(base.rank_panel[day, symbols, 8])), _rank01(np.asarray(base.rank_panel[day, symbols, 9]))])
            local_full = np.column_stack([local_full, _rank01(np.minimum(local_full[:, 8], local_full[:, 9]))])
        else:
            local_full = np.asarray(base.rank_panel[day, symbols], dtype=np.float32)
        rank_panel[day, symbols] = local_full
        selectors = (local_full[:, 0], local_full[:, 1], local_full[:, 7], local_full[:, 10])
        for order_idx, score in enumerate(selectors):
            ranked = np.lexsort((symbols, -np.asarray(score, dtype=np.float64)))
            orders[order_idx, day, : len(symbols)] = symbols[ranked]
    benchmark = _pool_benchmark(base, candidate_mask)
    files = {
        "rank_panel": _save_array(root / "rank_panel.npy", rank_panel),
        "candidate_orders": _save_array(root / "candidate_orders.npy", orders),
        "candidate_counts": _save_array(root / "candidate_counts.npy", counts),
        "pool_mask": _save_array(root / "pool_mask.npy", candidate_mask),
        "benchmark_returns": _save_array(root / "benchmark_returns.npy", benchmark),
        "signal_date_idx": dict(base.manifest["files"]["signal_date_idx"]),
        "next_open_buyable": dict(base.manifest["files"]["next_open_buyable"]),
        "next_open_sellable": dict(base.manifest["files"]["next_open_sellable"]),
    }
    manifest = {
        "schema": BOOK_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "book_id": book_id,
        "pool_name": pool_name,
        "rank_mode": rank_mode,
        "pool_membership_source": "PIT signal-date fields only",
        "minute_availability_is_membership_filter": False,
        "signal_date_count": base.day_count,
        "candidate_row_count": int(counts.sum()),
        "forbidden_2026_row_count": 0,
        "files": files,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _pool_benchmark(base: economic.SignalBook, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(base.day_count, dtype=np.float64)
    recovery = float(base.terminal_recovery_fraction)
    for day in range(1, base.day_count):
        previous_idx = int(base.signal_date_idx[day - 1])
        current_idx = int(base.signal_date_idx[day])
        symbols = np.flatnonzero(mask[day - 1]).astype(np.int32)
        if not len(symbols):
            output[day] = 0.0
            continue
        previous = np.asarray(base.adjusted_close[previous_idx, symbols], dtype=np.float64)
        current = np.asarray(base.adjusted_close[current_idx, symbols], dtype=np.float64)
        valid = np.isfinite(previous) & (previous > 0) & np.isfinite(current) & (current > 0)
        values = np.where(valid, current / previous - 1.0, recovery - 1.0)
        output[day] = float(np.mean(values))
    return output


class PoolSignalBook:
    def __init__(self, base: economic.SignalBook, manifest: Mapping[str, Any], output_root: Path) -> None:
        self._base = base
        self.manifest = dict(manifest)
        self.study = base.study
        files = dict(manifest["files"])
        self.signal_date_idx = base.signal_date_idx
        self.candidate_counts = np.load(_resolve(files["candidate_counts"]["path"]), mmap_mode="r", allow_pickle=False)
        self.rank_panel = np.load(_resolve(files["rank_panel"]["path"]), mmap_mode="r", allow_pickle=False)
        self.orders = np.load(_resolve(files["candidate_orders"]["path"]), mmap_mode="r", allow_pickle=False)
        self.next_buyable = base.next_buyable
        self.next_sellable = base.next_sellable
        self.benchmark_returns = np.load(_resolve(files["benchmark_returns"]["path"]), mmap_mode="r", allow_pickle=False)
        for name in ("pack", "date_values", "symbol_values", "symbol_count", "raw_open", "daily_raw", "adjusted_open", "adjusted_close", "amount", "status_valid", "has_bar", "is_delisted", "costs", "terminal_recovery_fraction"):
            setattr(self, name, getattr(base, name))

    @property
    def day_count(self) -> int:
        return len(self.signal_date_idx)

    def date_text(self, day: int) -> str:
        return self._base.date_text(day)

    def selector_column(self, family: str) -> int:
        return self._base.selector_column(family)

    def order_column(self, family: str) -> int:
        return self._base.order_column(family)

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        count = int(self.candidate_counts[int(day)])
        return np.asarray(self.orders[self.order_column(family), int(day), :count], dtype=np.int32)

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        return float(self.rank_panel[int(day), int(symbol_idx), int(column)])

    def candidate_count(self, day: int) -> int:
        return int(self.candidate_counts[int(day)])

    def trailing_amount(self, *, signal_date_idx: int, symbol_idx: int) -> float:
        return self._base.trailing_amount(signal_date_idx=signal_date_idx, symbol_idx=symbol_idx)


def _load_book(base: economic.SignalBook, book_id: str, output_root: Path) -> PoolSignalBook:
    manifest = json.loads((output_root / "signal_books" / book_id / "manifest.json").read_text(encoding="utf-8"))
    return PoolSignalBook(base, manifest, output_root)


def prepare(*, config_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    base = _base_book(config)
    keys = _candidate_keys(base, output_root)
    all_mask = np.asarray(np.isfinite(base.rank_panel[:, :, 0]), dtype=bool)
    csi_mask = _csi800_mask(base, output_root / "pool_inputs/candidate_keys.parquet")
    quality = _quality_mask(base, _pool_feature_frame(base, output_root), output_root)
    pool_masks = {"all_pit": all_mask, "csi800_pit": all_mask & csi_mask, "quality_liquidity_pit": all_mask & quality}
    mask_records: dict[str, Any] = {}
    for name, mask in pool_masks.items():
        mask_records[name] = _save_array(output_root / "pool_inputs" / f"{name}_mask.npy", mask)
    books: dict[str, Any] = {}
    for name, mask in pool_masks.items():
        for mode in RANK_MODES:
            books[f"{name}__{mode}"] = _build_book(base, pool_name=name, rank_mode=mode, pool_mask=mask, output_root=output_root)
    manifest = {
        "schema": SCHEMA,
        "status": "prepared",
        "study_id": STUDY_ID,
        "years": list(YEARS),
        "forbidden_2026_row_count": 0,
        "candidate_key_count": len(keys),
        "pool_masks": mask_records,
        "books": {key: {"path": str(output_root / "signal_books" / key / "manifest.json"), "sha256": _sha256(output_root / "signal_books" / key / "manifest.json")} for key in books},
        "all_pit_aliases_base_economic_book": True,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _oracle_arrays() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(ORACLE_MANIFEST.read_text(encoding="utf-8"))
    raw = np.load(_resolve(manifest["files"]["raw_label_panel"]["path"]), mmap_mode="r", allow_pickle=False)
    rank = np.load(_resolve(manifest["files"]["rank_panel"]["path"]), mmap_mode="r", allow_pickle=False)
    return raw, rank, manifest


def _prediction_metrics(
    base: economic.SignalBook,
    book: PoolSignalBook | economic.SignalBook,
    mask: np.ndarray,
    raw: np.ndarray,
    oracle_rank: np.ndarray,
    *,
    book_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in YEARS:
        for horizon, column in ((10, 0), (20, 1)):
            daily: list[dict[str, Any]] = []
            for day in range(book.day_count):
                if not book.date_text(day).startswith(str(year)):
                    continue
                symbols = np.flatnonzero(mask[day] & np.isfinite(raw[day, :, column]) & np.isfinite(book.rank_panel[day, :, column])).astype(np.int32)
                if len(symbols) < 20:
                    continue
                all_symbols = np.flatnonzero(
                    np.isfinite(raw[day, :, column])
                    & np.isfinite(base.rank_panel[day, :, column])
                ).astype(np.int32)
                if len(all_symbols) < 20:
                    continue
                pred = np.asarray(book.rank_panel[day, symbols, column], dtype=np.float64)
                truth = np.asarray(raw[day, symbols, column], dtype=np.float64)
                order = np.argsort(-pred, kind="stable")
                n = max(1, math.ceil(len(symbols) * 0.05))
                top = order[:n]
                actual_order = np.argsort(-truth, kind="stable")
                actual_top = set(actual_order[:n].tolist())
                global_truth = np.asarray(raw[day, all_symbols, column], dtype=np.float64)
                global_n = max(1, math.ceil(len(all_symbols) * 0.05))
                global_actual_top = set(all_symbols[np.argsort(-global_truth, kind="stable")[:global_n]].tolist())
                global_prediction_capture = len(set(symbols[top].tolist()) & global_actual_top) / global_n
                global_pool_retention = len(set(symbols.tolist()) & global_actual_top) / global_n
                retention = len(symbols) / len(all_symbols)
                corr = stats.spearmanr(pred, truth).statistic
                daily.append({
                    "rank_ic": float(corr) if math.isfinite(float(corr)) else math.nan,
                    "top5_mfe": float(np.mean(truth[top])),
                    "tail_hit": float(len(set(top.tolist()) & actual_top) / n),
                    "global_top5_prediction_capture": float(global_prediction_capture),
                    "global_top5_pool_retention": float(global_pool_retention),
                    "opportunity_density": float(global_pool_retention / retention) if retention > 0.0 else math.nan,
                    "candidate_count": len(symbols),
                    "candidate_retention": float(retention),
                })
            if daily:
                frame = pd.DataFrame(daily)
                rows.append({
                    "book_id": book_id,
                    "year": year,
                    "horizon": horizon,
                    "rank_ic": frame["rank_ic"].mean(),
                    "top5_mfe": frame["top5_mfe"].mean(),
                    "tail_hit": frame["tail_hit"].mean(),
                    "global_top5_prediction_capture": frame["global_top5_prediction_capture"].mean(),
                    "global_top5_pool_retention": frame["global_top5_pool_retention"].mean(),
                    "opportunity_density": frame["opportunity_density"].mean(),
                    "candidate_count_mean": frame["candidate_count"].mean(),
                    "candidate_retention": frame["candidate_retention"].mean(),
                })
    return pd.DataFrame(rows)


def _task_results_for_book(
    *,
    book_id: str,
    output_root: Path,
    study_sha256: str,
    signal_sha256: str,
) -> list[dict[str, Any]]:
    task_root = output_root / "tasks" / book_id
    results: list[dict[str, Any]] = []
    for spec in economic.task_specs():
        if not economic._task_complete(
            output_root=task_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        ):
            raise RuntimeError(f"account task is incomplete: {book_id}/{spec.task_id}")
        results.append(economic._task_result(output_root=task_root, spec=spec))
    return results


def _evaluate_account_book(
    *,
    book_id: str,
    output_root: Path,
    study: Mapping[str, Any],
    study_sha256: str,
    signal_sha256: str,
) -> dict[str, Any]:
    """Run the existing economic audit on one pool's 672 replay tasks."""
    task_root = output_root / "tasks" / book_id
    results = _task_results_for_book(
        book_id=book_id,
        output_root=output_root,
        study_sha256=study_sha256,
        signal_sha256=signal_sha256,
    )
    metrics = pd.DataFrame([economic._task_metric_row(item) for item in results])
    annual = economic._annual_metric_table(results)
    configurations = economic._configuration_table(metrics, study)
    family_stats, family_daily = economic._family_statistics(
        output_root=task_root,
        configurations=configurations,
        study=study,
    )
    comparisons = economic._paired_comparisons(output_root=task_root, metrics=metrics)
    evaluation_root = output_root / "evaluation" / "accounts" / book_id
    evaluation_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "task_metrics": evaluation_root / "task_metrics.parquet",
        "annual_metrics": evaluation_root / "annual_metrics.parquet",
        "configurations": evaluation_root / "configurations.parquet",
        "family_statistics": evaluation_root / "family_statistics.parquet",
        "family_daily_median_excess": evaluation_root / "family_daily_median_excess.parquet",
        "paired_comparisons": evaluation_root / "paired_comparisons.parquet",
    }
    for path, frame in (
        (paths["task_metrics"], metrics),
        (paths["annual_metrics"], annual),
        (paths["configurations"], configurations),
        (paths["family_statistics"], family_stats),
        (paths["family_daily_median_excess"], family_daily),
        (paths["paired_comparisons"], comparisons),
    ):
        frame.to_parquet(path, index=False, compression="zstd")
    negative = family_stats[family_stats["family"].astype(str).eq(economic.NEGATIVE_CONTROL)]
    formal = family_stats[family_stats["family"].astype(str).isin(economic.FORMAL_FAMILIES)]
    robust = formal[formal["robustly_monetizable"].astype(bool)]
    accounting_invalid = bool(
        metrics["maximum_conservation_error"].astype(float).max()
        > economic.STARTING_CASH_CNY * 1.0e-9
    )
    negative_invalid = bool(negative["robustly_monetizable"].astype(bool).any())
    if accounting_invalid or negative_invalid:
        overall = "audit_invalid_due_to_negative_control_or_accounting"
    elif not robust.empty:
        overall = "pool_economically_realizable_in_decision_folds"
    elif bool(formal["gross_rectangle_count"].astype(int).gt(0).any()):
        overall = "pool_signal_exists_but_not_net_monetized"
    else:
        overall = "pool_not_monetized_by_preregistered_policy_surface"
    decision = {
        "status": "completed",
        "book_id": book_id,
        "overall": overall,
        "task_count": len(results),
        "robust_pair_count": len(robust),
        "negative_control_invalid": negative_invalid,
        "accounting_invalid": accounting_invalid,
        "single_policy_winner_selected": False,
        "period": {
            "years": list(YEARS),
            "continuous_account": True,
            "annual_reset": False,
            "maximum_consumed_date": "2025-12-31",
            "forbidden_2026_row_count": 0,
        },
        "files": {
            name: _record(path, row_count=len(frame))
            for name, path, frame in (
                ("task_metrics", paths["task_metrics"], metrics),
                ("annual_metrics", paths["annual_metrics"], annual),
                ("configurations", paths["configurations"], configurations),
                ("family_statistics", paths["family_statistics"], family_stats),
                ("family_daily_median_excess", paths["family_daily_median_excess"], family_daily),
                ("paired_comparisons", paths["paired_comparisons"], comparisons),
            )
        },
    }
    _write_json(evaluation_root / "decision.json", decision)
    decision["files"]["decision"] = _record(evaluation_root / "decision.json")
    return decision


def run_pending(*, config_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    config = _load_config(config_path)
    prepare(config_path=config_path, output_root=output_root)
    base = _base_book(config)
    study_sha = economic._study_hash(ECONOMIC_STUDY)
    specs = economic.task_specs()
    completed = 0
    newly_completed = 0
    for book_id in NEW_BOOK_IDS:
        book = _load_book(base, book_id, output_root)
        task_root = output_root / "tasks" / book_id
        for spec in specs:
            if economic._task_complete(output_root=task_root, spec=spec, study_sha256=study_sha, signal_sha256=str(book.manifest["files"]["rank_panel"]["sha256"])):
                completed += 1
                continue
            result, equity, trades, monthly = economic.simulate_task(book=book, spec=spec)
            economic._write_task(output_root=task_root, result=result, equity=equity, trades=trades, monthly=monthly, study_sha256=study_sha)
            completed += 1
            newly_completed += 1
    for book_id in BOOK_IDS[:2]:
        source = ECONOMIC_OUTPUT
        if not (source / "tasks").is_dir():
            raise FileNotFoundError(source / "tasks")
        source_study_sha = economic._study_hash(ECONOMIC_STUDY)
        source_signal = json.loads((ECONOMIC_OUTPUT / "signal_book/manifest.json").read_text(encoding="utf-8"))
        source_signal_sha = str(source_signal["files"]["rank_panel"]["sha256"])
        for spec in specs:
            if not economic._task_complete(
                output_root=source,
                spec=spec,
                study_sha256=source_study_sha,
                signal_sha256=source_signal_sha,
            ):
                raise RuntimeError(f"cannot reuse incomplete all_pit task: {spec.task_id}")
        _write_json(
            output_root / "tasks" / book_id / "reuse.json",
            {
                "source_root": str(source),
                "source_signal_book": str(ECONOMIC_OUTPUT / "signal_book/manifest.json"),
                "source_study_sha256": source_study_sha,
                "source_signal_sha256": source_signal_sha,
                "task_count": len(specs),
                "verified": True,
            },
        )
    return {"status": "completed", "completed_tasks": completed, "newly_completed_tasks": newly_completed}


def evaluate(*, config_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    config = _load_config(config_path)
    prepared = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    base = _base_book(config)
    raw, oracle_rank, oracle_manifest = _oracle_arrays()
    rows: list[pd.DataFrame] = []
    masks = {name: np.load(_resolve(record["path"]), mmap_mode="r", allow_pickle=False) for name, record in prepared["pool_masks"].items()}
    for book_id in BOOK_IDS:
        if book_id.startswith("all_pit__"):
            book = base
        else:
            book = _load_book(base, book_id, output_root)
        pool_name = book_id.split("__", 1)[0]
        metrics = _prediction_metrics(
            base,
            book,
            masks[pool_name],
            raw,
            oracle_rank,
            book_id=book_id,
        )
        if not metrics.empty:
            rows.append(metrics)
    metrics_frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics_path = output_root / "evaluation/prediction_metrics.parquet"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_parquet(metrics_path, index=False, compression="zstd")
    study = economic.load_study(ECONOMIC_STUDY)
    study_sha = economic._study_hash(ECONOMIC_STUDY)
    task_status: dict[str, int] = {}
    account_decisions: dict[str, dict[str, Any]] = {}
    account_files: dict[str, Any] = {}
    for book_id in NEW_BOOK_IDS:
        book = _load_book(base, book_id, output_root)
        task_root = output_root / "tasks" / book_id
        valid_count = sum(
            economic._task_complete(
                output_root=task_root,
                spec=spec,
                study_sha256=study_sha,
                signal_sha256=str(book.manifest["files"]["rank_panel"]["sha256"]),
            )
            for spec in economic.task_specs()
        )
        task_status[book_id] = int(valid_count)
        if valid_count == len(economic.task_specs()):
            account = _evaluate_account_book(
                book_id=book_id,
                output_root=output_root,
                study=study,
                study_sha256=study_sha,
                signal_sha256=str(book.manifest["files"]["rank_panel"]["sha256"]),
            )
            account_decisions[book_id] = account
            account_files[book_id] = account["files"]
    all_pit_reuse = {
        book_id: json.loads(
            (output_root / "tasks" / book_id / "reuse.json").read_text(encoding="utf-8")
        )
        for book_id in BOOK_IDS[:2]
        if (output_root / "tasks" / book_id / "reuse.json").is_file()
    }
    all_complete = all(value == len(economic.task_specs()) for value in task_status.values())
    decision = {
        "status": "pending_account_tasks" if not all_complete else "evaluated",
        "default_pool": "all_pit",
        "pool_decision": "not_selected_by_single_configuration",
        "minimum_candidate_gate": 480,
        "minute_availability_is_membership_filter": False,
        "forbidden_2026_row_count": int(oracle_manifest.get("period", {}).get("forbidden_2026_row_count", 0)),
        "account_task_counts": task_status,
        "all_pit_reuse": all_pit_reuse,
        "account_decisions": account_decisions,
        "prediction_metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path), "row_count": len(metrics_frame)},
    }
    _write_json(output_root / "evaluation/decision.json", decision)
    result = {"schema": SCHEMA, "status": "completed", "study_id": STUDY_ID, "decision": decision, "files": {"manifest": _record(output_root / "manifest.json"), "prediction_metrics": _record(metrics_path), "decision": _record(output_root / "evaluation/decision.json"), "account_evaluations": account_files}}
    _write_json(DEFAULT_RECORD_ROOT / "result.json", result)
    return result


def self_test(*, config_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    config = _load_config(config_path)
    base = _base_book(config)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        prepare(config_path=config_path, output_root=output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("forbidden_2026_row_count") != 0:
        raise AssertionError("2026 entered pool audit")
    masks = {name: np.load(_resolve(record["path"]), mmap_mode="r", allow_pickle=False) for name, record in manifest["pool_masks"].items()}
    all_mask = masks["all_pit"]
    for name, mask in masks.items():
        if np.any(mask & ~all_mask):
            raise AssertionError(f"pool outside candidate support: {name}")
    for mode in RANK_MODES:
        book = _load_book(base, f"all_pit__{mode}", output_root)
        if not np.array_equal(np.asarray(book.candidate_counts), np.asarray(base.candidate_counts)):
            raise AssertionError("all_pit candidate counts changed")
        if not np.allclose(
            np.asarray(book.rank_panel),
            np.asarray(base.rank_panel),
            equal_nan=True,
        ):
            raise AssertionError("all_pit ranks changed")
    return {"status": "ok", "checks": {"years": list(YEARS), "forbidden_2026": True, "minute_independent": True, "all_pit_equivalence": True}}


def status(*, config_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    _load_config(config_path)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return {"study_id": STUDY_ID, "prepared": False, "pending_books": len(BOOK_IDS)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = {book_id: len(list((output_root / "tasks" / book_id).glob("*/task_result.json"))) for book_id in NEW_BOOK_IDS}
    return {"study_id": STUDY_ID, "prepared": manifest.get("status") == "prepared", "books": len(manifest.get("books", {})), "new_book_tasks": tasks, "evaluation_complete": (output_root / "evaluation/decision.json").is_file()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seq100_v4_pit_stock_pool_audit")
    parser.add_argument("--config", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    config = Path(args.config).resolve()
    output = Path(args.output_root).resolve()
    if args.status:
        payload = status(config_path=config, output_root=output)
    elif args.prepare:
        payload = prepare(config_path=config, output_root=output)
    elif args.run_pending:
        payload = run_pending(config_path=config, output_root=output)
    elif args.evaluate:
        payload = evaluate(config_path=config, output_root=output)
    else:
        payload = self_test(config_path=config, output_root=output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
