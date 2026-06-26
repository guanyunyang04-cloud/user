from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_data_platform.lake.canonical import DEFAULT_CANONICAL_START_DATE
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import DataDomain, NUMERIC_MARKET_COLUMNS, normalize_intraday_5m_frame
from quant_data_platform.ingest.import_tdx_daily import (
    DEFAULT_FIELDS,
    DEFAULT_TQCENTER_PATH,
    _chunks,
    _date_text,
    _fetch_market_data,
    _field_to_column,
    _json_safe,
    _load_tq_client,
    _normalize_symbols,
    _read_symbols_file,
    _read_symbols_from_dataset,
    _symbol_hash,
    _tdx_date,
    _utc_now,
    _write_json,
    _write_progress,
)


@dataclass(frozen=True)
class ImportTdx5mConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    tqcenter_path: Path = DEFAULT_TQCENTER_PATH
    initialize_path: Path | None = None
    symbols: tuple[str, ...] = ()
    symbols_file: Path | None = None
    symbols_dataset_id: str = ""
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    fields: tuple[str, ...] = DEFAULT_FIELDS
    period: str = "5m"
    count: int = -1
    dividend_type: str = "none"
    fill_data: bool = False
    adjusted_flag: str = "none"
    raw_amount_unit: str = "wan_yuan"
    amount_unit_factor: float = 10000.0
    standard_amount_unit: str = "yuan"
    source_name: str = "tdx_tqcenter_5m"
    chunk_size: int = 64
    max_chunks: int = 0
    retry_count: int = 1
    retry_sleep_seconds: float = 0.5
    dry_run: bool = False
    resume: bool = True
    reuse: bool = True
    progress_path: Path | None = None

    def normalized(self) -> "ImportTdx5mConfig":
        return ImportTdx5mConfig(
            lake_root=Path(self.lake_root),
            tqcenter_path=Path(self.tqcenter_path),
            initialize_path=Path(self.initialize_path) if self.initialize_path else None,
            symbols=_normalize_symbols(self.symbols),
            symbols_file=Path(self.symbols_file) if self.symbols_file else None,
            symbols_dataset_id=str(self.symbols_dataset_id or "").strip(),
            start_date=_date_text(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=_date_text(self.end_date) if str(self.end_date or "").strip() else _date_text(pd.Timestamp.now()),
            fields=tuple(str(item).strip() for item in (self.fields or DEFAULT_FIELDS) if str(item).strip()),
            period=str(self.period or "5m").strip().lower(),
            count=int(self.count),
            dividend_type=str(self.dividend_type or "none").strip().lower(),
            fill_data=bool(self.fill_data),
            adjusted_flag=str(self.adjusted_flag or "none").strip().lower(),
            raw_amount_unit=str(self.raw_amount_unit or "wan_yuan").strip().lower(),
            amount_unit_factor=float(self.amount_unit_factor if self.amount_unit_factor is not None else 10000.0),
            standard_amount_unit=str(self.standard_amount_unit or "yuan").strip().lower(),
            source_name=str(self.source_name or "tdx_tqcenter_5m").strip().lower(),
            chunk_size=max(1, int(self.chunk_size or 64)),
            max_chunks=max(0, int(self.max_chunks or 0)),
            retry_count=max(0, int(self.retry_count or 0)),
            retry_sleep_seconds=max(0.0, float(self.retry_sleep_seconds or 0.0)),
            dry_run=bool(self.dry_run),
            resume=bool(self.resume),
            reuse=bool(self.reuse),
            progress_path=Path(self.progress_path) if self.progress_path else None,
        )


@dataclass(frozen=True)
class ImportTdx5mResult:
    status: str
    dataset_id: str = ""
    shard_count: int = 0
    row_count: int = 0
    error_count: int = 0
    symbol_count: int = 0
    manifest_path: Path | None = None
    progress_path: Path | None = None
    errors: list[str] = field(default_factory=list)


def run_import_tdx_5m(config: ImportTdx5mConfig, *, tq_client: Any | None = None) -> ImportTdx5mResult:
    cfg = config.normalized()
    if cfg.period != "5m":
        raise ValueError(f"tdx_5m_import_requires_period_5m: {cfg.period}")
    lake = ResearchDataLake(cfg.lake_root)
    progress_path = cfg.progress_path or _default_progress_path(lake)
    owned_client = tq_client is None
    module = None
    tq = tq_client
    if tq is None and not cfg.dry_run:
        module, tq = _load_tq_client(cfg)  # type: ignore[arg-type]

    try:
        symbols, symbol_source = _resolve_symbols(lake=lake, cfg=cfg, tq=tq)
        chunks = list(_chunks(symbols, cfg.chunk_size))
        if cfg.max_chunks:
            chunks = chunks[: cfg.max_chunks]
        spec = _dataset_spec(cfg, symbols=symbols, symbol_source=symbol_source)
        identity_spec = {**spec, "domain": DataDomain.MARKET_INTRADAY_5M, "sharded": True}
        identity = lake.build_domain_dataset_identity(domain=DataDomain.MARKET_INTRADAY_5M, spec=identity_spec)
        shard_dir = Path(identity["dataset_dir"]) / "shards"
        manifest_path = Path(identity["dataset_dir"]) / "tdx_5m_import_manifest.json"
        _write_progress(
            progress_path,
            {
                "event": "start",
                "symbol_count": len(symbols),
                "chunk_count": len(chunks),
                "start_date": cfg.start_date,
                "end_date": cfg.end_date,
                "dry_run": cfg.dry_run,
            },
        )
        if cfg.dry_run:
            _write_json(
                manifest_path,
                {
                    "status": "dry_run",
                    "generated_at": _utc_now(),
                    "dataset_identity": identity,
                    "spec": spec,
                    "symbol_count": len(symbols),
                    "symbol_source": symbol_source,
                    "chunk_count": len(chunks),
                    "chunks": [_chunk_plan(index, chunk) for index, chunk in enumerate(chunks, start=1)],
                },
            )
            return ImportTdx5mResult(status="dry_run", shard_count=len(chunks), symbol_count=len(symbols), manifest_path=manifest_path, progress_path=progress_path)
        if tq is None:
            raise RuntimeError("tdx_client_required")

        shard_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            record = _import_one_chunk(cfg=cfg, tq=tq, shard_dir=shard_dir, chunk=chunk, index=index, total=len(chunks), progress_path=progress_path)
            records.append(record)
            if int(record.get("error_count", 0) or 0):
                errors.append(str(record.get("error", "") or f"chunk_{index}_failed"))

        records = sorted(records, key=lambda item: str(item.get("path", "")) or str(item.get("chunk_id", "")))
        record = lake.save_sharded_domain_dataset(
            domain=DataDomain.MARKET_INTRADAY_5M,
            spec=spec,
            shard_records=records,
            source=cfg.source_name,
            reuse=cfg.reuse,
        )
        row_count = int(sum(int(item.get("row_count", 0) or 0) for item in records))
        error_count = int(sum(int(item.get("error_count", 0) or 0) for item in records))
        status = "completed" if error_count == 0 else "completed_with_errors"
        _write_json(
            manifest_path,
            {
                "status": status,
                "generated_at": _utc_now(),
                "dataset_id": record.dataset_id,
                "dataset_identity": identity,
                "spec": spec,
                "symbol_count": len(symbols),
                "symbol_source": symbol_source,
                "chunk_count": len(records),
                "row_count": row_count,
                "error_count": error_count,
                "errors": errors[:200],
                "progress_path": str(progress_path.resolve()),
            },
        )
        _write_progress(progress_path, {"event": "completed", "status": status, "dataset_id": record.dataset_id, "row_count": row_count, "error_count": error_count})
        return ImportTdx5mResult(
            status=status,
            dataset_id=record.dataset_id,
            shard_count=len(records),
            row_count=row_count,
            error_count=error_count,
            symbol_count=len(symbols),
            manifest_path=manifest_path,
            progress_path=progress_path,
            errors=errors,
        )
    finally:
        if owned_client and tq is not None:
            try:
                tq.close()
            except Exception:
                pass
        _ = module


def _import_one_chunk(
    *,
    cfg: ImportTdx5mConfig,
    tq: Any,
    shard_dir: Path,
    chunk: Sequence[str],
    index: int,
    total: int,
    progress_path: Path,
) -> dict[str, Any]:
    chunk_id = f"tdx_5m_{index:05d}_{_symbol_hash(chunk)[:12]}"
    target_path = shard_dir / f"{chunk_id}.parquet"
    _write_progress(progress_path, {"event": "chunk_start", "index": index, "total": total, "symbol_count": len(chunk), "chunk_id": chunk_id})
    if cfg.resume and target_path.exists():
        frame = pd.read_parquet(target_path, columns=["trade_date", "symbol", "bar_time"])
        return _shard_record(target_path=target_path, frame=frame, chunk=chunk, chunk_id=chunk_id, status="stored", materialization="resume_hit")

    try:
        payload = _fetch_market_data(tq=tq, cfg=cfg, chunk=chunk)  # type: ignore[arg-type]
        frame = _tdx_5m_market_data_to_frame(payload, source=cfg.source_name, adjusted_flag=cfg.adjusted_flag, amount_unit_factor=cfg.amount_unit_factor)
        frame = _filter_intraday_frame(frame, start_date=cfg.start_date, end_date=cfg.end_date)
        frame.to_parquet(target_path, index=False)
        record = _shard_record(target_path=target_path, frame=frame, chunk=chunk, chunk_id=chunk_id, status="stored", materialization="fetched")
        _write_progress(
            progress_path,
            {
                "event": "chunk_stored",
                "index": index,
                "total": total,
                "chunk_id": chunk_id,
                "row_count": record["row_count"],
                "start_date": record["start_date"],
                "end_date": record["end_date"],
                "path": str(target_path.resolve()),
            },
        )
        return record
    except Exception as exc:  # noqa: BLE001
        message = f"{chunk_id}: {type(exc).__name__}: {exc}"
        _write_progress(progress_path, {"event": "chunk_error", "index": index, "total": total, "chunk_id": chunk_id, "error": message})
        return {
            "status": "error",
            "chunk_id": chunk_id,
            "path": str(target_path.resolve()),
            "row_count": 0,
            "error_count": 1,
            "error": message,
            "symbol_count": len(chunk),
            "symbols_sha256": _symbol_hash(chunk),
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
        }


def _tdx_5m_market_data_to_frame(
    payload: Mapping[str, pd.DataFrame],
    *,
    source: str,
    adjusted_flag: str,
    amount_unit_factor: float = 10000.0,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    lookup = {str(key).strip().lower(): key for key in payload.keys()}
    for field in DEFAULT_FIELDS:
        original_key = lookup.get(field.lower())
        if original_key is None:
            continue
        raw = payload.get(original_key)
        if not isinstance(raw, pd.DataFrame):
            continue
        column = _field_to_column(field)
        long = _wide_intraday_frame_to_long(raw, value_name=column)
        merged = long if merged is None else merged.merge(long, on=["trade_date", "symbol", "bar_time"], how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["trade_date", "symbol", "bar_time"])
    if "amount" in merged.columns:
        merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce") * float(amount_unit_factor)
    merged["source"] = source
    merged["adjusted_flag"] = adjusted_flag
    out = normalize_intraday_5m_frame(merged, source=source, adjusted_flag=adjusted_flag, require_columns=False)
    if out.empty:
        return out
    all_missing = out[NUMERIC_MARKET_COLUMNS].isna().all(axis=1)
    out = out.loc[~all_missing].copy()
    out = out.drop_duplicates(["symbol", "trade_date", "bar_time", "source"], keep="last")
    return out.sort_values(["trade_date", "symbol", "bar_time"]).reset_index(drop=True)


def _wide_intraday_frame_to_long(frame: pd.DataFrame, *, value_name: str) -> pd.DataFrame:
    working = frame.copy()
    timestamps = pd.to_datetime(working.index, errors="coerce")
    working = working.loc[timestamps.notna()].copy()
    timestamps = pd.to_datetime(working.index, errors="coerce")
    working.index = timestamps
    working.index.name = "bar_datetime"
    try:
        stacked = working.stack(future_stack=True)
    except TypeError:
        stacked = working.stack(dropna=False)
    long = stacked.rename(value_name).reset_index()
    long.columns = ["bar_datetime", "symbol", value_name]
    ts = pd.to_datetime(long["bar_datetime"], errors="coerce")
    long["trade_date"] = ts.dt.strftime("%Y-%m-%d")
    long["bar_time"] = ts.dt.strftime("%H:%M:%S")
    long["symbol"] = long["symbol"].astype(str).str.strip().str.upper()
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long[["trade_date", "symbol", "bar_time", value_name]]


def _filter_intraday_frame(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = dates.notna()
    if start_date:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date:
        mask &= dates <= pd.Timestamp(end_date)
    return frame.loc[mask].sort_values(["trade_date", "symbol", "bar_time"]).reset_index(drop=True)


def _resolve_symbols(*, lake: ResearchDataLake, cfg: ImportTdx5mConfig, tq: Any | None) -> tuple[tuple[str, ...], str]:
    if cfg.symbols:
        return cfg.symbols, "config_symbols"
    if cfg.symbols_file:
        return _read_symbols_file(cfg.symbols_file), f"symbols_file:{cfg.symbols_file}"
    if cfg.symbols_dataset_id:
        return _read_symbols_from_dataset(lake, cfg.symbols_dataset_id), f"symbols_dataset:{cfg.symbols_dataset_id}"
    if tq is None:
        return (), "tq_stock_list_unavailable_dry_run"
    symbols = _normalize_symbols(tq.get_stock_list())
    if not symbols:
        raise ValueError("tdx_stock_list_empty; pass --symbols, --symbols-file, or --symbols-dataset-id")
    return symbols, "tq_current_stock_list"


def _dataset_spec(cfg: ImportTdx5mConfig, *, symbols: Sequence[str], symbol_source: str) -> dict[str, Any]:
    return {
        "provider_plan": "tdx_5m_incremental_import",
        "canonical_start_date": DEFAULT_CANONICAL_START_DATE,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "source_name": cfg.source_name,
        "tqcenter_path": str(cfg.tqcenter_path),
        "initialize_path": str(cfg.initialize_path or "generated_session_file"),
        "period": cfg.period,
        "count": cfg.count,
        "dividend_type": cfg.dividend_type,
        "fill_data": cfg.fill_data,
        "adjusted_flag": cfg.adjusted_flag,
        "raw_amount_unit": cfg.raw_amount_unit,
        "amount_unit_factor": cfg.amount_unit_factor,
        "standard_amount_unit": cfg.standard_amount_unit,
        "amount_unit_policy": "multiply_by_factor" if float(cfg.amount_unit_factor) != 1.0 else "as_is",
        "fields": list(cfg.fields),
        "symbol_source": symbol_source,
        "symbol_count": len(symbols),
        "symbols_sha256": _symbol_hash(symbols),
        "chunk_size": cfg.chunk_size,
        "intraday_bar_policy": "tdx_native_5m",
        "expected_full_day_bars": 48,
    }


def _shard_record(*, target_path: Path, frame: pd.DataFrame, chunk: Sequence[str], chunk_id: str, status: str, materialization: str) -> dict[str, Any]:
    starts = pd.to_datetime(frame["trade_date"], errors="coerce") if "trade_date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "status": status,
        "chunk_id": chunk_id,
        "path": str(target_path.resolve()),
        "row_count": int(len(frame)),
        "error_count": 0,
        "symbol_count": int(len(chunk)),
        "symbols_sha256": _symbol_hash(chunk),
        "start_date": starts.min().strftime("%Y-%m-%d") if not starts.dropna().empty else "",
        "end_date": starts.max().strftime("%Y-%m-%d") if not starts.dropna().empty else "",
        "materialization": materialization,
        "bar_interval": "5m",
    }


def _chunk_plan(index: int, chunk: Sequence[str]) -> dict[str, Any]:
    return {
        "index": index,
        "symbol_count": len(chunk),
        "symbols_sha256": _symbol_hash(chunk),
        "first_symbol": str(chunk[0]) if chunk else "",
        "last_symbol": str(chunk[-1]) if chunk else "",
    }


def _default_progress_path(lake: ResearchDataLake) -> Path:
    return lake.root / "data_platform" / "tdx_5m_imports" / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import local TDX/tqcenter native 5m OHLCV into the research data lake.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--tqcenter-path", default=str(DEFAULT_TQCENTER_PATH))
    parser.add_argument("--initialize-path", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--symbols-dataset-id", default="")
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    parser.add_argument("--period", default="5m")
    parser.add_argument("--count", type=int, default=-1)
    parser.add_argument("--dividend-type", default="none", choices=("none", "front", "back"))
    parser.add_argument("--fill-data", action="store_true", default=False)
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--raw-amount-unit", default="wan_yuan")
    parser.add_argument("--amount-unit-factor", type=float, default=10000.0)
    parser.add_argument("--standard-amount-unit", default="yuan")
    parser.add_argument("--source-name", default="tdx_tqcenter_5m")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    parser.add_argument("--progress-path", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_import_tdx_5m(
        ImportTdx5mConfig(
            lake_root=Path(args.lake_root),
            tqcenter_path=Path(args.tqcenter_path),
            initialize_path=Path(args.initialize_path) if str(args.initialize_path or "").strip() else None,
            symbols=tuple(part.strip() for part in str(args.symbols or "").split(",") if part.strip()),
            symbols_file=Path(args.symbols_file) if str(args.symbols_file or "").strip() else None,
            symbols_dataset_id=args.symbols_dataset_id,
            start_date=args.start_date,
            end_date=args.end_date,
            fields=tuple(part.strip() for part in str(args.fields or "").split(",") if part.strip()),
            period=args.period,
            count=args.count,
            dividend_type=args.dividend_type,
            fill_data=args.fill_data,
            adjusted_flag=args.adjusted_flag,
            raw_amount_unit=args.raw_amount_unit,
            amount_unit_factor=args.amount_unit_factor,
            standard_amount_unit=args.standard_amount_unit,
            source_name=args.source_name,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            retry_count=args.retry_count,
            retry_sleep_seconds=args.retry_sleep_seconds,
            dry_run=args.dry_run,
            resume=args.resume,
            reuse=args.reuse,
            progress_path=Path(args.progress_path) if str(args.progress_path or "").strip() else None,
        )
    )
    payload = {
        "status": result.status,
        "dataset_id": result.dataset_id,
        "shard_count": result.shard_count,
        "row_count": result.row_count,
        "error_count": result.error_count,
        "symbol_count": result.symbol_count,
        "manifest_path": str(result.manifest_path.resolve()) if result.manifest_path else "",
        "progress_path": str(result.progress_path.resolve()) if result.progress_path else "",
    }
    if args.json:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: {result.dataset_id or result.manifest_path}")
    return 0 if result.status in {"completed", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
