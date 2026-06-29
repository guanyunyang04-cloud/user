from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import DataDomain, DOMAIN_STANDARD_COLUMNS, normalize_intraday_1m_frame
from quant_data_platform.ingest.import_external_quant_zip import normalize_intraday_1m_to_mootdx_240_frame

_PROGRESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class NormalizeIntraday1mContractConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_dataset_id: str = ""
    start_date: str = ""
    end_date: str = ""
    years: tuple[int, ...] = ()
    max_shards: int = 0
    workers: int = 1
    progress_path: Path | None = None
    dry_run: bool = False
    resume: bool = True
    reuse: bool = True
    engine: str = "duckdb"

    def normalized(self) -> "NormalizeIntraday1mContractConfig":
        engine = str(self.engine or "duckdb").strip().lower()
        if engine not in {"duckdb", "pandas"}:
            raise ValueError(f"unsupported_normalization_engine: {self.engine}")
        return NormalizeIntraday1mContractConfig(
            lake_root=Path(self.lake_root),
            source_dataset_id=str(self.source_dataset_id or "").strip(),
            start_date=str(self.start_date or "").strip(),
            end_date=str(self.end_date or "").strip(),
            years=tuple(sorted({int(item) for item in self.years})),
            max_shards=max(0, int(self.max_shards or 0)),
            workers=max(1, int(self.workers or 1)),
            progress_path=Path(self.progress_path) if self.progress_path else None,
            dry_run=bool(self.dry_run),
            resume=bool(self.resume),
            reuse=bool(self.reuse),
            engine=engine,
        )


@dataclass(frozen=True)
class NormalizeIntraday1mContractResult:
    status: str
    dataset_id: str = ""
    shard_count: int = 0
    row_count: int = 0
    source_row_count: int = 0
    removed_0930_rows: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_intraday_1m_contract(config: NormalizeIntraday1mContractConfig) -> NormalizeIntraday1mContractResult:
    cfg = config.normalized()
    if not cfg.source_dataset_id:
        raise ValueError("source_dataset_id_required")
    lake = ResearchDataLake(cfg.lake_root)
    source_metadata = lake.describe_dataset(cfg.source_dataset_id)
    source_domain = str(dict(source_metadata.get("parameters", {}) or {}).get("domain", "") or "")
    if source_domain != DataDomain.MARKET_INTRADAY_1M:
        raise ValueError(f"source_domain_mismatch: {source_domain}; expected={DataDomain.MARKET_INTRADAY_1M}")
    manifest_path = Path(str(dict(source_metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
    if not manifest_path.exists():
        raise FileNotFoundError(f"source_shard_manifest_not_found: {manifest_path}")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_shards = [
        dict(item)
        for item in list(source_manifest.get("shards", []) or [])
        if str(dict(item).get("status", "") or "") == "stored" and int(dict(item).get("row_count", 0) or 0) > 0
    ]
    if cfg.years:
        source_shards = [item for item in source_shards if _shard_intersects_years(item, set(cfg.years))]
    if cfg.max_shards:
        source_shards = source_shards[: cfg.max_shards]

    target_spec = _target_spec(cfg, source_metadata=source_metadata)
    identity = lake.build_domain_dataset_identity(domain=DataDomain.MARKET_INTRADAY_1M, spec={**target_spec, "domain": DataDomain.MARKET_INTRADAY_1M, "sharded": True})
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    progress_path = cfg.progress_path or lake.root / "canonical" / "imports" / f"intraday_1m_contract_normalize_{datetime.now().strftime('%Y%m%d_%H%M%S')}_progress.jsonl"
    _write_progress(
        progress_path,
        {
            "event": "start",
            "source_dataset_id": cfg.source_dataset_id,
            "target_dataset_id": str(identity.get("dataset_id", "") or ""),
            "source_shard_count": len(source_shards),
            "source_row_count": sum(int(item.get("row_count", 0) or 0) for item in source_shards),
            "workers": cfg.workers,
            "engine": cfg.engine,
            "dry_run": cfg.dry_run,
        },
    )
    if cfg.dry_run:
        _write_progress(progress_path, {"event": "completed", "dry_run": True, "shard_count": len(source_shards)})
        return NormalizeIntraday1mContractResult(
            status="dry_run",
            shard_count=len(source_shards),
            source_row_count=sum(int(item.get("row_count", 0) or 0) for item in source_shards),
        )

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if cfg.workers <= 1 or len(source_shards) <= 1:
        for index, source_record in enumerate(source_shards, start=1):
            try:
                records.append(
                    _normalize_one_shard(
                        shard_dir=shard_dir,
                        source_record=source_record,
                        cfg=cfg,
                        index=index,
                        total=len(source_shards),
                        progress_path=progress_path,
                    )
                )
            except Exception as exc:
                errors.append(f"{source_record.get('path', '')}: {exc}")
                break
    else:
        with ThreadPoolExecutor(max_workers=min(cfg.workers, len(source_shards))) as executor:
            futures = {
                executor.submit(
                    _normalize_one_shard,
                    shard_dir=shard_dir,
                    source_record=source_record,
                    cfg=cfg,
                    index=index,
                    total=len(source_shards),
                    progress_path=progress_path,
                ): source_record
                for index, source_record in enumerate(source_shards, start=1)
            }
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    source_record = futures[future]
                    errors.append(f"{source_record.get('path', '')}: {exc}")
                    break
    if errors:
        raise RuntimeError("; ".join(errors[:10]))

    records = sorted(records, key=lambda item: int(item.get("normalization_index", 0) or 0))
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec=target_spec,
        shard_records=records,
        source="intraday_1m_contract_normalization",
        reuse=cfg.reuse,
    )
    _write_progress(
        progress_path,
        {
            "event": "dataset_registered",
            "dataset_id": record.dataset_id,
            "shard_count": len(records),
            "row_count": sum(int(item.get("row_count", 0) or 0) for item in records),
            "removed_0930_rows": sum(int(item.get("removed_0930_rows", 0) or 0) for item in records),
        },
    )
    _write_progress(progress_path, {"event": "completed", "dataset_id": record.dataset_id})
    return NormalizeIntraday1mContractResult(
        status="completed",
        dataset_id=record.dataset_id,
        shard_count=len(records),
        row_count=sum(int(item.get("row_count", 0) or 0) for item in records),
        source_row_count=sum(int(item.get("source_row_count", 0) or 0) for item in records),
        removed_0930_rows=sum(int(item.get("removed_0930_rows", 0) or 0) for item in records),
        errors=[],
    )


def _normalize_one_shard(
    *,
    shard_dir: Path,
    source_record: dict[str, Any],
    cfg: NormalizeIntraday1mContractConfig,
    index: int,
    total: int,
    progress_path: Path,
) -> dict[str, Any]:
    source_path = Path(str(source_record.get("path", "") or ""))
    if not source_path.exists():
        raise FileNotFoundError(f"source_shard_not_found: {source_path}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    target_path = shard_dir / f"{_safe_stem(source_path.stem)}__mootdx_240.parquet"
    t0 = time.perf_counter()
    _write_progress(progress_path, {"event": "shard_start", "index": index, "total": total, "source_path": str(source_path.resolve())})
    if cfg.resume and target_path.exists():
        data = pd.read_parquet(target_path, columns=["trade_date", "symbol", "bar_time"])
        record = _shard_record(
            target_path=target_path,
            data=data,
            source_record=source_record,
            source_path=source_path,
            source_row_count=int(source_record.get("row_count", 0) or 0),
            removed_0930_rows=max(0, int(source_record.get("row_count", 0) or 0) - len(data)),
            materialization="resume_hit",
            index=index,
        )
        _write_progress(progress_path, {"event": "shard_stored", "index": index, "total": total, "row_count": record["row_count"], "materialization": "resume_hit", "elapsed_sec": round(time.perf_counter() - t0, 3)})
        return record

    if cfg.engine == "duckdb":
        source_row_count = _normalize_one_shard_duckdb(
            source_path=source_path,
            target_path=target_path,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
        )
        normalized = pd.read_parquet(target_path, columns=["trade_date", "symbol", "bar_time"])
    else:
        raw = pd.read_parquet(source_path)
        source_row_count = len(raw)
        normalized = normalize_intraday_1m_frame(raw, source="external_quant_zip", adjusted_flag="none", require_columns=False)
        normalized = normalize_intraday_1m_to_mootdx_240_frame(normalized)
        normalized = _filter_date_window(normalized, start_date=cfg.start_date, end_date=cfg.end_date)
        normalized = normalized.loc[:, DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_INTRADAY_1M]]
        normalized.to_parquet(target_path, index=False)
    record = _shard_record(
        target_path=target_path,
        data=normalized,
        source_record=source_record,
        source_path=source_path,
        source_row_count=source_row_count,
        removed_0930_rows=max(0, source_row_count - len(normalized)),
        materialization="normalized",
        index=index,
    )
    _write_progress(
        progress_path,
        {
            "event": "shard_stored",
            "index": index,
            "total": total,
            "path": str(target_path.resolve()),
            "source_path": str(source_path.resolve()),
            "source_row_count": source_row_count,
            "row_count": record["row_count"],
            "removed_0930_rows": record["removed_0930_rows"],
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "materialization": "normalized",
            "engine": cfg.engine,
        },
    )
    return record


def _normalize_one_shard_duckdb(
    *,
    source_path: Path,
    target_path: Path,
    start_date: str,
    end_date: str,
) -> int:
    import duckdb

    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = source_path.as_posix().replace("'", "''")
    target_text = target_path.as_posix().replace("'", "''")
    filters: list[str] = []
    if start_date:
        escaped_start = str(start_date).replace("'", "''")
        filters.append(f"trade_date >= '{escaped_start}'")
    if end_date:
        escaped_end = str(end_date).replace("'", "''")
        filters.append(f"trade_date <= '{escaped_end}'")
    where_sql = f"where {' and '.join(filters)}" if filters else ""
    columns = DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_INTRADAY_1M]
    select_columns = ",\n            ".join(columns)
    with duckdb.connect(":memory:") as con:
        con.execute("set threads=1")
        con.execute(
            f"""
            create temp table raw as
            select
                cast(symbol as varchar) as symbol,
                cast(trade_date as varchar) as trade_date,
                case
                    when length(regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g')) >= 9
                        then right(regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g'), 9)
                    when length(regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g')) >= 6
                        then left(regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g'), 6) || '000'
                    when length(regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g')) = 4
                        then regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') || '00000'
                    else regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g')
                end as bar_time,
                try_cast(open as double) as open,
                try_cast(high as double) as high,
                try_cast(low as double) as low,
                try_cast(close as double) as close,
                try_cast(volume as double) as volume,
                try_cast(amount as double) as amount,
                try_cast(turnover_rate as double) as turnover_rate,
                try_cast(float_share as double) as float_share,
                try_cast(total_share as double) as total_share,
                coalesce(nullif(cast(source as varchar), ''), 'external_quant_zip') as source,
                coalesce(nullif(cast(adjusted_flag as varchar), ''), 'none') as adjusted_flag
            from read_parquet('{source_text}')
            {where_sql}
            """
        )
        source_row_count = int(con.execute("select count(*) from raw").fetchone()[0] or 0)
        con.execute(
            """
            create temp table paired as
            select
                c.trade_date,
                c.symbol,
                c.source,
                c.adjusted_flag,
                coalesce(o.open, c.open) as open,
                greatest(o.high, c.high) as high,
                least(o.low, c.low) as low,
                c.close as close,
                coalesce(o.volume, 0) + coalesce(c.volume, 0) as volume,
                coalesce(o.amount, 0) + coalesce(c.amount, 0) as amount,
                coalesce(o.turnover_rate, 0) + coalesce(c.turnover_rate, 0) as turnover_rate,
                coalesce(c.float_share, o.float_share) as float_share,
                coalesce(c.total_share, o.total_share) as total_share
            from raw c
            join raw o
              on c.trade_date = o.trade_date
             and c.symbol = o.symbol
             and c.source = o.source
             and c.adjusted_flag = o.adjusted_flag
            where c.bar_time = '093100000'
              and o.bar_time = '093000000'
            qualify row_number() over (
                partition by c.trade_date, c.symbol, c.source, c.adjusted_flag
                order by c.trade_date
            ) = 1
            """
        )
        con.execute(
            """
            create temp table normalized as
            select
                r.symbol,
                r.trade_date,
                r.bar_time,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.amount,
                r.turnover_rate,
                r.float_share,
                r.total_share,
                r.source,
                r.adjusted_flag
            from raw r
            left join paired p
              on r.trade_date = p.trade_date
             and r.symbol = p.symbol
             and r.source = p.source
             and r.adjusted_flag = p.adjusted_flag
            where not (p.symbol is not null and r.bar_time in ('093000000', '093100000'))
            union all
            select
                symbol,
                trade_date,
                '093100000' as bar_time,
                open,
                high,
                low,
                close,
                volume,
                amount,
                turnover_rate,
                float_share,
                total_share,
                source,
                adjusted_flag
            from paired
            """
        )
        con.execute(
            f"""
            copy (
                select
                    {select_columns}
                from normalized
                order by trade_date, symbol, bar_time, source
            )
            to '{target_text}' (format parquet)
            """
        )
    return source_row_count


def _target_spec(cfg: NormalizeIntraday1mContractConfig, *, source_metadata: dict[str, Any]) -> dict[str, Any]:
    source_params = dict(source_metadata.get("parameters", {}) or {})
    start_date = cfg.start_date or str(source_metadata.get("start_date", "") or source_params.get("start_date", "") or "")
    end_date = cfg.end_date or str(source_metadata.get("end_date", "") or source_params.get("end_date", "") or "")
    spec = {
        "source": "intraday_1m_contract_normalization",
        "domain": DataDomain.MARKET_INTRADAY_1M,
        "source_dataset_id": cfg.source_dataset_id,
        "source_fingerprint": str(source_metadata.get("fingerprint", "") or ""),
        "start_date": start_date,
        "end_date": end_date,
        "years": list(cfg.years),
        "sharded": True,
        "one_minute_policy": "mootdx_240_0930_merged_into_0931",
        "one_minute_bar_count_contracts": {
            "all_sources": "240 bars/full trading day; 09:30 opening auction merged into 09:31 when present",
            "mootdx_online": "240 bars/full trading day, starts at 09:31",
        },
        "normalization_rule": "merge_0930_into_0931_open_high_low_volume_amount_then_drop_0930",
        "normalization_contract_version": "mootdx_240_v1",
    }
    if cfg.max_shards:
        spec["max_shards"] = int(cfg.max_shards)
        spec["normalization_scope"] = "partial_shard_sample"
    return spec


def _shard_record(
    *,
    target_path: Path,
    data: pd.DataFrame,
    source_record: dict[str, Any],
    source_path: Path,
    source_row_count: int,
    removed_0930_rows: int,
    materialization: str,
    index: int,
) -> dict[str, Any]:
    return {
        **source_record,
        "domain": DataDomain.MARKET_INTRADAY_1M,
        "status": "stored",
        "path": str(target_path.resolve()),
        "row_count": int(len(data)),
        "source_row_count": int(source_row_count),
        "removed_0930_rows": int(removed_0930_rows),
        "start_date": str(data["trade_date"].min()) if "trade_date" in data.columns and not data.empty else str(source_record.get("start_date", "") or ""),
        "end_date": str(data["trade_date"].max()) if "trade_date" in data.columns and not data.empty else str(source_record.get("end_date", "") or ""),
        "symbol_count": int(data["symbol"].nunique()) if "symbol" in data.columns and not data.empty else 0,
        "source_raw_path": str(source_path.resolve()),
        "normalization_materialization": materialization,
        "normalization_index": int(index),
        "one_minute_policy": "mootdx_240_0930_merged_into_0931",
    }


def _filter_date_window(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = dates.notna()
    if start_date:
        mask &= dates.ge(pd.Timestamp(start_date))
    if end_date:
        mask &= dates.le(pd.Timestamp(end_date))
    return frame.loc[mask].reset_index(drop=True)


def _shard_intersects_years(record: dict[str, Any], years: set[int]) -> bool:
    start = pd.to_datetime(str(record.get("start_date", "") or ""), errors="coerce")
    end = pd.to_datetime(str(record.get("end_date", "") or ""), errors="coerce")
    if pd.isna(start) and pd.isna(end):
        return True
    if pd.isna(start):
        start = end
    if pd.isna(end):
        end = start
    return any(int(start.year) <= year <= int(end.year) for year in years)


def _safe_stem(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(raw or "shard"))
    if len(safe) <= 180:
        return safe
    digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:150]}_{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"time": _utc_now(), **payload}, ensure_ascii=False, sort_keys=True) + "\n"
    with _PROGRESS_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize an existing sharded 1m dataset to the mootdx 240-bar contract.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--years", default="")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    parser.add_argument("--engine", choices=("duckdb", "pandas"), default="duckdb")
    return parser


def _parse_years(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(text or "").split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = normalize_intraday_1m_contract(
        NormalizeIntraday1mContractConfig(
            lake_root=Path(args.lake_root),
            source_dataset_id=args.source_dataset_id,
            start_date=args.start_date,
            end_date=args.end_date,
            years=_parse_years(args.years),
            max_shards=args.max_shards,
            workers=args.workers,
            progress_path=Path(args.progress_path) if str(args.progress_path or "").strip() else None,
            dry_run=args.dry_run,
            resume=args.resume,
            reuse=args.reuse,
            engine=args.engine,
        )
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
