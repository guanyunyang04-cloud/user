from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.providers import (
    _fetch_baostock_all_stock_frame_with_timeout,
    _fetch_baostock_industry_frame_with_timeout,
)


TARGET_DATASETS = {
    DataDomain.UNIVERSE_SNAPSHOT: "data_platform_universe_snapshot__7d0361a9af11039cd15b60fe",
    DataDomain.SECURITY_STATUS: "data_platform_security_status__c0c7df7228b45d3563e6dc13",
    DataDomain.INDUSTRY_CONCEPT: "data_platform_industry_concept__578368469c5bd89e05f82706",
}


def _log(path: Path, payload: dict[str, Any]) -> None:
    payload = {"ts": pd.Timestamp.utcnow().isoformat(), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_dataset_paths(lake: ResearchDataLake, dataset_id: str) -> list[str]:
    metadata = lake.describe_dataset(dataset_id)
    paths = dict(metadata.get("content_paths", {}) or {})
    manifest_path = Path(str(paths.get("shard_manifest", "") or ""))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        out = [str(item.get("path", "") or "") for item in manifest.get("shards", [])]
        return [path for path in out if path and Path(path).exists()]
    data_path = str(paths.get("silver_domain_data", "") or "")
    if "*" in data_path:
        return [str(path) for path in sorted(Path().glob(data_path))]
    return [data_path] if data_path and Path(data_path).exists() else []


def _trading_dates(lake: ResearchDataLake, *, calendar_dataset_id: str, start_date: str, end_date: str) -> list[str]:
    paths = _read_dataset_paths(lake, calendar_dataset_id)
    if not paths:
        raise RuntimeError(f"calendar_paths_missing: {calendar_dataset_id}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    is_open = frame.get("is_open", True)
    if not isinstance(is_open, pd.Series):
        frame["is_open"] = True
    else:
        frame["is_open"] = is_open.astype(str).str.lower().isin({"1", "true", "t", "yes"})
    mask = (frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date) & frame["is_open"]
    return sorted(frame.loc[mask, "trade_date"].dropna().astype(str).unique().tolist())


def _fetch_with_retries(
    *,
    domain: str,
    trade_date: str,
    retries: int,
    timeout_seconds: int,
    progress_path: Path,
) -> pd.DataFrame:
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            if domain == DataDomain.INDUSTRY_CONCEPT:
                frame = _fetch_baostock_industry_frame_with_timeout(
                    trade_date=trade_date,
                    timeout_seconds=timeout_seconds,
                )
            else:
                frame = _fetch_baostock_all_stock_frame_with_timeout(
                    domain=domain,
                    trade_date=trade_date,
                    timeout_seconds=timeout_seconds,
                )
            elapsed = round(time.perf_counter() - started, 3)
            _log(
                progress_path,
                {
                    "event": "date_domain_fetched",
                    "trade_date": trade_date,
                    "domain": domain,
                    "attempt": attempt,
                    "rows": int(len(frame)),
                    "elapsed_seconds": elapsed,
                },
            )
            return frame
        except Exception as exc:  # noqa: BLE001 - repair script records exact provider failure.
            last_error = f"{type(exc).__name__}: {exc}"
            _log(
                progress_path,
                {
                    "event": "date_domain_failed",
                    "trade_date": trade_date,
                    "domain": domain,
                    "attempt": attempt,
                    "error": last_error,
                },
            )
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"{domain} {trade_date} failed after {retries} attempts: {last_error}")


def _dataset_dir(lake: ResearchDataLake, dataset_id: str, domain: str) -> Path:
    fingerprint = dataset_id.split("__", 1)[1]
    return lake.parquet_root / "bronze_silver" / f"data_platform_{domain}" / fingerprint


def _write_2026_shards(
    *,
    lake: ResearchDataLake,
    dates: list[str],
    retries: int,
    all_stock_timeout_seconds: int,
    industry_timeout_seconds: int,
    progress_path: Path,
) -> None:
    frames: dict[str, list[pd.DataFrame]] = {
        DataDomain.UNIVERSE_SNAPSHOT: [],
        DataDomain.SECURITY_STATUS: [],
        DataDomain.INDUSTRY_CONCEPT: [],
    }
    for idx, trade_date in enumerate(dates, start=1):
        _log(progress_path, {"event": "date_started", "trade_date": trade_date, "index": idx, "date_count": len(dates)})
        universe = _fetch_with_retries(
            domain=DataDomain.UNIVERSE_SNAPSHOT,
            trade_date=trade_date,
            retries=retries,
            timeout_seconds=all_stock_timeout_seconds,
            progress_path=progress_path,
        )
        status = _fetch_with_retries(
            domain=DataDomain.SECURITY_STATUS,
            trade_date=trade_date,
            retries=retries,
            timeout_seconds=all_stock_timeout_seconds,
            progress_path=progress_path,
        )
        industry = _fetch_with_retries(
            domain=DataDomain.INDUSTRY_CONCEPT,
            trade_date=trade_date,
            retries=retries,
            timeout_seconds=industry_timeout_seconds,
            progress_path=progress_path,
        )
        if len(universe) < 3000 or len(status) < 3000:
            raise RuntimeError(f"low_density_all_stock: {trade_date} universe={len(universe)} status={len(status)}")
        if len(industry) < 3000:
            raise RuntimeError(f"low_density_industry: {trade_date} industry={len(industry)}")
        frames[DataDomain.UNIVERSE_SNAPSHOT].append(universe)
        frames[DataDomain.SECURITY_STATUS].append(status)
        frames[DataDomain.INDUSTRY_CONCEPT].append(industry)
        _log(
            progress_path,
            {
                "event": "date_completed",
                "trade_date": trade_date,
                "index": idx,
                "rows": {
                    DataDomain.UNIVERSE_SNAPSHOT: int(len(universe)),
                    DataDomain.SECURITY_STATUS: int(len(status)),
                    DataDomain.INDUSTRY_CONCEPT: int(len(industry)),
                },
            },
        )

    for domain, chunks in frames.items():
        data = pd.concat(chunks, ignore_index=True).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        shard_dir = _dataset_dir(lake, TARGET_DATASETS[domain], domain) / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        out_path = shard_dir / f"{domain}__2026_daily.parquet"
        data.to_parquet(out_path, index=False)
        _log(
            progress_path,
            {
                "event": "year_shard_written",
                "domain": domain,
                "path": str(out_path.resolve()),
                "rows": int(len(data)),
                "start_date": str(data["trade_date"].min()),
                "end_date": str(data["trade_date"].max()),
            },
        )


def _register_dataset(lake: ResearchDataLake, *, domain: str, source: str, progress_path: Path) -> None:
    dataset_id = TARGET_DATASETS[domain]
    dataset_kind = f"data_platform_{domain}"
    fingerprint = dataset_id.split("__", 1)[1]
    dataset_dir = _dataset_dir(lake, dataset_id, domain)
    shard_dir = dataset_dir / "shards"
    shard_manifest_path = dataset_dir / "shard_manifest.json"
    shard_paths = sorted(shard_dir.glob(f"{domain}__*_daily.parquet"))
    if len(shard_paths) != 17:
        raise RuntimeError(f"unexpected_shard_count: {domain} expected=17 actual={len(shard_paths)}")

    shard_records: list[dict[str, Any]] = []
    for path in shard_paths:
        frame = pd.read_parquet(path, columns=["trade_date"])
        start_date = pd.to_datetime(frame["trade_date"], errors="coerce").min().strftime("%Y-%m-%d")
        end_date = pd.to_datetime(frame["trade_date"], errors="coerce").max().strftime("%Y-%m-%d")
        shard_records.append(
            {
                "path": str(path.resolve()),
                "row_count": int(len(frame)),
                "start_date": start_date,
                "end_date": end_date,
                "status": "stored",
                "year": path.stem.split("__", 1)[1].split("_", 1)[0],
            }
        )

    manifest = {
        "dataset_id": dataset_id,
        "dataset_kind": dataset_kind,
        "domain": domain,
        "source": source,
        "sharded": True,
        "shard_count": len(shard_records),
        "shards": shard_records,
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    row_count = int(sum(int(item["row_count"]) for item in shard_records))
    start_date = min(str(item["start_date"]) for item in shard_records)
    end_date = max(str(item["end_date"]) for item in shard_records)
    content_paths = {
        "silver_domain_data": str((shard_dir / "*.parquet").resolve()),
        "shard_manifest": str(shard_manifest_path.resolve()),
    }
    lake._upsert_dataset(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        domain="bronze_silver",
        zone="research",
        source=source,
        spec={
            "dataset": dataset_kind,
            "domain": domain,
            "start_date": start_date,
            "end_date": end_date,
            "snapshot_frequency": "daily",
            "repair": "after_gc_20260629",
            "sharded": True,
        },
        label_completeness_summary={},
        content_paths=content_paths,
        row_counts={
            "silver_domain_data": row_count,
            "shards": len(shard_records),
            "stored_shards": len(shard_records),
            "error_count": 0,
            "_date_bounds": {"start_date": start_date, "end_date": end_date},
        },
        source_cache={"sharded": True, "shard_count": len(shard_records)},
        fingerprint=fingerprint,
        status="stored",
    )
    _log(
        progress_path,
        {
            "event": "dataset_registered",
            "domain": domain,
            "dataset_id": dataset_id,
            "rows": row_count,
            "shards": len(shard_records),
            "start_date": start_date,
            "end_date": end_date,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake-root", default="quant_data_platform/data/lake")
    parser.add_argument("--calendar-dataset-id", default="data_platform_trading_calendar__03298281bc6712b72d0b5a6b")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-06-26")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--all-stock-timeout-seconds", type=int, default=90)
    parser.add_argument("--industry-timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--progress-path",
        default="quant_data_platform/data/lake/canonical/imports/repair_daily_pit_2026_after_gc_timeout_progress.jsonl",
    )
    args = parser.parse_args()
    progress_path = Path(args.progress_path)
    source = "repair_daily_pit_after_gc_20260629_timeout_guard"
    lake = ResearchDataLake(Path(args.lake_root))
    dates = _trading_dates(lake, calendar_dataset_id=args.calendar_dataset_id, start_date=args.start_date, end_date=args.end_date)
    _log(progress_path, {"event": "start", "date_count": len(dates), "start_date": args.start_date, "end_date": args.end_date})
    if not dates:
        raise RuntimeError("no_trading_dates")
    _write_2026_shards(
        lake=lake,
        dates=dates,
        retries=max(1, int(args.retries)),
        all_stock_timeout_seconds=max(10, int(args.all_stock_timeout_seconds)),
        industry_timeout_seconds=max(10, int(args.industry_timeout_seconds)),
        progress_path=progress_path,
    )
    for domain in (DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS, DataDomain.INDUSTRY_CONCEPT):
        _register_dataset(lake, domain=domain, source=source, progress_path=progress_path)
    _log(progress_path, {"event": "completed", "dataset_ids": TARGET_DATASETS})
    print(json.dumps({"status": "ok", "dataset_ids": TARGET_DATASETS, "date_count": len(dates)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
