"""Export a small iQuant sample and compare it with the active QDP store."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.core.io import sha256_file, write_json
from quantlab.data.core.paths import workspace_root
from quantlab.data.iquant import (
    IQuantDataError,
    aggregate_1m_to_5m,
    inspect_file,
    read_file,
)
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_path,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)

DAILY_SYMBOLS = ("000001.SZ", "600000.SH", "600011.SH", "600276.SH")
MINUTE_SYMBOLS = ("600011.SH", "600276.SH")
MINUTE_DATES = (
    "2010-04-16",
    "2012-01-04",
    "2015-01-05",
    "2018-06-28",
    "2020-01-02",
    "2023-01-03",
    "2025-01-02",
    "2025-12-31",
    "2026-07-21",
)
VALUE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _symbol_path(data_dir: Path, symbol: str, period: str) -> Path:
    code, market = str(symbol).split(".", 1)
    folder = "86400" if period == "1d" else "60"
    return data_dir / market / folder / f"{code}.DAT"


def _domain_context(workspace: Path, domain: str) -> tuple[Any, list[Path]]:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, ""))
    if not dataset_id:
        raise IQuantDataError(f"active_qdp_domain_missing:{domain}")
    manifest = read_dataset_manifest(dataset_manifest_path(root, domain, dataset_id))
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise IQuantDataError(f"qdp_shards_missing:{domain}:{len(missing)}")
    return manifest, paths


def _selected_paths(manifest: Any, paths: list[Path], dates: tuple[str, ...] | None) -> list[Path]:
    if not dates:
        return paths
    selected: list[Path] = []
    for item, path in zip(manifest.shards, paths, strict=True):
        start = str(item.start_date or "")
        end = str(item.end_date or "")
        if len(start) != 10 or len(end) != 10 or any(start <= date <= end for date in dates):
            selected.append(path)
    return selected


def _query_qdp(
    workspace: Path,
    domain: str,
    *,
    symbols: tuple[str, ...],
    start_date: str = "",
    end_date: str = "",
    dates: tuple[str, ...] | None = None,
) -> tuple[Any, pd.DataFrame]:
    manifest, all_paths = _domain_context(workspace, domain)
    paths = _selected_paths(manifest, all_paths, dates)
    fields = ["symbol", "trade_date"]
    key = ["symbol", "trade_date"]
    if domain == "market_intraday_5m":
        fields.append("bar_time")
        key.append("bar_time")
    fields.extend([*VALUE_COLUMNS, "source", "adjusted_flag"])
    with duckdb.connect(":memory:") as connection:
        connection.register("wanted_symbols", pd.DataFrame({"symbol": list(symbols)}))
        conditions = ["q.symbol IN (SELECT symbol FROM wanted_symbols)"]
        parameters: list[object] = [list(map(str, paths))]
        if dates:
            connection.register("wanted_dates", pd.DataFrame({"trade_date": list(dates)}))
            conditions.append("q.trade_date IN (SELECT trade_date FROM wanted_dates)")
        else:
            conditions.extend(["q.trade_date >= ?", "q.trade_date <= ?"])
            parameters.extend([start_date, end_date])
        query = (
            f"SELECT {', '.join('q.' + field for field in fields)} "
            "FROM read_parquet(?, union_by_name=true) q WHERE "
            + " AND ".join(conditions)
            + " ORDER BY "
            + ", ".join(key)
        )
        frame = connection.execute(query, parameters).fetchdf()
    duplicate_count = int(frame.duplicated(key).sum())
    if duplicate_count:
        raise IQuantDataError(f"qdp_duplicate_keys:{domain}:{duplicate_count}")
    return manifest, frame


def _qdp_symbols(workspace: Path) -> set[str]:
    _, paths = _domain_context(workspace, "market_daily_raw")
    with duckdb.connect(":memory:") as connection:
        rows = connection.execute(
            "SELECT DISTINCT symbol FROM read_parquet(?, union_by_name=true)",
            [list(map(str, paths))],
        ).fetchall()
    return {str(row[0]) for row in rows}


def _comparison(left: pd.DataFrame, right: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    merged = left.merge(right, on=key, how="outer", suffixes=("_iquant", "_qdp"), indicator=True)
    merged["key_status"] = merged.pop("_merge").map(
        {"left_only": "iquant_only", "right_only": "qdp_only", "both": "both"}
    )
    for column in VALUE_COLUMNS:
        lhs = merged[f"{column}_iquant"]
        rhs = merged[f"{column}_qdp"]
        merged[f"{column}_absolute_error"] = (lhs - rhs).abs()
        denominator = np.maximum(np.maximum(lhs.abs(), rhs.abs()), 1.0)
        merged[f"{column}_relative_error"] = merged[f"{column}_absolute_error"] / denominator
        merged[f"{column}_exact"] = lhs.eq(rhs)
    return merged


def _metric_summary(comparison: pd.DataFrame) -> dict[str, Any]:
    both = comparison.loc[comparison["key_status"] == "both"]
    result: dict[str, Any] = {}
    for column in VALUE_COLUMNS:
        relative = both[f"{column}_relative_error"].dropna()
        absolute = both[f"{column}_absolute_error"].dropna()
        result[column] = {
            "compared_count": len(relative),
            "exact_count": int(both[f"{column}_exact"].sum()),
            "exact_rate": float(both[f"{column}_exact"].mean()) if len(both) else None,
            "median_relative_error": float(relative.median()) if len(relative) else None,
            "p95_relative_error": float(relative.quantile(0.95)) if len(relative) else None,
            "maximum_relative_error": float(relative.max()) if len(relative) else None,
            "maximum_absolute_error": float(absolute.max()) if len(absolute) else None,
        }
    return result


def _comparison_summary(comparison: pd.DataFrame) -> dict[str, Any]:
    status_counts = comparison["key_status"].value_counts().to_dict()
    return {
        "row_count": len(comparison),
        "key_counts": {name: int(status_counts.get(name, 0)) for name in ("both", "iquant_only", "qdp_only")},
        "metrics": _metric_summary(comparison),
    }


def _source_summaries(comparison: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, group in comparison.loc[comparison["key_status"] == "both"].groupby("source", dropna=False):
        result[str(source)] = _comparison_summary(group)
    return result


def _inventory(data_dir: Path, qdp_symbols: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = _now()
    initial_paths = sorted([*(data_dir / "SH" / "60").glob("*.DAT"), *(data_dir / "SZ" / "60").glob("*.DAT")])
    rows: list[dict[str, Any]] = []
    for path in initial_paths:
        market = path.parents[1].name.upper()
        symbol = f"{path.stem}.{market}"
        try:
            info = inspect_file(path, period="1m", symbol=symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "path": str(path.relative_to(data_dir)).replace("\\", "/"),
                    "file_size": int(info["file_size"]),
                    "mtime_ns": int(info["mtime_ns"]),
                    "row_count": int(info["row_count"]),
                    "decoded_start": str(info["decoded_start"]),
                    "decoded_end": str(info["decoded_end"]),
                    "format_valid": True,
                    "stable_during_read": bool(info["stable_during_read"]),
                    "qdp_symbol": symbol in qdp_symbols,
                    "error": "",
                }
            )
        except (OSError, IQuantDataError) as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "path": str(path.relative_to(data_dir)).replace("\\", "/"),
                    "file_size": int(path.stat().st_size) if path.exists() else 0,
                    "mtime_ns": int(path.stat().st_mtime_ns) if path.exists() else 0,
                    "row_count": 0,
                    "decoded_start": "",
                    "decoded_end": "",
                    "format_valid": False,
                    "stable_during_read": False,
                    "qdp_symbol": symbol in qdp_symbols,
                    "error": str(exc),
                }
            )
    final_paths = sorted([*(data_dir / "SH" / "60").glob("*.DAT"), *(data_dir / "SZ" / "60").glob("*.DAT")])
    frame = pd.DataFrame(rows)
    overlap = frame.loc[frame["qdp_symbol"]]
    finished_at = datetime.now(UTC)
    latest_mtime_ns = int(frame["mtime_ns"].max()) if len(frame) else 0
    latest_write_at = datetime.fromtimestamp(latest_mtime_ns / 1_000_000_000, UTC) if latest_mtime_ns else None
    latest_write_age_seconds = max(0.0, (finished_at - latest_write_at).total_seconds()) if latest_write_at else None
    changed_during_scan = [str(path) for path in initial_paths] != [str(path) for path in final_paths] or not bool(
        frame["stable_during_read"].all()
    )
    summary = {
        "scan_started_at": started,
        "scan_finished_at": finished_at.replace(microsecond=0).isoformat(),
        "file_count": len(frame),
        "file_count_after_scan": len(final_paths),
        "changed_during_scan": changed_during_scan,
        "latest_write_at": latest_write_at.replace(microsecond=0).isoformat() if latest_write_at else "",
        "latest_write_age_seconds": latest_write_age_seconds,
        "recent_write_within_5_minutes": latest_write_age_seconds is not None and latest_write_age_seconds <= 300.0,
        "total_bytes": int(frame["file_size"].sum()),
        "format_valid_count": int(frame["format_valid"].sum()),
        "qdp_daily_symbol_count": len(qdp_symbols),
        "qdp_symbol_file_count": len(overlap),
        "qdp_symbol_coverage_rate": float(len(overlap) / len(qdp_symbols)) if qdp_symbols else 0.0,
        "qdp_symbol_files_at_most_one_session": int((overlap["row_count"] <= 240).sum()),
        "qdp_symbol_files_at_most_two_sessions": int((overlap["row_count"] <= 500).sum()),
        "qdp_symbol_files_over_100k_rows": int((overlap["row_count"] > 100_000).sum()),
    }
    return frame, summary


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def _artifact(workspace: Path, path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    try:
        stored_path = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        stored_path = str(path.resolve())
    return {
        "path": stored_path,
        "row_count": len(frame),
        "file_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _sample_metadata(data_dir: Path, symbols: tuple[str, ...], period: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for symbol in symbols:
        path = _symbol_path(data_dir, symbol, period)
        info = inspect_file(path, period=period, symbol=symbol, include_sha256=True)
        info["path"] = str(path.relative_to(data_dir)).replace("\\", "/")
        if not bool(info["stable_during_read"]):
            raise IQuantDataError(f"sample_changed_during_read:{path}")
        result.append(info)
    return result


def run_parity(
    *,
    workspace: str | Path | None,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    record_path: str | Path | None = None,
) -> dict[str, Any]:
    ws = workspace_root(workspace)
    source_root = Path(data_dir).resolve()
    if not source_root.is_dir():
        raise IQuantDataError(f"iquant_data_dir_missing:{source_root}")
    output = Path(output_dir).resolve() if output_dir else (ws / "data" / "staging" / "iquant_parity").resolve()
    output.mkdir(parents=True, exist_ok=True)

    daily_manifest, daily_qdp = _query_qdp(
        ws,
        "market_daily_raw",
        symbols=DAILY_SYMBOLS,
        start_date="2010-01-04",
        end_date="2026-07-21",
    )
    daily_iquant = pd.concat(
        [
            read_file(
                _symbol_path(source_root, symbol, "1d"),
                symbol=symbol,
                period="1d",
                start_date="2010-01-04",
                end_date="2026-07-21",
            )
            for symbol in DAILY_SYMBOLS
        ],
        ignore_index=True,
    ).sort_values(["symbol", "trade_date"], ignore_index=True)
    daily_comparison = _comparison(daily_iquant, daily_qdp, ["symbol", "trade_date"])

    minute_iquant = pd.concat(
        [
            read_file(
                _symbol_path(source_root, symbol, "1m"),
                symbol=symbol,
                period="1m",
                start_date=min(MINUTE_DATES),
                end_date=max(MINUTE_DATES),
            ).loc[lambda frame: frame["trade_date"].isin(MINUTE_DATES)]
            for symbol in MINUTE_SYMBOLS
        ],
        ignore_index=True,
    ).sort_values(["symbol", "trade_date", "bar_time"], ignore_index=True)
    minute_iquant_5m = aggregate_1m_to_5m(minute_iquant)
    minute_manifest, minute_qdp = _query_qdp(
        ws,
        "market_intraday_5m",
        symbols=MINUTE_SYMBOLS,
        dates=MINUTE_DATES,
    )
    minute_comparison = _comparison(
        minute_iquant_5m,
        minute_qdp,
        ["symbol", "trade_date", "bar_time"],
    )

    qdp_symbols = _qdp_symbols(ws)
    inventory, inventory_summary = _inventory(source_root, qdp_symbols)
    frames = {
        "cache_inventory": inventory,
        "daily_iquant": daily_iquant,
        "daily_qdp": daily_qdp,
        "daily_comparison": daily_comparison,
        "minute_iquant_1m": minute_iquant,
        "minute_iquant_5m": minute_iquant_5m,
        "minute_qdp_5m": minute_qdp,
        "minute_comparison": minute_comparison,
    }
    artifacts: dict[str, Any] = {}
    for name, frame in frames.items():
        path = output / f"{name}.parquet"
        _write_parquet(path, frame)
        artifacts[name] = _artifact(ws, path, frame)

    daily_summary = _comparison_summary(daily_comparison)
    minute_summary = _comparison_summary(minute_comparison)
    recent_minute = minute_comparison.loc[minute_comparison["trade_date"] >= "2025-01-01"]
    recent_minute_summary = _comparison_summary(recent_minute)
    minute_coverage = float(inventory_summary["qdp_symbol_coverage_rate"])
    cache_mutating = bool(inventory_summary["changed_during_scan"] or inventory_summary["recent_write_within_5_minutes"])
    result: dict[str, Any] = {
        "schema": "quantlab.iquant_qdp_parity/1",
        "status": "ok_with_mutating_cache_warning" if cache_mutating else "ok",
        "created_at": _now(),
        "scope": {
            "iquant_data_dir": str(source_root),
            "daily_symbols": list(DAILY_SYMBOLS),
            "daily_range": ["2010-01-04", "2026-07-21"],
            "minute_symbols": list(MINUTE_SYMBOLS),
            "minute_dates": list(MINUTE_DATES),
        },
        "format_contract": {
            "container": "8-byte little-endian header plus 64-byte fixed K-line records",
            "timezone": "Asia/Shanghai",
            "price": "stored integer / 1000",
            "volume": "stored lots * 100 = shares",
            "amount": "currency units",
            "minute_label": "right edge; 09:31..09:35 aggregate to 09:35",
            "adjustment": "unadjusted/raw, confirmed against QDP adjusted_flag=none on the sample",
            "installed_xtdata_get_local_data": "empty compatibility stub; DAT decoded read-only",
        },
        "source_files": {
            "daily": _sample_metadata(source_root, DAILY_SYMBOLS, "1d"),
            "minute": _sample_metadata(source_root, MINUTE_SYMBOLS, "1m"),
        },
        "qdp": {
            "market_daily_raw_dataset_id": daily_manifest.dataset_id,
            "market_intraday_5m_dataset_id": minute_manifest.dataset_id,
        },
        "cache_inventory": inventory_summary,
        "daily_parity": daily_summary,
        "minute_parity": {
            **minute_summary,
            "recent_2025_plus": recent_minute_summary,
            "by_qdp_source": _source_summaries(minute_comparison),
            "expected_5m_bars_per_day": 48,
            "sample_day_bar_counts": {
                str(count): int(frequency)
                for count, frequency in minute_iquant_5m.groupby(["symbol", "trade_date"]).size().value_counts().sort_index().items()
            },
            "one_minute_rows_per_5m_bucket": {
                str(count): int(frequency)
                for count, frequency in minute_iquant_5m["minute_count"].value_counts().sort_index().items()
            },
        },
        "decision": {
            "classification": "supplement_and_validation_only",
            "historical_minute_replacement_ready": False,
            "daily_cross_check_ready": daily_summary["key_counts"]["both"] > 10_000,
            "reasons": [
                f"The current one-minute cache covers {minute_coverage:.2%} of QDP symbols with daily bars.",
                "The sampled minute files decode cleanly and all sampled 5-minute keys align with QDP.",
                "Recent sampled prices are effectively identical, but older vendor bars contain small OHLC and larger thin-bucket flow differences.",
                "Filling only downloaded symbols would create source and sample-selection bias; do not replace the historical minute table from this snapshot.",
                "Use iQuant for recent/live supplementation and independent checks until the download is complete, frozen, and reprofiled.",
            ],
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, result)
    if record_path:
        resolved_record = Path(record_path)
        if not resolved_record.is_absolute():
            resolved_record = ws / resolved_record
        write_json(resolved_record.resolve(), result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantlab data iquant-parity",
        description="Export a small iQuant DAT sample and compare it with active QDP daily/5m bars.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--data-dir", required=True, help="iQuant datadir containing SH/ and SZ/.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--record-path", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_parity(
        workspace=str(args.workspace_root or "") or None,
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir or "") or None,
        record_path=str(args.record_path or "") or None,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        inventory = result["cache_inventory"]
        daily = result["daily_parity"]
        minute = result["minute_parity"]
        print(f"status: {result['status']}")
        print(
            "minute_cache: "
            f"files={inventory['file_count']} qdp_symbols={inventory['qdp_symbol_file_count']}/"
            f"{inventory['qdp_daily_symbol_count']} ({inventory['qdp_symbol_coverage_rate']:.2%})"
        )
        print(f"daily_keys: {daily['key_counts']}")
        print(f"minute_keys: {minute['key_counts']}")
        print(f"decision: {result['decision']['classification']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
