"""Compare a selected local minute extract with QDP and iQuant."""

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
from quantlab.data.iquant import aggregate_1m_to_5m, inspect_file, read_file
from quantlab.data.qdp_v2 import resolve_active_domain

VALUE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
KEY_COLUMNS = ("symbol", "trade_date", "bar_time")


class MinuteParityError(ValueError):
    """Raised when parity inputs are incomplete or ambiguous."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _iquant_path(data_dir: Path, symbol: str) -> Path:
    code, market = str(symbol).split(".", 1)
    return data_dir / market / "60" / f"{code}.DAT"


def _qdp_frame(
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = resolve_active_domain("market_intraday_5m", workspace_root=workspace_root)
    with duckdb.connect(":memory:") as connection:
        connection.register("wanted_symbols", pd.DataFrame({"symbol": symbols}))
        frame = connection.execute(
            "SELECT q.symbol,q.trade_date,q.bar_time,q.open,q.high,q.low,q.close,q.volume,q.amount,"
            "q.source,q.adjusted_flag FROM read_parquet(?, union_by_name=true) q "
            "JOIN wanted_symbols s USING(symbol) WHERE q.trade_date>=? AND q.trade_date<=? "
            "ORDER BY q.symbol,q.trade_date,q.bar_time",
            [[str(path) for path in context.shard_paths], str(start_date), str(end_date)],
        ).fetchdf()
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteParityError("minute_qdp_duplicate_keys")
    sources = {str(key): int(value) for key, value in frame["source"].value_counts(dropna=False).items()}
    return frame, {"dataset_id": context.dataset_id, "source_counts": sources}


def compare_bar_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    merged = left.merge(
        right,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        indicator=True,
    )
    counts = merged["_merge"].value_counts().to_dict()
    both = merged.loc[merged["_merge"] == "both"]
    metrics: dict[str, Any] = {}
    for column in VALUE_COLUMNS:
        lhs = pd.to_numeric(both[f"{column}_{left_name}"], errors="coerce")
        rhs = pd.to_numeric(both[f"{column}_{right_name}"], errors="coerce")
        valid = lhs.notna() & rhs.notna()
        lhs = lhs.loc[valid]
        rhs = rhs.loc[valid]
        absolute = (lhs - rhs).abs()
        denominator = np.maximum(np.maximum(lhs.abs(), rhs.abs()), 1.0)
        relative = absolute / denominator
        metrics[column] = {
            "compared_count": int(len(relative)),
            "exact_count": int(lhs.eq(rhs).sum()),
            "exact_rate": float(lhs.eq(rhs).mean()) if len(relative) else None,
            "median_relative_error": float(relative.median()) if len(relative) else None,
            "p95_relative_error": float(relative.quantile(0.95)) if len(relative) else None,
            "maximum_relative_error": float(relative.max()) if len(relative) else None,
            "maximum_absolute_error": float(absolute.max()) if len(relative) else None,
        }
    return {
        "row_count": int(len(merged)),
        "key_counts": {
            "both": int(counts.get("both", 0)),
            f"{left_name}_only": int(counts.get("left_only", 0)),
            f"{right_name}_only": int(counts.get("right_only", 0)),
        },
        "metrics": metrics,
    }


def _session_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.groupby(["symbol", "trade_date"], sort=True).size().value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _read_iquant_minutes(
    data_dir: Path,
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return pd.concat(
        [
            read_file(
                _iquant_path(data_dir, symbol),
                symbol=symbol,
                period="1m",
                start_date=start_date,
                end_date=end_date,
            )
            for symbol in symbols
        ],
        ignore_index=True,
    )


def _source_payload(
    *,
    local_path: Path,
    local_row_count: int,
    data_dir: Path,
    symbols: list[str],
    qdp: dict[str, Any],
) -> dict[str, Any]:
    return {
        "local": {
            "path": str(local_path),
            "row_count": int(local_row_count),
            "file_size": int(local_path.stat().st_size),
            "sha256": sha256_file(local_path),
        },
        "qdp": qdp,
        "iquant": [
            inspect_file(_iquant_path(data_dir, symbol), period="1m", symbol=symbol)
            for symbol in symbols
        ],
    }


def run_minute_parity(
    *,
    local_parquet: str | Path,
    iquant_data_dir: str | Path,
    symbols: list[str] | None = None,
    start_date: str = "",
    end_date: str = "",
    workspace_root: str | Path | None = None,
    record_path: str | Path | None = None,
) -> dict[str, Any]:
    local_path = Path(local_parquet).resolve()
    data_dir = Path(iquant_data_dir).resolve()
    if not local_path.is_file() or not data_dir.is_dir():
        raise MinuteParityError("minute_parity_source_missing")
    local = pd.read_parquet(
        local_path,
        columns=[*KEY_COLUMNS, *VALUE_COLUMNS],
    )
    wanted = sorted({str(value) for value in (symbols or local["symbol"].unique().tolist())})
    local = local.loc[local["symbol"].isin(wanted)].copy()
    first_date = str(start_date or local["trade_date"].min())
    last_date = str(end_date or local["trade_date"].max())
    local = local.loc[
        (local["trade_date"] >= first_date) & (local["trade_date"] <= last_date)
    ].copy()
    if local.empty or local.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteParityError("minute_local_keys_invalid")
    local_qdp_5m = aggregate_1m_to_5m(local, include_opening_auction=True)
    local_continuous_5m = aggregate_1m_to_5m(local, include_opening_auction=False)
    iquant_1m = _read_iquant_minutes(
        data_dir,
        symbols=wanted,
        start_date=first_date,
        end_date=last_date,
    )
    iquant_5m = aggregate_1m_to_5m(iquant_1m)
    qdp_5m, qdp = _qdp_frame(
        symbols=wanted,
        start_date=first_date,
        end_date=last_date,
        workspace_root=workspace_root,
    )
    result = {
        "schema": "quantlab.minute_local_parity/2",
        "status": "ok",
        "created_at": _now(),
        "scope": {
            "symbols": wanted,
            "start_date": first_date,
            "end_date": last_date,
            "date_count": int(local["trade_date"].nunique()),
        },
        "contracts": {
            "local_archive": "raw/unadjusted; standalone 09:30 auction retained in this probe",
            "qdp_5m": "09:30 auction folded into the right-edge 09:35 bucket",
            "iquant_5m": "continuous 09:31..09:35 maps to the right-edge 09:35 bucket",
        },
        "sources": _source_payload(
            local_path=local_path,
            local_row_count=len(local),
            data_dir=data_dir,
            symbols=wanted,
            qdp=qdp,
        ),
        "row_counts": {
            "local_qdp_5m": int(len(local_qdp_5m)),
            "local_continuous_5m": int(len(local_continuous_5m)),
            "qdp_5m": int(len(qdp_5m)),
            "iquant_1m": int(len(iquant_1m)),
            "iquant_5m": int(len(iquant_5m)),
        },
        "session_bar_counts": {
            "local_qdp_5m": _session_counts(local_qdp_5m),
            "local_continuous_5m": _session_counts(local_continuous_5m),
            "iquant_5m": _session_counts(iquant_5m),
        },
        "local_qdp": compare_bar_frames(
            local_qdp_5m,
            qdp_5m,
            left_name="local",
            right_name="qdp",
        ),
        "local_iquant": compare_bar_frames(
            local_continuous_5m,
            iquant_5m,
            left_name="local",
            right_name="iquant",
        ),
        "independence": {
            "local_vs_qdp": "conversion parity only; QDP source counts show whether the underlying vendor archive is shared",
            "local_vs_iquant": "independent storage path and broker-client cache; useful as a source cross-check",
        },
    }
    if record_path:
        write_json(Path(record_path).resolve(), result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a local minute extract with QDP and iQuant")
    parser.add_argument("--local-parquet", required=True)
    parser.add_argument("--iquant-data-dir", required=True)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--record-path", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_minute_parity(
        local_parquet=args.local_parquet,
        iquant_data_dir=args.iquant_data_dir,
        symbols=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        workspace_root=args.workspace_root or None,
        record_path=args.record_path or None,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"status: {result['status']}")
        print(f"local_qdp_keys: {result['local_qdp']['key_counts']}")
        print(f"local_iquant_keys: {result['local_iquant']['key_counts']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MinuteParityError", "compare_bar_frames", "run_minute_parity"]
