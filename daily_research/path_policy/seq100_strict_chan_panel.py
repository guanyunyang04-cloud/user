"""Resumable, resource-adaptive all-market strict Chan structure panel builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_intraday as intraday
from daily_research.path_policy import seq100_strict_chan_parser as parser

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "output"
    / "path_policy"
    / "studies"
    / "seq100_strict_chan_panel_v1"
)
STUDY_ID = "seq100_strict_chan_panel_v1"
SCHEMA_VERSION = "seq100_strict_chan_panel/1"
EVENT_SCHEMA_VERSION = "seq100_strict_chan_panel_events/1"
SNAPSHOT_SCHEMA_VERSION = "seq100_strict_chan_panel_snapshots/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _parquet_safe_event_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in frame.select_dtypes(include=["object"]).columns:
        values = [
            value
            for value in frame[column].tolist()
            if value is not None and not (isinstance(value, float) and pd.isna(value))
        ]
        value_types = {type(value) for value in values}
        has_nested = any(
            isinstance(value, (Mapping, list, tuple, set)) for value in values
        )
        if len(value_types) <= 1 and not has_nested:
            continue

        def encode(value: Any) -> str | None:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, set):
                value = sorted(value)
            if isinstance(value, (Mapping, list, tuple)):
                return json.dumps(
                    value, ensure_ascii=False, sort_keys=True, default=str
                )
            return str(value)

        frame[column] = frame[column].map(encode)
    return frame


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket_members(
    symbols: Sequence[str], *, bucket_id: int, bucket_count: int
) -> list[str]:
    if bucket_count <= 0 or bucket_id < 0 or bucket_id >= bucket_count:
        raise ValueError("strict_chan_panel_bucket_invalid")
    ordered = sorted({str(symbol) for symbol in symbols})
    start = len(ordered) * bucket_id // bucket_count
    stop = len(ordered) * (bucket_id + 1) // bucket_count
    return ordered[start:stop]


def _event_rows(
    symbol: str,
    episode_id: int,
    records: Sequence[Mapping[str, Any]],
    *,
    formal_start: str,
    formal_end: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item.update(
            {
                "symbol": symbol,
                "episode_id": episode_id,
                "event_date": str(item["confirmed_time"])[:10],
                "is_formal": formal_start
                <= str(item["confirmed_time"])[:10]
                <= formal_end,
                "schema": EVENT_SCHEMA_VERSION,
            }
        )
        rows.append(item)
    return rows


def _snapshot_rows(
    symbol: str,
    episode_id: int,
    frame: pd.DataFrame,
    records: Sequence[Mapping[str, Any]],
    *,
    formal_start: str,
    formal_end: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records_sorted = sorted(
        records,
        key=lambda item: (
            str(item["confirmed_time"]),
            str(item["event_type"]),
            0 if str(item.get("action", "")) == "opened" else 1,
            str(item["id"]),
        ),
    )
    state_by_candidate: dict[str, str] = {}
    visible_counts: Counter[str] = Counter()
    latest_confirmed_time: str | None = None
    cursor = 0
    rows: list[dict[str, Any]] = []
    for trade_date, day in frame.groupby("trade_date", sort=True):
        trade_date = str(trade_date)
        if trade_date < formal_start or trade_date > formal_end:
            continue
        cutoff = pd.Timestamp(str(trade_date)) + pd.Timedelta(hours=23, minutes=59)
        while (
            cursor < len(records_sorted)
            and pd.Timestamp(str(records_sorted[cursor]["confirmed_time"])) <= cutoff
        ):
            item = records_sorted[cursor]
            visible_counts[str(item["event_type"])] += 1
            latest_confirmed_time = str(item["confirmed_time"])
            if str(item["event_type"]) == "segment_state":
                state_by_candidate[str(item["candidate_id"])] = str(item["action"])
            cursor += 1
        rows.append(
            {
                "schema": SNAPSHOT_SCHEMA_VERSION,
                "symbol": symbol,
                "episode_id": episode_id,
                "trade_date": trade_date,
                "five_minute_bars": len(day),
                "last_bar_timestamp": pd.Timestamp(day["timestamp"].max()).isoformat(),
                "confirmed_event_count": cursor,
                "confirmed_event_counts_json": json.dumps(
                    dict(sorted(visible_counts.items())), sort_keys=True
                ),
                "latest_confirmed_time": latest_confirmed_time,
                "pending_open_count_visible": sum(
                    action == "opened" for action in state_by_candidate.values()
                ),
                "data_complete": len(day) == len(intraday.EXPECTED_5M_TIMES),
            }
        )
    return rows


def _query_symbols(
    spec: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    _, pinned, paths, manifests = intraday._snapshot(spec)
    cache_path = output_root / "symbol_universe.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("pinned_datasets") == pinned:
            return [str(symbol) for symbol in cached["symbols"]], dict(cached["source"])
    connection, runtime = intraday._connect(output_root / "duckdb_tmp")
    try:
        scan = intraday._scan(paths["market_intraday_5m"])
        frame = connection.execute(
            f"""
            SELECT DISTINCT symbol
            FROM {scan}
            WHERE trade_date BETWEEN '2010-01-01' AND '2025-12-31'
            ORDER BY symbol
            """
        ).fetchdf()
    finally:
        connection.close()
    symbols = frame["symbol"].astype(str).tolist()
    source = {
        "pinned_datasets": pinned,
        "dataset_manifests": manifests,
        "runtime": runtime,
    }
    _write_json(
        cache_path,
        {
            "schema": "seq100_strict_chan_panel_symbol_universe/1",
            "pinned_datasets": pinned,
            "symbols": symbols,
            "source": source,
        },
    )
    return symbols, source


def _panel_spec() -> dict[str, Any]:
    spec = parser.load_definition_spec()
    return spec


def _config_fingerprint(
    *,
    bucket_id: int,
    bucket_count: int,
    symbols: Sequence[str],
    definition_sha256: str,
    implementation_sha256: str,
    source: Mapping[str, Any],
) -> str:
    payload = {
        "study_id": STUDY_ID,
        "bucket_id": bucket_id,
        "bucket_count": bucket_count,
        "symbols": list(symbols),
        "definition_sha256": definition_sha256,
        "implementation_sha256": implementation_sha256,
        "pinned_datasets": dict(source.get("pinned_datasets", {})),
        "formal_start": "2012-01-01",
        "formal_end": "2025-12-31",
        "burn_in_start": "2010-01-01",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_panel_partition(
    *,
    bucket_id: int,
    bucket_count: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    maximum_symbols: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if bucket_id < 0 or bucket_id >= bucket_count:
        raise ValueError("strict_chan_panel_bucket_id_invalid")
    spec = _panel_spec()
    symbols, source = _query_symbols(spec, output_root=root)
    members = _bucket_members(
        symbols,
        bucket_id=bucket_id,
        bucket_count=bucket_count,
    )
    if maximum_symbols is not None:
        members = members[: max(int(maximum_symbols), 0)]
    bucket_root = root / f"bucket={bucket_id:04d}"
    done_path = bucket_root / "done.json"
    definition_sha256 = _sha256_file(parser.DEFAULT_DEFINITION_PATH)
    implementation_sha256 = _sha256_file(Path(__file__))
    input_fingerprint = _config_fingerprint(
        bucket_id=bucket_id,
        bucket_count=bucket_count,
        symbols=members,
        definition_sha256=definition_sha256,
        implementation_sha256=implementation_sha256,
        source=source,
    )
    if resume and done_path.is_file() and maximum_symbols is None and not dry_run:
        completed = json.loads(done_path.read_text(encoding="utf-8"))
        if completed.get("input_fingerprint") == input_fingerprint:
            return completed
    result: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "dry_run" if dry_run else "running",
        "study_id": STUDY_ID,
        "bucket_id": bucket_id,
        "bucket_count": bucket_count,
        "symbol_count_total": len(symbols),
        "symbol_count_selected": len(members),
        "symbols": members,
        "source": source,
        "definition_path": str(parser.DEFAULT_DEFINITION_PATH.resolve()),
        "definition_sha256": definition_sha256,
        "implementation_sha256": implementation_sha256,
        "input_fingerprint": input_fingerprint,
        "formal_start": "2012-01-01",
        "formal_end": "2025-12-31",
        "burn_in_start": "2010-01-01",
        "forbidden_year": 2026,
        "events_path": str((bucket_root / "events.parquet").resolve()),
        "snapshots_path": str((bucket_root / "daily_snapshots.parquet").resolve()),
        "done_path": str(done_path.resolve()),
        "profit_claim": False,
    }
    if dry_run:
        return result

    episodes_by_symbol = intraday.load_symbols_episodes(
        members,
        start_date="2010-01-01",
        end_date="2025-12-31",
        definition_path=parser.DEFAULT_DEFINITION_PATH,
        temporary_root=bucket_root / "duckdb_tmp",
        contiguous_symbol_range=True,
    )
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    symbol_summaries: list[dict[str, Any]] = []
    for index, symbol in enumerate(members, start=1):
        episodes = episodes_by_symbol[symbol]
        symbol_event_count = 0
        symbol_snapshot_count = 0
        for episode_id, frame in episodes.frame.groupby("episode_id", sort=True):
            parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(
                drop=True
            )
            parsed = parser.parse_strict_chan(parse_frame)
            records = parsed.event_records()
            event_rows = _event_rows(
                symbol,
                int(episode_id),
                records,
                formal_start="2012-01-01",
                formal_end="2025-12-31",
            )
            snapshot_rows = _snapshot_rows(
                symbol,
                int(episode_id),
                frame,
                records,
                formal_start="2012-01-01",
                formal_end="2025-12-31",
            )
            events.extend(event_rows)
            snapshots.extend(snapshot_rows)
            symbol_event_count += len(event_rows)
            symbol_snapshot_count += len(snapshot_rows)
        symbol_summaries.append(
            {
                "symbol": symbol,
                "index": index,
                "event_count": symbol_event_count,
                "snapshot_count": symbol_snapshot_count,
                "usable_days": int(episodes.audit["usable_adjusted_days"]),
                "missing_positive_days": int(episodes.audit["missing_positive_days"]),
                "episode_count": int(episodes.audit["episode_count"]),
            }
        )

    _write_parquet(bucket_root / "events.parquet", _parquet_safe_event_frame(events))
    _write_parquet(bucket_root / "daily_snapshots.parquet", pd.DataFrame(snapshots))
    result.update(
        {
            "status": "completed",
            "processed_symbols": len(symbol_summaries),
            "event_rows": len(events),
            "formal_event_rows": sum(bool(item["is_formal"]) for item in events),
            "burn_in_event_rows": sum(not bool(item["is_formal"]) for item in events),
            "snapshot_rows": len(snapshots),
            "symbol_summaries": symbol_summaries,
        }
    )
    _write_json(done_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Build a resumable strict Chan panel bucket."
    )
    argument_parser.add_argument("--bucket-id", type=int, required=True)
    argument_parser.add_argument("--bucket-count", type=int, default=256)
    argument_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    argument_parser.add_argument("--max-symbols", type=int, default=None)
    argument_parser.add_argument("--no-resume", action="store_true")
    argument_parser.add_argument("--dry-run", action="store_true")
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_panel_partition(
        bucket_id=args.bucket_id,
        bucket_count=args.bucket_count,
        output_root=args.output_root,
        maximum_symbols=args.max_symbols,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
