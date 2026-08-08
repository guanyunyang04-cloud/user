"""QDP five-minute adapter and bounded real-data runner for the strict parser."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

from daily_research.path_policy import seq100_strict_chan_parser as parser

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "output"
    / "path_policy"
    / "studies"
    / parser.STUDY_ID
)
EXPECTED_5M_TIMES = tuple(
    pd.date_range("09:35", "11:30", freq="5min").strftime("%H%M00000")
) + tuple(pd.date_range("13:05", "15:00", freq="5min").strftime("%H%M00000"))
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class IntradayEpisodes:
    frame: pd.DataFrame
    audit: Mapping[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _variant_profile(variant_overrides: Mapping[str, str] | None) -> str:
    if not variant_overrides:
        return "primary"
    parts = []
    for name, value in sorted(dict(variant_overrides).items()):
        token = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{name}-{value}")
        parts.append(token.strip("_"))
    return "__".join(parts)


def _scan(paths: Sequence[Path]) -> str:
    literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], union_by_name=true, hive_partitioning=false)"


def _connect(
    temporary_root: Path | None = None,
) -> tuple[duckdb.DuckDBPyConnection, dict[str, Any]]:
    available = int(psutil.virtual_memory().available)
    cpu_count = int(psutil.cpu_count(logical=True) or 1)
    reserve = 2 * 1024**3
    usable = max(available - reserve, 512 * 1024**2)
    memory_limit = min(max(int(usable * 0.60), 512 * 1024**2), 16 * 1024**3)
    threads = max(1, min(8, cpu_count, int(max(usable, 1) / (1024**3)) + 1))
    connection = duckdb.connect()
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit='{max(memory_limit // (1024**2), 512)}MB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET enable_progress_bar=false")
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
        escaped = str(temporary_root.resolve()).replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped}'")
    return connection, {
        "available_memory_gb_at_start": available / (1024**3),
        "duckdb_memory_limit_gb": memory_limit / (1024**3),
        "duckdb_threads": threads,
        "reserve_memory_gb": reserve / (1024**3),
    }


def _snapshot(
    spec: Mapping[str, Any],
) -> tuple[Path, dict[str, str], dict[str, tuple[Path, ...]], dict[str, Any]]:
    root = qdp_v2_root(WORKSPACE_ROOT)
    active = active_dataset_map(read_active_manifest(root))
    pinned = {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(spec["data_contract"])["datasets"].items()
    }
    stale = {
        domain: {"pinned": dataset_id, "active": active.get(domain, "")}
        for domain, dataset_id in pinned.items()
        if str(active.get(domain, "")) != dataset_id
    }
    if stale:
        raise ValueError(
            f"strict_chan_pinned_qdp_dataset_not_active:{json.dumps(stale, sort_keys=True)}"
        )
    paths: dict[str, tuple[Path, ...]] = {}
    manifests: dict[str, Any] = {}
    for domain, dataset_id in pinned.items():
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            raise ValueError(f"strict_chan_qdp_manifest_missing:{domain}:{dataset_id}")
        manifest = read_dataset_manifest(manifest_path)
        resolved = tuple(
            resolve_manifest_path(item.path, root=root) for item in manifest.shards
        )
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise ValueError(f"strict_chan_qdp_shards_missing:{domain}:{len(missing)}")
        paths[domain] = resolved
        malformed = [
            {
                "path": item.path,
                "start_date": item.start_date,
                "end_date": item.end_date,
            }
            for item in manifest.shards
            if not ISO_DATE_PATTERN.fullmatch(str(item.start_date))
            or not ISO_DATE_PATTERN.fullmatch(str(item.end_date))
        ]
        manifests[domain] = {
            "dataset_id": manifest.dataset_id,
            "row_count": manifest.row_count,
            "shard_count": len(manifest.shards),
            "malformed_shard_date_metadata": malformed,
            "date_filter_uses_actual_parquet_column": True,
        }
    return root, pinned, paths, manifests


def _query_symbol_inputs(
    connection: duckdb.DuckDBPyConnection,
    paths: Mapping[str, Sequence[Path]],
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intraday_scan = _scan(paths["market_intraday_5m"])
    factor_scan = _scan(paths["adjust_factor"])
    daily_scan = _scan(paths["market_daily_raw"])
    intraday = connection.execute(
        f"""
        SELECT symbol, trade_date, bar_time, open, high, low, close,
               volume, amount, source, adjusted_flag
        FROM {intraday_scan}
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date, bar_time
        """,
        [symbol, start_date, end_date],
    ).fetchdf()
    factors = connection.execute(
        f"""
        SELECT symbol, trade_date, adjust_factor, factor_source_date,
               ffill_days, factor_semantics, factor_provider
        FROM {factor_scan}
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        QUALIFY row_number() OVER (
            PARTITION BY symbol, trade_date
            ORDER BY factor_source_date DESC NULLS LAST
        ) = 1
        ORDER BY trade_date
        """,
        [symbol, start_date, end_date],
    ).fetchdf()
    daily = connection.execute(
        f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM {daily_scan}
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        QUALIFY row_number() OVER (
            PARTITION BY symbol, trade_date ORDER BY source DESC NULLS LAST
        ) = 1
        ORDER BY trade_date
        """,
        [symbol, start_date, end_date],
    ).fetchdf()
    return intraday, factors, daily


def _complete_intraday_dates(intraday: pd.DataFrame) -> tuple[set[str], dict[str, Any]]:
    expected = set(EXPECTED_5M_TIMES)
    complete: set[str] = set()
    unexpected_rows = 0
    duplicate_rows = int(
        intraday.duplicated(["symbol", "trade_date", "bar_time"]).sum()
    )
    day_records: list[dict[str, Any]] = []
    for trade_date, day in intraday.groupby("trade_date", sort=True):
        times = day["bar_time"].astype(str)
        unique_times = set(times)
        unexpected = unique_times - expected
        missing = expected - unique_times
        unexpected_rows += int(times.isin(unexpected).sum())
        valid_prices = (
            np.isfinite(
                day[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
            ).all()
            and (day[["open", "high", "low", "close"]] > 0).all().all()
            and (day["high"] >= day[["open", "close", "low"]].max(axis=1)).all()
            and (day["low"] <= day[["open", "close", "high"]].min(axis=1)).all()
        )
        is_complete = (
            len(day) == len(EXPECTED_5M_TIMES)
            and len(unique_times) == len(EXPECTED_5M_TIMES)
            and not unexpected
            and not missing
            and valid_prices
        )
        if is_complete:
            complete.add(str(trade_date))
        elif len(day_records) < 50:
            day_records.append(
                {
                    "trade_date": str(trade_date),
                    "rows": len(day),
                    "distinct_times": len(unique_times),
                    "missing_times": sorted(missing),
                    "unexpected_times": sorted(unexpected),
                    "valid_prices": bool(valid_prices),
                }
            )
    return complete, {
        "duplicate_primary_key_rows": duplicate_rows,
        "unexpected_time_rows": unexpected_rows,
        "invalid_day_examples": day_records,
    }


def _factor_dates(factors: pd.DataFrame) -> tuple[set[str], dict[str, Any]]:
    if factors.empty:
        return set(), {"invalid_factor_rows": 0, "future_source_rows": 0}
    values = pd.to_numeric(factors["adjust_factor"], errors="coerce")
    source = pd.to_datetime(factors["factor_source_date"], errors="coerce")
    dates = pd.to_datetime(factors["trade_date"], errors="coerce")
    future = source.notna() & dates.notna() & source.gt(dates)
    valid = np.isfinite(values) & values.gt(0) & ~future
    return set(factors.loc[valid, "trade_date"].astype(str)), {
        "factor_rows": len(factors),
        "invalid_factor_rows": int((~(np.isfinite(values) & values.gt(0))).sum()),
        "future_source_rows": int(future.sum()),
        "factor_semantics": sorted(
            factors.loc[valid, "factor_semantics"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "factor_providers": sorted(
            factors.loc[valid, "factor_provider"].dropna().astype(str).unique().tolist()
        ),
    }


def _positive_daily_dates(daily: pd.DataFrame) -> set[str]:
    if daily.empty:
        return set()
    numeric = daily[["open", "high", "low", "close", "volume", "amount"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
        & numeric["open"].gt(0)
        & numeric["high"].gt(0)
        & numeric["low"].gt(0)
        & numeric["close"].gt(0)
        & numeric["volume"].gt(0)
        & numeric["amount"].gt(0)
    )
    return set(daily.loc[valid, "trade_date"].astype(str))


def _assign_episode_ids(
    dates: Sequence[str], missing_positive_dates: set[str]
) -> dict[str, int]:
    missing_sorted = sorted(missing_positive_dates)
    episode = 0
    previous: str | None = None
    mapping: dict[str, int] = {}
    for trade_date in sorted(dates):
        if previous is not None and any(
            previous < missing < trade_date for missing in missing_sorted
        ):
            episode += 1
        mapping[trade_date] = episode
        previous = trade_date
    return mapping


def load_symbol_episodes(
    symbol: str,
    *,
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    temporary_root: str | Path | None = None,
) -> IntradayEpisodes:
    spec = parser.load_definition_spec(definition_path)
    contract = dict(spec["data_contract"])
    if start_date < str(contract["burn_in_start"]):
        raise ValueError("strict_chan_intraday_start_before_contract")
    if end_date > str(contract["formal_end"]) or end_date.startswith(
        str(contract["forbidden_year"])
    ):
        raise ValueError("strict_chan_intraday_forbidden_date")
    _, pinned, paths, manifests = _snapshot(spec)
    temporary = Path(temporary_root) if temporary_root else None
    connection, runtime = _connect(temporary)
    try:
        intraday, factors, daily = _query_symbol_inputs(
            connection,
            paths,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        connection.close()
    if intraday.empty:
        raise ValueError(f"strict_chan_intraday_symbol_empty:{symbol}")
    intraday["trade_date"] = intraday["trade_date"].astype(str)
    intraday["bar_time"] = intraday["bar_time"].astype(str).str.zfill(9)
    factors["trade_date"] = factors["trade_date"].astype(str)
    daily["trade_date"] = daily["trade_date"].astype(str)
    complete_dates, bar_audit = _complete_intraday_dates(intraday)
    factor_dates, factor_audit = _factor_dates(factors)
    positive_dates = _positive_daily_dates(daily)
    usable_dates = complete_dates & factor_dates & positive_dates
    missing_positive = positive_dates - usable_dates
    episode_by_date = _assign_episode_ids(sorted(usable_dates), missing_positive)
    factor_values = factors.loc[
        factors["trade_date"].isin(usable_dates), ["trade_date", "adjust_factor"]
    ].copy()
    factor_values["adjust_factor"] = pd.to_numeric(
        factor_values["adjust_factor"], errors="raise"
    )
    selected = intraday[intraday["trade_date"].isin(usable_dates)].copy()
    selected = selected.merge(
        factor_values,
        on="trade_date",
        how="inner",
        validate="many_to_one",
    )
    selected["timestamp"] = pd.to_datetime(
        selected["trade_date"] + selected["bar_time"].str[:6],
        format="%Y-%m-%d%H%M%S",
        errors="raise",
    )
    for name in ("open", "high", "low", "close"):
        selected[f"raw_{name}"] = pd.to_numeric(selected[name], errors="raise")
        selected[name] = selected[f"raw_{name}"] * selected["adjust_factor"]
    selected["volume"] = pd.to_numeric(selected["volume"], errors="raise")
    selected["amount"] = pd.to_numeric(selected["amount"], errors="raise")
    selected["episode_id"] = selected["trade_date"].map(episode_by_date).astype(int)
    selected = selected.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    if (selected["trade_date"] >= "2026-01-01").any():
        raise ValueError("strict_chan_intraday_forbidden_rows")
    audit = {
        "schema": "seq100_strict_chan_intraday_input/1",
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "pinned_datasets": pinned,
        "dataset_manifests": manifests,
        "runtime": runtime,
        "raw_intraday_rows": len(intraday),
        "daily_rows": len(daily),
        "positive_daily_days": len(positive_dates),
        "complete_intraday_days": len(complete_dates),
        "usable_adjusted_days": len(usable_dates),
        "missing_positive_days": len(missing_positive),
        "missing_positive_day_examples": sorted(missing_positive)[:50],
        "episode_count": int(selected["episode_id"].nunique()) if len(selected) else 0,
        "usable_rows": len(selected),
        "expected_rows_from_days": len(usable_dates) * len(EXPECTED_5M_TIMES),
        "forbidden_2026_rows": int((selected["trade_date"] >= "2026-01-01").sum()),
        "price_adjustment": "raw_ohlc * same_day_adjust_factor",
        "missing_positive_day_breaks_episode": True,
        "suspension_day_fabrication": False,
        "bar_audit": bar_audit,
        "factor_audit": factor_audit,
    }
    if len(selected) != audit["expected_rows_from_days"]:
        raise ValueError("strict_chan_intraday_usable_row_count_mismatch")
    return IntradayEpisodes(frame=selected, audit=audit)


def run_symbol_study(
    symbol: str,
    *,
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variant_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    safe_symbol = symbol.replace(".", "_")
    profile = _variant_profile(variant_overrides)
    root = Path(output_root).resolve() / "symbols" / safe_symbol / profile
    episodes = load_symbol_episodes(
        symbol,
        start_date=start_date,
        end_date=end_date,
        definition_path=definition_path,
        temporary_root=root / "duckdb_tmp",
    )
    event_frames: dict[str, list[pd.DataFrame]] = {}
    episode_summaries: list[dict[str, Any]] = []
    for episode_id, frame in episodes.frame.groupby("episode_id", sort=True):
        parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(drop=True)
        result = parser.parse_strict_chan(
            parse_frame,
            definition_path=definition_path,
            variant_overrides=variant_overrides,
        )
        summary = result.summary()
        summary.update(
            {
                "episode_id": int(episode_id),
                "raw_bars": len(parse_frame),
                "start_time": _iso_min(parse_frame["timestamp"]),
                "end_time": _iso_max(parse_frame["timestamp"]),
            }
        )
        episode_summaries.append(summary)
        for event_type, event_frame in result.event_frames().items():
            if event_frame.empty:
                continue
            event_frame.insert(0, "episode_id", int(episode_id))
            event_frame.insert(0, "symbol", symbol)
            event_frames.setdefault(event_type, []).append(event_frame)
    events_root = root / "events"
    events_root.mkdir(parents=True, exist_ok=True)
    event_counts: dict[str, int] = {}
    for event_type, frames in event_frames.items():
        combined = pd.concat(frames, ignore_index=True, sort=False)
        path = events_root / f"{event_type}.parquet"
        combined.to_parquet(path, index=False)
        event_counts[event_type] = len(combined)
    _write_json(root / "input_audit.json", dict(episodes.audit))
    summary = {
        "schema": "seq100_strict_chan_symbol_study/1",
        "status": "completed",
        "study_id": parser.STUDY_ID,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "variant_overrides": dict(variant_overrides or {}),
        "variant_profile": profile,
        "episode_count": len(episode_summaries),
        "event_counts": event_counts,
        "episodes": episode_summaries,
        "input_audit_path": str((root / "input_audit.json").resolve()),
        "events_root": str(events_root.resolve()),
        "profit_claim": False,
    }
    _write_json(root / "summary.json", summary)
    return summary


def _event_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    optional = (
        "direction",
        "kind",
        "side",
        "point_type",
        "level",
        "break_case",
        "action",
        "relation",
        "metric",
    )
    return (
        str(record["event_type"]),
        str(record["id"]),
        int(record["event_index"]),
        int(record["confirmed_index"]),
        *(record.get(name) for name in optional),
    )


def _profile_diagnostics(
    records: Sequence[Mapping[str, Any]], raw_bars: int
) -> dict[str, Any]:
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_type.setdefault(str(record["event_type"]), []).append(record)
    event_types: dict[str, Any] = {}
    for event_type, values in sorted(by_type.items()):
        lag = np.asarray(
            [
                int(item["confirmed_index"]) - int(item["event_index"])
                for item in values
            ],
            dtype=np.int64,
        )
        event_types[event_type] = {
            "count": len(values),
            "per_1000_raw_bars": len(values) * 1000.0 / max(raw_bars, 1),
            "confirmation_lag_bars_median": float(np.median(lag)),
            "confirmation_lag_bars_p90": float(np.quantile(lag, 0.90)),
            "confirmation_lag_bars_maximum": int(lag.max()),
        }
    return {
        "raw_bars": raw_bars,
        "events": len(records),
        "event_types": event_types,
    }


def compare_symbol_variants(
    symbol: str,
    *,
    profiles: Mapping[str, Mapping[str, str]],
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    if "primary" not in profiles:
        raise ValueError("strict_chan_variant_comparison_primary_missing")
    safe_symbol = symbol.replace(".", "_")
    root = Path(output_root).resolve()
    loaded = load_symbol_episodes(
        symbol,
        start_date=start_date,
        end_date=end_date,
        definition_path=definition_path,
        temporary_root=root / "variant_comparisons" / "duckdb_tmp",
    )
    profile_records: dict[str, list[dict[str, Any]]] = {}
    profile_summaries: dict[str, Any] = {}
    for profile_name, overrides in profiles.items():
        records: list[dict[str, Any]] = []
        episode_summaries: list[dict[str, Any]] = []
        for episode_id, frame in loaded.frame.groupby("episode_id", sort=True):
            parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(
                drop=True
            )
            result = parser.parse_strict_chan(
                parse_frame,
                definition_path=definition_path,
                variant_overrides=overrides,
            )
            for record in result.event_records():
                record = dict(record)
                record["episode_id"] = int(episode_id)
                records.append(record)
            episode_summary = result.summary()
            episode_summary["episode_id"] = int(episode_id)
            episode_summaries.append(episode_summary)
        profile_records[profile_name] = records
        profile_summaries[profile_name] = {
            "variant_overrides": dict(overrides),
            "diagnostics": _profile_diagnostics(records, len(loaded.frame)),
            "episodes": episode_summaries,
        }
    primary_by_type: dict[str, set[tuple[Any, ...]]] = {}
    for record in profile_records["primary"]:
        primary_by_type.setdefault(str(record["event_type"]), set()).add(
            (int(record["episode_id"]), *_event_signature(record))
        )
    disagreements: dict[str, Any] = {}
    for profile_name, records in profile_records.items():
        if profile_name == "primary":
            continue
        comparison_by_type: dict[str, set[tuple[Any, ...]]] = {}
        for record in records:
            comparison_by_type.setdefault(str(record["event_type"]), set()).add(
                (int(record["episode_id"]), *_event_signature(record))
            )
        event_types = sorted(set(primary_by_type) | set(comparison_by_type))
        per_type: dict[str, Any] = {}
        for event_type in event_types:
            primary_set = primary_by_type.get(event_type, set())
            comparison_set = comparison_by_type.get(event_type, set())
            union = primary_set | comparison_set
            intersection = primary_set & comparison_set
            per_type[event_type] = {
                "primary": len(primary_set),
                "comparison": len(comparison_set),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
            }
        disagreements[profile_name] = per_type
    result = {
        "schema": "seq100_strict_chan_variant_comparison/1",
        "status": "completed",
        "study_id": parser.STUDY_ID,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "profit_or_return_test_performed": False,
        "input_audit": dict(loaded.audit),
        "profiles": profile_summaries,
        "disagreement_against_primary": disagreements,
    }
    output_path = (
        root / "variant_comparisons" / f"{safe_symbol}_{start_date}_{end_date}.json"
    )
    result["output_path"] = str(output_path.resolve())
    _write_json(output_path, result)
    return result


def _iso_min(values: pd.Series) -> str:
    return pd.Timestamp(values.min()).isoformat()


def _iso_max(values: pd.Series) -> str:
    return pd.Timestamp(values.max()).isoformat()


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Parse one real QDP symbol with the strict causal Chan grammar."
    )
    argument_parser.add_argument("--symbol", required=True)
    argument_parser.add_argument("--start-date", default="2010-01-01")
    argument_parser.add_argument("--end-date", default="2025-12-31")
    argument_parser.add_argument(
        "--definition", default=str(parser.DEFAULT_DEFINITION_PATH)
    )
    argument_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    argument_parser.add_argument(
        "--stroke-rule",
        choices=(
            "disjoint_fractals_and_price_ranges",
            "disjoint_fractals_only",
        ),
        default="",
    )
    argument_parser.add_argument(
        "--base-component",
        choices=("segment", "stroke_diagnostic"),
        default="",
    )
    argument_parser.add_argument("--compare-stroke-rules", action="store_true")
    argument_parser.add_argument("--compare-frozen-variants", action="store_true")
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.compare_frozen_variants:
        result = compare_symbol_variants(
            args.symbol,
            profiles={
                "primary": {},
                "strict_containment": {"inclusion_equality": "strict_containment"},
                "outer_bar_tiebreak": {"fractal_equality": "outer_bar_tiebreak"},
                "initial_direction_lookahead": {
                    "initial_inclusion_direction": "first_noninclusive_lookahead"
                },
                "disjoint_fractals_only": {"stroke_rule": "disjoint_fractals_only"},
                "stroke_base_diagnostic": {
                    "lowest_center_component": "stroke_diagnostic"
                },
                "fluctuation_interval_relation": {
                    "center_relation": "fluctuation_interval_overlap"
                },
            },
            start_date=args.start_date,
            end_date=args.end_date,
            definition_path=args.definition,
            output_root=args.output_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.compare_stroke_rules:
        result = compare_symbol_variants(
            args.symbol,
            profiles={
                "primary": {},
                "disjoint_fractals_only": {"stroke_rule": "disjoint_fractals_only"},
            },
            start_date=args.start_date,
            end_date=args.end_date,
            definition_path=args.definition,
            output_root=args.output_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    variants = {}
    if args.stroke_rule:
        variants["stroke_rule"] = args.stroke_rule
    if args.base_component:
        variants["lowest_center_component"] = args.base_component
    result = run_symbol_study(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        definition_path=args.definition,
        output_root=args.output_root,
        variant_overrides=variants,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
