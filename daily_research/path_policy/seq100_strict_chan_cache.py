"""Immutable bucket cache for strict Chan five-minute panel inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from daily_research.path_policy import seq100_strict_chan_intraday as intraday
from daily_research.path_policy import seq100_strict_chan_parser as parser

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/cache/seq100_strict_chan_input_cache_v1"
)
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_strict_chan_input_cache_v1.json"
)
STUDY_ID = "seq100_strict_chan_input_cache_v1"
SCHEMA_VERSION = "seq100_strict_chan_input_cache/1"
DOMAIN_ORDER = ("market_intraday_5m", "adjust_factor", "market_daily_raw")
DOMAIN_COLUMNS: dict[str, tuple[str, ...]] = {
    "market_intraday_5m": (
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
        "adjusted_flag",
    ),
    "adjust_factor": (
        "symbol",
        "trade_date",
        "adjust_factor",
        "factor_source_date",
        "ffill_days",
        "factor_semantics",
        "factor_provider",
    ),
    "market_daily_raw": (
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
    ),
}


@dataclass(frozen=True)
class StrictChanInputCache:
    cache_root: Path
    generation_root: Path
    fingerprint: str
    bucket_count: int
    symbols: tuple[str, ...]
    pinned_datasets: Mapping[str, str]
    manifest: Mapping[str, Any]

    def paths_for_bucket(self, bucket_id: int) -> dict[str, tuple[Path, ...]]:
        if bucket_id < 0 or bucket_id >= self.bucket_count:
            raise ValueError("strict_chan_cache_bucket_id_invalid")
        result: dict[str, tuple[Path, ...]] = {}
        for domain in DOMAIN_ORDER:
            domain_root = self.generation_root / "domains" / domain
            bucket_roots = [
                path
                for path in domain_root.glob("bucket_id=*")
                if path.is_dir() and int(path.name.split("=", 1)[1]) == int(bucket_id)
            ]
            paths = tuple(
                sorted(
                    path for root in bucket_roots for path in root.rglob("*.parquet")
                )
            )
            if not paths:
                raise ValueError(
                    f"strict_chan_cache_bucket_domain_missing:{bucket_id}:{domain}"
                )
            result[domain] = paths
        return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _safe_remove_tree(path: Path, *, root: Path) -> None:
    resolved = path.resolve()
    base = root.resolve()
    resolved.relative_to(base)
    if resolved == base:
        raise ValueError("strict_chan_cache_refuse_remove_root")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def _load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = Path(path).resolve()
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("study_id") != STUDY_ID:
        raise ValueError("strict_chan_cache_study_id_mismatch")
    if int(study["partitioning"]["bucket_count"]) <= 0:
        raise ValueError("strict_chan_cache_bucket_count_invalid")
    if tuple(study["domains"]) != DOMAIN_ORDER:
        raise ValueError("strict_chan_cache_domains_mismatch")
    study["_study_path"] = str(study_path)
    return study


def _bucket_assignment_frame(
    symbols: Sequence[str], *, bucket_count: int
) -> pd.DataFrame:
    if bucket_count <= 0:
        raise ValueError("strict_chan_cache_bucket_count_invalid")
    ordered = sorted({str(symbol) for symbol in symbols})
    rows: list[dict[str, Any]] = []
    for position, symbol in enumerate(ordered):
        bucket_id = min(
            ((position + 1) * bucket_count - 1) // max(len(ordered), 1),
            bucket_count - 1,
        )
        rows.append(
            {
                "symbol": symbol,
                "symbol_position": position,
                "bucket_id": bucket_id,
            }
        )
    return pd.DataFrame(rows, columns=("symbol", "symbol_position", "bucket_id"))


def _query_symbols(
    connection: duckdb.DuckDBPyConnection,
    paths: Mapping[str, Sequence[Path]],
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    scan = intraday._scan(paths["market_intraday_5m"])
    frame = connection.execute(
        f"""
        SELECT DISTINCT symbol
        FROM {scan}
        WHERE trade_date BETWEEN ? AND ?
          AND (symbol LIKE '%.SH' OR symbol LIKE '%.SZ')
        ORDER BY symbol
        """,
        [start_date, end_date],
    ).fetchdf()
    return frame["symbol"].astype(str).tolist()


def _cache_fingerprint(
    *,
    study: Mapping[str, Any],
    pinned_datasets: Mapping[str, str],
    symbols: Sequence[str],
    bucket_count: int,
    maximum_symbols: int | None,
) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "study_sha256": _sha256_file(str(study["_study_path"])),
        "pinned_datasets": dict(sorted(pinned_datasets.items())),
        "symbols": list(symbols),
        "bucket_count": bucket_count,
        "maximum_symbols": maximum_symbols,
        "domain_columns": {key: list(value) for key, value in DOMAIN_COLUMNS.items()},
        "start_date": study["period"]["burn_in_start"],
        "end_date": study["period"]["formal_end"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_relation(
    domain: str,
    paths: Mapping[str, Sequence[Path]],
    *,
    start_date: str,
    end_date: str,
) -> str:
    scan = intraday._scan(paths[domain])
    columns = ", ".join(f'"{column}"' for column in DOMAIN_COLUMNS[domain])
    base = f"""
        SELECT {columns}
        FROM {scan}
        WHERE trade_date BETWEEN {_sql_literal(start_date)} AND {_sql_literal(end_date)}
    """
    if domain == "adjust_factor":
        base += """
        QUALIFY row_number() OVER (
            PARTITION BY symbol, trade_date
            ORDER BY factor_source_date DESC NULLS LAST
        ) = 1
        """
    elif domain == "market_daily_raw":
        base += """
        QUALIFY row_number() OVER (
            PARTITION BY symbol, trade_date ORDER BY source DESC NULLS LAST
        ) = 1
        """
    return f"({base})"


def _validate_domain(
    connection: duckdb.DuckDBPyConnection,
    domain_root: Path,
    *,
    expected_rows: int | None,
    formal_end: str,
) -> dict[str, Any]:
    paths = tuple(sorted(domain_root.rglob("*.parquet")))
    if not paths:
        raise ValueError(f"strict_chan_cache_domain_empty:{domain_root.name}")
    scan = intraday._scan(paths)
    row = connection.execute(
        f"""
        SELECT count(*) AS rows,
               min(cast(trade_date AS VARCHAR)) AS start_date,
               max(cast(trade_date AS VARCHAR)) AS end_date,
               count(DISTINCT symbol) AS symbols,
               count(*) FILTER (
                   WHERE cast(trade_date AS VARCHAR) > ?
               ) AS forbidden_rows
        FROM {scan}
        """,
        [formal_end],
    ).fetchone()
    row_count = int(row[0])
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f"strict_chan_cache_row_count_mismatch:{domain_root.name}:"
            f"{row_count}!={expected_rows}"
        )
    if int(row[4]) != 0:
        raise ValueError(f"strict_chan_cache_forbidden_rows:{domain_root.name}")
    records = []
    for path in paths:
        metadata = pq.read_metadata(path)
        records.append(
            {
                "path": str(path.resolve()),
                "file_size": path.stat().st_size,
                "row_count": int(metadata.num_rows),
            }
        )
    return {
        "row_count": row_count,
        "start_date": str(row[1]),
        "end_date": str(row[2]),
        "symbol_count": int(row[3]),
        "forbidden_rows": int(row[4]),
        "file_count": len(paths),
        "files": records,
    }


def _completed_domain_manifest(
    path: Path, *, fingerprint: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "completed"
        or payload.get("fingerprint") != fingerprint
    ):
        return None
    if any(not Path(item["path"]).is_file() for item in payload.get("files", [])):
        return None
    return payload


def _build_domain(
    connection: duckdb.DuckDBPyConnection,
    *,
    domain: str,
    source_paths: Mapping[str, Sequence[Path]],
    generation_root: Path,
    mapping_relation: str,
    fingerprint: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    manifest_path = generation_root / "domain_manifests" / f"{domain}.json"
    completed = _completed_domain_manifest(manifest_path, fingerprint=fingerprint)
    if completed is not None:
        return completed
    final_root = generation_root / "domains" / domain
    if final_root.is_dir():
        _safe_remove_tree(final_root, root=generation_root)
    runtime_root = generation_root / "runtime" / domain
    _safe_remove_tree(runtime_root, root=generation_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    source = _source_relation(
        domain,
        source_paths,
        start_date=start_date,
        end_date=end_date,
    )
    selected = ", ".join(f's."{column}"' for column in DOMAIN_COLUMNS[domain])
    copy_result = connection.execute(
        f"""
        COPY (
            SELECT {selected}, m.bucket_id
            FROM {source} AS s
            INNER JOIN {mapping_relation} AS m ON s.symbol = m.symbol
        ) TO {_sql_literal(final_root)} (
            FORMAT PARQUET,
            PARTITION_BY (bucket_id),
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    ).fetchone()
    copied_rows = (
        int(copy_result[0]) if copy_result and copy_result[0] is not None else None
    )
    validation = _validate_domain(
        connection,
        final_root,
        expected_rows=copied_rows,
        formal_end=end_date,
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "fingerprint": fingerprint,
        "domain": domain,
        **validation,
    }
    _write_json(manifest_path, payload)
    return payload


def build_input_cache(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    bucket_count: int | None = None,
    maximum_symbols: int | None = None,
    domains: Sequence[str] = DOMAIN_ORDER,
    resume: bool = True,
) -> dict[str, Any]:
    study = _load_study(study_path)
    selected_domains = tuple(str(domain) for domain in domains)
    unknown = set(selected_domains) - set(DOMAIN_ORDER)
    if unknown:
        raise ValueError(f"strict_chan_cache_unknown_domains:{sorted(unknown)}")
    configured_bucket_count = int(study["partitioning"]["bucket_count"])
    actual_bucket_count = int(bucket_count or configured_bucket_count)
    if actual_bucket_count <= 0:
        raise ValueError("strict_chan_cache_bucket_count_invalid")
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = parser.load_definition_spec()
    _, pinned, source_paths, source_manifests = intraday._snapshot(spec)
    connection, runtime = intraday._connect(root / "duckdb_tmp")
    try:
        symbols = _query_symbols(
            connection,
            source_paths,
            start_date=str(study["period"]["burn_in_start"]),
            end_date=str(study["period"]["formal_end"]),
        )
        if maximum_symbols is not None:
            symbols = symbols[: max(int(maximum_symbols), 0)]
        if not symbols:
            raise ValueError("strict_chan_cache_symbol_universe_empty")
        mapping = _bucket_assignment_frame(symbols, bucket_count=actual_bucket_count)
        fingerprint = _cache_fingerprint(
            study=study,
            pinned_datasets=pinned,
            symbols=symbols,
            bucket_count=actual_bucket_count,
            maximum_symbols=maximum_symbols,
        )
        generation_root = root / f"generation={fingerprint[:16]}"
        summary_path = generation_root / "manifest.json"
        if resume and summary_path.is_file():
            completed = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                completed.get("status") == "completed"
                and completed.get("fingerprint") == fingerprint
            ):
                if bool(completed.get("full_market")):
                    _write_json(
                        root / "active.json",
                        {
                            "schema": SCHEMA_VERSION,
                            "fingerprint": fingerprint,
                            "manifest_path": str(summary_path.resolve()),
                        },
                    )
                return completed
        generation_root.mkdir(parents=True, exist_ok=True)
        relation = "strict_chan_cache_bucket_mapping"
        connection.register(relation, mapping)
        try:
            domain_manifests = {
                domain: _build_domain(
                    connection,
                    domain=domain,
                    source_paths=source_paths,
                    generation_root=generation_root,
                    mapping_relation=relation,
                    fingerprint=fingerprint,
                    start_date=str(study["period"]["burn_in_start"]),
                    end_date=str(study["period"]["formal_end"]),
                )
                for domain in selected_domains
            }
        finally:
            connection.unregister(relation)
    finally:
        connection.close()
    full_market = maximum_symbols is None and set(selected_domains) == set(DOMAIN_ORDER)
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "completed" if full_market else "partial",
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "full_market": full_market,
        "bucket_count": actual_bucket_count,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "pinned_datasets": pinned,
        "source_manifests": source_manifests,
        "runtime": runtime,
        "domains": domain_manifests,
        "generation_root": str(generation_root.resolve()),
        "manifest_path": str(summary_path.resolve()),
        "forbidden_year": 2026,
    }
    _write_json(summary_path, summary)
    if full_market:
        _write_json(
            root / "active.json",
            {
                "schema": SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "manifest_path": str(summary_path.resolve()),
            },
        )
    return summary


def resolve_active_cache(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    bucket_count: int,
) -> StrictChanInputCache | None:
    root = Path(cache_root).resolve()
    active_path = root / "active.json"
    if not active_path.is_file():
        return None
    active = json.loads(active_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(active["manifest_path"])).resolve()
    manifest_path.relative_to(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or not bool(manifest.get("full_market")):
        raise ValueError("strict_chan_cache_active_not_complete")
    if str(manifest.get("fingerprint")) != str(active.get("fingerprint")):
        raise ValueError("strict_chan_cache_active_fingerprint_mismatch")
    if int(manifest.get("bucket_count", -1)) != int(bucket_count):
        return None
    spec = parser.load_definition_spec(definition_path)
    expected = {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(spec["data_contract"])["datasets"].items()
    }
    if dict(manifest.get("pinned_datasets", {})) != expected:
        return None
    generation_root = Path(str(manifest["generation_root"])).resolve()
    generation_root.relative_to(root)
    view = StrictChanInputCache(
        cache_root=root,
        generation_root=generation_root,
        fingerprint=str(manifest["fingerprint"]),
        bucket_count=int(manifest["bucket_count"]),
        symbols=tuple(str(symbol) for symbol in manifest["symbols"]),
        pinned_datasets=expected,
        manifest=manifest,
    )
    view.paths_for_bucket(0)
    return view


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    argument_parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    argument_parser.add_argument("--bucket-count", type=int, default=None)
    argument_parser.add_argument("--max-symbols", type=int, default=None)
    argument_parser.add_argument(
        "--domains", nargs="+", choices=DOMAIN_ORDER, default=list(DOMAIN_ORDER)
    )
    argument_parser.add_argument("--no-resume", action="store_true")
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_input_cache(
        study_path=args.study,
        cache_root=args.cache_root,
        bucket_count=args.bucket_count,
        maximum_symbols=args.max_symbols,
        domains=args.domains,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
