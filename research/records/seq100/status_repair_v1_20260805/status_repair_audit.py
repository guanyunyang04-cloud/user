from __future__ import annotations

"""Audit, prepare, apply, and verify the in-place QDP ST-status repair.

The default command is read-only with respect to the active QDP store.  It
materializes reconciled evidence and replacement Parquet files, then runs all
pre-apply invariants.  ``--apply`` is the only path that mutates active data.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[3]
QDP_ROOT = WORKSPACE / "quant_data_platform" / "data" / "qdp_v2"
PACKAGE_SRC = WORKSPACE / "quant_data_platform" / "src"
ACTIVE_PATH = QDP_ROOT / "active" / "active.json"
BACKUP_ROOT = SCRIPT_DIR / "backup_pre_repair"
FREEZE_PATH = SCRIPT_DIR / "freeze_manifest.json"
TUSHARE_PATH = SCRIPT_DIR / "evidence" / "tushare_namechange" / "normalized.parquet"
SSE_EVENTS_PATH = (
    WORKSPACE
    / "quant_data_platform"
    / "src"
    / "quant_data_platform"
    / "qdp_v2"
    / "resources"
    / "sse_st_transitions_2010_2011.csv"
)
ARCHIVE_ROOT = (
    WORKSPACE
    / "daily_research"
    / "data"
    / "research_store"
    / "traditional_quant_baostock_archive_v1"
    / "raw"
)
ARCHIVE_PATHS = (
    ARCHIVE_ROOT
    / "baostock_daily_mainboard_v2_pit_recovery_2012_2015"
    / "daily_bars.parquet",
    ARCHIVE_ROOT
    / "baostock_daily_mainboard_v2_pit"
    / "baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603"
    / "daily_bars.parquet",
)
BURN_IN_START = "2010-01-04"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not_json_serializable:{type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _sql_paths(paths: list[Path] | tuple[Path, ...]) -> str:
    return "[" + ",".join(f"'{_sql_path(path)}'" for path in paths) + "]"


def _dataset_manifest(domain: str, dataset_id: str) -> Path:
    return QDP_ROOT / "datasets" / domain / dataset_id / "dataset.json"


def _dataset_paths(domain: str, dataset_id: str) -> list[Path]:
    manifest = read_json(_dataset_manifest(domain, dataset_id))
    result: list[Path] = []
    for shard in manifest["shards"]:
        path = Path(str(shard["path"]))
        result.append((path if path.is_absolute() else QDP_ROOT / path).resolve())
    return result


def _require_inputs() -> None:
    required = [
        ACTIVE_PATH,
        FREEZE_PATH,
        TUSHARE_PATH,
        SSE_EVENTS_PATH,
        *ARCHIVE_PATHS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("status_repair_inputs_missing:" + "|".join(missing))


def _create_source_views(
    con: duckdb.DuckDBPyConnection,
    active: dict[str, Any],
) -> tuple[str, str]:
    status_id = str(active["datasets"]["security_status"])
    name_id = str(active["datasets"]["name_change"])
    universe_id = str(active["datasets"]["universe_snapshot"])
    con.execute(
        f"CREATE VIEW active_status AS SELECT * FROM read_parquet("
        f"{_sql_paths(_dataset_paths('security_status', status_id))}, union_by_name=true)"
    )
    con.execute(
        f"CREATE VIEW active_names AS SELECT * FROM read_parquet("
        f"{_sql_paths(_dataset_paths('name_change', name_id))}, union_by_name=true)"
    )
    con.execute(
        f"CREATE VIEW active_universe AS SELECT symbol,trade_date,name FROM read_parquet("
        f"{_sql_paths(_dataset_paths('universe_snapshot', universe_id))}, union_by_name=true)"
    )
    con.execute(
        f"""CREATE VIEW archive_status AS
        SELECT upper(code) AS symbol, strftime(date, '%Y-%m-%d') AS trade_date,
               CASE
                 WHEN try_cast(isST AS INTEGER)=1 THEN true
                 WHEN try_cast(isST AS INTEGER)=0 THEN false
                 ELSE NULL
               END AS archive_is_st
        FROM read_parquet({_sql_paths(ARCHIVE_PATHS)}, union_by_name=true)"""
    )
    con.execute(
        f"""CREATE VIEW tushare_intervals AS
        WITH ordered AS (
          SELECT upper(ts_code) AS symbol, trim(name) AS name,
                 try_cast(start_date AS DATE) AS start_date,
                 try_cast(end_date AS DATE) AS supplied_end_date,
                 lead(try_cast(start_date AS DATE)) OVER (
                   PARTITION BY upper(ts_code) ORDER BY try_cast(start_date AS DATE)
                 ) AS next_start
          FROM read_parquet('{_sql_path(TUSHARE_PATH)}')
          WHERE try_cast(start_date AS DATE) IS NOT NULL
            AND trim(coalesce(name, '')) <> ''
        )
        SELECT symbol, name, start_date,
               CASE
                 WHEN supplied_end_date IS NOT NULL AND next_start IS NOT NULL
                   THEN least(
                     supplied_end_date,
                     cast(next_start - INTERVAL 1 DAY AS DATE)
                   )
                 WHEN supplied_end_date IS NOT NULL THEN supplied_end_date
                 WHEN next_start IS NOT NULL
                   THEN cast(next_start - INTERVAL 1 DAY AS DATE)
                 ELSE DATE '2099-12-31'
               END AS end_date,
               regexp_matches(
                 upper(replace(replace(name, ' ', ''), chr(12288), '')),
                 '^(\\*?ST|S\\*?ST|G\\*?ST)'
               ) AS tushare_is_st
        FROM ordered"""
    )
    con.execute(
        f"""CREATE VIEW official_events AS
        SELECT upper(symbol) AS symbol,
               try_cast(effective_date AS DATE) AS event_date,
               try_cast(is_st AS BOOLEAN) AS event_is_st,
               source AS official_source
        FROM read_csv_auto('{_sql_path(SSE_EVENTS_PATH)}', header=true)"""
    )
    con.execute(
        """CREATE VIEW official_first AS
        SELECT symbol, min(event_date) AS first_event_date,
               arg_min(event_is_st, event_date) AS first_event_is_st
        FROM official_events GROUP BY symbol"""
    )
    return status_id, name_id


def _clean_prepared_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory.resolve()
    if SCRIPT_DIR.resolve() not in resolved.parents:
        raise RuntimeError(f"prepared_directory_outside_repair_root:{resolved}")
    for path in resolved.glob("*.parquet"):
        path.unlink()


def _records(frame: Any) -> list[dict[str, Any]]:
    return list(frame.to_dict(orient="records"))


def build_dry_run() -> dict[str, Any]:
    _require_inputs()
    active = read_json(ACTIVE_PATH)
    active_hash = sha256_file(ACTIVE_PATH)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    status_id, name_id = _create_source_views(con, active)
    active_end = str(con.execute("SELECT max(trade_date) FROM active_status").fetchone()[0])

    audit_dir = SCRIPT_DIR / "audit"
    prepared_status_dir = SCRIPT_DIR / "prepared" / "security_status"
    prepared_name_dir = SCRIPT_DIR / "prepared" / "name_change"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _clean_prepared_directory(prepared_status_dir)
    _clean_prepared_directory(prepared_name_dir)
    evidence_path = audit_dir / "status_reconciliation.parquet"
    candidates_path = audit_dir / "status_repair_candidates.parquet"
    evidence_path.unlink(missing_ok=True)
    candidates_path.unlink(missing_ok=True)

    evidence_query = f"""
    WITH joined AS (
      SELECT a.symbol, a.trade_date, a.is_st AS active_is_st,
             a.is_suspended, a.is_delisted,
             a.status_reason AS active_status_reason,
             u.name AS universe_name,
             b.archive_is_st,
             t.name AS tushare_name,
             t.start_date AS tushare_start_date,
             t.end_date AS tushare_end_date,
             t.tushare_is_st,
             f.first_event_date, f.first_event_is_st,
             oe.event_date AS latest_official_event_date,
             oe.event_is_st AS latest_official_event_is_st
      FROM active_status a
      LEFT JOIN active_universe u USING(symbol, trade_date)
      LEFT JOIN archive_status b USING(symbol, trade_date)
      LEFT JOIN LATERAL (
        SELECT name, start_date, end_date, tushare_is_st
        FROM tushare_intervals ti
        WHERE ti.symbol=a.symbol
          AND ti.start_date <= try_cast(a.trade_date AS DATE)
          AND try_cast(a.trade_date AS DATE) <= ti.end_date
        ORDER BY ti.start_date DESC
        LIMIT 1
      ) t ON true
      LEFT JOIN official_first f ON f.symbol=a.symbol
      LEFT JOIN LATERAL (
        SELECT event_date, event_is_st
        FROM official_events e
        WHERE e.symbol=a.symbol
          AND e.event_date <= try_cast(a.trade_date AS DATE)
        ORDER BY event_date DESC
        LIMIT 1
      ) oe ON true
    ), derived AS (
      SELECT *,
        CASE
          WHEN symbol LIKE '%.SH' AND trade_date <= '2011-12-31'
               AND first_event_date IS NOT NULL
            THEN coalesce(latest_official_event_is_st, NOT first_event_is_st)
          ELSE NULL::BOOLEAN
        END AS official_is_st,
        CASE
          WHEN trim(coalesce(universe_name, ''))='' THEN NULL::BOOLEAN
          ELSE regexp_matches(
            upper(replace(replace(universe_name, ' ', ''), chr(12288), '')),
            '^(\\*?ST|S\\*?ST|G\\*?ST)'
          )
        END AS universe_name_is_st
      FROM joined
    ), canonical AS (
      SELECT *,
        coalesce(archive_is_st, official_is_st, tushare_is_st)
          AS canonical_is_st,
        CASE
          WHEN archive_is_st IS NOT NULL THEN 'baostock_archive.daily_isST'
          WHEN official_is_st IS NOT NULL THEN 'sse_official.st_transition'
          WHEN tushare_is_st IS NOT NULL THEN 'tushare.namechange_interval'
          ELSE 'status_unknown'
        END AS status_source,
        CASE
          WHEN archive_is_st IS NOT NULL OR official_is_st IS NOT NULL THEN 'high'
          WHEN tushare_is_st IS NOT NULL THEN 'medium'
          ELSE 'unknown'
        END AS confidence,
        (
          (
            archive_is_st IS NOT NULL
            AND tushare_is_st IS NOT NULL
            AND archive_is_st<>tushare_is_st
          )
          OR
          (
            official_is_st IS NOT NULL
            AND tushare_is_st IS NOT NULL
            AND official_is_st<>tushare_is_st
          )
        ) AS independent_conflict
      FROM derived
    )
    SELECT *,
           canonical_is_st IS DISTINCT FROM active_is_st AS status_changed,
           md5(symbol || '|' || trade_date) AS reconciliation_id
    FROM canonical
    WHERE trade_date BETWEEN '{BURN_IN_START}' AND '{active_end}'
    """
    con.execute(
        f"COPY ({evidence_query}) TO '{_sql_path(evidence_path)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"""COPY (
          SELECT symbol, trade_date, active_is_st, canonical_is_st,
                 status_source, confidence, independent_conflict,
                 archive_is_st, official_is_st, tushare_is_st,
                 universe_name_is_st, tushare_start_date, tushare_end_date,
                 reconciliation_id
          FROM read_parquet('{_sql_path(evidence_path)}')
          WHERE status_changed OR independent_conflict OR canonical_is_st IS NULL
          ORDER BY trade_date, symbol
        ) TO '{_sql_path(candidates_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )

    prepared_status_paths: list[Path] = []
    for year in range(int(BURN_IN_START[:4]), int(active_end[:4]) + 1):
        target = prepared_status_dir / f"security_status_{year}.parquet"
        con.execute(
            f"""COPY (
              SELECT symbol, trade_date, canonical_is_st AS is_st,
                     is_suspended, is_delisted,
                     CASE
                       WHEN is_delisted THEN 'delisted'
                       WHEN canonical_is_st IS NULL AND is_suspended
                         THEN 'status_unknown;suspended'
                       WHEN canonical_is_st IS NULL THEN 'status_unknown'
                       WHEN canonical_is_st AND is_suspended THEN 'st;suspended'
                       WHEN canonical_is_st THEN 'st'
                       WHEN is_suspended THEN 'suspended'
                       ELSE 'tradeable'
                     END AS status_reason,
                     status_source AS source
              FROM read_parquet('{_sql_path(evidence_path)}')
              WHERE year(try_cast(trade_date AS DATE))={year}
              ORDER BY trade_date, symbol
            ) TO '{_sql_path(target)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)"""
        )
        prepared_status_paths.append(target)

    prepared_name_path = prepared_name_dir / "name_change.parquet"
    con.execute(
        f"""COPY (
          SELECT symbol, trade_date, old_name, new_name, change_type, source
          FROM active_names
          WHERE lower(trim(coalesce(source, '')))
                <> 'qdp_universe_tail_transition'
          ORDER BY trade_date, symbol, change_type
        ) TO '{_sql_path(prepared_name_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )

    prepared_scan = (
        f"read_parquet({_sql_paths(prepared_status_paths)}, union_by_name=true)"
    )
    source_summary = con.execute(
        f"""SELECT status_source, confidence, count(*) AS row_count,
                   count(DISTINCT symbol) AS symbol_count,
                   sum(status_changed) AS changed_rows,
                   sum(canonical_is_st IS NULL) AS unknown_rows
            FROM read_parquet('{_sql_path(evidence_path)}')
            GROUP BY ALL ORDER BY row_count DESC"""
    ).fetchdf()
    source_summary.to_csv(
        audit_dir / "source_summary.csv", index=False, encoding="utf-8-sig"
    )
    impact_by_year = con.execute(
        f"""SELECT year(try_cast(trade_date AS DATE)) AS year,
                   count(*) AS row_count,
                   sum(status_changed) AS changed_rows,
                   count(DISTINCT CASE WHEN status_changed THEN symbol END)
                     AS changed_symbols,
                   sum(NOT active_is_st AND canonical_is_st) AS newly_st,
                   sum(active_is_st AND NOT canonical_is_st) AS no_longer_st,
                   sum(canonical_is_st IS NULL) AS unknown_rows
            FROM read_parquet('{_sql_path(evidence_path)}')
            GROUP BY 1 ORDER BY 1"""
    ).fetchdf()
    impact_by_year.to_csv(
        audit_dir / "impact_by_year.csv", index=False, encoding="utf-8-sig"
    )
    conflicts = con.execute(
        f"""SELECT symbol, trade_date, archive_is_st, official_is_st,
                   tushare_is_st, tushare_start_date, tushare_end_date,
                   status_source
            FROM read_parquet('{_sql_path(evidence_path)}')
            WHERE independent_conflict
            ORDER BY trade_date, symbol"""
    ).fetchdf()
    conflicts.to_csv(
        audit_dir / "independent_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    active_rows = int(con.execute("SELECT count(*) FROM active_status").fetchone()[0])
    prepared_rows = int(con.execute(f"SELECT count(*) FROM {prepared_scan}").fetchone()[0])
    active_name_rows = int(con.execute("SELECT count(*) FROM active_names").fetchone()[0])
    prepared_name_rows = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(prepared_name_path)}')"
        ).fetchone()[0]
    )
    checks: dict[str, Any] = {
        "active_pointer_unchanged_during_dry_run": (
            sha256_file(ACTIVE_PATH) == active_hash
        ),
        "status_row_count_equal": prepared_rows == active_rows,
        "status_primary_key_duplicates": int(
            con.execute(
                f"""SELECT count(*) FROM (
                    SELECT trade_date,symbol,count(*) n
                    FROM {prepared_scan} GROUP BY ALL HAVING n>1)"""
            ).fetchone()[0]
        ),
        "missing_prepared_status_keys": int(
            con.execute(
                f"""SELECT count(*) FROM (
                    SELECT trade_date,symbol FROM active_status
                    EXCEPT
                    SELECT trade_date,symbol FROM {prepared_scan})"""
            ).fetchone()[0]
        ),
        "extra_prepared_status_keys": int(
            con.execute(
                f"""SELECT count(*) FROM (
                    SELECT trade_date,symbol FROM {prepared_scan}
                    EXCEPT
                    SELECT trade_date,symbol FROM active_status)"""
            ).fetchone()[0]
        ),
        "changed_non_st_status_fields": int(
            con.execute(
                f"""SELECT count(*)
                    FROM active_status a JOIN {prepared_scan} p USING(symbol,trade_date)
                    WHERE a.is_suspended IS DISTINCT FROM p.is_suspended
                       OR a.is_delisted IS DISTINCT FROM p.is_delisted"""
            ).fetchone()[0]
        ),
        "archive_mismatches_2012_2025": int(
            con.execute(
                f"""SELECT count(*)
                    FROM {prepared_scan} p JOIN archive_status b USING(symbol,trade_date)
                    WHERE p.trade_date BETWEEN '2012-01-04' AND '2025-12-31'
                      AND p.is_st IS DISTINCT FROM b.archive_is_st"""
            ).fetchone()[0]
        ),
        "independent_conflict_rows": int(len(conflicts)),
        "name_change_rows_removed": active_name_rows - prepared_name_rows,
        "prepared_name_primary_key_duplicates": int(
            con.execute(
                f"""SELECT count(*) FROM (
                    SELECT symbol,trade_date,change_type,count(*) n
                    FROM read_parquet('{_sql_path(prepared_name_path)}')
                    GROUP BY ALL HAVING n>1)"""
            ).fetchone()[0]
        ),
        "prepared_name_synthetic_rows": int(
            con.execute(
                f"""SELECT count(*)
                    FROM read_parquet('{_sql_path(prepared_name_path)}')
                    WHERE lower(trim(coalesce(source,'')))
                          ='qdp_universe_tail_transition'"""
            ).fetchone()[0]
        ),
        "unknown_status_rows": int(
            con.execute(
                f"SELECT count(*) FROM {prepared_scan} WHERE is_st IS NULL"
            ).fetchone()[0]
        ),
        "historical_unknown_status_rows": int(
            con.execute(
                f"""SELECT count(*) FROM {prepared_scan}
                    WHERE is_st IS NULL AND trade_date<='2011-12-31'"""
            ).fetchone()[0]
        ),
        "post_archive_unknown_status_rows": int(
            con.execute(
                f"""SELECT count(*) FROM {prepared_scan}
                    WHERE is_st IS NULL AND trade_date>'2026-06-01'"""
            ).fetchone()[0]
        ),
    }
    boundary = con.execute(
        f"""SELECT trade_date, sum(is_st) AS st_rows
            FROM {prepared_scan}
            WHERE trade_date IN ('2011-11-21','2011-11-22')
            GROUP BY 1 ORDER BY 1"""
    ).fetchdf()
    checks["boundary_st_rows"] = _records(boundary)
    checks["boundary_st_jump"] = (
        abs(int(boundary.iloc[1]["st_rows"]) - int(boundary.iloc[0]["st_rows"]))
        if len(boundary) == 2
        else None
    )
    checks["000972_wrong_st_before_2012_05_02"] = int(
        con.execute(
            f"""SELECT count(*) FROM {prepared_scan}
                WHERE symbol='000972.SZ'
                  AND trade_date BETWEEN '2011-11-22' AND '2012-04-27'
                  AND is_st"""
        ).fetchone()[0]
    )
    checks["000972_is_st_on_2012_05_02"] = bool(
        con.execute(
            f"""SELECT is_st FROM {prepared_scan}
                WHERE symbol='000972.SZ' AND trade_date='2012-05-02'"""
        ).fetchone()[0]
    )

    zero_checks = (
        "status_primary_key_duplicates",
        "missing_prepared_status_keys",
        "extra_prepared_status_keys",
        "changed_non_st_status_fields",
        "archive_mismatches_2012_2025",
        "prepared_name_primary_key_duplicates",
        "prepared_name_synthetic_rows",
        "000972_wrong_st_before_2012_05_02",
    )
    ready = (
        bool(checks["active_pointer_unchanged_during_dry_run"])
        and bool(checks["status_row_count_equal"])
        and all(int(checks[name]) == 0 for name in zero_checks)
        and int(checks["name_change_rows_removed"]) == 24
        and checks["boundary_st_jump"] is not None
        and int(checks["boundary_st_jump"]) <= 5
        and bool(checks["000972_is_st_on_2012_05_02"])
    )
    summary = {
        "schema": "seq100_status_repair_dry_run/v2",
        "created_at": now_utc(),
        "status": "ready_to_apply" if ready else "dry_run_failed",
        "active_dataset_ids": {
            "security_status": status_id,
            "name_change": name_id,
        },
        "active_manifest_sha256": active_hash,
        "scope": {"start_date": BURN_IN_START, "end_date": active_end},
        "source_priority": [
            "baostock_archive.daily_isST",
            "sse_official.st_transition",
            "tushare.namechange_interval",
            "status_unknown",
        ],
        "tushare_interval_rule": (
            "min(supplied_end_date, next_start_date - 1 day)"
        ),
        "checks": checks,
        "source_summary": _records(source_summary),
        "impact_by_year": _records(impact_by_year),
        "independent_conflicts": _records(conflicts),
        "outputs": {
            "evidence": str(evidence_path),
            "candidates": str(candidates_path),
            "prepared_security_status": [
                str(path) for path in prepared_status_paths
            ],
            "prepared_name_change": str(prepared_name_path),
        },
        "active_write_performed": False,
        "model_retrained": False,
    }
    write_json(audit_dir / "audit_summary.json", summary)
    con.close()
    return summary


def verify_backup() -> dict[str, Any]:
    freeze = read_json(FREEZE_PATH)
    frozen_active = SCRIPT_DIR / "frozen" / "active.json"
    backup_active = BACKUP_ROOT / "active.json"
    errors: list[str] = []
    if (
        not backup_active.is_file()
        or not frozen_active.is_file()
        or sha256_file(backup_active) != sha256_file(frozen_active)
    ):
        errors.append("active_json_hash_mismatch")
    for domain in ("security_status", "name_change"):
        record = dict(freeze["datasets"][domain])
        dataset_id = str(record["dataset_id"])
        backup_dataset = BACKUP_ROOT / domain / dataset_id
        backup_manifest = backup_dataset / "dataset.json"
        if (
            not backup_manifest.is_file()
            or sha256_file(backup_manifest) != str(record["manifest_sha256"])
        ):
            errors.append(f"{domain}:manifest_hash_mismatch")
        for shard in record["shards"]:
            target = backup_dataset / "shards" / Path(str(shard["path"])).name
            if (
                not target.is_file()
                or sha256_file(target) != str(shard["sha256"])
            ):
                errors.append(f"{domain}:shard_hash_mismatch:{target.name}")
    if errors:
        raise RuntimeError("backup_verification_failed:" + "|".join(errors))
    return {"status": "verified", "root": str(BACKUP_ROOT), "errors": []}


def verify_active(summary: dict[str, Any]) -> dict[str, Any]:
    active = read_json(ACTIVE_PATH)
    expected = dict(summary["active_dataset_ids"])
    if any(
        str(active["datasets"].get(domain, "")) != str(dataset_id)
        for domain, dataset_id in expected.items()
    ):
        raise RuntimeError("active_dataset_id_changed")
    con = duckdb.connect()
    status_paths = _dataset_paths("security_status", expected["security_status"])
    name_paths = _dataset_paths("name_change", expected["name_change"])
    prepared_paths = [
        Path(item) for item in summary["outputs"]["prepared_security_status"]
    ]
    status_scan = f"read_parquet({_sql_paths(status_paths)}, union_by_name=true)"
    prepared_scan = (
        f"read_parquet({_sql_paths(prepared_paths)}, union_by_name=true)"
    )
    result = {
        "dataset_ids_unchanged": True,
        "active_pointer_sha256_unchanged": (
            sha256_file(ACTIVE_PATH) == summary["active_manifest_sha256"]
        ),
        "security_status_row_count": int(
            con.execute(f"SELECT count(*) FROM {status_scan}").fetchone()[0]
        ),
        "security_status_difference_rows": int(
            con.execute(
                f"""SELECT count(*) FROM (
                  (SELECT * FROM {status_scan} EXCEPT SELECT * FROM {prepared_scan})
                  UNION ALL
                  (SELECT * FROM {prepared_scan} EXCEPT SELECT * FROM {status_scan})
                )"""
            ).fetchone()[0]
        ),
        "security_status_primary_key_duplicates": int(
            con.execute(
                f"""SELECT count(*) FROM (
                    SELECT trade_date,symbol,count(*) n
                    FROM {status_scan} GROUP BY ALL HAVING n>1)"""
            ).fetchone()[0]
        ),
        "name_change_synthetic_rows": int(
            con.execute(
                f"""SELECT count(*)
                    FROM read_parquet({_sql_paths(name_paths)}, union_by_name=true)
                    WHERE lower(trim(coalesce(source,'')))
                          ='qdp_universe_tail_transition'"""
            ).fetchone()[0]
        ),
    }
    result["passed"] = (
        result["active_pointer_sha256_unchanged"]
        and result["security_status_difference_rows"] == 0
        and result["security_status_primary_key_duplicates"] == 0
        and result["name_change_synthetic_rows"] == 0
    )
    con.close()
    if not result["passed"]:
        raise RuntimeError(
            "post_apply_verification_failed:"
            + json.dumps(result, ensure_ascii=False, default=_json_default)
        )
    return result


def apply_repair(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "ready_to_apply":
        raise RuntimeError("status_repair_dry_run_not_ready")
    backup = verify_backup()
    if str(PACKAGE_SRC) not in sys.path:
        sys.path.insert(0, str(PACKAGE_SRC))
    from quant_data_platform.qdp_v2.repair import (
        replace_active_table_from_parquet,
    )

    status_result = replace_active_table_from_parquet(
        "security_status",
        [Path(item) for item in summary["outputs"]["prepared_security_status"]],
        reason="repair ST status from dated independent evidence",
        workspace_root=WORKSPACE,
        source_updates={
            "st_status_policy": (
                "baostock_daily_then_sse_transition_then_tushare_interval_"
                "else_unknown"
            ),
            "st_status_repaired_at": now_utc(),
            "st_status_evidence": str(summary["outputs"]["evidence"]),
        },
        quality_updates={
            "primary_key_unique": True,
            "unknown_is_st_preserved_as_null": True,
            "archive_mismatches_2012_2025": 0,
            "independent_conflict_rows": int(
                summary["checks"]["independent_conflict_rows"]
            ),
        },
    )
    name_result = replace_active_table_from_parquet(
        "name_change",
        Path(summary["outputs"]["prepared_name_change"]),
        reason="remove synthetic universe-tail name transitions",
        workspace_root=WORKSPACE,
        source_updates={
            "synthetic_universe_tail_transitions_removed_at": now_utc(),
        },
        quality_updates={"synthetic_universe_tail_transition_rows": 0},
    )
    verification = verify_active(summary)
    result = {
        "status": "applied_and_verified",
        "created_at": now_utc(),
        "backup": backup,
        "security_status": status_result,
        "name_change": name_result,
        "verification": verification,
    }
    write_json(SCRIPT_DIR / "audit" / "apply_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace both active tables after a passing dry-run",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        summary = read_json(SCRIPT_DIR / "audit" / "audit_summary.json")
        result = verify_active(summary)
    else:
        summary = build_dry_run()
        result = apply_repair(summary) if args.apply else summary
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
