"""Complete event-level research over already materialized minute samples.

The minute strategy runner is intentionally expensive because it builds causal
states from raw bars.  This tool never rebuilds those states.  It scans the
existing outcome partitions with DuckDB, audits normalized artifacts, and
writes compact summaries for every strategy, MA period, horizon, year, month,
market regime, and matched control comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest, ttest_1samp

GIB = 1024**3
CONTROL_IDS = ("s0_random_matched", "s0_liquidity_matched")
HORIZONS = ("5m", "15m", "30m", "60m", "1d", "2d", "3d", "5d")
METRIC_PREFIXES = ("gross_return_", "net_return_")
FEATURE_FLAGS = (
    "recent_high_breakout",
    "prior_acceleration",
    "breakout_recent",
    "daily_trend_positive",
    "market_supportive",
    "sector_strong",
    "leader_sync",
    "auction_confirmed",
    "volume_normal",
    "not_repeated_cross",
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"not_jsonable:{type(value).__name__}")


def _csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _scan(paths: list[str]) -> str:
    return "read_parquet(?, union_by_name=true)"


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sample_roots(workspace_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return canonical sources and an inventory of every minute output root."""

    runs = workspace_root / "runs"
    all_roots = sorted(
        p for p in runs.iterdir() if p.is_dir() and p.name.startswith("minute_")
    ) if runs.is_dir() else []
    canonical_specs = (
        (
            "primary_development_2022_2023_partial",
            runs / "minute_ma_development_2022_2024",
            "primary current-contract partitions; interrupted after 2023-05-18",
        ),
        (
            "supplemental_2023_09_legacy",
            runs / "minute_ma_month_2023_09",
            "independent complete month; old VWAP revision is flagged separately",
        ),
    )
    canonical: list[dict[str, Any]] = []
    canonical_dates: set[str] = set()
    for group, root, reason in canonical_specs:
        files = sorted((root / "outcomes").glob("date=*/outcomes.parquet"))
        dates = {p.parent.name.split("=", 1)[-1] for p in files}
        overlap = sorted(canonical_dates.intersection(dates))
        canonical_dates.update(dates)
        canonical.append(
            {
                "sample_group": group,
                "root": str(root),
                "outcome_paths": [str(p) for p in files],
                "date_count": len(files),
                "date_start": min(dates) if dates else None,
                "date_end": max(dates) if dates else None,
                "overlap_with_previous_canonical": overlap,
                "selection_reason": reason,
            }
        )
    inventory: list[dict[str, Any]] = []
    canonical_names = {Path(item["root"]).name for item in canonical}
    for root in all_roots:
        files = sorted((root / "outcomes").glob("date=*/outcomes.parquet"))
        dates = {p.parent.name.split("=", 1)[-1] for p in files}
        if root.name in canonical_names:
            status = "included"
            reason = next(item["selection_reason"] for item in canonical if Path(item["root"]).name == root.name)
        elif root.name in {"minute_ma_month_2022_04", "minute_ma_month_2022_04_t4_c128_v2"}:
            status = "duplicate_audit_only"
            reason = "overlaps canonical 2022-04 dates; retained only for duplicate/parity audit"
        elif root.name.startswith("minute_strategy_portfolio"):
            status = "derived_account_artifact"
            reason = "portfolio result, not an independent event sample"
        else:
            status = "implementation_probe_or_failed_run"
            reason = "smoke, benchmark, probe, or failed partial output; not a research sample"
        inventory.append(
            {
                "root": str(root),
                "name": root.name,
                "status": status,
                "reason": reason,
                "outcome_file_count": len(files),
                "date_start": min(dates) if dates else None,
                "date_end": max(dates) if dates else None,
                "outcome_bytes": sum(p.stat().st_size for p in files),
            }
        )
    return {"canonical": canonical, "inventory": inventory}, canonical


def _source_cte(canonical: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    parts: list[str] = []
    args: list[Any] = []
    for item in canonical:
        paths = item["outcome_paths"]
        if not paths:
            continue
        parts.append(
            "SELECT ? AS sample_group, * FROM read_parquet(?, union_by_name=true)"
        )
        args.extend([item["sample_group"], paths])
    if not parts:
        raise RuntimeError("minute_sample_outcomes_empty")
    return " UNION ALL ".join(parts), args


def _normalised_source_cte(canonical: list[dict[str, Any]], kind: str) -> tuple[str, list[Any]]:
    parts: list[str] = []
    args: list[Any] = []
    for item in canonical:
        root = Path(item["root"])
        files = sorted((root / "normalized" / kind).glob("date=*/*.parquet"))
        if not files:
            continue
        parts.append(
            "SELECT ? AS sample_group, * FROM read_parquet(?, union_by_name=true)"
        )
        args.extend([item["sample_group"], [str(p) for p in files]])
    if not parts:
        return "", []
    return " UNION ALL ".join(parts), args


def _typed_source(cte: str) -> str:
    return f"""
    WITH raw AS ({cte}), source AS (
        SELECT
            CAST(sample_group AS VARCHAR) AS sample_group,
            CAST(strategy_id AS VARCHAR) AS strategy_id,
            TRY_CAST(ma_period AS INTEGER) AS ma_period,
            CAST(signal_id AS VARCHAR) AS signal_id,
            CAST(symbol AS VARCHAR) AS symbol,
            CAST(signal_date AS DATE) AS signal_date,
            CAST(signal_time AS VARCHAR) AS signal_time,
            TRY_CAST(sixty_minute_bucket AS INTEGER) AS sixty_minute_bucket,
            CAST(event_trigger AS VARCHAR) AS event_trigger,
            COALESCE(TRY_CAST(causal_only AS BOOLEAN), FALSE) AS causal_only,
            COALESCE(TRY_CAST(diagnostic_only AS BOOLEAN), FALSE) AS diagnostic_only,
            COALESCE(TRY_CAST(entry_observed AS BOOLEAN), FALSE) AS entry_observed,
            COALESCE(TRY_CAST(entry_executable AS BOOLEAN), FALSE) AS entry_executable,
            CAST(reference_signal_id AS VARCHAR) AS reference_signal_id,
            CAST(reference_symbol AS VARCHAR) AS reference_symbol,
            TRY_CAST(liquidity_match_ratio AS DOUBLE) AS liquidity_match_ratio,
            TRY_CAST(mfe_same_day AS DOUBLE) AS mfe_same_day,
            TRY_CAST(mae_same_day AS DOUBLE) AS mae_same_day,
            TRY_CAST(market_regime AS VARCHAR) AS market_regime,
            TRY_CAST(sector_strength_rank AS DOUBLE) AS sector_strength_rank,
            TRY_CAST(sector_breadth AS DOUBLE) AS sector_breadth,
            TRY_CAST(leader_relative_return AS DOUBLE) AS leader_relative_return,
            TRY_CAST(vwap_deviation AS DOUBLE) AS vwap_deviation,
            TRY_CAST(volume_acceleration_5_20 AS DOUBLE) AS volume_acceleration_5_20,
            TRY_CAST(amount_curve_surprise AS DOUBLE) AS amount_curve_surprise,
            {", ".join(f"COALESCE(TRY_CAST({_quote(name)} AS BOOLEAN), FALSE) AS {_quote(name)}" for name in FEATURE_FLAGS)},
            {", ".join(f"TRY_CAST({_quote(f'net_return_{h}') } AS DOUBLE) AS {_quote(f'net_return_{h}')}" for h in HORIZONS)},
            {", ".join(f"TRY_CAST({_quote(f'gross_return_{h}') } AS DOUBLE) AS {_quote(f'gross_return_{h}')}" for h in HORIZONS)}
        FROM raw
    )
    """


def _audit_files(canonical: list[dict[str, Any]], output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_schema: list[str] | None = None
    for item in canonical:
        root = Path(item["root"])
        normalized_events = {p.parent.name.split("=", 1)[-1]: p for p in (root / "normalized" / "events").glob("date=*/*.parquet")}
        normalized_paths = {p.parent.name.split("=", 1)[-1]: p for p in (root / "normalized" / "paths").glob("date=*/*.parquet")}
        normalized_refs = {p.parent.name.split("=", 1)[-1]: p for p in (root / "normalized" / "references").glob("date=*/*.parquet")}
        for path in item["outcome_paths"]:
            outcome = Path(path)
            date_text = outcome.parent.name.split("=", 1)[-1]
            schema = pq.ParquetFile(outcome).schema.names
            if expected_schema is None:
                expected_schema = schema
            schema_match = schema == expected_schema
            outcome_rows = pq.ParquetFile(outcome).metadata.num_rows
            event_path = normalized_events.get(date_text)
            path_path = normalized_paths.get(date_text)
            ref_path = normalized_refs.get(date_text)
            complete = event_path is not None and path_path is not None and ref_path is not None
            event_rows = pq.ParquetFile(event_path).metadata.num_rows if event_path else None
            path_rows = pq.ParquetFile(path_path).metadata.num_rows if path_path else None
            ref_rows = pq.ParquetFile(ref_path).metadata.num_rows if ref_path else None
            path_unique = None
            event_orphans = None
            ref_orphans = None
            if complete:
                con = duckdb.connect()
                try:
                    con.execute("PRAGMA memory_limit='512MB'")
                    con.execute("PRAGMA threads=1")
                    path_unique = int(con.execute("SELECT COUNT(*) = COUNT(DISTINCT path_id) FROM read_parquet(?)", [str(path_path)]).fetchone()[0])
                    event_orphans = int(con.execute("SELECT COUNT(*) FROM read_parquet(?) e LEFT JOIN read_parquet(?) p USING(path_id) WHERE p.path_id IS NULL", [str(event_path), str(path_path)]).fetchone()[0])
                    ref_orphans = int(con.execute("SELECT COUNT(*) FROM read_parquet(?) r LEFT JOIN read_parquet(?) e USING(path_id) WHERE e.path_id IS NULL", [str(ref_path), str(path_path)]).fetchone()[0])
                finally:
                    con.close()
            rows.append(
                {
                    "sample_group": item["sample_group"],
                    "trade_date": date_text,
                    "outcome_path": str(outcome),
                    "outcome_rows": outcome_rows,
                    "schema_match": schema_match,
                    "normalized_complete": complete,
                    "event_rows": event_rows,
                    "path_rows": path_rows,
                    "reference_rows": ref_rows,
                    "path_id_unique": path_unique,
                    "event_orphan_path_count": event_orphans,
                    "reference_orphan_path_count": ref_orphans,
                    "normalized_row_parity": bool(complete and event_rows == outcome_rows and ref_rows == outcome_rows),
                }
            )
    result = pd.DataFrame(rows).sort_values(["sample_group", "trade_date"], ignore_index=True)
    _csv_write(output / "partition_audit.csv", result)
    return result


def _event_summary(con: duckdb.DuckDBPyConnection, source_sql: str, args: list[Any], output: Path) -> pd.DataFrame:
    # Keep the grouping state for one metric at a time.  A single query with
    # all medians materializes too many per-group sketches for the 128M-row
    # development sample and can consume the machine reserve.
    base_query = f"""
    {source_sql}
    SELECT sample_group, strategy_id, ma_period,
           COUNT(*) AS signal_count,
           COUNT(DISTINCT signal_date) AS date_count,
           COUNT(DISTINCT symbol) AS symbol_count,
           COUNT(DISTINCT event_trigger) AS event_trigger_count,
           COUNT(*) FILTER (WHERE entry_observed) AS entry_observed_count,
           COUNT(*) FILTER (WHERE entry_executable) AS entry_executable_count
    FROM source
    WHERE NOT diagnostic_only AND causal_only
    GROUP BY ALL
    ORDER BY sample_group, strategy_id, ma_period
    """
    base = con.execute(base_query, args).fetchdf()
    rows: list[dict[str, Any]] = []
    metric_specs = [
        (f"{prefix}{horizon}", prefix[:-1], horizon)
        for prefix in METRIC_PREFIXES
        for horizon in HORIZONS
    ] + [(name, name, "same_day") for name in ("mfe_same_day", "mae_same_day")]
    for metric_name, metric_label, horizon in metric_specs:
        q = _quote(metric_name)
        query = f"""
        {source_sql}
        SELECT sample_group, strategy_id, ma_period,
               COUNT(*) FILTER (WHERE isfinite({q})) AS observed_count,
               AVG({q}) FILTER (WHERE isfinite({q})) AS mean,
               MEDIAN({q}) FILTER (WHERE isfinite({q})) AS median,
               STDDEV_POP({q}) FILTER (WHERE isfinite({q})) AS std,
               COUNT(*) FILTER (WHERE isfinite({q}) AND {q} > 0) AS positive_count,
               SUM({q}) FILTER (WHERE isfinite({q}) AND {q} > 0) AS positive_sum,
               SUM(-{q}) FILTER (WHERE isfinite({q}) AND {q} < 0) AS negative_abs
        FROM source
        WHERE NOT diagnostic_only AND causal_only
        GROUP BY ALL
        """
        metric_frame = con.execute(query, args).fetchdf()
        for record in metric_frame.to_dict("records"):
            observed = int(record.get("observed_count") or 0)
            positive = int(record.get("positive_count") or 0)
            positive_sum = record.get("positive_sum")
            negative_abs = record.get("negative_abs")
            rows.append(
                {
                    "sample_group": record.get("sample_group"),
                    "strategy_id": record.get("strategy_id"),
                    "ma_period": record.get("ma_period"),
                    "metric": metric_label,
                    "horizon": horizon,
                    "observed_count": observed,
                    "mean": record.get("mean"),
                    "median": record.get("median"),
                    "std": record.get("std"),
                    "win_rate": positive / observed if observed else None,
                    "profit_factor": (
                        float(positive_sum) / float(negative_abs)
                        if positive_sum is not None
                        and negative_abs is not None
                        and float(negative_abs) > 0
                        else None
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.merge(base, on=["sample_group", "strategy_id", "ma_period"], how="left", validate="many_to_one")
    _csv_write(output / "event_metrics.csv", result)
    return result


def _daily_summary(con: duckdb.DuckDBPyConnection, source_sql: str, args: list[Any], output: Path) -> pd.DataFrame:
    metrics = [f"AVG({_quote(f'net_return_{h}')}) AS {_quote(f'net_return_{h}')}" for h in HORIZONS]
    query = f"""
    {source_sql}
    SELECT sample_group, strategy_id, ma_period, signal_date,
           COUNT(*) AS signal_count,
           COUNT(*) FILTER (WHERE entry_executable) AS entry_executable_count,
           {", ".join(metrics)}
    FROM source
    WHERE NOT diagnostic_only AND causal_only
    GROUP BY ALL
    ORDER BY sample_group, strategy_id, ma_period, signal_date
    """
    result = con.execute(query, args).fetchdf()
    result["signal_date"] = result["signal_date"].astype(str)
    _csv_write(output / "daily_metrics.csv", result)
    return result


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        order = np.argsort(values[valid], kind="stable")
        ranked = values[valid][order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
        target = np.flatnonzero(valid)[order]
        result[target] = adjusted
    return pd.Series(result, index=pvalues.index)


def _daily_consistency(daily: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in daily.groupby(["sample_group", "strategy_id", "ma_period"], dropna=False):
        sample_group, strategy_id, ma_period = keys
        for horizon in HORIZONS:
            values = pd.to_numeric(group[f"net_return_{horizon}"], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            positive = int((values > 0).sum())
            rows.append(
                {
                    "sample_group": sample_group,
                    "strategy_id": strategy_id,
                    "ma_period": ma_period,
                    "horizon": horizon,
                    "date_count": len(values),
                    "mean_daily_return": float(values.mean()),
                    "median_daily_return": float(np.median(values)),
                    "std_daily_return": float(values.std(ddof=0)),
                    "positive_date_count": positive,
                    "positive_date_fraction": positive / len(values),
                    "ttest_p_value": float(ttest_1samp(values, 0.0).pvalue) if len(values) >= 2 else None,
                    "sign_test_p_value": float(binomtest(positive, len(values), 0.5, alternative="two-sided").pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["bh_q_value"] = np.nan
        for (_sample, _horizon), indices in result.groupby(["sample_group", "horizon"]).groups.items():
            result.loc[indices, "bh_q_value"] = _bh_adjust(result.loc[indices, "ttest_p_value"])
    _csv_write(output / "daily_consistency.csv", result)
    return result


def _grouped_return_summary(con: duckdb.DuckDBPyConnection, source_sql: str, args: list[Any], group_sql: str, output_name: str, output: Path) -> pd.DataFrame:
    # Keep one horizon's median sketch at a time.  Grouped summaries such as
    # month and regime still scan the same source, but never retain all eight
    # horizon sketches in DuckDB simultaneously.
    base_query = f"""
    {source_sql}
    SELECT {group_sql}, COUNT(*) AS signal_count
    FROM source
    WHERE NOT diagnostic_only AND causal_only
    GROUP BY ALL
    ORDER BY ALL
    """
    result = con.execute(base_query, args).fetchdf()
    key_columns = [name for name in result.columns if name != "signal_count"]
    for horizon in HORIZONS:
        q = _quote(f"net_return_{horizon}")
        query = f"""
        {source_sql}
        SELECT {group_sql},
               AVG({q}) FILTER (WHERE isfinite({q})) AS {_quote(f'net_return_{horizon}_mean')},
               MEDIAN({q}) FILTER (WHERE isfinite({q})) AS {_quote(f'net_return_{horizon}_median')},
               COUNT(*) FILTER (WHERE isfinite({q}) AND {q} > 0) * 1.0
                   / NULLIF(COUNT(*) FILTER (WHERE isfinite({q})), 0)
                   AS {_quote(f'net_return_{horizon}_win_rate')}
        FROM source
        WHERE NOT diagnostic_only AND causal_only
        GROUP BY ALL
        ORDER BY ALL
        """
        part = con.execute(query, args).fetchdf()
        result = result.merge(part, on=key_columns, how="left", validate="one_to_one")
    _csv_write(output / output_name, result)
    return result


def _feature_summary(con: duckdb.DuckDBPyConnection, source_sql: str, args: list[Any], output: Path) -> pd.DataFrame:
    expressions = [f"AVG(CASE WHEN {_quote(name)} THEN 1.0 ELSE 0.0 END) AS {_quote(name + '_true_fraction')}" for name in FEATURE_FLAGS]
    expressions += [f"AVG({_quote(name)}) AS {_quote(name + '_mean')}" for name in ("sector_strength_rank", "sector_breadth", "leader_relative_return", "vwap_deviation", "volume_acceleration_5_20", "amount_curve_surprise")]
    query = f"""
    {source_sql}
    SELECT sample_group, strategy_id, ma_period, {", ".join(expressions)}
    FROM source
    WHERE NOT diagnostic_only AND causal_only
    GROUP BY ALL ORDER BY ALL
    """
    result = con.execute(query, args).fetchdf()
    _csv_write(output / "signal_feature_summary.csv", result)
    return result


def _control_pairs_legacy(
    con: duckdb.DuckDBPyConnection,
    source_sql: str,
    args: list[Any],
    output: Path,
    *,
    write_outputs: bool = True,
    return_daily: bool = False,
) -> pd.DataFrame:
    differences = []
    for horizon in HORIZONS:
        q = _quote(f"net_return_{horizon}")
        differences.append(f"r.{q} - c.{q} AS {_quote('difference_' + horizon)}")
    query = f"""
    {source_sql}
    , regular AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY sample_group, strategy_id, symbol, signal_date, sixty_minute_bucket, ma_period
            ORDER BY signal_id
        ) AS rn
        FROM source
        WHERE NOT diagnostic_only AND causal_only
          AND strategy_id NOT IN ('s0_random_matched', 's0_liquidity_matched')
    ), random_control AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY sample_group, symbol, signal_date, sixty_minute_bucket, ma_period
            ORDER BY signal_id
        ) AS rn
        FROM source
        WHERE NOT diagnostic_only AND causal_only AND strategy_id = 's0_random_matched'
    ), liquidity_control AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY sample_group, reference_signal_id
            ORDER BY signal_id
        ) AS rn
        FROM source
        WHERE NOT diagnostic_only AND causal_only AND strategy_id = 's0_liquidity_matched'
          AND reference_signal_id IS NOT NULL
    ), pairs AS (
        SELECT r.sample_group, r.strategy_id, r.ma_period, r.signal_date,
               's0_random_matched' AS control_id, {", ".join(f"{d.replace('r.', 'r.').replace('c.', 'c.')}" for d in differences)}
        FROM regular r JOIN random_control c
          ON c.sample_group = r.sample_group AND c.symbol = r.symbol
         AND c.signal_date = r.signal_date AND c.sixty_minute_bucket = r.sixty_minute_bucket
         AND c.ma_period = r.ma_period
        WHERE r.rn = 1 AND c.rn = 1
        UNION ALL
        SELECT r.sample_group, r.strategy_id, r.ma_period, r.signal_date,
               's0_liquidity_matched' AS control_id, {", ".join(differences)}
        FROM regular r JOIN liquidity_control c
          ON c.sample_group = r.sample_group AND c.reference_signal_id = r.signal_id
        WHERE r.rn = 1 AND c.rn = 1
    )
    SELECT sample_group, strategy_id, ma_period, signal_date, control_id,
           COUNT(*) AS matched_count,
           {", ".join(f"AVG({_quote('difference_' + h)}) AS {_quote('difference_' + h)}" for h in HORIZONS)}
    FROM pairs
    GROUP BY ALL ORDER BY ALL
    """
    result = con.execute(query, args).fetchdf()
    result["signal_date"] = result["signal_date"].astype(str)
    if write_outputs:
        _csv_write(output / "control_pair_daily.csv", result)
    if return_daily:
        return result
    rows: list[dict[str, Any]] = []
    for keys, group in result.groupby(["sample_group", "strategy_id", "ma_period", "control_id"], dropna=False):
        sample, strategy, ma, control = keys
        for horizon in HORIZONS:
            values = pd.to_numeric(group[f"difference_{horizon}"], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            positive = int((values > 0).sum())
            rows.append(
                {
                    "sample_group": sample,
                    "strategy_id": strategy,
                    "ma_period": ma,
                    "control_id": control,
                    "horizon": horizon,
                    "matched_date_count": len(values),
                    "mean_daily_difference": float(values.mean()),
                    "median_daily_difference": float(np.median(values)),
                    "std_daily_difference": float(values.std(ddof=0)),
                    "positive_date_fraction": positive / len(values),
                    "ttest_p_value": float(ttest_1samp(values, 0.0).pvalue) if len(values) >= 2 else None,
                    "sign_test_p_value": float(binomtest(positive, len(values), 0.5, alternative="two-sided").pvalue),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["bh_q_value"] = np.nan
        for (_sample, _control, _horizon), indices in summary.groupby(["sample_group", "control_id", "horizon"]).groups.items():
            summary.loc[indices, "bh_q_value"] = _bh_adjust(summary.loc[indices, "ttest_p_value"])
    if write_outputs:
        _csv_write(output / "control_pair_summary.csv", summary)
    return summary


def _control_pairs(con: duckdb.DuckDBPyConnection, source_sql: str, args: list[Any], output: Path) -> pd.DataFrame:
    """Compute daily control differences with date-local bounded joins.

    The global outcome relation contains more than 128M rows.  Even a narrow
    window join over that relation can exceed the process reserve.  Each
    canonical outcome partition is an independent signal date, so joining one
    partition at a time preserves the pairing contract while bounding every
    intermediate relation to roughly one trading day's data.
    """

    output.mkdir(parents=True, exist_ok=True)
    daily_parts: list[pd.DataFrame] = []
    # ``_source_cte`` stores arguments as alternating sample-group and path
    # list values. Reconstruct those lists so each date can be queried in
    # isolation without changing the public helper contract.
    for index in range(0, len(args), 2):
        sample_group = str(args[index])
        paths = args[index + 1]
        for path_text in paths:
            path = Path(path_text)
            one = [
                {
                    "sample_group": sample_group,
                    "root": str(path.parents[2]),
                    "outcome_paths": [str(path)],
                }
            ]
            local_cte, local_args = _source_cte(one)
            local_sql = _typed_source(local_cte)
            daily_parts.append(
                _control_pairs_legacy(
                    con,
                    local_sql,
                    local_args,
                    output,
                    write_outputs=False,
                    return_daily=True,
                )
            )
    result = pd.concat(
        [part for part in daily_parts if not part.empty], ignore_index=True
    ) if daily_parts else pd.DataFrame()
    if result.empty:
        _csv_write(output / "control_pair_daily.csv", result)
        _csv_write(output / "control_pair_summary.csv", result)
        return result
    result["signal_date"] = result["signal_date"].astype(str)
    _csv_write(output / "control_pair_daily.csv", result)
    rows: list[dict[str, Any]] = []
    for keys_value, group in result.groupby(
        ["sample_group", "strategy_id", "ma_period", "control_id"],
        dropna=False,
    ):
        sample, strategy, ma, control = keys_value
        for horizon in HORIZONS:
            values = pd.to_numeric(
                group[f"difference_{horizon}"], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            positive = int((values > 0).sum())
            rows.append(
                {
                    "sample_group": sample,
                    "strategy_id": strategy,
                    "ma_period": ma,
                    "control_id": control,
                    "horizon": horizon,
                    "matched_date_count": len(values),
                    "mean_daily_difference": float(values.mean()),
                    "median_daily_difference": float(np.median(values)),
                    "std_daily_difference": float(values.std(ddof=0)),
                    "positive_date_fraction": positive / len(values),
                    "ttest_p_value": float(ttest_1samp(values, 0.0).pvalue)
                    if len(values) >= 2
                    else None,
                    "sign_test_p_value": float(
                        binomtest(
                            positive,
                            len(values),
                            0.5,
                            alternative="two-sided",
                        ).pvalue
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["bh_q_value"] = np.nan
        for (_sample, _control, _horizon), indices in summary.groupby(
            ["sample_group", "control_id", "horizon"]
        ).groups.items():
            summary.loc[indices, "bh_q_value"] = _bh_adjust(
                summary.loc[indices, "ttest_p_value"]
            )
    _csv_write(output / "control_pair_summary.csv", summary)
    return summary

def _candidate_screen(metrics: pd.DataFrame, daily: pd.DataFrame, controls: pd.DataFrame, output: Path) -> pd.DataFrame:
    if metrics.empty:
        result = pd.DataFrame()
        _csv_write(output / "candidate_screen.csv", result)
        return result
    selected = metrics.loc[
        metrics["metric"].eq("net_return") & metrics["horizon"].isin(["60m", "1d", "2d", "5d"])
    ].copy()
    pivot = selected.pivot_table(
        index=["sample_group", "strategy_id", "ma_period"],
        columns="horizon",
        values=["signal_count", "mean", "median", "win_rate", "profit_factor"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    consistency = _daily_consistency_frame(daily)
    if not consistency.empty:
        c = consistency.loc[consistency["horizon"].isin(["60m", "1d", "2d", "5d"]), ["sample_group", "strategy_id", "ma_period", "horizon", "positive_date_fraction", "mean_daily_return", "bh_q_value"]]
        c = c.pivot_table(index=["sample_group", "strategy_id", "ma_period"], columns="horizon", values=["positive_date_fraction", "mean_daily_return", "bh_q_value"], aggfunc="first")
        c.columns = [f"daily_{a}_{b}" for a, b in c.columns]
        pivot = pivot.merge(c.reset_index(), on=["sample_group", "strategy_id", "ma_period"], how="left")
    if not controls.empty:
        c = controls.loc[controls["horizon"].isin(["60m", "1d", "2d", "5d"]), ["sample_group", "strategy_id", "ma_period", "control_id", "horizon", "mean_daily_difference", "positive_date_fraction"]]
        c = c.pivot_table(index=["sample_group", "strategy_id", "ma_period"], columns=["control_id", "horizon"], values=["mean_daily_difference", "positive_date_fraction"], aggfunc="first")
        c.columns = ["control_" + "_".join(str(x) for x in col) for col in c.columns]
        pivot = pivot.merge(c.reset_index(), on=["sample_group", "strategy_id", "ma_period"], how="left")
    pivot["screen_note"] = "descriptive screen only; no winner is selected"
    _csv_write(output / "candidate_screen.csv", pivot)
    return pivot


def _daily_consistency_frame(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in daily.groupby(["sample_group", "strategy_id", "ma_period"], dropna=False):
        for horizon in HORIZONS:
            values = pd.to_numeric(group[f"net_return_{horizon}"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                positive = int((values > 0).sum())
                rows.append({"sample_group": keys[0], "strategy_id": keys[1], "ma_period": keys[2], "horizon": horizon, "positive_date_fraction": positive / len(values), "mean_daily_return": float(values.mean()), "bh_q_value": None})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["ttest_p_value"] = [
            float(ttest_1samp(pd.to_numeric(daily.loc[(daily.sample_group == r.sample_group) & (daily.strategy_id == r.strategy_id) & (daily.ma_period == r.ma_period), f"net_return_{r.horizon}"].dropna(), errors="coerce"), 0.0).pvalue)
            if len(daily.loc[(daily.sample_group == r.sample_group) & (daily.strategy_id == r.strategy_id) & (daily.ma_period == r.ma_period), f"net_return_{r.horizon}"].dropna()) >= 2 else np.nan
            for r in result.itertuples()
        ]
        for (_sample, _horizon), indices in result.groupby(["sample_group", "horizon"]).groups.items():
            result.loc[indices, "bh_q_value"] = _bh_adjust(result.loc[indices, "ttest_p_value"])
    return result


def _account_record_inventory(workspace_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in (
        "minute_ma_portfolio_development_v1",
        "minute_ma_portfolio_seed_stability_v1",
        "minute_strategy_portfolio_causal_rankers_v1",
    ):
        path = workspace_root / "research" / "records" / name / "result.json"
        if path.exists():
            records.append({"record": name, **json.loads(path.read_text(encoding="utf-8"))})
    return records


def run(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory, canonical = _sample_roots(workspace_root)
    _json_dump(output_root / "sample_inventory.json", inventory)
    partition_audit = _audit_files(canonical, output_root)
    cte, source_args = _source_cte(canonical)
    temp_directory = workspace_root / "tmp" / "minute_existing_samples_analysis_duckdb"
    temp_directory.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='512MB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    temp_sql = str(temp_directory).replace("'", "''")
    con.execute(f"PRAGMA temp_directory='{temp_sql}'")
    try:
        source_sql = _typed_source(cte)
        event_metrics = _event_summary(con, source_sql, source_args, output_root)
        daily = _daily_summary(con, source_sql, source_args, output_root)
        consistency = _daily_consistency(daily, output_root)
        _grouped_return_summary(
            con,
            source_sql,
            source_args,
            "sample_group, strategy_id, ma_period, event_trigger",
            "event_trigger_summary.csv",
            output_root,
        )
        _grouped_return_summary(
            con,
            source_sql,
            source_args,
            "sample_group, CAST(EXTRACT(year FROM signal_date) AS INTEGER) AS signal_year, strategy_id, ma_period",
            "year_summary.csv",
            output_root,
        )
        _grouped_return_summary(
            con,
            source_sql,
            source_args,
            "sample_group, STRFTIME(signal_date, '%Y-%m') AS signal_month, strategy_id, ma_period",
            "month_summary.csv",
            output_root,
        )
        _grouped_return_summary(
            con,
            source_sql,
            source_args,
            "sample_group, COALESCE(market_regime, 'missing') AS market_regime, strategy_id, ma_period",
            "market_regime_summary.csv",
            output_root,
        )
        _feature_summary(con, source_sql, source_args, output_root)
        controls = _control_pairs(con, source_sql, source_args, output_root)
    finally:
        con.close()
    candidate = _candidate_screen(event_metrics, daily, controls, output_root)
    account_records = _account_record_inventory(workspace_root)
    audit_payload = {
        "partition_count": int(len(partition_audit)),
        "schema_mismatch_count": int((~partition_audit["schema_match"]).sum()) if not partition_audit.empty else 0,
        "normalized_incomplete_count": int((~partition_audit["normalized_complete"]).sum()) if not partition_audit.empty else 0,
        "normalized_row_parity_failure_count": int((~partition_audit["normalized_row_parity"]).sum()) if not partition_audit.empty else 0,
        "event_orphan_path_count": int(partition_audit["event_orphan_path_count"].fillna(0).sum()) if not partition_audit.empty else 0,
        "reference_orphan_path_count": int(partition_audit["reference_orphan_path_count"].fillna(0).sum()) if not partition_audit.empty else 0,
    }
    result = {
        "schema": "quantlab.minute_strategy_existing_samples_research/1",
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_contract": {
            "canonical_sample_groups": [item["sample_group"] for item in canonical],
            "excluded_duplicate_and_probe_roots": [item for item in inventory["inventory"] if item["status"] != "included"],
            "diagnostic_rows_excluded": True,
            "causal_only": True,
            "horizons": list(HORIZONS),
            "metrics": ["gross", "net", "mfe_same_day", "mae_same_day"],
            "daily_inference_unit": "signal_date mean, not independent signal rows",
            "multiple_comparison_adjustment": "Benjamini-Hochberg within sample_group and horizon",
        },
        "partition_audit": audit_payload,
        "event_metrics_rows": int(len(event_metrics)),
        "daily_metrics_rows": int(len(daily)),
        "daily_consistency_rows": int(len(consistency)),
        "control_pair_summary_rows": int(len(controls)),
        "candidate_screen_rows": int(len(candidate)),
        "account_records_already_available": [
            {
                "record": item["record"],
                "status": item.get("status"),
                "signal_date_count": item.get("signal_date_count"),
                "account_result_count": item.get("account_result_count"),
            }
            for item in account_records
        ],
        "account_replay_note": "Existing account records are retained as derived evidence; this event report does not silently treat them as a full replay of every canonical date.",
    }
    _json_dump(output_root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-root", default="research/records/minute_strategy_existing_samples_v1")
    args = parser.parse_args()
    result = run(Path(args.workspace_root).resolve(), Path(args.output_root).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
