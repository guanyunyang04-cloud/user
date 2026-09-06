"""Study existing minute samples after a causal main-board limit-up streak filter.

The minute event files are already restricted to the point-in-time main board.
This tool adds one orthogonal, causal label: the number of *completed* daily
main-board limit-up closes immediately before a minute signal.  It never
rebuilds minute events.  Daily limit prices are reconstructed from the local
PIT OHLCV and ST-status contracts, with the rounding rule recorded in the
output manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil

GIB = 1024**3
HORIZONS = ("5m", "15m", "30m", "60m", "1d", "2d", "3d", "5d")
THRESHOLDS = (
    ("all", "TRUE"),
    ("ge1", "prior_limit_up_streak >= 1"),
    ("ge2", "prior_limit_up_streak >= 2"),
    ("ge3", "prior_limit_up_streak >= 3"),
    ("ge4", "prior_limit_up_streak >= 4"),
    ("ge5", "prior_limit_up_streak >= 5"),
)
CONTROL_IDS = ("s0_random_matched", "s0_liquidity_matched")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _memory_floor(floor_gib: float = 0.5) -> None:
    available = float(psutil.virtual_memory().available)
    if available < float(floor_gib) * GIB:
        raise RuntimeError(
            f"mainboard_consecutive_memory_floor_breached:available={available / GIB:.3f}:floor={floor_gib}"
        )


def _scan(paths: list[Path] | tuple[Path, ...]) -> str:
    return "read_parquet(?, union_by_name=true)"


def _sample_specs(workspace: Path) -> list[dict[str, Any]]:
    runs = workspace / "runs"
    specs = (
        (
            "primary_development_2022_2023_partial",
            runs / "minute_ma_development_2022_2024",
            "current_contract",
        ),
        (
            "supplemental_2023_09_legacy",
            runs / "minute_ma_month_2023_09",
            "legacy_vwap",
        ),
    )
    result: list[dict[str, Any]] = []
    for sample_group, root, variant in specs:
        files = sorted((root / "outcomes").glob("date=*/outcomes.parquet"))
        dates = sorted(path.parent.name.split("=", 1)[-1] for path in files)
        if not files:
            raise FileNotFoundError(f"mainboard_consecutive_sample_empty:{root}")
        result.append(
            {
                "sample_group": sample_group,
                "root": root,
                "variant": variant,
                "files": files,
                "dates": dates,
            }
        )
    return result


def _active_paths(workspace: Path, domain: str) -> list[Path]:
    # Import lazily so this standalone research utility remains easy to inspect.
    sys.path.insert(0, str(workspace / "src"))
    from quantlab.data.qdp_v2.active import resolve_active_domain

    return [Path(path) for path in resolve_active_domain(domain, workspace_root=workspace).shard_paths]


def _create_daily_streak(
    con: duckdb.DuckDBPyConnection,
    *,
    daily_paths: list[Path],
    status_paths: list[Path],
    start_date: str,
    end_date: str,
) -> None:
    """Build a compact causal daily streak relation in DuckDB."""

    daily_scan = _scan(daily_paths)
    status_scan = _scan(status_paths)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE daily_limit_streak AS
        WITH daily AS (
            SELECT CAST(symbol AS VARCHAR) AS symbol,
                   CAST(trade_date AS DATE) AS trade_date,
                   TRY_CAST(high AS DOUBLE) AS high_price,
                   TRY_CAST(\"close\" AS DOUBLE) AS close_price
            FROM {daily_scan}
            WHERE trade_date BETWEEN ? AND ?
        ), status AS (
            SELECT CAST(symbol AS VARCHAR) AS symbol,
                   CAST(trade_date AS DATE) AS trade_date,
                   BOOL_OR(COALESCE(is_st, FALSE)) AS is_st
            FROM {status_scan}
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY 1, 2
        ), ordered AS (
            SELECT d.*, COALESCE(s.is_st, FALSE) AS is_st,
                   LAG(close_price) OVER (
                       PARTITION BY d.symbol ORDER BY d.trade_date
                   ) AS previous_close,
                   LAG(trade_date) OVER (
                       PARTITION BY d.symbol ORDER BY d.trade_date
                   ) AS previous_trade_date
            FROM daily d
            LEFT JOIN status s USING (symbol, trade_date)
        ), marked AS (
            SELECT *,
                   CASE WHEN is_st THEN 0.05 ELSE 0.10 END AS limit_pct,
                   ROUND(
                       previous_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END),
                       2
                   ) AS theoretical_up_limit,
                   CASE
                       WHEN previous_close > 0
                        AND ABS(
                            close_price - ROUND(
                                previous_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END),
                                2
                            )
                        ) <= 0.0051
                       THEN TRUE ELSE FALSE
                   END AS is_limit_up_close,
                   CASE
                       WHEN previous_close > 0
                        AND high_price >= ROUND(
                            previous_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END),
                            2
                        ) - 0.0051
                       THEN TRUE ELSE FALSE
                   END AS touched_limit_up
            FROM ordered
        ), groups AS (
            SELECT *,
                   SUM(CASE WHEN is_limit_up_close THEN 0 ELSE 1 END) OVER (
                       PARTITION BY symbol ORDER BY trade_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS streak_group
            FROM marked
        ), streaks AS (
            SELECT *,
                   CASE WHEN is_limit_up_close THEN ROW_NUMBER() OVER (
                       PARTITION BY symbol, streak_group ORDER BY trade_date
                   ) ELSE 0 END AS limit_up_streak
            FROM groups
        )
        SELECT symbol, trade_date, is_st, previous_close, theoretical_up_limit,
               is_limit_up_close, touched_limit_up,
               CAST(limit_up_streak AS BIGINT) AS limit_up_streak,
               CAST(LAG(limit_up_streak) OVER (
                   PARTITION BY symbol ORDER BY trade_date
               ) AS BIGINT) AS prior_limit_up_streak,
               COALESCE(LAG(touched_limit_up) OVER (
                   PARTITION BY symbol ORDER BY trade_date
               ), FALSE) AS prior_day_touched_limit_up,
               previous_trade_date
        FROM streaks
        """,
        [
            list(map(str, daily_paths)),
            str(start_date),
            str(end_date),
            list(map(str, status_paths)),
            str(start_date),
            str(end_date),
        ],
    )


def _typed_outcome_cte(files: list[Path]) -> tuple[str, list[Any]]:
    # Keep only columns needed by this study.  This materially lowers the
    # temporary relation compared with selecting the complete wide outcome.
    columns = [
        "signal_id",
        "strategy_id",
        "symbol",
        "signal_date",
        "signal_time",
        "sixty_minute_bucket",
        "ma_period",
        "causal_only",
        "diagnostic_only",
        "reference_signal_id",
        *[f"net_return_{h}" for h in HORIZONS],
        *[f"gross_return_{h}" for h in HORIZONS],
    ]
    projection = ", ".join(
        [
            f"CAST({name} AS VARCHAR) AS {name}"
            if name in {"signal_id", "strategy_id", "symbol", "signal_date", "signal_time", "reference_signal_id"}
            else f"TRY_CAST({name} AS DOUBLE) AS {name}"
            for name in columns
            if name not in {"causal_only", "diagnostic_only"}
        ]
    )
    projection = (
        projection
        + ", COALESCE(TRY_CAST(causal_only AS BOOLEAN), FALSE) AS causal_only"
        + ", COALESCE(TRY_CAST(diagnostic_only AS BOOLEAN), FALSE) AS diagnostic_only"
    )
    return f"SELECT {projection} FROM {_scan(files)}", [list(map(str, files))]


def _label_join_cte(outcome_cte: str) -> str:
    return f"""
    WITH outcomes AS ({outcome_cte}), labeled AS (
        SELECT o.*,
               COALESCE(d.prior_limit_up_streak, 0) AS prior_limit_up_streak,
               COALESCE(d.prior_day_touched_limit_up, FALSE) AS prior_day_touched_limit_up,
               (d.symbol IS NOT NULL) AS streak_label_observed
        FROM outcomes o
        LEFT JOIN daily_limit_streak d
          ON d.symbol = o.symbol AND d.trade_date = TRY_CAST(o.signal_date AS DATE)
        WHERE o.causal_only AND NOT o.diagnostic_only
    )
    """


def _metric_query(outcome_cte: str, sample_group: str) -> tuple[str, list[Any]]:
    fields: list[str] = []
    for threshold_name, condition in THRESHOLDS:
        for horizon in HORIZONS:
            value = f"net_return_{horizon}"
            guard = f"({condition}) AND isfinite({value})"
            prefix = f"{threshold_name}_{horizon}"
            fields.extend(
                [
                    f"COUNT(*) FILTER (WHERE {guard}) AS {prefix}_observed_count",
                    f"AVG({value}) FILTER (WHERE {guard}) AS {prefix}_mean",
                    f"MEDIAN({value}) FILTER (WHERE {guard}) AS {prefix}_median",
                    f"SUM({value}) FILTER (WHERE {guard} AND {value} > 0) AS {prefix}_positive_sum",
                    f"SUM({value}) FILTER (WHERE {guard} AND {value} < 0) AS {prefix}_negative_sum",
                    f"COUNT(*) FILTER (WHERE {guard} AND {value} > 0) AS {prefix}_positive_count",
                    f"COUNT(DISTINCT signal_date) FILTER (WHERE {guard}) AS {prefix}_date_count",
                    f"COUNT(DISTINCT symbol) FILTER (WHERE {guard}) AS {prefix}_symbol_count",
                ]
            )
    query = (
        _label_join_cte(outcome_cte)
        + " SELECT "
        + repr(sample_group)
        + " AS sample_group, strategy_id, TRY_CAST(ma_period AS INTEGER) AS ma_period, "
        + ", ".join(fields)
        + " FROM labeled GROUP BY ALL ORDER BY ALL"
    )
    return query, []


def _melt_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    id_columns = ["sample_group", "strategy_id", "ma_period"]
    for row in frame.to_dict("records"):
        for threshold_name, _ in THRESHOLDS:
            for horizon in HORIZONS:
                prefix = f"{threshold_name}_{horizon}"
                observed = int(row.get(f"{prefix}_observed_count") or 0)
                positive_sum = float(row.get(f"{prefix}_positive_sum") or 0.0)
                negative_sum = float(row.get(f"{prefix}_negative_sum") or 0.0)
                rows.append(
                    {
                        **{key: row.get(key) for key in id_columns},
                        "threshold": threshold_name,
                        "horizon": horizon,
                        "observed_count": observed,
                        "mean": row.get(f"{prefix}_mean"),
                        "median": row.get(f"{prefix}_median"),
                        "win_rate": (
                            float(row.get(f"{prefix}_positive_count") or 0) / observed
                            if observed
                            else None
                        ),
                        "profit_factor": (
                            positive_sum / abs(negative_sum)
                            if negative_sum < 0
                            else (math.inf if positive_sum > 0 else None)
                        ),
                        "date_count": int(row.get(f"{prefix}_date_count") or 0),
                        "symbol_count": int(row.get(f"{prefix}_symbol_count") or 0),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["profit_factor"] = result["profit_factor"].replace([np.inf, -np.inf], np.nan)
    return result


def _daily_query(outcome_cte: str, sample_group: str) -> tuple[str, list[Any]]:
    fields: list[str] = []
    for threshold_name, condition in THRESHOLDS:
        for horizon in ("60m", "1d", "5d"):
            value = f"net_return_{horizon}"
            prefix = f"{threshold_name}_{horizon}"
            fields.extend(
                [
                    f"AVG({value}) FILTER (WHERE ({condition}) AND isfinite({value})) AS {prefix}_mean",
                    f"COUNT(*) FILTER (WHERE ({condition}) AND isfinite({value})) AS {prefix}_count",
                ]
            )
    query = (
        _label_join_cte(outcome_cte)
        + " SELECT "
        + repr(sample_group)
        + " AS sample_group, strategy_id, TRY_CAST(ma_period AS INTEGER) AS ma_period, "
        + "CAST(signal_date AS DATE) AS signal_date, "
        + ", ".join(fields)
        + " FROM labeled GROUP BY ALL"
    )
    return query, []


def _summarize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sample, strategy, ma), group in frame.groupby(
        ["sample_group", "strategy_id", "ma_period"], sort=True
    ):
        for threshold_name, _ in THRESHOLDS:
            for horizon in ("60m", "1d", "5d"):
                mean_col = f"{threshold_name}_{horizon}_mean"
                count_col = f"{threshold_name}_{horizon}_count"
                values = pd.to_numeric(group[mean_col], errors="coerce")
                counts = pd.to_numeric(group[count_col], errors="coerce").fillna(0)
                observed = values.notna() & counts.gt(0)
                values = values.loc[observed]
                rows.append(
                    {
                        "sample_group": sample,
                        "strategy_id": strategy,
                        "ma_period": ma,
                        "threshold": threshold_name,
                        "horizon": horizon,
                        "date_count": int(len(values)),
                        "mean_daily_return": float(values.mean()) if len(values) else None,
                        "median_daily_return": float(values.median()) if len(values) else None,
                        "positive_date_fraction": float((values > 0).mean()) if len(values) else None,
                    }
                )
    return pd.DataFrame(rows)


def _control_query(outcome_cte: str, sample_group: str) -> tuple[str, list[Any]]:
    fields: list[str] = []
    for threshold_name, condition in THRESHOLDS:
        # The same threshold is applied to both sides of a matched pair.  This
        # preserves the control comparison instead of conditioning only one leg.
        pair_condition = condition.replace("prior_limit_up_streak", "target_streak")
        pair_condition_control = condition.replace("prior_limit_up_streak", "control_streak")
        guard = f"({pair_condition}) AND ({pair_condition_control})"
        for horizon in HORIZONS:
            diff = f"target_{horizon} - control_{horizon}"
            prefix = f"{threshold_name}_{horizon}"
            fields.extend(
                [
                    f"COUNT(*) FILTER (WHERE {guard} AND isfinite({diff})) AS {prefix}_matched_count",
                    f"AVG({diff}) FILTER (WHERE {guard} AND isfinite({diff})) AS {prefix}_mean_difference",
                    f"MEDIAN({diff}) FILTER (WHERE {guard} AND isfinite({diff})) AS {prefix}_median_difference",
                ]
            )
    # ``source`` is intentionally selected once; the window de-duplication
    # mirrors the existing control contract.
    query = (
        _label_join_cte(outcome_cte)
        + ", regular AS ("
        + " SELECT *, ROW_NUMBER() OVER (PARTITION BY strategy_id, symbol, signal_date, sixty_minute_bucket, ma_period ORDER BY signal_id) AS rn"
        + " FROM labeled WHERE strategy_id NOT IN ('s0_random_matched','s0_liquidity_matched')"
        + " ), random_control AS ("
        + " SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, signal_date, sixty_minute_bucket, ma_period ORDER BY signal_id) AS rn"
        + " FROM labeled WHERE strategy_id = 's0_random_matched'"
        + " ), liquidity_control AS ("
        + " SELECT *, ROW_NUMBER() OVER (PARTITION BY reference_signal_id ORDER BY signal_id) AS rn"
        + " FROM labeled WHERE strategy_id = 's0_liquidity_matched' AND reference_signal_id IS NOT NULL"
        + " ), pairs AS ("
        + " SELECT r.strategy_id, r.ma_period, 's0_random_matched' AS control_id,"
        + " r.prior_limit_up_streak AS target_streak, c.prior_limit_up_streak AS control_streak, "
        + ", ".join(
            [
                f"r.net_return_{horizon} AS target_{horizon}, c.net_return_{horizon} AS control_{horizon}"
                for horizon in HORIZONS
            ]
        )
        + " FROM regular r JOIN random_control c ON c.symbol=r.symbol AND c.signal_date=r.signal_date"
        + " AND c.sixty_minute_bucket=r.sixty_minute_bucket AND c.ma_period=r.ma_period"
        + " WHERE r.rn=1 AND c.rn=1"
        + " UNION ALL SELECT r.strategy_id, r.ma_period, 's0_liquidity_matched' AS control_id,"
        + " r.prior_limit_up_streak AS target_streak, c.prior_limit_up_streak AS control_streak, "
        + ", ".join(
            [
                f"r.net_return_{horizon} AS target_{horizon}, c.net_return_{horizon} AS control_{horizon}"
                for horizon in HORIZONS
            ]
        )
        + " FROM regular r JOIN liquidity_control c ON c.reference_signal_id=r.signal_id"
        + " WHERE r.rn=1 AND c.rn=1"
        + " ) SELECT "
        + repr(sample_group)
        + " AS sample_group, strategy_id, ma_period, control_id, "
        + ", ".join(fields)
        + " FROM pairs GROUP BY ALL ORDER BY ALL"
    )
    return query, []


def _melt_controls(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        for threshold_name, _ in THRESHOLDS:
            for horizon in HORIZONS:
                prefix = f"{threshold_name}_{horizon}"
                rows.append(
                    {
                        "sample_group": row.get("sample_group"),
                        "strategy_id": row.get("strategy_id"),
                        "ma_period": row.get("ma_period"),
                        "control_id": row.get("control_id"),
                        "threshold": threshold_name,
                        "horizon": horizon,
                        "matched_count": int(row.get(f"{prefix}_matched_count") or 0),
                        "mean_difference": row.get(f"{prefix}_mean_difference"),
                        "median_difference": row.get(f"{prefix}_median_difference"),
                    }
                )
    return pd.DataFrame(rows)


def _board_audit(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    universe_paths: list[Path],
    sample_group: str,
) -> pd.DataFrame:
    # The active universe is already PIT main-board; this query verifies that
    # the actual symbols/dates in each sample resolve to board=main.
    query = f"""
    WITH keys AS (
        SELECT DISTINCT CAST(symbol AS VARCHAR) AS symbol, CAST(signal_date AS DATE) AS trade_date
        FROM {_scan(files)}
    ), universe AS (
        SELECT CAST(symbol AS VARCHAR) AS symbol, CAST(trade_date AS DATE) AS trade_date,
               ANY_VALUE(CAST(board AS VARCHAR)) AS board
        FROM {_scan(universe_paths)}
        WHERE trade_date BETWEEN ? AND ?
        GROUP BY 1, 2
    )
    SELECT {repr(sample_group)} AS sample_group,
           COUNT(*) AS sample_symbol_date_count,
           COUNT(*) FILTER (WHERE u.board = 'main') AS main_count,
           COUNT(*) FILTER (WHERE u.board IS NULL) AS unresolved_count,
           COUNT(*) FILTER (WHERE u.board IS NOT NULL AND u.board <> 'main') AS non_main_count
    FROM keys k LEFT JOIN universe u USING(symbol, trade_date)
    """
    dates = sorted(path.parent.name.split("=", 1)[-1] for path in (files[0].parent.parent).glob("date=*/*.parquet"))
    return con.execute(query, [list(map(str, files)), list(map(str, universe_paths)), dates[0], dates[-1]]).fetchdf()


def _write_labels(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    output: Path,
    sample_group: str,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    query = f"""
    WITH keys AS (
        SELECT DISTINCT CAST(symbol AS VARCHAR) AS symbol, CAST(signal_date AS DATE) AS signal_date
        FROM {_scan(files)}
    )
    SELECT {repr(sample_group)} AS sample_group, k.symbol, k.signal_date,
           COALESCE(d.prior_limit_up_streak, 0) AS prior_limit_up_streak,
           COALESCE(d.prior_day_touched_limit_up, FALSE) AS prior_day_touched_limit_up,
           (d.symbol IS NOT NULL) AS streak_label_observed
    FROM keys k LEFT JOIN daily_limit_streak d
      ON d.symbol=k.symbol AND d.trade_date=k.signal_date
    ORDER BY k.signal_date, k.symbol
    """
    # DuckDB writes the result directly, avoiding a pandas copy of the label table.
    con.execute(
        f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [list(map(str, files)), str(output)],
    )
    return int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(output)]).fetchone()[0])


def _write_report(output: Path, manifest: dict[str, Any], metrics: pd.DataFrame, daily: pd.DataFrame, controls: pd.DataFrame) -> None:
    lines = [
        "# Main-board consecutive limit-up filter study",
        "",
        "This is a causal re-slicing of already generated minute outcomes. It does not regenerate minute events.",
        "",
        "## Definition",
        "",
        "The sample universe is already point-in-time Shanghai/Shenzhen main board. For each signal date, `prior_limit_up_streak` is the number of completed prior trading days whose unadjusted daily close equals the reconstructed main-board/ST upper limit. The signal-day bar is never used for the primary label.",
        "",
        "The local daily contract has OHLCV but no historical limit-status table. The label therefore uses current-day PIT ST status, 10% for ordinary main-board shares and 5% for ST, a two-decimal theoretical limit price, and half-tick tolerance. A touched-but-not-closed label is retained only as a sensitivity field.",
        "",
        "## Coverage",
        "",
        f"- Daily label rows: {manifest.get('label_row_count_total', 0):,}",
        f"- Main-board audit non-main rows: {manifest.get('board_non_main_count_total', 0):,}; unresolved rows: {manifest.get('board_unresolved_count_total', 0):,}",
        "- `all` is the unfiltered baseline. `ge1` through `ge5` require at least that many completed prior limit-up closes.",
        "",
        "## Interpretation",
        "",
        "A filter can improve the conditional event mean simply by selecting a much narrower, high-volatility state. It is not evidence of a tradable strategy unless the event count, day-level stability, matched-control difference, costs, and account replay all survive a new sample.",
        "",
        "See `event_metrics.csv`, `daily_metrics.csv`, `control_summary.csv`, `streak_distribution.csv`, `board_audit.csv`, and the machine-readable manifest for the complete slice.",
    ]
    if not metrics.empty:
        focus = metrics.loc[
            metrics["horizon"].isin(["60m", "1d", "5d"])
            & metrics["strategy_id"].isin(["fu_r3_core_pullback", "ng_r1_auction_reclaim", "s1_touch_reclaim", "s2_reclaim_bull_stack"])
            & metrics["threshold"].isin(["all", "ge1", "ge2", "ge3"])
        ].sort_values(["sample_group", "strategy_id", "ma_period", "threshold", "horizon"])
        if not focus.empty:
            lines += ["", "## Selected rows (descriptive)", "", "| Sample | Strategy | MA | Filter | Horizon | N | Mean | Median | Win rate |", "|---|---|---:|---|---|---:|---:|---:|---:|"]
            for _, row in focus.iterrows():
                mean = pd.to_numeric(pd.Series([row.get("mean")]), errors="coerce").iloc[0]
                median = pd.to_numeric(pd.Series([row.get("median")]), errors="coerce").iloc[0]
                lines.append(
                    f"| {row['sample_group']} | {row['strategy_id']} | {int(row['ma_period'])} | {row['threshold']} | {row['horizon']} | {int(row['observed_count']):,} | "
                    f"{mean:.4%} | {median:.4%} | {float(row['win_rate']):.2%} |"
                )
    (output / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    _memory_floor()
    specs = _sample_specs(workspace_root)
    all_signal_dates = [date for spec in specs for date in spec["dates"]]
    min_date = min(all_signal_dates)
    max_date = max(all_signal_dates)
    history_start = (pd.Timestamp(min_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    daily_paths = _active_paths(workspace_root, "market_daily_raw")
    status_paths = _active_paths(workspace_root, "security_status")
    universe_paths = _active_paths(workspace_root, "universe_snapshot")
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir = workspace_root / "tmp" / "mainboard_consecutive_filter_duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='512MB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{str(temp_dir).replace(chr(39), chr(39) * 2)}'")
    metrics_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
    board_parts: list[pd.DataFrame] = []
    label_count = 0
    try:
        _create_daily_streak(
            con,
            daily_paths=daily_paths,
            status_paths=status_paths,
            start_date=history_start,
            end_date=max_date,
        )
        streak_distribution = con.execute(
            """
            SELECT prior_limit_up_streak, COUNT(*) AS row_count,
                   COUNT(DISTINCT symbol) AS symbol_count,
                   MIN(trade_date) AS first_date, MAX(trade_date) AS last_date
            FROM daily_limit_streak
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY prior_limit_up_streak ORDER BY prior_limit_up_streak
            """,
            [min_date, max_date],
        ).fetchdf()
        for spec in specs:
            _memory_floor()
            files = spec["files"]
            outcome_cte, args = _typed_outcome_cte(files)
            metric_query, _ = _metric_query(outcome_cte, spec["sample_group"])
            # The outcome path is the only bound argument in the CTE.
            metrics_parts.append(con.execute(metric_query, args).fetchdf())
            daily_query, _ = _daily_query(outcome_cte, spec["sample_group"])
            daily_parts.append(con.execute(daily_query, args).fetchdf())
            control_query, _ = _control_query(outcome_cte, spec["sample_group"])
            control_parts.append(con.execute(control_query, args).fetchdf())
            board_parts.append(_board_audit(con, files, universe_paths, spec["sample_group"]))
            label_path = output_root / f"signal_streak_labels_{spec['sample_group']}.parquet"
            label_count += _write_labels(con, files, label_path, spec["sample_group"])
            _memory_floor()
            gc.collect()
    finally:
        con.close()
    metrics = _melt_metrics(pd.concat(metrics_parts, ignore_index=True))
    daily = _summarize_daily(pd.concat(daily_parts, ignore_index=True))
    controls = _melt_controls(pd.concat(control_parts, ignore_index=True))
    board = pd.concat(board_parts, ignore_index=True)
    _csv(output_root / "event_metrics.csv", metrics)
    _csv(output_root / "daily_metrics.csv", daily)
    _csv(output_root / "control_summary.csv", controls)
    _csv(output_root / "streak_distribution.csv", streak_distribution)
    _csv(output_root / "board_audit.csv", board)
    manifest = {
        "schema": "quantlab.minute_strategy_mainboard_consecutive_filter/1",
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "samples": [
            {
                "sample_group": item["sample_group"],
                "variant": item["variant"],
                "root": str(item["root"]),
                "signal_date_count": len(item["dates"]),
                "signal_date_start": item["dates"][0],
                "signal_date_end": item["dates"][-1],
                "outcome_file_count": len(item["files"]),
            }
            for item in specs
        ],
        "label_definition": {
            "primary": "completed_prior_daily_limit_up_close",
            "ordinary_mainboard_limit_pct": 0.10,
            "st_limit_pct": 0.05,
            "price_rounding_decimals": 2,
            "half_tick_tolerance": 0.0051,
            "signal_day_used": False,
            "touched_limit_up_is_sensitivity_only": True,
            "source_domains": ["market_daily_raw", "security_status", "universe_snapshot"],
        },
        "history_start": history_start,
        "history_end": max_date,
        "label_row_count_total": int(label_count),
        "board_non_main_count_total": int(pd.to_numeric(board["non_main_count"], errors="coerce").sum()),
        "board_unresolved_count_total": int(pd.to_numeric(board["unresolved_count"], errors="coerce").sum()),
        "event_metric_rows": int(len(metrics)),
        "daily_metric_rows": int(len(daily)),
        "control_metric_rows": int(len(controls)),
        "files": {
            "event_metrics": "event_metrics.csv",
            "daily_metrics": "daily_metrics.csv",
            "control_summary": "control_summary.csv",
            "streak_distribution": "streak_distribution.csv",
            "board_audit": "board_audit.csv",
            "research_summary": "research_summary.md",
        },
    }
    _dump(output_root / "result.json", manifest)
    _write_report(output_root, manifest, metrics, daily, controls)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--output-root",
        default="research/records/minute_strategy_mainboard_consecutive_v1",
    )
    args = parser.parse_args(argv)
    result = run(Path(args.workspace_root).resolve(), Path(args.output_root).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
