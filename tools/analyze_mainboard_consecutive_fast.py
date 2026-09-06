"""Fast, causal re-slice of the already generated minute samples.

The minute sample is already point-in-time main-board data.  This tool only
adds a daily state label: the number of completed limit-up closes immediately
before each signal date.  It scans the existing outcomes once per sample and
does not regenerate minute bars or events.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil

GIB = 1024**3
HORIZONS = ("60m", "1d", "5d")
DAILY_HORIZONS = ("60m", "1d", "5d")
THRESHOLDS = (
    ("all", "TRUE"),
    ("ge1", "prior_limit_up_streak >= 1"),
    ("ge2", "prior_limit_up_streak >= 2"),
    ("ge3", "prior_limit_up_streak >= 3"),
    ("ge4", "prior_limit_up_streak >= 4"),
    ("ge5", "prior_limit_up_streak >= 5"),
)
CONTROL_IDS = ("s0_random_matched", "s0_liquidity_matched")


def _scan(paths: list[Path]) -> str:
    return "read_parquet(?, union_by_name=true)"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _check_memory(floor_gib: float = 0.5) -> None:
    available = psutil.virtual_memory().available / GIB
    if available < float(floor_gib):
        raise RuntimeError(f"mainboard_consecutive_memory_floor_breached:{available:.3f}<{floor_gib}")


def _active_paths(workspace: Path, domain: str) -> list[Path]:
    sys.path.insert(0, str(workspace / "src"))
    from quantlab.data.qdp_v2.active import resolve_active_domain

    return [Path(path) for path in resolve_active_domain(domain, workspace_root=workspace).shard_paths]


def _samples(workspace: Path) -> list[dict[str, Any]]:
    specs = (
        ("primary_development_2022_2023_partial", workspace / "runs/minute_ma_development_2022_2024", "current_contract"),
        ("supplemental_2023_09_legacy", workspace / "runs/minute_ma_month_2023_09", "legacy_vwap"),
    )
    result = []
    for name, root, variant in specs:
        files = sorted((root / "outcomes").glob("date=*/outcomes.parquet"))
        dates = sorted(p.parent.name.split("=", 1)[-1] for p in files)
        if not files:
            raise FileNotFoundError(f"mainboard_consecutive_sample_empty:{root}")
        result.append({"sample_group": name, "root": root, "variant": variant, "files": files, "dates": dates})
    return result


def _build_streaks(
    con: duckdb.DuckDBPyConnection,
    daily_paths: list[Path],
    status_paths: list[Path],
    start_date: str,
    end_date: str,
) -> None:
    """Build daily labels once; all later queries join this small relation."""

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE daily_limit_streak AS
        WITH daily AS (
            SELECT CAST(symbol AS VARCHAR) symbol,
                   CAST(trade_date AS DATE) trade_date,
                   TRY_CAST(high AS DOUBLE) high_price,
                   TRY_CAST(\"close\" AS DOUBLE) close_price
            FROM {_scan(daily_paths)}
            WHERE trade_date BETWEEN ? AND ?
        ), status AS (
            SELECT CAST(symbol AS VARCHAR) symbol,
                   CAST(trade_date AS DATE) trade_date,
                   BOOL_OR(COALESCE(is_st, FALSE)) is_st
            FROM {_scan(status_paths)}
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY 1, 2
        ), ordered AS (
            SELECT d.*, COALESCE(s.is_st, FALSE) is_st,
                   LAG(close_price) OVER (PARTITION BY d.symbol ORDER BY trade_date) previous_close
            FROM daily d LEFT JOIN status s USING (symbol, trade_date)
        ), marked AS (
            SELECT *,
                   CASE WHEN previous_close > 0 AND ABS(
                       close_price - ROUND(previous_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END), 2)
                   ) <= 0.0051 THEN TRUE ELSE FALSE END is_limit_up_close,
                   CASE WHEN previous_close > 0 AND high_price >=
                       ROUND(previous_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END), 2) - 0.0051
                   THEN TRUE ELSE FALSE END touched_limit_up
            FROM ordered
        ), grouped AS (
            SELECT *, COALESCE(SUM(CASE WHEN is_limit_up_close THEN 0 ELSE 1 END) OVER (
                PARTITION BY symbol ORDER BY trade_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0) streak_group
            FROM marked
        ), streaks AS (
            SELECT *, CASE WHEN is_limit_up_close THEN ROW_NUMBER() OVER (
                PARTITION BY symbol, streak_group ORDER BY trade_date
            ) ELSE 0 END limit_up_streak
            FROM grouped
        )
        SELECT symbol, trade_date, is_st, is_limit_up_close, touched_limit_up,
               CAST(LAG(limit_up_streak) OVER (PARTITION BY symbol ORDER BY trade_date) AS BIGINT)
                   AS prior_limit_up_streak,
               COALESCE(LAG(touched_limit_up) OVER (PARTITION BY symbol ORDER BY trade_date), FALSE)
                   AS prior_day_touched_limit_up
        FROM streaks
        """,
        [list(map(str, daily_paths)), start_date, end_date, list(map(str, status_paths)), start_date, end_date],
    )


def _outcome_cte(files: list[Path]) -> str:
    text_cols = {"signal_id", "strategy_id", "symbol", "signal_date", "signal_time", "reference_signal_id"}
    columns = [
        "signal_id", "strategy_id", "symbol", "signal_date", "signal_time",
        "sixty_minute_bucket", "ma_period", "causal_only", "diagnostic_only", "reference_signal_id",
        *[f"net_return_{h}" for h in HORIZONS],
    ]
    projection = []
    for name in columns:
        if name in {"causal_only", "diagnostic_only"}:
            projection.append(f"COALESCE(TRY_CAST({name} AS BOOLEAN), FALSE) AS {name}")
        elif name in text_cols:
            projection.append(f"CAST({name} AS VARCHAR) AS {name}")
        else:
            projection.append(f"TRY_CAST({name} AS DOUBLE) AS {name}")
    return f"SELECT {', '.join(projection)} FROM {_scan(files)}"


def _labeled_cte(files: list[Path]) -> tuple[str, list[Any]]:
    cte = _outcome_cte(files)
    query = f"""
    WITH outcomes AS ({cte}), labeled AS (
        SELECT o.*, COALESCE(d.prior_limit_up_streak, 0) prior_limit_up_streak,
               COALESCE(d.prior_day_touched_limit_up, FALSE) prior_day_touched_limit_up,
               (d.symbol IS NOT NULL) streak_label_observed
        FROM outcomes o
        LEFT JOIN daily_limit_streak d
          ON d.symbol=o.symbol AND d.trade_date=TRY_CAST(o.signal_date AS DATE)
        WHERE o.causal_only AND NOT o.diagnostic_only
    )
    """
    return query, [list(map(str, files))]


def _metric_query(files: list[Path], sample_group: str) -> tuple[str, list[Any]]:
    fields: list[str] = []
    for threshold, condition in THRESHOLDS:
        for horizon in HORIZONS:
            for kind, expression in (("net", f"net_return_{horizon}"),):
                guard = f"({condition}) AND isfinite({expression})"
                prefix = f"{threshold}_{kind}_{horizon}"
                fields += [
                    f"COUNT(*) FILTER (WHERE {guard}) {prefix}_count",
                    f"AVG({expression}) FILTER (WHERE {guard}) {prefix}_mean",
                    f"SUM({expression}) FILTER (WHERE {guard} AND {expression}>0) {prefix}_positive_sum",
                    f"SUM({expression}) FILTER (WHERE {guard} AND {expression}<0) {prefix}_negative_sum",
                    f"COUNT(*) FILTER (WHERE {guard} AND {expression}>0) {prefix}_positive_count",
                    f"COUNT(*) FILTER (WHERE {guard}) {prefix}_row_count",
                ]
    cte, args = _labeled_cte(files)
    query = cte + " SELECT " + repr(sample_group) + " sample_group, strategy_id, CAST(ma_period AS INTEGER) ma_period, " + ", ".join(fields) + " FROM labeled GROUP BY ALL ORDER BY ALL"
    return query, args


def _melt_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        for threshold, _ in THRESHOLDS:
            for kind in ("net",):
                for horizon in HORIZONS:
                    prefix = f"{threshold}_{kind}_{horizon}"
                    count = int(row.get(f"{prefix}_count") or 0)
                    pos_sum = float(row.get(f"{prefix}_positive_sum") or 0.0)
                    neg_sum = float(row.get(f"{prefix}_negative_sum") or 0.0)
                    rows.append(
                        {
                            "sample_group": row.get("sample_group"), "strategy_id": row.get("strategy_id"),
                            "ma_period": row.get("ma_period"), "threshold": threshold, "metric": kind,
                            "horizon": horizon, "observed_count": count,
                            "mean": row.get(f"{prefix}_mean"), "median": None,
                            "win_rate": (float(row.get(f"{prefix}_positive_count") or 0) / count if count else None),
                            "profit_factor": (pos_sum / abs(neg_sum) if neg_sum < 0 else None),
                            "row_count": int(row.get(f"{prefix}_row_count") or 0),
                            "date_count": None,
                            "symbol_count": None,
                        }
                    )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["profit_factor"] = result["profit_factor"].replace([np.inf, -np.inf], np.nan)
    return result


def _daily_query(files: list[Path], sample_group: str) -> tuple[str, list[Any]]:
    fields: list[str] = []
    for threshold, condition in THRESHOLDS:
        for horizon in DAILY_HORIZONS:
            value = f"net_return_{horizon}"
            prefix = f"{threshold}_{horizon}"
            fields += [
                f"AVG({value}) FILTER (WHERE ({condition}) AND isfinite({value})) {prefix}_mean",
                f"COUNT(*) FILTER (WHERE ({condition}) AND isfinite({value})) {prefix}_count",
            ]
    cte, args = _labeled_cte(files)
    query = cte + " SELECT " + repr(sample_group) + " sample_group, strategy_id, CAST(ma_period AS INTEGER) ma_period, CAST(signal_date AS DATE) signal_date, " + ", ".join(fields) + " FROM labeled GROUP BY ALL"
    return query, args


def _summarize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["sample_group", "strategy_id", "ma_period"], sort=True):
        sample, strategy, ma = keys
        for threshold, _ in THRESHOLDS:
            for horizon in DAILY_HORIZONS:
                values = pd.to_numeric(group[f"{threshold}_{horizon}_mean"], errors="coerce")
                counts = pd.to_numeric(group[f"{threshold}_{horizon}_count"], errors="coerce").fillna(0)
                values = values.loc[values.notna() & counts.gt(0)]
                rows.append({
                    "sample_group": sample, "strategy_id": strategy, "ma_period": ma,
                    "threshold": threshold, "horizon": horizon, "date_count": int(len(values)),
                    "mean_daily_return": float(values.mean()) if len(values) else None,
                    "median_daily_return": float(values.median()) if len(values) else None,
                    "positive_date_fraction": float((values > 0).mean()) if len(values) else None,
                })
    return pd.DataFrame(rows)


def _control_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare each rule to same-filter cross-sectional control baselines.

    This is deliberately labelled a baseline comparison, not the stricter
    signal-to-reference matched test already present in the main report.
    """

    if metrics.empty:
        return pd.DataFrame()
    controls = metrics.loc[metrics.strategy_id.isin(CONTROL_IDS)].copy()
    targets = metrics.loc[~metrics.strategy_id.isin(CONTROL_IDS)].copy()
    if controls.empty or targets.empty:
        return pd.DataFrame()
    control_mean = controls.groupby(["sample_group", "strategy_id", "ma_period", "threshold", "metric", "horizon"], as_index=False).agg(
        control_count=("observed_count", "sum"),
        control_mean=("mean", "mean"),
    )
    rows = []
    for _, target in targets.iterrows():
        for _, control in control_mean.loc[
            (control_mean.sample_group == target.sample_group)
            & (control_mean.ma_period == target.ma_period)
            & (control_mean.threshold == target.threshold)
            & (control_mean.metric == target.metric)
            & (control_mean.horizon == target.horizon)
        ].iterrows():
            rows.append({
                "sample_group": target.sample_group, "strategy_id": target.strategy_id,
                "ma_period": target.ma_period, "threshold": target.threshold,
                "metric": target.metric, "horizon": target.horizon,
                "control_id": control.strategy_id, "target_count": int(target.observed_count),
                "control_count": int(control.control_count), "target_mean": target["mean"],
                "control_mean": control.control_mean,
                "mean_difference": target["mean"] - control.control_mean
                if pd.notna(target["mean"]) and pd.notna(control.control_mean) else np.nan,
            })
    return pd.DataFrame(rows)


def _labels_and_board_audit(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    universe_paths: list[Path],
    output: Path,
    sample_group: str,
    dates: list[str],
) -> tuple[int, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    label_path = output / f"signal_streak_labels_{sample_group}.parquet"
    query = f"""
    WITH keys AS (SELECT DISTINCT CAST(symbol AS VARCHAR) symbol, CAST(signal_date AS DATE) signal_date FROM {_scan(files)})
    SELECT {repr(sample_group)} sample_group, k.symbol, k.signal_date,
           COALESCE(d.prior_limit_up_streak, 0) prior_limit_up_streak,
           COALESCE(d.prior_day_touched_limit_up, FALSE) prior_day_touched_limit_up,
           (d.symbol IS NOT NULL) streak_label_observed
    FROM keys k LEFT JOIN daily_limit_streak d ON d.symbol=k.symbol AND d.trade_date=k.signal_date
    """
    target_sql = str(label_path).replace("'", "''")
    con.execute(
        f"COPY ({query}) TO '{target_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)",
        [list(map(str, files))],
    )
    label_count = int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(label_path)]).fetchone()[0])
    board_query = f"""
    WITH keys AS (SELECT DISTINCT CAST(symbol AS VARCHAR) symbol, CAST(signal_date AS DATE) trade_date FROM {_scan(files)}),
    u AS (SELECT CAST(symbol AS VARCHAR) symbol, CAST(trade_date AS DATE) trade_date, ANY_VALUE(CAST(board AS VARCHAR)) board FROM {_scan(universe_paths)} WHERE trade_date BETWEEN ? AND ? GROUP BY 1,2)
    SELECT {repr(sample_group)} sample_group, COUNT(*) sample_symbol_date_count,
           COUNT(*) FILTER (WHERE u.board='main') main_count,
           COUNT(*) FILTER (WHERE u.board IS NULL) unresolved_count,
           COUNT(*) FILTER (WHERE u.board IS NOT NULL AND u.board<>'main') non_main_count,
           COUNT(*) FILTER (WHERE d.symbol IS NOT NULL) label_observed_count,
           COUNT(*) FILTER (WHERE d.symbol IS NULL) label_unobserved_count
    FROM keys k LEFT JOIN u USING(symbol,trade_date)
    LEFT JOIN daily_limit_streak d ON d.symbol=k.symbol AND d.trade_date=k.trade_date
    """
    audit = con.execute(board_query, [list(map(str, files)), list(map(str, universe_paths)), dates[0], dates[-1]]).fetchdf()
    return label_count, audit


def _write_report(output: Path, metrics: pd.DataFrame, daily: pd.DataFrame, controls: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Main-board consecutive limit-up filter study", "",
        "The existing minute samples are already point-in-time Shanghai/Shenzhen main-board data. This study adds only a causal prior-day limit-up streak label; it does not regenerate minute events.", "",
        "## Label contract", "",
        "`prior_limit_up_streak` counts completed prior trading days whose unadjusted close equals the reconstructed upper limit. Ordinary main-board shares use 10%, ST shares 5%, the theoretical limit is rounded to two decimals, and half a tick is tolerated. The signal-day bar is not used. A touched-but-not-closed flag is retained for sensitivity only.", "",
        "`control_comparison.csv` is a same-filter cross-sectional baseline comparison. It is weaker than the matched-control test in the main existing-sample report and must not be read as causal proof.", "",
        f"Samples: {', '.join(item['sample_group'] for item in manifest['samples'])}",
        f"Label rows: {manifest['label_row_count_total']:,}; unobserved daily-label rows: {manifest['label_unobserved_count_total']:,}; non-main audit rows: {manifest['board_non_main_count_total']:,}; unresolved board rows: {manifest['board_unresolved_count_total']:,}.",
        "",
        "## Reading the result", "",
        "A better conditional mean on `geN` can be a narrow high-volatility selection effect. Require enough events, positive day-level consistency, a positive control difference, and a fresh forward sample before treating it as a candidate.",
    ]
    if not metrics.empty:
        focus = metrics.loc[
            (metrics.metric == "net") & metrics.horizon.isin(["60m", "1d", "5d"])
            & metrics.threshold.isin(["all", "ge1", "ge2", "ge3"])
        ].sort_values(["sample_group", "mean"], ascending=[True, False]).head(30)
        if not focus.empty:
            lines += ["", "## Highest conditional rows (descriptive only)", "", "| Sample | Strategy | MA | Filter | Horizon | N | Mean | Median | Win rate |", "|---|---|---:|---|---|---:|---:|---:|---:|"]
            for _, row in focus.iterrows():
                median = f"{float(row['median']):.3%}" if pd.notna(row["median"]) else "n/a"
                lines.append(f"| {row.sample_group} | {row.strategy_id} | {int(row.ma_period)} | {row.threshold} | {row.horizon} | {int(row.observed_count):,} | {float(row['mean']):.3%} | {median} | {float(row.win_rate):.2%} |" if pd.notna(row["mean"]) else "")
    (output / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    _check_memory()
    specs = _samples(workspace_root)
    dates = [date for item in specs for date in item["dates"]]
    history_start = (pd.Timestamp(min(dates)) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    daily_paths = _active_paths(workspace_root, "market_daily_raw")
    status_paths = _active_paths(workspace_root, "security_status")
    universe_paths = _active_paths(workspace_root, "universe_snapshot")
    output_root.mkdir(parents=True, exist_ok=True)
    temp = workspace_root / "tmp" / "mainboard_consecutive_fast_duckdb"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='512MB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{str(temp).replace(chr(39), chr(39) * 2)}'")
    metric_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    board_parts: list[pd.DataFrame] = []
    label_count = 0
    try:
        _build_streaks(con, daily_paths, status_paths, history_start, max(dates))
        streak_distribution = con.execute(
            "SELECT prior_limit_up_streak, COUNT(*) row_count, COUNT(DISTINCT symbol) symbol_count, MIN(trade_date) first_date, MAX(trade_date) last_date FROM daily_limit_streak WHERE trade_date BETWEEN ? AND ? GROUP BY 1 ORDER BY 1",
            [min(dates), max(dates)],
        ).fetchdf()
        for item in specs:
            _check_memory()
            query, args = _metric_query(item["files"], item["sample_group"])
            metric_parts.append(con.execute(query, args).fetchdf())
            query, args = _daily_query(item["files"], item["sample_group"])
            daily_parts.append(con.execute(query, args).fetchdf())
            count, audit = _labels_and_board_audit(con, item["files"], universe_paths, output_root, item["sample_group"], item["dates"])
            label_count += count
            board_parts.append(audit)
            gc.collect()
            _check_memory()
    finally:
        con.close()
    metrics = _melt_metrics(pd.concat(metric_parts, ignore_index=True))
    daily = _summarize_daily(pd.concat(daily_parts, ignore_index=True))
    # The daily table has one row per signal date, so use it for the exact
    # horizon-specific date coverage in the event table.  Do not label a raw
    # event-row count as a date count.
    if not metrics.empty and not daily.empty:
        date_lookup = daily.loc[:, ["sample_group", "strategy_id", "ma_period", "threshold", "horizon", "date_count"]]
        metrics = metrics.drop(columns=["date_count"], errors="ignore").merge(
            date_lookup,
            on=["sample_group", "strategy_id", "ma_period", "threshold", "horizon"],
            how="left",
            validate="one_to_one",
        )
    controls = _control_comparison(metrics)
    board = pd.concat(board_parts, ignore_index=True)
    _csv(output_root / "event_metrics.csv", metrics)
    _csv(output_root / "daily_metrics.csv", daily)
    _csv(output_root / "control_comparison.csv", controls)
    _csv(output_root / "streak_distribution.csv", streak_distribution)
    _csv(output_root / "board_audit.csv", board)
    manifest = {
        "schema": "quantlab.minute_strategy_mainboard_consecutive_filter/2", "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_seconds": round(time.perf_counter() - started, 3),
        "samples": [{"sample_group": i["sample_group"], "variant": i["variant"], "root": str(i["root"]), "signal_date_count": len(i["dates"]), "signal_date_start": i["dates"][0], "signal_date_end": i["dates"][-1], "outcome_file_count": len(i["files"])} for i in specs],
        "history_start": history_start, "history_end": max(dates), "label_row_count_total": int(label_count),
        "board_non_main_count_total": int(pd.to_numeric(board["non_main_count"], errors="coerce").sum()),
        "board_unresolved_count_total": int(pd.to_numeric(board["unresolved_count"], errors="coerce").sum()),
        "label_unobserved_count_total": int(pd.to_numeric(board["label_unobserved_count"], errors="coerce").sum()),
        "label_definition": {"primary": "completed_prior_daily_limit_up_close", "ordinary_mainboard_limit_pct": 0.10, "st_limit_pct": 0.05, "price_rounding_decimals": 2, "half_tick_tolerance": 0.0051, "signal_day_used": False, "touched_limit_up_is_sensitivity_only": True, "source_domains": ["market_daily_raw", "security_status", "universe_snapshot"]},
        "event_metric_rows": int(len(metrics)), "daily_metric_rows": int(len(daily)), "control_comparison_rows": int(len(controls)),
        "files": {"event_metrics": "event_metrics.csv", "daily_metrics": "daily_metrics.csv", "control_comparison": "control_comparison.csv", "streak_distribution": "streak_distribution.csv", "board_audit": "board_audit.csv", "research_summary": "research_summary.md"},
    }
    _dump(output_root / "result.json", manifest)
    _write_report(output_root, metrics, daily, controls, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-root", default="research/records/minute_strategy_mainboard_consecutive_v2")
    args = parser.parse_args(argv)
    result = run(Path(args.workspace_root).resolve(), Path(args.output_root).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
