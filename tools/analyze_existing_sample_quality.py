"""Audit tails, missing outcomes, feature quality and author hypotheses.

The tool reads only existing minute outcome partitions and the compact event
summaries.  It never rebuilds causal minute states or creates new signals.
DuckDB is kept bounded so the audit can run on the 16 GiB workstation while
leaving the machine-wide memory reserve intact.
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

HORIZONS = ("5m", "15m", "30m", "60m", "1d", "2d", "3d", "5d")
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
NUMERIC_FEATURES = (
    "sector_strength_rank",
    "sector_breadth",
    "leader_relative_return",
    "vwap_deviation",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
)
AUTHOR_MAP = {
    "fu_r3_core_pullback": ("fu_ge_long_kong_long", "FU-R3", "付哥龙空龙"),
    "ht_r3_core_pullback": ("hai_tang_jun", "HT-R3", "海棠君_"),
    "cs_r3_leader_confirmation": (
        "chong_sheng_san_hu_yi_ge",
        "CS-R3",
        "重生之散户一哥",
    ),
    "ng_r1_auction_reclaim": ("niu_ge_xun_shi", "NG-R1", "牛哥寻势"),
    "ng_r3_volume_quality": ("niu_ge_xun_shi", "NG-R3", "牛哥寻势"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not_jsonable:{type(value).__name__}")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_inventory(workspace_root: Path, event_record: Path) -> list[dict[str, Any]]:
    payload = json.loads((event_record / "sample_inventory.json").read_text(encoding="utf-8"))
    canonical = payload.get("canonical", [])
    if not canonical:
        raise RuntimeError("existing_sample_inventory_empty")
    return canonical


def _source_cte(paths: list[str]) -> tuple[str, list[Any]]:
    return "read_parquet(?, union_by_name=true)", [paths]


def _feature_quality(
    con: duckdb.DuckDBPyConnection,
    canonical: list[dict[str, Any]],
    output: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    flag_expr = []
    for name in FEATURE_FLAGS:
        flag = f"COALESCE(TRY_CAST(\"{name}\" AS BOOLEAN), FALSE)"
        flag_expr.extend(
            [
                f"SUM(CASE WHEN {flag} THEN 1 ELSE 0 END) AS \"{name}_true_count\"",
                f"AVG(CASE WHEN {flag} THEN 1.0 ELSE 0.0 END) AS \"{name}_true_fraction\"",
            ]
        )
    numeric_expr = []
    for name in NUMERIC_FEATURES:
        value = f"TRY_CAST(\"{name}\" AS DOUBLE)"
        numeric_expr.extend(
            [
                f"COUNT(*) FILTER (WHERE isfinite({value})) AS \"{name}_finite_count\"",
                f"COUNT(*) FILTER (WHERE {value} IS NULL OR NOT isfinite({value})) AS \"{name}_invalid_count\"",
                f"AVG({value}) FILTER (WHERE isfinite({value})) AS \"{name}_finite_mean\"",
                f"MEDIAN({value}) FILTER (WHERE isfinite({value})) AS \"{name}_finite_median\"",
            ]
        )
    for item in canonical:
        paths = item.get("outcome_paths", [])
        if not paths:
            continue
        cte, args = _source_cte(paths)
        query = f"""
        SELECT
            ? AS sample_group,
            CAST(strategy_id AS VARCHAR) AS strategy_id,
            TRY_CAST(ma_period AS INTEGER) AS ma_period,
            COUNT(*) AS signal_count,
            {", ".join(flag_expr)},
            {", ".join(numeric_expr)}
        FROM {cte}
        WHERE COALESCE(TRY_CAST(causal_only AS BOOLEAN), FALSE)
          AND NOT COALESCE(TRY_CAST(diagnostic_only AS BOOLEAN), FALSE)
        GROUP BY ALL
        ORDER BY ALL
        """
        rows.append(con.execute(query, [item["sample_group"], *args]).fetchdf())
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _csv_write(output / "signal_feature_quality.csv", result)
    return result


def _tail_summary(
    con: duckdb.DuckDBPyConnection,
    canonical: list[dict[str, Any]],
    output: Path,
) -> pd.DataFrame:
    metric_specs = [
        (prefix, horizon, f"{prefix}_{horizon}")
        for prefix in ("net_return", "gross_return")
        for horizon in HORIZONS
    ]
    periods = (10, 20, 40, 60, 120, 240)
    frames: list[pd.DataFrame] = []
    for item in canonical:
        paths = item.get("outcome_paths", [])
        if not paths:
            continue
        expressions: list[str] = []
        for prefix, horizon, column in metric_specs:
            value = f"TRY_CAST(\"{column}\" AS DOUBLE)"
            alias = f"{prefix}_{horizon}"
            expressions.extend(
                [
                    f"COUNT(*) FILTER (WHERE isfinite({value})) AS \"{alias}__observed\"",
                    f"COUNT(*) FILTER (WHERE isfinite({value}) AND {value} > 0) AS \"{alias}__positive\"",
                    f"AVG({value}) FILTER (WHERE isfinite({value})) AS \"{alias}__mean\"",
                    f"MIN({value}) FILTER (WHERE isfinite({value})) AS \"{alias}__min\"",
                    f"MAX({value}) FILTER (WHERE isfinite({value})) AS \"{alias}__max\"",
                    f"approx_quantile({value}, 0.01) FILTER (WHERE isfinite({value})) AS \"{alias}__q01\"",
                    f"approx_quantile({value}, 0.05) FILTER (WHERE isfinite({value})) AS \"{alias}__q05\"",
                    f"approx_quantile({value}, 0.25) FILTER (WHERE isfinite({value})) AS \"{alias}__q25\"",
                    f"approx_quantile({value}, 0.50) FILTER (WHERE isfinite({value})) AS \"{alias}__q50\"",
                    f"approx_quantile({value}, 0.75) FILTER (WHERE isfinite({value})) AS \"{alias}__q75\"",
                    f"approx_quantile({value}, 0.95) FILTER (WHERE isfinite({value})) AS \"{alias}__q95\"",
                    f"approx_quantile({value}, 0.99) FILTER (WHERE isfinite({value})) AS \"{alias}__q99\"",
                    f"approx_quantile({value}, 0.99) FILTER (WHERE isfinite({value}) AND {value} > 0) AS \"{alias}__positive_q99\"",
                ]
            )
        for period in periods:
            query = f"""
            SELECT CAST(strategy_id AS VARCHAR) AS strategy_id,
                   TRY_CAST(ma_period AS INTEGER) AS ma_period,
                   {", ".join(expressions)}
            FROM read_parquet(?, union_by_name=true)
            WHERE COALESCE(TRY_CAST(causal_only AS BOOLEAN), FALSE)
              AND NOT COALESCE(TRY_CAST(diagnostic_only AS BOOLEAN), FALSE)
              AND TRY_CAST(ma_period AS INTEGER) = ?
            GROUP BY ALL
            ORDER BY ALL
            """
            wide = con.execute(query, [paths, period]).fetchdf()
            if wide.empty:
                continue
            rows: list[dict[str, Any]] = []
            for record in wide.to_dict("records"):
                for prefix, horizon, _column in metric_specs:
                    alias = f"{prefix}_{horizon}"
                    observed = int(record.get(f"{alias}__observed") or 0)
                    rows.append(
                        {
                            "sample_group": item["sample_group"],
                            "metric": prefix,
                            "horizon": horizon,
                            "strategy_id": record.get("strategy_id"),
                            "ma_period": record.get("ma_period"),
                            "observed_count": observed,
                            "positive_count": int(record.get(f"{alias}__positive") or 0),
                            "mean_exact": record.get(f"{alias}__mean"),
                            "min_exact": record.get(f"{alias}__min"),
                            "max_exact": record.get(f"{alias}__max"),
                            "q01": record.get(f"{alias}__q01"),
                            "q05": record.get(f"{alias}__q05"),
                            "q25": record.get(f"{alias}__q25"),
                            "q50": record.get(f"{alias}__q50"),
                            "q75": record.get(f"{alias}__q75"),
                            "q95": record.get(f"{alias}__q95"),
                            "q99": record.get(f"{alias}__q99"),
                            "positive_q99": record.get(f"{alias}__positive_q99"),
                            "tail_estimate_method": "duckdb_approx_quantile_per_ma_period",
                        }
                    )
            frames.append(pd.DataFrame(rows))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _csv_write(output / "event_tail_summary.csv", result)
    return result


def _outcome_quality(event_record: Path, output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(event_record / "event_metrics.csv")
    metrics = metrics.loc[metrics["metric"].isin(["gross_return", "net_return"])].copy()
    metrics["missing_count"] = metrics["signal_count"] - metrics["observed_count"]
    metrics["missing_fraction"] = metrics["missing_count"] / metrics["signal_count"].replace(0, np.nan)
    metrics["observed_fraction"] = metrics["observed_count"] / metrics["signal_count"].replace(0, np.nan)
    metrics = metrics.sort_values(
        ["sample_group", "metric", "horizon", "strategy_id", "ma_period"],
        kind="stable",
    )
    _csv_write(output / "outcome_quality_summary.csv", metrics)
    return metrics


def _cost_drag(event_record: Path, output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(event_record / "event_metrics.csv")
    keys = ["sample_group", "strategy_id", "ma_period", "horizon"]
    gross = metrics.loc[metrics["metric"].eq("gross_return"), keys + ["observed_count", "mean", "median", "win_rate", "profit_factor"]].rename(
        columns={
            "observed_count": "gross_observed_count",
            "mean": "gross_mean",
            "median": "gross_median",
            "win_rate": "gross_win_rate",
            "profit_factor": "gross_profit_factor",
        }
    )
    net = metrics.loc[metrics["metric"].eq("net_return"), keys + ["observed_count", "mean", "median", "win_rate", "profit_factor"]].rename(
        columns={
            "observed_count": "net_observed_count",
            "mean": "net_mean",
            "median": "net_median",
            "win_rate": "net_win_rate",
            "profit_factor": "net_profit_factor",
        }
    )
    result = gross.merge(net, on=keys, how="outer", validate="one_to_one")
    result["mean_cost_drag"] = result["gross_mean"] - result["net_mean"]
    result["median_cost_drag"] = result["gross_median"] - result["net_median"]
    result["win_rate_change"] = result["net_win_rate"] - result["gross_win_rate"]
    result["observed_count_difference"] = result["gross_observed_count"] - result["net_observed_count"]
    result = result.sort_values(keys, kind="stable")
    _csv_write(output / "event_cost_drag.csv", result)
    return result


def _strategy_manifest(event_record: Path, output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(event_record / "event_metrics.csv")
    base = metrics.loc[metrics["metric"].eq("net_return")].groupby(
        ["sample_group", "strategy_id", "ma_period"], as_index=False
    ).agg(signal_count=("signal_count", "first"), date_count=("date_count", "first"), symbol_count=("symbol_count", "first"))
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        author_id, hypothesis_id, author_label = AUTHOR_MAP.get(
            str(row.strategy_id), (None, None, "rule_or_control")
        )
        strategy = str(row.strategy_id)
        rows.append(
            {
                "sample_group": row.sample_group,
                "strategy_id": strategy,
                "ma_period": int(row.ma_period),
                "signal_count": int(row.signal_count),
                "date_count": int(row.date_count),
                "symbol_count": int(row.symbol_count),
                "author_id": author_id,
                "hypothesis_id": hypothesis_id,
                "author_label": author_label,
                "is_control": strategy in {"s0_random_matched", "s0_liquidity_matched"},
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["sample_group", "strategy_id", "ma_period"], kind="stable"
    )
    _csv_write(output / "strategy_manifest.csv", result)
    return result


def _author_summary(event_record: Path, output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(event_record / "event_metrics.csv")
    daily = pd.read_csv(event_record / "daily_consistency.csv")
    net = metrics.loc[metrics["metric"].eq("net_return")].copy()
    net = net.loc[net["horizon"].isin(HORIZONS)]
    net = net.rename(
        columns={
            "mean": "event_mean",
            "median": "event_median",
            "win_rate": "event_win_rate",
            "profit_factor": "event_profit_factor",
            "observed_count": "event_observed_count",
        }
    )
    net = net[
        [
            "sample_group",
            "strategy_id",
            "ma_period",
            "horizon",
            "event_observed_count",
            "event_mean",
            "event_median",
            "event_win_rate",
            "event_profit_factor",
        ]
    ]
    daily = daily.rename(
        columns={
            "mean_daily_return": "daily_mean",
            "median_daily_return": "daily_median",
            "positive_date_fraction": "positive_date_fraction",
            "bh_q_value": "daily_bh_q_value",
            "date_count": "daily_date_count",
        }
    )
    daily = daily[
        [
            "sample_group",
            "strategy_id",
            "ma_period",
            "horizon",
            "daily_date_count",
            "daily_mean",
            "daily_median",
            "positive_date_fraction",
            "daily_bh_q_value",
        ]
    ]
    result = net.merge(
        daily,
        on=["sample_group", "strategy_id", "ma_period", "horizon"],
        how="left",
        validate="one_to_one",
    )
    result["author_id"] = result["strategy_id"].map(
        lambda value: AUTHOR_MAP.get(str(value), (None, None, "rule_or_control"))[0]
    )
    result["hypothesis_id"] = result["strategy_id"].map(
        lambda value: AUTHOR_MAP.get(str(value), (None, None, "rule_or_control"))[1]
    )
    result["author_label"] = result["strategy_id"].map(
        lambda value: AUTHOR_MAP.get(str(value), (None, None, "rule_or_control"))[2]
    )
    result["is_control"] = result["strategy_id"].isin(
        ["s0_random_matched", "s0_liquidity_matched"]
    )
    result = result.sort_values(
        ["sample_group", "author_label", "strategy_id", "ma_period", "horizon"],
        kind="stable",
    )
    _csv_write(output / "author_hypothesis_summary.csv", result)
    return result


def run(workspace_root: Path, event_record: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    canonical = _load_inventory(workspace_root, event_record)
    _outcome_quality(event_record, output)
    _cost_drag(event_record, output)
    _strategy_manifest(event_record, output)
    _author_summary(event_record, output)
    temp_directory = workspace_root / "tmp" / "minute_existing_quality_duckdb"
    temp_directory.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='512MB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{str(temp_directory).replace(chr(39), chr(39) * 2)}'")
    try:
        feature = _feature_quality(con, canonical, output)
        tail = _tail_summary(con, canonical, output)
    finally:
        con.close()
    result = {
        "schema": "quantlab.minute_strategy_existing_sample_quality/1",
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "sample_group_count": len(canonical),
        "feature_quality_rows": int(len(feature)),
        "tail_summary_rows": int(len(tail)),
        "files": {
            "signal_feature_quality": "signal_feature_quality.csv",
            "event_tail_summary": "event_tail_summary.csv",
            "outcome_quality_summary": "outcome_quality_summary.csv",
            "event_cost_drag": "event_cost_drag.csv",
            "strategy_manifest": "strategy_manifest.csv",
            "author_hypothesis_summary": "author_hypothesis_summary.csv",
        },
    }
    _json_dump(output / "quality_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--event-record",
        default="research/records/minute_strategy_existing_samples_v1",
    )
    parser.add_argument(
        "--output-root",
        default="research/records/minute_strategy_existing_samples_v1",
    )
    args = parser.parse_args()
    result = run(
        Path(args.workspace_root).resolve(),
        Path(args.event_record).resolve(),
        Path(args.output_root).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
