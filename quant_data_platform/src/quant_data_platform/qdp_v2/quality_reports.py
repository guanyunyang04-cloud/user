from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile


PRICE_TICK = 0.0101
FIVE_TICKS = 0.0501


def build_intraday_quality_report(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    focus_year: int = 2026,
    sample_limit: int = 50,
    write_chart: bool = True,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    raw = dict(active.get("raw", {}) or {})
    daily_id = str(raw.get("market_daily_raw", "") or "")
    five_id = str(raw.get("market_intraday_5m", "") or "")
    if not daily_id or not five_id:
        return {"status": "error", "errors": ["market_daily_raw_or_market_intraday_5m_missing"]}

    daily = _read_manifest(root, daily_id, "market_daily_raw")
    five = _read_manifest(root, five_id, "market_intraday_5m")
    daily_paths = _existing_shard_paths(daily, root=root)
    five_paths = _existing_shard_paths(five, root=root)
    if not daily_paths or not five_paths:
        return {"status": "error", "errors": ["daily_or_5m_shards_missing"]}

    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = max(1, int(threads or profile.duckdb_threads or 1))
    stamp = _stamp()
    audit_dir = root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    examples_path = audit_dir / f"intraday_quality_examples_{stamp}.csv"
    chart_path = audit_dir / f"intraday_quality_summary_{stamp}.png"

    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        _configure(con, memory_limit=memory_limit, threads=thread_count)
        con.execute(
            f"""
            create temp table intra_daily as
            with parsed as (
              select
                symbol,
                trade_date,
                regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt,
                cast(open as double) as open,
                cast(high as double) as high,
                cast(low as double) as low,
                cast(close as double) as close,
                cast(volume as double) as volume,
                cast(amount as double) as amount,
                source as intraday_source
              from read_parquet({_path_list_sql(five_paths)}, union_by_name=true)
            ),
            shaped as (
              select
                *,
                cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
              from parsed
              where length(bt) >= 4
            )
            select
              symbol,
              trade_date,
              arg_min(open, minute_of_day) as intraday_open,
              max(high) as intraday_high,
              min(low) as intraday_low,
              arg_max(close, minute_of_day) as intraday_close,
              sum(volume) as intraday_volume,
              sum(amount) as intraday_amount,
              count(*) as intraday_bars,
              any_value(intraday_source) as intraday_source
            from shaped
            group by symbol, trade_date
            """
        )
        con.execute(
            f"""
            create temp table joined_quality as
            with daily as (
              select
                symbol,
                trade_date,
                cast(open as double) as daily_open,
                cast(high as double) as daily_high,
                cast(low as double) as daily_low,
                cast(close as double) as daily_close,
                cast(volume as double) as daily_volume,
                cast(amount as double) as daily_amount,
                source as daily_source
              from read_parquet({_path_list_sql(daily_paths)}, union_by_name=true)
            )
            select
              coalesce(d.symbol, i.symbol) as symbol,
              coalesce(d.trade_date, i.trade_date) as trade_date,
              substr(coalesce(d.trade_date, i.trade_date), 1, 4) as year,
              substr(coalesce(d.trade_date, i.trade_date), 1, 7) as month,
              d.symbol is not null as has_daily,
              i.symbol is not null as has_intraday,
              d.daily_source,
              i.intraday_source,
              d.daily_open,
              d.daily_high,
              d.daily_low,
              d.daily_close,
              d.daily_volume,
              d.daily_amount,
              i.intraday_open,
              i.intraday_high,
              i.intraday_low,
              i.intraday_close,
              i.intraday_volume,
              i.intraday_amount,
              i.intraday_bars,
              abs(d.daily_open - i.intraday_open) as open_diff,
              abs(d.daily_high - i.intraday_high) as high_diff,
              abs(d.daily_low - i.intraday_low) as low_diff,
              abs(d.daily_close - i.intraday_close) as close_diff,
              abs(d.daily_volume - i.intraday_volume) as volume_diff,
              abs(d.daily_amount - i.intraday_amount) as amount_diff,
              case
                when greatest(abs(d.daily_amount), abs(i.intraday_amount)) > 0
                then abs(d.daily_amount - i.intraday_amount) / greatest(abs(d.daily_amount), abs(i.intraday_amount))
                else 0
              end as amount_rel_diff,
              greatest(
                coalesce(abs(d.daily_open - i.intraday_open), 0),
                coalesce(abs(d.daily_high - i.intraday_high), 0),
                coalesce(abs(d.daily_low - i.intraday_low), 0),
                coalesce(abs(d.daily_close - i.intraday_close), 0)
              ) as max_price_diff,
              case
                when d.daily_close >= 100 then '>=100'
                when d.daily_close >= 50 then '50-100'
                when d.daily_close >= 20 then '20-50'
                when d.daily_close >= 10 then '10-20'
                when d.daily_close is not null then '<10'
                else 'missing_daily_close'
              end as price_band
            from daily d
            full outer join intra_daily i using(symbol, trade_date)
            """
        )
        overall = _one(con, _summary_sql("joined_quality"))
        by_year = _rows(con, _summary_sql("joined_quality", group_by="year", select_prefix="year,", order_by="year"))
        by_price_band = _rows(con, _summary_sql("joined_quality", group_by="price_band", select_prefix="price_band,", order_by="price_band"))
        by_source = _rows(
            con,
            _summary_sql(
                "joined_quality",
                group_by="coalesce(daily_source, '<missing_daily>'), coalesce(intraday_source, '<missing_intraday>')",
                select_prefix="coalesce(daily_source, '<missing_daily>') as daily_source, coalesce(intraday_source, '<missing_intraday>') as intraday_source,",
                order_by="price_diff_gt_5tick_rows desc, amount_rel_diff_gt_10bp_rows desc",
                limit=50,
            ),
        )
        focus_by_month = _rows(
            con,
            _summary_sql(
                "joined_quality",
                where=f"year = '{int(focus_year)}'",
                group_by="month",
                select_prefix="month,",
                order_by="month",
            ),
        )
        examples = _rows(
            con,
            f"""
            select
              symbol,
              trade_date,
              daily_open,
              intraday_open,
              open_diff,
              daily_high,
              intraday_high,
              high_diff,
              daily_low,
              intraday_low,
              low_diff,
              daily_close,
              intraday_close,
              close_diff,
              daily_volume,
              intraday_volume,
              volume_diff,
              daily_amount,
              intraday_amount,
              amount_rel_diff,
              daily_source,
              intraday_source,
              price_band
            from joined_quality
            where
              not (has_daily and has_intraday)
              or max_price_diff > {FIVE_TICKS}
              or amount_rel_diff > 0.01
              or volume_diff > 100
            order by max_price_diff desc nulls last, amount_rel_diff desc nulls last
            limit {max(1, int(sample_limit))}
            """,
        )
        if examples:
            _write_csv(examples_path, examples)

    _add_rates(overall)
    for group in (by_year, by_price_band, by_source, focus_by_month):
        for row in group:
            _add_rates(row)

    payload: dict[str, Any] = {
        "status": "ok",
        "audit_type": "intraday_quality",
        "audited_at": utc_now(),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "runtime": profile.name,
        "duckdb_memory_limit": memory_limit,
        "threads": thread_count,
        "datasets": {
            "market_daily_raw": daily.dataset_id,
            "market_intraday_5m": five.dataset_id,
        },
        "method": "aggregate_active_5m_to_daily_then_compare_with_market_daily_raw",
        "interpretation": {
            "price_diff_gt_5tick_rate": "Main headline for price-source conflict risk.",
            "close_diff_gt_1tick_rate": "Most important for close-to-close short-term signals.",
            "amount_rel_diff_gt_10bp_rate": "Amount/source unit or rounding drift candidate.",
        },
        "overall": overall,
        "by_year": by_year,
        "by_price_band": by_price_band,
        "by_source_pair_top": by_source,
        f"by_month_{int(focus_year)}": focus_by_month,
        "examples_path": str(examples_path.resolve()) if examples else "",
        "runtime_environment": runtime_environment(),
    }
    if write_chart:
        chart_written = _write_chart(payload, chart_path)
        if chart_written:
            payload["chart_path"] = str(chart_path.resolve())
    json_path = audit_dir / f"intraday_quality_report_{stamp}.json"
    atomic_write_json(json_path, payload)
    payload["audit_path"] = str(json_path.resolve())
    return payload


def _summary_sql(
    table: str,
    *,
    where: str = "",
    group_by: str = "",
    select_prefix: str = "",
    order_by: str = "",
    limit: int = 0,
) -> str:
    where_clause = f"where {where}" if where else ""
    group_clause = f"group by {group_by}" if group_by else ""
    order_clause = f"order by {order_by}" if order_by else ""
    limit_clause = f"limit {int(limit)}" if limit else ""
    return f"""
    select
      {select_prefix}
      count(*) as joined_rows,
      sum(case when has_daily then 1 else 0 end) as daily_rows,
      sum(case when has_intraday then 1 else 0 end) as intraday_symbol_days,
      sum(case when has_daily and not has_intraday then 1 else 0 end) as daily_missing_in_intraday,
      sum(case when has_intraday and not has_daily then 1 else 0 end) as intraday_missing_in_daily,
      sum(case when has_daily and has_intraday then 1 else 0 end) as paired_rows,
      sum(case when has_daily and has_intraday and max_price_diff <= 0.001 then 1 else 0 end) as price_exact_rows,
      sum(case when has_daily and has_intraday and max_price_diff > 0.001 then 1 else 0 end) as price_any_diff_rows,
      sum(case when has_daily and has_intraday and max_price_diff > 0.001 and max_price_diff <= {PRICE_TICK} then 1 else 0 end) as price_diff_le_1tick_rows,
      sum(case when has_daily and has_intraday and max_price_diff > {PRICE_TICK} then 1 else 0 end) as price_diff_gt_1tick_rows,
      sum(case when has_daily and has_intraday and max_price_diff > {FIVE_TICKS} then 1 else 0 end) as price_diff_gt_5tick_rows,
      sum(case when has_daily and has_intraday and open_diff > {PRICE_TICK} then 1 else 0 end) as open_diff_gt_1tick_rows,
      sum(case when has_daily and has_intraday and high_diff > {PRICE_TICK} then 1 else 0 end) as high_diff_gt_1tick_rows,
      sum(case when has_daily and has_intraday and low_diff > {PRICE_TICK} then 1 else 0 end) as low_diff_gt_1tick_rows,
      sum(case when has_daily and has_intraday and close_diff > {PRICE_TICK} then 1 else 0 end) as close_diff_gt_1tick_rows,
      sum(case when has_daily and has_intraday and volume_diff > 100.0 then 1 else 0 end) as volume_diff_gt_100_rows,
      sum(case when has_daily and has_intraday and amount_rel_diff > 0.001 then 1 else 0 end) as amount_rel_diff_gt_10bp_rows,
      sum(case when has_daily and has_intraday and amount_rel_diff > 0.01 then 1 else 0 end) as amount_rel_diff_gt_1pct_rows,
      avg(case when has_daily and has_intraday then amount_rel_diff else null end) as avg_amount_rel_diff,
      quantile_cont(case when has_daily and has_intraday then amount_rel_diff else null end, 0.99) as p99_amount_rel_diff,
      max(case when has_daily and has_intraday then amount_rel_diff else null end) as max_amount_rel_diff
    from {table}
    {where_clause}
    {group_clause}
    {order_clause}
    {limit_clause}
    """


def _add_rates(row: dict[str, Any]) -> None:
    paired = float(row.get("paired_rows", 0) or 0)
    daily = float(row.get("daily_rows", 0) or 0)
    denom = paired if paired else daily if daily else float(row.get("joined_rows", 0) or 0)
    for key in [
        "price_exact_rows",
        "price_any_diff_rows",
        "price_diff_gt_1tick_rows",
        "price_diff_gt_5tick_rows",
        "open_diff_gt_1tick_rows",
        "high_diff_gt_1tick_rows",
        "low_diff_gt_1tick_rows",
        "close_diff_gt_1tick_rows",
        "volume_diff_gt_100_rows",
        "amount_rel_diff_gt_10bp_rows",
        "amount_rel_diff_gt_1pct_rows",
    ]:
        row[key.replace("_rows", "_rate")] = (float(row.get(key, 0) or 0) / denom) if denom else 0.0


def _write_chart(payload: Mapping[str, Any], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    overall = dict(payload.get("overall", {}) or {})
    by_year = list(payload.get("by_year", []) or [])
    focus_key = next((key for key in payload if str(key).startswith("by_month_")), "")
    by_month = list(payload.get(focus_key, []) or [])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("QDP Intraday vs Daily Quality Audit", fontsize=14)

    metrics = [
        ("exact", overall.get("price_exact_rate", 0)),
        (">1 tick", overall.get("price_diff_gt_1tick_rate", 0)),
        (">5 tick", overall.get("price_diff_gt_5tick_rate", 0)),
        ("close >1 tick", overall.get("close_diff_gt_1tick_rate", 0)),
    ]
    axes[0, 0].bar([m[0] for m in metrics], [float(m[1]) * 100 for m in metrics], color=["#2f855a", "#d69e2e", "#c05621", "#2b6cb0"])
    axes[0, 0].set_title("Price Difference Rates")
    axes[0, 0].set_ylabel("% of paired symbol-days")
    axes[0, 0].tick_params(axis="x", labelrotation=20)

    years = [str(row.get("year", "")) for row in by_year]
    axes[0, 1].plot(years, [float(row.get("price_diff_gt_5tick_rate", 0)) * 100 for row in by_year], marker="o", label="price >5 tick")
    axes[0, 1].plot(years, [float(row.get("close_diff_gt_1tick_rate", 0)) * 100 for row in by_year], marker="o", label="close >1 tick")
    axes[0, 1].set_title("Price Differences by Year")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].tick_params(axis="x", labelrotation=45)
    axes[0, 1].legend()

    months = [str(row.get("month", "")) for row in by_month]
    axes[1, 0].bar(months, [float(row.get("amount_rel_diff_gt_10bp_rate", 0)) * 100 for row in by_month], color="#805ad5")
    axes[1, 0].set_title(f"Amount >10bp by Month ({focus_key.replace('by_month_', '')})")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].tick_params(axis="x", labelrotation=45)

    qty_metrics = [
        ("volume >100", overall.get("volume_diff_gt_100_rate", 0)),
        ("amount >10bp", overall.get("amount_rel_diff_gt_10bp_rate", 0)),
        ("amount >1%", overall.get("amount_rel_diff_gt_1pct_rate", 0)),
    ]
    axes[1, 1].bar([m[0] for m in qty_metrics], [float(m[1]) * 100 for m in qty_metrics], color=["#3182ce", "#6b46c1", "#c53030"])
    axes[1, 1].set_title("Volume / Amount Difference Rates")
    axes[1, 1].set_ylabel("% of paired symbol-days")
    axes[1, 1].tick_params(axis="x", labelrotation=20)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _read_manifest(root: Path, dataset_id: str, domain: str) -> DatasetManifest:
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        raise FileNotFoundError(f"dataset_manifest_missing:{domain}:{dataset_id}")
    return read_dataset_manifest(path)


def _existing_shard_paths(manifest: DatasetManifest, *, root: Path) -> list[Path]:
    return [resolve_manifest_path(shard.path, root=root) for shard in manifest.shards if resolve_manifest_path(shard.path, root=root).exists()]


def _configure(con: Any, *, memory_limit: str, threads: int) -> None:
    safe_memory = str(memory_limit).replace("'", "")
    con.execute(f"set memory_limit='{safe_memory}'")
    con.execute(f"set threads={max(1, int(threads or 1))}")
    con.execute("set preserve_insertion_order=false")


def _path_list_sql(paths: Iterable[Path]) -> str:
    return "[" + ", ".join("'" + str(path).replace("\\", "/").replace("'", "''") + "'" for path in paths) + "]"


def _one(con: Any, sql: str) -> dict[str, Any]:
    rows = _rows(con, sql)
    return rows[0] if rows else {}


def _rows(con: Any, sql: str) -> list[dict[str, Any]]:
    frame = con.execute(sql).fetchdf()
    return [dict(item) for item in frame.to_dict(orient="records")]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp audit intraday-quality", description="Explain 5m/daily cross-frequency differences and amount drift.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--focus-year", type=int, default=2026)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_intraday_quality_report(
        workspace_root=str(args.workspace_root or "") or None,
        runtime=str(args.runtime or "balanced"),
        duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
        threads=int(args.threads or 0),
        focus_year=int(args.focus_year or 2026),
        sample_limit=int(args.sample_limit or 50),
        write_chart=not bool(args.no_chart),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items() if key in {"status", "audit_path", "chart_path", "examples_path"}))
    return 0 if str(payload.get("status", "")) == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
