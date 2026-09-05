"""Summarize completed minute-strategy portfolio replays.

This tool only reads finished portfolio artifacts.  It does not rebuild minute
signals or touch the interrupted raw-data generation job.  The output keeps
every account combination and adds bounded summaries for trades, equity,
seed stability, and the two non-identical sample windows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

KEY_COLUMNS = ("selection_ranker", "selection_seed", "strategy_id", "policy_id")
RANKERS = ("random_hash", "sector_leader", "trend_structure", "flow_quality")
SEEDS = (0, 1, 2, 3, 4)
POLICIES = (
    "next_open_1d",
    "next_open_2d",
    "next_open_3d",
    "next_open_5d",
    "protect_3pct_target6pct_3d",
    "momentum_trail_4pct_5d",
    "fast_failure_2pct_target4pct_2d",
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _flatten_results(
    rows: list[dict[str, Any]],
    *,
    sample_group: str,
    spec: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    frame.insert(0, "sample_group", sample_group)
    frame.insert(1, "sample_variant", spec["sample_variant"])
    frame.insert(2, "source_signal_root", spec["source_signal_root"])
    exit_keys: set[str] = set()
    for value in frame.get("exit_reason_counts", pd.Series(dtype=object)).dropna():
        if isinstance(value, dict):
            exit_keys.update(str(key) for key in value)
    for key in sorted(exit_keys):
        name = f"exit_reason_{_slug(key)}_count"
        frame[name] = [
            int(value.get(key, 0)) if isinstance(value, dict) else 0
            for value in frame["exit_reason_counts"]
        ]
    if "exit_reason_counts" in frame:
        frame = frame.drop(columns=["exit_reason_counts"])
    frame["fully_resolved"] = (
        frame["unresolved_position_count"].fillna(0).eq(0)
        & frame["unresolved_exit_count"].fillna(0).eq(0)
    )
    frame["positive_return"] = frame["net_return"].gt(0)
    return frame


def _flatten_annual(
    rows: list[dict[str, Any]], *, sample_group: str, sample_variant: str
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for row in rows:
        for annual in row.get("annual", []) or []:
            output.append(
                {
                    "sample_group": sample_group,
                    "sample_variant": sample_variant,
                    "selection_ranker": row.get("selection_ranker"),
                    "selection_seed": row.get("selection_seed"),
                    "strategy_id": row.get("strategy_id"),
                    "policy_id": row.get("policy_id"),
                    "signal_year": annual.get("year"),
                    "net_return": annual.get("net_return"),
                    "maximum_drawdown": annual.get("maximum_drawdown"),
                }
            )
    return pd.DataFrame(output)


def _flatten_seed_summary(
    payload: dict[str, Any], *, sample_group: str, sample_variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for account in payload.get("accounts", []):
        row = {
            "sample_group": sample_group,
            "sample_variant": sample_variant,
            "selection_ranker": account.get("selection_ranker"),
            "strategy_id": account.get("strategy_id"),
            "policy_id": account.get("policy_id"),
            "seed_count": account.get("seed_count"),
            "resolved_seed_count": account.get("resolved_seed_count"),
            "positive_seed_count": account.get("positive_seed_count"),
            "positive_seed_fraction": account.get("positive_seed_fraction"),
            "resolved_positive_seed_count": account.get("resolved_positive_seed_count"),
            "resolved_positive_seed_fraction": account.get("resolved_positive_seed_fraction"),
            "all_periods_positive_seed_count": account.get("all_periods_positive_seed_count"),
            "resolved_all_periods_positive_seed_count": account.get(
                "resolved_all_periods_positive_seed_count"
            ),
        }
        for prefix, key in (
            ("combined_net_return", "combined_net_return"),
            ("maximum_drawdown", "maximum_drawdown"),
        ):
            distribution = account.get(key, {}) or {}
            for stat in ("mean", "median", "standard_deviation", "minimum", "q25", "q75", "maximum"):
                row[f"{prefix}_{stat}"] = distribution.get(stat)
        rows.append(row)
        for annual in account.get("annual", []) or []:
            annual_row = {
                "sample_group": sample_group,
                "sample_variant": sample_variant,
                "selection_ranker": account.get("selection_ranker"),
                "strategy_id": account.get("strategy_id"),
                "policy_id": account.get("policy_id"),
                "signal_year": annual.get("year"),
                "observed_seed_count": annual.get("observed_seed_count"),
                "positive_seed_count": annual.get("positive_seed_count"),
                "positive_seed_fraction": annual.get("positive_seed_fraction"),
            }
            distribution = annual.get("net_return", {}) or {}
            for stat in ("mean", "median", "standard_deviation", "minimum", "q25", "q75", "maximum"):
                annual_row[f"net_return_{stat}"] = distribution.get(stat)
            annual_rows.append(annual_row)
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def _group_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        net = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        drawdown = pd.to_numeric(group["maximum_drawdown"], errors="coerce").dropna()
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "account_count": int(len(group)),
                "positive_count": int(group["positive_return"].sum()),
                "positive_fraction": float(group["positive_return"].mean()),
                "fully_resolved_count": int(group["fully_resolved"].sum()),
                "fully_resolved_fraction": float(group["fully_resolved"].mean()),
                "mean_net_return": float(net.mean()) if len(net) else None,
                "median_net_return": float(net.median()) if len(net) else None,
                "std_net_return": float(net.std(ddof=0)) if len(net) else None,
                "minimum_net_return": float(net.min()) if len(net) else None,
                "maximum_net_return": float(net.max()) if len(net) else None,
                "mean_maximum_drawdown": float(drawdown.mean()) if len(drawdown) else None,
                "median_maximum_drawdown": float(drawdown.median()) if len(drawdown) else None,
                "worst_maximum_drawdown": float(drawdown.min()) if len(drawdown) else None,
                "mean_closed_trade_count": float(group["closed_trade_count"].mean()),
                "total_closed_trade_count": int(group["closed_trade_count"].sum()),
                "total_delayed_exit_count": int(group["delayed_exit_count"].sum()),
                "total_unresolved_position_count": int(group["unresolved_position_count"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _trade_frames(
    con: duckdb.DuckDBPyConnection,
    *,
    root: Path,
    sample_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = str(root / "trades.parquet")
    base = ["selection_ranker", "selection_seed", "strategy_id", "policy_id"]
    summary = con.execute(
        f"""
        SELECT ? AS sample_group, {", ".join(base)},
               COUNT(*) AS trade_row_count,
               COUNT(DISTINCT signal_id) AS distinct_signal_count,
               COUNT(DISTINCT symbol) AS distinct_symbol_count,
               SUM(gross_entry_notional) AS total_entry_notional,
               SUM(buy_cost) AS total_buy_cost,
               SUM(sell_cost) AS total_sell_cost,
               SUM(buy_cost + sell_cost) AS total_cost,
               AVG((buy_cost + sell_cost) / NULLIF(gross_entry_notional, 0)) AS mean_cost_rate,
               SUM(pnl) AS total_pnl,
               SUM(proceeds) AS total_proceeds,
               AVG(pnl) AS mean_pnl,
               AVG(trade_net_return) AS mean_trade_net_return,
               MEDIAN(trade_net_return) AS median_trade_net_return,
               AVG(CASE WHEN trade_net_return > 0 THEN 1.0 ELSE 0.0 END) AS trade_win_rate,
               AVG(holding_trading_days) AS mean_holding_trading_days,
               MAX(holding_trading_days) AS max_holding_trading_days,
               SUM(exit_attempt_count) AS total_exit_attempt_count
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY ALL
        """,
        [sample_group, trade_path],
    ).fetchdf()
    year = con.execute(
        f"""
        SELECT ? AS sample_group, {", ".join(base)},
               EXTRACT(year FROM TRY_CAST(entry_date AS DATE))::INTEGER AS entry_year,
               COUNT(*) AS trade_row_count,
               SUM(pnl) AS total_pnl,
               AVG(trade_net_return) AS mean_trade_net_return,
               AVG(CASE WHEN trade_net_return > 0 THEN 1.0 ELSE 0.0 END) AS trade_win_rate,
               AVG(holding_trading_days) AS mean_holding_trading_days
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY ALL
        """,
        [sample_group, trade_path],
    ).fetchdf()
    exits = con.execute(
        f"""
        SELECT ? AS sample_group, {", ".join(base)}, exit_reason,
               COUNT(*) AS exit_count,
               SUM(pnl) AS total_pnl,
               AVG(trade_net_return) AS mean_trade_net_return
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY ALL
        """,
        [sample_group, trade_path],
    ).fetchdf()
    equity_path = str(root / "equity.parquet")
    equity = con.execute(
        f"""
        SELECT ? AS sample_group, {", ".join(base)},
               COUNT(*) AS equity_row_count,
               MIN(trade_date) AS equity_start_date,
               MAX(trade_date) AS equity_end_date,
               MIN(equity) AS minimum_marked_equity,
               MAX(equity) AS maximum_marked_equity,
               AVG(daily_return) AS mean_daily_return,
               STDDEV_POP(daily_return) AS daily_return_std,
               CASE WHEN STDDEV_POP(daily_return) > 0
                    THEN SQRT(252.0) * AVG(daily_return) / STDDEV_POP(daily_return)
                    ELSE NULL END AS daily_sharpe,
               EXP(
                   SUM(LN(GREATEST(1.0 + daily_return, 1.0e-12)))
                   * 252.0 / NULLIF(COUNT(*), 0)
               ) - 1.0 AS annualized_geometric_return,
               MIN(daily_return) AS minimum_daily_return,
               MAX(daily_return) AS maximum_daily_return,
               AVG(position_count) AS mean_position_count,
               SUM(CASE WHEN position_count > 0 THEN 1 ELSE 0 END) * 1.0
                   / NULLIF(COUNT(*), 0) AS invested_day_fraction,
               MAX(position_count) AS maximum_position_count
        FROM read_parquet(?)
        GROUP BY ALL
        ORDER BY ALL
        """,
        [sample_group, equity_path],
    ).fetchdf()
    return summary, year, exits, equity


def _load_spec(workspace_root: Path, name: str, sample_group: str, variant: str, source_signal_root: str, guard_name: str) -> dict[str, Any]:
    root = workspace_root / "runs" / name
    return {
        "root": root,
        "sample_group": sample_group,
        "sample_variant": variant,
        "source_signal_root": source_signal_root,
        "guard_log": workspace_root / "tmp" / guard_name,
    }


def _signal_dates(signal_root: Path) -> list[str]:
    return sorted(
        path.parent.name.split("=", 1)[-1]
        for path in (signal_root / "outcomes").glob("date=*/outcomes.parquet")
    )


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _write_report(
    output: Path,
    *,
    integrity: list[dict[str, Any]],
    account_results: pd.DataFrame,
    ranker_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    cross_sample: pd.DataFrame,
) -> None:
    lines = [
        "# Existing minute-strategy portfolio research",
        "",
        "This report summarizes completed account replays over already generated samples.",
        "It is descriptive research evidence; it does not freeze a live strategy or treat the best grid cell as validation.",
        "The companion event-layer files in this record cover every canonical outcome row, six MA periods, eight forward horizons, controls, regimes, features, tails, costs and author-hypothesis mappings.",
        "",
        "## Coverage and integrity",
        "",
        "| Sample | Signal dates | Signal window | Outcome window | Strategy count | Account results | Lowest available RAM |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for item in integrity:
        lines.append(
            f"| {item['sample_group']} | {item['signal_date_count']} | {item['signal_date_start']} to {item['signal_date_end']} | "
            f"{item.get('outcome_calendar_start', 'n/a')} to {item.get('outcome_calendar_end', 'n/a')} | "
            f"{item['strategy_count']} | {item['actual_account_result_count']} | {item.get('lowest_available_gb', 'n/a')} GiB |"
        )
    lines += [
        "",
        "The primary sample is the current-contract partial development run ending at signal date 2023-05-18; its later calendar end reflects forward outcome availability. The supplemental sample is a separate 20-day September 2023 legacy-VWAP run. They are never pooled as one time series.",
        "",
        "## Account-grid result",
        "",
        "| Sample | Accounts | Positive | Positive fraction | Fully resolved | Mean return | Median return | Worst drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample_group, group in account_results.groupby("sample_group", sort=True):
        lines.append(
            f"| {sample_group} | {len(group)} | {int(group['positive_return'].sum())} | "
            f"{group['positive_return'].mean():.2%} | {int(group['fully_resolved'].sum())} | "
            f"{_format_pct(group['net_return'].mean())} | {_format_pct(group['net_return'].median())} | "
            f"{_format_pct(group['maximum_drawdown'].min())} |"
        )
    lines += [
        "",
        "The full account result table is `account_results.csv`; it contains every strategy, exit policy, ranker and seed combination, including losing and unresolved cases.",
        "",
        "The primary window has many holdings that could not be legally closed before the available bars ended or remained locked. Therefore `account_resolved_results.csv` is the cleanly closed subset, while `account_results.csv` keeps marked-to-market unresolved positions. Neither view is silently substituted for the other.",
        "",
        "| Sample | Fully resolved accounts | Resolved positive fraction | Resolved mean return | Resolved median return | Resolved worst drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sample_group, group in account_results.groupby("sample_group", sort=True):
        resolved = group.loc[group["fully_resolved"]]
        lines.append(
            f"| {sample_group} | {len(resolved)} | {resolved['positive_return'].mean():.2%} | "
            f"{_format_pct(resolved['net_return'].mean())} | {_format_pct(resolved['net_return'].median())} | "
            f"{_format_pct(resolved['maximum_drawdown'].min())} |"
        )
    lines += [
        "",
        "## Execution and risk diagnostics",
        "",
        "The trade and equity summaries expose cost drag, turnover, time in market, daily volatility and unresolved/blocked exits. The portfolio assumptions use 3 bps commission, 0.1 bps transfer fee, 7 bps slippage and the declared stamp-tax schedule; these costs are not omitted from net returns.",
        "",
        "## Ranker and exit summaries",
        "",
        "| Sample | Ranker | Policy | Accounts | Positive fraction | Mean return | Median return | Mean drawdown |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    ranked = ranker_summary.sort_values(
        ["sample_group", "mean_net_return"], ascending=[True, False]
    )
    for row in ranked.itertuples(index=False):
        lines.append(
            f"| {row.sample_group} | {row.selection_ranker} | {row.policy_id} | {row.account_count} | "
            f"{row.positive_fraction:.2%} | {_format_pct(row.mean_net_return)} | "
            f"{_format_pct(row.median_net_return)} | {_format_pct(row.mean_maximum_drawdown)} |"
        )
    lines += [
        "",
        "## Best cells (descriptive only)",
        "",
    ]
    for sample_group, group in account_results.groupby("sample_group", sort=True):
        lines.append(f"### {sample_group}")
        lines.append("")
        lines.append("| Ranker | Seed | Strategy | Policy | Net return | Max drawdown | Closed trades | Resolved |")
        lines.append("|---|---:|---|---|---:|---:|---:|---|")
        for row in group.sort_values("net_return", ascending=False).head(10).itertuples(index=False):
            lines.append(
                f"| {row.selection_ranker} | {row.selection_seed} | {row.strategy_id} | {row.policy_id} | "
                f"{_format_pct(row.net_return)} | {_format_pct(row.maximum_drawdown)} | {row.closed_trade_count} | "
                f"{'yes' if row.fully_resolved else 'no'} |"
            )
        lines.append("")
    lines += [
        "These cells are not selected as deployable candidates. They are the extrema of a large comparison grid and therefore carry selection bias, especially in the short supplemental window.",
        "",
        "## Cross-sample comparison",
        "",
        f"The shared-key comparison contains {len(cross_sample)} rows for the 22 strategies present in both samples. It is a descriptive stability check, not an out-of-sample test: the primary and supplemental windows use different sample lengths and the supplemental run uses the legacy VWAP revision.",
        "",
        "See `account_seed_stability.csv`, `account_trade_summary.csv`, `account_trade_year.csv`, `account_exit_reason.csv`, `account_equity_summary.csv`, and `account_cross_sample.csv` for the complete machine-readable detail. The event-layer and combined conclusions are in `research_summary.md`.",
        "",
        "## Boundaries",
        "",
        "- No ungenerated 2023-05-19 onward or 2024 raw minute events were reconstructed.",
        "- No 2025 data was read.",
        "- Duplicate April 2022 and probe directories remain audit-only and are excluded from the canonical event sample.",
        "- Returns include the portfolio runner's declared fees, slippage, cash limits, five-position cap, T+1 and legal-exit handling.",
    ]
    (output / "portfolio_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    specs = [
        _load_spec(
            workspace_root,
            "minute_strategy_portfolio_existing_primary_331",
            "primary_development_2022_2023_partial",
            "current_contract",
            "runs/minute_ma_development_2022_2024",
            "minute_existing_primary_portfolio_memory_guard.json",
        ),
        _load_spec(
            workspace_root,
            "minute_strategy_portfolio_existing_supplemental_2023_09",
            "supplemental_2023_09_legacy",
            "legacy_vwap",
            "runs/minute_ma_month_2023_09",
            "minute_existing_supplemental_portfolio_memory_guard.json",
        ),
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: list[pd.DataFrame] = []
    all_annual: list[pd.DataFrame] = []
    all_seed: list[pd.DataFrame] = []
    all_seed_annual: list[pd.DataFrame] = []
    all_trade: list[pd.DataFrame] = []
    all_trade_year: list[pd.DataFrame] = []
    all_exits: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    integrity: list[dict[str, Any]] = []
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='512MB'")
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA preserve_insertion_order=false")
    temp_directory = workspace_root / "tmp" / "minute_existing_portfolio_summary_duckdb"
    temp_directory.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{str(temp_directory).replace(chr(39), chr(39) * 2)}'")
    try:
        for spec in specs:
            root = spec["root"]
            manifest_path = root / "run_manifest.json"
            summary_path = root / "portfolio_summary.json"
            seed_path = root / "selection_seed_summary.json"
            if not manifest_path.exists() or not summary_path.exists() or not seed_path.exists():
                raise FileNotFoundError(f"portfolio_artifacts_missing:{root}")
            manifest = _read_json(manifest_path)
            raw_rows = _read_json(summary_path)
            if manifest.get("status") != "ok" or not isinstance(raw_rows, list):
                raise RuntimeError(f"portfolio_artifact_not_complete:{root}")
            strategy_ids = list(manifest.get("strategy_ids", []))
            signal_dates = _signal_dates(workspace_root / spec["source_signal_root"])
            expected = set(
                itertools.product(
                    manifest.get("selection_rankers", list(RANKERS)),
                    manifest.get("selection_seeds", list(SEEDS)),
                    strategy_ids,
                    [item["policy_id"] for item in manifest.get("policies", [])],
                )
            )
            actual = {
                tuple(row.get(column) for column in KEY_COLUMNS)
                for row in raw_rows
            }
            duplicate_count = int(
                pd.DataFrame(raw_rows).duplicated(list(KEY_COLUMNS)).sum()
            )
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            frame = _flatten_results(
                raw_rows,
                sample_group=spec["sample_group"],
                spec=spec,
            )
            all_results.append(frame)
            all_annual.append(
                _flatten_annual(
                    raw_rows,
                    sample_group=spec["sample_group"],
                    sample_variant=spec["sample_variant"],
                )
            )
            seed, seed_annual = _flatten_seed_summary(
                _read_json(seed_path),
                sample_group=spec["sample_group"],
                sample_variant=spec["sample_variant"],
            )
            all_seed.append(seed)
            all_seed_annual.append(seed_annual)
            trade, trade_year, exits, equity = _trade_frames(
                con, root=root, sample_group=spec["sample_group"]
            )
            all_trade.append(trade)
            all_trade_year.append(trade_year)
            all_exits.append(exits)
            all_equity.append(equity)
            guard = _read_json(spec["guard_log"]) if spec["guard_log"].exists() else {}
            integrity.append(
                {
                    "sample_group": spec["sample_group"],
                    "sample_variant": spec["sample_variant"],
                    "source_signal_root": spec["source_signal_root"],
                    "run_root": str(root),
                    "manifest_status": manifest.get("status"),
                    "signal_date_count": len(signal_dates),
                    "signal_date_start": signal_dates[0] if signal_dates else None,
                    "signal_date_end": signal_dates[-1] if signal_dates else None,
                    "outcome_calendar_start": manifest.get("calendar_start"),
                    "outcome_calendar_end": manifest.get("calendar_end"),
                    "strategy_count": len(strategy_ids),
                    "policy_count": len(manifest.get("policies", [])),
                    "ranker_count": len(manifest.get("selection_rankers", [])),
                    "seed_count": len(manifest.get("selection_seeds", [])),
                    "expected_account_result_count": len(expected),
                    "actual_account_result_count": len(raw_rows),
                    "duplicate_account_key_count": duplicate_count,
                    "missing_account_key_count": len(missing),
                    "unexpected_account_key_count": len(unexpected),
                    "missing_account_key_examples": [list(item) for item in missing[:10]],
                    "unexpected_account_key_examples": [list(item) for item in unexpected[:10]],
                    "trade_row_count": int(trade["trade_row_count"].sum()) if not trade.empty else 0,
                    "equity_row_count": int(equity["equity_row_count"].sum()) if not equity.empty else 0,
                    "lowest_available_gb": guard.get("lowest_available_gb"),
                    "memory_guard_status": guard.get("status"),
                    "memory_guard_exit_code": guard.get("exit_code"),
                    "elapsed_seconds": manifest.get("elapsed_seconds"),
                }
            )
    finally:
        con.close()

    account_results = pd.concat(all_results, ignore_index=True)
    account_annual = pd.concat([frame for frame in all_annual if not frame.empty], ignore_index=True)
    account_seed = pd.concat(all_seed, ignore_index=True)
    account_seed_annual = pd.concat(
        [frame for frame in all_seed_annual if not frame.empty], ignore_index=True
    )
    account_trade = pd.concat(all_trade, ignore_index=True)
    account_trade_year = pd.concat(all_trade_year, ignore_index=True)
    account_exits = pd.concat(all_exits, ignore_index=True)
    account_equity = pd.concat(all_equity, ignore_index=True)
    account_results = account_results.sort_values(
        ["sample_group", *KEY_COLUMNS], kind="stable"
    ).reset_index(drop=True)
    resolved_results = account_results.loc[account_results["fully_resolved"]].copy()
    unresolved_results = account_results.loc[~account_results["fully_resolved"]].copy()
    for frame, name in (
        (account_results, "account_results.csv"),
        (resolved_results, "account_resolved_results.csv"),
        (unresolved_results, "account_unresolved_results.csv"),
        (account_annual, "account_annual.csv"),
        (account_seed, "account_seed_stability.csv"),
        (account_seed_annual, "account_seed_annual.csv"),
        (account_trade, "account_trade_summary.csv"),
        (account_trade_year, "account_trade_year.csv"),
        (account_exits, "account_exit_reason.csv"),
        (account_equity, "account_equity_summary.csv"),
    ):
        _csv_write(output_root / name, frame)

    ranker_summary = _group_summary(
        account_results, ["sample_group", "selection_ranker", "policy_id"]
    )
    policy_summary = _group_summary(account_results, ["sample_group", "policy_id"])
    strategy_summary = _group_summary(account_results, ["sample_group", "strategy_id"])
    ranker_only_summary = _group_summary(account_results, ["sample_group", "selection_ranker"])
    resolved_ranker_policy_summary = _group_summary(
        resolved_results, ["sample_group", "selection_ranker", "policy_id"]
    )
    resolved_policy_summary = _group_summary(resolved_results, ["sample_group", "policy_id"])
    resolved_strategy_summary = _group_summary(resolved_results, ["sample_group", "strategy_id"])
    unresolved_summary = _group_summary(
        unresolved_results, ["sample_group", "selection_ranker", "policy_id"]
    )
    cross_keys = ["selection_ranker", "selection_seed", "strategy_id", "policy_id"]
    primary = account_results.loc[
        account_results["sample_group"].eq("primary_development_2022_2023_partial"),
        cross_keys + ["net_return", "maximum_drawdown", "fully_resolved"],
    ]
    supplemental = account_results.loc[
        account_results["sample_group"].eq("supplemental_2023_09_legacy"),
        cross_keys + ["net_return", "maximum_drawdown", "fully_resolved"],
    ]
    cross_sample = primary.merge(
        supplemental,
        on=cross_keys,
        how="inner",
        suffixes=("_primary", "_supplemental"),
        validate="one_to_one",
    )
    if not cross_sample.empty:
        cross_sample["same_return_sign"] = np.sign(
            cross_sample["net_return_primary"]
        ).eq(np.sign(cross_sample["net_return_supplemental"]))
        cross_sample["return_difference_primary_minus_supplemental"] = (
            cross_sample["net_return_primary"] - cross_sample["net_return_supplemental"]
        )
    cross_summary = _group_summary(
        account_results, ["sample_group", "strategy_id"]
    )
    # The cross summary is intentionally kept separate from single-sample
    # account summaries; the source windows are not one continuous series.
    if not cross_sample.empty:
        cross_summary = (
            cross_sample.groupby(["strategy_id", "selection_ranker", "policy_id"], sort=True)
            .agg(
                shared_account_count=("strategy_id", "size"),
                same_return_sign_fraction=("same_return_sign", "mean"),
                primary_mean_return=("net_return_primary", "mean"),
                supplemental_mean_return=("net_return_supplemental", "mean"),
                primary_median_return=("net_return_primary", "median"),
                supplemental_median_return=("net_return_supplemental", "median"),
            )
            .reset_index()
        )
    _csv_write(output_root / "account_ranker_policy_summary.csv", ranker_summary)
    _csv_write(output_root / "account_policy_summary.csv", policy_summary)
    _csv_write(output_root / "account_strategy_summary.csv", strategy_summary)
    _csv_write(output_root / "account_ranker_summary.csv", ranker_only_summary)
    _csv_write(
        output_root / "account_resolved_ranker_policy_summary.csv",
        resolved_ranker_policy_summary,
    )
    _csv_write(output_root / "account_resolved_policy_summary.csv", resolved_policy_summary)
    _csv_write(output_root / "account_resolved_strategy_summary.csv", resolved_strategy_summary)
    _csv_write(output_root / "account_unresolved_summary.csv", unresolved_summary)
    _csv_write(output_root / "account_cross_sample.csv", cross_sample)
    _csv_write(output_root / "account_cross_sample_summary.csv", cross_summary)
    _json_dump(output_root / "account_integrity.json", integrity)
    summary = {
        "schema": "quantlab.minute_strategy_existing_portfolio_research/1",
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "sample_count": len(integrity),
        "account_result_count": int(len(account_results)),
        "resolved_account_result_count": int(len(resolved_results)),
        "unresolved_account_result_count": int(len(unresolved_results)),
        "account_seed_variant_count": int(len(account_seed)),
        "trade_summary_rows": int(len(account_trade)),
        "trade_year_rows": int(len(account_trade_year)),
        "exit_reason_rows": int(len(account_exits)),
        "equity_summary_rows": int(len(account_equity)),
        "cross_sample_rows": int(len(cross_sample)),
        "integrity": integrity,
        "files": {
            "account_results": "account_results.csv",
            "account_resolved_results": "account_resolved_results.csv",
            "account_unresolved_results": "account_unresolved_results.csv",
            "account_annual": "account_annual.csv",
            "account_seed_stability": "account_seed_stability.csv",
            "account_seed_annual": "account_seed_annual.csv",
            "account_trade_summary": "account_trade_summary.csv",
            "account_trade_year": "account_trade_year.csv",
            "account_exit_reason": "account_exit_reason.csv",
            "account_equity_summary": "account_equity_summary.csv",
            "account_ranker_policy_summary": "account_ranker_policy_summary.csv",
            "account_policy_summary": "account_policy_summary.csv",
            "account_strategy_summary": "account_strategy_summary.csv",
            "account_ranker_summary": "account_ranker_summary.csv",
            "account_resolved_ranker_policy_summary": "account_resolved_ranker_policy_summary.csv",
            "account_resolved_policy_summary": "account_resolved_policy_summary.csv",
            "account_resolved_strategy_summary": "account_resolved_strategy_summary.csv",
            "account_unresolved_summary": "account_unresolved_summary.csv",
            "account_cross_sample": "account_cross_sample.csv",
            "account_cross_sample_summary": "account_cross_sample_summary.csv",
            "account_integrity": "account_integrity.json",
            "portfolio_research": "portfolio_research.md",
        },
    }
    _json_dump(output_root / "account_research_result.json", summary)
    _write_report(
        output_root,
        integrity=integrity,
        account_results=account_results,
        ranker_summary=ranker_summary,
        policy_summary=policy_summary,
        strategy_summary=strategy_summary,
        cross_sample=cross_sample,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--output-root",
        default="research/records/minute_strategy_existing_samples_v1",
    )
    args = parser.parse_args()
    result = run(Path(args.workspace_root).resolve(), Path(args.output_root).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
