"""Search short-horizon limit-up event strategies under A-share T+1 rules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_EVENT_RUN_DIR = Path(
    "traditional_quant_research/output/experiments/all_limitup_event_study/"
    "all_limitup_event_study_vector_20260604_192738"
)
DEFAULT_EVENT_FILE = DEFAULT_EVENT_RUN_DIR / "all_limitup_events_tplus1_vector.csv"
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/short_limitup_strategy_search")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-04_short_limitup_strategy_search.md")

DEFAULT_SELL_WINDOWS = (1, 3, 5, 10, 20)
DEFAULT_STOP_LOSSES = (5.0, 7.0, 10.0)
DEFAULT_TARGETS = (10.0, 15.0, 20.0, 30.0)
DEFAULT_MIN_TRADES = 500

REQUIRED_EVENT_COLUMNS = [
    "date",
    "year",
    "code",
    "next_gap_bucket",
    "entry_access",
    "board_stage",
    "one_word_limit_like",
    "near_one_word_limit_like",
    "entry_limit_up",
    "entry_one_word_limit",
    "entry_day_close_ret_pct",
    "first_sell_open_ret_pct",
    "first_sell_close_ret_pct",
]


def run_short_limitup_strategy_search(
    *,
    event_file: str | Path = DEFAULT_EVENT_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sell_windows: Sequence[int] = DEFAULT_SELL_WINDOWS,
    stop_losses: Sequence[float] = DEFAULT_STOP_LOSSES,
    targets: Sequence[float] = DEFAULT_TARGETS,
    min_trades: int = DEFAULT_MIN_TRADES,
    top_event_rows: int = 5000,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run a grid search over limit-up event filters and T+2+ management rules."""

    if min_trades <= 0:
        raise ValueError("min_trades must be positive")
    event_path = Path(event_file)
    events = load_limitup_events(event_path)
    run_id = f"short_limitup_strategy_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    grid, yearly = search_limitup_strategies(
        events,
        sell_windows=sell_windows,
        stop_losses=stop_losses,
        targets=targets,
    )
    ranked = rank_strategy_grid(grid, min_trades=min_trades)
    best = ranked.head(1)
    best_events = pd.DataFrame()
    if not best.empty:
        row = best.iloc[0].to_dict()
        best_events = materialize_strategy_events(events, row).head(int(top_event_rows))

    summary = summarize_strategy_search(
        events,
        grid,
        ranked,
        yearly,
        run_id=run_id,
        run_dir=run_dir,
        event_file=event_path,
        min_trades=min_trades,
    )
    markdown = render_strategy_search_markdown(summary, ranked, yearly)

    grid.to_csv(run_dir / "strategy_grid.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(run_dir / "strategy_grid_ranked.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(run_dir / "strategy_yearly.csv", index=False, encoding="utf-8-sig")
    best_events.to_csv(run_dir / "best_strategy_events_sample.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def load_limitup_events(event_file: str | Path) -> pd.DataFrame:
    """Load the prebuilt all-limit-up T+1 event table and normalize dtypes."""

    path = Path(event_file)
    if not path.exists():
        raise FileNotFoundError(f"limit-up event file not found: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"event file missing required columns: {missing}")
    needed = set(REQUIRED_EVENT_COLUMNS)
    for window in DEFAULT_SELL_WINDOWS:
        needed.update(
            {
                f"sell{window}_close_ret_pct",
                f"sell{window}_max_high_pct",
                f"sell{window}_min_low_pct",
            }
        )
    available = [column for column in columns if column in needed]
    events = pd.read_csv(path, usecols=available)
    return normalize_limitup_events(events)


def normalize_limitup_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"events missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    for column in [
        "one_word_limit_like",
        "near_one_word_limit_like",
        "entry_limit_up",
        "entry_one_word_limit",
    ]:
        frame[column] = frame[column].astype(bool)
    numeric_columns = [
        column
        for column in frame.columns
        if column.endswith("_ret_pct") or column.endswith("_high_pct") or column.endswith("_low_pct")
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)


def search_limitup_strategies(
    events: pd.DataFrame,
    *,
    sell_windows: Sequence[int] = DEFAULT_SELL_WINDOWS,
    stop_losses: Sequence[float] = DEFAULT_STOP_LOSSES,
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate short-line setup filters and conservative T+2+ exits."""

    if events.empty:
        return pd.DataFrame(columns=_grid_columns()), pd.DataFrame(columns=_yearly_columns())
    frame = normalize_limitup_events(events)
    setup_filters = build_setup_filters(frame)
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for setup_name, setup_mask in setup_filters.items():
        selected = frame.loc[setup_mask].copy()
        if selected.empty:
            continue
        for sell_window in _parse_int_values(sell_windows, "sell_windows"):
            close_col = f"sell{sell_window}_close_ret_pct"
            high_col = f"sell{sell_window}_max_high_pct"
            low_col = f"sell{sell_window}_min_low_pct"
            if close_col not in selected.columns or high_col not in selected.columns or low_col not in selected.columns:
                continue
            base = selected.dropna(subset=[close_col, high_col, low_col]).copy()
            if base.empty:
                continue
            for stop_loss in _parse_float_values(stop_losses, "stop_losses"):
                for target in _parse_float_values(targets, "targets"):
                    returns = conservative_stop_target_return(
                        base[close_col],
                        base[high_col],
                        base[low_col],
                        stop_loss=float(stop_loss),
                        target=float(target),
                    )
                    row = summarize_returns(
                        returns,
                        setup_name=setup_name,
                        sell_window=sell_window,
                        stop_loss=float(stop_loss),
                        target=float(target),
                        source_rows=len(base),
                    )
                    rows.append(row)
                    yearly = summarize_yearly_returns(
                        base,
                        returns,
                        setup_name=setup_name,
                        sell_window=sell_window,
                        stop_loss=float(stop_loss),
                        target=float(target),
                    )
                    yearly_rows.extend(yearly)
    grid = pd.DataFrame(rows, columns=_grid_columns())
    yearly = pd.DataFrame(yearly_rows, columns=_yearly_columns())
    if not grid.empty and not yearly.empty:
        stability_columns = [
            "year_count",
            "positive_year_rate",
            "min_year_mean_ret_pct",
            "max_year_drawdown_proxy_pct",
        ]
        grid = grid.drop(columns=[column for column in stability_columns if column in grid.columns])
        stability = (
            yearly.groupby(["setup_name", "sell_window", "stop_loss", "target"], dropna=False)
            .agg(
                year_count=("year", "nunique"),
                positive_year_rate=("mean_ret_pct", lambda values: float((values > 0).mean())),
                min_year_mean_ret_pct=("mean_ret_pct", "min"),
                max_year_drawdown_proxy_pct=("p10_ret_pct", "min"),
            )
            .reset_index()
        )
        grid = grid.merge(stability, on=["setup_name", "sell_window", "stop_loss", "target"], how="left")
    return grid.reindex(columns=_grid_columns()), yearly


def build_setup_filters(events: pd.DataFrame) -> dict[str, pd.Series]:
    """Return auditable short-line candidate filters.

    Filters deliberately separate path strength from executable entry quality.
    """

    normal = events["entry_access"].eq("normal")
    near_limit = events["next_gap_bucket"].eq("near_limit_up_open")
    high_open_failed = events["next_gap_bucket"].eq("gap_ge_6") & ~events["entry_limit_up"]
    third_plus = events["board_stage"].isin(["third_board", "fourth_plus_board"])
    second_plus = events["board_stage"].isin(["second_board", "third_board", "fourth_plus_board"])
    t1_limit = events["entry_limit_up"]
    t1_not_limit = ~events["entry_limit_up"]
    signal_one_word = events["one_word_limit_like"] | events["near_one_word_limit_like"]
    low_or_flat_open = events["next_gap_bucket"].isin(["gap_le_minus3", "gap_minus3_to_0", "flat", "gap_0_to_3"])
    tradable_low_open_continue = normal & low_or_flat_open & t1_limit

    return {
        "all_limitup_events": pd.Series(True, index=events.index),
        "normal_entry": normal,
        "normal_entry_t1_limit_up": normal & t1_limit,
        "normal_entry_t1_not_limit_up": normal & t1_not_limit,
        "low_or_flat_open_t1_limit_up": tradable_low_open_continue,
        "gap_3_to_6_t1_limit_up": normal & events["next_gap_bucket"].eq("gap_3_to_6") & t1_limit,
        "gap_ge_6_t1_limit_up": normal & events["next_gap_bucket"].eq("gap_ge_6") & t1_limit,
        "gap_ge_6_failed": normal & high_open_failed,
        "second_plus_t1_limit_up": second_plus & t1_limit,
        "third_plus_t1_limit_up": third_plus & t1_limit,
        "signal_one_word": signal_one_word,
        "signal_one_word_t1_limit_up": signal_one_word & t1_limit,
        "near_limit_open": near_limit,
        "near_limit_open_signal_one_word": near_limit & signal_one_word,
        "near_limit_but_traded": events["entry_access"].eq("near_limit_but_traded"),
    }


def conservative_stop_target_return(
    close_ret_pct: pd.Series,
    max_high_pct: pd.Series,
    min_low_pct: pd.Series,
    *,
    stop_loss: float,
    target: float,
) -> pd.Series:
    """Approximate daily-bar exit return from T+2 onward.

    If stop and target are both touched in the same holding window, the stop is
    counted first. This is intentionally conservative because daily bars do not
    contain intraday order.
    """

    close_ret = pd.to_numeric(close_ret_pct, errors="coerce")
    max_high = pd.to_numeric(max_high_pct, errors="coerce")
    min_low = pd.to_numeric(min_low_pct, errors="coerce")
    output = close_ret.copy()
    stop_mask = min_low <= -abs(float(stop_loss))
    target_mask = (max_high >= float(target)) & ~stop_mask
    output.loc[stop_mask] = -abs(float(stop_loss))
    output.loc[target_mask] = float(target)
    return output


def summarize_returns(
    returns: pd.Series,
    *,
    setup_name: str,
    sell_window: int,
    stop_loss: float,
    target: float,
    source_rows: int,
) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    wins = values[values > 0]
    losses = values[values < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff = float(avg_win / abs(avg_loss)) if avg_loss < 0 else np.nan
    return {
        "setup_name": setup_name,
        "sell_window": int(sell_window),
        "stop_loss": float(stop_loss),
        "target": float(target),
        "n": int(len(values)),
        "source_rows": int(source_rows),
        "mean_ret_pct": float(values.mean()) if not values.empty else np.nan,
        "median_ret_pct": float(values.median()) if not values.empty else np.nan,
        "p10_ret_pct": float(values.quantile(0.10)) if not values.empty else np.nan,
        "p90_ret_pct": float(values.quantile(0.90)) if not values.empty else np.nan,
        "win_rate": float((values > 0).mean()) if not values.empty else np.nan,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "payoff": payoff,
        "stop_hit_rate": float((values <= -abs(float(stop_loss))).mean()) if not values.empty else np.nan,
        "target_hit_rate": float((values >= float(target)).mean()) if not values.empty else np.nan,
    }


def summarize_yearly_returns(
    events: pd.DataFrame,
    returns: pd.Series,
    *,
    setup_name: str,
    sell_window: int,
    stop_loss: float,
    target: float,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"year": events["year"].to_numpy(), "ret": returns.to_numpy()}).dropna()
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        values = pd.to_numeric(group["ret"], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "setup_name": setup_name,
                "sell_window": int(sell_window),
                "stop_loss": float(stop_loss),
                "target": float(target),
                "year": int(year),
                "n": int(len(values)),
                "mean_ret_pct": float(values.mean()),
                "median_ret_pct": float(values.median()),
                "p10_ret_pct": float(values.quantile(0.10)),
                "win_rate": float((values > 0).mean()),
            }
        )
    return rows


def rank_strategy_grid(grid: pd.DataFrame, *, min_trades: int = DEFAULT_MIN_TRADES) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame(columns=_grid_columns() + ["rank_score", "evidence_grade"])
    ranked = grid.copy()
    ranked = ranked[pd.to_numeric(ranked["n"], errors="coerce").fillna(0) >= int(min_trades)].copy()
    if ranked.empty:
        return pd.DataFrame(columns=list(grid.columns) + ["rank_score", "evidence_grade"])
    ranked["positive_year_rate"] = pd.to_numeric(ranked["positive_year_rate"], errors="coerce").fillna(0.0)
    ranked["mean_ret_pct"] = pd.to_numeric(ranked["mean_ret_pct"], errors="coerce")
    ranked["payoff"] = pd.to_numeric(ranked["payoff"], errors="coerce").fillna(0.0)
    ranked["stop_hit_rate"] = pd.to_numeric(ranked["stop_hit_rate"], errors="coerce").fillna(1.0)
    ranked["rank_score"] = (
        ranked["mean_ret_pct"]
        + 2.0 * ranked["positive_year_rate"]
        + 0.25 * ranked["payoff"]
        - 2.0 * ranked["stop_hit_rate"]
    )
    ranked["evidence_grade"] = np.where(
        ranked["positive_year_rate"].ge(0.8) & ranked["mean_ret_pct"].gt(0),
        "shortline_backtest_candidate",
        "shortline_diagnostic_only",
    )
    return ranked.sort_values(
        ["rank_score", "mean_ret_pct", "payoff", "positive_year_rate", "n"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def materialize_strategy_events(events: pd.DataFrame, strategy_row: Mapping[str, Any]) -> pd.DataFrame:
    frame = normalize_limitup_events(events)
    setup_name = str(strategy_row["setup_name"])
    filters = build_setup_filters(frame)
    if setup_name not in filters:
        raise ValueError(f"unknown setup_name: {setup_name}")
    sell_window = int(strategy_row["sell_window"])
    stop_loss = float(strategy_row["stop_loss"])
    target = float(strategy_row["target"])
    selected = frame.loc[filters[setup_name]].copy()
    close_col = f"sell{sell_window}_close_ret_pct"
    high_col = f"sell{sell_window}_max_high_pct"
    low_col = f"sell{sell_window}_min_low_pct"
    selected["managed_return_pct"] = conservative_stop_target_return(
        selected[close_col],
        selected[high_col],
        selected[low_col],
        stop_loss=stop_loss,
        target=target,
    )
    return selected.sort_values(["date", "code"]).reset_index(drop=True)


def summarize_strategy_search(
    events: pd.DataFrame,
    grid: pd.DataFrame,
    ranked: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    event_file: Path,
    min_trades: int,
) -> dict[str, Any]:
    best = ranked.iloc[0].to_dict() if not ranked.empty else {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "event_file": str(event_file),
        "event_count": int(len(events)),
        "date_min": str(pd.to_datetime(events["date"]).min().date()) if not events.empty else None,
        "date_max": str(pd.to_datetime(events["date"]).max().date()) if not events.empty else None,
        "grid_rows": int(len(grid)),
        "ranked_rows": int(len(ranked)),
        "yearly_rows": int(len(yearly)),
        "min_trades": int(min_trades),
        "best_strategy": _json_ready(best),
        "decision": "shortline_backtest_grid_ready" if not ranked.empty else "shortline_no_strategy_passed_min_trades",
        "candidate_count": int((ranked.get("evidence_grade", pd.Series(dtype=str)) == "shortline_backtest_candidate").sum())
        if not ranked.empty
        else 0,
        "tplus1_note": "signal T; buy T+1 open; managed exits start from T+2 windows",
    }


def render_strategy_search_markdown(summary: Mapping[str, Any], ranked: pd.DataFrame, yearly: pd.DataFrame) -> str:
    lines = [
        "# Short Limit-Up Strategy Search",
        "",
        "## Scope",
        "",
        f"- Event file: `{summary.get('event_file')}`",
        f"- Events: {summary.get('event_count')}",
        f"- Date range: {summary.get('date_min')} to {summary.get('date_max')}",
        "- Execution: signal on T, buy at T+1 open, earliest managed sell window starts on T+2.",
        "- Daily-bar stop/target handling is conservative: if stop and target both occur in a window, stop wins.",
        "",
        "## Decision",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Candidate count: {summary.get('candidate_count')}",
        f"- Ranked rows: {summary.get('ranked_rows')}",
        "",
    ]
    best = summary.get("best_strategy") or {}
    if best:
        lines.extend(
            [
                "## Best Ranked Strategy",
                "",
                f"- Setup: `{best.get('setup_name')}`",
                f"- Sell window: T+2 through T+{int(best.get('sell_window', 0)) + 1} close/high/low window",
                f"- Stop/target: {best.get('stop_loss')}% / {best.get('target')}%",
                f"- Trades: {best.get('n')}",
                f"- Mean return: {_fmt_pct(best.get('mean_ret_pct'))}",
                f"- Median return: {_fmt_pct(best.get('median_ret_pct'))}",
                f"- Win rate: {_fmt_rate(best.get('win_rate'))}",
                f"- Payoff: {_fmt_num(best.get('payoff'))}",
                f"- Positive year rate: {_fmt_rate(best.get('positive_year_rate'))}",
                f"- Evidence grade: `{best.get('evidence_grade')}`",
                "",
            ]
        )
    lines.extend(["## Top Strategies", ""])
    if ranked.empty:
        lines.append("No ranked strategy met the minimum trade threshold.")
    else:
        cols = [
            "setup_name",
            "sell_window",
            "stop_loss",
            "target",
            "n",
            "mean_ret_pct",
            "median_ret_pct",
            "win_rate",
            "payoff",
            "positive_year_rate",
            "evidence_grade",
        ]
        lines.append(_markdown_table(ranked.head(12), cols))
    if not yearly.empty and best:
        lines.extend(["", "## Best Strategy Yearly Detail", ""])
        mask = (
            yearly["setup_name"].astype(str).eq(str(best.get("setup_name")))
            & yearly["sell_window"].eq(int(best.get("sell_window")))
            & yearly["stop_loss"].eq(float(best.get("stop_loss")))
            & yearly["target"].eq(float(best.get("target")))
        )
        lines.append(_markdown_table(yearly.loc[mask].sort_values("year"), ["year", "n", "mean_ret_pct", "median_ret_pct", "win_rate"]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This search is a short-line backtest layer, not a live trading guarantee.",
            "- One-word and near-limit-open paths are high-convexity but may be unfilled in reality.",
            "- Tradable candidates should be judged separately from path-strength diagnostics.",
            "- The next upgrade should add minute-level open-board/reseal and orderability evidence.",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No rows."
    output = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in output.columns:
        if column.endswith("_pct"):
            output[column] = output[column].map(_fmt_pct)
        elif column.endswith("_rate"):
            output[column] = output[column].map(_fmt_rate)
        elif column == "payoff":
            output[column] = output[column].map(_fmt_num)
    return output.to_markdown(index=False)


def _fmt_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_rate(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _fmt_num(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _parse_int_values(values: Sequence[int] | str, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        parsed = tuple(int(value.strip()) for value in values.split(",") if value.strip())
    else:
        parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _parse_float_values(values: Sequence[float] | str, name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    else:
        parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _grid_columns() -> list[str]:
    return [
        "setup_name",
        "sell_window",
        "stop_loss",
        "target",
        "n",
        "source_rows",
        "mean_ret_pct",
        "median_ret_pct",
        "p10_ret_pct",
        "p90_ret_pct",
        "win_rate",
        "avg_win_pct",
        "avg_loss_pct",
        "payoff",
        "stop_hit_rate",
        "target_hit_rate",
        "year_count",
        "positive_year_rate",
        "min_year_mean_ret_pct",
        "max_year_drawdown_proxy_pct",
    ]


def _yearly_columns() -> list[str]:
    return [
        "setup_name",
        "sell_window",
        "stop_loss",
        "target",
        "year",
        "n",
        "mean_ret_pct",
        "median_ret_pct",
        "p10_ret_pct",
        "win_rate",
    ]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", default=str(DEFAULT_EVENT_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sell-windows", default=",".join(str(value) for value in DEFAULT_SELL_WINDOWS))
    parser.add_argument("--stop-losses", default=",".join(str(value) for value in DEFAULT_STOP_LOSSES))
    parser.add_argument("--targets", default=",".join(str(value) for value in DEFAULT_TARGETS))
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--top-event-rows", type=int, default=5000)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_short_limitup_strategy_search(
        event_file=args.event_file,
        output_dir=args.output_dir,
        sell_windows=_parse_int_values(args.sell_windows, "sell_windows"),
        stop_losses=_parse_float_values(args.stop_losses, "stop_losses"),
        targets=_parse_float_values(args.targets, "targets"),
        min_trades=args.min_trades,
        top_event_rows=args.top_event_rows,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
