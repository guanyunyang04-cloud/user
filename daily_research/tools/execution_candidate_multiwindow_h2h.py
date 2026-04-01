from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from daily_research.baseline.backtest import _annualized_vol, _max_drawdown


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a multi-window formal head-to-head for execution candidates from existing equity curves."
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--bridge-start", default="2025-03-18")
    parser.add_argument("--weak-start", default="2025-09-05")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.2%}"


def _num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def _sanitize_name(name: str) -> str:
    out = []
    for char in str(name):
        if char.isalnum() or char in {"_", "-"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def _json_default(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _resolve_output_dir(explicit: str | None, label_a: str, label_b: str) -> Path:
    if explicit:
        out = Path(explicit)
        return out if out.is_absolute() else (WORKSPACE_ROOT / out).resolve()
    prefix = f"execution_candidate_multiwindow_h2h_{_sanitize_name(label_a)}_vs_{_sanitize_name(label_b)}_{datetime.now():%Y%m%d}_r"
    existing = sorted(OUTPUT_ROOT.glob(f"{prefix}*"))
    next_index = 1
    for item in existing:
        suffix = item.name.replace(prefix, "", 1)
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    return OUTPUT_ROOT / f"{prefix}{next_index}"


def _annualized_return_from_total(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or np.isnan(total_return):
        return float("nan")
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return equity, metrics


def _segment_metrics(equity_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, Any]:
    segment = equity_df.loc[(equity_df["date"] >= start_date) & (equity_df["date"] <= end_date)].copy()
    if segment.empty:
        raise ValueError(f"Empty segment for {start_date} -> {end_date}")
    start_idx = int(segment.index[0])
    end_idx = int(segment.index[-1])
    prev_idx = start_idx - 1 if start_idx > 0 else start_idx
    path = equity_df.iloc[prev_idx : end_idx + 1].copy()

    portfolio_path = path["portfolio_equity"].astype(float) / float(path["portfolio_equity"].iloc[0])
    benchmark_path = path["benchmark_equity"].astype(float) / float(path["benchmark_equity"].iloc[0])
    excess_path = path["excess_equity"].astype(float) / float(path["excess_equity"].iloc[0])

    portfolio_returns = segment["portfolio_return"].astype(float)
    benchmark_returns = segment["benchmark_return"].astype(float)
    excess_returns = segment["excess_return"].astype(float)

    total_return = float(portfolio_path.iloc[-1] - 1.0)
    benchmark_total_return = float(benchmark_path.iloc[-1] - 1.0)
    excess_total_return = float(excess_path.iloc[-1] - 1.0)
    annual_return = _annualized_return_from_total(total_return, len(segment))
    benchmark_annual_return = _annualized_return_from_total(benchmark_total_return, len(segment))
    excess_annual_return = _annualized_return_from_total(excess_total_return, len(segment))
    annual_vol = _annualized_vol(portfolio_returns)
    excess_annual_vol = _annualized_vol(excess_returns)

    year_set = set(int(y) for y in segment["date"].dt.year.unique())
    partial_year = len(year_set) == 1 and (
        segment["date"].iloc[0].month > 1 or segment["date"].iloc[-1].month < 12
    )

    return {
        "start_date": str(segment["date"].iloc[0].date()),
        "end_date": str(segment["date"].iloc[-1].date()),
        "trading_days": int(len(segment)),
        "total_return": total_return,
        "annual_return": annual_return,
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annual_return": benchmark_annual_return,
        "excess_total_return": excess_total_return,
        "excess_annual_return": excess_annual_return,
        "excess_sharpe": float(excess_annual_return / excess_annual_vol) if excess_annual_vol > 0 else float("nan"),
        "max_drawdown": float(_max_drawdown(portfolio_path)),
        "excess_max_drawdown": float(_max_drawdown(excess_path)),
        "avg_holding_count": float(segment["holding_count"].mean()),
        "avg_turnover": float(segment["turnover"].mean()),
        "regime_active_ratio": float(segment["regime_on"].fillna(False).astype(bool).mean()) if "regime_on" in segment.columns else float("nan"),
        "partial_year": bool(partial_year),
    }


def _build_windows(common_dates: pd.DatetimeIndex, bridge_start: pd.Timestamp, weak_start: pd.Timestamp) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    full_start = pd.Timestamp(common_dates.min())
    full_end = pd.Timestamp(common_dates.max())
    windows.append({"window_label": "full_available", "start": full_start, "end": full_end, "window_type": "full"})

    for year in sorted(set(int(x.year) for x in common_dates)):
        year_dates = common_dates[common_dates.year == year]
        if len(year_dates) == 0:
            continue
        windows.append(
            {
                "window_label": f"year_{year}",
                "start": pd.Timestamp(year_dates.min()),
                "end": pd.Timestamp(year_dates.max()),
                "window_type": "calendar_year",
            }
        )

    if bridge_start <= full_end:
        bridge_dates = common_dates[common_dates >= bridge_start]
        if len(bridge_dates) > 0:
            windows.append(
                {
                    "window_label": "bridge_full",
                    "start": pd.Timestamp(bridge_dates.min()),
                    "end": full_end,
                    "window_type": "named_window",
                }
            )

    if weak_start <= full_end:
        weak_dates = common_dates[common_dates >= weak_start]
        if len(weak_dates) > 0:
            windows.append(
                {
                    "window_label": f"weak_window_{weak_start:%Y%m%d}",
                    "start": pd.Timestamp(weak_dates.min()),
                    "end": full_end,
                    "window_type": "named_window",
                }
            )

    return windows


def _build_window_metrics(equity_df: pd.DataFrame, label: str, windows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in windows:
        row = _segment_metrics(equity_df, window["start"], window["end"])
        row.update(
            {
                "candidate_label": label,
                "window_label": window["window_label"],
                "window_type": window["window_type"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _winner(value_a: float, value_b: float, *, higher: bool) -> str:
    if pd.isna(value_a) or pd.isna(value_b):
        return "unknown"
    if higher:
        if value_a > value_b:
            return "a"
        if value_b > value_a:
            return "b"
    else:
        if value_a < value_b:
            return "a"
        if value_b < value_a:
            return "b"
    return "tie"


def _build_head2head(frame_a: pd.DataFrame, frame_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    cols = [
        "window_label",
        "window_type",
        "start_date",
        "end_date",
        "trading_days",
        "partial_year",
        "annual_return",
        "excess_annual_return",
        "excess_sharpe",
        "max_drawdown",
        "excess_max_drawdown",
        "avg_turnover",
        "avg_holding_count",
    ]
    merged = frame_a[cols].merge(frame_b[cols], on=["window_label"], suffixes=(f"_{label_a}", f"_{label_b}"))
    merged["annual_return_delta"] = merged[f"annual_return_{label_a}"] - merged[f"annual_return_{label_b}"]
    merged["excess_annual_return_delta"] = merged[f"excess_annual_return_{label_a}"] - merged[f"excess_annual_return_{label_b}"]
    merged["excess_sharpe_delta"] = merged[f"excess_sharpe_{label_a}"] - merged[f"excess_sharpe_{label_b}"]
    merged["max_drawdown_delta"] = merged[f"max_drawdown_{label_a}"] - merged[f"max_drawdown_{label_b}"]
    merged["avg_turnover_delta"] = merged[f"avg_turnover_{label_a}"] - merged[f"avg_turnover_{label_b}"]
    merged["winner_excess_annual_return"] = merged.apply(
        lambda row: label_a if _winner(row[f"excess_annual_return_{label_a}"], row[f"excess_annual_return_{label_b}"], higher=True) == "a"
        else label_b if _winner(row[f"excess_annual_return_{label_a}"], row[f"excess_annual_return_{label_b}"], higher=True) == "b"
        else "tie",
        axis=1,
    )
    merged["winner_excess_sharpe"] = merged.apply(
        lambda row: label_a if _winner(row[f"excess_sharpe_{label_a}"], row[f"excess_sharpe_{label_b}"], higher=True) == "a"
        else label_b if _winner(row[f"excess_sharpe_{label_a}"], row[f"excess_sharpe_{label_b}"], higher=True) == "b"
        else "tie",
        axis=1,
    )
    merged["winner_excess_max_drawdown"] = merged.apply(
        lambda row: label_a if _winner(row[f"excess_max_drawdown_{label_a}"], row[f"excess_max_drawdown_{label_b}"], higher=True) == "a"
        else label_b if _winner(row[f"excess_max_drawdown_{label_a}"], row[f"excess_max_drawdown_{label_b}"], higher=True) == "b"
        else "tie",
        axis=1,
    )
    return merged.sort_values([f"start_date_{label_a}", "window_label"]).reset_index(drop=True)


def _write_markdown(path: Path, label_a: str, label_b: str, h2h_df: pd.DataFrame, metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> None:
    sharpe_wins_a = int((h2h_df["winner_excess_sharpe"] == label_a).sum())
    sharpe_wins_b = int((h2h_df["winner_excess_sharpe"] == label_b).sum())
    annual_wins_a = int((h2h_df["winner_excess_annual_return"] == label_a).sum())
    annual_wins_b = int((h2h_df["winner_excess_annual_return"] == label_b).sum())
    full_row = h2h_df.loc[h2h_df["window_label"] == "full_available"].iloc[0]
    weak_rows = h2h_df.loc[h2h_df["window_label"].str.startswith("weak_window_")]
    weak_row = weak_rows.iloc[0] if not weak_rows.empty else None

    lines = [
        "# Execution Candidate Multi-Window Head-to-Head",
        "",
        "## Full Period",
        f"- {label_a}: annual {_pct(metrics_a.get('annual_return'))} | excess annual {_pct(metrics_a.get('excess_annual_return'))} | excess Sharpe {_num(metrics_a.get('excess_sharpe'))} | max drawdown {_pct(metrics_a.get('max_drawdown'))}",
        f"- {label_b}: annual {_pct(metrics_b.get('annual_return'))} | excess annual {_pct(metrics_b.get('excess_annual_return'))} | excess Sharpe {_num(metrics_b.get('excess_sharpe'))} | max drawdown {_pct(metrics_b.get('max_drawdown'))}",
        "",
        "## Multi-Window Wins",
        f"- excess annual return wins: {label_a} {annual_wins_a}, {label_b} {annual_wins_b}",
        f"- excess Sharpe wins: {label_a} {sharpe_wins_a}, {label_b} {sharpe_wins_b}",
        "",
        "## Key Windows",
        f"- full_available: excess annual delta {_pct(full_row['excess_annual_return_delta'])} | excess Sharpe delta {_num(full_row['excess_sharpe_delta'])}",
    ]
    if weak_row is not None:
        lines.append(
            f"- {weak_row['window_label']}: excess annual delta {_pct(weak_row['excess_annual_return_delta'])} | excess Sharpe delta {_num(weak_row['excess_sharpe_delta'])}"
        )
    lines.extend(
        [
            "",
            "## Direct Answer",
            f"- The better upside candidate is `{label_a if metrics_a.get('annual_return', 0.0) >= metrics_b.get('annual_return', 0.0) else label_b}` on the full window.",
            f"- The more phase-robust candidate is `{label_a if sharpe_wins_a >= sharpe_wins_b else label_b}` across the named windows in this report.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    if not run_a.is_absolute():
        run_a = (WORKSPACE_ROOT / run_a).resolve()
    if not run_b.is_absolute():
        run_b = (WORKSPACE_ROOT / run_b).resolve()
    equity_a, metrics_a = _load_run(run_a)
    equity_b, metrics_b = _load_run(run_b)

    common_dates = pd.DatetimeIndex(sorted(set(equity_a["date"]).intersection(set(equity_b["date"]))))
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between the two run directories.")

    windows = _build_windows(common_dates, pd.Timestamp(args.bridge_start), pd.Timestamp(args.weak_start))
    frame_a = _build_window_metrics(equity_a, args.label_a, windows)
    frame_b = _build_window_metrics(equity_b, args.label_b, windows)
    h2h_df = _build_head2head(frame_a, frame_b, args.label_a, args.label_b)

    output_dir = _resolve_output_dir(args.output_dir, args.label_a, args.label_b)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_a.to_csv(output_dir / f"window_metrics_{args.label_a}.csv", index=False, encoding="utf-8-sig")
    frame_b.to_csv(output_dir / f"window_metrics_{args.label_b}.csv", index=False, encoding="utf-8-sig")
    h2h_df.to_csv(output_dir / "head2head_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "label_a": args.label_a,
        "run_a": str(run_a),
        "label_b": args.label_b,
        "run_b": str(run_b),
        "bridge_start": args.bridge_start,
        "weak_start": args.weak_start,
        "window_count": int(len(h2h_df)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", args.label_a, args.label_b, h2h_df, metrics_a, metrics_b)
    print(json.dumps({"output_dir": str(output_dir), "window_count": int(len(h2h_df))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
