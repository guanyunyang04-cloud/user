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
        description="Build a reusable performance-dispersion report with calendar-year and weak-window splits."
    )
    parser.add_argument("--run-dir", required=True, help="Formal output directory to diagnose.")
    parser.add_argument(
        "--strict-label",
        default="",
        help="Required when the run only has strict_compare_summary.csv and contains multiple labels.",
    )
    parser.add_argument(
        "--auto-window-days",
        default="63,126,252",
        help="Comma-separated rolling window sizes for automatic weak-window detection.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output dir. Defaults to daily_research/output/performance_dispersion_<run>_<date>_rN",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _safe_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _format_yyyymmdd(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def _sanitize_name(name: str) -> str:
    out = []
    for char in str(name):
        if char.isalnum() or char in {"_", "-"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "report"


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


def _resolve_output_dir(explicit: str | None, run_dir: Path) -> Path:
    if explicit:
        out = Path(explicit)
        return out if out.is_absolute() else (WORKSPACE_ROOT / out).resolve()

    prefix = f"performance_dispersion_{_sanitize_name(run_dir.name)}_{datetime.now():%Y%m%d}_r"
    existing = sorted(OUTPUT_ROOT.glob(f"{prefix}*"))
    next_index = 1
    for item in existing:
        suffix = item.name.replace(prefix, "", 1)
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    return OUTPUT_ROOT / f"{prefix}{next_index}"


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((1.0 + valid).prod() - 1.0)


def _annualized_return_from_total(total_return: float, periods: int) -> float:
    if periods <= 0 or np.isnan(total_return):
        return float("nan")
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (252.0 / periods) - 1.0)


def _pick_row(frame: pd.DataFrame, metric: str, side: str) -> pd.Series | None:
    valid = frame.dropna(subset=[metric])
    if valid.empty:
        return None
    if side == "best":
        return valid.loc[valid[metric].idxmax()]
    if side == "worst":
        return valid.loc[valid[metric].idxmin()]
    raise ValueError(f"Unsupported side: {side}")


def _detect_mode(run_dir: Path) -> str:
    if (run_dir / "equity_curve.csv").exists():
        return "equity_curve"
    if (run_dir / "strict_compare_summary.csv").exists():
        return "strict_compare"
    raise FileNotFoundError(f"No equity_curve.csv or strict_compare_summary.csv found in {run_dir}")


def _build_segment_metrics(equity_df: pd.DataFrame, start_idx: int, end_idx: int) -> dict[str, Any]:
    segment = equity_df.iloc[start_idx : end_idx + 1].copy()
    if segment.empty:
        raise ValueError("Empty segment")

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
    benchmark_annual_vol = _annualized_vol(benchmark_returns)
    excess_annual_vol = _annualized_vol(excess_returns)

    return {
        "start_date": str(segment["date"].iloc[0].date()),
        "end_date": str(segment["date"].iloc[-1].date()),
        "trading_days": int(len(segment)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": float(annual_return / annual_vol) if annual_vol > 0 else float("nan"),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annual_return": benchmark_annual_return,
        "benchmark_annual_vol": benchmark_annual_vol,
        "excess_total_return": excess_total_return,
        "excess_annual_return": excess_annual_return,
        "excess_annual_vol": excess_annual_vol,
        "excess_sharpe": float(excess_annual_return / excess_annual_vol) if excess_annual_vol > 0 else float("nan"),
        "max_drawdown": float(_max_drawdown(portfolio_path)),
        "excess_max_drawdown": float(_max_drawdown(excess_path)),
        "avg_holding_count": float(segment["holding_count"].mean()) if "holding_count" in segment.columns else float("nan"),
        "avg_turnover": float(segment["turnover"].mean()) if "turnover" in segment.columns else float("nan"),
        "hit_rate": float((portfolio_returns > 0).mean()) if not portfolio_returns.empty else float("nan"),
        "excess_hit_rate": float((excess_returns > 0).mean()) if not excess_returns.empty else float("nan"),
        "regime_active_ratio": float(segment["regime_on"].fillna(False).astype(bool).mean()) if "regime_on" in segment.columns else float("nan"),
    }


def _load_equity_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    metrics = _load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    run_config = _load_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    return equity, metrics, run_config


def _build_calendar_year_summary(equity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_year = int(equity_df["date"].dt.year.min())
    last_year = int(equity_df["date"].dt.year.max())
    for year, frame in equity_df.groupby(equity_df["date"].dt.year, sort=True):
        start_idx = int(frame.index[0])
        end_idx = int(frame.index[-1])
        row = _build_segment_metrics(equity_df, start_idx, end_idx)
        row["year"] = int(year)
        start_dt = pd.Timestamp(row["start_date"])
        end_dt = pd.Timestamp(row["end_date"])
        partial_start = int(year) == first_year and start_dt.month > 1
        partial_end = int(year) == last_year and end_dt.month < 12
        row["is_partial_year"] = bool(partial_start or partial_end)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_rolling_summary(equity_df: pd.DataFrame, window_days: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in window_days:
        if window <= 1 or len(equity_df) < window:
            continue
        scan_rows: list[dict[str, Any]] = []
        for start_idx in range(0, len(equity_df) - window + 1):
            end_idx = start_idx + window - 1
            row = _build_segment_metrics(equity_df, start_idx, end_idx)
            row["window_days"] = int(window)
            scan_rows.append(row)
        scan_df = pd.DataFrame(scan_rows)
        for metric in ("annual_return", "excess_annual_return", "excess_sharpe"):
            for side in ("best", "worst"):
                picked = _pick_row(scan_df, metric, side)
                if picked is None:
                    continue
                out = dict(picked)
                out["rank_metric"] = metric
                out["side"] = side
                rows.append(out)
    if not rows:
        return pd.DataFrame()
    cols = [
        "window_days",
        "rank_metric",
        "side",
        "start_date",
        "end_date",
        "trading_days",
        "total_return",
        "annual_return",
        "excess_total_return",
        "excess_annual_return",
        "excess_sharpe",
        "max_drawdown",
        "excess_max_drawdown",
        "avg_turnover",
        "regime_active_ratio",
    ]
    return pd.DataFrame(rows).loc[:, cols].sort_values(["window_days", "rank_metric", "side"]).reset_index(drop=True)


def _build_named_windows_from_equity(equity_df: pd.DataFrame, run_config: dict[str, Any]) -> pd.DataFrame:
    windows = run_config.get("windows") or []
    if not windows:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for item in windows:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        start = pd.Timestamp(_format_yyyymmdd(item.get("start")))
        end = pd.Timestamp(_format_yyyymmdd(item.get("end")))
        frame = equity_df.loc[(equity_df["date"] >= start) & (equity_df["date"] <= end)]
        if frame.empty:
            continue
        start_idx = int(frame.index[0])
        end_idx = int(frame.index[-1])
        row = _build_segment_metrics(equity_df, start_idx, end_idx)
        row["window_label"] = name
        row["window_source"] = "run_config"
        rows.append(row)
    return pd.DataFrame(rows)


def _select_strict_row(df: pd.DataFrame, strict_label: str) -> pd.Series:
    if strict_label:
        filtered = df.loc[df["label"] == strict_label]
        if filtered.empty:
            raise ValueError(f"Label {strict_label!r} not found in strict_compare_summary.csv")
        if len(filtered) > 1:
            raise ValueError(f"Label {strict_label!r} is not unique in strict_compare_summary.csv")
        return filtered.iloc[0]
    if len(df) == 1:
        return df.iloc[0]
    raise ValueError("Provide --strict-label because strict_compare_summary.csv contains multiple labels.")


def _extract_summary_metric(row: pd.Series, prefix: str, name: str) -> float:
    key = f"{prefix}{name}"
    return _safe_float(row[key]) if key in row.index else float("nan")


def _build_named_windows_from_strict_compare(run_dir: Path, strict_label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_df = pd.read_csv(run_dir / "strict_compare_summary.csv")
    run_config = _load_json(run_dir / "run_config.json")
    row = _select_strict_row(summary_df, strict_label)
    focus_state = str(run_config.get("focus_state", "")).strip()
    history_window = run_config.get("history_window") or {}

    rows: list[dict[str, Any]] = []

    def append_row(window_label: str, prefix: str, start: str, end: str, source: str) -> None:
        rows.append(
            {
                "window_label": window_label,
                "window_source": source,
                "start_date": start,
                "end_date": end,
                "trading_days": _extract_summary_metric(row, prefix, "days"),
                "total_return": _extract_summary_metric(row, prefix, "total_return"),
                "annual_return": _extract_summary_metric(row, prefix, "annual_return"),
                "annual_vol": _extract_summary_metric(row, prefix, "annual_vol"),
                "sharpe": _extract_summary_metric(row, prefix, "sharpe"),
                "benchmark_total_return": _extract_summary_metric(row, prefix, "benchmark_total_return"),
                "benchmark_annual_return": _extract_summary_metric(row, prefix, "benchmark_annual_return"),
                "excess_total_return": _extract_summary_metric(row, prefix, "excess_total_return"),
                "excess_annual_return": _extract_summary_metric(row, prefix, "excess_annual_return"),
                "excess_sharpe": _extract_summary_metric(row, prefix, "excess_sharpe"),
                "max_drawdown": _extract_summary_metric(row, prefix, "max_drawdown"),
                "excess_max_drawdown": _extract_summary_metric(row, prefix, "excess_max_drawdown"),
                "avg_holding_count": _extract_summary_metric(row, prefix, "avg_holding_count"),
                "avg_turnover": _extract_summary_metric(row, prefix, "avg_turnover"),
                "hit_rate": _extract_summary_metric(row, prefix, "hit_rate"),
                "regime_active_ratio": _extract_summary_metric(row, prefix, "regime_active_ratio"),
            }
        )

    append_row(
        "full",
        "full_",
        _format_yyyymmdd(history_window.get("effective_start_date")),
        _format_yyyymmdd(history_window.get("end_date")),
        "history_window",
    )

    for item in run_config.get("windows") or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        start = _format_yyyymmdd(item.get("start"))
        end = _format_yyyymmdd(item.get("end"))
        append_row(name, f"{name}_", start, end, "run_config")
        if focus_state:
            append_row(f"{focus_state}/{name}", f"{focus_state}_{name}_", start, end, "focus_state")

    result = pd.DataFrame(rows)
    result = result.dropna(how="all", subset=["annual_return", "excess_annual_return", "excess_sharpe"]).reset_index(drop=True)
    meta = {
        "selected_label": str(row["label"]),
        "focus_state": focus_state,
        "history_start": _format_yyyymmdd(history_window.get("effective_start_date")),
        "history_end": _format_yyyymmdd(history_window.get("end_date")),
    }
    return result, meta


def _build_equity_report(run_dir: Path, window_days: list[int]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    equity_df, metrics, run_config = _load_equity_run(run_dir)
    calendar_year_df = _build_calendar_year_summary(equity_df)
    rolling_df = _build_rolling_summary(equity_df, window_days)
    named_window_df = _build_named_windows_from_equity(equity_df, run_config)

    best_year = _pick_row(calendar_year_df, "annual_return", "best") if not calendar_year_df.empty else None
    worst_year = _pick_row(calendar_year_df, "annual_return", "worst") if not calendar_year_df.empty else None
    worst_126_excess_sharpe = None
    if not rolling_df.empty:
        weak_candidates = rolling_df.loc[
            (rolling_df["window_days"] == 126)
            & (rolling_df["rank_metric"] == "excess_sharpe")
            & (rolling_df["side"] == "worst")
        ]
        if not weak_candidates.empty:
            worst_126_excess_sharpe = weak_candidates.iloc[0].to_dict()

    dispersion_reasons: list[str] = []
    if best_year is not None and worst_year is not None:
        if _safe_float(best_year["annual_return"]) - _safe_float(worst_year["annual_return"]) >= 0.20:
            dispersion_reasons.append("calendar_year_annual_return_range_ge_20pct")
        if _safe_float(best_year["excess_annual_return"]) > 0 and _safe_float(worst_year["excess_annual_return"]) < 0:
            dispersion_reasons.append("calendar_year_excess_sign_flip")
    if worst_126_excess_sharpe is not None and _safe_float(worst_126_excess_sharpe["excess_sharpe"]) < 0:
        dispersion_reasons.append("rolling_126d_worst_excess_sharpe_negative")

    report = {
        "mode": "equity_curve",
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "full_metrics": metrics,
        "calendar_year_available": not calendar_year_df.empty,
        "calendar_year_count": int(len(calendar_year_df)),
        "calendar_year_partial_count": int(calendar_year_df["is_partial_year"].sum()) if "is_partial_year" in calendar_year_df.columns else 0,
        "best_calendar_year": None if best_year is None else dict(best_year),
        "worst_calendar_year": None if worst_year is None else dict(worst_year),
        "named_window_count": int(len(named_window_df)),
        "auto_window_days": window_days,
        "worst_126d_excess_sharpe_window": worst_126_excess_sharpe,
        "dispersion_exists": bool(dispersion_reasons),
        "dispersion_reasons": dispersion_reasons,
    }
    frames = {
        "calendar_year_summary": calendar_year_df,
        "rolling_window_summary": rolling_df,
        "named_window_summary": named_window_df,
    }
    return report, frames


def _build_strict_report(run_dir: Path, strict_label: str) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    named_window_df, meta = _build_named_windows_from_strict_compare(run_dir, strict_label)
    best_named = _pick_row(named_window_df, "annual_return", "best") if not named_window_df.empty else None
    worst_named = _pick_row(named_window_df, "annual_return", "worst") if not named_window_df.empty else None
    report = {
        "mode": "strict_compare",
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "selected_label": meta["selected_label"],
        "focus_state": meta["focus_state"],
        "calendar_year_available": False,
        "best_named_window": None if best_named is None else dict(best_named),
        "worst_named_window": None if worst_named is None else dict(worst_named),
        "dispersion_exists": bool(
            best_named is not None
            and worst_named is not None
            and _safe_float(best_named["annual_return"]) - _safe_float(worst_named["annual_return"]) >= 0.20
        ),
        "notes": [
            "No equity_curve.csv found, so natural-year and rolling-window splits are unavailable.",
            "This report uses named windows embedded in strict_compare_summary.csv and run_config.json.",
        ],
    }
    frames = {
        "named_window_summary": named_window_df,
    }
    return report, frames


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_markdown(path: Path, report: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Performance Dispersion Report",
        "",
        "## Run",
        f"- run_dir: `{report['run_dir']}`",
        f"- mode: `{report['mode']}`",
    ]
    if report.get("selected_label"):
        lines.append(f"- selected_label: `{report['selected_label']}`")
    if report.get("focus_state"):
        lines.append(f"- focus_state: `{report['focus_state']}`")

    if report["mode"] == "equity_curve":
        full = report.get("full_metrics", {})
        lines.extend(
            [
                "",
                "## Full Period",
                f"- annual_return: {_pct(full.get('annual_return'))}",
                f"- excess_annual_return: {_pct(full.get('excess_annual_return'))}",
                f"- excess_sharpe: {_num(full.get('excess_sharpe'))}",
                f"- max_drawdown: {_pct(full.get('max_drawdown'))}",
            ]
        )
        calendar_year_df = frames.get("calendar_year_summary", pd.DataFrame())
        if not calendar_year_df.empty:
            best_year = report.get("best_calendar_year") or {}
            worst_year = report.get("worst_calendar_year") or {}
            lines.extend(
                [
                    "",
                    "## Calendar Year Split",
                    f"- best year: `{best_year.get('year')}` | annual_return {_pct(best_year.get('annual_return'))} | excess_annual_return {_pct(best_year.get('excess_annual_return'))} | partial_year `{best_year.get('is_partial_year')}`",
                    f"- worst year: `{worst_year.get('year')}` | annual_return {_pct(worst_year.get('annual_return'))} | excess_annual_return {_pct(worst_year.get('excess_annual_return'))} | partial_year `{worst_year.get('is_partial_year')}`",
                ]
            )
        rolling_df = frames.get("rolling_window_summary", pd.DataFrame())
        if not rolling_df.empty:
            weak_63 = _pick_row(
                rolling_df.loc[(rolling_df["window_days"] == 63) & (rolling_df["rank_metric"] == "excess_sharpe")],
                "excess_sharpe",
                "worst",
            )
            weak_126 = _pick_row(
                rolling_df.loc[(rolling_df["window_days"] == 126) & (rolling_df["rank_metric"] == "excess_sharpe")],
                "excess_sharpe",
                "worst",
            )
            if weak_63 is not None or weak_126 is not None:
                lines.extend(["", "## Auto Weak Windows"])
            if weak_63 is not None:
                lines.append(
                    f"- worst 63d by excess_sharpe: `{weak_63['start_date']} -> {weak_63['end_date']}` | excess_sharpe {_num(weak_63['excess_sharpe'])} | annual_return {_pct(weak_63['annual_return'])}"
                )
            if weak_126 is not None:
                lines.append(
                    f"- worst 126d by excess_sharpe: `{weak_126['start_date']} -> {weak_126['end_date']}` | excess_sharpe {_num(weak_126['excess_sharpe'])} | annual_return {_pct(weak_126['annual_return'])}"
                )
        named_window_df = frames.get("named_window_summary", pd.DataFrame())
        if not named_window_df.empty:
            lines.extend(["", "## Named Windows"])
            for _, row in named_window_df.iterrows():
                lines.append(
                    f"- `{row['window_label']}`: {row['start_date']} -> {row['end_date']} | annual_return {_pct(row['annual_return'])} | excess_annual_return {_pct(row['excess_annual_return'])} | excess_sharpe {_num(row['excess_sharpe'])}"
                )
        lines.extend(
            [
                "",
                "## Direct Answer",
                f"- dispersion_exists: `{report['dispersion_exists']}`",
                f"- reasons: `{', '.join(report.get('dispersion_reasons', [])) or 'none'}`",
                "- Read this report as a phase-dispersion diagnostic, not as a replacement for full-period CAGR.",
            ]
        )
    else:
        named_window_df = frames.get("named_window_summary", pd.DataFrame())
        lines.extend(["", "## Named Windows"])
        if named_window_df.empty:
            lines.append("- No named windows were available.")
        else:
            for _, row in named_window_df.iterrows():
                lines.append(
                    f"- `{row['window_label']}`: {row['start_date']} -> {row['end_date']} | annual_return {_pct(row['annual_return'])} | excess_annual_return {_pct(row['excess_annual_return'])} | excess_sharpe {_num(row['excess_sharpe'])}"
                )
        lines.extend(
            [
                "",
                "## Direct Answer",
                f"- dispersion_exists: `{report['dispersion_exists']}`",
                "- Natural-year and rolling-window splits are unavailable because this run only stores strict summary artifacts.",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (WORKSPACE_ROOT / run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    window_days = _parse_int_list(args.auto_window_days)
    output_dir = _resolve_output_dir(args.output_dir, run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = _detect_mode(run_dir)
    if mode == "equity_curve":
        report, frames = _build_equity_report(run_dir, window_days)
    else:
        report, frames = _build_strict_report(run_dir, args.strict_label)

    for name, frame in frames.items():
        _write_frame(output_dir / f"{name}.csv", frame)

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "summary.md", report, frames)

    print(json.dumps({"output_dir": str(output_dir), "mode": report["mode"], "dispersion_exists": report["dispersion_exists"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
