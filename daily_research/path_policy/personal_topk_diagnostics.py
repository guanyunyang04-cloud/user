from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_TOP_KS = (1, 3, 5, 10, 20)
DEFAULT_SCORE_COLUMNS = (
    "pred_decision_score",
    "pred_cum_mu_1d",
    "pred_cum_mu_3d",
    "pred_cum_mu_5d",
    "pred_cum_mu_10d",
    "pred_cum_mu_20d",
    "pred_aux_upside_20d",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _parse_csv_ints(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [int(item) for item in raw]
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _parse_csv_strings(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _header(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)


def _available_score_columns(columns: set[str], requested: tuple[str, ...]) -> tuple[str, ...]:
    candidates = requested or DEFAULT_SCORE_COLUMNS
    return tuple(column for column in candidates if column in columns)


def _target_columns(columns: set[str], horizons: tuple[int, ...]) -> list[str]:
    out: list[str] = []
    for horizon in horizons:
        for column in (
            f"future_cum_excess_return_{horizon}d",
            f"future_rank_{horizon}d",
            f"future_path_max_drawdown_{horizon}d",
            f"future_path_worst_1d_{horizon}d",
            f"future_path_upside_capture_{horizon}d",
        ):
            if column in columns:
                out.append(column)
    return out


def _load_predictions(
    prediction_csv: Path,
    *,
    score_columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    columns = _header(prediction_csv)
    missing_base = sorted({"date", "stock"} - columns)
    if missing_base:
        raise ValueError(f"prediction CSV missing required columns {missing_base}: {prediction_csv}")
    available_scores = _available_score_columns(columns, score_columns)
    if not available_scores:
        raise ValueError(f"no requested score columns found in {prediction_csv}")
    target_cols = _target_columns(columns, horizons)
    if not any(column.startswith("future_cum_excess_return_") for column in target_cols):
        raise ValueError(f"prediction CSV has no future_cum_excess_return columns for horizons={horizons}: {prediction_csv}")
    optional = [column for column in ("role", "model_family", "history_bucket", "history_valid_ratio") if column in columns]
    usecols = ["date", "stock", *optional, *available_scores, *target_cols]
    frame = pd.read_csv(prediction_csv, usecols=usecols)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["stock"] = frame["stock"].astype(str).str.strip().str.upper()
    frame = frame.loc[frame["date"].notna() & frame["stock"].ne("")].copy()
    for column in [*available_scores, *target_cols]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _series_metrics(values: pd.Series, *, annualization_horizon: int) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype("float64")
    if numeric.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_rate": 0.0,
            "window_sharpe_like": 0.0,
        }
    std = float(numeric.std(ddof=0))
    scale = math.sqrt(252.0 / max(int(annualization_horizon), 1))
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": std,
        "p25": float(numeric.quantile(0.25)),
        "p75": float(numeric.quantile(0.75)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "positive_rate": float((numeric > 0.0).mean()),
        "window_sharpe_like": float(numeric.mean() / std * scale) if std > 0.0 else 0.0,
    }


def _monthly_metrics(daily: pd.DataFrame, *, value_column: str) -> dict[str, Any]:
    if daily.empty:
        return {"month_count": 0, "positive_month_rate": 0.0, "worst_month": "", "worst_month_mean": 0.0}
    work = daily[["date", value_column]].copy()
    work["month"] = pd.to_datetime(work["date"], errors="coerce").dt.to_period("M").astype(str)
    monthly = work.groupby("month", as_index=False)[value_column].mean()
    if monthly.empty:
        return {"month_count": 0, "positive_month_rate": 0.0, "worst_month": "", "worst_month_mean": 0.0}
    worst = monthly.sort_values(value_column, ascending=True).iloc[0]
    best = monthly.sort_values(value_column, ascending=False).iloc[0]
    return {
        "month_count": int(len(monthly)),
        "positive_month_rate": float((monthly[value_column] > 0.0).mean()),
        "worst_month": str(worst["month"]),
        "worst_month_mean": float(worst[value_column]),
        "best_month": str(best["month"]),
        "best_month_mean": float(best[value_column]),
    }


def _build_daily_rows(
    frame: pd.DataFrame,
    *,
    score_columns: tuple[str, ...],
    horizons: tuple[int, ...],
    top_ks: tuple[int, ...],
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cost = float(round_trip_cost_bps) / 10000.0
    max_top_k = max(top_ks) if top_ks else 0
    for score_column in score_columns:
        if score_column not in frame.columns:
            continue
        for date, group in frame.groupby("date", sort=True):
            score = pd.to_numeric(group[score_column], errors="coerce")
            daily = group.loc[score.notna()].copy()
            if daily.empty:
                continue
            daily["_score"] = score.loc[daily.index].astype("float64")
            daily = daily.sort_values("_score", ascending=False, kind="mergesort")
            top_symbols_by_k: dict[int, str] = {}
            for top_k in top_ks:
                selected = daily.head(min(int(top_k), len(daily))).copy()
                if selected.empty:
                    continue
                top_symbols_by_k[int(top_k)] = ",".join(selected["stock"].astype(str).head(max_top_k).tolist())
                for horizon in horizons:
                    target_col = f"future_cum_excess_return_{int(horizon)}d"
                    if target_col not in selected.columns:
                        continue
                    selected_target = pd.to_numeric(selected[target_col], errors="coerce").dropna()
                    all_target = pd.to_numeric(daily[target_col], errors="coerce").dropna()
                    if selected_target.empty or all_target.empty:
                        continue
                    all_mean = float(all_target.mean())
                    gross = float(selected_target.mean())
                    row: dict[str, Any] = {
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "score_column": str(score_column),
                        "top_k": int(top_k),
                        "horizon": int(horizon),
                        "available_count": int(len(daily)),
                        "selected_count": int(len(selected_target)),
                        "selected_stocks": top_symbols_by_k[int(top_k)],
                        "score_mean": float(selected["_score"].mean()),
                        "score_min": float(selected["_score"].min()),
                        "score_max": float(selected["_score"].max()),
                        "gross_forward_excess_return": gross,
                        "net_forward_excess_return": gross - cost,
                        "all_mean_forward_excess_return": all_mean,
                        "excess_vs_all_mean": gross - all_mean,
                        "hit_rate": float((selected_target > 0.0).mean()),
                        "outperform_all_mean_rate": float((selected_target > all_mean).mean()),
                    }
                    rank_col = f"future_rank_{int(horizon)}d"
                    if rank_col in selected.columns:
                        rank_values = pd.to_numeric(selected[rank_col], errors="coerce").dropna()
                        row["selected_future_rank_mean"] = float(rank_values.mean()) if not rank_values.empty else np.nan
                    drawdown_col = f"future_path_max_drawdown_{int(horizon)}d"
                    if drawdown_col in selected.columns:
                        drawdown = pd.to_numeric(selected[drawdown_col], errors="coerce").dropna()
                        row["selected_max_drawdown_mean"] = float(drawdown.mean()) if not drawdown.empty else np.nan
                    worst_col = f"future_path_worst_1d_{int(horizon)}d"
                    if worst_col in selected.columns:
                        worst = pd.to_numeric(selected[worst_col], errors="coerce").dropna()
                        row["selected_worst_1d_mean"] = float(worst.mean()) if not worst.empty else np.nan
                    upside_col = f"future_path_upside_capture_{int(horizon)}d"
                    if upside_col in selected.columns:
                        upside = pd.to_numeric(selected[upside_col], errors="coerce").dropna()
                        row["selected_upside_capture_mean"] = float(upside.mean()) if not upside.empty else np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def _summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["score_column", "top_k", "horizon"]
    for keys, group in daily.groupby(group_cols, sort=True):
        score_column, top_k, horizon = keys
        net = _series_metrics(group["net_forward_excess_return"], annualization_horizon=int(horizon))
        gross = _series_metrics(group["gross_forward_excess_return"], annualization_horizon=int(horizon))
        monthly = _monthly_metrics(group, value_column="net_forward_excess_return")
        row: dict[str, Any] = {
            "score_column": str(score_column),
            "top_k": int(top_k),
            "horizon": int(horizon),
            "date_count": int(group["date"].nunique()),
            "selected_count_mean": float(group["selected_count"].mean()),
            "available_count_mean": float(group["available_count"].mean()),
            "gross_mean": gross["mean"],
            "gross_median": gross["median"],
            "gross_p25": gross["p25"],
            "gross_p75": gross["p75"],
            "net_mean": net["mean"],
            "net_median": net["median"],
            "net_p25": net["p25"],
            "net_p75": net["p75"],
            "net_min": net["min"],
            "net_max": net["max"],
            "net_positive_rate": net["positive_rate"],
            "window_sharpe_like": net["window_sharpe_like"],
            "all_mean_forward_excess_return": float(group["all_mean_forward_excess_return"].mean()),
            "excess_vs_all_mean": float(group["excess_vs_all_mean"].mean()),
            "hit_rate_mean": float(group["hit_rate"].mean()),
            "outperform_all_mean_rate": float(group["outperform_all_mean_rate"].mean()),
            **monthly,
        }
        for optional in (
            "selected_future_rank_mean",
            "selected_max_drawdown_mean",
            "selected_worst_1d_mean",
            "selected_upside_capture_mean",
        ):
            if optional in group.columns:
                values = pd.to_numeric(group[optional], errors="coerce").dropna()
                row[optional] = float(values.mean()) if not values.empty else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["horizon", "net_mean", "hit_rate_mean"], ascending=[True, False, False]).reset_index(drop=True)


def _leaderboard(summary: pd.DataFrame, *, limit: int = 20) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    cols = [
        "score_column",
        "top_k",
        "horizon",
        "date_count",
        "net_mean",
        "gross_mean",
        "excess_vs_all_mean",
        "hit_rate_mean",
        "net_positive_rate",
        "positive_month_rate",
        "window_sharpe_like",
        "selected_future_rank_mean",
    ]
    available = [column for column in cols if column in summary.columns]
    ranked = summary.sort_values(
        ["net_mean", "positive_month_rate", "hit_rate_mean"],
        ascending=[False, False, False],
    ).head(int(limit))
    return [dict(row) for row in ranked.loc[:, available].to_dict(orient="records")]


def build_personal_topk_diagnostics(
    *,
    prediction_csv: str | Path,
    output_root: str | Path | None = None,
    run_tag: str = "personal_topk_diagnostics",
    score_columns: str | Iterable[str] | None = None,
    horizons: str | Iterable[int] | None = None,
    top_ks: str | Iterable[int] | None = None,
    round_trip_cost_bps: float = 20.0,
    leaderboard_limit: int = 20,
) -> dict[str, Any]:
    prediction_path = Path(prediction_csv)
    resolved_horizons = _parse_csv_ints(horizons, default=DEFAULT_HORIZONS)
    requested_scores = _parse_csv_strings(score_columns)
    columns = _header(prediction_path)
    resolved_scores = _available_score_columns(columns, requested_scores)
    if not resolved_scores:
        raise ValueError(f"no usable score columns found: requested={requested_scores or DEFAULT_SCORE_COLUMNS}")
    resolved_top_ks = _parse_csv_ints(top_ks, default=DEFAULT_TOP_KS)
    root = Path(output_root) if output_root is not None else prediction_path.parent / str(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    frame = _load_predictions(prediction_path, score_columns=resolved_scores, horizons=resolved_horizons)
    daily = _build_daily_rows(
        frame,
        score_columns=resolved_scores,
        horizons=resolved_horizons,
        top_ks=resolved_top_ks,
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    summary = _summarize_daily(daily)
    daily_path = _write_frame(root / "personal_topk_daily.csv", daily)
    summary_path = _write_frame(root / "personal_topk_summary.csv", summary)
    report = {
        "status": "completed" if not summary.empty else "empty",
        "created_at": _now(),
        "run_tag": str(run_tag),
        "prediction_csv": str(prediction_path.resolve()),
        "output_root": str(root.resolve()),
        "contract": {
            "diagnostic_kind": "forward_selection_topk",
            "not_a_backtest": True,
            "return_semantics": "future cumulative excess return from prediction CSV labels",
            "cost_semantics": "net_forward_excess_return subtracts one approximate round-trip cost from each selected forward window",
            "round_trip_cost_bps": float(round_trip_cost_bps),
        },
        "input": {
            "row_count": int(len(frame)),
            "date_count": int(frame["date"].nunique()) if not frame.empty else 0,
            "symbol_count": int(frame["stock"].nunique()) if not frame.empty else 0,
            "start_date": frame["date"].min().strftime("%Y-%m-%d") if not frame.empty else "",
            "end_date": frame["date"].max().strftime("%Y-%m-%d") if not frame.empty else "",
            "score_columns": list(resolved_scores),
            "horizons": list(resolved_horizons),
            "top_ks": list(resolved_top_ks),
        },
        "outputs": {
            "daily_csv": daily_path,
            "summary_csv": summary_path,
        },
        "leaderboard": _leaderboard(summary, limit=int(leaderboard_limit)),
    }
    report_path = _write_json(root / "personal_topk_report.json", report)
    report["outputs"]["report_json"] = report_path
    _write_markdown(root / "personal_topk_report.md", report)
    report["outputs"]["report_md"] = str((root / "personal_topk_report.md").resolve())
    _write_json(root / "personal_topk_report.json", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Personal Top-K Diagnostics",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Prediction CSV: `{report.get('prediction_csv', '')}`",
        f"- Round-trip cost bps: `{report.get('contract', {}).get('round_trip_cost_bps', '')}`",
        f"- Diagnostic only, not backtest: `{report.get('contract', {}).get('not_a_backtest', True)}`",
        "",
        "## Input",
        "",
        f"- Rows: `{report.get('input', {}).get('row_count', 0)}`",
        f"- Dates: `{report.get('input', {}).get('date_count', 0)}`",
        f"- Symbols: `{report.get('input', {}).get('symbol_count', 0)}`",
        f"- Range: `{report.get('input', {}).get('start_date', '')}` to `{report.get('input', {}).get('end_date', '')}`",
        f"- Score columns: `{','.join(report.get('input', {}).get('score_columns', []))}`",
        "",
        "## Leaderboard",
        "",
    ]
    leaderboard = list(report.get("leaderboard", []) or [])
    if leaderboard:
        lines.append("| score | top_k | horizon | net_mean | hit_rate | month_pos | rank_mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in leaderboard[:20]:
            lines.append(
                "| {score} | {top_k} | {horizon} | {net:.4f} | {hit:.3f} | {month:.3f} | {rank:.3f} |".format(
                    score=row.get("score_column", ""),
                    top_k=int(row.get("top_k", 0) or 0),
                    horizon=int(row.get("horizon", 0) or 0),
                    net=float(row.get("net_mean", 0.0) or 0.0),
                    hit=float(row.get("hit_rate_mean", 0.0) or 0.0),
                    month=float(row.get("positive_month_rate", 0.0) or 0.0),
                    rank=float(row.get("selected_future_rank_mean", 0.0) or 0.0),
                )
            )
    else:
        lines.append("- No usable rows.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Daily CSV: `{report.get('outputs', {}).get('daily_csv', '')}`",
            f"- Summary CSV: `{report.get('outputs', {}).get('summary_csv', '')}`",
            f"- Report JSON: `{report.get('outputs', {}).get('report_json', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate forecast predictions from a personal small-capital top-K selection perspective.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-tag", default="personal_topk_diagnostics")
    parser.add_argument("--score-columns", default="")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--top-ks", default="1,3,5,10,20")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--leaderboard-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_personal_topk_diagnostics(
        prediction_csv=args.prediction_csv,
        output_root=args.output_root or None,
        run_tag=args.run_tag,
        score_columns=args.score_columns or None,
        horizons=args.horizons,
        top_ks=args.top_ks,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        leaderboard_limit=int(args.leaderboard_limit),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report["outputs"]["report_json"])
    return report


if __name__ == "__main__":
    main()
