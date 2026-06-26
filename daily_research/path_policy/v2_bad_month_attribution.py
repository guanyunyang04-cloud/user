from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow, load_raw_data_with_cache
from daily_research.path_policy import v2_research_reset_baseline as v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "quant_data_platform/data/lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_bad_month_attribution_20260602_01"
MATRIX_RUN_TAG = "v2_candidate_review_matrix_20260602_01"
DEFAULT_VARIANT_ID = "h20_mw080_rb5d_all_c3_7_10_regime_off"


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
    return str(value)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _default_backtest_dir(matrix_run_tag: str, variant_id: str) -> Path:
    return STUDIES_ROOT / matrix_run_tag / "shared_backtest" / f"{matrix_run_tag}_{variant_id}"


def _wide_from_long_csv(path: Path, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "stock", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["stock"] = frame["stock"].astype(str).str.strip().str.upper()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.loc[frame["date"].notna() & frame["stock"].ne("")].copy()
    return frame.pivot_table(index="date", columns="stock", values=value_column, aggfunc="mean").sort_index()


def _month_to_analyze(monthly_summary: pd.DataFrame, requested_month: str) -> str:
    if requested_month:
        return str(requested_month)
    work = monthly_summary.copy()
    work["excess_return"] = pd.to_numeric(work.get("excess_return"), errors="coerce")
    work = work.dropna(subset=["excess_return"])
    if work.empty:
        raise ValueError("monthly_backtest_summary has no usable excess_return rows.")
    return str(work.sort_values("excess_return", ascending=True).iloc[0]["month"])


def _load_close_panel(
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
    dataset_id: str,
    data_lake_root: str | Path,
    benchmark: str,
    use_cache: bool,
) -> pd.DataFrame:
    raw, _meta = load_raw_data_with_cache(
        data_source="lake",
        csv_folder=None,
        universe=symbols,
        benchmark=benchmark,
        history_window=HistoryWindow(
            mode="infer",
            requested_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
            effective_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
            required_trading_days=1,
        ),
        lake_dataset_id=dataset_id,
        data_lake_root=str(data_lake_root),
        use_cache=use_cache,
        refresh_cache=False,
        progress_desc="v2 bad-month attribution raw load",
    )
    close = raw.get("Close")
    if close is None or close.empty:
        raise ValueError("Unable to load Close panel for attribution.")
    return close.reindex(columns=symbols).sort_index()


def _safe_quantile_bucket(series: pd.Series, *, q: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 2:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index)
    try:
        bucket = pd.qcut(numeric.rank(method="first"), q=min(q, int(valid.nunique())), labels=False, duplicates="drop")
    except ValueError:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index)
    return bucket.map(lambda item: f"{prefix}_{int(item) + 1}" if pd.notna(item) else f"{prefix}_missing")


def _contribution_rows(
    *,
    close: pd.DataFrame,
    target_weight: pd.DataFrame,
    score: pd.DataFrame,
    month: str,
    recent_return_window: int = 5,
    volatility_window: int = 20,
) -> pd.DataFrame:
    returns = close.pct_change()
    prev_weight = target_weight.shift(1).reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    aligned_score = score.shift(1).reindex(index=returns.index, columns=returns.columns)
    recent_return = close.pct_change(max(int(recent_return_window), 1)).shift(1).reindex(index=returns.index, columns=returns.columns)
    volatility = (
        returns.rolling(max(int(volatility_window), 2), min_periods=max(2, min(5, int(volatility_window))))
        .std()
        .shift(1)
        .reindex(index=returns.index, columns=returns.columns)
    )
    reversal = returns.shift(1).reindex(index=returns.index, columns=returns.columns)
    contribution = prev_weight * returns
    month_period = pd.Period(month, freq="M")
    mask = pd.Series(returns.index.to_period("M") == month_period, index=returns.index)
    rows = []
    for date in returns.index[mask]:
        daily = pd.DataFrame(
            {
                "date": date,
                "stock": returns.columns,
                "prev_weight": prev_weight.loc[date].values,
                "stock_return": returns.loc[date].values,
                "score": aligned_score.loc[date].values,
                "recent_return": recent_return.loc[date].values,
                "volatility": volatility.loc[date].values,
                "reversal": reversal.loc[date].values,
                "contribution": contribution.loc[date].values,
            }
        )
        daily = daily.loc[daily["prev_weight"].abs() > 1e-12].copy()
        if daily.empty:
            continue
        daily["score_bucket"] = _safe_quantile_bucket(daily["score"], q=5, prefix="score_q").values
        daily["weight_bucket"] = _safe_quantile_bucket(daily["prev_weight"], q=5, prefix="weight_q").values
        daily["recent_return_bucket"] = _safe_quantile_bucket(daily["recent_return"], q=5, prefix="recent_return_q").values
        daily["volatility_bucket"] = _safe_quantile_bucket(daily["volatility"], q=5, prefix="volatility_q").values
        daily["reversal_bucket"] = _safe_quantile_bucket(daily["reversal"], q=5, prefix="reversal_q").values
        rows.append(daily)
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "stock",
                "prev_weight",
                "stock_return",
                "score",
                "recent_return",
                "volatility",
                "reversal",
                "contribution",
                "score_bucket",
                "weight_bucket",
                "recent_return_bucket",
                "volatility_bucket",
                "reversal_bucket",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["date", "contribution"]).reset_index(drop=True)


def _group_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "contribution_sum", "contribution_mean", "row_count", "avg_prev_weight", "avg_stock_return"])
    grouped = frame.groupby(group_columns, as_index=False).agg(
        contribution_sum=("contribution", "sum"),
        contribution_mean=("contribution", "mean"),
        row_count=("stock", "count"),
        avg_prev_weight=("prev_weight", "mean"),
        avg_stock_return=("stock_return", "mean"),
    )
    return grouped.sort_values("contribution_sum").reset_index(drop=True)


def _top_records(frame: pd.DataFrame, *, n: int, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cols = [
        "date",
        "stock",
        "prev_weight",
        "stock_return",
        "score",
        "recent_return",
        "volatility",
        "reversal",
        "contribution",
        "score_bucket",
        "weight_bucket",
        "recent_return_bucket",
        "volatility_bucket",
        "reversal_bucket",
    ]
    cols = [col for col in cols if col in frame.columns]
    return frame.sort_values("contribution", ascending=ascending).head(int(n)).loc[:, cols].to_dict("records")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}) or {})
    worst = list(report.get("top_negative_contributors", []) or [])[:5]
    lines = [
        "# V2 Bad Month Attribution",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Matrix run tag: `{report.get('matrix_run_tag', '')}`",
        f"- Variant: `{report.get('variant_id', '')}`",
        f"- Month: `{report.get('month', '')}`",
        f"- Approx contribution sum: `{summary.get('approx_contribution_sum', '')}`",
        f"- Portfolio month return: `{summary.get('portfolio_return', '')}`",
        f"- Benchmark month return: `{summary.get('benchmark_return', '')}`",
        f"- Excess month return: `{summary.get('excess_return', '')}`",
        "",
        "## Top Negative Contributors",
        "",
    ]
    for item in worst:
        lines.append(
            f"- `{item.get('date')}` `{item.get('stock')}` contribution `{item.get('contribution')}` "
            f"return `{item.get('stock_return')}` weight `{item.get('prev_weight')}` score `{item.get('score')}`"
        )
    lines.extend(
        [
            "",
            "## Local Risk Buckets",
            "",
            f"- Recent return window: `{summary.get('recent_return_window', '')}`",
            f"- Volatility window: `{summary.get('volatility_window', '')}`",
            "- Additional CSV outputs cover recent-return, volatility, reversal, score x volatility, and score x reversal buckets.",
            "",
            "",
            "## Boundary",
            "",
            "- research_only: `true`",
            "- promotion_allowed: `false`",
            "- active_execution_strategy_expected_diff: `none`",
            "- Attribution is approximate: previous target weight x close-to-close stock return.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bad_month_attribution(
    *,
    run_tag: str = RUN_TAG,
    matrix_run_tag: str = MATRIX_RUN_TAG,
    variant_id: str = DEFAULT_VARIANT_ID,
    backtest_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    benchmark: str = "000300.SH",
    month: str = "",
    recent_return_window: int = 5,
    volatility_window: int = 20,
    use_cache: bool = True,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    source_dir = Path(backtest_dir) if backtest_dir is not None else _default_backtest_dir(matrix_run_tag, variant_id)
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    monthly_summary = pd.read_csv(source_dir / "monthly_backtest_summary.csv")
    selected_month = _month_to_analyze(monthly_summary, month)
    month_row = monthly_summary.loc[monthly_summary["month"].astype(str).eq(selected_month)].iloc[0].to_dict()
    target_weight = _wide_from_long_csv(source_dir / "aligned_daily_target_weight_panel.csv", "target_weight")
    score = _wide_from_long_csv(source_dir / "aligned_daily_score_panel.csv", "score")
    symbols = sorted(set(target_weight.columns.astype(str)) | set(score.columns.astype(str)))
    start_date = min(pd.Timestamp(target_weight.index.min()), pd.Timestamp(score.index.min())).strftime("%Y-%m-%d")
    end_date = max(pd.Timestamp(target_weight.index.max()), pd.Timestamp(score.index.max())).strftime("%Y-%m-%d")
    close = _load_close_panel(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        benchmark=benchmark,
        use_cache=use_cache,
    )
    rows = _contribution_rows(
        close=close,
        target_weight=target_weight,
        score=score,
        month=selected_month,
        recent_return_window=recent_return_window,
        volatility_window=volatility_window,
    )
    symbol_summary = _group_summary(rows, ["stock"])
    date_summary = _group_summary(rows, ["date"])
    score_bucket_summary = _group_summary(rows, ["score_bucket"])
    weight_bucket_summary = _group_summary(rows, ["weight_bucket"])
    recent_return_bucket_summary = _group_summary(rows, ["recent_return_bucket"])
    volatility_bucket_summary = _group_summary(rows, ["volatility_bucket"])
    reversal_bucket_summary = _group_summary(rows, ["reversal_bucket"])
    score_volatility_bucket_summary = _group_summary(rows, ["score_bucket", "volatility_bucket"])
    score_reversal_bucket_summary = _group_summary(rows, ["score_bucket", "reversal_bucket"])
    contribution_csv = root / "bad_month_contributions.csv"
    symbol_csv = root / "bad_month_symbol_summary.csv"
    date_csv = root / "bad_month_date_summary.csv"
    score_bucket_csv = root / "bad_month_score_bucket_summary.csv"
    weight_bucket_csv = root / "bad_month_weight_bucket_summary.csv"
    recent_return_bucket_csv = root / "bad_month_recent_return_bucket_summary.csv"
    volatility_bucket_csv = root / "bad_month_volatility_bucket_summary.csv"
    reversal_bucket_csv = root / "bad_month_reversal_bucket_summary.csv"
    score_volatility_bucket_csv = root / "bad_month_score_volatility_bucket_summary.csv"
    score_reversal_bucket_csv = root / "bad_month_score_reversal_bucket_summary.csv"
    rows.to_csv(contribution_csv, index=False, encoding="utf-8-sig")
    symbol_summary.to_csv(symbol_csv, index=False, encoding="utf-8-sig")
    date_summary.to_csv(date_csv, index=False, encoding="utf-8-sig")
    score_bucket_summary.to_csv(score_bucket_csv, index=False, encoding="utf-8-sig")
    weight_bucket_summary.to_csv(weight_bucket_csv, index=False, encoding="utf-8-sig")
    recent_return_bucket_summary.to_csv(recent_return_bucket_csv, index=False, encoding="utf-8-sig")
    volatility_bucket_summary.to_csv(volatility_bucket_csv, index=False, encoding="utf-8-sig")
    reversal_bucket_summary.to_csv(reversal_bucket_csv, index=False, encoding="utf-8-sig")
    score_volatility_bucket_summary.to_csv(score_volatility_bucket_csv, index=False, encoding="utf-8-sig")
    score_reversal_bucket_summary.to_csv(score_reversal_bucket_csv, index=False, encoding="utf-8-sig")
    report = {
        "schema_version": 1,
        "status": "completed",
        "run_tag": str(run_tag),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_bad_month_attribution",
        "matrix_run_tag": str(matrix_run_tag),
        "variant_id": str(variant_id),
        "backtest_dir": str(source_dir),
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "month": str(selected_month),
        "summary": {
            "row_count": int(len(rows)),
            "symbol_count": int(rows["stock"].nunique()) if not rows.empty else 0,
            "date_count": int(rows["date"].nunique()) if not rows.empty else 0,
            "approx_contribution_sum": float(rows["contribution"].sum()) if not rows.empty else 0.0,
            "portfolio_return": float(month_row.get("portfolio_return", 0.0)),
            "benchmark_return": float(month_row.get("benchmark_return", 0.0)),
            "excess_return": float(month_row.get("excess_return", 0.0)),
            "total_trading_cost_return": float(month_row.get("total_trading_cost_return", 0.0)),
            "recent_return_window": int(recent_return_window),
            "volatility_window": int(volatility_window),
        },
        "top_negative_contributors": _top_records(rows, n=20, ascending=True),
        "top_positive_contributors": _top_records(rows, n=20, ascending=False),
        "outputs": {
            "contribution_csv": str(contribution_csv),
            "symbol_summary_csv": str(symbol_csv),
            "date_summary_csv": str(date_csv),
            "score_bucket_summary_csv": str(score_bucket_csv),
            "weight_bucket_summary_csv": str(weight_bucket_csv),
            "recent_return_bucket_summary_csv": str(recent_return_bucket_csv),
            "volatility_bucket_summary_csv": str(volatility_bucket_csv),
            "reversal_bucket_summary_csv": str(reversal_bucket_csv),
            "score_volatility_bucket_summary_csv": str(score_volatility_bucket_csv),
            "score_reversal_bucket_summary_csv": str(score_reversal_bucket_csv),
            "report_json": str(root / "v2_bad_month_attribution_report.json"),
            "report_md": str(root / "v2_bad_month_attribution_report.md"),
        },
        "boundary": {
            "research_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "attribution_method": "previous_target_weight_x_close_to_close_return",
            "risk_state_method": "previous_close_derived_recent_return_volatility_reversal",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_bad_month_attribution_report.json", report)
    _write_json(root / "v2_bad_month_attribution_manifest.json", {k: v for k, v in report.items() if k not in {"top_negative_contributors", "top_positive_contributors"}})
    _write_markdown(root / "v2_bad_month_attribution_report.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attribute bad months for daily_research v2 candidate backtests.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--matrix-run-tag", default=MATRIX_RUN_TAG)
    parser.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    parser.add_argument("--backtest-dir", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--month", default="")
    parser.add_argument("--recent-return-window", type=int, default=5)
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_bad_month_attribution(
        run_tag=args.run_tag,
        matrix_run_tag=args.matrix_run_tag,
        variant_id=args.variant_id,
        backtest_dir=args.backtest_dir or None,
        output_root=args.output_root or None,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        month=args.month,
        recent_return_window=args.recent_return_window,
        volatility_window=args.volatility_window,
        use_cache=not bool(args.no_cache),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status={report.get('status')} run_tag={args.run_tag} output_root={args.output_root or STUDIES_ROOT / args.run_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
