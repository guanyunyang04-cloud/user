"""Audit v2.1 daily metrics semantics beyond missingness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_tradeable_panel


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_metrics_semantics_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-03_v2_metrics_semantics_audit.md")
VALUATION_FIELDS = ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")


def run_v2_metrics_semantics_audit(
    *,
    root: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
    max_outliers: int = 50,
) -> dict[str, Any]:
    run_id = f"v2_metrics_semantics_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pit_manifest(root)
    panel = load_tradeable_panel(root, start_date=start_date, end_date=end_date, include_metrics=True)
    audit = build_v2_metrics_semantics_audit(panel, max_outliers=max_outliers)
    summary = summarize_v2_metrics_semantics_audit(
        manifest=manifest,
        panel=panel,
        audit=audit,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
    )
    markdown = render_v2_metrics_semantics_audit_markdown(summary, audit)

    for name, frame in audit.items():
        frame.to_csv(run_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.parent.mkdir(parents=True, exist_ok=True)
        research_log_path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_v2_metrics_semantics_audit(panel: pd.DataFrame, *, max_outliers: int = 50) -> dict[str, pd.DataFrame]:
    frame = _prepare_panel(panel)
    pctchg = _pctchg_comparison(frame)
    return {
        "pctchg_summary": _pctchg_summary(pctchg),
        "pctchg_yearly_summary": _pctchg_yearly_summary(pctchg),
        "pctchg_outliers": _pctchg_outliers(pctchg, max_outliers=max_outliers),
        "valuation_summary": _valuation_summary(frame),
        "valuation_yearly_summary": _valuation_yearly_summary(frame),
    }


def summarize_v2_metrics_semantics_audit(
    *,
    manifest: Mapping[str, Any],
    panel: pd.DataFrame,
    audit: Mapping[str, pd.DataFrame],
    run_id: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    pct_summary = audit["pctchg_summary"]
    valuation_summary = audit["valuation_summary"]
    p95_abs_diff = _first_float(pct_summary, "abs_diff_p95")
    compare_rows = _first_int(pct_summary, "compare_rows")
    pctchg_close_return_aligned = bool(compare_rows > 0 and p95_abs_diff <= 0.05)
    valuation_non_null_min = float(valuation_summary["non_null_rate"].min()) if not valuation_summary.empty else 0.0
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_id": manifest.get("snapshot_id", ""),
        "date_range_requested": {"start_date": start_date, "end_date": end_date},
        "tradeable_panel_rows": int(len(panel)),
        "code_count": int(panel["code"].nunique()) if not panel.empty and "code" in panel.columns else 0,
        "date_count": int(pd.to_datetime(panel["date"]).nunique()) if not panel.empty and "date" in panel.columns else 0,
        "pctchg_compare_rows": compare_rows,
        "pctchg_abs_diff_p95": p95_abs_diff,
        "pctchg_close_return_aligned": pctchg_close_return_aligned,
        "valuation_non_null_min": valuation_non_null_min,
        "valuation_missingness_ready": bool(valuation_non_null_min >= 0.95),
        "valuation_pit_timing_ready": False,
        "candidate_count": 0,
        "status": "metrics_semantics_timing_audit",
        "limitations": [
            "pctChg comparison uses unadjusted close-to-close returns from the v2 daily bars and can diverge around corporate actions or source-specific price bases.",
            "Valuation fields have coverage but this audit cannot prove Baostock's publication timestamp or absence of post-close revisions.",
            "Until PIT timing is proven, valuation fields should be lagged at least one trading day before factor or exposure use.",
            "This audit does not solve true market-cap, float-market-cap, or share-base availability.",
        ],
    }


def render_v2_metrics_semantics_audit_markdown(summary: Mapping[str, Any], audit: Mapping[str, pd.DataFrame]) -> str:
    lines = [
        "# v2.1 Metrics Semantics and Timing Audit",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- snapshot_id: `{summary.get('snapshot_id', '')}`",
        f"- status: `{summary.get('status', '')}`",
        f"- tradeable_panel_rows: `{summary.get('tradeable_panel_rows', 0)}`",
        f"- pctchg_compare_rows: `{summary.get('pctchg_compare_rows', 0)}`",
        f"- pctchg_abs_diff_p95: `{_fmt(summary.get('pctchg_abs_diff_p95'))}`",
        f"- pctchg_close_return_aligned: `{summary.get('pctchg_close_return_aligned')}`",
        f"- valuation_missingness_ready: `{summary.get('valuation_missingness_ready')}`",
        f"- valuation_pit_timing_ready: `{summary.get('valuation_pit_timing_ready')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        "",
        "## pctChg vs close-to-close",
        "",
        _markdown_table(audit["pctchg_summary"]),
        "",
        "## pctChg Yearly",
        "",
        _markdown_table(audit["pctchg_yearly_summary"], max_rows=20),
        "",
        "## Valuation Fields",
        "",
        _markdown_table(audit["valuation_summary"]),
        "",
        "## Interpretation",
        "",
        "The audit checks field semantics after the missingness audit. `pctChg` is compared with v2 close-to-close returns "
        "to detect source or adjustment differences. Valuation fields are treated as coverage-ready but not PIT-timing-ready: "
        "they should be lagged by at least one trading day until publication timing and revision behavior are independently verified. "
        "No strategy candidate is promoted by this audit.",
        "",
    ]
    return "\n".join(lines)


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "close", "pctChg", *VALUATION_FIELDS]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    frame = panel.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    for column in ["close", "pctChg", *VALUATION_FIELDS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[columns].sort_values(["code", "date"]).reset_index(drop=True)


def _pctchg_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "code", "prev_close", "close", "close_ret_pct", "pctChg", "diff", "abs_diff"])
    output = frame[["date", "code", "close", "pctChg"]].copy()
    output["prev_close"] = output.groupby("code")["close"].shift(1)
    output["close_ret_pct"] = (output["close"] / output["prev_close"] - 1.0) * 100.0
    output["diff"] = output["pctChg"] - output["close_ret_pct"]
    output["abs_diff"] = output["diff"].abs()
    valid = output.replace([np.inf, -np.inf], np.nan)
    return valid.loc[valid[["pctChg", "close_ret_pct", "abs_diff"]].notna().all(axis=1)].reset_index(drop=True)


def _pctchg_summary(pctchg: pd.DataFrame) -> pd.DataFrame:
    if pctchg.empty:
        return pd.DataFrame([_pctchg_stats(pctchg, scope="tradeable")])
    return pd.DataFrame([_pctchg_stats(pctchg, scope="tradeable")])


def _pctchg_yearly_summary(pctchg: pd.DataFrame) -> pd.DataFrame:
    if pctchg.empty:
        return pd.DataFrame(columns=["year", *list(_pctchg_stats(pctchg, scope="").keys())[1:]])
    frame = pctchg.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows = []
    for year, group in frame.groupby("year", sort=True):
        row = _pctchg_stats(group, scope="")
        row.pop("scope", None)
        rows.append({"year": int(year), **row})
    return pd.DataFrame(rows)


def _pctchg_stats(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    values = pd.to_numeric(frame["abs_diff"], errors="coerce") if "abs_diff" in frame.columns else pd.Series(dtype="float64")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "scope": scope,
        "compare_rows": int(len(values)),
        "abs_diff_mean": float(values.mean()) if len(values) else np.nan,
        "abs_diff_median": float(values.median()) if len(values) else np.nan,
        "abs_diff_p95": _quantile(values, 0.95),
        "abs_diff_p99": _quantile(values, 0.99),
        "abs_diff_max": float(values.max()) if len(values) else np.nan,
        "within_1bp_rate": float((values <= 0.01).mean()) if len(values) else 0.0,
        "within_5bp_rate": float((values <= 0.05).mean()) if len(values) else 0.0,
        "gt_100bp_rows": int((values > 1.0).sum()) if len(values) else 0,
    }


def _pctchg_outliers(pctchg: pd.DataFrame, *, max_outliers: int) -> pd.DataFrame:
    columns = ["date", "code", "prev_close", "close", "close_ret_pct", "pctChg", "diff", "abs_diff"]
    if pctchg.empty:
        return pd.DataFrame(columns=columns)
    outliers = pctchg.sort_values("abs_diff", ascending=False).head(max_outliers).copy()
    outliers["date"] = pd.to_datetime(outliers["date"]).dt.strftime("%Y-%m-%d")
    return outliers[columns]


def _valuation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=["field", "rows", "non_null_rate", "finite_rate", "zero_rate", "negative_rate", "change_rate", "corr_with_close_ret"])
    close_ret = frame.groupby("code")["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    for field in VALUATION_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce")
        finite = np.isfinite(values)
        previous = values.groupby(frame["code"]).shift(1)
        comparable = values.notna() & previous.notna()
        changed = comparable & (values != previous)
        value_ret = values.groupby(frame["code"]).pct_change().replace([np.inf, -np.inf], np.nan)
        corr_frame = pd.DataFrame({"value_ret": value_ret, "close_ret": close_ret}).dropna()
        rows.append(
            {
                "field": field,
                "rows": int(len(values)),
                "non_null_rate": float(values.notna().mean()) if len(values) else 0.0,
                "finite_rate": float(finite.mean()) if len(values) else 0.0,
                "zero_rate": float((values == 0).mean()) if len(values) else 0.0,
                "negative_rate": float((values < 0).mean()) if len(values) else 0.0,
                "change_rate": float(changed.sum() / comparable.sum()) if int(comparable.sum()) else np.nan,
                "corr_with_close_ret": float(corr_frame["value_ret"].corr(corr_frame["close_ret"])) if len(corr_frame) >= 2 else np.nan,
                "p01": _quantile(values.dropna(), 0.01),
                "p50": _quantile(values.dropna(), 0.50),
                "p99": _quantile(values.dropna(), 0.99),
                "min": float(values.min()) if values.notna().any() else np.nan,
                "max": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _valuation_yearly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["year", "field", "non_null_rate", "negative_rate", "zero_rate"])
    work = frame.copy()
    work["year"] = pd.to_datetime(work["date"]).dt.year
    rows: list[dict[str, Any]] = []
    for (year, field), group in _year_field_groups(work):
        values = pd.to_numeric(group[field], errors="coerce")
        rows.append(
            {
                "year": int(year),
                "field": field,
                "rows": int(len(values)),
                "non_null_rate": float(values.notna().mean()) if len(values) else 0.0,
                "negative_rate": float((values < 0).mean()) if len(values) else 0.0,
                "zero_rate": float((values == 0).mean()) if len(values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _year_field_groups(frame: pd.DataFrame):
    for year, group in frame.groupby("year", sort=True):
        for field in VALUATION_FIELDS:
            yield (year, field), group


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.quantile(q)) if len(clean) else np.nan


def _first_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    value = frame.iloc[0][column]
    return float(value) if pd.notna(value) else np.nan


def _first_int(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    value = frame.iloc[0][column]
    return int(value) if pd.notna(value) else 0


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Snapshot root or v2 root containing latest_manifest.json.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    parser.add_argument("--max-outliers", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_metrics_semantics_audit(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
        max_outliers=int(args.max_outliers),
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
