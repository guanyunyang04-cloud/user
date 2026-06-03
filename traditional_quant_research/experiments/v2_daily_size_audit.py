"""Audit v2.2 daily size coverage and missingness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import (
    load_daily_universe,
    load_pit_daily_size,
    load_pit_manifest,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_daily_size_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_v2_daily_size_audit.md")
SIZE_FIELDS = ("total_market_cap", "float_market_cap", "total_share", "float_share", "free_share")
UNIT_FIELDS = ("market_cap_unit", "share_unit")


def run_v2_daily_size_audit(
    *,
    root: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    run_id = f"v2_daily_size_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_pit_manifest(root)
    universe = load_daily_universe(root, start_date=start_date, end_date=end_date)
    size = load_pit_daily_size(root, start_date=start_date, end_date=end_date)
    audit = build_daily_size_audit(universe, size)
    summary = summarize_daily_size_audit(
        manifest=manifest,
        universe=universe,
        size=size,
        audit=audit,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
    )
    markdown = render_daily_size_audit_markdown(summary, audit)

    audit["field_summary"].to_csv(run_dir / "field_summary.csv", index=False, encoding="utf-8-sig")
    audit["yearly_summary"].to_csv(run_dir / "yearly_summary.csv", index=False, encoding="utf-8-sig")
    audit["daily_summary"].to_csv(run_dir / "daily_summary.csv", index=False, encoding="utf-8-sig")
    audit["source_unit_summary"].to_csv(run_dir / "source_unit_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.parent.mkdir(parents=True, exist_ok=True)
        research_log_path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_daily_size_audit(universe: pd.DataFrame, size: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keys = _expected_keys(universe)
    merged = _merge_expected_size(keys, size)
    return {
        "field_summary": _field_summary(merged),
        "yearly_summary": _yearly_summary(merged),
        "daily_summary": _daily_summary(merged),
        "source_unit_summary": _source_unit_summary(merged),
    }


def summarize_daily_size_audit(
    *,
    manifest: Mapping[str, Any],
    universe: pd.DataFrame,
    size: pd.DataFrame,
    audit: Mapping[str, pd.DataFrame],
    run_id: str,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    field_summary = audit["field_summary"]
    tradeable = field_summary[field_summary["scope"] == "tradeable"].copy()
    min_tradeable_coverage = float(tradeable["coverage_rate"].min()) if not tradeable.empty else 0.0
    source_unit_summary = audit["source_unit_summary"]
    required_units_present = bool(
        not source_unit_summary.empty
        and source_unit_summary["market_cap_unit"].notna().all()
        and source_unit_summary["share_unit"].notna().all()
    )
    table_present = bool(not size.empty)
    ready = bool(table_present and min_tradeable_coverage >= 0.95 and required_units_present)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_id": manifest.get("snapshot_id", ""),
        "date_range_requested": {"start_date": start_date, "end_date": end_date},
        "universe_rows": int(len(universe)),
        "tradeable_rows": int(universe["is_tradeable"].sum()) if "is_tradeable" in universe.columns else 0,
        "size_rows": int(len(size)),
        "size_date_count": int(pd.to_datetime(size["date"]).nunique()) if not size.empty and "date" in size.columns else 0,
        "size_code_count": int(size["code"].nunique()) if not size.empty and "code" in size.columns else 0,
        "min_tradeable_coverage": min_tradeable_coverage,
        "required_units_present": required_units_present,
        "daily_size_ready_for_research": ready,
        "candidate_count": 0,
        "status": "daily_size_missingness_audit" if table_present else "daily_size_absent",
        "limitations": [
            "Coverage and units do not prove PIT timing or revision behavior.",
            "Size fields must come from a live-validated external source before strategy promotion.",
            "This audit does not fetch or build daily_size; it only audits a snapshot table if present.",
        ],
    }


def render_daily_size_audit_markdown(summary: Mapping[str, Any], audit: Mapping[str, pd.DataFrame]) -> str:
    lines = [
        "# v2.2 Daily Size Audit",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- snapshot_id: `{summary.get('snapshot_id', '')}`",
        f"- status: `{summary.get('status', '')}`",
        f"- universe_rows: `{summary.get('universe_rows', 0)}`",
        f"- tradeable_rows: `{summary.get('tradeable_rows', 0)}`",
        f"- size_rows: `{summary.get('size_rows', 0)}`",
        f"- min_tradeable_coverage: `{_fmt(summary.get('min_tradeable_coverage'))}`",
        f"- required_units_present: `{summary.get('required_units_present')}`",
        f"- daily_size_ready_for_research: `{summary.get('daily_size_ready_for_research')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        "",
        "## Field Summary",
        "",
        _markdown_table(audit["field_summary"]),
        "",
        "## Source/Unit Summary",
        "",
        _markdown_table(audit["source_unit_summary"]),
        "",
        "## Yearly Summary",
        "",
        _markdown_table(audit["yearly_summary"]),
        "",
        "## Interpretation",
        "",
        "This audit checks whether the optional v2.2 daily_size table has enough coverage and unit metadata "
        "to become a research input. It does not validate PIT timing, does not fetch external data, "
        "and does not promote any strategy candidate.",
        "",
    ]
    return "\n".join(lines)


def _expected_keys(universe: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "is_tradeable"]
    if universe.empty:
        return pd.DataFrame(columns=columns)
    keys = universe.copy()
    for column in columns:
        if column not in keys.columns:
            keys[column] = False if column == "is_tradeable" else pd.NA
    keys["date"] = pd.to_datetime(keys["date"])
    keys["code"] = keys["code"].astype(str)
    keys["is_tradeable"] = keys["is_tradeable"].fillna(False).astype(bool)
    return keys[columns].drop_duplicates(["date", "code"]).reset_index(drop=True)


def _merge_expected_size(keys: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", *SIZE_FIELDS, *UNIT_FIELDS, "source", "has_size_row"]
    if keys.empty:
        return pd.DataFrame(columns=["date", "code", "is_tradeable", *SIZE_FIELDS, *UNIT_FIELDS, "source", "has_size_row"])
    size_frame = size.copy()
    if size_frame.empty:
        size_frame = pd.DataFrame(columns=columns[:-1])
    for column in ["date", "code", *SIZE_FIELDS, *UNIT_FIELDS, "source"]:
        if column not in size_frame.columns:
            size_frame[column] = pd.NA
    size_frame["date"] = pd.to_datetime(size_frame["date"])
    size_frame["code"] = size_frame["code"].astype(str)
    for field in SIZE_FIELDS:
        size_frame[field] = pd.to_numeric(size_frame[field], errors="coerce")
    size_frame["has_size_row"] = True
    output = keys.merge(size_frame[columns], on=["date", "code"], how="left")
    output["has_size_row"] = output["has_size_row"].eq(True)
    return output


def _field_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, frame in _scopes(merged):
        expected_rows = int(len(frame))
        size_row_coverage = float(frame["has_size_row"].mean()) if expected_rows else 0.0
        for field in SIZE_FIELDS:
            values = pd.to_numeric(frame[field], errors="coerce") if field in frame.columns else pd.Series(dtype="float64")
            non_null_rows = int(values.notna().sum())
            finite_rows = int(np.isfinite(values.dropna()).sum()) if non_null_rows else 0
            positive_rows = int((values.dropna() > 0).sum()) if non_null_rows else 0
            rows.append(
                {
                    "scope": scope,
                    "field": field,
                    "expected_rows": expected_rows,
                    "size_row_coverage": size_row_coverage,
                    "non_null_rows": non_null_rows,
                    "coverage_rate": non_null_rows / expected_rows if expected_rows else 0.0,
                    "finite_rows": finite_rows,
                    "positive_rows": positive_rows,
                    "mean": float(values.mean()) if non_null_rows else np.nan,
                    "min": float(values.min()) if non_null_rows else np.nan,
                    "max": float(values.max()) if non_null_rows else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _yearly_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["year", "scope", "expected_rows", "size_row_coverage", "field", "coverage_rate"])
    frame = merged.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows: list[dict[str, Any]] = []
    for (year, scope), group in _with_scope(frame).groupby(["year", "scope"]):
        expected_rows = int(len(group))
        size_row_coverage = float(group["has_size_row"].mean()) if expected_rows else 0.0
        for field in SIZE_FIELDS:
            values = pd.to_numeric(group[field], errors="coerce") if field in group.columns else pd.Series(dtype="float64")
            rows.append(
                {
                    "year": int(year),
                    "scope": str(scope),
                    "expected_rows": expected_rows,
                    "size_row_coverage": size_row_coverage,
                    "field": field,
                    "coverage_rate": float(values.notna().mean()) if expected_rows else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _daily_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["date", "expected_rows", "tradeable_rows", "size_row_coverage"])
    grouped = (
        merged.groupby("date")
        .agg(
            expected_rows=("code", "count"),
            tradeable_rows=("is_tradeable", "sum"),
            size_row_coverage=("has_size_row", "mean"),
        )
        .reset_index()
    )
    grouped["date"] = pd.to_datetime(grouped["date"]).dt.strftime("%Y-%m-%d")
    return grouped


def _source_unit_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty or not merged["has_size_row"].any():
        return pd.DataFrame(columns=["source", "market_cap_unit", "share_unit", "rows", "code_count", "date_count"])
    frame = merged.loc[merged["has_size_row"]].copy()
    for column in ["source", *UNIT_FIELDS]:
        if column not in frame.columns:
            frame[column] = pd.NA
    grouped = (
        frame.groupby(["source", "market_cap_unit", "share_unit"], dropna=False)
        .agg(rows=("code", "count"), code_count=("code", "nunique"), date_count=("date", "nunique"))
        .reset_index()
    )
    return grouped


def _scopes(merged: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if merged.empty:
        empty = pd.DataFrame(columns=["date", "code", "is_tradeable", *SIZE_FIELDS, *UNIT_FIELDS, "source", "has_size_row"])
        return [("all", empty), ("tradeable", empty)]
    return [("all", merged), ("tradeable", merged.loc[merged["is_tradeable"]].copy())]


def _with_scope(frame: pd.DataFrame) -> pd.DataFrame:
    all_scope = frame.copy()
    all_scope["scope"] = "all"
    tradeable_scope = frame.loc[frame["is_tradeable"]].copy()
    tradeable_scope["scope"] = "tradeable"
    return pd.concat([all_scope, tradeable_scope], ignore_index=True)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_daily_size_audit(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
