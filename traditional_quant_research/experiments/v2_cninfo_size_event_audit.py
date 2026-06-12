"""Audit AKShare/CNInfo share-change events for free daily_size reconstruction."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.size_source import normalize_cninfo_share_change_events, normalize_project_symbol


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_cninfo_size_event_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-03_v2_cninfo_size_event_audit.md")
DEFAULT_SYMBOLS = ("600000.SH", "000001.SZ", "000002.SZ", "601398.SH")
DEFAULT_YEARS = tuple(range(2016, 2027))

EVENT_AUDIT_COLUMNS = [
    "code",
    "status",
    "raw_row_count",
    "event_count",
    "first_event_date",
    "last_event_date",
    "raw_columns",
    "fields_found",
    "total_share_coverage",
    "float_share_coverage",
    "free_share_coverage",
    "total_share_positive_rate",
    "float_share_positive_rate",
    "share_unit",
    "pit_semantics",
    "usable_for_reconstruction",
    "promotion_evidence_ready",
    "failure_type",
    "message",
]

_DATE_ALIASES = ("变动日期", "公告日期", "截止日期", "日期", "change_date", "date", "end_date")
_PUBLISH_ALIASES = ("公告日期", "披露日期", "发布日期", "publish_date", "announcement_date", "ann_date")
_TOTAL_SHARE_ALIASES = ("总股本", "总股本(股)", "总股本（股）", "total_share", "total_shares")
_FLOAT_SHARE_ALIASES = ("流通股", "流通股本", "流通A股", "无限售流通股", "已流通股份", "流通股份", "float_share", "float_shares")
_FREE_SHARE_ALIASES = ("自由流通股", "free_share", "free_shares")


def run_v2_cninfo_size_event_audit(
    *,
    symbols: str | Sequence[str] = DEFAULT_SYMBOLS,
    years: str | Sequence[int] = DEFAULT_YEARS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    akshare_module: Any | None = None,
) -> dict[str, Any]:
    """Fetch and audit CNInfo share-change events without writing daily_size cache."""

    run_id = f"v2_cninfo_size_event_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbol_list = _normalize_symbols(symbols)
    year_list = _normalize_years(years)
    start_date = f"{min(year_list)}0101" if year_list else "20160101"
    end_date = f"{max(year_list)}1231" if year_list else "20261231"
    ak, ak_error = _load_module("akshare", akshare_module)

    audit_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    raw_field_rows: list[dict[str, Any]] = []
    for code in symbol_list:
        row, events, raw_fields = audit_cninfo_symbol(
            code,
            akshare_module=ak,
            import_error=ak_error,
            start_date=start_date,
            end_date=end_date,
        )
        audit_rows.append(row)
        if not events.empty:
            event_frames.append(events)
        raw_field_rows.extend(raw_fields)

    symbol_audit = pd.DataFrame(audit_rows, columns=EVENT_AUDIT_COLUMNS)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else _empty_events_frame()
    raw_fields = pd.DataFrame(raw_field_rows, columns=["code", "field", "matched_role"])
    summary = summarize_cninfo_event_audit(
        symbol_audit,
        run_id=run_id,
        symbols=symbol_list,
        years=year_list,
        start_date=start_date,
        end_date=end_date,
    )
    markdown = render_cninfo_event_audit_markdown(summary, symbol_audit)

    symbol_audit.to_csv(run_dir / "cninfo_symbol_event_audit.csv", index=False, encoding="utf-8-sig")
    events.to_csv(run_dir / "cninfo_share_events.csv", index=False, encoding="utf-8-sig")
    raw_fields.to_csv(run_dir / "cninfo_raw_field_map.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def audit_cninfo_symbol(
    code: str,
    *,
    akshare_module: Any | None,
    import_error: str,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, str]]]:
    normalized = normalize_project_symbol(code)
    if akshare_module is None:
        return (
            _audit_row(
                code=normalized,
                status="package_missing",
                failure_type="package_missing",
                message=import_error,
            ),
            _empty_events_frame(),
            [],
        )
    try:
        raw = akshare_module.stock_share_change_cninfo(
            symbol=normalized.split(".", 1)[0],
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _audit_row(
                code=normalized,
                status="failed",
                failure_type=type(exc).__name__,
                message=str(exc),
            ),
            _empty_events_frame(),
            [],
        )
    raw = pd.DataFrame() if raw is None else raw.copy()
    raw_fields = _raw_field_rows(normalized, raw)
    events = normalize_cninfo_share_change_events(raw, code=normalized)
    if raw.empty:
        return (
            _audit_row(
                code=normalized,
                status="empty_raw",
                raw_columns="",
                failure_type="empty_raw",
                message="stock_share_change_cninfo returned no raw rows.",
            ),
            events,
            raw_fields,
        )
    if events.empty:
        return (
            _audit_row(
                code=normalized,
                status="empty_events",
                raw_row_count=len(raw),
                raw_columns=",".join(str(column) for column in raw.columns),
                fields_found=",".join(sorted({item["matched_role"] for item in raw_fields})),
                pit_semantics=_pit_semantics(raw),
                failure_type="empty_events",
                message="No parseable date/share events after normalization.",
            ),
            events,
            raw_fields,
        )
    total = pd.to_numeric(events["total_share"], errors="coerce")
    floating = pd.to_numeric(events["float_share"], errors="coerce")
    free = pd.to_numeric(events["free_share"], errors="coerce")
    total_coverage = float(total.notna().mean()) if len(events) else 0.0
    float_coverage = float(floating.notna().mean()) if len(events) else 0.0
    total_positive = float((total.dropna() > 0).sum() / len(events)) if len(events) else 0.0
    float_positive = float((floating.dropna() > 0).sum() / len(events)) if len(events) else 0.0
    usable = bool(total_coverage > 0 and float_coverage > 0 and total_positive > 0 and float_positive > 0)
    pit = _pit_semantics(raw)
    return (
        _audit_row(
            code=normalized,
            status="passed" if usable else "fields_incomplete",
            raw_row_count=len(raw),
            event_count=len(events),
            first_event_date=_date_or_empty(events["date"].min()),
            last_event_date=_date_or_empty(events["date"].max()),
            raw_columns=",".join(str(column) for column in raw.columns),
            fields_found=",".join(sorted({item["matched_role"] for item in raw_fields})),
            total_share_coverage=total_coverage,
            float_share_coverage=float_coverage,
            free_share_coverage=float(free.notna().mean()) if len(events) else 0.0,
            total_share_positive_rate=total_positive,
            float_share_positive_rate=float_positive,
            share_unit="shares",
            pit_semantics=pit,
            usable_for_reconstruction=usable,
            promotion_evidence_ready=False,
            failure_type="" if usable else "fields_incomplete",
            message=(
                "Events can be used for diagnostic reconstruction; current cross-check and coverage audit still required."
                if usable
                else "Total or float share coverage/positive-rate is insufficient."
            ),
        ),
        events,
        raw_fields,
    )


def summarize_cninfo_event_audit(
    symbol_audit: pd.DataFrame,
    *,
    run_id: str,
    symbols: Sequence[str],
    years: Sequence[int],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    usable = symbol_audit.loc[symbol_audit["usable_for_reconstruction"].eq(True)] if not symbol_audit.empty else pd.DataFrame()
    ready = symbol_audit.loc[symbol_audit["promotion_evidence_ready"].eq(True)] if not symbol_audit.empty else pd.DataFrame()
    failure_types = (
        symbol_audit.loc[symbol_audit["failure_type"].astype(str) != "", "failure_type"].astype(str).value_counts().to_dict()
        if not symbol_audit.empty and "failure_type" in symbol_audit.columns
        else {}
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": list(symbols),
        "years": [int(year) for year in years],
        "start_date": start_date,
        "end_date": end_date,
        "symbol_count": int(len(symbol_audit)),
        "usable_for_reconstruction_count": int(len(usable)),
        "promotion_evidence_ready_count": int(len(ready)),
        "failure_types": failure_types,
        "source_decision": "diagnostic_ready_for_cross_check" if len(usable) else "free_source_insufficient_for_size_gate",
        "evidence_grade": "diagnostic",
        "candidate_count": 0,
        "limitations": [
            "CNInfo share events alone do not prove daily PIT market cap.",
            "Promotion evidence requires current cross-check and full daily_size coverage audit.",
            "No formal daily_size cache is written by this audit.",
        ],
    }


def render_cninfo_event_audit_markdown(summary: Mapping[str, Any], symbol_audit: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# V2 CNInfo Size Event Audit",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- source_decision: `{summary.get('source_decision', '')}`",
            f"- evidence_grade: `{summary.get('evidence_grade', '')}`",
            f"- usable_for_reconstruction_count: `{summary.get('usable_for_reconstruction_count', 0)}`",
            f"- promotion_evidence_ready_count: `{summary.get('promotion_evidence_ready_count', 0)}`",
            f"- candidate_count: `{summary.get('candidate_count', 0)}`",
            "",
            "## Symbol Audit",
            "",
            _markdown_table(symbol_audit),
            "",
            "## Interpretation",
            "",
            "This audit verifies whether CNInfo share-change events can support diagnostic size reconstruction. "
            "It does not prove full PIT daily_size readiness and does not promote any strategy candidate.",
            "",
        ]
    )


def _audit_row(**kwargs: Any) -> dict[str, Any]:
    row = {column: "" for column in EVENT_AUDIT_COLUMNS}
    row.update(
        {
            "raw_row_count": 0,
            "event_count": 0,
            "total_share_coverage": 0.0,
            "float_share_coverage": 0.0,
            "free_share_coverage": 0.0,
            "total_share_positive_rate": 0.0,
            "float_share_positive_rate": 0.0,
            "usable_for_reconstruction": False,
            "promotion_evidence_ready": False,
        }
    )
    row.update(kwargs)
    return row


def _raw_field_rows(code: str, raw: pd.DataFrame) -> list[dict[str, str]]:
    roles = {
        "date": _DATE_ALIASES,
        "publish_date": _PUBLISH_ALIASES,
        "total_share": _TOTAL_SHARE_ALIASES,
        "float_share": _FLOAT_SHARE_ALIASES,
        "free_share": _FREE_SHARE_ALIASES,
    }
    rows: list[dict[str, str]] = []
    for field in raw.columns:
        for role, aliases in roles.items():
            if _field_matches(field, aliases):
                rows.append({"code": code, "field": str(field), "matched_role": role})
    return rows


def _pit_semantics(raw: pd.DataFrame) -> str:
    if raw.empty:
        return "unknown"
    if any(_field_matches(column, _PUBLISH_ALIASES) for column in raw.columns):
        return "publication_date_available"
    if any(_field_matches(column, _DATE_ALIASES) for column in raw.columns):
        return "effective_date_only"
    return "unknown"


def _field_matches(field: Any, aliases: Sequence[str]) -> bool:
    text = str(field).strip().lower()
    return any(text == str(alias).strip().lower() for alias in aliases)


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "code", "total_share", "float_share", "free_share", "source", "source_trade_date"])


def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
    raw = [item.strip() for item in symbols.split(",")] if isinstance(symbols, str) else [str(item).strip() for item in symbols]
    return [normalize_project_symbol(item) for item in raw if normalize_project_symbol(item)]


def _normalize_years(years: str | Sequence[int]) -> list[int]:
    raw = [item.strip() for item in years.split(",")] if isinstance(years, str) else [str(item).strip() for item in years]
    output = sorted({int(item) for item in raw if item})
    return output


def _load_module(name: str, module: Any | None) -> tuple[Any | None, str]:
    if module is not None:
        return module, ""
    try:
        return importlib.import_module(name), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _date_or_empty(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d")


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "nan")
    return view.to_markdown(index=False)


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
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_cninfo_size_event_audit(
        symbols=args.symbols,
        years=args.years,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
