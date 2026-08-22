"""Research Event Update: report_prepare responsibilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    END_DATE,
    FORECAST_COLUMNS,
    REPORT_COLUMNS,
    REPORT_EASTMONEY_START,
    REPORT_RC_FIELDS,
    START_DATE,
    ResearchEventUpdateError,
)
from .context import (
    _first,
    _json_list,
    _next_open_date,
    _normalized_institution,
    _normalized_text,
    _numeric,
    _open_dates,
    _read_state,
    _report_id,
    _report_rc_page_path,
    _report_request_dates,
    _report_source_key,
    _report_year_source_coverage,
    _runtime,
    _text,
    _workspace,
    _write_parquet,
    _write_state,
)
from .report_download import (
    _eastmoney_report_paths,
)


def _empty_tushare_report_year() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return (
        pd.DataFrame(columns=REPORT_COLUMNS),
        pd.DataFrame(columns=FORECAST_COLUMNS),
        {
            "raw_row_count": 0,
            "valid_prediction_row_count": 0,
            "unique_report_count": 0,
            "forecast_row_count": 0,
            "duplicate_prediction_row_count": 0,
            "symbol_count": 0,
            "institution_count": 0,
            "analyst_count": 0,
            "earliest_report_date": "",
            "latest_report_date": "",
            "field_nonnull_rate": {},
        },
    )


def _normalize_tushare_report_input(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    data = raw.copy()
    data["symbol"] = data.get("ts_code", "").fillna("").astype(str).str.upper()
    data["source_date"] = pd.to_datetime(data.get("report_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    data["title"] = data.get("report_title", "").fillna("").astype(str)
    data["institution"] = data.get("org_name", "").fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["normalized_institution"] = data["institution"].map(_normalized_institution)
    data["source_report_key"] = [
        _report_source_key(*values)
        for values in zip(data["symbol"], data["source_date"], data["title"], data["institution"], strict=True)
    ]
    data["report_id"] = data["source_report_key"].map(_report_id)
    valid = (
        data["symbol"].str.endswith((".SH", ".SZ"))
        & data["source_date"].between(START_DATE, END_DATE)
        & data["normalized_title"].ne("")
        & data["normalized_institution"].ne("")
    )
    data = data.loc[valid].reset_index(drop=True)
    data["feature_available_date"] = _next_open_date(data["source_date"], open_dates)
    return data


def _tushare_report_rows(data: pd.DataFrame) -> pd.DataFrame:
    report_rows: list[dict[str, Any]] = []
    for source_key, group in data.groupby("source_report_key", sort=True):
        analysts = [_text(item) for item in group.get("author_name", []) if _text(item)]
        disagreements = any(
            group[column].dropna().astype(str).nunique() > 1
            for column in (
                "report_title",
                "org_name",
                "rating",
                "min_price",
                "max_price",
            )
            if column in group
        )
        report_rows.append(
            {
                "report_id": _report_id(str(source_key)),
                "source_report_key": source_key,
                "symbol": _first(group["symbol"]),
                "trade_date": _first(group["source_date"]),
                "source_date": _first(group["source_date"]),
                "feature_available_date": _first(group["feature_available_date"]),
                "title": _first(group["title"]),
                "normalized_title": _first(group["normalized_title"]),
                "institution": _first(group["institution"]),
                "normalized_institution": _first(group["normalized_institution"]),
                "analyst": ";".join(sorted(set(analysts))),
                "report_type": _first(group.get("report_type", [])),
                "classification": _first(group.get("classify", [])),
                "rating": _first(group.get("rating", [])),
                "rating_change": "",
                "target_price_min": _numeric(_first(group.get("min_price", []))),
                "target_price_max": _numeric(_first(group.get("max_price", []))),
                "tushare_present": True,
                "eastmoney_present": False,
                "tushare_source_ids": json.dumps([f"report_rc:{source_key}"], ensure_ascii=False),
                "eastmoney_info_codes": "[]",
                "url": "",
                "pdf_file_size_kb": math.nan,
                "pdf_pages": math.nan,
                "source_disagreement": bool(disagreements),
                "identity_conflict_reason": ("analyst_union" if len(set(analysts)) > 1 and len(group) > 1 else ""),
                "source": "tushare_report_rc",
            }
        )
    return pd.DataFrame(report_rows, columns=REPORT_COLUMNS)


def _tushare_forecast_rows(data: pd.DataFrame) -> pd.DataFrame:
    forecast = pd.DataFrame(
        {
            "report_id": data["report_id"],
            "symbol": data["symbol"],
            "trade_date": data["source_date"],
            "source_date": data["source_date"],
            "feature_available_date": data["feature_available_date"],
            "forecast_quarter": data.get("quarter", "").fillna("").astype(str),
            "forecast_year": pd.to_numeric(
                data.get("quarter", "").astype(str).str.extract(r"(20\d{2})")[0],
                errors="coerce",
            ),
            "operating_revenue": pd.to_numeric(data.get("op_rt"), errors="coerce"),
            "operating_profit": pd.to_numeric(data.get("op_pr"), errors="coerce"),
            "total_profit": pd.to_numeric(data.get("tp"), errors="coerce"),
            "net_profit": pd.to_numeric(data.get("np"), errors="coerce"),
            "eps": pd.to_numeric(data.get("eps"), errors="coerce"),
            "pe": pd.to_numeric(data.get("pe"), errors="coerce"),
            "research_development": pd.to_numeric(data.get("rd"), errors="coerce"),
            "roe": pd.to_numeric(data.get("roe"), errors="coerce"),
            "ev_ebitda": pd.to_numeric(data.get("ev_ebitda"), errors="coerce"),
            "source_disagreement": False,
            "source": "tushare_report_rc",
        }
    )
    forecast = forecast.loc[forecast["forecast_quarter"].str.strip().ne("")].copy()
    value_columns = list(FORECAST_COLUMNS[7:16])
    forecast["_complete"] = forecast[value_columns].notna().sum(axis=1)
    duplicate_groups = forecast.groupby(["report_id", "forecast_quarter", "source"], dropna=False).size()
    conflict_keys = set(duplicate_groups.loc[duplicate_groups.gt(1)].index.tolist())
    forecast = forecast.sort_values(["report_id", "forecast_quarter", "_complete"], kind="stable").drop_duplicates(
        ["report_id", "forecast_quarter", "source"], keep="last"
    )
    forecast["source_disagreement"] = [
        (row.report_id, row.forecast_quarter, row.source) in conflict_keys for row in forecast.itertuples(index=False)
    ]
    forecast = forecast.loc[:, list(FORECAST_COLUMNS)].reset_index(drop=True)
    return forecast


def _tushare_report_statistics(
    *,
    raw: pd.DataFrame,
    data: pd.DataFrame,
    reports: pd.DataFrame,
    forecast: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "raw_row_count": len(raw),
        "valid_prediction_row_count": len(data),
        "unique_report_count": len(reports),
        "forecast_row_count": len(forecast),
        "duplicate_prediction_row_count": int(len(data) - len(forecast)),
        "symbol_count": int(reports["symbol"].nunique()),
        "institution_count": int(reports["institution"].nunique()),
        "analyst_count": len({name for value in reports["analyst"] for name in str(value).split(";") if name}),
        "earliest_report_date": str(reports["source_date"].min()),
        "latest_report_date": str(reports["source_date"].max()),
        "field_nonnull_rate": {
            column: float(data[column].notna().mean()) for column in REPORT_RC_FIELDS if column in data
        },
    }


def _normalize_tushare_report_year(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if raw.empty:
        return _empty_tushare_report_year()
    data = _normalize_tushare_report_input(raw, open_dates=open_dates)
    reports = _tushare_report_rows(data)
    forecast = _tushare_forecast_rows(data)
    statistics = _tushare_report_statistics(
        raw=raw,
        data=data,
        reports=reports,
        forecast=forecast,
    )
    return reports, forecast, statistics


def _normalize_eastmoney_reports(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    data = raw.copy()
    data["symbol"] = data.get("_query_symbol", "").fillna("").astype(str).str.upper()
    data["source_date"] = pd.to_datetime(data.get("publishDate"), errors="coerce").dt.strftime("%Y-%m-%d")
    data["title"] = data.get("title", "").fillna("").astype(str)
    institution = data.get("orgName", data.get("orgSName", ""))
    data["institution"] = institution.fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["normalized_institution"] = data["institution"].map(_normalized_institution)
    data["source_report_key"] = [
        _report_source_key(*values)
        for values in zip(data["symbol"], data["source_date"], data["title"], data["institution"], strict=True)
    ]
    data["feature_available_date"] = _next_open_date(data["source_date"], open_dates)
    valid = (
        data["source_date"].between(REPORT_EASTMONEY_START, END_DATE)
        & data["normalized_title"].ne("")
        & data["normalized_institution"].ne("")
    )
    data = data.loc[valid].copy()
    rows: list[dict[str, Any]] = []
    for source_key, group in data.groupby("source_report_key", sort=True):
        analysts: set[str] = set()
        for value in group.get("researcher", []):
            analysts.update(item.strip() for item in _text(value).split(",") if item.strip())
        info_codes = [_text(value) for value in group.get("infoCode", []) if _text(value)]
        info_code = info_codes[0] if info_codes else ""
        rating_values = group.get("sRatingName", group.get("emRatingName", []))
        rows.append(
            {
                "report_id": _report_id(str(source_key)),
                "source_report_key": source_key,
                "symbol": _first(group["symbol"]),
                "trade_date": _first(group["source_date"]),
                "source_date": _first(group["source_date"]),
                "feature_available_date": _first(group["feature_available_date"]),
                "title": _first(group["title"]),
                "normalized_title": _first(group["normalized_title"]),
                "institution": _first(group["institution"]),
                "normalized_institution": _first(group["normalized_institution"]),
                "analyst": ";".join(sorted(analysts)),
                "report_type": _text(_first(group.get("reportType", []))),
                "classification": _text(_first(group.get("column", []))),
                "rating": _text(_first(rating_values)),
                "rating_change": _text(_first(group.get("ratingChange", []))),
                "target_price_min": _numeric(_first(group.get("indvAimPriceL", []))),
                "target_price_max": _numeric(_first(group.get("indvAimPriceT", []))),
                "tushare_present": False,
                "eastmoney_present": True,
                "tushare_source_ids": "[]",
                "eastmoney_info_codes": _json_list(info_codes),
                "url": (f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf" if info_code else ""),
                "pdf_file_size_kb": _numeric(_first(group.get("attachSize", []))),
                "pdf_pages": _numeric(_first(group.get("attachPages", []))),
                "source_disagreement": bool(len(set(info_codes)) > 1),
                "identity_conflict_reason": ("multiple_eastmoney_info_codes" if len(set(info_codes)) > 1 else ""),
                "source": "eastmoney_report_metadata",
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def _merge_reports(
    tushare: pd.DataFrame,
    eastmoney: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([tushare, eastmoney], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for source_key, group in combined.groupby("source_report_key", sort=True):
        ts = group.loc[group["tushare_present"].astype(bool)]
        em = group.loc[group["eastmoney_present"].astype(bool)]

        def preferred(
            column: str,
            ts_frame: pd.DataFrame = ts,
            em_frame: pd.DataFrame = em,
        ) -> Any:
            left = _first(ts_frame[column]) if column in ts_frame else ""
            return left if _text(left) else _first(em_frame[column])

        disagreements = bool(group["source_disagreement"].astype(bool).any())
        for column in ("rating", "target_price_min", "target_price_max"):
            left = {_text(value) for value in ts.get(column, []) if _text(value)}
            right = {_text(value) for value in em.get(column, []) if _text(value)}
            disagreements = disagreements or bool(left and right and left.isdisjoint(right))
        analysts = {item for value in group["analyst"] for item in str(value).split(";") if item}
        reasons = {_text(value) for value in group["identity_conflict_reason"] if _text(value)}
        rows.append(
            {
                "report_id": _report_id(str(source_key)),
                "source_report_key": source_key,
                "symbol": preferred("symbol"),
                "trade_date": preferred("source_date"),
                "source_date": preferred("source_date"),
                "feature_available_date": preferred("feature_available_date"),
                "title": preferred("title"),
                "normalized_title": preferred("normalized_title"),
                "institution": preferred("institution"),
                "normalized_institution": preferred("normalized_institution"),
                "analyst": ";".join(sorted(analysts)),
                "report_type": preferred("report_type"),
                "classification": preferred("classification"),
                "rating": preferred("rating"),
                "rating_change": preferred("rating_change"),
                "target_price_min": _numeric(preferred("target_price_min")),
                "target_price_max": _numeric(preferred("target_price_max")),
                "tushare_present": not ts.empty,
                "eastmoney_present": not em.empty,
                "tushare_source_ids": _json_list(
                    item for value in ts.get("tushare_source_ids", []) for item in json.loads(value or "[]")
                ),
                "eastmoney_info_codes": _json_list(
                    item for value in em.get("eastmoney_info_codes", []) for item in json.loads(value or "[]")
                ),
                "url": preferred("url"),
                "pdf_file_size_kb": _numeric(preferred("pdf_file_size_kb")),
                "pdf_pages": _numeric(preferred("pdf_pages")),
                "source_disagreement": disagreements,
                "identity_conflict_reason": ";".join(sorted(reasons)),
                "source": (
                    "tushare_report_rc+eastmoney_report_metadata"
                    if not ts.empty and not em.empty
                    else _first(group["source"])
                ),
            }
        )
    return (
        pd.DataFrame(rows, columns=REPORT_COLUMNS)
        .sort_values(["trade_date", "symbol", "report_id"], kind="stable")
        .reset_index(drop=True)
    )


def _report_year_task_coverage(
    year: int,
    *,
    request_dates: tuple[str, ...],
    task_days: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, int]]:
    expected_dates = tuple(item for item in request_dates if item.startswith(f"{year}-"))
    year_tasks = {report_date: dict(task_days.get(report_date, {}) or {}) for report_date in expected_dates}
    terminal = sum(item.get("status") in {"observed", "confirmed_empty"} for item in year_tasks.values())
    failed = sum(item.get("status") == "failed" for item in year_tasks.values())
    pending = len(expected_dates) - terminal - failed
    if terminal != len(expected_dates):
        raise ResearchEventUpdateError(
            f"tushare_report_year_tasks_incomplete:{year}:terminal={terminal}:failed={failed}:pending={pending}"
        )
    return expected_dates, {
        "requested_date_count": len(expected_dates),
        "terminal_date_count": terminal,
        "observed_date_count": sum(item.get("status") == "observed" for item in year_tasks.values()),
        "confirmed_empty_date_count": sum(item.get("status") == "confirmed_empty" for item in year_tasks.values()),
        "failed_date_count": failed,
        "pending_date_count": pending,
    }


def _prepare_tushare_report_years(
    workspace: Path,
    *,
    open_dates: np.ndarray,
    task_days: dict[str, Any],
    request_dates: tuple[str, ...],
    normalized_root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    report_parts: list[Path] = []
    statistics: list[dict[str, Any]] = []
    for year in range(2010, 2026):
        year_root = _report_rc_page_path(workspace, f"{year}-01-01", 0).parent.parent
        raw_paths = sorted(year_root.rglob("*.parquet"))
        if not raw_paths:
            raise ResearchEventUpdateError(f"tushare_report_year_missing:{year}")
        _, task_coverage = _report_year_task_coverage(
            year,
            request_dates=request_dates,
            task_days=task_days,
        )
        raw = pd.concat([pd.read_parquet(path) for path in raw_paths], ignore_index=True)
        reports, _, stats = _normalize_tushare_report_year(
            raw,
            open_dates=open_dates,
        )
        report_path = normalized_root / f"reports_{year}.parquet"
        _write_parquet(reports, report_path)
        report_parts.append(report_path)
        statistics.append(
            {
                "year": year,
                "request_page_count": len(raw_paths),
                **stats,
                **_report_year_source_coverage(year, stats, task_coverage),
            }
        )
    return report_parts, statistics


def _load_eastmoney_reports(workspace: Path, *, open_dates: np.ndarray) -> pd.DataFrame:
    paths = _eastmoney_report_paths(workspace)
    if not paths:
        raise ResearchEventUpdateError("eastmoney_report_cache_missing")
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = _normalize_eastmoney_reports(
            pd.read_parquet(path),
            open_dates=open_dates,
        )
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REPORT_COLUMNS)


def _report_coverage_years(
    statistics: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    unavailable = [
        int(item["year"]) for item in statistics if item["tushare_source_coverage_status"] == "source_unavailable"
    ]
    partial = [
        int(item["year"]) for item in statistics if item["tushare_source_coverage_status"] == "partial_year_span"
    ]
    incomplete = [
        int(item["year"])
        for item in statistics
        if item["tushare_source_coverage_status"] != "complete_daily_task_ledger"
    ]
    return unavailable, partial, incomplete


def prepare_reports(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    open_dates = _open_dates(workspace)
    normalized_root = _runtime(workspace) / "normalized" / "tushare_reports"
    state = _read_state(workspace)
    report_state = dict(state.get("tushare_report_rc", {}) or {})
    task_days = dict(report_state.get("days", {}) or {})
    request_dates = _report_request_dates()
    if int(report_state.get("requested_date_count", 0) or 0) != len(request_dates):
        raise ResearchEventUpdateError("tushare_report_daily_ledger_incomplete")
    report_parts, statistics = _prepare_tushare_report_years(
        workspace,
        open_dates=open_dates,
        task_days=task_days,
        request_dates=request_dates,
        normalized_root=normalized_root,
    )
    tushare_reports = pd.concat([pd.read_parquet(path) for path in report_parts], ignore_index=True)
    eastmoney_reports = _load_eastmoney_reports(workspace, open_dates=open_dates)
    reports = _merge_reports(tushare_reports, eastmoney_reports)
    prepared = _runtime(workspace) / "prepared"
    report_path = prepared / "research_report.parquet"
    stats_path = prepared / "report_annual_statistics.parquet"
    _write_parquet(reports, report_path)
    _write_parquet(pd.DataFrame(statistics), stats_path)
    unavailable_years, partial_years, incomplete_years = _report_coverage_years(statistics)
    result = {
        "status": "completed",
        "report_path": str(report_path),
        "report_row_count": len(reports),
        "annual_statistics_path": str(stats_path),
        "tushare_only_report_count": int((reports["tushare_present"] & ~reports["eastmoney_present"]).sum()),
        "eastmoney_only_report_count": int((~reports["tushare_present"] & reports["eastmoney_present"]).sum()),
        "matched_report_count": int((reports["tushare_present"] & reports["eastmoney_present"]).sum()),
        "earliest_report_date": str(reports["source_date"].min()),
        "latest_report_date": str(reports["source_date"].max()),
        "tushare_source_unavailable_years": unavailable_years,
        "tushare_source_partial_years": partial_years,
        "tushare_incomplete_task_years": incomplete_years,
        "source_coverage_semantics": (
            "coverage is proven by a complete daily request ledger; confirmed-empty "
            "dates are distinct from provider failures"
        ),
        "forecast_domain_retired": True,
    }
    state = _read_state(workspace)
    state["reports_prepared"] = result
    state["status"] = "reports_prepared"
    _write_state(workspace, state)
    return result
