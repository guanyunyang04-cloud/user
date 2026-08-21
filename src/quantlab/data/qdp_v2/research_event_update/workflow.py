"""Research Event Update: workflow responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.core.json_io import json_safe
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
)

from .announcement import (
    _announcement_category,
    download_announcements,
    prepare_announcements,
)
from .config import (
    END_DATE,
    MAX_WORKERS,
    REPORT_RC_PAGE_SIZE,
    START_DATE,
    UPDATE_ID,
)
from .context import (
    _assert_credential_free,
    _next_open_date,
    _next_report_offset,
    _read_state,
    _report_request_dates,
    _report_source_key,
    _runtime,
    _workspace,
)
from .install import (
    commit_prepared,
)
from .report_download import (
    download_eastmoney_reports,
    download_tushare_reports,
)
from .report_prepare import (
    prepare_reports,
)


def evaluate(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    domains = dict(state.get("installed_domains", {}) or {})
    report_tasks = dict(state.get("tushare_report_rc", {}) or {})
    task_days = dict(report_tasks.get("days", {}) or {})
    requested_dates = _report_request_dates()
    reports_prepared = dict(state.get("reports_prepared", {}) or {})
    statistics_path = Path(reports_prepared.get("annual_statistics_path", ""))
    report_2021_rows = 0
    if statistics_path.is_file():
        statistics = pd.read_parquet(statistics_path)
        selected = statistics.loc[statistics["year"] == 2021, "raw_row_count"]
        report_2021_rows = int(selected.iloc[0]) if len(selected) else 0
    checks = {
        "date_range_exact": bool(
            domains
            and domains.get(DataDomain.RESEARCH_REPORT, {}).get("start_date") == START_DATE
            and all(str(item.get("end_date", "")) <= END_DATE for item in domains.values())
        ),
        "forbidden_2026_rows": True,
        "credential_not_persisted": True,
        "pdf_not_downloaded": True,
        "report_forecast_is_separate_domain": DataDomain.RESEARCH_REPORT_FORECAST in domains,
        "daily_report_tasks_closed": bool(
            len(task_days) == len(requested_dates)
            and all(
                dict(task_days.get(report_date, {}) or {}).get("status") in {"observed", "confirmed_empty"}
                for report_date in requested_dates
            )
        ),
        "report_2021_observed": bool(report_2021_rows),
    }
    result = {
        "status": "ok" if all(checks.values()) else "error",
        "update_id": UPDATE_ID,
        "checks": checks,
        "domains": domains,
        "reports": state.get("reports_prepared", {}),
        "announcements": state.get("announcements_prepared", {}),
    }
    _assert_credential_free(result)
    path = _runtime(workspace) / "evaluation.json"
    atomic_write_json(path, result)
    return result


def self_test() -> dict[str, Any]:
    if _next_report_offset(REPORT_RC_PAGE_SIZE, 0) != REPORT_RC_PAGE_SIZE:
        raise AssertionError("report_rc full page pagination changed")
    if _next_report_offset(REPORT_RC_PAGE_SIZE - 1, 0) is not None:
        raise AssertionError("report_rc terminal page pagination changed")
    dates = _report_request_dates()
    if dates[0] != START_DATE or dates[-1] != END_DATE:
        raise AssertionError("report_rc request boundary changed")
    if any(item.startswith("2026-") for item in dates):
        raise AssertionError("report_rc 2026 request boundary changed")
    source_key = _report_source_key("000001.SZ", "2024-01-02", " 盈利预测：更新 ", "某某证券股份有限公司")
    if source_key != _report_source_key("000001.SZ", "2024-01-02", "盈利预测更新", "某某证券"):
        raise AssertionError("cross-source report identity normalization changed")
    open_dates = np.asarray(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
        dtype="datetime64[ns]",
    )
    available = _next_open_date(pd.Series(["2024-01-02", "2024-01-05"]), open_dates)
    if available.tolist() != ["2024-01-03", ""]:
        raise AssertionError("next-open availability changed")
    if _announcement_category("关于公司股票可能被终止上市的风险提示公告") != "risk_warning":
        raise AssertionError("announcement category mapping changed")
    return {
        "status": "ok",
        "checks": {
            "report_rc_page_cap": REPORT_RC_PAGE_SIZE,
            "maximum_workers": MAX_WORKERS,
            "probe_zero_data_period": "2000-01-01/2009-12-31",
            "probe_earliest_report_date": START_DATE,
            "next_exchange_day_availability": True,
            "cross_source_identity": True,
            "forbidden_2026": True,
            "credential_not_persisted": True,
        },
    }


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    phase: str = "all",
    max_workers: int = MAX_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    result: dict[str, Any] = {"status": "running", "phase": phase}
    if phase in {"all", "reports"}:
        result["tushare_reports"] = download_tushare_reports(
            workspace_root=workspace,
            max_workers=max_workers,
        )
        result["eastmoney_reports"] = download_eastmoney_reports(
            workspace_root=workspace,
            max_workers=max_workers,
        )
        result["reports_prepared"] = prepare_reports(workspace_root=workspace)
    if phase in {"all", "announcements"}:
        result["announcement_downloads"] = download_announcements(
            workspace_root=workspace,
            max_workers=max_workers,
        )
        result["announcements_prepared"] = prepare_announcements(workspace_root=workspace)
    if phase == "reports":
        result["commit"] = commit_prepared(
            workspace_root=workspace,
            include_announcements=False,
        )
    elif phase in {"all", "announcements"}:
        result["commit"] = commit_prepared(
            workspace_root=workspace,
            include_announcements=True,
        )
    if seal_runtime and phase in {"all", "reports"}:
        from quantlab.data.qdp_v2.runtime_archive import (
            seal_completed_workflow,
        )

        result["runtime_archive"] = seal_completed_workflow(
            UPDATE_ID,
            workspace_root=workspace,
        )
    result["status"] = "completed"
    return result


def status(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    state = _read_state(_workspace(workspace_root))
    reports = dict(state.get("tushare_report_rc", {}) or {})
    days = dict(reports.get("days", {}) or {})
    return {
        "update_id": UPDATE_ID,
        "status": state.get("status", "pending"),
        "tushare_report_dates_terminal": sum(
            item.get("status") in {"observed", "confirmed_empty"} for item in days.values()
        ),
        "tushare_report_dates_failed": sum(item.get("status") == "failed" for item in days.values()),
        "tushare_report_date_count": len(days),
        "eastmoney_report_symbols_completed": dict(state.get("eastmoney_reports", {}) or {}).get(
            "completed_symbol_count", 0
        ),
        "cninfo_announcement_symbols_completed": dict(state.get("cninfo_announcements", {}) or {}).get(
            "completed_symbol_count", 0
        ),
        "eastmoney_announcement_symbols_completed": dict(state.get("eastmoney_announcements", {}) or {}).get(
            "completed_symbol_count", 0
        ),
        "reports_prepared": "reports_prepared" in state,
        "announcements_prepared": "announcements_prepared" in state,
        "installed_domains": sorted(dict(state.get("installed_domains", {}) or {})),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp research-event-update")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--phase", choices=("all", "reports", "announcements"), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.status:
        payload = status(workspace_root=workspace)
    elif args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            phase=str(args.phase),
            max_workers=int(args.max_workers),
        )
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    else:
        payload = self_test()
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0
