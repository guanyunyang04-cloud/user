"""Tushare Extended Backfill: workflow responsibilities."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.provider_credentials import (
    tushare_provider_status,
)

from .audit import (
    audit,
    cleanup_prepared_cache,
)
from .config import (
    EXPECTED_FACTOR_FIELD_COUNT,
    EXPECTED_FACTOR_SCHEMA_HASH,
    FORBIDDEN_YEAR,
    MAX_REQUESTS_PER_MINUTE,
    MAX_WORKERS,
    PAGE_SIZE,
    SPECS,
    UPDATE_ID,
    TushareExtendedBackfillError,
    _resolve_tushare_api_url,
)
from .context import (
    _read_state,
    _workspace,
)
from .download import (
    _download_inventory,
    _selected_specs,
    _task_params,
)
from .install import (
    prepare_and_install,
)
from .probe import (
    probe,
)


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    domains: str | Sequence[str] | None = None,
    workers: int = MAX_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    specs = _selected_specs(domains)
    downloads = _download_inventory(
        workspace=workspace,
        specs=specs,
        workers=workers,
    )
    installed = prepare_and_install(workspace_root=workspace, domains=domains)
    result = {"status": "completed", "downloads": downloads, "install": installed}
    if seal_runtime:
        from quantlab.data.qdp_v2.runtime_archive import (
            seal_completed_workflow,
        )

        result["runtime_archive"] = seal_completed_workflow(
            UPDATE_ID,
            workspace_root=workspace,
        )
    return result


def status(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    probe_state = dict(state.get("probe", {}) or {})
    return {
        "update_id": UPDATE_ID,
        "status": state.get("status", "pending"),
        "probe_status": probe_state.get("status", "pending"),
        "endpoint_status": {
            name: dict(record).get("status", "pending")
            for name, record in dict(probe_state.get("endpoints", {}) or {}).items()
        },
        "download_progress": state.get("download_progress", {}),
        "installed_domains": state.get("installed_domains", {}),
        "provider_profile": tushare_provider_status(workspace),
        "training_performed": state.get("training_performed", False),
    }


def self_test() -> dict[str, Any]:
    if len(SPECS) != 5 or MAX_WORKERS != 3:
        raise AssertionError("endpoint or concurrency policy changed")
    if PAGE_SIZE != 5_000 or EXPECTED_FACTOR_FIELD_COUNT != 261:
        raise AssertionError("provider pagination or factor schema policy changed")
    if any(str(year) == str(FORBIDDEN_YEAR) for year in range(2010, 2026)):
        raise AssertionError("forbidden year entered download inventory")
    if _task_params(SPECS["margin"], "2011", 0)["end_date"] != "20111231":
        raise AssertionError("year task boundary changed")
    if _task_params(SPECS["moneyflow"], "2011-01-04", 0)["trade_date"] != "20110104":
        raise AssertionError("daily task normalization changed")
    return {
        "status": "ok",
        "checks": {
            "endpoint_count": len(SPECS),
            "factor_field_count": EXPECTED_FACTOR_FIELD_COUNT,
            "factor_schema_hash": EXPECTED_FACTOR_SCHEMA_HASH,
            "page_size": PAGE_SIZE,
            "maximum_workers": MAX_WORKERS,
            "maximum_requests_per_minute": MAX_REQUESTS_PER_MINUTE,
            "burn_in_year": 2010,
            "research_start_year": 2011,
            "forbidden_2026": True,
            "training_performed": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tushare-extended-backfill")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--domains", default="")
    parser.add_argument("--workers", "--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--credential-stdin",
        action="store_true",
        help="Read the provider token without echo and keep it process-local.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--cleanup-prepared", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.credential_stdin:
        token = getpass.getpass("Token: ").strip()
        if not token:
            raise TushareExtendedBackfillError("tushare_token_required")
        os.environ["QDP_TUSHARE_PROXY_TOKEN"] = token
        os.environ["QDP_TUSHARE_PREFER_ENV"] = "1"
        try:
            _resolve_tushare_api_url(workspace)
        except TushareExtendedBackfillError:
            api_url = getpass.getpass("API URL: ").strip()
            if not api_url:
                raise TushareExtendedBackfillError("tushare_api_url_missing") from None
            os.environ["QDP_TUSHARE_API_URL"] = api_url
    domains = str(args.domains or "") or None
    if args.probe:
        payload = probe(workspace_root=workspace)
    elif args.status:
        payload = status(workspace_root=workspace)
    elif args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            domains=domains,
            workers=int(args.workers),
        )
    elif args.prepare:
        payload = prepare_and_install(workspace_root=workspace, domains=domains)
    elif args.audit:
        payload = audit(workspace_root=workspace)
    elif args.cleanup_prepared:
        payload = cleanup_prepared_cache(
            workspace_root=workspace,
            delete=bool(args.delete),
            yes=bool(args.yes),
        )
    else:
        payload = self_test()
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0
