from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_platform.contracts import DataDomain, DomainFetchRequest, normalize_domain
from daily_research.data_platform.manager import UnsupportedDomainError
from daily_research.data_platform.providers import (
    FORMAL_FREE_V3_REQUIRED_DOMAINS,
    build_default_providers,
    provider_capability_matrix,
)


@dataclass(frozen=True)
class ProviderHealthConfig:
    provider_plan: str = "formal_free_v3"
    as_of_date: str = ""
    domains: tuple[str, ...] = FORMAL_FREE_V3_REQUIRED_DOMAINS
    symbols: tuple[str, ...] = ("000001.SZ", "600000.SH", "000300.SH")
    adjusted_flag: str = "none"


def run_provider_health(config: ProviderHealthConfig | None = None, *, providers: Iterable[Any] | None = None) -> dict[str, Any]:
    resolved = config or ProviderHealthConfig()
    as_of_date = _date_text(resolved.as_of_date or pd.Timestamp.now().strftime("%Y-%m-%d"))
    domains = tuple(dict.fromkeys(normalize_domain(item) for item in (resolved.domains or FORMAL_FREE_V3_REQUIRED_DOMAINS)))
    provider_chain = list(providers) if providers is not None else build_default_providers(resolved.provider_plan)
    matrix = provider_capability_matrix(resolved.provider_plan)
    providers_payload: list[dict[str, Any]] = []
    total_checked = 0
    ok_count = 0
    error_count = 0
    total_steps = len(provider_chain) * len(domains)
    last_provider_name = ""
    last_domain = ""
    for provider in provider_chain:
        provider_name = str(getattr(provider, "name", "") or "")
        provider_entry = {"provider": provider_name, "domains": {}}
        for domain in domains:
            last_provider_name = provider_name
            last_domain = domain
            stage = f"Checking {provider_name} / {domain}"
            request = DomainFetchRequest(
                domain=domain,
                symbols=tuple(resolved.symbols or ()),
                start_date=as_of_date,
                end_date=as_of_date,
                adjusted_flag=resolved.adjusted_flag,
            )
            _update_execution_job_progress(
                stage=stage,
                completed_steps=total_checked,
                total_steps=total_steps,
                current_provider=provider_name,
                current_domain=domain,
                current_item=f"{provider_name} / {domain}",
            )
            try:
                result = provider.fetch_domain(request)
                row_count = int(len(result.data)) if result.data is not None else 0
                status = "ok" if row_count > 0 else "no_data"
                if status == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                provider_entry["domains"][domain] = {
                    "status": status,
                    "row_count": row_count,
                    "coverage_report": dict(result.coverage_report or {}),
                    "error_report": list(result.error_report or []),
                }
            except UnsupportedDomainError as exc:
                error_count += 1
                provider_entry["domains"][domain] = {"status": "unsupported", "row_count": 0, "error": str(exc)}
            except Exception as exc:
                error_count += 1
                provider_entry["domains"][domain] = {
                    "status": "error",
                    "row_count": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            total_checked += 1
            _update_execution_job_progress(
                stage=stage,
                completed_steps=total_checked,
                total_steps=total_steps,
                current_provider=provider_name,
                current_domain=domain,
                current_item=f"{provider_name} / {domain}",
            )
        providers_payload.append(provider_entry)
    summary = {
        "checked_provider_count": len(providers_payload),
        "checked_domain_count": total_checked,
        "ok_domain_count": ok_count,
        "error_count": error_count,
    }
    required_domains = set(FORMAL_FREE_V3_REQUIRED_DOMAINS)
    domain_status: dict[str, str] = {}
    for domain in domains:
        has_ok_provider = any(
            str(dict(provider_payload.get("domains", {})).get(domain, {}).get("status", "")) == "ok"
            for provider_payload in providers_payload
        )
        domain_status[domain] = "ok" if has_ok_provider else "missing"
    requested_required_domains = set(domains).intersection(required_domains)
    required_domain_status = {domain: domain_status.get(domain, "missing") for domain in sorted(requested_required_domains)}
    summary["domain_status"] = domain_status
    summary["required_domain_status"] = required_domain_status
    summary["missing_domain_count"] = sum(1 for value in domain_status.values() if value != "ok")
    status = "ok" if domain_status and all(value == "ok" for value in domain_status.values()) else "degraded"
    payload = {
        "status": status,
        "provider_plan": str(resolved.provider_plan),
        "as_of_date": as_of_date,
        "domains": list(domains),
        "symbols": list(resolved.symbols),
        "summary": summary,
        "domain_matrix": matrix,
        "providers": providers_payload,
    }
    _update_execution_job_progress(
        stage="Provider health completed",
        completed_steps=total_steps,
        total_steps=total_steps,
        current_item="completed",
        current_provider=last_provider_name,
        current_domain=last_domain,
        status=status,
        provider_health=payload,
    )
    return payload


def _update_execution_job_progress(
    *,
    stage: str,
    completed_steps: int,
    total_steps: int,
    current_provider: str = "",
    current_domain: str = "",
    current_item: str = "",
    status: str = "running",
    provider_health: dict[str, Any] | None = None,
) -> None:
    job_id = str(os.getenv("EXECUTION_APP_JOB_ID", "") or "").strip()
    if not job_id:
        return
    try:
        from daily_research.execution.app_runtime import update_job_progress

        total = max(int(total_steps or 0), 0)
        completed = max(int(completed_steps or 0), 0)
        percent = round((completed / total) * 100.0, 2) if total else None
        update_job_progress(
            job_id,
            mode="determinate" if total else "indeterminate",
            stage=stage,
            completed_steps=completed,
            total_steps=total,
            percent=percent,
            current_item=current_item,
            current_provider=current_provider,
            current_domain=current_domain,
            status=status,
            provider_health=provider_health,
        )
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run read-only health checks for Daily Research data providers.")
    parser.add_argument("--provider-plan", default="formal_free_v3")
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--domains", default=",".join(FORMAL_FREE_V3_REQUIRED_DOMAINS))
    parser.add_argument("--symbols", default="000001.SZ,600000.SH,000300.SH")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_provider_health(
        ProviderHealthConfig(
            provider_plan=args.provider_plan,
            as_of_date=args.as_of_date,
            domains=_split_csv(args.domains),
            symbols=_split_csv(args.symbols),
        )
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if payload.get("status") in {"ok", "degraded"} else 2


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _date_text(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value or "")[:10]


if __name__ == "__main__":
    raise SystemExit(main())
