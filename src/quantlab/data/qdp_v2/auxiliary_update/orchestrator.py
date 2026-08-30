"""Auxiliary update orchestrator operations."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantlab.core.io import json_safe

from .context import (
    AUXILIARY_DOMAINS,
    FREE_SOURCE_POLICY,
    LEGACY_SOURCE_POLICY,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _context,
    _identity_dependent_repair_required,
    _manifest,
    _runtime_state_path,
    _write_state,
    plan_auxiliary_update,
)
from .corporate import (
    repair_corporate_actions,
    validate_corporate_actions_secondary,
)
from .index import (
    repair_index_constituents,
)
from .industry import (
    repair_industry,
)
from .names import (
    repair_name_change,
)
from .secondary import (
    validate_index_secondary,
    validate_industry_secondary,
    validate_share_capital_secondary,
)
from .shares import (
    _legacy_cninfo_share_symbols,
    repair_share_capital,
)
from .valuation import (
    _valuation_market_cap_error_count,
    repair_valuation,
)


def _validate_auxiliary_domains(
    ctx: AuxiliaryContext,
    domains: Sequence[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for domain in domains:
        if domain == "industry_concept":
            results[domain] = validate_industry_secondary(ctx)
        elif domain == "share_capital":
            results[domain] = validate_share_capital_secondary(ctx)
        elif domain == "corporate_actions":
            results[domain] = validate_corporate_actions_secondary(ctx)
        elif domain == "index_constituents":
            results[domain] = validate_index_secondary(ctx)
        else:
            manifest = _manifest(ctx.root, ctx.datasets, domain)
            status = str(dict(manifest.source or {}).get("secondary_validation_status", ""))
            if status not in {"ok", "not_comparable"}:
                raise AuxiliaryUpdateError(f"auxiliary_secondary_validation_pending:{domain}:{status}")
            results[domain] = {
                "status": "already_validated",
                "secondary_validation_status": status,
            }
    return results


def run_auxiliary_validation(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = AUXILIARY_DOMAINS,
) -> dict[str, Any]:
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(AUXILIARY_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_auxiliary_domains:{','.join(unknown)}")
    return {
        "status": "validated",
        "as_of_date": ctx.target_date,
        "domains": _validate_auxiliary_domains(ctx, selected),
    }


def _selected_domains(domains: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(AUXILIARY_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_auxiliary_domains:{','.join(unknown)}")
    return selected


def _run_free_source_repair(
    *,
    as_of_date: str,
    workspace_root: str | Path | None,
    domains: tuple[str, ...],
) -> dict[str, Any]:
    from quantlab.data.qdp_v2.auxiliary_tail_update.orchestrator import run_auxiliary_tail_update

    result = run_auxiliary_tail_update(
        as_of_date=as_of_date,
        workspace_root=workspace_root,
        domains=domains,
    )
    result.update(
        {
            "mode": "free_source_tail",
            "strict_historical_repair": "skipped",
            "legacy_tushare_required_for": [
                "historical_name_intervals",
                "historical_share_capital_gap_repair",
                "historical_dividend_repair",
            ],
        }
    )
    return result


def _legacy_repair_state(ctx: AuxiliaryContext, domains: tuple[str, ...]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "status": "repairing",
        "as_of_date": ctx.target_date,
        "domains": list(domains),
        "provider_policy": LEGACY_SOURCE_POLICY,
        "completed": {},
        "failed_stage": "",
    }
    state_path = _runtime_state_path(ctx)
    if state_path.is_file():
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            if str(prior.get("as_of_date", "")) == ctx.target_date:
                state["completed"] = dict(prior.get("completed", {}) or {})
        except (OSError, ValueError):
            pass
    return state


def _legacy_domain_requires_repair(ctx: AuxiliaryContext, domain: str) -> bool:
    if domain == "share_capital":
        return bool(_legacy_cninfo_share_symbols(ctx))
    if domain == "valuation":
        return _valuation_market_cap_error_count(ctx) > 0
    if domain in {"industry_concept", "index_constituents"}:
        return _identity_dependent_repair_required(ctx, domain)
    return False


def _run_legacy_domain_repair(
    ctx: AuxiliaryContext,
    domain: str,
    daily_basic_parts: list[Path],
) -> tuple[dict[str, Any], list[Path]]:
    if domain == "industry_concept":
        return repair_industry(ctx), daily_basic_parts
    if domain == "name_change":
        return repair_name_change(ctx), daily_basic_parts
    if domain == "corporate_actions":
        return repair_corporate_actions(ctx), daily_basic_parts
    if domain == "index_constituents":
        return repair_index_constituents(ctx), daily_basic_parts
    if domain == "share_capital":
        return repair_share_capital(ctx)
    if domain == "valuation":
        return repair_valuation(ctx, daily_basic_parts=daily_basic_parts), daily_basic_parts
    raise AssertionError(domain)  # pragma: no cover - validated by _selected_domains


def _repair_legacy_domains(
    ctx: AuxiliaryContext,
    *,
    domains: tuple[str, ...],
    force: bool,
    state: dict[str, Any],
) -> None:
    daily_basic_parts: list[Path] = []
    for domain in domains:
        manifest = _manifest(ctx.root, ctx.datasets, domain)
        checked = str(dict(manifest.source or {}).get("checked_through", ""))
        source_contract = str(dict(manifest.source or {}).get("source_contract", ""))
        needs_repair = _legacy_domain_requires_repair(ctx, domain)
        if checked == ctx.target_date and "strict" in source_contract.lower() and not needs_repair and not force:
            state["completed"][domain] = {
                "status": "already_current",
                "checked_through": checked,
            }
            _write_state(ctx, state)
            continue
        state["stage"] = domain
        _write_state(ctx, state)
        result, daily_basic_parts = _run_legacy_domain_repair(ctx, domain, daily_basic_parts)
        state["completed"][domain] = result
        _write_state(ctx, state)


def _finish_legacy_repair(ctx: AuxiliaryContext, domains: tuple[str, ...], state: dict[str, Any]) -> None:
    state["stage"] = "secondary_validation"
    _write_state(ctx, state)
    state["validation"] = _validate_auxiliary_domains(ctx, domains)
    state.update({"status": "repaired", "stage": ""})
    _write_state(ctx, state)
    shutil.rmtree(ctx.runtime / "daily_basic_parts", ignore_errors=True)
    shutil.rmtree(ctx.runtime / "duckdb_spill", ignore_errors=True)
    _runtime_state_path(ctx).unlink(missing_ok=True)
    try:
        ctx.runtime.rmdir()
    except OSError:
        pass


def _failed_legacy_repair(ctx: AuxiliaryContext, state: dict[str, Any], exc: Exception) -> dict[str, Any]:
    state.update(
        {
            "status": "failed",
            "failed_stage": str(state.get("stage", "")),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
    )
    _write_state(ctx, state)
    return state


def run_auxiliary_repair(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = AUXILIARY_DOMAINS,
    force: bool = False,
    allow_legacy_tushare: bool = False,
) -> dict[str, Any]:
    selected = _selected_domains(domains)
    if not allow_legacy_tushare:
        return _run_free_source_repair(
            as_of_date=as_of_date,
            workspace_root=workspace_root,
            domains=selected,
        )
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    state = _legacy_repair_state(ctx, selected)
    _write_state(ctx, state)
    try:
        _repair_legacy_domains(ctx, domains=selected, force=force, state=state)
        _finish_legacy_repair(ctx, selected, state)
        return state
    except Exception as exc:
        return _failed_legacy_repair(ctx, state, exc)


def run_auxiliary_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    baostock_valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a free-source auxiliary tail after strict contracts are installed."""

    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    stale: list[str] = []
    for domain in AUXILIARY_DOMAINS:
        manifest = _manifest(ctx.root, ctx.datasets, domain)
        if str(
            dict(manifest.source or {}).get("checked_through", "")
        ) != ctx.target_date or _identity_dependent_repair_required(ctx, domain):
            stale.append(domain)
    if not stale:
        if baostock_valuation_cache_path is not None:
            Path(baostock_valuation_cache_path).resolve().unlink(missing_ok=True)
        return {
            "status": "current",
            "as_of_date": ctx.target_date,
            "domains": list(AUXILIARY_DOMAINS),
            "provider_policy": FREE_SOURCE_POLICY,
        }
    from quantlab.data.qdp_v2.auxiliary_tail_update.orchestrator import run_auxiliary_tail_update

    return run_auxiliary_tail_update(
        as_of_date=ctx.target_date,
        workspace_root=ctx.workspace,
        domains=stale,
        baostock_valuation_cache_path=baostock_valuation_cache_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp auxiliary-update",
        description="Repair or update the six strict-PIT auxiliary domains.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--legacy-tushare",
        action="store_true",
        help=(
            "Enable the historical repair path that still requires a valid "
            "Tushare token; the default uses free-source tail updates."
        ),
    )
    parser.add_argument("--domains", default=",".join(AUXILIARY_DOMAINS))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    domains = tuple(item.strip() for item in str(args.domains).split(",") if item.strip())
    if args.dry_run and args.validate_only:
        raise ValueError("dry_run_and_validate_only_are_mutually_exclusive")
    payload = (
        plan_auxiliary_update(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
        )
        if args.dry_run
        else run_auxiliary_validation(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
            domains=domains,
        )
        if args.validate_only
        else run_auxiliary_repair(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
            domains=domains,
            force=bool(args.force),
            allow_legacy_tushare=bool(args.legacy_tushare),
        )
    )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return (
        0
        if payload.get("status")
        in {
            "planned",
            "repaired",
            "updated",
            "validated",
        }
        else 2
    )


