from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.data_platform.contracts import DataDomain, DomainFetchRequest
from daily_research.data_platform.providers import build_default_providers
from daily_research.execution.app_service import FORMAL_DATA_PLATFORM_PROVIDER_PLAN
from daily_research.execution.app_runtime import PROJECT_ROOT, read_json_file


DEFAULT_SYMBOLS = ("000001.SZ", "600000.SH", "000300.SH")


def _date_text(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value or "")[:10]


def _extract_error(result: Any) -> str:
    errors = list(getattr(result, "error_report", []) or [])
    if not errors:
        return ""
    parts: list[str] = []
    for item in errors[:5]:
        if isinstance(item, dict):
            parts.append(str(item.get("message") or item.get("code") or item))
        else:
            parts.append(str(item))
    return " | ".join(part for part in parts if part)


def resolve_provider_ready_trading_date(
    *,
    candidate_date: str = "",
    provider_plan: str = FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
    min_coverage_ratio: float = 0.80,
    symbols: tuple[str, ...] | list[str] = DEFAULT_SYMBOLS,
    runs_root: str | Path | None = None,
    provider_builder: Callable[[str], list[Any]] | None = None,
) -> dict[str, Any]:
    resolved_date = _date_text(candidate_date or get_latest_completed_trading_date())
    manifest_blocker = _blocked_manifest_for_date(resolved_date, runs_root=runs_root)
    if manifest_blocker:
        return manifest_blocker
    requested_symbols = tuple(str(item).strip().upper() for item in symbols if str(item).strip())
    request = DomainFetchRequest(
        domain=DataDomain.MARKET_DAILY,
        symbols=requested_symbols,
        start_date=resolved_date,
        end_date=resolved_date,
        adjusted_flag="none",
    )
    builder = provider_builder or build_default_providers
    provider_payloads: list[dict[str, Any]] = []
    best_payload: dict[str, Any] = {
        "row_count": 0,
        "coverage_ratio": 0.0,
        "provider": "",
        "provider_error": "",
    }
    for provider in builder(str(provider_plan or FORMAL_DATA_PLATFORM_PROVIDER_PLAN)):
        provider_name = str(getattr(provider, "name", "") or "")
        try:
            result = provider.fetch_domain(request)
            frame = result.data if result is not None else pd.DataFrame()
            row_count = int(len(frame)) if frame is not None else 0
            coverage = dict(getattr(result, "coverage_report", {}) or {})
            coverage_ratio = float(coverage.get("coverage_ratio", 0.0) or 0.0)
            if coverage_ratio <= 0.0 and requested_symbols:
                coverage_ratio = float(row_count / len(requested_symbols)) if requested_symbols else 0.0
            payload = {
                "provider": provider_name,
                "status": "ok" if row_count > 0 else "no_data",
                "row_count": row_count,
                "coverage_ratio": coverage_ratio,
                "coverage_report": coverage,
                "provider_error": _extract_error(result),
            }
        except Exception as exc:
            payload = {
                "provider": provider_name,
                "status": "error",
                "row_count": 0,
                "coverage_ratio": 0.0,
                "provider_error": f"{type(exc).__name__}: {exc}",
            }
        provider_payloads.append(payload)
        if (
            int(payload.get("row_count", 0) or 0) > int(best_payload.get("row_count", 0) or 0)
            or float(payload.get("coverage_ratio", 0.0) or 0.0) > float(best_payload.get("coverage_ratio", 0.0) or 0.0)
        ):
            best_payload = dict(payload)
        if int(payload.get("row_count", 0) or 0) > 0 and float(payload.get("coverage_ratio", 0.0) or 0.0) >= float(min_coverage_ratio):
            return {
                "status": "ready",
                "candidate_date": resolved_date,
                "provider_ready_date": resolved_date,
                "provider_plan": str(provider_plan),
                "provider": provider_name,
                "row_count": int(payload.get("row_count", 0) or 0),
                "coverage_ratio": float(payload.get("coverage_ratio", 0.0) or 0.0),
                "coverage_report": dict(payload.get("coverage_report", {}) or {}),
                "providers": provider_payloads,
            }
    row_count = int(best_payload.get("row_count", 0) or 0)
    coverage_ratio = float(best_payload.get("coverage_ratio", 0.0) or 0.0)
    blocker = "data_not_ready" if row_count <= 0 else "coverage_below_threshold"
    provider_error = str(best_payload.get("provider_error", "") or "")
    if row_count <= 0 and not provider_error:
        provider_error = "empty_canonical_market"
    return {
        "status": "blocked",
        "blocker_code": blocker,
        "candidate_date": resolved_date,
        "provider_ready_date": "",
        "provider_plan": str(provider_plan),
        "provider": str(best_payload.get("provider", "") or ""),
        "row_count": row_count,
        "coverage_ratio": coverage_ratio,
        "provider_error": provider_error,
        "providers": provider_payloads,
    }


def _blocked_manifest_for_date(candidate_date: str, *, runs_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(runs_root) if runs_root is not None else PROJECT_ROOT / "output" / "research_data_lake" / "data_platform" / "runs"
    if not root.exists():
        return {}
    manifests = [path for path in root.glob("*/refresh_manifest.json") if path.is_file()]
    matched: list[tuple[float, Path, dict[str, Any]]] = []
    for path in manifests:
        payload = read_json_file(path)
        if _date_text(payload.get("as_of_date", "")) != candidate_date:
            continue
        if str(payload.get("status", "") or "").lower() != "blocked":
            continue
        matched.append((path.stat().st_mtime, path, payload))
    if not matched:
        return {}
    _, path, payload = sorted(matched, key=lambda item: item[0])[-1]
    coverage = dict(payload.get("coverage_report", {}) or {})
    blockers = [str(item) for item in payload.get("blockers", [])] if isinstance(payload.get("blockers"), list) else []
    row_count = int(coverage.get("row_count", 0) or 0)
    blocker_code = "data_not_ready" if row_count <= 0 or "empty_canonical_market" in blockers else "coverage_below_threshold"
    return {
        "status": "blocked",
        "blocker_code": blocker_code,
        "candidate_date": candidate_date,
        "provider_ready_date": "",
        "row_count": row_count,
        "coverage_ratio": float(coverage.get("coverage_ratio", 0.0) or 0.0),
        "expected_rows": int(coverage.get("expected_rows", 0) or 0),
        "provider_error": ",".join(blockers),
        "refresh_manifest_path": str(path.resolve()),
        "refresh_manifest_status": str(payload.get("status", "") or ""),
        "blockers": blockers,
        "coverage_report": coverage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check provider readiness for a daily execution target date.")
    parser.add_argument("--candidate-date", default="")
    parser.add_argument("--provider-plan", default=FORMAL_DATA_PLATFORM_PROVIDER_PLAN)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.80)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(item.strip() for item in str(args.symbols or "").split(",") if item.strip())
    payload = resolve_provider_ready_trading_date(
        candidate_date=args.candidate_date,
        provider_plan=args.provider_plan,
        min_coverage_ratio=float(args.min_coverage_ratio),
        symbols=symbols or DEFAULT_SYMBOLS,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if payload.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
