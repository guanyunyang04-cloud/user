from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from daily_research.data_lake import ResearchDataLake
from daily_research.data_platform.contracts import (
    FetchRequest,
    NUMERIC_MARKET_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    market_business_dates,
    next_business_date,
)
from daily_research.data_platform.manager import ProviderManager
from daily_research.data_platform.providers import build_default_providers


@dataclass(frozen=True)
class RefreshConfig:
    lake_root: Path | str = Path("daily_research/output/research_data_lake")
    as_of_date: str = ""
    start_date: str = ""
    symbols: tuple[str, ...] = ()
    benchmark: str = "000300.SH"
    provider_plan: str = "default_free"
    adjusted_flag: str = "none"
    min_coverage_ratio: float = 0.80
    conflict_tolerance_pct: float = 0.005
    severe_conflict_limit: int = 0
    timeout_seconds: float = 30.0
    allow_tdx_family: bool = False
    run_id: str = ""


@dataclass(frozen=True)
class RefreshResult:
    status: str
    refresh_run_id: str
    manifest_path: str
    bronze_paths: dict[str, str]
    silver_market_path: str
    conflict_report_path: str
    registered_market_dataset_id: str = ""
    coverage_report: dict[str, Any] = field(default_factory=dict)
    conflict_summary: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)


def run_refresh(config: RefreshConfig, *, providers: Iterable[Any] | None = None) -> RefreshResult:
    resolved = _resolve_config(config)
    lake = ResearchDataLake(resolved.lake_root)
    refresh_run_id = resolved.run_id or _build_run_id(resolved)
    run_root = Path(resolved.lake_root) / "data_platform" / "runs" / refresh_run_id
    bronze_root = run_root / "bronze"
    silver_root = run_root / "silver"
    bronze_root.mkdir(parents=True, exist_ok=True)
    silver_root.mkdir(parents=True, exist_ok=True)

    request_symbols = _request_symbols(resolved)
    request_start = _resolve_incremental_start(lake, resolved)
    if pd.Timestamp(request_start) > pd.Timestamp(resolved.as_of_date):
        manifest_path = run_root / "refresh_manifest.json"
        payload = {
            "status": "skipped",
            "refresh_run_id": refresh_run_id,
            "reason": "lake already covers requested as_of_date",
            "as_of_date": resolved.as_of_date,
            "request_start_date": request_start,
        }
        _write_json(manifest_path, payload)
        return RefreshResult(
            status="skipped",
            refresh_run_id=refresh_run_id,
            manifest_path=str(manifest_path.resolve()),
            bronze_paths={},
            silver_market_path="",
            conflict_report_path="",
            blockers=[],
        )

    request = FetchRequest(
        symbols=request_symbols,
        start_date=request_start,
        end_date=resolved.as_of_date,
        adjusted_flag=resolved.adjusted_flag,
    ).normalized()
    resolved_providers = list(providers) if providers is not None else build_default_providers(resolved.provider_plan)
    manager = ProviderManager(
        resolved_providers,
        timeout_seconds=resolved.timeout_seconds,
        allow_tdx_family=resolved.allow_tdx_family,
    )
    provider_result = manager.fetch_market_bars(request)
    bronze_paths = _write_bronze(provider_result.data, bronze_root, [getattr(item, "name", "") for item in resolved_providers])
    canonical, conflict_report, conflict_summary = _build_silver_market(
        provider_result.data,
        provider_priority=[getattr(item, "name", "") for item in resolved_providers],
        conflict_tolerance_pct=resolved.conflict_tolerance_pct,
    )
    silver_market_path = silver_root / "silver_market.parquet"
    conflict_report_path = silver_root / "source_conflict_report.parquet"
    canonical.to_parquet(silver_market_path, index=False)
    conflict_report.to_parquet(conflict_report_path, index=False)

    coverage_report = _coverage_report(canonical, request)
    blockers = _refresh_blockers(
        coverage_report=coverage_report,
        conflict_summary=conflict_summary,
        min_coverage_ratio=resolved.min_coverage_ratio,
        severe_conflict_limit=resolved.severe_conflict_limit,
    )
    if resolved.benchmark and (
        canonical.empty
        or not canonical["symbol"].astype(str).str.upper().eq(str(resolved.benchmark).upper()).any()
    ):
        blockers.append("missing_benchmark")
    registered_dataset_id = ""
    status = "blocked" if blockers else "ok"
    if not blockers:
        registered_dataset_id = _register_policy_input_bundle(
            lake=lake,
            config=resolved,
            refresh_run_id=refresh_run_id,
            canonical=canonical,
            provider_chain=[getattr(item, "name", "") for item in resolved_providers],
            coverage_report=coverage_report,
            conflict_summary=conflict_summary,
        )
    manifest_path = run_root / "refresh_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": status,
        "refresh_run_id": refresh_run_id,
        "provider_plan": resolved.provider_plan,
        "provider_chain": [str(getattr(item, "name", "")) for item in resolved_providers],
        "as_of_date": resolved.as_of_date,
        "request_start_date": request.start_date,
        "request_end_date": request.end_date,
        "symbols": list(request.symbols),
        "benchmark": resolved.benchmark,
        "adjusted_flag": resolved.adjusted_flag,
        "requested_symbols": list(request.symbols),
        "refresh_universe_key": _universe_key(request.symbols),
        "bronze_paths": bronze_paths,
        "silver_market_path": str(silver_market_path.resolve()),
        "conflict_report_path": str(conflict_report_path.resolve()),
        "coverage_report": coverage_report,
        "conflict_summary": conflict_summary,
        "provider_coverage_report": provider_result.coverage_report,
        "provider_error_report": provider_result.error_report,
        "blockers": blockers,
        "registered_market_dataset_id": registered_dataset_id,
    }
    _write_json(manifest_path, manifest)
    lake.write_catalog_manifest()
    return RefreshResult(
        status=status,
        refresh_run_id=refresh_run_id,
        manifest_path=str(manifest_path.resolve()),
        bronze_paths=bronze_paths,
        silver_market_path=str(silver_market_path.resolve()),
        conflict_report_path=str(conflict_report_path.resolve()),
        registered_market_dataset_id=registered_dataset_id,
        coverage_report=coverage_report,
        conflict_summary=conflict_summary,
        blockers=blockers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh TDX-free daily_research market data platform lake.")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--provider-plan", default="default_free")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. If empty, provider must supply a universe in a later phase.")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.80)
    parser.add_argument("--conflict-tolerance-pct", type=float, default=0.005)
    parser.add_argument("--severe-conflict-limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(item.strip().upper() for item in str(args.symbols or "").split(",") if item.strip())
    if not symbols:
        raise SystemExit("--symbols is required for the first TDX-free refresh implementation.")
    result = run_refresh(
        RefreshConfig(
            lake_root=Path(args.data_lake_root) if str(args.data_lake_root or "").strip() else Path("daily_research/output/research_data_lake"),
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            symbols=symbols,
            benchmark=args.benchmark,
            provider_plan=args.provider_plan,
            adjusted_flag=args.adjusted_flag,
            min_coverage_ratio=float(args.min_coverage_ratio),
            conflict_tolerance_pct=float(args.conflict_tolerance_pct),
            severe_conflict_limit=int(args.severe_conflict_limit),
            timeout_seconds=float(args.timeout_seconds),
        )
    )
    payload = {
        "status": result.status,
        "refresh_run_id": result.refresh_run_id,
        "manifest_path": result.manifest_path,
        "registered_market_dataset_id": result.registered_market_dataset_id,
        "blockers": result.blockers,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: {result.manifest_path}")
    return 0 if result.status in {"ok", "skipped"} else 2


def _resolve_config(config: RefreshConfig) -> RefreshConfig:
    as_of = pd.Timestamp(config.as_of_date).strftime("%Y-%m-%d")
    start = pd.Timestamp(config.start_date or as_of).strftime("%Y-%m-%d")
    return RefreshConfig(
        lake_root=Path(config.lake_root),
        as_of_date=as_of,
        start_date=start,
        symbols=tuple(str(item).strip().upper() for item in config.symbols if str(item).strip()),
        benchmark=str(config.benchmark or "000300.SH").strip().upper(),
        provider_plan=str(config.provider_plan or "default_free"),
        adjusted_flag=str(config.adjusted_flag or "none"),
        min_coverage_ratio=float(config.min_coverage_ratio),
        conflict_tolerance_pct=float(config.conflict_tolerance_pct),
        severe_conflict_limit=int(config.severe_conflict_limit),
        timeout_seconds=float(config.timeout_seconds),
        allow_tdx_family=bool(config.allow_tdx_family),
        run_id=str(config.run_id or ""),
    )


def _build_run_id(config: RefreshConfig) -> str:
    return f"refresh_daily_{config.provider_plan}_{config.as_of_date.replace('-', '')}_{pd.Timestamp.now().strftime('%H%M%S')}"


def _resolve_incremental_start(lake: ResearchDataLake, config: RefreshConfig) -> str:
    expected_universe_key = _universe_key(_request_symbols(config))
    expected_provider_plan = str(config.provider_plan or "")
    expected_benchmark = str(config.benchmark or "").upper()
    expected_adjusted_flag = str(config.adjusted_flag or "none")
    latest = ""
    try:
        rows = lake.list_datasets(dataset_kind="policy_input_bundle")
    except Exception:
        rows = pd.DataFrame()
    for _, row in rows.iterrows():
        try:
            metadata = lake.describe_dataset(str(row["dataset_id"]))
        except Exception:
            continue
        parameters = dict(metadata.get("parameters", {}) or {})
        if str(parameters.get("source", "") or metadata.get("source", "")).lower() != "data_platform_refresh":
            continue
        if str(parameters.get("refresh_universe_key", "") or "") != expected_universe_key:
            continue
        if str(parameters.get("provider_plan", "") or "") != expected_provider_plan:
            continue
        if str(parameters.get("benchmark", "") or "").upper() != expected_benchmark:
            continue
        if str(parameters.get("adjusted_flag", "none") or "none") != expected_adjusted_flag:
            continue
        end_date = str(metadata.get("end_date", "") or parameters.get("end_date", "") or "")
        if end_date and (not latest or pd.Timestamp(end_date) > pd.Timestamp(latest)):
            latest = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if latest:
        return next_business_date(latest)
    return config.start_date


def _write_bronze(data: pd.DataFrame, bronze_root: Path, provider_chain: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for provider in provider_chain:
        safe = _safe_name(provider)
        path = bronze_root / f"{safe}.parquet"
        subset = data.loc[data["source"].astype(str).str.lower() == str(provider).lower()].copy() if not data.empty else pd.DataFrame(columns=STANDARD_MARKET_COLUMNS)
        subset.to_parquet(path, index=False)
        paths[str(provider)] = str(path.resolve())
    return paths


def _build_silver_market(
    data: pd.DataFrame,
    *,
    provider_priority: list[str],
    conflict_tolerance_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if data.empty:
        empty = pd.DataFrame(columns=STANDARD_MARKET_COLUMNS)
        return empty, pd.DataFrame(columns=["trade_date", "symbol", "field", "min_value", "max_value", "relative_diff", "providers"]), {
            "source_conflict_count": 0,
            "severe_conflict_count": 0,
        }
    working = data.copy()
    priority = {str(provider).lower(): idx for idx, provider in enumerate(provider_priority)}
    working["_priority"] = working["source"].astype(str).str.lower().map(priority).fillna(9999).astype(int)
    working["_valid"] = _valid_mask(working)
    ordered = working.sort_values(["trade_date", "symbol", "_valid", "_priority"], ascending=[True, True, False, True])
    canonical = ordered.drop_duplicates(subset=["trade_date", "symbol"], keep="first").copy()
    canonical = canonical.drop(columns=["_priority", "_valid"], errors="ignore")
    conflicts: list[dict[str, Any]] = []
    for (trade_date, symbol), group in working.groupby(["trade_date", "symbol"]):
        if len(group) <= 1:
            continue
        providers = ",".join(sorted(group["source"].astype(str).unique()))
        for field in ["open", "high", "low", "close", "volume", "amount"]:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            if len(values) <= 1:
                continue
            min_value = float(values.min())
            max_value = float(values.max())
            denom = max(abs(min_value), 1e-12)
            relative_diff = abs(max_value - min_value) / denom
            if relative_diff > float(conflict_tolerance_pct):
                conflicts.append(
                    {
                        "trade_date": str(trade_date),
                        "symbol": str(symbol),
                        "field": field,
                        "min_value": min_value,
                        "max_value": max_value,
                        "relative_diff": float(relative_diff),
                        "providers": providers,
                    }
                )
    conflict_report = pd.DataFrame(conflicts)
    severe = int(len(conflict_report))
    return canonical[STANDARD_MARKET_COLUMNS].sort_values(["trade_date", "symbol"]).reset_index(drop=True), conflict_report, {
        "source_conflict_count": severe,
        "severe_conflict_count": severe,
        "conflict_tolerance_pct": float(conflict_tolerance_pct),
    }


def _coverage_report(canonical: pd.DataFrame, request: FetchRequest) -> dict[str, Any]:
    expected_dates = market_business_dates(request.start_date, request.end_date)
    expected_rows = int(len(request.symbols) * len(expected_dates))
    row_count = int(len(canonical))
    coverage_ratio = float(row_count / expected_rows) if expected_rows else 0.0
    return {
        "status": "ok" if row_count and coverage_ratio >= 1.0 else "partial" if row_count else "empty",
        "start_date": request.start_date,
        "end_date": request.end_date,
        "expected_rows": expected_rows,
        "row_count": row_count,
        "coverage_ratio": coverage_ratio,
        "symbol_count": int(canonical["symbol"].nunique()) if not canonical.empty else 0,
        "trade_date_count": int(canonical["trade_date"].nunique()) if not canonical.empty else 0,
    }


def _refresh_blockers(
    *,
    coverage_report: dict[str, Any],
    conflict_summary: dict[str, Any],
    min_coverage_ratio: float,
    severe_conflict_limit: int,
) -> list[str]:
    blockers: list[str] = []
    if float(coverage_report.get("coverage_ratio", 0.0) or 0.0) < float(min_coverage_ratio):
        blockers.append("coverage_below_threshold")
    if int(conflict_summary.get("severe_conflict_count", 0) or 0) > int(severe_conflict_limit):
        blockers.append("severe_source_conflict")
    if int(coverage_report.get("row_count", 0) or 0) <= 0:
        blockers.append("empty_canonical_market")
    return [item for item in blockers if not item.startswith("_")]


def _register_policy_input_bundle(
    *,
    lake: ResearchDataLake,
    config: RefreshConfig,
    refresh_run_id: str,
    canonical: pd.DataFrame,
    provider_chain: list[str],
    coverage_report: dict[str, Any],
    conflict_summary: dict[str, Any],
) -> str:
    benchmark = str(config.benchmark).upper()
    market = canonical.loc[canonical["symbol"] != benchmark].copy()
    benchmark_frame = canonical.loc[canonical["symbol"] == benchmark].copy()
    request_symbols = _request_symbols(config)
    if market.empty:
        raise ValueError("refresh_blocked: canonical market has no non-benchmark rows")
    if benchmark_frame.empty:
        raise ValueError(f"refresh_blocked: canonical market is missing benchmark {benchmark}")
    market_frames = {field.title() if field != "amount" else "Amount": _pivot(market, field) for field in ["open", "high", "low", "close", "volume", "amount"]}
    market_frames = {
        "Open": market_frames["Open"],
        "High": market_frames["High"],
        "Low": market_frames["Low"],
        "Close": market_frames["Close"],
        "Volume": market_frames["Volume"],
        "Amount": market_frames["Amount"],
    }
    close = market_frames["Close"]
    membership = close.notna().astype(bool)
    feature_zero = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    record = lake.save_market_data_bundle(
        spec={
            "dataset": "policy_input_bundle",
            "source": "data_platform_refresh",
            "provider_plan": config.provider_plan,
            "provider_chain": list(provider_chain),
            "refresh_run_id": refresh_run_id,
            "adjusted_flag": config.adjusted_flag,
            "requested_symbols": list(request_symbols),
            "refresh_universe_key": _universe_key(request_symbols),
            "start_date": str(coverage_report.get("start_date", "")),
            "end_date": str(coverage_report.get("end_date", "")),
            "benchmark": benchmark,
            "coverage_report": dict(coverage_report),
            "conflict_summary": dict(conflict_summary),
        },
        market_frames=market_frames,
        benchmark_close=_series(benchmark_frame, "close", name=benchmark),
        benchmark_open=_series(benchmark_frame, "open", name=benchmark),
        membership_frame=membership,
        feature_frames={
            "score_none": feature_zero,
            "score_v2": feature_zero.copy(),
            "score_blend": feature_zero.copy(),
        },
        source="data_platform_refresh",
        reuse=False,
    )
    return record.dataset_id


def _pivot(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    out = frame.pivot(index="trade_date", columns="symbol", values=field).sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = None
    out.columns.name = None
    return out


def _series(frame: pd.DataFrame, field: str, *, name: str) -> pd.Series:
    ordered = frame.sort_values("trade_date")
    out = pd.Series(pd.to_numeric(ordered[field], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(ordered["trade_date"]), name=name)
    out.index.name = None
    return out


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[NUMERIC_MARKET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    positive_prices = numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
    range_valid = numeric["high"].ge(numeric[["open", "close", "low"]].max(axis=1)) & numeric["low"].le(numeric[["open", "close", "high"]].min(axis=1))
    activity_valid = numeric[["volume", "amount"]].ge(0).all(axis=1)
    return positive_prices & range_valid & activity_valid


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "provider"))


def _request_symbols(config: RefreshConfig) -> tuple[str, ...]:
    return FetchRequest(
        symbols=tuple(dict.fromkeys([*config.symbols, config.benchmark])),
        start_date=config.start_date or config.as_of_date,
        end_date=config.as_of_date,
        adjusted_flag=config.adjusted_flag,
    ).normalized().symbols


def _universe_key(symbols: Iterable[str]) -> str:
    normalized = sorted(dict.fromkeys(str(item).strip().upper() for item in symbols if str(item).strip()))
    payload = "\n".join(normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
