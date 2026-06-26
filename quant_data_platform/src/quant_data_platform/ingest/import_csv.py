from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data_platform.lake import ResearchDataLake
from quant_data_platform.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    normalize_domain,
    normalize_domain_frame,
)
from quant_data_platform.ingest.refresh_daily import (
    RefreshConfig,
    _build_silver_domain,
    _domain_coverage_report,
    _json_safe,
    _register_policy_input_bundle,
    _safe_name,
    _universe_key,
)


@dataclass(frozen=True)
class CsvImportConfig:
    input_path: Path | str
    lake_root: Path | str = Path("daily_research/output/research_data_lake")
    domain: str = DataDomain.MARKET_DAILY
    as_of_date: str = ""
    source_name: str = "manual_csv"
    benchmark: str = "000300.SH"
    adjusted_flag: str = "none"
    min_coverage_ratio: float = 0.0
    run_id: str = ""


@dataclass(frozen=True)
class CsvImportResult:
    status: str
    import_run_id: str
    manifest_path: str
    bronze_path: str
    silver_path: str
    registered_dataset_id: str = ""
    blockers: list[str] = field(default_factory=list)
    coverage_report: dict[str, Any] = field(default_factory=dict)


def run_import_csv(config: CsvImportConfig) -> CsvImportResult:
    resolved = _resolve_config(config)
    domain = normalize_domain(resolved.domain)
    if domain != DataDomain.MARKET_DAILY:
        raise ValueError("CSV import v1 supports domain=market_daily only")
    lake = ResearchDataLake(resolved.lake_root)
    run_id = resolved.run_id or f"csv_import_{_safe_name(resolved.source_name)}_{resolved.as_of_date.replace('-', '')}_{pd.Timestamp.now().strftime('%H%M%S')}"
    run_root = Path(resolved.lake_root) / "data_platform" / "csv_imports" / run_id
    bronze_root = run_root / "bronze"
    silver_root = run_root / "silver"
    bronze_root.mkdir(parents=True, exist_ok=True)
    silver_root.mkdir(parents=True, exist_ok=True)

    raw = _load_csv_input(resolved.input_path)
    normalized = normalize_domain_frame(
        raw,
        domain=domain,
        source=resolved.source_name,
        as_of_date=resolved.as_of_date,
        adjusted_flag=resolved.adjusted_flag,
        require_columns=True,
    )
    bronze_path = bronze_root / f"{_safe_name(resolved.source_name)}.parquet"
    normalized.to_parquet(bronze_path, index=False)
    canonical, conflict_report, conflict_summary = _build_silver_domain(
        normalized,
        domain=domain,
        provider_priority=[resolved.source_name],
        conflict_tolerance_pct=0.0,
    )
    silver_path = silver_root / "silver_market_daily.parquet"
    conflict_path = silver_root / "source_conflict_report_market_daily.parquet"
    canonical.to_parquet(silver_path, index=False)
    conflict_report.to_parquet(conflict_path, index=False)
    symbols = tuple(dict.fromkeys(str(item).upper() for item in canonical["symbol"].dropna().astype(str))) if not canonical.empty else ()
    request = DomainFetchRequest(
        domain=domain,
        symbols=symbols,
        start_date=str(canonical["trade_date"].min()) if not canonical.empty else resolved.as_of_date,
        end_date=str(canonical["trade_date"].max()) if not canonical.empty else resolved.as_of_date,
        adjusted_flag=resolved.adjusted_flag,
    )
    coverage = _domain_coverage_report(canonical, request, calendar=None)
    blockers: list[str] = []
    benchmark = str(resolved.benchmark).upper()
    if canonical.empty:
        blockers.append("empty_canonical_market")
    if not canonical.empty and not canonical["symbol"].astype(str).str.upper().eq(benchmark).any():
        blockers.append("missing_benchmark")
    if float(coverage.get("coverage_ratio", 0.0) or 0.0) < float(resolved.min_coverage_ratio):
        blockers.append("coverage_below_threshold")
    dataset_id = ""
    status = "blocked" if blockers else "ok"
    if not blockers:
        refresh_config = RefreshConfig(
            lake_root=resolved.lake_root,
            as_of_date=resolved.as_of_date,
            start_date=str(coverage.get("start_date", request.start_date) or request.start_date),
            symbols=symbols,
            benchmark=benchmark,
            provider_plan="csv_import",
            adjusted_flag=resolved.adjusted_flag,
            run_id=run_id,
        )
        dataset_id = _register_policy_input_bundle(
            lake=lake,
            config=refresh_config,
            refresh_run_id=run_id,
            request_symbols=symbols,
            canonical=canonical,
            provider_chain=[resolved.source_name],
            coverage_report=coverage,
            conflict_summary=conflict_summary,
        )
    manifest_path = run_root / "csv_import_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": status,
        "import_run_id": run_id,
        "provider_plan": "csv_import",
        "source_name": resolved.source_name,
        "domain": domain,
        "as_of_date": resolved.as_of_date,
        "input_path": str(resolved.input_path),
        "bronze_path": str(bronze_path.resolve()),
        "silver_path": str(silver_path.resolve()),
        "conflict_report_path": str(conflict_path.resolve()),
        "coverage_report": coverage,
        "conflict_summary": conflict_summary,
        "blockers": blockers,
        "registered_dataset_id": dataset_id,
        "refresh_universe_key": _universe_key(symbols),
    }
    _write_json(manifest_path, manifest)
    lake.write_catalog_manifest()
    return CsvImportResult(
        status=status,
        import_run_id=run_id,
        manifest_path=str(manifest_path.resolve()),
        bronze_path=str(bronze_path.resolve()),
        silver_path=str(silver_path.resolve()),
        registered_dataset_id=dataset_id,
        blockers=blockers,
        coverage_report=coverage,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import CSV data into the QDP Bronze/Silver lake.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--domain", default=DataDomain.MARKET_DAILY)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--source-name", default="manual_csv")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_import_csv(
        CsvImportConfig(
            input_path=Path(args.input),
            lake_root=Path(args.data_lake_root) if str(args.data_lake_root or "").strip() else Path("daily_research/output/research_data_lake"),
            domain=args.domain,
            as_of_date=args.as_of_date,
            source_name=args.source_name,
            benchmark=args.benchmark,
            adjusted_flag=args.adjusted_flag,
        )
    )
    payload = {
        "status": result.status,
        "import_run_id": result.import_run_id,
        "manifest_path": result.manifest_path,
        "registered_dataset_id": result.registered_dataset_id,
        "blockers": result.blockers,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: {result.manifest_path}")
    return 0 if result.status == "ok" else 2


def _resolve_config(config: CsvImportConfig) -> CsvImportConfig:
    return CsvImportConfig(
        input_path=Path(config.input_path),
        lake_root=Path(config.lake_root),
        domain=normalize_domain(config.domain),
        as_of_date=pd.Timestamp(config.as_of_date).strftime("%Y-%m-%d"),
        source_name=str(config.source_name or "manual_csv").strip().lower(),
        benchmark=str(config.benchmark or "000300.SH").strip().upper(),
        adjusted_flag=str(config.adjusted_flag or "none"),
        min_coverage_ratio=float(config.min_coverage_ratio),
        run_id=str(config.run_id or ""),
    )


def _load_csv_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    if not path.is_dir():
        raise ValueError(f"CSV input path does not exist: {path}")
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(path.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if "symbol" not in frame.columns:
            frame["symbol"] = csv_path.stem.upper()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
