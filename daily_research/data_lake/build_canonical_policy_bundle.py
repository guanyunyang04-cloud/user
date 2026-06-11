from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from daily_research.data_lake.canonical import (
    DEFAULT_CANONICAL_ALIAS,
    DEFAULT_CANONICAL_START_DATE,
    build_coverage_report,
    write_canonical_manifest,
)
from daily_research.data_lake.catalog import (
    DATA_LAKE_SCHEMA_VERSION,
    DEFAULT_DATA_LAKE_ROOT,
    LakeDatasetRecord,
    ResearchDataLake,
    _stable_hash,
    _write_json,
)
from daily_research.data_lake.policy_input_loader import _read_table_paths
from daily_research.data_platform.contracts import DataDomain


@dataclass(frozen=True)
class BuildCanonicalPolicyBundleConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    market_daily_dataset_id: str = ""
    sidecar_dataset_ids: Mapping[str, str] = field(default_factory=dict)
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    benchmark: str = "000300.SH"
    benchmark_source_dataset_id: str = ""
    source: str = "canonical_data_v1"
    alias: str = DEFAULT_CANONICAL_ALIAS
    write_manifest: bool = True
    reuse: bool = True

    def normalized(self) -> "BuildCanonicalPolicyBundleConfig":
        sidecars = {
            str(key).strip(): str(value).strip()
            for key, value in dict(self.sidecar_dataset_ids or {}).items()
            if str(key).strip() and str(value).strip()
        }
        return BuildCanonicalPolicyBundleConfig(
            lake_root=Path(self.lake_root),
            market_daily_dataset_id=str(self.market_daily_dataset_id or "").strip(),
            sidecar_dataset_ids=sidecars,
            start_date=str(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=str(self.end_date or "").strip(),
            benchmark=str(self.benchmark or "000300.SH").strip().upper(),
            benchmark_source_dataset_id=str(self.benchmark_source_dataset_id or "").strip(),
            source=str(self.source or "canonical_data_v1"),
            alias=str(self.alias or DEFAULT_CANONICAL_ALIAS),
            write_manifest=bool(self.write_manifest),
            reuse=bool(self.reuse),
        )


@dataclass(frozen=True)
class BuildCanonicalPolicyBundleResult:
    status: str
    dataset_id: str
    canonical_manifest_path: str = ""
    row_count: int = 0
    membership_rows: int = 0
    membership_symbols: int = 0
    benchmark_rows: int = 0
    sidecar_dataset_ids: dict[str, str] = field(default_factory=dict)


def build_canonical_policy_bundle(config: BuildCanonicalPolicyBundleConfig) -> BuildCanonicalPolicyBundleResult:
    cfg = config.normalized()
    if not cfg.market_daily_dataset_id:
        raise ValueError("market_daily_dataset_id_required")
    lake = ResearchDataLake(cfg.lake_root)
    market_metadata = lake.describe_dataset(cfg.market_daily_dataset_id)
    market_paths = dict(market_metadata.get("content_paths", {}) or {})
    market = _normalize_market_frame(_read_market_dataset_frame(market_metadata))
    if market.empty:
        raise ValueError(f"canonical_bundle_blocker: empty_market_daily_dataset: {cfg.market_daily_dataset_id}")
    start_ts = pd.Timestamp(cfg.start_date)
    end_ts = pd.Timestamp(cfg.end_date or str(market["trade_date"].max()))
    market = market.loc[(market["trade_date"] >= start_ts) & (market["trade_date"] <= end_ts)].copy()
    if market.empty:
        raise ValueError(
            "canonical_bundle_blocker: no_market_rows_after_date_filter: "
            f"{cfg.market_daily_dataset_id} {cfg.start_date}->{end_ts.strftime('%Y-%m-%d')}"
        )
    start_date = market["trade_date"].min().strftime("%Y-%m-%d")
    end_date = market["trade_date"].max().strftime("%Y-%m-%d")
    benchmark = _build_benchmark_frame(
        lake=lake,
        market=market,
        benchmark=cfg.benchmark,
        benchmark_source_dataset_id=cfg.benchmark_source_dataset_id,
        start_date=start_date,
        end_date=end_date,
    )
    membership = _build_membership_frame(market=market, benchmark=cfg.benchmark)
    if membership.empty or len(membership.columns) <= 1:
        raise ValueError("canonical_bundle_blocker: empty_membership")

    spec = _bundle_spec(
        cfg,
        market_metadata=market_metadata,
        start_date=start_date,
        end_date=end_date,
        membership_symbols=max(0, len(membership.columns) - 1),
    )
    record = _save_reference_policy_bundle(
        lake=lake,
        spec=spec,
        source=cfg.source,
        market_paths=market_paths,
        market_metadata=market_metadata,
        benchmark=benchmark,
        membership=membership,
        row_count=int(len(market)),
        reuse=cfg.reuse,
    )
    manifest_path = ""
    if cfg.write_manifest:
        coverage = build_coverage_report(lake, start_date=cfg.start_date)
        manifest_path = str(
            write_canonical_manifest(
                lake,
                dataset_id=record.dataset_id,
                alias=cfg.alias,
                start_date=start_date,
                end_date=end_date,
                sidecar_dataset_ids=dict(cfg.sidecar_dataset_ids),
                coverage_report=coverage,
                notes="canonical policy bundle built from manifest-referenced market_daily and sidecar datasets",
            )
        )
    return BuildCanonicalPolicyBundleResult(
        status=record.status,
        dataset_id=record.dataset_id,
        canonical_manifest_path=manifest_path,
        row_count=int(len(market)),
        membership_rows=int(len(membership)),
        membership_symbols=max(0, len(membership.columns) - 1),
        benchmark_rows=int(len(benchmark)),
        sidecar_dataset_ids=dict(cfg.sidecar_dataset_ids),
    )


def _read_market_dataset_frame(metadata: Mapping[str, Any]) -> pd.DataFrame:
    paths = dict(metadata.get("content_paths", {}) or {})
    if str(metadata.get("dataset_kind", "") or "") == "policy_input_bundle":
        return _read_table_paths(paths, data_key="bronze_market_data", manifest_keys=("bronze_market_shard_manifest", "shard_manifest"))
    return _read_table_paths(paths, data_key="silver_domain_data", manifest_keys=("shard_manifest",))


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"])
    data = frame.copy()
    rename = {}
    if "date" in data.columns and "trade_date" not in data.columns:
        rename["date"] = "trade_date"
    if "stock" in data.columns and "symbol" not in data.columns:
        rename["stock"] = "symbol"
    if "code" in data.columns and "symbol" not in data.columns:
        rename["code"] = "symbol"
    if rename:
        data = data.rename(columns=rename)
    required = {"trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"canonical_bundle_blocker: market_daily_missing_columns: {sorted(missing)}")
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["trade_date", "symbol"])
    data = data.loc[data["symbol"].ne("")]
    return data.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _build_benchmark_frame(
    *,
    lake: ResearchDataLake,
    market: pd.DataFrame,
    benchmark: str,
    benchmark_source_dataset_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    benchmark_symbol = str(benchmark or "000300.SH").strip().upper()
    data = market.loc[market["symbol"].eq(benchmark_symbol)].copy()
    if data.empty and benchmark_source_dataset_id:
        data = _read_benchmark_from_policy_bundle(
            lake=lake,
            dataset_id=benchmark_source_dataset_id,
            benchmark=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
        )
    if data.empty:
        raise ValueError(f"canonical_bundle_blocker: missing_benchmark_symbol: {benchmark_symbol}")
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.loc[(data["trade_date"] >= pd.Timestamp(start_date)) & (data["trade_date"] <= pd.Timestamp(end_date))].copy()
    out = pd.DataFrame(
        {
            "trade_date": data["trade_date"].dt.strftime("%Y-%m-%d"),
            "benchmark": benchmark_symbol,
            "close": pd.to_numeric(data["close"], errors="coerce"),
        }
    )
    if "open" in data.columns:
        out["open"] = pd.to_numeric(data["open"], errors="coerce")
    return out.drop_duplicates(subset=["trade_date", "benchmark"], keep="last").sort_values("trade_date").reset_index(drop=True)


def _read_benchmark_from_policy_bundle(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    benchmark: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    metadata = lake.describe_dataset(dataset_id)
    if str(metadata.get("dataset_kind", "") or "") != "policy_input_bundle":
        data = _normalize_market_frame(_read_market_dataset_frame(metadata))
        data = data.loc[data["symbol"].astype(str).str.strip().str.upper().eq(benchmark)].copy()
        if data.empty:
            return pd.DataFrame()
        return data.loc[(data["trade_date"] >= pd.Timestamp(start_date)) & (data["trade_date"] <= pd.Timestamp(end_date))].copy()
    paths = dict(metadata.get("content_paths", {}) or {})
    path = Path(str(paths.get("silver_benchmark", "") or ""))
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_parquet(path)
    if "trade_date" not in data.columns:
        return pd.DataFrame()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    if "benchmark" in data.columns:
        data = data.loc[data["benchmark"].astype(str).str.strip().str.upper().eq(benchmark)].copy()
    data = data.loc[(data["trade_date"] >= pd.Timestamp(start_date)) & (data["trade_date"] <= pd.Timestamp(end_date))].copy()
    if data.empty:
        return pd.DataFrame()
    data["symbol"] = benchmark
    for column in ["open", "close"]:
        if column not in data.columns:
            data[column] = data["close"] if "close" in data.columns else pd.NA
    return data


def _build_membership_frame(*, market: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    data = market.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    benchmark_symbol = str(benchmark or "").strip().upper()
    data = data.loc[data["symbol"].map(_is_stock_symbol) & data["symbol"].ne(benchmark_symbol)].copy()
    value_columns = [column for column in ("open", "high", "low", "close", "volume", "amount") if column in data.columns]
    data["_present"] = data[value_columns].notna().any(axis=1) if value_columns else True
    membership = data.pivot_table(index="trade_date", columns="symbol", values="_present", aggfunc="max", fill_value=False)
    membership = membership.astype(bool).sort_index()
    membership.insert(0, "date", membership.index.strftime("%Y-%m-%d"))
    membership = membership.reset_index(drop=True)
    membership.columns.name = None
    return membership


def _is_stock_symbol(value: str) -> bool:
    text = str(value or "").strip().upper()
    if len(text) != 9 or text[6] != ".":
        return False
    return text[:6].isdigit() and text[-2:] in {"SH", "SZ", "BJ"}


def _bundle_spec(
    cfg: BuildCanonicalPolicyBundleConfig,
    *,
    market_metadata: Mapping[str, Any],
    start_date: str,
    end_date: str,
    membership_symbols: int,
) -> dict[str, Any]:
    return {
        "dataset": "canonical_policy_input_bundle",
        "canonical_alias": cfg.alias,
        "canonical_start_date": DEFAULT_CANONICAL_START_DATE,
        "start_date": start_date,
        "end_date": end_date,
        "benchmark": cfg.benchmark,
        "source": cfg.source,
        "market_daily_dataset_id": cfg.market_daily_dataset_id,
        "market_daily_dataset_kind": str(market_metadata.get("dataset_kind", "") or ""),
        "market_linkage_policy": "manifest_reference_no_ohlcv_rewrite",
        "raw_ohlcv_policy": "raw_prices_preserved_adjusted_prices_as_sidecars",
        "sidecar_dataset_ids": dict(cfg.sidecar_dataset_ids),
        "benchmark_source_dataset_id": cfg.benchmark_source_dataset_id,
        "membership_policy": "available_stock_rows_from_market_daily",
        "membership_symbols": int(membership_symbols),
    }


def _save_reference_policy_bundle(
    *,
    lake: ResearchDataLake,
    spec: Mapping[str, Any],
    source: str,
    market_paths: Mapping[str, Any],
    market_metadata: Mapping[str, Any],
    benchmark: pd.DataFrame,
    membership: pd.DataFrame,
    row_count: int,
    reuse: bool,
) -> LakeDatasetRecord:
    dataset_kind = "policy_input_bundle"
    zone = "research"
    fingerprint = _stable_hash(
        {
            "schema_version": DATA_LAKE_SCHEMA_VERSION,
            "dataset_kind": dataset_kind,
            "zone": zone,
            "spec": dict(spec),
        }
    )
    dataset_id = f"{dataset_kind}__{fingerprint}"
    dataset_dir = lake.parquet_root / "bronze_silver" / dataset_kind / fingerprint
    feature_dir = dataset_dir / "silver_feature_panels"
    content_paths = {
        "bronze_market_data": _market_data_path(market_paths),
        "bronze_market_shard_manifest": str(market_paths.get("shard_manifest", "") or ""),
        "silver_benchmark": str((dataset_dir / "silver_benchmark.parquet").resolve()),
        "silver_membership": str((dataset_dir / "silver_membership.parquet").resolve()),
        "silver_feature_values": str((feature_dir / "*.parquet").resolve()),
        "silver_feature_panels": str((feature_dir / "*.parquet").resolve()),
        "bundle_manifest": str((dataset_dir / "bundle_manifest.json").resolve()),
    }
    existing = lake._existing_by_fingerprint(fingerprint)
    if reuse and existing is not None and Path(content_paths["silver_benchmark"]).exists() and Path(content_paths["silver_membership"]).exists():
        metadata = lake.describe_dataset(str(existing["dataset_id"]))
        return LakeDatasetRecord(
            dataset_id=str(metadata["dataset_id"]),
            dataset_kind=str(metadata["dataset_kind"]),
            zone=str(metadata["zone"]),
            fingerprint=str(metadata["fingerprint"]),
            status="hit",
            root=lake.root,
            content_paths=dict(metadata.get("content_paths", {}) or {}),
            row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
            metadata=metadata,
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_parquet(content_paths["silver_benchmark"], index=False)
    membership.to_parquet(content_paths["silver_membership"], index=False)
    row_counts: dict[str, Any] = {
        "bronze_market_data": int(row_count),
        "silver_benchmark": int(len(benchmark)),
        "silver_membership": int(len(membership)),
        "silver_feature_panels": 0,
        "silver_feature_cells": 0,
        "_date_bounds": {"start_date": str(spec.get("start_date", "") or ""), "end_date": str(spec.get("end_date", "") or "")},
    }
    manifest = {
        "dataset_id": dataset_id,
        "dataset_kind": dataset_kind,
        "fingerprint": fingerprint,
        "status": "stored",
        "parameters": dict(spec),
        "content_paths": content_paths,
        "market_daily_dataset_id": str(spec.get("market_daily_dataset_id", "") or ""),
        "market_daily_source_paths": dict(market_paths),
        "market_daily_source_metadata": {
            "dataset_id": str(market_metadata.get("dataset_id", "") or ""),
            "dataset_kind": str(market_metadata.get("dataset_kind", "") or ""),
            "start_date": str(market_metadata.get("start_date", "") or ""),
            "end_date": str(market_metadata.get("end_date", "") or ""),
            "row_counts": dict(market_metadata.get("row_counts", {}) or {}),
        },
    }
    _write_json(Path(content_paths["bundle_manifest"]), manifest)
    lake._upsert_dataset(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        domain="bronze_silver",
        zone=zone,
        source=str(source or ""),
        spec=dict(spec),
        label_completeness_summary={},
        content_paths=content_paths,
        row_counts=row_counts,
        source_cache={
            "market_daily_dataset_id": str(spec.get("market_daily_dataset_id", "") or ""),
            "market_linkage_policy": str(spec.get("market_linkage_policy", "") or ""),
            "sidecar_dataset_ids": dict(spec.get("sidecar_dataset_ids", {}) or {}),
            "bundle_manifest": content_paths["bundle_manifest"],
        },
        fingerprint=fingerprint,
        status="stored",
    )
    metadata = lake.describe_dataset(dataset_id)
    return LakeDatasetRecord(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        zone=zone,
        fingerprint=fingerprint,
        status="stored",
        root=lake.root,
        content_paths=dict(metadata.get("content_paths", {}) or {}),
        row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
        metadata=metadata,
    )


def _market_data_path(paths: Mapping[str, Any]) -> str:
    for key in ("silver_domain_data", "bronze_market_data"):
        value = str(paths.get(key, "") or "")
        if value:
            return value
    return ""


def _parse_sidecars(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(text or "").split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid_sidecar_dataset_id: {part}; expected domain=dataset_id")
        key, value = part.split("=", 1)
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if not normalized_key or not normalized_value:
            raise ValueError(f"invalid_sidecar_dataset_id: {part}")
        out[normalized_key] = normalized_value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical policy_input_bundle from canonical market_daily and sidecars.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--market-daily-dataset-id", required=True)
    parser.add_argument("--sidecar-dataset-ids", default="")
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--benchmark-source-dataset-id", default="")
    parser.add_argument("--source", default="canonical_data_v1")
    parser.add_argument("--alias", default=DEFAULT_CANONICAL_ALIAS)
    parser.add_argument("--write-canonical-manifest", dest="write_manifest", action="store_true", default=True)
    parser.add_argument("--no-write-canonical-manifest", dest="write_manifest", action="store_false")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_canonical_policy_bundle(
        BuildCanonicalPolicyBundleConfig(
            lake_root=Path(args.lake_root),
            market_daily_dataset_id=args.market_daily_dataset_id,
            sidecar_dataset_ids=_parse_sidecars(args.sidecar_dataset_ids),
            start_date=args.start_date,
            end_date=args.end_date,
            benchmark=args.benchmark,
            benchmark_source_dataset_id=args.benchmark_source_dataset_id,
            source=args.source,
            alias=args.alias,
            write_manifest=bool(args.write_manifest),
            reuse=bool(args.reuse),
        )
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
