from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("intraday_", "intraday_context"),
    ("cs_rank_intraday_", "intraday_context"),
    ("cs_z_intraday_", "intraday_context"),
    ("valuation_", "valuation_context"),
    ("turn", "turnover_context"),
    ("cs_rank_turn", "turnover_context"),
    ("cs_z_turn", "turnover_context"),
    ("industry_rank_turn", "turnover_context"),
    ("industry_z_turn", "turnover_context"),
    ("adjust_", "adjust_context"),
    ("cs_rank_adjust_", "adjust_context"),
    ("cs_z_adjust_", "adjust_context"),
    ("index_", "index_context"),
    ("raw_", "raw_kline"),
    ("market_", "market_context"),
    ("benchmark_", "market_context"),
    ("industry_", "sector_relative_context"),
    ("stock_ret_", "sector_relative_context"),
    ("peer_", "peer_context"),
    ("relative_to_peer_", "peer_context"),
    ("local_", "local_state_context"),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _infer_group(name: str, manifest_group_map: Mapping[str, str]) -> str:
    feature = str(name)
    if feature in manifest_group_map:
        return str(manifest_group_map[feature])
    for prefix, group in DEFAULT_GROUP_PREFIXES:
        if feature.startswith(prefix):
            return group
    return "unknown"


def _feature_group_map_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    for shard in list(manifest.get("shards", []) or []):
        if str(shard.get("status", "")) != "completed":
            continue
        shard_manifest_path = str(shard.get("shard_manifest_json", "") or "")
        if not shard_manifest_path or not Path(shard_manifest_path).exists():
            continue
        shard_manifest = _read_json(shard_manifest_path)
        feature_manifest = dict(shard_manifest.get("feature_manifest", {}) or {})
        column_groups = dict(feature_manifest.get("column_groups", {}) or {})
        out: dict[str, str] = {}
        for group, columns in column_groups.items():
            if isinstance(columns, list):
                for column in columns:
                    out[str(column)] = str(group)
        if out:
            return out
    return {}


def _iter_completed_shards(
    manifest: Mapping[str, Any],
    *,
    years: set[int] | None = None,
    max_shards: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shard in list(manifest.get("shards", []) or []):
        if str(shard.get("status", "")) != "completed":
            continue
        year = int(shard.get("year", 0) or 0)
        if years is not None and year not in years:
            continue
        out.append(dict(shard))
        if int(max_shards) > 0 and len(out) >= int(max_shards):
            break
    return out


def _parse_years(value: str | Iterable[int] | None) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        years: set[int] = set()
        for part in text.split(","):
            item = part.strip()
            if not item:
                continue
            if "-" in item:
                start, end = item.split("-", 1)
                years.update(range(int(start), int(end) + 1))
            else:
                years.add(int(item))
        return years
    return {int(item) for item in value}


def audit_sharded_memmap_feature_coverage(
    *,
    manifest_json: str | Path,
    output_root: str | Path,
    run_tag: str = "qdp_sharded_feature_coverage_audit",
    years: str | Iterable[int] | None = None,
    max_shards: int = 0,
    chunk_features: int = 32,
    train_year_start: int = 2012,
    train_year_end: int = 2023,
    recent_year_start: int = 2024,
    recent_year_end: int = 2025,
    low_rate_threshold: float = 0.50,
    train_warn_threshold: float = 0.75,
    recent_warn_threshold: float = 0.90,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    manifest = _read_json(manifest_path)
    if str(manifest.get("artifact_type", "")) != "qdp_sharded_memmap":
        raise ValueError(f"requires qdp_sharded_memmap manifest: {manifest_path}")
    features = [str(item) for item in list(manifest.get("feature_columns", []) or [])]
    if not features:
        raise ValueError("manifest has no feature_columns")
    selected_years = _parse_years(years)
    shards = _iter_completed_shards(manifest, years=selected_years, max_shards=int(max_shards))
    if not shards:
        raise ValueError("no completed shards selected")

    output_dir = Path(output_root) / str(run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_map = _feature_group_map_from_manifest(manifest)
    feature_groups = {feature: _infer_group(feature, group_map) for feature in features}

    year_feature_counts: dict[int, np.ndarray] = {}
    year_rows: dict[int, int] = {}
    for shard in shards:
        year = int(shard.get("year", 0) or 0)
        shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        if len(shape) != 3 or int(shape[2]) != len(features):
            raise ValueError(f"invalid feature_store_shape for shard {shard.get('shard_key', '')}: {shape}")
        store = np.memmap(str(shard["feature_store_path"]), dtype="float32", mode="r", shape=shape)
        row_count = int(shape[0] * shape[1])
        year_rows[year] = int(year_rows.get(year, 0) + row_count)
        counts = year_feature_counts.setdefault(year, np.zeros(len(features), dtype=np.int64))
        for start in range(0, len(features), int(chunk_features)):
            end = min(start + int(chunk_features), len(features))
            block = np.asarray(store[:, :, start:end], dtype=np.float32)
            counts[start:end] += np.isfinite(block).reshape(-1, end - start).sum(axis=0)
        del store

    feature_year_rows: list[dict[str, Any]] = []
    for year in sorted(year_feature_counts):
        total_rows = int(year_rows[year])
        counts = year_feature_counts[year]
        for idx, feature in enumerate(features):
            finite_count = int(counts[idx])
            feature_year_rows.append(
                {
                    "year": int(year),
                    "feature": feature,
                    "group": feature_groups[feature],
                    "rows": total_rows,
                    "finite_count": finite_count,
                    "finite_rate": float(finite_count / total_rows) if total_rows else float("nan"),
                }
            )
    feature_year = pd.DataFrame(feature_year_rows)

    def _window_mean(series: pd.Series, start: int, end: int) -> float:
        years_for_rows = feature_year.loc[series.index, "year"].astype(int)
        values = series.loc[years_for_rows.between(int(start), int(end))]
        return float(values.mean()) if len(values) else float("nan")

    feature_summary = feature_year.groupby(["feature", "group"], as_index=False).agg(
        min_finite_rate=("finite_rate", "min"),
        mean_finite_rate=("finite_rate", "mean"),
        train_window_mean_finite_rate=("finite_rate", lambda s: _window_mean(s, train_year_start, train_year_end)),
        recent_window_mean_finite_rate=("finite_rate", lambda s: _window_mean(s, recent_year_start, recent_year_end)),
        zero_years=("finite_rate", lambda s: int((s <= 0.0).sum())),
        low_years=("finite_rate", lambda s: int((s < float(low_rate_threshold)).sum())),
    )
    group_year = feature_year.groupby(["year", "group"], as_index=False).agg(
        feature_count=("feature", "count"),
        mean_finite_rate=("finite_rate", "mean"),
        min_finite_rate=("finite_rate", "min"),
        zero_feature_count=("finite_rate", lambda s: int((s <= 0.0).sum())),
        low_feature_count=("finite_rate", lambda s: int((s < float(low_rate_threshold)).sum())),
    )
    group_summary = feature_year.groupby("group", as_index=False).agg(
        feature_count=("feature", "nunique"),
        min_year_feature_rate=("finite_rate", "min"),
        mean_rate=("finite_rate", "mean"),
        zero_feature_year_cells=("finite_rate", lambda s: int((s <= 0.0).sum())),
        low_feature_year_cells=("finite_rate", lambda s: int((s < float(low_rate_threshold)).sum())),
    )
    suspicious = feature_summary[
        feature_summary["zero_years"].astype(int).gt(1)
        | feature_summary["train_window_mean_finite_rate"].astype(float).lt(float(train_warn_threshold))
        | feature_summary["recent_window_mean_finite_rate"].astype(float).lt(float(recent_warn_threshold))
    ].copy()
    if not suspicious.empty:
        suspicious = suspicious.sort_values(
            ["group", "train_window_mean_finite_rate", "recent_window_mean_finite_rate", "feature"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )

    feature_year_csv = _write_frame(output_dir / "feature_year_finite_rates.csv", feature_year)
    feature_summary_csv = _write_frame(output_dir / "feature_coverage_summary.csv", feature_summary)
    group_year_csv = _write_frame(output_dir / "group_year_finite_rates.csv", group_year)
    group_summary_csv = _write_frame(output_dir / "group_coverage_summary.csv", group_summary)
    suspicious_csv = _write_frame(output_dir / "suspicious_feature_coverage.csv", suspicious)

    suspicious_by_group = (
        suspicious.groupby("group").size().sort_values(ascending=False).astype(int).to_dict() if not suspicious.empty else {}
    )
    key_findings = []
    if any(str(item).startswith("turn") or "turn" in str(item) for item in suspicious.get("feature", [])):
        key_findings.append("turnover_context_has_historical_coverage_gap")
    if any(str(item).startswith("valuation_") for item in suspicious.get("feature", [])):
        key_findings.append("valuation_context_has_historical_or_field_specific_coverage_gap")
    if "cs_z_intraday_last_5m_ret" in set(suspicious.get("feature", pd.Series(dtype=str)).astype(str)):
        key_findings.append("cs_z_intraday_last_5m_ret_has_derived_feature_coverage_anomaly")

    report = {
        "status": "completed",
        "run_tag": str(run_tag),
        "artifact_type": "qdp_sharded_feature_coverage_audit",
        "manifest_json": str(manifest_path.resolve()),
        "contract": {
            "read_only": True,
            "registry_impact": "unchanged",
            "memmap_impact": "unchanged",
            "active_artifact_impact": "unchanged",
            "low_rate_threshold": float(low_rate_threshold),
            "train_window": [int(train_year_start), int(train_year_end)],
            "recent_window": [int(recent_year_start), int(recent_year_end)],
            "train_warn_threshold": float(train_warn_threshold),
            "recent_warn_threshold": float(recent_warn_threshold),
        },
        "input": {
            "scanned_shards": int(len(shards)),
            "scanned_rows": int(sum(year_rows.values())),
            "feature_count": int(len(features)),
            "years": sorted(int(item) for item in year_rows),
            "source_feature_profile": str(manifest.get("feature_profile", "")),
            "canonical_dataset_id": str(manifest.get("canonical_dataset_id", "")),
        },
        "summary": {
            "group_count": int(group_summary["group"].nunique()) if not group_summary.empty else 0,
            "suspicious_feature_count": int(len(suspicious)),
            "suspicious_by_group": suspicious_by_group,
            "key_findings": key_findings,
        },
        "outputs": {
            "feature_year_csv": feature_year_csv,
            "feature_summary_csv": feature_summary_csv,
            "group_year_csv": group_year_csv,
            "group_summary_csv": group_summary_csv,
            "suspicious_csv": suspicious_csv,
        },
    }
    report_json = _write_json(output_dir / "coverage_audit_report.json", report)
    report["outputs"]["report_json"] = report_json
    _write_json(output_dir / "coverage_audit_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit feature finite coverage in a QDP sharded memmap.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-tag", default="qdp_sharded_feature_coverage_audit")
    parser.add_argument("--years", default="")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--chunk-features", type=int, default=32)
    parser.add_argument("--train-year-start", type=int, default=2012)
    parser.add_argument("--train-year-end", type=int, default=2023)
    parser.add_argument("--recent-year-start", type=int, default=2024)
    parser.add_argument("--recent-year-end", type=int, default=2025)
    parser.add_argument("--low-rate-threshold", type=float, default=0.50)
    parser.add_argument("--train-warn-threshold", type=float, default=0.75)
    parser.add_argument("--recent-warn-threshold", type=float, default=0.90)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_sharded_memmap_feature_coverage(
        manifest_json=args.manifest_json,
        output_root=args.output_root,
        run_tag=args.run_tag,
        years=args.years or None,
        max_shards=int(args.max_shards),
        chunk_features=int(args.chunk_features),
        train_year_start=int(args.train_year_start),
        train_year_end=int(args.train_year_end),
        recent_year_start=int(args.recent_year_start),
        recent_year_end=int(args.recent_year_end),
        low_rate_threshold=float(args.low_rate_threshold),
        train_warn_threshold=float(args.train_warn_threshold),
        recent_warn_threshold=float(args.recent_warn_threshold),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
