from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_DATA_LAKE_ROOT = Path("daily_research/output/research_data_lake")
DATA_LAKE_SCHEMA_VERSION = 1


def _require_duckdb():
    try:
        import duckdb  # type: ignore

        return duckdb
    except Exception as exc:  # pragma: no cover - exercised only on missing optional dependency
        raise RuntimeError(
            "duckdb is required for daily_research.data_lake. "
            "Install it in the yolos environment or update daily_research/environment.yml."
        ) from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_date(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_bounds(frame: pd.DataFrame, column: str = "date") -> tuple[str, str]:
    if frame.empty or column not in frame.columns:
        return "", ""
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return "", ""
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _available_strict_end(available_trade_dates: Sequence[Any], max_forward_horizon: int) -> str:
    dates = pd.to_datetime(pd.Series(list(available_trade_dates)), errors="coerce").dropna().sort_values().unique()
    if len(dates) == 0:
        return ""
    horizon = max(int(max_forward_horizon or 0), 0)
    index = max(0, len(dates) - horizon - 1)
    return pd.Timestamp(dates[index]).strftime("%Y-%m-%d")


def build_label_completeness_summary(
    *,
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    zone: str,
    available_trade_dates: Sequence[Any],
    max_forward_horizon: int,
) -> dict[str, Any]:
    resolved_zone = str(zone or "strict_train")
    strict_end_date = _available_strict_end(available_trade_dates, max_forward_horizon)
    if strict_end_date:
        sample_dates = pd.to_datetime(sample_frame.get("date", pd.Series([], dtype=str)), errors="coerce")
        daily_dates = pd.to_datetime(daily_frame.get("date", pd.Series([], dtype=str)), errors="coerce")
        strict_end_ts = pd.Timestamp(strict_end_date)
        sample_observed = sample_dates.le(strict_end_ts).fillna(False)
        daily_observed = daily_dates.le(strict_end_ts).fillna(False)
    else:
        sample_observed = pd.Series([False] * len(sample_frame), index=sample_frame.index)
        daily_observed = pd.Series([False] * len(daily_frame), index=daily_frame.index)
    sample_rows = int(len(sample_frame))
    daily_rows = int(len(daily_frame))
    observed_rows = int(sample_observed.sum()) if sample_rows else 0
    daily_observed_rows = int(daily_observed.sum()) if daily_rows else 0
    return {
        "zone": resolved_zone,
        "max_forward_horizon": int(max_forward_horizon or 0),
        "strict_end_date": strict_end_date,
        "sample_rows": sample_rows,
        "daily_rows": daily_rows,
        "observed_label_rows": observed_rows,
        "unobserved_label_rows": int(sample_rows - observed_rows),
        "observed_label_row_ratio": float(observed_rows / sample_rows) if sample_rows else 0.0,
        "daily_observed_rows": daily_observed_rows,
        "daily_unobserved_rows": int(daily_rows - daily_observed_rows),
        "is_training_safe": bool(resolved_zone == "strict_train" and sample_rows > 0 and observed_rows == sample_rows),
    }


@dataclass(frozen=True)
class LakeDatasetRecord:
    dataset_id: str
    dataset_kind: str
    zone: str
    fingerprint: str
    status: str
    root: Path
    content_paths: dict[str, str]
    row_counts: dict[str, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LakeTrainingDatasetRecord(LakeDatasetRecord):
    sample_frame: pd.DataFrame
    daily_frame: pd.DataFrame
    teacher_summary: dict[str, Any]
    label_completeness_summary: dict[str, Any]


class ResearchDataLake:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_DATA_LAKE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.parquet_root = self.root / "parquet"
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.duckdb"
        self._duckdb = _require_duckdb()
        self._ensure_schema()

    def _connect(self):
        return self._duckdb.connect(str(self.catalog_path))

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                create table if not exists datasets (
                    dataset_id varchar primary key,
                    dataset_kind varchar,
                    domain varchar,
                    zone varchar,
                    source varchar,
                    universe_name varchar,
                    benchmark varchar,
                    start_date date,
                    end_date date,
                    strict_end_date date,
                    max_forward_horizon integer,
                    schema_version integer,
                    fingerprint varchar unique,
                    parameters_json varchar,
                    label_completeness_json varchar,
                    content_paths_json varchar,
                    row_counts_json varchar,
                    source_cache_json varchar,
                    created_at timestamp,
                    status varchar
                )
                """
            )

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        with self._connect() as con:
            return con.execute(sql, list(params or [])).fetchdf()

    def list_datasets(self, *, dataset_kind: str | None = None, zone: str | None = None) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_kind:
            clauses.append("dataset_kind = ?")
            params.append(str(dataset_kind))
        if zone:
            clauses.append("zone = ?")
            params.append(str(zone))
        where = f" where {' and '.join(clauses)}" if clauses else ""
        return self.query(f"select * from datasets{where} order by created_at, dataset_id", params)

    def describe_dataset(self, dataset_id: str) -> dict[str, Any]:
        rows = self.query("select * from datasets where dataset_id = ?", [str(dataset_id)])
        if rows.empty:
            raise KeyError(f"Unknown data lake dataset_id: {dataset_id}")
        row = rows.iloc[0].to_dict()
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key.endswith("_json"):
                out[key.removesuffix("_json")] = json.loads(value) if isinstance(value, str) and value else {}
            elif isinstance(value, pd.Timestamp):
                out[key] = value.strftime("%Y-%m-%d") if key.endswith("_date") else value.isoformat()
            elif pd.isna(value):
                out[key] = ""
            else:
                out[key] = value
        out["label_completeness_summary"] = out.pop("label_completeness", {})
        return out

    def build_catalog_manifest(self) -> dict[str, Any]:
        rows = self.list_datasets()
        datasets = [
            self.describe_dataset(str(row["dataset_id"]))
            for _, row in rows.iterrows()
        ]
        return {
            "status": "ok",
            "data_lake_root": str(self.root.resolve()),
            "catalog_path": str(self.catalog_path.resolve()),
            "dataset_count": int(len(datasets)),
            "datasets": datasets,
        }

    def write_catalog_manifest(self, path: str | Path | None = None) -> Path:
        manifest_path = Path(path) if path is not None else self.root / "manifest_latest.json"
        _write_json(manifest_path, self.build_catalog_manifest())
        return manifest_path

    def _existing_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        rows = self.query("select * from datasets where fingerprint = ?", [str(fingerprint)])
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def _upsert_dataset(
        self,
        *,
        dataset_id: str,
        dataset_kind: str,
        domain: str,
        zone: str,
        source: str,
        spec: Mapping[str, Any],
        label_completeness_summary: Mapping[str, Any],
        content_paths: Mapping[str, str],
        row_counts: Mapping[str, int],
        source_cache: Mapping[str, Any] | None,
        fingerprint: str,
        status: str,
    ) -> None:
        start_date = str(spec.get("start_date", "") or "")
        end_date = str(spec.get("end_date", "") or "")
        if not start_date or not end_date:
            bounds = row_counts.get("_date_bounds", None)
            if isinstance(bounds, Mapping):
                start_date = start_date or str(bounds.get("start_date", "") or "")
                end_date = end_date or str(bounds.get("end_date", "") or "")
        params_json = json.dumps(_json_safe(dict(spec)), ensure_ascii=False, sort_keys=True)
        label_json = json.dumps(_json_safe(dict(label_completeness_summary)), ensure_ascii=False, sort_keys=True)
        content_json = json.dumps(_json_safe(dict(content_paths)), ensure_ascii=False, sort_keys=True)
        row_counts_json = json.dumps(
            _json_safe({key: value for key, value in dict(row_counts).items() if not key.startswith("_")}),
            ensure_ascii=False,
            sort_keys=True,
        )
        source_cache_json = json.dumps(_json_safe(dict(source_cache or {})), ensure_ascii=False, sort_keys=True)
        strict_end_date = str(dict(label_completeness_summary).get("strict_end_date", "") or "")
        max_forward_horizon = int(dict(label_completeness_summary).get("max_forward_horizon", 0) or 0)
        with self._connect() as con:
            con.execute(
                "delete from datasets where dataset_id = ? or fingerprint = ?",
                [dataset_id, fingerprint],
            )
            con.execute(
                """
                insert into datasets values (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp, ?
                )
                """,
                [
                    dataset_id,
                    dataset_kind,
                    domain,
                    zone,
                    source,
                    str(spec.get("pool_name", spec.get("universe", "")) or ""),
                    str(spec.get("benchmark", "") or ""),
                    _normalize_date(start_date) or None,
                    _normalize_date(end_date) or None,
                    _normalize_date(strict_end_date) or None,
                    max_forward_horizon,
                    DATA_LAKE_SCHEMA_VERSION,
                    fingerprint,
                    params_json,
                    label_json,
                    content_json,
                    row_counts_json,
                    source_cache_json,
                    status,
                ],
            )

    def _dataset_dir(self, dataset_kind: str, zone: str, fingerprint: str) -> Path:
        return self.parquet_root / "gold" / dataset_kind / str(zone) / fingerprint

    def build_training_dataset_identity(self, *, spec: Mapping[str, Any], zone: str) -> dict[str, str]:
        resolved_zone = str(zone or "strict_train")
        dataset_kind = "continuous_policy_training_matrices"
        fingerprint = _stable_hash(
            {
                "schema_version": DATA_LAKE_SCHEMA_VERSION,
                "dataset_kind": dataset_kind,
                "zone": resolved_zone,
                "spec": dict(spec),
            }
        )
        return {
            "dataset_kind": dataset_kind,
            "zone": resolved_zone,
            "fingerprint": fingerprint,
            "dataset_id": f"{dataset_kind}__{resolved_zone}__{fingerprint}",
            "dataset_dir": str(self._dataset_dir(dataset_kind, resolved_zone, fingerprint).resolve()),
        }

    def save_training_dataset(
        self,
        *,
        spec: Mapping[str, Any],
        sample_frame: pd.DataFrame,
        daily_frame: pd.DataFrame,
        teacher_summary: Mapping[str, Any],
        zone: str,
        label_completeness_summary: Mapping[str, Any] | None = None,
        source_cache: Mapping[str, Any] | None = None,
        reuse: bool = True,
    ) -> LakeTrainingDatasetRecord:
        resolved_zone = str(zone or "strict_train")
        dataset_kind = "continuous_policy_training_matrices"
        fingerprint = _stable_hash(
            {
                "schema_version": DATA_LAKE_SCHEMA_VERSION,
                "dataset_kind": dataset_kind,
                "zone": resolved_zone,
                "spec": dict(spec),
            }
        )
        dataset_id = f"{dataset_kind}__{resolved_zone}__{fingerprint}"
        dataset_dir = self._dataset_dir(dataset_kind, resolved_zone, fingerprint)
        content_paths = {
            "sample_frame": str((dataset_dir / "sample_frame.parquet").resolve()),
            "daily_frame": str((dataset_dir / "daily_frame.parquet").resolve()),
            "teacher_summary": str((dataset_dir / "teacher_summary.json").resolve()),
        }
        existing = self._existing_by_fingerprint(fingerprint)
        if reuse and existing is not None and all(Path(path).exists() for path in content_paths.values()):
            return self.load_training_dataset(str(existing["dataset_id"]), status="hit")

        label_summary = dict(label_completeness_summary or {})
        if not label_summary:
            label_summary = {
                "zone": resolved_zone,
                "sample_rows": int(len(sample_frame)),
                "daily_rows": int(len(daily_frame)),
                "observed_label_row_ratio": 1.0 if resolved_zone == "strict_train" else 0.0,
            }
        if resolved_zone == "strict_train" and float(label_summary.get("observed_label_row_ratio", 0.0) or 0.0) < 1.0:
            raise ValueError("strict_train datasets cannot contain unobserved forward labels.")

        dataset_dir.mkdir(parents=True, exist_ok=True)
        sample_out = sample_frame.copy()
        daily_out = daily_frame.copy()
        sample_out.to_parquet(content_paths["sample_frame"], index=False)
        daily_out.to_parquet(content_paths["daily_frame"], index=False)
        _write_json(Path(content_paths["teacher_summary"]), dict(teacher_summary or {}))
        start_date, end_date = _date_bounds(daily_out if "date" in daily_out.columns else sample_out)
        row_counts: dict[str, Any] = {
            "sample_frame": int(len(sample_out)),
            "daily_frame": int(len(daily_out)),
            "_date_bounds": {"start_date": start_date, "end_date": end_date},
        }
        merged_spec = {**dict(spec), "start_date": str(spec.get("start_date", "") or start_date), "end_date": str(spec.get("end_date", "") or end_date)}
        self._upsert_dataset(
            dataset_id=dataset_id,
            dataset_kind=dataset_kind,
            domain="gold",
            zone=resolved_zone,
            source=str(spec.get("data_source", spec.get("source", "")) or ""),
            spec=merged_spec,
            label_completeness_summary=label_summary,
            content_paths=content_paths,
            row_counts=row_counts,
            source_cache=source_cache,
            fingerprint=fingerprint,
            status="stored",
        )
        return self.load_training_dataset(dataset_id, status="stored")

    def save_sharded_training_dataset(
        self,
        *,
        spec: Mapping[str, Any],
        zone: str,
        shard_records: Sequence[Mapping[str, Any]],
        teacher_summary: Mapping[str, Any],
        label_completeness_summary: Mapping[str, Any],
        source_cache: Mapping[str, Any] | None = None,
        status: str = "stored",
    ) -> LakeTrainingDatasetRecord:
        resolved_zone = str(zone or "strict_train")
        identity = self.build_training_dataset_identity(spec=spec, zone=resolved_zone)
        dataset_kind = identity["dataset_kind"]
        fingerprint = identity["fingerprint"]
        dataset_id = identity["dataset_id"]
        dataset_dir = Path(identity["dataset_dir"])
        shard_manifest_path = dataset_dir / "shard_manifest.json"
        teacher_summary_path = dataset_dir / "teacher_summary.json"
        audit_report_path = dataset_dir / "audit_report.json"
        content_paths = {
            "dataset_root": str(dataset_dir.resolve()),
            "sample_frame_shards": str((dataset_dir / "sample_frame" / "*.parquet").resolve()),
            "daily_frame_shards": str((dataset_dir / "daily_frame" / "*.parquet").resolve()),
            "teacher_summary": str(teacher_summary_path.resolve()),
            "shard_manifest": str(shard_manifest_path.resolve()),
            "portfolio_checkpoints": str((dataset_dir / "portfolio_checkpoint" / "*.json").resolve()),
            "audit_report": str(audit_report_path.resolve()),
        }
        label_summary = dict(label_completeness_summary or {})
        if resolved_zone == "strict_train" and int(label_summary.get("unobserved_label_rows", 0) or 0) != 0:
            raise ValueError("strict_train sharded datasets cannot contain unobserved forward labels.")
        if resolved_zone == "strict_train" and not bool(label_summary.get("is_training_safe", False)):
            raise ValueError("strict_train sharded datasets must be marked training safe.")

        dataset_dir.mkdir(parents=True, exist_ok=True)
        safe_shards = [_json_safe(dict(item)) for item in shard_records]
        _write_json(
            shard_manifest_path,
            {
                "dataset_id": dataset_id,
                "fingerprint": fingerprint,
                "zone": resolved_zone,
                "sharded": True,
                "shard_count": len(safe_shards),
                "completed_shard_count": int(sum(1 for item in safe_shards if str(item.get("status", "")) == "stored")),
                "shards": safe_shards,
            },
        )
        _write_json(teacher_summary_path, dict(teacher_summary or {}))
        if not audit_report_path.exists():
            _write_json(audit_report_path, {"status": "not_run", "dataset_id": dataset_id})

        sample_rows = int(sum(int(item.get("sample_rows", 0) or 0) for item in safe_shards))
        daily_rows = int(sum(int(item.get("daily_rows", 0) or 0) for item in safe_shards))
        completed = int(sum(1 for item in safe_shards if str(item.get("status", "")) == "stored"))
        row_counts: dict[str, Any] = {
            "sample_frame": sample_rows,
            "daily_frame": daily_rows,
            "shard_count": int(len(safe_shards)),
            "completed_shard_count": completed,
        }
        start_dates = [str(item.get("start_date", "") or "") for item in safe_shards if str(item.get("start_date", "") or "")]
        end_dates = [str(item.get("end_date", "") or "") for item in safe_shards if str(item.get("end_date", "") or "")]
        if start_dates and end_dates:
            row_counts["_date_bounds"] = {"start_date": min(start_dates), "end_date": max(end_dates)}
        merged_spec = {
            **dict(spec),
            "sharded": True,
            "start_date": str(spec.get("start_date", "") or (min(start_dates) if start_dates else "")),
            "end_date": str(spec.get("end_date", "") or (max(end_dates) if end_dates else "")),
        }
        merged_source_cache = {
            **dict(source_cache or {}),
            "sharded": True,
            "shards": safe_shards,
        }
        self._upsert_dataset(
            dataset_id=dataset_id,
            dataset_kind=dataset_kind,
            domain="gold",
            zone=resolved_zone,
            source=str(spec.get("data_source", spec.get("source", "")) or ""),
            spec=merged_spec,
            label_completeness_summary=label_summary,
            content_paths=content_paths,
            row_counts=row_counts,
            source_cache=merged_source_cache,
            fingerprint=fingerprint,
            status=str(status or "stored"),
        )
        metadata = self.describe_dataset(dataset_id)
        return LakeTrainingDatasetRecord(
            dataset_id=dataset_id,
            dataset_kind=dataset_kind,
            zone=resolved_zone,
            fingerprint=fingerprint,
            status=str(status or "stored"),
            root=self.root,
            content_paths=dict(metadata.get("content_paths", {}) or {}),
            row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
            metadata=metadata,
            sample_frame=pd.DataFrame(),
            daily_frame=pd.DataFrame(),
            teacher_summary=dict(teacher_summary or {}),
            label_completeness_summary=label_summary,
        )

    def load_training_dataset(self, dataset_id: str, *, status: str = "loaded") -> LakeTrainingDatasetRecord:
        metadata = self.describe_dataset(dataset_id)
        paths = dict(metadata.get("content_paths", {}) or {})
        source_cache = dict(metadata.get("source_cache", {}) or {})
        is_sharded = bool(source_cache.get("sharded", False) or paths.get("shard_manifest"))
        if is_sharded:
            shard_rows = list(source_cache.get("shards", []) or [])
            if not shard_rows and paths.get("shard_manifest"):
                shard_rows = list(_read_json(Path(paths["shard_manifest"])).get("shards", []) or [])
            sample_paths = [Path(str(item.get("sample_path", ""))) for item in shard_rows if str(item.get("sample_path", "") or "")]
            daily_paths = [Path(str(item.get("daily_path", ""))) for item in shard_rows if str(item.get("daily_path", "") or "")]
            sample_frames = [pd.read_parquet(path) for path in sample_paths if path.exists()]
            daily_frames = [pd.read_parquet(path) for path in daily_paths if path.exists()]
            sample_frame = (
                pd.concat(sample_frames, ignore_index=True)
                if sample_frames
                else pd.DataFrame()
            )
            daily_frame = (
                pd.concat(daily_frames, ignore_index=True)
                if daily_frames
                else pd.DataFrame()
            )
            if not sample_frame.empty and {"date", "stock"}.issubset(sample_frame.columns):
                sample_frame = sample_frame.sort_values(["date", "stock"]).reset_index(drop=True)
            if not daily_frame.empty and "date" in daily_frame.columns:
                daily_frame = daily_frame.sort_values(["date"]).reset_index(drop=True)
        else:
            sample_frame = pd.read_parquet(paths["sample_frame"])
            daily_frame = pd.read_parquet(paths["daily_frame"])
        teacher_summary = _read_json(Path(paths["teacher_summary"]))
        return LakeTrainingDatasetRecord(
            dataset_id=str(metadata["dataset_id"]),
            dataset_kind=str(metadata["dataset_kind"]),
            zone=str(metadata["zone"]),
            fingerprint=str(metadata["fingerprint"]),
            status=status,
            root=self.root,
            content_paths={str(key): str(value) for key, value in paths.items()},
            row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
            metadata=metadata,
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            teacher_summary=teacher_summary,
            label_completeness_summary=dict(metadata.get("label_completeness_summary", {}) or {}),
        )

    def find_training_dataset(self, *, spec: Mapping[str, Any], zone: str) -> LakeTrainingDatasetRecord | None:
        identity = self.build_training_dataset_identity(spec=spec, zone=zone)
        existing = self._existing_by_fingerprint(identity["fingerprint"])
        if existing is None:
            return None
        try:
            return self.load_training_dataset(str(existing["dataset_id"]), status="hit")
        except Exception:
            return None

    def update_dataset_audit_report(self, dataset_id: str, audit_report: Mapping[str, Any]) -> None:
        metadata = self.describe_dataset(dataset_id)
        paths = dict(metadata.get("content_paths", {}) or {})
        audit_path_text = str(paths.get("audit_report", "") or "").strip()
        if audit_path_text:
            audit_path = Path(audit_path_text)
            _write_json(audit_path, dict(audit_report))
        source_cache = dict(metadata.get("source_cache", {}) or {})
        source_cache["audit_report"] = _json_safe(dict(audit_report))
        self._upsert_dataset(
            dataset_id=str(metadata["dataset_id"]),
            dataset_kind=str(metadata["dataset_kind"]),
            domain=str(metadata.get("domain", "") or "gold"),
            zone=str(metadata["zone"]),
            source=str(metadata.get("source", "") or ""),
            spec=dict(metadata.get("parameters", {}) or {}),
            label_completeness_summary=dict(metadata.get("label_completeness_summary", {}) or {}),
            content_paths=paths,
            row_counts=dict(metadata.get("row_counts", {}) or {}),
            source_cache=source_cache,
            fingerprint=str(metadata["fingerprint"]),
            status=str(metadata.get("status", "") or "stored"),
        )

    def save_market_data_bundle(
        self,
        *,
        spec: Mapping[str, Any],
        market_frames: Mapping[str, pd.DataFrame],
        benchmark_close: pd.Series,
        benchmark_open: pd.Series | None = None,
        membership_frame: pd.DataFrame,
        feature_frames: Mapping[str, pd.DataFrame],
        source: str,
        reuse: bool = True,
    ) -> LakeDatasetRecord:
        dataset_kind = "policy_input_bundle"
        zone = "research"
        effective_spec = dict(spec)
        if benchmark_open is not None and "benchmark_fields" not in effective_spec:
            effective_spec["benchmark_fields"] = ["open", "close"]
        fingerprint = _stable_hash(
            {
                "schema_version": DATA_LAKE_SCHEMA_VERSION,
                "dataset_kind": dataset_kind,
                "zone": zone,
                "spec": effective_spec,
            }
        )
        dataset_id = f"{dataset_kind}__{fingerprint}"
        dataset_dir = self.parquet_root / "bronze_silver" / dataset_kind / fingerprint
        content_paths = {
            "bronze_market_data": str((dataset_dir / "bronze_market_data.parquet").resolve()),
            "silver_benchmark": str((dataset_dir / "silver_benchmark.parquet").resolve()),
            "silver_membership": str((dataset_dir / "silver_membership.parquet").resolve()),
            "silver_feature_values": str((dataset_dir / "silver_feature_panels" / "*.parquet").resolve()),
            "silver_feature_panels": str((dataset_dir / "silver_feature_panels" / "*.parquet").resolve()),
        }
        existing = self._existing_by_fingerprint(fingerprint)
        if reuse and existing is not None and Path(content_paths["bronze_market_data"]).exists():
            metadata = self.describe_dataset(str(existing["dataset_id"]))
            return LakeDatasetRecord(
                dataset_id=str(metadata["dataset_id"]),
                dataset_kind=str(metadata["dataset_kind"]),
                zone=str(metadata["zone"]),
                fingerprint=str(metadata["fingerprint"]),
                status="hit",
                root=self.root,
                content_paths=dict(metadata.get("content_paths", {}) or {}),
                row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
                metadata=metadata,
            )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        market_long = self._market_frames_to_long(market_frames, source=source)
        market_long.to_parquet(content_paths["bronze_market_data"], index=False)
        close_series = pd.Series(pd.to_numeric(benchmark_close, errors="coerce"), index=pd.to_datetime(benchmark_close.index), name=benchmark_close.name)
        benchmark_dates = pd.Index(pd.to_datetime(close_series.index)).dropna().unique()
        open_series: pd.Series | None = None
        if benchmark_open is not None:
            open_series = pd.Series(pd.to_numeric(benchmark_open, errors="coerce"), index=pd.to_datetime(benchmark_open.index), name=benchmark_open.name)
            benchmark_dates = benchmark_dates.union(pd.Index(pd.to_datetime(open_series.index)).dropna().unique())
        benchmark_dates = pd.Index(sorted(benchmark_dates))
        benchmark = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(benchmark_dates).strftime("%Y-%m-%d"),
                "benchmark": str(effective_spec.get("benchmark", benchmark_close.name or "") or ""),
                "close": close_series.reindex(benchmark_dates).to_numpy(dtype=float),
            }
        )
        if open_series is not None:
            benchmark["open"] = open_series.reindex(benchmark_dates).to_numpy(dtype=float)
        benchmark.to_parquet(content_paths["silver_benchmark"], index=False)
        membership_out = membership_frame.copy()
        if "date" in membership_out.columns and "trade_date" not in membership_out.columns:
            membership_out = membership_out.rename(columns={"date": "trade_date"})
        membership_out.to_parquet(content_paths["silver_membership"], index=False)
        feature_dir = dataset_dir / "silver_feature_panels"
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_cells = 0
        feature_panels = 0
        for feature_name, frame in feature_frames.items():
            feature_panel = self._feature_frame_to_panel(frame)
            feature_cells += int(frame.size)
            feature_panels += 1
            safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(feature_name))
            feature_panel.to_parquet(feature_dir / f"{safe_name}.parquet", index=False)
        start_date = str(market_long["trade_date"].min()) if not market_long.empty else ""
        end_date = str(market_long["trade_date"].max()) if not market_long.empty else ""
        row_counts: dict[str, Any] = {
            "bronze_market_data": int(len(market_long)),
            "silver_benchmark": int(len(benchmark)),
            "silver_membership": int(len(membership_out)),
            "silver_feature_panels": int(feature_panels),
            "silver_feature_cells": int(feature_cells),
            "_date_bounds": {"start_date": start_date, "end_date": end_date},
        }
        merged_spec = {
            **effective_spec,
            "start_date": str(effective_spec.get("start_date", "") or start_date),
            "end_date": str(effective_spec.get("end_date", "") or end_date),
        }
        self._upsert_dataset(
            dataset_id=dataset_id,
            dataset_kind=dataset_kind,
            domain="bronze_silver",
            zone=zone,
            source=str(source or ""),
            spec=merged_spec,
            label_completeness_summary={},
            content_paths=content_paths,
            row_counts=row_counts,
            source_cache={},
            fingerprint=fingerprint,
            status="stored",
        )
        metadata = self.describe_dataset(dataset_id)
        return LakeDatasetRecord(
            dataset_id=dataset_id,
            dataset_kind=dataset_kind,
            zone=zone,
            fingerprint=fingerprint,
            status="stored",
            root=self.root,
            content_paths=dict(metadata.get("content_paths", {}) or {}),
            row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
            metadata=metadata,
        )

    @staticmethod
    def _field_to_column(field: str) -> str:
        normalized = str(field or "").strip().lower()
        aliases = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "amount": "amount"}
        return aliases.get(normalized, normalized)

    @classmethod
    def _wide_to_long(cls, frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        working = frame.copy()
        working.index = pd.to_datetime(working.index)
        working.index.name = "trade_date"
        try:
            stacked = working.stack(future_stack=True)
        except TypeError:  # pandas < 2.1
            stacked = working.stack(dropna=False)
        long = stacked.rename(value_name).reset_index()
        long.columns = ["trade_date", "symbol", value_name]
        long["trade_date"] = pd.to_datetime(long["trade_date"]).dt.strftime("%Y-%m-%d")
        long["symbol"] = long["symbol"].astype(str)
        long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
        return long

    @classmethod
    def _market_frames_to_long(cls, market_frames: Mapping[str, pd.DataFrame], *, source: str) -> pd.DataFrame:
        merged: pd.DataFrame | None = None
        for field, frame in market_frames.items():
            column = cls._field_to_column(field)
            long = cls._wide_to_long(frame, column)
            if merged is None:
                merged = long
            else:
                merged = merged.merge(long, on=["trade_date", "symbol"], how="outer")
        if merged is None:
            merged = pd.DataFrame(columns=["trade_date", "symbol"])
        merged["source"] = str(source or "")
        merged["ingest_batch_id"] = _stable_hash(
            {
                "source": str(source or ""),
                "trade_date_min": str(merged["trade_date"].min()) if not merged.empty else "",
                "trade_date_max": str(merged["trade_date"].max()) if not merged.empty else "",
                "symbol_count": int(merged["symbol"].nunique()) if "symbol" in merged.columns else 0,
            }
        )
        ordered = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "ingest_batch_id"]
        for column in ordered:
            if column not in merged.columns:
                merged[column] = np.nan if column not in {"source", "ingest_batch_id"} else ""
        return merged[ordered].sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    @classmethod
    @staticmethod
    def _feature_frame_to_panel(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working.index = pd.to_datetime(working.index)
        working.insert(0, "trade_date", working.index.strftime("%Y-%m-%d"))
        return working.reset_index(drop=True)
