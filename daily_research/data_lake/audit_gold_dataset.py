from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_lake.catalog import ResearchDataLake


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _problem(code: str, detail: str, *, severity: str = "error") -> dict[str, str]:
    return {"severity": severity, "code": code, "detail": detail}


def _safe_count_duplicate_keys(frame: pd.DataFrame, columns: list[str]) -> int:
    if frame.empty or any(column not in frame.columns for column in columns):
        return 0
    return int(frame.duplicated(columns).sum())


def _observed_counts(frame: pd.DataFrame, *, strict_end_date: str) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    if "is_observed" in frame.columns:
        observed = frame["is_observed"].astype(bool)
    elif strict_end_date and "date" in frame.columns:
        observed = pd.to_datetime(frame["date"], errors="coerce").le(pd.Timestamp(strict_end_date)).fillna(False)
    else:
        observed = pd.Series([False] * len(frame), index=frame.index)
    observed_count = int(observed.sum())
    return observed_count, int(len(frame) - observed_count)


def audit_gold_dataset(*, lake: ResearchDataLake, dataset_id: str, writeback: bool = True) -> dict[str, Any]:
    metadata = lake.describe_dataset(dataset_id)
    if str(metadata.get("dataset_kind", "") or "") != "continuous_policy_training_matrices":
        raise ValueError(f"Dataset is not a continuous-policy Gold training dataset: {dataset_id}")
    paths = dict(metadata.get("content_paths", {}) or {})
    source_cache = dict(metadata.get("source_cache", {}) or {})
    label_summary = dict(metadata.get("label_completeness_summary", {}) or {})
    row_counts = dict(metadata.get("row_counts", {}) or {})
    zone = str(metadata.get("zone", "") or "")
    strict_end_date = str(label_summary.get("strict_end_date", "") or "")
    findings: list[dict[str, str]] = []

    shard_manifest_path = Path(str(paths.get("shard_manifest", "") or ""))
    shard_rows = list(source_cache.get("shards", []) or [])
    manifest = _read_json(shard_manifest_path) if shard_manifest_path else {}
    if not shard_rows:
        shard_rows = list(manifest.get("shards", []) or [])
    is_sharded = bool(source_cache.get("sharded", False) or manifest.get("sharded", False))
    if is_sharded and not shard_rows:
        findings.append(_problem("missing_shard_manifest", "Sharded Gold dataset has no shard records."))

    sample_total = 0
    daily_total = 0
    sample_observed_total = 0
    sample_unobserved_total = 0
    daily_observed_total = 0
    daily_unobserved_total = 0
    duplicate_sample_keys = 0
    duplicate_daily_dates = 0
    sample_key_frames: list[pd.DataFrame] = []
    daily_key_frames: list[pd.DataFrame] = []

    if is_sharded:
        for shard in shard_rows:
            sample_path = Path(str((shard or {}).get("sample_path", "") or ""))
            daily_path = Path(str((shard or {}).get("daily_path", "") or ""))
            checkpoint_path_text = str((shard or {}).get("checkpoint_path", "") or "").strip()
            checkpoint_path = Path(checkpoint_path_text) if checkpoint_path_text else None
            if not sample_path.exists():
                findings.append(_problem("missing_sample_shard", f"Missing sample shard: {sample_path}"))
                continue
            if not daily_path.exists():
                findings.append(_problem("missing_daily_shard", f"Missing daily shard: {daily_path}"))
                continue
            if checkpoint_path is not None and not checkpoint_path.exists():
                findings.append(_problem("missing_checkpoint", f"Missing portfolio checkpoint: {checkpoint_path}"))
            elif checkpoint_path is not None:
                checkpoint = _read_json(checkpoint_path)
                if str(checkpoint.get("fingerprint", "") or "") != str(metadata.get("fingerprint", "") or ""):
                    findings.append(
                        _problem(
                            "checkpoint_fingerprint_mismatch",
                            f"Checkpoint fingerprint mismatch: {checkpoint_path}",
                        )
                    )
            sample = pd.read_parquet(sample_path)
            daily = pd.read_parquet(daily_path)
            sample_total += int(len(sample))
            daily_total += int(len(daily))
            obs, unobs = _observed_counts(sample, strict_end_date=strict_end_date)
            daily_obs, daily_unobs = _observed_counts(daily, strict_end_date=strict_end_date)
            sample_observed_total += obs
            sample_unobserved_total += unobs
            daily_observed_total += daily_obs
            daily_unobserved_total += daily_unobs
            if {"date", "stock"}.issubset(sample.columns):
                sample_key_frames.append(sample[["date", "stock"]].copy())
            if "date" in daily.columns:
                daily_key_frames.append(daily[["date"]].copy())
    else:
        sample_path = Path(str(paths.get("sample_frame", "") or ""))
        daily_path = Path(str(paths.get("daily_frame", "") or ""))
        if not sample_path.exists():
            findings.append(_problem("missing_sample_frame", f"Missing sample frame: {sample_path}"))
        if not daily_path.exists():
            findings.append(_problem("missing_daily_frame", f"Missing daily frame: {daily_path}"))
        if sample_path.exists() and daily_path.exists():
            sample = pd.read_parquet(sample_path)
            daily = pd.read_parquet(daily_path)
            sample_total = int(len(sample))
            daily_total = int(len(daily))
            sample_observed_total, sample_unobserved_total = _observed_counts(sample, strict_end_date=strict_end_date)
            daily_observed_total, daily_unobserved_total = _observed_counts(daily, strict_end_date=strict_end_date)
            if {"date", "stock"}.issubset(sample.columns):
                sample_key_frames.append(sample[["date", "stock"]].copy())
            if "date" in daily.columns:
                daily_key_frames.append(daily[["date"]].copy())

    if sample_key_frames:
        all_sample_keys = pd.concat(sample_key_frames, ignore_index=True)
        duplicate_sample_keys = int(all_sample_keys.duplicated(["date", "stock"]).sum())
    if daily_key_frames:
        all_daily_keys = pd.concat(daily_key_frames, ignore_index=True)
        duplicate_daily_dates = int(all_daily_keys.duplicated(["date"]).sum())

    if duplicate_sample_keys > 0:
        findings.append(_problem("duplicate_sample_keys", f"Duplicate (date, stock) rows: {duplicate_sample_keys}"))
    if duplicate_daily_dates > 0:
        findings.append(_problem("duplicate_daily_dates", f"Duplicate daily date rows: {duplicate_daily_dates}"))
    if sample_total != int(row_counts.get("sample_frame", sample_total) or 0):
        findings.append(
            _problem(
                "sample_row_count_mismatch",
                f"Catalog sample rows {row_counts.get('sample_frame')} != shard rows {sample_total}",
            )
        )
    if daily_total != int(row_counts.get("daily_frame", daily_total) or 0):
        findings.append(
            _problem(
                "daily_row_count_mismatch",
                f"Catalog daily rows {row_counts.get('daily_frame')} != shard rows {daily_total}",
            )
        )
    if zone == "strict_train" and sample_unobserved_total != 0:
        findings.append(_problem("strict_unobserved_labels", f"Strict Gold has {sample_unobserved_total} unobserved sample labels."))
    if zone == "strict_train" and not bool(label_summary.get("is_training_safe", False)):
        findings.append(_problem("strict_not_training_safe", "Strict Gold catalog label summary is not training safe."))
    if zone == "realtime_research" and sample_total > 0 and sample_unobserved_total == 0:
        findings.append(
            _problem(
                "realtime_tail_not_marked",
                "Realtime Gold has no unobserved rows; tail label marking may be missing.",
                severity="warning",
            )
        )
    if int(label_summary.get("sample_rows", sample_total) or 0) != sample_total:
        findings.append(
            _problem(
                "label_summary_sample_mismatch",
                f"Label summary sample rows {label_summary.get('sample_rows')} != actual {sample_total}",
            )
        )
    if int(label_summary.get("unobserved_label_rows", sample_unobserved_total) or 0) != sample_unobserved_total:
        findings.append(
            _problem(
                "label_summary_unobserved_mismatch",
                f"Label summary unobserved rows {label_summary.get('unobserved_label_rows')} != actual {sample_unobserved_total}",
            )
        )

    error_count = sum(1 for item in findings if item["severity"] == "error")
    warning_count = sum(1 for item in findings if item["severity"] == "warning")
    report = {
        "status": "ok" if error_count == 0 else "failed",
        "dataset_id": dataset_id,
        "zone": zone,
        "sharded": is_sharded,
        "sample_rows": sample_total,
        "daily_rows": daily_total,
        "observed_label_rows": sample_observed_total,
        "unobserved_label_rows": sample_unobserved_total,
        "daily_observed_rows": daily_observed_total,
        "daily_unobserved_rows": daily_unobserved_total,
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": findings,
    }
    if writeback:
        lake.update_dataset_audit_report(dataset_id, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a Gold continuous-policy training dataset in the research data lake.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--no-writeback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    report = audit_gold_dataset(lake=lake, dataset_id=args.dataset_id, writeback=not bool(args.no_writeback))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
