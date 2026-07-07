from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_STORE_ROOT = Path("daily_research/data/research_store")
DEFAULT_SEQUENCE_PACK_ROOTS = (
    DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack",
)
DEFAULT_STUDIES_ROOT = Path("daily_research/output/path_policy/studies")
DEFAULT_REFERENCE_ROOTS = (
    Path("daily_research/brain"),
    Path("daily_research/brain/references"),
    Path("brain"),
    Path("brain/references"),
)
DEFAULT_REPORT_ROOT = Path("daily_research/output/path_policy/research_gc")
DELETE_CONFIRMATION = "DELETE_RESEARCH_ARTIFACTS"
TRIM_CONFIRMATION = "TRIM_PREDICTION_OUTPUTS"
SUMMARY_FILE_NAMES = (
    "study_summary.json",
    "sequence_path_training_summary.json",
    "sequence_flat_lgbm_summary.json",
)


@dataclass(frozen=True)
class DirectorySize:
    size_bytes: int
    file_count: int
    latest_mtime: float


@dataclass(frozen=True)
class ArtifactItem:
    artifact_type: str
    name: str
    path: str
    size_bytes: int
    file_count: int
    size_gb: float
    last_modified: str
    classification: str
    recommendation: str
    cleanup_action: str
    safe_to_delete_directory: bool
    referenced_by_brain: bool
    reasons: list[str]
    manifest_path: str = ""
    summary_path: str = ""
    large_files: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["large_files"] = list(self.large_files or [])
        return payload


def _workspace_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _relative(path: str | Path) -> str:
    resolved = _workspace_path(path)
    try:
        return resolved.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except Exception:
        return resolved.as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _relative(value)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return target


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_mtime(value: float) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def _directory_size(path: str | Path) -> DirectorySize:
    root = _workspace_path(path)
    total = 0
    files = 0
    latest = 0.0
    if not root.exists():
        return DirectorySize(0, 0, 0.0)
    if root.is_file():
        stat = root.stat()
        return DirectorySize(int(stat.st_size), 1, float(stat.st_mtime))
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                stat = file_path.stat()
            except OSError:
                continue
            total += int(stat.st_size)
            files += 1
            latest = max(latest, float(stat.st_mtime))
    return DirectorySize(total, files, latest)


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(_workspace_path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_reference_text(reference_roots: Iterable[str | Path]) -> str:
    chunks: list[str] = []
    for root in reference_roots:
        base = _workspace_path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".txt"}:
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
    return "\n".join(chunks).lower()


def _is_referenced(name: str, reference_text: str) -> bool:
    clean = str(name or "").strip().lower()
    return bool(clean and clean in reference_text)


def _path_is_inside(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(_workspace_path(root).resolve())
            return True
        except ValueError:
            continue
    return False


def _large_files(path: str | Path, *, threshold_bytes: int, max_files: int) -> list[dict[str, Any]]:
    root = _workspace_path(path)
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                stat = file_path.stat()
            except OSError:
                continue
            if stat.st_size < threshold_bytes:
                continue
            out.append(
                {
                    "path": _relative(file_path),
                    "size_bytes": int(stat.st_size),
                    "size_gb": round(float(stat.st_size) / 1024**3, 4),
                    "mtime": _format_mtime(float(stat.st_mtime)),
                    "suffix": file_path.suffix.lower(),
                    "kind": _large_file_kind(file_path),
                }
            )
    return sorted(out, key=lambda item: int(item["size_bytes"]), reverse=True)[:max_files]


def _large_file_kind(path: Path) -> str:
    name = path.name.lower()
    if "prediction" in name and path.suffix.lower() in {".csv", ".parquet", ".feather"}:
        return "prediction_output"
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if path.suffix.lower() in {".dat", ".npy", ".npz"}:
        return "array_or_memmap"
    if path.suffix.lower() in {".csv", ".parquet"}:
        return "tabular_output"
    return "large_file"


def _study_summary_paths(path: str | Path) -> list[Path]:
    artifact_dir = _workspace_path(path)
    return [artifact_dir / name for name in SUMMARY_FILE_NAMES if (artifact_dir / name).exists()]


def _primary_study_summary_path(path: str | Path) -> Path:
    summaries = _study_summary_paths(path)
    return summaries[0] if summaries else _workspace_path(path) / "study_summary.json"


def _large_prediction_files(path: str | Path, *, threshold_bytes: int) -> list[dict[str, Any]]:
    root = _workspace_path(path)
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            file_path = Path(dirpath) / name
            if _large_file_kind(file_path) != "prediction_output":
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            if stat.st_size < threshold_bytes:
                continue
            out.append(
                {
                    "path": _relative(file_path),
                    "size_bytes": int(stat.st_size),
                    "size_gb": round(float(stat.st_size) / 1024**3, 4),
                    "mtime": _format_mtime(float(stat.st_mtime)),
                    "suffix": file_path.suffix.lower(),
                    "kind": "prediction_output",
                }
            )
    return sorted(out, key=lambda item: int(item["size_bytes"]), reverse=True)


def _manifest_references_outside_dir(manifest: Mapping[str, Any], artifact_dir: Path) -> bool:
    root = _workspace_path(artifact_dir).resolve()
    candidates: list[str] = []
    for section in ("feature_channels", "label_arrays"):
        value = manifest.get(section)
        if isinstance(value, dict):
            for item in value.values():
                if not isinstance(item, dict):
                    continue
                if item.get("path"):
                    candidates.append(str(item["path"]))
                for shard in item.get("shards", []) or []:
                    if isinstance(shard, dict) and shard.get("path"):
                        candidates.append(str(shard["path"]))
    if manifest.get("sample_index_path"):
        candidates.append(str(manifest["sample_index_path"]))
    for raw in candidates:
        try:
            path = Path(raw)
            resolved = (path if path.is_absolute() else WORKSPACE_ROOT / path).resolve()
            resolved.relative_to(root)
        except ValueError:
            return True
        except Exception:
            continue
    return False


def classify_sequence_pack(path: str | Path, *, reference_text: str, large_file_threshold_bytes: int) -> ArtifactItem:
    artifact_dir = _workspace_path(path)
    name = artifact_dir.name
    manifest_path = artifact_dir / "manifest.json"
    progress_path = artifact_dir / "build_progress.json"
    if not progress_path.exists():
        progress_path = artifact_dir / "progress.json"
    progress = _read_json(progress_path)
    manifest = _read_json(manifest_path)
    size = _directory_size(artifact_dir)
    referenced = _is_referenced(name, reference_text)
    reasons: list[str] = []
    classification = "sequence_pack_review_required"
    recommendation = "review_before_cleanup"
    cleanup_action = "manual_review"
    safe_delete = False

    if not manifest_path.exists():
        classification = "partial_or_invalid_sequence_pack"
        recommendation = "delete_if_not_referenced"
        cleanup_action = "delete_directory"
        reasons.append("missing_manifest_json")
        safe_delete = not referenced
    elif "smoke" in name.lower():
        classification = "smoke_sequence_pack"
        recommendation = "delete_if_not_referenced"
        cleanup_action = "delete_directory"
        reasons.append("smoke_pack")
        safe_delete = not referenced
    elif str(progress.get("status", "") or "").strip().lower() not in {"", "completed"}:
        classification = "partial_or_interrupted_sequence_pack"
        recommendation = "delete_if_not_referenced"
        cleanup_action = "delete_directory"
        reasons.append(f"progress_status={progress.get('status')}")
        safe_delete = not referenced
    elif _manifest_references_outside_dir(manifest, artifact_dir):
        classification = "view_or_reanchor_sequence_pack"
        recommendation = "keep_or_migrate_view_manifest"
        cleanup_action = "keep"
        reasons.append("manifest_references_arrays_outside_own_directory")
    elif "full" in name.lower():
        classification = "full_sequence_pack"
        recommendation = "keep_until_shared_store_view_replaces_it"
        cleanup_action = "keep"
        reasons.append("full_pack")
    else:
        classification = "sequence_pack_review_required"
        recommendation = "review_manifest_and_references"
        cleanup_action = "manual_review"
        reasons.append("not_classified_as_smoke_partial_or_full")

    if referenced:
        reasons.append("referenced_by_brain_docs")
        safe_delete = False
    return ArtifactItem(
        artifact_type="sequence_pack",
        name=name,
        path=_relative(artifact_dir),
        size_bytes=size.size_bytes,
        file_count=size.file_count,
        size_gb=round(float(size.size_bytes) / 1024**3, 4),
        last_modified=_format_mtime(size.latest_mtime),
        classification=classification,
        recommendation=recommendation,
        cleanup_action=cleanup_action,
        safe_to_delete_directory=safe_delete,
        referenced_by_brain=referenced,
        reasons=reasons,
        manifest_path=_relative(manifest_path) if manifest_path.exists() else "",
        large_files=_large_files(artifact_dir, threshold_bytes=large_file_threshold_bytes, max_files=8),
    )


def classify_study(path: str | Path, *, reference_text: str, large_file_threshold_bytes: int) -> ArtifactItem:
    artifact_dir = _workspace_path(path)
    name = artifact_dir.name
    summary_path = _primary_study_summary_path(artifact_dir)
    size = _directory_size(artifact_dir)
    referenced = _is_referenced(name, reference_text)
    large_files = _large_files(artifact_dir, threshold_bytes=large_file_threshold_bytes, max_files=12)
    large_prediction_files = [item for item in large_files if item.get("kind") == "prediction_output"]
    reasons: list[str] = []
    classification = "study_review_required"
    recommendation = "review_before_cleanup"
    cleanup_action = "manual_review"
    safe_delete = False
    lower_name = name.lower()

    if not summary_path.exists():
        classification = "partial_or_intermediate_study"
        recommendation = "delete_if_not_referenced"
        cleanup_action = "delete_directory"
        reasons.append("missing_study_summary_json")
        safe_delete = not referenced
    elif any(token in lower_name for token in ("smoke", "speed", "throughput", "debug")):
        classification = "smoke_or_throughput_study"
        recommendation = "delete_if_not_referenced_or_archive_summary"
        cleanup_action = "delete_directory"
        reasons.append("smoke_speed_or_throughput_run")
        safe_delete = not referenced
    elif large_prediction_files:
        classification = "study_with_large_prediction_outputs"
        recommendation = "trim_prediction_outputs_after_summary_and_metrics_are_preserved"
        cleanup_action = "trim_large_prediction_files"
        reasons.append("large_prediction_outputs")
    else:
        classification = "study_with_summary"
        recommendation = "keep"
        cleanup_action = "keep"
        reasons.append("study_summary_exists")

    if referenced:
        reasons.append("referenced_by_brain_docs")
        safe_delete = False
    return ArtifactItem(
        artifact_type="study",
        name=name,
        path=_relative(artifact_dir),
        size_bytes=size.size_bytes,
        file_count=size.file_count,
        size_gb=round(float(size.size_bytes) / 1024**3, 4),
        last_modified=_format_mtime(size.latest_mtime),
        classification=classification,
        recommendation=recommendation,
        cleanup_action=cleanup_action,
        safe_to_delete_directory=safe_delete,
        referenced_by_brain=referenced,
        reasons=reasons,
        summary_path=_relative(summary_path) if summary_path.exists() else "",
        large_files=large_files,
    )


def _scan_child_dirs(roots: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        base = _workspace_path(root)
        if not base.exists():
            continue
        out.extend(sorted(path for path in base.iterdir() if path.is_dir()))
    return out


def _group_totals(items: Iterable[ArtifactItem]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for item in items:
        entry = totals.setdefault(item.classification, {"count": 0, "size_bytes": 0, "size_gb": 0.0})
        entry["count"] += 1
        entry["size_bytes"] += item.size_bytes
    for entry in totals.values():
        entry["size_gb"] = round(float(entry["size_bytes"]) / 1024**3, 4)
    return dict(sorted(totals.items(), key=lambda kv: (-int(kv[1]["size_bytes"]), kv[0])))


def build_research_gc_report(
    *,
    sequence_pack_roots: Iterable[str | Path] = DEFAULT_SEQUENCE_PACK_ROOTS,
    studies_root: str | Path = DEFAULT_STUDIES_ROOT,
    reference_roots: Iterable[str | Path] = DEFAULT_REFERENCE_ROOTS,
    large_file_threshold_mb: float = 256.0,
    max_items: int = 200,
) -> dict[str, Any]:
    threshold_bytes = max(int(float(large_file_threshold_mb) * 1024 * 1024), 1)
    reference_text = _collect_reference_text(reference_roots)
    sequence_items = [
        classify_sequence_pack(path, reference_text=reference_text, large_file_threshold_bytes=threshold_bytes)
        for path in _scan_child_dirs(sequence_pack_roots)
    ]
    study_items = [
        classify_study(path, reference_text=reference_text, large_file_threshold_bytes=threshold_bytes)
        for path in _scan_child_dirs([studies_root])
    ]
    all_items = sorted([*sequence_items, *study_items], key=lambda item: item.size_bytes, reverse=True)
    safe_delete = [item for item in all_items if item.safe_to_delete_directory]
    trim_candidates = [item for item in all_items if item.cleanup_action == "trim_large_prediction_files"]
    total_size = sum(item.size_bytes for item in all_items)
    safe_delete_size = sum(item.size_bytes for item in safe_delete)
    trim_candidate_large_file_size = sum(
        int(file_item.get("size_bytes", 0))
        for item in trim_candidates
        for file_item in list(item.large_files or [])
        if file_item.get("kind") == "prediction_output"
    )
    return {
        "schema_version": 1,
        "status": "dry_run",
        "generated_at": _now(),
        "workspace_root": str(WORKSPACE_ROOT.resolve()),
        "policy": {
            "deletes_active_qdp_data": False,
            "delete_requires": ["--delete", f"--confirm-delete {DELETE_CONFIRMATION}"],
            "safe_delete_directory_rule": "only unreferenced smoke/partial/interrupted artifacts under allowed research roots",
            "prediction_trim_rule": f"use trim-predictions --delete --confirm-trim {TRIM_CONFIRMATION}; directory GC does not delete whole studies for prediction bloat",
        },
        "roots": {
            "sequence_pack_roots": [_relative(path) for path in sequence_pack_roots],
            "studies_root": _relative(studies_root),
            "reference_roots": [_relative(path) for path in reference_roots],
            "preferred_research_store_root": _relative(DEFAULT_RESEARCH_STORE_ROOT),
        },
        "totals": {
            "artifact_count": len(all_items),
            "size_bytes": total_size,
            "size_gb": round(float(total_size) / 1024**3, 4),
            "safe_delete_candidate_count": len(safe_delete),
            "safe_delete_candidate_size_bytes": safe_delete_size,
            "safe_delete_candidate_size_gb": round(float(safe_delete_size) / 1024**3, 4),
            "prediction_trim_candidate_count": len(trim_candidates),
            "prediction_trim_candidate_size_bytes": trim_candidate_large_file_size,
            "prediction_trim_candidate_size_gb": round(float(trim_candidate_large_file_size) / 1024**3, 4),
        },
        "classification_totals": _group_totals(all_items),
        "safe_delete_candidates": [item.to_dict() for item in safe_delete[:max_items]],
        "prediction_trim_candidates": [item.to_dict() for item in trim_candidates[:max_items]],
        "largest_artifacts": [item.to_dict() for item in all_items[:max_items]],
    }


def write_markdown_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    totals = dict(report.get("totals", {}) or {})
    lines = [
        "# Research Store GC Dry Run",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Total scanned size: `{totals.get('size_gb', 0)} GB`",
        f"- Safe delete candidates: `{totals.get('safe_delete_candidate_count', 0)}` / `{totals.get('safe_delete_candidate_size_gb', 0)} GB`",
        f"- Prediction trim candidates: `{totals.get('prediction_trim_candidate_count', 0)}` / `{totals.get('prediction_trim_candidate_size_gb', 0)} GB`",
        "",
        "## Policy",
        "",
        "- Active QDP datasets are not delete candidates.",
        "- Directory deletion requires explicit delete flags and only applies to unreferenced smoke/partial/interrupted research artifacts.",
        "- Large prediction output trimming uses the separate `trim-predictions` command; directory GC does not delete whole studies for prediction bloat.",
        "",
        "## Largest Artifacts",
        "",
        "| Size GB | Type | Classification | Recommendation | Path |",
        "|---:|---|---|---|---|",
    ]
    for item in list(report.get("largest_artifacts", []) or [])[:50]:
        lines.append(
            f"| {item.get('size_gb', 0)} | {item.get('artifact_type', '')} | {item.get('classification', '')} | "
            f"{item.get('recommendation', '')} | `{item.get('path', '')}` |"
        )
    lines.extend(["", "## Safe Delete Candidates", "", "| Size GB | Type | Reason | Path |", "|---:|---|---|---|"])
    for item in list(report.get("safe_delete_candidates", []) or [])[:50]:
        lines.append(
            f"| {item.get('size_gb', 0)} | {item.get('artifact_type', '')} | {', '.join(item.get('reasons', []))} | `{item.get('path', '')}` |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_prediction_trim_markdown_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    totals = dict(report.get("totals", {}) or {})
    lines = [
        "# Prediction Output Trim Report",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Status: `{report.get('status', '')}`",
        f"- Candidate studies: `{totals.get('candidate_study_count', 0)}`",
        f"- Candidate files: `{totals.get('candidate_file_count', 0)}`",
        f"- Candidate size: `{totals.get('candidate_size_gb', 0)} GB`",
        f"- Deleted files: `{totals.get('deleted_file_count', 0)}`",
        f"- Reclaimed size: `{totals.get('deleted_size_gb', 0)} GB`",
        "",
        "## Policy",
        "",
        "- Only large prediction CSV/parquet/feather outputs under the configured studies root are candidates.",
        "- Summary JSON, metrics CSV, reports and checkpoints are retained.",
        "- Delete mode requires `--delete --confirm-trim TRIM_PREDICTION_OUTPUTS`.",
        "",
        "## Candidates",
        "",
        "| Size GB | Files | Deleted | Study |",
        "|---:|---:|---:|---|",
    ]
    for item in list(report.get("candidates", []) or [])[:100]:
        lines.append(
            f"| {item.get('candidate_size_gb', 0)} | {item.get('candidate_file_count', 0)} | "
            f"{item.get('deleted_file_count', 0)} | `{item.get('study_path', '')}` |"
        )
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _rewrite_path_string(value: str, replacements: Mapping[str, str]) -> str:
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _rewrite_json_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _rewrite_path_string(value, replacements)
    if isinstance(value, list):
        return [_rewrite_json_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_json_strings(item, replacements) for key, item in value.items()}
    return value


def _path_replacements(source_root: Path, target_root: Path) -> dict[str, str]:
    source_abs = str(source_root.resolve())
    target_abs = str(target_root.resolve())
    source_rel = _relative(source_root)
    target_rel = _relative(target_root)
    return {
        source_abs: target_abs,
        source_abs.replace("\\", "/"): target_abs.replace("\\", "/"),
        source_rel: target_rel,
        source_rel.replace("/", "\\"): target_rel.replace("/", "\\"),
    }


def _rewrite_pack_json_files(pack_dir: Path, *, source_root: Path, target_root: Path, source_dir: Path, target_dir: Path) -> None:
    replacements = {
        **_path_replacements(source_root, target_root),
        **_path_replacements(source_dir, target_dir),
    }
    for json_path in [path for path in pack_dir.rglob("*.json") if path.is_file()]:
        payload = _read_json(json_path)
        if not payload:
            continue
        payload = _rewrite_json_strings(payload, replacements)
        if json_path.name == "manifest.json":
            payload.pop("artifact_view", None)
            payload["artifact_migration"] = {
                "schema_version": 1,
                "migrated_at": _now(),
                "owner": "daily_research",
                "storage_policy": "physical",
                "artifact_name": target_dir.name,
                "reason": "sequence packs are daily_research research artifacts, not QDP active data base",
            }
        _write_json(json_path, payload)


def migrate_legacy_sequence_pack(
    source_dir: str | Path,
    *,
    source_root: str | Path = Path("quant_data_platform/data/qdp_v2/research/sequence_pack"),
    target_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack",
) -> dict[str, Any]:
    source_root_path = _workspace_path(source_root).resolve()
    target_root_path = _workspace_path(target_root).resolve()
    source_path = _workspace_path(source_dir).resolve()
    source_path.relative_to(source_root_path)
    target_root_path.relative_to(WORKSPACE_ROOT.resolve())
    target_path = target_root_path / source_path.name
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(source_path)
    if not (source_path / "manifest.json").exists():
        raise ValueError(f"source pack has no manifest.json: {source_path}")
    if target_path.exists():
        raise FileExistsError(target_path)
    target_root_path.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    _rewrite_pack_json_files(
        target_path,
        source_root=source_root_path,
        target_root=target_root_path,
        source_dir=source_path,
        target_dir=target_path,
    )
    return {
        "status": "migrated",
        "source_dir": _relative(source_path),
        "target_dir": _relative(target_path),
        "manifest_path": _relative(target_path / "manifest.json"),
    }


def migrate_legacy_sequence_packs(
    *,
    source_root: str | Path = Path("quant_data_platform/data/qdp_v2/research/sequence_pack"),
    target_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack",
    reference_roots: Iterable[str | Path] = DEFAULT_REFERENCE_ROOTS,
) -> dict[str, Any]:
    source_root_path = _workspace_path(source_root)
    target_root_path = _workspace_path(target_root)
    reference_text = _collect_reference_text(reference_roots)
    migrated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for artifact_dir in _scan_child_dirs([source_root_path]):
        item = classify_sequence_pack(artifact_dir, reference_text=reference_text, large_file_threshold_bytes=1024**4)
        if item.classification not in {"full_sequence_pack", "view_or_reanchor_sequence_pack"}:
            skipped.append(
                {
                    "path": item.path,
                    "classification": item.classification,
                    "reason": "not_full_or_view_pack",
                }
            )
            continue
        result = migrate_legacy_sequence_pack(artifact_dir, source_root=source_root_path, target_root=target_root_path)
        result["classification"] = item.classification
        migrated.append(result)
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": _now(),
        "source_root": _relative(source_root_path),
        "target_root": _relative(target_root_path),
        "migrated_count": len(migrated),
        "skipped_count": len(skipped),
        "migrated": migrated,
        "skipped": skipped,
    }


def delete_safe_candidates(report: Mapping[str, Any], *, allowed_roots: Iterable[str | Path], confirm_delete: str) -> dict[str, Any]:
    if str(confirm_delete or "") != DELETE_CONFIRMATION:
        raise ValueError(f"delete requires --confirm-delete {DELETE_CONFIRMATION}")
    roots = [_workspace_path(root) for root in allowed_roots]
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in list(report.get("safe_delete_candidates", []) or []):
        path = _workspace_path(str(item.get("path", "") or ""))
        if not item.get("safe_to_delete_directory"):
            skipped.append({"path": _relative(path), "reason": "not_marked_safe_to_delete_directory"})
            continue
        if not _path_is_inside(path, roots):
            skipped.append({"path": _relative(path), "reason": "outside_allowed_roots"})
            continue
        if not path.exists() or not path.is_dir():
            skipped.append({"path": _relative(path), "reason": "missing_or_not_directory"})
            continue
        shutil.rmtree(path)
        deleted.append({"path": _relative(path), "size_bytes": int(item.get("size_bytes", 0) or 0)})
    return {
        "deleted_count": len(deleted),
        "deleted_size_bytes": sum(int(item.get("size_bytes", 0) or 0) for item in deleted),
        "deleted": deleted,
        "skipped": skipped,
    }


def _retained_study_files(study_dir: Path) -> list[str]:
    retained: list[str] = []
    exact_names = {
        *SUMMARY_FILE_NAMES,
        "split_metrics.csv",
        "topk_metrics.csv",
        "daily_rank_ic.csv",
        "training_history.csv",
        "sequence_path_training_report.md",
        "sequence_flat_lgbm_summary.json",
        "feature_importance.csv",
        "best_model.pt",
    }
    for path in sorted(study_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in exact_names or path.suffix.lower() in {".pt", ".md"}:
            retained.append(_relative(path))
    return retained[:200]


def _write_prediction_trim_manifest(
    study_dir: Path,
    *,
    candidate_files: list[dict[str, Any]],
    deleted_files: list[dict[str, Any]],
    skipped_files: list[dict[str, Any]],
) -> Path:
    manifest_path = study_dir / "prediction_trim_manifest.json"
    payload = {
        "schema_version": 1,
        "generated_at": _now(),
        "study_path": _relative(study_dir),
        "policy": {
            "trimmed_kind": "prediction_output",
            "retained": "summary JSON, metrics CSV, reports and checkpoints",
            "regeneration_note": "Full prediction rows were removed to control research-output bloat; regenerate explicitly from the pack/checkpoint only when needed.",
        },
        "candidate_file_count": len(candidate_files),
        "deleted_file_count": len(deleted_files),
        "deleted_size_bytes": sum(int(item.get("size_bytes", 0) or 0) for item in deleted_files),
        "deleted_size_gb": round(
            sum(int(item.get("size_bytes", 0) or 0) for item in deleted_files) / 1024**3,
            4,
        ),
        "candidate_files": candidate_files,
        "deleted_files": deleted_files,
        "skipped_files": skipped_files,
        "retained_files": _retained_study_files(study_dir),
    }
    return _write_json(manifest_path, payload)


def _mark_summaries_prediction_trimmed(study_dir: Path, manifest_path: Path, *, deleted_files: list[dict[str, Any]]) -> None:
    trimmed_size = sum(int(item.get("size_bytes", 0) or 0) for item in deleted_files)
    for summary_path in _study_summary_paths(study_dir):
        payload = _read_json(summary_path)
        if not payload:
            continue
        payload["prediction_outputs_trimmed"] = True
        payload["prediction_trimmed_at"] = _now()
        payload["prediction_trim_manifest"] = _relative(manifest_path)
        payload["prediction_trimmed_file_count"] = len(deleted_files)
        payload["prediction_trimmed_size_bytes"] = trimmed_size
        payload["prediction_trimmed_size_gb"] = round(float(trimmed_size) / 1024**3, 4)
        _write_json(summary_path, payload)


def trim_prediction_outputs(
    *,
    studies_root: str | Path = DEFAULT_STUDIES_ROOT,
    reference_roots: Iterable[str | Path] = DEFAULT_REFERENCE_ROOTS,
    large_file_threshold_mb: float = 256.0,
    max_items: int = 1000,
    delete: bool = False,
    confirm_trim: str = "",
) -> dict[str, Any]:
    if delete and str(confirm_trim or "") != TRIM_CONFIRMATION:
        raise ValueError(f"prediction trim requires --confirm-trim {TRIM_CONFIRMATION}")
    threshold_bytes = max(int(float(large_file_threshold_mb) * 1024 * 1024), 1)
    root = _workspace_path(studies_root)
    reference_text = _collect_reference_text(reference_roots)
    candidates: list[dict[str, Any]] = []
    totals = {
        "candidate_study_count": 0,
        "candidate_file_count": 0,
        "candidate_size_bytes": 0,
        "candidate_size_gb": 0.0,
        "deleted_file_count": 0,
        "deleted_size_bytes": 0,
        "deleted_size_gb": 0.0,
        "skipped_file_count": 0,
    }
    for study_dir in _scan_child_dirs([root]):
        item = classify_study(study_dir, reference_text=reference_text, large_file_threshold_bytes=threshold_bytes)
        if item.cleanup_action != "trim_large_prediction_files":
            continue
        files = _large_prediction_files(study_dir, threshold_bytes=threshold_bytes)
        if not files:
            continue
        candidate_size = sum(int(file_item.get("size_bytes", 0) or 0) for file_item in files)
        deleted_files: list[dict[str, Any]] = []
        skipped_files: list[dict[str, Any]] = []
        manifest_path = ""
        if delete:
            if not _path_is_inside(study_dir, [root]):
                skipped_files.append({"path": _relative(study_dir), "reason": "study_outside_allowed_root"})
            else:
                for file_item in files:
                    file_path = _workspace_path(str(file_item.get("path", "") or ""))
                    if not _path_is_inside(file_path, [study_dir]):
                        skipped_files.append({**file_item, "reason": "file_outside_study_dir"})
                        continue
                    if not file_path.exists() or not file_path.is_file():
                        skipped_files.append({**file_item, "reason": "missing_or_not_file"})
                        continue
                    try:
                        file_path.unlink()
                    except OSError as exc:
                        skipped_files.append({**file_item, "reason": f"unlink_failed:{exc.__class__.__name__}"})
                        continue
                    deleted_files.append(file_item)
                manifest_path_obj = _write_prediction_trim_manifest(
                    study_dir,
                    candidate_files=files,
                    deleted_files=deleted_files,
                    skipped_files=skipped_files,
                )
                manifest_path = _relative(manifest_path_obj)
                if deleted_files:
                    _mark_summaries_prediction_trimmed(study_dir, manifest_path_obj, deleted_files=deleted_files)
        deleted_size = sum(int(file_item.get("size_bytes", 0) or 0) for file_item in deleted_files)
        candidates.append(
            {
                "study_name": item.name,
                "study_path": item.path,
                "summary_path": item.summary_path,
                "referenced_by_brain": item.referenced_by_brain,
                "candidate_file_count": len(files),
                "candidate_size_bytes": candidate_size,
                "candidate_size_gb": round(float(candidate_size) / 1024**3, 4),
                "deleted_file_count": len(deleted_files),
                "deleted_size_bytes": deleted_size,
                "deleted_size_gb": round(float(deleted_size) / 1024**3, 4),
                "skipped_file_count": len(skipped_files),
                "trim_manifest": manifest_path,
                "files": files[:20],
                "skipped_files": skipped_files,
            }
        )
        totals["candidate_study_count"] += 1
        totals["candidate_file_count"] += len(files)
        totals["candidate_size_bytes"] += candidate_size
        totals["deleted_file_count"] += len(deleted_files)
        totals["deleted_size_bytes"] += deleted_size
        totals["skipped_file_count"] += len(skipped_files)
    totals["candidate_size_gb"] = round(float(totals["candidate_size_bytes"]) / 1024**3, 4)
    totals["deleted_size_gb"] = round(float(totals["deleted_size_bytes"]) / 1024**3, 4)
    candidates = sorted(candidates, key=lambda row: int(row.get("candidate_size_bytes", 0)), reverse=True)
    return {
        "schema_version": 1,
        "status": "trim_executed" if delete else "dry_run",
        "generated_at": _now(),
        "workspace_root": str(WORKSPACE_ROOT.resolve()),
        "policy": {
            "deletes_active_qdp_data": False,
            "delete_requires": ["--delete", f"--confirm-trim {TRIM_CONFIRMATION}"],
            "trim_rule": "delete only large prediction_output files under studies root; keep summaries, metrics, reports and checkpoints",
            "large_file_threshold_mb": float(large_file_threshold_mb),
        },
        "roots": {
            "studies_root": _relative(root),
            "reference_roots": [_relative(path) for path in reference_roots],
        },
        "totals": totals,
        "candidates": candidates[:max(int(max_items), 1)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run and guarded cleanup for daily_research model-ready artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Build a dry-run research artifact GC report.")
    scan.add_argument("--sequence-pack-root", action="append", type=Path, default=None)
    scan.add_argument("--studies-root", type=Path, default=DEFAULT_STUDIES_ROOT)
    scan.add_argument("--reference-root", action="append", type=Path, default=None)
    scan.add_argument("--large-file-threshold-mb", type=float, default=256.0)
    scan.add_argument("--max-items", type=int, default=200)
    scan.add_argument("--write-report", action="store_true")
    scan.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    scan.add_argument("--delete", action="store_true")
    scan.add_argument("--confirm-delete", default="")
    scan.add_argument("--json", action="store_true")
    migrate = sub.add_parser("migrate-legacy-packs", help="Physically move kept legacy sequence packs into daily_research research_store and rewrite manifests.")
    migrate.add_argument("--source-root", type=Path, default=Path("quant_data_platform/data/qdp_v2/research/sequence_pack"))
    migrate.add_argument("--target-root", type=Path, default=DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack")
    migrate.add_argument("--reference-root", action="append", type=Path, default=None)
    migrate.add_argument("--write-report", action="store_true")
    migrate.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    migrate.add_argument("--json", action="store_true")
    trim = sub.add_parser("trim-predictions", help="Dry-run or delete large prediction output files while keeping study summaries and metrics.")
    trim.add_argument("--studies-root", type=Path, default=DEFAULT_STUDIES_ROOT)
    trim.add_argument("--reference-root", action="append", type=Path, default=None)
    trim.add_argument("--large-file-threshold-mb", type=float, default=256.0)
    trim.add_argument("--max-items", type=int, default=1000)
    trim.add_argument("--write-report", action="store_true")
    trim.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    trim.add_argument("--delete", action="store_true")
    trim.add_argument("--confirm-trim", default="")
    trim.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        sequence_pack_roots = tuple(args.sequence_pack_root) if args.sequence_pack_root else DEFAULT_SEQUENCE_PACK_ROOTS
        reference_roots = tuple(args.reference_root) if args.reference_root else DEFAULT_REFERENCE_ROOTS
        report = build_research_gc_report(
            sequence_pack_roots=sequence_pack_roots,
            studies_root=args.studies_root,
            reference_roots=reference_roots,
            large_file_threshold_mb=float(args.large_file_threshold_mb),
            max_items=max(int(args.max_items), 1),
        )
        if args.write_report:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = _write_json(args.report_root / f"research_gc_dry_run_{stamp}.json", report)
            md_path = write_markdown_report(args.report_root / f"research_gc_dry_run_{stamp}.md", report)
            report = dict(report)
            report["report_paths"] = {"json": _relative(json_path), "markdown": _relative(md_path)}
        if args.delete:
            delete_result = delete_safe_candidates(
                report,
                allowed_roots=[*sequence_pack_roots, args.studies_root],
                confirm_delete=str(args.confirm_delete or ""),
            )
            report = dict(report)
            report["status"] = "delete_executed"
            report["delete_result"] = delete_result
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "migrate-legacy-packs":
        reference_roots = tuple(args.reference_root) if args.reference_root else DEFAULT_REFERENCE_ROOTS
        report = migrate_legacy_sequence_packs(
            source_root=args.source_root,
            target_root=args.target_root,
            reference_roots=reference_roots,
        )
        if args.write_report:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = _write_json(args.report_root / f"legacy_sequence_pack_migration_{stamp}.json", report)
            report = dict(report)
            report["report_path"] = _relative(report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "trim-predictions":
        reference_roots = tuple(args.reference_root) if args.reference_root else DEFAULT_REFERENCE_ROOTS
        report = trim_prediction_outputs(
            studies_root=args.studies_root,
            reference_roots=reference_roots,
            large_file_threshold_mb=float(args.large_file_threshold_mb),
            max_items=max(int(args.max_items), 1),
            delete=bool(args.delete),
            confirm_trim=str(args.confirm_trim or ""),
        )
        if args.write_report:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "prediction_trim_execute" if bool(args.delete) else "prediction_trim_dry_run"
            json_path = _write_json(args.report_root / f"{prefix}_{stamp}.json", report)
            md_path = write_prediction_trim_markdown_report(args.report_root / f"{prefix}_{stamp}.md", report)
            report = dict(report)
            report["report_paths"] = {"json": _relative(json_path), "markdown": _relative(md_path)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
