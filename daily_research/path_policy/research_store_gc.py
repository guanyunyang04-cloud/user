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
    Path("quant_data_platform/data/qdp_v2/research/sequence_pack"),
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
    summary_path = artifact_dir / "study_summary.json"
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
            "prediction_trim_rule": "reported only; not deleted by directory GC",
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
        "- Large prediction output trimming is reported but not deleted by directory GC.",
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


def register_sequence_pack_view(
    source_manifest: str | Path,
    *,
    view_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack",
    overwrite: bool = False,
) -> dict[str, Any]:
    source = _workspace_path(source_manifest)
    if not source.exists():
        raise FileNotFoundError(source)
    manifest = _read_json(source)
    if manifest.get("artifact_type") != "qdp_v2_sequence_path_pack":
        raise ValueError(f"not a qdp_v2 sequence path pack manifest: {source}")
    view_dir = _workspace_path(view_root) / source.parent.name
    view_manifest = view_dir / "manifest.json"
    if view_manifest.exists() and not overwrite:
        return {
            "status": "exists",
            "source_manifest": _relative(source),
            "view_manifest": _relative(view_manifest),
            "view_dir": _relative(view_dir),
        }

    payload = dict(manifest)
    payload["artifact_view"] = {
        "schema_version": 1,
        "view_type": "legacy_sequence_pack_zero_copy",
        "created_at": _now(),
        "owner": "daily_research",
        "source_manifest": str(source.resolve()),
        "source_artifact_root": str(source.parent.resolve()),
        "view_dir": str(view_dir.resolve()),
        "storage_policy": "zero_copy",
        "note": "Array paths intentionally remain in the source compatibility location; this view changes ownership/entrypoint, not physical storage.",
    }
    _write_json(view_manifest, payload)
    (view_dir / "VIEW.md").write_text(
        "\n".join(
            [
                "# Zero-Copy Sequence Pack View",
                "",
                f"- Source manifest: `{_relative(source)}`",
                "- Owner: `daily_research`",
                "- Storage policy: `zero_copy`",
                "- The manifest is readable by existing training code because array paths are preserved.",
                "- Do not delete or move the source artifact until a physical migration or rebuild replaces these paths.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "status": "created",
        "source_manifest": _relative(source),
        "view_manifest": _relative(view_manifest),
        "view_dir": _relative(view_dir),
    }


def register_legacy_sequence_pack_views(
    *,
    source_roots: Iterable[str | Path] = (Path("quant_data_platform/data/qdp_v2/research/sequence_pack"),),
    view_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack",
    reference_roots: Iterable[str | Path] = DEFAULT_REFERENCE_ROOTS,
    overwrite: bool = False,
) -> dict[str, Any]:
    reference_text = _collect_reference_text(reference_roots)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for artifact_dir in _scan_child_dirs(source_roots):
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
        result = register_sequence_pack_view(
            artifact_dir / "manifest.json",
            view_root=view_root,
            overwrite=overwrite,
        )
        result["classification"] = item.classification
        created.append(result)
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": _now(),
        "source_roots": [_relative(path) for path in source_roots],
        "view_root": _relative(view_root),
        "created_or_existing_count": len(created),
        "skipped_count": len(skipped),
        "views": created,
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
    views = sub.add_parser("register-legacy-views", help="Create zero-copy daily_research view manifests for legacy sequence packs.")
    views.add_argument("--source-root", action="append", type=Path, default=None)
    views.add_argument("--view-root", type=Path, default=DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack")
    views.add_argument("--reference-root", action="append", type=Path, default=None)
    views.add_argument("--overwrite", action="store_true")
    views.add_argument("--write-report", action="store_true")
    views.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    views.add_argument("--json", action="store_true")
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
    if args.command == "register-legacy-views":
        source_roots = tuple(args.source_root) if args.source_root else (Path("quant_data_platform/data/qdp_v2/research/sequence_pack"),)
        reference_roots = tuple(args.reference_root) if args.reference_root else DEFAULT_REFERENCE_ROOTS
        report = register_legacy_sequence_pack_views(
            source_roots=source_roots,
            view_root=args.view_root,
            reference_roots=reference_roots,
            overwrite=bool(args.overwrite),
        )
        if args.write_report:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = _write_json(args.report_root / f"legacy_sequence_pack_views_{stamp}.json", report)
            report = dict(report)
            report["report_path"] = _relative(report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
