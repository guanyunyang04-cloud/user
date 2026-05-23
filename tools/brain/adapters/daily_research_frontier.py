from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOTS = {
    "path_policy": Path("daily_research/output/path_policy/studies"),
    "continuous_policy": Path("daily_research/output/continuous_policy/studies"),
}
SUMMARY_FILENAMES = (
    "study_summary.json",
    "forecast_walkforward_summary.json",
    "forecast_training_summary.json",
)
BRAIN_REFERENCE_ROOT = Path("daily_research/brain/references")
EVIDENCE_REGISTRY_PATH = BRAIN_REFERENCE_ROOT / "evidence_registry.json"
DATASET_ID_PATTERN = re.compile(
    r"\b(?:policy_[A-Za-z0-9_]+|continuous_policy_[A-Za-z0-9_]+)(?:__[A-Za-z0-9_]+)*__[0-9a-f]{16,32}\b"
)
FAILED_STATUS_VALUES = {
    "failed",
    "failure",
    "error",
    "interrupted",
    "timeout",
    "timed_out",
    "cancelled",
    "canceled",
    "running",
    "in_progress",
    "pending",
}
FAILED_VERDICT_MARKERS = (
    "failed",
    "failure",
    "interrupted",
    "timeout",
    "blocked",
    "incomplete",
    "insufficient",
)


def _workspace_root(workspace_root: str | Path | None = None) -> Path:
    return Path(workspace_root).resolve() if workspace_root is not None else WORKSPACE_ROOT


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _mtime_epoch(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _iso_from_epoch(epoch: float) -> str:
    if epoch <= 0:
        return ""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def _dataset_ids(payload: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for value in _walk_values(payload):
        if not isinstance(value, str):
            continue
        values.extend(DATASET_ID_PATTERN.findall(value.strip()))
    return _dedupe(values)


def _model_families(payload: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("forecast_model_families", "model_families", "models"):
        value = payload.get(key)
        if isinstance(value, list):
            out.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, tuple):
            out.extend(str(item) for item in value if str(item).strip())
        elif isinstance(value, Mapping):
            out.extend(str(item) for item in value.keys() if str(item).strip())
        elif isinstance(value, str) and value.strip():
            out.extend(part.strip() for part in value.split(",") if part.strip())
    selected = payload.get("selected_model_summary")
    if isinstance(selected, Mapping):
        for key in ("model_family", "family", "model_type", "selected_family"):
            value = selected.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
    return _dedupe(out)


def _tag_from(study_dir: Path, payload: Mapping[str, Any]) -> str:
    for key in ("study_tag", "run_tag", "protocol_tag", "tag"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return study_dir.name


def _completed_evidence(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status", "") or "").strip().lower()
    verdict = str(payload.get("evidence_verdict", "") or "").strip().lower()
    if status != "completed":
        return False
    if status in FAILED_STATUS_VALUES:
        return False
    return not any(marker in verdict for marker in FAILED_VERDICT_MARKERS)


def _summary_path_for_study(study_dir: Path) -> Path | None:
    for filename in SUMMARY_FILENAMES:
        candidate = study_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _scan_output_studies(root: Path) -> list[dict[str, Any]]:
    studies: list[dict[str, Any]] = []
    for workflow, rel_root in STUDY_ROOTS.items():
        abs_root = root / rel_root
        if not abs_root.exists():
            continue
        for study_dir in sorted(path for path in abs_root.iterdir() if path.is_dir()):
            summary_path = _summary_path_for_study(study_dir)
            if summary_path is None:
                continue
            payload = _load_json(summary_path)
            mtime_epoch = _mtime_epoch(summary_path)
            studies.append(
                {
                    "workflow": workflow,
                    "tag": _tag_from(study_dir, payload),
                    "mtime": _iso_from_epoch(mtime_epoch),
                    "mtime_epoch": mtime_epoch,
                    "status": str(payload.get("status", "") or ""),
                    "evidence_verdict": str(payload.get("evidence_verdict", "") or ""),
                    "stage": str(payload.get("stage", "") or ""),
                    "dataset_ids": _dataset_ids(payload),
                    "model_families": _model_families(payload),
                    "summary_path": _relative(summary_path, root),
                    "completed_evidence": _completed_evidence(payload),
                }
            )
    return sorted(studies, key=lambda item: float(item.get("mtime_epoch", 0.0) or 0.0), reverse=True)


def _latest_brain_reference(root: Path) -> tuple[str, float]:
    ref_root = root / BRAIN_REFERENCE_ROOT
    if not ref_root.exists():
        return "", 0.0
    latest = 0.0
    for path in ref_root.iterdir():
        if path.is_file() and path.suffix.lower() in {".md", ".json"}:
            latest = max(latest, _mtime_epoch(path))
    return _iso_from_epoch(latest), latest


def _registered_study_tags(root: Path) -> list[str]:
    registry_path = root / EVIDENCE_REGISTRY_PATH
    if not registry_path.exists():
        return []
    registry = _load_json(registry_path)
    tags: list[str] = []
    records = registry.get("records", [])
    if not isinstance(records, list):
        return []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for key in ("study_tags", "study_tag", "tag", "id"):
            value = record.get(key)
            if isinstance(value, list):
                tags.extend(str(item) for item in value)
            elif isinstance(value, str):
                tags.append(value)
    return _dedupe(tags)


def build_frontier_report(workspace_root: str | Path | None = None, max_studies: int = 12) -> dict[str, Any]:
    """Build a read-only freshness report for daily_research handoff decisions."""

    root = _workspace_root(workspace_root)
    all_studies = _scan_output_studies(root)
    latest_studies = all_studies[: max(int(max_studies), 0)]
    latest_brain_reference_time, latest_brain_reference_epoch = _latest_brain_reference(root)
    registered_tags = set(_registered_study_tags(root))
    latest_tags = [str(study.get("tag", "") or "") for study in latest_studies if study.get("tag")]
    unregistered_latest_tags = [tag for tag in latest_tags if tag not in registered_tags]
    latest_output_epoch = max((float(study.get("mtime_epoch", 0.0) or 0.0) for study in latest_studies), default=0.0)
    output_newer_than_brain = bool(latest_output_epoch and latest_output_epoch > latest_brain_reference_epoch)
    warnings: list[str] = []
    if output_newer_than_brain:
        warnings.append("output_newer_than_brain_references")
    if unregistered_latest_tags:
        warnings.append("unregistered_latest_output_tags")
    return {
        "schema_version": 1,
        "latest_output_studies": latest_studies,
        "latest_brain_reference_time": latest_brain_reference_time,
        "latest_brain_reference_epoch": latest_brain_reference_epoch,
        "registered_study_tag_count": len(registered_tags),
        "unregistered_latest_tags": unregistered_latest_tags,
        "output_newer_than_brain": output_newer_than_brain,
        "brain_may_be_stale": output_newer_than_brain or bool(unregistered_latest_tags),
        "warnings": warnings,
        "guidance": "If brain_may_be_stale=true, read output explicit tags before answering current state or next-step questions.",
    }

