from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.brain.adapters import daily_research_evidence
from tools.brain.platform import WORKSPACE_ROOT, read_text, write_json


REGISTRY_PATH = daily_research_evidence.REGISTRY_PATH
MAIN_REFERENCE_ROOTS = (Path("brain/references"),)
BRAIN_NATIVE_DOC_PATTERN = re.compile(r"^(brain_native|brain_system|api_agent)_.+\.md$")
R_ID_PATTERN = re.compile(r"\br\d+[a-z]?\b", re.IGNORECASE)
DATASET_ID_PATTERN = re.compile(r"\b[A-Za-z0-9_]+(?:__[A-Za-z0-9_]+)*__[0-9a-f]{16,32}\b")


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    date: str
    r_id: str
    path: str
    tags: list[str]
    workflow: str
    research_programs: list[str]
    study_families: list[str]
    run_tags: list[str]
    verdict: str
    blockers: list[str]
    next_allowed_actions: list[str]
    active_artifact_impact: str
    dataset_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reference_files() -> list[Path]:
    files: list[Path] = []
    for root_rel in MAIN_REFERENCE_ROOTS:
        root = WORKSPACE_ROOT / root_rel
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.glob("*.md")
            if path.is_file()
            and (
                BRAIN_NATIVE_DOC_PATTERN.match(path.name)
                or path.name.startswith("r")
                or path.name.startswith("workspace_")
            )
        )
    for root_rel in daily_research_evidence.REFERENCE_ROOTS:
        root = WORKSPACE_ROOT / root_rel
        if not root.exists():
            continue
        files.extend(path for path in root.glob("*.md") if daily_research_evidence.is_reference_file(path))
    return sorted(files)


def _first_date(text: str, path: Path) -> str:
    for pattern in (r"\b(20\d{2}-\d{2}-\d{2})\b", r"_(20\d{6})\.md$"):
        match = re.search(pattern, text if pattern.startswith(r"\b") else path.name)
        if match:
            value = match.group(1)
            if len(value) == 8:
                return f"{value[:4]}-{value[4:6]}-{value[6:]}"
            return value
    return ""


def _r_id_from(path: Path, text: str) -> str:
    match = R_ID_PATTERN.search(path.name)
    if not match:
        match = R_ID_PATTERN.search(text)
    return match.group(0).lower() if match else ""


def _normalize_bullet(line: str) -> str:
    return line.strip().lstrip("- ").strip()


def _heading_text(line: str) -> str:
    return line.strip().lstrip("#").strip().lower()


def _section_bullets(lines: Sequence[str], heading_keywords: Iterable[str], *, limit: int = 8) -> list[str]:
    out: list[str] = []
    lowered = [keyword.lower() for keyword in heading_keywords]
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = _heading_text(stripped)
            if in_section:
                break
            in_section = any(keyword in heading for keyword in lowered)
            continue
        if not in_section or not stripped.startswith("-"):
            continue
        out.append(_normalize_bullet(stripped))
        if len(out) >= limit:
            break
    return _dedupe(out)


def _inline_keyword_bullets(lines: Sequence[str], keywords: Iterable[str], *, limit: int = 8) -> list[str]:
    out: list[str] = []
    lowered = [keyword.lower() for keyword in keywords]
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        text = _normalize_bullet(stripped)
        haystack = text.lower()
        if any(keyword in haystack for keyword in lowered):
            out.append(text)
        if len(out) >= limit:
            break
    return _dedupe(out)


def _inline_next_action_bullets(lines: Sequence[str], *, limit: int = 6) -> list[str]:
    out: list[str] = []
    prefixes = ("next allowed", "next work", "next research", "next focus", "next gate", "next decision", "下一步", "后续")
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        text = _normalize_bullet(stripped)
        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            out.append(text)
        if len(out) >= limit:
            break
    return _dedupe(out)


def _evidence_summary(lines: Sequence[str]) -> tuple[str, list[str], list[str]]:
    verdict = _section_bullets(lines, ("verdict", "current verdict", "interpretation", "summary", "结论", "判定"), limit=4)
    blockers = _section_bullets(
        lines,
        ("blocker", "remaining blocker", "behavior evidence", "safe screening evidence", "阻塞", "失败"),
        limit=8,
    )
    next_actions = _section_bullets(lines, ("next", "next focus", "next gate", "next decision", "下一步", "后续"), limit=6)
    if not verdict:
        verdict = _inline_keyword_bullets(lines, ("current verdict", "verdict", "boundary", "边界"), limit=4)
    if not blockers:
        blockers = _inline_keyword_bullets(lines, ("blocker", "failed", "failure", "dead", "insufficient"), limit=8)
    if not next_actions:
        next_actions = _inline_next_action_bullets(lines, limit=6)
    return "; ".join(verdict), blockers, next_actions


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


def _workflow_from(path: Path, text: str) -> str:
    if daily_research_evidence.owns_reference_path(path, WORKSPACE_ROOT):
        return daily_research_evidence.workflow_from(path, text)
    lower = f"{path.name}\n{text}".lower()
    if path.name.startswith(("brain_native", "brain_system", "api_agent")):
        return "brain"
    if "brain maintenance" in lower or "brain-skill" in lower or "brain_skill" in lower:
        return "brain"
    if "brain" in lower:
        return "brain"
    return "workspace"


def _tags_from(path: Path, text: str, r_id: str, workflow: str) -> list[str]:
    tags = [r_id, workflow] if r_id else [workflow]
    if daily_research_evidence.owns_reference_path(path, WORKSPACE_ROOT):
        return _dedupe([*tags, *daily_research_evidence.extra_tags(path, text, workflow)])
    if workflow == "brain":
        return _dedupe([*tags, "brain"])
    return _dedupe(tags)


def _active_artifact_impact(text: str) -> str:
    lower = text.lower()
    if "active_execution_strategy.json remains unchanged" in lower or "active artifact" in lower and "unchanged" in lower:
        return "unchanged"
    if "不触碰" in text and "active" in lower:
        return "unchanged"
    if "active_execution_strategy.json" in lower and "diff" in lower:
        return "guarded"
    return "not_stated"


def build_evidence_record(path: Path) -> EvidenceRecord:
    rel_path = path.relative_to(WORKSPACE_ROOT).as_posix() if path.is_absolute() else path.as_posix()
    text = read_text(rel_path)
    lines = text.splitlines()
    r_id = _r_id_from(path, text)
    record_id = r_id or path.stem
    workflow = _workflow_from(path, text)
    verdict, blockers, next_allowed_actions = _evidence_summary(lines)
    return EvidenceRecord(
        id=record_id,
        date=_first_date(text, path),
        r_id=r_id,
        path=rel_path,
        tags=_tags_from(path, text, r_id, workflow),
        workflow=workflow,
        research_programs=daily_research_evidence.research_programs(text),
        study_families=daily_research_evidence.study_families(text),
        run_tags=daily_research_evidence.run_tags(text),
        verdict=verdict,
        blockers=blockers,
        next_allowed_actions=next_allowed_actions,
        active_artifact_impact=_active_artifact_impact(text),
        dataset_ids=_dedupe(DATASET_ID_PATTERN.findall(text)),
    )


def build_evidence_registry() -> dict[str, Any]:
    records = [build_evidence_record(path) for path in _reference_files()]
    r_id_counts: dict[str, int] = {}
    for record in records:
        if record.r_id:
            r_id_counts[record.r_id] = r_id_counts.get(record.r_id, 0) + 1
    records = [
        EvidenceRecord(
            id=Path(record.path).stem if record.r_id and r_id_counts.get(record.r_id, 0) > 1 else record.id,
            date=record.date,
            r_id=record.r_id,
            path=record.path,
            tags=record.tags,
            workflow=record.workflow,
            research_programs=record.research_programs,
            study_families=record.study_families,
            run_tags=record.run_tags,
            verdict=record.verdict,
            blockers=record.blockers,
            next_allowed_actions=record.next_allowed_actions,
            active_artifact_impact=record.active_artifact_impact,
            dataset_ids=record.dataset_ids,
        )
        for record in records
    ]
    ids: set[str] = set()
    duplicates: list[str] = []
    missing_paths: list[str] = []
    for record in records:
        if record.id in ids:
            duplicates.append(record.id)
        ids.add(record.id)
        if not (WORKSPACE_ROOT / record.path).exists():
            missing_paths.append(record.path)
    return {
        "schema_version": 3,
        "status": "ok" if not duplicates and not missing_paths else "invalid",
        "record_count": len(records),
        "duplicates": duplicates,
        "missing_paths": missing_paths,
        "records": [record.to_dict() for record in records],
    }


def write_evidence_registry(path: str | Path = REGISTRY_PATH) -> Path:
    payload = build_evidence_registry()
    return write_json(path, payload)


def load_evidence_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    target = WORKSPACE_ROOT / Path(path)
    if not target.exists():
        return build_evidence_registry()
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def query_evidence_registry(query: str, *, path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    needle = str(query or "").strip().lower()
    registry = load_evidence_registry(path)
    records = registry.get("records", [])
    if not needle:
        matches = records
    else:
        exact_matches = [
            record
            for record in records
            if isinstance(record, dict)
            and (
                needle in {str(record.get("id", "")).lower(), str(record.get("r_id", "")).lower()}
                or needle in {str(item).lower() for item in record.get("research_programs", []) if item}
                or needle in {str(item).lower() for item in record.get("study_families", []) if item}
                or needle in {str(item).lower() for item in record.get("run_tags", []) if item}
                or needle in {str(item).lower() for item in record.get("dataset_ids", []) if item}
            )
        ]
        fuzzy_matches = [
            record
            for record in records
            if isinstance(record, dict)
            and needle in json.dumps(record, ensure_ascii=False).lower()
        ]
        matches = exact_matches or fuzzy_matches
    return {
        "query": query,
        "match_count": len(matches),
        "matches": matches,
        "registry_status": registry.get("status", ""),
        "registry_path": Path(path).as_posix(),
    }
