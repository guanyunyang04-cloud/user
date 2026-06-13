from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.brain.evidence_registry import load_evidence_registry


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAIN_MANIFEST = Path("brain/brain_manifest.json")


@dataclass(frozen=True)
class RouteCandidate:
    brain_id: str
    score: int
    matched_terms: list[str]
    sources: list[str]
    evidence_strength: str = "soft"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteTarget:
    id: str
    kind: str
    domain: str
    bootstrap_aliases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrainRouteDescriptor:
    brain_id: str
    body_root: str
    manifest_path: str
    aliases: list[str]
    path_prefixes: list[str]
    routing_terms: list[str]
    weak_terms: list[str]


WORKSPACE_TARGET = RouteTarget(
    id="workspace",
    kind="workspace",
    domain="workspace_governance",
    bootstrap_aliases=["workspace", "workspace_root", "workspace_governance"],
)

WORKSPACE_TERMS = (
    "主脑",
    "分脑",
    "脑区",
    "brain",
    "capsule",
    "接管",
    "治理",
    "main-only",
    "branch",
    "worktree",
    "workspace-brain",
    "workspace brain",
    "主分脑",
)

WORKSPACE_OVERRIDE_TERMS = (
    "brain路由",
    "路由机制",
    "route",
    "routing",
    "routing policy",
    "route_task_to_brain",
    "catalog",
    "brain_catalog",
    "workspace-brain",
    "workspace brain",
    "brain architecture",
    "agent learning",
    "agent meta",
    "skills",
    "skill",
    "脑区治理",
    "主脑",
    "分脑",
    "脑区",
    "项目大脑",
    "接管规则",
    "治理规则",
    "主分脑",
)

SOFT_ONLY_TERMS = {
    "brain",
    "workspace",
    "study",
    "training",
    "数据集",
    "模型",
    "复盘",
}


def _workspace_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(_workspace_path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _child_target(brain_id: str) -> RouteTarget:
    return RouteTarget(
        id=brain_id,
        kind="child",
        domain=brain_id,
        bootstrap_aliases=[brain_id],
    )


def _ambiguous_target() -> RouteTarget:
    return RouteTarget(id="", kind="ambiguous", domain="", bootstrap_aliases=[])


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _path_terms(path_text: str) -> list[str]:
    text = str(path_text or "").strip().replace("\\", "/").strip("/")
    if not text or text == ".":
        return []
    return [text, f"{text}/", text.replace("/", "\\") + "\\"]


def _routing_terms(child_manifest: dict[str, Any]) -> list[str]:
    routing_hints = child_manifest.get("routing_hints") if isinstance(child_manifest.get("routing_hints"), dict) else {}
    return _dedupe(_as_list(routing_hints.get("terms")))


def _weak_manifest_terms(child_ref: dict[str, Any], child_manifest: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    regional = child_manifest.get("regional_specialization") if isinstance(child_manifest.get("regional_specialization"), dict) else {}
    body_map = child_manifest.get("body_map") if isinstance(child_manifest.get("body_map"), dict) else {}
    terms.extend(_as_list(regional.get("body_entry_priority")))
    terms.extend(str(key) for key in body_map)
    for values in body_map.values():
        terms.extend(_as_list(values))
    for key in ("role", "regional_role"):
        value = str(child_ref.get(key, "") or child_manifest.get(key, "") or "").strip()
        if value:
            terms.append(value)
    return _dedupe(terms)


def load_route_descriptors() -> list[BrainRouteDescriptor]:
    main_manifest = _load_json(MAIN_MANIFEST)
    descriptors: list[BrainRouteDescriptor] = []
    for child in main_manifest.get("child_brains", []) if isinstance(main_manifest.get("child_brains"), list) else []:
        if not isinstance(child, dict):
            continue
        brain_id = str(child.get("id", "") or "").strip()
        manifest_path = str(child.get("path", "") or "").strip().replace("\\", "/")
        body_root = str(child.get("body_root", "") or "").strip().replace("\\", "/")
        if not brain_id or not manifest_path:
            continue
        child_manifest = _load_json(manifest_path)
        routing_hints = child_manifest.get("routing_hints") if isinstance(child_manifest.get("routing_hints"), dict) else {}
        hint_prefixes = _as_list(routing_hints.get("path_prefixes"))
        path_prefixes = _dedupe([body_root, *hint_prefixes])
        aliases = _dedupe([brain_id, body_root, *[Path(body_root).name if body_root else ""], *_as_list(routing_hints.get("aliases"))])
        descriptors.append(
            BrainRouteDescriptor(
                brain_id=brain_id,
                body_root=body_root,
                manifest_path=manifest_path,
                aliases=aliases,
                path_prefixes=path_prefixes,
                routing_terms=_routing_terms(child_manifest),
                weak_terms=_weak_manifest_terms(child, child_manifest),
            )
        )
    return descriptors


def _matches(task: str, terms: list[str] | tuple[str, ...]) -> list[str]:
    lower = task.lower()
    matches: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if not text:
            continue
        if text.lower() in lower or text in task:
            matches.append(text)
    return _dedupe(matches)


def _is_soft_only_term(term: str) -> bool:
    text = str(term or "").strip().lower()
    if not text:
        return True
    if text in SOFT_ONLY_TERMS:
        return True
    return False


def _split_term_strength(matches: list[str]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    for term in matches:
        (soft if _is_soft_only_term(term) else hard).append(term)
    return _dedupe(hard), _dedupe(soft)


def _project_path_matches(task: str, descriptors: list[BrainRouteDescriptor]) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    for descriptor in descriptors:
        terms: list[str] = []
        for prefix in descriptor.path_prefixes:
            terms.extend(_path_terms(prefix))
        matches = _matches(task, terms)
        if matches:
            candidates.append(
                RouteCandidate(
                    brain_id=descriptor.brain_id,
                    score=100 + len(matches),
                    matched_terms=matches,
                    sources=["path_prefix_match"],
                    evidence_strength="hard",
                )
            )
    return candidates


def _descriptor_term_matches(task: str, descriptors: list[BrainRouteDescriptor]) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    for descriptor in descriptors:
        alias_matches = _matches(task, descriptor.aliases)
        routing_matches = _matches(task, descriptor.routing_terms)
        weak_matches = _matches(task, descriptor.weak_terms)
        hard_routing_matches, soft_routing_matches = _split_term_strength(routing_matches)
        if not alias_matches and not hard_routing_matches and not soft_routing_matches and not weak_matches:
            continue
        sources: list[str] = []
        score = 0
        evidence_strength = "soft"
        if alias_matches:
            sources.append("alias_match")
            score += 20 + len(alias_matches)
            evidence_strength = "hard"
        if hard_routing_matches:
            sources.append("manifest_term_match")
            score += 10 + len(hard_routing_matches)
            evidence_strength = "hard"
        if soft_routing_matches:
            sources.append("manifest_soft_term_match")
            score += len(soft_routing_matches)
        if weak_matches:
            sources.append("body_map_term_match")
            score += len(weak_matches)
        candidates.append(
            RouteCandidate(
                brain_id=descriptor.brain_id,
                score=score,
                matched_terms=_dedupe([*alias_matches, *hard_routing_matches, *soft_routing_matches, *weak_matches]),
                sources=sources,
                evidence_strength=evidence_strength,
            )
        )
    return candidates


def _workspace_matches(task: str) -> list[RouteCandidate]:
    matches = _matches(task, WORKSPACE_TERMS)
    if not matches:
        return []
    return [
        RouteCandidate(
            brain_id="workspace_governance",
            score=len(matches),
            matched_terms=matches,
            sources=["workspace_term_match"],
            evidence_strength="hard" if _matches(task, WORKSPACE_OVERRIDE_TERMS) else "soft",
        )
    ]


def _workspace_override_matches(task: str) -> list[str]:
    return _matches(task, WORKSPACE_OVERRIDE_TERMS)


def _registry_matches(task: str, descriptors: list[BrainRouteDescriptor]) -> list[RouteCandidate]:
    text = str(task or "").strip()
    if not text:
        return []
    try:
        payload = load_evidence_registry()
    except Exception:
        return []
    by_root = {descriptor.body_root.replace("\\", "/").strip("/"): descriptor.brain_id for descriptor in descriptors if descriptor.body_root}
    matched_hard: dict[str, set[str]] = {}
    matched_soft: dict[str, set[str]] = {}
    for record in payload.get("records", []) or []:
        if not isinstance(record, dict):
            continue
        path = str(record.get("path", "") or "").replace("\\", "/")
        brain_id = ""
        for root, candidate_id in by_root.items():
            if path.startswith(f"{root}/brain/references/"):
                brain_id = candidate_id
                break
        if not brain_id:
            continue
        hard_values = [
            str(record.get("id", "") or ""),
            *[str(item) for item in record.get("research_programs", []) or []],
            *[str(item) for item in record.get("study_families", []) or []],
            *[str(item) for item in record.get("run_tags", []) or []],
            *[str(item) for item in record.get("dataset_ids", []) or []],
        ]
        path_values = [path, Path(path).name if path else ""]
        soft_values = [str(record.get("workflow", "") or ""), *[str(item) for item in record.get("tags", []) or []]]
        for value in hard_values:
            clean = str(value or "").strip()
            if clean and not _is_soft_only_term(clean) and clean.lower() in text.lower():
                matched_hard.setdefault(brain_id, set()).add(clean)
        for value in path_values:
            clean = str(value or "").strip()
            if clean and len(clean) >= 8 and clean.lower() in text.lower():
                matched_hard.setdefault(brain_id, set()).add(clean)
        for value in soft_values:
            clean = str(value or "").strip()
            if clean and clean.lower() in text.lower():
                matched_soft.setdefault(brain_id, set()).add(clean)

    candidates: list[RouteCandidate] = []
    for brain_id, values in matched_hard.items():
        candidates.append(
            RouteCandidate(
                brain_id=brain_id,
                score=80 + len(values),
                matched_terms=sorted(values),
                sources=["registry_exact_match"],
                evidence_strength="hard",
            )
        )
    for brain_id, values in matched_soft.items():
        candidates.append(
            RouteCandidate(
                brain_id=brain_id,
                score=len(values),
                matched_terms=sorted(values),
                sources=["registry_soft_match"],
                evidence_strength="soft",
            )
        )
    return candidates


def _merge_candidates(candidates: list[RouteCandidate]) -> list[RouteCandidate]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        item = merged.setdefault(candidate.brain_id, {"score": 0, "matched_terms": [], "sources": [], "evidence_strength": "soft"})
        item["score"] += int(candidate.score)
        item["matched_terms"].extend(candidate.matched_terms)
        item["sources"].extend(candidate.sources)
        if candidate.evidence_strength == "hard":
            item["evidence_strength"] = "hard"
    return [
        RouteCandidate(
            brain_id=brain_id,
            score=int(payload["score"]),
            matched_terms=sorted(set(payload["matched_terms"])),
            sources=sorted(set(payload["sources"])),
            evidence_strength=str(payload["evidence_strength"]),
        )
        for brain_id, payload in merged.items()
    ]


def route_task_to_brain(task: str) -> dict[str, Any]:
    text = str(task or "").strip()
    descriptors = load_route_descriptors()
    raw_candidates = [
        *_project_path_matches(text, descriptors),
        *_descriptor_term_matches(text, descriptors),
        *_workspace_matches(text),
        *_registry_matches(text, descriptors),
    ]
    candidates = _merge_candidates(raw_candidates)
    candidates = sorted(candidates, key=lambda item: (-item.score, item.brain_id))
    child_ids = {descriptor.brain_id for descriptor in descriptors}
    hard_children = [item for item in candidates if item.brain_id in child_ids and item.evidence_strength == "hard"]
    soft_children = [item for item in candidates if item.brain_id in child_ids and item.evidence_strength != "hard"]
    workspace_override_matches = _workspace_override_matches(text)
    has_workspace_candidate = any(item.brain_id == "workspace_governance" for item in candidates)
    selected = "workspace"
    target = WORKSPACE_TARGET
    status = "selected"
    reason = "no manifest, catalog, path, or registry evidence matched; default to workspace governance"
    confidence = "medium"
    decision_required = False
    decision_reason = ""
    recommended_default = "workspace"

    if len(hard_children) > 1:
        selected = ""
        target = _ambiguous_target()
        status = "ambiguous"
        confidence = "low"
        decision_required = True
        recommended_default = "none"
        reason = "multiple child brains matched; main brain must not default to a child"
        decision_reason = "multiple child brains have hard routing evidence"
    elif hard_children:
        selected = hard_children[0].brain_id
        target = _child_target(selected)
        confidence = "high"
        recommended_default = selected
        reason = f"task matched {selected} via {', '.join(hard_children[0].sources)}"
        if workspace_override_matches:
            reason += "; workspace governance terms are advisory only because a single child has hard evidence"
    elif workspace_override_matches or has_workspace_candidate:
        confidence = "high" if workspace_override_matches else "medium"
        reason = "task matched workspace governance terms"
    elif soft_children:
        selected = ""
        target = _ambiguous_target()
        status = "needs_agent_decision"
        confidence = "low"
        decision_required = True
        recommended_default = "workspace"
        reason = "only soft child routing evidence matched; runtime will not choose a child"
        decision_reason = "agent must inspect user intent before bootstrapping any child brain"

    candidate_summary = [
        {
            "brain_id": item.brain_id,
            "evidence_strength": item.evidence_strength,
            "matched_terms": item.matched_terms,
            "sources": item.sources,
        }
        for item in candidates
    ]
    return {
        "status": status,
        "selected_brain_id": selected,
        "target": target.to_dict(),
        "candidates": [item.to_dict() for item in candidates],
        "routing_sources": sorted({source for item in candidates for source in item.sources}),
        "reason": reason,
        "confidence": confidence,
        "decision_required": decision_required,
        "decision_reason": decision_reason,
        "recommended_default": recommended_default,
        "workspace_governance_signal": {
            "matched": bool(workspace_override_matches or has_workspace_candidate),
            "matched_terms": workspace_override_matches,
            "advisory_only": bool(hard_children and len(hard_children) == 1),
        },
        "candidate_summary": candidate_summary,
    }
