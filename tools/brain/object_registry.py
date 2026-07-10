from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OBJECT_REGISTRY = Path("brain/object_registry.json")


@dataclass(frozen=True)
class ObjectMatch:
    object_id: str
    owner: str
    type: str
    matched_terms: list[str]
    matched_write_terms: list[str]
    matched_paths: list[str]
    route_reason: str
    default_mode: str = "read_only"
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _workspace_path(path: str | Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


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


def load_object_registry(path: str | Path = OBJECT_REGISTRY) -> dict[str, Any]:
    payload = json.loads(_workspace_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"object registry must be a JSON object: {path}")
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("object registry must declare objects as a list")
    return payload


def registry_objects(path: str | Path = OBJECT_REGISTRY) -> list[dict[str, Any]]:
    return [dict(item) for item in load_object_registry(path).get("objects", []) if isinstance(item, dict)]


def object_by_id(object_id: str, path: str | Path = OBJECT_REGISTRY) -> dict[str, Any]:
    target = str(object_id or "").strip()
    for item in registry_objects(path):
        if str(item.get("object_id", "") or "").strip() == target:
            return item
    return {}


def terms_for_owner(owner: str, field: str = "routing_terms") -> tuple[str, ...]:
    owner_text = str(owner or "").strip()
    out: list[str] = []
    for item in registry_objects():
        if str(item.get("owner", "") or "").strip() != owner_text:
            continue
        out.extend(_as_list(item.get(field)))
    return tuple(_dedupe(out))


def terms_for_object(object_id: str, field: str = "routing_terms") -> tuple[str, ...]:
    item = object_by_id(object_id)
    return tuple(_as_list(item.get(field)))


def _matches_text(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    matches: list[str] = []
    for term in terms:
        clean = str(term or "").strip()
        if not clean:
            continue
        if clean.lower() in lower or clean in text:
            matches.append(clean)
    return _dedupe(matches)


def _normalize_path_text(path: str | Path) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    try:
        candidate = Path(text)
        if candidate.is_absolute():
            text = candidate.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except Exception:
        pass
    return text.strip("/")


def _path_matches(path_text: str, prefixes: list[str]) -> list[str]:
    normalized = _normalize_path_text(path_text).lower()
    if not normalized:
        return []
    matches: list[str] = []
    for prefix in prefixes:
        raw_prefix = str(prefix or "").strip()
        clean = _normalize_path_text(raw_prefix).lower()
        if not clean:
            continue
        is_dir_prefix = raw_prefix.replace("\\", "/").endswith("/")
        if normalized == clean or (is_dir_prefix and normalized.startswith(clean.rstrip("/") + "/")):
            matches.append(raw_prefix)
    return _dedupe(matches)


def match_objects_for_task(task: str, *, paths: list[str] | None = None) -> list[ObjectMatch]:
    text = str(task or "").strip()
    path_values = paths or []
    matches: list[ObjectMatch] = []
    for item in registry_objects():
        object_id = str(item.get("object_id", "") or "").strip()
        owner = str(item.get("owner", "") or "").strip()
        object_type = str(item.get("type", "") or "").strip()
        routing_matches = _matches_text(text, _as_list(item.get("routing_terms")))
        write_matches = _matches_text(text, _as_list(item.get("write_terms")))
        path_matches: list[str] = []
        for path in path_values:
            path_matches.extend(_path_matches(path, _as_list(item.get("path_prefixes"))))
        path_matches = _dedupe(path_matches)
        if not object_id or not owner or not object_type:
            continue
        # Generic mutation verbs (update/cleanup/write) only set the mode after
        # object-specific routing or path evidence has selected the object.
        if not routing_matches and not path_matches:
            continue
        reason = str(item.get("write_reason" if write_matches or path_matches else "read_reason", "") or "")
        matches.append(
            ObjectMatch(
                object_id=object_id,
                owner=owner,
                type=object_type,
                matched_terms=routing_matches,
                matched_write_terms=write_matches,
                matched_paths=path_matches,
                route_reason=reason,
                default_mode=str(item.get("default_mode", "") or "read_only"),
                protected=bool(item.get("protected", False)),
            )
        )
    return matches


def match_objects_for_paths(paths: list[str]) -> list[ObjectMatch]:
    return match_objects_for_task("", paths=paths)


def owner_matched(task: str, owner: str) -> bool:
    return any(item.owner == owner for item in match_objects_for_task(task))


def owner_write_matched(task: str, owner: str) -> bool:
    return any(item.owner == owner and item.matched_write_terms for item in match_objects_for_task(task))


def object_matched(task: str, object_id: str) -> bool:
    return any(item.object_id == object_id for item in match_objects_for_task(task))


def object_write_matched(task: str, object_id: str) -> bool:
    return any(item.object_id == object_id and item.matched_write_terms for item in match_objects_for_task(task))


def registry_writeback_targets(object_ids: list[str]) -> list[str]:
    targets: list[str] = []
    wanted = set(object_ids)
    for item in registry_objects():
        if str(item.get("object_id", "") or "") in wanted:
            targets.extend(_as_list(item.get("writeback_targets")))
    return _dedupe(targets)


def registry_validation_commands(object_ids: list[str]) -> list[str]:
    commands: list[str] = []
    wanted = set(object_ids)
    for item in registry_objects():
        if str(item.get("object_id", "") or "") in wanted:
            commands.extend(_as_list(item.get("validation_commands")))
    return _dedupe(commands)


def changed_paths_from_git() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        body = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in body:
            body = body.split(" -> ", 1)[1].strip()
        if body:
            paths.append(body.replace("\\", "/"))
    return _dedupe(paths)
