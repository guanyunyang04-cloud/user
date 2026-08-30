"""Audit and migrate path references embedded in research study JSON files.

Study specifications outlived the ``daily_research`` and
``quant_data_platform`` directory names.  This utility keeps the JSON layout
and historical identifiers intact while making resolvable references point at
the current workspace.  References to deleted process artifacts are marked
with an explicit ``legacy://`` URI instead of being silently redirected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "quantlab.research_study_path_migration/1"
LEGACY_URI_PREFIX = "legacy://"
LEGACY_PREFIXES = ("daily_research/", "quant_data_platform/", "tmp/")

# These are namespace moves whose current counterpart is unambiguous.  A
# reference is only called ``workspace_resolved`` when the mapped target is
# present in the workspace; otherwise it remains explicit provenance.
CURRENT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("daily_research/studies/", "research/studies/"),
    ("daily_research/research_records/", "research/records/"),
    ("daily_research/models/", "research/models/"),
    ("quant_data_platform/data/qdp_v2/", "data/qdp/qdp_v2/"),
    ("quant_data_platform/data/qdp_runtime/", "data/qdp/qdp_runtime/"),
)


@dataclass(frozen=True)
class ReferenceChange:
    """One path-valued JSON field and its migration decision."""

    json_path: str
    original: str
    replacement: str
    status: str
    resolved_path: str | None


@dataclass(frozen=True)
class FilePlan:
    """Migration plan for one study specification."""

    path: Path
    original_sha256: str
    rewritten_text: str
    changes: tuple[ReferenceChange, ...]

    @property
    def changed(self) -> bool:
        return _sha256_bytes(self.rewritten_text.encode("utf-8")) != self.original_sha256


def _normalise(value: str) -> str:
    return value.replace("\\", "/")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_pointer(parts: Iterable[str]) -> str:
    encoded = []
    for part in parts:
        encoded.append(str(part).replace("~", "~0").replace("/", "~1"))
    return "/" + "/".join(encoded)


def _walk_strings(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, path + (str(index),))
    elif isinstance(value, str):
        yield _json_pointer(path), value


def _is_legacy_path(value: str) -> bool:
    value = _normalise(value)
    return value.startswith(LEGACY_PREFIXES)


def _is_output_destination(json_path: str) -> bool:
    """Return whether a legacy path is a new run destination, not an input."""

    parts = [part for part in json_path.split("/") if part]
    if not parts:
        return False
    # ``source.*`` and ``sources.*`` fields point at old evidence, even when
    # their leaf name contains ``output_root``.
    if parts[0] in {"source", "sources", "source_artifacts"}:
        return False
    leaf = parts[-1].lower()
    if leaf in {"output_root", "output_dir", "record_dir"}:
        return True
    return parts[0] == "outputs"


def _relative_if_inside(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _mapped_current(value: str, root: Path) -> tuple[str, Path] | None:
    value = _normalise(value)
    for old_prefix, new_prefix in CURRENT_PREFIXES:
        if value.startswith(old_prefix):
            candidate = root / (new_prefix + value[len(old_prefix) :])
            if candidate.exists():
                return new_prefix + value[len(old_prefix) :], candidate
            return None
    return None


def _mapped_output(value: str, json_path: str) -> str | None:
    value = _normalise(value)
    if not _is_output_destination(json_path):
        return None
    if value.startswith("daily_research/output/path_policy/studies/"):
        return "runs/studies/" + value[len("daily_research/output/path_policy/studies/") :]
    if value.startswith("daily_research/research_records/"):
        # Durable records are immutable evidence.  A migrated output target
        # must not point into that tree, even when the old path used to do so.
        return "runs/studies/records/" + value[len("daily_research/research_records/") :]
    if value.startswith("tmp/"):
        return "runs/tmp/" + value[len("tmp/") :]
    return None


def classify_reference(value: str, json_path: str, root: Path) -> ReferenceChange | None:
    """Classify one legacy path and return its canonical representation."""

    normalised = _normalise(value)
    if normalised.startswith(LEGACY_URI_PREFIX):
        return None
    if not _is_legacy_path(normalised):
        return None

    output = _mapped_output(normalised, json_path)
    if output is not None:
        return ReferenceChange(
            json_path=json_path,
            original=value,
            replacement=output,
            status="workspace_output",
            resolved_path=str(root / output),
        )

    mapped = _mapped_current(normalised, root)
    if mapped is not None:
        replacement, target = mapped
        return ReferenceChange(
            json_path=json_path,
            original=value,
            replacement=replacement,
            status="workspace_resolved",
            resolved_path=_relative_if_inside(target, root) or target.as_posix(),
        )

    # Keep the original path readable, but make its non-runnable status
    # explicit so a future consumer cannot mistake it for a current file.
    return ReferenceChange(
        json_path=json_path,
        original=value,
        replacement=LEGACY_URI_PREFIX + normalised,
        status="provenance_only",
        resolved_path=None,
    )


_JSON_STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def _rewrite_string_tokens(
    text: str,
    replacements: dict[str, list[str]],
) -> str:
    """Rewrite JSON string values without reformatting the surrounding file."""

    output: list[str] = []
    cursor = 0
    for match in _JSON_STRING.finditer(text):
        token = match.group(0)
        try:
            value = json.loads(token)
        except json.JSONDecodeError:
            continue
        # Object keys are not path fields.  Looking ahead is sufficient for
        # valid JSON and avoids changing a deliberately path-shaped key.
        tail = text[match.end() :]
        if tail.lstrip().startswith(":"):
            continue
        queue = replacements.get(value)
        if not queue:
            continue
        replacement = queue.pop(0)
        output.append(text[cursor : match.start()])
        output.append(json.dumps(replacement, ensure_ascii=False))
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def plan_file(path: Path, root: Path) -> FilePlan:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid study JSON: {path}: {exc}") from exc

    changes: list[ReferenceChange] = []
    replacements: dict[str, list[str]] = defaultdict(list)
    for json_path, value in _walk_strings(payload):
        change = classify_reference(value, json_path, root)
        if change is None:
            continue
        changes.append(change)
        replacements[value].append(change.replacement)

    rewritten = _rewrite_string_tokens(text, replacements)
    # Ensure every planned token was actually found and replaced.  This is a
    # guard against accidentally editing a duplicate value in the wrong field.
    leftovers = {key: values for key, values in replacements.items() if values}
    if leftovers:
        raise ValueError(f"could not locate JSON string token(s) in {path}: {sorted(leftovers)}")

    return FilePlan(
        path=path,
        original_sha256=_sha256_bytes(raw),
        rewritten_text=rewritten,
        changes=tuple(changes),
    )


def study_files(root: Path) -> tuple[Path, ...]:
    directory = root / "research" / "studies"
    return tuple(sorted(path for path in directory.glob("*.json") if path.name != "path_migration_manifest.json"))


def build_manifest(root: Path, plans: Iterable[FilePlan]) -> dict[str, Any]:
    plans = tuple(plans)
    references = [
        {
            "json_path": change.json_path,
            "original": change.original,
            "replacement": change.replacement,
            "status": change.status,
            "resolved_path": change.resolved_path,
        }
        for plan in plans
        for change in plan.changes
    ]
    counts = Counter(item["status"] for item in references)
    return {
        "schema": SCHEMA,
        "workspace_root": ".",
        "study_directory": "research/studies",
        "legacy_uri_prefix": LEGACY_URI_PREFIX,
        "summary": {
            "study_files": len(plans),
            "files_changed": sum(plan.changed for plan in plans),
            "references_migrated": len(references),
            "workspace_resolved": counts.get("workspace_resolved", 0),
            "workspace_output": counts.get("workspace_output", 0),
            "provenance_only": counts.get("provenance_only", 0),
        },
        "files": [
            {
                "path": _relative_if_inside(plan.path, root),
                "original_sha256": plan.original_sha256,
                "references": [
                    {
                        "json_path": change.json_path,
                        "original": change.original,
                        "replacement": change.replacement,
                        "status": change.status,
                        "resolved_path": change.resolved_path,
                    }
                    for change in plan.changes
                ],
            }
            for plan in plans
            if plan.changes
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("schema") == SCHEMA else None


def _portable_manifest(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    """Remove machine-specific absolute paths from an existing audit record."""

    portable = json.loads(json.dumps(payload, ensure_ascii=False))
    for file_entry in portable.get("files", []):
        if not isinstance(file_entry, dict):
            continue
        for reference in file_entry.get("references", []):
            if not isinstance(reference, dict):
                continue
            resolved = reference.get("resolved_path")
            if not isinstance(resolved, str) or not Path(resolved).is_absolute():
                continue
            relative = _relative_if_inside(Path(resolved), root)
            if relative is not None:
                reference["resolved_path"] = relative
    portable["workspace_root"] = "."
    return portable


def _merge_manifests(
    existing: dict[str, Any],
    current: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Append a later migration without discarding the first audit trail."""

    merged = _portable_manifest(existing, root)
    by_path: dict[str, dict[str, Any]] = {
        str(item.get("path")): item
        for item in merged.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    for item in current.get("files", []):
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = str(item["path"])
        if path not in by_path:
            by_path[path] = item
            continue
        target = by_path[path]
        known = {
            (ref.get("json_path"), ref.get("replacement"))
            for ref in target.get("references", [])
            if isinstance(ref, dict)
        }
        for ref in item.get("references", []):
            if not isinstance(ref, dict):
                continue
            key = (ref.get("json_path"), ref.get("replacement"))
            if key not in known:
                target.setdefault("references", []).append(ref)
                known.add(key)
    merged["files"] = [by_path[key] for key in sorted(by_path)]
    references = [
        ref
        for item in merged["files"]
        for ref in item.get("references", [])
        if isinstance(ref, dict)
    ]
    counts = Counter(ref.get("status") for ref in references)
    merged["summary"] = {
        "study_files": len(study_files(root)),
        "files_changed": len(merged["files"]),
        "references_migrated": len(references),
        "workspace_resolved": counts.get("workspace_resolved", 0),
        "workspace_output": counts.get("workspace_output", 0),
        "provenance_only": counts.get("provenance_only", 0),
    }
    return merged


def run(root: Path, *, apply: bool, manifest_path: Path) -> dict[str, Any]:
    plans = tuple(plan_file(path, root) for path in study_files(root))
    changed = any(plan.changed for plan in plans)
    existing = _load_manifest(manifest_path) if apply and manifest_path.is_file() else None
    if apply:
        for plan in plans:
            if plan.changed:
                _write_text_atomic(plan.path, plan.rewritten_text)
    manifest = build_manifest(root, plans)
    if apply:
        # Keep the original audit trail on an idempotent second invocation.
        # A migration manifest is evidence of the first rewrite, not a cache
        # that should be replaced by an empty report.
        if existing is not None:
            manifest = _merge_manifests(existing, manifest, root) if changed else _portable_manifest(existing, root)
        if changed or existing is None or manifest != existing:
            _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="write migrated study JSON files and the audit manifest",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="return 1 when any bare legacy path still needs migration",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default: <workspace>/research/path_migration_manifest.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.workspace_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
    else root / "research" / "path_migration_manifest.json"
    )
    try:
        manifest = run(root, apply=bool(args.apply), manifest_path=manifest_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    pending = int(manifest["summary"].get("references_migrated", 0)) > 0
    if args.check:
        print(f"check={'failed' if pending else 'ok'}")
        return 1 if pending else 0
    if not args.apply:
        print("dry_run=true")
        print("rerun with --apply to write migrated files")
    else:
        print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
