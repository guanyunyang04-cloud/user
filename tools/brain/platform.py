from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.brain.adapters import daily_research as daily_research_adapter
from tools.brain.project_profiles import load_project_profile


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAIN_MANIFEST = Path("brain/brain_manifest.json")
BRAIN_CATALOG = Path("brain/brain_catalog.json")
WORKFLOW_REGISTRY = Path("brain/workflows/registry.json")
SPLIT_WORKFLOW_REGISTRY = Path("brain/workflows/registry.json")
OUTPUT_ROOT = WORKSPACE_ROOT / "brain/output/brain_workflow"
PYTHON_EXECUTABLE = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SHARED_CONTRACT_KEY = "shared_regional_brain_contract"
LANGUAGE_POLICY_ID = "zh_semantic_en_identifiers_v1"
WORKSPACE_BOOTSTRAP_ALIASES = frozenset({"", "workspace", "workspace_root", "workspace_governance"})
OPTIONAL_BRAIN_KEYS = (
    "identity_path",
    "state_path",
    "knowledge_path",
    "operations_path",
    "governance_path",
    "operating_protocol_path",
)
MOJIBAKE_MARKERS = (
    "\ufffd",
    "\u93bf",
    "\u942d",
    "\u7ecb",
    "\u951b",
    "\u9286",
    "\u6b5a",
    "\u4e63",
    "\u6d93",
    "\u6d60",
)
ArtifactRecord = daily_research_adapter.ArtifactRecord
ArtifactFreshnessReport = daily_research_adapter.ArtifactFreshnessReport
StudyTrialEvidence = daily_research_adapter.StudyTrialEvidence
RunEvidenceReport = daily_research_adapter.RunEvidenceReport


@dataclass(frozen=True)
class EncodingReport:
    label: str
    is_utf8: bool
    has_replacement_char: bool
    suspicious_mojibake_count: int
    line_count: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrainBootstrapState:
    root: str
    main_manifest: str
    main_entrypoint: str
    main_boot_order: list[str]
    shared_module_order: list[str]
    shared_region_bindings: dict[str, Any]
    shared_fast_handoff_modules: list[str]
    boot_order: list[str]
    target_kind: str = "workspace"
    workflow_domain: str = "workspace_governance"
    child_brain: str = ""
    child_manifest: str = ""
    child_entrypoint: str = ""
    child_attach_status: str = ""
    child_regional_specialization: dict[str, Any] = field(default_factory=dict)
    child_fast_handoff_paths: list[str] = field(default_factory=list)
    child_boot_order: list[str] = field(default_factory=list)
    main_optional_paths: dict[str, str] = field(default_factory=dict)
    child_optional_paths: dict[str, str] = field(default_factory=dict)
    child_write_routes: dict[str, str] = field(default_factory=dict)
    artifact_freshness: dict[str, Any] = field(default_factory=dict)
    project_profile: dict[str, Any] = field(default_factory=dict)
    workflow_hints: dict[str, Any] = field(default_factory=dict)
    encoding_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in self.main_optional_paths.items():
            payload[f"main_{key}"] = value
        for key, value in self.child_optional_paths.items():
            payload[f"child_{key}"] = value
        return payload


@dataclass(frozen=True)
class BrainHealthReport:
    status: str
    checks: dict[str, dict[str, Any]]
    elapsed_seconds: float
    mode: str = "compact"
    brain: str = "workspace"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowState:
    workflow_id: str
    status: str
    read_only: bool
    writes_tracked_files: bool
    registry_entry: dict[str, Any]
    artifact_freshness: dict[str, Any]
    evidence_gaps: list[str]
    next_allowed_actions: list[str]
    resource_risk: str
    run_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def read_text(path: str | Path) -> str:
    return workspace_path(path).read_text(encoding="utf-8-sig")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _load_playbook(playbook_ref: str, workflow_id: str) -> dict[str, Any]:
    path_text, _, fragment = str(playbook_ref).partition("#")
    if not path_text:
        raise ValueError(f"workflow {workflow_id} has empty playbook path")
    payload = json.loads(read_text(Path(path_text)))
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow playbook must be a JSON object: {path_text}")
    if fragment:
        workflows = payload.get("workflows", payload)
        if not isinstance(workflows, dict) or fragment not in workflows:
            raise KeyError(f"Workflow playbook fragment not found: {playbook_ref}")
        entry = workflows[fragment]
    else:
        entry = payload
    if not isinstance(entry, dict):
        raise ValueError(f"Workflow playbook entry must be an object: {playbook_ref}")
    out = dict(entry)
    out.setdefault("artifacts", [])
    out.setdefault("capability_hints", [])
    out.setdefault("risk_signals", [])
    out.setdefault("verification_hints", [])
    out.setdefault("stop_conditions", [])
    return out


def _load_split_workflow_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError("split workflow registry must be a JSON object")
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        raise ValueError("split workflow registry must declare workflows as a list")
    out: dict[str, dict[str, Any]] = {}
    for item in workflows:
        if not isinstance(item, dict):
            continue
        workflow_id = str(item.get("id", "")).strip()
        playbook = str(item.get("playbook", "")).strip()
        if not workflow_id or not playbook:
            continue
        entry = _load_playbook(playbook, workflow_id)
        entry.setdefault("title", str(item.get("title", "") or workflow_id))
        entry.setdefault("category", str(item.get("category", "") or "general"))
        entry.setdefault("playbook", playbook)
        out[workflow_id] = entry
    return out


def _child_workflow_registry_path(child_brain: str | None) -> Path | None:
    if not child_brain:
        return None
    try:
        main_manifest = load_manifest(MAIN_MANIFEST)
        child_ref = _resolve_child(main_manifest, child_brain)
    except Exception:
        return None
    manifest_path = Path(str(child_ref.get("path", "") or ""))
    if not manifest_path.as_posix():
        return None
    return manifest_path.parent / "workflows" / "registry.json"


def load_workflow_registry(child_brain: str | None = None) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    if workspace_path(SPLIT_WORKFLOW_REGISTRY).exists():
        registry.update(_load_split_workflow_registry(SPLIT_WORKFLOW_REGISTRY))

    child_registry = _child_workflow_registry_path(child_brain)
    if child_registry is not None and workspace_path(child_registry).exists():
        registry.update(_load_split_workflow_registry(child_registry))

    if registry:
        return registry

    payload = json.loads(read_text(WORKFLOW_REGISTRY))
    if not isinstance(payload, dict):
        raise ValueError("workflow_registry.json must be a JSON object")
    redirect = str(payload.get("redirect", "") or "").strip()
    if redirect:
        return _load_split_workflow_registry(Path(redirect))
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


def _catalog_record(
    *,
    brain_id: str,
    root: Path,
    manifest_path: Path | str,
    status: str,
    body_root: Path | str,
) -> dict[str, Any]:
    root_text = root.as_posix()
    references = root / "references"
    return {
        "brain_id": brain_id,
        "root": root_text,
        "manifest_path": Path(str(manifest_path)).as_posix() if str(manifest_path).strip() else "",
        "status": status,
        "body_root": Path(str(body_root)).as_posix() if str(body_root).strip() else "",
        "references_path": references.as_posix() if workspace_path(references).exists() else "",
        "language_policy": LANGUAGE_POLICY_ID,
    }


def _discover_brain_dirs() -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("*/brain", "*/*/brain"):
        for path in sorted(WORKSPACE_ROOT.glob(pattern)):
            if not path.is_dir():
                continue
            rel = path.relative_to(WORKSPACE_ROOT)
            if any(part in {"output", "archive", "node_modules", ".git", ".pytest_cache"} for part in rel.parts):
                continue
            candidates.append(Path(rel.as_posix()))
    return _dedupe(candidates)


def build_brain_catalog() -> dict[str, Any]:
    main_manifest = load_manifest(MAIN_MANIFEST)
    registered_catalog: dict[str, dict[str, Any]] = {}
    if workspace_path(BRAIN_CATALOG).exists():
        try:
            file_catalog = json.loads(read_text(BRAIN_CATALOG))
            if isinstance(file_catalog, dict):
                registered_catalog = {
                    str(item.get("brain_id", "")): item
                    for item in file_catalog.get("brains", [])
                    if isinstance(item, dict) and str(item.get("brain_id", ""))
                }
        except Exception:
            registered_catalog = {}
    records: list[dict[str, Any]] = [
        _catalog_record(
            brain_id="workspace_root",
            root=Path("brain"),
            manifest_path=MAIN_MANIFEST,
            status="canonical_root",
            body_root=Path("."),
        )
    ]
    canonical_roots = {Path("brain").as_posix()}

    for child in main_manifest.get("child_brains", []):
        if not isinstance(child, dict):
            continue
        child_id = str(child.get("id", "")).strip()
        child_path = Path(str(child.get("path", "")).strip())
        root = child_path.parent if child_path.as_posix() else Path(str(child.get("body_root", ""))) / "brain"
        if not child_id:
            continue
        canonical_roots.add(root.as_posix())
        records.append(
            _catalog_record(
                brain_id=child_id,
                root=root,
                manifest_path=child_path,
                status="attached" if str(child.get("attach_status", "")).startswith("attached") else "discovered_untracked",
                body_root=Path(str(child.get("body_root", "") or root.parent.as_posix())),
            )
        )

    acknowledged_non_truth_statuses = {
        "cache_legacy",
        "non_truth_tooling",
        "external_or_inactive_missing_manifest",
    }
    action_needed_noncanonical_statuses = {"discovered_untracked", "missing_manifest"}
    noncanonical_statuses = acknowledged_non_truth_statuses | action_needed_noncanonical_statuses

    for root in _discover_brain_dirs():
        if root.as_posix() in canonical_roots:
            continue
        manifest_path = root / "brain_manifest.json"
        if root.as_posix() == "daily_research/cache/brain":
            brain_id = "daily_research_cache_legacy"
            status = "cache_legacy"
            manifest_ref: Path | str = manifest_path if workspace_path(manifest_path).exists() else ""
        elif root.as_posix() == "tools/brain":
            brain_id = "tools"
            status = "non_truth_tooling"
            manifest_ref = ""
        else:
            brain_id = root.parent.as_posix().replace("/", "_").replace("-", "_")
            status = "discovered_untracked" if workspace_path(manifest_path).exists() else "missing_manifest"
            manifest_ref = manifest_path if workspace_path(manifest_path).exists() else ""
        records.append(
            _catalog_record(
                brain_id=brain_id,
                root=root,
                manifest_path=manifest_ref,
                status=status,
                body_root=root.parent,
            )
        )

    current_ids = {str(item.get("brain_id", "")) for item in records}
    for brain_id, record in registered_catalog.items():
        status = str(record.get("status", "") or "")
        if brain_id in current_ids or status not in noncanonical_statuses:
            continue
        root = Path(str(record.get("root", "") or ""))
        if not root.as_posix():
            continue
        records.append(
            _catalog_record(
                brain_id=brain_id,
                root=root,
                manifest_path=str(record.get("manifest_path", "") or ""),
                status=status,
                body_root=Path(str(record.get("body_root", "") or root.parent.as_posix())),
            )
        )

    return {
        "schema_version": 2,
        "language_policy": LANGUAGE_POLICY_ID,
        "brains": records,
    }


def child_brain_ids() -> list[str]:
    main_manifest = load_manifest(MAIN_MANIFEST)
    out: list[str] = []
    for child in main_manifest.get("child_brains", []):
        if isinstance(child, dict) and str(child.get("id", "")).strip():
            out.append(str(child["id"]).strip())
    return out


def resolve_workflow_child_brain(workflow_id: str) -> str | None:
    normalized = "continuous_policy_result_review" if workflow_id == "continuous_policy" else str(workflow_id or "").strip()
    if not normalized:
        return None
    main_registry = load_workflow_registry()
    if normalized in main_registry:
        return None
    for child_id in child_brain_ids():
        child_registry_path = _child_workflow_registry_path(child_id)
        if child_registry_path is None or not workspace_path(child_registry_path).exists():
            continue
        if normalized in _load_split_workflow_registry(child_registry_path):
            return child_id
    return None


def check_text_encoding_text(text: str, *, label: str) -> EncodingReport:
    return EncodingReport(
        label=label,
        is_utf8=True,
        has_replacement_char="\ufffd" in text,
        suspicious_mojibake_count=sum(text.count(marker) for marker in MOJIBAKE_MARKERS),
        line_count=len(text.splitlines()),
    )


def check_text_encoding(path: str | Path) -> EncodingReport:
    rel = Path(path)
    label = rel.as_posix()
    try:
        raw = workspace_path(rel).read_bytes()
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return EncodingReport(label=label, is_utf8=False, has_replacement_char=False, suspicious_mojibake_count=0, line_count=0, error=str(exc))
    return check_text_encoding_text(text, label=label)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        normalized = path.as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(path)
    return out


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _split_contract_source(source: str) -> tuple[Path, str]:
    path_text, _, key = str(source).partition("#")
    return Path(path_text), key


def _shared_contract(main_manifest: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    contract = main_manifest.get(SHARED_CONTRACT_KEY)
    if not isinstance(contract, dict):
        raise ValueError(f"Main manifest must declare {SHARED_CONTRACT_KEY}")
    if source:
        path, key = _split_contract_source(source)
        if path.as_posix() != MAIN_MANIFEST.as_posix() or key != SHARED_CONTRACT_KEY:
            raise ValueError(f"Unsupported shared_contract_source: {source}")
    return contract


def _resolve_child(main_manifest: dict[str, Any], child_id: str) -> dict[str, Any]:
    for child in main_manifest.get("child_brains", []):
        if isinstance(child, dict) and (child.get("id") == child_id or child.get("body_root") == child_id):
            return child
    raise KeyError(f"Unknown child brain: {child_id}")


def normalize_bootstrap_target(target_id: str | None) -> str | None:
    text = str(target_id or "").strip()
    return None if text in WORKSPACE_BOOTSTRAP_ALIASES else text


def target_kind_for_bootstrap(target_id: str | None) -> str:
    return "workspace" if normalize_bootstrap_target(target_id) is None else "child"


def workflow_domain_for_bootstrap(target_id: str | None) -> str:
    normalized = normalize_bootstrap_target(target_id)
    return "workspace_governance" if normalized is None else normalized


def _module_path_map(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for module in manifest.get("modules", []):
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("id", "")).strip()
        paths = module.get("paths")
        if module_id and isinstance(paths, list):
            out[module_id] = [Path(str(path)) for path in paths if str(path).strip()]
    return out


def _parse_module_order_item(raw: Any) -> tuple[str, bool]:
    item = str(raw).strip()
    optional = item.endswith("?")
    return (item[:-1] if optional else item), optional


def _derive_child_read_order(main_manifest: dict[str, Any], child_manifest: dict[str, Any]) -> list[Path]:
    contract = _shared_contract(main_manifest, str(child_manifest.get("shared_contract_source", "")).strip() or None)
    module_map = _module_path_map(child_manifest)
    ordered: list[Path] = []
    seen: set[str] = set()
    for raw in contract.get("default_module_order", []):
        module_id, optional = _parse_module_order_item(raw)
        paths = module_map.get(module_id, [])
        if not paths and optional:
            continue
        if not paths:
            raise FileNotFoundError(f"Child brain missing required module: {module_id}")
        for path in paths:
            if path.as_posix() not in seen:
                seen.add(path.as_posix())
                ordered.append(path)
    for module in child_manifest.get("modules", []):
        if not isinstance(module, dict):
            continue
        for raw_path in module.get("paths", []):
            path = Path(str(raw_path))
            normalized = path.as_posix()
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(path)
    return _dedupe(ordered)


def _build_main_order(main_manifest: dict[str, Any]) -> list[Path]:
    return _dedupe([MAIN_MANIFEST, *[Path(str(item)) for item in main_manifest.get("read_order", [])]])


def _build_child_order(main_manifest: dict[str, Any], child_manifest_path: Path, child_manifest: dict[str, Any]) -> list[Path]:
    handoff = child_manifest.get("handoff_contract", {})
    entry_sequence = handoff.get("entry_sequence") if isinstance(handoff, dict) else None
    if isinstance(entry_sequence, list) and entry_sequence:
        return _dedupe([Path(str(item)) for item in entry_sequence])
    read_order = child_manifest.get("read_order")
    if isinstance(read_order, list) and read_order:
        return _dedupe([child_manifest_path, *[Path(str(item)) for item in read_order]])
    return _dedupe([child_manifest_path, *_derive_child_read_order(main_manifest, child_manifest)])


def _collect_optional_paths(manifest: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in OPTIONAL_BRAIN_KEYS:
        value = str(manifest.get(key, "") or "").strip()
        if value:
            out[key] = value
    return out


def _collect_fast_handoff_paths(main_manifest: dict[str, Any], child_manifest: dict[str, Any]) -> list[str]:
    contract = _shared_contract(main_manifest, str(child_manifest.get("shared_contract_source", "")).strip() or None)
    module_map = _module_path_map(child_manifest)
    out: list[str] = []
    for module_id in contract.get("fast_handoff_modules", []):
        for path in module_map.get(str(module_id), []):
            normalized = path.as_posix()
            if normalized not in out:
                out.append(normalized)
    return out


def _encoding_summary(paths: list[str]) -> dict[str, Any]:
    reports = {path: check_text_encoding(path).to_dict() for path in paths}
    return {
        "reports": reports,
        "has_errors": any((not item["is_utf8"]) or item["has_replacement_char"] or item["suspicious_mojibake_count"] > 0 for item in reports.values()),
    }


def resolve_bootstrap(child_id: str | None = None) -> BrainBootstrapState:
    child_id = normalize_bootstrap_target(child_id)
    main_manifest = load_manifest(MAIN_MANIFEST)
    shared_contract = _shared_contract(main_manifest)
    main_order = _build_main_order(main_manifest)
    main_paths = [path.as_posix() for path in main_order]
    project_profile = load_project_profile(child_id or "workspace")
    guard_profile = project_profile.get("guard_profile", {}) if isinstance(project_profile.get("guard_profile"), dict) else {}
    artifact_freshness = resolve_artifact_freshness().to_dict() if bool(guard_profile.get("artifact_freshness")) else {}
    registry = load_workflow_registry(child_id)

    if child_id is None:
        return BrainBootstrapState(
            root=WORKSPACE_ROOT.as_posix(),
            main_manifest=MAIN_MANIFEST.as_posix(),
            main_entrypoint=str(main_manifest.get("entrypoint", "")),
            main_boot_order=main_paths,
            shared_module_order=list(shared_contract.get("default_module_order", [])),
            shared_region_bindings=dict(shared_contract.get("region_bindings", {})),
            shared_fast_handoff_modules=list(shared_contract.get("fast_handoff_modules", [])),
            boot_order=main_paths,
            target_kind="workspace",
            workflow_domain="workspace_governance",
            main_optional_paths=_collect_optional_paths(main_manifest),
            artifact_freshness=artifact_freshness,
            project_profile=project_profile,
            workflow_hints={"available_workflows": sorted(registry)},
            encoding_report=_encoding_summary(main_paths),
        )

    child_ref = _resolve_child(main_manifest, child_id)
    child_manifest_path = Path(str(child_ref["path"]))
    child_manifest = load_manifest(child_manifest_path)
    child_order = _build_child_order(main_manifest, child_manifest_path, child_manifest)
    child_paths = [path.as_posix() for path in child_order]
    boot_order = [path.as_posix() for path in _dedupe([*main_order, *child_order])]
    return BrainBootstrapState(
        root=WORKSPACE_ROOT.as_posix(),
        main_manifest=MAIN_MANIFEST.as_posix(),
        main_entrypoint=str(main_manifest.get("entrypoint", "")),
        main_boot_order=main_paths,
        shared_module_order=list(shared_contract.get("default_module_order", [])),
        shared_region_bindings=dict(shared_contract.get("region_bindings", {})),
        shared_fast_handoff_modules=list(shared_contract.get("fast_handoff_modules", [])),
        boot_order=boot_order,
        target_kind="child",
        workflow_domain=str(child_ref["id"]),
        child_brain=str(child_ref["id"]),
        child_manifest=child_manifest_path.as_posix(),
        child_entrypoint=str(child_manifest.get("entrypoint", "")),
        child_attach_status=str(child_manifest.get("attach_status", "")),
        child_regional_specialization=dict(child_manifest.get("regional_specialization", {})),
        child_fast_handoff_paths=_collect_fast_handoff_paths(main_manifest, child_manifest),
        child_boot_order=child_paths,
        main_optional_paths=_collect_optional_paths(main_manifest),
        child_optional_paths=_collect_optional_paths(child_manifest),
        child_write_routes=dict(child_manifest.get("write_routes", {}) or {}),
        artifact_freshness=artifact_freshness,
        project_profile=project_profile,
        workflow_hints={"available_workflows": sorted(registry), "recommended_default": "brain_handoff"},
        encoding_report=_encoding_summary(boot_order),
    )


def resolve_artifact_freshness() -> ArtifactFreshnessReport:
    return daily_research_adapter.resolve_artifact_freshness()


def _path_line_count(path: str | Path) -> int:
    target = workspace_path(path)
    if not target.exists() or not target.is_file():
        return 0
    return len(target.read_text(encoding="utf-8-sig").splitlines())


def _brain_core_doc_stats(root: str) -> dict[str, Any]:
    core_names = (
        "identity_layer.md",
        "state_center.md",
        "knowledge_center.md",
        "brain_architecture.md",
        "operations_center.md",
        "governance_layer.md",
        "episodic_memory.md",
        "brain_manifest.json",
    )
    stats: dict[str, Any] = {}
    for name in core_names:
        rel = Path(root) / name
        target = workspace_path(rel)
        if target.exists() and target.is_file():
            stats[rel.as_posix()] = {
                "line_count": _path_line_count(rel),
                "encoding": check_text_encoding(rel).to_dict(),
            }
    return stats


def _active_artifact_diff_status() -> dict[str, Any]:
    return daily_research_adapter.active_artifact_diff_status()


def audit_brain_system(*, scope: str = "all") -> dict[str, Any]:
    catalog = build_brain_catalog()
    workflow_entries = load_workflow_registry()
    freshness = resolve_artifact_freshness()
    active_guard = _active_artifact_diff_status()

    brain_stats = {
        item["brain_id"]: {
            "status": item["status"],
            "root": item["root"],
            "core_docs": _brain_core_doc_stats(str(item["root"])),
            "references_count": item.get("references_count", 0),
            "manifest_present": bool(item.get("manifest_path")),
        }
        for item in catalog["brains"]
    }
    acknowledged_non_truth_statuses = {
        "cache_legacy",
        "non_truth_tooling",
        "external_or_inactive_missing_manifest",
    }
    action_needed_noncanonical_statuses = {"discovered_untracked", "missing_manifest"}
    acknowledged_non_truth = [
        item
        for item in catalog["brains"]
        if item.get("status") in acknowledged_non_truth_statuses
    ]
    actionable_noncanonical = [
        item
        for item in catalog["brains"]
        if item.get("status") in action_needed_noncanonical_statuses
    ]
    split_exists = workspace_path(SPLIT_WORKFLOW_REGISTRY).exists()
    legacy_exists = workspace_path(WORKFLOW_REGISTRY).exists()
    legacy_line_count = _path_line_count(WORKFLOW_REGISTRY) if legacy_exists else 0
    split_line_count = _path_line_count(SPLIT_WORKFLOW_REGISTRY) if split_exists else 0
    workflow_registry = {
        "mode": "split" if split_exists else "legacy",
        "legacy_path": WORKFLOW_REGISTRY.as_posix(),
        "legacy_line_count": legacy_line_count,
        "split_path": SPLIT_WORKFLOW_REGISTRY.as_posix() if split_exists else "",
        "split_line_count": split_line_count,
        "workflow_count": len(workflow_entries),
        "workflow_ids": sorted(workflow_entries),
    }
    language = {
        "policy": LANGUAGE_POLICY_ID,
        "policy_path": "brain/language_policy.md",
        "user_facing_language": "zh_cn",
        "engineering_identifiers": "en",
    }

    warnings: list[str] = []
    if actionable_noncanonical:
        warnings.append("noncanonical_brains_discovered")
    if freshness.is_stale_risk:
        warnings.append("loose_latest_stale_requires_explicit_tag")
    if legacy_line_count > 220 and not split_exists:
        warnings.append("workflow_registry_line_count_warning")
    if active_guard["status"] != "clean":
        warnings.append("active_execution_artifact_dirty")

    if active_guard["status"] != "clean":
        verdict = "blocked"
    elif legacy_line_count > 320 and not split_exists:
        verdict = "needs_refactor"
    elif warnings:
        verdict = "usable_with_warnings"
    else:
        verdict = "clean"

    return {
        "schema_version": 1,
        "scope": scope,
        "verdict": verdict,
        "warnings": warnings,
        "catalog": catalog,
        "attached_brains": [item for item in catalog["brains"] if item.get("status") in {"canonical_root", "attached"}],
        "discovered_noncanonical_brains": actionable_noncanonical,
        "acknowledged_non_truth_brains": acknowledged_non_truth,
        "brain_stats": brain_stats,
        "language": language,
        "workflow_registry": workflow_registry,
        "loose_latest": freshness.to_dict(),
        "active_artifact_guard": active_guard,
    }


def resolve_run_evidence(run_tag: str, workflow: str | None = None) -> RunEvidenceReport:
    return daily_research_adapter.resolve_run_evidence(run_tag, workflow=workflow)


def _openmp_strict_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_check(name: str, command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    elapsed = time.perf_counter() - started

    def excerpt(text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        half = max(int(limit / 2), 1)
        return f"{text[:half]}\n...<truncated>...\n{text[-half:]}"

    return {
        "name": name,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout_tail": excerpt(result.stdout or "", limit=4000),
        "stderr_tail": excerpt(result.stderr or "", limit=2000),
        "elapsed_seconds": round(elapsed, 3),
    }


def _project_brain_doc_guard_files(profile: dict[str, Any]) -> list[str]:
    body_root = str(profile.get("body_root", "") or "").strip().replace("\\", "/")
    if not body_root or body_root == ".":
        return []
    candidates = [
        f"{body_root}/brain/identity_layer.md",
        f"{body_root}/brain/state_center.md",
        f"{body_root}/brain/knowledge_center.md",
        f"{body_root}/brain/brain_architecture.md",
        f"{body_root}/brain/operations_center.md",
        f"{body_root}/brain/governance_layer.md",
        f"{body_root}/brain/brain_manifest.json",
    ]
    return [path for path in candidates if (WORKSPACE_ROOT / path).exists()]


def check_brain_health(
    brain: str | None = None,
    *,
    mode: str = "compact",
    include_doc_guard: bool = False,
    include_project_consistency: bool = False,
    include_openmp_strict: bool = False,
) -> BrainHealthReport:
    started = time.perf_counter()
    normalized_mode = str(mode or "compact").strip().lower()
    if normalized_mode not in {"compact", "standard", "full"}:
        normalized_mode = "compact"
    brain_id = brain or "workspace"
    profile = load_project_profile(brain_id)
    guard_profile = profile.get("guard_profile", {}) if isinstance(profile.get("guard_profile"), dict) else {}
    check_commands = {
        "brain_integrity": {
            "command": [PYTHON_EXECUTABLE, "-m", "tools.brain.integrity_check", "--json"],
            "env": None,
        },
    }
    should_run_doc_guard = normalized_mode in {"standard", "full"} or include_doc_guard
    if should_run_doc_guard:
        doc_guard_command = [PYTHON_EXECUTABLE, "-m", "tools.brain.doc_guard", "check"]
        if normalized_mode != "full":
            project_files = _project_brain_doc_guard_files(profile)
            if project_files:
                doc_guard_command.extend(["--files", *project_files])
            else:
                doc_guard_command.extend(["--scope", "changed"])
        check_commands["doc_guard"] = {
            "command": doc_guard_command,
            "env": None,
        }
    should_run_project_consistency = bool(guard_profile.get("project_consistency")) and (
        normalized_mode == "full" or include_project_consistency
    )
    if should_run_project_consistency:
        project_mode = "full" if normalized_mode == "full" else "research"
        check_commands["project_consistency"] = {
            "command": [PYTHON_EXECUTABLE, "daily_research/tools/project_consistency_check.py", "--mode", project_mode],
            "env": None,
        }
    should_run_openmp_strict = bool(guard_profile.get("openmp_strict")) and (
        normalized_mode == "full" or include_openmp_strict
    )
    if should_run_openmp_strict:
        check_commands["openmp_strict"] = {
            "command": [PYTHON_EXECUTABLE, "daily_research/tools/openmp_runtime_check.py", "--mode", "strict", "--strict"],
            "env": _openmp_strict_env(),
        }
    checks: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(check_commands)) as executor:
        futures = {
            executor.submit(_run_check, name, spec["command"], env=spec["env"]): name
            for name, spec in check_commands.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                checks[name] = future.result()
            except Exception as exc:  # defensive: health should report every lane, not crash on one lane
                checks[name] = {
                    "name": name,
                    "returncode": -1,
                    "ok": False,
                    "stdout_tail": "",
                    "stderr_tail": str(exc),
                    "elapsed_seconds": 0.0,
                }
    checks = {name: checks[name] for name in check_commands}
    status = "ok" if all(item["ok"] for item in checks.values()) else "failed"
    return BrainHealthReport(
        status=status,
        checks=checks,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        mode=normalized_mode,
        brain=brain_id,
    )


def _continuous_policy_evidence_gaps(freshness: ArtifactFreshnessReport) -> list[str]:
    return daily_research_adapter.artifact_freshness_evidence_gaps(freshness)


def _profile_validation_commands(child_brain: str | None) -> list[str]:
    profile = load_project_profile(child_brain or "workspace")
    verification_profile = profile.get("verification_profile", {}) if isinstance(profile.get("verification_profile"), dict) else {}
    return [str(item) for item in verification_profile.get("always_commands", []) or []]


def _expand_profile_validation_commands(commands: list[Any], *, child_brain: str | None) -> list[str]:
    out: list[str] = []
    for command in commands:
        text = str(command or "").strip()
        if not text:
            continue
        if text == "profile:verification_profile.always_commands":
            out.extend(_profile_validation_commands(child_brain))
        else:
            out.append(text)
    return _dedupe_text(out)


def build_workflow_state(workflow_id: str, *, run_tag: str | None = None, child_brain: str | None = None) -> WorkflowState:
    registry = load_workflow_registry(child_brain)
    normalized = "continuous_policy_result_review" if workflow_id == "continuous_policy" else workflow_id
    if normalized not in registry:
        raise KeyError(f"Unknown workflow: {workflow_id}")
    profile = load_project_profile(child_brain or "workspace")
    guard_profile = profile.get("guard_profile", {}) if isinstance(profile.get("guard_profile"), dict) else {}
    freshness = resolve_artifact_freshness() if bool(guard_profile.get("artifact_freshness")) else None
    run_evidence: dict[str, Any] = {}
    if run_tag and normalized.startswith("continuous_policy"):
        evidence = resolve_run_evidence(run_tag, workflow="continuous_policy")
        run_evidence = evidence.to_dict()
        gaps = list(evidence.evidence_gaps)
    else:
        gaps = _continuous_policy_evidence_gaps(freshness) if freshness is not None and normalized.startswith("continuous_policy") else []
    next_allowed = list(registry[normalized].get("allowed_commands", []) or [])
    is_stale_risk = bool(freshness.is_stale_risk) if freshness is not None else False
    resource_risk = "safe" if "safe_screening" in normalized else ("stale_latest_risk" if is_stale_risk else "normal")
    return WorkflowState(
        workflow_id=normalized,
        status="blocked" if gaps and normalized == "continuous_policy_result_review" else "ready",
        read_only=True,
        writes_tracked_files=False,
        registry_entry=registry[normalized],
        artifact_freshness=freshness.to_dict() if freshness is not None else {},
        evidence_gaps=gaps,
        next_allowed_actions=next_allowed,
        resource_risk=("normal_with_stale_latest" if run_evidence and is_stale_risk else resource_risk),
        run_evidence=run_evidence,
    )


def build_workflow_guide(workflow_id: str, *, child_brain: str | None = None) -> dict[str, Any]:
    registry = load_workflow_registry(child_brain)
    normalized = "continuous_policy_result_review" if workflow_id == "continuous_policy" else str(workflow_id or "").strip()
    if normalized not in registry:
        raise KeyError(f"Unknown workflow: {workflow_id}")
    entry = dict(registry[normalized])
    completion_review_workflows = {
        "brain_maintenance",
        "brain_architecture_refactor",
        "brain_writeback_verified",
    }
    return {
        "workflow_id": normalized,
        "description": str(entry.get("description", "") or ""),
        "workflow_mode": "brain_native_contract",
        "skill_boundary": "Brain workflow owns this project-specific contract.",
        "completion_review_required": normalized in completion_review_workflows,
        "preflight": list(entry.get("preflight", []) or []),
        "checklist": list(entry.get("checklist", []) or []),
        "allowed_commands": list(entry.get("allowed_commands", []) or []),
        "capability_hints": list(entry.get("capability_hints", []) or []),
        "risk_signals": list(entry.get("risk_signals", []) or []),
        "verification_hints": list(entry.get("verification_hints", []) or []),
        "stop_conditions": list(entry.get("stop_conditions", []) or []),
        "completion": list(entry.get("completion", []) or []),
        "validation_commands": _expand_profile_validation_commands(
            list(entry.get("validation_commands", []) or []),
            child_brain=child_brain,
        ),
        "writeback_routes": dict(entry.get("writeback_routes", {}) or {}),
    }


def select_workflow_for_task(task: str, *, intent: str = "read") -> dict[str, Any]:
    text = str(task or "").strip()
    lower = text.lower()
    normalized_intent = str(intent or "read").strip().lower() or "read"

    matched_terms: list[str] = []

    def has_any(*needles: str, source: str = "term_match") -> bool:
        found = [needle for needle in needles if needle in lower or needle in text]
        if found:
            matched_terms.extend(found)
            if source not in decision_sources:
                decision_sources.append(source)
            return True
        return False

    decision_sources: list[str] = []
    ambiguous_signals: list[str] = []
    intent_override_applied = False

    method_terms = (
        "继续实施",
        "实施计划",
        "implement",
        "execute",
        "执行计划",
        "please implement",
        "please implement this plan",
        "按计划实现",
        "开始落地",
        "全部完成",
        "完成这个计划",
        "详细计划",
        "计划",
        "方案",
        "设计",
        "深入思考",
        "深入分析",
        "评估",
        "方案设计",
        "不要修改代码",
        "只做计划",
        "plan",
        "design",
        "报错",
        "失败",
        "bug",
        "error",
        "oom",
        "cuda",
        "异常",
        "中断",
        "debug",
        "修复",
        "是否全部完成",
        "完成了吗",
        "是否完成",
        "是否通过",
        "检查一下",
        "verify",
        "verification",
        "完成没",
        "长训练",
        "长任务",
        "long task",
        "训练轮询",
        "估算剩余时间",
        "eta",
        "wait-process",
    )

    has_any(*method_terms, source="external_skill_signal")

    architecture_terms = (
        "强重构",
        "脑区迁移",
        "结构重做",
        "brain refactor",
        "architecture refactor",
    )
    audit_terms = (
        "脑区乱",
        "乱不乱",
        "复杂不复杂",
        "有没有错",
        "全中文",
        "需要优化",
        "脑区检查",
        "brain audit",
    )
    maintenance_terms = (
        "脑区规则",
        "规则修改",
        "workspace-brain",
        "本机 skill",
        "global skill",
        "skill_install",
        "skill sync",
        "workflow registry",
        "playbook",
        "capsule",
        "brain selector",
        "selector 漏判",
        "workflow selector",
    )
    writeback_terms = (
        "更新脑区",
        "写回",
        "writeback",
        "writeback verified",
        "证据登记",
        "登记证据",
        "evidence registry",
        "evidence-index",
        "evidence index",
        "rebuild evidence",
        "主线审阅",
        "审阅文档",
        "入库",
    )

    architecture_match = has_any(*architecture_terms)
    writeback_match = False if architecture_match else has_any(*writeback_terms)
    audit_match = False if architecture_match or normalized_intent == "writeback" or writeback_match else has_any(*audit_terms)
    maintenance_match = (
        False
        if architecture_match or normalized_intent == "writeback" or writeback_match or audit_match
        else has_any("脑区维护", "主脑维护", "brain maintenance", "health", *maintenance_terms)
    )

    if architecture_match:
        selected = "brain_architecture_refactor"
        reason = "task asks for a brain architecture refactor or migration"
    elif normalized_intent == "writeback" or writeback_match:
        selected = "brain_writeback_verified"
        reason = "task asks for brain writeback, evidence registry, or mainline review updates"
    elif audit_match:
        selected = "brain_system_audit"
        reason = "task asks to audit brain structure, complexity, language, or optimization need"
    elif maintenance_match:
        selected = "brain_maintenance"
        reason = "task asks to maintain or health-check the workspace brain"
    else:
        selected = "brain_handoff"
        reason = "task method is left to local skills; brain only supplies project handoff and guards"
    return {
        "task": text,
        "intent": normalized_intent,
        "selected_workflow": selected,
        "reason": reason,
        "matched_terms": sorted(set(matched_terms)),
        "decision_sources": sorted(set(decision_sources)),
        "confidence": "high" if not ambiguous_signals else "medium",
        "intent_override_applied": intent_override_applied,
        "ambiguous_signals": ambiguous_signals,
    }


def _routes_from_writeback_targets(targets: list[str]) -> dict[str, str]:
    routes: dict[str, str] = {}
    for target in targets:
        text = str(target or "").strip().replace("\\", "/")
        if not text:
            continue
        if text == "brain/state_center.md":
            routes["workspace_state"] = text
        elif text == "brain/references/":
            routes["workspace_references"] = text
        elif text == "daily_research/brain/state_center.md":
            routes["daily_research_state"] = text
        elif text == "daily_research/brain/references/":
            routes["daily_research_references"] = text
        elif text == "quant_data_platform/brain/state_center.md":
            routes["quant_data_platform_state"] = text
        elif text == "quant_data_platform/brain/references/":
            routes["quant_data_platform_references"] = text
        else:
            key = text.strip("/").replace("/", "_").replace(".", "_").replace("-", "_")
            routes[key] = text
    return routes


def build_writeback_plan(source: str, *, task: str | None = None, apply_brain_writeback: bool = False) -> dict[str, Any]:
    registry = load_workflow_registry()
    routes = dict(registry["brain_writeback"].get("writeback_routes", {}) or {})
    task_text = str(task or "").strip()
    routing: dict[str, Any] = {}
    if task_text:
        from tools.brain.routing import route_task_to_brain

        routing = route_task_to_brain(task_text)
        task_routes = _routes_from_writeback_targets(list(routing.get("writeback_targets", []) or []))
        if task_routes:
            routes = task_routes
    freshness = resolve_artifact_freshness().to_dict()
    source_text = str(source or "latest").strip() or "latest"
    run_evidence: dict[str, Any] = {}
    if source_text.startswith("run:"):
        run_source = source_text.split(":", 1)[1]
        workflow = None
        run_tag = run_source
        if ":" in run_source:
            workflow, run_tag = run_source.split(":", 1)
        run_evidence = resolve_run_evidence(run_tag, workflow=workflow).to_dict()
    return {
        "source": source_text,
        "task": task_text,
        "apply_brain_writeback": bool(apply_brain_writeback),
        "routes": routes,
        "brain_orchestration": {
            "primary_brain_id": str(routing.get("primary_brain_id", "") or ""),
            "supporting_brain_ids": list(routing.get("supporting_brain_ids", []) or []),
            "object_routes": list(routing.get("object_routes", []) or []),
            "writeback_targets": list(routing.get("writeback_targets", []) or []),
        } if routing else {},
        "artifact_freshness": freshness,
        "run_evidence": run_evidence,
        "planned_updates": [
            {"route": "state", "reason": "current status, boundaries, and priority changes"},
            {"route": "operations", "reason": "new command or workflow entry changes"},
            {"route": "episodic", "reason": "action-after review and evidence index"},
            {"route": "references", "reason": "large raw evidence or dated logs"},
        ],
        "requires_explicit_apply": not bool(apply_brain_writeback),
    }


def write_workflow_output(kind: str, payload: dict[str, Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_kind = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in kind)
    path = OUTPUT_ROOT / f"{safe_kind}_{stamp}.json"
    write_json(path, payload)
    return str(path.resolve())


def print_json(payload: dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def ensure_workspace_on_path() -> None:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
