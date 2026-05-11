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


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAIN_MANIFEST = Path("brain/brain_manifest.json")
WORKFLOW_REGISTRY = Path("daily_research/brain/workflow_registry.json")
OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research/output/brain_workflow"
PYTHON_EXECUTABLE = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SHARED_CONTRACT_KEY = "shared_regional_brain_contract"
OPTIONAL_BRAIN_KEYS = (
    "identity_path",
    "state_path",
    "knowledge_path",
    "operations_path",
    "governance_path",
)
MOJIBAKE_MARKERS = (
    "\ufffd",
    "鎿",
    "鐭",
    "绋",
    "锛",
    "銆",
    "歚",
    "乣",
    "涓",
    "浠",
)
LATEST_ARTIFACTS = {
    "latest_study_summary": Path("daily_research/output/continuous_policy/latest_study_summary.json"),
    "latest_protocol_summary": Path("daily_research/output/continuous_policy/latest_protocol_summary.json"),
    "latest_behavior_audit_summary": Path("daily_research/output/continuous_policy/latest_behavior_audit_summary.json"),
    "latest_conclusion_ledger": Path("daily_research/output/continuous_policy/latest_conclusion_ledger.json"),
}
CONTINUOUS_POLICY_OUTPUT_ROOT = Path("daily_research/output/continuous_policy")
STUDY_SUMMARY_METRIC_KEYS = (
    "annual_return",
    "max_drawdown",
    "monthly_return_mean",
    "cash_timing_quality_1d",
    "portfolio_daily_source_target_count",
    "portfolio_daily_source_realized_sell_rate",
    "portfolio_daily_exposure_utilization",
    "portfolio_daily_receiver_unrealized_deploy_share",
    "allocation_layer_native_target_used",
    "allocation_layer_native_fallback_used",
    "native_target_valid",
    "native_receiver_executable_mask_count",
    "native_positive_delta_count",
    "native_positive_delta_unsupported_share",
    "native_receiver_mask_mismatch_count",
    "native_target_constraint_violations",
    "native_target_invalid_sum_count",
    "native_target_invalid_turnover_count",
    "native_target_invalid_cap_count",
    "native_target_invalid_negative_weight_count",
    "native_target_invalid_unsupported_receiver_count",
    "native_target_invalid_sell_nonheld_count",
    "native_source_target_count",
    "training_evidence_status",
)


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
class ArtifactRecord:
    name: str
    path: str
    exists: bool
    tag: str
    executed_at: str
    generated_at: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactFreshnessReport:
    artifacts: dict[str, ArtifactRecord]
    is_stale_risk: bool
    mismatch_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": {key: value.to_dict() for key, value in self.artifacts.items()},
            "is_stale_risk": self.is_stale_risk,
            "mismatch_reason": self.mismatch_reason,
        }


@dataclass(frozen=True)
class BrainHealthReport:
    status: str
    checks: dict[str, dict[str, Any]]
    elapsed_seconds: float

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
    study_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyTrialEvidence:
    trial_id: int
    trial_tag: str
    status: str
    phase: str
    protocol_summary_json: str
    evaluation_summary_json: str
    training_diagnostics_json: str
    loss_profile: str
    sample_model_type: str
    training_evidence_status: str
    metrics: dict[str, Any]
    training_terms: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyEvidenceReport:
    study_tag: str
    study_summary_json: str
    exists: bool
    coherent: bool
    artifact_freshness: dict[str, Any]
    trial_count: int
    completed_trial_count: int
    failed_trial_count: int
    search_profile: str
    objective_profile: str
    confirmatory_enabled: bool
    resource_limits: dict[str, Any]
    trials: list[StudyTrialEvidence]
    evidence_gaps: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trials"] = [trial.to_dict() for trial in self.trials]
        return payload


def workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def read_text(path: str | Path) -> str:
    return workspace_path(path).read_text(encoding="utf-8-sig")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def load_workflow_registry() -> dict[str, dict[str, Any]]:
    payload = json.loads(read_text(WORKFLOW_REGISTRY))
    if not isinstance(payload, dict):
        raise ValueError("workflow_registry.json must be a JSON object")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


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
    main_manifest = load_manifest(MAIN_MANIFEST)
    shared_contract = _shared_contract(main_manifest)
    main_order = _build_main_order(main_manifest)
    main_paths = [path.as_posix() for path in main_order]
    artifact_freshness = resolve_artifact_freshness().to_dict()
    registry = load_workflow_registry()

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
            main_optional_paths=_collect_optional_paths(main_manifest),
            artifact_freshness=artifact_freshness,
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
        workflow_hints={"available_workflows": sorted(registry), "recommended_default": "brain_handoff"},
        encoding_report=_encoding_summary(boot_order),
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    full = workspace_path(path)
    if not full.exists():
        return {}
    try:
        payload = json.loads(full.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_path_text(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    return _read_json_if_exists(Path(path_text))


def _extract_artifact_tag(payload: dict[str, Any]) -> str:
    for key in ("study_tag", "protocol_tag", "run_tag", "tag"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _artifact_record(name: str, path: Path) -> ArtifactRecord:
    payload = _read_json_if_exists(path)
    promotion_gate = payload.get("promotion_gate") if isinstance(payload.get("promotion_gate"), dict) else {}
    return ArtifactRecord(
        name=name,
        path=path.as_posix(),
        exists=workspace_path(path).exists(),
        tag=_extract_artifact_tag(payload),
        executed_at=str(payload.get("executed_at", "") or ""),
        generated_at=str(payload.get("generated_at", "") or ""),
        status=str(payload.get("status", "") or promotion_gate.get("status", "") or payload.get("phase", "") or ""),
    )


def resolve_artifact_freshness() -> ArtifactFreshnessReport:
    records = {name: _artifact_record(name, path) for name, path in LATEST_ARTIFACTS.items()}
    study_tag = records["latest_study_summary"].tag
    protocol_tag = records["latest_protocol_summary"].tag
    mismatch = bool(study_tag and protocol_tag and study_tag != protocol_tag)
    reason = ""
    if mismatch:
        reason = f"latest study tag differs from latest protocol tag: {study_tag} != {protocol_tag}"
    return ArtifactFreshnessReport(records, mismatch, reason)


def _study_summary_path(study_tag: str) -> Path:
    return CONTINUOUS_POLICY_OUTPUT_ROOT / "studies" / study_tag / "study_summary.json"


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _first_path_payload(*path_texts: str) -> tuple[str, dict[str, Any]]:
    for path_text in path_texts:
        if not str(path_text or "").strip():
            continue
        payload = _read_json_path_text(str(path_text))
        if payload:
            return str(path_text), payload
    return "", {}


def _trial_training_diagnostics(protocol_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    embedded = _nested_dict(protocol_payload, "train", "training_diagnostics")
    path_text = str(
        embedded.get("training_diagnostics_json", "")
        or protocol_payload.get("training_diagnostics_json", "")
        or protocol_payload.get("model_training_diagnostics_json", "")
        or ""
    )
    path_payload = _read_json_path_text(path_text)
    if path_payload:
        return path_text, path_payload
    return path_text, embedded


def _trial_evaluation_payload(protocol_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    embedded = _nested_dict(protocol_payload, "evaluation")
    path_text = str(
        embedded.get("evaluation_summary_json", "")
        or protocol_payload.get("evaluation_summary_json", "")
        or ""
    )
    path_payload = _read_json_path_text(path_text)
    if path_payload:
        return path_text, path_payload
    return path_text, embedded


def _metric_from_sources(key: str, *sources: dict[str, Any]) -> Any:
    for source in sources:
        if key in source:
            return source[key]
    for source in sources:
        for nested_key in ("continuity_metrics", "continuous_policy_metrics", "primary_metrics", "training_evidence"):
            nested = source.get(nested_key) if isinstance(source, dict) else None
            if isinstance(nested, dict) and key in nested:
                return nested[key]
    return None


def _build_trial_evidence(trial_payload: dict[str, Any]) -> StudyTrialEvidence:
    protocol_path = str(trial_payload.get("protocol_summary_json", "") or "")
    protocol_payload = _read_json_path_text(protocol_path)
    evaluation_path, evaluation_payload = _trial_evaluation_payload(protocol_payload)
    diagnostics_path, diagnostics_payload = _trial_training_diagnostics(protocol_payload)
    primary_metrics = trial_payload.get("primary_metrics") if isinstance(trial_payload.get("primary_metrics"), dict) else {}
    continuity_metrics = _nested_dict(evaluation_payload, "continuity_metrics")
    training_evidence = _nested_dict(protocol_payload, "training_evidence")
    metrics: dict[str, Any] = {}
    for key in STUDY_SUMMARY_METRIC_KEYS:
        value = _metric_from_sources(
            key,
            trial_payload,
            primary_metrics,
            continuity_metrics,
            evaluation_payload,
            training_evidence,
            diagnostics_payload,
        )
        if value is not None:
            metrics[key] = value
    terms = diagnostics_payload.get("portfolio_day_set_native_allocation_vector_terms", {})
    if not isinstance(terms, dict):
        terms = {}
    trial_config = trial_payload.get("trial_config") if isinstance(trial_payload.get("trial_config"), dict) else {}
    return StudyTrialEvidence(
        trial_id=int(trial_payload.get("trial_id", 0) or 0),
        trial_tag=str(trial_payload.get("trial_tag", "") or protocol_payload.get("run_tag", "") or ""),
        status=str(trial_payload.get("status", "") or ""),
        phase=str(trial_payload.get("phase", "") or ""),
        protocol_summary_json=protocol_path,
        evaluation_summary_json=evaluation_path,
        training_diagnostics_json=diagnostics_path,
        loss_profile=str(trial_config.get("loss_profile", "") or protocol_payload.get("loss_profile", "") or diagnostics_payload.get("loss_profile", "") or ""),
        sample_model_type=str(diagnostics_payload.get("sample_model_type", "") or ""),
        training_evidence_status=str(
            _metric_from_sources("training_evidence_status", trial_payload, primary_metrics, training_evidence, diagnostics_payload)
            or training_evidence.get("status", "")
            or ""
        ),
        metrics=metrics,
        training_terms=dict(terms),
    )


def resolve_study_evidence(study_tag: str) -> StudyEvidenceReport:
    tag = str(study_tag or "").strip()
    if not tag:
        raise ValueError("study_tag is required")
    path = _study_summary_path(tag)
    summary = _read_json_if_exists(path)
    gaps: list[str] = []
    trials: list[StudyTrialEvidence] = []
    if not summary:
        gaps.append(f"study summary not found: {path.as_posix()}")
    for trial_payload in summary.get("screening_trials", []) if isinstance(summary.get("screening_trials"), list) else []:
        if isinstance(trial_payload, dict):
            evidence = _build_trial_evidence(trial_payload)
            trials.append(evidence)
            if not evidence.protocol_summary_json:
                gaps.append(f"trial {evidence.trial_id} missing protocol_summary_json")
            if not evidence.training_diagnostics_json:
                gaps.append(f"trial {evidence.trial_id} missing training diagnostics path")
    if summary and str(summary.get("study_tag", "") or summary.get("run_tag", "") or "") != tag:
        gaps.append("study summary tag differs from requested tag")
    return StudyEvidenceReport(
        study_tag=tag,
        study_summary_json=path.as_posix(),
        exists=bool(summary),
        coherent=bool(summary) and not gaps,
        artifact_freshness=resolve_artifact_freshness().to_dict(),
        trial_count=int(summary.get("trial_count", 0) or 0),
        completed_trial_count=int(summary.get("completed_trial_count", 0) or 0),
        failed_trial_count=int(summary.get("failed_trial_count", 0) or 0),
        search_profile=str(summary.get("search_profile", "") or ""),
        objective_profile=str(summary.get("objective_profile", "") or ""),
        confirmatory_enabled=bool(summary.get("confirmatory_enabled", False)),
        resource_limits=dict(summary.get("resource_limits", {}) or {}),
        trials=trials,
        evidence_gaps=gaps,
    )


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
        check=False,
        env=env,
    )
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-2000:],
        "elapsed_seconds": round(elapsed, 3),
    }


def check_brain_health() -> BrainHealthReport:
    started = time.perf_counter()
    check_commands = {
        "brain_integrity": {
            "command": [PYTHON_EXECUTABLE, "daily_research/tools/brain_integrity_check.py", "--json"],
            "env": None,
        },
        "doc_guard": {
            "command": [PYTHON_EXECUTABLE, "daily_research/tools/doc_guard.py", "check"],
            "env": None,
        },
        "project_consistency": {
            "command": [PYTHON_EXECUTABLE, "daily_research/tools/project_consistency_check.py"],
            "env": None,
        },
        "openmp_strict": {
            "command": [PYTHON_EXECUTABLE, "daily_research/tools/openmp_runtime_check.py", "--strict"],
            "env": _openmp_strict_env(),
        },
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
    return BrainHealthReport(status=status, checks=checks, elapsed_seconds=round(time.perf_counter() - started, 3))


def _continuous_policy_evidence_gaps(freshness: ArtifactFreshnessReport) -> list[str]:
    gaps: list[str] = []
    if freshness.is_stale_risk:
        gaps.append("latest_study_summary and latest_protocol_summary tags differ")
    study = freshness.artifacts["latest_study_summary"]
    protocol = freshness.artifacts["latest_protocol_summary"]
    if not study.exists:
        gaps.append("latest study summary is missing")
    if not protocol.exists:
        gaps.append("latest protocol summary is missing")
    return gaps


def build_workflow_state(workflow_id: str, *, study_tag: str | None = None) -> WorkflowState:
    registry = load_workflow_registry()
    normalized = "continuous_policy_result_review" if workflow_id == "continuous_policy" else workflow_id
    if normalized not in registry:
        raise KeyError(f"Unknown workflow: {workflow_id}")
    freshness = resolve_artifact_freshness()
    study_evidence: dict[str, Any] = {}
    if study_tag and normalized.startswith("continuous_policy"):
        evidence = resolve_study_evidence(study_tag)
        study_evidence = evidence.to_dict()
        gaps = list(evidence.evidence_gaps)
    else:
        gaps = _continuous_policy_evidence_gaps(freshness) if normalized.startswith("continuous_policy") else []
    next_allowed = list(registry[normalized].get("allowed_commands", []) or [])
    resource_risk = "safe" if "safe_screening" in normalized else ("stale_latest_risk" if freshness.is_stale_risk else "normal")
    return WorkflowState(
        workflow_id=normalized,
        status="blocked" if gaps and normalized == "continuous_policy_result_review" else "ready",
        read_only=True,
        writes_tracked_files=False,
        registry_entry=registry[normalized],
        artifact_freshness=freshness.to_dict(),
        evidence_gaps=gaps,
        next_allowed_actions=next_allowed,
        resource_risk=("normal_with_stale_latest" if study_evidence and freshness.is_stale_risk else resource_risk),
        study_evidence=study_evidence,
    )


def build_writeback_plan(source: str, *, apply_brain_writeback: bool = False) -> dict[str, Any]:
    registry = load_workflow_registry()
    routes = dict(registry["brain_writeback"].get("writeback_routes", {}) or {})
    freshness = resolve_artifact_freshness().to_dict()
    source_text = str(source or "latest").strip() or "latest"
    study_evidence: dict[str, Any] = {}
    if source_text.startswith("study:"):
        study_evidence = resolve_study_evidence(source_text.split(":", 1)[1]).to_dict()
    return {
        "source": source_text,
        "apply_brain_writeback": bool(apply_brain_writeback),
        "routes": routes,
        "artifact_freshness": freshness,
        "study_evidence": study_evidence,
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def ensure_workspace_on_path() -> None:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
