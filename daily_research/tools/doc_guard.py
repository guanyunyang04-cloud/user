from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from daily_research.tools.brain_integrity_check import run_checks as run_brain_integrity_checks


DEFAULT_DOCS = [
    "brain/identity_layer.md",
    "brain/state_center.md",
    "brain/knowledge_center.md",
    "brain/master_brain.md",
    "brain/brain_architecture.md",
    "brain/operations_center.md",
    "brain/governance_layer.md",
    "brain/brain_manifest.json",
    "daily_research/brain/identity_layer.md",
    "daily_research/brain/state_center.md",
    "daily_research/brain/knowledge_center.md",
    "daily_research/brain/continuous_policy_design_contract.md",
    "daily_research/brain/brain_architecture.md",
    "daily_research/brain/operations_center.md",
    "daily_research/brain/governance_layer.md",
    "daily_research/brain/episodic_memory.md",
    "daily_research/brain/brain_manifest.json",
    "t0_project/brain/identity_layer.md",
    "t0_project/brain/state_center.md",
    "t0_project/brain/knowledge_center.md",
    "t0_project/brain/brain_architecture.md",
    "t0_project/brain/operations_center.md",
    "t0_project/brain/governance_layer.md",
    "t0_project/brain/episodic_memory.md",
    "t0_project/brain/brain_manifest.json",
    "daily_stock_analysis-main/brain/identity_layer.md",
    "daily_stock_analysis-main/brain/state_center.md",
    "daily_stock_analysis-main/brain/knowledge_center.md",
    "daily_stock_analysis-main/brain/brain_architecture.md",
    "daily_stock_analysis-main/brain/operations_center.md",
    "daily_stock_analysis-main/brain/governance_layer.md",
    "daily_stock_analysis-main/brain/episodic_memory.md",
    "daily_stock_analysis-main/brain/brain_manifest.json",
]


@dataclass(frozen=True)
class DocRule:
    warn_lines: int | None = None
    max_lines: int | None = None
    forbidden_heading_patterns: tuple[tuple[str, str], ...] = ()
    forbidden_text_patterns: tuple[tuple[str, str], ...] = ()
    enforce_non_decreasing_dated_headings: bool = False


DOC_RULES = {
    "brain/identity_layer.md": DocRule(
        warn_lines=160,
        max_lines=240,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main identity layer should stay a current identity document instead of becoming a dated log"),
        ),
    ),
    "brain/master_brain.md": DocRule(
        warn_lines=160,
        max_lines=240,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main brain should stay a current topology document instead of becoming a dated log"),
        ),
    ),
    "brain/brain_architecture.md": DocRule(
        warn_lines=160,
        max_lines=240,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "brain/governance_layer.md": DocRule(
        warn_lines=160,
        max_lines=240,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main governance layer should stay a governance document instead of becoming a dated log"),
        ),
    ),
    "brain/brain_manifest.json": DocRule(
        warn_lines=200,
        max_lines=280,
    ),
    "daily_research/brain/identity_layer.md": DocRule(
        warn_lines=180,
        max_lines=260,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research identity layer should stay a current identity document instead of becoming a dated log"),
        ),
        forbidden_text_patterns=(
            (r"当前\s*live\s*默认执行", "mutable live default belongs in state_center and active_execution_strategy, not identity_layer"),
            (r"regoff_k2_5d_ensemble_native_anchor", "stale live default marker must not be kept in identity_layer"),
        ),
    ),
    "daily_research/brain/state_center.md": DocRule(
        warn_lines=180,
        max_lines=260,
    ),
    "daily_research/brain/operations_center.md": DocRule(
        warn_lines=180,
        max_lines=260,
    ),
    "daily_research/brain/continuous_policy_design_contract.md": DocRule(
        warn_lines=180,
        max_lines=260,
    ),
    "daily_research/brain/brain_architecture.md": DocRule(
        warn_lines=160,
        max_lines=240,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/governance_layer.md": DocRule(
        warn_lines=180,
        max_lines=260,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research governance layer should stay a governance document instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/episodic_memory.md": DocRule(
        enforce_non_decreasing_dated_headings=True,
    ),
    "daily_research/brain/brain_manifest.json": DocRule(
        warn_lines=180,
        max_lines=260,
    ),
    "t0_project/brain/identity_layer.md": DocRule(
        warn_lines=160,
        max_lines=240,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 identity layer should stay a current identity document instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/brain_architecture.md": DocRule(
        warn_lines=140,
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/governance_layer.md": DocRule(
        warn_lines=160,
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 governance layer should stay a governance document instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/episodic_memory.md": DocRule(
        warn_lines=120,
        max_lines=180,
    ),
    "t0_project/brain/brain_manifest.json": DocRule(
        warn_lines=180,
        max_lines=260,
    ),
    "daily_stock_analysis-main/brain/identity_layer.md": DocRule(
        warn_lines=180,
        max_lines=260,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis identity layer should stay a current identity document instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/brain_architecture.md": DocRule(
        warn_lines=140,
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/governance_layer.md": DocRule(
        warn_lines=160,
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis governance layer should stay a governance document instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/episodic_memory.md": DocRule(
        warn_lines=120,
        max_lines=180,
    ),
    "daily_stock_analysis-main/brain/brain_manifest.json": DocRule(
        warn_lines=180,
        max_lines=260,
    ),
}


BRAIN_DOC_PREFIXES = (
    "brain/",
    "daily_research/brain/",
    "t0_project/brain/",
    "daily_stock_analysis-main/brain/",
)

ALLOWED_EXTERNAL_DOC_PREFIXES = (
    "daily_research/cache/",
    "daily_research/output/",
    "daily_research/archive/output/",
    "daily_research/archive/manifests/",
    "daily_stock_analysis-main/docs/",
    "daily_stock_analysis-main/.github/",
    "daily_stock_analysis-main/.claude/skills/",
)

DOCUMENT_REDIRECTS = {
    "daily_research/archive/README.md": "daily_research/brain/references/archive_rules.md",
    "t0_project/先进技术通达信TQ策略研发.md": "t0_project/brain/references/tdx_tq_strategy_research.md",
    "t0_project/rl_agent/结合AI的通达信做T策略.md": "t0_project/brain/references/rl_t0_strategy_research.md",
    "daily_stock_analysis-main/review.md": "daily_stock_analysis-main/brain/references/review_audit_20260319.md",
}

ALLOWED_EXTERNAL_DOCS = {
    "daily_research/README.md",
    "daily_research/execution/使用教程.md",
    "daily_stock_analysis-main/AGENTS.md",
    "daily_stock_analysis-main/CLAUDE.md",
    "daily_stock_analysis-main/README.md",
    "daily_stock_analysis-main/SKILL.md",
    "daily_stock_analysis-main/strategies/README.md",
    *DOCUMENT_REDIRECTS.keys(),
}

REQUIRED_DOC_SNIPPETS = {
    "daily_research/README.md": (
        "权威接管真源",
        "daily_research/brain/",
    ),
    "daily_research/brain/identity_layer.md": (
        "## 5. 当前事实入口",
        "daily_research/brain/state_center.md",
        "daily_research/output/active_execution_strategy.json",
    ),
    "daily_research/execution/使用教程.md": (
        "权威操作真源",
        "daily_research/brain/operations_center.md",
    ),
    "daily_research/brain/episodic_memory.md": (
        "## 当前结论",
        "## 证据索引",
        "## 历史原文",
        "episodic_memory_history_raw_20260317_20260422.md",
    ),
    "daily_research/brain/state_center.md": (
        "## 当前结论",
        "## 历史归档入口",
        "split_heads_release_translation_deploy_r12",
        "split_heads_action_value_unification_r13",
        "split_heads_direct_action_value_r14",
        "monthly_consistency_score",
        "release_translation_deploy_v1",
        "state_center_history_raw_20260424.md",
        "state_center_evidence_index_20260424.md",
    ),
    "daily_research/brain/operations_center.md": (
        "## 默认操作纪律",
        "## 历史归档入口",
        "split_heads_release_translation_deploy_r12",
        "split_heads_action_value_unification_r13",
        "split_heads_direct_action_value_r14",
        "monthly_returns.csv",
        "operations_center_history_raw_20260424.md",
        "operations_center_evidence_index_20260424.md",
    ),
    "daily_research/brain/continuous_policy_design_contract.md": (
        "## 北极星",
        "## 历史归档入口",
        "alpha_result_value_budget_split_v12",
        "alpha_result_value_budget_split_v13",
        "alpha_result_value_budget_split_v14",
        "release_translation_deploy_health_score",
        "action_value_consistency_score",
        "direct_action_value_mode_share",
        "monthly_return_mean",
        "continuous_policy_design_contract_history_raw_20260424.md",
        "continuous_policy_design_contract_evidence_index_20260424.md",
    ),
    "daily_research/brain/references/episodic_memory_evidence_index_20260422.md": (
        "# episodic_memory 证据索引",
        "历史原文",
        "## 标题索引",
    ),
    "daily_research/brain/references/state_center_evidence_index_20260424.md": (
        "# state_center evidence index",
        "state_center_history_raw_20260424.md",
        "## Heading Index",
    ),
    "daily_research/brain/references/operations_center_evidence_index_20260424.md": (
        "# operations_center evidence index",
        "operations_center_history_raw_20260424.md",
        "## Heading Index",
    ),
    "daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md": (
        "# continuous_policy_design_contract evidence index",
        "continuous_policy_design_contract_history_raw_20260424.md",
        "## Heading Index",
    ),
    "daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md": (
        "# 研究日志",
        "## 2026-04-22 主分脑系统维护与兼容入口收口",
    ),
    "daily_stock_analysis-main/README.md": (
        "AI 接管真源",
        "daily_stock_analysis-main/brain/",
    ),
    "daily_stock_analysis-main/AGENTS.md": (
        "兼容入口",
        "brain/state_center.md",
        "brain/operations_center.md",
    ),
    "daily_stock_analysis-main/CLAUDE.md": (
        "AGENTS.md",
    ),
    "daily_stock_analysis-main/SKILL.md": (
        "AI 接管真源",
        "daily_stock_analysis-main/brain/",
    ),
    "daily_stock_analysis-main/strategies/README.md": (
        "AI 接管真源",
        "daily_stock_analysis-main/brain/",
    ),
    "daily_stock_analysis-main/.github/copilot-instructions.md": (
        "工作区接管真源",
        "brain/state_center.md",
        "brain/operations_center.md",
    ),
    "daily_stock_analysis-main/.github/instructions/governance.instructions.md": (
        "工作区级接管真源",
        "brain/state_center.md",
        "brain/operations_center.md",
    ),
}
MAIN_MANIFEST = Path("brain/brain_manifest.json")
SHARED_CONTRACT_KEY = "shared_regional_brain_contract"
MOJIBAKE_TOKENS = (
    "銆",
    "锛",
    "€",
    "identity銆",
    "state銆",
    "knowledge銆",
    "operations銆",
    "governance锛",
    "鏃犵姸鎬",
    "鍏堟帴",
    "澶ц剳",
    "韬唤",
    "body 鍦板浘",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _load_main_manifest() -> dict[str, Any]:
    return json.loads(_read_text(MAIN_MANIFEST))


def _split_contract_source(source: str) -> tuple[Path, str]:
    path_text, _, key = source.partition("#")
    return Path(path_text), key


def _parse_module_order_item(raw: Any) -> tuple[str, bool]:
    item = str(raw).strip()
    optional = item.endswith("?")
    return (item[:-1] if optional else item), optional


def _module_path_map(data: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for module in data.get("modules", []):
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("id", "")).strip()
        paths = module.get("paths")
        if not module_id or not isinstance(paths, list):
            continue
        out[module_id] = [str(path).strip() for path in paths if str(path).strip()]
    return out


def _shared_contract_from_main(main_manifest: dict[str, Any], source: str | None = None) -> dict[str, Any] | None:
    contract = main_manifest.get(SHARED_CONTRACT_KEY)
    if not isinstance(contract, dict):
        return None
    if source:
        path, key = _split_contract_source(source)
        if path.as_posix() != MAIN_MANIFEST.as_posix() or key != SHARED_CONTRACT_KEY:
            return None
    return contract


def _derive_sub_brain_read_order(data: dict[str, Any], main_manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    shared_source = str(data.get("shared_contract_source", "")).strip()
    if not shared_source:
        return [], ["shared_contract_source_missing_or_empty"]

    contract = _shared_contract_from_main(main_manifest, shared_source)
    if contract is None:
        return [], [f"shared_contract_source_invalid:{shared_source}"]
    if not contract.get("derive_read_order_from_modules"):
        return [], ["main_shared_regional_contract_must_enable_derived_read_order"]

    module_map = _module_path_map(data)
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in contract.get("default_module_order", []):
        module_id, optional = _parse_module_order_item(raw)
        paths = module_map.get(module_id, [])
        if not paths:
            if optional:
                continue
            issues.append(f"shared_contract_required_module_missing:{module_id}")
            continue
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)

    for module in data.get("modules", []):
        if not isinstance(module, dict):
            continue
        for raw_path in module.get("paths", []):
            path = str(raw_path).strip()
            if not path or path in seen:
                continue
            seen.add(path)
            ordered.append(path)

    return ordered, issues


def _tail_lines(text: str, count: int) -> list[str]:
    lines = text.splitlines()
    return lines[-count:] if count > 0 else lines


def _suspicious_question_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if "?" in line:
            out.append(line)
    return out


def _suspicious_mojibake_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if any(token in line for token in MOJIBAKE_TOKENS):
            out.append(line)
    return out


def _normalized_path(path: Path) -> str:
    return path.as_posix()


def _resolve_rule(path: Path) -> DocRule | None:
    normalized = _normalized_path(path)
    for suffix, rule in sorted(DOC_RULES.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized.endswith(suffix):
            return rule
    return None


def _matching_lines(lines: Iterable[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return [line for line in lines if regex.search(line)]


def _dated_headings(lines: Iterable[str]) -> list[tuple[int, str, str]]:
    regex = re.compile(r"^##\s+(20\d{2}-\d{2}-\d{2})\b")
    headings: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        match = regex.search(line)
        if match:
            headings.append((lineno, match.group(1), line))
    return headings


def _check_manifest_semantics(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid_json={exc}"]

    normalized = _normalized_path(path)
    if normalized == "brain/brain_manifest.json":
        if data.get("brain_type") != "main":
            issues.append("main_manifest_brain_type_must_be_main")
        shared_contract = data.get(SHARED_CONTRACT_KEY)
        if not isinstance(shared_contract, dict):
            issues.append("main_shared_regional_contract_missing_or_invalid")
        else:
            default_module_order = shared_contract.get("default_module_order")
            if not isinstance(default_module_order, list) or not default_module_order:
                issues.append("main_shared_regional_contract_default_module_order_missing_or_empty")
            required_modules = shared_contract.get("required_modules")
            if not isinstance(required_modules, list) or not required_modules:
                issues.append("main_shared_regional_contract_required_modules_missing_or_empty")
            optional_modules = shared_contract.get("optional_modules")
            if not isinstance(optional_modules, list):
                issues.append("main_shared_regional_contract_optional_modules_missing_or_invalid")
            fast_handoff_modules = shared_contract.get("fast_handoff_modules")
            if not isinstance(fast_handoff_modules, list) or not fast_handoff_modules:
                issues.append("main_shared_regional_contract_fast_handoff_modules_missing_or_empty")
            region_bindings = shared_contract.get("region_bindings")
            if not isinstance(region_bindings, dict) or not region_bindings:
                issues.append("main_shared_regional_contract_region_bindings_missing_or_invalid")

        for optional_key in (
            "identity_path",
            "state_path",
            "knowledge_path",
            "operations_path",
            "governance_path",
        ):
            optional_path = str(data.get(optional_key, "")).strip()
            if optional_path and not Path(optional_path).exists():
                issues.append(f"main_optional_brain_path_missing:{optional_key}:{optional_path}")
        child_brains = data.get("child_brains")
        if not isinstance(child_brains, list) or not child_brains:
            issues.append("main_manifest_child_brains_missing_or_empty")
            return issues

        for child in child_brains:
            if not isinstance(child, dict):
                issues.append("main_manifest_child_entry_must_be_object")
                continue
            for key in ("id", "path", "body_root", "entrypoint", "attach_status"):
                if key not in child:
                    issues.append(f"main_manifest_child_missing_key:{key}")

            child_path = child.get("path")
            if not isinstance(child_path, str):
                continue
            child_manifest = Path(child_path)
            if not child_manifest.exists():
                issues.append(f"child_manifest_missing:{child_path}")
                continue

            try:
                child_data = json.loads(_read_text(child_manifest))
            except json.JSONDecodeError as exc:
                issues.append(f"child_manifest_invalid_json:{child_path}:{exc}")
                continue

            if child_data.get("brain_id") != child.get("id"):
                issues.append(f"child_brain_id_mismatch:{child_path}")
            if child_data.get("parent_brain") != "brain/brain_manifest.json":
                issues.append(f"child_parent_brain_mismatch:{child_path}")
            if child_data.get("body_root") != child.get("body_root"):
                issues.append(f"child_body_root_mismatch:{child_path}")
            if child_data.get("entrypoint") != child.get("entrypoint"):
                issues.append(f"child_entrypoint_mismatch:{child_path}")
            if not str(child_data.get("attach_status", "")).startswith("attached"):
                issues.append(f"child_attach_status_invalid:{child_path}")
        return issues

    if normalized.endswith("/brain/brain_manifest.json"):
        required = {
            "brain_type",
            "brain_id",
            "parent_brain",
            "attach_status",
            "entrypoint",
            "body_root",
            "shared_contract_source",
            "regional_specialization",
            "write_routes",
            "body_map",
            "modules",
            "handoff_contract",
        }
        missing = sorted(required.difference(data))
        if missing:
            issues.append(f"sub_manifest_missing_keys:{','.join(missing)}")
            return issues

        if data.get("brain_type") != "sub_brain":
            issues.append("sub_manifest_brain_type_must_be_sub_brain")
        if not str(data.get("attach_status", "")).startswith("attached"):
            issues.append("sub_manifest_attach_status_must_start_with_attached")

        parent_path = Path(str(data.get("parent_brain", "")))
        if not parent_path.exists():
            issues.append(f"parent_brain_missing:{parent_path.as_posix()}")

        body_root = Path(str(data.get("body_root", "")))
        if not body_root.exists():
            issues.append(f"body_root_missing:{body_root.as_posix()}")

        entrypoint = Path(str(data.get("entrypoint", "")))
        if not entrypoint.exists():
            issues.append(f"entrypoint_missing:{entrypoint.as_posix()}")

        main_manifest = _load_main_manifest()
        shared_source = str(data.get("shared_contract_source", "")).strip()
        if shared_source:
            shared_path, shared_key = _split_contract_source(shared_source)
            if shared_path.as_posix() != MAIN_MANIFEST.as_posix() or shared_key != SHARED_CONTRACT_KEY:
                issues.append(f"shared_contract_source_invalid:{shared_source}")
        else:
            issues.append("shared_contract_source_missing_or_empty")

        module_map = _module_path_map(data)
        read_order = data.get("read_order")
        resolved_read_order: list[str] = []
        if isinstance(read_order, list) and read_order:
            resolved_read_order = [str(item) for item in read_order]
        else:
            resolved_read_order, derivation_issues = _derive_sub_brain_read_order(data, main_manifest)
            issues.extend(derivation_issues)
            if not resolved_read_order:
                issues.append("read_order_missing_and_shared_contract_derivation_failed")

        if resolved_read_order:
            if Path(str(resolved_read_order[0])) != entrypoint:
                issues.append("read_order_first_item_should_match_entrypoint")
            for item in resolved_read_order:
                if not Path(str(item)).exists():
                    issues.append(f"read_order_path_missing:{item}")

        write_routes = data.get("write_routes")
        if not isinstance(write_routes, dict) or not write_routes:
            issues.append("write_routes_missing_or_empty")
        else:
            for route_path in write_routes.values():
                if not Path(str(route_path)).exists():
                    issues.append(f"write_route_missing:{route_path}")

        body_map = data.get("body_map")
        if not isinstance(body_map, dict) or not body_map:
            issues.append("body_map_missing_or_empty")
        else:
            for mapped_paths in body_map.values():
                if not isinstance(mapped_paths, list) or not mapped_paths:
                    issues.append("body_map_group_missing_paths")
                    continue
                for mapped_path in mapped_paths:
                    if not Path(str(mapped_path)).exists():
                        issues.append(f"body_map_path_missing:{mapped_path}")

        modules = data.get("modules")
        if not isinstance(modules, list) or not modules:
            issues.append("modules_missing_or_empty")
        else:
            for module in modules:
                if not isinstance(module, dict):
                    issues.append("module_entry_must_be_object")
                    continue
                if "id" not in module or "paths" not in module:
                    issues.append("module_missing_id_or_paths")
                    continue
                paths = module.get("paths")
                if not isinstance(paths, list) or not paths:
                    issues.append(f"module_paths_missing_or_empty:{module.get('id', 'unknown')}")
                    continue
                for module_path in paths:
                    if not Path(str(module_path)).exists():
                        issues.append(f"module_path_missing:{module_path}")

        shared_contract = _shared_contract_from_main(main_manifest, shared_source or None)
        if isinstance(shared_contract, dict):
            for module_id in shared_contract.get("required_modules", []):
                if str(module_id) not in module_map:
                    issues.append(f"shared_contract_required_module_missing:{module_id}")

        regional_specialization = data.get("regional_specialization")
        if not isinstance(regional_specialization, dict) or not regional_specialization:
            issues.append("regional_specialization_missing_or_invalid")
        else:
            role = str(regional_specialization.get("role", "")).strip()
            if not role:
                issues.append("regional_specialization_role_missing_or_empty")

            priority_regions = regional_specialization.get("priority_regions")
            if not isinstance(priority_regions, list) or not priority_regions:
                issues.append("regional_specialization_priority_regions_missing_or_empty")
            elif isinstance(shared_contract, dict):
                region_bindings = shared_contract.get("region_bindings", {})
                for region in priority_regions:
                    if str(region) not in region_bindings:
                        issues.append(f"regional_specialization_unknown_region:{region}")

            focus_modules = regional_specialization.get("focus_modules")
            if not isinstance(focus_modules, list) or not focus_modules:
                issues.append("regional_specialization_focus_modules_missing_or_empty")
            else:
                for module_id in focus_modules:
                    if str(module_id) not in module_map:
                        issues.append(f"regional_specialization_focus_module_missing:{module_id}")

        handoff = data.get("handoff_contract")
        if not isinstance(handoff, dict):
            issues.append("handoff_contract_missing_or_invalid")
        else:
            entry_sequence = handoff.get("entry_sequence")
            if not isinstance(entry_sequence, list) or not entry_sequence:
                if not handoff.get("derive_entry_sequence_from_shared_contract"):
                    issues.append("handoff_entry_sequence_missing_and_not_derived")
                elif not resolved_read_order:
                    issues.append("handoff_entry_sequence_missing_and_read_order_unavailable")
                else:
                    derived_entry_sequence = [path.as_posix(), *resolved_read_order]
                    for item in derived_entry_sequence:
                        if not Path(str(item)).exists():
                            issues.append(f"handoff_entry_path_missing:{item}")
            else:
                if Path(str(entry_sequence[0])) != path:
                    issues.append("handoff_entry_sequence_should_start_with_manifest")
                for item in entry_sequence:
                    if not Path(str(item)).exists():
                        issues.append(f"handoff_entry_path_missing:{item}")

        for optional_key in (
            "identity_path",
            "state_path",
            "knowledge_path",
            "operations_path",
            "governance_path",
        ):
            optional_path = str(data.get(optional_key, "")).strip()
            if not optional_path:
                continue
            if not Path(optional_path).exists():
                issues.append(f"optional_brain_path_missing:{optional_key}:{optional_path}")
            if optional_key == "identity_path" and Path(optional_path) != entrypoint:
                issues.append("identity_path_should_match_entrypoint")

    return issues


def _is_allowed_doc_path(relative_path: str) -> bool:
    if relative_path in ALLOWED_EXTERNAL_DOCS:
        return True
    if relative_path.startswith(BRAIN_DOC_PREFIXES):
        return True
    return relative_path.startswith(ALLOWED_EXTERNAL_DOC_PREFIXES)


def _check_document_layout() -> list[str]:
    issues: list[str] = []
    workspace_root = Path(__file__).resolve().parents[2]

    for path in workspace_root.rglob("*.md"):
        relative_path = _normalized_path(path.relative_to(workspace_root))
        if _is_allowed_doc_path(relative_path):
            continue
        issues.append(f"doc_outside_brain_without_exception:{relative_path}")

    for relative_path, target in DOCUMENT_REDIRECTS.items():
        redirect_path = workspace_root / relative_path
        target_path = workspace_root / target
        if not redirect_path.exists():
            issues.append(f"redirect_doc_missing:{relative_path}")
            continue
        if not target_path.exists():
            issues.append(f"redirect_target_missing:{target}")
            continue
        text = _read_text(redirect_path)
        if "Canonical brain source:" not in text or target not in text:
            issues.append(f"redirect_doc_missing_canonical_marker:{relative_path}")

    for relative_path, snippets in REQUIRED_DOC_SNIPPETS.items():
        path = workspace_root / relative_path
        if not path.exists():
            issues.append(f"required_doc_missing:{relative_path}")
            continue
        text = _read_text(path)
        for snippet in snippets:
            if snippet not in text:
                issues.append(f"required_doc_snippet_missing:{relative_path}:{snippet}")

    return issues


def _check_active_execution_brain_alignment() -> list[str]:
    issues: list[str] = []
    active_path = WORKSPACE_ROOT / "daily_research/output/active_execution_strategy.json"
    if not active_path.exists():
        return issues
    try:
        active = json.loads(active_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return [f"active_execution_strategy_invalid_json:{exc}"]
    if not isinstance(active, dict):
        return ["active_execution_strategy_must_be_object"]

    active_label = str(active.get("trade_plan_candidate_label") or active.get("candidate_label") or "").strip()
    active_profile = str(
        active.get("effective_live_execution_profile")
        or active.get("execution_alignment_profile")
        or active.get("execution_policy_label")
        or ""
    ).strip()
    if not active_label:
        issues.append("active_execution_strategy_label_missing")
        return issues

    for relative_path in (
        "daily_research/brain/state_center.md",
        "daily_research/brain/knowledge_center.md",
    ):
        path = WORKSPACE_ROOT / relative_path
        if not path.exists():
            issues.append(f"active_execution_brain_target_missing:{relative_path}")
            continue
        text = _read_text(path)
        if active_label not in text:
            issues.append(f"active_execution_label_missing_from_brain:{relative_path}:{active_label}")
        if active_profile and active_profile not in text:
            issues.append(f"active_execution_profile_missing_from_brain:{relative_path}:{active_profile}")

    identity_path = WORKSPACE_ROOT / "daily_research/brain/identity_layer.md"
    if identity_path.exists():
        identity_text = _read_text(identity_path)
        if active_label in identity_text or "当前 live 默认执行" in identity_text:
            issues.append("mutable_active_execution_state_leaked_into_identity_layer")
    return issues


def cmd_check(args: argparse.Namespace) -> int:
    has_issue = False
    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"[missing] {path}")
            has_issue = True
            continue

        text = _read_text(path)
        lines = text.splitlines()
        line_count = len(lines)
        replacement_count = text.count("\ufffd")
        tail = _tail_lines(text, args.tail_lines)
        tail_question_lines = _suspicious_question_lines(tail)
        mojibake_lines = _suspicious_mojibake_lines(lines)
        rule = _resolve_rule(path)

        print(f"[check] {path}")
        print(f"  line_count={line_count}")
        print(f"  replacement_char_count={replacement_count}")
        print(f"  suspicious_question_lines_in_tail={len(tail_question_lines)}")
        print(f"  suspicious_mojibake_lines={len(mojibake_lines)}")

        if replacement_count or tail_question_lines or mojibake_lines:
            has_issue = True
            for line in tail_question_lines[: args.show_lines]:
                print(f"    ? {line}")
            for line in mojibake_lines[: args.show_lines]:
                print(f"    ! {line}")

        if rule and rule.warn_lines is not None and line_count > rule.warn_lines:
            print(f"  structural_warning=line_count_exceeds_warning ({line_count} > {rule.warn_lines})")

        if rule and rule.max_lines is not None and line_count > rule.max_lines:
            has_issue = True
            print(f"  structural_issue=line_count_exceeds_limit ({line_count} > {rule.max_lines})")

        if rule:
            for pattern, reason in rule.forbidden_heading_patterns:
                matches = _matching_lines(lines, pattern)
                print(f"  forbidden_heading_matches={len(matches)} for rule: {reason}")
                if matches:
                    has_issue = True
                    for line in matches[: args.show_lines]:
                        print(f"    ! {line}")

            for pattern, reason in rule.forbidden_text_patterns:
                matches = _matching_lines(lines, pattern)
                print(f"  forbidden_text_matches={len(matches)} for rule: {reason}")
                if matches:
                    has_issue = True
                    for line in matches[: args.show_lines]:
                        print(f"    ! {line}")

            if rule.enforce_non_decreasing_dated_headings:
                headings = _dated_headings(lines)
                out_of_order_pairs: list[tuple[tuple[int, str, str], tuple[int, str, str]]] = []
                for previous, current in zip(headings, headings[1:]):
                    if current[1] < previous[1]:
                        out_of_order_pairs.append((previous, current))

                print(f"  dated_heading_order_issues={len(out_of_order_pairs)}")
                if out_of_order_pairs:
                    has_issue = True
                    for previous, current in out_of_order_pairs[: args.show_lines]:
                        print(f"    ! {path}:{current[0]} date {current[1]} appears after later date {previous[1]}")
                        print(f"      prev={previous[2]}")
                        print(f"      curr={current[2]}")

        if path.suffix.lower() == ".json" and _normalized_path(path).endswith("brain_manifest.json"):
            manifest_issues = _check_manifest_semantics(path, text)
            print(f"  manifest_semantic_issues={len(manifest_issues)}")
            if manifest_issues:
                has_issue = True
                for issue in manifest_issues[: args.show_lines]:
                    print(f"    ! {issue}")

    layout_issues = _check_document_layout()
    print(f"[layout] documentation_layout_issues={len(layout_issues)}")
    if layout_issues:
        has_issue = True
        for issue in layout_issues[: args.show_lines]:
            print(f"    ! {issue}")

    active_alignment_issues = _check_active_execution_brain_alignment()
    print(f"[active-execution] brain_alignment_issues={len(active_alignment_issues)}")
    if active_alignment_issues:
        has_issue = True
        for issue in active_alignment_issues[: args.show_lines]:
            print(f"    ! {issue}")

    brain_findings = run_brain_integrity_checks()
    brain_errors = [finding for finding in brain_findings if finding.severity == "error"]
    brain_warnings = [finding for finding in brain_findings if finding.severity == "warning"]
    print(f"[brain-integrity] errors={len(brain_errors)} warnings={len(brain_warnings)}")
    if brain_errors:
        has_issue = True
    for finding in brain_findings[: args.show_lines]:
        path = f" path={finding.path}" if finding.path else ""
        print(f"    ! [{finding.severity}] {finding.code}{path}: {finding.detail}")

    return 1 if has_issue else 0


def cmd_append(args: argparse.Namespace) -> int:
    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_text(path) if path.exists() else ""
    body = Path(args.body_file).read_text(encoding="utf-8")
    parts = [existing.rstrip(), args.header.rstrip(), "", body.strip(), ""]
    path.write_text("\n".join(part for part in parts if part != ""), encoding="utf-8")
    print(f"[append] wrote UTF-8 section to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UTF-8-safe and structure-aware helper for workspace brains.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check brain docs for encoding damage and structure drift.")
    check.add_argument("--files", nargs="+", default=DEFAULT_DOCS)
    check.add_argument("--tail-lines", type=int, default=120)
    check.add_argument("--show-lines", type=int, default=12)
    check.set_defaults(func=cmd_check)

    append = sub.add_parser("append", help="Append a markdown section using explicit UTF-8 writes.")
    append.add_argument("--file", required=True)
    append.add_argument("--header", required=True, help="Markdown header line, e.g. ## 2026-03-20 ...")
    append.add_argument("--body-file", required=True, help="UTF-8 markdown fragment to append under the header.")
    append.set_defaults(func=cmd_append)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
