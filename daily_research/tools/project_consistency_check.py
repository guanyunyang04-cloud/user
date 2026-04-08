from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "daily_research"
OUTPUT_ROOT = PROJECT_ROOT / "output"

EXPECTED_TARGET_WEIGHT_SEMANTICS = "research_raw_target_weight"
EXPECTED_TARGET_WEIGHT_CAP_MODE = "follow_research_raw_no_global_cap"
MIN_DEFAULT_EPOCHS = 32
MIN_DEFAULT_MIN_EPOCHS = 16


@dataclass(frozen=True)
class CheckResult:
    code: str
    detail: str


def _read_text(relative_path: str) -> str:
    return (WORKSPACE_ROOT / relative_path).read_text(encoding="utf-8")


def _read_json(relative_path: str) -> dict[str, Any]:
    payload = json.loads((WORKSPACE_ROOT / relative_path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_default_int(text: str, flag: str) -> int:
    pattern = rf'add_argument\("{re.escape(flag)}".*?default=([0-9]+)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse default integer for {flag}")
    return int(match.group(1))


def _has_default_expression(text: str, flag: str, snippet_regex: str) -> bool:
    pattern = rf'add_argument\("{re.escape(flag)}".*?default={snippet_regex}'
    return re.search(pattern, text, flags=re.DOTALL) is not None


def _parse_default_budget_start(text: str, flag: str) -> int:
    pattern = rf'add_argument\("{re.escape(flag)}".*?default="([^"]+)"'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse default budget list for {flag}")
    first = str(match.group(1)).split(",", 1)[0].strip()
    return int(first)


def _require(condition: bool, failures: list[CheckResult], code: str, detail: str) -> None:
    if not condition:
        failures.append(CheckResult(code=code, detail=detail))


def _check_train_entrypoints(failures: list[CheckResult]) -> None:
    train_text = _read_text("daily_research/deep_alpha/run_deep_alpha_research.py")
    pretrain_text = _read_text("daily_research/deep_alpha/pretrain_deep_alpha_encoder.py")

    _require(
        _has_default_expression(train_text, "--epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
        failures,
        "train_default_epochs_too_low",
        "run_deep_alpha_research.py default --epochs must start at >= 32.",
    )
    _require(
        _has_default_expression(train_text, "--min-epochs", "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)"),
        failures,
        "train_default_min_epochs_too_low",
        "run_deep_alpha_research.py default --min-epochs must stay aligned with the 32-start policy.",
    )
    _require(
        "No CUDA device is available" in train_text and 'torch.device("cuda")' in train_text,
        failures,
        "train_gpu_guard_missing",
        "run_deep_alpha_research.py must require CUDA instead of silently falling back to CPU.",
    )
    _require(
        'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in train_text,
        failures,
        "train_cpu_fallback_present",
        "run_deep_alpha_research.py still contains a CPU fallback path.",
    )

    _require(
        _has_default_expression(pretrain_text, "--epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
        failures,
        "pretrain_default_epochs_too_low",
        "pretrain_deep_alpha_encoder.py default --epochs must start at >= 32.",
    )
    _require(
        _has_default_expression(pretrain_text, "--min-epochs", "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)"),
        failures,
        "pretrain_default_min_epochs_too_low",
        "pretrain_deep_alpha_encoder.py default --min-epochs must stay aligned with the 32-start policy.",
    )
    _require(
        _parse_default_int(pretrain_text, "--max-total-epochs") >= MIN_DEFAULT_EPOCHS,
        failures,
        "pretrain_max_total_epochs_too_low",
        "pretrain_deep_alpha_encoder.py default --max-total-epochs must not cap below the start budget.",
    )
    _require(
        "No CUDA device is available" in pretrain_text and 'torch.device("cuda")' in pretrain_text,
        failures,
        "pretrain_gpu_guard_missing",
        "pretrain_deep_alpha_encoder.py must require CUDA instead of silently falling back to CPU.",
    )
    _require(
        'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in pretrain_text,
        failures,
        "pretrain_cpu_fallback_present",
        "pretrain_deep_alpha_encoder.py still contains a CPU fallback path.",
    )


def _check_aux_defaults(failures: list[CheckResult]) -> None:
    scripts = {
        "daily_research/deep_alpha/run_minimal_matrix.py": (
            ("--epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
            ("--min-epochs", "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)"),
            ("--pretrain-epochs", "DEFAULT_MIN_START_EPOCH_BUDGET"),
            ("--pretrain-min-epochs", "default_min_epochs_for_budget\\(DEFAULT_MIN_START_EPOCH_BUDGET\\)"),
            ("--max-total-epochs", MIN_DEFAULT_EPOCHS),
        ),
        "daily_research/deep_alpha/run_epoch_budget_formal_matrix.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
        "daily_research/deep_alpha/run_graph_off_plain_budget_review.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
        "daily_research/deep_alpha/run_encoder_transformer_stability_review.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
        "daily_research/deep_alpha/run_short_alpha_production_epoch_extension.py": (("--epoch-budgets", MIN_DEFAULT_EPOCHS),),
    }
    for relative_path, checks in scripts.items():
        text = _read_text(relative_path)
        for flag, minimum in checks:
            if flag == "--epoch-budgets":
                value = _parse_default_budget_start(text, flag)
                _require(
                    value >= int(minimum),
                    failures,
                    "aux_default_too_low",
                    f"{relative_path} has stale default for {flag}: {value} < {minimum}.",
                )
            elif isinstance(minimum, str):
                _require(
                    _has_default_expression(text, flag, minimum),
                    failures,
                    "aux_default_too_low",
                    f"{relative_path} is missing the shared current default expression for {flag}.",
                )
            else:
                value = _parse_default_int(text, flag)
                _require(
                    value >= int(minimum),
                    failures,
                    "aux_default_too_low",
                    f"{relative_path} has stale default for {flag}: {value} < {minimum}.",
                )


def _check_execution_semantics(failures: list[CheckResult]) -> None:
    manifest = _read_json("daily_research/output/active_execution_strategy.json")
    _require(
        str(manifest.get("target_weight_semantics", "")).strip() == EXPECTED_TARGET_WEIGHT_SEMANTICS,
        failures,
        "active_manifest_target_weight_semantics_mismatch",
        "active_execution_strategy.json is missing research_raw_target_weight semantics.",
    )
    _require(
        str(manifest.get("target_weight_cap_mode", "")).strip() == EXPECTED_TARGET_WEIGHT_CAP_MODE,
        failures,
        "active_manifest_target_weight_cap_mode_mismatch",
        "active_execution_strategy.json is missing follow_research_raw_no_global_cap semantics.",
    )


def _check_no_hardcoded_operational_roots(failures: list[CheckResult]) -> None:
    files = (
        "daily_research/execution/run_short_alpha_execution_single_mapping_candidate_pipeline.py",
        "daily_research/execution/activate_execution_single_mapping_candidate.py",
    )
    forbidden_patterns = (
        "short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_review_20",
        "short_alpha_production_execution_policy_audit_20",
        "short_alpha_execution_single_mapping_candidate_pipeline_20",
    )
    for relative_path in files:
        text = _read_text(relative_path)
        for pattern in forbidden_patterns:
            _require(
                pattern not in text,
                failures,
                "hardcoded_operational_root_present",
                f"{relative_path} still hardcodes a dated operational root containing `{pattern}`.",
            )


def _check_memory_sync(failures: list[CheckResult]) -> None:
    required_strings = {
        "daily_research/brain/working_memory.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            EXPECTED_TARGET_WEIGHT_CAP_MODE,
            "32",
            "最高优先级",
        ),
        "daily_research/brain/procedural_memory.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            EXPECTED_TARGET_WEIGHT_CAP_MODE,
            "strict resume",
            "GPU",
            "最高优先级",
        ),
        "daily_research/brain/action_system.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            EXPECTED_TARGET_WEIGHT_CAP_MODE,
            "最新提出的规则默认覆盖旧的本地假设",
        ),
        "daily_research/brain/project_map.md": (
            EXPECTED_TARGET_WEIGHT_SEMANTICS,
            EXPECTED_TARGET_WEIGHT_CAP_MODE,
            "最高优先级",
        ),
    }
    for relative_path, snippets in required_strings.items():
        text = _read_text(relative_path)
        for snippet in snippets:
            _require(
                snippet in text,
                failures,
                "brain_memory_not_synced",
                f"{relative_path} is missing current consistency marker: {snippet}",
            )


def run_checks() -> list[CheckResult]:
    failures: list[CheckResult] = []
    _check_train_entrypoints(failures)
    _check_aux_defaults(failures)
    _check_execution_semantics(failures)
    _check_no_hardcoded_operational_roots(failures)
    _check_memory_sync(failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check project-wide consistency rules that must not regress.")
    parser.parse_args()
    failures = run_checks()
    payload = {
        "status": "ok" if not failures else "failed",
        "failure_count": len(failures),
        "failures": [failure.__dict__ for failure in failures],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
