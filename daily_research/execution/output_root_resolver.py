from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"

SINGLE_MAPPING_REVIEW_PREFIX = "short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_review_"
SINGLE_MAPPING_PIPELINE_PREFIX = "short_alpha_execution_single_mapping_candidate_pipeline_"
RECENT_EXECUTION_AUDIT_PREFIX = "short_alpha_production_execution_policy_audit_"
FORMAL_EXECUTION_AUDIT_WINDOW_PREFIXES = (
    "short_alpha_formal_execution_policy_audit_20230216_20240229_",
    "short_alpha_formal_execution_policy_audit_20240301_20250317_",
)
FORMAL_EXECUTION_AUDIT_WINDOW_TOKENS = (
    "20230216_20240229",
    "20240301_20250317",
)
FORMAL_EXECUTION_AUDIT_PREFIX = "short_alpha_formal_execution_policy_audit_"
STATIC_PRODUCTION_ROOT = OUTPUT_ROOT / "deep_alpha_short_alpha_execalign_production_default"


def latest_output_dir(prefix: str) -> Path:
    matches = sorted(OUTPUT_ROOT.glob(f"{prefix}*"), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No output root matched prefix: {prefix}")
    return matches[-1].resolve()


def latest_output_dir_by_glob(pattern: str) -> Path:
    matches = sorted(OUTPUT_ROOT.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No output root matched pattern: {pattern}")
    return matches[-1].resolve()


def resolve_output_dir_arg(raw_path: str | Path | None, *, latest_prefix: str) -> Path:
    if raw_path not in {None, ""}:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Resolved path does not exist: {candidate}")
        return candidate
    return latest_output_dir(latest_prefix)


def resolve_single_mapping_review_root(raw_path: str | Path | None) -> Path:
    return resolve_output_dir_arg(raw_path, latest_prefix=SINGLE_MAPPING_REVIEW_PREFIX)


def resolve_recent_execution_audit_root(raw_path: str | Path | None) -> Path:
    return resolve_output_dir_arg(raw_path, latest_prefix=RECENT_EXECUTION_AUDIT_PREFIX)


def resolve_single_mapping_pipeline_root(raw_path: str | Path | None) -> Path:
    return resolve_output_dir_arg(raw_path, latest_prefix=SINGLE_MAPPING_PIPELINE_PREFIX)


def resolve_formal_execution_audit_roots(raw_value: str | None) -> list[Path]:
    if str(raw_value or "").strip():
        roots: list[Path] = []
        for token in str(raw_value).split(","):
            item = token.strip()
            if not item:
                continue
            roots.append(resolve_output_dir_arg(item, latest_prefix=FORMAL_EXECUTION_AUDIT_PREFIX))
        if not roots:
            raise FileNotFoundError("No formal audit roots were resolved from --audit-roots.")
        return roots
    roots = [latest_output_dir(prefix) for prefix in FORMAL_EXECUTION_AUDIT_WINDOW_PREFIXES]
    generic_candidates = [
        path.resolve()
        for path in OUTPUT_ROOT.glob(f"{FORMAL_EXECUTION_AUDIT_PREFIX}*")
        if all(token not in path.name for token in FORMAL_EXECUTION_AUDIT_WINDOW_TOKENS)
    ]
    if not generic_candidates:
        raise FileNotFoundError("No latest-window formal execution audit root was found.")
    roots.append(max(generic_candidates, key=lambda item: item.stat().st_mtime))
    return roots


def resolve_single_mapping_root_tag(raw_tag: str | None) -> str:
    tag = str(raw_tag or "").strip()
    if tag:
        return tag
    try:
        return latest_output_dir(SINGLE_MAPPING_PIPELINE_PREFIX).name
    except FileNotFoundError:
        return f"{SINGLE_MAPPING_PIPELINE_PREFIX}{datetime.now().strftime('%Y%m%d')}_r1"
