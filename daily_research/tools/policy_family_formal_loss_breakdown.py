from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_FORMAL_SUMMARY = OUTPUT_ROOT / "short_alpha_policy_v2_family_formal_review_20260411_r1" / "summary.json"
DEFAULT_RECENT_SUMMARY = OUTPUT_ROOT / "short_alpha_policy_v2_family_recent_eval_20260411_r1" / "summary.json"
DEFAULT_CONSTRAINED_SUMMARY = OUTPUT_ROOT / "short_alpha_policy_v2_family_constrained_execution_review_20260411_r2" / "summary.json"
DEFAULT_ROOT_TAG = "short_alpha_policy_family_formal_loss_breakdown_20260411_r1"
DEFAULT_FOCUS_PROFILES = "short_expert_policy_v2b,short_expert_policy_v2c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Break down where the current learned-control family wins or still loses across formal / constrained / recent."
    )
    parser.add_argument("--formal-summary", default=str(DEFAULT_FORMAL_SUMMARY))
    parser.add_argument("--recent-summary", default=str(DEFAULT_RECENT_SUMMARY))
    parser.add_argument("--constrained-summary", default=str(DEFAULT_CONSTRAINED_SUMMARY))
    parser.add_argument("--focus-profiles", default=DEFAULT_FOCUS_PROFILES)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _format_num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _profile_key(profile_name: str) -> str:
    return str(profile_name).replace("short_expert_", "")


def _recent_row(recent_summary: dict[str, Any], profile_name: str) -> dict[str, Any]:
    key = _profile_key(profile_name)
    payload = recent_summary.get(key)
    return payload if isinstance(payload, dict) else {}


def _formal_row_map(formal_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = formal_summary.get("family_rows", [])
    return {str(row.get("profile_name", "")): row for row in rows if isinstance(row, dict)}


def _build_diagnosis(
    *,
    profile_name: str,
    recent_row: dict[str, Any],
    formal_row: dict[str, Any],
    constrained_best: dict[str, Any],
    constrained_selected: dict[str, Any],
    current_reference: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    current_robust = _safe_float(current_reference.get("monthly_robust_score"))
    best_robust = _safe_float(constrained_best.get("monthly_robust_score"))
    selected_robust = _safe_float(constrained_selected.get("monthly_robust_score"))
    recent_profile = str(recent_row.get("execution_alignment_profile", "") or "")
    formal_profile = str(formal_row.get("execution_alignment_profile", "") or "")
    selected_profile = str(constrained_selected.get("profile_name", "") or "")
    best_profile = str(constrained_best.get("profile_name", "") or "")

    if best_profile and selected_profile and best_profile != selected_profile:
        notes.append(f"constrained best bridge moved from `{selected_profile}` to `{best_profile}`")
    if recent_profile and best_profile and recent_profile == best_profile and recent_profile != formal_profile:
        notes.append(f"recent-preferred bridge `{recent_profile}` transfers into constrained formal")
    if best_robust - selected_robust >= 0.005:
        notes.append(f"selected formal bridge leaves about `{_format_num(best_robust - selected_robust)}` monthly robust on the table")
    if best_robust >= current_robust:
        notes.append("constrained formal already clears current mainline")
    elif current_robust - best_robust <= 0.01:
        notes.append("constrained formal gap is narrow and likely bridge/control related")
    else:
        notes.append("constrained formal gap remains materially open")

    current_names = _safe_float(current_reference.get("avg_positive_count"))
    current_top1 = _safe_float(current_reference.get("avg_top1_weight"))
    current_hhi = _safe_float(current_reference.get("avg_hhi"))
    best_names = _safe_float(constrained_best.get("aligned_avg_positive_count"))
    best_top1 = _safe_float(constrained_best.get("aligned_avg_top1_weight"))
    best_hhi = _safe_float(constrained_best.get("aligned_avg_hhi"))
    if current_names > 0 and best_names > 0 and best_names < current_names * 0.6:
        notes.append("best constrained answer is much more concentrated than current mainline")
    if current_top1 > 0 and best_top1 > current_top1 * 2.0:
        notes.append("top-1 weight is materially above current mainline")
    if current_hhi > 0 and best_hhi > current_hhi * 2.0:
        notes.append("portfolio HHI is materially above current mainline")

    if profile_name.endswith("policy_v2b") and best_profile == "regoff_k1_20d_ensemble_native_anchor":
        notes.append("v2b looks like a stable slow-bridge winner rather than a fast-bridge chaser")
    if profile_name.endswith("policy_v2c") and best_profile == "regoff_k1_5d_ensemble_native_anchor":
        notes.append("v2c gains come from a faster bridge, but concentration stays high")
    return notes


def _next_design_hypotheses(focus_profiles: list[str]) -> list[str]:
    ordered = [str(name).strip() for name in focus_profiles if str(name).strip()]
    if not ordered:
        return [
            "Prioritize execution-stability regularization when recent and constrained formal disagree on bridge speed.",
            "Prioritize candidate-count / concentration regularization when deployable variants still rely on too few names.",
            "Only keep gross-teacher distillation if gross-control remains the main bottleneck after stability + concentration fixes.",
        ]
    labels = ordered + ["candidate_branch", "teacher_branch"]
    return [
        f"{labels[0]} should prioritize execution-stability regularization when recent and constrained formal disagree on bridge speed.",
        f"{labels[1]} should prioritize candidate-count / concentration regularization when the best constrained answer still relies on very few names or high top-1 weight.",
        f"{labels[2]} should only keep gross-teacher distillation if the family still loses mainly on gross-control after stability + concentration fixes.",
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_summary_path = Path(args.formal_summary).resolve()
    recent_summary_path = Path(args.recent_summary).resolve()
    constrained_summary_path = Path(args.constrained_summary).resolve()
    formal_summary = _load_json(formal_summary_path)
    recent_summary = _load_json(recent_summary_path)
    constrained_summary = _load_json(constrained_summary_path)
    if not formal_summary or not recent_summary or not constrained_summary:
        raise FileNotFoundError("formal/recent/constrained summary is missing.")

    formal_rows = _formal_row_map(formal_summary)
    focus_profiles = [part.strip() for part in str(args.focus_profiles).split(",") if part.strip()]
    current_reference = constrained_summary.get("current_mainline_reference", {})
    analyses: list[dict[str, Any]] = []

    for profile_name in focus_profiles:
        formal_row = formal_rows.get(profile_name, {})
        recent_row = _recent_row(recent_summary, profile_name)
        constrained_model = constrained_summary.get("model_summaries", {}).get(profile_name, {})
        constrained_best = constrained_model.get("best_variant", {})
        constrained_selected = constrained_model.get("selected_formal_variant", {})
        if not formal_row or not recent_row or not constrained_best:
            continue
        analyses.append(
            {
                "profile_name": profile_name,
                "formal_row": formal_row,
                "recent_row": recent_row,
                "constrained_best": constrained_best,
                "constrained_selected": constrained_selected,
                "delta_recent_minus_formal": _safe_float(recent_row.get("recent_monthly_robust_score")) - _safe_float(formal_row.get("monthly_robust_score")),
                "delta_constrained_minus_formal": _safe_float(constrained_best.get("monthly_robust_score")) - _safe_float(formal_row.get("monthly_robust_score")),
                "delta_constrained_minus_current": _safe_float(constrained_best.get("monthly_robust_score")) - _safe_float(current_reference.get("monthly_robust_score")),
                "diagnosis": _build_diagnosis(
                    profile_name=profile_name,
                    recent_row=recent_row,
                    formal_row=formal_row,
                    constrained_best=constrained_best,
                    constrained_selected=constrained_selected,
                    current_reference=current_reference,
                ),
            }
        )

    summary_payload = {
        "formal_summary_path": str(formal_summary_path),
        "recent_summary_path": str(recent_summary_path),
        "constrained_summary_path": str(constrained_summary_path),
        "current_mainline_reference": current_reference,
        "family_best_constrained_variant": constrained_summary.get("family_best_variant", {}),
        "analyses": analyses,
        "next_design_hypotheses": _next_design_hypotheses(focus_profiles),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Policy Family Formal Loss Breakdown",
        "",
        "## Direct Answer",
        f"- current mainline formal robust: `{_format_num(current_reference.get('monthly_robust_score'))}`",
        f"- family best constrained variant: `{constrained_summary.get('family_best_variant', {}).get('variant_name', 'n/a')}`",
        f"- family best constrained robust: `{_format_num(constrained_summary.get('family_best_variant', {}).get('monthly_robust_score'))}`",
        f"- delta vs current mainline: `{_format_num(_safe_float(constrained_summary.get('family_best_variant', {}).get('monthly_robust_score')) - _safe_float(current_reference.get('monthly_robust_score')))}`",
        "",
        "## Focus Profiles",
    ]
    for analysis in analyses:
        formal_row = analysis["formal_row"]
        recent_row = analysis["recent_row"]
        constrained_best = analysis["constrained_best"]
        constrained_selected = analysis["constrained_selected"]
        lines.append(
            f"- `{analysis['profile_name']}`: formal `{_format_num(formal_row.get('monthly_robust_score'))}` "
            f"@ `{formal_row.get('execution_alignment_profile', 'n/a')}`, recent `{_format_num(recent_row.get('recent_monthly_robust_score'))}` "
            f"@ `{recent_row.get('execution_alignment_profile', 'n/a')}`, constrained best "
            f"`{_format_num(constrained_best.get('monthly_robust_score'))}` @ `{constrained_best.get('profile_name', 'n/a')}`, "
            f"selected constrained `{_format_num(constrained_selected.get('monthly_robust_score'))}`"
        )
        for note in analysis["diagnosis"]:
            lines.append(f"  - {note}")
        lines.append(
            f"  - concentration readout: aligned names `{_format_num(constrained_best.get('aligned_avg_positive_count'))}`, "
            f"top1 `{_format_pct(constrained_best.get('aligned_avg_top1_weight'))}`, "
            f"top2 `{_format_pct(constrained_best.get('aligned_avg_top2_share'))}`, "
            f"HHI `{_format_num(constrained_best.get('aligned_avg_hhi'))}`"
        )
    lines.extend(
        [
            "",
            "## Next Design Hypotheses",
        ]
    )
    for note in _next_design_hypotheses(focus_profiles):
        lines.append(f"- {note}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
