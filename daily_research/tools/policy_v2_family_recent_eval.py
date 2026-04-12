from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.tools.recent_model_protocol import (
    DEFAULT_RECENT_MODEL_ROOT_TAG,
    OUTPUT_ROOT,
    ensure_recent_model_matrix,
    format_num,
    format_pct,
)


DEFAULT_ROOT_TAG = "short_alpha_policy_v2_family_recent_eval_20260411_r1"
PROFILE_NAMES = [
    "baseline_current",
    "short_expert_monthly_v1",
    "short_expert_policy_v2",
    "short_expert_policy_v2a",
    "short_expert_policy_v2b",
    "short_expert_policy_v2c",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recent 12-month evaluation for the narrow policy_v2 family under the independent recent-start protocol."
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--recent-end-date", default="")
    parser.add_argument("--recent-window-months", type=int, default=12)
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--recent-model-root-tag", default=DEFAULT_RECENT_MODEL_ROOT_TAG)
    parser.add_argument("--force-rerun-recent-models", action="store_true")
    return parser.parse_args()


def _notes_by_profile() -> dict[str, str]:
    return {
        "baseline_current": "recent winner baseline reference",
        "short_expert_monthly_v1": "current default strongest research model",
        "short_expert_policy_v2": "existing policy_v2 learned control branch",
        "short_expert_policy_v2a": "policy_v2a tighter candidate branch",
        "short_expert_policy_v2b": "policy_v2b tighter gross branch",
        "short_expert_policy_v2c": "policy_v2c hold-aware control branch",
    }


def _build_summary(recent_df: pd.DataFrame, recent_summary: dict[str, object], output_dir: Path) -> None:
    recent_df.to_csv(output_dir / "recent_family_scoreboard.csv", index=False, encoding="utf-8-sig")

    lookup = recent_df.set_index("profile_name").to_dict("index")
    baseline = lookup.get("baseline_current", {})
    current = lookup.get("short_expert_monthly_v1", {})
    policy_v2 = lookup.get("short_expert_policy_v2", {})
    policy_v2a = lookup.get("short_expert_policy_v2a", {})
    policy_v2b = lookup.get("short_expert_policy_v2b", {})
    policy_v2c = lookup.get("short_expert_policy_v2c", {})
    winner = recent_df.iloc[0].to_dict() if not recent_df.empty else {}

    family_rows = recent_df.loc[
        recent_df["profile_name"].isin(["short_expert_policy_v2", "short_expert_policy_v2a", "short_expert_policy_v2b", "short_expert_policy_v2c"])
    ].copy()
    family_winner = family_rows.iloc[0].to_dict() if not family_rows.empty else {}

    summary = {
        "recent_validation_protocol": "independent_recent_model_as_of_recent_start",
        "recent_model_root_tag": str(recent_summary.get("recent_model_root_tag", "")),
        "recent_start_date": str(recent_summary.get("recent_start_date", "")),
        "recent_end_date": str(recent_summary.get("recent_end_date", "")),
        "recent_requested_end_date": str(recent_summary.get("recent_requested_end_date", "")),
        "recent_validation_end_date": str(recent_summary.get("recent_validation_end_date", "")),
        "recent_train_end_date": str(recent_summary.get("recent_train_end_date", "")),
        "recent_winner_profile_name": str(recent_summary.get("recent_winner_profile_name", "")),
        "family_recent_winner_profile_name": str(family_winner.get("profile_name", "")),
        "baseline_current": baseline,
        "current_default": current,
        "policy_v2": policy_v2,
        "policy_v2a": policy_v2a,
        "policy_v2b": policy_v2b,
        "policy_v2c": policy_v2c,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Policy V2 Family Recent Eval",
        "",
        "## Scope",
        "- protocol: `independent recent-start latest model`",
        f"- recent model root tag: `{recent_summary.get('recent_model_root_tag', '')}`",
        f"- requested recent cutoff: `{recent_summary.get('recent_requested_end_date', '')}`",
        f"- effective validation window: `{recent_summary.get('recent_start_date', '')} -> {recent_summary.get('recent_end_date', '')}`",
        f"- recent train_end: `{recent_summary.get('recent_train_end_date', '')}`",
        "- compared profiles: `baseline_current`, `short_expert_monthly_v1`, `short_expert_policy_v2`, `short_expert_policy_v2a`, `short_expert_policy_v2b`, `short_expert_policy_v2c`",
        "",
        "## Direct Answer",
        f"- overall recent winner: `{winner.get('profile_name', 'n/a')}`",
        f"- family recent winner: `{family_winner.get('profile_name', 'n/a')}`",
        f"- family winner monthly robust: `{format_num(family_winner.get('recent_monthly_robust_score'))}`",
        f"- delta vs existing policy_v2: `{format_num(float(family_winner.get('recent_monthly_robust_score', 0.0) or 0.0) - float(policy_v2.get('recent_monthly_robust_score', 0.0) or 0.0))}`",
        f"- delta vs current default: `{format_num(float(family_winner.get('recent_monthly_robust_score', 0.0) or 0.0) - float(current.get('recent_monthly_robust_score', 0.0) or 0.0))}`",
        f"- delta vs baseline_current: `{format_num(float(family_winner.get('recent_monthly_robust_score', 0.0) or 0.0) - float(baseline.get('recent_monthly_robust_score', 0.0) or 0.0))}`",
        "",
        "## Scoreboard",
    ]
    for _, row in recent_df.iterrows():
        lines.append(
            f"- `{row['profile_name']}`: recent excess annual `{format_pct(row['recent_excess_annual_return'])}`, "
            f"recent excess Sharpe `{format_num(row['recent_excess_sharpe'])}`, "
            f"recent positive-month `{format_pct(row['recent_positive_month_ratio'])}`, "
            f"recent median monthly `{format_pct(row['recent_median_monthly_return'])}`, "
            f"recent worst month `{format_pct(row['recent_worst_monthly_return'])}`, "
            f"recent monthly robust `{format_num(row['recent_monthly_robust_score'])}`, "
            f"execution profile `{row.get('execution_alignment_profile', '')}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "- The next learned-control branch must improve or at least preserve policy_v2's recent advantage while meaningfully shrinking the formal gap to the current mainline.",
            "- Beating `short_expert_monthly_v1` is necessary for relevance; materially closing the gap to `baseline_current` is the stronger target.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    recent_df, recent_summary, recent_source_runs = ensure_recent_model_matrix(
        profile_names=PROFILE_NAMES,
        root_tag=str(args.recent_model_root_tag),
        recent_end_date=str(args.recent_end_date),
        recent_window_months=int(args.recent_window_months),
        python_executable=str(args.python_executable),
        force_rerun=bool(args.force_rerun_recent_models),
        notes_by_profile=_notes_by_profile(),
    )
    (output_dir / "recent_model_source_runs.json").write_text(
        json.dumps(recent_source_runs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _build_summary(recent_df, recent_summary, output_dir)


if __name__ == "__main__":
    main()
