from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.tools.recent_model_protocol import (
    DEFAULT_RECENT_MODEL_ROOT_TAG,
    OUTPUT_ROOT,
    ensure_recent_model_matrix,
    format_num,
    format_pct,
)


DEFAULT_ROOT_TAG = "short_alpha_policy_v3_recent_eval_20260410_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recent 12-month validation for short_expert_policy_v3 using an independent "
            "recent-start latest-model protocol."
        )
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--recent-end-date", default="")
    parser.add_argument("--recent-window-months", type=int, default=12)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--recent-model-root-tag", default=DEFAULT_RECENT_MODEL_ROOT_TAG)
    parser.add_argument("--force-rerun-recent-models", action="store_true")
    return parser.parse_args()


def _notes_by_profile() -> dict[str, str]:
    return {
        "short_expert_monthly_v1": "current default strongest research model",
        "short_expert_policy_v2": "policy_v2 learned control branch",
        "short_expert_policy_v3": "policy_v3 constrained learned control branch",
        "state_liquidity_listwise_v1": "recent companion baseline",
    }


def _build_summary(recent_df: pd.DataFrame, recent_summary: dict[str, object], output_dir: Path) -> None:
    rows = recent_df.copy()
    rows.to_csv(output_dir / "recent_policy_scoreboard.csv", index=False, encoding="utf-8-sig")

    lookup = rows.set_index("profile_name").to_dict("index")
    current = lookup.get("short_expert_monthly_v1", {})
    policy_v2 = lookup.get("short_expert_policy_v2", {})
    policy_v3 = lookup.get("short_expert_policy_v3", {})
    companion = lookup.get("state_liquidity_listwise_v1", {})
    winner = rows.iloc[0].to_dict() if not rows.empty else {}

    summary = {
        "recent_validation_protocol": "independent_recent_model_as_of_recent_start",
        "recent_model_root_tag": str(recent_summary.get("recent_model_root_tag", "")),
        "recent_start_date": str(recent_summary.get("recent_start_date", "")),
        "recent_end_date": str(recent_summary.get("recent_end_date", "")),
        "recent_train_end_date": str(recent_summary.get("recent_train_end_date", "")),
        "recent_winner_profile_name": str(recent_summary.get("recent_winner_profile_name", "")),
        "policy_v3": policy_v3,
        "policy_v2": policy_v2,
        "current_default": current,
        "companion": companion,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_v3_robust = float(policy_v3.get("recent_monthly_robust_score", 0.0) or 0.0)
    policy_v2_robust = float(policy_v2.get("recent_monthly_robust_score", 0.0) or 0.0)
    current_robust = float(current.get("recent_monthly_robust_score", 0.0) or 0.0)
    companion_robust = float(companion.get("recent_monthly_robust_score", 0.0) or 0.0)

    lines = [
        "# Policy V3 Recent Eval",
        "",
        "## Scope",
        "- protocol: `independent recent-start latest model`",
        f"- recent model root tag: `{recent_summary.get('recent_model_root_tag', '')}`",
        f"- recent window: `{recent_summary.get('recent_start_date', '')} -> {recent_summary.get('recent_end_date', '')}`",
        f"- recent train_end: `{recent_summary.get('recent_train_end_date', '')}`",
        "- compared branches: `short_expert_monthly_v1`, `short_expert_policy_v2`, `short_expert_policy_v3`, `state_liquidity_listwise_v1`",
        "",
        "## Direct Answer",
        f"- recent winner: `{winner.get('profile_name', 'n/a')}`",
        f"- policy_v3 monthly robust: `{format_num(policy_v3_robust)}`",
        f"- delta vs policy_v2: `{format_num(policy_v3_robust - policy_v2_robust)}`",
        f"- delta vs current default: `{format_num(policy_v3_robust - current_robust)}`",
        f"- delta vs companion: `{format_num(policy_v3_robust - companion_robust)}`",
        "",
        "## Scoreboard",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"- `{row['profile_name']}`: recent excess annual `{format_pct(row['recent_excess_annual_return'])}`, "
            f"recent excess Sharpe `{format_num(row['recent_excess_sharpe'])}`, "
            f"recent positive-month ratio `{format_pct(row['recent_positive_month_ratio'])}`, "
            f"recent median monthly `{format_pct(row['recent_median_monthly_return'])}`, "
            f"recent worst month `{format_pct(row['recent_worst_monthly_return'])}`, "
            f"recent monthly robust `{format_num(row['recent_monthly_robust_score'])}`, "
            f"execution profile `{row.get('execution_alignment_profile', '')}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "- 这份 recent 结果只认独立 recent-start 模型，不再复用 formal run_dir 做 replay。",
            "- `policy_v3` 只有在同时打赢 `policy_v2`、current default 和 companion 时，才具备默认候选资格。",
            "- 如果 `policy_v3` 仍弱于 `policy_v2`，说明“再多学一点执行动作”这一步还没有转化成更强的最近一年月度兑现。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_names = [
        "short_expert_monthly_v1",
        "short_expert_policy_v2",
        "short_expert_policy_v3",
        "state_liquidity_listwise_v1",
    ]
    recent_df, recent_summary, recent_source_runs = ensure_recent_model_matrix(
        profile_names=profile_names,
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
