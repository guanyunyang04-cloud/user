from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.tools.recent_model_protocol import (
    DEFAULT_RECENT_MODEL_ROOT_TAG,
    ensure_recent_model_matrix,
    format_num,
    format_pct,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a strongest-model verdict for the current daily_research scope "
            "using rolling formal latest-model artifacts plus independent recent validation."
        )
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_strongest_model_verdict_20260409_r1")
    parser.add_argument("--recent-end-date", default="")
    parser.add_argument("--recent-window-months", type=int, default=12)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--recent-model-root-tag", default=DEFAULT_RECENT_MODEL_ROOT_TAG)
    parser.add_argument("--force-rerun-recent-models", action="store_true")
    return parser.parse_args()


def _source_rows() -> list[dict[str, Any]]:
    return [
        {
            "entry_name": "short_expert_monthly_v1__monthly_first_formal",
            "profile_name": "short_expert_monthly_v1",
            "scope_label": "eligible_monthly_first_formal_liquid500",
            "family_key": "short_alpha",
            "checkpoint_selection_objective": "primary_monthly_robust_score",
            "source_csv": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48",
            "eligible_for_winner": True,
            "notes": "current strongest monthly-first short-horizon expert candidate under 3 formal windows",
        },
        {
            "entry_name": "state_liquidity_listwise_v1__monthly_first_formal",
            "profile_name": "state_liquidity_listwise_v1",
            "scope_label": "eligible_monthly_first_formal_liquid500",
            "family_key": "short_alpha",
            "checkpoint_selection_objective": "primary_monthly_robust_score",
            "source_csv": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48",
            "eligible_for_winner": True,
            "notes": "current strongest stable base model under the same monthly-first 3-window protocol",
        },
        {
            "entry_name": "baseline_current__monthly_first_formal",
            "profile_name": "baseline_current",
            "scope_label": "eligible_monthly_first_formal_liquid500",
            "family_key": "baseline",
            "checkpoint_selection_objective": "primary_monthly_robust_score",
            "source_csv": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48",
            "eligible_for_winner": True,
            "notes": "baseline reference inside the same monthly-first 3-window protocol",
        },
        {
            "entry_name": "state_liquidity_listwise_v1__annual_checkpoint_reference",
            "profile_name": "state_liquidity_listwise_v1",
            "scope_label": "reference_annual_checkpoint_formal_liquid500",
            "family_key": "short_alpha",
            "checkpoint_selection_objective": "primary_annual_return",
            "source_csv": OUTPUT_ROOT / "short_alpha_formal_head2head_20260409_recheck_r1" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_formal_head2head_20260409_recheck_r1",
            "eligible_for_winner": False,
            "notes": "reference only; same 3 windows but annual-checkpoint protocol, not the current monthly-first winner gate",
        },
        {
            "entry_name": "dynamic_graph_no_priors__cross_family_reference",
            "profile_name": "dynamic_graph_no_priors",
            "scope_label": "reference_cross_family_formal_liquid500",
            "family_key": "dynamic_graph",
            "checkpoint_selection_objective": "primary_annual_return",
            "source_csv": OUTPUT_ROOT / "dynamic_graph_liquid500_challenger_20260405_r1" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "dynamic_graph_liquid500_challenger_20260405_r1",
            "eligible_for_winner": False,
            "notes": "cross-family reference only; useful for scale, but not included in the current monthly-first winner gate",
        },
    ]


def _read_row(source: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(source["source_csv"])
    row = frame.loc[frame["profile_name"] == source["profile_name"]]
    if row.empty:
        raise KeyError(f"profile {source['profile_name']} not found in {source['source_csv']}")
    payload = row.iloc[0].to_dict()
    return {
        "entry_name": str(source["entry_name"]),
        "profile_name": str(source["profile_name"]),
        "scope_label": str(source["scope_label"]),
        "family_key": str(source["family_key"]),
        "checkpoint_selection_objective": str(source["checkpoint_selection_objective"]),
        "source_csv": str(source["source_csv"]),
        "source_root": str(source["source_root"]),
        "eligible_for_winner": bool(source["eligible_for_winner"]),
        "notes": str(source["notes"]),
        "window_count": int(payload.get("window_count", 0) or 0),
        "epoch_budget": int(payload.get("epoch_budget", payload.get("mean_epoch_budget", 0)) or 0),
        "mean_excess_annual_return": float(payload.get("mean_excess_annual_return", 0.0) or 0.0),
        "mean_excess_sharpe": float(payload.get("mean_excess_sharpe", 0.0) or 0.0),
        "mean_avg_turnover": float(payload.get("mean_avg_turnover", 0.0) or 0.0),
        "mean_excess_max_drawdown": float(payload.get("mean_excess_max_drawdown", 0.0) or 0.0),
        "mean_monthly_positive_ratio": float(payload.get("mean_monthly_positive_ratio", 0.0) or 0.0),
        "mean_monthly_median_excess_return": float(payload.get("mean_monthly_median_excess_return", 0.0) or 0.0),
        "worst_monthly_excess_return": float(payload.get("worst_monthly_excess_return", 0.0) or 0.0),
        "mean_monthly_top3_positive_share": float(payload.get("mean_monthly_top3_positive_share", 0.0) or 0.0),
        "max_monthly_negative_streak": int(payload.get("max_monthly_negative_streak", 0) or 0),
        "reused_window_count": int(payload.get("reused_window_count", 0) or 0),
    }


def _build_summary(rows: pd.DataFrame, recent_df: pd.DataFrame, recent_summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eligible = rows.loc[rows["eligible_for_winner"]].copy()
    eligible = eligible.sort_values(
        [
            "mean_excess_annual_return",
            "mean_excess_sharpe",
            "mean_monthly_positive_ratio",
            "worst_monthly_excess_return",
            "mean_avg_turnover",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No eligible rows found for strongest-model verdict.")

    winner = eligible.iloc[0].to_dict()
    runner_up = eligible.iloc[1].to_dict() if len(eligible) > 1 else {}
    references = rows.loc[~rows["eligible_for_winner"]].copy()
    references = references.sort_values(
        ["mean_excess_annual_return", "mean_excess_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)

    recent_lookup = recent_df.set_index("profile_name").to_dict("index")
    winner_recent = recent_lookup.get(str(winner["profile_name"]), {})
    runner_up_recent = recent_lookup.get(str(runner_up.get("profile_name", "")), {})
    recent_winner = recent_df.iloc[0].to_dict() if not recent_df.empty else {}
    promotable_winner = dict(winner)
    promotable_recent = recent_lookup.get(str(promotable_winner.get("profile_name", "")), {})
    formal_recent_agreement = str(winner["profile_name"]) == str(recent_summary.get("recent_winner_profile_name", ""))

    summary = {
        "scope": "liquid500 + execution_first + rolling formal 3 windows + as-of-window latest model + monthly-first checkpoint + window_count=3",
        "recent_validation_required": True,
        "recent_validation_complete": True,
        "recent_validation_protocol": "independent_recent_model_as_of_recent_start",
        "recent_start_date": str(recent_summary.get("recent_start_date", "")),
        "recent_end_date": str(recent_summary.get("recent_end_date", "")),
        "recent_train_end_date": str(recent_summary.get("recent_train_end_date", "")),
        "recent_model_root_tag": str(recent_summary.get("recent_model_root_tag", "")),
        "winner_entry_name": str(winner["entry_name"]),
        "winner_profile_name": str(winner["profile_name"]),
        "winner_source_root": str(winner["source_root"]),
        "winner_mean_excess_annual_return": float(winner["mean_excess_annual_return"]),
        "winner_mean_excess_sharpe": float(winner["mean_excess_sharpe"]),
        "winner_mean_monthly_positive_ratio": float(winner["mean_monthly_positive_ratio"]),
        "winner_mean_monthly_median_excess_return": float(winner["mean_monthly_median_excess_return"]),
        "winner_worst_monthly_excess_return": float(winner["worst_monthly_excess_return"]),
        "winner_recent": winner_recent,
        "runner_up_entry_name": str(runner_up.get("entry_name", "")),
        "runner_up_profile_name": str(runner_up.get("profile_name", "")),
        "runner_up_mean_excess_annual_return": float(runner_up.get("mean_excess_annual_return", 0.0) or 0.0),
        "runner_up_mean_excess_sharpe": float(runner_up.get("mean_excess_sharpe", 0.0) or 0.0),
        "runner_up_recent": runner_up_recent,
        "recent_winner_profile_name": str(recent_summary.get("recent_winner_profile_name", "")),
        "recent_winner": recent_winner,
        "promotable_winner_profile_name": str(promotable_winner.get("profile_name", "")),
        "promotable_rule": "formal_winner_direct_default_under_current_policy",
        "promotable_winner_recent": promotable_recent,
        "formal_recent_agreement": bool(formal_recent_agreement),
        "references": references.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Strongest Model Verdict",
        "",
        "## Scope",
        "- current winner gate: `liquid500 + execution_first + rolling formal 3 windows + as-of-window latest model + primary_monthly_robust_score + window_count=3`",
        "- every formal window must use the latest label-available model as of that window",
        "- recent validation remains mandatory companion evidence and must not be omitted",
        "- recent layer now uses an independent recent-start latest model instead of replaying a formal run",
        f"- current recent window: `{recent_summary.get('recent_start_date', '')} -> {recent_summary.get('recent_end_date', '')}`",
        f"- recent train_end: `{recent_summary.get('recent_train_end_date', '')}`",
        "",
        "## Formal Winner",
        f"- formal winner: `{winner['profile_name']}`",
        f"- source_root: `{winner['source_root']}`",
        f"- mean excess annual: `{format_pct(winner['mean_excess_annual_return'])}`",
        f"- mean excess Sharpe: `{format_num(winner['mean_excess_sharpe'])}`",
        f"- mean positive-month ratio: `{format_pct(winner['mean_monthly_positive_ratio'])}`",
        f"- mean monthly median excess: `{format_pct(winner['mean_monthly_median_excess_return'])}`",
        f"- worst month excess: `{format_pct(winner['worst_monthly_excess_return'])}`",
    ]
    if runner_up:
        lines.extend(
            [
                "",
                "## Formal Runner-Up",
                f"- runner_up: `{runner_up['profile_name']}`",
                f"- mean excess annual: `{format_pct(runner_up['mean_excess_annual_return'])}`",
                f"- mean excess Sharpe: `{format_num(runner_up['mean_excess_sharpe'])}`",
                f"- mean positive-month ratio: `{format_pct(runner_up['mean_monthly_positive_ratio'])}`",
                f"- mean monthly median excess: `{format_pct(runner_up['mean_monthly_median_excess_return'])}`",
                f"- worst month excess: `{format_pct(runner_up['worst_monthly_excess_return'])}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Recent Winner",
            f"- recent winner: `{recent_summary.get('recent_winner_profile_name', '')}`",
        ]
    )
    for _, recent_row in recent_df.iterrows():
        lines.append(
            f"- `{recent_row['profile_name']}`: recent excess annual `{format_pct(recent_row['recent_excess_annual_return'])}`, "
            f"recent excess Sharpe `{format_num(recent_row['recent_excess_sharpe'])}`, "
            f"recent positive-month ratio `{format_pct(recent_row['recent_positive_month_ratio'])}`, "
            f"recent monthly robust `{format_num(recent_row['recent_monthly_robust_score'])}`, "
            f"execution profile `{recent_row.get('execution_alignment_profile', '')}`"
        )

    lines.extend(
        [
            "",
            "## Promotable Winner",
            f"- promotable winner: `{promotable_winner.get('profile_name', '')}`",
            "- rule: `formal_winner_direct_default_under_current_policy`",
            f"- formal / recent agreement: `{formal_recent_agreement}`",
            f"- promotable recent monthly robust: `{format_num(promotable_recent.get('recent_monthly_robust_score', 0.0) or 0.0)}`",
        ]
    )

    if not references.empty:
        lines.extend(["", "## Reference Tracks"])
        for _, ref in references.iterrows():
            lines.append(
                f"- `{ref['entry_name']}`: excess annual `{format_pct(ref['mean_excess_annual_return'])}`, "
                f"excess Sharpe `{format_num(ref['mean_excess_sharpe'])}`, notes `{ref['notes']}`"
            )

    lines.extend(
        [
            "",
            "## Decision",
            f"- 当前 formal strongest research model 是 `{winner['profile_name']}`，recent 已按独立 recent-start 最新模型补齐。",
            f"- 当前 recent 12 个月 winner 是 `{recent_summary.get('recent_winner_profile_name', '')}`；它只回答最近一年兑现质量，不替代 formal strongest gate。",
            f"- 当前 promotable winner 单独展示为 `{promotable_winner.get('profile_name', '')}`；这样 formal / recent / default 三层语义不再混写。",
            f"- 按当前治理协议，后续默认执行若要变更，仍应先把 `{promotable_winner.get('profile_name', '')}` 物化成 latest-data `production full-fit + highest family budget`。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame([_read_row(source) for source in _source_rows()])
    rows.to_csv(output_dir / "verdict_leaderboard.csv", index=False, encoding="utf-8-sig")

    recent_profiles = sorted(rows.loc[rows["eligible_for_winner"], "profile_name"].astype(str).unique().tolist())
    notes_by_profile = {
        str(row["profile_name"]): str(row["notes"])
        for _, row in rows.loc[rows["eligible_for_winner"]].iterrows()
    }
    recent_df, recent_summary, recent_source_runs = ensure_recent_model_matrix(
        profile_names=recent_profiles,
        root_tag=str(args.recent_model_root_tag),
        recent_end_date=str(args.recent_end_date),
        recent_window_months=int(args.recent_window_months),
        python_executable=str(args.python_executable),
        force_rerun=bool(args.force_rerun_recent_models),
        notes_by_profile=notes_by_profile,
    )
    recent_df.to_csv(output_dir / "recent_model_scoreboard.csv", index=False, encoding="utf-8-sig")
    (output_dir / "recent_model_source_runs.json").write_text(
        json.dumps(recent_source_runs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _build_summary(rows, recent_df, recent_summary, output_dir)


if __name__ == "__main__":
    main()
