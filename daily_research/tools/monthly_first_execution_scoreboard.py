from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"

DEFAULT_VERDICT_ROOT = OUTPUT_ROOT / "short_alpha_execution_semantic_concentration_verdict_20260409_r1"
DEFAULT_RECENT_AUDIT_ROOT = OUTPUT_ROOT / "short_alpha_production_execution_policy_audit_20260409_r2"
DEFAULT_TARGETED_REVIEW_ROOT = (
    OUTPUT_ROOT / "short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3"
)

COMMON_VARIANT_LABELS = {
    "raw_regoff_k1_5d_ensemble_native_anchor": "raw + k1 5d static",
    "raw_regoff_k2_5d_ensemble_native_anchor": "raw + k2 5d static",
    "raw_regoff_k3_5d_ensemble_native_anchor": "raw + k3 5d static",
}
RECENT_PROFILE_LABELS = {
    "regoff_k1_5d_ensemble_native_anchor": "k1 5d static",
    "regoff_k2_5d_ensemble_native_anchor": "k2 5d static",
    "regoff_k3_5d_ensemble_native_anchor": "k3 5d static",
    "topk3_1d_regoff": "topk3 1d regoff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a monthly-first execution scoreboard for the current short_alpha main line."
    )
    parser.add_argument("--verdict-root", default=str(DEFAULT_VERDICT_ROOT))
    parser.add_argument("--recent-audit-root", default=str(DEFAULT_RECENT_AUDIT_ROOT))
    parser.add_argument("--targeted-review-root", default=str(DEFAULT_TARGETED_REVIEW_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_monthly_first_execution_scoreboard_20260409_r1")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _monthly_robust_score(diag: dict[str, Any]) -> float:
    positive_ratio = float(diag.get("positive_month_ratio", 0.0) or 0.0)
    median_monthly_return = float(diag.get("median_monthly_return", 0.0) or 0.0)
    mean_monthly_return = float(diag.get("mean_monthly_return", 0.0) or 0.0)
    worst_monthly_return = float(diag.get("worst_monthly_return", 0.0) or 0.0)
    top3_positive_share = float(diag.get("top3_positive_month_share", 0.0) or 0.0)
    longest_negative_streak = int(diag.get("longest_negative_streak", 0) or 0)
    downside_penalty = max(-worst_monthly_return, 0.0)
    concentration_penalty = max(top3_positive_share - 0.60, 0.0)
    streak_penalty = max(longest_negative_streak - 2, 0)
    return float(
        mean_monthly_return
        + median_monthly_return
        + 0.05 * (positive_ratio - 0.50)
        - 0.35 * downside_penalty
        - 0.05 * concentration_penalty
        - 0.01 * float(streak_penalty)
    )


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _build_common_window_scoreboard(verdict_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(verdict_root / "variant_summary.csv")
    rows: list[dict[str, Any]] = []
    for variant_name, label in COMMON_VARIANT_LABELS.items():
        match = frame.loc[frame["variant_name"].astype(str) == variant_name]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        run_dir = Path(str(row["run_dir"])).resolve()
        monthly_diag = _load_json(run_dir / "monthly_backtest_diagnostics.json")
        rows.append(
            {
                "variant_name": variant_name,
                "label": label,
                "annual_return": float(row.get("annual_return", 0.0) or 0.0),
                "excess_annual_return": float(row.get("excess_annual_return", 0.0) or 0.0),
                "excess_sharpe": float(row.get("excess_sharpe", 0.0) or 0.0),
                "avg_turnover": float(row.get("avg_turnover", 0.0) or 0.0),
                "positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
                "median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
                "mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
                "worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
                "longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
                "top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
                "monthly_robust_score": _monthly_robust_score(monthly_diag),
                "march_excess_return": float(row.get("march_excess_return", 0.0) or 0.0),
                "april_excess_return": float(row.get("april_excess_return", 0.0) or 0.0),
                "march_action_count": int(float(row.get("march_action_count", 0.0) or 0.0)),
                "april_action_count": int(float(row.get("april_action_count", 0.0) or 0.0)),
                "mean_max_name_weight": float(row.get("mean_max_name_weight", 0.0) or 0.0),
                "recent20_mean_max_name_weight": float(row.get("recent20_mean_max_name_weight", 0.0) or 0.0),
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["monthly_robust_score", "mean_monthly_return", "excess_annual_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_recent_window_scoreboard(recent_audit_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(recent_audit_root / "execution_policy_summary.csv")
    frame = frame.loc[frame["profile_name"].astype(str).isin(RECENT_PROFILE_LABELS.keys())].copy()
    frame["label"] = frame["profile_name"].map(RECENT_PROFILE_LABELS)
    return frame.sort_values(
        by=["monthly_robust_score", "mean_monthly_return", "excess_annual_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_targeted_scoreboard(targeted_review_root: Path) -> tuple[pd.DataFrame, list[str]]:
    compare = pd.read_csv(targeted_review_root / "window_compare.csv")
    compare = compare[
        [
            "window_key",
            "selected_plan_label",
            "targeted_excess_annual_return",
            "static_excess_annual_return",
            "delta_excess_annual_return",
            "targeted_excess_sharpe",
            "static_excess_sharpe",
            "delta_excess_sharpe",
            "targeted_avg_turnover",
            "static_avg_turnover",
            "delta_avg_turnover",
            "targeted_run_dir",
            "static_run_dir",
            "static_source_run_dir",
        ]
    ].copy()
    compare.insert(1, "window_type", "formal_window")

    recent_gate = _load_json(targeted_review_root / "recent_gate_summary.json")
    if recent_gate:
        compare = pd.concat(
            [
                compare,
                pd.DataFrame(
                    [
                        {
                            "window_key": str(recent_gate.get("window_key", "")),
                            "window_type": "recent_gate",
                            "selected_plan_label": str(recent_gate.get("selected_plan_label", "")),
                            "targeted_excess_annual_return": float(recent_gate.get("targeted_excess_annual_return", 0.0) or 0.0),
                            "static_excess_annual_return": float(recent_gate.get("static_excess_annual_return", 0.0) or 0.0),
                            "delta_excess_annual_return": float(recent_gate.get("delta_excess_annual_return", 0.0) or 0.0),
                            "targeted_excess_sharpe": float(recent_gate.get("targeted_excess_sharpe", 0.0) or 0.0),
                            "static_excess_sharpe": float(recent_gate.get("static_excess_sharpe", 0.0) or 0.0),
                            "delta_excess_sharpe": float(recent_gate.get("delta_excess_sharpe", 0.0) or 0.0),
                            "targeted_avg_turnover": float("nan"),
                            "static_avg_turnover": float("nan"),
                            "delta_avg_turnover": float("nan"),
                            "targeted_run_dir": str(recent_gate.get("targeted_run_dir", "")),
                            "static_run_dir": str(recent_gate.get("static_run_dir", "")),
                            "static_source_run_dir": str(recent_gate.get("static_source_run_dir", "")),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    recent_month_choices = targeted_review_root / "recent_month_policy_choices.csv"
    recent_selected_profiles: list[str] = []
    if recent_month_choices.exists():
        recent_choice_df = pd.read_csv(recent_month_choices)
        recent_selected_profiles = sorted(
            {
                str(item).strip()
                for item in recent_choice_df["selected_profile"].astype(str)
                if str(item).strip()
            }
        )
    return compare.reset_index(drop=True), recent_selected_profiles


def _write_summary(
    output_dir: Path,
    *,
    common_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    targeted_df: pd.DataFrame,
    targeted_summary: dict[str, Any],
    recent_selected_profiles: list[str],
) -> None:
    common_winner = common_df.iloc[0].to_dict() if not common_df.empty else {}
    recent_winner = recent_df.iloc[0].to_dict() if not recent_df.empty else {}
    targeted_formal = targeted_df.loc[targeted_df["window_type"].astype(str) == "formal_window"].copy()
    recent_gate = targeted_df.loc[targeted_df["window_type"].astype(str) == "recent_gate"].copy()

    summary = {
        "common_window_winner": str(common_winner.get("variant_name", "")),
        "recent_window_winner": str(recent_winner.get("profile_name", "")),
        "targeted_formal_window_count": int(len(targeted_formal)),
        "targeted_annual_wins": int((targeted_formal["delta_excess_annual_return"] > 0.0).sum()) if not targeted_formal.empty else 0,
        "targeted_sharpe_wins": int((targeted_formal["delta_excess_sharpe"] > 0.0).sum()) if not targeted_formal.empty else 0,
        "targeted_recent_delta_excess_annual_return": 0.0 if recent_gate.empty else float(recent_gate.iloc[0]["delta_excess_annual_return"]),
        "targeted_recent_delta_excess_sharpe": 0.0 if recent_gate.empty else float(recent_gate.iloc[0]["delta_excess_sharpe"]),
        "recent_selected_profiles": recent_selected_profiles,
        "targeted_review_selection_objective": str(targeted_summary.get("selection_objective", "")),
        "targeted_review_root": str(targeted_summary.get("review_root", "")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    k2_common = common_df.loc[common_df["variant_name"].eq("raw_regoff_k2_5d_ensemble_native_anchor")].iloc[0]
    k1_common = common_df.loc[common_df["variant_name"].eq("raw_regoff_k1_5d_ensemble_native_anchor")].iloc[0]
    k3_common = common_df.loc[common_df["variant_name"].eq("raw_regoff_k3_5d_ensemble_native_anchor")].iloc[0]
    k2_recent = recent_df.loc[recent_df["profile_name"].eq("regoff_k2_5d_ensemble_native_anchor")].iloc[0]
    topk3_recent = recent_df.loc[recent_df["profile_name"].eq("topk3_1d_regoff")].iloc[0]

    lines = [
        "# Monthly-First Execution Scoreboard",
        "",
        "## Direct Answer",
        f"- common-window winner: `{common_winner.get('variant_name', 'n/a')}` ({common_winner.get('label', 'n/a')})",
        f"- recent-window winner: `{recent_winner.get('profile_name', 'n/a')}` ({recent_winner.get('label', 'n/a')})",
        f"- targeted repair formal annual wins: `{summary['targeted_annual_wins']}/{summary['targeted_formal_window_count']}`",
        f"- targeted repair formal Sharpe wins: `{summary['targeted_sharpe_wins']}/{summary['targeted_formal_window_count']}`",
        f"- targeted recent delta excess annual: `{_pct(summary['targeted_recent_delta_excess_annual_return'])}`",
        f"- targeted recent delta excess Sharpe: `{_num(summary['targeted_recent_delta_excess_sharpe'])}`",
        "",
        "## Common Window",
        f"- `k2` monthly_robust_score: `{_num(k2_common['monthly_robust_score'])}`",
        f"- `k1` monthly_robust_score: `{_num(k1_common['monthly_robust_score'])}`",
        f"- `k3` monthly_robust_score: `{_num(k3_common['monthly_robust_score'])}`",
        f"- `k2` recent20 mean max weight: `{_pct(k2_common['recent20_mean_max_name_weight'])}`",
        "",
        "## Recent Window",
        "- recent audit ranks `k2 > k1 > k3 > topk3` by `monthly_robust_score`.",
        f"- `k2` recent monthly_robust_score: `{_num(k2_recent['monthly_robust_score'])}`",
        f"- `topk3` recent monthly_robust_score: `{_num(topk3_recent['monthly_robust_score'])}`",
        "",
        "## Targeted Review",
        "- this review uses same-window clipped static replays before comparing targeted vs static, so it avoids the old one-day mismatch.",
        "- full mapping stayed `trend_up_low_vol|expand|stable -> topk3_1d_regoff`, but it did not beat static `k2` on the leave-window-out formal comparison.",
        f"- recent selected profiles: `{','.join(recent_selected_profiles) if recent_selected_profiles else 'n/a'}`",
        "",
        "## Conclusion",
        "- keep `regoff_k2_5d_ensemble_native_anchor` as the execution-side short-line mainline.",
        "- downgrade `trend_up_low_vol|expand|stable -> topk3_1d_regoff` to observation-only until new same-window monthly-first evidence appears.",
        "- next execution work should move into signal-to-weight or month-trigger design, not more simple regime swapping.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    verdict_root = Path(args.verdict_root).resolve()
    recent_audit_root = Path(args.recent_audit_root).resolve()
    targeted_review_root = Path(args.targeted_review_root).resolve()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_df = _build_common_window_scoreboard(verdict_root)
    recent_df = _build_recent_window_scoreboard(recent_audit_root)
    targeted_df, recent_selected_profiles = _build_targeted_scoreboard(targeted_review_root)
    targeted_summary = _load_json(targeted_review_root / "summary.json")

    common_df.to_csv(output_dir / "common_window_scoreboard.csv", index=False, encoding="utf-8-sig")
    recent_df.to_csv(output_dir / "recent_window_scoreboard.csv", index=False, encoding="utf-8-sig")
    targeted_df.to_csv(output_dir / "targeted_vs_k2_scoreboard.csv", index=False, encoding="utf-8-sig")
    _write_summary(
        output_dir,
        common_df=common_df,
        recent_df=recent_df,
        targeted_df=targeted_df,
        targeted_summary=targeted_summary,
        recent_selected_profiles=recent_selected_profiles,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "common_window_winner": "" if common_df.empty else str(common_df.iloc[0]["variant_name"]),
                "recent_window_winner": "" if recent_df.empty else str(recent_df.iloc[0]["profile_name"]),
                "targeted_review_root": str(targeted_review_root),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
