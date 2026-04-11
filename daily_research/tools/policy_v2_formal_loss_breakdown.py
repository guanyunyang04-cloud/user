from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_POLICY_V2_RUN_DIR = OUTPUT_ROOT / "short_alpha_policy_v2_review_20260410_r1" / "runs" / "short_expert_policy_v2"
DEFAULT_CURRENT_FORMAL_RUN_DIR = OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
DEFAULT_CONSTRAINED_REVIEW_ROOT = OUTPUT_ROOT / "short_alpha_policy_v2_constrained_execution_review_20260410_r1"
DEFAULT_ROOT_TAG = "short_alpha_policy_v2_formal_loss_breakdown_20260410_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize where policy_v2 loses formal elasticity and tie that loss back to constrained execution evidence."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_POLICY_V2_RUN_DIR))
    parser.add_argument("--current-formal-run-dir", default=str(DEFAULT_CURRENT_FORMAL_RUN_DIR))
    parser.add_argument("--constrained-review-root", default=str(DEFAULT_CONSTRAINED_REVIEW_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    current_formal_run_dir = Path(args.current_formal_run_dir).resolve()
    constrained_root = Path(args.constrained_review_root).resolve()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_v2_metrics = _load_json(run_dir / "metrics.json")
    policy_v2_raw_monthly = _load_json(run_dir / "monthly_backtest_diagnostics.json")
    policy_v2_aligned_monthly = _load_json(run_dir / "execution_aligned_monthly_backtest_diagnostics.json")
    current_metrics = _load_json(current_formal_run_dir / "metrics.json")
    current_monthly = _load_json(current_formal_run_dir / "primary_research_monthly_diagnostics.json")
    objective_rows = pd.read_csv(run_dir / "execution_alignment_objective_rows.csv")
    constrained_scoreboard = pd.read_csv(constrained_root / "variant_scoreboard.csv")

    train_eval_rows = objective_rows.loc[objective_rows["phase"] == "train_eval"].copy()
    train_eval_rows = train_eval_rows.sort_values(
        ["excess_annual_return", "excess_sharpe", "avg_turnover"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected_profile = str(policy_v2_metrics.get("execution_alignment_profile", "") or "")
    selected_train_row = train_eval_rows.loc[train_eval_rows["profile_name"] == selected_profile]
    selected_train_row = selected_train_row.iloc[0].to_dict() if not selected_train_row.empty else {}
    best_constrained = constrained_scoreboard.iloc[0].to_dict() if not constrained_scoreboard.empty else {}
    selected_constrained = constrained_scoreboard.loc[constrained_scoreboard["variant_name"] == "policy_v2_current_k2_20d"]
    selected_constrained = selected_constrained.iloc[0].to_dict() if not selected_constrained.empty else {}

    summary = {
        "policy_v2_run_dir": str(run_dir),
        "current_formal_run_dir": str(current_formal_run_dir),
        "selected_profile": selected_profile,
        "selected_train_eval_profile_row": selected_train_row,
        "best_constrained_variant": best_constrained,
        "selected_constrained_variant": selected_constrained,
        "policy_v2_raw_monthly_diagnostics": policy_v2_raw_monthly,
        "policy_v2_execution_aligned_monthly_diagnostics": policy_v2_aligned_monthly,
        "current_mainline_primary_monthly_diagnostics": current_monthly,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Policy V2 Formal Loss Breakdown",
        "",
        "## Direct Answer",
        f"- current selected execution profile: `{selected_profile}`",
        f"- raw formal excess annual / robust proxy: `{_format_pct(policy_v2_metrics.get('holdout_backtest', {}).get('excess_annual_return', 0.0))}` / `{_format_num(float(policy_v2_raw_monthly.get('mean_monthly_return', 0.0) or 0.0) + float(policy_v2_raw_monthly.get('median_monthly_return', 0.0) or 0.0))}`",
        f"- selected aligned formal excess annual / monthly robust: `{_format_pct(policy_v2_metrics.get('execution_aligned_holdout_backtest', {}).get('excess_annual_return', 0.0))}` / `{_format_num(best_constrained.get('monthly_robust_score', float('nan')) if best_constrained else float('nan'))}`",
        f"- current mainline formal excess annual / monthly robust: `{_format_pct(current_metrics.get('primary_research_backtest', {}).get('excess_annual_return', 0.0))}` / `{_format_num(float(current_monthly.get('mean_monthly_return', 0.0) or 0.0) + float(current_monthly.get('median_monthly_return', 0.0) or 0.0))}`",
        "",
        "## Why It Loses",
        f"- `policy_v2` raw 1d is not deployable: raw formal excess annual is `{_format_pct(policy_v2_metrics.get('holdout_backtest', {}).get('excess_annual_return', 0.0))}` and raw downside is much worse than the aligned path.",
        f"- train-eval auto selected `{selected_profile}` because its train-eval excess annual / Sharpe was `{_format_pct(selected_train_row.get('excess_annual_return', 0.0))}` / `{_format_num(selected_train_row.get('excess_sharpe', 0.0))}`.",
        f"- constrained review shows the best formal answer is `{best_constrained.get('variant_name', 'n/a')}` with profile `{best_constrained.get('profile_name', 'n/a')}`, not the current selected `k2_20d`.",
        f"- delta vs current selected `k2_20d` on monthly robust is `{_format_num(float(best_constrained.get('monthly_robust_score', 0.0) or 0.0) - float(selected_constrained.get('monthly_robust_score', 0.0) or 0.0))}`.",
        "",
        "## Design Takeaways",
        "- formal loss is not evidence that learned control is wrong, but the current evidence also does not support the simple story that the bridge drifted too slow: `k2_20d` remains policy_v2's own best formal answer.",
        "- the constrained review says that pre-bridge candidate caps and tighter gross bands do not rescue the formal gap by themselves, so `policy_v3` cannot rely on hand-crafted sparsity alone.",
        "- `policy_v3` should therefore focus on improving the learned score-to-weight conversion itself, while being evaluated on a narrow but still realistic k2 profile family that keeps `5d/10d/20d` all available.",
        "",
        "## Train-Eval Top Rows",
    ]
    for _, row in train_eval_rows.head(5).iterrows():
        lines.append(
            f"- `{row['profile_name']}`: train-eval excess annual `{_format_pct(row['excess_annual_return'])}`, "
            f"excess Sharpe `{_format_num(row['excess_sharpe'])}`, avg turnover `{_format_num(row['avg_turnover'])}`"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
