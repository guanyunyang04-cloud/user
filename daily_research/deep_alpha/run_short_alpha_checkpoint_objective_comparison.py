from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
SHORT_ALPHA_FORMAL_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_short_alpha_formal_head2head.py"

DEFAULT_ANNUAL_ROOT = "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1"
DEFAULT_MONTHLY_ROOT = "short_alpha_formal_head2head_20260405_monthly_checkpoint_r1"
DEFAULT_COMPARISON_ROOT = "short_alpha_checkpoint_objective_comparison_20260405_r1"

BASELINE_PROFILE = "baseline_current"
CANDIDATE_PROFILE = "state_liquidity_listwise_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the liquid500 short-alpha main line under annual-return checkpoint selection "
            "vs monthly-robust checkpoint selection."
        )
    )
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--annual-root-tag", default=DEFAULT_ANNUAL_ROOT)
    parser.add_argument("--monthly-root-tag", default=DEFAULT_MONTHLY_ROOT)
    parser.add_argument("--comparison-root-tag", default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--rerun-annual", action="store_true")
    parser.add_argument("--rerun-monthly", action="store_true")
    parser.add_argument("--family-epoch-budget-manifest", default="daily_research/output/deep_alpha_family_epoch_budget_latest.json")
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _root_dir(root_tag: str) -> Path:
    return OUTPUT_ROOT / root_tag


def _build_formal_command(
    *,
    python_executable: str,
    root_tag: str,
    checkpoint_selection_objective: str,
    family_epoch_budget_manifest: str,
) -> list[str]:
    return [
        str(python_executable),
        str(SHORT_ALPHA_FORMAL_SCRIPT),
        "--python-executable",
        str(python_executable),
        "--root-tag",
        root_tag,
        "--family-epoch-budget-manifest",
        str(family_epoch_budget_manifest),
        "--checkpoint-selection-objective",
        checkpoint_selection_objective,
        "--research-objective-mode",
        "execution_first",
        "--execution-alignment-objective",
        "robust_composite",
    ]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _extract_profile_row(summary_df: pd.DataFrame, profile_name: str) -> pd.Series:
    frame = summary_df.loc[summary_df["profile_name"] == profile_name]
    if frame.empty:
        raise KeyError(f"Profile not found in summary: {profile_name}")
    return frame.iloc[0]


def _as_float(row: pd.Series, key: str) -> float:
    value = row.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_int(row: pd.Series, key: str) -> int:
    value = row.get(key, 0)
    try:
        return int(value)
    except Exception:
        return 0


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


def _collect_objective_row(
    *,
    objective_label: str,
    root_tag: str,
) -> dict[str, Any]:
    root_dir = _root_dir(root_tag)
    summary_df = _load_csv(root_dir / "profile_summary.csv")
    source_runs = _load_json(root_dir / "source_runs.json")
    candidate = _extract_profile_row(summary_df, CANDIDATE_PROFILE)
    baseline = _extract_profile_row(summary_df, BASELINE_PROFILE)
    return {
        "objective_label": objective_label,
        "root_tag": root_tag,
        "candidate_mean_excess_annual_return": _as_float(candidate, "mean_excess_annual_return"),
        "baseline_mean_excess_annual_return": _as_float(baseline, "mean_excess_annual_return"),
        "candidate_mean_excess_sharpe": _as_float(candidate, "mean_excess_sharpe"),
        "baseline_mean_excess_sharpe": _as_float(baseline, "mean_excess_sharpe"),
        "candidate_positive_month_ratio": _as_float(candidate, "mean_monthly_positive_ratio"),
        "baseline_positive_month_ratio": _as_float(baseline, "mean_monthly_positive_ratio"),
        "candidate_median_monthly_return": _as_float(candidate, "mean_monthly_median_excess_return"),
        "baseline_median_monthly_return": _as_float(baseline, "mean_monthly_median_excess_return"),
        "candidate_worst_monthly_return": _as_float(candidate, "worst_monthly_excess_return"),
        "baseline_worst_monthly_return": _as_float(baseline, "worst_monthly_excess_return"),
        "candidate_top3_positive_share": _as_float(candidate, "mean_monthly_top3_positive_share"),
        "baseline_top3_positive_share": _as_float(baseline, "mean_monthly_top3_positive_share"),
        "candidate_max_negative_streak": _as_int(candidate, "max_monthly_negative_streak"),
        "baseline_max_negative_streak": _as_int(baseline, "max_monthly_negative_streak"),
        "candidate_wins_annual": _as_int(candidate, "wins_by_excess_annual_return"),
        "candidate_wins_sharpe": _as_int(candidate, "wins_by_excess_sharpe"),
        "candidate_source_recent_run": str(source_runs.get(CANDIDATE_PROFILE, {}).get("20250318_20260331", "")),
    }


def _finalize_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    frame = summary_df.copy()
    frame["delta_mean_excess_annual_return"] = (
        frame["candidate_mean_excess_annual_return"] - frame["baseline_mean_excess_annual_return"]
    )
    frame["delta_mean_excess_sharpe"] = (
        frame["candidate_mean_excess_sharpe"] - frame["baseline_mean_excess_sharpe"]
    )
    frame["delta_positive_month_ratio"] = (
        frame["candidate_positive_month_ratio"] - frame["baseline_positive_month_ratio"]
    )
    frame["delta_median_monthly_return"] = (
        frame["candidate_median_monthly_return"] - frame["baseline_median_monthly_return"]
    )
    frame["delta_worst_monthly_return"] = (
        frame["candidate_worst_monthly_return"] - frame["baseline_worst_monthly_return"]
    )
    frame["delta_top3_positive_share"] = (
        frame["candidate_top3_positive_share"] - frame["baseline_top3_positive_share"]
    )
    frame = frame.sort_values(
        [
            "delta_positive_month_ratio",
            "delta_median_monthly_return",
            "delta_worst_monthly_return",
            "delta_top3_positive_share",
            "delta_mean_excess_sharpe",
            "delta_mean_excess_annual_return",
            "candidate_wins_annual",
            "candidate_wins_sharpe",
            "candidate_positive_month_ratio",
            "candidate_median_monthly_return",
        ],
        ascending=[False, False, False, True, False, False, False, False, False, False],
    ).reset_index(drop=True)
    return frame


def _write_summary(output_dir: Path, comparison_df: pd.DataFrame) -> None:
    best = comparison_df.iloc[0]
    lines = [
        "# Short Alpha Checkpoint Objective Comparison",
        "",
        "- main line: `baseline_current` vs `state_liquidity_listwise_v1`",
        "- research objective: `execution_first`",
        "- execution alignment: `train_eval_auto + robust_composite`",
        "- annual objective baseline root: "
        f"`{comparison_df.loc[comparison_df['objective_label'] == 'annual_return', 'root_tag'].iloc[0]}`",
        "- monthly objective root: "
        f"`{comparison_df.loc[comparison_df['objective_label'] == 'monthly_robust', 'root_tag'].iloc[0]}`",
        "",
        "## Direct Answer",
        f"- current better checkpoint objective for the liquid500 short-alpha main line is `{best['objective_label']}`.",
        f"- candidate positive-month ratio vs baseline delta: `{_pct(best['delta_positive_month_ratio'])}`.",
        f"- candidate median monthly excess vs baseline delta: `{_pct(best['delta_median_monthly_return'])}`.",
        f"- candidate mean excess annual vs baseline delta: `{_pct(best['delta_mean_excess_annual_return'])}`.",
        f"- candidate mean excess Sharpe vs baseline delta: `{_num(best['delta_mean_excess_sharpe'])}`.",
        "",
        "## Objective Table",
    ]
    for _, row in comparison_df.iterrows():
        lines.append(
            "- "
            f"`{row['objective_label']}`: "
            f"candidate positive-month ratio `{_pct(row['candidate_positive_month_ratio'])}` vs baseline `{_pct(row['baseline_positive_month_ratio'])}`; "
            f"candidate median monthly excess `{_pct(row['candidate_median_monthly_return'])}` vs baseline `{_pct(row['baseline_median_monthly_return'])}`; "
            f"candidate mean excess annual `{_pct(row['candidate_mean_excess_annual_return'])}` vs baseline `{_pct(row['baseline_mean_excess_annual_return'])}`; "
            f"candidate mean excess Sharpe `{_num(row['candidate_mean_excess_sharpe'])}` vs baseline `{_num(row['baseline_mean_excess_sharpe'])}`; "
            f"annual/sharpe wins `{int(row['candidate_wins_annual'])}/{int(row['candidate_wins_sharpe'])}`"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    annual_root = _root_dir(args.annual_root_tag)
    if args.rerun_annual or not (annual_root / "profile_summary.csv").exists():
        annual_cmd = _build_formal_command(
            python_executable=args.python_executable,
            root_tag=str(args.annual_root_tag),
            checkpoint_selection_objective="primary_annual_return",
            family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
        )
        _run_command(annual_cmd)

    monthly_root = _root_dir(args.monthly_root_tag)
    if args.rerun_monthly or not (monthly_root / "profile_summary.csv").exists():
        monthly_cmd = _build_formal_command(
            python_executable=args.python_executable,
            root_tag=str(args.monthly_root_tag),
            checkpoint_selection_objective="primary_monthly_robust_score",
            family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
        )
        _run_command(monthly_cmd)

    comparison_rows = [
        _collect_objective_row(objective_label="annual_return", root_tag=str(args.annual_root_tag)),
        _collect_objective_row(objective_label="monthly_robust", root_tag=str(args.monthly_root_tag)),
    ]
    comparison_df = _finalize_comparison(pd.DataFrame(comparison_rows))

    output_dir = _root_dir(args.comparison_root_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "objective_comparison.csv", index=False, encoding="utf-8-sig")
    _write_summary(output_dir, comparison_df)

    best = comparison_df.iloc[0]
    payload = {
        "output_dir": str(output_dir),
        "winning_objective": str(best["objective_label"]),
        "winning_root_tag": str(best["root_tag"]),
        "winning_candidate_recent_source_run": str(best["candidate_source_recent_run"]),
        "winning_candidate_positive_month_ratio": float(best["candidate_positive_month_ratio"]),
        "winning_candidate_median_monthly_return": float(best["candidate_median_monthly_return"]),
        "winning_candidate_mean_excess_annual_return": float(best["candidate_mean_excess_annual_return"]),
        "winning_candidate_mean_excess_sharpe": float(best["candidate_mean_excess_sharpe"]),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
