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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
FORMAL_POLICY_REVIEW_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_formal_execution_policy_review.py"
WEAK_MONTH_REVIEW_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_short_alpha_weak_month_review.py"
DEFAULT_FORMAL_ROOT = OUTPUT_ROOT / "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review weak-month repair candidates for the liquid500 short_alpha main line "
            "by expanding the score-to-weight / bridge search space and re-running the "
            "formal execution-policy audit."
        )
    )
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    parser.add_argument("--candidate-profile", default="state_liquidity_listwise_v1")
    parser.add_argument("--baseline-profile", default="baseline_current")
    parser.add_argument("--current-profile", default="regoff_k1_5d_ensemble_native_anchor")
    parser.add_argument("--profile-set", default="weak_month_repair_v1")
    parser.add_argument("--selection-objective", default="excess_annual_return")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_score_weight_repair_review_20260406_r1")
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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


def _discover_candidate_runs(formal_root: Path, candidate_profile: str) -> list[Path]:
    runs_root = formal_root / "runs"
    out: list[Path] = []
    prefix = f"{candidate_profile}_"
    for run_dir in sorted(runs_root.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith(prefix):
            out.append(run_dir.resolve())
    if not out:
        raise FileNotFoundError(f"No formal runs found for {candidate_profile} under {formal_root}")
    return out


def _write_report(
    output_dir: Path,
    *,
    leaderboard: dict[str, Any],
    profile_mean_summary: pd.DataFrame,
    weak_month_df: pd.DataFrame,
    regime_summary: pd.DataFrame,
    current_profile: str,
) -> None:
    best_profile = str(leaderboard.get("best_profile", "") or "")
    current_row = profile_mean_summary.loc[profile_mean_summary["profile_name"].astype(str).eq(current_profile)]
    current_row_payload = current_row.iloc[0].to_dict() if not current_row.empty else {}
    best_row = profile_mean_summary.iloc[0].to_dict() if not profile_mean_summary.empty else {}
    weak_policy_summary = (
        weak_month_df.groupby("best_policy_name", dropna=False)
        .agg(
            weak_month_wins=("month", "count"),
            avg_lift_vs_current=("best_policy_delta_vs_current", "mean"),
            avg_candidate_gap=("candidate_gap_vs_best_policy", "mean"),
        )
        .reset_index()
        .sort_values(["weak_month_wins", "avg_lift_vs_current"], ascending=[False, False])
    )
    weak_policy_summary.to_csv(output_dir / "weak_policy_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Short Alpha Score-to-Weight Repair Review",
        "",
        f"- profile_set: `{leaderboard.get('profile_set', '')}`",
        f"- current_profile: `{current_profile}`",
        f"- best_repair_profile: `{best_profile}`",
        f"- best_mean_excess_annual_return: `{_pct(leaderboard.get('best_mean_excess_annual_return'))}`",
        f"- best_mean_excess_sharpe: `{_num(leaderboard.get('best_mean_excess_sharpe'))}`",
    ]
    if current_row_payload:
        lines.extend(
            [
                f"- current_mean_excess_annual_return: `{_pct(current_row_payload.get('mean_excess_annual_return'))}`",
                f"- current_mean_excess_sharpe: `{_num(current_row_payload.get('mean_excess_sharpe'))}`",
                f"- delta_vs_current_excess_annual_return: `{_pct(_safe_float(best_row.get('mean_excess_annual_return')) - _safe_float(current_row_payload.get('mean_excess_annual_return')))}`",
                f"- delta_vs_current_excess_sharpe: `{_num(_safe_float(best_row.get('mean_excess_sharpe')) - _safe_float(current_row_payload.get('mean_excess_sharpe')))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Weak-Month Coverage",
            f"- weak_month_count: `{int(len(weak_month_df))}`",
            f"- weak_month_ratio: `{_pct(len(weak_month_df) / 36.0 if len(weak_month_df) else 0.0)}`",
            f"- avg_best_policy_lift_vs_current: `{_pct(weak_month_df['best_policy_delta_vs_current'].mean() if not weak_month_df.empty else float('nan'))}`",
            f"- avg_candidate_gap_vs_best_policy: `{_pct(weak_month_df['candidate_gap_vs_best_policy'].mean() if not weak_month_df.empty else float('nan'))}`",
            "",
            "## Weak Regimes",
        ]
    )
    for _, row in regime_summary.iterrows():
        lines.append(
            "- "
            f"{str(row['month_start_regime'])}: weak months `{int(row['weak_month_count'])}`, "
            f"candidate delta vs baseline `{_pct(row['avg_candidate_delta_vs_baseline'])}`, "
            f"best-policy lift `{_pct(row['avg_best_policy_delta_vs_current'])}`"
        )
    lines.extend(["", "## Top Repair Profiles"])
    for _, row in weak_policy_summary.head(5).iterrows():
        lines.append(
            "- "
            f"{str(row['best_policy_name'])}: weak-month wins `{int(row['weak_month_wins'])}`, "
            f"avg lift vs current `{_pct(row['avg_lift_vs_current'])}`, "
            f"avg candidate gap `{_pct(row['avg_candidate_gap'])}`"
        )
    lines.extend(
        [
            "",
            "## Direct Answer",
        ]
    )
    if best_profile and best_profile != current_profile and _safe_float(best_row.get("mean_excess_annual_return")) > _safe_float(current_row_payload.get("mean_excess_annual_return")):
        lines.append(
            "- The expanded score-to-weight / bridge search space finds a formal winner above the current policy, so weak-month repair should continue on this branch before touching the backbone again."
        )
    else:
        lines.append(
            "- The expanded score-to-weight / bridge search space still does not overturn the current formal execution winner, so the remaining repair work is more about targeted weak-month gating than about a new static bridge profile."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    formal_root = Path(args.formal_root).resolve()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_runs = _discover_candidate_runs(formal_root, str(args.candidate_profile).strip())

    formal_review_tag = f"{args.root_tag}/formal_policy_review"
    formal_review_cmd = [
        str(args.python_executable),
        str(FORMAL_POLICY_REVIEW_SCRIPT),
        "--candidate-name",
        "short_alpha_score_weight_repair",
        "--universe",
        "liquid500",
        "--panel-scope",
        "formal",
        "--profile-set",
        str(args.profile_set),
        "--selection-objective",
        str(args.selection_objective),
        "--current-profile",
        str(args.current_profile),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--experiment-tag",
        formal_review_tag,
        "--python-executable",
        str(args.python_executable),
    ]
    for run_dir in candidate_runs:
        formal_review_cmd.extend(["--run-dir", str(run_dir)])
    _run_command(formal_review_cmd)

    formal_review_dir = Path(args.output_root).resolve() / formal_review_tag
    leaderboard = _load_json(formal_review_dir / "leaderboard.json")
    profile_mean_summary = pd.read_csv(formal_review_dir / "profile_mean_summary.csv")
    window_profile_summary = pd.read_csv(formal_review_dir / "window_profile_summary.csv")
    audit_roots = sorted(dict.fromkeys(str(item).strip() for item in window_profile_summary["audit_root"].tolist() if str(item).strip()))
    if not audit_roots:
        raise RuntimeError("No audit roots were found in the formal policy review output.")
    (output_dir / "audit_roots.txt").write_text("\n".join(audit_roots) + "\n", encoding="utf-8")

    weak_month_tag = f"{args.root_tag}/weak_month_review"
    weak_month_cmd = [
        str(args.python_executable),
        str(WEAK_MONTH_REVIEW_SCRIPT),
        "--formal-root",
        str(formal_root),
        "--audit-roots",
        ",".join(audit_roots),
        "--candidate-profile",
        str(args.candidate_profile),
        "--baseline-profile",
        str(args.baseline_profile),
        "--current-policy",
        str(args.current_profile),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--root-tag",
        weak_month_tag,
    ]
    _run_command(weak_month_cmd)

    weak_month_dir = Path(args.output_root).resolve() / weak_month_tag
    weak_month_df = pd.read_csv(weak_month_dir / "weak_month_review.csv")
    regime_summary = pd.read_csv(weak_month_dir / "weak_month_regime_summary.csv")

    profile_mean_summary.to_csv(output_dir / "profile_mean_summary.csv", index=False, encoding="utf-8-sig")
    window_profile_summary.to_csv(output_dir / "window_profile_summary.csv", index=False, encoding="utf-8-sig")
    weak_month_df.to_csv(output_dir / "weak_month_review.csv", index=False, encoding="utf-8-sig")
    regime_summary.to_csv(output_dir / "weak_month_regime_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "leaderboard.json").write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_report(
        output_dir,
        leaderboard=leaderboard,
        profile_mean_summary=profile_mean_summary,
        weak_month_df=weak_month_df,
        regime_summary=regime_summary,
        current_profile=str(args.current_profile).strip(),
    )
    print(json.dumps({"output_dir": str(output_dir), "best_profile": leaderboard.get("best_profile", "")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
