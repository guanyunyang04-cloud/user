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

from daily_research.deep_alpha.family_epoch_budget import DEFAULT_LATEST_MANIFEST_PATH, resolve_epoch_budget_for_family
from daily_research.deep_alpha.short_alpha_profiles import build_profile_cli_args, get_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
DEFAULT_ROOT_TAG = "short_alpha_policy_v2_family_formal_review_20260411_r1"
DEFAULT_EXECUTION_PROFILES = (
    "raw_1d,topk1_1d_regoff,"
    "regoff_k1_3d_ensemble_native_anchor,regoff_k1_5d_ensemble_native_anchor,"
    "regoff_k1_10d_ensemble_native_anchor,regoff_k1_20d_ensemble_native_anchor,"
    "regoff_k2_3d_ensemble_native_anchor,regoff_k2_5d_ensemble_native_anchor,"
    "regoff_k2_10d_ensemble_native_anchor,regoff_k2_20d_ensemble_native_anchor"
)
DEFAULT_EXISTING_POLICY_V2_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_policy_v2_review_20260410_r1" / "runs" / "short_expert_policy_v2"
)
DEFAULT_CURRENT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
)
DEFAULT_COMPANION_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_formal_head2head_20260405_monthly_checkpoint_r1" / "runs" / "state_liquidity_listwise_v1_20250318_20260331"
)
FAMILY_PROFILES = (
    "short_expert_policy_v2",
    "short_expert_policy_v2a",
    "short_expert_policy_v2b",
    "short_expert_policy_v2c",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a narrow formal family review for policy_v2 / v2a / v2b / v2c against the current mainline."
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--execution-alignment-candidate-profiles", default=DEFAULT_EXECUTION_PROFILES)
    parser.add_argument("--family-epoch-budget-manifest", default=str(DEFAULT_LATEST_MANIFEST_PATH))
    parser.add_argument("--existing-policy-v2-run-dir", default=str(DEFAULT_EXISTING_POLICY_V2_RUN_DIR))
    parser.add_argument("--current-formal-run-dir", default=str(DEFAULT_CURRENT_FORMAL_RUN_DIR))
    parser.add_argument("--companion-formal-run-dir", default=str(DEFAULT_COMPANION_FORMAL_RUN_DIR))
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_monthly_diag(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    explicit = metrics.get("primary_research_monthly_diagnostics")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)
    fallback = _load_json(run_dir / "primary_research_monthly_diagnostics.json")
    return fallback if fallback else {}


def _monthly_robust_score(monthly_diag: dict[str, Any]) -> float:
    positive_ratio = float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0)
    median_monthly_return = float(monthly_diag.get("median_monthly_return", 0.0) or 0.0)
    mean_monthly_return = float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0)
    worst_monthly_return = float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0)
    top3_positive_share = float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0)
    longest_negative_streak = int(monthly_diag.get("longest_negative_streak", 0) or 0)
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


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def _run_dir_for_profile(root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / str(root_tag).strip() / "runs" / str(profile_name).strip()


def _ensure_profile_run(profile_name: str, args: argparse.Namespace) -> tuple[Path, str]:
    if profile_name == "short_expert_policy_v2" and not args.force_rerun:
        existing = Path(args.existing_policy_v2_run_dir).resolve()
        if (existing / "metrics.json").exists():
            return existing, "policy_v2_reused_existing"

    run_dir = _run_dir_for_profile(str(args.root_tag), profile_name)
    if (run_dir / "metrics.json").exists() and not args.force_rerun:
        return run_dir, "family_reused_existing"
    resume_run_dir = None if args.force_rerun else run_dir if (run_dir / "deep_alpha_model.pt").exists() else None

    epoch_budget = resolve_epoch_budget_for_family(
        "short_alpha",
        manifest_path=str(args.family_epoch_budget_manifest),
        fallback_epochs=8,
    )
    profile = get_profile(profile_name)
    command = [
        str(args.python_executable),
        str(RUN_SCRIPT),
        "--data-source",
        "tq",
        "--start-date",
        "20210101",
        "--end-date",
        "20260401",
        "--benchmark",
        "000300.SH",
        "--liquidity-pool",
        "liquid500",
        "--research-time-unit",
        "calendar_months",
        "--train-end-date",
        "2025-03-17",
        "--valid-start-date",
        "2025-03-18",
        "--valid-days",
        "0",
        "--valid-months",
        "12",
        "--lookback-window",
        "120",
        "--batch-size",
        "256",
        "--hidden-dim",
        "96",
        "--encoder-family",
        "patch_transformer",
        "--patch-len",
        "5",
        "--transformer-heads",
        "4",
        "--transformer-layers",
        "2",
        "--dropout",
        "0.1",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--epochs",
        str(epoch_budget),
        "--min-epochs",
        "1",
        "--early-stop-patience",
        str(max(int(epoch_budget), 8)),
        "--lr-plateau-patience",
        "4",
        "--lr-plateau-factor",
        "0.5",
        "--min-improvement",
        "0.0001",
        "--return-loss-mode",
        "top_bottom_bce",
        "--return-target-transform",
        "raw",
        "--score-risk-mode",
        "subtract",
        "--execution-alignment-mode",
        "train_eval_auto",
        "--execution-alignment-objective",
        "robust_composite",
        "--execution-alignment-candidate-profiles",
        str(args.execution_alignment_candidate_profiles),
        "--execution-alignment-transaction-cost-bps",
        "3",
        "--execution-alignment-slippage-bps",
        "7",
        "--execution-alignment-sell-tax-bps",
        "10",
        "--train-eval-window-days",
        "0",
        "--train-eval-window-months",
        "6",
        "--dynamic-graph-layer",
        "--dynamic-graph-top-k",
        "8",
        "--dynamic-graph-temperature",
        "0.35",
        "--dynamic-graph-industry-boost",
        "0.15",
        "--dynamic-graph-style-boost",
        "0.05",
        "--num-workers",
        "0",
        "--pin-memory",
        "--use-amp",
        "--no-safe-runtime-profile",
        "--experiment-tag",
        f"{str(args.root_tag).strip()}/runs/{profile_name}",
    ]
    command.extend(build_profile_cli_args(profile, include_objective_overrides=True))
    if resume_run_dir is not None:
        command.extend(["--resume-run-dir", str(resume_run_dir), "--resume-mode", "strict"])
    _run_command(command)
    return run_dir, "family_strict_resumed" if resume_run_dir is not None else "family_fresh"


def _build_row(profile_name: str, run_dir: Path, source_label: str) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    monthly_diag = _load_monthly_diag(run_dir)
    backtest = metrics.get("primary_research_backtest", {})
    return {
        "profile_name": profile_name,
        "source_label": source_label,
        "run_dir": str(run_dir),
        "execution_alignment_profile": str(metrics.get("execution_alignment_profile", "") or ""),
        "score_head_method": str(metrics.get("score_head_method", "") or ""),
        "excess_annual_return": float(backtest.get("excess_annual_return", 0.0) or 0.0),
        "excess_sharpe": float(backtest.get("excess_sharpe", 0.0) or 0.0),
        "avg_turnover": float(backtest.get("avg_turnover", 0.0) or 0.0),
        "positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
        "median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
        "mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
        "worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
        "top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
        "longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
        "monthly_robust_score": _monthly_robust_score(monthly_diag),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_runs: dict[str, str] = {}
    family_rows: list[dict[str, Any]] = []
    for profile_name in FAMILY_PROFILES:
        run_dir, source_label = _ensure_profile_run(profile_name, args)
        source_runs[profile_name] = str(run_dir)
        family_rows.append(_build_row(profile_name, run_dir, source_label))

    current_formal_run_dir = Path(args.current_formal_run_dir).resolve()
    companion_formal_run_dir = Path(args.companion_formal_run_dir).resolve()
    reference_rows = [
        _build_row("short_expert_monthly_v1", current_formal_run_dir, "current_mainline_reference"),
        _build_row("state_liquidity_listwise_v1", companion_formal_run_dir, "formal_companion_reference"),
    ]

    rows = family_rows + reference_rows
    scoreboard = pd.DataFrame(rows).sort_values(
        ["monthly_robust_score", "positive_month_ratio", "median_monthly_return", "excess_annual_return", "excess_sharpe"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    scoreboard.to_csv(output_dir / "formal_scoreboard.csv", index=False, encoding="utf-8-sig")
    (output_dir / "family_source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    overall_winner = scoreboard.iloc[0].to_dict() if not scoreboard.empty else {}
    family_scoreboard = scoreboard.loc[scoreboard["profile_name"].isin(FAMILY_PROFILES)].copy()
    family_winner = family_scoreboard.iloc[0].to_dict() if not family_scoreboard.empty else {}
    current_row = scoreboard.loc[scoreboard["profile_name"] == "short_expert_monthly_v1"].iloc[0].to_dict()
    companion_row = scoreboard.loc[scoreboard["profile_name"] == "state_liquidity_listwise_v1"].iloc[0].to_dict()

    summary = {
        "overall_formal_winner_profile_name": str(overall_winner.get("profile_name", "")),
        "family_formal_winner_profile_name": str(family_winner.get("profile_name", "")),
        "family_formal_winner_row": family_winner,
        "current_row": current_row,
        "companion_row": companion_row,
        "family_rows": family_scoreboard.to_dict("records"),
        "execution_alignment_candidate_profiles": str(args.execution_alignment_candidate_profiles),
        "source_runs": source_runs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Policy V2 Family Formal Review",
        "",
        "## Scope",
        "- window: `2025-03-18 -> 2026-02-27`",
        "- protocol: `execution_first + primary_monthly_robust_score`",
        f"- constrained execution profile set: `{args.execution_alignment_candidate_profiles}`",
        "- compared family: `short_expert_policy_v2`, `short_expert_policy_v2a`, `short_expert_policy_v2b`, `short_expert_policy_v2c`",
        "",
        "## Direct Answer",
        f"- overall formal winner: `{overall_winner.get('profile_name', 'n/a')}`",
        f"- family formal winner: `{family_winner.get('profile_name', 'n/a')}`",
        f"- family winner execution profile: `{family_winner.get('execution_alignment_profile', '')}`",
        f"- family winner monthly robust: `{_format_num(family_winner.get('monthly_robust_score'))}`",
        f"- delta vs current mainline monthly robust: `{_format_num(float(family_winner.get('monthly_robust_score', 0.0) or 0.0) - float(current_row.get('monthly_robust_score', 0.0) or 0.0))}`",
        f"- delta vs existing policy_v2 monthly robust: `{_format_num(float(family_winner.get('monthly_robust_score', 0.0) or 0.0) - float(family_scoreboard.loc[family_scoreboard['profile_name'] == 'short_expert_policy_v2', 'monthly_robust_score'].iloc[0] if 'short_expert_policy_v2' in set(family_scoreboard['profile_name']) else 0.0))}`",
        "",
        "## Family Scoreboard",
    ]
    for _, row in family_scoreboard.iterrows():
        lines.append(
            f"- `{row['profile_name']}` [{row['source_label']}]: profile `{row['execution_alignment_profile']}`, "
            f"excess annual `{_format_pct(row['excess_annual_return'])}`, excess Sharpe `{_format_num(row['excess_sharpe'])}`, "
            f"positive-month `{_format_pct(row['positive_month_ratio'])}`, median monthly `{_format_pct(row['median_monthly_return'])}`, "
            f"worst month `{_format_pct(row['worst_monthly_return'])}`, monthly robust `{_format_num(row['monthly_robust_score'])}`"
        )
    lines.extend(
        [
            "",
            "## References",
            f"- current mainline `short_expert_monthly_v1`: monthly robust `{_format_num(current_row.get('monthly_robust_score'))}`, excess annual `{_format_pct(current_row.get('excess_annual_return'))}`",
            f"- formal companion `state_liquidity_listwise_v1`: monthly robust `{_format_num(companion_row.get('monthly_robust_score'))}`, excess annual `{_format_pct(companion_row.get('excess_annual_return'))}`",
            "",
            "## Decision",
            "- Only a family candidate that materially shrinks the formal gap should move on to constrained execution review as the next learned-control branch.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
