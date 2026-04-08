from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.backtest import summarize_backtest_by_month, summarize_monthly_diagnostics
from daily_research.deep_alpha.execution_alignment import default_auto_profile_argument
from daily_research.deep_alpha.family_epoch_budget import DEFAULT_LATEST_MANIFEST_PATH, resolve_epoch_budget_for_family
from daily_research.deep_alpha.research_objective import resolve_primary_backtest
from daily_research.deep_alpha.short_alpha_profiles import build_profile_cli_args, get_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXECUTION_ALIGNMENT_PROFILE_SET = default_auto_profile_argument()


@dataclass(frozen=True)
class RecentFormalWindow:
    label: str = "20250318_20260331"
    train_end: str = "2025-03-17"
    valid_start: str = "2025-03-18"
    valid_months: int = 12


WINDOW = RecentFormalWindow()
RESEARCH_TIME_UNIT = "calendar_months"
TRAIN_EVAL_WINDOW_MONTHS = 6
DEFAULT_ROOT_TAG = "short_alpha_short_horizon_expert_review_20260406_r1"
DEFAULT_PROFILES = (
    "baseline_current",
    "state_liquidity_listwise_v1",
    "short_expert_monthly_v1",
)
REUSE_METRICS: dict[str, str] = {
    "baseline_current": (
        "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/"
        "runs/baseline_current_20250318_20260331/metrics.json"
    ),
    "state_liquidity_listwise_v1": (
        "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/"
        "runs/state_liquidity_listwise_v1_20250318_20260331/metrics.json"
    ),
    "short_expert_monthly_v1": (
        "daily_research/output/short_alpha_short_horizon_expert_review_20260406_r2_fullbudget/"
        "runs/short_expert_monthly_v1/metrics.json"
    ),
    "short_expert_monthly_v2": (
        "daily_research/output/short_alpha_short_horizon_expert_v2_review_20260407_r3_shortalpha48/"
        "runs/short_expert_monthly_v2/metrics.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current liquid500 short-alpha main line against a short-horizon "
            "expert profile on the latest formal monthly window."
        )
    )
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated short-alpha profile names from short_alpha_profiles.py",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--family-epoch-budget-manifest", default=str(DEFAULT_LATEST_MANIFEST_PATH))
    parser.add_argument(
        "--short-alpha-epoch-budget-override",
        type=int,
        default=0,
        help="Optional explicit epoch budget for non-baseline short-alpha profiles in this review.",
    )
    return parser.parse_args()


def _resolve_family_key(profile_name: str) -> str:
    return "baseline" if str(profile_name).strip().lower() == "baseline_current" else "short_alpha"


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _resolve_metrics_path(root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / profile_name / "metrics.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _load_primary_monthly_diagnostics(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    explicit = metrics.get("primary_research_monthly_diagnostics")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)
    monthly_label = str(metrics.get("primary_research_monthly_summary_label", "")).strip()
    if not monthly_label:
        monthly_label = (
            "execution_aligned_monthly_backtest_summary"
            if str(metrics.get("primary_research_backtest_label", "")).strip() == "execution_aligned_holdout_backtest"
            else "monthly_backtest_summary"
        )
    monthly_path = run_dir / f"{monthly_label}.csv"
    if monthly_path.exists():
        monthly_df = pd.read_csv(monthly_path)
        if "date" in monthly_df.columns:
            monthly_df["date"] = pd.to_datetime(monthly_df["date"])
        return summarize_monthly_diagnostics(monthly_df)
    backtest_label, _ = resolve_primary_backtest(metrics, research_objective_mode=str(metrics.get("research_objective_mode", "")))
    fallback_path = run_dir / f"{backtest_label}.csv"
    if fallback_path.exists():
        backtest_df = pd.read_csv(fallback_path)
        if "date" in backtest_df.columns:
            backtest_df["date"] = pd.to_datetime(backtest_df["date"])
        monthly_df = summarize_backtest_by_month(backtest_df)
        return summarize_monthly_diagnostics(monthly_df)
    return {}


def _build_command(
    *,
    python_executable: str,
    experiment_tag: str,
    profile_name: str,
    family_epoch_budget_manifest: str,
    short_alpha_epoch_budget_override: int,
) -> list[str]:
    profile = get_profile(profile_name)
    family_key = _resolve_family_key(profile_name)
    epoch_budget = resolve_epoch_budget_for_family(family_key, manifest_path=family_epoch_budget_manifest, fallback_epochs=8)
    if family_key == "short_alpha" and int(short_alpha_epoch_budget_override) > 0:
        epoch_budget = int(short_alpha_epoch_budget_override)
    cmd = [
        python_executable,
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
        RESEARCH_TIME_UNIT,
        "--train-end-date",
        WINDOW.train_end,
        "--valid-start-date",
        WINDOW.valid_start,
        "--valid-days",
        "0",
        "--valid-months",
        str(WINDOW.valid_months),
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
        EXECUTION_ALIGNMENT_PROFILE_SET,
        "--execution-alignment-transaction-cost-bps",
        "3",
        "--execution-alignment-slippage-bps",
        "7",
        "--execution-alignment-sell-tax-bps",
        "10",
        "--train-eval-window-days",
        "0",
        "--train-eval-window-months",
        str(TRAIN_EVAL_WINDOW_MONTHS),
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
        experiment_tag,
    ]
    cmd.extend(build_profile_cli_args(profile, include_objective_overrides=True))
    return cmd


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


def main() -> None:
    args = parse_args()
    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No short-alpha profiles were provided.")

    rows: list[dict[str, Any]] = []
    source_runs: dict[str, str] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        reused_existing = False
        metrics_path = _resolve_metrics_path(root_tag, profile.name)
        reuse_path_str = REUSE_METRICS.get(profile.name, "")
        if reuse_path_str and not args.force_rerun:
            candidate_path = (PROJECT_ROOT / reuse_path_str).resolve()
            if candidate_path.exists():
                metrics_path = candidate_path
                reused_existing = True
        if not metrics_path.exists() or args.force_rerun:
            experiment_tag = f"{root_tag}/runs/{profile.name}"
            command = _build_command(
                python_executable=str(args.python_executable),
                experiment_tag=experiment_tag,
                profile_name=profile.name,
                family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
                short_alpha_epoch_budget_override=int(args.short_alpha_epoch_budget_override),
            )
            _run_command(command)
            metrics_path = _resolve_metrics_path(root_tag, profile.name)
            reused_existing = False
        metrics = _load_json(metrics_path)
        _, holdout = resolve_primary_backtest(metrics, research_objective_mode=str(metrics.get("research_objective_mode", "")))
        holdout = dict(holdout)
        monthly_diagnostics = _load_primary_monthly_diagnostics(metrics_path.parent, metrics)
        source_runs[profile.name] = str(metrics_path.resolve())
        rows.append(
            {
                "profile_name": profile.name,
                "description": profile.description,
                "metrics_path": str(metrics_path.resolve()),
                "reused_existing": reused_existing,
                "feature_count": int(metrics.get("feature_count", 0)),
                "target_names": ",".join(metrics.get("target_names", [])),
                "score_head_method": str(metrics.get("score_head_method", profile.score_head_method or "manual")),
                "adaptive_task_weights": bool(metrics.get("adaptive_task_weights", profile.adaptive_task_weights)),
                "checkpoint_selection_objective": str(
                    metrics.get("checkpoint_selection_objective", profile.checkpoint_selection_objective or "primary_annual_return")
                ),
                "research_objective_mode": str(metrics.get("research_objective_mode", profile.research_objective_mode or "execution_first")),
                "annual_return": float(holdout.get("annual_return", 0.0)),
                "excess_total_return": float(holdout.get("excess_total_return", 0.0)),
                "excess_annual_return": float(holdout.get("excess_annual_return", 0.0)),
                "excess_sharpe": float(holdout.get("excess_sharpe", 0.0)),
                "excess_max_drawdown": float(holdout.get("excess_max_drawdown", 0.0)),
                "avg_turnover": float(holdout.get("avg_turnover", 0.0)),
                "avg_holding_count": float(holdout.get("avg_holding_count", 0.0)),
                "monthly_positive_ratio": float(monthly_diagnostics.get("positive_month_ratio", 0.0)),
                "monthly_median_excess_return": float(monthly_diagnostics.get("median_monthly_return", 0.0)),
                "monthly_worst_excess_return": float(monthly_diagnostics.get("worst_monthly_return", 0.0)),
                "monthly_top3_positive_share": float(monthly_diagnostics.get("top3_positive_month_share", 0.0)),
                "monthly_longest_negative_streak": int(monthly_diagnostics.get("longest_negative_streak", 0)),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        [
            "monthly_positive_ratio",
            "monthly_median_excess_return",
            "monthly_worst_excess_return",
            "monthly_top3_positive_share",
            "excess_sharpe",
            "excess_annual_return",
        ],
        ascending=[False, False, False, True, False, False],
    ).reset_index(drop=True)

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    best = summary_df.iloc[0]
    current_row = summary_df.loc[summary_df["profile_name"] == "state_liquidity_listwise_v1"].iloc[0]
    lines = [
        "# Short Alpha Short-Horizon Expert Review",
        "",
        "## Protocol",
        f"- window: `{WINDOW.valid_start} -> 2026-04-01`",
        f"- research_time_unit: `{RESEARCH_TIME_UNIT}` with `valid_months={WINDOW.valid_months}` and `train_eval_window_months={TRAIN_EVAL_WINDOW_MONTHS}`",
        "- universe: `liquid500`",
        "- benchmark: `000300.SH`",
        "- backbone: `patch_transformer + dynamic_graph_v1`",
        "- comparison goal: current monthly-first short-alpha line vs short-horizon expert profile",
        "",
        "## Direct Answer",
        f"- current best profile under monthly-first ranking on the latest formal window is `{best['profile_name']}`.",
        f"- best positive-month ratio: `{_pct(best['monthly_positive_ratio'])}`.",
        f"- best median monthly excess: `{_pct(best['monthly_median_excess_return'])}`.",
        f"- best worst-month excess: `{_pct(best['monthly_worst_excess_return'])}`.",
        f"- best excess annual / Sharpe: `{_pct(best['excess_annual_return'])} / {_num(best['excess_sharpe'])}`.",
        "",
        "## Current Main Line vs Short-Horizon Expert",
    ]
    for _, row in summary_df.iterrows():
        delta_month = float(row["monthly_median_excess_return"]) - float(current_row["monthly_median_excess_return"])
        delta_annual = float(row["excess_annual_return"]) - float(current_row["excess_annual_return"])
        reuse_label = "reused" if bool(row["reused_existing"]) else "fresh"
        lines.append(
            "- "
            f"`{row['profile_name']}` [{reuse_label}]: "
            f"positive-month `{_pct(row['monthly_positive_ratio'])}`, "
            f"median monthly excess `{_pct(row['monthly_median_excess_return'])}`, "
            f"worst month `{_pct(row['monthly_worst_excess_return'])}`, "
            f"top3 share `{_pct(row['monthly_top3_positive_share'])}`, "
            f"excess annual `{_pct(row['excess_annual_return'])}`, "
            f"excess Sharpe `{_num(row['excess_sharpe'])}`, "
            f"delta vs current median monthly `{_pct(delta_month)}`, "
            f"delta vs current excess annual `{_pct(delta_annual)}`, "
            f"score_head `{row['score_head_method']}`, "
            f"checkpoint `{row['checkpoint_selection_objective']}`"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_dir={output_dir}")


if __name__ == "__main__":
    main()
