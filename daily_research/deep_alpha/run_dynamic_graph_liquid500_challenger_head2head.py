from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.backtest import summarize_backtest_by_month, summarize_monthly_diagnostics
from daily_research.deep_alpha.dynamic_graph_profiles import get_profile as get_dynamic_graph_profile
from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.execution_alignment import default_auto_profile_argument
from daily_research.deep_alpha.family_epoch_budget import DEFAULT_LATEST_MANIFEST_PATH, resolve_epoch_budget_for_family
from daily_research.deep_alpha.research_objective import (
    CHECKPOINT_SELECTION_OBJECTIVES,
    DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    DEFAULT_RESEARCH_OBJECTIVE_MODE,
    resolve_primary_backtest,
)
from daily_research.deep_alpha.short_alpha_profiles import (
    build_profile_cli_args as build_short_alpha_profile_cli_args,
    get_profile as get_short_alpha_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXECUTION_ALIGNMENT_PROFILE_SET = default_auto_profile_argument()
DEFAULT_PROFILES = ("baseline_current", "state_liquidity_listwise_v1", "dynamic_graph_no_priors")


@dataclass(frozen=True)
class FormalWindow:
    label: str
    train_end: str
    valid_start: str
    valid_months: int = 12


WINDOWS: tuple[FormalWindow, ...] = (
    FormalWindow(label="20230216_20240229", train_end="2023-02-15", valid_start="2023-02-16"),
    FormalWindow(label="20240301_20250317", train_end="2024-02-29", valid_start="2024-03-01"),
    FormalWindow(label="20250318_20260331", train_end="2025-03-17", valid_start="2025-03-18"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run liquid500 same-protocol formal head-to-head for dynamic_graph_no_priors against the current short_alpha main line."
    )
    parser.add_argument("--root-tag", default="dynamic_graph_liquid500_challenger_20260405_r1")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--family-epoch-budget-manifest", default=str(DEFAULT_LATEST_MANIFEST_PATH))
    parser.add_argument(
        "--research-objective-mode",
        choices=["execution_first", "raw_holdout"],
        default=DEFAULT_RESEARCH_OBJECTIVE_MODE,
    )
    parser.add_argument(
        "--checkpoint-selection-objective",
        choices=list(CHECKPOINT_SELECTION_OBJECTIVES),
        default=DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    )
    parser.add_argument("--checkpoint-selection-min-improvement", type=float, default=0.0001)
    parser.add_argument("--execution-alignment-objective", default="robust_composite")
    return parser.parse_args()


def _resolve_family_key(profile_name: str) -> str:
    normalized = str(profile_name or "").strip().lower()
    if normalized == "baseline_current":
        return "baseline"
    if normalized == "state_liquidity_listwise_v1":
        return "short_alpha"
    return "dynamic_graph"


def _is_short_alpha_profile(profile_name: str) -> bool:
    return str(profile_name).strip().lower() in {"baseline_current", "state_liquidity_listwise_v1"}


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_command(
    *,
    python_executable: str,
    experiment_tag: str,
    profile_name: str,
    window: FormalWindow,
    family_epoch_budget_manifest: str,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
) -> list[str]:
    family_key = _resolve_family_key(profile_name)
    epoch_budget = resolve_epoch_budget_for_family(family_key, manifest_path=family_epoch_budget_manifest, fallback_epochs=8)
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
        "calendar_months",
        "--train-end-date",
        window.train_end,
        "--valid-start-date",
        window.valid_start,
        "--valid-days",
        "0",
        "--valid-months",
        str(window.valid_months),
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
        "--research-objective-mode",
        research_objective_mode,
        "--checkpoint-selection-objective",
        checkpoint_selection_objective,
        "--checkpoint-selection-min-improvement",
        str(checkpoint_selection_min_improvement),
        "--return-loss-mode",
        "top_bottom_bce",
        "--return-target-transform",
        "raw",
        "--prediction-horizons",
        "5,10,20",
        "--task-loss-weights",
        "5:0.2,10:0.3,20:0.5,downside:0.35",
        "--score-horizon-weights",
        "5:0.2,10:0.3,20:0.5",
        "--score-risk-mode",
        "subtract",
        "--score-head-method",
        "manual",
        "--execution-alignment-mode",
        "train_eval_auto",
        "--execution-alignment-objective",
        execution_alignment_objective,
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
        "6",
        "--num-workers",
        "0",
        "--pin-memory",
        "--use-amp",
        "--no-safe-runtime-profile",
        "--experiment-tag",
        experiment_tag,
    ]

    if _is_short_alpha_profile(profile_name):
        profile = get_short_alpha_profile(profile_name)
        cmd.extend(
            [
                "--dynamic-graph-layer",
                "--dynamic-graph-top-k",
                "8",
                "--dynamic-graph-temperature",
                "0.35",
                "--dynamic-graph-industry-boost",
                "0.15",
                "--dynamic-graph-style-boost",
                "0.05",
            ]
        )
        cmd.extend(build_short_alpha_profile_cli_args(profile, include_objective_overrides=False))
        return cmd

    profile = get_dynamic_graph_profile(profile_name)
    if profile.dynamic_graph_layer:
        cmd.append("--dynamic-graph-layer")
    cmd.extend(
        [
            "--dynamic-graph-top-k",
            str(profile.top_k),
            "--dynamic-graph-temperature",
            str(profile.temperature),
            "--dynamic-graph-industry-boost",
            str(profile.industry_boost),
            "--dynamic-graph-style-boost",
            str(profile.style_boost),
            "--ranking-loss-weight",
            "0.0",
            "--listwise-loss-weight",
            "0.0",
            "--listwise-temperature",
            "0.35",
        ]
    )
    return cmd


def _resolve_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / f"{profile_name}_{window.label}" / "metrics.json"


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_primary_monthly_diagnostics(run_dir: Path, metrics: dict) -> dict:
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
        return summarize_monthly_diagnostics(pd.read_csv(monthly_path), return_column="excess_return")
    equity_filename = "execution_aligned_equity_curve.csv" if monthly_label.startswith("execution_aligned_") else "equity_curve.csv"
    actions_filename = "execution_aligned_actions.csv" if monthly_label.startswith("execution_aligned_") else "actions.csv"
    equity_path = run_dir / equity_filename
    if not equity_path.exists():
        return summarize_monthly_diagnostics(pd.DataFrame(), return_column="excess_return")
    equity_df = pd.read_csv(equity_path)
    if "date" in equity_df.columns:
        equity_df["date"] = pd.to_datetime(equity_df["date"], errors="coerce")
        equity_df = equity_df.dropna(subset=["date"]).set_index("date")
    actions_path = run_dir / actions_filename
    action_df = pd.read_csv(actions_path) if actions_path.exists() else pd.DataFrame()
    monthly_df = summarize_backtest_by_month(equity_df, action_df)
    return summarize_monthly_diagnostics(monthly_df, return_column="excess_return")


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_num(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles or "").split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No profiles were provided.")

    rows: list[dict] = []
    source_runs: dict[str, dict[str, str]] = {}
    for profile_name in profile_names:
        source_runs[profile_name] = {}
        family_key = _resolve_family_key(profile_name)
        for window in WINDOWS:
            metrics_path = _resolve_metrics_path(root_tag, profile_name, window)
            if not metrics_path.exists() or args.force_rerun:
                command = _build_command(
                    python_executable=str(args.python_executable),
                    experiment_tag=f"{root_tag}/runs/{profile_name}_{window.label}",
                    profile_name=profile_name,
                    window=window,
                    family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
                    research_objective_mode=str(args.research_objective_mode),
                    checkpoint_selection_objective=str(args.checkpoint_selection_objective),
                    checkpoint_selection_min_improvement=float(args.checkpoint_selection_min_improvement),
                    execution_alignment_objective=str(args.execution_alignment_objective),
                )
                _run_command(command)
            metrics = _load_metrics(metrics_path)
            _, holdout = resolve_primary_backtest(metrics)
            monthly = _load_primary_monthly_diagnostics(metrics_path.parent, metrics)
            source_runs[profile_name][window.label] = str(metrics_path.resolve())
            rows.append(
                {
                    "profile_name": profile_name,
                    "family_key": family_key,
                    "window_label": window.label,
                    "metrics_path": str(metrics_path.resolve()),
                    "epoch_budget": int(metrics.get("epochs", 0) or resolve_epoch_budget_for_family(family_key, manifest_path=str(args.family_epoch_budget_manifest), fallback_epochs=8)),
                    "excess_annual_return": float(holdout.get("excess_annual_return", 0.0)),
                    "excess_sharpe": float(holdout.get("excess_sharpe", 0.0)),
                    "avg_turnover": float(holdout.get("avg_turnover", 0.0)),
                    "excess_max_drawdown": float(holdout.get("excess_max_drawdown", 0.0)),
                    "monthly_positive_ratio": float(monthly.get("positive_month_ratio", 0.0)),
                    "monthly_median_excess_return": float(monthly.get("median_monthly_return", 0.0)),
                    "monthly_worst_excess_return": float(monthly.get("worst_monthly_return", 0.0)),
                    "monthly_top3_positive_share": float(monthly.get("top3_positive_month_share", 0.0)),
                }
            )

    detail_df = pd.DataFrame(rows).sort_values(["profile_name", "window_label"]).reset_index(drop=True)
    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_dir / "window_detail.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_df = (
        detail_df.groupby("profile_name", dropna=False)
        .agg(
            family_key=("family_key", "first"),
            window_count=("window_label", "count"),
            epoch_budget=("epoch_budget", "max"),
            mean_excess_annual_return=("excess_annual_return", "mean"),
            mean_excess_sharpe=("excess_sharpe", "mean"),
            mean_avg_turnover=("avg_turnover", "mean"),
            mean_excess_max_drawdown=("excess_max_drawdown", "mean"),
            mean_monthly_positive_ratio=("monthly_positive_ratio", "mean"),
            mean_monthly_median_excess_return=("monthly_median_excess_return", "mean"),
            worst_monthly_excess_return=("monthly_worst_excess_return", "min"),
            mean_monthly_top3_positive_share=("monthly_top3_positive_share", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "mean_monthly_positive_ratio",
                "mean_monthly_median_excess_return",
                "worst_monthly_excess_return",
                "mean_monthly_top3_positive_share",
                "mean_excess_sharpe",
                "mean_excess_annual_return",
            ],
            ascending=[False, False, False, True, False, False],
        )
        .reset_index(drop=True)
    )
    summary_df.to_csv(output_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")

    current_default = summary_df.loc[summary_df["profile_name"] == "state_liquidity_listwise_v1"].iloc[0]
    challenger = summary_df.loc[summary_df["profile_name"] == "dynamic_graph_no_priors"].iloc[0]
    baseline = summary_df.loc[summary_df["profile_name"] == "baseline_current"].iloc[0]
    challenger_vs_current = detail_df.loc[detail_df["profile_name"] == "dynamic_graph_no_priors"][
        ["window_label", "excess_annual_return", "excess_sharpe"]
    ].merge(
        detail_df.loc[detail_df["profile_name"] == "state_liquidity_listwise_v1"][
            ["window_label", "excess_annual_return", "excess_sharpe"]
        ],
        on="window_label",
        suffixes=("_challenger", "_current"),
        how="inner",
    )
    challenger_vs_current["delta_excess_annual_return"] = challenger_vs_current["excess_annual_return_challenger"] - challenger_vs_current["excess_annual_return_current"]
    challenger_vs_current["delta_excess_sharpe"] = challenger_vs_current["excess_sharpe_challenger"] - challenger_vs_current["excess_sharpe_current"]
    challenger_vs_current.to_csv(output_dir / "challenger_vs_current_default.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Dynamic Graph Liquid500 Challenger Head-to-Head",
        "",
        "## Protocol",
        "- universe: `liquid500`",
        "- research_time_unit: `calendar_months` with `valid_months=12` and `train_eval_window_months=6`",
        f"- objective: `{args.research_objective_mode} + train_eval_auto execution alignment`",
        f"- checkpoint_selection_objective: `{args.checkpoint_selection_objective}`",
        "- current default line: `state_liquidity_listwise_v1`",
        "- challenger: `dynamic_graph_no_priors`",
        "",
        "## Monthly Priority Summary",
        f"- current default positive-month ratio: `{_format_pct(float(current_default['mean_monthly_positive_ratio']))}`",
        f"- current default median monthly excess: `{_format_pct(float(current_default['mean_monthly_median_excess_return']))}`",
        f"- current default worst month excess: `{_format_pct(float(current_default['worst_monthly_excess_return']))}`",
        f"- challenger positive-month ratio: `{_format_pct(float(challenger['mean_monthly_positive_ratio']))}`",
        f"- challenger median monthly excess: `{_format_pct(float(challenger['mean_monthly_median_excess_return']))}`",
        f"- challenger worst month excess: `{_format_pct(float(challenger['worst_monthly_excess_return']))}`",
        "",
        "## Mean Summary",
        f"- baseline_current: `{_format_pct(float(baseline['mean_excess_annual_return']))} / {_format_num(float(baseline['mean_excess_sharpe']))}`",
        f"- state_liquidity_listwise_v1: `{_format_pct(float(current_default['mean_excess_annual_return']))} / {_format_num(float(current_default['mean_excess_sharpe']))}`",
        f"- dynamic_graph_no_priors: `{_format_pct(float(challenger['mean_excess_annual_return']))} / {_format_num(float(challenger['mean_excess_sharpe']))}`",
        "",
        "## Challenger Vs Current Default",
    ]
    for _, row in challenger_vs_current.iterrows():
        lines.append(
            f"- `{row['window_label']}`: challenger excess annual `{_format_pct(float(row['excess_annual_return_challenger']))}`, "
            f"current `{_format_pct(float(row['excess_annual_return_current']))}`, "
            f"delta `{_format_pct(float(row['delta_excess_annual_return']))}`; "
            f"challenger excess Sharpe `{_format_num(float(row['excess_sharpe_challenger']))}`, "
            f"current `{_format_num(float(row['excess_sharpe_current']))}`, "
            f"delta `{_format_num(float(row['delta_excess_sharpe']))}`"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_dir={output_dir}")


if __name__ == "__main__":
    main()
