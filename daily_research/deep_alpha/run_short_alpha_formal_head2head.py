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
from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.execution_alignment import default_auto_profile_argument
from daily_research.deep_alpha.short_alpha_profiles import build_profile_cli_args, get_profile
from daily_research.deep_alpha.family_epoch_budget import (
    DEFAULT_LATEST_MANIFEST_PATH,
    default_min_epochs_for_budget,
    resolve_epoch_budget_for_family,
)
from daily_research.deep_alpha.research_objective import (
    CHECKPOINT_SELECTION_OBJECTIVES,
    DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    DEFAULT_RESEARCH_OBJECTIVE_MODE,
    resolve_primary_backtest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXECUTION_ALIGNMENT_PROFILE_SET = default_auto_profile_argument()


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
RESEARCH_TIME_UNIT = "calendar_months"
TRAIN_EVAL_WINDOW_MONTHS = 6

PROFILES_TO_RUN: tuple[str, ...] = ("baseline_current", "state_liquidity_listwise_v1")

REUSE_METRICS: dict[str, dict[str, str]] = {}
REUSE_METRICS = {
    "baseline_current": {
        "20230216_20240229": "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/runs/baseline_current_20230216_20240229/metrics.json",
        "20240301_20250317": "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/runs/baseline_current_20240301_20250317/metrics.json",
        "20250318_20260331": "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/runs/baseline_current_20250318_20260331/metrics.json",
    },
    "state_liquidity_listwise_v1": {
        "20230216_20240229": "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/runs/state_liquidity_listwise_v1_20230216_20240229/metrics.json",
        "20240301_20250317": "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/runs/state_liquidity_listwise_v1_20240301_20250317/metrics.json",
        "20250318_20260331": "daily_research/output/short_alpha_formal_head2head_20260405_monthly_checkpoint_r1/runs/state_liquidity_listwise_v1_20250318_20260331/metrics.json",
    },
    "short_expert_monthly_v1": {
        "20240301_20250317": "daily_research/output/short_alpha_short_expert_formal_head2head_20260407_r3_shortalpha32/runs/short_expert_monthly_v1_20240301_20250317/metrics.json",
        "20250318_20260331": "daily_research/output/short_alpha_short_horizon_expert_review_20260406_r2_fullbudget/runs/short_expert_monthly_v1/metrics.json",
    },
}


def _resolve_family_key(profile_name: str) -> str:
    normalized = str(profile_name or "").strip().lower()
    return "baseline" if normalized == "baseline_current" else "short_alpha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal multi-window head-to-head for the short-alpha state_liquidity_listwise candidate.")
    parser.add_argument("--root-tag", default="short_alpha_formal_head2head_20260403_monthly_r1")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--force-rerun-profiles",
        default="",
        help="Comma-separated profile names that must rerun fresh under this root even when reuse metrics exist.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES_TO_RUN),
        help="Comma-separated short-alpha profile names from short_alpha_profiles.py",
    )
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


def _parse_name_list(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


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
    profile = get_profile(profile_name)
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
        RESEARCH_TIME_UNIT,
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
        str(default_min_epochs_for_budget(epoch_budget)),
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
        "--score-risk-mode",
        "subtract",
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
    cmd.extend(build_profile_cli_args(profile, include_objective_overrides=False))
    return cmd


def _resolve_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / f"{profile_name}_{window.label}" / "metrics.json"


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    monthly_filename = f"{monthly_label}.csv"
    monthly_path = run_dir / monthly_filename
    if monthly_path.exists():
        monthly_df = pd.read_csv(monthly_path)
        return summarize_monthly_diagnostics(monthly_df, return_column="excess_return")
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


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    force_rerun_profiles = _parse_name_list(args.force_rerun_profiles)
    if not profile_names:
        raise ValueError("No short-alpha profiles were provided.")

    collected_rows: list[dict] = []
    source_runs: dict[str, dict[str, str]] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        source_runs[profile.name] = {}
        for window in WINDOWS:
            reuse_path_str = REUSE_METRICS.get(profile.name, {}).get(window.label, "")
            profile_force_rerun = bool(args.force_rerun or profile.name in force_rerun_profiles)
            if reuse_path_str and not profile_force_rerun:
                metrics_path = (PROJECT_ROOT / reuse_path_str).resolve()
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Expected reuse metrics not found: {metrics_path}")
                reused_existing = True
            else:
                metrics_path = _resolve_metrics_path(root_tag, profile.name, window)
                reused_existing = False
                if not metrics_path.exists() or profile_force_rerun:
                    experiment_tag = f"{root_tag}/runs/{profile.name}_{window.label}"
                    command = _build_command(
                        python_executable=args.python_executable,
                        experiment_tag=experiment_tag,
                        profile_name=profile.name,
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
            holdout = dict(holdout)
            monthly_diagnostics = _load_primary_monthly_diagnostics(metrics_path.parent, metrics)
            family_key = _resolve_family_key(profile.name)
            source_runs[profile.name][window.label] = str(metrics_path.resolve())
            collected_rows.append(
                {
                    "profile_name": profile.name,
                    "window_label": window.label,
                    "metrics_path": str(metrics_path.resolve()),
                    "reused_existing": reused_existing,
                    "family_key": family_key,
                    "epoch_budget": int(metrics.get("epochs", 0) or resolve_epoch_budget_for_family(family_key, manifest_path=str(args.family_epoch_budget_manifest), fallback_epochs=8)),
                    "feature_count": int(metrics.get("feature_count", 0)),
                    "state_context": bool(metrics.get("state_context", False)),
                    "liquidity_context": bool(metrics.get("liquidity_context", False)),
                    "ranking_loss_weight": float(metrics.get("ranking_loss_weight", 0.0)),
                    "listwise_loss_weight": float(metrics.get("listwise_loss_weight", 0.0)),
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
                    "monthly_issue_flags": ",".join(monthly_diagnostics.get("issue_flags", [])),
                }
            )

    detail_df = pd.DataFrame(collected_rows).sort_values(["profile_name", "window_label"]).reset_index(drop=True)
    baseline_name = "baseline_current"
    candidate_name = "state_liquidity_listwise_v1"

    baseline_lookup = (
        detail_df[detail_df["profile_name"] == baseline_name]
        .set_index("window_label")[["excess_annual_return", "excess_sharpe", "avg_turnover", "excess_max_drawdown"]]
        .to_dict(orient="index")
    )
    compare_rows: list[dict] = []
    for _, row in detail_df[detail_df["profile_name"] == candidate_name].iterrows():
        base = baseline_lookup[str(row["window_label"])]
        compare_rows.append(
            {
                "window_label": row["window_label"],
                "candidate_excess_annual_return": row["excess_annual_return"],
                "baseline_excess_annual_return": base["excess_annual_return"],
                "candidate_excess_sharpe": row["excess_sharpe"],
                "baseline_excess_sharpe": base["excess_sharpe"],
                "candidate_avg_turnover": row["avg_turnover"],
                "baseline_avg_turnover": base["avg_turnover"],
                "candidate_excess_max_drawdown": row["excess_max_drawdown"],
                "baseline_excess_max_drawdown": base["excess_max_drawdown"],
                "delta_excess_annual_return": float(row["excess_annual_return"] - base["excess_annual_return"]),
                "delta_excess_sharpe": float(row["excess_sharpe"] - base["excess_sharpe"]),
                "delta_turnover": float(row["avg_turnover"] - base["avg_turnover"]),
                "wins_excess_annual_return": bool(row["excess_annual_return"] > base["excess_annual_return"]),
                "wins_excess_sharpe": bool(row["excess_sharpe"] > base["excess_sharpe"]),
            }
        )

    compare_df = pd.DataFrame(compare_rows)
    summary_rows: list[dict] = []
    for profile_name in profile_names:
        frame = detail_df[detail_df["profile_name"] == profile_name]
        summary_rows.append(
            {
                "profile_name": profile_name,
                "window_count": int(len(frame)),
                "mean_excess_total_return": float(frame["excess_total_return"].mean()),
                "mean_excess_annual_return": float(frame["excess_annual_return"].mean()),
                "mean_excess_sharpe": float(frame["excess_sharpe"].mean()),
                "mean_excess_max_drawdown": float(frame["excess_max_drawdown"].mean()),
                "mean_avg_turnover": float(frame["avg_turnover"].mean()),
                "mean_monthly_positive_ratio": float(frame["monthly_positive_ratio"].mean()),
                "mean_monthly_median_excess_return": float(frame["monthly_median_excess_return"].mean()),
                "worst_monthly_excess_return": float(frame["monthly_worst_excess_return"].min()),
                "mean_monthly_top3_positive_share": float(frame["monthly_top3_positive_share"].mean()),
                "max_monthly_negative_streak": int(frame["monthly_longest_negative_streak"].max()),
                "epoch_budget": int(frame["epoch_budget"].max()),
                "wins_by_excess_annual_return": int(compare_df["wins_excess_annual_return"].sum()) if profile_name == candidate_name else 0,
                "wins_by_excess_sharpe": int(compare_df["wins_excess_sharpe"].sum()) if profile_name == candidate_name else 0,
                "reused_window_count": int(frame["reused_existing"].sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        [
            "mean_monthly_positive_ratio",
            "mean_monthly_median_excess_return",
            "worst_monthly_excess_return",
            "mean_monthly_top3_positive_share",
            "mean_excess_sharpe",
            "mean_excess_annual_return",
        ],
        ascending=[False, False, False, True, False, False],
    ).reset_index(drop=True)

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_dir / "window_detail.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(output_dir / "head2head_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_summary = summary_df.loc[summary_df["profile_name"] == candidate_name].iloc[0]
    baseline_summary = summary_df.loc[summary_df["profile_name"] == baseline_name].iloc[0]
    lines = [
        "# Short Alpha Formal Head-to-Head",
        "",
        "## Protocol",
        "- window_count: `3`",
        f"- research_time_unit: `{RESEARCH_TIME_UNIT}` with `valid_months={WINDOWS[0].valid_months}` and `train_eval_window_months={TRAIN_EVAL_WINDOW_MONTHS}`",
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- backbone: `patch_transformer + dynamic_graph_v1`",
        f"- objective: `{args.research_objective_mode} + train_eval_auto execution alignment`",
        f"- checkpoint_selection_objective: `{args.checkpoint_selection_objective}`",
        "- candidate: `state_liquidity_listwise_v1`",
        "- baseline: `baseline_current`",
        f"- epoch_budget_baseline: `{int(baseline_summary['epoch_budget'])}`",
        f"- epoch_budget_candidate: `{int(candidate_summary['epoch_budget'])}`",
        "",
        "## Monthly Priority Summary",
        f"- candidate positive-month ratio: `{float(candidate_summary['mean_monthly_positive_ratio']):.2%}`",
        f"- candidate median monthly excess: `{_format_pct(float(candidate_summary['mean_monthly_median_excess_return']))}`",
        f"- candidate worst month excess: `{_format_pct(float(candidate_summary['worst_monthly_excess_return']))}`",
        f"- candidate top3 positive-month share: `{float(candidate_summary['mean_monthly_top3_positive_share']):.2%}`",
        f"- baseline positive-month ratio: `{float(baseline_summary['mean_monthly_positive_ratio']):.2%}`",
        f"- baseline median monthly excess: `{_format_pct(float(baseline_summary['mean_monthly_median_excess_return']))}`",
        f"- baseline worst month excess: `{_format_pct(float(baseline_summary['worst_monthly_excess_return']))}`",
        f"- baseline top3 positive-month share: `{float(baseline_summary['mean_monthly_top3_positive_share']):.2%}`",
        "",
        "## Mean Summary",
        f"- candidate excess annual: `{_format_pct(float(candidate_summary['mean_excess_annual_return']))}`",
        f"- candidate excess Sharpe: `{_format_float(float(candidate_summary['mean_excess_sharpe']))}`",
        f"- baseline excess annual: `{_format_pct(float(baseline_summary['mean_excess_annual_return']))}`",
        f"- baseline excess Sharpe: `{_format_float(float(baseline_summary['mean_excess_sharpe']))}`",
        f"- wins by excess annual: `{int(candidate_summary['wins_by_excess_annual_return'])}/{len(WINDOWS)}`",
        f"- wins by excess Sharpe: `{int(candidate_summary['wins_by_excess_sharpe'])}/{len(WINDOWS)}`",
        "",
        "## Window Head-to-Head",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            f"- `{row['window_label']}`: candidate excess annual {_format_pct(float(row['candidate_excess_annual_return']))}, "
            f"baseline {_format_pct(float(row['baseline_excess_annual_return']))}, "
            f"delta {_format_pct(float(row['delta_excess_annual_return']))}; "
            f"candidate excess Sharpe {_format_float(float(row['candidate_excess_sharpe']))}, "
            f"baseline {_format_float(float(row['baseline_excess_sharpe']))}, "
            f"delta {_format_float(float(row['delta_excess_sharpe']))}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_dir={output_dir}")


if __name__ == "__main__":
    main()
