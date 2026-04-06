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

from daily_research.deep_alpha.short_alpha_profiles import (
    DEFAULT_SHORT_ALPHA_PROFILE,
    build_profile_cli_args,
    get_profile,
    list_profile_lines,
)
from daily_research.deep_alpha.research_objective import resolve_primary_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"


@dataclass(frozen=True)
class RecentFormalWindow:
    train_end: str = "2025-03-17"
    valid_start: str = "2025-03-18"
    valid_months: int = 12


WINDOW = RecentFormalWindow()
RESEARCH_TIME_UNIT = "calendar_months"
TRAIN_EVAL_WINDOW_MONTHS = 6
DEFAULT_PROFILES = (
    "baseline_current",
    "state_context_v1",
    "state_liquidity_listwise_v1",
    "short_target_v1",
    "short_input_v1",
    "short_combo_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the short-alpha experiment matrix on the current deep_alpha winner.")
    parser.add_argument("--root-tag", default="deep_alpha_short_alpha_matrix_20260403_monthly_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated short-alpha profile names from short_alpha_profiles.py",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_command(*, python_executable: str, experiment_tag: str, profile_name: str) -> list[str]:
    profile = get_profile(profile_name)
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
        "8",
        "--min-epochs",
        "1",
        "--early-stop-patience",
        "8",
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


def _resolve_metrics_path(root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / profile_name / "metrics.json"


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        print("\n".join(list_profile_lines()))
        return

    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No short-alpha profiles were provided.")

    rows: list[dict] = []
    source_runs: dict[str, str] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        metrics_path = _resolve_metrics_path(root_tag, profile.name)
        if not metrics_path.exists() or args.force_rerun:
            experiment_tag = f"{root_tag}/runs/{profile.name}"
            command = _build_command(
                python_executable=args.python_executable,
                experiment_tag=experiment_tag,
                profile_name=profile.name,
            )
            _run_command(command)
        metrics = _load_metrics(metrics_path)
        _, holdout = resolve_primary_backtest(metrics)
        holdout = dict(holdout)
        source_runs[profile.name] = str(metrics_path.resolve())
        rows.append(
            {
                "profile_name": profile.name,
                "description": profile.description,
                "metrics_path": str(metrics_path.resolve()),
                "feature_count": int(metrics.get("feature_count", 0)),
                "target_names": ",".join(metrics.get("target_names", [])),
                "state_context": bool(metrics.get("state_context", False)),
                "liquidity_context": bool(metrics.get("liquidity_context", False)),
                "short_alpha_features": bool(metrics.get("short_alpha_features", False)),
                "score_head_method": str(metrics.get("score_head_method", profile.score_head_method or "manual")),
                "adaptive_task_weights": bool(metrics.get("adaptive_task_weights", profile.adaptive_task_weights)),
                "checkpoint_selection_objective": str(
                    metrics.get("checkpoint_selection_objective", profile.checkpoint_selection_objective or "primary_annual_return")
                ),
                "research_objective_mode": str(
                    metrics.get("research_objective_mode", profile.research_objective_mode or "execution_first")
                ),
                "ranking_loss_weight": float(metrics.get("ranking_loss_weight", 0.0)),
                "listwise_loss_weight": float(metrics.get("listwise_loss_weight", 0.0)),
                "breakout_event_loss_weight": float(metrics.get("breakout_event_loss_weight", 0.0)),
                "clean_breakout_event_loss_weight": float(metrics.get("clean_breakout_event_loss_weight", 0.0)),
                "annual_return": float(holdout.get("annual_return", 0.0)),
                "excess_total_return": float(holdout.get("excess_total_return", 0.0)),
                "excess_annual_return": float(holdout.get("excess_annual_return", 0.0)),
                "excess_sharpe": float(holdout.get("excess_sharpe", 0.0)),
                "excess_max_drawdown": float(holdout.get("excess_max_drawdown", 0.0)),
                "avg_turnover": float(holdout.get("avg_turnover", 0.0)),
                "avg_holding_count": float(holdout.get("avg_holding_count", 0.0)),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["excess_annual_return", "excess_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    baseline_row = summary_df.loc[summary_df["profile_name"] == DEFAULT_SHORT_ALPHA_PROFILE].iloc[0]
    summary_df["delta_excess_annual_return"] = summary_df["excess_annual_return"] - float(baseline_row["excess_annual_return"])
    summary_df["delta_excess_sharpe"] = summary_df["excess_sharpe"] - float(baseline_row["excess_sharpe"])
    summary_df["delta_turnover"] = summary_df["avg_turnover"] - float(baseline_row["avg_turnover"])

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(
        json.dumps(source_runs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_row = summary_df.iloc[0]
    lines = [
        "# Short Alpha Experiment Matrix",
        "",
        "## Protocol",
        f"- window: `{WINDOW.valid_start} -> 2026-04-01`",
        f"- research_time_unit: `{RESEARCH_TIME_UNIT}` with `valid_months={WINDOW.valid_months}` and `train_eval_window_months={TRAIN_EVAL_WINDOW_MONTHS}`",
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- backbone: `patch_transformer + dynamic_graph_v1` unless a profile explicitly changes inputs/targets/losses",
        "- objective: `execution_first + train_eval_auto execution alignment`",
        "",
        "## Winner",
        f"- best_excess_annual_return: `{best_row['profile_name']}` = {_format_pct(float(best_row['excess_annual_return']))}",
        f"- excess_sharpe: {_format_float(float(best_row['excess_sharpe']))}",
        f"- avg_turnover: {_format_float(float(best_row['avg_turnover']))}",
        "",
        "## Full Ranking",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['profile_name']}`: excess annual {_format_pct(float(row['excess_annual_return']))}, "
            f"excess Sharpe {_format_float(float(row['excess_sharpe']))}, "
            f"drawdown {_format_pct(float(row['excess_max_drawdown']))}, "
            f"turnover {_format_float(float(row['avg_turnover']))}, "
            f"delta vs baseline excess annual {_format_pct(float(row['delta_excess_annual_return']))}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_dir={output_dir}")


if __name__ == "__main__":
    main()
