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

from daily_research.deep_alpha.short_alpha_profiles import get_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"


@dataclass(frozen=True)
class FormalWindow:
    label: str
    train_end: str
    valid_start: str
    valid_days: int = 252


WINDOWS: tuple[FormalWindow, ...] = (
    FormalWindow(label="20230216_20240229", train_end="2023-02-15", valid_start="2023-02-16"),
    FormalWindow(label="20240301_20250317", train_end="2024-02-29", valid_start="2024-03-01"),
    FormalWindow(label="20250318_20260331", train_end="2025-03-17", valid_start="2025-03-18"),
)

PROFILES_TO_RUN: tuple[str, ...] = ("baseline_current", "state_liquidity_listwise_v1")

REUSE_METRICS: dict[str, dict[str, str]] = {
    "baseline_current": {
        "20250318_20260331": "daily_research/output/deep_alpha_short_alpha_matrix_20260402_r1/runs/baseline_current/metrics.json",
    },
    "state_liquidity_listwise_v1": {
        "20250318_20260331": "daily_research/output/deep_alpha_short_alpha_matrix_20260402_r1/runs/state_liquidity_listwise_v1/metrics.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal multi-window head-to-head for the short-alpha state_liquidity_listwise candidate.")
    parser.add_argument("--root-tag", default="short_alpha_formal_head2head_20260402_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES_TO_RUN),
        help="Comma-separated short-alpha profile names from short_alpha_profiles.py",
    )
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_command(*, python_executable: str, experiment_tag: str, profile_name: str, window: FormalWindow) -> list[str]:
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
        "--train-end-date",
        window.train_end,
        "--valid-start-date",
        window.valid_start,
        "--valid-days",
        str(window.valid_days),
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
        "--score-head-method",
        "manual",
        "--train-eval-window-days",
        "126",
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
        "--prediction-horizons",
        profile.prediction_horizons,
        "--task-loss-weights",
        profile.task_loss_weights,
        "--score-horizon-weights",
        profile.score_horizon_weights,
        "--ranking-loss-weight",
        str(profile.ranking_loss_weight),
        "--listwise-loss-weight",
        str(profile.listwise_loss_weight),
        "--listwise-temperature",
        str(profile.listwise_temperature),
        "--breakout-event-horizon",
        str(profile.breakout_event_horizon),
        "--breakout-event-threshold",
        str(profile.breakout_event_threshold),
        "--breakout-event-pullback-limit",
        str(profile.breakout_event_pullback_limit),
        "--breakout-event-loss-weight",
        str(profile.breakout_event_loss_weight),
        "--clean-breakout-event-loss-weight",
        str(profile.clean_breakout_event_loss_weight),
        "--experiment-tag",
        experiment_tag,
    ]
    if profile.state_context:
        cmd.append("--state-context")
    if profile.liquidity_context:
        cmd.append("--liquidity-context")
    if profile.short_alpha_features:
        cmd.append("--short-alpha-features")
    return cmd


def _resolve_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / f"{profile_name}_{window.label}" / "metrics.json"


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No short-alpha profiles were provided.")

    collected_rows: list[dict] = []
    source_runs: dict[str, dict[str, str]] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        source_runs[profile.name] = {}
        for window in WINDOWS:
            reuse_path_str = REUSE_METRICS.get(profile.name, {}).get(window.label, "")
            if reuse_path_str and not args.force_rerun:
                metrics_path = (PROJECT_ROOT / reuse_path_str).resolve()
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Expected reuse metrics not found: {metrics_path}")
                reused_existing = True
            else:
                metrics_path = _resolve_metrics_path(root_tag, profile.name, window)
                reused_existing = False
                if not metrics_path.exists() or args.force_rerun:
                    experiment_tag = f"{root_tag}/runs/{profile.name}_{window.label}"
                    command = _build_command(
                        python_executable=args.python_executable,
                        experiment_tag=experiment_tag,
                        profile_name=profile.name,
                        window=window,
                    )
                    _run_command(command)
            metrics = _load_metrics(metrics_path)
            holdout = dict(metrics.get("holdout_backtest", {}))
            source_runs[profile.name][window.label] = str(metrics_path.resolve())
            collected_rows.append(
                {
                    "profile_name": profile.name,
                    "window_label": window.label,
                    "metrics_path": str(metrics_path.resolve()),
                    "reused_existing": reused_existing,
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
                "wins_by_excess_annual_return": int(compare_df["wins_excess_annual_return"].sum()) if profile_name == candidate_name else 0,
                "wins_by_excess_sharpe": int(compare_df["wins_excess_sharpe"].sum()) if profile_name == candidate_name else 0,
                "reused_window_count": int(frame["reused_existing"].sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_excess_sharpe", ascending=False).reset_index(drop=True)

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
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- backbone: `patch_transformer + dynamic_graph_v1`",
        "- candidate: `state_liquidity_listwise_v1`",
        "- baseline: `baseline_current`",
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
