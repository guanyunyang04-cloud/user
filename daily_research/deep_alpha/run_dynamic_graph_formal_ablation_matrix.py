from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.dynamic_graph_profiles import get_profile
from daily_research.deep_alpha.family_epoch_budget import DEFAULT_LATEST_MANIFEST_PATH, resolve_epoch_budget_for_family
from daily_research.deep_alpha.research_objective import resolve_primary_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"


@dataclass(frozen=True)
class FormalWindow:
    label: str
    train_end: str
    valid_start: str


WINDOWS: tuple[FormalWindow, ...] = (
    FormalWindow(label="20230216_20240229", train_end="2023-02-15", valid_start="2023-02-16"),
    FormalWindow(label="20240301_20250317", train_end="2024-02-29", valid_start="2024-03-01"),
    FormalWindow(label="20250318_20260331", train_end="2025-03-17", valid_start="2025-03-18"),
)
RESEARCH_TIME_UNIT = "calendar_months"
FORMAL_VALID_MONTHS = 12
TRAIN_EVAL_WINDOW_MONTHS = 6

PROFILES_TO_RUN: tuple[str, ...] = (
    "plain_baseline",
    "dynamic_graph_v1",
    "dynamic_graph_no_priors",
    "dynamic_graph_topk4",
    "dynamic_graph_topk12",
)

REUSE_METRICS: dict[str, dict[str, str]] = {}


def _resolve_family_key(profile_name: str) -> str:
    normalized = str(profile_name or "").strip().lower()
    return "baseline" if normalized == "plain_baseline" else "dynamic_graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal dynamic-graph ablation matrix under the strict liquid800 mainboard protocol.")
    parser.add_argument("--root-tag", default="dynamic_graph_ablation_formal_20260403_monthly_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--force-raw-cache-path", default="")
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES_TO_RUN),
        help="Comma-separated profile names from dynamic_graph_profiles.py",
    )
    parser.add_argument("--family-epoch-budget-manifest", default=str(DEFAULT_LATEST_MANIFEST_PATH))
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_base_command(
    python_executable: str,
    experiment_tag: str,
    window: FormalWindow,
    epoch_budget: int,
    *,
    force_raw_cache_path: str = "",
) -> list[str]:
    command = [
        python_executable,
        str(RUN_SCRIPT),
        "--data-source",
        "tq",
        "--rolling-liquidity-pool",
        "liquid800",
        "--pool-rebalance-days",
        "21",
        "--pool-adv-window",
        "20",
        "--start-date",
        "20220101",
        "--end-date",
        "20260401",
        "--benchmark",
        "000300.SH",
        "--lookback-window",
        "120",
        "--prediction-horizons",
        "5,10,20",
        "--research-time-unit",
        RESEARCH_TIME_UNIT,
        "--train-end-date",
        window.train_end,
        "--valid-start-date",
        window.valid_start,
        "--valid-days",
        "0",
        "--valid-months",
        str(FORMAL_VALID_MONTHS),
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
        "4",
        "--early-stop-patience",
        str(max(int(epoch_budget), 4)),
        "--lr-plateau-patience",
        "1",
        "--lr-plateau-factor",
        "0.5",
        "--min-improvement",
        "0.0001",
        "--return-loss-mode",
        "regression",
        "--return-target-transform",
        "raw",
        "--score-risk-mode",
        "state_gate",
        "--score-risk-state-thresholds",
        "0.0,0.2,0.35,0.5,0.65",
        "--score-head-method",
        "manual",
        "--train-eval-window-days",
        "0",
        "--train-eval-window-months",
        str(TRAIN_EVAL_WINDOW_MONTHS),
        "--ranking-loss-weight",
        "0.0",
        "--listwise-loss-weight",
        "0.0",
        "--random-seed",
        "7",
        "--market-state-count",
        "4",
        "--holding-count",
        "5",
        "--rebalance-freq",
        "1d",
        "--max-weight",
        "0.25",
        "--min-adv20",
        "50000.0",
        "--min-price",
        "2.0",
        "--max-price",
        "300.0",
        "--num-workers",
        "0",
        "--pin-memory",
        "--use-amp",
        "--safe-runtime-profile",
        "--experiment-tag",
        experiment_tag,
    ]
    if str(force_raw_cache_path or "").strip():
        command.extend(["--force-raw-cache-path", str(force_raw_cache_path).strip()])
    return command


def _resolve_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    run_name = f"{profile_name}_{window.label}"
    return OUTPUT_ROOT / root_tag / "runs" / run_name / "metrics.json"


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_holdout(metrics: dict) -> dict:
    _, holdout = resolve_primary_backtest(metrics)
    return dict(holdout)


def _summarize_profile(profile_name: str, profile_rows: list[dict]) -> dict:
    frame = pd.DataFrame(profile_rows)
    return {
        "profile_name": profile_name,
        "window_count": int(len(frame)),
        "mean_epoch_budget": float(frame["epoch_budget"].mean()),
        "mean_excess_total_return": float(frame["excess_total_return"].mean()),
        "mean_excess_annual_return": float(frame["excess_annual_return"].mean()),
        "mean_excess_sharpe": float(frame["excess_sharpe"].mean()),
        "mean_excess_max_drawdown": float(frame["excess_max_drawdown"].mean()),
        "mean_avg_turnover": float(frame["avg_turnover"].mean()),
        "mean_avg_holding_count": float(frame["avg_holding_count"].mean()),
        "mean_feature_count": float(frame["feature_count"].mean()),
        "reused_window_count": int(frame["reused_existing"].sum()),
    }


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No ablation profiles were provided.")

    source_runs: dict[str, dict[str, str]] = {}
    collected_rows: list[dict] = []

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
                    family_key = _resolve_family_key(profile.name)
                    epoch_budget = resolve_epoch_budget_for_family(
                        family_key,
                        manifest_path=str(args.family_epoch_budget_manifest),
                        fallback_epochs=8,
                    )
                    command = _build_base_command(
                        args.python_executable,
                        experiment_tag,
                        window,
                        epoch_budget,
                        force_raw_cache_path=str(args.force_raw_cache_path),
                    )
                    if profile.dynamic_graph_layer:
                        command.extend(
                            [
                                "--dynamic-graph-layer",
                                "--dynamic-graph-top-k",
                                str(profile.top_k),
                                "--dynamic-graph-temperature",
                                str(profile.temperature),
                                "--dynamic-graph-industry-boost",
                                str(profile.industry_boost),
                                "--dynamic-graph-style-boost",
                                str(profile.style_boost),
                            ]
                        )
                    _run_command(command)
            metrics = _load_metrics(metrics_path)
            holdout = _extract_holdout(metrics)
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
                    "dynamic_graph_layer": bool(metrics.get("dynamic_graph_layer", False)),
                    "dynamic_graph_top_k": int(metrics.get("dynamic_graph_top_k", 0)) if metrics.get("dynamic_graph_layer", False) else 0,
                    "dynamic_graph_industry_boost": float(metrics.get("dynamic_graph_industry_boost", 0.0)) if metrics.get("dynamic_graph_layer", False) else 0.0,
                    "dynamic_graph_style_boost": float(metrics.get("dynamic_graph_style_boost", 0.0)) if metrics.get("dynamic_graph_layer", False) else 0.0,
                    "annual_return": float(holdout.get("annual_return", 0.0)),
                    "excess_total_return": float(holdout.get("excess_total_return", 0.0)),
                    "excess_annual_return": float(holdout.get("excess_annual_return", 0.0)),
                    "excess_sharpe": float(holdout.get("excess_sharpe", 0.0)),
                    "excess_max_drawdown": float(holdout.get("excess_max_drawdown", 0.0)),
                    "avg_turnover": float(holdout.get("avg_turnover", 0.0)),
                    "avg_holding_count": float(holdout.get("avg_holding_count", 0.0)),
                }
            )

    detailed_df = pd.DataFrame(collected_rows).sort_values(["profile_name", "window_label"]).reset_index(drop=True)
    summary_rows = [
        _summarize_profile(profile_name, detailed_df[detailed_df["profile_name"] == profile_name].to_dict(orient="records"))
        for profile_name in profile_names
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_excess_sharpe", ascending=False).reset_index(drop=True)

    plain_lookup = (
        detailed_df[detailed_df["profile_name"] == "plain_baseline"]
        .set_index("window_label")[["excess_annual_return", "excess_sharpe"]]
        .to_dict(orient="index")
    )
    winner_lookup = (
        detailed_df[detailed_df["profile_name"] == "dynamic_graph_v1"]
        .set_index("window_label")[["excess_annual_return", "excess_sharpe"]]
        .to_dict(orient="index")
    )
    compare_rows: list[dict] = []
    for profile_name in profile_names:
        if profile_name == "plain_baseline":
            continue
        profile_df = detailed_df[detailed_df["profile_name"] == profile_name]
        for _, row in profile_df.iterrows():
            window_label = str(row["window_label"])
            plain_row = plain_lookup[window_label]
            winner_row = winner_lookup[window_label]
            compare_rows.append(
                {
                    "profile_name": profile_name,
                    "window_label": window_label,
                    "minus_plain_excess_annual_return": float(row["excess_annual_return"] - plain_row["excess_annual_return"]),
                    "minus_plain_excess_sharpe": float(row["excess_sharpe"] - plain_row["excess_sharpe"]),
                    "minus_winner_excess_annual_return": float(row["excess_annual_return"] - winner_row["excess_annual_return"]),
                    "minus_winner_excess_sharpe": float(row["excess_sharpe"] - winner_row["excess_sharpe"]),
                    "wins_plain_by_excess_annual": bool(row["excess_annual_return"] > plain_row["excess_annual_return"]),
                    "wins_plain_by_excess_sharpe": bool(row["excess_sharpe"] > plain_row["excess_sharpe"]),
                    "wins_winner_by_excess_annual": bool(row["excess_annual_return"] > winner_row["excess_annual_return"]),
                    "wins_winner_by_excess_sharpe": bool(row["excess_sharpe"] > winner_row["excess_sharpe"]),
                }
            )
    compare_df = pd.DataFrame(compare_rows)

    verdict_lines = [
        "# Dynamic Graph Formal Ablation Verdict",
        "",
        f"- generated_at: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- root_tag: `{root_tag}`",
        (
            "- protocol: strict rolling `liquid800` / mainboard-only / `next_open` / 3 equal walk-forward windows / "
            f"`{RESEARCH_TIME_UNIT}` with `valid_months={FORMAL_VALID_MONTHS}` and `train_eval_window_months={TRAIN_EVAL_WINDOW_MONTHS}`"
        ),
        "- reused_existing: `0`; all profiles were rerun under the monthly execution-first protocol.",
        "",
        "## Mean Summary",
    ]
    for _, row in summary_df.iterrows():
        verdict_lines.append(
            "- "
            f"{row['profile_name']}: "
            f"epoch budget {int(row['mean_epoch_budget'])}, "
            f"excess annual {_format_pct(row['mean_excess_annual_return'])}, "
            f"excess Sharpe {_format_float(row['mean_excess_sharpe'])}, "
            f"excess max drawdown {_format_pct(row['mean_excess_max_drawdown'])}, "
            f"avg turnover {_format_float(row['mean_avg_turnover'])}"
        )

    if "dynamic_graph_v1" in summary_df["profile_name"].values:
        winner_row = summary_df.loc[summary_df["profile_name"] == "dynamic_graph_v1"].iloc[0]
        verdict_lines.extend(
            [
                "",
                "## Default Winner Check",
                "- "
                f"`dynamic_graph_v1` mean excess annual = {_format_pct(winner_row['mean_excess_annual_return'])}, "
                f"mean excess Sharpe = {_format_float(winner_row['mean_excess_sharpe'])}.",
            ]
        )
    if "dynamic_graph_no_priors" in summary_df["profile_name"].values:
        noprior_row = summary_df.loc[summary_df["profile_name"] == "dynamic_graph_no_priors"].iloc[0]
        verdict_lines.append(
            "- "
            f"`dynamic_graph_no_priors` mean excess annual = {_format_pct(noprior_row['mean_excess_annual_return'])}, "
            f"mean excess Sharpe = {_format_float(noprior_row['mean_excess_sharpe'])}. "
            "Use this row to judge how much of the edge survives without industry/style priors."
        )
    verdict_lines.extend(
        [
            "",
            "## Output Files",
            f"- summary_csv: `daily_research/output/{root_tag}/ablation_summary.csv`",
            f"- detailed_csv: `daily_research/output/{root_tag}/ablation_window_details.csv`",
            f"- compare_csv: `daily_research/output/{root_tag}/ablation_vs_plain_and_winner.csv`",
            f"- source_runs_json: `daily_research/output/{root_tag}/source_runs.json`",
        ]
    )

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    detailed_df.to_csv(output_dir / "ablation_window_details.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(output_dir / "ablation_vs_plain_and_winner.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "source_runs.json", "w", encoding="utf-8") as f:
        json.dump(source_runs, f, ensure_ascii=False, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(verdict_lines) + "\n")

    print(f"output_dir={output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
