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

from daily_research.deep_alpha.architecture_profiles import get_profile
from daily_research.deep_alpha.execution_alignment import default_auto_profile_argument
from daily_research.deep_alpha.research_objective import (
    CHECKPOINT_SELECTION_OBJECTIVES,
    DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    DEFAULT_RESEARCH_OBJECTIVE_MODE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"

EXECUTION_ALIGNMENT_CANDIDATES: tuple[str, ...] = (default_auto_profile_argument(),)


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
TRAIN_EVAL_WINDOW_MONTHS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a monthly execution-first formal sensitivity study for finetune epoch budget."
    )
    parser.add_argument("--root-tag", default="deep_alpha_epoch_budget_formal_20260404_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--profile", default="baseline_current")
    parser.add_argument("--epoch-budgets", default="32,48,64")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
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


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _parse_epoch_budgets(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Epoch budget must be positive: {token}")
        values.append(value)
    ordered = sorted(set(values))
    if not ordered:
        raise ValueError("No epoch budgets were provided.")
    return ordered


def _build_research_command(
    *,
    python_executable: str,
    experiment_tag: str,
    profile_name: str,
    epoch_budget: int,
    window: FormalWindow,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
) -> list[str]:
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
        str(profile.hidden_dim),
        "--encoder-family",
        profile.encoder_family,
        "--patch-len",
        str(profile.patch_len),
        "--transformer-heads",
        str(profile.transformer_heads),
        "--transformer-layers",
        str(profile.transformer_layers),
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
        "--score-risk-mode",
        "subtract",
        "--score-head-method",
        "manual",
        "--train-eval-window-days",
        "0",
        "--train-eval-window-months",
        str(TRAIN_EVAL_WINDOW_MONTHS),
        "--num-workers",
        "0",
        "--pin-memory",
        "--use-amp",
        "--no-safe-runtime-profile",
        "--prediction-horizons",
        "5,10,20",
        "--task-loss-weights",
        "5:0.2,10:0.3,20:0.5,downside:0.35",
        "--score-horizon-weights",
        "5:0.2,10:0.3,20:0.5",
        "--execution-alignment-mode",
        "train_eval_auto",
        "--execution-alignment-objective",
        execution_alignment_objective,
        "--execution-alignment-candidate-profiles",
        ",".join(EXECUTION_ALIGNMENT_CANDIDATES),
        "--execution-alignment-transaction-cost-bps",
        str(transaction_cost_bps),
        "--execution-alignment-slippage-bps",
        str(slippage_bps),
        "--execution-alignment-sell-tax-bps",
        str(sell_tax_bps),
        "--experiment-tag",
        experiment_tag,
    ]
    if profile.dynamic_graph_layer:
        cmd.extend(
            [
                "--dynamic-graph-layer",
                "--dynamic-graph-top-k",
                str(profile.dynamic_graph_top_k),
                "--dynamic-graph-temperature",
                str(profile.dynamic_graph_temperature),
                "--dynamic-graph-industry-boost",
                str(profile.dynamic_graph_industry_boost),
                "--dynamic-graph-style-boost",
                str(profile.dynamic_graph_style_boost),
            ]
        )
    if profile.relation_layer:
        cmd.append("--relation-layer")
    if profile.state_context:
        cmd.append("--state-context")
    if profile.liquidity_context:
        cmd.append("--liquidity-context")
    if profile.structure_context:
        cmd.append("--structure-context")
    if profile.aux_structure_task:
        cmd.extend(
            [
                "--aux-structure-task",
                "--aux-structure-loss-weight",
                str(profile.aux_structure_loss_weight),
                "--aux-structure-label-smoothing",
                str(profile.aux_structure_label_smoothing),
            ]
        )
    if profile.structure_prototype_task:
        cmd.extend(
            [
                "--structure-prototype-task",
                "--structure-prototype-loss-weight",
                str(profile.structure_prototype_loss_weight),
                "--structure-prototype-temperature",
                str(profile.structure_prototype_temperature),
            ]
        )
    return cmd


def _build_external_replay_command(
    *,
    python_executable: str,
    output_root: Path,
    replay_tag: str,
    candidate_label: str,
    score_panel_csv: Path,
    target_weight_panel_csv: Path,
    valid_start: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> list[str]:
    return [
        python_executable,
        str(EXTERNAL_REPLAY_SCRIPT),
        "--data-source",
        "tq",
        "--benchmark",
        "000300.SH",
        "--start-date",
        valid_start.replace("-", ""),
        "--end-date",
        "20260401",
        "--score-panel-csv",
        str(score_panel_csv),
        "--target-weight-panel-csv",
        str(target_weight_panel_csv),
        "--candidate-label",
        candidate_label,
        "--rebalance-freq",
        "1d",
        "--rebalance-offset-mode",
        "single",
        "--rebalance-anchor-date",
        valid_start.replace("-", ""),
        "--transaction-cost-bps",
        str(transaction_cost_bps),
        "--slippage-bps",
        str(slippage_bps),
        "--sell-tax-bps",
        str(sell_tax_bps),
        "--no-market-regime-filter",
        "--output-dir",
        str(output_root),
        "--experiment-tag",
        replay_tag,
    ]


def _resolve_metrics_path(root_tag: str, profile_name: str, epoch_budget: int, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / f"{profile_name}_e{epoch_budget}_{window.label}" / "metrics.json"


def _resolve_replay_metrics_path(root_tag: str, profile_name: str, epoch_budget: int, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "replays" / f"{profile_name}_e{epoch_budget}_{window.label}" / "metrics.json"


def _load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"metrics.json is not a JSON object: {path}")
    return payload


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    root_tag = str(args.root_tag).strip()
    profile_name = str(args.profile).strip()
    epoch_budgets = _parse_epoch_budgets(args.epoch_budgets)
    profile = get_profile(profile_name)

    collected_rows: list[dict[str, Any]] = []
    source_runs: dict[str, dict[str, str]] = {}
    replay_runs: dict[str, dict[str, str]] = {}

    for epoch_budget in epoch_budgets:
        budget_key = f"e{epoch_budget}"
        source_runs[budget_key] = {}
        replay_runs[budget_key] = {}
        for window in WINDOWS:
            metrics_path = _resolve_metrics_path(root_tag, profile.name, epoch_budget, window)
            if not metrics_path.exists() or args.force_rerun:
                experiment_tag = f"{root_tag}/runs/{profile.name}_e{epoch_budget}_{window.label}"
                command = _build_research_command(
                    python_executable=args.python_executable,
                    experiment_tag=experiment_tag,
                    profile_name=profile.name,
                    epoch_budget=epoch_budget,
                    window=window,
                    transaction_cost_bps=args.transaction_cost_bps,
                    slippage_bps=args.slippage_bps,
                    sell_tax_bps=args.sell_tax_bps,
                    research_objective_mode=str(args.research_objective_mode),
                    checkpoint_selection_objective=str(args.checkpoint_selection_objective),
                    checkpoint_selection_min_improvement=float(args.checkpoint_selection_min_improvement),
                    execution_alignment_objective=str(args.execution_alignment_objective),
                )
                _run_command(command)

            metrics = _load_metrics(metrics_path)
            run_dir = metrics_path.parent
            replay_metrics_path = _resolve_replay_metrics_path(root_tag, profile.name, epoch_budget, window)
            if not replay_metrics_path.exists() or args.force_rerun:
                replay_command = _build_external_replay_command(
                    python_executable=args.python_executable,
                    output_root=OUTPUT_ROOT / root_tag,
                    replay_tag=f"replays/{profile.name}_e{epoch_budget}_{window.label}",
                    candidate_label=f"{profile.name}_e{epoch_budget}_{window.label}",
                    score_panel_csv=run_dir / "execution_aligned_daily_score_panel.csv",
                    target_weight_panel_csv=run_dir / "execution_aligned_daily_target_weight_panel.csv",
                    valid_start=window.valid_start,
                    transaction_cost_bps=args.transaction_cost_bps,
                    slippage_bps=args.slippage_bps,
                    sell_tax_bps=args.sell_tax_bps,
                )
                _run_command(replay_command)
            replay_metrics = _load_metrics(replay_metrics_path)

            raw_holdout = dict(metrics.get("holdout_backtest", {}))
            aligned_holdout = dict(metrics.get("execution_aligned_holdout_backtest", {}))
            training = dict(metrics.get("training_diagnostics", {}))
            selected_train = dict(metrics.get("execution_alignment_selected_train_metrics", {}))

            source_runs[budget_key][window.label] = str(metrics_path.resolve())
            replay_runs[budget_key][window.label] = str(replay_metrics_path.resolve())
            epochs_completed = int(training.get("epochs_completed", 0) or 0)
            selected_epoch = int(training.get("selected_epoch", 0) or 0)
            best_epoch = int(training.get("best_epoch", 0) or 0)
            collected_rows.append(
                {
                    "profile_name": profile.name,
                    "epoch_budget": int(epoch_budget),
                    "window_label": window.label,
                    "metrics_path": str(metrics_path.resolve()),
                    "replay_metrics_path": str(replay_metrics_path.resolve()),
                    "selected_execution_profile": str(metrics.get("execution_alignment_profile", "")),
                    "raw_excess_annual_return": float(raw_holdout.get("excess_annual_return", 0.0)),
                    "raw_excess_sharpe": float(raw_holdout.get("excess_sharpe", 0.0)),
                    "aligned_excess_annual_return": float(aligned_holdout.get("excess_annual_return", 0.0)),
                    "aligned_excess_sharpe": float(aligned_holdout.get("excess_sharpe", 0.0)),
                    "aligned_avg_turnover": float(aligned_holdout.get("avg_turnover", 0.0)),
                    "replay_excess_annual_return": float(replay_metrics.get("excess_annual_return", 0.0)),
                    "replay_excess_sharpe": float(replay_metrics.get("excess_sharpe", 0.0)),
                    "replay_excess_max_drawdown": float(replay_metrics.get("excess_max_drawdown", 0.0)),
                    "replay_avg_turnover": float(replay_metrics.get("avg_turnover", 0.0)),
                    "epochs_requested": int(training.get("epochs_requested", 0) or 0),
                    "epochs_completed": epochs_completed,
                    "best_epoch": best_epoch,
                    "selected_epoch": selected_epoch,
                    "best_valid_loss": float(training.get("best_valid_loss", float("nan"))),
                    "final_valid_loss": float(training.get("final_valid_loss", float("nan"))),
                    "training_status": str(training.get("status", "")),
                    "still_improving": bool(training.get("still_improving", False)),
                    "selected_metric_name": str(training.get("selected_metric_name", "")),
                    "selected_metric_value": float(training.get("selected_metric_value", float("nan"))),
                    "selected_epoch_hits_cap": bool(selected_epoch == epochs_completed and epochs_completed > 0),
                    "best_epoch_hits_cap": bool(best_epoch == epochs_completed and epochs_completed > 0),
                    "train_mean_window_excess_annual_return": float(
                        selected_train.get("mean_window_excess_annual_return", float("nan"))
                    ),
                    "train_weak_window_excess_annual_return": float(
                        selected_train.get("weak_window_excess_annual_return", float("nan"))
                    ),
                    "train_worst_window_excess_sharpe": float(
                        selected_train.get("worst_window_excess_sharpe", float("nan"))
                    ),
                }
            )

    detail_df = pd.DataFrame(collected_rows).sort_values(["epoch_budget", "window_label"]).reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    for epoch_budget in epoch_budgets:
        frame = detail_df[detail_df["epoch_budget"] == epoch_budget]
        summary_rows.append(
            {
                "epoch_budget": int(epoch_budget),
                "window_count": int(len(frame)),
                "selected_execution_profiles": ",".join(sorted(set(str(x) for x in frame["selected_execution_profile"]))),
                "stable_count": int(frame["training_status"].eq("stable").sum()),
                "undertrained_count": int(frame["training_status"].eq("undertrained").sum()),
                "plateaued_count": int(frame["training_status"].eq("plateaued").sum()),
                "selected_epoch_hits_cap_count": int(frame["selected_epoch_hits_cap"].sum()),
                "best_epoch_hits_cap_count": int(frame["best_epoch_hits_cap"].sum()),
                "mean_selected_epoch": float(frame["selected_epoch"].mean()),
                "mean_best_epoch": float(frame["best_epoch"].mean()),
                "mean_raw_excess_annual_return": float(frame["raw_excess_annual_return"].mean()),
                "mean_raw_excess_sharpe": float(frame["raw_excess_sharpe"].mean()),
                "mean_aligned_excess_annual_return": float(frame["aligned_excess_annual_return"].mean()),
                "mean_aligned_excess_sharpe": float(frame["aligned_excess_sharpe"].mean()),
                "mean_replay_excess_annual_return": float(frame["replay_excess_annual_return"].mean()),
                "mean_replay_excess_sharpe": float(frame["replay_excess_sharpe"].mean()),
                "mean_replay_excess_max_drawdown": float(frame["replay_excess_max_drawdown"].mean()),
                "mean_replay_avg_turnover": float(frame["replay_avg_turnover"].mean()),
                "mean_selected_metric_value": float(frame["selected_metric_value"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_replay_excess_sharpe", "mean_replay_excess_annual_return"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_budget = int(summary_df.iloc[0]["epoch_budget"])
    current_budget = 8 if 8 in epoch_budgets else int(epoch_budgets[0])
    current_row = summary_df[summary_df["epoch_budget"] == current_budget].iloc[0]
    best_row = summary_df.iloc[0]
    weakest_replay_row = detail_df.sort_values("replay_excess_annual_return").iloc[0]

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_dir / "window_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "epoch_budget_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "replay_runs.json").write_text(json.dumps(replay_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Finetune Epoch Budget Formal Sensitivity",
        "",
        "## Protocol",
        f"- profile: `{profile.name}`",
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- window_count: `3`",
        f"- epoch_budgets: `{', '.join(str(x) for x in epoch_budgets)}`",
        "- model protocol: `top_bottom_bce + manual score head + next_open`",
        f"- research objective: `{args.research_objective_mode}`",
        f"- checkpoint selection: `{args.checkpoint_selection_objective}` (min improvement `{float(args.checkpoint_selection_min_improvement):.4f}`)",
        f"- execution alignment: `train_eval_auto / {args.execution_alignment_objective}`",
        f"- realistic cost: transaction `{float(args.transaction_cost_bps):.1f}` bps, slippage `{float(args.slippage_bps):.1f}` bps, sell-tax `{float(args.sell_tax_bps):.1f}` bps",
        "",
        "## Mean Summary",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "- "
            f"`{int(row['epoch_budget'])}` epochs: "
            f"mean replay excess annual {_format_pct(row['mean_replay_excess_annual_return'])}, "
            f"mean replay excess Sharpe {_format_float(row['mean_replay_excess_sharpe'])}, "
            f"mean aligned excess annual {_format_pct(row['mean_aligned_excess_annual_return'])}, "
            f"undertrained windows {int(row['undertrained_count'])}/3, "
            f"selected-at-cap {int(row['selected_epoch_hits_cap_count'])}/3, "
            f"selected profiles `{row['selected_execution_profiles']}`"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "- "
            f"best replay budget: `{best_budget}` epochs "
            f"(mean replay excess annual {_format_pct(best_row['mean_replay_excess_annual_return'])}, "
            f"mean replay excess Sharpe {_format_float(best_row['mean_replay_excess_sharpe'])})",
            "- "
            f"current reference budget: `{current_budget}` epochs "
            f"(mean replay excess annual {_format_pct(current_row['mean_replay_excess_annual_return'])}, "
            f"mean replay excess Sharpe {_format_float(current_row['mean_replay_excess_sharpe'])})",
            "- "
            f"weakest replay window overall: budget `{int(weakest_replay_row['epoch_budget'])}` / "
            f"`{weakest_replay_row['window_label']}` "
            f"(excess annual {_format_pct(weakest_replay_row['replay_excess_annual_return'])}, "
            f"excess Sharpe {_format_float(weakest_replay_row['replay_excess_sharpe'])})",
            "",
            "## Artifacts",
            f"- summary_csv: `daily_research/output/{root_tag}/epoch_budget_summary.csv`",
            f"- detail_csv: `daily_research/output/{root_tag}/window_detail.csv`",
            f"- source_runs: `daily_research/output/{root_tag}/source_runs.json`",
            f"- replay_runs: `daily_research/output/{root_tag}/replay_runs.json`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
