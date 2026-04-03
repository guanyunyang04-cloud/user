from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.architecture_profiles import get_profile, list_profile_lines
from daily_research.deep_alpha.research_objective import (
    DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    DEFAULT_RESEARCH_OBJECTIVE_MODE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"
RECENT_H2H_SCRIPT = PROJECT_ROOT / "daily_research" / "tools" / "execution_candidate_multiwindow_h2h.py"

CURRENT_DEFAULT_REALISTIC_RUN = (
    PROJECT_ROOT / "daily_research" / "output" / "execution_costreview_regoff_k2_realistic_20260401_r1"
)

EXECUTION_ALIGNMENT_CANDIDATES: tuple[str, ...] = (
    "raw_1d",
    "topk2_1d_regoff",
    "regoff_k2_10d_ensemble_native_anchor",
    "regon_k1_10d_ensemble_native_anchor",
)


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

DEFAULT_PROFILES: tuple[str, ...] = (
    "baseline_current",
    "structure_context_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run architecture execution-objective formal head-to-head with realistic external replay."
    )
    parser.add_argument("--root-tag", default="deep_alpha_architecture_execalign_formal_20260403_r2")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated architecture profile names from architecture_profiles.py",
    )
    parser.add_argument("--list-profiles", action="store_true")
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
        choices=["valid_loss", "primary_annual_return", "primary_excess_annual_return", "primary_excess_sharpe"],
        default=DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    )
    parser.add_argument("--checkpoint-selection-min-improvement", type=float, default=0.0001)
    parser.add_argument("--execution-alignment-objective", default="robust_composite")
    return parser.parse_args()


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_research_command(
    *,
    python_executable: str,
    experiment_tag: str,
    profile_name: str,
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
        "252",
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


def _build_recent_h2h_command(
    *,
    python_executable: str,
    run_a: Path,
    label_a: str,
    run_b: Path,
    label_b: str,
    output_dir: Path,
) -> list[str]:
    return [
        python_executable,
        str(RECENT_H2H_SCRIPT),
        "--run-a",
        str(run_a),
        "--label-a",
        label_a,
        "--run-b",
        str(run_b),
        "--label-b",
        label_b,
        "--bridge-start",
        "2025-03-18",
        "--weak-start",
        "2025-09-05",
        "--output-dir",
        str(output_dir),
    ]


def _resolve_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / f"{profile_name}_{window.label}" / "metrics.json"


def _resolve_replay_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "replays" / f"{profile_name}_{window.label}" / "metrics.json"


def _resolve_model_path(metrics_path: Path) -> Path:
    return metrics_path.parent / "deep_alpha_model.pt"


def _load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"metrics.json is not a JSON object: {path}")
    return payload


def _extract_artifact_summary(model_path: Path) -> dict[str, Any]:
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"Unexpected model artifact payload: {model_path}")
    config = artifact.get("config") if isinstance(artifact.get("config"), dict) else {}
    state_dict = artifact.get("model_state_dict") if isinstance(artifact.get("model_state_dict"), dict) else {}
    parameter_count = int(sum(int(tensor.numel()) for tensor in state_dict.values()))
    return {
        "config": config,
        "parameter_count": parameter_count,
    }


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
        raise ValueError("No architecture profiles were provided.")
    if "baseline_current" not in profile_names:
        raise ValueError("The execution-objective head-to-head requires baseline_current to be included.")
    if "structure_context_only" not in profile_names:
        raise ValueError("The execution-objective head-to-head requires structure_context_only to be included.")

    collected_rows: list[dict[str, Any]] = []
    source_runs: dict[str, dict[str, str]] = {}
    replay_runs: dict[str, dict[str, str]] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        source_runs[profile.name] = {}
        replay_runs[profile.name] = {}
        for window in WINDOWS:
            metrics_path = _resolve_metrics_path(root_tag, profile.name, window)
            if not metrics_path.exists() or args.force_rerun:
                experiment_tag = f"{root_tag}/runs/{profile.name}_{window.label}"
                command = _build_research_command(
                    python_executable=args.python_executable,
                    experiment_tag=experiment_tag,
                    profile_name=profile.name,
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
            model_summary = _extract_artifact_summary(_resolve_model_path(metrics_path))
            run_dir = metrics_path.parent
            replay_metrics_path = _resolve_replay_metrics_path(root_tag, profile.name, window)
            if not replay_metrics_path.exists() or args.force_rerun:
                replay_command = _build_external_replay_command(
                    python_executable=args.python_executable,
                    output_root=OUTPUT_ROOT / root_tag,
                    replay_tag=f"replays/{profile.name}_{window.label}",
                    candidate_label=f"{profile.name}_execalign_realistic_{window.label}",
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
            selected_train = dict(metrics.get("execution_alignment_selected_train_metrics", {}))

            source_runs[profile.name][window.label] = str(metrics_path.resolve())
            replay_runs[profile.name][window.label] = str(replay_metrics_path.resolve())
            collected_rows.append(
                {
                    "profile_name": profile.name,
                    "category": profile.category,
                    "window_label": window.label,
                    "metrics_path": str(metrics_path.resolve()),
                    "replay_metrics_path": str(replay_metrics_path.resolve()),
                    "parameter_count": int(model_summary["parameter_count"]),
                    "feature_count": int(metrics.get("feature_count", 0)),
                    "selected_execution_profile": str(metrics.get("execution_alignment_profile", "")),
                    "selected_execution_profile_description": str(metrics.get("execution_alignment_profile_description", "")),
                    "raw_excess_annual_return": float(raw_holdout.get("excess_annual_return", 0.0)),
                    "raw_excess_sharpe": float(raw_holdout.get("excess_sharpe", 0.0)),
                    "raw_excess_max_drawdown": float(raw_holdout.get("excess_max_drawdown", 0.0)),
                    "raw_avg_turnover": float(raw_holdout.get("avg_turnover", 0.0)),
                    "aligned_excess_annual_return": float(aligned_holdout.get("excess_annual_return", 0.0)),
                    "aligned_excess_sharpe": float(aligned_holdout.get("excess_sharpe", 0.0)),
                    "aligned_excess_max_drawdown": float(aligned_holdout.get("excess_max_drawdown", 0.0)),
                    "aligned_avg_turnover": float(aligned_holdout.get("avg_turnover", 0.0)),
                    "replay_excess_annual_return": float(replay_metrics.get("excess_annual_return", 0.0)),
                    "replay_excess_sharpe": float(replay_metrics.get("excess_sharpe", 0.0)),
                    "replay_excess_max_drawdown": float(replay_metrics.get("excess_max_drawdown", 0.0)),
                    "replay_avg_turnover": float(replay_metrics.get("avg_turnover", 0.0)),
                    "replay_annual_return": float(replay_metrics.get("annual_return", 0.0)),
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

    detail_df = pd.DataFrame(collected_rows).sort_values(["profile_name", "window_label"]).reset_index(drop=True)
    baseline_name = "baseline_current"
    baseline_lookup = (
        detail_df[detail_df["profile_name"] == baseline_name]
        .set_index("window_label")[
            [
                "raw_excess_annual_return",
                "raw_excess_sharpe",
                "aligned_excess_annual_return",
                "aligned_excess_sharpe",
                "replay_excess_annual_return",
                "replay_excess_sharpe",
            ]
        ]
        .to_dict(orient="index")
    )

    comparison_rows: list[dict[str, Any]] = []
    for _, row in detail_df[detail_df["profile_name"] != baseline_name].iterrows():
        base = baseline_lookup[str(row["window_label"])]
        comparison_rows.append(
            {
                "profile_name": row["profile_name"],
                "window_label": row["window_label"],
                "selected_execution_profile": row["selected_execution_profile"],
                "delta_raw_excess_annual_return": float(row["raw_excess_annual_return"] - base["raw_excess_annual_return"]),
                "delta_raw_excess_sharpe": float(row["raw_excess_sharpe"] - base["raw_excess_sharpe"]),
                "delta_aligned_excess_annual_return": float(
                    row["aligned_excess_annual_return"] - base["aligned_excess_annual_return"]
                ),
                "delta_aligned_excess_sharpe": float(row["aligned_excess_sharpe"] - base["aligned_excess_sharpe"]),
                "delta_replay_excess_annual_return": float(
                    row["replay_excess_annual_return"] - base["replay_excess_annual_return"]
                ),
                "delta_replay_excess_sharpe": float(row["replay_excess_sharpe"] - base["replay_excess_sharpe"]),
                "wins_aligned_excess_annual_return": bool(
                    row["aligned_excess_annual_return"] > base["aligned_excess_annual_return"]
                ),
                "wins_aligned_excess_sharpe": bool(row["aligned_excess_sharpe"] > base["aligned_excess_sharpe"]),
                "wins_replay_excess_annual_return": bool(
                    row["replay_excess_annual_return"] > base["replay_excess_annual_return"]
                ),
                "wins_replay_excess_sharpe": bool(row["replay_excess_sharpe"] > base["replay_excess_sharpe"]),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["profile_name", "window_label"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for profile_name in profile_names:
        frame = detail_df[detail_df["profile_name"] == profile_name]
        compare_frame = comparison_df[comparison_df["profile_name"] == profile_name]
        summary_rows.append(
            {
                "profile_name": profile_name,
                "window_count": int(len(frame)),
                "selected_execution_profiles": ",".join(sorted(set(str(x) for x in frame["selected_execution_profile"]))),
                "mean_raw_excess_annual_return": float(frame["raw_excess_annual_return"].mean()),
                "mean_raw_excess_sharpe": float(frame["raw_excess_sharpe"].mean()),
                "mean_aligned_excess_annual_return": float(frame["aligned_excess_annual_return"].mean()),
                "mean_aligned_excess_sharpe": float(frame["aligned_excess_sharpe"].mean()),
                "mean_replay_excess_annual_return": float(frame["replay_excess_annual_return"].mean()),
                "mean_replay_excess_sharpe": float(frame["replay_excess_sharpe"].mean()),
                "mean_replay_excess_max_drawdown": float(frame["replay_excess_max_drawdown"].mean()),
                "mean_replay_avg_turnover": float(frame["replay_avg_turnover"].mean()),
                "wins_aligned_excess_annual_return": int(compare_frame["wins_aligned_excess_annual_return"].sum())
                if not compare_frame.empty
                else 0,
                "wins_aligned_excess_sharpe": int(compare_frame["wins_aligned_excess_sharpe"].sum())
                if not compare_frame.empty
                else 0,
                "wins_replay_excess_annual_return": int(compare_frame["wins_replay_excess_annual_return"].sum())
                if not compare_frame.empty
                else 0,
                "wins_replay_excess_sharpe": int(compare_frame["wins_replay_excess_sharpe"].sum())
                if not compare_frame.empty
                else 0,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_replay_excess_sharpe", "mean_replay_excess_annual_return"], ascending=[False, False]
    ).reset_index(drop=True)

    recent_window_label = "20250318_20260331"
    recent_baseline_replay_run = (_resolve_replay_metrics_path(root_tag, baseline_name, WINDOWS[-1])).parent
    current_default_run = CURRENT_DEFAULT_REALISTIC_RUN.resolve()
    current_default_metrics = _load_metrics(current_default_run / "metrics.json")
    recent_structure_run = (_resolve_replay_metrics_path(root_tag, "structure_context_only", WINDOWS[-1])).parent
    current_h2h_dir = OUTPUT_ROOT / root_tag / "recent_h2h_structure_vs_current_default"
    baseline_h2h_dir = OUTPUT_ROOT / root_tag / "recent_h2h_structure_vs_baseline_execalign"
    baseline_vs_current_h2h_dir = OUTPUT_ROOT / root_tag / "recent_h2h_baseline_execalign_vs_current_default"
    _run_command(
        _build_recent_h2h_command(
            python_executable=args.python_executable,
            run_a=recent_structure_run,
            label_a="structure_execalign_realistic",
            run_b=current_default_run,
            label_b="regoff_k2_realistic",
            output_dir=current_h2h_dir,
        )
    )
    _run_command(
        _build_recent_h2h_command(
            python_executable=args.python_executable,
            run_a=recent_structure_run,
            label_a="structure_execalign_realistic",
            run_b=recent_baseline_replay_run,
            label_b="baseline_execalign_realistic",
            output_dir=baseline_h2h_dir,
        )
    )
    _run_command(
        _build_recent_h2h_command(
            python_executable=args.python_executable,
            run_a=recent_baseline_replay_run,
            label_a="baseline_execalign_realistic",
            run_b=current_default_run,
            label_b="regoff_k2_realistic",
            output_dir=baseline_vs_current_h2h_dir,
        )
    )

    weakest_row = comparison_df.loc[comparison_df["profile_name"] == "structure_context_only"].sort_values(
        "delta_replay_excess_annual_return"
    ).iloc[0]
    recent_structure_metrics = _load_metrics(recent_structure_run / "metrics.json")
    recent_baseline_metrics = _load_metrics(recent_baseline_replay_run / "metrics.json")

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_dir / "window_detail.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "replay_runs.json").write_text(json.dumps(replay_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Architecture Execution Objective Head-to-Head",
        "",
        "## Protocol",
        "- window_count: `3`",
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- model protocol: `top_bottom_bce + manual score head + next_open`",
        f"- research objective: `{args.research_objective_mode}`",
        f"- checkpoint selection: `{args.checkpoint_selection_objective}` (min improvement `{float(args.checkpoint_selection_min_improvement):.4f}`)",
        f"- execution alignment: `train_eval_auto / {args.execution_alignment_objective}`",
        f"- realistic cost: transaction `{float(args.transaction_cost_bps):.1f}` bps, slippage `{float(args.slippage_bps):.1f}` bps, sell-tax `{float(args.sell_tax_bps):.1f}` bps",
        f"- execution candidates scanned: `{', '.join(EXECUTION_ALIGNMENT_CANDIDATES)}`",
        "",
        "## Mean Summary",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "- "
            f"`{row['profile_name']}`: "
            f"mean aligned excess annual {_format_pct(row['mean_aligned_excess_annual_return'])}, "
            f"mean aligned excess Sharpe {_format_float(row['mean_aligned_excess_sharpe'])}, "
            f"mean replay excess annual {_format_pct(row['mean_replay_excess_annual_return'])}, "
            f"mean replay excess Sharpe {_format_float(row['mean_replay_excess_sharpe'])}, "
            f"selected profiles `{row['selected_execution_profiles']}`"
        )

    lines.extend(
        [
            "",
            "## Baseline vs Structure",
            "- "
            f"`structure_context_only` aligned wins: excess annual {int(summary_df.loc[summary_df['profile_name'] == 'structure_context_only', 'wins_aligned_excess_annual_return'].iloc[0])}/3, "
            f"excess Sharpe {int(summary_df.loc[summary_df['profile_name'] == 'structure_context_only', 'wins_aligned_excess_sharpe'].iloc[0])}/3",
            "- "
            f"`structure_context_only` external replay wins: excess annual {int(summary_df.loc[summary_df['profile_name'] == 'structure_context_only', 'wins_replay_excess_annual_return'].iloc[0])}/3, "
            f"excess Sharpe {int(summary_df.loc[summary_df['profile_name'] == 'structure_context_only', 'wins_replay_excess_sharpe'].iloc[0])}/3",
            "- "
            f"weakest replay window vs baseline: `{weakest_row['window_label']}` "
            f"(delta excess annual {_format_pct(weakest_row['delta_replay_excess_annual_return'])}, "
            f"delta excess Sharpe {_format_float(weakest_row['delta_replay_excess_sharpe'])})",
            "",
            "## Recent Upgrade Gate",
            "- "
            f"recent `structure_execalign_realistic`: annual {_format_pct(recent_structure_metrics['annual_return'])}, "
            f"excess annual {_format_pct(recent_structure_metrics['excess_annual_return'])}, "
            f"excess Sharpe {_format_float(recent_structure_metrics['excess_sharpe'])}, "
            f"max drawdown {_format_pct(recent_structure_metrics['max_drawdown'])}",
            "- "
            f"recent `baseline_execalign_realistic`: annual {_format_pct(recent_baseline_metrics['annual_return'])}, "
            f"excess annual {_format_pct(recent_baseline_metrics['excess_annual_return'])}, "
            f"excess Sharpe {_format_float(recent_baseline_metrics['excess_sharpe'])}, "
            f"max drawdown {_format_pct(recent_baseline_metrics['max_drawdown'])}",
            "- "
            f"current default `regoff_k2_realistic`: annual {_format_pct(current_default_metrics['annual_return'])}, "
            f"excess annual {_format_pct(current_default_metrics['excess_annual_return'])}, "
            f"excess Sharpe {_format_float(current_default_metrics['excess_sharpe'])}, "
            f"max drawdown {_format_pct(current_default_metrics['max_drawdown'])}",
            "",
            "## Weak-Window Review",
            "- recent named-window H2H vs current default:",
            f"  - `{current_h2h_dir.resolve() / 'summary.md'}`",
            "- recent named-window H2H vs baseline execalign:",
            f"  - `{baseline_h2h_dir.resolve() / 'summary.md'}`",
            "- recent named-window H2H: baseline execalign vs current default:",
            f"  - `{baseline_vs_current_h2h_dir.resolve() / 'summary.md'}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
