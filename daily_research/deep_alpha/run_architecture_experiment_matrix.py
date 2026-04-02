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

from daily_research.deep_alpha.architecture_profiles import (
    DEFAULT_ARCHITECTURE_PROFILE,
    get_profile,
    list_profile_lines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"


@dataclass(frozen=True)
class RecentFormalWindow:
    end_date: str = "20260401"
    train_end: str = "2025-03-17"
    valid_start: str = "2025-03-18"
    valid_days: int = 252


WINDOW = RecentFormalWindow()
DEFAULT_PROFILES: tuple[str, ...] = (
    "baseline_current",
    "capacity_small_h64",
    "capacity_large_h160",
    "depth_shallow_l1",
    "depth_deep_l4",
    "encoder_transformer_v1",
    "encoder_mamba_v1",
    "graph_off_plain",
    "graph_relation_only",
    "graph_topk4",
    "state_context_only",
    "state_liquidity_context",
    "structure_context_only",
    "structure_aux_task_v1",
    "structure_prototype_task_v1",
)

REUSE_METRICS: dict[str, str] = {
    "baseline_current": "daily_research/output/deep_alpha_short_alpha_matrix_20260402_r1/runs/baseline_current/metrics.json",
    "state_context_only": "daily_research/output/deep_alpha_short_alpha_matrix_20260402_r1/runs/state_context_v1/metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recent-formal deep_alpha architecture matrix for complexity, depth, and structure effects.")
    parser.add_argument("--root-tag", default="deep_alpha_architecture_matrix_20260402_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated architecture profile names from architecture_profiles.py",
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
        WINDOW.end_date,
        "--benchmark",
        "000300.SH",
        "--liquidity-pool",
        "liquid500",
        "--train-end-date",
        WINDOW.train_end,
        "--valid-start-date",
        WINDOW.valid_start,
        "--valid-days",
        str(WINDOW.valid_days),
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


def _resolve_metrics_path(root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / profile_name / "metrics.json"


def _resolve_model_path(metrics_path: Path) -> Path:
    return metrics_path.parent / "deep_alpha_model.pt"


def _load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"metrics.json is not a JSON object: {path}")
    return payload


def _load_model_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"Unexpected model artifact payload: {path}")
    return artifact


def _extract_artifact_summary(model_path: Path) -> dict[str, Any]:
    artifact = _load_model_artifact(model_path)
    config = artifact.get("config") if isinstance(artifact.get("config"), dict) else {}
    state_dict = artifact.get("model_state_dict") if isinstance(artifact.get("model_state_dict"), dict) else {}
    parameter_count = int(sum(int(tensor.numel()) for tensor in state_dict.values()))
    return {
        "config": config,
        "parameter_count": parameter_count,
        "model_artifact_mb": float(model_path.stat().st_size / (1024.0 * 1024.0)),
    }


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _format_signed_pct(value: float) -> str:
    return f"{value:+.2%}"


def _format_signed_float(value: float) -> str:
    return f"{value:+.3f}"


def _category_block(summary_df: pd.DataFrame, category: str) -> list[str]:
    block = summary_df[summary_df["category"] == category].sort_values(
        ["excess_annual_return", "excess_sharpe"],
        ascending=[False, False],
    )
    if block.empty:
        return []
    lines = [f"### {category.title()}"]
    for _, row in block.iterrows():
        lines.append(
            f"- `{row['profile_name']}`: excess annual {_format_pct(float(row['excess_annual_return']))}, "
            f"excess Sharpe {_format_float(float(row['excess_sharpe']))}, "
            f"params `{int(row['parameter_count'])}`, "
            f"delta vs baseline excess annual {_format_signed_pct(float(row['delta_excess_annual_return']))}"
        )
    return lines


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        print("\n".join(list_profile_lines()))
        return

    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No architecture profiles were provided.")

    rows: list[dict[str, Any]] = []
    source_runs: dict[str, str] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        reuse_path_str = REUSE_METRICS.get(profile.name, "")
        if reuse_path_str and not args.force_rerun:
            metrics_path = (PROJECT_ROOT / reuse_path_str).resolve()
            if not metrics_path.exists():
                raise FileNotFoundError(f"Expected reuse metrics not found: {metrics_path}")
            reused_existing = True
        else:
            metrics_path = _resolve_metrics_path(root_tag, profile.name)
            reused_existing = False
            if not metrics_path.exists() or args.force_rerun:
                experiment_tag = f"{root_tag}/runs/{profile.name}"
                command = _build_command(
                    python_executable=args.python_executable,
                    experiment_tag=experiment_tag,
                    profile_name=profile.name,
                )
                _run_command(command)

        metrics = _load_metrics(metrics_path)
        model_path = _resolve_model_path(metrics_path)
        artifact_info = _extract_artifact_summary(model_path)
        config = artifact_info["config"] if isinstance(artifact_info.get("config"), dict) else {}
        holdout = dict(metrics.get("holdout_backtest", {}))
        source_runs[profile.name] = str(metrics_path.resolve())
        rows.append(
            {
                "profile_name": profile.name,
                "category": profile.category,
                "description": profile.description,
                "metrics_path": str(metrics_path.resolve()),
                "model_path": str(model_path.resolve()),
                "reused_existing": reused_existing,
                "parameter_count": int(artifact_info.get("parameter_count", 0)),
                "model_artifact_mb": float(artifact_info.get("model_artifact_mb", 0.0)),
                "feature_count": int(metrics.get("feature_count", 0)),
                "hidden_dim": int(config.get("hidden_dim", profile.hidden_dim)),
                "encoder_family": str(metrics.get("encoder_family", config.get("encoder_family", profile.encoder_family))),
                "transformer_heads": int(config.get("transformer_heads", profile.transformer_heads)),
                "transformer_layers": int(config.get("transformer_layers", profile.transformer_layers)),
                "dynamic_graph_layer": bool(metrics.get("dynamic_graph_layer", profile.dynamic_graph_layer)),
                "dynamic_graph_top_k": int(metrics.get("dynamic_graph_top_k", profile.dynamic_graph_top_k)),
                "dynamic_graph_industry_boost": float(metrics.get("dynamic_graph_industry_boost", profile.dynamic_graph_industry_boost)),
                "dynamic_graph_style_boost": float(metrics.get("dynamic_graph_style_boost", profile.dynamic_graph_style_boost)),
                "relation_layer": bool(metrics.get("relation_layer", profile.relation_layer)),
                "state_context": bool(metrics.get("state_context", profile.state_context)),
                "liquidity_context": bool(metrics.get("liquidity_context", profile.liquidity_context)),
                "structure_context": bool(metrics.get("structure_context", profile.structure_context)),
                "aux_structure_task": bool(metrics.get("aux_structure_task", profile.aux_structure_task)),
                "structure_prototype_task": bool(metrics.get("structure_prototype_task", profile.structure_prototype_task)),
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
    baseline_row = summary_df.loc[summary_df["profile_name"] == DEFAULT_ARCHITECTURE_PROFILE].iloc[0]
    summary_df["delta_excess_annual_return"] = summary_df["excess_annual_return"] - float(baseline_row["excess_annual_return"])
    summary_df["delta_excess_sharpe"] = summary_df["excess_sharpe"] - float(baseline_row["excess_sharpe"])
    summary_df["delta_turnover"] = summary_df["avg_turnover"] - float(baseline_row["avg_turnover"])
    summary_df["delta_parameter_count"] = summary_df["parameter_count"] - int(baseline_row["parameter_count"])

    category_summary_df = (
        summary_df.sort_values(["category", "excess_annual_return", "excess_sharpe"], ascending=[True, False, False])
        .groupby("category", as_index=False)
        .first()
        .sort_values(["category"])
        .reset_index(drop=True)
    )

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    category_summary_df.to_csv(output_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(
        json.dumps(source_runs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    winner = summary_df.iloc[0]
    complexity_block = _category_block(summary_df, "complexity")
    depth_block = _category_block(summary_df, "depth")
    encoder_block = _category_block(summary_df, "encoder")
    graph_block = _category_block(summary_df, "graph")
    context_block = _category_block(summary_df, "context")
    structure_block = _category_block(summary_df, "structure")

    baseline_lookup = summary_df.set_index("profile_name").to_dict(orient="index")
    lines = [
        "# Deep Alpha Architecture Experiment Matrix",
        "",
        "## Protocol",
        f"- window: `{WINDOW.valid_start} -> 2026-03-31`",
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- invariant settings: `top_bottom_bce + manual score head + no execution alignment + next_open`",
        "- purpose: isolate model complexity / depth / structure changes under the current recent-formal protocol",
        "",
        "## Winner",
        f"- best_excess_annual_return: `{winner['profile_name']}` = {_format_pct(float(winner['excess_annual_return']))}",
        f"- excess_sharpe: `{_format_float(float(winner['excess_sharpe']))}`",
        f"- parameter_count: `{int(winner['parameter_count'])}`",
        f"- avg_turnover: `{_format_float(float(winner['avg_turnover']))}`",
        "",
        "## Key Reads",
        (
            f"- complexity: `capacity_small_h64` {_format_pct(float(baseline_lookup['capacity_small_h64']['excess_annual_return']))}, "
            f"`baseline_current` {_format_pct(float(baseline_lookup['baseline_current']['excess_annual_return']))}, "
            f"`capacity_large_h160` {_format_pct(float(baseline_lookup['capacity_large_h160']['excess_annual_return']))}"
        ),
        (
            f"- depth: `depth_shallow_l1` {_format_pct(float(baseline_lookup['depth_shallow_l1']['excess_annual_return']))}, "
            f"`baseline_current` {_format_pct(float(baseline_lookup['baseline_current']['excess_annual_return']))}, "
            f"`depth_deep_l4` {_format_pct(float(baseline_lookup['depth_deep_l4']['excess_annual_return']))}"
        ),
        (
            f"- encoder family: `patch_transformer` baseline {_format_pct(float(baseline_lookup['baseline_current']['excess_annual_return']))}, "
            f"`transformer` {_format_pct(float(baseline_lookup['encoder_transformer_v1']['excess_annual_return']))}, "
            f"`mamba` {_format_pct(float(baseline_lookup['encoder_mamba_v1']['excess_annual_return']))}"
        ),
        (
            f"- graph structure: `graph_off_plain` {_format_pct(float(baseline_lookup['graph_off_plain']['excess_annual_return']))}, "
            f"`graph_relation_only` {_format_pct(float(baseline_lookup['graph_relation_only']['excess_annual_return']))}, "
            f"`graph_topk4` {_format_pct(float(baseline_lookup['graph_topk4']['excess_annual_return']))}, "
            f"`baseline_current` {_format_pct(float(baseline_lookup['baseline_current']['excess_annual_return']))}"
        ),
        (
            f"- context: `state_context_only` {_format_pct(float(baseline_lookup['state_context_only']['excess_annual_return']))}, "
            f"`state_liquidity_context` {_format_pct(float(baseline_lookup['state_liquidity_context']['excess_annual_return']))}"
        ),
        (
            f"- structure regularization: `structure_context_only` {_format_pct(float(baseline_lookup['structure_context_only']['excess_annual_return']))}, "
            f"`structure_aux_task_v1` {_format_pct(float(baseline_lookup['structure_aux_task_v1']['excess_annual_return']))}, "
            f"`structure_prototype_task_v1` {_format_pct(float(baseline_lookup['structure_prototype_task_v1']['excess_annual_return']))}"
        ),
        "",
        "## Category Ranking",
    ]
    for block in (complexity_block, depth_block, encoder_block, graph_block, context_block, structure_block):
        lines.extend(block)
        if block:
            lines.append("")
    lines.append("## Full Ranking")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['profile_name']}` [{row['category']}]: excess annual {_format_pct(float(row['excess_annual_return']))}, "
            f"excess Sharpe {_format_float(float(row['excess_sharpe']))}, "
            f"drawdown {_format_pct(float(row['excess_max_drawdown']))}, "
            f"turnover {_format_float(float(row['avg_turnover']))}, "
            f"params `{int(row['parameter_count'])}`, "
            f"delta vs baseline excess annual {_format_signed_pct(float(row['delta_excess_annual_return']))}, "
            f"delta Sharpe {_format_signed_float(float(row['delta_excess_sharpe']))}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_dir={output_dir}")


if __name__ == "__main__":
    main()
