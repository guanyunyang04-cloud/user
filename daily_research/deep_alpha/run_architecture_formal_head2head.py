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

from daily_research.deep_alpha.architecture_profiles import get_profile
from daily_research.deep_alpha.research_objective import resolve_primary_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"


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

DEFAULT_PROFILES: tuple[str, ...] = (
    "baseline_current",
    "capacity_large_h160",
    "depth_shallow_l1",
    "depth_deep_l4",
    "graph_off_plain",
    "structure_context_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal multi-window head-to-head for key deep_alpha architecture variants.")
    parser.add_argument("--root-tag", default="deep_alpha_architecture_formal_head2head_20260403_monthly_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated architecture profile names from architecture_profiles.py",
    )
    parser.add_argument(
        "--recent-root-tag",
        default="deep_alpha_architecture_matrix_20260403_monthly_r1",
        help="Recent architecture matrix root used to reuse the latest-window metrics when available.",
    )
    parser.add_argument("--force-rerun", action="store_true")
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


def _resolve_metrics_path(root_tag: str, profile_name: str, window: FormalWindow) -> Path:
    return OUTPUT_ROOT / root_tag / "runs" / f"{profile_name}_{window.label}" / "metrics.json"


def _resolve_recent_metrics_path(recent_root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / recent_root_tag / "runs" / profile_name / "metrics.json"


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
    root_tag = str(args.root_tag).strip()
    profile_names = [item.strip() for item in str(args.profiles).split(",") if item.strip()]
    if not profile_names:
        raise ValueError("No architecture profiles were provided.")
    if "baseline_current" not in profile_names:
        raise ValueError("The formal head-to-head requires baseline_current to be included in --profiles.")

    collected_rows: list[dict[str, Any]] = []
    source_runs: dict[str, dict[str, str]] = {}

    for profile_name in profile_names:
        profile = get_profile(profile_name)
        source_runs[profile.name] = {}
        for window in WINDOWS:
            reused_existing = False
            if window.label == "20250318_20260331" and not args.force_rerun:
                recent_metrics_path = _resolve_recent_metrics_path(args.recent_root_tag, profile.name)
                if recent_metrics_path.exists():
                    metrics_path = recent_metrics_path
                    reused_existing = True
                else:
                    metrics_path = _resolve_metrics_path(root_tag, profile.name, window)
            else:
                metrics_path = _resolve_metrics_path(root_tag, profile.name, window)

            if (not metrics_path.exists()) or args.force_rerun:
                experiment_tag = f"{root_tag}/runs/{profile.name}_{window.label}"
                command = _build_command(
                    python_executable=args.python_executable,
                    experiment_tag=experiment_tag,
                    profile_name=profile.name,
                    window=window,
                )
                _run_command(command)
                metrics_path = _resolve_metrics_path(root_tag, profile.name, window)
                reused_existing = False

            metrics = _load_metrics(metrics_path)
            artifact_info = _extract_artifact_summary(_resolve_model_path(metrics_path))
            _, holdout = resolve_primary_backtest(metrics)
            holdout = dict(holdout)
            source_runs[profile.name][window.label] = str(metrics_path.resolve())
            collected_rows.append(
                {
                    "profile_name": profile.name,
                    "category": profile.category,
                    "window_label": window.label,
                    "metrics_path": str(metrics_path.resolve()),
                    "reused_existing": reused_existing,
                    "parameter_count": int(artifact_info.get("parameter_count", 0)),
                    "hidden_dim": int(artifact_info.get("config", {}).get("hidden_dim", profile.hidden_dim)),
                    "encoder_family": str(metrics.get("encoder_family", profile.encoder_family)),
                    "transformer_layers": int(artifact_info.get("config", {}).get("transformer_layers", profile.transformer_layers)),
                    "dynamic_graph_layer": bool(metrics.get("dynamic_graph_layer", profile.dynamic_graph_layer)),
                    "relation_layer": bool(metrics.get("relation_layer", profile.relation_layer)),
                    "structure_context": bool(metrics.get("structure_context", profile.structure_context)),
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
    baseline_lookup = (
        detail_df[detail_df["profile_name"] == "baseline_current"]
        .set_index("window_label")[["excess_annual_return", "excess_sharpe", "avg_turnover", "excess_max_drawdown"]]
        .to_dict(orient="index")
    )

    compare_rows: list[dict[str, Any]] = []
    for _, row in detail_df[detail_df["profile_name"] != "baseline_current"].iterrows():
        base = baseline_lookup[str(row["window_label"])]
        compare_rows.append(
            {
                "profile_name": row["profile_name"],
                "category": row["category"],
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

    compare_df = pd.DataFrame(compare_rows).sort_values(["profile_name", "window_label"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    for profile_name in profile_names:
        frame = detail_df[detail_df["profile_name"] == profile_name]
        compare_frame = compare_df[compare_df["profile_name"] == profile_name]
        summary_rows.append(
            {
                "profile_name": profile_name,
                "category": str(frame["category"].iloc[0]),
                "window_count": int(len(frame)),
                "mean_excess_total_return": float(frame["excess_total_return"].mean()),
                "mean_excess_annual_return": float(frame["excess_annual_return"].mean()),
                "mean_excess_sharpe": float(frame["excess_sharpe"].mean()),
                "mean_excess_max_drawdown": float(frame["excess_max_drawdown"].mean()),
                "mean_avg_turnover": float(frame["avg_turnover"].mean()),
                "wins_by_excess_annual_return": int(compare_frame["wins_excess_annual_return"].sum()) if not compare_frame.empty else 0,
                "wins_by_excess_sharpe": int(compare_frame["wins_excess_sharpe"].sum()) if not compare_frame.empty else 0,
                "mean_delta_excess_annual_return_vs_baseline": float(compare_frame["delta_excess_annual_return"].mean()) if not compare_frame.empty else 0.0,
                "mean_delta_excess_sharpe_vs_baseline": float(compare_frame["delta_excess_sharpe"].mean()) if not compare_frame.empty else 0.0,
                "reused_window_count": int(frame["reused_existing"].sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_excess_annual_return", "mean_excess_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_dir / "window_detail.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "profile_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "source_runs.json").write_text(json.dumps(source_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_summary = summary_df.loc[summary_df["profile_name"] == "baseline_current"].iloc[0]
    lines = [
        "# Deep Alpha Architecture Formal Head-to-Head",
        "",
        "## Protocol",
        "- window_count: `3`",
        f"- research_time_unit: `{RESEARCH_TIME_UNIT}` with `valid_months={WINDOWS[0].valid_months}` and `train_eval_window_months={TRAIN_EVAL_WINDOW_MONTHS}`",
        "- benchmark: `000300.SH`",
        "- universe: `liquid500`",
        "- invariant settings: `execution_first + train_eval_auto execution alignment + top_bottom_bce + manual score head + next_open`",
        "- baseline: `baseline_current`",
        "",
        "## Baseline Mean",
        f"- mean_excess_annual_return: `{_format_pct(float(baseline_summary['mean_excess_annual_return']))}`",
        f"- mean_excess_sharpe: `{_format_float(float(baseline_summary['mean_excess_sharpe']))}`",
        "",
        "## Candidate Mean Summary",
    ]
    for _, row in summary_df[summary_df["profile_name"] != "baseline_current"].iterrows():
        lines.append(
            f"- `{row['profile_name']}` [{row['category']}]: mean excess annual {_format_pct(float(row['mean_excess_annual_return']))}, "
            f"mean excess Sharpe {_format_float(float(row['mean_excess_sharpe']))}, "
            f"wins annual `{int(row['wins_by_excess_annual_return'])}/{len(WINDOWS)}`, "
            f"wins Sharpe `{int(row['wins_by_excess_sharpe'])}/{len(WINDOWS)}`, "
            f"mean delta annual vs baseline {_format_pct(float(row['mean_delta_excess_annual_return_vs_baseline']))}"
        )
    lines.extend(["", "## Window Head-to-Head"])
    for profile_name in [item for item in profile_names if item != "baseline_current"]:
        lines.append(f"### {profile_name}")
        profile_frame = compare_df[compare_df["profile_name"] == profile_name]
        for _, row in profile_frame.iterrows():
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
