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

from daily_research.baseline.backtest import summarize_monthly_diagnostics
from daily_research.deep_alpha.architecture_profiles import get_profile, list_profile_lines
from daily_research.deep_alpha.execution_alignment import default_auto_profile_argument
from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.family_epoch_budget import (
    DEFAULT_LATEST_MANIFEST_PATH,
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


@dataclass(frozen=True)
class WindowSpec:
    label: str
    train_end: str
    valid_start: str
    valid_months: int = 12


RECENT_WINDOW = WindowSpec(
    label="20250318_20260331",
    train_end="2025-03-17",
    valid_start="2025-03-18",
)
FORMAL_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(label="20230216_20240229", train_end="2023-02-15", valid_start="2023-02-16"),
    WindowSpec(label="20240301_20250317", train_end="2024-02-29", valid_start="2024-03-01"),
    RECENT_WINDOW,
)
RESEARCH_TIME_UNIT = "calendar_months"
TRAIN_EVAL_WINDOW_MONTHS = 6
DEFAULT_RECENT_PROFILES: tuple[str, ...] = (
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
EXECUTION_ALIGNMENT_CANDIDATES: tuple[str, ...] = (default_auto_profile_argument(),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the deep_alpha architecture complexity/depth/structure formal experiments under the current protocol."
    )
    parser.add_argument("--root-tag", default="deep_alpha_architecture_protocol_refresh_20260406_r1")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument(
        "--recent-profiles",
        default=",".join(DEFAULT_RECENT_PROFILES),
        help="Comma-separated architecture profile names for the recent-window full matrix.",
    )
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--end-date", default="20260403")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--liquidity-pool", default="liquid500")
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
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--family-epoch-budget-manifest", default=str(DEFAULT_LATEST_MANIFEST_PATH))
    return parser.parse_args()


def _parse_name_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _family_key_for_profile(profile_name: str) -> str:
    profile = get_profile(profile_name)
    return "structure" if str(profile.category) == "structure" else "baseline"


def _epoch_budget_for_profile(profile_name: str, manifest_path: str) -> int:
    family_key = _family_key_for_profile(profile_name)
    fallback = 8 if family_key == "baseline" else 12
    return resolve_epoch_budget_for_family(
        family_key,
        manifest_path=manifest_path,
        fallback_epochs=fallback,
    )


def _resolve_recent_metrics_path(root_tag: str, profile_name: str) -> Path:
    preferred = OUTPUT_ROOT / f"{root_tag}__{profile_name}" / "metrics.json"
    legacy = OUTPUT_ROOT / root_tag / "metrics.json"
    if str(profile_name) == "baseline_current" and legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def _resolve_formal_metrics_path(root_tag: str, profile_name: str, window: WindowSpec) -> Path:
    run_key = f"{profile_name}_{window.label}"
    preferred = OUTPUT_ROOT / f"{root_tag}__{run_key}" / "metrics.json"
    legacy = OUTPUT_ROOT / root_tag / "metrics.json"
    if str(profile_name) == "baseline_current" and str(window.label) == RECENT_WINDOW.label and legacy.exists() and not preferred.exists():
        return legacy
    return preferred


def _build_research_command(
    *,
    python_executable: str,
    experiment_tag: str,
    profile_name: str,
    window: WindowSpec,
    end_date: str,
    benchmark: str,
    liquidity_pool: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
    family_epoch_budget_manifest: str,
) -> list[str]:
    profile = get_profile(profile_name)
    epoch_budget = _epoch_budget_for_profile(profile_name, family_epoch_budget_manifest)
    command = [
        python_executable,
        str(RUN_SCRIPT),
        "--data-source",
        "tq",
        "--start-date",
        "20210101",
        "--end-date",
        str(end_date),
        "--benchmark",
        str(benchmark),
        "--liquidity-pool",
        str(liquidity_pool),
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
        command.extend(
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
        command.append("--relation-layer")
    if profile.state_context:
        command.append("--state-context")
    if profile.liquidity_context:
        command.append("--liquidity-context")
    if profile.structure_context:
        command.append("--structure-context")
    if profile.aux_structure_task:
        command.extend(
            [
                "--aux-structure-task",
                "--aux-structure-loss-weight",
                str(profile.aux_structure_loss_weight),
                "--aux-structure-label-smoothing",
                str(profile.aux_structure_label_smoothing),
            ]
        )
    if profile.structure_prototype_task:
        command.extend(
            [
                "--structure-prototype-task",
                "--structure-prototype-loss-weight",
                str(profile.structure_prototype_loss_weight),
                "--structure-prototype-temperature",
                str(profile.structure_prototype_temperature),
            ]
        )
    return command


def _extract_artifact_summary(run_dir: Path) -> dict[str, Any]:
    model_path = run_dir / "deep_alpha_model.pt"
    if not model_path.exists():
        return {"parameter_count": float("nan"), "model_artifact_mb": float("nan")}
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        return {"parameter_count": float("nan"), "model_artifact_mb": float("nan")}
    state_dict = artifact.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return {"parameter_count": float("nan"), "model_artifact_mb": float("nan")}
    parameter_count = int(sum(int(tensor.numel()) for tensor in state_dict.values()))
    return {
        "parameter_count": parameter_count,
        "model_artifact_mb": float(model_path.stat().st_size / (1024.0 * 1024.0)),
    }


def _load_monthly_summary(run_dir: Path, metrics: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    monthly_candidates: list[tuple[str, str]] = []
    primary_label = str(metrics.get("primary_research_monthly_summary_label", "")).strip()
    if primary_label:
        monthly_candidates.append((f"{primary_label}.csv", primary_label))
    monthly_candidates.extend(
        [
            ("execution_aligned_monthly_backtest_summary.csv", "execution_aligned_monthly_backtest_summary"),
            ("primary_research_monthly_summary.csv", "primary_research_monthly_summary"),
            ("monthly_backtest_summary.csv", "monthly_backtest_summary"),
        ]
    )
    seen: set[str] = set()
    for filename, label in monthly_candidates:
        if filename in seen:
            continue
        seen.add(filename)
        path = run_dir / filename
        if not path.exists():
            continue
        frame = _read_csv(path)
        if "month" in frame.columns:
            frame["month"] = frame["month"].astype(str)
        return frame, label
    return pd.DataFrame(), "missing"


def _load_rankic_summary(run_dir: Path) -> dict[str, float]:
    path = run_dir / "validation_rankic_monthly_summary.csv"
    if not path.exists():
        return {
            "alpha_rankic_mean": float("nan"),
            "alpha_rankic_ir_mean": float("nan"),
            "risk_rankic_mean": float("nan"),
        }
    frame = _read_csv(path)
    if frame.empty or "target" not in frame.columns:
        return {
            "alpha_rankic_mean": float("nan"),
            "alpha_rankic_ir_mean": float("nan"),
            "risk_rankic_mean": float("nan"),
        }
    alpha_frame = frame[frame["target"].astype(str).str.startswith("fwd_excess_")].copy()
    risk_frame = frame[frame["target"].astype(str).str.startswith("risk_")].copy()
    return {
        "alpha_rankic_mean": float(alpha_frame["rankic_mean"].mean()) if not alpha_frame.empty else float("nan"),
        "alpha_rankic_ir_mean": float(alpha_frame["rankic_ir"].mean()) if not alpha_frame.empty else float("nan"),
        "risk_rankic_mean": float(risk_frame["rankic_mean"].mean()) if not risk_frame.empty else float("nan"),
    }


def _training_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    training = metrics.get("training_diagnostics", {})
    if not isinstance(training, dict):
        training = {}
    epochs_requested = int(training.get("epochs_requested", metrics.get("epochs", 0) or 0) or 0)
    selected_epoch = int(training.get("selected_epoch", training.get("best_epoch", 0) or 0) or 0)
    return {
        "training_status": str(training.get("status", "")).strip(),
        "epochs_requested": epochs_requested,
        "epochs_completed": int(training.get("epochs_completed", 0) or 0),
        "selected_epoch": selected_epoch,
        "selected_epoch_ratio": float(selected_epoch / epochs_requested) if epochs_requested > 0 else float("nan"),
        "selected_metric_name": str(training.get("selected_metric_name", "")).strip(),
        "selected_metric_value": float(training.get("selected_metric_value", float("nan")) or float("nan")),
        "best_valid_loss": float(training.get("best_valid_loss", float("nan")) or float("nan")),
        "objective_aligned_budget_pressure": bool(training.get("objective_aligned_budget_pressure", False)),
        "selected_in_tail": bool(training.get("selected_in_tail", False)),
        "selected_at_right_boundary": bool(training.get("selected_at_right_boundary", False)),
        "still_improving": bool(training.get("still_improving", False)),
    }


def _recent_row_for_profile(root_tag: str, profile_name: str, metrics_path: Path) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    label, primary = resolve_primary_backtest(metrics)
    run_dir = metrics_path.parent
    profile = get_profile(profile_name)
    monthly_frame, monthly_source = _load_monthly_summary(run_dir, metrics)
    monthly_diag = summarize_monthly_diagnostics(monthly_frame, return_column="excess_return")
    row: dict[str, Any] = {
        "profile_name": profile.name,
        "category": str(profile.category),
        "metrics_path": str(metrics_path),
        "run_dir": str(run_dir),
        "root_tag": root_tag,
        "primary_backtest_label": label,
        "monthly_source": monthly_source,
        "execution_alignment_profile": str(metrics.get("execution_alignment_profile", "")).strip(),
        "panel_mode": "execution_aligned" if label == "execution_aligned_holdout_backtest" else "raw",
        "annual_return": float(primary.get("annual_return", float("nan"))),
        "excess_annual_return": float(primary.get("excess_annual_return", float("nan"))),
        "excess_total_return": float(primary.get("excess_total_return", float("nan"))),
        "excess_sharpe": float(primary.get("excess_sharpe", float("nan"))),
        "excess_max_drawdown": float(primary.get("excess_max_drawdown", float("nan"))),
        "avg_turnover": float(primary.get("avg_turnover", float("nan"))),
        "avg_holding_count": float(primary.get("avg_holding_count", float("nan"))),
    }
    row.update(monthly_diag)
    row.update(_load_rankic_summary(run_dir))
    row.update(_training_summary(metrics))
    row.update(_extract_artifact_summary(run_dir))
    return row


def _formal_row_for_profile_window(root_tag: str, profile_name: str, window: WindowSpec, metrics_path: Path) -> dict[str, Any]:
    row = _recent_row_for_profile(root_tag, profile_name, metrics_path)
    row["window_label"] = window.label
    return row


def _category_winners(recent_frame: pd.DataFrame) -> pd.DataFrame:
    candidate_frame = recent_frame[recent_frame["category"] != "baseline"].copy()
    if candidate_frame.empty:
        return pd.DataFrame()
    candidate_frame = candidate_frame.sort_values(
        [
            "category",
            "objective_aligned_budget_pressure",
            "positive_month_ratio",
            "median_monthly_return",
            "worst_monthly_return",
            "top3_positive_month_share",
            "excess_sharpe",
            "excess_annual_return",
            "alpha_rankic_mean",
        ],
        ascending=[True, True, False, False, False, True, False, False, False],
    ).reset_index(drop=True)
    winners = candidate_frame.groupby("category", as_index=False, sort=False).first()
    return winners.reset_index(drop=True)


def _formal_summary(formal_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_name, frame in formal_frame.groupby("profile_name", sort=False):
        ordered = frame.sort_values("window_label").reset_index(drop=True)
        rows.append(
            {
                "profile_name": profile_name,
                "category": str(ordered["category"].iloc[0]),
                "window_count": int(len(ordered)),
                "mean_excess_annual_return": float(ordered["excess_annual_return"].mean()),
                "mean_excess_sharpe": float(ordered["excess_sharpe"].mean()),
                "mean_positive_month_ratio": float(ordered["positive_month_ratio"].mean()),
                "mean_median_monthly_return": float(ordered["median_monthly_return"].mean()),
                "worst_monthly_return": float(ordered["worst_monthly_return"].min()),
                "mean_top3_positive_month_share": float(ordered["top3_positive_month_share"].mean()),
                "mean_avg_turnover": float(ordered["avg_turnover"].mean()),
                "mean_alpha_rankic": float(ordered["alpha_rankic_mean"].mean()),
                "budget_pressure_count": int(ordered["objective_aligned_budget_pressure"].sum()),
                "tail_selected_count": int(ordered["selected_in_tail"].sum()),
                "boundary_selected_count": int(ordered["selected_at_right_boundary"].sum()),
                "selected_profiles": ",".join(sorted({str(v) for v in ordered["execution_alignment_profile"] if str(v).strip()})),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        [
            "mean_positive_month_ratio",
            "mean_median_monthly_return",
            "worst_monthly_return",
            "mean_top3_positive_month_share",
            "mean_excess_sharpe",
            "mean_excess_annual_return",
        ],
        ascending=[False, False, False, True, False, False],
    ).reset_index(drop=True)


def _format_pct(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "n/a"
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def _format_float(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "n/a"
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _build_analysis_payload(
    recent_frame: pd.DataFrame,
    winners: pd.DataFrame,
    formal_frame: pd.DataFrame,
    formal_summary: pd.DataFrame,
) -> dict[str, Any]:
    baseline_recent = recent_frame[recent_frame["profile_name"] == "baseline_current"].head(1)
    baseline_formal = formal_summary[formal_summary["profile_name"] == "baseline_current"].head(1)
    pressure_profiles = recent_frame[
        recent_frame["objective_aligned_budget_pressure"] | (recent_frame["selected_epoch_ratio"] >= 0.80)
    ]["profile_name"].astype(str).tolist()
    conversion_gap_profiles = []
    if not baseline_recent.empty:
        baseline_rankic = float(baseline_recent.iloc[0]["alpha_rankic_mean"])
        baseline_excess = float(baseline_recent.iloc[0]["excess_annual_return"])
        for _, row in recent_frame.iterrows():
            if str(row["profile_name"]) == "baseline_current":
                continue
            rankic = float(row["alpha_rankic_mean"])
            excess = float(row["excess_annual_return"])
            if pd.notna(rankic) and pd.notna(excess) and rankic > baseline_rankic and excess < baseline_excess:
                conversion_gap_profiles.append(str(row["profile_name"]))
    return {
        "baseline_recent": baseline_recent.to_dict(orient="records"),
        "baseline_formal": baseline_formal.to_dict(orient="records"),
        "recent_winners": winners.to_dict(orient="records"),
        "formal_summary": formal_summary.to_dict(orient="records"),
        "budget_pressure_profiles": sorted(set(pressure_profiles)),
        "conversion_gap_profiles": sorted(set(conversion_gap_profiles)),
        "formal_best_profile": str(formal_summary.iloc[0]["profile_name"]) if not formal_summary.empty else "",
        "formal_frame": formal_frame.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    if args.list_profiles:
        print("\n".join(list_profile_lines()))
        return

    recent_profiles = _parse_name_list(args.recent_profiles)
    if not recent_profiles:
        raise ValueError("No recent profiles were provided.")
    if "baseline_current" not in recent_profiles:
        raise ValueError("recent profiles must include baseline_current.")

    root_dir = OUTPUT_ROOT / str(args.root_tag).strip()
    recent_root_tag = f"{str(args.root_tag).strip()}_recent"
    formal_root_tag = f"{str(args.root_tag).strip()}_formal"
    root_dir.mkdir(parents=True, exist_ok=True)

    recent_rows: list[dict[str, Any]] = []
    for profile_name in recent_profiles:
        metrics_path = _resolve_recent_metrics_path(recent_root_tag, profile_name)
        if args.force_rerun or not metrics_path.exists():
            command = _build_research_command(
                python_executable=str(args.python_executable),
                experiment_tag=f"{recent_root_tag}__{profile_name}",
                profile_name=profile_name,
                window=RECENT_WINDOW,
                end_date=str(args.end_date),
                benchmark=str(args.benchmark),
                liquidity_pool=str(args.liquidity_pool),
                transaction_cost_bps=float(args.transaction_cost_bps),
                slippage_bps=float(args.slippage_bps),
                sell_tax_bps=float(args.sell_tax_bps),
                research_objective_mode=str(args.research_objective_mode),
                checkpoint_selection_objective=str(args.checkpoint_selection_objective),
                checkpoint_selection_min_improvement=float(args.checkpoint_selection_min_improvement),
                execution_alignment_objective=str(args.execution_alignment_objective),
                family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
            )
            _run_command(command)
        recent_rows.append(_recent_row_for_profile(recent_root_tag, profile_name, metrics_path))

    recent_frame = pd.DataFrame(recent_rows).sort_values(
        [
            "objective_aligned_budget_pressure",
            "positive_month_ratio",
            "median_monthly_return",
            "worst_monthly_return",
            "top3_positive_month_share",
            "excess_sharpe",
            "excess_annual_return",
        ],
        ascending=[True, False, False, False, True, False, False],
    ).reset_index(drop=True)
    winners = _category_winners(recent_frame)
    formal_profiles = ["baseline_current"] + [
        profile_name
        for profile_name in winners["profile_name"].astype(str).tolist()
        if str(profile_name).strip() and str(profile_name) != "baseline_current"
    ]

    formal_rows: list[dict[str, Any]] = []
    source_runs: dict[str, dict[str, str]] = {}
    for profile_name in formal_profiles:
        source_runs[profile_name] = {}
        for window in FORMAL_WINDOWS:
            if window.label == RECENT_WINDOW.label:
                metrics_path = _resolve_recent_metrics_path(recent_root_tag, profile_name)
            else:
                metrics_path = _resolve_formal_metrics_path(formal_root_tag, profile_name, window)
                if args.force_rerun or not metrics_path.exists():
                    command = _build_research_command(
                        python_executable=str(args.python_executable),
                        experiment_tag=f"{formal_root_tag}__{profile_name}_{window.label}",
                        profile_name=profile_name,
                        window=window,
                        end_date=str(args.end_date),
                        benchmark=str(args.benchmark),
                        liquidity_pool=str(args.liquidity_pool),
                        transaction_cost_bps=float(args.transaction_cost_bps),
                        slippage_bps=float(args.slippage_bps),
                        sell_tax_bps=float(args.sell_tax_bps),
                        research_objective_mode=str(args.research_objective_mode),
                        checkpoint_selection_objective=str(args.checkpoint_selection_objective),
                        checkpoint_selection_min_improvement=float(args.checkpoint_selection_min_improvement),
                        execution_alignment_objective=str(args.execution_alignment_objective),
                        family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
                    )
                    _run_command(command)
            source_runs[profile_name][window.label] = str(metrics_path)
            formal_rows.append(_formal_row_for_profile_window(formal_root_tag, profile_name, window, metrics_path))

    formal_frame = pd.DataFrame(formal_rows).sort_values(
        ["profile_name", "window_label"], ascending=[True, True]
    ).reset_index(drop=True)
    formal_summary = _formal_summary(formal_frame)
    analysis_payload = _build_analysis_payload(recent_frame, winners, formal_frame, formal_summary)

    recent_frame.to_csv(root_dir / "recent_matrix_summary.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(root_dir / "recent_category_winners.csv", index=False, encoding="utf-8-sig")
    formal_frame.to_csv(root_dir / "formal_window_detail.csv", index=False, encoding="utf-8-sig")
    formal_summary.to_csv(root_dir / "formal_profile_summary.csv", index=False, encoding="utf-8-sig")
    with (root_dir / "source_runs.json").open("w", encoding="utf-8") as f:
        json.dump(source_runs, f, ensure_ascii=False, indent=2)
    with (root_dir / "analysis_payload.json").open("w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, ensure_ascii=False, indent=2)

    report_lines: list[str] = [
        "# Architecture Protocol Refresh",
        "",
        "## Protocol",
        f"- recent_root_tag: `{recent_root_tag}`",
        f"- formal_root_tag: `{formal_root_tag}`",
        f"- universe: `{args.liquidity_pool}`",
        f"- benchmark: `{args.benchmark}`",
        f"- research objective: `{args.research_objective_mode}`",
        f"- checkpoint selection: `{args.checkpoint_selection_objective}`",
        f"- execution alignment: `train_eval_auto / {args.execution_alignment_objective}` with `{','.join(EXECUTION_ALIGNMENT_CANDIDATES)}`",
        f"- realistic cost: transaction `{args.transaction_cost_bps}` bps, slippage `{args.slippage_bps}` bps, sell-tax `{args.sell_tax_bps}` bps",
        f"- family epoch budget manifest: `{args.family_epoch_budget_manifest}`",
        "",
        "## Recent Category Winners",
    ]
    for _, row in winners.iterrows():
        report_lines.append(
            f"- `{row['category']}` -> `{row['profile_name']}`: excess annual {_format_pct(row['excess_annual_return'])}, "
            f"excess Sharpe {_format_float(row['excess_sharpe'])}, positive-month ratio {_format_pct(row['positive_month_ratio'])}, "
            f"median monthly excess {_format_pct(row['median_monthly_return'])}, worst month {_format_pct(row['worst_monthly_return'])}, "
            f"budget pressure `{bool(row['objective_aligned_budget_pressure'])}`"
        )
    report_lines.extend(["", "## Formal Summary"])
    for _, row in formal_summary.iterrows():
        report_lines.append(
            f"- `{row['profile_name']}` [{row['category']}]: mean excess annual {_format_pct(row['mean_excess_annual_return'])}, "
            f"mean excess Sharpe {_format_float(row['mean_excess_sharpe'])}, mean positive-month ratio {_format_pct(row['mean_positive_month_ratio'])}, "
            f"mean median monthly excess {_format_pct(row['mean_median_monthly_return'])}, worst month {_format_pct(row['worst_monthly_return'])}, "
            f"budget pressure windows `{int(row['budget_pressure_count'])}`"
        )
    report_lines.extend(["", "## Diagnosis"])
    if not formal_summary.empty and str(formal_summary.iloc[0]["profile_name"]) == "baseline_current":
        report_lines.append("- Current evidence says architecture alone is still not the main bottleneck; `baseline_current` remains formal rank 1 under the current protocol.")
    else:
        report_lines.append(
            f"- The formal rank 1 profile has switched to `{formal_summary.iloc[0]['profile_name']}`, which means architecture itself has become the main bottleneck again."
        )
    pressure_profiles = analysis_payload["budget_pressure_profiles"]
    if pressure_profiles:
        report_lines.append(f"- These profiles still show budget pressure or tail-selected checkpoints: `{','.join(pressure_profiles)}`.")
    conversion_gap_profiles = analysis_payload["conversion_gap_profiles"]
    if conversion_gap_profiles:
        report_lines.append(f"- These profiles show a conversion gap where RankIC improves but realized net return does not: `{','.join(conversion_gap_profiles)}`.")
    if not winners.empty:
        strongest_nonbaseline = formal_summary[formal_summary["profile_name"] != "baseline_current"].head(1)
        if not strongest_nonbaseline.empty:
            row = strongest_nonbaseline.iloc[0]
            report_lines.append(
                f"- The most interesting architecture challenger to keep studying is `{row['profile_name']}`, with formal means of "
                f"{_format_pct(row['mean_excess_annual_return'])} / {_format_float(row['mean_excess_sharpe'])}."
            )
    report_lines.extend(["", "## Next Directions"])
    report_lines.append("- If `baseline_current` still ranks first, priority should stay on weak-month repair, execution policy, and objective design instead of blindly making the network deeper or larger.")
    report_lines.append("- If a challenger has better RankIC but worse realized return, inspect score-to-weight mapping, concentration, and execution dilution before changing the backbone again.")
    report_lines.append("- Do not make final negative calls on branches that still show budget pressure; finish budget extension on those branches first.")
    with (root_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines).strip() + "\n")

    print(f"Architecture protocol refresh complete: {root_dir}")


if __name__ == "__main__":
    main()
