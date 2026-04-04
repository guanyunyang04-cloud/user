import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.architecture_profiles import get_profile as get_architecture_profile
from daily_research.deep_alpha.dynamic_graph_profiles import get_profile as get_dynamic_graph_profile
from daily_research.deep_alpha.family_epoch_budget import (
    DEFAULT_LATEST_MANIFEST_PATH,
    get_family_budget_spec,
    list_family_budget_lines,
    load_family_epoch_budget_manifest,
)
from daily_research.deep_alpha.research_objective import DEFAULT_RESEARCH_OBJECTIVE_MODE
from daily_research.deep_alpha.short_alpha_profiles import get_profile as get_short_alpha_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"

EXECUTION_ALIGNMENT_CANDIDATES: tuple[str, ...] = (
    "raw_1d",
    "topk2_1d_regoff",
    "regoff_k2_10d_ensemble_native_anchor",
    "regon_k1_10d_ensemble_native_anchor",
)


@dataclass(frozen=True)
class CalibrationWindow:
    label: str
    train_end: str
    valid_start: str
    valid_months: int


DEFAULT_CALIBRATION_WINDOWS: tuple[CalibrationWindow, ...] = (
    CalibrationWindow(label="20220401_20220630", train_end="2022-03-31", valid_start="2022-04-01", valid_months=3),
    CalibrationWindow(label="20220701_20220930", train_end="2022-06-30", valid_start="2022-07-01", valid_months=3),
    CalibrationWindow(label="20221010_20221230", train_end="2022-09-30", valid_start="2022-10-10", valid_months=3),
)

DYNAMIC_GRAPH_CALIBRATION_WINDOWS: tuple[CalibrationWindow, ...] = (
    CalibrationWindow(label="20220901_20221031", train_end="2022-08-31", valid_start="2022-09-01", valid_months=2),
    CalibrationWindow(label="20221101_20221230", train_end="2022-10-31", valid_start="2022-11-01", valid_months=2),
    CalibrationWindow(label="20230103_20230131", train_end="2022-12-30", valid_start="2023-01-03", valid_months=1),
)

LATE_PREFORMAL_CALIBRATION_WINDOWS: tuple[CalibrationWindow, ...] = (
    CalibrationWindow(label="20220801_20221031", train_end="2022-07-29", valid_start="2022-08-01", valid_months=3),
    CalibrationWindow(label="20221101_20230131", train_end="2022-10-31", valid_start="2022-11-01", valid_months=3),
    CalibrationWindow(label="20230103_20230131", train_end="2022-12-30", valid_start="2023-01-03", valid_months=1),
)


def get_calibration_windows(family_key: str, preset: str = "family_default") -> tuple[CalibrationWindow, ...]:
    normalized = str(family_key or "").strip().lower()
    preset_name = str(preset or "family_default").strip().lower()
    if preset_name == "late_preformal" and normalized != "dynamic_graph":
        return LATE_PREFORMAL_CALIBRATION_WINDOWS
    if normalized == "dynamic_graph":
        return DYNAMIC_GRAPH_CALIBRATION_WINDOWS
    return DEFAULT_CALIBRATION_WINDOWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate family-specific epoch budgets with strict resume and execution-first frontier stop rules."
    )
    parser.add_argument("--root-tag", default="deep_alpha_family_epoch_frontier_20260404_r1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--families",
        default="baseline,structure,short_alpha,dynamic_graph",
        help="Comma-separated family keys from family_epoch_budget.py",
    )
    parser.add_argument("--list-families", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--initial-budgets", default="4,8,12,16")
    parser.add_argument("--extension-budgets", default="24,32")
    parser.add_argument(
        "--calibration-window-preset",
        choices=["family_default", "late_preformal"],
        default="family_default",
        help="Choose family-default calibration windows or a later pre-formal preset for non-dynamic families.",
    )
    parser.add_argument(
        "--frontier-objective",
        choices=["replay_excess_annual_return", "replay_excess_sharpe"],
        default="replay_excess_annual_return",
    )
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
        default="primary_excess_annual_return",
    )
    parser.add_argument("--checkpoint-selection-min-improvement", type=float, default=0.0001)
    parser.add_argument("--execution-alignment-objective", default="robust_composite")
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
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
        raise ValueError("At least one epoch budget must be provided.")
    return ordered


def _parse_family_list(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        raise ValueError("At least one family key must be provided.")
    ordered: list[str] = []
    for value in values:
        spec = get_family_budget_spec(value)
        if spec.family_key not in ordered:
            ordered.append(spec.family_key)
    return ordered


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _family_run_root(root_tag: str, family_key: str) -> Path:
    return OUTPUT_ROOT / root_tag / "families" / family_key


def _run_dir(root_tag: str, family_key: str, profile_name: str, epoch_budget: int, window: CalibrationWindow) -> Path:
    return _family_run_root(root_tag, family_key) / "runs" / f"{profile_name}_e{epoch_budget}_{window.label}"


def _replay_dir(root_tag: str, family_key: str, profile_name: str, epoch_budget: int, window: CalibrationWindow) -> Path:
    return _family_run_root(root_tag, family_key) / "replays" / f"{profile_name}_e{epoch_budget}_{window.label}"


def _load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _build_common_alignment_args(
    *,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
) -> list[str]:
    return [
        "--research-objective-mode",
        research_objective_mode,
        "--checkpoint-selection-objective",
        checkpoint_selection_objective,
        "--checkpoint-selection-min-improvement",
        str(checkpoint_selection_min_improvement),
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
    ]


def _build_architecture_command(
    *,
    python_executable: str,
    root_tag: str,
    family_key: str,
    profile_name: str,
    epoch_budget: int,
    window: CalibrationWindow,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
    resume_run_dir: Path | None,
) -> list[str]:
    profile = get_architecture_profile(profile_name)
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
        str(max(epoch_budget, 8)),
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
        "12",
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
        f"{root_tag}/families/{family_key}/runs/{profile.name}_e{epoch_budget}_{window.label}",
    ]
    cmd.extend(
        _build_common_alignment_args(
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            research_objective_mode=research_objective_mode,
            checkpoint_selection_objective=checkpoint_selection_objective,
            checkpoint_selection_min_improvement=checkpoint_selection_min_improvement,
            execution_alignment_objective=execution_alignment_objective,
        )
    )
    if resume_run_dir is not None:
        cmd.extend(["--resume-run-dir", str(resume_run_dir), "--resume-mode", "strict"])
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


def _build_short_alpha_command(
    *,
    python_executable: str,
    root_tag: str,
    family_key: str,
    profile_name: str,
    epoch_budget: int,
    window: CalibrationWindow,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
    resume_run_dir: Path | None,
) -> list[str]:
    profile = get_short_alpha_profile(profile_name)
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
        str(max(epoch_budget, 8)),
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
        "6",
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
        f"{root_tag}/families/{family_key}/runs/{profile.name}_e{epoch_budget}_{window.label}",
    ]
    cmd.extend(
        _build_common_alignment_args(
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            research_objective_mode=research_objective_mode,
            checkpoint_selection_objective=checkpoint_selection_objective,
            checkpoint_selection_min_improvement=checkpoint_selection_min_improvement,
            execution_alignment_objective=execution_alignment_objective,
        )
    )
    if resume_run_dir is not None:
        cmd.extend(["--resume-run-dir", str(resume_run_dir), "--resume-mode", "strict"])
    if profile.state_context:
        cmd.append("--state-context")
    if profile.liquidity_context:
        cmd.append("--liquidity-context")
    if profile.short_alpha_features:
        cmd.append("--short-alpha-features")
    return cmd


def _build_dynamic_graph_command(
    *,
    python_executable: str,
    root_tag: str,
    family_key: str,
    profile_name: str,
    epoch_budget: int,
    window: CalibrationWindow,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
    resume_run_dir: Path | None,
) -> list[str]:
    profile = get_dynamic_graph_profile(profile_name)
    cmd = [
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
        "calendar_months",
        "--train-end-date",
        window.train_end,
        "--valid-start-date",
        window.valid_start,
        "--valid-days",
        "0",
        "--valid-months",
        str(window.valid_months),
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
        str(max(epoch_budget, 8)),
        "--lr-plateau-patience",
        "4",
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
        "6",
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
        "--task-loss-weights",
        "5:0.2,10:0.3,20:0.5,downside:0.35",
        "--score-horizon-weights",
        "5:0.2,10:0.3,20:0.5",
        "--experiment-tag",
        f"{root_tag}/families/{family_key}/runs/{profile.name}_e{epoch_budget}_{window.label}",
    ]
    cmd.extend(
        _build_common_alignment_args(
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            research_objective_mode=research_objective_mode,
            checkpoint_selection_objective=checkpoint_selection_objective,
            checkpoint_selection_min_improvement=checkpoint_selection_min_improvement,
            execution_alignment_objective=execution_alignment_objective,
        )
    )
    if resume_run_dir is not None:
        cmd.extend(["--resume-run-dir", str(resume_run_dir), "--resume-mode", "strict"])
    if profile.dynamic_graph_layer:
        cmd.extend(
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
    return cmd


def _build_research_command(
    *,
    family_key: str,
    python_executable: str,
    root_tag: str,
    profile_name: str,
    epoch_budget: int,
    window: CalibrationWindow,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    research_objective_mode: str,
    checkpoint_selection_objective: str,
    checkpoint_selection_min_improvement: float,
    execution_alignment_objective: str,
    resume_run_dir: Path | None,
) -> list[str]:
    spec = get_family_budget_spec(family_key)
    if spec.runner_kind == "architecture":
        return _build_architecture_command(
            python_executable=python_executable,
            root_tag=root_tag,
            family_key=family_key,
            profile_name=profile_name,
            epoch_budget=epoch_budget,
            window=window,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            research_objective_mode=research_objective_mode,
            checkpoint_selection_objective=checkpoint_selection_objective,
            checkpoint_selection_min_improvement=checkpoint_selection_min_improvement,
            execution_alignment_objective=execution_alignment_objective,
            resume_run_dir=resume_run_dir,
        )
    if spec.runner_kind == "short_alpha":
        return _build_short_alpha_command(
            python_executable=python_executable,
            root_tag=root_tag,
            family_key=family_key,
            profile_name=profile_name,
            epoch_budget=epoch_budget,
            window=window,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            research_objective_mode=research_objective_mode,
            checkpoint_selection_objective=checkpoint_selection_objective,
            checkpoint_selection_min_improvement=checkpoint_selection_min_improvement,
            execution_alignment_objective=execution_alignment_objective,
            resume_run_dir=resume_run_dir,
        )
    if spec.runner_kind == "dynamic_graph":
        return _build_dynamic_graph_command(
            python_executable=python_executable,
            root_tag=root_tag,
            family_key=family_key,
            profile_name=profile_name,
            epoch_budget=epoch_budget,
            window=window,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            research_objective_mode=research_objective_mode,
            checkpoint_selection_objective=checkpoint_selection_objective,
            checkpoint_selection_min_improvement=checkpoint_selection_min_improvement,
            execution_alignment_objective=execution_alignment_objective,
            resume_run_dir=resume_run_dir,
        )
    raise ValueError(f"Unsupported runner_kind: {spec.runner_kind}")


def _build_external_replay_command(
    *,
    python_executable: str,
    root_tag: str,
    family_key: str,
    profile_name: str,
    epoch_budget: int,
    window: CalibrationWindow,
    score_panel_csv: Path,
    target_weight_panel_csv: Path,
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
        window.valid_start.replace("-", ""),
        "--end-date",
        "20260401",
        "--score-panel-csv",
        str(score_panel_csv),
        "--target-weight-panel-csv",
        str(target_weight_panel_csv),
        "--candidate-label",
        f"{family_key}_{profile_name}_e{epoch_budget}_{window.label}",
        "--rebalance-freq",
        "1d",
        "--rebalance-offset-mode",
        "single",
        "--rebalance-anchor-date",
        window.valid_start.replace("-", ""),
        "--transaction-cost-bps",
        str(transaction_cost_bps),
        "--slippage-bps",
        str(slippage_bps),
        "--sell-tax-bps",
        str(sell_tax_bps),
        "--no-market-regime-filter",
        "--output-dir",
        str(OUTPUT_ROOT / root_tag),
        "--experiment-tag",
        f"families/{family_key}/replays/{profile_name}_e{epoch_budget}_{window.label}",
    ]


def _collect_budget_rows(
    *,
    root_tag: str,
    family_key: str,
    profile_name: str,
    epoch_budget: int,
    windows: tuple[CalibrationWindow, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        metrics_path = _run_dir(root_tag, family_key, profile_name, epoch_budget, window) / "metrics.json"
        replay_metrics_path = _replay_dir(root_tag, family_key, profile_name, epoch_budget, window) / "metrics.json"
        metrics = _load_metrics(metrics_path)
        replay_metrics = _load_metrics(replay_metrics_path)
        training = dict(metrics.get("training_diagnostics", {}))
        rows.append(
            {
                "family_key": family_key,
                "profile_name": profile_name,
                "epoch_budget": int(epoch_budget),
                "window_label": window.label,
                "metrics_path": str(metrics_path.resolve()),
                "replay_metrics_path": str(replay_metrics_path.resolve()),
                "selected_execution_profile": str(metrics.get("execution_alignment_profile", "")),
                "replay_excess_annual_return": float(replay_metrics.get("excess_annual_return", 0.0)),
                "replay_excess_sharpe": float(replay_metrics.get("excess_sharpe", 0.0)),
                "replay_excess_max_drawdown": float(replay_metrics.get("excess_max_drawdown", 0.0)),
                "replay_avg_turnover": float(replay_metrics.get("avg_turnover", 0.0)),
                "selected_epoch": int(training.get("selected_epoch", 0) or 0),
                "epochs_completed": int(training.get("epochs_completed", 0) or 0),
                "selected_epoch_ratio": float(training.get("selected_epoch_ratio", 0.0) or 0.0),
                "selected_in_tail": bool(training.get("selected_in_tail", False)),
                "selected_at_right_boundary": bool(training.get("selected_at_right_boundary", False)),
                "still_improving": bool(training.get("still_improving", False)),
                "objective_aligned_budget_pressure": bool(training.get("objective_aligned_budget_pressure", False)),
                "training_status": str(training.get("status", "")),
            }
        )
    return rows


def _summarize_budget(frame: pd.DataFrame, epoch_budget: int) -> dict[str, Any]:
    budget_frame = frame.loc[frame["epoch_budget"] == int(epoch_budget)].copy()
    return {
        "epoch_budget": int(epoch_budget),
        "window_count": int(len(budget_frame)),
        "mean_replay_excess_annual_return": float(budget_frame["replay_excess_annual_return"].mean()),
        "mean_replay_excess_sharpe": float(budget_frame["replay_excess_sharpe"].mean()),
        "mean_replay_excess_max_drawdown": float(budget_frame["replay_excess_max_drawdown"].mean()),
        "mean_replay_avg_turnover": float(budget_frame["replay_avg_turnover"].mean()),
        "mean_selected_epoch_ratio": float(budget_frame["selected_epoch_ratio"].mean()),
        "selected_in_tail_count": int(budget_frame["selected_in_tail"].sum()),
        "selected_at_right_boundary_count": int(budget_frame["selected_at_right_boundary"].sum()),
        "still_improving_count": int(budget_frame["still_improving"].sum()),
        "budget_pressure_count": int(budget_frame["objective_aligned_budget_pressure"].sum()),
        "undertrained_count": int((budget_frame["training_status"] == "undertrained").sum()),
    }


def _summary_objective_key(frontier_objective: str) -> str:
    objective = str(frontier_objective).strip()
    if objective.startswith("mean_"):
        return objective
    return f"mean_{objective}"


def _pick_best_budget(summary_rows: list[dict[str, Any]], frontier_objective: str) -> dict[str, Any]:
    objective_key = _summary_objective_key(frontier_objective)
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            float(row.get(objective_key, float("-inf"))),
            float(row.get("mean_replay_excess_sharpe", float("-inf"))),
            -int(row.get("epoch_budget", 0)),
        ),
        reverse=True,
    )
    return dict(ordered[0])


def _should_extend(summary_rows: list[dict[str, Any]], frontier_objective: str) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    best_row = _pick_best_budget(summary_rows, frontier_objective)
    right_boundary = max(summary_rows, key=lambda row: int(row["epoch_budget"]))
    right_boundary_is_best = int(best_row["epoch_budget"]) == int(right_boundary["epoch_budget"])
    right_boundary_pressure = int(right_boundary.get("budget_pressure_count", 0)) > 0
    return bool(right_boundary_is_best or right_boundary_pressure), best_row, dict(right_boundary)


def _write_summary_markdown(root_tag: str, family_rows: list[dict[str, Any]], family_summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Deep Alpha Family Epoch Frontier Calibration",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Recommended Budgets",
        "",
        "| Family | Profile | Recommended Epoch Budget | Frontier Objective | Right Boundary | Stop Reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in family_summaries:
        lines.append(
            "| "
            f"{summary['family_key']} | {summary['profile_name']} | {summary['recommended_epoch_budget']} | "
            f"{summary['recommended_objective_value']:.4f} | {summary['right_boundary_budget']} | {summary['stop_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Budget Summary",
            "",
            "| Family | Epoch Budget | Mean Replay Excess Annual | Mean Replay Excess Sharpe | Budget Pressure Windows | Mean Selected Epoch Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in family_rows:
        lines.append(
            "| "
            f"{row['family_key']} | {row['epoch_budget']} | {row['mean_replay_excess_annual_return']:.4f} | "
            f"{row['mean_replay_excess_sharpe']:.4f} | {row['budget_pressure_count']} | {row['mean_selected_epoch_ratio']:.3f} |"
        )
    summary_path = OUTPUT_ROOT / root_tag / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.list_families:
        for line in list_family_budget_lines():
            print(line)
        return

    root_tag = str(args.root_tag).strip()
    families = _parse_family_list(args.families)
    initial_budgets = _parse_int_list(args.initial_budgets)
    extension_budgets = _parse_int_list(args.extension_budgets)
    queued_budgets_template = list(initial_budgets)
    for extension_budget in extension_budgets:
        if extension_budget > max(initial_budgets) and extension_budget not in queued_budgets_template:
            queued_budgets_template.append(extension_budget)

    family_budget_rows: list[dict[str, Any]] = []
    family_recommendations: dict[str, Any] = {}

    for family_key in families:
        spec = get_family_budget_spec(family_key)
        profile_name = spec.profile_name
        family_windows = get_calibration_windows(family_key, args.calibration_window_preset)
        tested_budgets: list[int] = []
        budget_summaries: list[dict[str, Any]] = []

        for budget in queued_budgets_template:
            if tested_budgets and budget > max(initial_budgets):
                continue_flag, _, _ = _should_extend(budget_summaries, args.frontier_objective)
                if not continue_flag:
                    break

            for window in family_windows:
                run_dir = _run_dir(root_tag, family_key, profile_name, budget, window)
                metrics_path = run_dir / "metrics.json"
                if not metrics_path.exists() or args.force_rerun:
                    prior_budget = None
                    for candidate in sorted((value for value in tested_budgets if value < budget), reverse=True):
                        prior_run_dir = _run_dir(root_tag, family_key, profile_name, candidate, window)
                        if (prior_run_dir / "metrics.json").exists():
                            prior_budget = candidate
                            break
                    resume_run_dir = None if prior_budget is None else _run_dir(root_tag, family_key, profile_name, prior_budget, window)
                    command = _build_research_command(
                        family_key=family_key,
                        python_executable=args.python_executable,
                        root_tag=root_tag,
                        profile_name=profile_name,
                        epoch_budget=budget,
                        window=window,
                        transaction_cost_bps=args.transaction_cost_bps,
                        slippage_bps=args.slippage_bps,
                        sell_tax_bps=args.sell_tax_bps,
                        research_objective_mode=str(args.research_objective_mode),
                        checkpoint_selection_objective=str(args.checkpoint_selection_objective),
                        checkpoint_selection_min_improvement=float(args.checkpoint_selection_min_improvement),
                        execution_alignment_objective=str(args.execution_alignment_objective),
                        resume_run_dir=resume_run_dir,
                    )
                    _run_command(command)

                replay_dir = _replay_dir(root_tag, family_key, profile_name, budget, window)
                replay_metrics_path = replay_dir / "metrics.json"
                if not replay_metrics_path.exists() or args.force_rerun:
                    replay_command = _build_external_replay_command(
                        python_executable=args.python_executable,
                        root_tag=root_tag,
                        family_key=family_key,
                        profile_name=profile_name,
                        epoch_budget=budget,
                        window=window,
                        score_panel_csv=run_dir / "execution_aligned_daily_score_panel.csv",
                        target_weight_panel_csv=run_dir / "execution_aligned_daily_target_weight_panel.csv",
                        transaction_cost_bps=args.transaction_cost_bps,
                        slippage_bps=args.slippage_bps,
                        sell_tax_bps=args.sell_tax_bps,
                    )
                    _run_command(replay_command)

            tested_budgets.append(int(budget))
            current_frame = pd.DataFrame(
                row
                for tested_budget in tested_budgets
                for row in _collect_budget_rows(
                    root_tag=root_tag,
                    family_key=family_key,
                    profile_name=profile_name,
                    epoch_budget=tested_budget,
                    windows=family_windows,
                )
            )
            budget_summaries = [_summarize_budget(current_frame, tested_budget) for tested_budget in tested_budgets]

        if not budget_summaries:
            raise RuntimeError(f"No budget summaries were produced for family {family_key}")

        best_row = _pick_best_budget(budget_summaries, args.frontier_objective)
        right_boundary = max(budget_summaries, key=lambda row: int(row["epoch_budget"]))
        stop_reason = "right_boundary_not_best_and_no_budget_pressure"
        if int(best_row["epoch_budget"]) == int(right_boundary["epoch_budget"]):
            stop_reason = "hit_frontier_limit_with_right_boundary_best"
        elif int(right_boundary.get("budget_pressure_count", 0)) > 0:
            stop_reason = "hit_frontier_limit_with_right_boundary_pressure"

        family_recommendations[family_key] = {
            "family_key": family_key,
            "profile_name": profile_name,
            "runner_kind": spec.runner_kind,
            "recommended_epoch_budget": int(best_row["epoch_budget"]),
            "recommended_objective_name": str(args.frontier_objective),
            "recommended_objective_value": float(best_row[_summary_objective_key(args.frontier_objective)]),
            "mean_replay_excess_sharpe": float(best_row["mean_replay_excess_sharpe"]),
            "right_boundary_budget": int(right_boundary["epoch_budget"]),
            "right_boundary_budget_pressure_count": int(right_boundary.get("budget_pressure_count", 0)),
            "tested_budgets": [int(item["epoch_budget"]) for item in budget_summaries],
            "stop_reason": stop_reason,
            "budget_summaries": budget_summaries,
            "source_root": str(_family_run_root(root_tag, family_key).resolve()),
        }
        for row in budget_summaries:
            family_budget_rows.append({"family_key": family_key, "profile_name": profile_name, **row})

    budget_frame = pd.DataFrame(family_budget_rows).sort_values(["family_key", "epoch_budget"])
    existing_manifest = load_family_epoch_budget_manifest(DEFAULT_LATEST_MANIFEST_PATH)
    existing_families = existing_manifest.get("families", {}) if isinstance(existing_manifest, dict) else {}
    existing_windows = existing_manifest.get("calibration_windows_by_family", {}) if isinstance(existing_manifest, dict) else {}
    merged_family_recommendations = dict(existing_families) if isinstance(existing_families, dict) else {}
    merged_family_recommendations.update(family_recommendations)
    merged_windows = dict(existing_windows) if isinstance(existing_windows, dict) else {}
    merged_windows.update(
        {
            family_key: [
                {
                    "label": window.label,
                    "train_end": window.train_end,
                    "valid_start": window.valid_start,
                    "valid_months": int(window.valid_months),
                }
                for window in get_calibration_windows(family_key, args.calibration_window_preset)
            ]
            for family_key in families
        }
    )
    manifest = {
        "framework": "deep_alpha_family_epoch_frontier",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root_tag": root_tag,
        "frontier_objective": str(args.frontier_objective),
        "research_objective_mode": str(args.research_objective_mode),
        "checkpoint_selection_objective": str(args.checkpoint_selection_objective),
        "calibration_window_preset": str(args.calibration_window_preset),
        "calibration_windows_by_family": merged_windows,
        "families": merged_family_recommendations,
    }

    output_dir = OUTPUT_ROOT / root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_frame.to_csv(output_dir / "family_epoch_budget_summary.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "family_epoch_budget_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(DEFAULT_LATEST_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _write_summary_markdown(root_tag, budget_frame.to_dict(orient="records"), list(family_recommendations.values()))
    print(json.dumps({"root_tag": root_tag, "families": merged_family_recommendations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
