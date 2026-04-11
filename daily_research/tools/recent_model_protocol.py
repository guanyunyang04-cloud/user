from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.backtest import summarize_backtest_by_month, summarize_monthly_diagnostics
from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.deep_alpha.execution_alignment import default_auto_profile_argument
from daily_research.deep_alpha.family_epoch_budget import (
    DEFAULT_LATEST_MANIFEST_PATH,
    default_min_epochs_for_budget,
    resolve_epoch_budget_for_family,
)
from daily_research.deep_alpha.research_objective import resolve_primary_backtest
from daily_research.deep_alpha.short_alpha_profiles import build_profile_cli_args, get_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
RUN_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_deep_alpha_research.py"
EXECUTION_ALIGNMENT_PROFILE_SET = default_auto_profile_argument()
DEFAULT_RECENT_MODEL_ROOT_TAG = "short_alpha_recent_model_protocol_20260410_r1"


@dataclass(frozen=True)
class RecentModelWindow:
    label: str
    recent_start_date: str
    recent_end_date: str
    train_end_date: str
    valid_start_date: str
    valid_months: int


def build_recent_model_window(end_date: str = "", months: int = 12) -> RecentModelWindow:
    end_ts = pd.Timestamp(end_date or get_latest_completed_trading_date()).normalize()
    start_ts = (end_ts - pd.DateOffset(months=int(months)) + pd.Timedelta(days=1)).normalize()
    train_end_ts = (start_ts - pd.Timedelta(days=1)).normalize()
    return RecentModelWindow(
        label=f"{start_ts.strftime('%Y%m%d')}_{end_ts.strftime('%Y%m%d')}",
        recent_start_date=start_ts.strftime("%Y%m%d"),
        recent_end_date=end_ts.strftime("%Y%m%d"),
        train_end_date=train_end_ts.strftime("%Y-%m-%d"),
        valid_start_date=start_ts.strftime("%Y-%m-%d"),
        valid_months=int(months),
    )


def format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def format_num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def monthly_robust_score(monthly_diag: dict[str, Any]) -> float:
    positive_ratio = float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0)
    median_monthly_return = float(monthly_diag.get("median_monthly_return", 0.0) or 0.0)
    mean_monthly_return = float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0)
    worst_monthly_return = float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0)
    top3_positive_share = float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0)
    longest_negative_streak = int(monthly_diag.get("longest_negative_streak", 0) or 0)

    downside_penalty = max(-worst_monthly_return, 0.0)
    concentration_penalty = max(top3_positive_share - 0.60, 0.0)
    streak_penalty = max(longest_negative_streak - 2, 0)
    return float(
        mean_monthly_return
        + median_monthly_return
        + 0.05 * (positive_ratio - 0.50)
        - 0.35 * downside_penalty
        - 0.05 * concentration_penalty
        - 0.01 * float(streak_penalty)
    )


def _resolve_family_key(profile_name: str) -> str:
    return "baseline" if str(profile_name).strip().lower() == "baseline_current" else "short_alpha"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_primary_monthly_diagnostics(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
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
    monthly_path = run_dir / f"{monthly_label}.csv"
    if monthly_path.exists():
        monthly_df = pd.read_csv(monthly_path)
        return summarize_monthly_diagnostics(monthly_df, return_column="excess_return")
    backtest_label, _ = resolve_primary_backtest(metrics, research_objective_mode=str(metrics.get("research_objective_mode", "")))
    equity_path = run_dir / ("execution_aligned_equity_curve.csv" if backtest_label.startswith("execution_aligned_") else "equity_curve.csv")
    actions_path = run_dir / ("execution_aligned_actions.csv" if backtest_label.startswith("execution_aligned_") else "actions.csv")
    if not equity_path.exists():
        return {}
    equity_df = pd.read_csv(equity_path)
    if "date" in equity_df.columns:
        equity_df["date"] = pd.to_datetime(equity_df["date"], errors="coerce")
        equity_df = equity_df.dropna(subset=["date"]).set_index("date")
    action_df = pd.read_csv(actions_path) if actions_path.exists() else pd.DataFrame()
    monthly_df = summarize_backtest_by_month(equity_df, action_df)
    return summarize_monthly_diagnostics(monthly_df, return_column="excess_return")


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def resolve_recent_metrics_path(root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / f"{str(root_tag).strip()}__{str(profile_name).strip()}" / "metrics.json"


def _build_recent_research_command(
    *,
    python_executable: str,
    root_tag: str,
    profile_name: str,
    window: RecentModelWindow,
    family_epoch_budget_manifest: str,
    end_date: str,
    resume_run_dir: Path | None = None,
) -> list[str]:
    profile = get_profile(profile_name)
    family_key = _resolve_family_key(profile_name)
    epoch_budget = resolve_epoch_budget_for_family(
        family_key,
        manifest_path=family_epoch_budget_manifest,
        fallback_epochs=8,
    )
    command = [
        str(python_executable),
        str(RUN_SCRIPT),
        "--data-source",
        "tq",
        "--start-date",
        "20210101",
        "--end-date",
        str(end_date),
        "--benchmark",
        "000300.SH",
        "--liquidity-pool",
        "liquid500",
        "--research-time-unit",
        "calendar_months",
        "--train-end-date",
        str(window.train_end_date),
        "--valid-start-date",
        str(window.valid_start_date),
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
        "execution_first",
        "--checkpoint-selection-objective",
        "primary_monthly_robust_score",
        "--checkpoint-selection-min-improvement",
        "0.0001",
        "--checkpoint-eval-interval",
        "2",
        "--checkpoint-eval-start-epoch",
        "1",
        "--return-loss-mode",
        "top_bottom_bce",
        "--return-target-transform",
        "raw",
        "--score-risk-mode",
        "subtract",
        "--execution-alignment-mode",
        "train_eval_auto",
        "--execution-alignment-objective",
        "robust_composite",
        "--execution-alignment-shortlist-size",
        "8",
        "--execution-alignment-screen-window-days",
        "63",
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
        "--no-pin-memory",
        "--use-amp",
        "--no-safe-runtime-profile",
        "--experiment-tag",
        f"{str(root_tag).strip()}__{profile.name}",
    ]
    command.extend(build_profile_cli_args(profile, include_objective_overrides=True))
    if resume_run_dir is not None:
        command.extend(["--resume-run-dir", str(resume_run_dir.resolve()), "--resume-mode", "strict"])
    return command


def resolve_recent_run_dir(root_tag: str, profile_name: str) -> Path:
    return OUTPUT_ROOT / f"{str(root_tag).strip()}__{str(profile_name).strip()}"


def resolve_recent_resume_run_dir(root_tag: str, profile_name: str) -> Path | None:
    run_dir = resolve_recent_run_dir(root_tag, profile_name)
    if not run_dir.exists():
        return None
    if (run_dir / "deep_alpha_model.pt").exists():
        return run_dir
    return None


def ensure_recent_model_run(
    *,
    profile_name: str,
    root_tag: str,
    window: RecentModelWindow,
    python_executable: str,
    family_epoch_budget_manifest: str = str(DEFAULT_LATEST_MANIFEST_PATH),
    force_rerun: bool = False,
) -> Path:
    metrics_path = resolve_recent_metrics_path(root_tag, profile_name)
    if metrics_path.exists() and not force_rerun:
        return metrics_path
    resume_run_dir = None if force_rerun else resolve_recent_resume_run_dir(root_tag, profile_name)
    command = _build_recent_research_command(
        python_executable=python_executable,
        root_tag=root_tag,
        profile_name=profile_name,
        window=window,
        family_epoch_budget_manifest=family_epoch_budget_manifest,
        end_date=window.recent_end_date,
        resume_run_dir=resume_run_dir,
    )
    _run_command(command)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Recent model metrics missing after run: {metrics_path}")
    return metrics_path


def load_recent_model_row(*, profile_name: str, metrics_path: Path, notes: str = "") -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    run_dir = metrics_path.parent
    _, holdout = resolve_primary_backtest(metrics, research_objective_mode=str(metrics.get("research_objective_mode", "")))
    monthly_diag = _load_primary_monthly_diagnostics(run_dir, metrics)
    return {
        "profile_name": str(profile_name),
        "run_dir": str(run_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "notes": str(notes),
        "train_end": str(metrics.get("train_end", "") or ""),
        "valid_start": str(metrics.get("valid_start", "") or ""),
        "valid_end": str(metrics.get("valid_end", "") or ""),
        "checkpoint_selection_objective": str(metrics.get("checkpoint_selection_objective", "") or ""),
        "execution_alignment_profile": str(metrics.get("execution_alignment_profile", "") or ""),
        "score_head_method": str(metrics.get("score_head_method", "") or ""),
        "recent_annual_return": float(holdout.get("annual_return", 0.0) or 0.0),
        "recent_excess_annual_return": float(holdout.get("excess_annual_return", 0.0) or 0.0),
        "recent_excess_sharpe": float(holdout.get("excess_sharpe", 0.0) or 0.0),
        "recent_avg_turnover": float(holdout.get("avg_turnover", 0.0) or 0.0),
        "recent_positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
        "recent_median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
        "recent_mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
        "recent_worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
        "recent_top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
        "recent_longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
        "recent_monthly_robust_score": monthly_robust_score(monthly_diag),
    }


def ensure_recent_model_matrix(
    *,
    profile_names: list[str],
    root_tag: str = DEFAULT_RECENT_MODEL_ROOT_TAG,
    recent_end_date: str = "",
    recent_window_months: int = 12,
    python_executable: str = sys.executable,
    family_epoch_budget_manifest: str = str(DEFAULT_LATEST_MANIFEST_PATH),
    force_rerun: bool = False,
    notes_by_profile: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    window = build_recent_model_window(recent_end_date, recent_window_months)
    notes_by_profile = notes_by_profile or {}
    rows: list[dict[str, Any]] = []
    source_runs: dict[str, str] = {}
    for profile_name in profile_names:
        metrics_path = ensure_recent_model_run(
            profile_name=profile_name,
            root_tag=root_tag,
            window=window,
            python_executable=python_executable,
            family_epoch_budget_manifest=family_epoch_budget_manifest,
            force_rerun=force_rerun,
        )
        source_runs[str(profile_name)] = str(metrics_path.resolve())
        rows.append(
            load_recent_model_row(
                profile_name=profile_name,
                metrics_path=metrics_path,
                notes=str(notes_by_profile.get(str(profile_name), "")),
            )
        )
    frame = pd.DataFrame(rows).sort_values(
        [
            "recent_monthly_robust_score",
            "recent_positive_month_ratio",
            "recent_median_monthly_return",
            "recent_excess_annual_return",
            "recent_excess_sharpe",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary = {
        "recent_model_root_tag": str(root_tag),
        "recent_window_label": str(window.label),
        "recent_start_date": str(window.recent_start_date),
        "recent_end_date": str(window.recent_end_date),
        "recent_train_end_date": str(window.train_end_date),
        "recent_window_months": int(window.valid_months),
        "recent_winner_profile_name": str(frame.iloc[0]["profile_name"]) if not frame.empty else "",
        "recent_winner_monthly_robust_score": float(frame.iloc[0]["recent_monthly_robust_score"]) if not frame.empty else 0.0,
        "recent_rows": frame.to_dict("records"),
    }
    return frame, summary, source_runs
