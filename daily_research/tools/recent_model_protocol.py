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


def _load_long_value_panel(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["stock"] = frame["stock"].astype(str).str.upper().str.strip()
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    frame = frame.dropna(subset=["date", "stock", value_name])
    if frame.empty:
        return pd.DataFrame()
    return (
        frame[["date", "stock", value_name]]
        .drop_duplicates(subset=["date", "stock"], keep="last")
        .pivot(index="date", columns="stock", values=value_name)
        .sort_index()
        .fillna(0.0)
    )


def _load_regime_state_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    if "date" in frame.columns:
        date_col = "date"
    elif "Date" in frame.columns:
        date_col = "Date"
    else:
        date_col = str(frame.columns[0])
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return pd.DataFrame()
    frame = frame.set_index(date_col).sort_index()

    def _normalize_label_series(series: pd.Series) -> pd.Series:
        normalized = series.astype("string").str.strip().str.lower()
        normalized = normalized.mask(normalized.eq(""), pd.NA)
        return normalized

    if "market_state" in frame.columns:
        frame["market_state"] = _normalize_label_series(frame["market_state"])
    if "quadrant" in frame.columns:
        frame["quadrant"] = _normalize_label_series(frame["quadrant"])
    if "trend_bucket" in frame.columns:
        frame["trend_bucket"] = _normalize_label_series(frame["trend_bucket"])
    if "vol_bucket" in frame.columns:
        frame["vol_bucket"] = _normalize_label_series(frame["vol_bucket"])

    trend_source = ""
    for candidate in ("benchmark_trend_gap", "trend_gap_60", "trend_gap_20"):
        if candidate in frame.columns:
            trend_source = candidate
            break
    vol_source = ""
    for candidate in ("benchmark_annual_vol", "vol_20", "vol_60"):
        if candidate in frame.columns:
            vol_source = candidate
            break

    if trend_source:
        trend_gap = pd.to_numeric(frame[trend_source], errors="coerce")
        if "benchmark_trend_gap" not in frame.columns:
            frame["benchmark_trend_gap"] = trend_gap
        trend_pass = trend_gap.gt(0.0)
        if "trend_pass" not in frame.columns:
            frame["trend_pass"] = trend_pass.fillna(False).astype(bool)
        if "trend_bucket" not in frame.columns:
            trend_bucket = pd.Series("trend_flat", index=frame.index, dtype="object")
            trend_bucket.loc[trend_gap > 0.01] = "trend_up"
            trend_bucket.loc[trend_gap < -0.01] = "trend_down"
            trend_bucket.loc[trend_gap.isna()] = pd.NA
            frame["trend_bucket"] = _normalize_label_series(trend_bucket)

    if vol_source:
        annual_vol = pd.to_numeric(frame[vol_source], errors="coerce")
        if "benchmark_annual_vol" not in frame.columns:
            frame["benchmark_annual_vol"] = annual_vol
        if "benchmark_vol_gap" not in frame.columns:
            frame["benchmark_vol_gap"] = annual_vol.sub(0.28)
        if "benchmark_vol_ratio" not in frame.columns:
            frame["benchmark_vol_ratio"] = annual_vol.div(0.28).replace([float("inf"), float("-inf")], pd.NA)
        vol_pass = annual_vol.le(0.28)
        if "vol_pass" not in frame.columns:
            frame["vol_pass"] = vol_pass.fillna(False).astype(bool)
        if "vol_bucket" not in frame.columns:
            vol_bucket = pd.Series("vol_mid", index=frame.index, dtype="object")
            vol_bucket.loc[annual_vol <= 0.252] = "vol_low"
            vol_bucket.loc[annual_vol > 0.308] = "vol_high"
            vol_bucket.loc[annual_vol.isna()] = pd.NA
            frame["vol_bucket"] = _normalize_label_series(vol_bucket)

    if "trend_bucket" in frame.columns and "vol_bucket" in frame.columns and "market_state" not in frame.columns:
        trend_bucket = _normalize_label_series(frame["trend_bucket"])
        vol_bucket = _normalize_label_series(frame["vol_bucket"])
        market_state = trend_bucket.str.cat(vol_bucket, sep="_")
        market_state.loc[trend_bucket.isna() | vol_bucket.isna()] = pd.NA
        frame["market_state"] = _normalize_label_series(market_state)

    if "quadrant" not in frame.columns and "benchmark_trend_gap" in frame.columns and "benchmark_annual_vol" in frame.columns:
        trend_gap = pd.to_numeric(frame["benchmark_trend_gap"], errors="coerce")
        annual_vol = pd.to_numeric(frame["benchmark_annual_vol"], errors="coerce")
        quadrant = pd.Series(pd.NA, index=frame.index, dtype="object")
        trend_up = trend_gap.gt(0.0)
        low_vol = annual_vol.le(0.28)
        ready = trend_gap.notna() & annual_vol.notna()
        quadrant.loc[trend_up & low_vol] = "trend_up_low_vol"
        quadrant.loc[trend_up & ~low_vol] = "trend_up_high_vol"
        quadrant.loc[~trend_up & low_vol] = "trend_down_low_vol"
        quadrant.loc[~trend_up & ~low_vol] = "trend_down_high_vol"
        quadrant.loc[~ready] = pd.NA
        frame["quadrant"] = _normalize_label_series(quadrant)

    if "quadrant" not in frame.columns and "state_name" in frame.columns:
        state_name = _normalize_label_series(frame["state_name"])
        legacy_mask = state_name.isin(
            [
                "trend_up_low_vol",
                "trend_up_high_vol",
                "trend_down_low_vol",
                "trend_down_high_vol",
            ]
        )
        if bool(legacy_mask.any()):
            frame["quadrant"] = state_name.where(legacy_mask)

    if "market_state" not in frame.columns and "quadrant" in frame.columns:
        quadrant = _normalize_label_series(frame["quadrant"])
        quadrant_to_market = {
            "trend_up_low_vol": "trend_up_vol_low",
            "trend_up_high_vol": "trend_up_vol_high",
            "trend_down_low_vol": "trend_down_vol_low",
            "trend_down_high_vol": "trend_down_vol_high",
        }
        frame["market_state"] = _normalize_label_series(quadrant.map(quadrant_to_market))

    if "regime_ready" not in frame.columns:
        readiness_sources = [col for col in ("benchmark_trend_gap", "benchmark_annual_vol", "quadrant", "market_state") if col in frame.columns]
        if readiness_sources:
            ready_mask = pd.Series(True, index=frame.index, dtype=bool)
            for column_name in readiness_sources[:2]:
                ready_mask = ready_mask & frame[column_name].notna()
            frame["regime_ready"] = ready_mask.astype(bool)

    if "regime_on" not in frame.columns and "quadrant" in frame.columns:
        quadrant = _normalize_label_series(frame["quadrant"])
        ready_mask = frame["regime_ready"].astype(bool) if "regime_ready" in frame.columns else quadrant.notna()
        frame["regime_on"] = quadrant.isin({"trend_up_low_vol", "trend_up_high_vol"}).fillna(False) & ready_mask

    return frame


def resolve_recent_result_dir(summary_entry: dict[str, Any]) -> Path:
    for key in ("run_dir", "recent_replay_run_dir"):
        raw = str(summary_entry.get(key, "") or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return Path()


def resolve_repair_companion_entry(
    summary_payload: dict[str, Any],
    *,
    winner_entry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    winner_dir = resolve_recent_result_dir(winner_entry or {})
    for key in ("recent_winner", "runner_up_recent"):
        entry = summary_payload.get(key, {})
        if not isinstance(entry, dict) or not entry:
            continue
        entry_dir = resolve_recent_result_dir(entry)
        if winner_dir and entry_dir and entry_dir == winner_dir:
            continue
        return entry, key
    return {}, ""


def load_recent_protocol_bundle(summary_entry: dict[str, Any]) -> dict[str, Any]:
    run_dir = resolve_recent_result_dir(summary_entry)
    if not run_dir.exists():
        raise FileNotFoundError(f"Recent protocol run dir not found: {run_dir}")

    score_candidates = (
        "execution_aligned_daily_score_panel.csv",
        "aligned_daily_score_panel.csv",
        "daily_score_panel.csv",
    )
    target_candidates = (
        "execution_aligned_daily_target_weight_panel.csv",
        "aligned_daily_target_weight_panel.csv",
        "daily_target_weight_panel.csv",
    )
    regime_candidates = (
        "market_state_frame.csv",
        "regime_state.csv",
    )

    score_panel = pd.DataFrame()
    score_panel_csv = Path()
    for filename in score_candidates:
        candidate = run_dir / filename
        if candidate.exists():
            score_panel = _load_long_value_panel(candidate, "score")
            score_panel_csv = candidate
            break

    target_panel = pd.DataFrame()
    target_panel_csv = Path()
    for filename in target_candidates:
        candidate = run_dir / filename
        if candidate.exists():
            target_panel = _load_long_value_panel(candidate, "target_weight")
            target_panel_csv = candidate
            break

    regime_state = pd.DataFrame()
    regime_state_csv = Path()
    for filename in regime_candidates:
        candidate = run_dir / filename
        if candidate.exists():
            regime_state = _load_regime_state_panel(candidate)
            regime_state_csv = candidate
            break

    return {
        "run_dir": run_dir,
        "score_panel": score_panel,
        "target_panel": target_panel,
        "regime_state": regime_state,
        "score_panel_csv": score_panel_csv,
        "target_panel_csv": target_panel_csv,
        "regime_state_csv": regime_state_csv,
        "protocol": "corrected_recent_execution_aligned" if score_panel_csv.name.startswith("execution_aligned_") else "legacy_recent_replay",
    }


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
