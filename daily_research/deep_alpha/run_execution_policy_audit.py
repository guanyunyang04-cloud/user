from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.execution_alignment import (
    default_auto_profile_argument,
    get_profile,
    parse_profile_name_list,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a deep_alpha run under a wider execution-policy search space and "
            "report which execution policy maximizes net replay profit."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--panel-scope",
        choices=["auto", "formal", "live"],
        default="auto",
        help="formal uses daily_* panels; live uses daily_live_* panels.",
    )
    parser.add_argument(
        "--profile-set",
        default=default_auto_profile_argument(),
        help="Comma-separated execution profiles or a registered profile set such as profit_max_v1.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["excess_annual_return", "excess_sharpe", "monthly_robust_score"],
        default="excess_annual_return",
        help="How to rank execution policies in the final audit summary.",
    )
    parser.add_argument("--start-date", default="", help="Optional replay start-date override.")
    parser.add_argument("--end-date", default="", help="Optional replay end-date override.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _panel_latest_bounds(path: Path) -> tuple[str, str]:
    frame = pd.read_csv(path, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values()
    if dates.empty:
        raise RuntimeError(f"Panel has no valid dates: {path}")
    return pd.Timestamp(dates.iloc[0]).strftime("%Y%m%d"), pd.Timestamp(dates.iloc[-1]).strftime("%Y%m%d")


def _resolve_panel_paths(run_dir: Path, panel_scope: str) -> tuple[str, Path, Path]:
    candidates = {
        "formal": (
            run_dir / "daily_score_panel.csv",
            run_dir / "daily_target_weight_panel.csv",
        ),
        "live": (
            run_dir / "daily_live_score_panel.csv",
            run_dir / "daily_live_target_weight_panel.csv",
        ),
    }
    if panel_scope == "auto":
        for scope_name in ("formal", "live"):
            score_path, target_path = candidates[scope_name]
            if score_path.exists() and target_path.exists():
                return scope_name, score_path, target_path
        raise FileNotFoundError(f"No usable daily_* or daily_live_* panels found in {run_dir}")
    score_path, target_path = candidates[panel_scope]
    if not score_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            f"Requested panel_scope={panel_scope} but panels are missing: "
            f"{score_path.name}, {target_path.name}"
        )
    return panel_scope, score_path, target_path


def _monthly_robust_score(monthly_diag: dict[str, Any]) -> float:
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


def _rank_tuple(row: dict[str, Any], selection_objective: str) -> tuple[float, float, float, float]:
    if selection_objective == "excess_sharpe":
        return (
            float(row.get("excess_sharpe", float("-inf"))),
            float(row.get("excess_annual_return", float("-inf"))),
            float(row.get("median_monthly_return", float("-inf"))),
            -float(row.get("avg_turnover", float("inf"))),
        )
    if selection_objective == "monthly_robust_score":
        return (
            float(row.get("monthly_robust_score", float("-inf"))),
            float(row.get("excess_annual_return", float("-inf"))),
            float(row.get("excess_sharpe", float("-inf"))),
            -float(row.get("avg_turnover", float("inf"))),
        )
    return (
        float(row.get("excess_annual_return", float("-inf"))),
        float(row.get("excess_sharpe", float("-inf"))),
        float(row.get("median_monthly_return", float("-inf"))),
        -float(row.get("avg_turnover", float("inf"))),
    )


def _build_replay_command(
    *,
    python_executable: str,
    score_panel_csv: Path,
    target_weight_panel_csv: Path,
    profile_name: str,
    start_date: str,
    end_date: str,
    metrics: dict[str, Any],
    output_dir: Path,
    experiment_tag: str,
) -> list[str]:
    profile = get_profile(profile_name)
    resolved_anchor_date = str(profile.rebalance_anchor_date or "")
    if resolved_anchor_date:
        try:
            anchor_ts = pd.Timestamp(resolved_anchor_date)
            if anchor_ts > pd.Timestamp(end_date):
                resolved_anchor_date = str(pd.Timestamp(start_date).strftime("%Y-%m-%d"))
        except Exception:
            resolved_anchor_date = str(profile.rebalance_anchor_date or "")
    cmd = [
        str(python_executable),
        str(EXTERNAL_REPLAY_SCRIPT),
        "--score-panel-csv",
        str(score_panel_csv),
        "--target-weight-panel-csv",
        str(target_weight_panel_csv),
        "--data-source",
        "tq",
        "--benchmark",
        str(metrics.get("benchmark", "000300.SH") or "000300.SH"),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--rebalance-freq",
        str(profile.rebalance_freq),
        "--rebalance-offset-mode",
        str(profile.rebalance_offset_mode),
        "--target-weight-top-k",
        str(int(profile.target_weight_top_k)),
        "--target-weight-min-weight",
        str(float(profile.target_weight_min_weight)),
        "--target-weight-power",
        str(float(profile.target_weight_power)),
        "--transaction-cost-bps",
        str(float(metrics.get("execution_alignment_transaction_cost_bps", 3.0) or 3.0)),
        "--slippage-bps",
        str(float(metrics.get("execution_alignment_slippage_bps", 7.0) or 7.0)),
        "--sell-tax-bps",
        str(float(metrics.get("execution_alignment_sell_tax_bps", 10.0) or 10.0)),
        "--output-dir",
        str(output_dir),
        "--experiment-tag",
        experiment_tag,
        "--candidate-label",
        profile_name,
    ]
    if resolved_anchor_date:
        cmd.extend(["--rebalance-anchor-date", resolved_anchor_date])
    if profile.target_weight_full_invest:
        cmd.append("--target-weight-full-invest")
    if not profile.use_market_regime_filter:
        cmd.append("--no-market-regime-filter")
    return cmd


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    metrics = _load_json(run_dir / "metrics.json")
    if not metrics:
        raise FileNotFoundError(f"metrics.json not found or invalid under {run_dir}")

    resolved_scope, score_panel_csv, target_weight_panel_csv = _resolve_panel_paths(run_dir, args.panel_scope)
    panel_start, panel_end = _panel_latest_bounds(target_weight_panel_csv)
    start_date = str(args.start_date or panel_start)
    end_date = str(args.end_date or panel_end)
    profile_names = parse_profile_name_list(args.profile_set)
    current_profile = str(metrics.get("execution_alignment_profile", "") or "").strip()
    if current_profile and current_profile not in profile_names:
        profile_names.append(current_profile)

    output_root = Path(args.output_root).resolve()
    root_tag = args.experiment_tag.strip() or f"execution_policy_audit_{run_dir.name}_{resolved_scope}_{datetime.now():%Y%m%d_%H%M%S}"
    audit_root = output_root / root_tag
    profiles_root = audit_root / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for profile_name in profile_names:
        run_tag = f"{profile_name}_{start_date}_{end_date}"
        cmd = _build_replay_command(
            python_executable=args.python_executable,
            score_panel_csv=score_panel_csv,
            target_weight_panel_csv=target_weight_panel_csv,
            profile_name=profile_name,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            output_dir=profiles_root,
            experiment_tag=run_tag,
        )
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        profile_run_dir = profiles_root / run_tag
        replay_metrics = _load_json(profile_run_dir / "metrics.json")
        monthly_diag = _load_json(profile_run_dir / "monthly_backtest_diagnostics.json")
        row = {
            "profile_name": profile_name,
            "run_dir": str(profile_run_dir),
            "annual_return": float(replay_metrics.get("annual_return", 0.0) or 0.0),
            "excess_annual_return": float(replay_metrics.get("excess_annual_return", 0.0) or 0.0),
            "excess_sharpe": float(replay_metrics.get("excess_sharpe", 0.0) or 0.0),
            "avg_turnover": float(replay_metrics.get("avg_turnover", 0.0) or 0.0),
            "positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
            "median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
            "mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
            "worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
            "top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
            "longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
        }
        row["monthly_robust_score"] = _monthly_robust_score(monthly_diag)
        row["is_current_profile"] = bool(profile_name == current_profile)
        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("Execution policy audit produced no rows.")

    best_row = max(summary_rows, key=lambda row: _rank_tuple(row, args.selection_objective))
    current_row = next((row for row in summary_rows if bool(row.get("is_current_profile"))), None)

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        by=[
            "excess_annual_return",
            "excess_sharpe",
            "median_monthly_return",
        ],
        ascending=[False, False, False],
    )
    summary_frame.to_csv(audit_root / "execution_policy_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Execution Policy Audit",
        "",
        f"- run_dir: `{run_dir}`",
        f"- panel_scope: `{resolved_scope}`",
        f"- panel_window: `{start_date} -> {end_date}`",
        f"- profile_set: `{args.profile_set}`",
        f"- selection_objective: `{args.selection_objective}`",
        f"- current_profile: `{current_profile or 'none'}`",
        f"- best_profile: `{best_row['profile_name']}`",
        f"- best_excess_annual_return: `{float(best_row['excess_annual_return']):.2%}`",
        f"- best_excess_sharpe: `{float(best_row['excess_sharpe']):.3f}`",
        f"- best_positive_month_ratio: `{float(best_row['positive_month_ratio']):.2%}`",
        f"- best_median_monthly_return: `{float(best_row['median_monthly_return']):.2%}`",
    ]
    if current_row is not None:
        lines.extend(
            [
                "",
                "## Current vs Best",
                "",
                f"- current_excess_annual_return: `{float(current_row['excess_annual_return']):.2%}`",
                f"- current_excess_sharpe: `{float(current_row['excess_sharpe']):.3f}`",
                f"- delta_excess_annual_return: `{float(best_row['excess_annual_return']) - float(current_row['excess_annual_return']):.2%}`",
                f"- delta_excess_sharpe: `{float(best_row['excess_sharpe']) - float(current_row['excess_sharpe']):.3f}`",
                f"- should_switch: `{best_row['profile_name'] != current_row['profile_name']}`",
            ]
        )
    (audit_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    console_summary = {
        "output_dir": str(audit_root),
        "run_dir": str(run_dir),
        "panel_scope": resolved_scope,
        "selection_objective": args.selection_objective,
        "current_profile": current_profile,
        "best_profile": str(best_row["profile_name"]),
        "best_excess_annual_return": float(best_row["excess_annual_return"]),
        "best_excess_sharpe": float(best_row["excess_sharpe"]),
        "should_switch": bool(current_row is None or best_row["profile_name"] != current_row["profile_name"]),
    }
    print(json.dumps(console_summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
