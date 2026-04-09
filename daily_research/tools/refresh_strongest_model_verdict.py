from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from daily_research.baseline.data_provider import get_latest_completed_trading_date


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a strongest-model verdict for the current daily_research scope "
            "using rolling formal latest-model artifacts plus mandatory recent validation."
        )
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_strongest_model_verdict_20260409_r1")
    parser.add_argument("--recent-end-date", default="")
    parser.add_argument("--recent-window-months", type=int, default=12)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def _source_rows() -> list[dict[str, Any]]:
    short_alpha_monthly_root = OUTPUT_ROOT / "short_alpha_formal_head2head_20260405_monthly_checkpoint_r1"
    return [
        {
            "entry_name": "short_expert_monthly_v1__monthly_first_formal",
            "profile_name": "short_expert_monthly_v1",
            "scope_label": "eligible_monthly_first_formal_liquid500",
            "family_key": "short_alpha",
            "checkpoint_selection_objective": "primary_monthly_robust_score",
            "source_csv": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48",
            "run_dir": OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1",
            "eligible_for_winner": True,
            "notes": "current strongest monthly-first short-horizon expert candidate under 3 formal windows",
        },
        {
            "entry_name": "state_liquidity_listwise_v1__monthly_first_formal",
            "profile_name": "state_liquidity_listwise_v1",
            "scope_label": "eligible_monthly_first_formal_liquid500",
            "family_key": "short_alpha",
            "checkpoint_selection_objective": "primary_monthly_robust_score",
            "source_csv": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48",
            "run_dir": short_alpha_monthly_root / "runs" / "state_liquidity_listwise_v1_20250318_20260331",
            "eligible_for_winner": True,
            "notes": "current strongest stable base model under the same monthly-first 3-window protocol",
        },
        {
            "entry_name": "baseline_current__monthly_first_formal",
            "profile_name": "baseline_current",
            "scope_label": "eligible_monthly_first_formal_liquid500",
            "family_key": "baseline",
            "checkpoint_selection_objective": "primary_monthly_robust_score",
            "source_csv": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_short_expert_formal_head2head_20260407_r4_shortalpha48",
            "run_dir": short_alpha_monthly_root / "runs" / "baseline_current_20250318_20260331",
            "eligible_for_winner": True,
            "notes": "baseline reference inside the same monthly-first 3-window protocol",
        },
        {
            "entry_name": "state_liquidity_listwise_v1__annual_checkpoint_reference",
            "profile_name": "state_liquidity_listwise_v1",
            "scope_label": "reference_annual_checkpoint_formal_liquid500",
            "family_key": "short_alpha",
            "checkpoint_selection_objective": "primary_annual_return",
            "source_csv": OUTPUT_ROOT / "short_alpha_formal_head2head_20260409_recheck_r1" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "short_alpha_formal_head2head_20260409_recheck_r1",
            "run_dir": Path(),
            "eligible_for_winner": False,
            "notes": "reference only; same 3 windows but annual-checkpoint protocol, not the current monthly-first winner gate",
        },
        {
            "entry_name": "dynamic_graph_no_priors__cross_family_reference",
            "profile_name": "dynamic_graph_no_priors",
            "scope_label": "reference_cross_family_formal_liquid500",
            "family_key": "dynamic_graph",
            "checkpoint_selection_objective": "primary_annual_return",
            "source_csv": OUTPUT_ROOT / "dynamic_graph_liquid500_challenger_20260405_r1" / "profile_summary.csv",
            "source_root": OUTPUT_ROOT / "dynamic_graph_liquid500_challenger_20260405_r1",
            "run_dir": Path(),
            "eligible_for_winner": False,
            "notes": "cross-family reference only; useful for scale, but not included in the current monthly-first winner gate",
        },
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _panel_latest_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def _read_row(source: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(source["source_csv"])
    row = frame.loc[frame["profile_name"] == source["profile_name"]]
    if row.empty:
        raise KeyError(f"profile {source['profile_name']} not found in {source['source_csv']}")
    payload = row.iloc[0].to_dict()
    return {
        "entry_name": str(source["entry_name"]),
        "profile_name": str(source["profile_name"]),
        "scope_label": str(source["scope_label"]),
        "family_key": str(source["family_key"]),
        "checkpoint_selection_objective": str(source["checkpoint_selection_objective"]),
        "source_csv": str(source["source_csv"]),
        "source_root": str(source["source_root"]),
        "run_dir": "" if not source["run_dir"] else str(Path(source["run_dir"]).resolve()),
        "eligible_for_winner": bool(source["eligible_for_winner"]),
        "notes": str(source["notes"]),
        "window_count": int(payload.get("window_count", 0) or 0),
        "epoch_budget": int(payload.get("epoch_budget", payload.get("mean_epoch_budget", 0)) or 0),
        "mean_excess_annual_return": float(payload.get("mean_excess_annual_return", 0.0) or 0.0),
        "mean_excess_sharpe": float(payload.get("mean_excess_sharpe", 0.0) or 0.0),
        "mean_avg_turnover": float(payload.get("mean_avg_turnover", 0.0) or 0.0),
        "mean_excess_max_drawdown": float(payload.get("mean_excess_max_drawdown", 0.0) or 0.0),
        "mean_monthly_positive_ratio": float(payload.get("mean_monthly_positive_ratio", 0.0) or 0.0),
        "mean_monthly_median_excess_return": float(payload.get("mean_monthly_median_excess_return", 0.0) or 0.0),
        "worst_monthly_excess_return": float(payload.get("worst_monthly_excess_return", 0.0) or 0.0),
        "mean_monthly_top3_positive_share": float(payload.get("mean_monthly_top3_positive_share", 0.0) or 0.0),
        "max_monthly_negative_streak": int(payload.get("max_monthly_negative_streak", 0) or 0),
        "reused_window_count": int(payload.get("reused_window_count", 0) or 0),
    }


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_num(value: float) -> str:
    return f"{value:.3f}"


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


def _recent_window_bounds(end_date: str, months: int) -> tuple[str, str]:
    end_ts = pd.Timestamp(end_date or get_latest_completed_trading_date()).normalize()
    start_ts = (end_ts - pd.DateOffset(months=int(months)) + pd.Timedelta(days=1)).normalize()
    return start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d")


def _ensure_live_panels(run_dir: Path, recent_end_date: str) -> None:
    live_panel = run_dir / "daily_live_target_weight_panel.csv"
    latest_live_date = _panel_latest_date(live_panel)
    if latest_live_date is not None and latest_live_date >= pd.Timestamp(recent_end_date):
        return
    from daily_research.deep_alpha.export_live_panels_from_run import refresh_live_panels_for_run

    refresh_live_panels_for_run(run_dir, latest_end_date=recent_end_date)


def _resolve_recent_panels(run_dir: Path) -> tuple[Path, Path, str]:
    aligned_score = run_dir / "execution_aligned_daily_live_score_panel.csv"
    aligned_target = run_dir / "execution_aligned_daily_live_target_weight_panel.csv"
    if aligned_score.exists() and aligned_target.exists():
        return aligned_score, aligned_target, "execution_aligned_live"
    raw_score = run_dir / "daily_live_score_panel.csv"
    raw_target = run_dir / "daily_live_target_weight_panel.csv"
    if raw_score.exists() and raw_target.exists():
        return raw_score, raw_target, "raw_live"
    raise FileNotFoundError(f"No usable recent panels found under {run_dir}")


def _run_recent_validation(
    rows: pd.DataFrame,
    *,
    output_dir: Path,
    python_executable: str,
    recent_start_date: str,
    recent_end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recent_profiles_root = output_dir / "recent_validation" / "profiles"
    recent_profiles_root.mkdir(parents=True, exist_ok=True)

    recent_rows: list[dict[str, Any]] = []
    eligible = rows.loc[rows["eligible_for_winner"]].copy()
    for _, row in eligible.iterrows():
        run_dir = Path(str(row["run_dir"]))
        if not run_dir.exists():
            raise FileNotFoundError(f"Recent validation run_dir not found: {run_dir}")
        _ensure_live_panels(run_dir, recent_end_date)
        score_panel, target_panel, panel_mode = _resolve_recent_panels(run_dir)
        metrics = _load_json(run_dir / "metrics.json")
        replay_tag = f"{row['profile_name']}_recent_{recent_start_date}_{recent_end_date}"
        replay_dir = recent_profiles_root / replay_tag
        cmd = [
            str(python_executable),
            str(EXTERNAL_REPLAY_SCRIPT),
            "--score-panel-csv",
            str(score_panel),
            "--target-weight-panel-csv",
            str(target_panel),
            "--data-source",
            "tq",
            "--benchmark",
            str(metrics.get("benchmark", "000300.SH") or "000300.SH"),
            "--start-date",
            recent_start_date,
            "--end-date",
            recent_end_date,
            "--transaction-cost-bps",
            str(float(metrics.get("execution_alignment_transaction_cost_bps", 3.0) or 3.0)),
            "--slippage-bps",
            str(float(metrics.get("execution_alignment_slippage_bps", 7.0) or 7.0)),
            "--sell-tax-bps",
            str(float(metrics.get("execution_alignment_sell_tax_bps", 10.0) or 10.0)),
            "--output-dir",
            str(recent_profiles_root),
            "--experiment-tag",
            replay_tag,
            "--candidate-label",
            f"{row['profile_name']}_recent",
            "--no-market-regime-filter",
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        replay_metrics = _load_json(replay_dir / "metrics.json")
        monthly_diag = _load_json(replay_dir / "monthly_backtest_diagnostics.json")
        recent_rows.append(
            {
                "profile_name": str(row["profile_name"]),
                "run_dir": str(run_dir),
                "recent_panel_mode": panel_mode,
                "recent_replay_run_dir": str(replay_dir),
                "recent_annual_return": float(replay_metrics.get("annual_return", 0.0) or 0.0),
                "recent_excess_annual_return": float(replay_metrics.get("excess_annual_return", 0.0) or 0.0),
                "recent_excess_sharpe": float(replay_metrics.get("excess_sharpe", 0.0) or 0.0),
                "recent_avg_turnover": float(replay_metrics.get("avg_turnover", 0.0) or 0.0),
                "recent_positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
                "recent_median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
                "recent_mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
                "recent_worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
                "recent_top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
                "recent_longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
                "recent_monthly_robust_score": _monthly_robust_score(monthly_diag),
            }
        )

    recent_df = pd.DataFrame(recent_rows)
    if recent_df.empty:
        raise RuntimeError("Recent validation produced no rows.")
    recent_df = recent_df.sort_values(
        by=[
            "recent_monthly_robust_score",
            "recent_excess_annual_return",
            "recent_excess_sharpe",
            "recent_median_monthly_return",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    recent_df.to_csv(output_dir / "recent_validation_scoreboard.csv", index=False, encoding="utf-8-sig")

    best_recent = recent_df.iloc[0].to_dict()
    summary = {
        "recent_start_date": recent_start_date,
        "recent_end_date": recent_end_date,
        "recent_window_months": 12,
        "recent_winner_profile_name": str(best_recent["profile_name"]),
        "recent_winner_monthly_robust_score": float(best_recent["recent_monthly_robust_score"]),
        "recent_winner_excess_annual_return": float(best_recent["recent_excess_annual_return"]),
        "recent_winner_excess_sharpe": float(best_recent["recent_excess_sharpe"]),
        "recent_rows": recent_df.to_dict("records"),
    }
    (output_dir / "recent_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return recent_df, summary


def _build_summary(rows: pd.DataFrame, recent_df: pd.DataFrame, recent_summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eligible = rows.loc[rows["eligible_for_winner"]].copy()
    eligible = eligible.sort_values(
        [
            "mean_excess_annual_return",
            "mean_excess_sharpe",
            "mean_monthly_positive_ratio",
            "worst_monthly_excess_return",
            "mean_avg_turnover",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No eligible rows found for strongest-model verdict.")
    winner = eligible.iloc[0].to_dict()
    runner_up = eligible.iloc[1].to_dict() if len(eligible) > 1 else {}
    references = rows.loc[~rows["eligible_for_winner"]].copy()
    references = references.sort_values(
        ["mean_excess_annual_return", "mean_excess_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)

    recent_lookup = recent_df.set_index("profile_name").to_dict("index")
    winner_recent = recent_lookup.get(str(winner["profile_name"]), {})
    runner_up_recent = recent_lookup.get(str(runner_up.get("profile_name", "")), {})

    summary = {
        "scope": "liquid500 + execution_first + rolling formal 3 windows + as-of-window latest model + monthly-first checkpoint + window_count=3",
        "recent_validation_required": True,
        "recent_validation_complete": True,
        "recent_start_date": str(recent_summary.get("recent_start_date", "")),
        "recent_end_date": str(recent_summary.get("recent_end_date", "")),
        "winner_entry_name": str(winner["entry_name"]),
        "winner_profile_name": str(winner["profile_name"]),
        "winner_source_root": str(winner["source_root"]),
        "winner_mean_excess_annual_return": float(winner["mean_excess_annual_return"]),
        "winner_mean_excess_sharpe": float(winner["mean_excess_sharpe"]),
        "winner_mean_monthly_positive_ratio": float(winner["mean_monthly_positive_ratio"]),
        "winner_mean_monthly_median_excess_return": float(winner["mean_monthly_median_excess_return"]),
        "winner_worst_monthly_excess_return": float(winner["worst_monthly_excess_return"]),
        "winner_recent": winner_recent,
        "runner_up_entry_name": str(runner_up.get("entry_name", "")),
        "runner_up_profile_name": str(runner_up.get("profile_name", "")),
        "runner_up_mean_excess_annual_return": float(runner_up.get("mean_excess_annual_return", 0.0) or 0.0),
        "runner_up_mean_excess_sharpe": float(runner_up.get("mean_excess_sharpe", 0.0) or 0.0),
        "runner_up_recent": runner_up_recent,
        "recent_winner_profile_name": str(recent_summary.get("recent_winner_profile_name", "")),
        "references": references.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Strongest Model Verdict",
        "",
        "## Scope",
        "- current winner gate: `liquid500 + execution_first + rolling formal 3 windows + as-of-window latest model + primary_monthly_robust_score + window_count=3`",
        "- every formal window must use the latest label-available model as of that window",
        "- recent validation remains mandatory companion evidence and must not be omitted",
        f"- current recent window: `{recent_summary.get('recent_start_date', '')} -> {recent_summary.get('recent_end_date', '')}`",
        "",
        "## Winner",
        f"- winner: `{winner['profile_name']}`",
        f"- source_root: `{winner['source_root']}`",
        f"- mean excess annual: `{_format_pct(float(winner['mean_excess_annual_return']))}`",
        f"- mean excess Sharpe: `{_format_num(float(winner['mean_excess_sharpe']))}`",
        f"- mean positive-month ratio: `{_format_pct(float(winner['mean_monthly_positive_ratio']))}`",
        f"- mean monthly median excess: `{_format_pct(float(winner['mean_monthly_median_excess_return']))}`",
        f"- worst month excess: `{_format_pct(float(winner['worst_monthly_excess_return']))}`",
    ]
    if runner_up:
        lines.extend(
            [
                "",
                "## Runner-Up",
                f"- runner_up: `{runner_up['profile_name']}`",
                f"- mean excess annual: `{_format_pct(float(runner_up['mean_excess_annual_return']))}`",
                f"- mean excess Sharpe: `{_format_num(float(runner_up['mean_excess_sharpe']))}`",
                f"- mean positive-month ratio: `{_format_pct(float(runner_up['mean_monthly_positive_ratio']))}`",
                f"- mean monthly median excess: `{_format_pct(float(runner_up['mean_monthly_median_excess_return']))}`",
                f"- worst month excess: `{_format_pct(float(runner_up['worst_monthly_excess_return']))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recent Validation",
            f"- recent winner: `{recent_summary.get('recent_winner_profile_name', '')}`",
        ]
    )
    for _, recent_row in recent_df.iterrows():
        lines.append(
            f"- `{recent_row['profile_name']}`: recent excess annual `{_format_pct(float(recent_row['recent_excess_annual_return']))}`, "
            f"recent excess Sharpe `{_format_num(float(recent_row['recent_excess_sharpe']))}`, "
            f"recent positive-month ratio `{_format_pct(float(recent_row['recent_positive_month_ratio']))}`, "
            f"recent monthly robust `{_format_num(float(recent_row['recent_monthly_robust_score']))}`"
        )
    if not references.empty:
        lines.extend(
            [
                "",
                "## Reference Tracks",
            ]
        )
        for _, ref in references.iterrows():
            lines.append(
                f"- `{ref['entry_name']}`: excess annual `{_format_pct(float(ref['mean_excess_annual_return']))}`, "
                f"excess Sharpe `{_format_num(float(ref['mean_excess_sharpe']))}`, notes `{ref['notes']}`"
            )
    lines.extend(
        [
            "",
            "## Decision",
            f"- 当前 formal strongest research model 是 `{winner['profile_name']}`，并且 recent 必报证据已经补齐。",
            f"- 当前 recent 12 个月 companion winner 是 `{recent_summary.get('recent_winner_profile_name', '')}`；它用于回答最近一年兑现情况，不回填替代 formal strongest gate。",
            f"- 按当前协议，后续默认执行应直接物化 `{winner['profile_name']}`，并先做 latest-data `production full-fit + highest family budget`。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame([_read_row(source) for source in _source_rows()])
    rows.to_csv(output_dir / "verdict_leaderboard.csv", index=False, encoding="utf-8-sig")

    recent_start_date, recent_end_date = _recent_window_bounds(args.recent_end_date, args.recent_window_months)
    recent_df, recent_summary = _run_recent_validation(
        rows,
        output_dir=output_dir,
        python_executable=args.python_executable,
        recent_start_date=recent_start_date,
        recent_end_date=recent_end_date,
    )
    _build_summary(rows, recent_df, recent_summary, output_dir)


if __name__ == "__main__":
    main()
