from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.family_epoch_budget import DEFAULT_LATEST_MANIFEST_PATH
from daily_research.deep_alpha.research_objective import summarize_primary_monthly_objectives
from daily_research.execution.update_default_candidate_production import (
    DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST,
    _activate_strategy,
    _build_retrain_command,
    _sync_production_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"
RUN_TRADE_PLAN_SCRIPT = PROJECT_ROOT / "daily_research" / "execution" / "run_trade_plan.py"
DEFAULT_PRODUCTION_ROOT = OUTPUT_ROOT / "deep_alpha_short_alpha_execalign_production_default"
DEFAULT_PRODUCTION_MANIFEST = DEFAULT_PRODUCTION_ROOT / "production_retrain_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend the active short_alpha production recipe with strict resume, "
            "replay the candidates, and optionally promote the best epoch budget back into production."
        )
    )
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--production-manifest", default=str(DEFAULT_PRODUCTION_MANIFEST))
    parser.add_argument("--production-root", default=str(DEFAULT_PRODUCTION_ROOT))
    parser.add_argument("--strategy-manifest-path", default=str(DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST))
    parser.add_argument("--family-epoch-budget-manifest", default=str(DEFAULT_LATEST_MANIFEST_PATH))
    parser.add_argument("--epoch-budgets", default="32,48,64")
    parser.add_argument("--root-tag", default="short_alpha_production_epoch_extension_20260405_r1")
    parser.add_argument("--replay-start-date", default="")
    parser.add_argument("--replay-end-date", default="")
    parser.add_argument("--activate-best", dest="activate_best", action="store_true")
    parser.add_argument("--no-activate-best", dest="activate_best", action="store_false")
    parser.set_defaults(activate_best=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _parse_epoch_budgets(raw: str) -> list[int]:
    budgets: list[int] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Epoch budget must be positive: {value}")
        budgets.append(value)
    if not budgets:
        raise ValueError("At least one epoch budget is required.")
    return sorted(dict.fromkeys(budgets))


def _replace_arg(cmd: list[str], flag: str, value: Any) -> None:
    token = str(flag)
    replacement = str(value)
    for idx, item in enumerate(cmd[:-1]):
        if item == token:
            cmd[idx + 1] = replacement
            return
    cmd.extend([token, replacement])


def _append_resume_args(cmd: list[str], resume_run_dir: Path) -> None:
    cmd.extend(["--resume-run-dir", str(resume_run_dir), "--resume-mode", "strict"])


def _run_command(command: list[str], *, cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def _infer_replay_start_date(source_formal_run_dir: Path) -> str:
    tail = source_formal_run_dir.name.rsplit("_", 2)
    if len(tail) >= 3 and tail[-2].isdigit():
        return tail[-2]
    return "20250318"


def _build_resume_command(
    *,
    python_executable: str,
    resume_run_dir: Path,
    family_epoch_budget_manifest: str,
    epoch_budget: int,
    latest_completed_date: str,
    latest_trainable_date: str,
    internal_monitor_start_date: str,
    internal_monitor_days: int,
    experiment_tag: str,
) -> list[str]:
    cmd = _build_retrain_command(
        source_run_dir=resume_run_dir,
        family_epoch_budget_manifest=family_epoch_budget_manifest,
        latest_completed_date=latest_completed_date,
        latest_trainable_date=latest_trainable_date,
        internal_monitor_start_date=internal_monitor_start_date,
        internal_monitor_days=internal_monitor_days,
        experiment_tag=experiment_tag,
    )
    cmd[0] = str(python_executable)
    _replace_arg(cmd, "--epochs", epoch_budget)
    _replace_arg(cmd, "--early-stop-patience", max(epoch_budget, 8))
    _append_resume_args(cmd, resume_run_dir)
    return cmd


def _build_replay_command(
    *,
    python_executable: str,
    run_dir: Path,
    replay_output_root: Path,
    candidate_label: str,
    replay_start_date: str,
    replay_end_date: str,
) -> list[str]:
    return [
        str(python_executable),
        str(EXTERNAL_REPLAY_SCRIPT),
        "--data-source",
        "tq",
        "--benchmark",
        "000300.SH",
        "--start-date",
        str(replay_start_date),
        "--end-date",
        str(replay_end_date),
        "--score-panel-csv",
        str(run_dir / "execution_aligned_daily_live_score_panel.csv"),
        "--target-weight-panel-csv",
        str(run_dir / "execution_aligned_daily_live_target_weight_panel.csv"),
        "--candidate-label",
        candidate_label,
        "--rebalance-freq",
        "1d",
        "--rebalance-offset-mode",
        "single",
        "--rebalance-anchor-date",
        str(replay_start_date),
        "--transaction-cost-bps",
        "3",
        "--slippage-bps",
        "7",
        "--sell-tax-bps",
        "10",
        "--no-market-regime-filter",
        "--output-dir",
        str(replay_output_root),
        "--experiment-tag",
        f"recent_replays/{candidate_label}",
    ]


def _load_candidate_row(epoch_budget: int, run_dir: Path, replay_dir: Path) -> dict[str, Any]:
    run_metrics = _load_json(run_dir / "metrics.json")
    replay_metrics = _load_json(replay_dir / "metrics.json")
    monthly_diagnostics = _load_json(replay_dir / "monthly_backtest_diagnostics.json")
    monthly_objectives = summarize_primary_monthly_objectives(monthly_diagnostics)
    training = dict(run_metrics.get("training_diagnostics", {}))
    return {
        "epoch_budget": int(epoch_budget),
        "run_dir": str(run_dir),
        "replay_dir": str(replay_dir),
        "selected_epoch": int(training.get("selected_epoch", 0) or 0),
        "selected_epoch_ratio": float(training.get("selected_epoch_ratio", 0.0) or 0.0),
        "objective_aligned_budget_pressure": bool(training.get("objective_aligned_budget_pressure", False)),
        "annual_return": float(replay_metrics.get("annual_return", float("nan"))),
        "excess_annual_return": float(replay_metrics.get("excess_annual_return", float("nan"))),
        "excess_sharpe": float(replay_metrics.get("excess_sharpe", float("nan"))),
        "max_drawdown": float(replay_metrics.get("max_drawdown", float("nan"))),
        "avg_turnover": float(replay_metrics.get("avg_turnover", float("nan"))),
        "avg_holding_count": float(replay_metrics.get("avg_holding_count", float("nan"))),
        "positive_month_ratio": float(monthly_diagnostics.get("positive_month_ratio", 0.0) or 0.0),
        "median_monthly_return": float(monthly_diagnostics.get("median_monthly_return", 0.0) or 0.0),
        "worst_monthly_return": float(monthly_diagnostics.get("worst_monthly_return", 0.0) or 0.0),
        "top3_positive_month_share": float(monthly_diagnostics.get("top3_positive_month_share", 0.0) or 0.0),
        "longest_negative_streak": int(monthly_diagnostics.get("longest_negative_streak", 0) or 0),
        "monthly_robust_score": float(monthly_objectives.get("primary_monthly_robust_score", 0.0) or 0.0),
        "issue_flags": ",".join(monthly_diagnostics.get("issue_flags", [])),
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _select_best(summary_df: pd.DataFrame) -> pd.Series:
    ordered = summary_df.sort_values(
        [
            "monthly_robust_score",
            "excess_annual_return",
            "excess_sharpe",
            "positive_month_ratio",
            "median_monthly_return",
            "worst_monthly_return",
            "top3_positive_month_share",
            "objective_aligned_budget_pressure",
            "epoch_budget",
        ],
        ascending=[False, False, False, False, False, False, True, True, False],
    ).reset_index(drop=True)
    return ordered.iloc[0]


def _write_report(output_dir: Path, summary_df: pd.DataFrame, best_row: pd.Series, base_epoch_budget: int) -> None:
    lines = [
        "# Short Alpha Production Epoch Extension",
        "",
        f"- base_epoch_budget: `{base_epoch_budget}`",
        f"- best_epoch_budget: `{int(best_row['epoch_budget'])}`",
        "",
        "## Monthly-First Ranking",
    ]
    ordered = summary_df.sort_values(
        [
            "monthly_robust_score",
            "excess_annual_return",
            "excess_sharpe",
            "positive_month_ratio",
            "median_monthly_return",
            "worst_monthly_return",
            "top3_positive_month_share",
        ],
        ascending=[False, False, False, False, False, False, True],
    )
    for _, row in ordered.iterrows():
        lines.append(
            "- "
            f"e{int(row['epoch_budget'])}: "
            f"monthly score {_num(row['monthly_robust_score'])}, "
            f"positive-month ratio {_pct(row['positive_month_ratio'])}, "
            f"median monthly excess {_pct(row['median_monthly_return'])}, "
            f"worst month {_pct(row['worst_monthly_return'])}, "
            f"top3 share {_pct(row['top3_positive_month_share'])}, "
            f"recent replay excess annual {_pct(row['excess_annual_return'])}, "
            f"excess Sharpe {_num(row['excess_sharpe'])}, "
            f"budget pressure `{bool(row['objective_aligned_budget_pressure'])}`"
        )
    lines.extend(
        [
            "",
            "## Direct Answer",
            f"- Current best production recipe is `e{int(best_row['epoch_budget'])}` under monthly-first replay ranking.",
            f"- selected_epoch / budget = `{int(best_row['selected_epoch'])}/{int(best_row['epoch_budget'])}`.",
            f"- budget_pressure = `{bool(best_row['objective_aligned_budget_pressure'])}`.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    production_manifest = _load_json(Path(args.production_manifest))
    production_root = Path(args.production_root).resolve()
    strategy_manifest_path = Path(args.strategy_manifest_path).resolve()
    active_strategy_manifest = _load_json(strategy_manifest_path) if strategy_manifest_path.exists() else {}

    source_formal_run_dir = Path(production_manifest["source_formal_run_dir"]).resolve()
    current_run_dir = Path(production_manifest["active_production_run_dir"]).resolve()
    latest_completed_date = str(
        args.replay_end_date.strip()
        or production_manifest.get("launch_cutoff_date")
        or pd.Timestamp(get_latest_completed_trading_date()).strftime("%Y%m%d")
    )
    latest_trainable_date = str(production_manifest.get("train_end_date", ""))
    internal_monitor_start_date = str(production_manifest.get("internal_monitor_start_date", ""))
    internal_monitor_days = int(production_manifest.get("internal_monitor_days", 3) or 3)
    train_start_date = str(production_manifest.get("train_start_date", "20210101"))
    replay_start_date = str(args.replay_start_date.strip() or _infer_replay_start_date(source_formal_run_dir))
    epoch_budgets = _parse_epoch_budgets(args.epoch_budgets)

    current_metrics = _load_json(current_run_dir / "metrics.json")
    base_epoch_budget = int(current_metrics.get("epochs", 0) or current_metrics.get("training_diagnostics", {}).get("epochs_requested", 0) or 24)
    if base_epoch_budget not in epoch_budgets:
        epoch_budgets.insert(0, base_epoch_budget)
        epoch_budgets = sorted(dict.fromkeys(epoch_budgets))

    output_dir = OUTPUT_ROOT / args.root_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_output_root = output_dir.resolve()

    candidate_runs: dict[int, Path] = {base_epoch_budget: current_run_dir}
    ordered_budgets = sorted(epoch_budgets)
    previous_run_dir = current_run_dir
    previous_budget = base_epoch_budget
    for budget in ordered_budgets:
        if budget <= base_epoch_budget:
            continue
        experiment_tag = f"{args.root_tag}/runs/short_alpha_production_e{budget}"
        cmd = _build_resume_command(
            python_executable=args.python_executable,
            resume_run_dir=previous_run_dir,
            family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
            epoch_budget=budget,
            latest_completed_date=latest_completed_date,
            latest_trainable_date=latest_trainable_date,
            internal_monitor_start_date=internal_monitor_start_date,
            internal_monitor_days=internal_monitor_days,
            experiment_tag=experiment_tag,
        )
        _run_command(cmd, cwd=PROJECT_ROOT)
        run_dir = (OUTPUT_ROOT / args.root_tag / "runs" / f"short_alpha_production_e{budget}").resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Extended production run missing: {run_dir}")
        candidate_runs[budget] = run_dir
        previous_run_dir = run_dir
        previous_budget = budget

    summary_rows: list[dict[str, Any]] = []
    for budget in ordered_budgets:
        run_dir = candidate_runs[budget]
        candidate_label = f"short_alpha_production_e{budget}"
        replay_dir = replay_output_root / "recent_replays" / candidate_label
        if not replay_dir.exists():
            replay_cmd = _build_replay_command(
                python_executable=args.python_executable,
                run_dir=run_dir,
                replay_output_root=replay_output_root,
                candidate_label=candidate_label,
                replay_start_date=replay_start_date,
                replay_end_date=latest_completed_date,
            )
            _run_command(replay_cmd, cwd=PROJECT_ROOT)
        summary_rows.append(_load_candidate_row(budget, run_dir, replay_dir))

    summary_df = pd.DataFrame(summary_rows).sort_values("epoch_budget").reset_index(drop=True)
    best_row = _select_best(summary_df)

    summary_df.to_csv(output_dir / "epoch_extension_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir, summary_df, best_row, base_epoch_budget)

    promoted = False
    if args.activate_best and int(best_row["epoch_budget"]) != base_epoch_budget:
        best_run_dir = Path(str(best_row["run_dir"])).resolve()
        _sync_production_root(
            run_dir=best_run_dir,
            production_root=production_root,
            source_run_dir=source_formal_run_dir,
            latest_completed_date=latest_completed_date,
            latest_trainable_date=latest_trainable_date,
            internal_monitor_start_date=internal_monitor_start_date,
            internal_monitor_days=internal_monitor_days,
            train_start_date=train_start_date,
            strategy_manifest_path=strategy_manifest_path,
            activate_strategy=True,
        )
        strategy_name = str(active_strategy_manifest.get("strategy_name", "") or "state_liquidity_listwise_v1_execfirst_winner").strip()
        strategy_panel_mode = str(active_strategy_manifest.get("panel_mode", "") or "auto").strip()
        if strategy_panel_mode not in {"auto", "raw", "execution_aligned"}:
            strategy_panel_mode = "auto"
        _activate_strategy(
            source_run_dir=source_formal_run_dir,
            production_root=production_root,
            strategy_manifest_path=strategy_manifest_path,
            strategy_name=strategy_name,
            strategy_panel_mode=strategy_panel_mode,
        )
        promoted = True
        trade_plan_cmd = [str(args.python_executable), str(RUN_TRADE_PLAN_SCRIPT)]
        _run_command(trade_plan_cmd, cwd=PROJECT_ROOT)

    console_summary = {
        "output_dir": str(output_dir),
        "base_epoch_budget": base_epoch_budget,
        "tested_epoch_budgets": ordered_budgets,
        "best_epoch_budget": int(best_row["epoch_budget"]),
        "best_monthly_robust_score": float(best_row["monthly_robust_score"]),
        "best_excess_annual_return": float(best_row["excess_annual_return"]),
        "best_positive_month_ratio": float(best_row["positive_month_ratio"]),
        "best_budget_pressure": bool(best_row["objective_aligned_budget_pressure"]),
        "promoted": promoted,
    }
    print(json.dumps(console_summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
