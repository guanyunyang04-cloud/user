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

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.research_objective import resolve_primary_backtest, summarize_primary_monthly_objectives
from daily_research.deep_alpha.run_architecture_protocol_refresh import (
    FORMAL_WINDOWS,
    OUTPUT_ROOT,
    RECENT_WINDOW,
    _build_research_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_BY_LABEL = {window.label: window for window in FORMAL_WINDOWS}
DEFAULT_SOURCE_ROOT_TAG = "deep_alpha_architecture_protocol_refresh_20260406_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend graph_off_plain on the budget-pressured old formal windows with strict resume, "
            "refresh the three-window summary, and decide whether this branch deserves the liquid500 challenger gate."
        )
    )
    parser.add_argument("--source-root-tag", default=DEFAULT_SOURCE_ROOT_TAG)
    parser.add_argument("--root-tag", default="graph_off_plain_budget_review_20260406_r1")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--epoch-budgets", default="32,48,64")
    parser.add_argument("--window-labels", default="20230216_20240229,20240301_20250317")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--end-date", default="20260403")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--liquidity-pool", default="liquid500")
    parser.add_argument("--research-objective-mode", default="execution_first")
    parser.add_argument("--checkpoint-selection-objective", default="primary_annual_return")
    parser.add_argument("--checkpoint-selection-min-improvement", type=float, default=0.0001)
    parser.add_argument("--execution-alignment-objective", default="robust_composite")
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument(
        "--family-epoch-budget-manifest",
        default=str(OUTPUT_ROOT / "deep_alpha_family_epoch_budget_latest.json"),
    )
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Epoch budget must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one epoch budget is required.")
    return sorted(dict.fromkeys(values))


def _parse_name_list(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not values:
        raise ValueError("At least one window label is required.")
    return values


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _replace_arg(command: list[str], flag: str, value: Any) -> None:
    token = str(flag)
    replacement = str(value)
    for idx, item in enumerate(command[:-1]):
        if item == token:
            command[idx + 1] = replacement
            return
    command.extend([token, replacement])


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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


def _source_run_dir(source_root_tag: str, profile_name: str, window_label: str) -> Path:
    if window_label == RECENT_WINDOW.label:
        if profile_name == "baseline_current":
            return OUTPUT_ROOT / f"{source_root_tag}_recent"
        return OUTPUT_ROOT / f"{source_root_tag}_recent__{profile_name}"
    if profile_name == "baseline_current":
        return OUTPUT_ROOT / f"{source_root_tag}_formal__baseline_current_{window_label}"
    return OUTPUT_ROOT / f"{source_root_tag}_formal__{profile_name}_{window_label}"


def _load_run_row(run_dir: Path, *, window_label: str, epoch_budget: int, row_kind: str) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    training = dict(metrics.get("training_diagnostics", {}))
    _, primary_backtest = resolve_primary_backtest(metrics, research_objective_mode="execution_first")
    monthly_diag = metrics.get("primary_research_monthly_diagnostics")
    if not isinstance(monthly_diag, dict) or not monthly_diag:
        monthly_diag = metrics.get("execution_aligned_monthly_backtest_diagnostics")
    monthly_objectives = summarize_primary_monthly_objectives(monthly_diag if isinstance(monthly_diag, dict) else {})
    return {
        "window_label": window_label,
        "row_kind": row_kind,
        "run_dir": str(run_dir),
        "epoch_budget": int(epoch_budget),
        "annual_return": float(primary_backtest.get("annual_return", float("nan"))),
        "excess_annual_return": float(primary_backtest.get("excess_annual_return", float("nan"))),
        "excess_sharpe": float(primary_backtest.get("excess_sharpe", float("nan"))),
        "avg_turnover": float(primary_backtest.get("avg_turnover", float("nan"))),
        "positive_month_ratio": float(monthly_objectives.get("primary_monthly_positive_ratio", float("nan"))),
        "median_monthly_return": float(monthly_objectives.get("primary_monthly_median_return", float("nan"))),
        "worst_monthly_return": float(monthly_objectives.get("primary_monthly_worst_return", float("nan"))),
        "top3_positive_month_share": float(monthly_objectives.get("primary_monthly_top3_positive_share", float("nan"))),
        "longest_negative_streak": int(monthly_objectives.get("primary_monthly_longest_negative_streak", 0) or 0),
        "monthly_robust_score": float(monthly_objectives.get("primary_monthly_robust_score", float("nan"))),
        "execution_alignment_profile": str(metrics.get("execution_alignment_profile", "") or "").strip(),
        "selected_epoch": int(training.get("selected_epoch", 0) or 0),
        "selected_epoch_ratio": float(training.get("selected_epoch_ratio", 0.0) or 0.0),
        "objective_aligned_budget_pressure": bool(training.get("objective_aligned_budget_pressure", False)),
        "selected_at_right_boundary": bool(training.get("selected_at_right_boundary", False)),
        "still_improving": bool(training.get("still_improving", False)),
    }


def _rank_budget_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "monthly_robust_score",
            "excess_annual_return",
            "excess_sharpe",
            "positive_month_ratio",
            "median_monthly_return",
            "worst_monthly_return",
            "objective_aligned_budget_pressure",
            "epoch_budget",
        ],
        ascending=[False, False, False, False, False, False, True, True],
    ).reset_index(drop=True)


def _write_report(
    output_dir: Path,
    *,
    window_budget_summary: pd.DataFrame,
    selected_windows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    refreshed_summary: pd.DataFrame,
) -> None:
    graph_mean_annual = float(refreshed_summary.loc[refreshed_summary["profile_name"].eq("graph_off_plain"), "mean_excess_annual_return"].iloc[0])
    graph_mean_sharpe = float(refreshed_summary.loc[refreshed_summary["profile_name"].eq("graph_off_plain"), "mean_excess_sharpe"].iloc[0])
    baseline_mean_annual = float(refreshed_summary.loc[refreshed_summary["profile_name"].eq("baseline_current"), "mean_excess_annual_return"].iloc[0])
    baseline_mean_sharpe = float(refreshed_summary.loc[refreshed_summary["profile_name"].eq("baseline_current"), "mean_excess_sharpe"].iloc[0])
    annual_wins = int((selected_windows["excess_annual_return"].values > baseline_rows["excess_annual_return"].values).sum())
    sharpe_wins = int((selected_windows["excess_sharpe"].values > baseline_rows["excess_sharpe"].values).sum())
    gate_pass = bool(
        (not selected_windows["objective_aligned_budget_pressure"].any())
        and graph_mean_annual > baseline_mean_annual
        and graph_mean_sharpe > baseline_mean_sharpe
        and annual_wins >= 2
        and sharpe_wins >= 2
    )

    lines = [
        "# graph_off_plain Budget Review",
        "",
        "## Old-Window Frontier",
    ]
    for window_label, frame in window_budget_summary.groupby("window_label", sort=False):
        ordered = _rank_budget_rows(frame)
        best_row = ordered.iloc[0]
        lines.append(f"- `{window_label}` best budget `e{int(best_row['epoch_budget'])}`: monthly score `{_num(best_row['monthly_robust_score'])}`, excess annual `{_pct(best_row['excess_annual_return'])}`, excess Sharpe `{_num(best_row['excess_sharpe'])}`, budget pressure `{bool(best_row['objective_aligned_budget_pressure'])}`")
    lines.extend(
        [
            "",
            "## Refreshed Three-Window Summary",
            f"- graph_off_plain mean excess annual / Sharpe: `{_pct(graph_mean_annual)} / {_num(graph_mean_sharpe)}`",
            f"- baseline_current mean excess annual / Sharpe: `{_pct(baseline_mean_annual)} / {_num(baseline_mean_sharpe)}`",
            f"- annual wins vs baseline: `{annual_wins}/3`",
            f"- Sharpe wins vs baseline: `{sharpe_wins}/3`",
            "",
            "## Direct Answer",
        ]
    )
    if gate_pass:
        lines.append("- graph_off_plain clears the liquid500 challenger gate after budget completion; it is no longer just a budget-pressure branch.")
    else:
        lines.append("- graph_off_plain still does not clear the liquid500 challenger gate after budget completion; keep it as a monitored architecture branch, not a default-upgrade candidate.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = _parse_int_list(args.epoch_budgets)
    window_labels = _parse_name_list(args.window_labels)
    output_dir = OUTPUT_ROOT / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    budget_rows: list[dict[str, Any]] = []
    selected_window_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for window_label in window_labels:
        if window_label not in WINDOW_BY_LABEL:
            raise KeyError(f"Unknown formal window label: {window_label}")
        window = WINDOW_BY_LABEL[window_label]
        source_run_dir = _source_run_dir(str(args.source_root_tag), "graph_off_plain", window_label).resolve()
        if not source_run_dir.exists():
            raise FileNotFoundError(f"Source graph_off_plain run missing: {source_run_dir}")
        base_metrics = _load_json(source_run_dir / "metrics.json")
        base_budget = int(base_metrics.get("epochs", 0) or base_metrics.get("training_diagnostics", {}).get("epochs_requested", 0) or budgets[0])
        if base_budget not in budgets:
            budgets = sorted(dict.fromkeys([base_budget, *budgets]))
        previous_run_dir = source_run_dir
        budget_rows.append(_load_run_row(source_run_dir, window_label=window_label, epoch_budget=base_budget, row_kind="source"))

        for budget in budgets:
            if budget <= base_budget:
                continue
            experiment_tag = f"{args.root_tag}/runs/graph_off_plain_{window_label}_e{budget}"
            run_dir = OUTPUT_ROOT / args.root_tag / "runs" / f"graph_off_plain_{window_label}_e{budget}"
            if not (run_dir / "metrics.json").exists() or args.force_rerun:
                command = _build_research_command(
                    python_executable=str(args.python_executable),
                    experiment_tag=experiment_tag,
                    profile_name="graph_off_plain",
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
                _replace_arg(command, "--epochs", budget)
                _replace_arg(command, "--early-stop-patience", max(int(budget), 8))
                command.extend(["--resume-run-dir", str(previous_run_dir), "--resume-mode", "strict"])
                _run_command(command)
            previous_run_dir = run_dir.resolve()
            budget_rows.append(_load_run_row(previous_run_dir, window_label=window_label, epoch_budget=budget, row_kind="extended"))

        window_frame = pd.DataFrame([row for row in budget_rows if row["window_label"] == window_label and row["row_kind"] in {"source", "extended"}])
        best_row = _rank_budget_rows(window_frame).iloc[0].to_dict()
        selected_window_rows.append(best_row)

        baseline_run_dir = _source_run_dir(str(args.source_root_tag), "baseline_current", window_label).resolve()
        if not baseline_run_dir.exists():
            raise FileNotFoundError(f"Baseline source run missing: {baseline_run_dir}")
        baseline_rows.append(_load_run_row(baseline_run_dir, window_label=window_label, epoch_budget=int(_load_json(baseline_run_dir / 'metrics.json').get('epochs', 0) or 0), row_kind="baseline"))

    recent_graph_run_dir = _source_run_dir(str(args.source_root_tag), "graph_off_plain", RECENT_WINDOW.label).resolve()
    recent_baseline_run_dir = _source_run_dir(str(args.source_root_tag), "baseline_current", RECENT_WINDOW.label).resolve()
    selected_window_rows.append(_load_run_row(recent_graph_run_dir, window_label=RECENT_WINDOW.label, epoch_budget=int(_load_json(recent_graph_run_dir / "metrics.json").get("epochs", 0) or 0), row_kind="recent_existing"))
    baseline_rows.append(_load_run_row(recent_baseline_run_dir, window_label=RECENT_WINDOW.label, epoch_budget=int(_load_json(recent_baseline_run_dir / "metrics.json").get("epochs", 0) or 0), row_kind="baseline"))

    window_budget_summary = pd.DataFrame(budget_rows).sort_values(["window_label", "epoch_budget"]).reset_index(drop=True)
    selected_windows = pd.DataFrame(selected_window_rows).sort_values("window_label").reset_index(drop=True)
    baseline_frame = pd.DataFrame(baseline_rows).sort_values("window_label").reset_index(drop=True)

    refreshed_summary = pd.DataFrame(
        [
            {
                "profile_name": "graph_off_plain",
                "mean_excess_annual_return": float(selected_windows["excess_annual_return"].mean()),
                "mean_excess_sharpe": float(selected_windows["excess_sharpe"].mean()),
                "mean_positive_month_ratio": float(selected_windows["positive_month_ratio"].mean()),
                "mean_median_monthly_return": float(selected_windows["median_monthly_return"].mean()),
                "worst_monthly_return": float(selected_windows["worst_monthly_return"].min()),
                "budget_pressure_windows": int(selected_windows["objective_aligned_budget_pressure"].sum()),
            },
            {
                "profile_name": "baseline_current",
                "mean_excess_annual_return": float(baseline_frame["excess_annual_return"].mean()),
                "mean_excess_sharpe": float(baseline_frame["excess_sharpe"].mean()),
                "mean_positive_month_ratio": float(baseline_frame["positive_month_ratio"].mean()),
                "mean_median_monthly_return": float(baseline_frame["median_monthly_return"].mean()),
                "worst_monthly_return": float(baseline_frame["worst_monthly_return"].min()),
                "budget_pressure_windows": int(baseline_frame["objective_aligned_budget_pressure"].sum()),
            },
        ]
    )

    window_budget_summary.to_csv(output_dir / "window_budget_summary.csv", index=False, encoding="utf-8-sig")
    selected_windows.to_csv(output_dir / "selected_window_summary.csv", index=False, encoding="utf-8-sig")
    baseline_frame.to_csv(output_dir / "baseline_window_summary.csv", index=False, encoding="utf-8-sig")
    refreshed_summary.to_csv(output_dir / "refreshed_three_window_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(
        output_dir,
        window_budget_summary=window_budget_summary,
        selected_windows=selected_windows,
        baseline_rows=baseline_frame,
        refreshed_summary=refreshed_summary,
    )
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
