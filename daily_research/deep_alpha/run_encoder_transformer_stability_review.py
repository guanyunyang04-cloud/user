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
DEFAULT_EXTEND_WINDOW = "20240301_20250317"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review encoder_transformer_v1 stability by extending its budget-pressured weak window, "
            "then re-evaluating the refreshed three-window monthly diagnostics."
        )
    )
    parser.add_argument("--source-root-tag", default=DEFAULT_SOURCE_ROOT_TAG)
    parser.add_argument("--root-tag", default="encoder_transformer_stability_review_20260406_r1")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--extend-window-label", default=DEFAULT_EXTEND_WINDOW)
    parser.add_argument("--epoch-budgets", default="32,48,64")
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
    budget_summary: pd.DataFrame,
    encoder_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
) -> None:
    encoder_mean_annual = float(encoder_rows["excess_annual_return"].mean())
    encoder_mean_sharpe = float(encoder_rows["excess_sharpe"].mean())
    baseline_mean_annual = float(baseline_rows["excess_annual_return"].mean())
    baseline_mean_sharpe = float(baseline_rows["excess_sharpe"].mean())
    encoder_mean_pos = float(encoder_rows["positive_month_ratio"].mean())
    encoder_worst_month = float(encoder_rows["worst_monthly_return"].min())
    encoder_mean_top3 = float(encoder_rows["top3_positive_month_share"].mean())
    annual_wins = int((encoder_rows["excess_annual_return"].values > baseline_rows["excess_annual_return"].values).sum())
    sharpe_wins = int((encoder_rows["excess_sharpe"].values > baseline_rows["excess_sharpe"].values).sum())
    gate_pass = bool(
        (not encoder_rows["objective_aligned_budget_pressure"].any())
        and encoder_mean_annual > baseline_mean_annual
        and encoder_mean_sharpe > baseline_mean_sharpe
        and encoder_mean_pos >= 0.60
        and encoder_worst_month > -0.12
    )

    best_budget_row = _rank_budget_rows(budget_summary).iloc[0]
    lines = [
        "# encoder_transformer_v1 Stability Review",
        "",
        "## Budget-Pressured Window",
        f"- extended_window: `{best_budget_row['window_label']}`",
        f"- best_budget: `e{int(best_budget_row['epoch_budget'])}`",
        f"- best_monthly_score: `{_num(best_budget_row['monthly_robust_score'])}`",
        f"- best_excess_annual_return: `{_pct(best_budget_row['excess_annual_return'])}`",
        f"- best_excess_sharpe: `{_num(best_budget_row['excess_sharpe'])}`",
        f"- budget_pressure: `{bool(best_budget_row['objective_aligned_budget_pressure'])}`",
        "",
        "## Refreshed Three-Window Stability",
        f"- encoder mean excess annual / Sharpe: `{_pct(encoder_mean_annual)} / {_num(encoder_mean_sharpe)}`",
        f"- baseline mean excess annual / Sharpe: `{_pct(baseline_mean_annual)} / {_num(baseline_mean_sharpe)}`",
        f"- encoder mean positive-month ratio: `{_pct(encoder_mean_pos)}`",
        f"- encoder worst month: `{_pct(encoder_worst_month)}`",
        f"- encoder mean top3 positive-month share: `{_pct(encoder_mean_top3)}`",
        f"- annual wins vs baseline: `{annual_wins}/3`",
        f"- Sharpe wins vs baseline: `{sharpe_wins}/3`",
        "",
        "## Direct Answer",
    ]
    if gate_pass:
        lines.append("- encoder_transformer_v1 now clears the liquid500 challenger gate after the stability review.")
    else:
        lines.append("- encoder_transformer_v1 still does not clear the liquid500 challenger gate; recent upside is real, but weak-window stability is not good enough yet.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = _parse_int_list(args.epoch_budgets)
    extend_window_label = str(args.extend_window_label).strip()
    if extend_window_label not in WINDOW_BY_LABEL:
        raise KeyError(f"Unknown formal window label: {extend_window_label}")
    extend_window = WINDOW_BY_LABEL[extend_window_label]
    output_dir = OUTPUT_ROOT / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_run_dir = _source_run_dir(str(args.source_root_tag), "encoder_transformer_v1", extend_window_label).resolve()
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source encoder_transformer_v1 run missing: {source_run_dir}")
    base_metrics = _load_json(source_run_dir / "metrics.json")
    base_budget = int(base_metrics.get("epochs", 0) or base_metrics.get("training_diagnostics", {}).get("epochs_requested", 0) or budgets[0])
    if base_budget not in budgets:
        budgets = sorted(dict.fromkeys([base_budget, *budgets]))

    budget_rows: list[dict[str, Any]] = [
        _load_run_row(source_run_dir, window_label=extend_window_label, epoch_budget=base_budget, row_kind="source")
    ]
    for budget in budgets:
        if budget <= base_budget:
            continue
        experiment_tag = f"{args.root_tag}/runs/encoder_transformer_v1_{extend_window_label}_e{budget}"
        run_dir = OUTPUT_ROOT / args.root_tag / "runs" / f"encoder_transformer_v1_{extend_window_label}_e{budget}"
        if not (run_dir / "metrics.json").exists() or args.force_rerun:
            command = _build_research_command(
                python_executable=str(args.python_executable),
                experiment_tag=experiment_tag,
                profile_name="encoder_transformer_v1",
                window=extend_window,
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
            _run_command(command)
        budget_rows.append(_load_run_row(run_dir.resolve(), window_label=extend_window_label, epoch_budget=budget, row_kind="extended"))

    budget_summary = pd.DataFrame(budget_rows).sort_values("epoch_budget").reset_index(drop=True)
    best_extended_row = _rank_budget_rows(budget_summary).iloc[0].to_dict()

    encoder_rows = [
        _load_run_row(_source_run_dir(str(args.source_root_tag), "encoder_transformer_v1", "20230216_20240229").resolve(), window_label="20230216_20240229", epoch_budget=int(_load_json(_source_run_dir(str(args.source_root_tag), "encoder_transformer_v1", "20230216_20240229").resolve() / "metrics.json").get("epochs", 0) or 0), row_kind="existing"),
        best_extended_row,
        _load_run_row(_source_run_dir(str(args.source_root_tag), "encoder_transformer_v1", RECENT_WINDOW.label).resolve(), window_label=RECENT_WINDOW.label, epoch_budget=int(_load_json(_source_run_dir(str(args.source_root_tag), "encoder_transformer_v1", RECENT_WINDOW.label).resolve() / "metrics.json").get("epochs", 0) or 0), row_kind="existing"),
    ]
    baseline_rows = [
        _load_run_row(_source_run_dir(str(args.source_root_tag), "baseline_current", "20230216_20240229").resolve(), window_label="20230216_20240229", epoch_budget=int(_load_json(_source_run_dir(str(args.source_root_tag), "baseline_current", "20230216_20240229").resolve() / "metrics.json").get("epochs", 0) or 0), row_kind="baseline"),
        _load_run_row(_source_run_dir(str(args.source_root_tag), "baseline_current", extend_window_label).resolve(), window_label=extend_window_label, epoch_budget=int(_load_json(_source_run_dir(str(args.source_root_tag), "baseline_current", extend_window_label).resolve() / "metrics.json").get("epochs", 0) or 0), row_kind="baseline"),
        _load_run_row(_source_run_dir(str(args.source_root_tag), "baseline_current", RECENT_WINDOW.label).resolve(), window_label=RECENT_WINDOW.label, epoch_budget=int(_load_json(_source_run_dir(str(args.source_root_tag), "baseline_current", RECENT_WINDOW.label).resolve() / "metrics.json").get("epochs", 0) or 0), row_kind="baseline"),
    ]

    encoder_frame = pd.DataFrame(encoder_rows).sort_values("window_label").reset_index(drop=True)
    baseline_frame = pd.DataFrame(baseline_rows).sort_values("window_label").reset_index(drop=True)
    budget_summary.to_csv(output_dir / "budget_summary.csv", index=False, encoding="utf-8-sig")
    encoder_frame.to_csv(output_dir / "refreshed_encoder_window_summary.csv", index=False, encoding="utf-8-sig")
    baseline_frame.to_csv(output_dir / "baseline_window_summary.csv", index=False, encoding="utf-8-sig")
    _write_report(output_dir, budget_summary=budget_summary, encoder_rows=encoder_frame, baseline_rows=baseline_frame)
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
