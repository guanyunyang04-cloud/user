from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.execution_alignment import resolve_profile
from daily_research.execution.output_root_resolver import OUTPUT_ROOT, STATIC_PRODUCTION_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"
H2H_SCRIPT = PROJECT_ROOT / "daily_research" / "tools" / "execution_candidate_multiwindow_h2h.py"
DEFAULT_ROOT_TAG = "short_alpha_execution_semantic_concentration_verdict_20260409_r1"
RAW_PANEL_NAME = "daily_live_target_weight_panel.csv"
CAPPED_PANEL_NAME = "portfolio_capped_daily_live_target_weight_panel.csv"
SCORE_PANEL_NAME = "daily_live_score_panel.csv"
CURRENT_STATIC_PROFILE = "regoff_k1_5d_ensemble_native_anchor"


@dataclass(frozen=True)
class Variant:
    name: str
    panel_name: str
    family: str
    description: str
    profile_name: str = ""


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="capped_direct_1d",
        panel_name=CAPPED_PANEL_NAME,
        family="capped_direct",
        description="Legacy capped direct external target-weight panel with no extra bridge.",
    ),
    Variant(
        name="raw_direct_1d",
        panel_name=RAW_PANEL_NAME,
        family="raw_direct",
        description="Research raw no-cap direct external target-weight panel with no extra bridge.",
    ),
    Variant(
        name="raw_regoff_k1_5d_ensemble_native_anchor",
        panel_name=RAW_PANEL_NAME,
        family="raw_controlled",
        description="Current live static fallback profile over the raw no-cap target-weight panel.",
        profile_name="regoff_k1_5d_ensemble_native_anchor",
    ),
    Variant(
        name="raw_regoff_k2_5d_ensemble_native_anchor",
        panel_name=RAW_PANEL_NAME,
        family="raw_controlled",
        description="Raw no-cap target-weight panel bridged to top-k 2 on a 5d all-offset ensemble.",
        profile_name="regoff_k2_5d_ensemble_native_anchor",
    ),
    Variant(
        name="raw_regoff_k3_5d_ensemble_native_anchor",
        panel_name=RAW_PANEL_NAME,
        family="raw_controlled",
        description="Raw no-cap target-weight panel bridged to top-k 3 on a 5d all-offset ensemble.",
        profile_name="regoff_k3_5d_ensemble_native_anchor",
    ),
    Variant(
        name="raw_regoff_k2p15_5d_ensemble_native_anchor",
        panel_name=RAW_PANEL_NAME,
        family="raw_controlled",
        description="Raw no-cap target-weight panel bridged to top-k 2 with power 1.5 on a 5d all-offset ensemble.",
        profile_name="regoff_k2p15_5d_ensemble_native_anchor",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a same-protocol execution verdict for raw no-cap, capped, and a small set of "
            "raw concentration-repair bridge candidates, then summarize the winner."
        )
    )
    parser.add_argument("--production-root", default=str(STATIC_PRODUCTION_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--start-date", default="20250318")
    parser.add_argument("--end-date", default="20260408")
    parser.add_argument("--bridge-start", default="2025-03-18")
    parser.add_argument("--weak-start", default="2025-09-05")
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    return parser.parse_args()


def _load_long_panel(path: Path, value_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["stock"] = raw["stock"].astype(str).str.upper().str.strip()
    raw[value_name] = pd.to_numeric(raw[value_name], errors="coerce")
    raw = raw.dropna(subset=["date", "stock", value_name])
    return (
        raw.sort_values(["date", "stock"])
        .drop_duplicates(subset=["date", "stock"], keep="last")
        .pivot(index="date", columns="stock", values=value_name)
        .sort_index()
        .fillna(0.0)
    )


def _resolve_output_dir(output_root: str, root_tag: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root / root_tag


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def _profile_args(profile_name: str) -> list[str]:
    if not profile_name:
        return []
    profile = resolve_profile(name=profile_name)
    args = [
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
    ]
    if str(profile.rebalance_anchor_date or "").strip():
        args.extend(["--rebalance-anchor-date", str(profile.rebalance_anchor_date)])
    if bool(profile.target_weight_full_invest):
        args.append("--target-weight-full-invest")
    if not bool(profile.use_market_regime_filter):
        args.append("--no-market-regime-filter")
    return args


def _build_backtest_command(
    *,
    python_executable: str,
    score_panel: Path,
    target_weight_panel: Path,
    output_dir: Path,
    variant: Variant,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(python_executable),
        str(BACKTEST_SCRIPT),
        "--score-panel-csv",
        str(score_panel),
        "--target-weight-panel-csv",
        str(target_weight_panel),
        "--data-source",
        "tq",
        "--universe-scope",
        "all_a",
        "--benchmark",
        "000300.SH",
        "--start-date",
        str(args.start_date),
        "--end-date",
        str(args.end_date),
        "--transaction-cost-bps",
        str(float(args.transaction_cost_bps)),
        "--slippage-bps",
        str(float(args.slippage_bps)),
        "--sell-tax-bps",
        str(float(args.sell_tax_bps)),
        "--output-dir",
        str(output_dir),
        "--experiment-tag",
        str(variant.name),
        "--candidate-label",
        str(variant.name),
        "--no-market-regime-filter",
    ]
    command.extend(_profile_args(variant.profile_name))
    return command


def _load_run_metrics(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(run_dir / "monthly_backtest_summary.csv")
    aligned = _load_long_panel(run_dir / "aligned_daily_target_weight_panel.csv", "target_weight")
    return metrics if isinstance(metrics, dict) else {}, monthly, aligned


def _recent_month_value(monthly: pd.DataFrame, month_label: str, column: str) -> float:
    if monthly.empty or column not in monthly.columns:
        return float("nan")
    frame = monthly.loc[monthly["month"].astype(str).eq(month_label), [column]]
    if frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").iloc[-1])


def _concentration_stats(aligned_target_weights: pd.DataFrame) -> dict[str, float]:
    if aligned_target_weights.empty:
        return {
            "mean_positive_name_count": float("nan"),
            "median_positive_name_count": float("nan"),
            "recent20_mean_positive_name_count": float("nan"),
            "mean_max_name_weight": float("nan"),
            "max_max_name_weight": float("nan"),
            "recent20_mean_max_name_weight": float("nan"),
            "recent20_max_name_weight": float("nan"),
            "mean_effective_name_count": float("nan"),
        }
    panel = aligned_target_weights.fillna(0.0).astype(float)
    positive = panel.clip(lower=0.0)
    positive_name_count = positive.gt(0.0).sum(axis=1).astype(float)
    max_name_weight = positive.max(axis=1).astype(float)
    hhi = positive.pow(2).sum(axis=1).astype(float)
    hhi_values = hhi.to_numpy(dtype=float)
    effective_name_count_values = np.full(len(hhi_values), np.nan, dtype=float)
    positive_mask = hhi_values > 0.0
    effective_name_count_values[positive_mask] = 1.0 / hhi_values[positive_mask]
    effective_name_count = pd.Series(
        effective_name_count_values,
        index=positive.index,
        dtype=float,
    )
    recent_slice = slice(-20, None)
    return {
        "mean_positive_name_count": float(positive_name_count.mean()),
        "median_positive_name_count": float(positive_name_count.median()),
        "recent20_mean_positive_name_count": float(positive_name_count.iloc[recent_slice].mean()),
        "mean_max_name_weight": float(max_name_weight.mean()),
        "max_max_name_weight": float(max_name_weight.max()),
        "recent20_mean_max_name_weight": float(max_name_weight.iloc[recent_slice].mean()),
        "recent20_max_name_weight": float(max_name_weight.iloc[recent_slice].max()),
        "mean_effective_name_count": float(effective_name_count.mean()),
    }


def _run_h2h(
    *,
    python_executable: str,
    output_dir: Path,
    run_a: Path,
    label_a: str,
    run_b: Path,
    label_b: str,
    bridge_start: str,
    weak_start: str,
) -> Path:
    h2h_dir = output_dir / f"{label_a}_vs_{label_b}"
    command = [
        str(python_executable),
        str(H2H_SCRIPT),
        "--run-a",
        str(run_a),
        "--label-a",
        str(label_a),
        "--run-b",
        str(run_b),
        "--label-b",
        str(label_b),
        "--bridge-start",
        str(bridge_start),
        "--weak-start",
        str(weak_start),
        "--output-dir",
        str(h2h_dir),
    ]
    _run_command(command)
    return h2h_dir


def _select_best_raw_controlled(summary: pd.DataFrame) -> pd.Series:
    controlled = summary.loc[summary["family"].astype(str).eq("raw_controlled")].copy()
    if controlled.empty:
        raise RuntimeError("No raw_controlled variants were produced.")
    ordered = controlled.sort_values(
        [
            "annual_return",
            "gross_annual_return",
            "excess_annual_return",
            "mean_max_name_weight",
            "recent20_mean_max_name_weight",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    return ordered.iloc[0]


def _should_upgrade(best_raw_controlled: pd.Series, current_live_row: pd.Series) -> bool:
    best_annual = float(best_raw_controlled.get("annual_return", float("-inf")))
    current_annual = float(current_live_row.get("annual_return", float("-inf")))
    best_concentration = float(best_raw_controlled.get("mean_max_name_weight", float("inf")))
    current_concentration = float(current_live_row.get("mean_max_name_weight", float("inf")))
    if np.isnan(best_annual) or np.isnan(current_annual):
        return False
    if best_annual > current_annual and best_concentration <= current_concentration:
        return True
    annual_gap = best_annual - current_annual
    concentration_gain = current_concentration - best_concentration
    return annual_gap >= -0.0025 and concentration_gain >= 0.05


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.2%}"


def _num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def main() -> None:
    args = parse_args()
    production_root = Path(args.production_root).resolve()
    if not production_root.exists():
        raise FileNotFoundError(f"Production root does not exist: {production_root}")
    output_dir = _resolve_output_dir(args.output_root, args.root_tag)
    runs_dir = output_dir / "runs"
    h2h_root = output_dir / "h2h"
    runs_dir.mkdir(parents=True, exist_ok=True)
    h2h_root.mkdir(parents=True, exist_ok=True)

    score_panel = production_root / SCORE_PANEL_NAME
    raw_panel = production_root / RAW_PANEL_NAME
    capped_panel = production_root / CAPPED_PANEL_NAME
    for path in (score_panel, raw_panel, capped_panel):
        if not path.exists():
            raise FileNotFoundError(f"Required production panel is missing: {path}")

    summary_rows: list[dict[str, Any]] = []
    run_dirs: dict[str, Path] = {}
    for variant in DEFAULT_VARIANTS:
        target_weight_panel = production_root / variant.panel_name
        command = _build_backtest_command(
            python_executable=args.python_executable,
            score_panel=score_panel,
            target_weight_panel=target_weight_panel,
            output_dir=runs_dir,
            variant=variant,
            args=args,
        )
        _run_command(command)
        run_dir = runs_dir / variant.name
        run_dirs[variant.name] = run_dir
        metrics, monthly, aligned = _load_run_metrics(run_dir)
        row = {
            "variant_name": variant.name,
            "family": variant.family,
            "source_panel_name": variant.panel_name,
            "profile_name": variant.profile_name,
            "description": variant.description,
            "annual_return": float(metrics.get("annual_return", float("nan"))),
            "gross_annual_return": float(metrics.get("gross_annual_return", float("nan"))),
            "excess_annual_return": float(metrics.get("excess_annual_return", float("nan"))),
            "gross_excess_annual_return": float(metrics.get("gross_excess_annual_return", float("nan"))),
            "excess_sharpe": float(metrics.get("excess_sharpe", float("nan"))),
            "max_drawdown": float(metrics.get("max_drawdown", float("nan"))),
            "avg_turnover": float(metrics.get("avg_turnover", float("nan"))),
            "annual_return_cost_drag": float(metrics.get("annual_return_cost_drag", float("nan"))),
            "total_trading_cost_return": float(metrics.get("total_trading_cost_return", float("nan"))),
            "avg_holding_count_backtest": float(metrics.get("avg_holding_count", float("nan"))),
            "march_excess_return": _recent_month_value(monthly, "2026-03", "excess_return"),
            "march_action_count": _recent_month_value(monthly, "2026-03", "action_count"),
            "march_avg_turnover": _recent_month_value(monthly, "2026-03", "avg_turnover"),
            "april_excess_return": _recent_month_value(monthly, "2026-04", "excess_return"),
            "april_action_count": _recent_month_value(monthly, "2026-04", "action_count"),
            "april_avg_turnover": _recent_month_value(monthly, "2026-04", "avg_turnover"),
            "run_dir": str(run_dir.resolve()),
        }
        row.update(_concentration_stats(aligned))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "annual_return",
            "gross_annual_return",
            "excess_annual_return",
            "mean_max_name_weight",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    summary.to_csv(output_dir / "variant_summary.csv", index=False, encoding="utf-8-sig")

    current_live_row = summary.loc[
        summary["variant_name"].astype(str).eq(f"raw_{CURRENT_STATIC_PROFILE}")
    ]
    if current_live_row.empty:
        raise RuntimeError("Current live static fallback row was not produced.")
    current_live_row = current_live_row.iloc[0]
    best_raw_controlled = _select_best_raw_controlled(summary)
    capped_row = summary.loc[summary["variant_name"].astype(str).eq("capped_direct_1d")].iloc[0]
    raw_direct_row = summary.loc[summary["variant_name"].astype(str).eq("raw_direct_1d")].iloc[0]
    best_overall_row = summary.iloc[0]
    should_upgrade = _should_upgrade(best_raw_controlled, current_live_row)
    recommended_static_fallback_profile = (
        str(best_raw_controlled.get("profile_name", "")).strip()
        if should_upgrade and str(best_raw_controlled.get("profile_name", "")).strip()
        else CURRENT_STATIC_PROFILE
    )

    h2h_outputs = {
        "raw_direct_vs_capped_direct": str(
            _run_h2h(
                python_executable=args.python_executable,
                output_dir=h2h_root / "raw_direct_vs_capped_direct",
                run_a=run_dirs["raw_direct_1d"],
                label_a="raw_direct_1d",
                run_b=run_dirs["capped_direct_1d"],
                label_b="capped_direct_1d",
                bridge_start=args.bridge_start,
                weak_start=args.weak_start,
            )
        ),
        "current_live_vs_best_raw_controlled": str(
            _run_h2h(
                python_executable=args.python_executable,
                output_dir=h2h_root / "current_live_vs_best_raw_controlled",
                run_a=run_dirs[str(current_live_row["variant_name"])],
                label_a=str(current_live_row["variant_name"]),
                run_b=run_dirs[str(best_raw_controlled["variant_name"])],
                label_b=str(best_raw_controlled["variant_name"]),
                bridge_start=args.bridge_start,
                weak_start=args.weak_start,
            )
        ),
        "capped_direct_vs_best_raw_controlled": str(
            _run_h2h(
                python_executable=args.python_executable,
                output_dir=h2h_root / "capped_direct_vs_best_raw_controlled",
                run_a=run_dirs["capped_direct_1d"],
                label_a="capped_direct_1d",
                run_b=run_dirs[str(best_raw_controlled["variant_name"])],
                label_b=str(best_raw_controlled["variant_name"]),
                bridge_start=args.bridge_start,
                weak_start=args.weak_start,
            )
        ),
    }

    summary_payload = {
        "production_root": str(production_root),
        "output_dir": str(output_dir),
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "slippage_bps": float(args.slippage_bps),
        "sell_tax_bps": float(args.sell_tax_bps),
        "best_overall_variant": str(best_overall_row["variant_name"]),
        "best_raw_controlled_variant": str(best_raw_controlled["variant_name"]),
        "current_live_variant": str(current_live_row["variant_name"]),
        "should_upgrade_default_static_fallback": bool(should_upgrade),
        "recommended_static_fallback_profile": recommended_static_fallback_profile,
        "current_static_fallback_profile": CURRENT_STATIC_PROFILE,
        "h2h_outputs": h2h_outputs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Execution Semantic Concentration Verdict",
        "",
        f"- period: `{args.start_date} -> {args.end_date}`",
        f"- costs: `transaction {float(args.transaction_cost_bps):.1f} bps | slippage {float(args.slippage_bps):.1f} bps | sell_tax {float(args.sell_tax_bps):.1f} bps`",
        f"- best_overall_variant: `{best_overall_row['variant_name']}` | annual {_pct(best_overall_row['annual_return'])} | excess annual {_pct(best_overall_row['excess_annual_return'])} | excess Sharpe {_num(best_overall_row['excess_sharpe'])}",
        f"- best_raw_controlled_variant: `{best_raw_controlled['variant_name']}` | annual {_pct(best_raw_controlled['annual_return'])} | excess annual {_pct(best_raw_controlled['excess_annual_return'])} | mean max weight {_pct(best_raw_controlled['mean_max_name_weight'])}",
        f"- current_live_variant: `{current_live_row['variant_name']}` | annual {_pct(current_live_row['annual_return'])} | excess annual {_pct(current_live_row['excess_annual_return'])} | mean max weight {_pct(current_live_row['mean_max_name_weight'])}",
        f"- raw_direct_vs_capped_direct annual delta: `{_pct(float(raw_direct_row['annual_return']) - float(capped_row['annual_return']))}` | gross annual delta `{_pct(float(raw_direct_row['gross_annual_return']) - float(capped_row['gross_annual_return']))}`",
        f"- default_static_fallback_upgrade: `{should_upgrade}`",
        f"- recommended_static_fallback_profile: `{recommended_static_fallback_profile}`",
        "",
        "## Leaderboard",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['variant_name']}` | annual {_pct(row['annual_return'])} | gross annual {_pct(row['gross_annual_return'])} | "
            f"excess annual {_pct(row['excess_annual_return'])} | excess Sharpe {_num(row['excess_sharpe'])} | "
            f"avg turnover {_pct(row['avg_turnover'])} | mean max weight {_pct(row['mean_max_name_weight'])} | "
            f"recent20 mean max weight {_pct(row['recent20_mean_max_name_weight'])}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
