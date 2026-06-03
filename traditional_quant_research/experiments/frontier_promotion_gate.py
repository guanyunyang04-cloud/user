"""Promotion gate for current frontier strategies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_COMBINED_OUTPUT_ROOT = Path(
    "traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit"
)
DEFAULT_SIZE_AUDIT_ROOT = Path("traditional_quant_research/output/experiments/v2_daily_size_audit")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_promotion_gate")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_frontier_promotion_gate.md")

DEFAULT_REQUIRED_IMPACT_BPS = 10.0
DEFAULT_REQUIRED_FEE_BPS = 30.0
DEFAULT_MIN_EVAL_YEAR_COUNT = 3
DEFAULT_MIN_TOTAL_PERIODS = 24
DEFAULT_MIN_MEAN_ANNUALIZED_RETURN = 0.0
DEFAULT_MIN_ANNUALIZED_RETURN = 0.0
DEFAULT_MIN_POSITIVE_YEAR_RATE = 1.0
DEFAULT_MAX_WORST_DRAWDOWN = -0.25
DEFAULT_MAX_MONTHLY_MEAN_ABS_ACTIVE_EXPOSURE = 0.50
DEFAULT_EXPOSURE_FIELDS = (
    "log_amount_mean_20d_z",
    "neg_volatility_20d_z",
    "momentum_20d_z",
    "turn_xsec_z",
)


def run_frontier_promotion_gate(
    *,
    combined_run_dir: str | Path | None = None,
    size_audit_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    required_impact_bps: float = DEFAULT_REQUIRED_IMPACT_BPS,
    required_fee_bps: float = DEFAULT_REQUIRED_FEE_BPS,
    min_eval_year_count: int = DEFAULT_MIN_EVAL_YEAR_COUNT,
    min_total_periods: int = DEFAULT_MIN_TOTAL_PERIODS,
    max_monthly_mean_abs_active_exposure: float = DEFAULT_MAX_MONTHLY_MEAN_ABS_ACTIVE_EXPOSURE,
    exposure_fields: Sequence[str] = DEFAULT_EXPOSURE_FIELDS,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    run_id = f"frontier_promotion_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_dir = Path(combined_run_dir) if combined_run_dir is not None else latest_run_dir(DEFAULT_COMBINED_OUTPUT_ROOT)
    size_dir = Path(size_audit_run_dir) if size_audit_run_dir is not None else latest_run_dir(DEFAULT_SIZE_AUDIT_ROOT)
    aggregate, exposure_summary = read_combined_constraint_evidence(combined_dir)
    size_summary = read_size_audit_summary(size_dir)
    gate = evaluate_promotion_gates(
        aggregate,
        exposure_summary,
        size_summary=size_summary,
        required_impact_bps=required_impact_bps,
        required_fee_bps=required_fee_bps,
        min_eval_year_count=min_eval_year_count,
        min_total_periods=min_total_periods,
        max_monthly_mean_abs_active_exposure=max_monthly_mean_abs_active_exposure,
        exposure_fields=exposure_fields,
    )
    summary = summarize_promotion_gate(
        gate,
        run_id=run_id,
        combined_run_dir=combined_dir,
        size_audit_run_dir=size_dir,
        size_summary=size_summary,
        required_impact_bps=required_impact_bps,
        required_fee_bps=required_fee_bps,
        min_total_periods=min_total_periods,
        max_monthly_mean_abs_active_exposure=max_monthly_mean_abs_active_exposure,
    )
    markdown = render_promotion_gate_markdown(summary, gate)

    gate.to_csv(run_dir / "promotion_gate_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def evaluate_promotion_gates(
    aggregate: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    *,
    size_summary: Mapping[str, Any],
    required_impact_bps: float,
    required_fee_bps: float,
    min_eval_year_count: int,
    min_total_periods: int,
    max_monthly_mean_abs_active_exposure: float,
    exposure_fields: Sequence[str],
) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame(columns=_gate_columns())
    frame = aggregate.copy()
    for column in [
        "impact_bps_per_1pct",
        "fee_bps",
        "eval_year_count",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "exposure_penalty_strength",
        "constraint_fallback_count",
        "constraint_fallback_rate",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "constraint_variant" not in frame.columns:
        frame["constraint_variant"] = "baseline"
    if "constraint_fallback_count" not in frame.columns:
        frame["constraint_fallback_count"] = 0
    if "constraint_fallback_rate" not in frame.columns:
        frame["constraint_fallback_rate"] = 0.0
    frame = frame.loc[
        np.isclose(frame.get("impact_bps_per_1pct", np.nan), required_impact_bps)
        & np.isclose(frame.get("fee_bps", np.nan), required_fee_bps)
    ].copy()
    rows: list[dict[str, Any]] = []
    size_gate = bool(size_summary.get("daily_size_ready_for_research", False))
    for row in frame.to_dict("records"):
        signal = str(row.get("signal", ""))
        constraint_variant = str(row.get("constraint_variant", "baseline") or "baseline")
        strength = float(row.get("exposure_penalty_strength", 0.0) or 0.0)
        fallback_count = int(float(row.get("constraint_fallback_count", 0) or 0))
        fallback_rate = float(row.get("constraint_fallback_rate", 0.0) or 0.0)
        exposure_gate, max_abs_exposure, exposure_failures = _style_exposure_gate(
            exposure_summary,
            signal=signal,
            constraint_variant=constraint_variant,
            exposure_penalty_strength=strength,
            exposure_fields=exposure_fields,
            max_monthly_mean_abs_active_exposure=max_monthly_mean_abs_active_exposure,
        )
        if fallback_count > 0 or fallback_rate > 0:
            exposure_gate = False
            if "optimizer_fallback" not in exposure_failures:
                exposure_failures.append("optimizer_fallback")
        checks = {
            "size_gate": size_gate,
            "return_gate": float(row.get("mean_annualized_return", np.nan)) >= DEFAULT_MIN_MEAN_ANNUALIZED_RETURN
            and float(row.get("min_annualized_return", np.nan)) >= DEFAULT_MIN_ANNUALIZED_RETURN,
            "year_gate": int(float(row.get("eval_year_count", 0) or 0)) >= min_eval_year_count
            and float(row.get("positive_year_rate", 0.0) or 0.0) >= DEFAULT_MIN_POSITIVE_YEAR_RATE,
            "sample_gate": int(float(row.get("total_periods", 0) or 0)) >= min_total_periods,
            "drawdown_gate": float(row.get("worst_max_drawdown", np.nan)) >= DEFAULT_MAX_WORST_DRAWDOWN,
            "style_exposure_gate": exposure_gate,
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "constraint_variant": constraint_variant,
                "signal": signal,
                "impact_bps_per_1pct": float(row.get("impact_bps_per_1pct", np.nan)),
                "fee_bps": float(row.get("fee_bps", np.nan)),
                "exposure_penalty_strength": strength,
                "constraint_fallback_count": fallback_count,
                "constraint_fallback_rate": fallback_rate,
                "mean_annualized_return": float(row.get("mean_annualized_return", np.nan)),
                "min_annualized_return": float(row.get("min_annualized_return", np.nan)),
                "positive_year_rate": float(row.get("positive_year_rate", np.nan)),
                "worst_max_drawdown": float(row.get("worst_max_drawdown", np.nan)),
                "total_periods": int(float(row.get("total_periods", 0) or 0)),
                "max_monthly_mean_abs_active_exposure": max_abs_exposure,
                "size_gate": checks["size_gate"],
                "return_gate": checks["return_gate"],
                "year_gate": checks["year_gate"],
                "sample_gate": checks["sample_gate"],
                "drawdown_gate": checks["drawdown_gate"],
                "style_exposure_gate": checks["style_exposure_gate"],
                "promoted": not failed,
                "failed_gates": ",".join(failed),
                "exposure_failures": ",".join(exposure_failures),
                "promotion_level": "strategy_candidate" if not failed else "candidate-frontier/backtest_only",
            }
        )
    return pd.DataFrame(rows, columns=_gate_columns())


def summarize_promotion_gate(
    gate: pd.DataFrame,
    *,
    run_id: str,
    combined_run_dir: Path,
    size_audit_run_dir: Path,
    size_summary: Mapping[str, Any],
    required_impact_bps: float,
    required_fee_bps: float,
    min_total_periods: int,
    max_monthly_mean_abs_active_exposure: float,
) -> dict[str, Any]:
    promoted = gate.loc[gate["promoted"]] if not gate.empty else pd.DataFrame()
    fail_counter: Counter[str] = Counter()
    if not gate.empty:
        for item in gate["failed_gates"].dropna().astype(str):
            fail_counter.update(part for part in item.split(",") if part)
    best_row = {}
    if not gate.empty:
        best_row = gate.sort_values("mean_annualized_return", ascending=False).iloc[0].to_dict()
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "combined_run_dir": str(combined_run_dir),
        "size_audit_run_dir": str(size_audit_run_dir),
        "required_impact_bps": required_impact_bps,
        "required_fee_bps": required_fee_bps,
        "min_total_periods": min_total_periods,
        "max_monthly_mean_abs_active_exposure": max_monthly_mean_abs_active_exposure,
        "daily_size_ready_for_research": bool(size_summary.get("daily_size_ready_for_research", False)),
        "daily_size_status": str(size_summary.get("status", "")),
        "evaluated_rows": int(len(gate)),
        "candidate_count": int(len(promoted)),
        "decision": "promote_strategy_candidate" if len(promoted) else "keep_candidate_frontier_backtest_only",
        "best_signal_by_return": str(best_row.get("signal", "")),
        "best_signal_mean_annualized_return": float(best_row.get("mean_annualized_return", np.nan))
        if best_row
        else None,
        "top_failed_gates": dict(fail_counter.most_common()),
        "limitations": [
            "Promotion gate uses current structured audit outputs; it does not rerun backtests.",
            "Passing this gate would still require independent review before production deployment.",
            "Current true size control depends on daily_size_ready_for_research.",
        ],
    }


def render_promotion_gate_markdown(summary: Mapping[str, Any], gate: pd.DataFrame) -> str:
    lines = [
        "# Frontier Promotion Gate",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- daily_size_status: `{summary.get('daily_size_status', '')}`",
        f"- daily_size_ready_for_research: `{summary.get('daily_size_ready_for_research')}`",
        f"- required_impact_bps: `{summary.get('required_impact_bps')}`",
        f"- min_total_periods: `{summary.get('min_total_periods')}`",
        f"- max_monthly_mean_abs_active_exposure: `{summary.get('max_monthly_mean_abs_active_exposure')}`",
        f"- best_signal_by_return: `{summary.get('best_signal_by_return', '')}`",
        f"- best_signal_mean_annualized_return: `{_fmt(summary.get('best_signal_mean_annualized_return'))}`",
        "",
        "## Failed Gates",
        "",
    ]
    failed = summary.get("top_failed_gates", {})
    if failed:
        for key, value in failed.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Gate Summary", "", _markdown_table(gate), "", "## Interpretation", ""])
    lines.append(
        "This gate is a promotion decision layer. Positive constrained returns alone are insufficient: "
        "the frontier must also have enough independent periods, execution/cost evidence, true size readiness, "
        "and controlled active exposures. Rows that fail remain `candidate-frontier/backtest_only`."
    )
    lines.append("")
    return "\n".join(lines)


def read_combined_constraint_evidence(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_path = run_dir / "combined_constraint_aggregate.csv"
    exposure_path = run_dir / "combined_constraint_basket_exposure_summary.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"aggregate not found: {aggregate_path}")
    if not exposure_path.exists():
        raise FileNotFoundError(f"basket exposure summary not found: {exposure_path}")
    return pd.read_csv(aggregate_path), pd.read_csv(exposure_path)


def read_size_audit_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"size audit summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"run root not found: {root}")
    dirs = [item for item in root.iterdir() if item.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no run directories found under: {root}")
    return max(dirs, key=lambda item: item.stat().st_mtime)


def _style_exposure_gate(
    exposure_summary: pd.DataFrame,
    *,
    signal: str,
    constraint_variant: str,
    exposure_penalty_strength: float,
    exposure_fields: Sequence[str],
    max_monthly_mean_abs_active_exposure: float,
) -> tuple[bool, float, list[str]]:
    if exposure_summary.empty:
        return False, np.nan, list(exposure_fields)
    frame = exposure_summary.copy()
    if "constraint_variant" not in frame.columns:
        frame["constraint_variant"] = "baseline"
    for column in ["exposure_penalty_strength", "mean_abs_active_exposure"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    subset = frame.loc[
        (frame["signal"].astype(str) == signal)
        & (frame["constraint_variant"].astype(str) == constraint_variant)
        & np.isclose(frame["exposure_penalty_strength"], exposure_penalty_strength)
        & (frame["period_type"].astype(str) == "monthly")
        & (frame["factor"].astype(str).isin(set(exposure_fields)))
    ].copy()
    if subset.empty:
        return False, np.nan, list(exposure_fields)
    max_abs = float(subset["mean_abs_active_exposure"].max())
    failures = (
        subset.loc[subset["mean_abs_active_exposure"] > max_monthly_mean_abs_active_exposure, "factor"]
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return len(failures) == 0, max_abs, failures


def _gate_columns() -> list[str]:
    return [
        "constraint_variant",
        "signal",
        "impact_bps_per_1pct",
        "fee_bps",
        "exposure_penalty_strength",
        "constraint_fallback_count",
        "constraint_fallback_rate",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "max_monthly_mean_abs_active_exposure",
        "size_gate",
        "return_gate",
        "year_gate",
        "sample_gate",
        "drawdown_gate",
        "style_exposure_gate",
        "promoted",
        "failed_gates",
        "exposure_failures",
        "promotion_level",
    ]


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-run-dir", default=None)
    parser.add_argument("--size-audit-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--required-impact-bps", type=float, default=DEFAULT_REQUIRED_IMPACT_BPS)
    parser.add_argument("--required-fee-bps", type=float, default=DEFAULT_REQUIRED_FEE_BPS)
    parser.add_argument("--min-total-periods", type=int, default=DEFAULT_MIN_TOTAL_PERIODS)
    parser.add_argument(
        "--max-monthly-mean-abs-active-exposure",
        type=float,
        default=DEFAULT_MAX_MONTHLY_MEAN_ABS_ACTIVE_EXPOSURE,
    )
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_promotion_gate(
        combined_run_dir=args.combined_run_dir,
        size_audit_run_dir=args.size_audit_run_dir,
        output_dir=args.output_dir,
        required_impact_bps=args.required_impact_bps,
        required_fee_bps=args.required_fee_bps,
        min_total_periods=args.min_total_periods,
        max_monthly_mean_abs_active_exposure=args.max_monthly_mean_abs_active_exposure,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
