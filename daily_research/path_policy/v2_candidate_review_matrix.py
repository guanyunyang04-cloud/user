from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.execution.freeze_guard import assert_task_allowed
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy import v2_score_backtest_bridge as bridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "daily_research/output/research_data_lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_candidate_review_matrix_20260602_01"
SOURCE_BRIDGE_RUN_TAG = bridge.RUN_TAG
SMALL_CAPITAL_GATE_ID = "small_capital_balanced_return_v1"
SMALL_CAPITAL_RESEARCH_THRESHOLDS = {
    "excess_sharpe_min": 1.2,
    "excess_annual_return_min": 0.25,
    "positive_month_ratio_min": 0.60,
    "catastrophic_worst_month_floor": -0.15,
}


@dataclass(frozen=True)
class Variant:
    variant_id: str
    holding_count: int
    max_weight: float
    rebalance_freq: str
    rebalance_offset_mode: str
    transaction_cost_bps: float
    slippage_bps: float
    sell_tax_bps: float
    use_market_regime_filter: bool


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _parse_ints(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    values = [int(float(item.strip())) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_strings(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _source_bridge_report(source_bridge_root: str | Path) -> dict[str, Any]:
    report_path = Path(source_bridge_root) / "v2_score_backtest_bridge_report.json"
    manifest_path = Path(source_bridge_root) / "v2_score_backtest_bridge_manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if report:
        return report
    if manifest:
        return {"status": manifest.get("status", ""), "bridge": manifest}
    return {}


def _source_score_panel(source_bridge_root: str | Path) -> Path:
    report = _source_bridge_report(source_bridge_root)
    bridge_payload = dict(report.get("bridge", {}) or {})
    path = Path(str(bridge_payload.get("backtest_score_panel_csv") or bridge_payload.get("score_panel_csv") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Source bridge score panel not found: {path}")
    return path


def _variants(
    *,
    holding_counts: tuple[int, ...],
    max_weights: tuple[float, ...],
    rebalance_freqs: tuple[str, ...],
    rebalance_offset_modes: tuple[str, ...],
    transaction_cost_bps_values: tuple[float, ...],
    slippage_bps_values: tuple[float, ...],
    sell_tax_bps_values: tuple[float, ...],
    market_regime_filter_modes: tuple[bool, ...],
) -> list[Variant]:
    out: list[Variant] = []
    for holding_count in holding_counts:
        for max_weight in max_weights:
            for rebalance_freq in rebalance_freqs:
                for offset_mode in rebalance_offset_modes:
                    for transaction_cost_bps in transaction_cost_bps_values:
                        for slippage_bps in slippage_bps_values:
                            for sell_tax_bps in sell_tax_bps_values:
                                for use_regime in market_regime_filter_modes:
                                    suffix = "regime_on" if use_regime else "regime_off"
                                    variant_id = (
                                        f"h{int(holding_count)}_mw{int(round(max_weight * 1000)):03d}_"
                                        f"rb{rebalance_freq}_{offset_mode}_"
                                        f"c{transaction_cost_bps:g}_{slippage_bps:g}_{sell_tax_bps:g}_{suffix}"
                                    ).replace(".", "p")
                                    out.append(
                                        Variant(
                                            variant_id=variant_id,
                                            holding_count=int(holding_count),
                                            max_weight=float(max_weight),
                                            rebalance_freq=str(rebalance_freq),
                                            rebalance_offset_mode=str(offset_mode),
                                            transaction_cost_bps=float(transaction_cost_bps),
                                            slippage_bps=float(slippage_bps),
                                            sell_tax_bps=float(sell_tax_bps),
                                            use_market_regime_filter=bool(use_regime),
                                        )
                                    )
    return out


def _variant_command(
    variant: Variant,
    *,
    score_panel_csv: Path,
    output_root: Path,
    run_tag: str,
    dataset_id: str,
    data_lake_root: str | Path,
    start_date: str,
    end_date: str,
    benchmark: str,
    score_threshold: float,
    no_cache: bool,
) -> list[str]:
    return bridge._backtest_command(
        score_panel_csv=score_panel_csv,
        output_root=output_root,
        experiment_tag=f"{run_tag}_{variant.variant_id}",
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        holding_count=variant.holding_count,
        max_weight=variant.max_weight,
        rebalance_freq=variant.rebalance_freq,
        rebalance_offset_mode=variant.rebalance_offset_mode,
        rebalance_anchor_date=start_date,
        score_threshold=score_threshold,
        transaction_cost_bps=variant.transaction_cost_bps,
        slippage_bps=variant.slippage_bps,
        sell_tax_bps=variant.sell_tax_bps,
        use_market_regime_filter=variant.use_market_regime_filter,
        no_cache=no_cache,
    )


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    assert_task_allowed("candidate-backtest")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    return {"returncode": int(proc.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path)}


def _score_panel_dates(score_panel_csv: Path) -> tuple[str, str]:
    frame = pd.read_csv(score_panel_csv, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"Score panel has no usable dates: {score_panel_csv}")
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _small_capital_gate(
    *,
    metrics: dict[str, Any],
    monthly: dict[str, Any],
    run_result: dict[str, Any] | None,
    artifacts_available: bool,
) -> dict[str, Any]:
    excess_sharpe = _safe_float(metrics.get("excess_sharpe"))
    excess_annual_return = _safe_float(metrics.get("excess_annual_return"))
    annual_return = _safe_float(metrics.get("annual_return"))
    max_drawdown = _safe_float(metrics.get("max_drawdown"))
    positive_month_ratio = _safe_float(monthly.get("positive_month_ratio"))
    worst_month = _safe_float(monthly.get("worst_monthly_excess_return", monthly.get("worst_monthly_return")))
    completed = bool(run_result is None or int(run_result.get("returncode", 1)) == 0) and bool(artifacts_available)
    positive_transfer = bool(completed and excess_sharpe > 0.0 and excess_annual_return > 0.0)
    thresholds = SMALL_CAPITAL_RESEARCH_THRESHOLDS
    checks = {
        "backtest_completed": completed,
        "no_obvious_data_or_execution_issue": completed,
        "not_below_benchmark_with_negative_sharpe": not (excess_annual_return < 0.0 and excess_sharpe < 0.0),
        "positive_transfer": positive_transfer,
        "excess_sharpe_ge_1_2": excess_sharpe >= float(thresholds["excess_sharpe_min"]),
        "excess_annual_return_ge_025": excess_annual_return >= float(thresholds["excess_annual_return_min"]),
        "positive_month_ratio_ge_060": positive_month_ratio >= float(thresholds["positive_month_ratio_min"]),
        "worst_month_not_catastrophic": worst_month >= float(thresholds["catastrophic_worst_month_floor"]),
    }
    drawdown_penalty = max(0.0, abs(min(max_drawdown, 0.0)) - 0.30) * 0.75
    worst_month_penalty = max(0.0, float(thresholds["catastrophic_worst_month_floor"]) - worst_month) * 2.0
    non_transfer_penalty = 0.50 if completed and not positive_transfer else 0.0
    score = (
        2.0 * max(excess_annual_return, 0.0)
        + 0.85 * max(excess_sharpe, 0.0)
        + 0.35 * max(positive_month_ratio, 0.0)
        + 0.25 * max(annual_return, 0.0)
        - drawdown_penalty
        - worst_month_penalty
        - non_transfer_penalty
    )
    research_grade_checks = [
        checks["backtest_completed"],
        checks["no_obvious_data_or_execution_issue"],
        checks["positive_transfer"],
        checks["excess_sharpe_ge_1_2"],
        checks["excess_annual_return_ge_025"],
        checks["positive_month_ratio_ge_060"],
        checks["worst_month_not_catastrophic"],
    ]
    return {
        "gate_id": SMALL_CAPITAL_GATE_ID,
        "checks": checks,
        "positive_transfer": positive_transfer,
        "research_grade_candidate": bool(all(research_grade_checks)),
        "balanced_return_score": float(score),
        "worst_month_metric": {
            "value": worst_month,
            "source": "worst_monthly_excess_return" if "worst_monthly_excess_return" in monthly else "worst_monthly_return",
        },
    }


def _variant_result(
    variant: Variant,
    *,
    command: list[str],
    run_result: dict[str, Any] | None,
) -> dict[str, Any]:
    artifacts = bridge._collect_backtest_artifacts(command)
    metrics = dict(artifacts.get("core_metrics", {}) or {})
    monthly = dict(artifacts.get("monthly_backtest_diagnostics", {}) or {})
    issue_flags = list(monthly.get("issue_flags", []) or [])
    gate_checks = {
        "backtest_completed": bool(run_result is None or run_result.get("returncode") == 0) and bool(artifacts.get("available", False)),
        "monthly_positive_ratio_ge_075": float(monthly.get("positive_month_ratio") or 0.0) >= 0.75,
        "negative_month_count_le_2": int(monthly.get("negative_month_count") or 999) <= 2,
        "max_drawdown_ge_minus_025": float(metrics.get("max_drawdown") or -999.0) >= -0.25,
        "excess_sharpe_ge_1": float(metrics.get("excess_sharpe") or 0.0) >= 1.0,
        "no_deep_bad_month_flag": "deep_bad_month" not in issue_flags,
        "cost_drag_le_020": float(metrics.get("annual_return_cost_drag") or 999.0) <= 0.20,
    }
    small_capital_gate = _small_capital_gate(
        metrics=metrics,
        monthly=monthly,
        run_result=run_result,
        artifacts_available=bool(artifacts.get("available", False)),
    )
    return {
        "variant": variant.__dict__,
        "status": "completed" if all([gate_checks["backtest_completed"]]) else "not_run_or_failed",
        "command": command,
        "run_result": run_result or {"status": "not_requested"},
        "artifacts": artifacts,
        "gate_checks": gate_checks,
        "promotion_review_eligible": bool(all(gate_checks.values())),
        "small_capital_gate": small_capital_gate,
        "small_capital_positive_transfer": bool(small_capital_gate["positive_transfer"]),
        "small_capital_research_grade_candidate": bool(small_capital_gate["research_grade_candidate"]),
        "small_capital_balanced_return_score": float(small_capital_gate["balanced_return_score"]),
    }


def _summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        variant = dict(item.get("variant", {}) or {})
        artifacts = dict(item.get("artifacts", {}) or {})
        metrics = dict(artifacts.get("core_metrics", {}) or {})
        monthly = dict(artifacts.get("monthly_backtest_diagnostics", {}) or {})
        small_capital = dict(item.get("small_capital_gate", {}) or {})
        worst_month_metric = dict(small_capital.get("worst_month_metric", {}) or {})
        rows.append(
            {
                **variant,
                "status": item.get("status", ""),
                "promotion_review_eligible": bool(item.get("promotion_review_eligible", False)),
                "small_capital_gate_id": small_capital.get("gate_id", SMALL_CAPITAL_GATE_ID),
                "small_capital_positive_transfer": bool(item.get("small_capital_positive_transfer", False)),
                "small_capital_research_grade_candidate": bool(item.get("small_capital_research_grade_candidate", False)),
                "small_capital_balanced_return_score": item.get("small_capital_balanced_return_score"),
                "annual_return": metrics.get("annual_return"),
                "excess_annual_return": metrics.get("excess_annual_return"),
                "excess_sharpe": metrics.get("excess_sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "avg_turnover": metrics.get("avg_turnover"),
                "annual_return_cost_drag": metrics.get("annual_return_cost_drag"),
                "positive_month_ratio": monthly.get("positive_month_ratio"),
                "negative_month_count": monthly.get("negative_month_count"),
                "worst_monthly_return": monthly.get("worst_monthly_return"),
                "worst_monthly_excess_return": worst_month_metric.get("value"),
                "worst_month_metric_source": worst_month_metric.get("source", ""),
                "issue_flags": ",".join(str(flag) for flag in list(monthly.get("issue_flags", []) or [])),
            }
        )
    return pd.DataFrame(rows)


def _small_capital_report_payload(report: dict[str, Any]) -> dict[str, Any]:
    summary = dict(report.get("summary", {}) or {})
    return {
        "schema_version": 1,
        "status": report.get("status", ""),
        "run_tag": report.get("run_tag", ""),
        "study_family": "small_capital_balanced_return_v1",
        "gate_id": SMALL_CAPITAL_GATE_ID,
        "source_report": str(Path(str(report.get("summary_csv", ""))).with_name("v2_candidate_review_matrix_report.json")),
        "dataset_id": report.get("dataset_id", ""),
        "pool_view_id": report.get("pool_view_id", ""),
        "source_bridge_run_tag": report.get("source_bridge_run_tag", ""),
        "thresholds": dict(SMALL_CAPITAL_RESEARCH_THRESHOLDS),
        "summary": {
            "variant_count": int(summary.get("variant_count", 0)),
            "completed_backtest_count": int(summary.get("completed_backtest_count", 0)),
            "positive_transfer_count": int(summary.get("small_capital_positive_transfer_count", 0)),
            "research_grade_candidate_count": int(summary.get("small_capital_research_grade_candidate_count", 0)),
            "best_by_balanced_return_score": summary.get("best_by_small_capital_balanced_return_score", {}),
            "best_research_grade_candidate": summary.get("best_small_capital_research_grade_candidate", {}),
        },
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": report.get("updated_at", _now()),
    }


def _write_small_capital_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    best = dict(summary.get("best_by_balanced_return_score", {}) or {})
    candidate = dict(summary.get("best_research_grade_candidate", {}) or {})
    lines = [
        "# Small Capital Balanced Return V1",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Run tag: `{payload.get('run_tag', '')}`",
        f"- Gate id: `{payload.get('gate_id', '')}`",
        f"- Dataset: `{payload.get('dataset_id', '')}`",
        f"- Pool: `{payload.get('pool_view_id', '')}`",
        f"- Completed backtests: `{summary.get('completed_backtest_count', 0)}`",
        f"- Positive-transfer variants: `{summary.get('positive_transfer_count', 0)}`",
        f"- Research-grade candidates: `{summary.get('research_grade_candidate_count', 0)}`",
        "",
        "## Thresholds",
        "",
        "- excess_sharpe >= `1.2`",
        "- excess_annual_return >= `0.25`",
        "- positive_month_ratio >= `0.60`",
        "- worst_month >= `-0.15`",
        "- positive transfer requires positive excess annual return and positive excess Sharpe.",
        "",
        "## Best By Balanced Return Score",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Variant: `{best.get('variant_id', '')}`",
                f"- Score: `{best.get('small_capital_balanced_return_score', '')}`",
                f"- Excess annual return: `{best.get('excess_annual_return', '')}`",
                f"- Excess Sharpe: `{best.get('excess_sharpe', '')}`",
                f"- Positive month ratio: `{best.get('positive_month_ratio', '')}`",
                f"- Worst month: `{best.get('worst_monthly_excess_return', best.get('worst_monthly_return', ''))}`",
            ]
        )
    else:
        lines.append("- No completed small-capital variant yet.")
    if candidate:
        lines.extend(
            [
                "",
                "## Best Research-Grade Candidate",
                "",
                f"- Variant: `{candidate.get('variant_id', '')}`",
                f"- Score: `{candidate.get('small_capital_balanced_return_score', '')}`",
                f"- Excess annual return: `{candidate.get('excess_annual_return', '')}`",
                f"- Excess Sharpe: `{candidate.get('excess_sharpe', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- research_only: `true`",
            "- shadow_only: `true`",
            "- promotion_allowed: `false`",
            "- active_execution_strategy_expected_diff: `none`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}) or {})
    lines = [
        "# V2 Candidate Review Matrix",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Source bridge: `{report.get('source_bridge_run_tag', '')}`",
        f"- Dataset: `{report.get('dataset_id', '')}`",
        f"- Pool: `{report.get('pool_view_id', '')}`",
        f"- Variant count: `{summary.get('variant_count', 0)}`",
        f"- Completed backtests: `{summary.get('completed_backtest_count', 0)}`",
        f"- Promotion-review eligible variants: `{summary.get('promotion_review_eligible_count', 0)}`",
        f"- Small-capital positive-transfer variants: `{summary.get('small_capital_positive_transfer_count', 0)}`",
        f"- Small-capital research-grade candidates: `{summary.get('small_capital_research_grade_candidate_count', 0)}`",
        "",
        "## Boundary",
        "",
        "- research_only: `true`",
        "- shadow_only: `true`",
        "- promotion_allowed: `false`",
        "- active_execution_strategy_expected_diff: `none`",
        "- This matrix reviews score-to-weight candidate mappings; it is not a production target-weight panel.",
        "",
        "## Candidate Gate",
        "",
        "- monthly_positive_ratio >= `0.75`",
        "- negative_month_count <= `2`",
        "- max_drawdown >= `-0.25`",
        "- excess_sharpe >= `1.0`",
        "- no `deep_bad_month` flag",
        "- annual_return_cost_drag <= `0.20`",
    ]
    small_best = dict(summary.get("best_by_small_capital_balanced_return_score", {}) or {})
    if small_best:
        lines.extend(
            [
                "",
                f"## {SMALL_CAPITAL_GATE_ID}",
                "",
                "- This is a research-only scout gate, not a promotion gate.",
                "- positive transfer requires excess_annual_return > `0` and excess_sharpe > `0`.",
                "- research-grade threshold: excess_sharpe >= `1.2`, excess_annual_return >= `0.25`, positive_month_ratio >= `0.60`, worst_month >= `-0.15`.",
                f"- Best variant: `{small_best.get('variant_id', '')}`",
                f"- Balanced return score: `{small_best.get('small_capital_balanced_return_score', '')}`",
                f"- Excess annual return: `{small_best.get('excess_annual_return', '')}`",
                f"- Excess Sharpe: `{small_best.get('excess_sharpe', '')}`",
                f"- Positive month ratio: `{small_best.get('positive_month_ratio', '')}`",
                f"- Worst month: `{small_best.get('worst_monthly_excess_return', small_best.get('worst_monthly_return', ''))}`",
            ]
        )
    best = dict(summary.get("best_by_excess_sharpe", {}) or {})
    if best:
        lines.extend(
            [
                "",
                "## Best Completed Variant",
                "",
                f"- Variant: `{best.get('variant_id', '')}`",
                f"- Excess Sharpe: `{best.get('excess_sharpe', '')}`",
                f"- Positive month ratio: `{best.get('positive_month_ratio', '')}`",
                f"- Negative months: `{best.get('negative_month_count', '')}`",
                f"- Max drawdown: `{best.get('max_drawdown', '')}`",
                f"- Issue flags: `{best.get('issue_flags', '')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_review_matrix(
    *,
    run_tag: str = RUN_TAG,
    source_bridge_run_tag: str = SOURCE_BRIDGE_RUN_TAG,
    source_bridge_root: str | Path | None = None,
    output_root: str | Path | None = None,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    benchmark: str = "000300.SH",
    holding_counts: tuple[int, ...] = (10, 20, 30),
    max_weights: tuple[float, ...] = (0.08, 0.12, 0.16),
    rebalance_freqs: tuple[str, ...] = ("5d", "10d", "20d"),
    rebalance_offset_modes: tuple[str, ...] = ("all",),
    transaction_cost_bps_values: tuple[float, ...] = (10.0,),
    slippage_bps_values: tuple[float, ...] = (5.0,),
    sell_tax_bps_values: tuple[float, ...] = (10.0,),
    market_regime_filter_modes: tuple[bool, ...] = (False,),
    score_threshold: float = -999.0,
    run_backtests: bool = False,
    no_cache: bool = False,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_bridge_root) if source_bridge_root is not None else STUDIES_ROOT / str(source_bridge_run_tag)
    score_panel_csv = _source_score_panel(source_root)
    start_date, end_date = _score_panel_dates(score_panel_csv)
    variants = _variants(
        holding_counts=holding_counts,
        max_weights=max_weights,
        rebalance_freqs=rebalance_freqs,
        rebalance_offset_modes=rebalance_offset_modes,
        transaction_cost_bps_values=transaction_cost_bps_values,
        slippage_bps_values=slippage_bps_values,
        sell_tax_bps_values=sell_tax_bps_values,
        market_regime_filter_modes=market_regime_filter_modes,
    )
    results: list[dict[str, Any]] = []
    for variant in variants:
        command = _variant_command(
            variant,
            score_panel_csv=score_panel_csv,
            output_root=root,
            run_tag=run_tag,
            dataset_id=dataset_id,
            data_lake_root=data_lake_root,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            score_threshold=score_threshold,
            no_cache=no_cache,
        )
        run_result = None
        if run_backtests:
            run_result = _run_command(
                command,
                stdout_path=root / "logs" / f"{variant.variant_id}.stdout.log",
                stderr_path=root / "logs" / f"{variant.variant_id}.stderr.log",
            )
        results.append(_variant_result(variant, command=command, run_result=run_result))

    summary_csv = root / "v2_candidate_review_matrix_summary.csv"
    frame = _summary_frame(results)
    frame.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    completed = frame.loc[frame["status"].eq("completed")].copy() if not frame.empty else pd.DataFrame()
    best: dict[str, Any] = {}
    if not completed.empty and completed["excess_sharpe"].notna().any():
        best = completed.sort_values(["promotion_review_eligible", "excess_sharpe"], ascending=[False, False]).iloc[0].to_dict()
    small_completed = completed.loc[completed["small_capital_positive_transfer"].astype(bool)].copy() if not completed.empty else pd.DataFrame()
    small_best: dict[str, Any] = {}
    if not completed.empty and completed["small_capital_balanced_return_score"].notna().any():
        small_best = completed.sort_values(
            ["small_capital_research_grade_candidate", "small_capital_balanced_return_score", "excess_sharpe"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
    small_research_grade = (
        completed.loc[completed["small_capital_research_grade_candidate"].astype(bool)].copy() if not completed.empty else pd.DataFrame()
    )
    best_small_research_grade: dict[str, Any] = {}
    if not small_research_grade.empty:
        best_small_research_grade = small_research_grade.sort_values(
            ["small_capital_balanced_return_score", "excess_sharpe"],
            ascending=[False, False],
        ).iloc[0].to_dict()
    summary = {
        "variant_count": int(len(results)),
        "completed_backtest_count": int(frame["status"].eq("completed").sum()) if not frame.empty else 0,
        "promotion_review_eligible_count": int(frame["promotion_review_eligible"].sum()) if not frame.empty else 0,
        "small_capital_gate_id": SMALL_CAPITAL_GATE_ID,
        "small_capital_thresholds": dict(SMALL_CAPITAL_RESEARCH_THRESHOLDS),
        "small_capital_positive_transfer_count": int(len(small_completed)),
        "small_capital_research_grade_candidate_count": int(
            frame["small_capital_research_grade_candidate"].sum()
        )
        if not frame.empty
        else 0,
        "best_by_excess_sharpe": best,
        "best_by_small_capital_balanced_return_score": small_best,
        "best_small_capital_research_grade_candidate": best_small_research_grade,
    }
    report = {
        "schema_version": 1,
        "status": "completed" if not run_backtests or all(item.get("status") == "completed" for item in results) else "backtest_failed",
        "run_tag": str(run_tag),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_candidate_review_matrix",
        "source_bridge_run_tag": str(source_bridge_run_tag),
        "source_bridge_root": str(source_root),
        "score_panel_csv": str(score_panel_csv),
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "date_range": {"start": start_date, "end": end_date},
        "run_backtests": bool(run_backtests),
        "summary_csv": str(summary_csv),
        "summary": summary,
        "results": results,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "execution_state_required": "frozen_skeleton_only",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_candidate_review_matrix_report.json", report)
    _write_json(root / "v2_candidate_review_matrix_manifest.json", {k: v for k, v in report.items() if k != "results"})
    _write_markdown(root / "v2_candidate_review_matrix_report.md", report)
    small_capital_report = _small_capital_report_payload(report)
    _write_json(root / f"{SMALL_CAPITAL_GATE_ID}_report.json", small_capital_report)
    _write_small_capital_markdown(root / f"{SMALL_CAPITAL_GATE_ID}_report.md", small_capital_report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a research-only v2 execution-candidate review matrix.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--source-bridge-run-tag", default=SOURCE_BRIDGE_RUN_TAG)
    parser.add_argument("--source-bridge-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-counts", default="10,20,30")
    parser.add_argument("--max-weights", default="0.08,0.12,0.16")
    parser.add_argument("--rebalance-freqs", default="5d,10d,20d")
    parser.add_argument("--rebalance-offset-modes", default="all")
    parser.add_argument("--transaction-cost-bps-values", default="10")
    parser.add_argument("--slippage-bps-values", default="5")
    parser.add_argument("--sell-tax-bps-values", default="10")
    parser.add_argument("--market-regime-filter-modes", default="off")
    parser.add_argument("--score-threshold", type=float, default=-999.0)
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    regime_modes = tuple(
        item.strip().lower() in {"on", "true", "1", "yes"}
        for item in str(args.market_regime_filter_modes or "").split(",")
        if item.strip()
    ) or (False,)
    report = build_candidate_review_matrix(
        run_tag=args.run_tag,
        source_bridge_run_tag=args.source_bridge_run_tag,
        source_bridge_root=args.source_bridge_root or None,
        output_root=args.output_root or None,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        holding_counts=_parse_ints(args.holding_counts, (10, 20, 30)),
        max_weights=_parse_floats(args.max_weights, (0.08, 0.12, 0.16)),
        rebalance_freqs=_parse_strings(args.rebalance_freqs, ("5d", "10d", "20d")),
        rebalance_offset_modes=_parse_strings(args.rebalance_offset_modes, ("all",)),
        transaction_cost_bps_values=_parse_floats(args.transaction_cost_bps_values, (10.0,)),
        slippage_bps_values=_parse_floats(args.slippage_bps_values, (5.0,)),
        sell_tax_bps_values=_parse_floats(args.sell_tax_bps_values, (10.0,)),
        market_regime_filter_modes=regime_modes,
        score_threshold=float(args.score_threshold),
        run_backtests=bool(args.run_backtests),
        no_cache=bool(args.no_cache),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status={report.get('status')} run_tag={args.run_tag} output_root={args.output_root or STUDIES_ROOT / args.run_tag}")
    return 0 if str(report.get("status")) != "backtest_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
