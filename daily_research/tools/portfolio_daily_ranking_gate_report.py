from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daily_research.continuous_policy.run_self_optimizing_study import _score_protocol_summary  # noqa: E402
from daily_research.continuous_policy.runtime import now_iso  # noqa: E402


DEFAULT_STUDY_SUMMARY = (
    ROOT
    / "daily_research"
    / "output"
    / "continuous_policy"
    / "studies"
    / "cp_v3_portfolio_daily_ranking_r19__study_r1"
    / "study_summary.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _safe_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def _series_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    series = frame[column]
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "1.0", "true", "yes"})


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _masked_mean(frame: pd.DataFrame, column: str, mask: pd.Series) -> float:
    if column not in frame.columns or not bool(mask.any()):
        return 0.0
    value = pd.to_numeric(frame.loc[mask, column], errors="coerce").mean()
    return _safe_float(value)


def _quantiles(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": _safe_float(numeric.min()),
        "p25": _safe_float(numeric.quantile(0.25)),
        "p50": _safe_float(numeric.quantile(0.50)),
        "p75": _safe_float(numeric.quantile(0.75)),
        "max": _safe_float(numeric.max()),
    }


def _gate_checks(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    observed = (
        _safe_float(metrics.get("portfolio_daily_receiver_target_count")) >= 3.0
        or _safe_float(metrics.get("portfolio_daily_source_target_count")) >= 3.0
    )
    source_observed = _safe_float(metrics.get("portfolio_daily_source_target_count")) >= 5.0
    checks = [
        ("annual_return_positive", "年化收益必须为正", _safe_float(metrics.get("annual_return")) > 0.0),
        ("sharpe_positive", "Sharpe 必须为正", _safe_float(metrics.get("sharpe")) > 0.0),
        ("monthly_return_mean_positive", "月度均值必须为正", _safe_float(metrics.get("monthly_return_mean")) > 0.0),
        ("max_drawdown_floor", "最大回撤不应差于 -18%", _safe_float(metrics.get("max_drawdown")) >= -0.18),
        (
            "monthly_consistency_floor",
            "月度一致性不应低于 0.45",
            _safe_float(metrics.get("monthly_consistency_score")) >= 0.45,
        ),
        (
            "source_realized_sell_floor",
            "资金来源被观测到时 source realized sell rate 不应低于 0.35",
            (not source_observed) or _safe_float(metrics.get("portfolio_daily_source_realized_sell_rate")) >= 0.35,
        ),
        (
            "source_not_sold_ceiling",
            "资金来源被观测到时 source target 未真实卖出的占比不应高于 0.65",
            (not source_observed) or _safe_float(metrics.get("portfolio_daily_source_target_not_sold_share")) <= 0.65,
        ),
        (
            "order_translation_conflict_ceiling",
            "订单翻译冲突率不应高于 0.24",
            _safe_float(metrics.get("order_translation_conflict_rate")) <= 0.24
            and _safe_float(metrics.get("direct_action_order_translation_conflict_rate")) <= 0.24,
        ),
        (
            "add_to_hold_conflict_ceiling",
            "add-to-hold 冲突率不应高于 0.35",
            _safe_float(metrics.get("add_to_hold_conflict_share")) <= 0.35,
        ),
        (
            "cash_branch_alive",
            "组合日排名被观测到时 cash reserve 不能长期为 0",
            (not observed) or _safe_float(metrics.get("portfolio_daily_cash_reserve_rate")) > 0.0,
        ),
    ]
    return [
        {"name": name, "description": description, "passed": bool(passed)}
        for name, description, passed in checks
    ]


def _csv_diagnostics(evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    outcomes_path = _safe_path(evaluation_summary.get("action_outcomes_csv"))
    turnover_path = _safe_path(evaluation_summary.get("turnover_csv"))
    monthly_path = _safe_path(evaluation_summary.get("monthly_returns_csv"))

    if outcomes_path is not None:
        outcomes = pd.read_csv(outcomes_path)
        receiver_mask = _series_bool(outcomes, "portfolio_daily_receiver_target")
        source_mask = _series_bool(outcomes, "portfolio_daily_source_target")
        source_realized_mask = source_mask & outcomes.get("weight_change_action", pd.Series("", index=outcomes.index)).isin(
            ["reduce", "exit"]
        )
        source_not_sold_mask = source_mask & (~source_realized_mask)
        receiver_realized_mask = receiver_mask & outcomes.get("weight_change_action", pd.Series("", index=outcomes.index)).isin(
            ["open", "add"]
        )
        receiver_unrealized_mask = receiver_mask & (~receiver_realized_mask)
        receiver_exec_guard_mask = _series_bool(outcomes, "portfolio_daily_receiver_exec_guarded")
        source_reason_series = outcomes.get(
            "portfolio_daily_source_target_not_sold_reason",
            pd.Series("historical_missing_reason", index=outcomes.index),
        )
        result["action_outcomes"] = {
            "row_count": int(len(outcomes)),
            "receiver_target_count": int(receiver_mask.sum()),
            "source_target_count": int(source_mask.sum()),
            "source_realized_sell_count": int(source_realized_mask.sum()),
            "source_realized_sell_rate": _safe_float(source_realized_mask.sum() / source_mask.sum())
            if bool(source_mask.any())
            else 0.0,
            "source_target_not_sold_count": int(source_not_sold_mask.sum()),
            "source_target_not_sold_share": _safe_float(source_not_sold_mask.sum() / source_mask.sum())
            if bool(source_mask.any())
            else 0.0,
            "source_exec_guard_count": int(_series_bool(outcomes, "portfolio_daily_source_exec_guard").sum()),
            "source_exec_cap_guard_count": int(_series_bool(outcomes, "portfolio_daily_source_exec_cap_guarded").sum()),
            "receiver_exec_guard_count": int(receiver_exec_guard_mask.sum()),
            "receiver_realized_deploy_rate": _safe_float(receiver_realized_mask.sum() / receiver_mask.sum())
            if bool(receiver_mask.any())
            else 0.0,
            "receiver_unrealized_deploy_count": int(receiver_unrealized_mask.sum()),
            "receiver_unrealized_deploy_share": _safe_float(receiver_unrealized_mask.sum() / receiver_mask.sum())
            if bool(receiver_mask.any())
            else 0.0,
            "receiver_exec_guard_reason_counts": {
                str(key): int(value)
                for key, value in outcomes.get(
                    "portfolio_daily_receiver_exec_guard_reason",
                    pd.Series("historical_missing_reason", index=outcomes.index),
                )
                .loc[receiver_exec_guard_mask]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
            "source_realized_reduction_weight": _safe_float(
                _numeric(outcomes, "portfolio_daily_source_realized_reduction_weight").loc[source_mask].sum()
            ),
            "receiver_realized_deploy_count": int(receiver_realized_mask.sum()),
            "effective_capital_transfer_count": int(
                min(int(source_realized_mask.sum()), int(receiver_realized_mask.sum()))
            ),
            "source_not_sold_reason_counts": {
                str(key): int(value)
                for key, value in source_reason_series.loc[source_not_sold_mask]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
            "receiver_forward_excess_5d": _masked_mean(outcomes, "forward_excess_5d", receiver_mask),
            "source_forward_excess_5d": _masked_mean(outcomes, "forward_excess_5d", source_mask),
            "receiver_minus_source_forward_excess_5d": _masked_mean(outcomes, "forward_excess_5d", receiver_mask)
            - _masked_mean(outcomes, "forward_excess_5d", source_mask),
            "receiver_forward_excess_10d": _masked_mean(outcomes, "forward_excess_10d", receiver_mask),
            "source_forward_excess_10d": _masked_mean(outcomes, "forward_excess_10d", source_mask),
            "receiver_forward_excess_20d": _masked_mean(outcomes, "forward_excess_20d", receiver_mask),
            "source_forward_excess_20d": _masked_mean(outcomes, "forward_excess_20d", source_mask),
            "receiver_component_means": {
                column: _masked_mean(outcomes, column, receiver_mask)
                for column in [
                    "direct_action_deploy_rank_score",
                    "portfolio_daily_receiver_score",
                    "deploy_value_target",
                    "deploy_gate_target",
                    "deploy_executability_target",
                    "alpha_opportunity_value",
                    "value_arbitration_target",
                    "cash_defense_value",
                    "release_value_target",
                    "current_weight",
                ]
            },
            "source_component_means": {
                column: _masked_mean(outcomes, column, source_mask)
                for column in [
                    "direct_action_pair_source_release_score",
                    "portfolio_daily_source_gap",
                    "portfolio_daily_source_score",
                    "direct_action_pair_source_opportunity_cost",
                    "release_value_target",
                    "release_gate_target",
                    "sell_release_value",
                    "exit_timing_pressure",
                    "portfolio_daily_cash_score",
                    "hold_continuation_value",
                    "alpha_opportunity_value",
                    "deploy_executability_target",
                    "large_upside_1d_target",
                ]
            },
        }

    if turnover_path is not None:
        turnover = pd.read_csv(turnover_path)
        calibration = str(evaluation_summary.get("budget_calibration", "") or "")
        cash_threshold = (
            0.24
            if (
                "cash_aware_guard_v13" in calibration
                or "source_exec_guard_v14" in calibration
                or "receiver_exec_guard_v15" in calibration
            )
            else 0.58
        )
        cash_score = _numeric(turnover, "portfolio_daily_cash_score")
        reserve = _series_bool(turnover, "portfolio_daily_cash_reserve_signal")
        score_above_threshold = cash_score >= cash_threshold
        result["cash_branch"] = {
            "day_count": int(len(turnover)),
            "cash_threshold": cash_threshold,
            "cash_score_quantiles": _quantiles(cash_score),
            "cash_score_mean": _safe_float(cash_score.mean()),
            "cash_score_above_threshold_days": int(score_above_threshold.sum()),
            "cash_reserve_days": int(reserve.sum()),
            "cash_reserve_rate": _safe_float(reserve.mean()) if len(reserve) else 0.0,
            "likely_receiver_pressure_blocked_days": int((score_above_threshold & ~reserve).sum()),
            "avg_receiver_target_count": _safe_float(_numeric(turnover, "portfolio_daily_receiver_target_count").mean()),
            "avg_source_target_count": _safe_float(_numeric(turnover, "portfolio_daily_source_target_count").mean()),
            "avg_budget_deploy_score": _safe_float(_numeric(turnover, "budget_deploy_score").mean()),
            "avg_budget_cash_timing_signal": _safe_float(
                _numeric(turnover, "budget_model_cash_timing_signal").mean()
            ),
            "avg_budget_risk_signal": _safe_float(_numeric(turnover, "budget_model_risk_signal").mean()),
            "avg_order_translation_conflict_rate": _safe_float(
                _numeric(turnover, "order_translation_conflict_rate").mean()
            ),
            "max_order_translation_conflict_rate": _safe_float(
                _numeric(turnover, "order_translation_conflict_rate").max()
            ),
            "avg_add_to_hold_conflict_share": _safe_float(_numeric(turnover, "add_to_hold_conflict_share").mean()),
            "max_add_to_hold_conflict_share": _safe_float(_numeric(turnover, "add_to_hold_conflict_share").max()),
        }
        if "date" in turnover.columns:
            monthly = turnover.copy()
            monthly["month"] = pd.to_datetime(monthly["date"], errors="coerce").dt.to_period("M").astype(str)
            monthly_rows = []
            for month, group in monthly.groupby("month", dropna=True):
                monthly_rows.append(
                    {
                        "month": str(month),
                        "avg_order_translation_conflict_rate": _safe_float(
                            _numeric(group, "order_translation_conflict_rate").mean()
                        ),
                        "avg_add_to_hold_conflict_share": _safe_float(
                            _numeric(group, "add_to_hold_conflict_share").mean()
                        ),
                        "avg_cash_score": _safe_float(_numeric(group, "portfolio_daily_cash_score").mean()),
                        "cash_reserve_rate": _safe_float(
                            _series_bool(group, "portfolio_daily_cash_reserve_signal").mean()
                        ),
                    }
                )
            result["monthly_execution"] = monthly_rows

    if monthly_path is not None:
        monthly_returns = pd.read_csv(monthly_path)
        result["monthly_returns"] = [
            {
                "month": str(row.get("month", "")),
                "monthly_return": _safe_float(row.get("monthly_return")),
                "intramonth_max_drawdown": _safe_float(row.get("intramonth_max_drawdown")),
            }
            for row in monthly_returns.to_dict("records")
        ]

    return result


def _trial_record(item: dict[str, Any]) -> dict[str, Any]:
    protocol_path = Path(str(item.get("protocol_summary_json", "") or ""))
    protocol = _read_json(protocol_path)
    v1 = _score_protocol_summary(protocol, objective_profile="portfolio_daily_ranking_v1")
    v2 = _score_protocol_summary(protocol, objective_profile="portfolio_daily_ranking_v2_gated")
    metrics = dict(v2.get("primary_metrics", {}) or {})
    evaluation_path = _safe_path(dict(protocol.get("evaluation", {}) or {}).get("evaluation_summary_json"))
    csv_diagnostics = _csv_diagnostics(_read_json(evaluation_path)) if evaluation_path else {}
    action_diag = dict(csv_diagnostics.get("action_outcomes", {}) or {})
    if action_diag:
        metrics["portfolio_daily_source_target_not_sold_share"] = action_diag.get("source_target_not_sold_share")
        metrics["portfolio_daily_source_target_not_sold_count"] = action_diag.get("source_target_not_sold_count")
        metrics["portfolio_daily_effective_capital_transfer_count"] = action_diag.get("effective_capital_transfer_count")
        metrics["portfolio_daily_receiver_realized_deploy_count"] = action_diag.get("receiver_realized_deploy_count")
        metrics["portfolio_daily_receiver_exec_guard_count"] = action_diag.get("receiver_exec_guard_count")
        metrics["portfolio_daily_receiver_realized_deploy_rate"] = action_diag.get("receiver_realized_deploy_rate")
        metrics["portfolio_daily_receiver_unrealized_deploy_share"] = action_diag.get("receiver_unrealized_deploy_share")
        metrics["portfolio_daily_source_exec_cap_guard_count"] = action_diag.get("source_exec_cap_guard_count")
        metrics["portfolio_daily_source_realized_reduction_weight"] = action_diag.get("source_realized_reduction_weight")
        v2["primary_metrics"] = metrics
    gate_checks = _gate_checks(metrics)
    return {
        "trial_tag": str(item.get("trial_tag", "")),
        "phase": str(item.get("phase", "")),
        "role": str(item.get("role", "")),
        "source_trial_tag": str(item.get("source_trial_tag", "")),
        "protocol_summary_json": str(protocol_path.resolve()),
        "v1": v1,
        "v2": v2,
        "gate_checks": gate_checks,
        "gate_failed": [check["name"] for check in gate_checks if not check["passed"]],
        "csv_diagnostics": csv_diagnostics,
        "training_diagnostics": dict(
            dict(protocol.get("evaluation", {}) or {}).get("training_diagnostics", {})
            or dict(protocol.get("training_evidence", {}) or {}).get("training_diagnostics", {})
            or {}
        ),
    }


def _confirm_stability(record: dict[str, Any], source: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = dict(record["v2"].get("primary_metrics", {}) or {})
    source_metrics = dict(source["v2"].get("primary_metrics", {}) or {}) if source else {}
    gate_pass = not record.get("gate_failed")
    source_gate_pass = (not source.get("gate_failed")) if source else False
    annual = _safe_float(metrics.get("annual_return"))
    sharpe = _safe_float(metrics.get("sharpe"))
    monthly = _safe_float(metrics.get("monthly_return_mean"))
    drawdown = _safe_float(metrics.get("max_drawdown"))
    annual_delta = annual - _safe_float(source_metrics.get("annual_return"))
    sharpe_delta = sharpe - _safe_float(source_metrics.get("sharpe"))
    monthly_delta = monthly - _safe_float(source_metrics.get("monthly_return_mean"))
    drawdown_delta = drawdown - _safe_float(source_metrics.get("max_drawdown"))
    order_delta = _safe_float(metrics.get("order_translation_conflict_rate")) - _safe_float(
        source_metrics.get("order_translation_conflict_rate")
    )
    add_delta = _safe_float(metrics.get("add_to_hold_conflict_share")) - _safe_float(
        source_metrics.get("add_to_hold_conflict_share")
    )
    checks = {
        "confirm_gate_pass": gate_pass,
        "confirm_annual_return_floor": annual >= 0.12,
        "confirm_sharpe_floor": sharpe >= 0.50,
        "confirm_monthly_return_floor": monthly >= 0.006,
        "confirm_drawdown_floor": drawdown >= -0.18,
        "annual_return_decay_limit": (not source) or annual_delta >= -0.35,
        "sharpe_decay_limit": (not source) or sharpe_delta >= -1.20,
        "monthly_return_decay_limit": (not source) or monthly_delta >= -0.025,
        "drawdown_decay_limit": (not source) or drawdown_delta >= -0.07,
        "order_translation_decay_limit": (not source) or order_delta <= 0.18,
        "add_to_hold_decay_limit": (not source) or add_delta <= 0.22,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "trial_tag": record["trial_tag"],
        "source_trial_tag": record.get("source_trial_tag", ""),
        "source_found": source is not None,
        "stable_confirmatory": not failed,
        "failed_stability_checks": failed,
        "source_gate_pass": source_gate_pass,
        "confirm_gate_pass": gate_pass,
        "annual_return_delta": annual_delta if source else 0.0,
        "sharpe_delta": sharpe_delta if source else 0.0,
        "monthly_return_mean_delta": monthly_delta if source else 0.0,
        "max_drawdown_delta": drawdown_delta if source else 0.0,
        "order_translation_conflict_delta": order_delta if source else 0.0,
        "add_to_hold_conflict_delta": add_delta if source else 0.0,
    }


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    by_tag = {record["trial_tag"]: record for record in records}
    rows = []
    for record in records:
        metrics = dict(record["v2"].get("primary_metrics", {}) or {})
        stability = (
            _confirm_stability(record, by_tag.get(str(record.get("source_trial_tag", ""))))
            if record.get("phase") == "confirmatory"
            else {}
        )
        rows.append(
            {
                "trial_tag": record["trial_tag"],
                "phase": record["phase"],
                "role": record["role"],
                "source_trial_tag": record.get("source_trial_tag", ""),
                "v1_composite_score": record["v1"]["composite_score"],
                "v2_composite_score": record["v2"]["composite_score"],
                "v2_performance_score": record["v2"]["performance_score"],
                "v2_stability_score": record["v2"]["stability_score"],
                "v2_gate_failed": ",".join(record["gate_failed"]),
                "confirm_stable": stability.get("stable_confirmatory", ""),
                "confirm_stability_failed": ",".join(stability.get("failed_stability_checks", [])),
                "annual_return_delta_vs_source": stability.get("annual_return_delta", ""),
                "sharpe_delta_vs_source": stability.get("sharpe_delta", ""),
                "monthly_return_mean_delta_vs_source": stability.get("monthly_return_mean_delta", ""),
                "max_drawdown_delta_vs_source": stability.get("max_drawdown_delta", ""),
                "order_translation_conflict_delta_vs_source": stability.get("order_translation_conflict_delta", ""),
                "add_to_hold_conflict_delta_vs_source": stability.get("add_to_hold_conflict_delta", ""),
                "annual_return": metrics.get("annual_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "monthly_return_mean": metrics.get("monthly_return_mean"),
                "monthly_consistency_score": metrics.get("monthly_consistency_score"),
                "receiver_minus_source_5d": metrics.get("portfolio_daily_receiver_minus_source_forward_excess_5d"),
                "source_realized_sell_rate": metrics.get("portfolio_daily_source_realized_sell_rate"),
                "source_target_not_sold_share": metrics.get("portfolio_daily_source_target_not_sold_share"),
                "source_exec_cap_guard_count": metrics.get("portfolio_daily_source_exec_cap_guard_count"),
                "effective_capital_transfer_count": metrics.get("portfolio_daily_effective_capital_transfer_count"),
                "cash_reserve_rate": metrics.get("portfolio_daily_cash_reserve_rate"),
                "cash_score_mean": metrics.get("portfolio_daily_cash_score_mean"),
                "order_translation_conflict_rate": metrics.get("order_translation_conflict_rate"),
                "add_to_hold_conflict_share": metrics.get("add_to_hold_conflict_share"),
                "protocol_summary_json": record["protocol_summary_json"],
            }
        )
    pd.DataFrame(rows).sort_values("v2_composite_score", ascending=False).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def _fmt(value: Any, digits: int = 6) -> str:
    return f"{_safe_float(value):.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    ranking = report["v2_ranking"]
    v1_champion = report["v1_champion"]
    v2_top_ranked = report["v2_champion"]
    qualified_v2_champion = report.get("qualified_v2_champion")
    economic_champion = report["economic_champion"]
    cash = report["cash_branch_summary"]
    stability_pairs = report.get("confirm_stability_checks", [])
    cash_failed_count = sum(1 for record in ranking if "cash_branch_alive" in record.get("gate_failed", []))
    source_failed_count = sum(
        1
        for record in ranking
        if "source_realized_sell_floor" in record.get("gate_failed", [])
        or "source_not_sold_ceiling" in record.get("gate_failed", [])
    )
    order_failed_count = sum(1 for record in ranking if "order_translation_conflict_ceiling" in record.get("gate_failed", []))
    add_failed_count = sum(1 for record in ranking if "add_to_hold_conflict_ceiling" in record.get("gate_failed", []))
    stable_confirm_count = sum(1 for item in stability_pairs if item.get("stable_confirmatory"))
    core_conclusions = [
        "- v2 目标已把“亏损但 ranking 干净”的路线压低，receiver-source spread 不再能单独抵消负收益。",
    ]
    if cash_failed_count == len(ranking):
        core_conclusions.append("- 当前全部路线仍触发 cash_branch_alive 失败，现金保留分支尚未成为有效竞争项。")
    elif cash_failed_count:
        core_conclusions.append(f"- 当前仍有 {cash_failed_count} 条路线触发 cash_branch_alive 失败，现金保留需要按候选逐条审计。")
    else:
        core_conclusions.append("- 当前没有路线触发 cash_branch_alive 失败，现金保留分支已经从死分支变为可观测行为。")
    if order_failed_count or add_failed_count:
        core_conclusions.append(
            f"- 执行冲突仍需关注：order translation 失败路线 {order_failed_count} 条，add-to-hold 失败路线 {add_failed_count} 条。"
        )
    else:
        core_conclusions.append("- 当前 v2 gate 下没有路线触发 order translation 或 add-to-hold 上限失败。")
    if source_failed_count:
        core_conclusions.append(f"- source 执行耦合仍是核心瓶颈：{source_failed_count} 条路线未达到真实释放资金要求。")
    if stability_pairs:
        if stable_confirm_count:
            core_conclusions.append(f"- fresh confirm 稳定性已有 {stable_confirm_count} 条通过，但仍需结合收益、回撤和月度质量人工复核。")
        else:
            core_conclusions.append("- fresh confirm 稳定性尚未通过：screening 过 gate 的路线仍可能在 confirmatory 中塌陷。")

    lines = [
        "# portfolio_daily_ranking_v2_gated 离线诊断报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 输入 study：`{report['study_summary_json']}`",
        f"- v1 champion：`{v1_champion['trial_tag']}`，v1 composite={_fmt(v1_champion['v1']['composite_score'])}",
        f"- v2 top-ranked：`{v2_top_ranked['trial_tag']}`，v2 composite={_fmt(v2_top_ranked['v2']['composite_score'])}",
        "- 合格 v2 champion："
        + (
            f"`{qualified_v2_champion['trial_tag']}`"
            if qualified_v2_champion
            else "无，当前没有完全通过 v2 gates 的路线"
        ),
        f"- 经济冠军：`{economic_champion['trial_tag']}`，annual_return={_fmt(economic_champion['v2']['primary_metrics'].get('annual_return'))}",
        "",
        "## 核心结论",
        "",
        *core_conclusions,
        "",
        "## v2 重排",
        "",
        "| rank | trial | phase | v2 | v1 | annual | sharpe | mdd | spread5d | src_not_sold | transfer | cash | failed gates |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, record in enumerate(ranking, start=1):
        metrics = record["v2"]["primary_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{record['trial_tag']}`",
                    record["phase"],
                    _fmt(record["v2"]["composite_score"]),
                    _fmt(record["v1"]["composite_score"]),
                    _fmt(metrics.get("annual_return")),
                    _fmt(metrics.get("sharpe")),
                    _fmt(metrics.get("max_drawdown")),
                    _fmt(metrics.get("portfolio_daily_receiver_minus_source_forward_excess_5d")),
                    _fmt(metrics.get("portfolio_daily_source_target_not_sold_share")),
                    _fmt(metrics.get("portfolio_daily_effective_capital_transfer_count"), digits=0),
                    _fmt(metrics.get("portfolio_daily_cash_reserve_rate")),
                    ",".join(record["gate_failed"]) or "无",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 反冠军对比",
            "",
        ]
    )
    champion_rows = [("v1 champion", v1_champion), ("v2 top-ranked", v2_top_ranked), ("economic champion", economic_champion)]
    if qualified_v2_champion:
        champion_rows.insert(2, ("qualified v2 champion", qualified_v2_champion))
    for label, record in champion_rows:
        metrics = record["v2"]["primary_metrics"]
        csv_diag = record.get("csv_diagnostics", {})
        action_diag = csv_diag.get("action_outcomes", {})
        lines.extend(
            [
                f"### {label}: `{record['trial_tag']}`",
                "",
                f"- annual_return={_fmt(metrics.get('annual_return'))}, sharpe={_fmt(metrics.get('sharpe'))}, max_drawdown={_fmt(metrics.get('max_drawdown'))}",
                f"- receiver-source 5d={_fmt(metrics.get('portfolio_daily_receiver_minus_source_forward_excess_5d'))}, source realized sell rate={_fmt(metrics.get('portfolio_daily_source_realized_sell_rate'))}",
                f"- source not sold share={_fmt(metrics.get('portfolio_daily_source_target_not_sold_share'))}, effective capital transfer count={_fmt(metrics.get('portfolio_daily_effective_capital_transfer_count'), digits=0)}",
                f"- receiver exec guard count={_fmt(metrics.get('portfolio_daily_receiver_exec_guard_count'), digits=0)}, receiver realized deploy rate={_fmt(metrics.get('portfolio_daily_receiver_realized_deploy_rate'))}",
                f"- order_translation_conflict={_fmt(metrics.get('order_translation_conflict_rate'))}, add_to_hold_conflict={_fmt(metrics.get('add_to_hold_conflict_share'))}",
                f"- CSV 复算 receiver-source 5d={_fmt(action_diag.get('receiver_minus_source_forward_excess_5d'))}, source realized sell count={action_diag.get('source_realized_sell_count', 0)}",
                f"- source 未卖原因：{action_diag.get('source_not_sold_reason_counts', {})}",
                f"- 失败门槛：{', '.join(record['gate_failed']) or '无'}",
                "",
            ]
    )

    if stability_pairs:
        lines.extend(
            [
                "## confirm 稳定性",
                "",
                "| trial | source | stable | annual_delta | sharpe_delta | monthly_delta | mdd_delta | order_delta | add_delta | failed checks |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in stability_pairs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{item.get('trial_tag', '')}`",
                        f"`{item.get('source_trial_tag', '')}`",
                        "是" if item.get("stable_confirmatory") else "否",
                        _fmt(item.get("annual_return_delta")),
                        _fmt(item.get("sharpe_delta")),
                        _fmt(item.get("monthly_return_mean_delta")),
                        _fmt(item.get("max_drawdown_delta")),
                        _fmt(item.get("order_translation_conflict_delta")),
                        _fmt(item.get("add_to_hold_conflict_delta")),
                        ",".join(item.get("failed_stability_checks", [])) or "无",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## cash 分支诊断",
            "",
            f"- 观察 trial 数：{cash['trial_count']}",
            f"- 平均 cash_score={_fmt(cash['avg_cash_score_mean'])}",
            f"- 最大 cash_score={_fmt(cash['max_cash_score'])}",
            f"- cash_score >= 阈值的天数合计：{cash['cash_score_above_threshold_days']}",
            f"- cash reserve 天数合计：{cash['cash_reserve_days']}",
            f"- 疑似被 receiver pressure 压制天数：{cash['likely_receiver_pressure_blocked_days']}",
            "",
            "## 下一步执行判定",
            "",
            "- 未通过 stable confirm 前，不得进入 promotion 或 live 讨论。",
        ]
    )
    if source_failed_count:
        lines.extend(
            [
                "- 下一轮优先使用 v14 source-exec guard 验证 source target 是否真实 reduce/exit，而不是只扩大 v13 训练。",
                "- 如果 source not sold 或 effective transfer 仍失败，应继续收紧 source 保留底线、换手优先级和 receiver/source matching。",
            ]
        )
    else:
        lines.extend(
            [
                "- 当前 source execution gate 未失败时，下一轮应保持 v14，并优先压低 order translation 与 add-to-hold 冲突。",
                "- 不应因 source realized sell 改善就进入 promotion；必须补 fresh confirm 和执行冲突修复证据。",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_report(study_summary_path: Path, output_dir: Path) -> dict[str, Any]:
    study = _read_json(study_summary_path)
    records = [_trial_record(item) for item in study.get("trials", []) or []]
    if not records:
        raise ValueError(f"study 中没有可诊断 trial：{study_summary_path}")

    ranking = sorted(records, key=lambda record: float(record["v2"]["composite_score"]), reverse=True)
    v1_champion_tag = str(dict(study.get("champion", {}) or {}).get("trial_tag", ""))
    v1_champion = next((record for record in records if record["trial_tag"] == v1_champion_tag), records[0])
    v2_champion = ranking[0]
    qualified_v2_champion = next((record for record in ranking if not record.get("gate_failed")), None)
    economic_champion = max(
        records,
        key=lambda record: float(record["v2"]["primary_metrics"].get("annual_return", 0.0) or 0.0),
    )

    cash_rows = []
    for record in records:
        cash_diag = dict(record.get("csv_diagnostics", {}).get("cash_branch", {}) or {})
        if cash_diag:
            cash_rows.append(cash_diag)
    cash_summary = {
        "trial_count": len(cash_rows),
        "avg_cash_score_mean": _safe_float(
            sum(_safe_float(row.get("cash_score_mean")) for row in cash_rows) / len(cash_rows)
        )
        if cash_rows
        else 0.0,
        "max_cash_score": max((_safe_float(row.get("cash_score_quantiles", {}).get("max")) for row in cash_rows), default=0.0),
        "cash_score_above_threshold_days": int(
            sum(int(row.get("cash_score_above_threshold_days", 0) or 0) for row in cash_rows)
        ),
        "cash_reserve_days": int(sum(int(row.get("cash_reserve_days", 0) or 0) for row in cash_rows)),
        "likely_receiver_pressure_blocked_days": int(
            sum(int(row.get("likely_receiver_pressure_blocked_days", 0) or 0) for row in cash_rows)
        ),
    }
    by_tag = {record["trial_tag"]: record for record in records}
    confirm_stability_checks = [
        _confirm_stability(record, by_tag.get(str(record.get("source_trial_tag", ""))))
        for record in records
        if record.get("phase") == "confirmatory"
    ]

    report = {
        "generated_at": now_iso(),
        "study_summary_json": str(study_summary_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "v1_champion": v1_champion,
        "v2_champion": v2_champion,
        "qualified_v2_champion": qualified_v2_champion,
        "economic_champion": economic_champion,
        "v2_ranking": ranking,
        "cash_branch_summary": cash_summary,
        "confirm_stability_checks": confirm_stability_checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(records, output_dir / "portfolio_daily_ranking_v2_rescore.csv")
    (output_dir / "portfolio_daily_ranking_v2_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "portfolio_daily_ranking_v2_gate_report.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线重算 portfolio_daily_ranking_v2_gated 并生成反冠军诊断。")
    parser.add_argument("--study-summary", default=str(DEFAULT_STUDY_SUMMARY))
    parser.add_argument("--output-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study_summary_path = Path(args.study_summary).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir or "").strip()
        else study_summary_path.parent / "portfolio_daily_ranking_v2_gate_report"
    )
    report = build_report(study_summary_path, output_dir)
    print(json.dumps(
        {
            "generated_at": report["generated_at"],
            "output_dir": report["output_dir"],
            "v1_champion": report["v1_champion"]["trial_tag"],
            "v2_top_ranked": report["v2_champion"]["trial_tag"],
            "qualified_v2_champion": (
                report["qualified_v2_champion"]["trial_tag"]
                if report.get("qualified_v2_champion")
                else ""
            ),
            "economic_champion": report["economic_champion"]["trial_tag"],
            "report_md": str((output_dir / "portfolio_daily_ranking_v2_gate_report.md").resolve()),
            "rescore_csv": str((output_dir / "portfolio_daily_ranking_v2_rescore.csv").resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
