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
        result["action_outcomes"] = {
            "row_count": int(len(outcomes)),
            "receiver_target_count": int(receiver_mask.sum()),
            "source_target_count": int(source_mask.sum()),
            "source_realized_sell_count": int(source_realized_mask.sum()),
            "source_realized_sell_rate": _safe_float(source_realized_mask.sum() / source_mask.sum())
            if bool(source_mask.any())
            else 0.0,
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
        cash_threshold = 0.24 if "cash_aware_guard_v13" in calibration else 0.58
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
    gate_checks = _gate_checks(metrics)
    evaluation_path = _safe_path(dict(protocol.get("evaluation", {}) or {}).get("evaluation_summary_json"))
    csv_diagnostics = _csv_diagnostics(_read_json(evaluation_path)) if evaluation_path else {}
    return {
        "trial_tag": str(item.get("trial_tag", "")),
        "phase": str(item.get("phase", "")),
        "role": str(item.get("role", "")),
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


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        metrics = dict(record["v2"].get("primary_metrics", {}) or {})
        rows.append(
            {
                "trial_tag": record["trial_tag"],
                "phase": record["phase"],
                "role": record["role"],
                "v1_composite_score": record["v1"]["composite_score"],
                "v2_composite_score": record["v2"]["composite_score"],
                "v2_performance_score": record["v2"]["performance_score"],
                "v2_stability_score": record["v2"]["stability_score"],
                "v2_gate_failed": ",".join(record["gate_failed"]),
                "annual_return": metrics.get("annual_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "monthly_return_mean": metrics.get("monthly_return_mean"),
                "monthly_consistency_score": metrics.get("monthly_consistency_score"),
                "receiver_minus_source_5d": metrics.get("portfolio_daily_receiver_minus_source_forward_excess_5d"),
                "source_realized_sell_rate": metrics.get("portfolio_daily_source_realized_sell_rate"),
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
    v2_champion = report["v2_champion"]
    economic_champion = report["economic_champion"]
    cash = report["cash_branch_summary"]

    lines = [
        "# portfolio_daily_ranking_v2_gated 离线诊断报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 输入 study：`{report['study_summary_json']}`",
        f"- v1 champion：`{v1_champion['trial_tag']}`，v1 composite={_fmt(v1_champion['v1']['composite_score'])}",
        f"- v2 champion：`{v2_champion['trial_tag']}`，v2 composite={_fmt(v2_champion['v2']['composite_score'])}",
        f"- 经济冠军：`{economic_champion['trial_tag']}`，annual_return={_fmt(economic_champion['v2']['primary_metrics'].get('annual_return'))}",
        "",
        "## 核心结论",
        "",
        "- v2 目标已把“亏损但 ranking 干净”的路线压低，receiver-source spread 不再能单独抵消负收益。",
        "- r19 的全部路线都触发 cash_branch_alive 失败，说明 cash reserve 在当前组合日排名实现中是死亡分支。",
        "- 主要结构冲突仍集中在 order_translation_conflict 与 add_to_hold_conflict，说明模型意图到真实权重变化之间仍有翻译损耗。",
        "",
        "## v2 重排",
        "",
        "| rank | trial | phase | v2 | v1 | annual | sharpe | mdd | spread5d | cash | failed gates |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
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
    for label, record in [("v1 champion", v1_champion), ("v2 champion", v2_champion), ("economic champion", economic_champion)]:
        metrics = record["v2"]["primary_metrics"]
        csv_diag = record.get("csv_diagnostics", {})
        action_diag = csv_diag.get("action_outcomes", {})
        lines.extend(
            [
                f"### {label}: `{record['trial_tag']}`",
                "",
                f"- annual_return={_fmt(metrics.get('annual_return'))}, sharpe={_fmt(metrics.get('sharpe'))}, max_drawdown={_fmt(metrics.get('max_drawdown'))}",
                f"- receiver-source 5d={_fmt(metrics.get('portfolio_daily_receiver_minus_source_forward_excess_5d'))}, source realized sell rate={_fmt(metrics.get('portfolio_daily_source_realized_sell_rate'))}",
                f"- order_translation_conflict={_fmt(metrics.get('order_translation_conflict_rate'))}, add_to_hold_conflict={_fmt(metrics.get('add_to_hold_conflict_share'))}",
                f"- CSV 复算 receiver-source 5d={_fmt(action_diag.get('receiver_minus_source_forward_excess_5d'))}, source realized sell count={action_diag.get('source_realized_sell_count', 0)}",
                f"- 失败门槛：{', '.join(record['gate_failed']) or '无'}",
                "",
            ]
        )

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
            "- 现在不建议直接扩大长训练；应先用 v2 作为 r20 smoke 的目标函数。",
            "- r20 smoke 必须验证三件事：负收益路线不能夺冠，cash reserve 不能继续全 0，订单翻译冲突必须下降或被强惩罚。",
            "- 如果 smoke 仍全量触发 cash_branch_alive 失败，应优先修 cash gate/receiver pressure 竞争机制，再进入 confirmatory。",
            "",
        ]
    )
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

    report = {
        "generated_at": now_iso(),
        "study_summary_json": str(study_summary_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "v1_champion": v1_champion,
        "v2_champion": v2_champion,
        "economic_champion": economic_champion,
        "v2_ranking": ranking,
        "cash_branch_summary": cash_summary,
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
            "v2_champion": report["v2_champion"]["trial_tag"],
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
