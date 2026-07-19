from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from daily_research.path_policy import seq100_2026_fold_comparison as comparison
from daily_research.path_policy.seq100_development import WORKSPACE_ROOT


BUILDER_VERSION = "seq100_compact_research_report_v1"
BASE_STUDY_ROOT = comparison.BASE_STUDY_ROOT
REPORT_ROOT = BASE_STUDY_ROOT / "analysis" / "unit_time_exit_20260719"
FINITE_ROOT = BASE_STUDY_ROOT / "analysis" / "finite_capital_20260719"
FRESHNESS_ROOT = BASE_STUDY_ROOT / "analysis" / "checkpoint_freshness_20260719"
COMPARISON_ROOT = comparison.ANALYSIS_ROOT
ARTIFACT_PATH = REPORT_ROOT / "artifact.json"
INTEGRITY_PATH = REPORT_ROOT / "artifact.integrity.json"
ARCHIVE_ROOT = REPORT_ROOT / "archive" / "pre_compaction_v1"

LEGACY_REPORT_FILES = {
    "artifact.json": ARTIFACT_PATH,
    "chart_map.json": REPORT_ROOT / "chart_map.json",
    "artifact_append_audit.json": COMPARISON_ROOT / "artifact_append_audit.json",
    "freshness_report_extension_validation.json": FRESHNESS_ROOT
    / "report_extension_validation.json",
    "scripts/unit_time_build_report_artifact.py": REPORT_ROOT
    / "build_report_artifact.py",
    "scripts/finite_extend_report_artifact.py": FINITE_ROOT
    / "extend_report_artifact.py",
    "scripts/finite_extend_top1_report.py": FINITE_ROOT / "extend_top1_report.py",
    "scripts/freshness_extend_report_artifact.py": FRESHNESS_ROOT
    / "extend_report_artifact.py",
    "scripts/append_seq100_2026_artifact.py": WORKSPACE_ROOT
    / "daily_research"
    / "path_policy"
    / "append_seq100_2026_artifact.py",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _one_row(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    selected = frame.loc[mask].drop_duplicates()
    if len(selected) != 1:
        raise ValueError(f"expected one source row for {label}, got {len(selected)}")
    return selected.iloc[0]


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=5e-10):
        raise ValueError(f"headline source drift for {label}: {observed} != {expected}")


def _source(
    *,
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    tables: Sequence[str],
    filters: Sequence[str],
    definitions: Sequence[str],
    executed_at: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": list(tables),
            "filters": list(filters),
            "metric_definitions": list(definitions),
        },
    }


def _load_report_data() -> dict[str, Any]:
    horizon = pd.read_csv(REPORT_ROOT / "profile_horizon_curve.csv")
    slots = pd.read_csv(FINITE_ROOT / "preferred_fixed_slot_curve.csv")
    top1 = pd.read_csv(FINITE_ROOT / "top1_metrics.csv")
    cohort = pd.read_csv(FINITE_ROOT / "top1_cohort_metrics.csv")
    summary = _read_json(COMPARISON_ROOT / "comparison_summary.json")
    if int(summary.get("schema_version", 0)) != 2:
        raise ValueError("compact report requires the migrated 2026 integrity v2 summary")

    legal_top3 = _one_row(
        slots,
        slots["profile"].eq("legal_flat_baseline")
        & slots["policy"].eq("fixed_d4")
        & slots["slot_count"].eq(6)
        & slots["cost_scenario"].eq("base"),
        "Legal Top3 D4/6",
    )
    structured_top3 = _one_row(
        slots,
        slots["profile"].eq("structured_joint_turnover")
        & slots["policy"].eq("fixed_d16")
        & slots["slot_count"].eq(3)
        & slots["cost_scenario"].eq("base"),
        "Structured Top3 D16/3",
    )
    top1_base = top1[
        top1["cost_scenario"].eq("base") & ~top1["allow_pyramiding"].astype(bool)
    ]
    legal_top1 = _one_row(
        top1_base,
        top1_base["profile"].eq("legal_flat_baseline")
        & top1_base["policy"].eq("fixed_d10")
        & top1_base["slot_count"].eq(3),
        "Legal Top1 D10/3",
    )
    structured_top1 = _one_row(
        top1_base,
        top1_base["profile"].eq("structured_joint_turnover")
        & top1_base["policy"].eq("fixed_d18")
        & top1_base["slot_count"].eq(1),
        "Structured Top1 D18/1",
    )
    structured_stop = _one_row(
        top1_base,
        top1_base["profile"].eq("structured_joint_turnover")
        & top1_base["policy"].eq("stop_intraday_tp15_sl10")
        & top1_base["slot_count"].eq(1),
        "Structured Top1 +15/-10",
    )
    for observed, expected, label in (
        (
            legal_top3["signal_period_cagr_trading_days"],
            0.6061182316778575,
            "Legal D4 CAGR",
        ),
        (
            legal_top3["signal_period_maximum_drawdown"],
            -0.338244417686396,
            "Legal D4 drawdown",
        ),
        (
            structured_top3["signal_period_cagr_trading_days"],
            0.9515516697428784,
            "Structured D16 CAGR",
        ),
        (
            structured_top3["signal_period_maximum_drawdown"],
            -0.385556920430255,
            "Structured D16 drawdown",
        ),
        (legal_top1["signal_period_cagr_trading_days"], 1.1610664553229668, "Legal Top1"),
        (
            structured_top1["signal_period_cagr_trading_days"],
            1.3152838506898172,
            "Structured Top1",
        ),
        (
            structured_stop["signal_period_cagr_trading_days"],
            1.7070757337939164,
            "Structured stop",
        ),
    ):
        _assert_close(float(observed), float(expected), label)
    if int(structured_stop["closed_trade_count"]) != 42:
        raise ValueError("Structured Top1 stop trade count drifted")
    if (
        int(summary["signal_date_count"]) != 48
        or int(summary["candidate_count"]) != 143_840
        or int(summary["symbol_count"]) != 3_034
    ):
        raise ValueError("2026 report scope drifted")

    slot_rows = slots[
        slots["profile"].isin(comparison.PROFILE_ORDER)
        & slots["cost_scenario"].eq("base")
    ].copy()
    slot_rows["profile_label"] = slot_rows["profile"].map(
        {
            "legal_flat_baseline": "Legal flat / D4",
            "structured_joint_turnover": "Structured joint / D16",
        }
    )
    slot_rows["slot_label"] = slot_rows["slot_count"].astype(int).astype(str)
    slot_rows = slot_rows.rename(
        columns={
            "signal_period_cagr_trading_days": "cagr",
            "signal_period_total_return": "total_return",
            "signal_period_maximum_drawdown": "maximum_drawdown",
            "mean_signal_capital_utilization": "capital_utilization",
            "closed_trade_count": "closed_trades",
        }
    )
    slot_rows = slot_rows[
        [
            "profile",
            "profile_label",
            "policy",
            "slot_count",
            "slot_label",
            "cagr",
            "total_return",
            "maximum_drawdown",
            "capital_utilization",
            "closed_trades",
        ]
    ]

    equal_weight = cohort[cohort["period"].eq("three_year_equal_weight")].copy()
    alpha_rows: list[dict[str, Any]] = []
    for row in equal_weight.itertuples(index=False):
        profile_label = (
            "Legal flat"
            if row.profile == "legal_flat_baseline"
            else "Structured joint"
        )
        alpha_rows.extend(
            [
                {
                    "profile": row.profile,
                    "profile_label": profile_label,
                    "selection": "Top1",
                    "executable_alpha": float(row.top1_base_alpha),
                    "selected_return": float(row.top1_selected_base),
                    "universe_return": float(row.top1_universe_base),
                    "opportunity_alpha": float(row.top1_opportunity_alpha),
                },
                {
                    "profile": row.profile,
                    "profile_label": profile_label,
                    "selection": "Top3",
                    "executable_alpha": float(row.top3_base_alpha),
                    "selected_return": float(row.top3_selected_base),
                    "universe_return": float(row.top3_universe_base),
                    "opportunity_alpha": float(row.top3_opportunity_alpha),
                },
            ]
        )

    vintage = pd.DataFrame(summary["metrics"])
    vintage["profile_label"] = vintage["profile"].map(
        {
            "legal_flat_baseline": "Legal flat",
            "structured_joint_turnover": "Structured joint",
        }
    )
    vintage["checkpoint_label"] = vintage["checkpoint_vintage"].astype(int).astype(str)
    vintage["profile_vintage"] = (
        vintage["profile_label"] + " / " + vintage["checkpoint_label"]
    )
    vintage["gate_status"] = "未通过"

    strategy_rows = pd.DataFrame(
        [
            {
                "profile_selection": "Legal / Top3",
                "policy": "固定 D4",
                "slots": 6,
                "cagr": float(legal_top3["signal_period_cagr_trading_days"]),
                "maximum_drawdown": float(
                    legal_top3["signal_period_maximum_drawdown"]
                ),
                "closed_trades": int(legal_top3["closed_trade_count"]),
                "role": "较稳健研究基准",
            },
            {
                "profile_selection": "Structured / Top3",
                "policy": "固定 D16",
                "slots": 3,
                "cagr": float(structured_top3["signal_period_cagr_trading_days"]),
                "maximum_drawdown": float(
                    structured_top3["signal_period_maximum_drawdown"]
                ),
                "closed_trades": int(structured_top3["closed_trade_count"]),
                "role": "收益优先研究基准",
            },
            {
                "profile_selection": "Legal / Top1",
                "policy": "固定 D10",
                "slots": 3,
                "cagr": float(legal_top1["signal_period_cagr_trading_days"]),
                "maximum_drawdown": float(
                    legal_top1["signal_period_maximum_drawdown"]
                ),
                "closed_trades": int(legal_top1["closed_trade_count"]),
                "role": "高集中度变体",
            },
            {
                "profile_selection": "Structured / Top1",
                "policy": "固定 D18",
                "slots": 1,
                "cagr": float(structured_top1["signal_period_cagr_trading_days"]),
                "maximum_drawdown": float(
                    structured_top1["signal_period_maximum_drawdown"]
                ),
                "closed_trades": int(structured_top1["closed_trade_count"]),
                "role": "高集中度变体",
            },
        ]
    )
    return {
        "horizon": horizon,
        "slot_rows": slot_rows,
        "alpha_rows": pd.DataFrame(alpha_rows),
        "vintage": vintage,
        "strategy_rows": strategy_rows,
        "legal_top3": legal_top3,
        "structured_top3": structured_top3,
        "legal_top1": legal_top1,
        "structured_top1": structured_top1,
        "structured_stop": structured_stop,
        "summary": summary,
    }


def build_artifact_payload() -> dict[str, Any]:
    data = _load_report_data()
    generated_at = str(data["summary"]["completed_at"])
    source_paths = {
        "horizon": "daily_research/output/path_policy/studies/seq100_legal_structured_path_rolling_2023_2025_v1/analysis/unit_time_exit_20260719/profile_horizon_curve.csv",
        "finite": "daily_research/output/path_policy/studies/seq100_legal_structured_path_rolling_2023_2025_v1/analysis/finite_capital_20260719/preferred_fixed_slot_curve.csv",
        "finite_all": "daily_research/output/path_policy/studies/seq100_legal_structured_path_rolling_2023_2025_v1/analysis/finite_capital_20260719/portfolio_metrics.csv",
        "top1": "daily_research/output/path_policy/studies/seq100_legal_structured_path_rolling_2023_2025_v1/analysis/finite_capital_20260719/top1_metrics.csv",
        "top1_cohort": "daily_research/output/path_policy/studies/seq100_legal_structured_path_rolling_2023_2025_v1/analysis/finite_capital_20260719/top1_cohort_metrics.csv",
        "vintage": "daily_research/output/path_policy/studies/seq100_legal_structured_path_2026_fold_v1/analysis/checkpoint_vintage_2026_20260719/comparison_summary.json",
        "vintage_metrics": "daily_research/output/path_policy/studies/seq100_legal_structured_path_2026_fold_v1/analysis/checkpoint_vintage_2026_20260719/vintage_metrics.csv",
    }
    sources = [
        _source(
            source_id="horizon_source",
            label="Seq100 固定退出持有期曲线",
            path=source_paths["horizon"],
            sql=f"SELECT * FROM read_csv_auto('{source_paths['horizon']}') ORDER BY exit_day",
            description="比较两个 profile 在 D2-D60 固定退出下的 Top3 单位时间收益。",
            tables=[source_paths["horizon"]],
            filters=["validation years: 2023-2025", "Top3", "legal exits D2-D60"],
            definitions=[
                "单位时间收益为三年年份等权的绝对对数收益，按实际持有交易日换算为 bps。",
                "固定退出日使用相同 next-open entry、费用和不可卖递延合同。",
            ],
            executed_at=generated_at,
        ),
        _source(
            source_id="finite_capital_source",
            label="Top3 有限资金连续账户",
            path=source_paths["finite"],
            sql=f"SELECT * FROM read_csv_auto('{source_paths['finite']}') WHERE cost_scenario='base' ORDER BY profile, slot_count",
            description="100 万元单账户按日回放 Top3 固定退出策略。",
            tables=[source_paths["finite"], source_paths["finite_all"]],
            filters=[
                "signals: 2023-01-03 through 2025-12-31",
                "no pyramiding and no Top4 replacement",
                "slot counts: 3, 6, 12, 24, 48",
            ],
            definitions=[
                "交易日 CAGR = exp(log(期末权益/初始权益) × 252 / 727) - 1。",
                "最大回撤由连续账户逐日盯市权益相对历史高水位计算。",
                "资金利用率 = 1 - 现金/账户权益。",
            ],
            executed_at=generated_at,
        ),
        _source(
            source_id="top1_source",
            label="Top1 独立 cohort 与连续账户",
            path=source_paths["top1"],
            sql=f"SELECT * FROM read_csv_auto('{source_paths['top1']}') WHERE cost_scenario='base' AND allow_pyramiding=false",
            description="读取每日第一名的 cohort alpha 与固定退出连续账户结果。",
            tables=[source_paths["top1"], source_paths["top1_cohort"]],
            filters=["2023-2025", "daily selection count: 1", "no pyramiding"],
            definitions=[
                "Executable alpha = 每日 selected realized return - 同日可执行候选池 return。",
                "Top1 连续账户 CAGR 与 Top3 cohort alpha 回答不同问题，不可直接互换。",
            ],
            executed_at=generated_at,
        ),
        _source(
            source_id="vintage_source",
            label="2026 四年代 checkpoint 统一评价",
            path=source_paths["vintage"],
            sql=f"SELECT * FROM read_csv_auto('{source_paths['vintage_metrics']}') ORDER BY profile, checkpoint_vintage",
            description="比较两个 profile 的 2023-2026 checkpoint 在同一 2026 实现收益窗口上的指标。",
            tables=[source_paths["vintage"], source_paths["vintage_metrics"]],
            filters=[
                "signal window: 2026-01-05 through 2026-03-19",
                "48 dates, 143,840 candidates, 3,034 symbols",
                "score-only dates after 2026-03-19 excluded",
            ],
            definitions=[
                "Rank IC 为逐日横截面 Spearman 相关的均值。",
                "Path MAE 为 60 日 OHLC 相对价格路径的平均绝对误差。",
                "Freshness gate 要求 Top3、广度以及排名/路径支持同时成立。",
            ],
            executed_at=generated_at,
        ),
        _source(
            source_id="strategy_source",
            label="Top3 与 Top1 主要连续账户策略",
            path=source_paths["finite"],
            sql=(
                "SELECT profile, 'Top3' AS selection, policy, slot_count, "
                "signal_period_cagr_trading_days, signal_period_maximum_drawdown, closed_trade_count "
                f"FROM read_csv_auto('{source_paths['finite']}') WHERE cost_scenario='base' "
                "UNION ALL "
                "SELECT profile, 'Top1' AS selection, policy, slot_count, "
                "signal_period_cagr_trading_days, signal_period_maximum_drawdown, closed_trade_count "
                f"FROM read_csv_auto('{source_paths['top1']}') "
                "WHERE cost_scenario='base' AND allow_pyramiding=false"
            ),
            description="把两个 Top3 固定退出基准与两个 Top1 集中账户放到统一字段中比较。",
            tables=[source_paths["finite"], source_paths["top1"]],
            filters=[
                "Top3: Legal D4/6 and Structured D16/3",
                "Top1: Legal D10/3 and Structured D18/1",
                "base costs and no pyramiding",
            ],
            definitions=[
                "表内 CAGR、最大回撤和平仓笔数均来自各自完整连续账户。",
                "Top1 与 Top3 的每日选择数量不同，表格用于策略定位而非因果归因。",
            ],
            executed_at=generated_at,
        ),
    ]

    legal_top3 = data["legal_top3"]
    structured_top3 = data["structured_top3"]
    legal_top1 = data["legal_top1"]
    structured_top1 = data["structured_top1"]
    stop = data["structured_stop"]
    summary = data["summary"]
    cards = [
        {
            "id": "legal_top3_card",
            "dataset": "finite_headline",
            "sourceId": "finite_capital_source",
            "description": "Legal fixed D4，100 万元连续账户，6 个槽位。",
            "metrics": [
                {"label": "Legal Top3 CAGR", "field": "legal_cagr", "format": "percent"},
                {"label": "最大回撤", "field": "legal_drawdown", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "structured_top3_card",
            "dataset": "finite_headline",
            "sourceId": "finite_capital_source",
            "description": "Structured fixed D16，100 万元连续账户，3 个槽位。",
            "metrics": [
                {"label": "Structured Top3 CAGR", "field": "structured_cagr", "format": "percent"},
                {"label": "最大回撤", "field": "structured_drawdown", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "top1_card",
            "dataset": "top1_headline",
            "sourceId": "top1_source",
            "description": "Structured fixed D18/1 是高集中度样本内峰值，不是 Top3 的排名升级。",
            "metrics": [
                {"label": "Structured Top1 CAGR", "field": "structured_cagr", "format": "percent"},
                {"label": "Legal Top1 CAGR", "field": "legal_cagr", "format": "percent"},
                {"label": "Structured 回撤", "field": "structured_drawdown", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "freshness_card",
            "dataset": "freshness_headline",
            "sourceId": "vintage_source",
            "description": "0 表示四年代 freshness gate 未通过；2026 checkpoint 仅是研究 fold。",
            "metrics": [
                {"label": "2026 gate（1=通过）", "field": "gate_pass", "format": "number"},
                {"label": "信号日", "field": "signal_dates", "format": "number"},
                {"label": "Score 覆盖", "field": "score_coverage", "format": "percent"},
            ],
        },
    ]
    charts = [
        {
            "id": "profile_efficiency_chart",
            "title": "两个 profile 的固定退出资本效率",
            "subtitle": "D2-D60；Legal 在短端达峰，Structured 在 D14-D18 形成中期平台。",
            "type": "line",
            "dataset": "profile_horizon_curve",
            "sourceId": "horizon_source",
            "encodings": {
                "x": {"field": "exit_day", "type": "quantitative", "label": "固定退出日"},
                "y": {
                    "fields": [
                        "top3_log_bps_per_day_legal_flat",
                        "top3_log_bps_per_day_structured",
                    ],
                    "type": "quantitative",
                    "label": "绝对 bps/占用交易日",
                },
                "tooltip": [
                    {"field": "exit_day", "label": "退出日"},
                    {"field": "top3_log_bps_per_day_legal_flat", "label": "Legal flat"},
                    {"field": "top3_log_bps_per_day_structured", "label": "Structured joint"},
                    {"field": "top3_alpha_legal_flat", "label": "Legal alpha", "format": "percent"},
                    {"field": "top3_alpha_structured", "label": "Structured alpha", "format": "percent"},
                ],
            },
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "endpoints"},
            "settings": {"showPoints": "never"},
            "layout": "full",
        },
        {
            "id": "finite_slot_chart",
            "title": "Top3 固定退出的槽位与 CAGR",
            "subtitle": "2023-2025 连续账户；槽位增加后每笔资金被稀释。",
            "type": "bar",
            "dataset": "finite_slot_curve",
            "sourceId": "finite_capital_source",
            "encodings": {
                "x": {"field": "slot_label", "type": "ordinal", "label": "持仓槽位"},
                "y": {"field": "cagr", "type": "quantitative", "format": "percent", "label": "交易日 CAGR"},
                "color": {"field": "profile_label", "type": "nominal", "label": "策略"},
                "tooltip": [
                    {"field": "total_return", "format": "percent", "label": "总收益"},
                    {"field": "maximum_drawdown", "format": "percent", "label": "最大回撤"},
                    {"field": "capital_utilization", "format": "percent", "label": "资金利用率"},
                    {"field": "closed_trades", "label": "平仓笔数"},
                ],
            },
            "valueFormat": "percent",
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "all"},
            "settings": {"groupMode": "grouped"},
            "layout": "full",
        },
        {
            "id": "top1_vs_top3_chart",
            "title": "Top1 与 Top3 的三年等权 executable alpha",
            "subtitle": "两个 profile 的 Top1 均为正 alpha，但都略低于同 profile Top3。",
            "type": "bar",
            "dataset": "top1_vs_top3_alpha",
            "sourceId": "top1_source",
            "encodings": {
                "x": {"field": "profile_label", "type": "nominal", "label": "Profile"},
                "y": {"field": "executable_alpha", "type": "quantitative", "format": "percent", "label": "Executable alpha"},
                "color": {"field": "selection", "type": "nominal", "label": "每日选择"},
                "tooltip": [
                    {"field": "selected_return", "format": "percent", "label": "Selected return"},
                    {"field": "universe_return", "format": "percent", "label": "Universe return"},
                    {"field": "opportunity_alpha", "format": "percent", "label": "Opportunity alpha"},
                ],
            },
            "valueFormat": "percent",
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "all"},
            "settings": {"groupMode": "grouped"},
            "layout": "full",
        },
        {
            "id": "vintage_top3_chart",
            "title": "四年代 checkpoint 的 2026 Top3 executable alpha",
            "subtitle": "相同 48 个信号日与 143,840 个候选；2026 未同时超过三个旧版本。",
            "type": "bar",
            "dataset": "vintage_metrics",
            "sourceId": "vintage_source",
            "encodings": {
                "x": {"field": "checkpoint_label", "type": "ordinal", "label": "Checkpoint 年代"},
                "y": {"field": "top3_base_alpha", "type": "quantitative", "format": "percent", "label": "Top3 executable alpha"},
                "color": {"field": "profile_label", "type": "nominal", "label": "Profile"},
                "tooltip": [
                    {"field": "top3_stress_alpha", "format": "percent", "label": "Stress alpha"},
                    {"field": "rank_ic", "label": "Rank IC"},
                    {"field": "path_mae", "format": "percent", "label": "Path MAE"},
                    {"field": "candidate_score_coverage", "format": "percent", "label": "Score 覆盖"},
                ],
            },
            "valueFormat": "percent",
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "all"},
            "settings": {"groupMode": "grouped"},
            "layout": "full",
        },
    ]
    tables = [
        {
            "id": "strategy_table",
            "title": "四个主要连续账户策略",
            "subtitle": "Top3 是研究基准；Top1 是集中度更高的变体，均为同区间样本内结果。",
            "dataset": "strategy_comparison",
            "sourceId": "strategy_source",
            "defaultSort": {"field": "profile_selection", "direction": "asc"},
            "density": "spacious",
            "columns": [
                {"field": "profile_selection", "label": "Profile / selection", "type": "text"},
                {"field": "policy", "label": "退出", "type": "text"},
                {"field": "slots", "label": "槽位", "format": "number"},
                {"field": "cagr", "label": "CAGR", "format": "percent"},
                {"field": "maximum_drawdown", "label": "最大回撤", "format": "percent", "movement": True},
                {"field": "closed_trades", "label": "平仓笔数", "format": "number"},
                {"field": "role", "label": "研究定位", "type": "text"},
            ],
        },
        {
            "id": "vintage_table",
            "title": "八组 checkpoint 的统一 2026 指标",
            "subtitle": "两个 profile × 四年代；所有行共享候选、标签、执行和费用语义。",
            "dataset": "vintage_metrics",
            "sourceId": "vintage_source",
            "defaultSort": {"field": "profile_vintage", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "profile_vintage", "label": "Profile / checkpoint", "type": "text"},
                {"field": "rank_ic", "label": "Rank IC", "format": "number"},
                {"field": "path_mae", "label": "Path MAE", "format": "percent"},
                {"field": "top1_base_alpha", "label": "Top1 alpha", "format": "percent"},
                {"field": "top3_base_alpha", "label": "Top3 alpha", "format": "percent"},
                {"field": "top5_base_alpha", "label": "Top5 alpha", "format": "percent"},
                {"field": "top10_base_alpha", "label": "Top10 alpha", "format": "percent"},
                {"field": "top3_opportunity_alpha", "label": "Top3 opportunity", "format": "percent"},
                {"field": "exit_regret", "label": "Exit regret", "format": "percent"},
                {"field": "candidate_score_coverage", "label": "Score 覆盖", "format": "percent"},
                {"field": "execution_return_coverage", "label": "Execution 覆盖", "format": "percent"},
                {"field": "gate_status", "label": "Freshness", "type": "text"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Seq100 当前研究结论", "layout": "full"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 技术结论\n\n"
                "- **退出期限应由 profile 区分。** Legal flat 的资本效率集中在短端，Structured joint 在中期平台更强；固定 D4/D16 仍比模型计划日更适合作为研究基准。\n"
                "- **有限资金支持 Top3 基准，也允许 Top1 集中变体。** Legal D4/6 槽位为 **60.61% CAGR、-33.82% 回撤**；Structured D16/3 为 **95.16%、-38.56%**。Top1 能放大复利，但没有提升 cohort 排名质量。\n"
                "- **2026 数据更新没有通过 freshness promotion。** 两个 2026 checkpoint 都未同时超过三个旧版本，也没有获得完整排名和路径支持；它们仅是用户授权的扩展实验，不是部署模型。\n"
                "- **所有收益都是研究回测。** 2023-2025 的策略和槽位在同一区间选择；2026 只有 48 个信号日，不年化、不生成正式 moving-block 区间。"
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "exit_section",
            "type": "markdown",
            "sourceId": "horizon_source",
            "layout": "full",
            "body": (
                "## Legal 偏短持有，Structured 偏中期持有\n\n"
                "固定退出曲线显示两个 profile 不应共用一个持有期：Legal flat 在 D4 附近达到短端效率峰值，Structured joint 的优势则延伸至 D14-D18。模型计划日仍有远端坍缩，因此当前应把固定窗口作为退出基准，把路径计划日保留为待改进预测输出。"
            ),
        },
        {"id": "profile_efficiency_chart_block", "type": "chart", "chartId": "profile_efficiency_chart", "layout": "full"},
        {
            "id": "finite_capital_section",
            "type": "markdown",
            "sourceId": "finite_capital_source",
            "layout": "full",
            "body": (
                "## 有限资金下，少量匹配槽位优于盲目分散\n\n"
                "Legal D4 在 6 槽位达到 **60.61% CAGR、-33.82% 最大回撤**；Structured D16 在 3 槽位达到 **95.16%、-38.56%**。增加槽位会降低单笔资金并快速稀释复利。下表把 Top3 基准与 Top1 集中变体放在同一口径下，但不把样本内最高 CAGR 解释为未来承诺。"
            ),
        },
        {"id": "finite_slot_chart_block", "type": "chart", "chartId": "finite_slot_chart", "layout": "full"},
        {"id": "strategy_table_block", "type": "table", "tableId": "strategy_table", "layout": "full"},
        {
            "id": "top1_section",
            "type": "markdown",
            "sourceId": "top1_source",
            "layout": "full",
            "body": (
                "## Top1 是集中表达，不是更好的排名器\n\n"
                "三年年份等权下，Legal Top1 executable alpha 为 **4.40%**，低于 Top3 的 **4.59%**；Structured Top1 为 **5.37%**，低于 Top3 的 **5.71%**。连续账户中 Legal D10/3 和 Structured D18/1 可达到 **116.11%** 与 **131.53% CAGR**，代价是更少交易、更高集中度和更不稳定的回撤。"
            ),
        },
        {"id": "top1_vs_top3_chart_block", "type": "chart", "chartId": "top1_vs_top3_chart", "layout": "full"},
        {
            "id": "freshness_section",
            "type": "markdown",
            "sourceId": "vintage_source",
            "layout": "full",
            "body": (
                "## 2026 checkpoint 未形成一致增益\n\n"
                "八组结果共享 **2026-01-05 至 2026-03-19** 的 48 个信号日、143,840 个候选和 3,034 个 symbol，score 与 execution coverage 均为 100%。2026 checkpoint 在 Top3、Top1/5/10 广度以及排名/路径支持上没有同时战胜三个旧版本，因此 freshness gate 失败，不支持 promotion 或实盘替换。"
            ),
        },
        {"id": "vintage_top3_chart_block", "type": "chart", "chartId": "vintage_top3_chart", "layout": "full"},
        {"id": "vintage_table_block", "type": "table", "tableId": "vintage_table", "layout": "full"},
        {
            "id": "definitions_section",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 研究口径与方法\n\n"
                "### 模型与候选\n"
                "输入为过去 100×32 日线与状态特征，输出未来 60 日 OHLC 路径；两个 profile 均使用 seed 7。评价候选不按未来标签或 entry fill 过滤，每个 checkpoint 使用自身训练时 normalization。\n\n"
                "### 退出与收益\n"
                "合法退出域为 D2-D60，并列取最早合法日。Executable alpha 是 selected 净实现收益减同日可执行候选池收益；有限资金账户使用 100 万元、next-open entry、真实现金/手数、费用、不可卖递延与逐日盯市。\n\n"
                "### 完整性\n"
                "哈希只确认数据、候选、模型、评价合同和结果身份一致，不证明模型质量。默认报告不展示哈希明细，完整性记录保存在 sidecar 与归档中。"
            ),
        },
        {
            "id": "limitations_section",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 限制与稳健性边界\n\n"
                "- 2023-2025 的退出日、槽位和止盈止损均在同一历史区间扫描，存在选择偏差。\n"
                f"- Structured Top1 +15%/-10% 的 **{float(stop['signal_period_cagr_trading_days']):.2%} CAGR** 只有 **{int(stop['closed_trade_count'])} 笔交易**，不作为稳健推荐。\n"
                "- 回测尚未建模大资金容量、冲击成本和组合级行业暴露。\n"
                "- 2026 仅 48 个高度重叠信号日，不能支撑正式 moving-block 置信区间，也不能年化为有限资金 CAGR。"
            ),
        },
        {
            "id": "next_steps_section",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 下一步\n\n"
                "1. 将 Legal D4/6 与 Structured D16/3 保留为研究基准，后续新模型必须同时改善收益、回撤和退出稳定性。\n"
                "2. 在更长的 2026 实现收益窗口形成后重新做四年代配对评价，不提前 promotion 2026 checkpoint。\n"
                "3. 单独研究模型计划日的远端坍缩，再决定是否继续使用路径预测退出。\n\n"
                "### 待回答问题\n"
                "更长窗口能否保持 2024 checkpoint 的优势？固定退出的收益是否来自可重复的行业/状态结构？Top1 的容量与单股贡献上限应如何进入组合约束？\n\n"
                "压缩前的完整 73-block 报告和增量脚本保存在同目录 `archive/pre_compaction_v1`，用于审计而非默认阅读。"
            ),
        },
    ]
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "finite_headline": [
                {
                    "legal_cagr": float(legal_top3["signal_period_cagr_trading_days"]),
                    "legal_drawdown": float(legal_top3["signal_period_maximum_drawdown"]),
                    "structured_cagr": float(structured_top3["signal_period_cagr_trading_days"]),
                    "structured_drawdown": float(structured_top3["signal_period_maximum_drawdown"]),
                }
            ],
            "top1_headline": [
                {
                    "legal_cagr": float(legal_top1["signal_period_cagr_trading_days"]),
                    "legal_drawdown": float(legal_top1["signal_period_maximum_drawdown"]),
                    "structured_cagr": float(structured_top1["signal_period_cagr_trading_days"]),
                    "structured_drawdown": float(structured_top1["signal_period_maximum_drawdown"]),
                }
            ],
            "freshness_headline": [
                {
                    "gate_pass": int(bool(summary["overall_2026_freshness_gate_pass"])),
                    "signal_dates": int(summary["signal_date_count"]),
                    "candidate_count": int(summary["candidate_count"]),
                    "score_coverage": min(
                        float(row["candidate_score_coverage"])
                        for row in summary["metrics"]
                    ),
                }
            ],
            "profile_horizon_curve": _records(data["horizon"]),
            "finite_slot_curve": _records(data["slot_rows"]),
            "strategy_comparison": _records(data["strategy_rows"]),
            "top1_vs_top3_alpha": _records(data["alpha_rows"]),
            "vintage_metrics": _records(data["vintage"]),
        },
    }
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Seq100 当前研究结论",
        "description": "退出期限、有限资金、Top1 集中度与 checkpoint freshness 的精简技术报告。",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
        "package_info": {
            "builder_version": BUILDER_VERSION,
            "profiles": list(comparison.PROFILE_ORDER),
            "validation_years": [2023, 2024, 2025, 2026],
            "seed": comparison.SEED,
        },
    }


def validate_compact_artifact(artifact: Mapping[str, Any]) -> dict[str, int]:
    if set(artifact) != {"surface", "manifest", "snapshot", "sources", "package_info"}:
        raise ValueError("compact artifact top-level schema drifted")
    if str(artifact["surface"]) != "report":
        raise ValueError("compact artifact is not a report")
    manifest = dict(artifact["manifest"])
    snapshot = dict(artifact["snapshot"])
    collections = {
        name: list(manifest.get(name, []) or [])
        for name in ("cards", "charts", "tables", "sources", "blocks")
    }
    counts = {name: len(values) for name, values in collections.items()}
    if counts["blocks"] > 16 or counts["charts"] > 4 or counts["tables"] > 2 or counts["cards"] > 4:
        raise ValueError(f"compact report count limit exceeded: {counts}")
    if counts != {"cards": 4, "charts": 4, "tables": 2, "sources": 5, "blocks": 16}:
        raise ValueError(f"compact report shape drifted: {counts}")
    ids: dict[str, set[str]] = {}
    for name, values in collections.items():
        current = [str(item["id"]) for item in values]
        if len(current) != len(set(current)):
            raise ValueError(f"duplicate {name} ids")
        ids[name] = set(current)
    datasets = dict(snapshot.get("datasets", {}) or {})
    source_ids = ids["sources"]
    for card in collections["cards"]:
        if str(card["dataset"]) not in datasets or str(card["sourceId"]) not in source_ids:
            raise ValueError(f"invalid card reference: {card['id']}")
    for chart in collections["charts"]:
        if str(chart["dataset"]) not in datasets or str(chart["sourceId"]) not in source_ids:
            raise ValueError(f"invalid chart reference: {chart['id']}")
    for table in collections["tables"]:
        if str(table["dataset"]) not in datasets or str(table["sourceId"]) not in source_ids:
            raise ValueError(f"invalid table reference: {table['id']}")
    for index, block in enumerate(collections["blocks"]):
        block_type = str(block["type"])
        if block_type == "chart" and str(block["chartId"]) not in ids["charts"]:
            raise ValueError(f"invalid chart block reference: {block['id']}")
        if block_type == "table" and str(block["tableId"]) not in ids["tables"]:
            raise ValueError(f"invalid table block reference: {block['id']}")
        if block_type == "metric-strip" and not set(block["cardIds"]).issubset(ids["cards"]):
            raise ValueError(f"invalid metric block reference: {block['id']}")
        if block_type == "chart":
            previous = collections["blocks"][index - 1] if index else {}
            if str(previous.get("type", "")) != "markdown":
                raise ValueError(f"chart lacks adjacent narrative: {block['id']}")
    first = collections["blocks"][0]
    if str(first.get("body", "")) != f"# {manifest['title']}":
        raise ValueError("report title block does not match manifest title")
    if list(artifact["sources"]) != collections["sources"]:
        raise ValueError("top-level and manifest sources differ")
    serialized = json.dumps(artifact, ensure_ascii=False, allow_nan=False)
    if "sha256" in serialized.lower():
        raise ValueError("hash details leaked into the reader-facing report")
    required_topics = ("退出", "有限资金", "Top1", "freshness")
    body = "\n".join(
        str(block.get("body", "")) for block in collections["blocks"] if block["type"] == "markdown"
    )
    if any(topic not in body for topic in required_topics):
        raise ValueError("compact report is missing a required research topic")
    return counts


def _verify_archive() -> dict[str, Any]:
    manifest_path = ARCHIVE_ROOT / "archive_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    for item in manifest["files"]:
        path = ARCHIVE_ROOT / str(item["archive_path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(item["file_size"])
            or _file_sha256(path) != str(item["sha256"])
        ):
            raise ValueError(f"report archive drifted: {path}")
    return manifest


def _archive_current_report() -> dict[str, Any]:
    if ARCHIVE_ROOT.is_dir():
        return _verify_archive()
    staging = ARCHIVE_ROOT.with_name(f".{ARCHIVE_ROOT.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for archive_path, source in LEGACY_REPORT_FILES.items():
        if not source.is_file():
            continue
        target = staging / archive_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "source_path": str(source.resolve()),
                "archive_path": archive_path,
                "file_size": int(target.stat().st_size),
                "sha256": _file_sha256(target),
            }
        )
    if not any(item["archive_path"] == "artifact.json" for item in files):
        raise FileNotFoundError("the accumulated report was not available for archival")
    manifest = {
        "schema_version": 1,
        "artifact_type": "seq100_report_pre_compaction_archive",
        "source_block_count": 73,
        "files": sorted(files, key=lambda item: str(item["archive_path"])),
    }
    _atomic_write_json(staging / "archive_manifest.json", manifest)
    ARCHIVE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, ARCHIVE_ROOT)
    return _verify_archive()


def _cleanup_legacy_output_scripts() -> None:
    tracked_append = LEGACY_REPORT_FILES["scripts/append_seq100_2026_artifact.py"].resolve()
    for archive_path, source in LEGACY_REPORT_FILES.items():
        resolved = source.resolve()
        if archive_path == "artifact.json" or resolved == tracked_append:
            continue
        if source.is_file():
            source.unlink()


def _write_chart_map(artifact: Mapping[str, Any]) -> None:
    questions = {
        "profile_efficiency_chart": "两个 profile 是否应使用同一固定退出日？",
        "finite_slot_chart": "有限资金下的槽位扩张如何影响 CAGR？",
        "top1_vs_top3_chart": "Top1 是否比 Top3 具有更高 cohort alpha？",
        "vintage_top3_chart": "2026 checkpoint 是否在统一窗口超过旧版本？",
    }
    rows = []
    for chart in list(dict(artifact["manifest"])["charts"]):
        rows.append(
            {
                "chart_id": str(chart["id"]),
                "question": questions[str(chart["id"])],
                "type": str(chart["type"]),
                "dataset": str(chart["dataset"]),
                "source_id": str(chart["sourceId"]),
                "palette_policy": "hard two-root cap plus neutrals",
                "delivery_surface": "Seq100 compact MCP report artifact",
            }
        )
    _atomic_write_json(
        REPORT_ROOT / "chart_map.json",
        {
            "schema_version": 1,
            "builder_version": BUILDER_VERSION,
            "charts": rows,
        },
    )


def verify_compact_report(artifact_path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    artifact = _read_json(artifact_path)
    counts = validate_compact_artifact(artifact)
    sidecar_path = artifact_path.with_name("artifact.integrity.json")
    sidecar = _read_json(sidecar_path)
    if (
        str(sidecar.get("builder_version", "")) != BUILDER_VERSION
        or str(sidecar.get("artifact_sha256", "")) != _file_sha256(artifact_path)
        or dict(sidecar.get("counts", {}) or {}) != counts
    ):
        raise ValueError("compact report integrity sidecar drifted")
    chart_map = _read_json(artifact_path.with_name("chart_map.json"))
    if (
        str(chart_map.get("builder_version", "")) != BUILDER_VERSION
        or len(list(chart_map.get("charts", []) or [])) != counts["charts"]
    ):
        raise ValueError("compact report chart map drifted")
    archive = _verify_archive()
    return {
        "status": "ok",
        "artifact": str(artifact_path),
        "counts": counts,
        "archive_file_count": len(archive["files"]),
    }


def build_compact_report(artifact_path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    if artifact_path != ARTIFACT_PATH.resolve():
        raise ValueError("compact report may only replace the registered Seq100 artifact")
    from daily_research.path_policy import seq100_integrity_v2 as integrity

    integrity.verify_research_integrity(
        output_root=COMPARISON_ROOT, include_report=False
    )
    archive = _archive_current_report()
    artifact = build_artifact_payload()
    counts = validate_compact_artifact(artifact)
    staging = artifact_path.with_name("artifact.json.staging")
    staging.write_bytes(_json_bytes(artifact))
    reloaded = _read_json(staging)
    if reloaded != artifact:
        staging.unlink(missing_ok=True)
        raise ValueError("staged compact artifact differs after JSON round-trip")
    validate_compact_artifact(reloaded)
    os.replace(staging, artifact_path)
    sidecar = {
        "schema_version": 1,
        "artifact_type": "seq100_compact_report_integrity",
        "builder_version": BUILDER_VERSION,
        "generated_at": str(artifact["manifest"]["generatedAt"]),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _file_sha256(artifact_path),
        "counts": counts,
    }
    _atomic_write_json(artifact_path.with_name("artifact.integrity.json"), sidecar)
    _cleanup_legacy_output_scripts()
    _write_chart_map(artifact)
    verified = verify_compact_report(artifact_path)
    return {
        **verified,
        "archive": str(ARCHIVE_ROOT),
        "archived_file_count": len(archive["files"]),
    }


__all__ = [
    "ARTIFACT_PATH",
    "BUILDER_VERSION",
    "build_artifact_payload",
    "build_compact_report",
    "validate_compact_artifact",
    "verify_compact_report",
]
