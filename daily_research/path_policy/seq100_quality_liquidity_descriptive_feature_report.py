from __future__ import annotations

"""Build the canonical portable-report artifact for the Seq100 descriptive audit."""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_quality_liquidity_descriptive_feature_audit as audit,
)

WORKSPACE_ROOT = audit.WORKSPACE_ROOT
STUDY_ID = audit.STUDY_ID
DEFAULT_AUDIT_ROOT = audit.DEFAULT_OUTPUT_ROOT
TITLE = "Seq100 质量流动性池训练前描述性关系审计"

FAMILY_LABELS = {
    "announcements": "公告",
    "daily_cross_sectional_technical": "日频横截面技术",
    "daily_price_volume_technical": "日频价量技术",
    "financial_statement_extensions": "财务扩展字段",
    "financial_statements": "财务报表",
    "industry_context": "行业上下文",
    "margin_detail": "个股融资融券",
    "margin_market": "市场融资融券",
    "market_state": "市场状态",
    "research_reports": "研报与盈利预测",
    "same_day_5m": "当日五分钟",
    "size_liquidity_and_status": "市值流动性与状态",
    "structured_financial_summary": "结构化财务摘要",
    "traditional_moneyflow": "传统资金流",
    "traditional_technical_indicators": "传统技术指标",
}
CLASS_LABELS = {
    "first_model_formal_family": "首版消融候选族",
    "availability_gated_family": "需可用性门控",
    "diagnostic_only": "仅诊断",
    "defer_from_initial_model": "首版暂缓",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(pd.notna(clean), None)
    return [dict(row) for row in clean.to_dict(orient="records")]


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    tables: Sequence[str],
    filters: Sequence[str],
    metric_definitions: Mapping[str, str],
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_source = {"id": source_id, "label": label, "path": path}
    reader = "read_parquet" if path.endswith(".parquet") else "read_json_auto"
    canonical_source = {
        "id": source_id,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM {reader}('{path}')",
            "description": description,
            "executed_at": generated_at,
            "tables_used": list(tables),
            "filters": list(filters),
            "metric_definitions": dict(metric_definitions),
        },
    }
    return manifest_source, canonical_source


def _headline_rows(
    manifest: Mapping[str, Any], recommendations: pd.DataFrame
) -> pd.DataFrame:
    counts = recommendations["classification"].value_counts()
    return pd.DataFrame(
        [
            {
                "common_rows": int(manifest["scope"]["common_support_row_count"]),
                "numeric_features": 628,
                "formal_families": int(counts.get("first_model_formal_family", 0)),
                "gated_families": int(counts.get("availability_gated_family", 0)),
                "excluded_rows": int(manifest["aggregation"]["excluded_minute_rows"]),
                "redundancy_pairs": int(
                    manifest["aggregation"]["redundancy_pair_rows"]
                ),
            }
        ]
    )


def _representative_features(
    relation: pd.DataFrame, stability: pd.DataFrame
) -> pd.DataFrame:
    all_history = relation[
        relation["period"].eq("all_history") & relation["horizon"].eq(10)
    ].copy()
    stable = stability[
        stability["horizon"].eq(10) & stability["stable_without_recent_reversal"]
    ][["feature", "same_direction_year_count"]]
    all_history = all_history.merge(stable, on="feature", how="inner")
    rows: list[dict[str, Any]] = []
    for family, group in all_history.groupby("analytic_family", sort=False):
        if family in {"market_state", "margin_market"}:
            score = group["high_minus_low_state_high_rate"].abs()
            evidence_dimension = "真实高状态率差"
        else:
            score = group["high_minus_low_top5_rate"].abs()
            evidence_dimension = "Top 5% 命中率差"
        if score.notna().sum() == 0:
            continue
        row = group.loc[score.idxmax()]
        rows.append(
            {
                "family": FAMILY_LABELS.get(str(family), str(family)),
                "analytic_family": str(family),
                "feature": str(row["feature"]),
                "preferred_tail": "高值"
                if row["high_minus_low_top5_rate"] >= 0
                else "低值",
                "evidence_dimension": evidence_dimension,
                "top5_rate_gap": float(row["high_minus_low_top5_rate"]),
                "preferred_top5_lift": float(
                    row["high_top5_lift"]
                    if row["high_minus_low_top5_rate"] >= 0
                    else row["low_top5_lift"]
                ),
                "preferred_mfe_median": float(
                    row["high_mfe_median"]
                    if row["high_minus_low_top5_rate"] >= 0
                    else row["low_mfe_median"]
                ),
                "preferred_mae_median": float(
                    row["high_mae_median"]
                    if row["high_minus_low_top5_rate"] >= 0
                    else row["low_mae_median"]
                ),
                "state_high_rate_gap": float(row["high_minus_low_state_high_rate"]),
                "same_direction_years": int(row["same_direction_year_count"]),
            }
        )
    return pd.DataFrame(rows)


def _market_state_rows(relation: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "market_all__drawdown60_mean",
        "market_all__ret1_dispersion",
        "market_all__vol20_mean",
        "market_all__ret20_mean",
    ]
    frame = relation[
        relation["period"].eq("all_history")
        & relation["horizon"].eq(10)
        & relation["feature"].isin(preferred)
    ].copy()
    labels = {
        "market_all__drawdown60_mean": "60日回撤均值",
        "market_all__ret1_dispersion": "1日收益离散度",
        "market_all__vol20_mean": "20日波动均值",
        "market_all__ret20_mean": "20日收益均值",
    }
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        for tail, mfe, mae in (
            ("高分位交易日", row.high_mfe_median, row.high_mae_median),
            ("低分位交易日", row.low_mfe_median, row.low_mae_median),
        ):
            rows.append(
                {
                    "feature": labels[str(row.feature)],
                    "feature_tail": tail,
                    "mfe_median": float(mfe),
                    "mae_median": float(mae),
                    "state_high_rate_gap": float(row.high_minus_low_state_high_rate),
                    "valid_count": int(row.valid_count),
                }
            )
    return pd.DataFrame(rows)


def _coverage_family_rows(
    coverage: pd.DataFrame,
    drift: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    all_history = coverage[coverage["period"].eq("all_history")]
    drift_flags = drift.groupby("analytic_family")[
        "source_or_distribution_drift_flag"
    ].sum()
    rows = (
        all_history.groupby("analytic_family")
        .agg(
            feature_count=("feature", "nunique"),
            minimum_coverage=("nonnull_rate", "min"),
            median_coverage=("nonnull_rate", "median"),
            low_coverage_features=(
                "nonnull_rate",
                lambda values: int((values < 0.8).sum()),
            ),
        )
        .reset_index()
    )
    rows["drift_flag_features"] = (
        rows["analytic_family"].map(drift_flags).fillna(0).astype(int)
    )
    rows = rows.merge(
        recommendations[["analytic_family", "classification"]],
        on="analytic_family",
        how="left",
    )
    rows["family"] = (
        rows["analytic_family"].map(FAMILY_LABELS).fillna(rows["analytic_family"])
    )
    rows["classification_label"] = rows["classification"].map(CLASS_LABELS)
    return rows.sort_values(["classification", "family"]).reset_index(drop=True)


def _source_availability_rows(source_state: pd.DataFrame) -> pd.DataFrame:
    def state_row(
        period: str, field: str, state: str, block: str | None = None
    ) -> pd.Series:
        mask = (
            source_state["period"].eq(period)
            & source_state["field"].eq(field)
            & source_state["state"].eq(state)
            & source_state["horizon"].eq(10)
        )
        if block is not None:
            mask &= source_state["block"].eq(block)
        rows = source_state[mask]
        if len(rows) != 1:
            raise RuntimeError(
                f"source_state_not_unique:{period}:{block}:{field}:{state}"
            )
        return rows.iloc[0]

    all_rows = 4_476_851
    recent_rows = 1_234_550
    moneyflow = state_row(
        "all_history", "coverage_state", "observed", "traditional_moneyflow_features"
    )
    technical = state_row(
        "all_history", "coverage_state", "observed", "tushare_technical_candidates"
    )
    margin_all = state_row(
        "all_history",
        "margin_eligibility_state",
        "eligible_observed",
        "margin_features",
    )
    margin_recent = state_row(
        "decision_years_2023_2025",
        "margin_eligibility_state",
        "eligible_observed",
        "margin_features",
    )
    contract_all = state_row(
        "all_history",
        "balance_contract_liabilities_field_state",
        "observed",
        "financial_statement_extensions",
    )
    contract_recent = state_row(
        "decision_years_2023_2025",
        "balance_contract_liabilities_field_state",
        "observed",
        "financial_statement_extensions",
    )
    return pd.DataFrame(
        [
            {
                "source_family": "技术因子",
                "all_history_observed_rate": technical.row_count / all_rows,
                "recent_observed_rate": 1.0,
                "semantics": "全历史 observed",
            },
            {
                "source_family": "传统资金流",
                "all_history_observed_rate": moneyflow.row_count / all_rows,
                "recent_observed_rate": 1.0,
                "semantics": "未知保留为 source_unavailable",
            },
            {
                "source_family": "融资融券资格",
                "all_history_observed_rate": margin_all.row_count / all_rows,
                "recent_observed_rate": margin_recent.row_count / recent_rows,
                "semantics": "标的扩张，不解释为普通缺数",
            },
            {
                "source_family": "合同负债字段",
                "all_history_observed_rate": contract_all.row_count / all_rows,
                "recent_observed_rate": contract_recent.row_count / recent_rows,
                "semantics": "早年制度性 unknown；特殊报表 not_applicable",
            },
            {
                "source_family": "研报预测来源状态",
                "all_history_observed_rate": 1.0,
                "recent_observed_rate": 1.0,
                "semantics": "v2 日级账本 complete；2021 已恢复",
            },
        ]
    )


def prepare_artifact(audit_root: Path = DEFAULT_AUDIT_ROOT) -> dict[str, Any]:
    manifest = _read_json(audit_root / "manifest.json")
    if manifest.get("status") != "completed":
        raise RuntimeError("descriptive_audit_not_completed")
    generated_at = _now()
    target_year = pd.read_parquet(audit_root / "target_summary_by_year.parquet")
    target_period = pd.read_parquet(audit_root / "target_summary_by_period.parquet")
    relation = pd.read_parquet(audit_root / "feature_relation_by_period.parquet")
    stability = pd.read_parquet(audit_root / "feature_stability.parquet")
    recommendations = pd.read_parquet(audit_root / "family_recommendations.parquet")
    coverage = pd.read_parquet(audit_root / "coverage_distribution_by_period.parquet")
    drift = pd.read_parquet(audit_root / "source_and_distribution_drift.parquet")
    source_state = pd.read_parquet(audit_root / "source_state_by_period.parquet")
    strata = pd.read_parquet(audit_root / "strata_by_period.parquet")
    pool = pd.read_parquet(audit_root / "pool_bias_by_period.parquet")

    target_year = target_year[
        ["year", "horizon", "valid_count", "mfe_median", "mfe_q95", "mae_median"]
    ].copy()
    target_year["horizon_label"] = target_year["horizon"].map({10: "10日", 20: "20日"})
    turnover = strata[
        strata["period"].eq("all_history")
        & strata["dimension"].eq("turnover_band")
        & strata["horizon"].eq(10)
    ][
        [
            "stratum",
            "row_count",
            "valid_count",
            "top1_rate",
            "top5_rate",
            "mfe_median",
            "mae_median",
        ]
    ].copy()
    turnover["turnover_band"] = turnover["stratum"].map(
        {
            "q1": "最低20%",
            "q2": "20–40%",
            "q3": "40–60%",
            "q4": "60–80%",
            "q5": "最高20%",
        }
    )
    turnover["order"] = turnover["stratum"].str[1:].astype(int)
    turnover = turnover.sort_values("order")

    representative = _representative_features(relation, stability)
    market_state = _market_state_rows(relation)
    family_rows = _coverage_family_rows(coverage, drift, recommendations)
    availability = _source_availability_rows(source_state)
    pool_rows = pool[pool["period"].eq("all_history")][
        [
            "stratum",
            "horizon",
            "row_count",
            "valid_count",
            "top1_rate",
            "top5_rate",
            "mfe_median",
            "mae_median",
            "state_high_rate",
        ]
    ].copy()
    pool_rows["support_group"] = pool_rows["stratum"].map(
        {"retained": "分钟完整", "excluded_missing_5m": "剩余分钟缺口"}
    )
    pool_rows["horizon_label"] = pool_rows["horizon"].map({10: "10日", 20: "20日"})
    headline = _headline_rows(manifest, recommendations)

    source_specs = [
        _source(
            "audit_contract",
            "审计 manifest 与预注册口径",
            "manifest.json",
            "读取冻结范围、不训练约束、输入指纹和输出 hash。",
            [
                "manifest.json",
                "seq100_quality_liquidity_descriptive_feature_audit.json",
            ],
            ["2011-2025 only", "2010 excluded from statistics", "2026 read count = 0"],
            {"common_rows": "sum of annual complete-pool row spine counts"},
            generated_at,
        ),
        _source(
            "target_evidence",
            "真实目标、风险与状态汇总",
            "target_summary_by_year.parquet",
            "按年读取真实 MFE、pre-peak MAE 和固定状态坐标汇总。",
            ["target_summary_by_year.parquet", "target_summary_by_period.parquet"],
            ["quality_liquidity_complete_pit", "valid label flags only"],
            {
                "top1_rate": "daily cross-sectional true-MFE rank above 99th percentile",
                "top5_rate": "daily cross-sectional true-MFE rank above 95th percentile",
            },
            generated_at,
        ),
        _source(
            "feature_evidence",
            "特征关系、稳定性与族建议",
            "feature_relation_by_period.parquet",
            "读取全量年度关系、稳定性门控、冗余和特征族分类。",
            [
                "feature_relation_by_year.parquet",
                "feature_relation_by_period.parquet",
                "feature_stability.parquet",
                "family_recommendations.parquet",
                "redundancy_pairs.parquet",
            ],
            [
                "628 eligible numeric features",
                "no model fitting",
                "|rho| >= 0.995 for redundancy",
            ],
            {
                "tail_gap": "high feature decile Top-5% rate minus low feature decile Top-5% rate",
                "stable": "same signed opportunity/risk/state relationship in at least 10 years without equal-or-larger 2023-2025 reversal",
            },
            generated_at,
        ),
        _source(
            "strata_evidence",
            "市值、成交额与换手分层",
            "strata_by_period.parquet",
            "读取质量池中的市值、成交额、换手率和行业分层结果。",
            ["strata_by_year.parquet", "strata_by_period.parquet"],
            ["quality_liquidity_complete_pit", "all_history"],
            {
                "turnover_band": "within-trade-date turnover quintile inside the complete pool"
            },
            generated_at,
        ),
        _source(
            "quality_evidence",
            "覆盖率、来源状态与漂移",
            "source_state_by_period.parquet",
            "读取 observed、known_ineligible、source_unavailable、not_applicable 和字段状态。",
            [
                "coverage_distribution_by_period.parquet",
                "source_state_by_period.parquet",
                "source_and_distribution_drift.parquet",
            ],
            ["unknown is never zero", "availability states retained"],
            {
                "observed_rate": "state row count divided by period complete-pool row count"
            },
            generated_at,
        ),
        _source(
            "pool_evidence",
            "完整日频池与分钟共同支撑偏差",
            "pool_bias_by_period.parquet",
            "比较日频质量池中分钟完整与剩余 152 个缺口股票日。",
            [
                "pool_bias_by_year.parquet",
                "pool_bias_by_period.parquet",
                "excluded_minute_rows.parquet",
            ],
            ["quality_liquidity_pit", "2011-2025", "missing 5m only"],
            {
                "excluded_rows": "daily quality rows absent from the complete 48-bar support"
            },
            generated_at,
        ),
    ]
    manifest_sources = [item[0] for item in source_specs]
    canonical_sources = [item[1] for item in source_specs]

    first_formal = int(
        recommendations["classification"].eq("first_model_formal_family").sum()
    )
    gated = int(recommendations["classification"].eq("availability_gated_family").sum())
    all_target = target_period[target_period["period"].eq("all_history")].set_index(
        "horizon"
    )
    atr = relation[
        relation["period"].eq("all_history")
        & relation["feature"].eq("atr_10d")
        & relation["horizon"].eq(10)
    ].iloc[0]
    q1 = turnover[turnover["stratum"].eq("q1")].iloc[0]
    q5 = turnover[turnover["stratum"].eq("q5")].iloc[0]
    excluded_10 = pool_rows[
        pool_rows["stratum"].eq("excluded_missing_5m") & pool_rows["horizon"].eq(10)
    ].iloc[0]
    retained_10 = pool_rows[
        pool_rows["stratum"].eq("retained") & pool_rows["horizon"].eq(10)
    ].iloc[0]

    cards = [
        {
            "id": "common_rows_card",
            "description": "2011–2025 分钟完整质量池公共样本。",
            "dataset": "headline",
            "sourceId": "audit_contract",
            "metrics": [
                {"label": "公共样本", "field": "common_rows", "format": "number"}
            ],
        },
        {
            "id": "numeric_features_card",
            "description": "518 个既有字段、106 个正式新增候选和 4 个 availability-gated 数值字段。",
            "dataset": "headline",
            "sourceId": "feature_evidence",
            "metrics": [
                {
                    "label": "审计数值字段",
                    "field": "numeric_features",
                    "format": "number",
                }
            ],
        },
        {
            "id": "formal_families_card",
            "description": "通过稳定性与独立代表门控的首版消融候选族；不等于最终纳入。",
            "dataset": "headline",
            "sourceId": "feature_evidence",
            "metrics": [
                {"label": "消融候选族", "field": "formal_families", "format": "number"}
            ],
        },
        {
            "id": "gated_families_card",
            "description": "研报、融资融券和财务扩展等结构性来源必须显式门控。",
            "dataset": "headline",
            "sourceId": "feature_evidence",
            "metrics": [
                {"label": "可用性门控族", "field": "gated_families", "format": "number"}
            ],
        },
        {
            "id": "excluded_rows_card",
            "description": "日频质量池中仍缺完整 48 根五分钟的股票日。",
            "dataset": "headline",
            "sourceId": "pool_evidence",
            "metrics": [
                {"label": "剩余分钟缺口", "field": "excluded_rows", "format": "number"}
            ],
        },
        {
            "id": "redundancy_card",
            "description": "固定哈希样本中绝对 Spearman 相关不低于 0.995 的字段对。",
            "dataset": "headline",
            "sourceId": "feature_evidence",
            "metrics": [
                {
                    "label": "近重复字段对",
                    "field": "redundancy_pairs",
                    "format": "number",
                }
            ],
        },
    ]

    charts = [
        {
            "id": "target_year_chart",
            "title": "真实 MFE 中位数年度变化",
            "subtitle": "2011–2025，完整质量池，有效标签行；10日与20日目标",
            "type": "line",
            "intent": "trend",
            "question": "强机会目标的总体位置是否跨年度稳定？",
            "rationale": "15 个年度点足以显示制度与行情阶段变化，并保留两种持有期比较。",
            "dataset": "annual_targets",
            "sourceId": "target_evidence",
            "valueFormat": "percent",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "interactive": False},
            "labels": {"values": "endpoints"},
            "encodings": {
                "x": {"field": "year", "type": "ordinal", "label": "年份"},
                "y": {
                    "field": "mfe_median",
                    "type": "quantitative",
                    "label": "MFE 中位数",
                },
                "color": {
                    "field": "horizon_label",
                    "type": "nominal",
                    "label": "目标周期",
                },
                "tooltip": [
                    {
                        "field": "valid_count",
                        "type": "quantitative",
                        "label": "有效样本",
                    },
                    {
                        "field": "mfe_q95",
                        "type": "quantitative",
                        "label": "MFE 95分位",
                        "format": "percent",
                    },
                    {
                        "field": "mae_median",
                        "type": "quantitative",
                        "label": "MAE 中位数",
                        "format": "percent",
                    },
                ],
            },
        },
        {
            "id": "turnover_tail_chart",
            "title": "换手率分层中的真实 Top 5% 比例",
            "subtitle": "2011–2025；当日质量池内换手率五分位；10日 MFE",
            "type": "bar",
            "intent": "comparison",
            "question": "质量池内强机会是否随换手率单调富集？",
            "rationale": "五个有序分层适合从零开始的柱图，能直接比较 Top 5% 占比。",
            "dataset": "turnover_strata",
            "sourceId": "strata_evidence",
            "valueFormat": "percent",
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "all"},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0.050393,
                    "label": "全样本 5.04%",
                    "color": "neutral",
                    "lineStyle": "dashed",
                }
            ],
            "encodings": {
                "x": {
                    "field": "turnover_band",
                    "type": "ordinal",
                    "label": "换手率分位",
                },
                "y": {
                    "field": "top5_rate",
                    "type": "quantitative",
                    "label": "真实 Top 5% 比例",
                },
                "tooltip": [
                    {"field": "row_count", "type": "quantitative", "label": "样本"},
                    {
                        "field": "top1_rate",
                        "type": "quantitative",
                        "label": "Top 1%",
                        "format": "percent",
                    },
                    {
                        "field": "mfe_median",
                        "type": "quantitative",
                        "label": "MFE 中位数",
                        "format": "percent",
                    },
                    {
                        "field": "mae_median",
                        "type": "quantitative",
                        "label": "MAE 中位数",
                        "format": "percent",
                    },
                ],
            },
        },
        {
            "id": "market_state_chart",
            "title": "代表性市场状态高低分位的 10 日 MFE 中位数",
            "subtitle": "市场级字段按年度交易日时间分位；每个交易日内仍使用全部质量池股票",
            "type": "bar",
            "intent": "comparison",
            "question": "市场级状态变量是否改变绝对机会幅度？",
            "rationale": "四个代表性状态各保留高低分位，分组柱图能显示方向与幅度。",
            "dataset": "market_state",
            "sourceId": "feature_evidence",
            "valueFormat": "percent",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "interactive": False},
            "encodings": {
                "x": {"field": "feature", "type": "nominal", "label": "市场状态"},
                "y": {
                    "field": "mfe_median",
                    "type": "quantitative",
                    "label": "10日 MFE 中位数",
                },
                "color": {
                    "field": "feature_tail",
                    "type": "nominal",
                    "label": "年度交易日分位",
                },
                "tooltip": [
                    {
                        "field": "mae_median",
                        "type": "quantitative",
                        "label": "MAE 中位数",
                        "format": "percent",
                    },
                    {
                        "field": "state_high_rate_gap",
                        "type": "quantitative",
                        "label": "高状态率差",
                        "format": "percent",
                    },
                    {
                        "field": "valid_count",
                        "type": "quantitative",
                        "label": "有效样本",
                    },
                ],
            },
        },
    ]

    tables = [
        {
            "id": "family_table",
            "title": "特征族进入首版模型的建议",
            "subtitle": "15 个分析族；本阶段不冻结最终字段集合",
            "dataset": "family_recommendations",
            "sourceId": "feature_evidence",
            "defaultSort": {"field": "family", "direction": "asc"},
            "columns": [
                {"field": "family", "label": "特征族", "type": "text"},
                {"field": "classification_label", "label": "建议", "type": "text"},
                {"field": "feature_count", "label": "数值字段", "format": "number"},
                {
                    "field": "median_coverage",
                    "label": "中位覆盖率",
                    "format": "percent",
                },
                {
                    "field": "low_coverage_features",
                    "label": "覆盖<80%",
                    "format": "number",
                },
                {
                    "field": "drift_flag_features",
                    "label": "漂移标记",
                    "format": "number",
                },
            ],
        },
        {
            "id": "representative_table",
            "title": "各族代表性稳定关系",
            "subtitle": "10日目标；每族选择通过稳定门控且关系幅度最大的成员",
            "dataset": "representative_features",
            "sourceId": "feature_evidence",
            "defaultSort": {"field": "preferred_top5_lift", "direction": "desc"},
            "columns": [
                {"field": "family", "label": "特征族", "type": "text"},
                {"field": "feature", "label": "代表字段", "type": "text"},
                {"field": "preferred_tail", "label": "有利分位", "type": "text"},
                {
                    "field": "preferred_top5_lift",
                    "label": "Top5 lift",
                    "format": "number",
                },
                {
                    "field": "preferred_mfe_median",
                    "label": "MFE 中位数",
                    "format": "percent",
                },
                {
                    "field": "preferred_mae_median",
                    "label": "MAE 中位数",
                    "format": "percent",
                },
                {
                    "field": "same_direction_years",
                    "label": "同向年份",
                    "format": "number",
                },
            ],
        },
        {
            "id": "availability_table",
            "title": "主要新增来源的可用性",
            "subtitle": "全历史与 2023–2025；结构性资格不等于数据缺失",
            "dataset": "source_availability",
            "sourceId": "quality_evidence",
            "defaultSort": {"field": "all_history_observed_rate", "direction": "desc"},
            "columns": [
                {"field": "source_family", "label": "来源/字段族", "type": "text"},
                {
                    "field": "all_history_observed_rate",
                    "label": "2011–2025",
                    "format": "percent",
                },
                {
                    "field": "recent_observed_rate",
                    "label": "2023–2025",
                    "format": "percent",
                },
                {"field": "semantics", "label": "解释", "type": "text"},
            ],
        },
        {
            "id": "pool_bias_table",
            "title": "分钟共同支撑剩余缺口偏差",
            "subtitle": "152 个股票日；日频质量池资格不受分钟文件存在性影响",
            "dataset": "pool_bias",
            "sourceId": "pool_evidence",
            "defaultSort": {"field": "horizon_label", "direction": "asc"},
            "columns": [
                {"field": "support_group", "label": "组别", "type": "text"},
                {"field": "horizon_label", "label": "目标", "type": "text"},
                {"field": "row_count", "label": "股票日", "format": "number"},
                {"field": "valid_count", "label": "有效标签", "format": "number"},
                {"field": "top1_rate", "label": "Top1 比例", "format": "percent"},
                {"field": "top5_rate", "label": "Top5 比例", "format": "percent"},
                {"field": "mfe_median", "label": "MFE 中位数", "format": "percent"},
                {"field": "mae_median", "label": "MAE 中位数", "format": "percent"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术结论\n\n"
                f"- **数据已达到进入首版建模设计的条件。** 审计覆盖 {manifest['scope']['common_support_row_count']:,} 个 2011–2025 公共股票日、628 个可分析数值字段；2010 未进入统计，2026 读取为零。\n"
                f"- **建议保留 {first_formal} 个首版消融候选族，另有 {gated} 个族必须走 availability gate。** 这只是待比较的特征族，不代表最终纳入，尚未冻结字段，也没有训练模型。\n"
                f"- **强机会与风险来自同一套活跃度结构。** 10日 ATR 高分位的 Top 5% lift 为 {atr.high_top5_lift:.2f}×，但 MAE 中位数由低分位的 {atr.low_mae_median:.2%} 扩大到 {atr.high_mae_median:.2%}；不能把强机会富集直接等同为更好的可交易收益。\n"
                f"- **剩余分钟缺口足够小但并非随机。** 152 行中 10日 Top 5% 比例为 {excluded_10.top5_rate:.2%}，完整组为 {retained_10.top5_rate:.2%}，同时缺口组 MAE 更差；应保留双股票池和偏差报告。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "common_rows_card",
                "numeric_features_card",
                "formal_families_card",
                "gated_families_card",
                "excluded_rows_card",
                "redundancy_card",
            ],
        },
        {
            "id": "target_result",
            "type": "markdown",
            "sourceId": "target_evidence",
            "body": (
                "## 目标本身稳定存在，但机会幅度具有强烈年度状态依赖\n\n"
                f"全历史 10日/20日 MFE 中位数分别为 {all_target.loc[10, 'mfe_median']:.2%} 和 {all_target.loc[20, 'mfe_median']:.2%}，对应 pre-peak MAE 中位数为 {all_target.loc[10, 'mae_median']:.2%} 和 {all_target.loc[20, 'mae_median']:.2%}。年度曲线显示 2015 等高波动阶段明显抬升，说明后续模型必须显式吸收市场状态，不能依赖单一全历史阈值。"
            ),
        },
        {"id": "target_chart_block", "type": "chart", "chartId": "target_year_chart"},
        {
            "id": "turnover_result",
            "type": "markdown",
            "sourceId": "strata_evidence",
            "body": (
                "## 高换手显著富集强机会，也同步放大下行路径风险\n\n"
                f"10日 Top 5% 比例从换手最低组的 {q1.top5_rate:.2%} 单调升至最高组的 {q5.top5_rate:.2%}；MFE 中位数由 {q1.mfe_median:.2%} 升至 {q5.mfe_median:.2%}，但 MAE 中位数也由 {q1.mae_median:.2%} 恶化到 {q5.mae_median:.2%}。这支持把流动性和波动族纳入首版模型，同时要求风险头与机会头共同解释，而不是只做强机会分类。"
            ),
        },
        {
            "id": "turnover_chart_block",
            "type": "chart",
            "chartId": "turnover_tail_chart",
        },
        {
            "id": "feature_family_result",
            "type": "markdown",
            "sourceId": "feature_evidence",
            "body": (
                "## 第一版应按特征族进入，并在族内先去除近重复字段\n\n"
                "价量技术、横截面技术、五分钟、市场状态、行业、市值流动性、财务、公告、传统资金流和传统技术指标均有成员通过不少于 10 年同向且近三年未等幅反转的描述性门控。研报、融资融券和财务扩展字段虽然存在稳定关系，但覆盖含义由资格、制度或供应商年代决定，因此只能作为 availability-gated 族。固定哈希样本发现 292 对 |ρ|≥0.995 的近重复字段，首版输入不应把它们全部平铺。"
            ),
        },
        {"id": "family_table_block", "type": "table", "tableId": "family_table"},
        {
            "id": "representative_table_block",
            "type": "table",
            "tableId": "representative_table",
        },
        {
            "id": "market_state_result",
            "type": "markdown",
            "sourceId": "feature_evidence",
            "body": (
                "## 市场状态改变绝对 MFE、风险和状态坐标，但不会改变每日 Top 5% 配额\n\n"
                "市场级字段在同一天对所有股票相同，因此本审计改用年度内交易日时间分位；个股级字段仍用日内横截面分位。由于强机会主标签本来就是每天横截面 Top 1%/5%，市场状态不能改变该配额，却能明显改变绝对 MFE、MAE 和真实状态分布。这个区别防止把市场级常数误判为无效特征。"
            ),
        },
        {
            "id": "market_state_chart_block",
            "type": "chart",
            "chartId": "market_state_chart",
        },
        {
            "id": "availability_result",
            "type": "markdown",
            "sourceId": "quality_evidence",
            "body": (
                "## 数据修复消除了主要断点，但结构性缺失仍必须显式编码\n\n"
                "研报预测 v2 的日级任务账本在 2011–2025 全部闭合，2021 已恢复，不再存在年度分页截断。技术因子几乎全覆盖，资金流全历史覆盖超过 99.8%。融资融券 observed 比例反映标的范围扩张；合同负债的早年 unknown 和特殊报表 not_applicable 也不能填零。"
            ),
        },
        {
            "id": "availability_table_block",
            "type": "table",
            "tableId": "availability_table",
        },
        {
            "id": "pool_bias_result",
            "type": "markdown",
            "sourceId": "pool_evidence",
            "body": (
                "## 双股票池应继续保留，分钟共同支撑只用于统一训练行键\n\n"
                "修复后日频质量池有 4,477,003 行，分钟完整公共池有 4,476,851 行，仅差 152 行。缺口组的强机会比例更高、风险也更差，但样本极小且 2023–2025 只剩 3 行，不能据此修改股票池资格；它只说明需要持续保留偏差表。"
            ),
        },
        {"id": "pool_bias_table_block", "type": "table", "tableId": "pool_bias_table"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "audit_contract",
            "body": (
                "## 范围、口径与可复核定义\n\n"
                "- 研究总体：`quality_liquidity_complete_pit`，2011–2025 共 4,476,851 行；日频参照池 `quality_liquidity_pit` 共 4,477,003 行。\n"
                "- 强机会：每个交易日、每个目标周期内真实 MFE 的 Top 1% 与 Top 5%；绝对 MFE 另行报告。\n"
                "- 风险：从入场到峰值前的 `pre_peak_mae_10/20`；状态：既有固定三状态坐标，未重训。\n"
                "- 关系分位：个股字段为日内横截面高/低 10%；市场级字段为年度交易日高/低 10%。\n"
                "- 2023–2025 参与训练前研究，未来效果报告称为 retrospective rolling OOS。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## 方法与稳健性门控\n\n"
                "每个年度对全部公共行计算覆盖、分布、特征分位、Top 1%/5% lift、MFE、MAE 与状态差异，再用行数充分统计量聚合三个时期和全历史。近年反转使用排除 2023–2025 的历史基线；正式字段较多的族还需至少两个不落入 |ρ|≥0.995 冗余边的稳定代表，才列为首版消融候选。年度中位数聚合仍是近似统计，最终短名单冻结前需重算精确 pooled median。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 限制、不确定性与不能据此推出的结论\n\n"
                "- 这是描述性关系，不是因果识别，也不是预测性能；高换手、高波动和小市值的 lift 可能共同反映价格限制、横截面分布和行情状态。\n"
                "- 分期中位数是年度组中位数的加权中位数，均值和命中率则由年度充分统计量精确聚合；解读极端分位时应优先查看逐年表。\n"
                "- source-unavailable、known-ineligible、not-applicable 与真实零值严格分离；缺失状态与标签的差异不能直接当成可交易信号。\n"
                "- 本审计没有交易成本、容量、涨跌停可成交性或峰值兑现回放，因此不能覆盖旧 v4 的经济可实现性否定结论。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "feature_evidence",
            "body": (
                "## 下一步：冻结首版族级输入契约，再设计模型实验\n\n"
                "1. 首版正式输入保留 11 个 formal family，4 个 gated family 通过独立 availability mask 接入。\n"
                "2. 在每个族内优先保留 PIT 语义更可靠、覆盖更稳、与既有字段不近重复的代表字段；不把 292 对近重复成员全部输入。\n"
                "3. 为机会、风险和市场状态保留联合评估口径，避免只优化 broad Rank IC。\n"
                "4. 字段集合冻结后再开始时间滚动模型实验；本报告不授权直接训练。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 波动与换手带来的 Top 5% lift，在加入涨跌停、可成交性和容量约束后还剩多少？\n"
                "- 小市值强机会富集究竟是可预测结构，还是风险与极端行情的混合代理？\n"
                "- 研报和融资融券族在 availability gate 内，是否仍有跨年代一致的增量信息？\n"
                "- 族内去冗余后，Top 1%/5% 捕获能否显著缩小旧 v4 的 prediction-oracle gap？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "2011–2025 质量流动性池的训练前数据质量与特征关系技术审计。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": _json_rows(headline),
                "annual_targets": _json_rows(target_year),
                "turnover_strata": _json_rows(turnover),
                "market_state": _json_rows(market_state),
                "family_recommendations": _json_rows(family_rows),
                "representative_features": _json_rows(representative),
                "source_availability": _json_rows(availability),
                "pool_bias": _json_rows(pool_rows),
            },
            "accessIssues": [],
        },
        "sources": canonical_sources,
    }
    artifact_path = audit_root / "artifact.json"
    _write_json(artifact_path, artifact)
    chart_map = {
        "schema": "seq100_descriptive_audit_chart_map/v1",
        "charts": [
            {
                "section": "目标年度状态",
                "question": charts[0]["question"],
                "family": "trend",
                "type": "line",
                "fields": ["year", "horizon_label", "mfe_median"],
                "takeaway": "绝对机会幅度跨年变化明显",
                "palette": "hard two-root cap",
            },
            {
                "section": "换手分层",
                "question": charts[1]["question"],
                "family": "comparison",
                "type": "bar",
                "fields": ["turnover_band", "top5_rate"],
                "takeaway": "强机会随换手率单调富集且风险同步增大",
                "palette": "single-root preferred",
            },
            {
                "section": "市场状态",
                "question": charts[2]["question"],
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["feature", "feature_tail", "mfe_median"],
                "takeaway": "市场状态改变绝对 MFE 与风险",
                "palette": "hard two-root cap",
            },
        ],
    }
    _write_json(audit_root / "chart_map.json", chart_map)
    source_notes = {
        "schema": "seq100_descriptive_audit_report_notes/v1",
        "audience": "technical",
        "delivery_mode": "portable_html",
        "required_structure_mapping": {
            "technical_summary": "technical_summary",
            "key_findings": [
                "target_result",
                "turnover_result",
                "feature_family_result",
                "market_state_result",
                "availability_result",
                "pool_bias_result",
            ],
            "scope_and_definitions": "scope_definitions",
            "methodology": "methodology",
            "limitations": "limitations",
            "recommended_next_steps": "next_steps",
            "further_questions": "further_questions",
        },
        "quantitative_section_without_chart_reasons": {
            "feature_family_result": "15-row exact classification lookup is clearer as tables",
            "availability_result": "source-state semantics require exact text and rates",
            "pool_bias_result": "only two groups and two horizons; exact small-sample table is more honest",
        },
        "report_spine": {
            "question": "Which repaired data and feature families deserve entry into the first quality-pool model?",
            "answer": f"Use {first_formal} formal families and {gated} availability-gated families; freeze no final field set yet.",
            "population": "4,476,851 complete quality-pool stock-days, 2011-2025",
            "comparison_basis": "daily cross-sectional feature deciles, annual time deciles for market-wide fields, annual and era stability",
        },
    }
    _write_json(audit_root / "report_source_notes.json", source_notes)
    report_manifest = {
        "schema": "seq100_descriptive_audit_report_manifest/v1",
        "status": "artifact_ready",
        "study_id": STUDY_ID,
        "generated_at": generated_at,
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "size": int(artifact_path.stat().st_size),
        },
        "chart_map": {
            "path": str((audit_root / "chart_map.json").resolve()),
            "sha256": _sha256(audit_root / "chart_map.json"),
        },
        "source_notes": {
            "path": str((audit_root / "report_source_notes.json").resolve()),
            "sha256": _sha256(audit_root / "report_source_notes.json"),
        },
    }
    _write_json(audit_root / "report_manifest.json", report_manifest)
    return report_manifest


def finalize_report(audit_root: Path = DEFAULT_AUDIT_ROOT) -> dict[str, Any]:
    report_manifest = _read_json(audit_root / "report_manifest.json")
    report_path = audit_root / "report.html"
    receipt_path = audit_root / "report_delivery_receipt.json"
    if not report_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("portable_report_or_delivery_receipt_missing")
    receipt = _read_json(receipt_path)
    verification = receipt.get("stages", {}).get("verification")
    if verification not in {"passed", "structural_only"}:
        raise RuntimeError(f"portable_report_verification_failed:{verification}")
    report_manifest.update(
        {
            "status": "completed",
            "completed_at": _now(),
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _sha256(report_path),
                "size": int(report_path.stat().st_size),
            },
            "delivery_receipt": {
                "path": str(receipt_path.resolve()),
                "sha256": _sha256(receipt_path),
                "verification": verification,
            },
        }
    )
    _write_json(audit_root / "report_manifest.json", report_manifest)
    return report_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        finalize_report(args.audit_root)
        if args.finalize
        else prepare_artifact(args.audit_root)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
