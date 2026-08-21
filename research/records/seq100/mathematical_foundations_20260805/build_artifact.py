from __future__ import annotations

"""Build the bounded Data Analytics MCP report from reviewed audit outputs."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GENERATED_AT = "2026-08-05T23:00:00+08:00"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    selected = frame if columns is None else frame.loc[:, columns]
    return [
        {str(key): clean_value(value) for key, value in row.items()}
        for row in selected.to_dict(orient="records")
    ]


def source_audit() -> dict[str, Any]:
    return {
        "id": "audit_source",
        "label": "2011-2025 mathematical foundations audit",
        "path": "daily_research/research_records/seq100/mathematical_foundations_20260805/audit.py",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('daily_research/research_records/seq100/mathematical_foundations_20260805/summary.json');\nSELECT * FROM read_csv_auto('daily_research/research_records/seq100/mathematical_foundations_20260805/*.csv', union_by_name=true, filename=true);",
            "description": "Read-only reconstruction of finite D20 price paths, date-level dependence, threshold comparisons, and retained model/account diagnostics.",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "daily_research/research_records/seq100/mathematical_foundations_20260805/summary.json",
                "daily_research/research_records/seq100/mathematical_foundations_20260805/*.csv",
            ],
            "filters": [
                "formal research dates 2011-01-04 through 2025-12-31",
                "2010 burn-in only",
                "logical maximum outcome date 2025-12-31",
                "no 2026 reads",
            ],
            "metric_definitions": [
                "D20 endpoint return = close at signal index + 20 divided by next-open entry minus one.",
                "Date-equal means first average candidates within each signal date, then average dates.",
                "HAC bandwidths and moving blocks are measured in signal trading dates.",
                "ES5 is the mean of the worst 5 percent of finite returns.",
            ],
        },
    }


def source_raw() -> dict[str, Any]:
    return {
        "id": "raw_qdp_source",
        "label": "Pinned QDP raw OHLC and security status evidence",
        "path": "quant_data_platform/data/qdp_v2/datasets/market_daily_raw/market_daily_raw__0559f722eb9142e5f476854a/dataset.json",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT symbol, trade_date, open, high, low, close FROM read_parquet('quant_data_platform/data/qdp_v2/datasets/market_daily_raw/**/*.parquet') WHERE trade_date BETWEEN '2011-01-01' AND '2025-12-31'",
            "description": "Unadjusted OHLC is used only for the flat-price diagnostic; the result is explicitly not official ST truth.",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "market_daily_raw__0559f722eb9142e5f476854a",
                "security_status__dd7f01560245387dd417d53d",
            ],
            "filters": ["2011-01-01 through 2025-12-31", "pinned pack PIT universe"],
            "metric_definitions": [
                "Flat diagnostic requires open=high=low=close and rounded close at previous close +/- 5%.",
            ],
        },
    }


def source_policy() -> dict[str, Any]:
    return {
        "id": "policy_source",
        "label": "Quality-liquidity policy target labels",
        "path": "daily_research/output/path_policy/studies/seq100_quality_liquidity_policy_targets/labels/manifest.json",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('daily_research/research_records/seq100/mathematical_foundations_20260805/policy_label_delays.csv')",
            "description": "Audits fill-day delays and distinguishes direct finite endpoint returns from policy returns that retry blocked exits.",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "daily_research/research_records/seq100/mathematical_foundations_20260805/policy_label_delays.csv",
                "daily_research/output/path_policy/studies/seq100_quality_liquidity_policy_targets/labels/fill_day.npy",
                "daily_research/output/path_policy/studies/seq100_quality_liquidity_policy_targets/labels/policy_return.npy",
            ],
            "filters": ["horizons D10 and D20", "cutoff 2025-12-31"],
            "metric_definitions": [
                "Delayed open means a blocked event or timeout is filled after its trigger date.",
                "A finite-horizon policy label must censor unresolved delayed exits rather than retry indefinitely.",
            ],
        },
    }


def source_models() -> dict[str, Any]:
    return {
        "id": "model_source",
        "label": "Frozen 557-feature model and account evidence",
        "path": "daily_research/output/path_policy/studies/seq100_quality_liquidity_model/",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_csv_auto('daily_research/research_records/seq100/mathematical_foundations_20260805/st_feature_importance_summary.csv')",
            "description": "Retained model gain summaries, direct T+1 target comparisons, and paired account diagnostics; no new model was trained in this audit.",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "daily_research/research_records/seq100/mathematical_foundations_20260805/st_feature_importance_summary.csv",
                "seq100_quality_liquidity_model/tasks/*/model.txt",
                "seq100_quality_liquidity_t1_return_models/audit.json",
                "seq100_quality_liquidity_execution/decision.json",
            ],
            "filters": ["2011-2025 audit scope", "2023-2025 retrospective rolling OOS for model/account evidence"],
            "metric_definitions": [
                "LightGBM gain share is conditional split gain, not causal feature contribution.",
                "2023-2025 is retrospective rolling OOS because it has been repeatedly inspected.",
            ],
        },
    }


def source_t1_models() -> dict[str, Any]:
    return {
        "id": "t1_model_source",
        "label": "Strict T+1 direct-return model audit",
        "path": "daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/annual_metrics.parquet",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": "SELECT * FROM read_parquet('daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/annual_metrics.parquet') WHERE target = 'open_to_open_1' ORDER BY year, model_source",
            "description": "Compares direct open-to-open T+1 targets with the frozen return head on identical retrospective rolling-OOS support.",
            "executed_at": GENERATED_AT,
            "tables_used": [
                "daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/annual_metrics.parquet",
                "daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/paired_comparison.parquet",
                "daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/audit.json",
            ],
            "filters": [
                "target open_to_open_1",
                "rolling years 2023-2025",
                "identical candidate support",
            ],
            "metric_definitions": [
                "Rank IC is the within-date Spearman correlation averaged over the evaluation year.",
                "Top-5 capture is the share of realized cross-sectional extreme return captured by the five highest model scores.",
            ],
        },
    }


def source_literature() -> dict[str, Any]:
    return {
        "id": "literature_source",
        "label": "Mathematical foundations literature and DOI evidence",
        "path": "daily_research/research_records/seq100/mathematical_foundations_20260805/literature_evidence.md",
        "query": {
            "engine": "DOI/Crossref metadata plus Consensus/SciSpace abstracts",
            "language": "text",
            "description": "Primary literature identifiers and applicability notes; Scite was unavailable because its monthly quota was exhausted.",
            "executed_at": GENERATED_AT,
            "tables_used": ["literature_evidence.md"],
            "filters": ["peer-reviewed methodology and risk literature"],
            "metric_definitions": ["DOIs are canonical identifiers; citations do not prove trading profitability."],
        },
    }


def build() -> dict[str, Any]:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    barrier = load_csv("barrier_by_volatility_decile.csv")
    annual = load_csv("barrier_by_year.csv")
    st_daily = load_csv("st_rate_daily.csv")
    policy_delays = load_csv("policy_label_delays.csv")
    importance = load_csv("st_feature_importance_summary.csv")
    threshold = load_csv("threshold_robustness.csv")
    paired = load_csv("paired_model_accounts.csv")
    outcomes = load_csv("trade_outcomes.csv")
    findings = load_csv("data_quality_findings.csv")

    fixed_rate = float(np.average(barrier["fixed_8pct_hit_rate"], weights=barrier["count"]))
    scaled_rate = float(np.average(barrier["volatility_scaled_k1_hit_rate"], weights=barrier["count"]))
    headline = [{
        "valid_d20_paths": summary["barrier"]["valid_path_count"],
        "signal_dates": summary["dependence"]["signal_date_count"],
        "row_count": summary["row_index"]["row_count"],
        "arithmetic_endpoint_mean": summary["endpoint_distribution_all"]["mean"],
        "date_equal_log_mean": summary["dependence"]["date_equal_mean_log_return"],
        "date_equal_log_hac20_t": summary["dependence"]["date_equal_log_hac20_t_of_mean"],
        "date_equal_log_hac60_t": summary["dependence"]["date_equal_log_hac60_t_of_mean"],
        "endpoint_es05": summary["endpoint_distribution_all"]["es05"],
        "fixed_8pct_hit_rate": fixed_rate,
        "scaled_k1_hit_rate": scaled_rate,
        "st_splice_change": summary["st_state"]["largest_one_day_change"]["change"],
        "maximum_policy_delay_days": policy_delays["maximum_delay_days"].max(),
        "models_with_st_gain": int(importance["positive_gain_count"].sum()),
        "audited_model_count": int(importance["model_count"].sum()),
    }]

    barrier_long: list[dict[str, Any]] = []
    for row in barrier.to_dict(orient="records"):
        common = {
            "volatility_decile": int(row["volatility_decile"]),
            "volatility_median": row["volatility_median"],
            "standardized_distance_median": row["standardized_8pct_distance_median"],
            "endpoint_mean": row["endpoint_mean"],
            "endpoint_mean_log_return": row["endpoint_mean_log_return"],
            "endpoint_es05": row["endpoint_es05"],
            "candidate_count": row["count"],
        }
        barrier_long.append({**common, "barrier_type": "fixed 8%", "hit_rate": row["fixed_8pct_hit_rate"]})
        barrier_long.append({**common, "barrier_type": "volatility-scaled k=1", "hit_rate": row["volatility_scaled_k1_hit_rate"]})

    annual_long: list[dict[str, Any]] = []
    for row in annual.to_dict(orient="records"):
        common = {
            "year": int(row["year"]),
            "year_label": str(int(row["year"])),
            "candidate_count": row["count"],
            "date_count": row["date_count"],
            "endpoint_mean": row["endpoint_mean"],
            "endpoint_mean_log_return": row["endpoint_mean_log_return"],
            "endpoint_es05": row["endpoint_es05"],
        }
        annual_long.append({**common, "barrier_type": "fixed 8%", "hit_rate": row["fixed_8pct_hit_rate"]})
        annual_long.append({**common, "barrier_type": "volatility-scaled k=1", "hit_rate": row["volatility_scaled_k1_hit_rate"]})

    endpoint_long: list[dict[str, Any]] = []
    for row in annual.to_dict(orient="records"):
        common = {
            "year": int(row["year"]),
            "year_label": str(int(row["year"])),
            "candidate_count": row["count"],
            "endpoint_es05": row["endpoint_es05"],
        }
        endpoint_long.append({**common, "return_measure": "arithmetic endpoint", "value": row["endpoint_mean"]})
        endpoint_long.append({**common, "return_measure": "log endpoint", "value": row["endpoint_mean_log_return"]})

    boundary = st_daily.loc[st_daily["trade_date"].between("2011-11-15", "2011-11-30")].copy()
    st_long: list[dict[str, Any]] = []
    for row in boundary.to_dict(orient="records"):
        common = {
            "trade_date": row["trade_date"],
            "date_label": row["trade_date"],
            "universe_count": row["universe_count"],
            "is_st_count": row["is_st_count"],
        }
        st_long.append({**common, "rate_type": "calculated from is_st", "st_rate": row["calculated_st_rate"]})
        st_long.append({**common, "rate_type": "feature value", "st_rate": row["feature_st_rate"]})

    sources = [
        source_audit(),
        source_raw(),
        source_policy(),
        source_models(),
        source_t1_models(),
        source_literature(),
    ]
    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": "A股量化研究的数学地基：2011-2025 数据与模型审计",
        "description": "技术报告：从随机过程、条件分布、风险度量和稳健验证重新定义强模型；包含固定8%批注、标签延迟和ST状态数据审计。",
        "generatedAt": GENERATED_AT,
        "sources": sources,
        "cards": [
            {
                "id": "path_count_card",
                "dataset": "headline",
                "sourceId": "audit_source",
                "description": "完整的D20纯价格路径；不把未来买卖状态当作价格标签门槛。",
                "metrics": [{"label": "完整 D20 路径", "field": "valid_d20_paths", "format": "number"}],
            },
            {
                "id": "log_mean_card",
                "dataset": "headline",
                "sourceId": "audit_source",
                "description": "每天等权的D20对数财富变化；日期级而不是股票行级估计。",
                "metrics": [
                    {"label": "日期等权对数均值", "field": "date_equal_log_mean", "format": "percent", "signed": True},
                    {"label": "HAC20 t", "field": "date_equal_log_hac20_t", "format": "number", "signed": True},
                ],
            },
            {
                "id": "tail_card",
                "dataset": "headline",
                "sourceId": "audit_source",
                "description": "全体D20终点收益最差5%的平均值。",
                "metrics": [{"label": "终点 ES5", "field": "endpoint_es05", "format": "percent", "signed": True}],
            },
            {
                "id": "barrier_card",
                "dataset": "headline",
                "sourceId": "audit_source",
                "description": "固定8%事件的行加权观察率；仅作为描述性切片。",
                "metrics": [{"label": "固定 8% 观察命中率", "field": "fixed_8pct_hit_rate", "format": "percent"}],
            },
        ],
        "charts": [
            {
                "id": "barrier_volatility_chart",
                "title": "D20 障碍命中率与波动率十分位",
                "subtitle": "固定8%命中率随波动率上升，而k=1标准化障碍反向下降；每组约43.9万条路径。",
                "type": "bar",
                "dataset": "barrier_long",
                "sourceId": "audit_source",
                "encodings": {
                    "x": {"field": "volatility_decile", "type": "ordinal", "label": "历史波动率十分位"},
                    "y": {"field": "hit_rate", "type": "quantitative", "format": "percent", "label": "命中率"},
                    "color": {"field": "barrier_type", "type": "nominal", "label": "障碍定义"},
                    "tooltip": [
                        {"field": "volatility_median", "format": "percent", "label": "波动率中位数"},
                        {"field": "standardized_distance_median", "label": "8%标准化距离"},
                        {"field": "endpoint_mean_log_return", "format": "percent", "label": "终点对数均值"},
                        {"field": "endpoint_es05", "format": "percent", "label": "终点ES5"},
                    ],
                },
                "legend": {"position": "bottom", "sort": "spec"},
                "settings": {"groupMode": "grouped"},
                "layout": "full",
            },
            {
                "id": "annual_barrier_chart",
                "title": "年度障碍命中率",
                "subtitle": "年度状态差异远大于一个固定百分比标签可以解释的稳定信号。",
                "type": "line",
                "dataset": "annual_long",
                "sourceId": "audit_source",
                "encodings": {
                    "x": {"field": "year_label", "type": "ordinal", "label": "年份"},
                    "y": {"field": "hit_rate", "type": "quantitative", "format": "percent", "label": "命中率"},
                    "color": {"field": "barrier_type", "type": "nominal", "label": "障碍定义"},
                    "tooltip": [
                        {"field": "candidate_count", "format": "number", "label": "路径数"},
                        {"field": "endpoint_mean_log_return", "format": "percent", "label": "终点对数均值"},
                        {"field": "endpoint_es05", "format": "percent", "label": "终点ES5"},
                    ],
                },
                "legend": {"position": "bottom", "sort": "spec"},
                "layout": "full",
            },
            {
                "id": "endpoint_year_chart",
                "title": "年度 D20 终点收益：算术与对数口径",
                "subtitle": "算术均值的正尾不能替代可复利的对数增长；两种口径刻画不同问题。",
                "type": "bar",
                "dataset": "endpoint_long",
                "sourceId": "audit_source",
                "encodings": {
                    "x": {"field": "year_label", "type": "ordinal", "label": "年份"},
                    "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "终点均值"},
                    "color": {"field": "return_measure", "type": "nominal", "label": "收益口径"},
                    "tooltip": [
                        {"field": "candidate_count", "format": "number", "label": "路径数"},
                        {"field": "endpoint_es05", "format": "percent", "label": "ES5"},
                    ],
                },
                "legend": {"position": "bottom", "sort": "spec"},
                "settings": {"groupMode": "grouped"},
                "layout": "full",
            },
            {
                "id": "st_boundary_chart",
                "title": "ST 状态拼接边界诊断",
                "subtitle": "2011-11-21 到 2011-11-22 出现约5.42个百分点的一日跳变；该图是诊断而非官方ST真值。",
                "type": "line",
                "dataset": "st_boundary_long",
                "sourceId": "raw_qdp_source",
                "encodings": {
                    "x": {"field": "date_label", "type": "ordinal", "label": "交易日"},
                    "y": {"field": "st_rate", "type": "quantitative", "format": "percent", "label": "ST比例"},
                    "color": {"field": "rate_type", "type": "nominal", "label": "计算口径"},
                    "tooltip": [
                        {"field": "universe_count", "format": "number", "label": "池内行数"},
                        {"field": "is_st_count", "format": "number", "label": "is_st行数"},
                    ],
                },
                "legend": {"position": "bottom", "sort": "spec"},
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "annual_barrier_table",
                "title": "年度障碍与终点分布",
                "subtitle": "2011-2025；路径数、信号日数和ES5均按年度完整支持计算。",
                "dataset": "annual_barrier",
                "sourceId": "audit_source",
                "defaultSort": {"field": "year", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "year", "label": "年份", "type": "number"},
                    {"field": "count", "label": "路径数", "type": "number"},
                    {"field": "date_count", "label": "信号日数", "type": "number"},
                    {"field": "fixed_8pct_hit_rate", "label": "固定8%命中", "format": "percent"},
                    {"field": "volatility_scaled_k1_hit_rate", "label": "k=1命中", "format": "percent"},
                    {"field": "endpoint_mean", "label": "终点算术均值", "format": "percent", "signed": True},
                    {"field": "endpoint_mean_log_return", "label": "终点对数均值", "format": "percent", "signed": True},
                    {"field": "endpoint_es05", "label": "终点ES5", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "volatility_table",
                "title": "波动率十分位的障碍与风险",
                "subtitle": "每组约等量路径；保留波动率、标准化距离、终点和左尾字段以便审计。",
                "dataset": "barrier_deciles",
                "sourceId": "audit_source",
                "defaultSort": {"field": "volatility_decile", "direction": "asc"},
                "density": "dense",
                "columns": [
                    {"field": "volatility_decile", "label": "十分位", "type": "number"},
                    {"field": "count", "label": "路径数", "type": "number"},
                    {"field": "volatility_median", "label": "波动率中位数", "format": "percent"},
                    {"field": "standardized_8pct_distance_median", "label": "8%标准化距离", "format": "number"},
                    {"field": "fixed_8pct_hit_rate", "label": "固定8%命中", "format": "percent"},
                    {"field": "volatility_scaled_k1_hit_rate", "label": "k=1命中", "format": "percent"},
                    {"field": "endpoint_mean_log_return", "label": "终点对数均值", "format": "percent", "signed": True},
                    {"field": "endpoint_es05", "label": "终点ES5", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "quality_table",
                "title": "数据质量严重度清单",
                "subtitle": "高严重度问题会改变标签、特征或推断口径；可选限制没有伪装成缺数。",
                "dataset": "data_quality_findings",
                "sourceId": "audit_source",
                "defaultSort": {"field": "severity", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "severity", "label": "严重度", "type": "text"},
                    {"field": "confidence", "label": "置信度", "type": "text"},
                    {"field": "finding", "label": "问题", "type": "text"},
                    {"field": "evidence", "label": "证据", "type": "text"},
                    {"field": "impact", "label": "影响", "type": "text"},
                ],
            },
            {
                "id": "policy_delay_table",
                "title": "D10/D20 政策标签延迟",
                "subtitle": "受阻卖出只受2025截止约束；延迟超过目标期限的数量按股票行统计。",
                "dataset": "policy_delays",
                "sourceId": "policy_source",
                "defaultSort": {"field": "horizon", "direction": "asc"},
                "density": "spacious",
                "columns": [
                    {"field": "horizon", "label": "目标期限", "type": "number"},
                    {"field": "resolved_count", "label": "已解析行", "type": "number"},
                    {"field": "delayed_open_count", "label": "延迟开盘", "type": "number"},
                    {"field": "delay_over_20_count", "label": "延迟>20日", "type": "number"},
                    {"field": "delay_over_60_count", "label": "延迟>60日", "type": "number"},
                    {"field": "maximum_delay_days", "label": "最大延迟日数", "type": "number"},
                    {"field": "maximum_policy_return", "label": "最大政策收益", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "threshold_table",
                "title": "8% 与邻近障碍的配对稳健性",
                "subtitle": "2023-2025账户配对；HAC和移动块区间未校正本次及历史反复选择。",
                "dataset": "threshold_robustness",
                "sourceId": "audit_source",
                "defaultSort": {"field": "hac_t", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "slot_count", "label": "槽位K", "type": "number"},
                    {"field": "cost", "label": "成本", "type": "text"},
                    {"field": "reference_threshold", "label": "对照", "type": "text"},
                    {"field": "lag_or_block_days", "label": "带宽/块长", "type": "number"},
                    {"field": "annualized_mean_log_return_difference", "label": "年化对数差", "format": "percent", "signed": True},
                    {"field": "hac_t", "label": "HAC t", "format": "number", "signed": True},
                    {"field": "annualized_block_bootstrap_ci_low", "label": "区间下界", "format": "percent", "signed": True},
                    {"field": "annualized_block_bootstrap_ci_high", "label": "区间上界", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "model_importance_table",
                "title": "ST市场特征的条件增益摘要",
                "subtitle": "63个已训练头；gain是树分裂统计，不是因果贡献。",
                "dataset": "model_importance",
                "sourceId": "model_source",
                "defaultSort": {"field": "median_gain_share", "direction": "desc"},
                "density": "spacious",
                "columns": [
                    {"field": "model_group", "label": "模型组", "type": "text"},
                    {"field": "model_count", "label": "模型数", "type": "number"},
                    {"field": "positive_gain_count", "label": "正增益数", "type": "number"},
                    {"field": "median_gain_share", "label": "中位增益占比", "format": "percent"},
                    {"field": "minimum_gain_share", "label": "最小增益占比", "format": "percent"},
                    {"field": "maximum_gain_share", "label": "最大增益占比", "format": "percent"},
                    {"field": "median_gain_rank", "label": "中位增益排名", "type": "number"},
                ],
            },
            {
                "id": "paired_table",
                "title": "新旧失败分布标签的账户配对",
                "subtitle": "同一2023-2025信号日历；结果是方向诊断，不是确认性检验。",
                "dataset": "paired_accounts",
                "sourceId": "audit_source",
                "defaultSort": {"field": "hac_t", "direction": "desc"},
                "density": "dense",
                "columns": [
                    {"field": "slot_count", "label": "槽位K", "type": "number"},
                    {"field": "cost", "label": "成本", "type": "text"},
                    {"field": "lag_or_block_days", "label": "带宽/块长", "type": "number"},
                    {"field": "annualized_mean_log_return_difference", "label": "年化对数差", "format": "percent", "signed": True},
                    {"field": "hac_t", "label": "HAC t", "format": "number", "signed": True},
                    {"field": "annualized_block_bootstrap_ci_low", "label": "区间下界", "format": "percent", "signed": True},
                    {"field": "annualized_block_bootstrap_ci_high", "label": "区间上界", "format": "percent", "signed": True},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full", "body": "# A股量化研究的数学地基：2011-2025 数据与模型审计"},
            {"id": "technical_summary", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 技术摘要\n\n**结论先行：当前没有一个经过数据修复、时间依赖校正和选择偏差校正后仍被证明有用的强模型。** 固定8%不是数学常数；D20路径的正算术均值在日期等权对数口径下消失。下一步应预测执行相关结果的条件分布，并把低持仓作为后置组合约束。\n\n审计覆盖2011-2025正式期，2010仅作burn-in，2026读取为零。报告把描述性事实、统计推断、理论假设和尚未验证的建议分开。"},
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["path_count_card", "log_mean_card", "tail_card", "barrier_card"], "layout": "full"},
            {"id": "scope", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 范围、粒度与收益定义\n\n正式输入有 **4,476,851** 行、**3,644** 信号日、**3,005** 个实际模型证券；完整D20价格路径有 **4,385,062** 行、**3,624** 个有效信号日。路径从次日开盘进入，以D20收盘为终点，MFE/TP从D2开始以遵守T+1。`endpoint_return` 是纯价格终点收益，不等于受阻卖出后的政策收益。`log1p(return)` 才是重复资本的可加财富尺度。"},
            {"id": "fixed8", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 固定8%没有特殊数学地位\n\n令 `X_s = mu*s + sigma*W_s`，上穿 `a=log(1.08)` 的概率为 `Phi((mu*H-a)/(sigma*sqrt(H))) + exp(2*mu*a/sigma^2) Phi((-mu*H-a)/(sigma*sqrt(H)))`。无量纲输入是障碍 `k=a/(sigma*sqrt(H))` 与漂移 `delta=mu*sqrt(H)/sigma`；只有零漂移时才只依赖 `k`。当前8%的标准化距离中位数为 **0.731**，1%-99%为 **0.269-2.300**。行加权年度固定命中率 **32.58%-70.99%**，说明标签主要在选择波动率和市场状态。波动率标准化为 `k=1` 后年度仍为 **22.25%-41.68%**，所以波动率也不是充分状态变量。"},
            {"id": "barrier_vol_chart_block", "type": "chart", "chartId": "barrier_volatility_chart", "layout": "full"},
            {"id": "barrier_table_block", "type": "table", "tableId": "volatility_table", "layout": "full"},
            {"id": "endpoint", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 算术均值不能替代对数财富\n\n全体D20终点的算术均值是 **+0.775%**，中位数 **-0.437%**，ES5 **-22.76%**，ES1 **-33.23%**；偏度 **1.62**、超额峰度 **12.47**。行加权对数均值只有 **+0.011%**，日期等权对数均值为 **-0.0317%**。这正是右尾和波动拖累同时存在的情形。命中8%的路径中，**19.97%** 到D20反而为负，**4.68%** 低于-10%，平均峰后回吐 **10.35%**；未命中路径的终点均值 **-5.74%**、ES5 **-25.04%**。"},
            {"id": "endpoint_year_chart_block", "type": "chart", "chartId": "endpoint_year_chart", "layout": "full"},
            {"id": "annual_table_block", "type": "table", "tableId": "annual_barrier_table", "layout": "full"},
            {"id": "dependence", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 股票行数不是独立实验数\n\n日期固定效应解释约 **29.81%** 的横截面总方差；日均终点收益的lag-1自相关为 **0.951**。把股票行当独立样本的标准误约 **0.0061个百分点**，日期等权后为 **0.1274个百分点**，HAC20/60后为 **0.463%/0.493%**，相应有效日期数只有约 **274/242**。日期等权对数均值的HAC t为 **-0.067/-0.064**，20/60日块区间均跨零。所有模型比较应先按日期聚合损失，再做HAC、块自助或SPA；不能用行级p值证明稳定alpha。"},
            {"id": "st", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## ST状态存在可定位的数据拼接问题\n\n原始未复权OHLC与状态表的诊断显示，ST比例从 **2011-11-21的7.2068%** 一日跳到 **2011-11-22的1.7875%**；底层状态行数从 **145** 变为 **36**，而总覆盖没有相应断裂。该日期与历史档案范围边界重合，最可能是名称/状态来源拼接，而不是正常市场变化。固定8%平价诊断找到 **6,982** 个已标记、**4,398** 个未标记股票日；它不是官方真值，需用交易所历史简称区间重建。`market_all__st_rate` 与`is_st/pit_universe`内部误差仅 **3.71e-9**，但内部一致不等于外部正确。"},
            {"id": "st_chart_block", "type": "chart", "chartId": "st_boundary_chart", "layout": "full"},
            {"id": "quality_table_block", "type": "table", "tableId": "quality_table", "layout": "full"},
            {"id": "labels", "type": "markdown", "layout": "full", "sourceId": "policy_source", "body": "## 受阻卖出把D10/D20政策标签变成了不定期限标签\n\nD10有 **31,931** 次延迟开盘，其中 **5,744** 超过20日、**2,960** 超过60日；D20对应 **29,039**、**9,030**、**4,352**。最大延迟 **1,006** 个交易日，极端记录 `000520.SZ/2013-12-25` 的政策收益为 **+587.02%**，填充约两年后发生。`endpoint_return`不受这种重试机制直接污染，但`timeout_return`及其模型明显被污染。正确做法是有限期限加`complete/right_censored/unresolved/invalid`状态，而不是把多年后的可卖价格塞回原标签。"},
            {"id": "policy_table_block", "type": "table", "tableId": "policy_delay_table", "layout": "full"},
            {"id": "model_direct", "type": "markdown", "layout": "full", "sourceId": "t1_model_source", "body": "## 严格T+1目标提高了排序，但削弱了极端尾部捕获\n\n严格T+1直接收益模型把open-to-open Rank IC从冻结旧头的 **0.0033/0.0335/0.0306** 提到 **0.0420/0.0742/0.0774**（2023/2024/2025），说明目标契约会实质改变可学习信号。然而直接模型的Top-5极端尾部捕获率三年都下降，且所有低持仓账户执行面尚未找到稳定正收益区间；这是局部预测能力，不是强模型结论。"},
            {"id": "model_st", "type": "markdown", "layout": "full", "sourceId": "model_source", "body": "## 可疑ST状态进入了全部审计模型\n\nST特征在 **63/63** 个审计头有正增益；按组中位增益占比为直接收益 **3.09%**、原路径 **3.36%**、政策目标 **8.60%**、T+1直接收益 **0.94%**。这些是树的条件分裂增益，不能解释为因果贡献，尤其在状态源可疑时；必须修复后做删除/重建消融。"},
            {"id": "model_table_block", "type": "table", "tableId": "model_importance_table", "layout": "full"},
            {"id": "model_spec", "type": "markdown", "layout": "full", "sourceId": "literature_source", "body": "## 正确的预测对象：条件分布而不是8%二分类\n\n令`F_t`为收盘时可用信息，模型应估计 `P(Z_{i,t}|F_t)`，其中`Z`至少包含多期限open/open与open/close收益、MFE、MAE、峰值时间、峰后回吐、标准化障碍的首次穿越时间，以及买卖阻塞和右删失状态。\n\n分位数回归使用 pinball loss，并检查/修复分位数交叉；全分布使用CRPS或log score；VaR与ES用联合严格一致的Fissler-Ziegel分数；首次穿越用带生存项的离散hazard likelihood。右删失行贡献生存概率，不能被强行标成失败。对路径联合分布还需额外的边际/路径分数，因为单一多元分数可能看不出依赖结构错误。相关理论依据见文献证据文件。"},
            {"id": "decision", "type": "markdown", "layout": "full", "sourceId": "literature_source", "body": "## 低持仓应是后置决策约束\n\n预测层只负责校准 `P(Z|F_t)`；决策层再解 `max_w E[log(1+w'R_net)] - lambda*CVaR_alpha(-w'R_net)`，约束`w_i>=0`、`sum(w)<=1`、`||w||_0<=K`、容量和换手，并允许现金。没有稳健的正效用时，空仓是可行解。Top-1会放大顺序统计误差和winner's curse，所以“尽可能少仓位”不应反过来污染监督目标。预测后优化损失可以作为后续挑战者，但必须在执行模型冻结后才有可识别含义。"},
            {"id": "validation", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 能支撑强模型结论的验证协议\n\n先修复ST和有限期限标签，再冻结数据哈希、目标、候选模型数、损失、超参数预算，以及 `d_t=mean_i[L_baseline-L_model]` 的最小相关改善和预设状态容忍度。用nested purged rolling做外层确认与内层选择，purge至少覆盖最大有限标签依赖期；按日期聚合损失，使用HAC与20/60日块自助；对全候选族使用White Reality Check或Hansen SPA，并用同时区间检查最坏预设状态，对选中的Sharpe只补充DSR。PBO只有在近似独立块足够时才有解释力；本数据D20大约只有181个不重叠块。2023-2025已经反复用于研究选择，只能叫retrospective rolling OOS，不能叫untouched holdout。"},
            {"id": "annotation_conclusions", "type": "markdown", "layout": "full", "sourceId": "audit_source", "body": "## 两个批注的最终回答\n\n**Annotation 1：不一定是8%，而且8%没有特殊理论支撑。** 8%只是一个尺度依赖的事件切片；如果继续报告它，必须同时报告标准化障碍、完整分布和邻近阈值的预注册比较。\n\n**Annotation 2：失败分布建模方向成立，但当前改善未被证明。** 新的hit-plus-timeout标签降低了深亏率和ES5，说明控制失败分布有经济含义；然而与旧标签的配对账户改善在HAC/块区间下不显著，不能把它写成已胜出的模型。"},
            {"id": "next_steps", "type": "markdown", "layout": "full", "body": "## 推荐下一步\n\n1. 用交易所历史简称有效区间重建ST，并做删除/重建消融。\n2. 将受阻卖出改成有限期限、期限专属validity和右删失状态。\n3. 冻结一个小型分布式目标协议：日期先验、L2基线、分位数LightGBM、标准化障碍hazard。\n4. 先通过日期级proper score、尾部校准和SPA/Reality Check，再进入独立执行实验。\n5. 通过后才把场景映射到带现金的log-growth/CVaR低K组合优化，并保留新的未来确认期。"},
            {"id": "further_questions", "type": "markdown", "layout": "full", "sourceId": "literature_source", "body": "## 仍待回答的问题\n\n- 去除日期共同因子后，哪些特征仍能预测股票相对残差？\n- calibrated first-passage/hazard能否改善峰值实现，而不把未来卖得出去作为标签门槛？\n- 在审计过的佣金、印花税、价差和参与率成本下，最小可接受效果是多少？\n- 在校正目标、期限、特征、阈值和执行的全部试验后，尾部捕获还能剩多少？\n\n外部文献的DOI和适用边界保存在 `literature_evidence.md`；Scite本轮因月度额度耗尽未能提供Smart Citation。"},
        ],
    }

    snapshot: dict[str, Any] = {
        "version": 1,
        "status": "ready",
        "generatedAt": GENERATED_AT,
        "datasets": {
            "headline": headline,
            "barrier_long": barrier_long,
            "annual_long": annual_long,
            "endpoint_long": endpoint_long,
            "st_boundary_long": st_long,
            "barrier_deciles": records(barrier),
            "annual_barrier": records(annual),
            "data_quality_findings": [
                {**row, "severity_rank": {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(row["severity"], 9)}
                for row in records(findings)
            ],
            "policy_delays": records(policy_delays),
            "model_importance": records(importance),
            "threshold_robustness": records(threshold),
            "paired_accounts": records(paired),
            "trade_outcomes": records(outcomes),
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


if __name__ == "__main__":
    output = ROOT / "artifact.json"
    output.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(output)
