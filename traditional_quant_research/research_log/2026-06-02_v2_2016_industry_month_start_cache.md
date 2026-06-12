# v2 2016 Industry Month-Start Cache

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_v2_2016_industry_month_start_cache.md`.


日期：2026-06-02

## 目的

验证 `--industry-frequency month-start` 能够用更少 Baostock 请求生成按每日股票池展开的行业缓存，为后续补齐 `2016-2025` 行业字段提供可行路径。

## 命令

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-industry --year 2016 --start-date 2016-01-01 --end-date 2016-12-31 --industry-frequency month-start --request-interval-seconds 0.05

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 assemble --year 2016 --start-date 2016-01-01 --end-date 2016-12-31 --snapshot-id baostock_v2_2016_industry_month_start_smoke_20260602 --include-industry --industry-frequency month-start
```

## 结果

- 行业缓存：`cache/stock_industry/year=2016.parquet`
- 行业缓存频率：`month-start`
- Baostock 查询日期数：`12`
- 每日展开日期数：`244`
- 行业缓存股票数：`2467`
- 行业缓存行数：`577686`
- 非空行业行数：`575218`
- 空行业行数：`2468`
- 行业缓存失败数：`0`
- 验证 snapshot：`baostock_v2_2016_industry_month_start_smoke_20260602`
- snapshot `stock_industry_rows`：`577686`

## Loader 验证

- `2016-01-04` 行业表：`2318` 行，非空行业 `2315` 行。
- `2016-01-04` 可交易面板：`2065` 行，非空行业 `2062` 行。
- `load_tradeable_panel(..., include_industry=True)` 中行业来源字段使用 `industry_source`，避免与行情表 `source` 字段冲突。

## 字段审计

字段审计 run：`v2_industry_size_source_audit_20260602_204039`

- `true_industry_in_snapshot`: `True`
- `true_market_cap_in_snapshot`: `False`
- `share_base_in_snapshot`: `False`
- `turnover_in_snapshot`: `False`

## 解释

`month-start` 口径显著降低请求量：2016 年只需 `12` 次 Baostock 行业查询，再按股票向后填充到每日股票池。该口径适合先做长样本行业暴露审计，但不能声称捕捉月内行业分类变化。正式策略候选晋级前，仍需在研究日志中标注行业频率口径，并补齐市值/流通市值或维持 proxy-only 说明。
