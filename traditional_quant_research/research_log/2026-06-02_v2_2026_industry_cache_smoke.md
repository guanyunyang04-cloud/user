# v2 2026 Industry Cache Smoke

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_v2_2026_industry_cache_smoke.md`.


日期：2026-06-02

## 目的

验证 Baostock `query_stock_industry` 已能进入 v2 PIT 数据链路，并生成含 `stock_industry.parquet` 的可读 snapshot，为后续真实行业暴露/中性化审计做准备。

## 命令

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-industry --year 2026 --start-date 2026-01-01 --end-date 2026-06-01 --request-interval-seconds 0.05

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 assemble --year 2026 --start-date 2026-01-01 --end-date 2026-06-01 --snapshot-id baostock_v2_2026_to_0601_industry_smoke_20260602 --include-industry
```

## 结果

- 行业缓存：`cache/stock_industry/year=2026.parquet`
- 行业缓存行数：`306598`
- 行业缓存日期数：`96`
- 行业缓存股票数：`3203`
- 日期范围：`2026-01-05` 到 `2026-06-01`
- 行业缓存失败数：`0`
- 验证 snapshot：`baostock_v2_2026_to_0601_industry_smoke_20260602`
- snapshot `stock_industry_rows`：`306598`
- 单日 `2026-01-05` 行业表：`3189` 行，`3189` 个非空行业字段。
- 单日 `2026-01-05` 可交易面板：`3056` 行，`3056` 个非空行业字段。

## 字段审计

字段审计 run：`v2_industry_size_source_audit_20260602_201712`

- `true_industry_in_snapshot`: `True`
- `true_market_cap_in_snapshot`: `False`
- `share_base_in_snapshot`: `False`
- `turnover_in_snapshot`: `False`
- `size_liquidity_proxy_in_snapshot`: `True`
- `pit_status_fields_in_snapshot`: `True`

## 解释

v2 已经具备真实行业字段的生产缓存和 loader 合并路径。当前证据只覆盖 2026 年截至 `2026-06-01` 的验证 snapshot；默认 latest 仍指向全量 `baostock_v2_pit_20160101_20260601_stockbasic_fixed`，避免研究入口被半年度 smoke snapshot 覆盖。

策略候选数量仍为 `0`。下一步是按年补齐 `2016-2025` 行业缓存，assemble 含行业表的全量 v2 snapshot，然后重跑 frontier 信号的行业暴露/行业中性化审计。市值、流通市值、股本和换手字段仍未接入，不能声称真实市值中性化。
