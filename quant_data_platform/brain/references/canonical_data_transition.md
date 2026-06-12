# Canonical Data

这里是工作区级 canonical 数据基底的过渡入口和兼容指针，不直接承载大体量 parquet / memmap 正文。

长期治理归属已经迁到 `H:/quant_project/quant_data_platform`。本目录保留 registry/root manifest 兼容入口，直到 `quant_data_platform` 的 registry、canonical bundle 和 sharded memmap 全部验证完成。

- 默认别名：`canonical_data_v1`
- 默认研究窗口：`2010-01-01` 起
- 主数据湖：`H:/quant_project/daily_research/output/research_data_lake`
- canonical manifest：`H:/quant_project/daily_research/output/research_data_lake/canonical/canonical_manifest.json`
- 根级 registry：`H:/quant_project/canonical_data/registry/root_manifest.json`

原则：

- raw 数据尽量保留全历史，正式研究视图默认从 `2010-01-01` 起。
- 原始 OHLCV 不被复权价覆盖；复权价、复权收益、分钟聚合特征都作为派生 sidecar。
- BaoStock 仍可用于 PIT 状态、交易日历、上市/停牌/ST、估值、行业、指数成分等基础口径和日常更新。
- TDX/tqcenter 可作为快速外部日线源，入湖后必须写明来源、复权状态、`fill_data` 和成交额单位转换。
- 2010-2025 的 5 分钟历史数据优先复用已完成的 `aggregate_1m_to_5m` 数据集；通过抽样等效性审计后纳入 canonical，不为血统重建全历史。
- 2026 以后 5 分钟增量优先使用 TDX/tqcenter 原生 `period=5m` 补齐；未来若有原生 5m zip，则直接导入原生 5m。
- 1 分钟数据作为 cold archive 保留，不进入默认研究视图，也不再作为未来 5 分钟更新的默认派生来源。
- canonical 数据基底纳入日线、5 分钟日级特征、复权因子、估值、行业、指数成分和交易过滤域；训练 profile 再决定是否使用这些字段。
- 默认短线核心 profile 只纳入日线、5 分钟日级特征、复权收益派生等价格/量价相关信息。
- 5 分钟日级特征默认包含开盘/尾盘收益与成交额占比、高低点出现时间、先高后低、日内回撤/上冲、VWAP 位置和斜率、上午/下午差异、量能集中度、午间跳变和收盘压力。
- `valuation`、`industry_concept`、`index_constituents` 进入 canonical 数据基底；短线核心 profile 默认不用，结构风格或中线 profile 可显式启用。
- `financial_quarterly`、`performance_forecast`、`performance_express` 不进入 v1 默认基底；如以后做中长线或事件专题，必须先建立严格披露/可用时点审计。
- `trading_calendar`、`universe_snapshot`、`security_status` 只作为交易可用性和风险过滤，不作为触发信号。
- 共享 memmap 建成后由 registry 解析；实验只生成轻量 sample index 和 normalization manifest。当前 16GB 机器不安全执行全 A、256 特征、长窗口单进程全量 memmap，已加低内存保护；已先建立 capped validation memmap，下一步是分片 canonical feature/label store。
- 旧 bundle、旧 parquet、旧 `.dat` 清理必须先有 dry-run、替代指针和随机一致性验证。

已完成组件：

- canonical 短线 policy bundle：`policy_input_bundle__002270b729a4eabe6211ff6d`。
- canonical 日线主表：`data_platform_market_daily__618db3bf9609c6e2b7673f40`，覆盖 `2010-01-04` 到 `2026-06-10`。
- benchmark 日线：`data_platform_market_daily__76a2dc61ae52a429782208e1`，`000300.SH`，覆盖 `2010-01-04` 到 `2026-06-10`。
- canonical 5 分钟：`data_platform_market_intraday_5m__aa65b9aebaae19f98ece08e3`，覆盖 `2010-01-01` 到 `2026-06-10`。
- 5 分钟日级特征：`data_platform_intraday_daily_features__e7ee4203e25ae82160ea480e`，覆盖 `2010-01-04` 到 `2026-06-10`。
- 复权因子 sidecar：`data_platform_adjust_factor__30c0c748c33515a192346c3e`，覆盖 `2010-01-01` 到 `2026-06-08`。
- 当前已完成的 capped validation memmap 仍使用旧短线 feature profile：`raw_kline_context_v2_short_horizon_intraday_v1`，保留日线、5 分钟日级特征、复权收益派生；估值、行业、指数成分是否入模由后续 profile 和 sharded memmap registry 控制。
- capped validation memmap：`H:/quant_project/daily_research/output/path_policy/studies/canonical_short_horizon_capped100_memmap_20260611_01/forecast_dataset_manifest.json`，100 只股票，`2022-2025` 角色窗口，60 日 lookback，256 特征，1536 条样本，已注册到 `canonical_data/registry/memmap_registry.json`，scope 为 `capped_validation`。
- 低内存保护：`--max-universe-size 0` + 重型短线 profile + `--forecast-max-feature-columns >= 128` 在低于 32GB 物理内存时会拒绝启动，除非显式传 `--forecast-allow-low-memory-full-build`。
