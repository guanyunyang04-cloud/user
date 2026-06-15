# Data Catalog

当前状态：v1 TDX 快照用于本地行情缓存参考；v2 Baostock PIT 日频快照已落地为默认研究数据源。当前 latest v2.1 snapshot 已支持全量 `month-start` 行业表和 daily metrics 表；2016-2026 年度正式缓存、全样本缺失率审计和 metrics 语义审计已通过；v2.2 `daily_size` 年度缓存/装配链路已实现，本地已安装 `tushare=1.4.29`，但 token/live probe 尚未完成，市值/流通市值仍未接入，估值字段官方发布时间/修订行为仍未证明。

## 第一版研究范围

- 市场范围：上证主板 A 股与深证主板 A 股。
- 纳入代码：`.SH` 的 `600/601/603/605` 与 `.SZ` 的 `000/001/002/003` 普通 A 股股票。
- 默认排除：创业板、科创板、北交所、ST、停牌、退市、B 股、基金/ETF、指数、可转债、期货、港股和其它非普通 A 股标的。
- 数据源：`H:\new_tdx64\PYPlugins\user\t0_project\tqcenter.py` 调用 `TPythClient.dll`，以本机通达信缓存为主要原始来源。
- 首版频率：日频。
- 可用样本实测：代表股票日线可返回约 `2000-05-22` 到 `2026-06-01`，字段包括 `Open / High / Low / Close / Volume / Amount / ForwardFactor`。
- 分钟数据状态：可调用但缓存只覆盖近几个月，且最新样本约到 `2026-03-12`，不纳入第一版传统量化研究数据集。
- 数据使用方式：构造研究快照后，常规因子研究和回测读取本项目快照；只有刷新数据、补齐缺口或重建快照时再调用通达信缓存/DLL。

首批需要登记的字段：

- 交易日历：交易日、市场、是否开市。
- 证券主表：代码、名称、上市/退市日期、交易状态。
- 行情序列：开高低收、成交量、成交额、复权因子。
- 横截面字段：行业、市值、估值、换手率、停牌/ST/涨跌停状态。

任何研究结论必须写明数据来源、样本区间、复权方式和 survivorship bias 处理方式。

## v2 PIT Baostock 快照

- 默认快照根目录：`traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/`。
- 当前 latest snapshot：`baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`。
- 样本区间：`2016-01-04` 到 `2026-06-01`，交易日数 `2526`，证券数 `3393`。
- 核心表：`security_master.parquet`、`raw_daily_stock_lists.parquet`、`daily_universe.parquet`、`daily_bars.parquet`、`daily_status.parquet`、`manifest.json`、`quality_report.json`。
- 行业表：`stock_industry.parquet`，由 Baostock `query_stock_industry(code='', date='YYYY-MM-DD')` 构造，schema 为 `date, code, name_on_date, industry, industry_classification, industry_update_date, source`。
- v2.1 日频指标表：`daily_metrics.parquet`，由 Baostock `query_history_k_data_plus` 的 `turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM` 字段构造，schema 为 `date, code, turn, pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM, source`。
- v2.2 可选 size 表：`daily_size.parquet`，vendor-neutral schema 为 `date, code, total_market_cap, float_market_cap, total_share, float_share, free_share, market_cap_unit, share_unit, source, source_trade_date`；当前 latest snapshot 尚无该表，`dataset_v2.load_pit_daily_size()` 返回空表，`load_tradeable_panel(..., include_size=True)` 在无表时不改变旧面板。
- 全量含行业+metrics snapshot：`baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`，`daily_universe_rows=daily_bar_rows=daily_status_rows=daily_metrics_rows=stock_industry_rows=7451610`，`failure_count=0`，`tradeable_rows=7031085`。
- 2026-06-15 存储瘦身：历史 full/smoke/audit/build 快照、raw `cache/`、旧构建日志、v1 残留、可再生事件 cache 和大体积实验明细已清理；latest v2 PIT snapshot 与 `latest_manifest.json` 保留。清理记录见 `research_log/2026-06-15_dataset_storage_cleanup.md`。
- 全量行业缓存构建状态（历史构建记录；raw `cache/` 已在 2026-06-15 清理）：`cache/stock_industry/year=2016.parquet` 到 `year=2026.parquet` 曾完成并已装配进 latest snapshot；统一口径为 `month-start`，2016-2025 每年查询 `12` 个日期，2026 查询 2026-06-01 前 `6` 个日期并展开到每日股票池。
- 日频指标年度缓存状态（历史构建记录；raw `cache/` 已在 2026-06-15 清理）：`cache/daily_metrics/year=2016.parquet` 到 `year=2026.parquet` 曾完成并已装配进 latest snapshot。2016 样本区间 `2016-01-04` 到 `2016-12-30`，`577687` 行、`2467` 只股票、`244` 个交易日，失败数 `0`；2017 样本区间 `2017-01-03` 到 `2017-12-29`，`640615` 行、`2763` 只股票、`244` 个交易日，失败数 `0`；2018 样本区间 `2018-01-02` 到 `2018-12-28`，`680632` 行、`2835` 只股票、`243` 个交易日，失败数 `0`；2019 样本区间 `2019-01-02` 到 `2019-12-31`，`699809` 行、`2910` 只股票、`244` 个交易日，失败数 `0`；2020 样本区间 `2020-01-02` 到 `2020-12-31`，`717114` 行、`3043` 只股票、`243` 个交易日，失败数 `0`；2021 样本区间 `2021-01-04` 到 `2021-12-31`，`750951` 行、`3155` 只股票、`243` 个交易日，失败数 `0`；2022 样本区间 `2022-01-04` 到 `2022-12-30`，`762501` 行、`3207` 只股票、`242` 个交易日，失败数 `0`；2023 样本区间 `2023-01-03` 到 `2023-12-29`，`771522` 行、`3231` 只股票、`242` 个交易日，失败数 `0`；2024 样本区间 `2024-01-02` 到 `2024-12-31`，`771516` 行、`3221` 只股票、`242` 个交易日，失败数 `0`；2025 样本区间 `2025-01-02` 到 `2025-12-31`，`772807` 行、`3213` 只股票、`243` 个交易日，失败数 `0`；2026 样本区间 `2026-01-05` 到 `2026-06-01`，`306630` 行、`3203` 只股票、`96` 个交易日，失败数 `0`。
- 代表年度：2016 行业缓存 `577686` 行、`244` 个交易日、失败数 `0`；2026 行业缓存 `306630` 行、`96` 个交易日、失败数 `0`。
- 2026 metrics 审计 snapshot：`baostock_v2_2026_to_0601_industry_metrics_smoke_20260603`，含 `daily_metrics_rows=306630` 和 `stock_industry_rows=306630`；审计日志为 `research_log/2026-06-03_v2_2026_daily_metrics_audit.md`。
- 2026 metrics 缺失率审计结果：`tradeable_rows=292483`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn/pctChg` 非空率约 `0.997088`，估值字段全量非空。
- 2025 metrics 审计 snapshot：`baostock_v2_2025_industry_metrics_audit_20260603`，含 `daily_metrics_rows=772799` 和 `stock_industry_rows=772799`；审计日志为 `research_log/2026-06-03_v2_2025_daily_metrics_audit.md`。
- 2025 metrics 缺失率审计结果：`tradeable_rows=742360`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn/pctChg` 非空率约 `0.997311`，估值字段全量非空。
- 2024 metrics 审计 snapshot：`baostock_v2_2024_industry_metrics_audit_20260603`，含 `daily_metrics_rows=771470` 和 `stock_industry_rows=771470`；审计日志为 `research_log/2026-06-03_v2_2024_daily_metrics_audit.md`。
- 2024 metrics 缺失率审计结果：`tradeable_rows=745054`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn/pctChg` 非空率约 `0.996800`，估值字段全量非空。
- 2023 metrics 审计 snapshot：`baostock_v2_2023_industry_metrics_audit_20260603`，含 `daily_metrics_rows=771488` 和 `stock_industry_rows=771488`；审计日志为 `research_log/2026-06-03_v2_2023_daily_metrics_audit.md`。
- 2023 metrics 缺失率审计结果：`tradeable_rows=744550`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn/pctChg` 非空率约 `0.997654`，估值字段全量非空。
- 2022 metrics 审计 snapshot：`baostock_v2_2022_industry_metrics_audit_20260603`，含 `daily_metrics_rows=762466` 和 `stock_industry_rows=762466`；审计日志为 `research_log/2026-06-03_v2_2022_daily_metrics_audit.md`。
- 2022 metrics 缺失率审计结果：`tradeable_rows=728743`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn/pctChg` 非空率约 `0.996875`，估值字段全量非空。
- 2021 metrics 审计 snapshot：`baostock_v2_2021_industry_metrics_audit_20260603`，含 `daily_metrics_rows=750932` 和 `stock_industry_rows=750932`；审计日志为 `research_log/2026-06-03_v2_2021_daily_metrics_audit.md`。
- 2021 metrics 缺失率审计结果：`tradeable_rows=706460`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn` 非空率约 `0.995953`，`pctChg` 非空率约 `0.999527`，估值字段全量非空。
- 2020 metrics 审计 snapshot：`baostock_v2_2020_industry_metrics_audit_20260603`，含 `daily_metrics_rows=717104` 和 `stock_industry_rows=717104`；审计日志为 `research_log/2026-06-03_v2_2020_daily_metrics_audit.md`。
- 2020 metrics 缺失率审计结果：`tradeable_rows=669952`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn` 非空率约 `0.993062`，`pctChg` 非空率约 `0.999997`，估值字段全量非空。
- 2019 metrics 审计 snapshot：`baostock_v2_2019_industry_metrics_audit_20260603`，含 `daily_metrics_rows=699798` 和 `stock_industry_rows=699798`；审计日志为 `research_log/2026-06-03_v2_2019_daily_metrics_audit.md`。
- 2019 metrics 缺失率审计结果：`tradeable_rows=667879`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn` 非空率约 `0.993762`，`pctChg` 非空率约 `0.999984`，估值字段全量非空。
- 2018 metrics 审计 snapshot：`baostock_v2_2018_industry_metrics_audit_20260603`，含 `daily_metrics_rows=680626` 和 `stock_industry_rows=680626`；审计日志为 `research_log/2026-06-03_v2_2018_daily_metrics_audit.md`。
- 2018 metrics 缺失率审计结果：`tradeable_rows=631122`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn` 非空率约 `0.951722`，`pctChg` 非空率约 `0.999808`，估值字段全量非空。
- 2017 metrics 审计 snapshot：`baostock_v2_2017_industry_metrics_audit_20260603`，含 `daily_metrics_rows=640611` 和 `stock_industry_rows=640611`；审计日志为 `research_log/2026-06-03_v2_2017_daily_metrics_audit.md`。
- 2017 metrics 缺失率审计结果：`tradeable_rows=584379`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn` 非空率约 `0.931729`，`pctChg` 非空率约 `0.999753`，估值字段全量非空。
- 2016 metrics 审计 snapshot：`baostock_v2_2016_industry_metrics_audit_20260603`，含 `daily_metrics_rows=577686` 和 `stock_industry_rows=577686`；审计日志为 `research_log/2026-06-03_v2_2016_daily_metrics_audit.md`。
- 2016 metrics 缺失率审计结果：`tradeable_rows=518103`，tradeable-only 的 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 覆盖率均为 `1.0`；全 universe 的 `turn` 非空率约 `0.915643`，`pctChg` 非空率约 `0.998906`，估值字段全量非空。
- 全样本 metrics 缺失率审计 snapshot：`baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`，审计日志为 `research_log/2026-06-03_v2_full_daily_metrics_audit.md`；`universe_rows=7451610`，`tradeable_rows=7031085`，`metrics_rows=7451610`，tradeable-only 全字段覆盖率 `1.0`，全 universe 的 `turn` 非空率约 `0.980227`、`pctChg` 非空率约 `0.998534`，估值字段全量非空。
- 全样本 metrics 语义审计：`research_log/2026-06-03_v2_metrics_semantics_audit.md`。`tradeable_panel_rows=7031085`，`pctchg_compare_rows=7027693`，`pctchg_abs_diff_p95≈0.000047`，说明 `pctChg` 与 v2 未复权 close-to-close 收益主体对齐；但 `gt_100bp_rows=14179`，需视为公司行为/复权口径或源差异风险样本。估值字段缺失率通过，但 `valuation_pit_timing_ready=False`，使用前至少滞后一日。
- 市值/股本来源审计：`research_log/2026-06-03_v2_market_cap_source_audit.md`。latest snapshot 没有真实市值、流通市值、股本或 share-base 字段；Baostock `query_history_k_data_plus` 对 `totalShare/liqaShare/total_mv/float_mv/market_cap/float_market_cap` 均返回日线指标参数错误；`query_stock_basic` 只返回 `code,code_name,ipoDate,outDate,type,status`。结论：不能从当前 Baostock v2.1 单独构造真实市值/流通市值，后续需引入外部 PIT cap/float-cap 来源或继续保持 proxy-only。
- 外部 PIT size 来源侦察：`research_log/2026-06-03_v2_external_size_source_scout.md`。真实 run `v2_external_size_source_scout_20260603_115009` 生成外部来源矩阵，当前本地已安装 `tushare=1.4.29`、`akshare=1.18.63`、`efinance=0.5.8`、`baostock=0.9.1`，未安装 `jqdatasdk/rqdatac`，且没有已认证并 live-probe 通过的 PIT daily market-cap/float-cap 来源。首选 v2.2 验证路径为 Tushare `daily_basic` 的 `total_mv/circ_mv/total_share/float_share/free_share`；JoinQuant/RQData 是有订阅时的备选；AkShare/efinance 保留为公开端点交叉检查，不默认作为 PIT 主源。
- Tushare size live probe：`research_log/2026-06-03_v2_tushare_size_probe.md`。真实 run `v2_tushare_size_probe_20260603_114552` 对 `600000.SH/000001.SZ` 和 `20260525/20260601` 的 `daily_basic` 验证链路返回 `skipped/auth_missing`，`expected_pairs=4`、`returned_pairs=0`、`v2_2_ready=False`；这不是字段否定证据，只说明当前本地缺 `TUSHARE_TOKEN/TS_TOKEN`，尚未执行 live field probe。当前 probe 已补充输出标准化 `daily_size_probe.csv/parquet`，由 `size_source.py` 将 Tushare 字段映射为项目 `daily_size` schema。
- v2.2 daily size 生产化缓存/装配链路：`dataset_builder_v2 fetch-size` 已支持按年读取 v2 交易日/股票池缓存，调用 Tushare `daily_basic` 并写入 `cache/daily_size/year=YYYY.parquet`；带 `--symbols` 或 `--max-symbols` 的 smoke 会写入 `cache/daily_size/samples/sample=<hash>/` 及对应 meta/progress/parts，不覆盖正式年度 cache；`assemble --include-size` 只从正式年度缓存写出 snapshot 级 `daily_size.parquet`，并在 manifest 中记录 `daily_size_rows/daily_size_fields`。当前该链路通过离线 fake-source 测试，不代表真实 Tushare live 数据已落地。
- v2.2 daily size 审计：`research_log/2026-06-03_v2_daily_size_audit.md`。真实 run `v2_daily_size_audit_20260603_103517` 对 latest snapshot 返回 `daily_size_absent`，`universe_rows=7451610`、`tradeable_rows=7031085`、`size_rows=0`、`min_tradeable_coverage=0.0`、`required_units_present=False`、`daily_size_ready_for_research=False`；说明当前 size 表尚未生成，不是字段质量通过或失败。
- v2.1 metrics 暴露诊断：`research_log/2026-06-03_v2_metrics_exposure_diagnostics.md`。真实 run `v2_metrics_exposure_diagnostics_20260603_064734` 在 2024/2025/2026 三个前一年 fit、当年 eval 窗口中，对 rolling IC、IC-weighted 和 low-corr 三条 frontier 信号加入 `turn/pctChg`、流动性/动量/低波动 proxy 和一日滞后估值字段暴露诊断。最大平均绝对 signal-metric 相关为 `0.760402`，最大平均绝对篮子指标主动暴露为 `1.170983`；高暴露集中在 `neg_volatility_20d_z_xsec_z`、`log_amount_mean_20d_z_xsec_z`、`momentum_20d_z_xsec_z` 和 `turn_xsec_z`，估值滞后字段暴露相对较弱。结论：frontier 仍是 `candidate-frontier/backtest_only`，下一门禁应优先做组合层流动性/低波动/动量暴露约束或中性化，而不是升级策略候选。
- frontier metrics 硬中性化审计：`research_log/2026-06-03_low_corr_frontier_metrics_neutralization_audit_interpretation.md`。真实 run `low_corr_frontier_neutralization_audit_20260603_073116` 使用 `--include-metrics`，在同一 `20d/monthly/top_n=200/buffer=3.0/30bps/execution_constraints` 协议下，对 rolling IC、IC-weighted 和 low-corr 三条 frontier 信号按日横截面对 `log_amount_mean_20d_z,momentum_20d_z,neg_volatility_20d_z,turn_xsec_z` 同时残差化。信号层相关性被降到数值零，但 30 bps 均值年化从 rolling IC `0.350177` 降到 `0.173130`、IC-weighted `0.280980` 降到 `0.172495`、low-corr `0.262745` 降到 `0.135335`，且 worst drawdown 和 mean turnover 均恶化。篮子 active exposure 只部分降低，`turn_xsec_z` 甚至上升。结论：硬多指标中性化是诊断门禁，不是当前候选晋级路径；下一步应做组合层柔性暴露约束/优化器或外部 PIT cap/float-cap 来源评估。
- frontier 组合层暴露惩罚审计：`research_log/2026-06-03_low_corr_frontier_portfolio_exposure_penalty_audit_interpretation.md`。`horizon_backtest.horizon_aligned_top_n_backtest()` 已新增 `exposure_penalty_cols` 和 `exposure_penalty_strength`，在 Top-N/buffer 选股阶段按篮子平均暴露偏离扣分。真实 run `low_corr_frontier_neutralization_audit_20260603_081810` 使用 `log_amount_mean_20d_z,neg_volatility_20d_z,momentum_20d_z,turn_xsec_z` 和 `strength=0.25`。30 bps 下原始 rolling IC / IC-weighted / low-corr 均值年化为 `0.353015/0.277791/0.260064`，基本不伤原始 frontier；但原始篮子 active exposure 只小幅下降，例如 rolling IC `log_amount_mean_20d_z` 从 `1.001891` 到 `0.998860`。残差化变体的暴露下降更明显，但收益仍低于原始 frontier。结论：组合层惩罚基础设施可用，但 `0.25` 不是候选晋级门禁；下一步应做 penalty strength 网格并联动行业 cap/真实 PIT size 来源。
- frontier 组合层暴露惩罚强度网格：`research_log/2026-06-03_low_corr_frontier_exposure_penalty_grid_interpretation.md`。真实 run `low_corr_frontier_exposure_penalty_grid_20260603_085614` 在同一 `20d/monthly/top_n=200/buffer=3.0/30bps/execution_constraints/include_metrics` 协议下测试 `strength=0/0.25/0.5/1.0`。rolling IC 最优为 `0.25`，均值年化 `0.353015`、较无惩罚 `+0.002838`；IC-weighted 最优仍是无惩罚，均值年化 `0.280980`；low-corr 最优为 `1.0`，均值年化 `0.275757`、较无惩罚 `+0.013012`。惩罚会把 `log_amount_mean_20d_z/neg_volatility_20d_z/turn_xsec_z` 等篮子 active exposure 推向零，但幅度有限。结论：exposure penalty 是可用的组合构造工具，不是独立候选门禁；下一步应联动行业 cap、冲击成本和外部 PIT cap/float-cap 来源评估。
- frontier 组合约束联动门禁：`research_log/2026-06-03_low_corr_frontier_combined_constraint_audit_interpretation.md`。真实 run `low_corr_frontier_combined_constraint_audit_20260603_094139` 使用 rolling IC `strength=0.25`、IC-weighted `strength=0.0`、low-corr `strength=1.0`，叠加 `group_col=industry,max_group_weight=0.10`、30 bps、100m 和 `impact_bps_per_1pct=0/10`。无额外 impact 下 rolling IC / IC-weighted / low-corr 均值年化为 `0.352647/0.284596/0.277423`；10 bps impact 下为 `0.309775/0.247792/0.229035`，三条信号均保持三年正收益。行业 active weight 已被压低，monthly mean abs active industry weight 约 `0.0085-0.0091`；但 `log_amount_mean_20d_z` active exposure 仍约 `-0.98` 到 `-1.00`，低波动/弱动量/换手结构也仍明显。结论：combined gate 强化 `candidate-frontier/backtest_only`，但不完成候选晋级；下一步应引入外部 PIT market cap / float cap 或制定更严格 proxy-only 晋级政策。
- frontier 扩展组合约束审计：`research_log/2026-06-03_low_corr_frontier_extended_combined_constraint_audit.md`。真实 run `low_corr_frontier_combined_constraint_audit_20260603_122047` 将同一 `20d/monthly/top_n=200/buffer=3.0/30bps/100m/industry 10% cap/exposure penalty` 协议扩展到 2017-2026。10 bps impact 下 rolling IC / low-corr / IC-weighted 的均值年化为 `0.096737/0.056183/0.023878`，最差年均为负，正收益年份均为 `0.6`，total periods 为 `60/64/64`。
- frontier 晋级门禁：`research_log/2026-06-03_frontier_promotion_gate.md`。最新真实 run `frontier_promotion_gate_20260603_125120` 读取 2017-2026 扩展 combined constraint 输出和 daily size audit summary，要求 `30 bps / 100m / 10 bps impact` 下通过收益、年度、回撤、最少 `24` 个 periods、真实 size ready 和风格 active exposure gate。三条 frontier 均通过 `sample_gate` 和 `drawdown_gate`，但全部失败于 `size_gate`、`return_gate`、`year_gate` 和 `style_exposure_gate`；候选数量保持 `0`，promotion level 均为 `candidate-frontier/backtest_only`。
- frontier 失败归因：`research_log/2026-06-03_frontier_failure_attribution.md`。真实 run `frontier_failure_attribution_20260603_132813` 读取扩展 combined constraint 产物，生成 `yearly_failure_attribution.csv`、`signal_failure_summary.csv`、`exposure_failure_summary.csv` 和 `annualized_return_pivot.csv`。三条 frontier 的 weak years 均为 `2017,2018,2022,2023`，total weak signal-years 为 `12`；rolling IC 均值最强但 positive year rate 仍只有 `0.6`。
- frontier 弱年 regime 归因：`research_log/2026-06-03_frontier_weak_year_regime_attribution.md`。真实 run `frontier_weak_year_regime_attribution_20260603_134922` 读取 failure attribution 输出，并从 latest v2 PIT tradeable panel 生成 `market_regime_daily.csv`、`yearly_market_regime.csv`、`weak_year_regime_profile.csv`、`signal_year_regime_attribution.csv` 和 `weak_vs_positive_regime_summary.csv`。共同弱年仍为 `2017,2018,2022,2023`；相对正收益年，弱年 `breadth_20d_positive_rate` 低约 `0.069451`，`market_ret_20d_mean` 低约 `0.025886`，`breadth_5d_positive_rate` 低约 `0.032508`，说明当前 frontier 失效更接近广度/20日市场强度不足下的全信号共振失效。
- 行业缓存命令：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-industry --year 2025 --start-date 2025-01-01 --end-date 2025-12-31 --industry-frequency month-start
```

- 日频指标缓存命令：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-metrics --year 2025 --start-date 2025-01-01 --end-date 2025-12-31 --include-metrics
```

带 `--symbols` 或 `--max-symbols` 的 metrics smoke 会写入 `cache/daily_metrics/samples/sample=<hash>/`，对应进度写入 `cache/progress/daily_metrics/samples/sample=<hash>/`，不会覆盖正式年度 `cache/daily_metrics/year=YYYY.parquet` 或通用年度 progress。

- Tushare daily size 年度缓存命令：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 fetch-size --year 2026 --start-date 2026-01-01 --end-date 2026-06-01 --size-token <token>
```

`fetch-size` 只构建 size cache，不生成新 snapshot。未传 `--trade-dates` 时会优先读取 v2 交易日缓存；若交易日缓存缺失，则从年度 `daily_stock_lists` 缓存推导交易日。无 `--symbols/--max-symbols` 的正式全量写入 `cache/daily_size/year=YYYY.parquet`；带 `--symbols` 或 `--max-symbols` 的 smoke 写入 `cache/daily_size/samples/sample=<hash>/year=YYYY.parquet`，不能作为全样本 cache 命中，也不会被 `assemble --include-size` 当作正式年度 size 表读取。

- 日频指标缺失审计命令：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_daily_metrics_audit --root traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/<snapshot_id> --write-research-log
```

- 写入含日频指标表的新 snapshot：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 assemble --start-date 2016-01-01 --end-date 2026-06-01 --include-industry --industry-frequency month-start --include-metrics
```

- 写入含 daily size 表的新 snapshot：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 assemble --start-date 2016-01-01 --end-date 2026-06-01 --include-size
```

该命令只读取已存在的正式年度 `cache/daily_size/year=YYYY.parquet`；如果只有 `samples/sample=<hash>` 小样本缓存，或没有真实 size 缓存，snapshot 不会凭空生成正式 `daily_size.parquet`。

- 行业缓存频率：
  - `--industry-frequency daily`：逐交易日查询 Baostock，最接近日频 PIT，但速度慢；2026 半年实测约 `96` 次查询。
  - `--industry-frequency month-start`：只查询每月首个交易日，然后按 `code` 向后填充到每日股票池，meta 会记录 `industry_frequency=month-start` 和 `query_date_count`；适合先补全长样本行业暴露审计，但不能表述为逐日精确行业变更捕捉。

- 写入含行业表的新 snapshot：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.dataset_builder_v2 assemble --start-date 2016-01-01 --end-date 2026-06-01 --include-industry --industry-frequency month-start
```

研究侧读取：

- `load_pit_stock_industry(root=None, start_date=None, end_date=None, symbols=None)` 读取行业表；旧 snapshot 无该表时返回空表。
- `load_pit_daily_metrics(root=None, start_date=None, end_date=None, symbols=None)` 读取日频指标表；旧 snapshot 无该表时返回空表。
- `load_pit_daily_size(root=None, start_date=None, end_date=None, symbols=None)` 读取 v2.2 预留 size 表；旧 snapshot 无该表时返回固定 schema 的空表。
- `load_tradeable_panel(..., include_industry=True)` 在有 `stock_industry.parquet` 时按 `date, code` 合并行业字段。
- `load_tradeable_panel(..., include_metrics=True)` 在有 `daily_metrics.parquet` 时按 `date, code` 合并指标字段，指标来源字段为 `metrics_source`。
- `load_tradeable_panel(..., include_size=True)` 在有 `daily_size.parquet` 时按 `date, code` 合并 size 字段，来源字段为 `size_source`。
- 行业合并后行业来源字段为 `industry_source`，行业名称快照字段为 `industry_name_on_date`；研究代码不应依赖 pandas 自动生成的 `source_x/source_y` 后缀。
- size 表在研究侧使用统一字段名，不直接暴露 Tushare `total_mv/circ_mv` 等供应商字段名；供应商字段应在 ingestion 层映射到 `total_market_cap/float_market_cap`。
- size ingestion 契约：`size_source.standardize_tushare_daily_basic_size()` 将 Tushare `ts_code/trade_date/total_mv/circ_mv/total_share/float_share/free_share` 映射为 `daily_size` schema，单位固定记录为 `market_cap_unit=10k CNY`、`share_unit=10k shares`；`fetch_tushare_daily_size_cache()` 可按交易日生成年度标准化缓存，`write_daily_size_parquet()` 可写出 loader 兼容的 `daily_size.parquet`。该契约和 builder 装配链已离线测试通过，但不代表 latest snapshot 已生成真实 size 表。

当前限制：

- v2 已有 PIT 股票池、上市/退市、ST、停牌/可交易状态和 OHLCV/成交额。
- latest v2 snapshot 当前仍没有真实市值、流通市值、股本；`amount`、`volume` 只能作为流动性/规模 proxy。2016-2026 metrics 已验证 `turn` 可作为 Baostock 换手率字段候选，`pctChg` 主体对齐 close-to-close 收益；估值字段覆盖率通过但官方发布时间/修订行为未证明。
- 外部来源矩阵已确认：Tushare `daily_basic` 是 v2.2 PIT size/float-size 第一验证对象；当前 probe 已实现但真实 run 因缺少 `TUSHARE_TOKEN/TS_TOKEN` 而 `skipped/auth_missing`，`daily_size` 审计确认 latest snapshot 表缺失且覆盖率为 `0.0`，没有 token/auth/live probe 和正式 `daily_size.parquet` 前不能声称完成真实 size neutralization。
- 当前 frontier 在 2024-2026 有正收益压力证据，但扩展到 2017-2026 后收益/年度稳定性不足；结构化 promotion gate 显示真实 size、收益/年度稳定性和风格 active exposure 仍未达候选门槛，不能升级为正式策略候选。
- failure attribution 显示弱年集中在 `2017/2018/2022/2023`，弱年 regime attribution 进一步显示这些年份相对正收益年主要弱在 `breadth_20d_positive_rate`、`market_ret_20d_mean` 和 `breadth_5d_positive_rate`；说明当前 frontier 是近三年强窗口证据，不是全周期稳健候选。后续研究应先重建跨阶段稳定性，测试 fit/eval 分离的 regime 分层或新增因子族，而不是继续围绕 2024-2026 做参数微调。
- 2024-2026 frontier metrics 暴露诊断显示，当前最强信号和实际 Top-N 篮子仍显著偏向低流动性/小成交额、低波动和动量相关结构；在完成组合层暴露约束、外部 PIT 市值/流通市值来源或更强 proxy-neutral 门禁前，不能把这些 frontier 升级为正式策略候选。
- 当前行业口径为 `month-start` 前向填充，可用于长样本行业暴露审计；不能声称捕捉月内行业变更。
- Baostock live 日线字段 smoke 显示 `turn`、`pctChg`、`peTTM`、`pbMRQ`、`psTTM`、`pcfNcfTTM` 可返回；`turnover` 和 `turnover_rate` 不是有效 Baostock 日线字段名。2016-2026 metrics 已通过 missingness 和语义审计，但估值字段在官方发布时间/修订行为未证明前不纳入候选晋级门禁。
