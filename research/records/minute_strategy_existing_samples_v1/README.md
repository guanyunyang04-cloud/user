# 已生成分钟策略样本的完整研究记录

本记录只研究已经落盘的分钟策略样本，不恢复或补生成尚未完成的 2022–2024 原始分钟事件，也不读取 2025 数据。

## 样本边界

| 样本组 | 版本 | 信号日期 | 说明 |
|---|---|---|---|
| `primary_development_2022_2023_partial` | `current_contract` | 2022-01-04 – 2023-05-18，331 个交易日 | `minute_ma_development_2022_2024` 中已完成的当前数据合同分区；进程在 2023-05-18 后暂停。 |
| `supplemental_2023_09_legacy` | `legacy_vwap` | 2023-09-01 – 2023-09-28，20 个交易日 | 独立的旧 VWAP 版本补充样本，只做单独稳健性参考。 |

2022-04 的旧目录、smoke/probe 目录和已有 portfolio 输出不作为新的独立事件样本。它们只用于重复性或实现审计，避免重复计数。

## 已完成的研究层次

事件层位于同一目录的 `event_metrics.csv`、`daily_metrics.csv`、`daily_consistency.csv`、`event_trigger_summary.csv`、`year_summary.csv`、`month_summary.csv`、`market_regime_summary.csv`、`signal_feature_summary.csv`、`control_pair_daily.csv`、`control_pair_summary.csv` 和 `candidate_screen.csv`。它们覆盖全部可执行策略、六个 MA 周期、5m/15m/30m/60m/1d/2d/3d/5d 结果、gross/net、MFE/MAE、事件类型、年度/月度/市场状态、日级稳定性、随机与流动性匹配控制，以及 Benjamini–Hochberg 多重比较校正。

质量审计补充了 `outcome_quality_summary.csv`（每个远期指标的观测/缺失率）、`event_cost_drag.csv`（gross 到 net 的成本拖累）、`event_tail_summary.csv`（精确均值/极值与按 MA 周期的近似分位数）、`signal_feature_quality.csv`（特征有效值和无效值计数）、`strategy_manifest.csv` 以及 `author_hypothesis_summary.csv`。无效的比率特征保留为计数，不填充成零；尾部分位数明确标记为 DuckDB 的近似值。

账户层使用有限资金、五个最大同时持仓、100 股整手、手续费、滑点、T+1、涨跌停/停牌等合法退出约束，并遍历每个样本中的全部策略 × 退出规则 × 排序器 × seed 组合。账户结果分别落在：

- `runs/minute_strategy_portfolio_existing_primary_331`
- `runs/minute_strategy_portfolio_existing_supplemental_2023_09`

随后由 `tools/analyze_existing_portfolio_results.py` 生成账户层全量汇总：

- `account_results.csv`：每个组合的一行，保留盈利、亏损、未完全解析等所有结果；
- `account_resolved_results.csv` / `account_unresolved_results.csv`：分别查看完全平仓与仍有未平仓/未解析退出的账户，避免把两种状态混在一个统计量中；
- `account_seed_stability.csv` / `account_seed_annual.csv`：seed 稳定性；
- `account_trade_summary.csv` / `account_trade_year.csv` / `account_exit_reason.csv`：成交、年度成交和退出原因；
- `account_equity_summary.csv`：权益路径摘要；
- `account_ranker_policy_summary.csv`、`account_policy_summary.csv`、`account_strategy_summary.csv`、`account_ranker_summary.csv`：分层汇总；
- `account_cross_sample.csv`：两个样本共有组合的逐组合对照；
- `research_summary.md`：事件、作者命题、账户和质量审计的综合结论；
- `portfolio_research.md`：人类可读报告；
- `account_integrity.json`、`account_research_result.json`：覆盖和完整性审计。

## 解释边界

账户网格中的最高收益只是大量比较中的极值，不能单独作为冻结策略或实盘依据。主样本与旧 VWAP 补充样本不应拼成一条连续收益曲线；后者交易日很少，不能替代真正的跨阶段验证。下一步若要继续，应先从这些全量结果中提出少量、事先定义的候选，再用未参与选择的连续样本或 QMT 影子交易进行前向检验。
