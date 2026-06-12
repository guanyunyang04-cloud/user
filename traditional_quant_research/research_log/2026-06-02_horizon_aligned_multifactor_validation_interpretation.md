# 2026-06-02 Horizon-Aligned Multifactor Validation Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_horizon_aligned_multifactor_validation_interpretation.md`.


## Decision

严格持有期多因子验证结论：`multifactor_rolling_ic_weighted_score` 是当前最强候选输入，可以进入下一轮年度稳定性和执行约束压力测试；`multifactor_low_corr_rank_score` 仍保留为排序诊断对照，但在本轮 horizon-aligned Top-N 中表现明显弱于 rolling IC，暂不作为主候选输入。

当前仍不能升级为策略候选。原因是本轮只解决了 `5d` 标签重叠年化问题，尚未加入涨跌停不可成交、停牌持仓卖出约束、滑点/冲击成本、行业/规模中性化和年度切片。

阶段门禁：

- `go`: 以 `multifactor_rolling_ic_weighted_score` 为主候选输入，进入年度切片和执行约束压力测试。
- `go`: 保留 `multifactor_low_corr_rank_score` 作为 IC/分位排序对照组。
- `hold`: 把当前结果升级为 `out_of_sample_supported` 或 `production_candidate`。
- `hold`: 启动传统 ML 主线替代当前强 baseline。

## Evidence Used

本报告引用：

- 自动验证日志：`research_log/2026-06-02_multifactor_2024_2026_horizon_aligned_validation.md`。
- 产物目录：`traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_114857`。
- 严格回测输出：`horizon_backtest_summary.csv`。

主配置：

- 快照：`baostock_v2_pit_20160101_20260601_stockbasic_fixed`。
- 区间：`2024-01-01` 到 `2026-06-01`。
- 标签：`fwd_ret_5d`。
- horizon-aligned 回测：下一可交易日开盘入场，5 个可交易日后收盘退出，非重叠持有期。
- Top-N：`100`。
- 成本：`0/10/30` bps。
- 调仓候选：`daily/weekly`，但实际持有期非重叠。

## Findings

### 1. rolling IC 是严格持有期下最强信号

`multifactor_rolling_ic_weighted_score` 在 horizon-aligned Top-N 中排名第一：

- weekly Top-100, 0 bps: 年化收益 `0.380077`，Sharpe `2.188328`，最大回撤 `-0.076641`，持有期数 `59`。
- weekly Top-100, 10 bps: 年化收益 `0.290725`，Sharpe `1.746618`，最大回撤 `-0.080584`。
- weekly Top-100, 30 bps: 年化收益 `0.128685`，Sharpe `0.865740`，最大回撤 `-0.104835`。
- daily Top-100, 10 bps: 年化收益 `0.236348`，Sharpe `1.351714`，持有期数 `64`。

这说明 rolling IC 信号不是只靠旧 daily overlapping 标签列回测才好看。即使换成非重叠 5 日持有期，它仍有正向证据。

### 2. 成本压力很明显

rolling IC weekly Top-100 从 `0` 到 `30` bps：

- 年化收益从 `0.380077` 降到 `0.128685`。
- Sharpe 从 `2.188328` 降到 `0.865740`。
- mean net return 从 `0.006644` 降到 `0.002639`。

因此下一阶段必须重点控制换手和成本，不能只追求 IC 或分位 spread。

### 3. low-corr 排序强，但严格 Top-N 较弱

`multifactor_low_corr_rank_score` 在 IC 和分位 spread 中最强，但 horizon-aligned Top-N 表现较弱：

- daily Top-100, 0 bps: 年化收益 `0.129589`，Sharpe `0.569747`。
- daily Top-100, 10 bps: 年化收益 `0.046737`，Sharpe `0.307920`。
- daily Top-100, 30 bps: 年化收益 `-0.101497`，Sharpe `-0.215997`。
- weekly Top-100, 10 bps: 年化收益 `-0.006694`，Sharpe `0.101426`。

这说明 IC/分位排序和实际 Top-N 组合表现之间仍有差异。低相关组合适合作为排序对照组，但当前不应作为主策略输入。

### 4. 旧标签列 Top-N 与严格回测差距巨大

旧 `backtest_summary.csv` 中 rolling IC daily Top-100, 0 bps Sharpe 为 `3.498158`，年化收益为 `3.464384`。严格 horizon-aligned 后，rolling IC daily Top-100, 0 bps Sharpe 为 `1.740400`，年化收益为 `0.319117`。

这个差距证明前一轮 protocol warning 是必要的：多日 forward return 的 daily Top-N 诊断不能当真实 PnL。

## Current Classification

- `multifactor_rolling_ic_weighted_score`: `candidate_input_leader`。
- `multifactor_low_corr_rank_score`: `ranking_diagnostic_control`。
- horizon-aligned Top-N protocol: `protocol_foundation_plus`。
- 当前策略候选数量：`0`。
- 当前阶段：进入年度稳定性和执行约束压力测试。

## Required Next Work

下一步必须补齐：

1. 年度切片：2024、2025、2026 分年 horizon-aligned 表。
2. 成本压力：`0/5/10/20/30/50` bps。
3. 换手约束：限制单期换手或加入 turnover penalty。
4. 执行约束：涨停不可买入、跌停不可卖出、停牌持仓处理。
5. 中性化升级：正式行业/规模中性化替代当前 proxy neutralization。
6. 样本外协议：至少做 rolling walk-forward 或固定 train/validation/test 切分。

## Practical Recommendation

下一轮实验不要扩大到传统 ML。应围绕 `multifactor_rolling_ic_weighted_score` 做更严格的候选筛选：年度表、成本表、换手表和执行约束表。如果 rolling IC 在这些门禁下仍稳定，再把它定义为强 baseline，然后才有必要比较传统 ML 是否带来增益。
