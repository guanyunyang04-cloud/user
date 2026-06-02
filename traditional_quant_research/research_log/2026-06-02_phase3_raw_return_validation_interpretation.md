# 2026-06-02 Phase 3 Raw-Return Validation Interpretation

## Decision

原始收益口径验证结论：`multifactor_low_corr_rank_score` 和 `multifactor_rolling_ic_weighted_score` 继续保留为下一阶段重点候选输入，但当前 raw-return Top-N 结果仍不能升级为真实策略候选。

核心原因：本轮使用 `horizon=5` 的 `fwd_ret_5d`。实验框架会把所选日期的 `return_col` 当作该调仓期收益，因此 daily Top-N 行会使用重叠 5 日未来收益并按日年化，容易显著放大组合 PnL。该结果可以作为持有期排序诊断和候选信号筛选证据，不能当成真实逐日净值回测。

阶段门禁：

- `go`: 保留 `multifactor_low_corr_rank_score` 和 `multifactor_rolling_ic_weighted_score` 进入执行约束回测开发。
- `go`: 建立 horizon-aligned portfolio simulator，修正 `5d` 标签与 daily/weekly/monthly 调仓的收益解释。
- `hold`: 直接使用本轮 raw daily Top-N 年化收益或 Sharpe 声明策略有效。

## Evidence Used

本报告引用：

- 自动日志：`research_log/2026-06-02_multifactor_2024_2026_raw_return_validation.md`。
- 产物目录：`traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_113107`。
- 回测协议：`backtest_protocol.top_n_rebalance_backtest` 使用 `return_col` 的均值作为每个 rebalance row 的 `gross_return`。

主配置：

- 快照：`baostock_v2_pit_20160101_20260601_stockbasic_fixed`。
- 区间：`2024-01-01` 到 `2026-06-01`。
- 标签：`fwd_ret_5d`。
- 成本网格：`0/5/10/20/30` bps。
- Top-N：`50/100`。
- 调仓频率：`daily/weekly/monthly`。

## Findings

### 1. IC 和分位证据与 xsec-excess 一致

`raw` 与 `xsec-excess` 的 RankIC 排名一致，因为同日横截面减去均值不会改变横截面排序。最强排序诊断仍是：

- `multifactor_low_corr_rank_score`: `mean_rank_ic = 0.074406`。
- `multifactor_ic_weighted_score`: `mean_rank_ic = 0.072937`。
- `multifactor_rolling_ic_weighted_score`: `mean_rank_ic = 0.072146`。

分位 spread 也保持同样顺序：

- `multifactor_low_corr_rank_score`: high-minus-low `0.003073`。
- `multifactor_rolling_ic_weighted_score`: high-minus-low `0.002954`。
- `multifactor_ic_weighted_score`: high-minus-low `0.002471`。

这支持一个判断：多因子排序能力不是 `xsec-excess` 标签造成的假象。

### 2. daily Top-N 数字过强，主要用于暴露协议问题

raw daily Top-N 的最佳结果来自 `multifactor_rolling_ic_weighted_score`：

- daily Top-100, 0 bps: Sharpe `3.498158`，年化收益 `3.464384`。
- daily Top-100, 30 bps: Sharpe `2.476638`，年化收益 `1.801997`。

这些数字不应被视为真实策略收益。原因是 `fwd_ret_5d` 被每日滚动使用，收益窗口重叠，且回测协议没有模拟持有 5 日分层组合、每日换仓份额、不可交易持仓、滑点和冲击成本。

### 3. 周频结果更接近 5 日 horizon，但仍需谨慎

若只看更接近 `5d` horizon 的 weekly Top-100，`multifactor_rolling_ic_weighted_score` 仍有正证据：

- 0 bps: 年化收益 `0.328218`，Sharpe `1.479491`。
- 10 bps: 年化收益 `0.243330`，Sharpe `1.159028`。
- 30 bps: 年化收益 `0.089196`，Sharpe `0.517358`。

这个结果比 daily 数字更有参考价值，但仍不是最终回测，因为周频抽样并不等于真实 5 日持有组合，也没有处理涨跌停、停牌持仓和成交失败。

## Required Protocol Upgrade

下一步必须新增一个 horizon-aligned portfolio simulator，至少支持：

1. `horizon=1` 时，按次日收益和每日调仓构造真实日频净值。
2. `horizon=5/20` 时，构造重叠子组合或非重叠持有期组合，避免把同一未来收益窗口重复当作独立日收益。
3. 明确成交时间：收盘后信号，下一可交易日执行。
4. 买入过滤：停牌、涨停、缺 bar、非 tradeable 标的不可买入。
5. 卖出约束：停牌或跌停时持仓不能简单消失。
6. 成本模型：手续费、滑点、换手、可选冲击成本。
7. 输出年度稳定性：逐年收益、Sharpe、最大回撤、换手、成本敏感性。

## Current Classification

- raw IC/分位验证：`diagnostic_plus`。
- raw Top-N daily 数字：`protocol_warning`，不能用于策略升级。
- raw weekly Top-N：`backtest_only`，可作为候选筛选线索。
- 当前候选输入：`multifactor_low_corr_rank_score`、`multifactor_rolling_ic_weighted_score`。
- 当前策略候选数量：`0`。

## Practical Recommendation

下一步应先实现 horizon-aligned portfolio simulator，而不是继续扩大因子池或启动传统 ML。现在最有价值的问题不是“收益够不够高”，而是“在不重复计算 5 日未来收益、加入真实执行限制后，当前两个候选输入还能不能活下来”。
