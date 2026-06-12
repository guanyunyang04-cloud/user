# 2026-06-02 Phase 2 Multifactor Validation Interpretation

## Decision

第二阶段多因子验证结论：多因子 rank baseline 已经显著强于第一阶段 `baseline_score`，可以进入“候选策略筛选基础设施”阶段；但当前证据仍不能把任何信号升级为正式策略候选，也不建议立刻进入传统 ML 生产式建模。

推荐阶段门禁：

- `go`: 原始收益口径、执行约束更强的多因子候选筛选。
- `go`: 行业/规模中性化、年度稳定性、成本压力和换手约束补强。
- `hold`: 直接把当前多因子分数声明为可投资策略。
- `hold`: 在缺少强 baseline 门禁前启动 Ridge、LightGBM、随机森林等传统 ML 主线。

## Evidence Used

本解释报告引用以下第二阶段产物：

- 2026 半年多因子诊断：`research_log/2026-06-02_multifactor_baseline.md`。
- 2024-2026 多因子验证：`research_log/2026-06-02_multifactor_2024_2026_validation.md`。
- 产物目录：`traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_112236`。

主验证配置：

- 快照：`baostock_v2_pit_20160101_20260601_stockbasic_fixed`。
- 区间：`2024-01-01` 到 `2026-06-01`。
- 样本：`1,779,897` 行、`581` 个交易日、`3,233` 只证券。
- 标签：`xsec_excess_ret_5d`，原始标签为 `fwd_ret_5d`。
- rolling IC：窗口 `252`，最小历史 `60`，fallback date rate `0.110155`。
- 低相关因子：`momentum_20d_z`、`log_amount_mean_20d_z`、`neg_volatility_20d_z`、`reversal_5d_z`。

## Findings

### 1. 多因子 rank baseline 明显优于第一阶段 baseline_score

在 2024-2026 验证中，`baseline_score` 的 `mean_rank_ic` 为 `-0.002236`，基本不能作为强基线。多因子 rank 组合显著改善：

- `multifactor_low_corr_rank_score`: `mean_rank_ic = 0.074406`。
- `multifactor_ic_weighted_score`: `mean_rank_ic = 0.072937`。
- `multifactor_rolling_ic_weighted_score`: `mean_rank_ic = 0.072146`。
- `multifactor_equal_rank_score`: `mean_rank_ic = 0.071564`。

这说明第二阶段的方向校正、rank 合成、低相关筛选和 IC 加权不是装饰性改造，而是确实把第一阶段弱信号提升为更可研究的选股排序信号。

### 2. 低相关组合是当前最稳的排序诊断信号

`multifactor_low_corr_rank_score` 同时取得最佳 RankIC 和最佳分位 spread：

- 最佳 RankIC：`mean_rank_ic = 0.074406`。
- 最佳 high-minus-low spread：`0.003073`。

这条结果对研究很重要：低相关筛选虽然简单，但它减少了高相似因子的重复投票，使组合信号在 IC 和分位层面更一致。它应当成为下一阶段候选筛选的核心对照组之一。

### 3. rolling IC 信号的 Top-N 表现最好，但需要更严格解释

Top-N 网格中表现最好的信号是 `multifactor_rolling_ic_weighted_score`：

- daily Top-100, 0 bps: Sharpe `1.542878`，年化收益 `0.593572`。
- daily Top-100, 10 bps: Sharpe `1.081098`，年化收益 `0.363591`。

但该结果目前不能直接升级为策略候选，原因有三点：

- 当前标签模式是 `xsec-excess`，Top-N 指标更像 alpha-style 排序诊断，不是完整真实组合收益。
- 最佳 IC/分位信号和最佳 Top-N 信号不一致。
- 当前执行约束仍缺少涨跌停、真实滑点、冲击成本、持仓不可卖出和更完整换手约束。

因此 rolling IC 信号应被列为“重点候选输入”，而不是“候选策略”。

### 4. 2026 半年窗口和 2024-2026 验证存在阶段差异

2026 半年诊断里，最佳 IC、最佳分位 spread、最佳 Top-N 信号不一致，且分位方向出现明显不稳定。2024-2026 验证中，低相关组合在 IC 和分位 spread 上变得更一致，说明更长样本对多因子证据更友好。

这不是坏事，反而提醒下一阶段必须强制做年度切片和样本外滚动验证。单个短窗口可以作为 smoke evidence，但不能决定策略升级。

## Current Classification

- v2 PIT 数据集：`diagnostic_ready`。
- 标签体系：`diagnostic_ready`，但 `5d/20d` 极端收益仍需复核。
- 单因子方向：`diagnostic`。
- 多因子 rank baseline：`diagnostic_plus`，已经形成强对照组。
- Top-N 回测矩阵：`backtest_only`。
- 当前多因子信号：不能升级为 `out_of_sample_supported` 或 `production_candidate`。

## Next Stage

第三阶段应命名为“多因子候选策略筛选基础设施”，核心不是增加模型复杂度，而是把当前排序证据放进更真实的交易检验中。

优先顺序：

1. 回到 `raw` 标签模式，用 `fwd_ret_5d` 和 `fwd_ret_1d` 复验当前多因子信号。
2. 增加年度稳定性表：逐年 RankIC、分位 spread、Top-N 净收益、换手、最大回撤。
3. 扩展成本压力：至少覆盖 `0/5/10/20/30` bps，并报告信号在成本下的降级速度。
4. 加入更强执行约束：涨跌停不可买入/卖出、停牌持仓处理、次日成交口径和真实可交易 mask。
5. 补强中性化：从当前 `log_amount_mean_20d_z` proxy neutralization 升级到正式行业/规模中性化。
6. 复核极端标签：特别是 `5d/20d` 大幅收益样本和复权/除权影响。
7. 形成候选门禁：只有同时通过 IC、分位、年度、成本、换手和执行约束的信号，才能进入 `out_of_sample_supported`。

## Entry Criteria For Traditional ML

传统 ML 可以作为后续路线，但不应早于以下条件：

- `multifactor_low_corr_rank_score` 和 `multifactor_rolling_ic_weighted_score` 已在 raw-return 执行约束回测中形成强 baseline。
- 年度切片显示多数年份方向一致，且不是由单一年份贡献。
- 成本压力和换手约束后仍有正向净表现。
- 极端标签处理规则固定。
- 训练/验证/测试或 rolling walk-forward 协议已写入实验入口。

## Practical Recommendation

下一步不要追“更花的模型”。应优先把 `multifactor_low_corr_rank_score` 和 `multifactor_rolling_ic_weighted_score` 当作两个候选输入，做 raw-return、年度切片、成本压力和执行约束回测。若它们在更严格协议下仍能稳定胜出，再启动传统 ML 才有意义，因为那时 ML 的增益可以和强 baseline 比较，而不是和一个弱等权分数比较。
