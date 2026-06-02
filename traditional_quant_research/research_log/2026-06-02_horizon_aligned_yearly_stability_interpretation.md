# 2026-06-02 Horizon-Aligned Yearly Stability Interpretation

## Decision

年度稳定性验证结论：`multifactor_rolling_ic_weighted_score` 仍是当前最强候选输入，但没有通过样本外稳定门禁。它在 2024 和 2025 年表现强，2026 年明显降温，且 10 bps 成本后 2026 年转负。因此当前不能升级为策略候选，只能进入下一轮“换手/成本压缩 + 执行约束”测试。

`multifactor_low_corr_rank_score` 的年度稳定性更弱：2024 年显著为负，2025 年显著为正，2026 年接近无效。它继续保留为排序诊断对照组，不作为主候选输入。

阶段门禁：

- `go`: 继续围绕 `multifactor_rolling_ic_weighted_score` 做候选输入研究。
- `go`: 开发换手约束、成本压缩和执行约束回测。
- `hold`: 升级为 `out_of_sample_supported`。
- `hold`: 升级为 `production_candidate`。
- `hold`: 启动传统 ML 主线替代当前 baseline。

## Evidence Used

本报告引用：

- 自动验证日志：`research_log/2026-06-02_multifactor_2024_2026_horizon_aligned_yearly_validation.md`。
- 产物目录：`traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_115939`。
- 年度表：`horizon_backtest_yearly_summary.csv`。

主配置：

- 快照：`baostock_v2_pit_20160101_20260601_stockbasic_fixed`。
- 区间：`2024-01-01` 到 `2026-06-01`。
- horizon: `5`。
- Top-N: `100`。
- 成本：`0/10/30` bps。
- 调仓候选：`daily/weekly`。
- 回测口径：下一可交易日开盘入场，5 个可交易日后收盘退出，非重叠持有期。

## Findings

### 1. rolling IC 的年度表现不均衡

`multifactor_rolling_ic_weighted_score` weekly Top-100:

| year | fee_bps | annualized_return | sharpe | periods |
|---:|---:|---:|---:|---:|
| 2024 | 0 | 0.772556 | 3.745259 | 11 |
| 2025 | 0 | 0.492421 | 2.946255 | 31 |
| 2026 | 0 | 0.017635 | 0.188818 | 17 |
| 2024 | 10 | 0.689282 | 3.413227 | 11 |
| 2025 | 10 | 0.389920 | 2.434603 | 31 |
| 2026 | 10 | -0.052508 | -0.248033 | 17 |
| 2024 | 30 | 0.534033 | 2.759858 | 11 |
| 2025 | 30 | 0.205185 | 1.409549 | 31 |
| 2026 | 30 | -0.178877 | -1.128911 | 17 |

该信号在 2024/2025 年较强，但 2026 年半年度明显失效。由于 2026 是更靠近样本尾部的验证窗口，该失效不能忽略。

### 2. 成本压力暴露换手问题

rolling IC weekly Top-100 的 mean turnover 约为：

- 2024: `0.961818`。
- 2025: `1.421935`。
- 2026: `1.417647`。

10 bps 后 2026 年已经转负，30 bps 后 2025 年 Sharpe 也降至 `1.409549`，2026 年 Sharpe 降至 `-1.128911`。这说明当前信号对交易成本很敏感，必须优先做换手约束或信号平滑。

### 3. low-corr 年度稳定性不足

`multifactor_low_corr_rank_score` weekly Top-100:

| year | fee_bps | annualized_return | sharpe |
|---:|---:|---:|---:|
| 2024 | 0 | -0.717110 | -3.135282 |
| 2025 | 0 | 0.842173 | 3.614233 |
| 2026 | 0 | 0.011453 | 0.152703 |
| 2024 | 10 | -0.737705 | -3.331156 |
| 2025 | 10 | 0.704519 | 3.166351 |
| 2026 | 10 | -0.064138 | -0.289372 |
| 2024 | 30 | -0.774587 | -3.722357 |
| 2025 | 30 | 0.458772 | 2.267784 |
| 2026 | 30 | -0.199083 | -1.179101 |

这说明低相关组合的 IC/分位优势并未稳定转化为年度 Top-N 收益。当前只能作为排序诊断对照，而不是策略输入主线。

### 4. 2026 年是当前最大风险点

在 2026 半年，多个多因子信号在成本后转弱，甚至 `baseline_score` 在部分 daily 配置下反而更强。这提示当前多因子信号可能存在阶段依赖，或者 rolling IC 权重在样本尾部适应不足。

该问题必须通过以下方式拆解：

- 按月/季度切片查看 2026 失效集中在哪些月份。
- 检查 rolling IC 权重在 2026 的因子暴露变化。
- 比较 daily 与 weekly 真实成交口径。
- 加入换手约束后复验 2026。

## Current Classification

- `multifactor_rolling_ic_weighted_score`: `candidate_input_watchlist`。
- `multifactor_low_corr_rank_score`: `ranking_diagnostic_control`。
- 年度稳定性：`not_passed`。
- 成本鲁棒性：`not_passed`。
- 样本外支持：`not_established`。
- 当前策略候选数量：`0`。

## Next Required Work

下一步应优先实现：

1. 换手约束或信号平滑：限制单期换手，减少成本敏感性。
2. 月度/季度切片：定位 2026 失效区间。
3. rolling IC 权重审计：检查 2026 权重是否过拟合或滞后。
4. 执行约束：涨停不可买入、跌停不可卖出、停牌持仓处理。
5. 更完整成本网格：`0/5/10/20/30/50` bps。

## Practical Recommendation

当前最有价值的方向不是扩大模型复杂度，而是压缩交易层脆弱性。若 rolling IC 在换手约束和执行约束后仍能保住 2024/2025 的大部分收益，并改善 2026 的成本后表现，才可以进入更强的样本外协议。否则，应回到因子权重、因子池和风格暴露层面修正。
