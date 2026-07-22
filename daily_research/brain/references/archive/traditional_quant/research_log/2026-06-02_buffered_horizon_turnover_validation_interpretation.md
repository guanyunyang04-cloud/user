# 2026-06-02 Buffered Horizon Turnover Validation Interpretation

## Decision

Top-N buffer 机制已验证有效：它能显著降低 `multifactor_rolling_ic_weighted_score` 的换手，并在高成本情景下改善净表现。但它没有解决 2026 半年失效问题，因此当前仍不能把该信号升级为策略候选。

阶段门禁：

- `go`: 保留 buffer 作为后续候选筛选的成本压缩工具。
- `go`: 在后续执行约束回测中默认比较 `buffer_multiplier = 1.0/1.5/2.0`。
- `hold`: 因 buffer 改善高成本结果而升级为 `out_of_sample_supported`。
- `hold`: 因 buffer 改善高成本结果而升级为 `production_candidate`。

## Evidence Used

本报告引用：

- 自动验证日志：`research_log/2026-06-02_multifactor_2024_2026_buffered_horizon_validation.md`。
- 产物目录：`traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_123305`。
- 严格回测表：`horizon_backtest_summary.csv`。
- 年度表：`horizon_backtest_yearly_summary.csv`。

主配置：

- 快照：`baostock_v2_pit_20160101_20260601_stockbasic_fixed`。
- 区间：`2024-01-01` 到 `2026-06-01`。
- horizon: `5`。
- Top-N: `100`。
- 调仓候选：`weekly`。
- 成本：`0/10/30` bps。
- buffer multipliers: `1.0/1.5/2.0`。
- 回测口径：下一可交易日开盘入场，5 个可交易日后收盘退出，非重叠持有期。

## Findings

### 1. buffer 显著降低换手

`multifactor_rolling_ic_weighted_score` weekly Top-100:

| buffer | mean_turnover | gross_return | net_return, 30bps |
|---:|---:|---:|---:|
| 1.0 | 1.334915 | 0.006644 | 0.002639 |
| 1.5 | 1.133898 | 0.006157 | 0.002756 |
| 2.0 | 1.002712 | 0.005866 | 0.002858 |

buffer 越大，换手越低；同时毛收益略下降。这是预期中的取舍：少换仓会牺牲一部分排名新鲜度，但能降低成本拖累。

### 2. 高成本下 buffer 改善净表现

在 30 bps 下：

| buffer | annualized_return | sharpe | max_drawdown |
|---:|---:|---:|---:|
| 1.0 | 0.128685 | 0.865740 | -0.104835 |
| 1.5 | 0.135336 | 0.903045 | -0.105498 |
| 2.0 | 0.141538 | 0.948780 | -0.101225 |

这说明 buffer 是有效的成本压缩方向。尤其在高成本场景，`buffer_multiplier=2.0` 相比无 buffer 有更高 Sharpe 和略低最大回撤。

### 3. 中低成本下 buffer 会牺牲收益

在 0 bps 和 10 bps 下，无 buffer 的收益和 Sharpe 仍最高：

| fee_bps | best buffer | best sharpe |
|---:|---:|---:|
| 0 | 1.0 | 2.188328 |
| 10 | 1.0 | 1.746618 |
| 30 | 2.0 | 0.948780 |

所以 buffer 不是无条件更优，而是高成本/高换手环境下的防守工具。

### 4. 2026 问题仍未解决

年度表显示，rolling IC 在 2026 年仍然弱：

- buffer `1.0`, 10 bps: annualized return `-0.052508`, Sharpe `-0.248033`。
- buffer `1.5`, 10 bps: annualized return `-0.069544`, Sharpe `-0.364757`。
- buffer `2.0`, 10 bps: annualized return `-0.061666`, Sharpe `-0.314623`。
- buffer `2.0`, 30 bps: annualized return `-0.158269`, Sharpe `-0.994367`。

buffer 可以减少交易摩擦，但不能修复信号在 2026 半年的 alpha 衰减。该问题需要回到权重、因子暴露和市场阶段切片分析。

## Current Classification

- Top-N buffer protocol: `cost_control_tool_validated`。
- `multifactor_rolling_ic_weighted_score`: `candidate_input_watchlist`。
- 成本鲁棒性：`partially_improved_not_passed`。
- 年度稳定性：`not_passed`。
- 样本外支持：`not_established`。
- 当前策略候选数量：`0`。

## Next Required Work

下一步应做：

1. 2026 月度/季度切片，定位失效区间。
2. rolling IC 权重审计，检查 2026 是否因权重滞后或过拟合导致降温。
3. 执行约束回测，加入涨停不可买入、跌停不可卖出、停牌持仓处理。
4. 对 buffer 做更细网格：`1.0/1.25/1.5/2.0`，并比较换手、毛收益和净收益。
5. 若 2026 仍不改善，回到因子池和风格暴露层面修正，而不是升级模型复杂度。

## Practical Recommendation

后续候选筛选应把 `buffer_multiplier=1.0` 作为进攻口径，把 `buffer_multiplier=2.0` 作为高成本防守口径。若某信号只在无成本或低成本下有效，而在 30 bps 与 buffer 后仍无法保持年度稳定，就不能升级为策略候选。
