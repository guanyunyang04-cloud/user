# 60分钟均线代表日期验证

这是一份真实 QDP 1 分钟数据上的定义与可复核性检查，不是收益回测，也不是策略有效性结论。

## 范围

- 开发区间：2022-01-01 至 2024-12-31；每批按年份加载前一年的 10 月作为暖机。
- 2025 验证集：未读取、未用于选择或调整。
- 股票：000001.SZ, 000858.SZ, 600036.SH, 600519.SH, 600613.SH, 600664.SH, 600127.SH, 002412.SZ；代表日期 12 个。
- 日期标签是日线统计代理（上涨、下跌、离散度、流动性、集中/轮动），不是主观市场周期标签。
- 日期沿用此前的日线横截面候选清单，未根据本次分钟事件结果事后挑选。

## 批次结果

| 批次 | 目标日 | 目标股票日 | 240根完整 | 状态行 | 事件行 | 质量排除（整个加载区间） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 7 | 56 | 56 | 80640 | 249 | 1 |
| 2023 | 2 | 16 | 16 | 23040 | 119 | 0 |
| 2024 | 3 | 24 | 24 | 34560 | 56 | 0 |

合计：96 个目标股票日，138240 条分钟×均线状态，424 条事件。

## 事件分布

| 事件 | 数量 |
| --- | ---: |
| `break_reclaim` | 18 |
| `failed_break` | 62 |
| `near_touch` | 39 |
| `posthoc_catchup` | 71 |
| `reclaim_from_below` | 34 |
| `repeated_cross` | 184 |
| `touch_hold` | 16 |

## 因果性检查

- 通过：`2022:all_expected_stock_days_present`
- 通过：`2022:all_target_stock_days_have_240_bars`
- 通过：`2022:all_target_buckets_have_60_bars`
- 通过：`2022:all_target_ordinals_are_0_to_239`
- 通过：`2022:all_state_groups_have_60_rows`
- 通过：`2022:causal_intersection_constant_per_group`
- 通过：`2022:live_ma_formula_exact`
- 通过：`2022:posthoc_never_true_touch`
- 通过：`2022:confirmation_not_before_reference`
- 通过：`2023:all_expected_stock_days_present`
- 通过：`2023:all_target_stock_days_have_240_bars`
- 通过：`2023:all_target_buckets_have_60_bars`
- 通过：`2023:all_target_ordinals_are_0_to_239`
- 通过：`2023:all_state_groups_have_60_rows`
- 通过：`2023:causal_intersection_constant_per_group`
- 通过：`2023:live_ma_formula_exact`
- 通过：`2023:posthoc_never_true_touch`
- 通过：`2023:confirmation_not_before_reference`
- 通过：`2024:all_expected_stock_days_present`
- 通过：`2024:all_target_stock_days_have_240_bars`
- 通过：`2024:all_target_buckets_have_60_bars`
- 通过：`2024:all_target_ordinals_are_0_to_239`
- 通过：`2024:all_state_groups_have_60_rows`
- 通过：`2024:causal_intersection_constant_per_group`
- 通过：`2024:live_ma_formula_exact`
- 通过：`2024:posthoc_never_true_touch`
- 通过：`2024:confirmation_not_before_reference`
- 通过：`representative_event_samples_reference_recomputed`
- 通过：`representative_event_confirmation_order_causal`
- 通过：`representative_samples_mark_hour_end_diagnostics`

代表事件的原始分钟重算样本见同目录 `representative_validation.json`。样本同时记录了固定因果相交价、参考分钟高低点，以及确认时间顺序。

## 结论边界

- 交易时段切分、午休隔离、60 分钟完整性和六个 MA 周期在所选真实数据上可复核。
- `posthoc_catchup` 与真实穿越已分开；它只能用于事后诊断，不能回填成实时买入信号。
- 近期触碰计数使用通用字段 `prior_true_touch_count_window`，窗口长度另行记录（本次为 20 个完成小时）。
- 下一阶段才是把这些事件分别映射为竞争策略，并在 2022--2024 做开发期收益回放；届时必须记录所有尝试过的版本。
