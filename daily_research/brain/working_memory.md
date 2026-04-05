# Daily Research 当前判断

快照日期：`2026-04-05`

## 1. 当前总判断
- 当前默认执行已切换为：
  - `active_execution_strategy -> state_liquidity_listwise_v1_execfirst_winner`
- 当前研究判读顺序也已切到“月度优先”：
  - 先看月度分布、坏月、月度胜率、收益集中度
  - 再看整窗 mean annual / Sharpe
- 当前 liquid500 默认执行主线的最新结论是：
  - `state_liquidity_listwise_v1 + regoff_k2_10d_ensemble_native_anchor` 已完成 `formal -> recent gate -> production full-fit -> active default` 全链闭环
- 当前 liquid800 / mainboard 研究线最重要的新结论是：
  - `dynamic_graph_no_priors` 明显优于 `dynamic_graph_v1`
- 新版月度总判已经以预算归一化 formal + production replay 重做完毕：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1/report.md`
  - 旧 `2026-04-03 monthly_r1` 分析现在只保留为阶段性诊断，不再作为当前排序结论

## 2. 预算归一化 formal 结论
- family epoch budget 已冻结为：
  - `baseline_current -> 4`
  - `structure_context_only -> 12`
  - `state_liquidity_listwise_v1 -> 24`
  - `dynamic_graph_no_priors -> 16`
- architecture 线结论没有翻案：
  - `baseline_current` mean replay excess annual / Sharpe = `4.95% / 0.266`
  - `structure_context_only` = `1.66% / 0.156`
  - 因而 `structure_context_only` 不再占用执行升级主优先级
- short_alpha 线被显著改写：
  - `state_liquidity_listwise_v1` mean excess annual / Sharpe = `31.86% / 1.797`
  - `baseline_current` = `15.72% / 0.934`
  - 三窗 formal 为 `3/3` 同时取胜
- dynamic_graph 线被进一步强化：
  - `dynamic_graph_no_priors` mean excess annual / Sharpe = `34.94% / 1.393`
  - `dynamic_graph_v1` = `19.97% / 1.044`

## 3. 升级闭环验证
- `state_liquidity_listwise_v1` 已通过 recent realistic replay gate：
  - `short_alpha_execalign_realistic` full-period annual / excess annual / excess Sharpe = `71.19% / 51.70% / 2.964`
  - 旧默认 `regoff_k2_realistic` = `44.12% / 25.75% / 1.722`
  - named-window 胜负 = `4/5`
- `state_liquidity_listwise_v1` 也已通过 production full-fit replay gate：
  - `short_alpha_production_realistic` full-period annual / excess annual / excess Sharpe = `0.68% / -10.21% / -0.618`
  - 旧 active default `baseline_current_production_realistic` = `-10.36% / -20.06% / -1.288`
  - multi-window 胜负 = `4/4`
- 因此 liquid500 默认执行已不再停留在“研究赢家”，而是已经切到 production 复核后的新赢家。

## 4. 月度总判更新
- `structure_context_only` 的旧判断基本仍成立：
  - 它仍更像防守型结构先验
  - 月度兑现能力仍不足
  - 当前不再占用执行升级优先级
- `short_alpha` 的旧判断已被后续实验改写：
  - 它不再只是“更稳但不够爆”
  - 预算归一化后，它同时改善了月度胜率、月度中位数和坏月
  - 当前应把它视为 liquid500 主执行升级线，而不是备选 challenger
- `dynamic_graph_no_priors` 仍是 mainboard / liquid800 主研究线：
  - 但更精确的月度结论是：`v1` 在平滑性上未必更差
  - 真正的问题是 priors 没有带来更高的净收益兑现
- 当前月度研究的主任务已经从“谁看起来更稳”转成：
  - 谁能在按月统计的净收益上持续兑现
  - 谁的优势不是由极少数幸运月份撑起来

## 5. 当前默认执行状态
- 当前 active strategy 指向：
  - `daily_research/output/active_execution_strategy.json`
  - `strategy_name = state_liquidity_listwise_v1_execfirst_winner`
- 当前默认 production root 为：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- 当前默认计划已实跑验证：
  - `signal_date = 2026-04-03`
  - `execution_date = 2026-04-06`
  - `target_position_count = 19`
  - `production_model_retrain_status = fresh`
- 当前需要单独盯住的残余风险是：
  - `short_alpha` production full-fit 内部监控仍显示 `selected_epoch = 21 / 24`
  - `objective_aligned_budget_pressure = true`
  - 因此这条线虽然已上线，但下一步仍应继续做 production recipe 的 epoch extension

## 6. 当前优先级
1. 先把 `state_liquidity_listwise_v1` 的 production recipe 继续做 epoch frontier extension，重点看 `24 -> 32 -> 40` 是否还能稳定提升。
2. 把月度目标进一步前推到训练判决里，至少把月度胜率、中位数超额、坏月惩罚、收益集中度惩罚纳入正式 review。
3. 所有 rich experiment 与 candidate review 先读 `Monthly Priority Summary` / `primary_research_monthly_diagnostics`，用它定位问题和优化方向。
4. 保持 `dynamic_graph_no_priors` 在 liquid800 / mainboard 独立研究线推进，不与 liquid500 默认执行判断混写。
5. 把旧 baseline production root 保留为显式回退对照，而不是继续作为默认执行真源。

## 7. 暂不优先做的事
- 不再重开 `structure_context_only` 大矩阵。
- 不再手工给所有家族统一写死同一个 `--epochs`。
- 不把 liquid800 研究 winner 直接静默混入 liquid500 默认执行。

## 8. 当前关键证据目录
- active execution manifest：
  - `daily_research/output/active_execution_strategy.json`
- active production root：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- family budget manifest：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- short-alpha budget-normalized formal：
  - `daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1`
- short-alpha production promotion eval：
  - `daily_research/output/short_alpha_production_promotion_eval_20260405_r1`
- monthly landscape review：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- monthly-priority smoke：
  - `daily_research/output/deep_alpha_monthly_focus_smoke_20260405_r1`
- dynamic-graph budget-normalized formal：
  - `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
