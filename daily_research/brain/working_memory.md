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
- 上述默认执行主线还完成了 production recipe `24 -> 32 -> 40` 的 strict-resume epoch extension：
  - 当前 active production run 已更新到 `short_alpha_production_e40`
  - 最新 production recipe 已解除 `objective_aligned_budget_pressure`
- 当前 liquid800 / mainboard 研究线最重要的新结论是：
  - `dynamic_graph_no_priors` 明显优于 `dynamic_graph_v1`
- 新版月度总判已经以预算归一化 formal + production replay 重做完毕：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1/report.md`
  - 旧 `2026-04-03 monthly_r1` 分析现在只保留为阶段性诊断，不再作为当前排序结论
- 月度 checkpoint objective 已正式接入 `deep_alpha` 主链：
  - 后续 fresh run 可直接用 `primary_monthly_robust_score` 做 checkpoint 选择
  - 单次 run 现在会额外落盘 `primary_research_monthly_objectives.json`

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
- `state_liquidity_listwise_v1` production recipe 的 epoch extension 也已完成：
  - 输出目录：`daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
  - 月度优先 replay ranking 选择 `e40` 为当前 best recipe
  - `e24 -> e40` 的 recent replay excess annual / Sharpe 从 `-10.21% / -0.618` 改善到 `-8.76% / -0.499`
  - 当前 production root 内部 `training_diagnostics` 已变为 `selected_epoch = 25 / 40`、`objective_aligned_budget_pressure = false`
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
- 当前 active production run 为：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1/runs/short_alpha_production_e40`
- 当前默认计划已实跑验证：
  - `signal_date = 2026-04-03`
  - `execution_date = 2026-04-06`
  - `target_position_count = 13`
  - `candidate_total_rows = 50`
  - `candidate_usable_rows = 50`
  - `production_model_retrain_status = fresh`
- 当前需要明确记住的边界是：
  - 这次 production recipe extension 走的是 strict resume continuation
  - 为保证同一训练链可比性，当前 production chain 仍保持 `checkpoint_selection_objective = primary_annual_return`
  - 下一步若要检验月度 checkpoint objective，必须走 fresh run 或 warm-start restart，而不是在同一 strict-resume 链里中途改目标

## 6. 当前优先级
1. 先用 fresh run 或 warm-start restart 正式比较 `primary_annual_return` 与 `primary_monthly_robust_score`，重点放在 `short_alpha` 与 `baseline_current` 两条 liquid500 主线上。
2. 把确认后的月度 checkpoint objective 前推到后续 frontier、formal rich experiment 与 production fresh retrain 中，但不要在已有 strict-resume 链里中途切目标。
3. 所有 rich experiment 与 candidate review 先读 `Monthly Priority Summary` / `primary_research_monthly_diagnostics` / `primary_research_monthly_objectives`，用它定位问题和优化方向。
4. 继续累积 `short_alpha` production 上线后的月度样本，重点盯月度胜率、中位数超额、坏月惩罚与收益集中度是否继续占优。
5. 保持 `dynamic_graph_no_priors` 在 liquid800 / mainboard 独立研究线推进，不与 liquid500 默认执行判断混写。
6. 把旧 baseline production root 保留为显式回退对照，而不是继续作为默认执行真源。

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
- short-alpha production epoch extension：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
- monthly landscape review：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- monthly-priority smoke：
  - `daily_research/output/deep_alpha_monthly_focus_smoke_20260405_r1`
- dynamic-graph budget-normalized formal：
  - `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
