# Daily Research 当前判断

快照日期：`2026-04-06`

## 1. 当前总判断
- 当前全局 deployable leaderboard 已建立：
  - `daily_research/output/global_deployable_strategy_leaderboard_20260406_r1`
- 当前全局 deployable winner 为：
  - `state_liquidity_listwise_v1_execfirst_profitmax_global_winner`
  - `liquid500 + raw panel + regoff_k1_5d_ensemble_native_anchor`
- 当前 active strategy 真源固定为：
  - `daily_research/output/active_execution_strategy.json`
- 当前 production 主链已经完成：
  - `formal -> recent realistic gate -> production full-fit -> strict-resume epoch extension -> active default`
- 当前 liquid800 / mainboard 独立研究主线为：
  - `dynamic_graph_no_priors`
- 当前研究判读顺序固定为“月度优先”：
  - 先看月度胜率、中位数、坏月、收益集中度
  - 再看整窗 mean excess annual / Sharpe
- 当前若要讨论“全项目最高”，必须先进入：
  - `global_deployable_non_capacity_adjusted_v1`
  - 同一成本引擎
  - 同一 execution-policy audit 口径

## 2. 当前默认执行快照
- active manifest：
  - `daily_research/output/active_execution_strategy.json`
- strategy name：
  - `state_liquidity_listwise_v1_execfirst_profitmax_global_winner`
- liquidity pool：
  - `liquid500`
- panel mode：
  - `raw`
- execution policy：
  - `regoff_k1_5d_ensemble_native_anchor`
- production root：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- active production run：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1/runs/short_alpha_production_e40`
- 当前 production recipe 状态：
  - `selected_epoch = 25 / 40`
  - `objective_aligned_budget_pressure = false`
- 当前默认计划已实跑验证：
  - `signal_date = 2026-04-03`
  - `execution_date = 2026-04-06`
  - `target_position_count = 4`
  - `candidate_total_rows = 50`
  - `candidate_usable_rows = 50`
  - `production_model_retrain_status = fresh`

## 3. 当前正式判决
- family epoch budget 当前冻结为：
  - `baseline_current -> 4`
  - `structure_context_only -> 12`
  - `state_liquidity_listwise_v1 -> 24`
  - `dynamic_graph_no_priors -> 16`
- architecture 线：
  - `baseline_current = 4.95% / 0.266`
  - `structure_context_only = 1.66% / 0.156`
  - 结论：`structure_context_only` 保留为 raw 结构 challenger，但不占用执行升级优先级。
- short-alpha 线：
  - `state_liquidity_listwise_v1 = 31.86% / 1.797`
  - `baseline_current = 15.72% / 0.934`
  - 三窗 formal 为 `3/3` 同时取胜。
- short-alpha 执行法 formal 复核：
  - `regoff_k1_5d_ensemble_native_anchor = 53.29% / 2.170`
  - 旧 active policy `regoff_k2_10d_ensemble_native_anchor = 28.93% / 1.655`
  - 结论：当前 active strategy 已切到 profit-max execution policy。
- dynamic-graph 执行法 formal 复核：
  - `regoff_k3_5d_ensemble_native_anchor = 33.27% / 1.290`
  - 仍显著弱于 short-alpha 的 `53.29% / 2.170`
  - 结论：`dynamic_graph_no_priors` 仍是 liquid800 研究 winner，但当前全局 rank = `2`
- short-alpha production replay gate：
  - `short_alpha_production_realistic = -10.21% / -0.618`
  - 旧 baseline production = `-20.06% / -1.288`
  - multi-window 胜负为 `4/4`。
- short-alpha production recipe extension：
  - `e24 -> e40` 的 recent replay excess annual / Sharpe 从 `-10.21% / -0.618` 改善到 `-8.76% / -0.499`
  - 当前 best recipe 为 `e40`
  - 当前主线不再处于 recipe budget pressure。
- checkpoint objective fresh compare：
  - `primary_monthly_robust_score` 会同时抬高 candidate 与 baseline
  - 但 baseline 提升更多，导致 candidate 相对优势变弱
  - 结论：liquid500 short-alpha 主线默认仍保持 `primary_annual_return`。
- short-alpha 弱月诊断：
  - `state_liquidity_listwise_v1` 在 `36` 个月里有 `16` 个弱月
  - 弱月主要集中在 `trend_down_low_vol` 与 `trend_up_low_vol`
  - 弱月里 candidate 相对 baseline 的平均 excess 月收益差为 `-4.43%`
  - 同期最优执行法相对当前 `regoff_k1_5d_ensemble_native_anchor` 仍有平均 `8.04%` 的弱月 lift
- short-alpha 条件化执行法复核：
  - leave-window-out 的 month-start-regime conditioned policy 对静态 `regoff_k1_5d_ensemble_native_anchor` 为 `0/3` 全败
  - mean excess annual / Sharpe 从 `53.29% / 2.170` 降到 `37.28% / 1.366`
  - 结论：当前不把简单 regime-conditioned execution policy 推到 active default
- short-alpha profit-max production fresh refresh：
  - 显式按 `execution_alignment_mode = profile` + `regoff_k1_5d_ensemble_native_anchor` 做了一次 fresh production full-fit review
  - 同一执行法 recent live replay 下，review production 为 `-15.03% / -0.910`
  - 当前 production 在同一执行法下为 `7.77% / 0.517`
  - multi-window named windows 为 `0/2` 全败
  - 结论：fresh review 不晋升，当前 production root 保持不变
- dynamic_graph 线：
  - `dynamic_graph_no_priors = 34.94% / 1.393`
  - `dynamic_graph_v1 = 19.97% / 1.044`
  - 结论：`dynamic_graph_no_priors` 是当前 rolling liquid800 / mainboard 研究 winner。
- dynamic_graph liquid500 同宇宙 challenger：
  - `state_liquidity_listwise_v1 = 34.85% / 2.196`
  - `dynamic_graph_no_priors = 33.21% / 1.956`
  - challenger 前两窗领先，但最近窗 `20250318_20260331` 明显落后
  - 结论：`dynamic_graph_no_priors` 还不是 liquid500 默认执行升级答案

## 4. 当前优先级
1. 继续累计 liquid500 short-alpha 上线后的月度样本，重点监控：
   - positive-month ratio
   - median monthly excess
   - worst month
   - top3 positive-month share
2. liquid500 当前最值得继续做的是 short-alpha 弱月修复，优先盯：
   - `trend_down_low_vol`
   - `trend_up_low_vol`
   - `2025-07`
   - `2024-01`
   - `2025-10`
   且优先做 targeted weak-month repair，不优先做简单全局 regime-conditioned policy 切换。
3. 新的 liquid500 challenger 若要晋级，默认先做：
   - budget-normalized formal
   - recent realistic gate
   - production promotion
   - 必要时再做 execution policy audit
4. 月度 checkpoint objective 后续优先放在：
   - `dynamic_graph_no_priors`
   - 非默认 liquid500 challenger
   而不是直接改当前 active default 的 fresh retrain 默认值。
5. 保持 `dynamic_graph_no_priors` 在 liquid800 / mainboard 独立研究线推进；它虽已补完 liquid500 同宇宙 formal challenger，但当前证据仍不足以替代 short-alpha 主线。
6. 旧 baseline production root 仅保留为显式回退对照，不再作为默认执行真源。

## 5. 当前边界
- formal holdout 负责研究 winner 判决。
- recent realistic replay 负责执行 gate。
- production full-fit 负责默认执行候选。
- active strategy manifest 负责日常执行真源。
- strict-resume continuation 不允许在同一训练链中途切换 `checkpoint_selection_objective`。
- 旧的跨股票池 formal headline 不等于“当前全局最高”；若要统一排序，必须先进入全局 deployable leaderboard。

## 6. 当前关键证据目录
- active execution manifest：
  - `daily_research/output/active_execution_strategy.json`
- family budget manifest：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- liquid500 short-alpha budget-normalized formal：
  - `daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1`
- liquid500 short-alpha execution policy formal review：
  - `daily_research/output/short_alpha_execution_policy_formal_review_20260405_r1`
- liquid500 short-alpha production promotion eval：
  - `daily_research/output/short_alpha_production_promotion_eval_20260405_r1`
- liquid500 short-alpha production epoch extension：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
- liquid500 short-alpha checkpoint objective compare：
  - `daily_research/output/short_alpha_checkpoint_objective_comparison_20260405_r1`
- liquid500 short-alpha weak-month review：
  - `daily_research/output/short_alpha_weak_month_review_20260405_r1`
- liquid500 short-alpha conditional execution policy review：
  - `daily_research/output/short_alpha_conditional_execution_policy_review_20260405_r1`
- liquid500 short-alpha profit-max production refresh：
  - `daily_research/output/short_alpha_profitmax_production_refresh_20260405_r1`
- 月度总判：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- liquid800 / mainboard dynamic-graph budget-normalized formal：
  - `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
- liquid500 dynamic-graph challenger formal：
  - `daily_research/output/dynamic_graph_liquid500_challenger_20260405_r1`
