# Daily Research 当前判断

快照日期：`2026-04-06`

## 1. 当前总判断
- 当前全局 deployable winner 仍是：
  - `state_liquidity_listwise_v1_execfirst_profitmax_global_winner`
  - `liquid500 + raw panel + regoff_k1_5d_ensemble_native_anchor`
- 当前 active strategy 真源固定为：
  - `daily_research/output/active_execution_strategy.json`
- 当前 liquid800 / mainboard 独立研究 winner 仍是：
  - `dynamic_graph_no_priors`
- 当前研究判读顺序固定为“月度优先”：
  - 先看 `positive_month_ratio`
  - 再看 `median_monthly_return`
  - 再看 `worst_monthly_return`
  - 再看 `top3_positive_month_share`
  - 最后才看 mean excess annual / Sharpe

## 2. 当前默认执行快照
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
- 当前默认计划实跑状态：
  - `signal_date = 2026-04-03`
  - `execution_date = 2026-04-06`
  - `target_position_count = 4`
  - `production_model_retrain_status = fresh`

## 3. 当前正式结论
- family epoch budget 当前冻结为：
  - `baseline_current -> 4`
  - `structure_context_only -> 12`
  - `state_liquidity_listwise_v1 -> 24`
  - `dynamic_graph_no_priors -> 16`
- liquid500 short-alpha formal winner 仍是：
  - `state_liquidity_listwise_v1 = 31.86% / 1.797`
  - `baseline_current = 15.72% / 0.934`
  - 三窗 `3/3` 同时取胜
- liquid500 执行法 formal winner 仍是：
  - `regoff_k1_5d_ensemble_native_anchor = 53.29% / 2.170`
- `weak_month_repair_v1` 扩展静态桥接搜索已经完成：
  - 最优 profile 仍是 `regoff_k1_5d_ensemble_native_anchor`
  - 没有新的静态 score-to-weight / bridge profile 翻案
  - 结论转为：后续应做 targeted weak-month repair，而不是继续扩大静态桥接集合
- short-alpha weak-month 诊断仍有效：
  - `36` 个月里有 `16` 个 weak months
  - 主要集中在 `trend_down_low_vol` 与 `trend_up_low_vol`
  - weak months 里最优 policy 相对当前 policy 仍有平均 `7.55%` lift
- simple regime-conditioned execution policy 已被正式否定：
  - leave-window-out 对静态 `regoff_k1_5d_ensemble_native_anchor` 为 `0/3` 全败
  - mean excess annual / Sharpe 从 `53.29% / 2.170` 降到 `37.28% / 1.366`
- profit-max production fresh refresh 已被正式否定：
  - 同一 execution policy 下，fresh review production recent replay = `-15.03% / -0.910`
  - 当前 production = `7.77% / 0.517`
  - 结论：当前 production root 保持不变
- `dynamic_graph_no_priors` 仍是 liquid800 / mainboard 研究 winner：
  - budget-normalized formal = `34.94% / 1.393`
  - 同口径 profit-max execution-policy formal review = `33.27% / 1.290`
  - 当前全局 rank = `2`
- `dynamic_graph_no_priors` 已补 liquid500 同宇宙 challenger formal：
  - `state_liquidity_listwise_v1 = 34.85% / 2.196`
  - `dynamic_graph_no_priors = 33.21% / 1.956`
  - 最近窗明显落后，当前仍不是 liquid500 默认执行升级答案
- architecture current-protocol refresh 已重做完毕：
  - `baseline_current` 仍是 formal 月度优先 rank 1：`22.46% / 1.695`
  - `structure_context_only` 是当前最强 raw 架构 challenger：`31.57% / 1.717`
  - 但它坏月更深、月度胜率更低，不进入默认执行升级链
- `graph_off_plain` 预算补齐复核已完成：
  - 两条旧窗高预算时 annual / Sharpe 有改善
  - 但 monthly-first ranking 仍落回旧预算答案
  - refreshed three-window mean 仅 `17.47% / 1.027`
  - 结论：仍不通过 liquid500 challenger gate，只保留为 monitored architecture branch
- `encoder_transformer_v1` 稳定性复核已完成：
  - 弱窗 `20240301_20250317` 在 `e24` 改善到 `25.19% / 1.546`
  - refreshed three-window mean = `34.68% / 1.343`
  - 但旧弱窗稳定性仍不够，且 `budget_pressure = true`
  - 结论：high-upside but unstable，当前仍不通过 liquid500 challenger gate

## 4. 当前优先级
1. 持续累积 liquid500 short-alpha 上线后的月度样本，重点监控：
   - `positive_month_ratio`
   - `median monthly excess`
   - `worst month`
   - `top3 positive-month share`
2. liquid500 当前最高优先级是 short-alpha 的 targeted weak-month repair，优先盯：
   - `trend_down_low_vol`
   - `trend_up_low_vol`
   - `2025-07`
   - `2024-01`
   - `2025-10`
3. short-alpha 后续不再优先扩大静态 bridge/profile 搜索；优先做：
   - weak-month 定向修复
   - score-to-weight 映射
   - 执行兑现质量
4. 新的 liquid500 challenger 如要晋级，默认顺序仍是：
   - budget-normalized formal
   - recent realistic gate
   - production promotion
   - 必要时再做 execution policy audit
5. architecture 线若继续推进，优先顺序改为：
   - `encoder_transformer_v1` 稳定性与弱窗修复
   - `graph_off_plain` 作为监控分支按需复核
   - 不再把“更大、更深”本身视为默认升级方向
6. 月度 checkpoint objective 继续优先放在：
   - `dynamic_graph_no_priors`
   - 非默认 liquid500 challenger
   而不是直接改当前 active default 的 fresh retrain 默认值
7. 旧 baseline production root 仅保留为显式回退对照，不再作为默认执行真源

## 5. 当前边界
- formal holdout 负责研究 winner 判决
- recent realistic replay 负责执行 gate
- production full-fit 负责默认执行候选
- active strategy manifest 负责日常执行真源
- strict-resume continuation 不允许在同一训练链中途切换 `checkpoint_selection_objective`
- liquid500 默认执行 winner 与 liquid800 / mainboard 研究 winner必须分开叙述
- 如果要说“当前全项目最高净收益”，必须先进入：
  - `global_deployable_non_capacity_adjusted_v1`
  - 同一成本引擎
  - 同一 execution-policy audit 搜索空间

## 6. 当前关键证据目录
- active execution manifest：
  - `daily_research/output/active_execution_strategy.json`
- global deployable leaderboard：
  - `daily_research/output/global_deployable_strategy_leaderboard_20260406_r1`
- family budget manifest：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- liquid500 short-alpha budget-normalized formal：
  - `daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1`
- liquid500 short-alpha execution policy formal review：
  - `daily_research/output/short_alpha_execution_policy_formal_review_20260405_r1`
- liquid500 short-alpha score-to-weight repair review：
  - `daily_research/output/short_alpha_score_weight_repair_review_20260406_r1`
- liquid500 short-alpha weak-month review：
  - `daily_research/output/short_alpha_weak_month_review_20260405_r1`
- liquid500 short-alpha conditional execution policy review：
  - `daily_research/output/short_alpha_conditional_execution_policy_review_20260405_r1`
- liquid500 short-alpha profit-max production refresh：
  - `daily_research/output/short_alpha_profitmax_production_refresh_20260405_r1`
- liquid500 short-alpha production epoch extension：
  - `daily_research/output/short_alpha_production_epoch_extension_20260405_r1`
- liquid500 short-alpha checkpoint objective compare：
  - `daily_research/output/short_alpha_checkpoint_objective_comparison_20260405_r1`
- monthly landscape review：
  - `daily_research/output/deep_alpha_monthly_landscape_review_20260405_r1`
- architecture protocol refresh：
  - `daily_research/output/deep_alpha_architecture_protocol_refresh_20260406_r1`
- graph_off_plain budget review：
  - `daily_research/output/graph_off_plain_budget_review_20260406_r1`
- encoder_transformer_v1 stability review：
  - `daily_research/output/encoder_transformer_stability_review_20260406_r1`
- liquid800 / mainboard dynamic-graph formal：
  - `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
- dynamic_graph liquid500 challenger formal：
  - `daily_research/output/dynamic_graph_liquid500_challenger_20260405_r1`
- dynamic_graph 同口径 execution-policy review：
  - `daily_research/output/dynamic_graph_no_priors_execution_policy_formal_review_20260406_r1`
