# Daily Research 当前判断

快照日期：`2026-04-05`

## 1. 当前总判断
- 当前默认执行仍保持不变：
  - `active_execution_strategy -> baseline_current_execfirst_winner`
- 当前最重要的新结论是：
  - `state_liquidity_listwise_v1 + regoff_k2_10d_ensemble_native_anchor` 已成为新的 liquid500 execution-upgrade 首选
- 当前 liquid800 / mainboard 研究线最重要的新结论是：
  - `dynamic_graph_no_priors` 明显优于 `dynamic_graph_v1`

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

## 3. 最近窗升级门槛判断
- `state_liquidity_listwise_v1` 已通过 recent realistic replay gate：
  - `short_alpha_execalign_realistic` full-period annual / excess annual / excess Sharpe = `71.19% / 51.70% / 2.964`
  - 当前默认 `regoff_k2_realistic` = `44.12% / 25.75% / 1.722`
  - named-window 胜负 = `4/5`
- 当前最接近上线切换的不是 `structure`，而是 `short_alpha`。

## 4. 当前默认执行状态
- 当前 active strategy 仍指向：
  - `daily_research/output/active_execution_strategy.json`
  - `strategy_name = baseline_current_execfirst_winner`
- 当前默认 production root 仍为：
  - `daily_research/output/deep_alpha_liquid500_dynamic_graph_bridge_production_default`
- 当前结论是：
  - 新 winner 已经出现
  - 但还没有完成 production full-fit promotion
  - 因此默认执行不应静默切换

## 5. 当前优先级
1. 先把 `state_liquidity_listwise_v1` 跑成 production full-fit promotion。
2. 用 production 口径验证它是否仍能稳定赢过当前 active default。
3. 若 promotion 复现优势，再更新 `active_execution_strategy.json`。
4. `dynamic_graph_no_priors` 保持在 liquid800 / mainboard 独立研究线推进。

## 6. 暂不优先做的事
- 不再重开 `structure_context_only` 大矩阵。
- 不再手工给所有家族统一写死同一个 `--epochs`。
- 不把 liquid500 默认执行升级判断与 liquid800 研究方向判断混写。

## 7. 当前关键证据目录
- family budget manifest：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- architecture budget-normalized formal：
  - `daily_research/output/deep_alpha_architecture_execalign_formal_20260404_monthly_budgetnorm_r1`
- short-alpha budget-normalized formal：
  - `daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1`
- short-alpha recent realistic gate：
  - `daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1/recent_h2h_short_alpha_vs_current_default`
- dynamic-graph budget-normalized formal：
  - `daily_research/output/dynamic_graph_ablation_formal_20260404_monthly_budgetnorm_r1`
