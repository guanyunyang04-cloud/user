# Daily Research 当前判断

快照日期：`2026-04-09`

## 1. 当前主线
- 当前底层最强模型仍是 `state_liquidity_listwise_v1`，正式复核根为 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- 当前日常执行真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 active default 仍是 `state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`。
- 当前 active execution pipeline root 是 `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r1`。
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。

## 2. 执行侧结论
- 当前唯一正式保留的 execution repair 仍是 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`。
- 当前 live 月 `2026-04` 仍未触发这条映射，所以 live panel 走的是 `static_fallback_daily_live_target_weight_panel.csv`。
- 这个 fallback 已不再吃旧的 `25%` capped panel，而是从 production raw panel 重新桥接出来。
- 最新 trade plan 已按统一语义切到 `40% / 20% / 20% / 20%`，不再是假性 `25%` fallback。

## 3. 生产模型状态
- 最新 production full-fit fresh run 是 `daily_research/output/deep_alpha_short_alpha_execfirst_production_fullfit_20260409_r1`。
- 这轮 `32` 起训后被判为 undertrained，所以又按 strict resume 补到了 `64`。
- 当前稳定 production run 是 `daily_research/output/short_alpha_production_epoch_extension_20260409_r1/runs/short_alpha_production_e64`。
- 当前 production root `daily_research/output/deep_alpha_short_alpha_execalign_production_default` 已同步到这个 `e64` 版本。
- 当前 production root 同时保留三张 panel：
- `daily_live_target_weight_panel.csv` = research raw uncapped
- `portfolio_capped_daily_live_target_weight_panel.csv` = capped reference only
- `static_fallback_daily_live_target_weight_panel.csv` = raw panel 经过 `regoff_k1_5d_ensemble_native_anchor` bridge 的正式 fallback

## 4. 最新收益判断
- strongest model 本体能力仍成立，但当前短板仍在最近阶段的执行兑现。
- 最新 costed active-candidate recent recheck 根是 `daily_research/output/recheck_active_execution_candidate_20260409_r3_unified_costed`。
- 这轮结果是 annual `7.30%`、excess annual `-4.74%`、excess Sharpe `-0.226`。
- 它比旧的 capped fallback 语义更统一，但最近区间兑现更弱。
- 当前结论因此不是“统一后收益更高”，而是“统一后语义更干净，但最近收益更差”。

## 5. 当前问题
- 当前问题不是主模型失效，而是 live 月没有触发最强映射。
- 当前问题也不是预算不够；production 这条线已经补到 `64` 且预算压力解除。
- 当前真正的未解问题是：如果目标是最大真实收益，项目后续需要回答“研究 raw 无上限语义”是否应继续作为全链默认，还是应回到“全链统一 capped 语义”重新比较。

## 6. 当前纪律
- 用户最新提出的要求拥有最高优先级。
- 默认目标是最有效，不是最小改动。
- 训练默认从 `32` 起步。
- 同模型扩预算只允许 `strict resume`。
- 训练一律 `GPU`。
- active candidate backtest 默认必须带真实成本，不允许再靠手工 CLI 临时补。

## 7. 下一步
- 第一优先级不是再找新映射，而是正面回答 `raw no-cap` 与 `global cap` 哪条全链收益更高。
- 第二优先级才是继续观察 `expand|stable -> topk3` 的真实触发样本。
- 模型侧如重开，仍只从 `penalty-only` 窄修复继续。
- `2026-04-09` 一致性补充：
- active manifest 现在显式记录 `effective_live_target_weight_mode / effective_live_execution_profile / effective_live_execution_bridge_meta / effective_live_weight_generation_note`，不再只靠 `live_trigger_monitor.json` 侧面判断当前实际执行态。
- latest trade plan 现在会解释 bridge 语义；若当前 live 月走 `static_fallback + regoff_k1_5d_ensemble_native_anchor`，会明确提示这是 `5d / all-offset / topk1` bridge，`转权重前分数` 只是信号日 raw score，不要求与最终权重单调一致。
- `DeepAlphaConfig` 也已同步到 `32 / 16` 默认训练预算；训练侧旧 execution-alignment 默认 `regoff_k2_10d_ensemble_native_anchor` 已从主入口和 retrain-frequency fallback 中移除。
