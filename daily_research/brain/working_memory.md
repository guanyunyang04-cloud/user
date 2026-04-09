# Daily Research 当前判断

快照日期：`2026-04-09`

## 1. 当前锁定协议
- 用户最新提出的要求仍是最高优先级。
- 研究模型协议已经锁死：研究模型只允许使用 formal 评估开始前一天及更早的可标注数据，最近 `12` 个自然月完整预留做评估。
- 当前 formal 主窗示例是 `train_end = 2025-03-17`、`valid_start = 2025-03-18`、`valid_end = 2026-03-31`。
- 执行模型协议也已锁死：只有执行模型才允许使用最新可标注数据做 `production full-fit`，当前 production `launch_cutoff_date = 2026-04-08`。
- 当前 recent/live 监控窗口不是“最近一年”，而是 `2026-03-05 -> 2026-04-08`。
- 当前默认判决顺序是“月度收益优先、模型偏短线”。

## 2. 当前主线
- 当前底层最强模型仍是 `state_liquidity_listwise_v1`，正式复核根为 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- 当前日常执行真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 active strategy 是 `state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`。
- 当前 active execution pipeline root 是 `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`。
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。

## 3. 当前 live 执行态
- 当前 live 月 `2026-04` 仍未触发 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`，所以 live panel 走的是 `static_fallback_daily_live_target_weight_panel.csv`。
- 当前 live effective mode 是 `static_fallback`。
- 当前 live effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 这个 fallback 已不再吃旧的 `25%` capped panel，而是从 production raw panel 通过 `k2 / 5d / all-offset` bridge 重新生成。
- `topk3_1d_regoff` 现在只保留为 observation-only 的 targeted repair 分支，不再是当前默认 repair 叙事。

## 4. 当前 formal / recent / live 裁决
- formal base-model verdict：`state_liquidity_listwise_v1` 仍成立，根为 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- execution semantic verdict：`raw + k2` 明确胜过当前 `k1` 与 capped direct，根为 `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`。
- monthly-first execution verdict：`k2` 仍是 execution-side mainline，根为 `daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`。
- recent audit verdict：`k2 > k1 > k3 > topk3`，根为 `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`。
- recent attack recheck verdict：`k2 > k1 > k3` 仍成立，根为 `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r3_attack_recheck`。
- same-window targeted verdict：`topk3` 对 `k2` formal `0/3` 胜，recent gate 也无增益，根为 `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`。

## 5. 30% 强月裁决
- 30% 强月 verdict 根为 `daily_research/output/short_alpha_monthly_attack_signal_weight_verdict_20260409_r1`。
- 当前没有任何 challenger 命中 `portfolio_return >= 30%` 的稳定强月门槛。
- formal attack winner 是 `formal_current_equal_top5_k1_bridge`。
- recent/live gate winner 仍是 `recent_current_live_k2_static`。
- 当前最重要的新问题因此变成：`formal attack bridge = k1`，但 `recent live gate = k2`，两者还没有收敛成同一条可上线主线。

## 6. 当前真正问题
- 当前问题不是主模型失效，也不是 `k1 / k2 / topk3` 谁当主线还没定。
- 当前问题也不是预算不够；production 这条线已经补到 `64` 且预算压力解除。
- 当前真正未解的是：
  - 如何在 `k2` 主线之上改善月度分布、弱月修复与集中度，同时保持当前 raw 统一语义。
  - 如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 recent/live 上复现而不退化。

## 7. 下一步
- 第一优先级是围绕 `k2` 主线做 `signal-to-weight / month-trigger` 设计。
- 第二优先级是围绕 `formal_current_equal_top5_k1_bridge` 做阈值微调、月状态定义收敛和 recent/live 扩样。
- broad execution-policy sweep 继续停止。
- 训练默认仍从 `32` 起步；如不够，只允许同模型 `strict resume`。
- 模型侧如重开，仍只从短周期、弱月修补、`penalty-only` 一类窄修复继续。
- active manifest 现在显式记录 `effective_live_target_weight_mode / effective_live_execution_profile / effective_live_execution_bridge_meta / effective_live_weight_generation_note`，不再只靠 `live_trigger_monitor.json` 侧面判断当前实际执行态。
- latest trade plan 现在会解释 bridge 语义；若当前 live 月走 `static_fallback + regoff_k2_5d_ensemble_native_anchor`，会明确提示这是 `5d / all-offset / topk2` bridge，`转权重前分数` 只是信号日 raw score，不要求与最终权重单调一致。
