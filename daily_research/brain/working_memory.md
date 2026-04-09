# Daily Research 当前判断

快照日期：`2026-04-09`

## 1. 当前主线
- 当前底层最强模型仍是 `state_liquidity_listwise_v1`，正式复核根为 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- 当前日常执行真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 active strategy 是 `state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`。
- 当前 active execution pipeline root 是 `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`。
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。
- 当前默认判决顺序是“月度收益优先、模型偏短线”。

## 2. 当前 live 执行态
- 当前 live 月 `2026-04` 仍未触发 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`，所以 live panel 走的是 `static_fallback_daily_live_target_weight_panel.csv`。
- 当前 live effective mode 是 `static_fallback`。
- 当前 live effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 这个 fallback 已不再吃旧的 `25%` capped panel，而是从 production raw panel 通过 `k2 / 5d / all-offset` bridge 重新生成。
- `topk3_1d_regoff` 现在只保留为 observation-only 的 targeted repair 分支，不再是当前默认 repair 叙事。

## 3. 生产模型状态
- 最新 production full-fit fresh run 是 `daily_research/output/deep_alpha_short_alpha_execfirst_production_fullfit_20260409_r1`。
- 这轮 `32` 起训后被判为 undertrained，所以又按 strict resume 补到了 `64`。
- 当前稳定 production run 是 `daily_research/output/short_alpha_production_epoch_extension_20260409_r1/runs/short_alpha_production_e64`。
- 当前 production root `daily_research/output/deep_alpha_short_alpha_execalign_production_default` 已同步到这个 `e64` 版本。
- 当前 production root 同时保留三张 panel：
- `daily_live_target_weight_panel.csv` = research raw uncapped
- `portfolio_capped_daily_live_target_weight_panel.csv` = capped reference only
- `static_fallback_daily_live_target_weight_panel.csv` = raw panel 经过 `regoff_k2_5d_ensemble_native_anchor` bridge 的正式 fallback

## 4. 当前执行侧裁决
- strongest model 本体能力仍成立，当前执行侧主线已经稳定到 `raw + k2`。
- same-protocol 语义裁决根为 `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`，在同窗同成本下，`raw_regoff_k2_5d_ensemble_native_anchor` 明确胜过当前 `k1` 与 capped direct。
- 月度优先 scoreboard 根为 `daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`，在 common window 与 recent window 上都继续把 `k2` 排在第一。
- same-window targeted review 根为 `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`，强制 `topk3` 对 `k2` 的 leave-window-out formal 结果为 annual wins `0/3`、Sharpe wins `0/3`，recent gate 也为中性。
- 当前执行侧结论因此已经收敛为：`k2` 是 mainline，`topk3` 是观察分支。

## 5. 当前问题
- 当前问题不是主模型失效，也不是 `k1 / k2 / topk3` 谁当主线还没定。
- 当前问题也不是预算不够；production 这条线已经补到 `64` 且预算压力解除。
- 当前真正未解的是：如何在 `k2` 主线之上改善月度分布、弱月修复与集中度，同时保持当前 raw 统一语义。

## 6. 当前纪律
- 用户最新提出的要求拥有最高优先级。
- 默认目标是最有效，不是最小改动。
- 训练默认从 `32` 起步。
- 同模型扩预算只允许 `strict resume`。
- 训练一律 `GPU`。
- active candidate backtest 默认必须带真实成本，不允许再靠手工 CLI 临时补。

## 7. 下一步
- 第一优先级是围绕 `k2` 主线做 `signal-to-weight / month-trigger` 设计。
- 第二优先级才是继续观察 targeted repair 是否会出现新的同窗月度优先增益证据。
- broad execution-policy sweep 继续停止。
- 模型侧如重开，仍只从短周期、弱月修补、`penalty-only` 一类窄修复继续。
- `2026-04-09` 一致性补充：
- active manifest 现在显式记录 `effective_live_target_weight_mode / effective_live_execution_profile / effective_live_execution_bridge_meta / effective_live_weight_generation_note`，不再只靠 `live_trigger_monitor.json` 侧面判断当前实际执行态。
- latest trade plan 现在会解释 bridge 语义；若当前 live 月走 `static_fallback + regoff_k2_5d_ensemble_native_anchor`，会明确提示这是 `5d / all-offset / topk2` bridge，`转权重前分数` 只是信号日 raw score，不要求与最终权重单调一致。
- `DeepAlphaConfig` 也已同步到 `32 / 16` 默认训练预算；训练侧旧 execution-alignment 默认 `regoff_k2_10d_ensemble_native_anchor` 已从主入口和 retrain-frequency fallback 中移除。

## 8. 用户偏好更新
- 用户最新偏好：更看重“每个月的收益评估”，希望模型整体偏短线。
- 从现在开始，月度收益质量高于长窗口年化均值本身。
- 后续判断优先看：月度正收益占比、月度中位数、弱月回撤、坏月修复能力、最近 1-3 个月兑现质量。
- `annual_return / excess_annual_return / Sharpe` 继续保留，但不再单独作为最上位裁决标准。

## 9. 月度优先执行裁决
- monthly-first scoreboard root：`daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`
- recent audit root：`daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`
- targeted review root：`daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`
- execution-side short-line mainline 已明确为 `regoff_k2_5d_ensemble_native_anchor`
- 在 common window `2025-03-18 -> 2026-04-08` 上，`k2` 仍是月度优先评分下最好的 controlled raw bridge。
- 在 recent window `2026-03-05 -> 2026-04-08` 上，`k2` 也继续以 `monthly_robust_score` 排在 `k1 / k3 / topk3_1d_regoff` 之前。
- 强制 targeted repair `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 已不再打赢静态 `k2`；same-window clipped leave-window-out 结果为 annual wins `0/3`、Sharpe wins `0/3`。
- recent gate 为中性，因为最近两个月依然都选到 `regoff_k2_5d_ensemble_native_anchor`。
- 因此下一步执行侧工作应转向 `signal-to-weight / month-trigger design`，而不是继续做简单 regime swapping。
