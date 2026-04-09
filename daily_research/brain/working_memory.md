# Daily Research 当前判断

快照日期：`2026-04-09`

## 1. 当前锁定协议
- 用户最新提出的要求仍是最高优先级。
- formal 验证采用滚动窗口；每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
- 当前 formal 主窗示例仍是 `train_end = 2025-03-17`、`valid_start = 2025-03-18`、`valid_end = 2026-03-31`，它代表“该窗起点前最新模型”，不是“故意落后一整年”的旧解释。
- recent 验证现在是 strongest-model research verdict 的必备伴随证据，不允许只报 formal。
- 执行物化仍使用当前可标注最新数据做 `production full-fit`，当前 production `launch_cutoff_date = 2026-04-09`。
- 最后真正写入默认执行的 winner，必须先做 latest-data `production full-fit + highest family budget`，不允许省算力。
- 当前 recent 窗口按最近一年 `12` 个月定义；若以当前 latest completed date `2026-04-09` 计，默认 recent 区间应理解为 `2025-04-10 -> 2026-04-09`。
- 当前默认判决顺序是“月度收益优先、模型偏短线”。

## 2. 当前主线
- 当前 strongest-model gate 已固定为 `liquid500 + execution_first + rolling formal 3 windows + as-of-window latest model + primary_monthly_robust_score + window_count=3`，裁决根为 `daily_research/output/short_alpha_strongest_model_verdict_20260409_r1`。
- 当前最强研究模型是 `short_expert_monthly_v1`。
- 按当前新协议，研究最强模型可以直接作为执行默认；不再额外保留独立 promotion 哲学阻塞层。
- 当前稳定 base model 仍是 `state_liquidity_listwise_v1`，正式复核根为 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- 当前日常执行真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 active strategy 是 `deep_alpha_short_alpha_execalign_production_default`。
- 当前 active candidate label 是 `short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`。
- 当前 active production root 是 `daily_research/output/deep_alpha_short_alpha_execalign_production_default`。
- 当前 active production full-fit run 是 `daily_research/output/deep_alpha_short_alpha_execfirst_production_fullfit_20260409_r1`。
- 当前统一权重语义是 `research_raw_target_weight`。
- 当前统一上限语义是 `follow_research_raw_no_global_cap`。

## 3. 当前 live 执行态
- 当前默认交易计划直接读取 production root 下的 `execution_aligned_daily_live_target_weight_panel.csv` 与 `execution_aligned_daily_live_score_panel.csv`。
- 当前 live effective mode 是 `execution_aligned_live`。
- 当前 live effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 当前默认执行已经切到 `short_expert_monthly_v1 + k2` 这条 latest-data production full-fit 主线。
- `static_fallback_daily_live_target_weight_panel.csv` 仍保留在 production root 里作为运维安全基线，但它不再是当前默认 target-weight 来源。
- `topk3_1d_regoff` 现在只保留为 observation-only 的 targeted repair 分支，不再是当前默认 repair 叙事。
- 最新交易计划已经重刷到 `2026-04-09` 信号、`2026-04-10` 执行，候选标签与 production root 都已切到 `short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`。

## 4. 当前 formal / recent / live 裁决
- strongest-model verdict：`short_expert_monthly_v1` 已在当前 monthly-first formal strongest-model gate 下胜出，根为 `daily_research/output/short_alpha_strongest_model_verdict_20260409_r1`。
- strongest-model recent 验证已经补齐；同一根下最近一年 `12` 个月 companion winner 是 `state_liquidity_listwise_v1`。
- formal stable-base verdict：`state_liquidity_listwise_v1` 仍成立，根为 `daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`。
- execution semantic verdict：`raw + k2` 明确胜过当前 `k1` 与 capped direct，根为 `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`。
- monthly-first execution verdict：`k2` 仍是 execution-side mainline，根为 `daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`。
- recent audit verdict：`k2 > k1 > k3 > topk3`，根为 `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`。
- recent attack recheck verdict：`k2 > k1 > k3` 仍成立，根为 `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r3_attack_recheck`。
- same-window targeted verdict：`topk3` 对 `k2` formal `0/3` 胜，recent gate 也无增益，根为 `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`。

## 5. 30% 强月裁决
- 30% 强月 verdict 根为 `daily_research/output/short_alpha_monthly_attack_signal_weight_verdict_20260409_r1`。
- 当前没有任何 challenger 命中 `portfolio_return >= 30%` 的稳定强月门槛。
- formal attack winner 是 `formal_current_equal_top5_k1_bridge`。
- 但默认执行已经不再等待 `k1 attack bridge` 晋升；当前 production default 已经按 strongest research winner 物化到 `short_expert_monthly_v1 + k2`。
- 当前最重要的攻击分支问题变成：如何让 `formal_current_equal_top5_k1_bridge` 在最近一年 `12` 个月口径下，真正打赢当前已上线的 `short_expert + k2` 默认链。

## 6. 当前真正问题
- 当前问题不是“谁是最强研究模型”还没定；这件事已经由 strongest-model verdict 收口为 `short_expert_monthly_v1`，而且已经完成默认执行物化。
- 当前问题也不是 `recent` 定义还模糊；当前 recent 已固定为最近一年 `12` 个月。
- 当前问题也不是预算不够；current default 已按 latest-data `production full-fit + highest family budget` 物化。
- 如何让新的 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor` 默认链，在最近一年 `12` 个月口径下缩小或反超 `state_liquidity_listwise_v1` 的 companion 优势。
- 如何在当前 `k2` 主线之上改善月度分布、弱月修复与集中度，同时保持当前 raw 统一语义。
- 如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 same-protocol recent/live 上复现而不退化。

## 7. 下一步
- 第一优先级是用同协议 recent 一年窗口，直接比较“当前已上线 `short_expert + k2` 默认链”与“`state_liquidity + k2` companion 基线”的月度优先兑现质量。
- 第二优先级是在不推翻当前默认的前提下，继续围绕 `short_expert + k2` 做 `signal-to-weight / month-trigger` 设计。
- 第三优先级是围绕 `formal_current_equal_top5_k1_bridge` 做阈值微调、月状态定义收敛和 recent/live 扩样，只把它当 research attack branch。
- 第四优先级是持续保持 active manifest、production root、latest trade plan 和 brain 文档四者同源一致。
- broad execution-policy sweep 继续停止。
- 训练默认仍从 `32` 起步；如不够，只允许同模型 `strict resume`。
- 模型侧如重开，仍只从短周期、弱月修补、`penalty-only` 一类窄修复继续。
- active manifest 现在必须显式记录 `effective_live_target_weight_mode / effective_live_execution_profile / effective_live_execution_bridge_meta / effective_live_weight_generation_note`，不再允许 production promotion 后留空。
- latest trade plan 现在会解释当前 effective live mode 与权重生成语义；当前默认模式下，`转权重前分数` 是 promoted production full-fit 的 execution pre-weight score，不要求与最终权重单调一致。
