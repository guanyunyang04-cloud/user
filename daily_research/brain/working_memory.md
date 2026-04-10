# Daily Research 当前判断

快照日期：`2026-04-10`

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
- 当前最新“让模型学习 score -> candidate pool -> raw target weight -> gross exposure”的研究分支是 `short_expert_policy_v1`，设计稿在 `daily_research/output/short_alpha_policy_model_v1_design_20260409.md`。
- `short_expert_policy_v1` 已完成 validation-panel smoke：基于 `short_expert_monthly_v1` 的 `2025-03-18 -> 2026-02-27` 验证面板可稳定导出 learned target-weight，`230` 个交易日平均 gross exposure 约 `0.676`，平均正持仓数 `10`。
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
- learned policy review verdict：`short_expert_policy_v1` 已完成 latest formal single-window review，根为 `daily_research/output/short_alpha_policy_v1_review_20260409_r1`；结果为 excess annual `65.84%`、excess Sharpe `3.901`、positive month `75.00%`、median monthly excess `3.53%`。
- 但 `short_expert_policy_v1` 仍弱于同窗 `short_expert_monthly_v1` 的 excess annual `91.76%`、excess Sharpe `4.852`、positive month `83.33%`、median monthly excess `4.61%`，因此当前它还是 promising research branch，不是 strongest winner。
- deep capacity verdict：`short_alpha_deep_capacity_review_20260409_r1` 已补齐；纯加深 `patch_transformer` 没有带来 formal uplift，`short_expert_policy_v1_deep` 虽然接近，但仍未打赢 `short_expert_monthly_v1`。
- `short_expert_mamba_policy_v1` 已尝试启动，但在当前 `RTX 2060 6GB` 上同协议 wall-clock 吞吐过慢，未纳入这轮 formal winner 判定。
- deep capacity recent eval：`short_alpha_deep_capacity_recent_eval_20260410_r1` 已补齐；formal winner `short_expert_monthly_v1` 在 `2025-04-10 -> 2026-04-09` recent 一年里的 excess annual 为 `20.11%`、excess Sharpe 为 `1.154`、positive month ratio 为 `53.85%`、median monthly excess 仅 `0.02%`、worst month 为 `-6.89%`，结论是“有正超额，但月度分布偏弱”。
- current default signal/cash repair verdict：`short_alpha_current_default_signal_cash_repair_20260410_r1` 已补齐；overall recent winner 仍是 `state_liquidity_listwise_v1` companion baseline，但 current-default repair winner 已变成 `winner_current_target_market_state_guard_v1`。
- 这轮 current-default repair 的直接结论是：`cash-sizing guard` 比 `signal-to-weight` 更接近正确方向。`winner_current_target_market_state_guard_v1` 的 monthly_robust_score 为 `0.052`，高于当前 default `0.046`；而 `winner_score_weight_k2_static` 虽然 excess annual 冲到 `49.69%`，但 monthly_robust_score 只有 `0.014`，说明它更像高波动攻击桥，不是当前要的月度稳健修补。
- repair winner 的 live preview 已落到 `daily_research/output/short_alpha_current_default_signal_cash_repair_20260410_r1/live_preview/trade_plan/latest_trade_plan.txt`，当前 `2026-04-09` 信号对应的 market state 仍是 `trend_down_low_vol`，预览计划没有建议动作。
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
- `short_alpha_recent_root_cause_breakdown_20260410_r1` 已给出 recent 一年主因拆解：当前月度分布不稳的第一主因更像是“顺风状态兑现不足 + 现金/总仓位不够状态化 + score-to-weight 转换偏弱”，不是“模型完全不会看状态”。
- 同一根 recent 拆解还表明：股票池是天花板约束，但不是当前 winner 与 companion 差距的第一主因；两条线使用的是同一个固定 `liquid500` 池。
- 当前最明确的三条量化信号是：winner 在 `trend_up_low_vol` 的日均超额只有 `0.1683%`，低于 companion 的 `0.2745%`；winner 的 active-date score/weight Spearman 为 `0.206`，低于 companion 的 `0.235`；winner 的 down/up gross exposure 比例为 `1.032`，说明它没有在弱状态里明显更保守。
- 现在 current-default repair verdict 进一步把优先级压实了：第一包小修里，`market_state_guard_v1` 能把 monthly_robust_score 从 `0.046` 提到 `0.052`，但还打不到 companion 的 `0.073`；纯 `score_weight_k2` 路线虽然把年化拉高，却把月度稳健性拖坏了。
- 如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 same-protocol recent/live 上复现而不退化。
- 如何让 `short_expert_policy_v1` 在 formal 3 windows 与 recent 12 个月上证明“学出来的选股/持仓转换”优于当前 `short_expert_monthly_v1 + k2` 默认链。
- deepening 本身已经做过一轮同窗验证；下一步不该再把“单纯加深 backbone”当默认升级方向。

## 7. 下一步
- 第一优先级已经从“先修 signal-to-weight”收口成“先修 cash sizing”：当前最强 small repair 是 `winner_current_target_market_state_guard_v1`，说明 `short_expert + k2` 现阶段最值得先补的是状态化总仓位，而不是更激进的 score bridge。
- 第二优先级是在 current default 主线上继续微调 `market_state_guard` 的 gross map，让它尽量逼近 companion 的 recent 一年月度分布，而不是立刻切到 `score_weight_k2_static` 这类高波动攻击桥。
- 第三优先级才是围绕 `signal-to-weight` 做更窄修补，只保留不破坏 monthly_robust_score 的版本；当前 `power125` 与 `score_weight_k2` 的证据已经说明，纯加权进攻不能直接替代 cash guard。
- 第四优先级继续做“当前已上线 `short_expert + k2` 默认链”与“`state_liquidity + k2` companion 基线”的同协议 recent 一年月度优先对照，确保修补方向持续有效。
- 第五优先级并行保留 `short_expert_policy_v1` 研究分支，优先验证它能否把 score-to-weight 的手写桥接进一步内生化，而不是继续单纯堆深 backbone。
- 第六优先级才是单独评估“固定 `liquid500` vs 动态 rolling pool”；在同池 recent 拆解已经完成前，不允许先把问题归咎为池子太窄。
- 第七优先级是围绕 `formal_current_equal_top5_k1_bridge` 做阈值微调、月状态定义收敛和 recent/live 扩样，只把它当 research attack branch。
- 第八优先级是持续保持 active manifest、production root、latest trade plan 和 brain 文档四者同源一致。
- broad execution-policy sweep 继续停止。
- 训练默认仍从 `32` 起步；如不够，只允许同模型 `strict resume`。
- 模型侧如重开，仍只从短周期、弱月修补、`penalty-only` 一类窄修复继续。
- active manifest 现在必须显式记录 `effective_live_target_weight_mode / effective_live_execution_profile / effective_live_execution_bridge_meta / effective_live_weight_generation_note`，不再允许 production promotion 后留空。
- latest trade plan 现在会解释当前 effective live mode 与权重生成语义；当前默认模式下，`转权重前分数` 是 promoted production full-fit 的 execution pre-weight score，不要求与最终权重单调一致。
