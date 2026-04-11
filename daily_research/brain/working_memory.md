# Daily Research 当前判断

快照日期：`2026-04-10`

## 1. 当前锁定协议
- 用户最新提出的要求仍是最高优先级。
- formal 验证采用滚动窗口；每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
- 当前 formal 主窗示例仍是 `train_end = 2025-03-17`、`valid_start = 2025-03-18`、`valid_end = 2026-03-31`，它代表“该窗起点前最新模型”，不是“故意落后一整年”的旧解释。
- recent 验证现在是 strongest-model research verdict 的必备伴随证据，不允许只报 formal。
- 执行物化仍使用当前可标注最新数据做 `production full-fit`，当前 production `launch_cutoff_date = 2026-04-09`。
- 最后真正写入默认执行的 winner，必须先做 latest-data `production full-fit + highest family budget`，不允许省算力。
- 当前 recent 窗口按最近一年 `12` 个月定义；若以当前 latest completed date `2026-04-10` 计，默认 recent 区间应理解为 `2025-04-11 -> 2026-04-10`。
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
- current default follow-up repair verdict：`short_alpha_current_default_followup_repair_20260410_r1` 已补齐；第二轮 winner-side repair winner 已进一步收口到 `winner_current_target_market_state_guard_v2_balance`。
- follow-up 结论是：`market_state_guard_v2_balance` 把 monthly_robust_score 从当前 default 的 `0.046` 提到 `0.060`，也高于上一轮 `v1` 的 `0.052`；但它仍未追平 companion baseline 的 `0.073`，所以当前正确定位仍是“最强 repair candidate”，不是“已经足以直接替代 companion 的最终答案”。
- follow-up 里的控制层迁移结果是同向的：winner 借用 companion 的逐日 gross / 状态 gross 后，monthly_robust 分别提高约 `0.012 / 0.012`；companion 借用 winner 的逐日 gross / 状态 gross 后，monthly_robust 分别下降约 `0.006 / 0.002`。这说明 gross-control 的确是 current default 与 companion 差距的重要来源。
- follow-up 也说明：低波动 `score blend` 虽然能把 score/weight Spearman 从 `0.206` 推到约 `0.218`，但 monthly_robust 仍低于 `market_state_guard_v2_balance`；因此 narrow `signal-to-weight` 还不是当前第一修补主线。
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
- 第二轮 current-default follow-up repair verdict 又把方向继续压实了一步：`market_state_guard_v2_balance` 能把 monthly_robust_score 进一步抬到 `0.060`，同时控制层迁移实验显示 gross-control 改好时 winner 会同步改善、companion 会同步变差；因此“先修 gross-control / cash sizing，再修窄版 signal-to-weight”已经不再只是猜测，而是有同向迁移证据支持的判断。
- 如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 same-protocol recent/live 上复现而不退化。
- 如何让 `short_expert_policy_v1` 在 formal 3 windows 与 recent 12 个月上证明“学出来的选股/持仓转换”优于当前 `short_expert_monthly_v1 + k2` 默认链。
- deepening 本身已经做过一轮同窗验证；下一步不该再把“单纯加深 backbone”当默认升级方向。

## 7. 下一步
- 第一优先级已经从“先修 signal-to-weight”收口成“先修 cash sizing / gross-control”：当前最强 repair candidate 已从 `winner_current_target_market_state_guard_v1` 进到 `winner_current_target_market_state_guard_v2_balance`，说明 `short_expert + k2` 现阶段最值得先补的是状态化总仓位，而不是更激进的 score bridge。
- 第二优先级是在 current default 主线上继续微调 `market_state_guard_v2_balance` 的 gross map，让它尽量把 companion 的 `recent` 一年月度稳健差距从 `-0.013` 再往下压，而不是立刻切到 `score_weight_k2_static` 这类高波动攻击桥。
- 第三优先级才是围绕 `signal-to-weight` 做更窄修补，只保留不破坏 monthly_robust_score 的版本；第二轮 follow-up 里 `score blend` 虽然抬高了对分数排序的跟随度，但仍未超过 `market_state_guard_v2_balance`，因此它现在只能做 secondary branch。
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
- 2026-04-10 最新补充：
  - `short_alpha_current_default_gross_control_sweep_20260410_r1` 已补齐；winner-side 最强 gross-only 修补为 `winner_gross_map_u097_f098_d088`，`monthly_robust_score = 0.0616`，相对 current default `0.0456` 提升约 `+0.0160`，但仍低于 companion `0.0726`。
  - 这轮说明 hand-crafted 控制层已经接近当前上限：`month-trigger` overlay 与窄版 `score-to-weight` overlay 都没有继续推翻 gross-only winner，所以“先冻结 best gross-control，再让 learned control 去学它”已经成立。
  - `short_alpha_policy_v2_review_20260410_r1` 已补齐；`short_expert_policy_v2` 在 same-window formal 里把 `positive month ratio` 提到 `83.33%`、把 `worst month` 收窄到 `-1.12%`、把 `top3 positive share` 降到 `50.88%`，明显强于 `policy_v1` 的月度稳定性，但 `excess annual = 55.83%`，仍低于 `short_expert_monthly_v1` 的 `91.76%`，也低于 `policy_v1` 的 `65.84%`。
  - `short_alpha_policy_v2_recent_eval_20260410_r1` 也已补齐；`short_expert_policy_v2` 的 recent 一年 `monthly_robust_score = 0.0149`，显著高于 current default `-0.0117` 与 `policy_v1` 的 `-0.0470`，且仅略低于 companion `0.0154`；recent `excess annual = 22.21%`，也略高于 current default 的 `20.11%`。
  - learned-control 当前结论已经更新为：`policy_v2 > policy_v1`，并且 recent 一年几乎追平 companion 的 robust，但 formal 仍未打赢 `short_expert_monthly_v1`，所以它现在是最有前途的 learned-control research branch，还不是新的默认执行 winner。
  - 接下来 learned-control 主线不再是继续讨论 `policy_v1`，而是围绕 `short_expert_policy_v2` 压缩 formal 收益弹性损失，重点排查它为什么自动漂到 `regoff_k2_20d_ensemble_native_anchor`，以及如何在保住 recent 稳健性的同时，把 formal excess annual 拉回到当前主线附近。
  - 本轮 formal / recent 的正式训练、回放与汇总已统一锁定在 `yolos` 环境；其它环境结果不得再作为正式证据引用。
  - `short_alpha_policy_v2_constrained_execution_review_20260410_r1` 已补齐；`policy_v2` 的 constrained formal best 仍是 `regoff_k2_20d_ensemble_native_anchor`，`monthly_robust_score = 0.0919`，相对当前 mainline formal `0.1012` 仍落后约 `-0.0093`。这说明 `policy_v2` 的 formal 收益折损不能再简单归因成“桥太慢”。
  - `short_alpha_policy_v2_formal_loss_breakdown_20260410_r1` 已补齐；`raw_1d` 明确不可部署，手工候选数收缩与更紧 gross band 也没有单独救回 formal gap，因此下一代 learned-control 不能只靠继续手工稀疏化。
  - `short_alpha_policy_v3_review_20260410_r1` 已补齐；`short_expert_policy_v3` 的 formal profile 仍是 `regoff_k2_20d_ensemble_native_anchor`，formal `monthly_robust_score = 0.0786`，低于 `policy_v2 = 0.0830`、`state_liquidity_listwise_v1 = 0.0897` 和当前 mainline `0.1012`，不是 formal winner。
  - `short_alpha_policy_v3_recent_eval_20260410_r1` 已补齐；在 `2025-04-11 -> 2026-04-10` recent 一年窗口里，`short_expert_policy_v3` 的 `monthly_robust_score = -0.0143`，虽然高于 current default `-0.0370`，但仍低于 `policy_v2 = -0.0061` 与 companion `0.0588`，也不是 recent winner。
  - learned-control 当前主研究分支仍应保持为 `short_expert_policy_v2`；`policy_v3` 第一版证明“把更多控制动作学进去”本身可行，但它既没有解决 formal gap，也没有在 latest recent 一年里打赢 `policy_v2`。
  - 当前默认执行不变，继续维持 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`；本轮不触发新的 production promotion。
