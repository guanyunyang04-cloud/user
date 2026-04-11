# Daily Research 项目地图

快照日期：`2026-04-10`

## 1. 项目定义
- `daily_research` 是一条以执行后净收益最大化为目标的研究-执行统一链路。
- 当前默认研究目标是 `execution_first`，当前默认判决顺序是“月度收益优先、模型偏短线”。
- 当前默认执行真源是 `daily_research/output/active_execution_strategy.json`。
- 当前项目治理规则是：用户最新要求拥有最高优先级。

## 2. 接管前先问的三个问题
- 这次问题属于 `formal`、`recent` 还是 `live`。
- 这次工作属于研究环、执行环，还是 promotion 边界。
- 这次引用的结果是在比较 base model、execution mainline，还是 attack challenger。

## 3. 双环闭环
- 研究环：
  - 用滚动 formal 协议训练研究模型。
  - 每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
  - 每个窗口的评估段仍保持独立 holdout，不并回该窗训练集。
  - recent 验证是研究 winner 的必备伴随证据。
- 执行环：
  - strongest research winner 可直接进入默认执行物化。
  - `production full-fit` 负责用最新可标注数据 + 当前最高 family budget 物化 live panel、fallback 和真实执行默认值。
  - recent 负责回答“最近一年 `12` 个月有没有兑现”，live 负责回答“当前生产面板实际在跑什么”，两者都不回填成 formal 证据。

## 4. 当前统一语义
- `research_raw_target_weight`
- `follow_research_raw_no_global_cap`
- `monthly-first`
- `short-term bias`
- `strict resume`
- `GPU only`

## 5. 当前主线
- strongest research model：`short_expert_monthly_v1`
- 当前默认接管基线批次仍锚定在 `2026-04-09` 这轮 strongest-model / production 物化体系上。
- strongest-model verdict root：`daily_research/output/short_alpha_strongest_model_verdict_20260409_r1`
- strongest stable base model：`state_liquidity_listwise_v1`
- active execution strategy：`deep_alpha_short_alpha_execalign_production_default`
- active candidate label：`short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`
- active manifest：`daily_research/output/active_execution_strategy.json`
- production root：`daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- current active production run：`daily_research/output/deep_alpha_short_alpha_execfirst_production_fullfit_20260409_r1`
- single-mapping pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`（仅历史观察分支，不是当前默认）

## 6. 当前执行侧现状
- 当前 live effective mode 是 `execution_aligned_live`。
- 当前 live effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 当前实际默认执行来自 production root 下的 `execution_aligned_daily_live_target_weight_panel.csv`。
- `static_fallback_daily_live_target_weight_panel.csv` 继续作为 production safety baseline 保留，但不再是当前默认 target-weight 来源。
- `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 现在只保留为 observation-only 的历史 targeted repair 分支，不再作为当前默认 repair 叙事。
- `active_execution_strategy.json`、pipeline sidecar 与 `latest_trade_plan.txt` 现在都会同步写出 effective profile、bridge meta 和 weight-generation explanation。

## 7. 当前关键结果
- strongest-model verdict 根：`daily_research/output/short_alpha_strongest_model_verdict_20260409_r1`
- formal base-model verdict 根：`daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`
- execution semantic verdict 根：`daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
- monthly-first scoreboard 根：`daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`
- recent audit 根：`daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`
- recent attack recheck 根：`daily_research/output/short_alpha_production_execution_policy_audit_20260409_r3_attack_recheck`
- same-window targeted review 根：`daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`
- 30% 强月 verdict 根：`daily_research/output/short_alpha_monthly_attack_signal_weight_verdict_20260409_r1`
- recent 根因拆解根：`daily_research/output/short_alpha_recent_root_cause_breakdown_20260410_r1`
- current default signal/cash repair verdict 根：`daily_research/output/short_alpha_current_default_signal_cash_repair_20260410_r1`
- current default follow-up repair verdict 根：`daily_research/output/short_alpha_current_default_followup_repair_20260410_r1`
- current default gross-control sweep 根：`daily_research/output/short_alpha_current_default_gross_control_sweep_20260410_r1`
- policy_v2 constrained formal review 根：`daily_research/output/short_alpha_policy_v2_constrained_execution_review_20260410_r1`
- policy_v2 formal loss breakdown 根：`daily_research/output/short_alpha_policy_v2_formal_loss_breakdown_20260410_r1`
- policy_v3 latest formal review 根：`daily_research/output/short_alpha_policy_v3_review_20260410_r1`
- policy_v3 latest recent eval 根：`daily_research/output/short_alpha_policy_v3_recent_eval_20260410_r1`

## 8. 当前主问题
- strongest-model research 层问题已经收口完成：formal winner、recent companion evidence、latest-data production full-fit 与默认执行接管都已打通。
- live 层当前主问题：如何让 `short_expert_monthly_v1 + k2` 新默认链，在最近一年 `12` 个月口径下缩小或反超 `state_liquidity_listwise_v1` 的 companion 优势。
- current default 的 recent 一年主因已经有第一轮量化拆解：问题更像是“顺风状态兑现不足 + cash sizing 不够状态化 + score-to-weight 转换偏弱”，不是“模型完全不会看市场状态”。
- 股票池当前更像是天花板约束，而不是 recent 差距的第一主因；winner 与 companion 当前使用的是同一个固定 `liquid500` 池。
- execution optimization 层问题：如何在 `k2` 主线之上继续改善月度分布、弱月修复与集中度，同时维持当前 raw 统一语义。
- current-default 小修包已经有第一轮 same-protocol verdict：`winner_current_target_market_state_guard_v1` 是当前默认链自己内部的 repair winner，说明眼下最值得先补的是状态化总仓位，不是更激进的 `score_weight_k2`。
- current-default 第二轮 follow-up verdict 已把 winner-side 最强修补推进到 `winner_current_target_market_state_guard_v2_balance`，并且 gross-control 迁移实验出现了同向读数；这说明 current default 与 companion 的差距里，控制层已经不只是嫌疑，而是带量化证据的主因之一。
- research attack 层问题：如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 recent/live 上复现而不退化。
- 当前还没有任何 challenger 实现稳定 `30%+` 月收益门槛。
- 因此下一阶段最高优先级不再是重新决定谁上线，而是围绕当前已上线的 `short_expert + k2` 默认链，先做 `market_state_guard_v2_balance` 一类的 `cash sizing / gross-control` 修补，再做 `month-trigger` 与窄版 `signal-to-weight`，并用同协议 recent 一年窗口持续和 `state_liquidity` 对照。

## 9. 决策闭环
- rolling formal head-to-head + recent validation 负责确认 strongest model 与 stable base model。
- production full-fit 与 fallback refresh 负责把 strongest winner 物化成真实 live panel 和默认执行。
- single-mapping pipeline 与 activation 现在只负责历史 repair / observation 分支，不再负责当前默认执行。
- `run_trade_plan.py` 负责把当前可执行计划和 bridge 解释展示出来。
- monthly-first scoreboard 与 same-window targeted review 负责决定 repair 路径是 promotion、observation 还是 retire。
- 强月 verdict 负责回答“是否更接近 30% 强月目标”，不能替代普通 monthly-first verdict。
- 稳定结论再回写到 `semantic / project_map / working / procedural / action / episodic` 六层分脑。
- 2026-04-10 最新补充：
  - current default 主线的 hand-crafted 控制层目前已经基本收敛：best gross-only 修补是 `winner_gross_map_u097_f098_d088`，提升真实存在，但仍未超过 companion。
  - `policy_v2` 的 constrained formal review 已证明：best constrained answer 仍是 `regoff_k2_20d_ensemble_native_anchor`，所以当前 formal gap 不能再简单归因成“桥太慢”。
  - `policy_v2` 的 formal loss breakdown 已证明：`raw_1d` 不是可部署答案，手工收紧候选数与 gross band 也没有单独救回 formal gap；下一步 learned-control 应优先改 learned score-to-weight 本体，而不是继续堆更多手工稀疏化。
  - `policy_v3` 的 latest-window formal / recent 都已补齐；它 formal `monthly_robust_score = 0.0786`、recent `monthly_robust_score = -0.0143`，都没有打赢 `policy_v2` 或 companion，因此当前 learned-control 主研究分支仍是 `short_expert_policy_v2`，不是 `policy_v3`。
  - 本轮所有正式训练、formal / recent 回放与结论生成均已锁定 `yolos` 环境。
