# Daily Research 项目地图

快照日期：`2026-04-12`

## 1. 项目定义
- `daily_research` 是一条以执行后净收益最大化为目标的研究-执行统一链路。
- 当前默认研究目标是 `execution_first`，当前默认判决顺序是“月度收益优先、模型偏短线”。
- 当前默认执行真源是 `daily_research/output/active_execution_strategy.json`。
- 当前项目治理规则是：用户最新要求拥有最高优先级。

## 2. 接管前先问的三个问题
- 这次问题属于 `formal`、`recent` 还是 `live`。
- 这次工作属于研究环、执行环，还是 promotion 边界。
- 这次引用的结果是在比较 base model、execution mainline，还是 attack challenger。

## 2.1 当前接管快路
- 当前默认接管不再从 `episodic_memory.md` 开始。
- 当前接管快路是：
  - `identity_layer.md`
  - `handoff_packet.md`
  - `rule_memory.md`
  - `lesson_memory.md`
  - `temporal_state.md`
  - 然后再进入 `working / action / episodic`
- 目标是让新 agent 先接状态，再接历史，而不是先淹没在长日志里。

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
- 30% 强月 verdict 根：`daily_research/output/short_alpha_monthly_attack_signal_weight_verdict_20260409_r1`
- policy_v2 constrained formal review 根：`daily_research/output/short_alpha_policy_v2_constrained_execution_review_20260410_r1`
- policy_v2 formal loss breakdown 根：`daily_research/output/short_alpha_policy_v2_formal_loss_breakdown_20260410_r1`
- policy_v2 family formal review 根：`daily_research/output/short_alpha_policy_v2_family_formal_review_20260411_r1`
- policy_v2 family recent eval 根：`daily_research/output/short_alpha_policy_v2_family_recent_eval_20260411_r1`
- policy_v2 family constrained formal review 根：`daily_research/output/short_alpha_policy_v2_family_constrained_execution_review_20260411_r2`
- policy family formal loss breakdown 根：`daily_research/output/short_alpha_policy_family_formal_loss_breakdown_20260411_r1`
- policy_v2 family constrained pipeline 根：`daily_research/output/short_alpha_policy_v2_family_pipeline_20260411_r1_status.json`
- policy_v4 family formal review 根：`daily_research/output/short_alpha_policy_v4_family_formal_review_20260411_r1`
- policy_v4 family constrained formal review 根：`daily_research/output/short_alpha_policy_v4_family_constrained_execution_review_20260411_r1`
- policy_v4 family recent eval 根：`daily_research/output/short_alpha_policy_v4_family_recent_eval_20260411_r1`
- policy_v4 family pipeline 根：`daily_research/output/short_alpha_policy_v4_family_pipeline_20260411_r1_status.json`
- policy_v3 latest formal review 根：`daily_research/output/short_alpha_policy_v3_review_20260410_r1`
- policy_v3 latest recent eval 根：`daily_research/output/short_alpha_policy_v3_recent_eval_20260410_r1`
- 旧 replay-based / 历史机制参考索引：`daily_research/archive/output/replay_based_reference_index.md`

## 8. 当前主问题
- strongest-model research 层问题已经收口完成：formal winner、corrected recent evidence、latest-data production full-fit 与默认执行接管都已打通。
- live 层当前主问题：如何让 `short_expert_monthly_v1 + k2` 新默认链，在最近一年 `12` 个月 corrected recent 口径下解释并收敛与 `baseline_current` 的差异。
- current default 的 recent 一年主因已经有第一轮量化拆解：问题更像是“顺风状态兑现不足 + cash sizing 不够状态化 + score-to-weight 转换偏弱”，不是“模型完全不会看市场状态”。
- 股票池当前更像是天花板约束，而不是 recent 差距的第一主因；winner 与 companion 当前使用的是同一个固定 `liquid500` 池。
- execution optimization 层问题：如何在 `k2` 主线之上继续改善月度分布、弱月修复与集中度，同时维持当前 raw 统一语义。
- current-default 小修包已经有第一轮 same-protocol verdict：`winner_current_target_market_state_guard_v1` 是当前默认链自己内部的 repair winner，说明眼下最值得先补的是状态化总仓位，不是更激进的 `score_weight_k2`。
- current-default 第二轮 follow-up verdict 已把 winner-side 最强修补推进到 `winner_current_target_market_state_guard_v2_balance`，并且 gross-control 迁移实验出现了同向读数；这说明 current default 与 companion 的差距里，控制层已经不只是嫌疑，而是带量化证据的主因之一。
- research attack 层问题：如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 recent/live 上复现而不退化。
- 当前还没有任何 challenger 实现稳定 `30%+` 月收益门槛。
- 因此下一阶段最高优先级不再是重新决定谁上线，而是围绕当前已上线的 `short_expert + k2` 默认链，先做 `market_state_guard_v2_balance` 一类的 `cash sizing / gross-control` 修补，再做 `month-trigger` 与窄版 `signal-to-weight`，并用同协议 recent 一年窗口持续和 `state_liquidity` 对照。
- learned-control 层的新主矛盾已经再次更新为：`short_expert_policy_v2b` 已经给出 `0.1104` 的 constrained / deployable formal 证据，而 `short_expert_policy_v4b` 给出了 `0.1387` 的 corrected recent 前沿；当前最该补的不再是“有没有 constrained formal”，而是如何把 `v4b` 的 recent 强度转成不坍塌的 deployable formal。

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
  - `policy_v3` 的 latest-window formal / corrected recent 都已补齐；它 formal `monthly_robust_score = 0.0786`、corrected recent `monthly_robust_score = 0.0390`，都没有打赢 `policy_v2`，因此当前 learned-control 主研究分支仍是 `short_expert_policy_v2`，不是 `policy_v3`。
  - 本轮所有正式训练、formal / recent 回放与结论生成均已锁定 `yolos` 环境。
  - `policy_v2 family` constrained review 已补齐：`short_expert_policy_v2b__k1_20d = 0.1104` 已高于 current mainline `0.1012`，成为 learned-control 当前最强 constrained / deployable 候选；`policy_v2c` 的 constrained best 只有 `0.0953`，而 selected slow bridge 更低到 `0.0846`。
  - `policy_v4 family` 第一轮也已跑完：`policy_v4b` formal family best `0.1008`、recent `0.1387` 都很强，但 constrained best 只有 `0.0687`；`policy_v4a` formal / constrained 都没有解题。当前 learned-control 新主问题已经变成“如何保住 `policy_v4b` 的 recent 优势，同时不丢掉 `policy_v2b` 的 deployable constrained formal 能力”。

## 9. 当前地图修正
- strongest-model 主线现在要分三层看：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = baseline_current`
  - `promotable winner = short_expert_monthly_v1`
- 因此当前研究层主矛盾不再是“`short_expert` 如何打赢 `state_liquidity` 的 recent companion”，而是“`short_expert` 这条默认执行主线如何解释并收敛与 `baseline_current` 的 corrected recent 差异”。
- learned-control 层当前要分三层看：
  - `constrained formal front-runner = short_expert_policy_v2b__k1_20d`
  - `corrected recent winner = short_expert_policy_v4b`
  - `fresh formal family best = short_expert_policy_v4b`
- `short_expert_policy_v4b` 当前不是 promotion 答案，因为它的 constrained best 只有 `0.0687`；`short_expert_policy_v2b` 才是当前最强 deployable learned-control candidate。`policy_v3` 目前 formal 和 corrected recent 都没有打赢这些前沿，因此不进入默认执行晋升主线。
- 当前 learned-control 最重要的下一步不是广扫新模型，而是围绕“保住 `policy_v4b` 的 recent 强度并让 constrained formal 不坍塌”做窄迭代，同时保留 `policy_v2b` 作为 deployable 对照锚点。
- 旧 replay-recent 口径下的 recent root-cause、cash-sizing repair、follow-up repair、execution audit 与 targeted repair 原始根，现统一收口到 `daily_research/archive/output/replay_based_reference_index.md`，不再在主地图散落直引。
- 本机正式训练地图补充为：`yolos` + 前台 + `num_workers = 0` + `pin_memory = false`。
