# Daily Research 项目地图

快照日期：`2026-04-12`

## 1. 项目定义
- `daily_research` 是一条以执行后净收益最大化为目标的研究-执行统一链路。
- 当前默认研究目标是 `execution_first`，默认判决顺序是“月度收益优先、模型偏短线”。
- 当前默认执行真源是 `daily_research/output/active_execution_strategy.json`。
- 当前项目治理规则是：用户最新要求拥有最高优先级。

## 2. 接管前先问的三个问题
- 这次问题属于 `formal`、`recent`、`learned-control` 还是 `live`。
- 这次工作属于研究环、执行环，还是 promotion 边界。
- 这次引用的结果是在比较 strongest-model 主线，还是 learned-control 分支。

## 3. 当前接管快路
- 当前默认接管不再从 `episodic_memory.md` 开始。
- 当前接管快路是：
  - `identity_layer.md`
  - `handoff_packet.md`
  - `rule_memory.md`
  - `lesson_memory.md`
  - `temporal_state.md`
  - 然后再进入 `working / action / episodic`

## 4. 双环闭环
- 研究环：
  - 用滚动 formal 协议训练研究模型。
  - 用独立 recent 协议验证最近一年兑现。
  - 用 strongest-model gate 判定 short-alpha 主线 formal / recent / promotable。
- 执行环：
  - strongest research winner 可直接进入默认执行物化。
  - active manifest、production root、trade plan 和 brain 文档保持同源。
  - live 只回答“当前生产面板在跑什么”，不回填 formal 证据。

## 5. 当前统一语义
- `research_raw_target_weight`
- `follow_research_raw_no_global_cap`
- `monthly-first`
- `short-term bias`
- `strict resume`
- `foreground_only`
- `project-python`

## 6. 当前 strongest-model 主线
- strongest-model verdict root：
  - `daily_research/output/short_alpha_strongest_model_verdict_20260412_r1`
- strongest-model recent root：
  - `daily_research/output/short_alpha_recent_model_protocol_20260412_r1`
- 当前三层答案：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- 当前 strongest-model / production 物化锚点仍然落在 `2026-04-09` 这轮默认执行接管。
- strongest stable base model：
  - `state_liquidity_listwise_v1`

## 7. 当前 live 主线
- active execution strategy：
  - `deep_alpha_short_alpha_execalign_production_default`
- active candidate label：
  - `short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`
- active manifest：
  - `daily_research/output/active_execution_strategy.json`
- production root：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- current active production run：
  - `daily_research/output/deep_alpha_short_alpha_execfirst_production_fullfit_20260409_r1`
- single-mapping pipeline：
  - `daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`
  - 仅历史观察分支，不是当前默认

## 8. 当前 learned-control 主线
- current active v5 family formal review root：
  - `daily_research/output/short_alpha_policy_v5_family_formal_review_20260412_r1`
- current active v5 family constrained review root：
  - `daily_research/output/short_alpha_policy_v5_family_constrained_execution_review_20260412_r1`
- current active v5 family recent eval root：
  - `daily_research/output/short_alpha_policy_v5_family_recent_eval_20260412_r1`
- current active v5 family breakdown root：
  - `daily_research/output/short_alpha_policy_v5_family_formal_loss_breakdown_20260412_r1`
- current active v5 family pipeline status：
  - `daily_research/output/short_alpha_policy_v5_family_pipeline_20260412_r1_status.json`

## 9. 当前关键结果
- strongest-model 当前已重新对齐，主矛盾不再是“`baseline_current` recent 为什么更强”。
- learned-control 当前四层要这样讲：
  - deployable constrained front-runner = `short_expert_policy_v5b__k1_20d = 0.1177`
  - current recent winner = `short_expert_policy_v5b = 0.1200`
  - active v5 family fresh formal best = `short_expert_policy_v5b = 0.0839`
  - historical cross-family fresh formal best = `short_expert_policy_v4b = 0.1008`
- `policy_v5a__k1_20d = 0.1018` 与 `policy_v5c__k1_20d = 0.1010` 提供了 near-mainline constrained 支撑。

## 10. 当前主问题
- strongest-model 层：
  - 当前已经重新对齐，live 默认也未改变。
- learned-control 层：
  - `policy_v5b` 已经拿到 recent + constrained 的当前主导权。
  - 当前最该补的不是“有没有 deployable evidence”，而是“如何缩小 `policy_v5b` 的 fresh-formal gap”。
- live 层：
  - 继续保持 `short_expert_monthly_v1 + k2_5d` 稳定，不做静默切换。

## 11. 决策闭环
- strongest-model verdict 负责 short-alpha 主线 formal / recent / promotable。
- `policy_v5 family` 负责 learned-control 当前主研究分支。
- production full-fit 负责把 default winner 物化成真实 live panel 和默认执行。
- 真正写入默认执行前，仍必须使用当前最高 family budget。
- `run_trade_plan.py` 负责把当前可执行计划和 bridge 解释展示出来。
- 稳定结论回写到 `semantic / project_map / working / procedural / action / episodic`。

## 12. 当前地图修正
- 不再把 `baseline_current` 写成 strongest-model current recent winner。
- 不再把 `policy_v2b` 写成 current learned-control 默认答案。
- 不再把 `policy_v4b` 写成 current learned-control recent frontier。
- 当前 learned-control 最重要的下一步不是广扫新模型，而是围绕 `policy_v5b` 做窄迭代，同时保留 `policy_v5a / policy_v5c` 作为对照。
- 旧 replay-based 机制根统一经 `daily_research/archive/output/replay_based_reference_index.md` 引用，不再在主地图散落直引。
