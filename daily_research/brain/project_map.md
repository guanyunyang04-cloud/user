# Daily Research 项目地图

快照日期：`2026-04-09`

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
  - 用 formal 协议训练研究模型。
  - 训练数据只能到 formal 评估开始前一天。
  - 最近 `12` 个自然月完整预留做评估。
  - 研究 winner 先在这条环里成立，才能进入 promotion 讨论。
- 执行环：
  - winner 冻结后，才允许用最新可标注数据做 `production full-fit`。
  - `production full-fit` 负责物化 live panel、fallback 和真实执行默认值。
  - recent/live 监控负责回答“最近有没有兑现”，但不回填成 formal 证据。

## 4. 当前统一语义
- `research_raw_target_weight`
- `follow_research_raw_no_global_cap`
- `monthly-first`
- `short-term bias`
- `strict resume`
- `GPU only`

## 5. 当前主线
- strongest base model：`state_liquidity_listwise_v1`
- active execution strategy：`state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
- active manifest：`daily_research/output/active_execution_strategy.json`
- active pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`
- production root：`daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- current stable production run：`daily_research/output/short_alpha_production_epoch_extension_20260409_r1/runs/short_alpha_production_e64`

## 6. 当前执行侧现状
- 当前 live effective mode 是 `static_fallback`。
- 当前 live effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 当前实际执行来自 `static_fallback_daily_live_target_weight_panel.csv`，并且它是从 raw production panel 经 `k2 / 5d / all-offset` bridge 生成，不再吃旧 capped fallback。
- `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 现在只保留为 observation-only 的历史 targeted repair 分支，不再作为当前默认 repair 叙事。
- `active_execution_strategy.json`、pipeline sidecar 与 `latest_trade_plan.txt` 现在都会同步写出 effective profile、bridge meta 和 weight-generation explanation。

## 7. 当前关键结果
- formal base-model verdict 根：`daily_research/output/short_alpha_formal_head2head_20260409_recheck_r1`
- execution semantic verdict 根：`daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`
- monthly-first scoreboard 根：`daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`
- recent audit 根：`daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`
- recent attack recheck 根：`daily_research/output/short_alpha_production_execution_policy_audit_20260409_r3_attack_recheck`
- same-window targeted review 根：`daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`
- 30% 强月 verdict 根：`daily_research/output/short_alpha_monthly_attack_signal_weight_verdict_20260409_r1`

## 8. 当前主问题
- live 层问题：如何在 `k2` 主线之上继续改善月度分布、弱月修复与集中度，同时维持当前 raw 统一语义。
- research attack 层问题：如何把 `formal_current_equal_top5_k1_bridge` 这类更强攻击桥，在 recent/live 上复现而不退化。
- 当前还没有任何 challenger 实现稳定 `30%+` 月收益门槛。
- 因此下一阶段最高优先级是围绕 `k1 attack branch vs k2 live branch` 做阈值微调、状态收敛和 recent/live 扩样，而不是继续 broad repair sweep 或大范围重训。

## 9. 决策闭环
- formal head-to-head 负责确认 base model 是否仍成立。
- production full-fit 与 fallback refresh 负责物化真实 live panel。
- single-mapping pipeline 与 activation 负责把 current live mode / effective profile 写回 active manifest。
- `run_trade_plan.py` 负责把当前可执行计划和 bridge 解释展示出来。
- monthly-first scoreboard 与 same-window targeted review 负责决定 repair 路径是 promotion、observation 还是 retire。
- 强月 verdict 负责回答“是否更接近 30% 强月目标”，不能替代普通 monthly-first verdict。
- 稳定结论再回写到 `semantic / project_map / working / procedural / action / episodic` 六层分脑。
