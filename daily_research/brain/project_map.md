# Daily Research 项目地图

快照日期：`2026-04-09`

## 1. 项目定义
- `daily_research` 是一条以执行后净收益最大化为目标的研究-执行统一链路。
- 当前默认研究目标仍是 `execution_first`。
- 当前默认执行真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前项目治理规则是：用户最新要求拥有最高优先级。

## 2. 当前统一语义
- `research_raw_target_weight`
- `follow_research_raw_no_global_cap`
- `strict resume`
- `GPU only`

## 3. 当前主线
- strongest base model：`state_liquidity_listwise_v1`
- active execution strategy：`state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
- active pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r1`
- production root：`daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- current stable production run：`daily_research/output/short_alpha_production_epoch_extension_20260409_r1/runs/short_alpha_production_e64`

## 4. 执行侧现状
- 当前唯一正式保留的 execution repair 是 `trend_up_low_vol|expand|stable -> topk3_1d_regoff`。
- 这条映射 historical formal 证据仍成立，但当前 live 月没有触发。
- 因此当前实际执行来自 `static_fallback_daily_live_target_weight_panel.csv`。
- 这个 static fallback 已经改成从 raw production panel 统一桥接，不再吃旧 capped fallback。

## 5. 模型侧现状
- 模型侧当前没有新的 clean promotion answer。
- 如需重开，只从 `short_expert_penalty_only_monthly_v1` 一类窄修复继续。
- `v2`、`heavy penalty` 等大包路线继续保持停止。

## 6. 最新关键结果
- production full-fit fresh `32` 起训后被判 undertrained。
- 同模型 strict resume 到 `64` 后预算压力解除，当前 production 已以 `e64` 为准。
- active execution recent costed recheck 根为 `daily_research/output/recheck_active_execution_candidate_20260409_r3_unified_costed`。
- 这轮结果说明：统一后的 raw no-cap 语义更一致，但最近区间收益不如旧 capped fallback。

## 7. 当前真正问题
- 当前问题不是“模型坏了”，而是“研究 raw no-cap 语义”和“最近真实兑现能力”之间出现了冲突。
- 项目下一阶段最重要的问题，已经变成：哪条全链语义真的带来更高的真实收益。
- `2026-04-09` 项目统一补充：
- 现在项目里的“当前执行态”不再只靠 pipeline sidecar 文件解释；`active_execution_strategy.json` 也会同步记录 current live mode、effective execution profile、bridge meta 和 weight-generation note。
- 现在项目里的“分数展示”也不再假装是权重直接父因子；若当前 live 权重来自 bridge，trade plan 会明确把 `转权重前分数` 标成 raw pre-weight score，并解释 bridge 如何产生最终权重。
