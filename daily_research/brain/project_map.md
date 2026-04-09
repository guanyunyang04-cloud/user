# Daily Research 项目地图

快照日期：`2026-04-09`

## 1. 项目定义
- `daily_research` 是一条以执行后净收益最大化为目标的研究-执行统一链路。
- 当前默认研究目标是 `execution_first`，当前默认判决顺序是“月度收益优先、模型偏短线”。
- 当前默认执行真源是 `daily_research/output/active_execution_strategy.json`。
- 当前项目治理规则是：用户最新要求拥有最高优先级。

## 2. 当前统一语义
- `research_raw_target_weight`
- `follow_research_raw_no_global_cap`
- `monthly-first`
- `short-term bias`
- `strict resume`
- `GPU only`

## 3. 当前主线
- strongest base model：`state_liquidity_listwise_v1`
- active execution strategy：`state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
- active manifest：`daily_research/output/active_execution_strategy.json`
- active pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260409_r2`
- production root：`daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- current stable production run：`daily_research/output/short_alpha_production_epoch_extension_20260409_r1/runs/short_alpha_production_e64`

## 4. 执行侧现状
- 当前 live effective mode 是 `static_fallback`。
- 当前 live effective profile 是 `regoff_k2_5d_ensemble_native_anchor`。
- 当前实际执行来自 `static_fallback_daily_live_target_weight_panel.csv`，并且它是从 raw production panel 经 `k2 / 5d / all-offset` bridge 生成，不再吃旧 capped fallback。
- `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 现在只保留为 observation-only 的历史 targeted repair 分支，不再作为当前默认 repair 叙事。
- `active_execution_strategy.json`、pipeline sidecar 与 `latest_trade_plan.txt` 现在都会同步写出 effective profile、bridge meta 和 weight-generation explanation。

## 5. 模型侧现状
- 模型侧当前没有新的 clean promotion answer。
- 如需重开，只从 `short_expert_penalty_only_monthly_v1` 一类窄修复继续。
- `v2`、`heavy penalty` 等大包路线继续保持停止。

## 6. 最新关键结果
- production full-fit fresh `32` 起训后被判 undertrained，同模型 strict resume 到 `64` 后预算压力解除，当前 production 已以 `e64` 为准。
- same-protocol 语义裁决根为 `daily_research/output/short_alpha_execution_semantic_concentration_verdict_20260409_r1`，结论是 `raw + k2` 明确胜过当前 `k1` 与 capped direct。
- 月度优先 scoreboard 根为 `daily_research/output/short_alpha_monthly_first_execution_scoreboard_20260409_r1`，结论是 `k2` 仍是 execution-side mainline。
- recent execution-policy audit 根为 `daily_research/output/short_alpha_production_execution_policy_audit_20260409_r2`，近期排序明确为 `k2 > k1 > k3 > topk3`。
- same-window targeted review 根为 `daily_research/output/short_alpha_targeted_weak_month_repair_regime_firstweek_combo_expand_stable_topk3_monthly_k2_review_20260409_r3`，`topk3` 对 `k2` formal `0/3` 胜，recent gate 也无增益。

## 7. 当前真正问题
- 当前问题已经不再是“`raw no-cap` 还是旧 global cap”或“要不要继续把 `topk3` 当默认 repair 故事”。
- 当前未解主问题变成：如何在 `k2` 主线之上，进一步改善月度分布、弱月修复与集中度，同时维持当前 raw 统一语义。
- 因此下一阶段最高优先级是 `signal-to-weight / month-trigger design`，而不是继续 broad repair sweep 或大范围重训。

## 8. 决策闭环
- formal head-to-head 负责确认 base model 是否仍成立。
- production full-fit 与 fallback refresh 负责物化真实 live panel。
- single-mapping pipeline 与 activation 负责把 current live mode / effective profile 写回 active manifest。
- `run_trade_plan.py` 负责把当前可执行计划和 bridge 解释展示出来。
- monthly-first scoreboard 与 same-window targeted review 负责决定 repair 路径是 promotion、observation 还是 retire。
- 稳定结论再回写到 `semantic / project_map / working / procedural / action / episodic` 六层分脑。
