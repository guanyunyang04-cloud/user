# Daily Research 当前判断

快照日期：`2026-04-08`

## 1. 当前总判断
- 用户的北极星目标仍是“月度正收益 `> 30%`”，但它当前只作为长期方向，不作为正式晋级门槛。
- 当前全局 deployable winner 仍是：
  - `state_liquidity_listwise_v1`
  - `liquid500`
  - 基础执行壳仍来自 `regoff_k1_5d_ensemble_native_anchor`
- 当前日常执行真源仍是：
  - `daily_research/output/active_execution_strategy.json`

## 2. 执行侧正式结论
- broad execution-policy sweep 已结束。
- 当前唯一高证据 repair candidate 是：
  - `trend_up_low_vol|expand|stable -> topk3_1d_regoff`
- 它已经物化成 active default：
  - strategy：`state_liquidity_listwise_v1_execfirst_single_mapping_candidate_active`
  - pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260408_r2`
- 当前 active execution 的语义也已经收口：
  - `research_raw_target_weight`
  - `follow_research_raw_no_global_cap`
  - 研究面板给多少 raw target weight，执行就按多少走；external target-weight 链不再隐含通用 `max_weight=0.25`
- 当前 formal full-period H2H 结果是：
  - candidate `51.42% / 2.253`
  - static `39.85% / 1.738`
  - full-period delta `+11.54% / +0.514`
- recent realistic gate 也保持相对正增量：
  - targeted `-13.60% / -0.788`
  - static `-34.61% / -1.850`
  - delta 仍为正
- 这条映射的触发覆盖率是：
  - `3/36 = 8.33%`
  - triggered mean monthly delta `+8.50%`
- 当前 live 月份没有触发这条映射：
  - `2026-04` 仍回到 `regoff_k1_5d_ensemble_native_anchor`
  - 因此当前 active live panel 实际等价于旧静态默认，但 manifest 已切成 candidate pipeline
- 当前 candidate pipeline 已完成执行层轻量化：
  - trade plan refresh 改走 `live-only`，不再每日重跑 formal replay / H2H
  - pipeline root 内部现已同时物化 `daily_live_target_weight_panel.csv` 与 companion `daily_live_score_panel.csv`
  - score 明确只作为 static reference view，真实执行仍由 target weight panel 决定

## 3. 模型侧正式结论
- `penalty-only narrow ablation` 已完整收口。
- 统一结论是：
  - no narrow split produced a clean promotion answer over `state_liquidity_listwise_v1`
- 当前 best reusable restart point：
  - `short_expert_penalty_only_monthly_v1`
  - `63.93% / 4.696`
  - monthly robust `0.0835`
- strongest split challenger：
  - `short_expert_penalty_only_light_monthly_v1`
  - positive `83.33%`
  - median `2.92%`
  - monthly robust `0.0822`
  - 仍未超过当前线 `median 5.26% / robust 0.0897`
- `short_expert_penalty_only_heavy_monthly_v1`
  - 已 strict resume `48 -> 64`
  - 预算压力清除后稳定到 `39.46% / 3.054`
  - monthly robust `0.0458`
  - 已确认为负证据

## 4. 当前一致性修正
- 训练/预训练默认起训已经统一到 `32`，同模型扩预算默认只走 strict resume。
- GPU only 已下沉到训练与预训练入口，不能再静默回落到 CPU。
- single-mapping candidate pipeline 与 activation 脚本已移除 dated operational root 默认值，改为按最新产物动态解析。
- 项目已新增 `daily_research/tools/project_consistency_check.py`，以后用它兜住这类前后口径分叉。
- 当前新增的项目级规则是：
  - 用户最新要求拥有最高优先级
  - 不允许把旧口径、旧默认值、旧包装脚本继续留在默认主链里生效

## 4. 当前核心问题
- 当前问题不是“默认主线错了”。
- 当前问题是：
  - 执行侧 candidate 已升级为 active default，但触发稀疏，真正的收益兑现仍取决于未来 live 月份是否出现命中
  - 模型侧已有有效增量，但还没有干净跨过当前线的月度中位数与整体 monthly robust 边界

## 5. 当前明确停止项
- 不再继续 broad execution-policy sweep。
- 不再继续 `short_expert_monthly_v2`、`heavy penalty` 这类已证伪分支。
- 不再把“可能没训够”当作默认解释。
- 不再允许 execution-bound / monthly-refresh 结果来自：
  - 旧模型
  - 低预算模型
  - CPU 训练
  - fresh rerun 取代 strict resume 的同模型扩预算

## 6. 当前下一步
1. 执行侧
- 继续围绕 `expand|stable -> topk3` 收集真实触发与最近月样本。
- 月更或 production refresh 后，优先用最新模型重刷 single-mapping candidate pipeline。

2. 模型侧
- 如重新开线，只从 `short_expert_penalty_only_monthly_v1` 或 `light` 这种窄修复继续。
- 目标只盯月度中位数和 monthly robust 修复。

3. 生产纪律
- 所有 execution / monthly refresh 候选都必须先补成：
  - latest model
  - highest family budget
  - GPU training
  - strict resume if same-model budget extension
- `2026-04-08` 已完成一次不覆盖 active candidate 的 production full-fit refresh：
  - refreshed root = `daily_research/output/deep_alpha_short_alpha_execfirst_production_fullfit_20260408_r1`
  - production root launch cutoff 已更新到 `20260408`
  - 本轮按新纪律从 `32` epoch 起训
  - train history best epoch 落在 `15/32`，不是右边界，因此当前没有继续续训的证据
