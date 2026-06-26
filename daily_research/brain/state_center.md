# Daily Research 状态程序
快照日期：`2026-06-27`

本文件是 `daily_research` 的当前程序实例，不是历史长卷；它只保存接管时需要激活的对象、函数和过程入口。

## Module Interface
`imports`: `qdp_data_substrate` from `quant_data_platform`；`evidence_registry` from `daily_research/brain/references/evidence_registry.json`；`active_execution_artifact` from `daily_research/output/active_execution_strategy.json`，只在执行相关任务中激活。
`exports`: `current_research_pointer = shortline_after_close_research`；`execution_state = frozen_skeleton_only`；`data_access_policy = qdp_only`。

## Object Instances
### object `daily_research_project`
`type`: project_brain
`state`: 正式生产研究与执行主线分脑；研究端已从旧 path20 / alpha_v2 复杂模型优先，切到 QDP 数据基底上的收盘后短线选股。
`owns`: 研究计划、模型、回测、执行候选、证据解释。
`consumes`: QDP explicit lake dataset id、manifest、memmap、training pack。
`methods`: `inspect_state()`；`write_research_evidence(reference)`；`request_qdp_data_change(requirement)`。

### object `execution_surface`
`type`: protected_execution_object
`state`: `frozen_skeleton_only / awaiting_research_rebuild`；历史 active 标签 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
`activation`: active/default、paper/live、broker、trade plan、execution restore 相关任务会激活；数据源评估、研究计划、scorer 设计、普通文档整理不激活。
`methods`: `inspect_active_artifact()`；`keep_frozen()`；`restore_or_activate(explicit_user_authorization, promotion_evidence)`。

### object `qdp_consumption`
`type`: data_dependency
`state`: QDP lake 已迁到 `quant_data_platform/data/lake`；`daily_research/output/research_data_lake` 已删除。
`active_pack`: `quant_data_platform/data/memmap/training_pack/tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_training_pack_20260626_01/qdp_training_pack_manifest.json`
`active_memmap`: `tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_2010_2026_20260626_01`；`qdp validate-memmap` status `ok`；feature_count `307`；symbol_count `3025`。
`methods`: `inspect_qdp_manifest()`；`consume_training_pack(manifest)`；`request_provider_or_canonical_update(requirement) -> QDP`。

### object `provider_boundary`
`type`: access_policy
`state`: `daily_research` 不直连在线 provider；`mootdx_online`、`BaoStock`、`CNInfo` 只作为 QDP 上游 ingest / raw archive / canonical 治理对象。
`function`: `resolve_data_access(task) -> qdp_object`

### object `shortline_after_close_research`
`type`: active_research_program
`state`: 当前目标是收盘后短线选股；D 收盘后更新 QDP，D+1 只有入场条件满足时买入，T+1 约束下最早 D+2 卖出。
`mechanisms`: 强势启动/延续、强势回踩低吸、行业扩散补涨。
`facts`: Stage 0 固定 D+1 open 诊断未通过 validation-selected same-candidate test；307 特征画像和 raw 特征诊断支持 anti-overheat / anti-chase；`pullback_intraday_recovery` 是当前最强条件族。
`methods`: `build_upside_or_entry_scorer_baseline()`；`validate_scorer_by_decile_and_topk()`；`narrow_condition_matrix(priors)`。
`next_method`: `build_upside_or_entry_scorer_baseline()`

### object `alpha_v2_history`
`type`: archived_research_line
`state`: 旧复杂模型线保留为历史研究和可复用基础设施；anchor `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` 仍是历史比较点，但不是 multi-seed、candidate matrix、execution review 或 active/default 依据。

## Pure Functions
- `select_relevant_objects(task)`: 从任务文本和路径选择对象；无关对象不激活。
- `classify_evidence(run)`: 把 smoke、dry-run、short-window、interrupted、insufficient、failed、completed run 分到对应证据等级。
- `activate_execution_boundary(objects, method)`: 只有 `execution_surface` 的 restore/activate/trade-plan 方法被调用时返回 true。
- `derive_next_action(shortline_state, evidence)`: 当前返回 `build_upside_or_entry_scorer_baseline()`。
- `resolve_data_access(task)`: 任何新增数据需求都返回 QDP provider/canonical 对象，不返回 direct online provider。

## Procedures
### procedure `shortline_scorer_baseline`
`input`: QDP replacement training pack、raw/engineering feature diagnostics、shortline condition priors
`steps`: 读取 explicit QDP pack / raw feature sidecar；构建 upside 或 entry scorer baseline；用 score decile、top-decile 期望、validation-selected same-candidate test 和成本后表现验证；结果写入 shortline reference，本文件只回写当前结论摘要。
`side_effects`: research artifacts only；不触碰 active/default/live。

### procedure `execution_change`
`input`: explicit user authorization、promotion-grade evidence、active artifact diff
`steps`: 激活 `execution_surface`；inspect active artifact；对齐 promotion 边界；再讨论 restore / activate / trade plan。
`side_effects`: protected；非执行任务不调用。

## Evidence Entrypoints
- QDP replacement activation：`daily_research/brain/references/qdp_alpha_v2_replacement_data_base_activation_20260626.md`
- Shortline plan：`daily_research/brain/references/shortline_after_close_research_plan_20260624.md`
- Stage 0 diagnostic：`daily_research/brain/references/shortline_stage0_fixed_next_open_diagnostic_20260624.md`
- Feature profile / semantics：`daily_research/brain/references/shortline_upside_feature_profile_20260624.md`、`daily_research/brain/references/shortline_upside_feature_semantics_20260624.md`
- Raw diagnostic / condition matrix：`daily_research/brain/references/shortline_raw_upside_diagnostic_20260624.md`、`daily_research/brain/references/shortline_raw_condition_matrix_20260624.md`
- Machine index：`daily_research/brain/references/evidence_registry.json`
