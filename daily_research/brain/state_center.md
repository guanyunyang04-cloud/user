# Daily Research 状态程序
快照日期：`2026-07-07`

本文件是 `daily_research` 的当前程序实例，不是历史长卷；它只保存接管时需要激活的对象、函数和过程入口。

## Module Interface
`imports`: `qdp_v2_data_base` from `quant_data_platform`；`evidence_registry` from `daily_research/brain/references/evidence_registry.json`；`active_execution_artifact` from `daily_research/output/active_execution_strategy.json`，只在执行相关任务中激活。
`exports`: `current_research_pointer = seq100_todayclose_path_only`；`execution_state = frozen_skeleton_only`；`data_access_policy = qdp_only`。

## Object Instances
### object `daily_research_project`
`type`: project_brain
`state`: 正式生产研究与执行主线分脑；研究端已从旧 path20 / alpha_v2 复杂模型优先，切到 QDP 数据基底上的 seq100 today-close path-value 主线。
`owns`: 研究计划、模型、回测、执行候选、证据解释。
`consumes`: QDP v2 table/domain names, dataset manifests, and explicit downstream research packs when selected.
`methods`: `inspect_state()`；`write_research_evidence(reference)`；`request_qdp_data_change(requirement)`。

### object `execution_surface`
`type`: protected_execution_object
`state`: `frozen_skeleton_only / awaiting_research_rebuild`；历史 active 标签 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
`activation`: active/default、paper/live、broker、trade plan、execution restore 相关任务会激活；数据源评估、研究计划、scorer 设计、普通文档整理不激活。
`methods`: `inspect_active_artifact()`；`keep_frozen()`；`restore_or_activate(explicit_user_authorization, promotion_evidence)`。

### object `qdp_consumption`
`type`: data_dependency
`state`: QDP 当前 active 数据基底是 v2 manifest-first：`quant_data_platform/data/qdp_v2/active/active.json` + dataset manifests + parquet。
`active_data_scope`: `2011-11-22..2026-06-26`；沪深 A 股主板，剔除创业板/科创板/ST/退市；`600036.SH` from `2016-07-25`。
`downstream_pack_note`: memmap、sequence pack、normalization、sample_index、labels and model-ready panels are `daily_research` research artifacts under `daily_research/data/research_store/`; they are not the QDP active data base.
`methods`: `inspect_qdp_status()`；`consume_qdp_table(domain)`；`consume_training_pack(manifest)`；`request_qdp_update_or_table(requirement) -> QDP`。

### object `provider_boundary`
`type`: access_policy
`state`: `daily_research` 不直连在线 provider；`mootdx_online`、`BaoStock`、`CNInfo` 只作为 QDP 上游 ingest / raw archive / canonical 治理对象。
`function`: `resolve_data_access(task) -> qdp_object`

### object `shortline_after_close_research`
`type`: supporting_research_prior
`state`: 收盘后短线选股经验保留为执行层和条件入场先验；它不再是默认下一步。D 收盘后更新 QDP，D+1 只有入场条件满足时买入，T+1 约束下最早 D+2 卖出。
`mechanisms`: 强势启动/延续、强势回踩低吸、行业扩散补涨。
`facts`: Stage 0 固定 D+1 open 诊断未通过 validation-selected same-candidate test；307 特征画像和 raw 特征诊断支持 anti-overheat / anti-chase；`pullback_intraday_recovery` 是当前最强条件族。
`methods`: `build_upside_or_entry_scorer_baseline()`；`validate_scorer_by_decile_and_topk()`；`narrow_condition_matrix(priors)`。
`next_method`: `reuse_as_execution_layer_prior_when_needed()`

### object `seq100_path_value_research`
`type`: active_research_program
`state`: 当前唯一默认研究主线；用 QDP v2 active 数据构造 model-ready research artifacts，由 `daily_research` 拥有和解释。
`default_mainline`: `seq100_todayclose_path_only`。
`active_concept_surface`: `research_store_view`、`seq100_x84_input`、`today_close_anchor`、`future60_ohlc_path`、`path_trade_value_v2`、`path_value_spread`。
`input_principle`: 过去 100 日 daily raw、daily state、intraday summary、limit structure 序列；不使用 symbol embedding 作为默认主线。
`output_principle`: 预测未来路径；路径摘要和 path trade value 从预测路径派生，排序使用预测路径价值而不是独立固定标签。
`default_entry`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --json`。
`comparison_surface`: `table_path60_baseline`、`path_only_next_open`、`rank_heavy_top1`。
`archived_or_paused_surface`: `alpha_v2`、`path20`、`symbol_embedding`、`residual_score`、`richer_target`、`ohlcva_unified`。
`current_evidence`: today-close path-only is the default research mainline; validation/test rank IC and TopK path-value spread are promising research evidence, not promotion evidence.
`artifact_owner`: `daily_research`; preferred new root is `daily_research/data/research_store/<artifact_id>/`.
`historical_artifacts`: old QDP research artifacts were physically migrated or deleted on 2026-07-07; `quant_data_platform/data/qdp_v2/research/` no longer exists as a training-pack location.
`resource_state`: H: pressure is driven mainly by repeated research packs and `daily_research/output/path_policy/studies`, not by the QDP active data base itself. On 2026-07-07, guarded research GC deleted 74 unreferenced smoke/partial/interrupted artifacts and reclaimed 14.41GB; prediction output trim deleted 92 large forecast CSV files across 46 studies and reclaimed 63.55GB. The old self-contained full sequence packs were replaced by a unified research store: shared panel_store + label_store + sample_index + lightweight views. Post-unification GC scan sees 162 artifacts / 94.29GB, including 75.11GB shared research_store components, with zero safe directory candidates and zero prediction trim candidates.
`gc_report`: `daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_091946.md`
`research_store_physical`: active model-ready entrypoints are lightweight view manifests under `daily_research/data/research_store/views/`; shared inputs live under `panel_store/`; labels live under `label_store/`; sample indexes live under `sample_index/`; old alpha_v2 sharded memmap and training pack live under `sharded_memmap/` and `training_pack/`.
`research_store_views`: `seq100_path60_nextopen_ohlc`；`seq100_path60_nextopen_ohlcva`；`seq100_path60_todayclose_ohlcva`；`seq100_path20_nextopen_ohlc_from_path60`.
`store_view_note`: path20 cannot be fully replaced by slicing path60 labels because early valid path20 samples have no path60 label; the view shares input panels but keeps an independent path20 label store with `label_symbol_idx`.
`next_method`: `run_execution_layer_backtest()`

### object `alpha_v2_history`
`type`: archived_research_line
`state`: 旧复杂模型线保留为历史研究和可复用基础设施；anchor `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` 仍是历史比较点，但不是 multi-seed、candidate matrix、execution review 或 active/default 依据。

## Pure Functions
- `select_relevant_objects(task)`: 从任务文本和路径选择对象；无关对象不激活。
- `classify_evidence(run)`: 把 smoke、dry-run、short-window、interrupted、insufficient、failed、completed run 分到对应证据等级。
- `activate_execution_boundary(objects, method)`: 只有 `execution_surface` 的 restore/activate/trade-plan 方法被调用时返回 true。
- `derive_next_action(seq100_state, evidence)`: 当前返回 `run_execution_layer_backtest()`；旧 shortline 只在需要执行层先验时显式调用。
- `resolve_data_access(task)`: 任何新增数据需求都返回 QDP v2 provider/update/table request，不返回 direct online provider。

## Procedures
### procedure `shortline_scorer_baseline`
`input`: QDP v2 tables or explicit downstream training pack、raw/engineering feature diagnostics、shortline condition priors
`steps`: 读取 explicit QDP table/pack；构建 upside 或 entry scorer baseline；用 score decile、top-decile 期望、validation-selected same-candidate test 和成本后表现验证；结果写入 shortline reference，本文件只回写当前结论摘要。
`side_effects`: research artifacts only；不触碰 active/default/live。

### procedure `execution_change`
`input`: explicit user authorization、promotion-grade evidence、active artifact diff
`steps`: 激活 `execution_surface`；inspect active artifact；对齐 promotion 边界；再讨论 restore / activate / trade plan。
`side_effects`: protected；非执行任务不调用。

## Evidence Entrypoints
- QDP replacement activation：`daily_research/brain/references/qdp_alpha_v2_replacement_data_base_activation_20260626.md`
- Seq100 path-value orchestration：`daily_research/brain/references/seq100_path_value_research_orchestration_20260706.md`
- Seq100 mainline slimming contract：`daily_research/brain/references/path_policy_seq100_mainline_slimming_contract_20260707.md`
- Shortline plan：`daily_research/brain/references/shortline_after_close_research_plan_20260624.md`
- Stage 0 diagnostic：`daily_research/brain/references/shortline_stage0_fixed_next_open_diagnostic_20260624.md`
- Feature profile / semantics：`daily_research/brain/references/shortline_upside_feature_profile_20260624.md`、`daily_research/brain/references/shortline_upside_feature_semantics_20260624.md`
- Raw diagnostic / condition matrix：`daily_research/brain/references/shortline_raw_upside_diagnostic_20260624.md`、`daily_research/brain/references/shortline_raw_condition_matrix_20260624.md`
- Machine index：`daily_research/brain/references/evidence_registry.json`
