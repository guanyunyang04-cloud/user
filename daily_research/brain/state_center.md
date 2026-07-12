# Daily Research 状态程序
快照日期：`2026-07-12`

本文件是 `daily_research` 的当前程序实例，不是历史长卷；它只保存接管时需要激活的对象、函数和过程入口。

## Module Interface
`imports`: `qdp_v2_data_base` from `quant_data_platform`；`evidence_registry` from `daily_research/brain/references/evidence_registry.json`；`active_execution_artifact` from `daily_research/output/active_execution_strategy.json`，只在执行相关任务中激活。
`exports`: `current_research_pointer = seq100_todayclose_path_only/daily_only_summary_v2_ohlcva_aux_low`；`execution_state = frozen_skeleton_only`；`data_access_policy = qdp_only`。

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
`default_mainline`: `seq100_todayclose_path_only/daily_only_summary_v2_ohlcva_aux_low`。
`active_concept_surface`: `research_store_view`、`seq100_x32_daily_input`、`today_close_anchor`、`future60_ohlc_path`、`summary_v2_multi_horizon_ohlc`、`low_weight_va_auxiliary`、`path_trade_value_v2`、`path_value_spread`。
`input_principle`: 过去 100 日 `daily_raw + daily_state` 序列；`intraday_summary` 和 `limit_structure` 不再作为默认输入，但旧 all-channel base 保留为 broad-TopK 对照基线；不使用 symbol embedding 作为默认主线。
`output_principle`: 预测未来 OHLC 路径，并用低权重 VA auxiliary path 约束量价表征；路径摘要和 path trade value 从预测 OHLC 路径派生，排序使用预测路径价值而不是独立固定标签。
`model_training_entry`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --json`；兼容的单模型训练入口，不能单独形成正式 verdict。
`formal_evaluation_entry`: 先用 `seq100_research_generation init-development` 冻结候选与四折合同，再以 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe tools/memory_guard.py --min-available-gb 1.0 -- <yolos-python> -m daily_research.path_policy.seq100_research_generation run-development --root <study-root> --registry <study-root>/development_registry.json` 运行完整矩阵，最后调用 `select-development --registry ... --ledger ...`；不得绕过 registry 直接形成正式 verdict。
`comparison_surface`: `table_path60_baseline`、`path_only_next_open`、`rank_heavy_top1`、`all_channels_base_summary`、`summary_v2_multi_horizon_ohlc`、`summary_v2_no60`、`summary_v2_price_delta`、`summary_v2_ohlcva_aux`、`summary_v2_ohlcva_aux_low`、`summary_v2_ohlcva_aux_low_price_delta`、`summary_v2_ohlcva_path_equal`、`daily_only_no_minute`、`daily_only_summary_v2`、`daily_only_summary_v2_price_delta`、`daily_only_summary_v2_ohlcva_aux`、`daily_only_summary_v2_ohlcva_aux_low_price_delta`、`daily_only_summary_v2_ohlcva_path_equal`、`path_value_v2_hard_st`、`global_tail_512`、`no_intraday_summary`、`no_limit_structure`、`direct_value_rank_5d`、`direct_value_rank_10d`、`direct_value_rank_60d`。
`archived_or_paused_surface`: `alpha_v2`、`path20`、`symbol_embedding`、`residual_score`、`richer_target`、`ohlcva_unified`。
`primary_evaluation_policy`: 正式模型/profile 判断使用 2022-2025 purged expanding `train/development`，四年直接参与 checkpoint、loss 设计和冠军选择；无历史 test/outer audit。每条训练标签必须满足 `max_label_dependency_date_idx < development_start_date_idx`，归一化只用 development 首日前已公开 feature dates；每折使用全部合格训练行，至少完成一个完整 epoch，最多 10 epoch，仅按 `development_total_loss`、patience 2 早停并恢复 best checkpoint；Top1/3/5/10 不选择折内 checkpoint，但用于候选资格和设计决策。真正 lockbox 从未来冠军冻结后的 2026 新预测/订单开始。
`current_evidence`: candidate-complete v3 正式矩阵已完成 `8/8` jobs：seed 7、baseline/hard-ST、2022-2025 四折均 `best_epoch=1`、`completed_epochs=3`、early-stop=true。四年等权成本后 realized-plan alpha：baseline Top1/3/5/10 为 `-2.89%/-1.83%/-1.24%/-0.06%`，hard-ST 为 `-10.36%/-7.99%/-5.26%/-2.28%`；Top3 正收益年份分别 `0/4` 与 `1/4`，两者均未通过资格门，`winner=null`，不得 freeze。baseline Top3 opportunity alpha 仍为 `+39.87%`，但可执行 alpha 为 `-1.83%`、oracle regret `0.5651`，说明主要问题是预测退出/执行目标错配；hard-ST 放大错配，不直接继续 global-tail。候选与执行收益覆盖均为 100%，不是数据缺口；QDP active 与 execution 均未改变。
`next_generation_implementation`: additive PIT/后复权/成交/停牌语义、candidate/supervision 双索引、availability masks、realized-plan/oracle-regret、hard-ST、global-tail、pinned prefetch、loss-based early stopping、candidate-complete 四折 registry/ledger/selection 与 portfolio freeze gate 均已实现。v3 已实际训练 `318,564` optimizer steps、`146,867,538` sample exposures；8 个作业均使用 `1.0 GiB` 可用物理内存硬保护且未触发，训练器内部 `2.0 GiB` 工作集软回收线保留。baseline/2022 的 exit 120 已证实仅为 stdout transport 中断，证据完整且未重训。
`gate0_state`: 原 active survivor-scope 的 `402,987` 个缺行情键阻塞已由 non-active 五域 view `seq100_pit_2012_2025_formal__9d6feb2a5ba7f15a7635b42f` 消除；独立 research-eligible/non-suspended anti-join、pack price coverage 和 factor coverage 均为 0 缺口。
`corrected_source_pack`: `daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_candidate_complete_2012_2025_v7/manifest.json`，SHA-256 `710b0421e554c31912ef249ca0a3df8fc1b0b3ba8b595a06784cae8fa34b7bda`；3509 panel dates、3404 PIT symbols、8,204,961 supervised samples、8,208,431 signal-day candidates，2012-2025 source role 无固定 validation/test；candidate eligibility 不依赖未来标签或 entry fill，执行/价格覆盖 gate 无缺口。
`artifact_owner`: `daily_research`; preferred new root is `daily_research/data/research_store/<artifact_id>/`.
`historical_artifacts`: old QDP research artifacts were physically migrated or deleted on 2026-07-07; `quant_data_platform/data/qdp_v2/research/` no longer exists as a training-pack location.
`resource_state`: 2026-07-10 repository/QDP/research cleanup has reclaimed `136,151,736,627` bytes (`126.82 GiB`) in total. The research-store phase archived and deleted 9 unreachable cold components (`57.353 GiB`), trimmed 86 prediction files across 43 runs (`16.376 GiB`), and deleted 63 unreferenced partial/smoke directories (`3.046 GiB`). Post-cleanup scan has zero safe-directory, prediction-trim, or cold-component candidates.
`gc_report`: `brain/references/repository_retention_cleanup_20260710.md`；cold component manifests/hashes/provenance are in `daily_research/brain/references/research_store_cold_assets_archive_20260710_105437.{md,json}`.
`research_store_physical`: schema-v2 pointer index + retention policy protect one source view, four purged walk-forward fold views, and the deprecated legacy 2025 view for historical reproducibility; all share one panel store and one today-close OHLCVA label store. Current index has 6 views and 14 reachable components.
`research_store_views`: source `seq100_path60_todayclose_ohlcva`；formal folds `seq100_path60_todayclose_ohlcva_purged_oos{2022,2023,2024,2025}`；deprecated pre-purge `seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025`.
`rollforward_views`: formal fold views expose only `train/oos`, carry label-end purge and complete-case audit fields, and refit normalization before each OOS start. The legacy duplicated validation/test view is retained only for historical reproducibility and must not be used for a new verdict.
`store_view_note`: next-open/path20 views are historical evidence only; their former manifests and component hashes are preserved in the cold-asset archive, not as runnable active views.
`next_method`: `audit_fixed_vs_predicted_vs_oracle_exit_then_train_execution_aligned_soft_exit()`；先在冻结候选和同一成本合同上拆解固定退出、当前预测退出和 oracle executable exit，再把 soft exit distribution 的 expected executable net return/value 作为单一核心改动；path/summary 保留为辅助。该候选通过前不跑 `global_tail_512`，execution remains separate and frozen.

### object `alpha_v2_history`
`type`: archived_research_line
`state`: 旧复杂模型线保留为历史研究和可复用基础设施；anchor `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` 仍是历史比较点，但不是 multi-seed、candidate matrix、execution review 或 active/default 依据。

## Pure Functions
- `select_relevant_objects(task)`: 从任务文本和路径选择对象；无关对象不激活。
- `classify_evidence(run)`: 把 smoke、dry-run、short-window、interrupted、insufficient、failed、completed run 分到对应证据等级。
- `activate_execution_boundary(objects, method)`: 只有 `execution_surface` 的 restore/activate/trade-plan 方法被调用时返回 true。
- `derive_next_action(seq100_state, evidence)`: 当前优先返回 `audit_exit_policy_gap_then_train_one_soft_execution_aligned_candidate()`；无合格候选时返回 `winner_null_keep_execution_frozen()`。
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
- Seq100 summary_v2 result (pre-purge historical evidence)：`daily_research/brain/references/seq100_summary_v2_multi_horizon_ohlc_result_20260707.md`
- Seq100 daily-only no-minute result (pre-purge historical evidence)：`daily_research/brain/references/seq100_daily_only_no_minute_result_20260708.md`
- Seq100 input ablation and summary_v2 no60 result (pre-purge historical evidence)：`daily_research/brain/references/seq100_input_ablation_and_summary_v2_no60_result_20260708.md`
- Seq100 daily-only summary_v2 result (pre-purge historical evidence)：`daily_research/brain/references/seq100_daily_only_summary_v2_result_20260708.md`
- Seq100 direct-value rank result (pre-purge historical evidence)：`daily_research/brain/references/seq100_direct_value_rank_result_20260708.md`
- Seq100 price-delta and low OHLCVA auxiliary result (pre-purge historical evidence)：`daily_research/brain/references/path_policy_seq100_price_delta_ohlcva_aux_low_result_20260709.md`
- Seq100 OHLCVA equal path-loss result (pre-purge historical evidence)：`daily_research/brain/references/path_policy_seq100_ohlcva_path_equal_result_20260709.md`
- Seq100 2025 roll-forward profile result (pre-purge historical evidence)：`daily_research/brain/references/path_policy_seq100_rollforward_2025_result_20260709.md`
- Seq100 all-profile 2024 validation + 2025 forward result (pre-purge historical evidence)：`daily_research/brain/references/path_policy_seq100_all_profiles_val2024_forward2025_result_20260709.md`
- Seq100 all-channel summary_v2 combo result (pre-purge historical evidence)：`daily_research/brain/references/path_policy_seq100_summary_v2_all_channels_combo_result_20260710.md`
- Seq100 purged 2022-2025 walk-forward result (authoritative current evidence)：`daily_research/brain/references/path_policy_seq100_purged_walkforward_2022_2025_result_20260710.md`
- Seq100 PIT-adjusted/global-tail approved contract：`daily_research/brain/references/seq100_pit_adjusted_global_tail_contract_20260711.md`
- Seq100 corrected pack semantics implementation and Gate-0 blocker：`daily_research/brain/references/seq100_pit_pack_semantics_implementation_20260711.md`
- Seq100 formal PIT view and corrected source-pack completion：`daily_research/brain/references/seq100_pit_formal_view_and_pack_20260711.md`
- Seq100 hard-ST/global-tail training-core implementation and throughput：`daily_research/brain/references/seq100_training_core_implementation_20260711.md`
- Seq100 two-stage candidate/freeze/one-time outer-audit orchestrator implementation：`daily_research/brain/references/seq100_research_generation_orchestration_implementation_20260711.md`
- Seq100 candidate-complete 2022-2025 development v3 result (`winner=null`, authoritative current evidence)：`daily_research/brain/references/path_policy_seq100_candidate_complete_development_v3_result_20260712.md`
- Archived Path20/alpha-v2 July run reconciliation：`daily_research/brain/references/path_policy_alpha_v2_july_archived_runs_reconciliation_20260710.md`
- Shortline plan：`daily_research/brain/references/shortline_after_close_research_plan_20260624.md`
- Stage 0 diagnostic：`daily_research/brain/references/shortline_stage0_fixed_next_open_diagnostic_20260624.md`
- Feature profile / semantics：`daily_research/brain/references/shortline_upside_feature_profile_20260624.md`、`daily_research/brain/references/shortline_upside_feature_semantics_20260624.md`
- Raw diagnostic / condition matrix：`daily_research/brain/references/shortline_raw_upside_diagnostic_20260624.md`、`daily_research/brain/references/shortline_raw_condition_matrix_20260624.md`
- Machine index：`daily_research/brain/references/evidence_registry.json`
