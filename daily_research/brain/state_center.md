# Daily Research 状态中枢

快照日期：`2026-05-11`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- active 执行物化真源：`daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile：`regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root：`daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前执行权重语义：`research_raw_target_weight`；权重上限语义：`follow_research_raw_no_global_cap`。
- continuous_policy 当前仍是 `research / shadow_only`；未过 formal evidence、v2 gate、stable confirm 与 promotion gate 前，不得替代 active 执行链。
- 当前有效研究证据基线仍是 r39 allocation objective consolidation 的 `alpha_result_value_budget_split_v25` 与 `portfolio_daily_ranking_v2_gated`。
- r48 formal screening + confirmatory 已完成但 stable confirm 为空；r49-r52 是 research 入口与结构升级，不是策略有效 verdict。
- r52 native source-delta closure 已修通部分 source 通道，但未完成策略闭合：2/3 screening trials 恢复 source target，3/3 仍为 `training_evidence_status = insufficient`。
- 2026-05-11 已新增 explicit study evidence capsule：`brain_workflow status --workflow continuous_policy --study-tag <tag> --json` 可按指定 study 聚合 study / protocol / training / evaluation 证据；loose `latest_*` 仍显示 r52 study 与 r34 protocol/audit/ledger 不同源，不能直接作为真源。
- r52b 结构入口为 `split_heads_portfolio_daily_day_set_native_target_validity_closure_r52b` / `alpha_result_value_budget_split_v38`；它是 r52 的 native target validity 修复入口，不是 formal verdict。
- r52b safe screening `self_opt_study_r52b_native_target_validity_closure_screening_safe_20260511_01` 已完成 2/3 screening trials、0 failed，并因 resource gate 早停；结论是运行通道可用但 validity 未改善，不能进入 confirmatory 或 22 epoch resume。
- r52c 结构入口为 `split_heads_portfolio_daily_day_set_native_executable_receiver_closure_r52c` / `alpha_result_value_budget_split_v39`；safe screening 已完成 3/3 trials、0 failed，receiver executable closure 在 simulator 边界有效，但仍不是 formal verdict。
- r52d 代码合同已存在：`split_heads_portfolio_daily_day_set_native_validation_closure_r52d` / `alpha_result_value_budget_split_v40`，并已有 validation closure、train/sim alignment、deadband 常量共享的合同测试；dry-run `self_opt_study_r52d_validation_closure_dryrun_20260511_01` 通过；safe screening `self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01` 已完成 3/3 trials、0 failed、confirmatory disabled、true solver disabled；explicit capsule 已判定 `r52d screening-only failed confirmatory eligibility`，不得 promotion / live / active artifact。

## 当前接管入口
- 读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- `episodic_memory.md` 只作为过程复盘入口；历史原文、长命令和标题索引默认进入 `daily_research/brain/references/`。
- 运行 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- Windows 默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- OpenMP 根治验收口径：无 `KMP_DUPLICATE_LIB_OK` 运行 `daily_research/tools/openmp_runtime_check.py --strict` 必须通过。
- PowerShell 出现中文乱码时，先用显式 UTF-8 复读，不得直接判定文档损坏。

## 当前主问题
- production 执行侧不是当前阻塞点；默认 active 继续由 `short_expert_policy_v5b` 承担。
- continuous_policy 的深层瓶颈是组合日资金分配：同一天谁是 receiver、谁是 source、留多少 cash、承受多少 turnover / cost / drawdown。
- r31 / r33 / r34-r39 保留为 receiver/source/cash 合同与证据基线；r40-r52 是 allocation layer 升级链。
- r52c 之后当前最新问题不再是 unsupported receiver validity；瓶颈已转为 deployment / cash timing 闭合、exposure utilization 偏低、source depth 不稳与 formal training evidence 不足。
- r52c safe screening 中 `native_target_valid = 0.975~1.0`，`allocation_layer_native_fallback_used = 0~0.025`，`native_target_invalid_unsupported_receiver_count = 0`；但全部 trials 仍为 `training_evidence_status = insufficient`，`cash_timing_quality_1d < 0`，`portfolio_daily_exposure_utilization ~= 0.33`。
- r52d 完整 screening-only 证据已用 explicit capsule 读取；当前结论是不进入 confirmatory、resume 或加长同 tag 训练。下一轮若继续 r52 系列，应优先修 deployment / cash timing / exposure utilization / training evidence 闭合，而不是回到 receiver mask 修补。

## 近期研究索引
- r31：`split_heads_portfolio_daily_receiver_semantic_closure_r31`，保留 receiver executable closure 合同。
- r31 审计 marker：`direct_action_authorization_subset_violation_count`、`authorized_add_no_weight_change_share`、`deploy_intent_unrealized_share`。
- r33：`split_heads_portfolio_daily_source_forward_proxy_r33`，保留 `portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction` 与 `portfolio_daily_source_distribution_clean_pass`。
- r34：`split_heads_portfolio_daily_allocation_breadth_r34`，保留 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth` 与 `portfolio_daily_joint_economic_quality_gate`。
- r35：`split_heads_portfolio_daily_unified_allocation_r35`，保留 `portfolio_daily_unified_allocation_objective`。
- r36-r39：risk-aware / decision-focused / source hard-negative / allocation objective consolidation 是当前有效证据链，保留 `portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_receiver_source_spread_reward`。
- r39 当前仍对应 `portfolio_daily_ranking_v2_gated` 与 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15` 证据基线。
- r40-r48：end-to-end allocation layer、convex/OPE 方向可用但未过 stable confirm。
- r49-r52d：capital-flow closure、true solver 入口、native allocation vector、day-set native allocation vector 与 validation closure 均为 research / shadow；`source/receiver/cash listwise allocation teacher` 仍只是 teacher / warm start。

## 当前优先级
- P0：冻结 live / active artifact，只在 research / shadow 范围推进。
- P1：保持主分脑入口精炼，dated log 必须进入 `references/` archive。
- P2：继续把 receiver 可执行性、source 分布质量、monthly return、exposure utilization、realized deploy、cash timing 与 drawdown 写入 objective / feedback / gate。
- P3：不再优先追加单边 source/cash/reduce guard；优先推进统一 allocation objective / native allocation vector 的结构闭合。
- P4：r50 true solver 保留为 research 入口，但因本机负荷过高，不作为当前默认长训路径。

## 当前边界
- formal、recent、promotion、live 不得混写。
- smoke、dry-run、short-window check、repaired confirm、insufficient evidence 都不能升级为正式 verdict。
- `receiver_unrealized_deploy_share = 0`、`source_positive_forward_sell_share = 0` 或 source 通道恢复都只是必要条件，不是完成态。
- 不得用 safe screening 或代码合同结果改 `active_execution_strategy.json`。

## 当前风险
- 继续堆 simulator guard 会让系统退回“翻译器补漏洞”，不是学习组合资金分配。
- 放宽 source clean-pass 容易从 source dormant 退回强势误卖。
- 只加训练资源可能掩盖 native target 约束问题，并再次触发本机资源风险。
- 主文档若继续按日期堆积，接管会被历史细节淹没。

## 历史归档入口
- 本文件归档前完整快照：`daily_research/brain/references/state_center_archive_20260510.md`。
- 早期状态原文：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 早期状态索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- 过程复盘归档：`daily_research/brain/references/episodic_memory_archive_20260510.md`。
- 知识中枢归档：`daily_research/brain/references/knowledge_center_archive_20260510.md`。
- 操作中枢归档：`daily_research/brain/references/operations_center_archive_20260510.md`。
- 早期历史原文与索引仍保留在 `daily_research/brain/references/*_history_raw_*.md` 与 `*_evidence_index_*.md`。
## 2026-05-11 r52c Safe Screening Latest Verdict
- study tag: `self_opt_study_r52c_native_executable_receiver_closure_screening_safe_20260511_01` completed 3/3 trials, 0 failed, confirmatory disabled.
- receiver executable closure is effective in this round: `native_target_valid=0.975~1.0`, `allocation_layer_native_fallback_used=0~0.025`, `native_target_invalid_unsupported_receiver_count=0`.
- remaining blockers are no longer unsupported receiver validity; now the bottleneck is deployment/cash timing closure with weak utilization.
- stop conditions still triggered: `training_evidence_status=insufficient`; one trial has `source_target_count=2<3`; all trials have `cash_timing_quality_1d<0`; `portfolio_daily_exposure_utilization~0.33`.
- decision boundary: do not enter confirmatory and do not start 22-epoch resume on r52c; keep this branch as research/shadow-only.

## 2026-05-11 r52d Screening Verdict
- Fact: r52d profile and v40 loss contract are present in code and covered by focused tests for validation closure, train/sim alignment fields, and shared deadband constants.
- Fact: dry-run `self_opt_study_r52d_validation_closure_dryrun_20260511_01` passed with 3 v40 trials, `confirmatory_enabled=false`, `native_validation_closure_support=true`, and true solver disabled.
- Fact: safe screening `self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01` completed 3/3 trials, 0 failed, with `confirmatory_enabled=false`, `native_validation_closure_support=true`, and true solver disabled.
- Evidence capsule: `daily_research/brain/references/r52d_native_validation_closure_status_20260511.md`.
- Confirmatory gate failed: all trials have `training_evidence_status=insufficient`, `composite_score<0`, `cash_timing_quality_1d<0`, and `portfolio_daily_exposure_utilization~0.33`; trial 03 also has `source_target_count=0` and negative return.
- Current verdict: `r52d screening-only failed confirmatory eligibility`; there is no confirmatory run, stable confirm, promotion verdict, or live/default decision.
- Boundary: r52d may only be treated as a research/shadow screening result; do not promote it into live/default/promotion language.

## 2026-05-11 r52e Deployment/Cash/Exposure Closure Contract
- Fact: r52d is frozen as a failed screening-only baseline; it must not be extended into confirmatory, strict resume, promotion, live/default, or active artifact changes.
- Fact: r52e code contract now exists as `split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e` with loss profile `alpha_result_value_budget_split_v41`.
- Fact: r52e adds first-class allocation closure diagnostics for actual cash weight, actual gross exposure, deployable idle cash, target-vs-budget gap, receiver target support, source target support, native fallback, and `cash_semantics_mismatch`.
- Fact: scoring and resource gates now penalize high actual idle cash, excessive actual cash weight, low exposure utilization, receiver-target underuse, source dead days, and cash semantics mismatch.
- Boundary update: r52e now has a failed safe-screening verdict; it remains blocked from confirmatory, strict resume, promotion, live/default, and active artifact changes.

## 2026-05-12 r52e Safe Screening Verdict
- Fact: `self_opt_study_r52e_deployment_cash_exposure_closure_screening_safe_20260511_01` ran with safe resources, confirmatory disabled, and true solver disabled.
- Fact: resource gate stopped after 1/3 screening trials, saving 2 trials; original failures were `source_release_dead`, `economic_signal_too_weak`, `exposure_utilization_low`, and `actual_cash_weight_high`.
- Fact: trial 01 was not close to acceptance: `composite_score=-45.938456`, `annual_return=-0.268027`, `cash_timing_quality_1d=-0.053696`, `portfolio_daily_exposure_utilization=0.331584`, and `training_evidence_status=insufficient`.
- Fact: recomputed closure audit after the export fix shows `deployable_idle_cash_mean=0.540194`, `cash_semantics_mismatch=1.0`, and `receiver_candidate_without_target_day_share=0.925926`; patched resource gate would also fail `actual_cash_idle_high` and `cash_semantics_mismatch`.
- Evidence capsule: `daily_research/brain/references/r52e_deployment_cash_exposure_closure_status_20260512.md`.
- Current verdict: `r52e safe-screening failed resume and confirmatory eligibility`; next work must inspect target-weight underdeployment and source release dead, not add epochs or run confirmatory.

## 2026-05-12 r53 Cash-Funded Allocation Core Verdict
- Fact: r53 is a new research line, not a continuation of r52e patching. Code contract exists as `allocation_core_v2.py`, profile `split_heads_portfolio_daily_cash_funded_allocation_core_r53`, and loss profile `alpha_result_value_budget_split_v43`.
- Fact: r53 safe screening `self_opt_study_r53_cash_funded_allocation_core_screening_safe_20260512_03` completed 3/3 trials with confirmatory disabled and no live/default/active change.
- Fact: r53 fixed the main cash/exposure closure failure shape: best screening trial has actual cash about `0.189`, actual gross about `0.811`, target gap about `0.0049`, native fallback `0`, and `native_target_valid=1.0`.
- Fact: patched r53 resource gate re-evaluates the best trial as only `cash_timing_bad`; it no longer mislabels a budget-closed day as `cash_funded_deployment_failed`.
- Current verdict: r53 is code-contract plus safe-screening improvement, not a strategy verdict. Confirmatory, strict resume, promotion, live/default, and active artifact changes remain blocked because training evidence is insufficient, cash timing is negative, and composite score remains strongly negative.
- Evidence capsule: `daily_research/brain/references/r53_cash_funded_allocation_core_status_20260512.md`.

## 2026-05-12 r54 Semantic Budget Controller Verdict
- Fact: r54 code contract exists as `split_heads_portfolio_daily_semantic_budget_controller_r54` / `alpha_result_value_budget_split_v44`; it makes target-weight intent the main policy surface and derives execution actions from final target-weight deltas.
- Fact: r54 corrected r53 scorer/audit semantics: budget-closed days no longer require fresh receiver activity, and intent translation conflict is now deadband-aware.
- Fact: dry-run `self_opt_study_r54_semantic_budget_controller_dryrun_20260512_02` passed with confirmatory disabled and true solver disabled.
- Fact: safe screening `self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02` completed with 2 completed trials and 1 failed trial; confirmatory remained disabled.
- Best completed trial: `trial_02`, `composite_score=4.80437`, `annual_return=0.153993`, actual cash about `0.1885`, actual gross about `0.8115`, exposure utilization about `1.083`, target gap about `0.0048`, native fallback `0`, intent translation conflict `0`.
- Remaining blockers: training evidence is insufficient (`best_epoch=8` at edge), `cash_timing_quality_1d=-0.168248`, reduce/exit quality still fails, and trial 03 exited with code `3221226505` despite writing a protocol summary.
- Evidence capsule: `daily_research/brain/references/r54_semantic_budget_controller_status_20260512.md`.
- Current verdict: r54 is a code-contract and safe-screening improvement, not a strategy-success verdict; do not run confirmatory, promotion, live/default, or active artifact changes from this evidence.

## 2026-05-12 r55 Cash Timing Release Controller Verdict
- Fact: r55 code contract exists as `split_heads_portfolio_daily_cash_timing_release_controller_r55` / `alpha_result_value_budget_split_v45`; it adds failed-trial artifact diagnostics, per-trial behavior bottleneck reports, and explicit cash-timing/source-release/reduce-exit native loss terms.
- Fact: dry-run `self_opt_study_r55_cash_timing_release_controller_dryrun_20260512_01` passed with confirmatory disabled and true solver disabled.
- Fact: safe screening `self_opt_study_r55_cash_timing_release_controller_screening_safe_20260512_01` completed 2/2 trials, 0 failed, confirmatory disabled.
- Best screening trial: `trial_02`, `composite_score=5.128541`, `cash_timing_quality_1d=-0.168208`, actual cash about `0.187927`, target gap about `0.004817`, exposure utilization about `1.078645`, intent translation conflict `0`.
- Remaining blockers: training evidence remains insufficient with best epoch at edge, cash timing is still strongly negative, and source/reduce/exit are still all dead (`source_target_count=0`, realized sell rate `0`, reduce/exit `0`).
- Evidence capsule: `daily_research/brain/references/r55_cash_timing_release_controller_status_20260512.md`.
- Current verdict: r55 is research/shadow-only and not eligible for strict resume, confirmatory, promotion, live/default, or active artifact changes.

## 2026-05-13 GPU Training Runtime Acceleration
- Fact: seq_v3 training already used CUDA in r55; the runtime gap was missing AMP/GradScaler, pinned DataLoader memory, non-blocking transfers, and timing diagnostics.
- Fact: `training_runtime_acceleration.py` now configures CUDA AMP for non-cvxpy profiles and keeps cvxpy-layer profiles out of AMP while preserving fast transfers.
- Fact: seq_v3 diagnostics now write `gpu_acceleration`, `amp_enabled`, `data_loader_pin_memory`, `non_blocking_transfer`, and per-epoch timing fields.
- Evidence capsule: `daily_research/brain/references/r55_gpu_training_runtime_acceleration_status_20260513.md`.
- Boundary: this is training infrastructure and observability only; it does not change r55 verdict, active strategy, live/default, promotion, or confirmatory eligibility.

## 2026-05-13 r56 Release-First Constrained Decoder Status
- Fact: r56 code contract exists as `split_heads_portfolio_daily_release_first_constrained_decoder_r56` / `alpha_result_value_budget_split_v46`, with `allocation_core_v3.py`, `derive_release_first_intent(...)`, simulator `release_first_allocation_v3_mode`, and release-first diagnostics.
- Fact: focused required tests passed after implementation: `136 passed, 24 warnings`.
- Fact: dry run `self_opt_study_r56_release_first_constrained_decoder_dryrun_20260513_01` passed with confirmatory disabled, true solver disabled, and safe resource resolution.
- Fact: safe screening tags `self_opt_study_r56_release_first_constrained_decoder_screening_safe_20260513_01`, `_02`, and `_03` failed before completed protocol summaries due AMP/runtime issues; tag `_04` timed out after more than one hour and was stopped without a protocol summary.
- Fact: completed r56 safe-screening evidence is `0`; failed/timeout attempts are diagnostics only and must not enter completed evidence.
- Fact: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Inference: r56 allocator/source behavior cannot yet be judged; the current blocker is r56 training runtime completion under day-set v46, not source/reduce/exit semantics.
- Boundary: r56 remains research / shadow-only. Do not enter strict resume, confirmatory, promotion, live/default, or active artifact changes from this evidence.
- Next: add a shorter r56 GPU smoke/safe checkpoint or stronger epoch-level progress diagnostics before another full safe screening; do not repackage this as "add loss/epoch".

## 2026-05-13 r58 Framework Simplification Status
- Fact: r58 removes the r57 parent/child subprocess protocol runner and watchdog path from `run_self_optimizing_study.py`.
- Fact: r58 keeps low-complexity runtime observability: protocol progress JSONL, training progress events, study progress events, and explicit study-tag collision protection.
- Fact: active new-study search profiles are now limited to `focused_seq_v1`, r53, r54, r55, and r56 through `research_profile_registry.py`; legacy r19-r52 profiles remain historical compatibility data, not default new-study entrypoints.
- Fact: focused regression passed with `136 passed, 24 warnings`.
- Evidence capsule: `daily_research/brain/references/r58_framework_simplification_status_20260513.md`.
- Current verdict: r58 is research framework simplification only. It is not strategy evidence, confirmatory evidence, promotion support, live/default change, or active artifact change.
- Next workflow: use direct foreground protocol smoke before dry-run study and safe screening; do not use parent/child subprocess watchdog as the default research path.

## 2026-05-13 r59 Parallel Core V4 Status
- Fact: r59 adds parallel backend `formal_torch_core_v4` for the release-first research line, without replacing v3 artifacts or adding r59 logic to `model_seq_v3.py`.
- Fact: core-v4 training contract is epoch-based, resume-capable, and GPU-required, but explicitly `promotable=False`.
- Fact: core-v4 artifact type is `continuous_policy_torch_core_v4`; prediction emits target-weight intent, target-delta intent, release-first support fields, and `release_first_allocation_v3_mode=1.0`.
- Fact: active new-study search profiles are now limited to `focused_seq_v1`, r56 v3 reference, and r59 core-v4; r53-r55 remain legacy-compatible history, not default new-study entrypoints.
- Fact: focused regression passed with `136 passed, 24 warnings`.
- Fact: direct smoke `protocol_r59_core_v4_direct_smoke_20260513_03` completed with backend `formal_torch_core_v4`, CUDA AMP/pinned-memory/non-blocking diagnostics, parseable protocol summary, and `training_evidence_status=insufficient`.
- Fact: dry run `self_opt_study_r59_core_v4_release_first_dryrun_20260513_01` selected only the r59 core-v4 profile with `alpha_result_value_budget_split_v46`, `allocation_layer_v1`, `result_value_v10`, and `active_execution_strategy` prior.
- Fact: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Evidence capsule: `daily_research/brain/references/r59_core_v4_release_first_status_20260513.md`.
- Current verdict: r59 is research / shadow-only / core-v4 infrastructure. It is not safe-screening evidence, strategy-effectiveness evidence, confirmatory evidence, promotion support, live/default change, or active artifact change.
- Next: if continuing r59, run explicit-tag safe screening only after smoke and dry-run remain clean; do not infer strategy quality from smoke metrics.

## 2026-05-13 r60 Profile-Bound Core V4 Screening Status
- Fact: r60 fixes direct protocol profile binding. `--search-profile split_heads_portfolio_daily_release_first_core_v4_r59` now applies r59 base-trial defaults unless explicitly overridden.
- Fact: protocol summaries now include `profile_binding` and enriched release-first/source/cash/intent continuity diagnostics from turnover CSV.
- Fact: focused regression passed with `140 passed, 24 warnings`.
- Fact: direct smoke `protocol_r60_profile_bound_core_v4_smoke_20260513_01` completed with `profile_binding.profile_applied=true`, backend `formal_torch_core_v4`, loss `alpha_result_value_budget_split_v46`, allocation-layer semantics, result-value objective, and active strategy prior.
- Fact: dry run `self_opt_study_r60_profile_bound_core_v4_dryrun_20260513_01` selected only the r59 core-v4 profile.
- Fact: safe screening `self_opt_study_r60_profile_bound_core_v4_screening_safe_20260513_01` completed 1/1 trials, 0 failed, confirmatory disabled.
- Safe-screening verdict: completed evidence is negative. Trial 01 has `composite_score=-36.684195`, `release_first_source_intent_count=0`, `portfolio_daily_source_target_count=0`, `portfolio_daily_target_sum_gap~0.6148`, `portfolio_daily_actual_cash_weight_mean~0.8468`, `intent_translation_conflict_rate~0.8434`, reduce/exit still `0`, and resource gate stopped on source-dead/underdeployment/high-cash failures.
- Positive note: training evidence became sufficient (`best_epoch=6`, `completed_epochs=12`) and cash timing improved to `0.026076`, but this does not offset source/release/deployment failure.
- Fact: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Evidence capsule: `daily_research/brain/references/r60_profile_bound_core_v4_screening_status_20260513.md`.
- Current verdict: r60 is research / shadow-only. It proves correct profile binding and diagnostic surfacing, but it is not eligible for r61 strict resume, confirmatory, promotion, live/default, or active artifact changes.
- Next: inspect core-v4 release/source target generation and allocator-consumable target-delta wiring; do not reframe this as simply needing more epochs or loss weight.

## 2026-05-14 r61 Decision-Focused Core V4 Status
- Fact: r61 adds release-flow trace diagnostics, core-v4 release-first target construction, receiver support outputs, v47 decision-focused core-v4 loss, active r61 registry entry, and reusable training dataset cache.
- Fact: all constructed training dataset surfaces are now reusable by fingerprint: `sample_frame.pkl`, `daily_frame.pkl`, `teacher_summary.json`, and `metadata.json`; `train_policy.py` defaults to `--training-dataset-cache-mode auto`.
- Fact: focused regression passed with `154 passed, 24 warnings`; doc guard passed; active artifact diff is empty.
- Fact: smoke `protocol_r61_release_first_decision_core_v4_smoke_20260514_03` completed with profile binding, core-v4/v47, CUDA AMP, cache hit, and parseable release-flow diagnostics.
- Fact: dry run `self_opt_study_r61_release_first_decision_core_v4_dryrun_20260514_02` selected only the r61 active profile and kept `protocol_runner=in_process`.
- Fact: safe protocol `self_opt_study_r61_release_first_decision_core_v4_screening_safe_20260514_01__trial_01` completed and wrote a parseable protocol summary, but the outer study wrapper was interrupted before final `study_summary.json`.
- Screening protocol verdict: wiring improved but behavior remains blocked. Evaluation has tiny nonzero source intent/target `0.0241`, clean target gap `0.0026`, but conflict rate `0.99998`, cash timing `-0.16636`, and trace blocker `no_held_negative_delta`; shadow remains source-dead with `release_flow_primary_blocker=source_executable_dead`.
- Evidence capsule: `daily_research/brain/references/r61_release_first_decision_core_v4_status_20260514.md`.
- Current verdict: r61 is research / shadow-only. Do not enter confirmatory, strict promotion, live/default, or active artifact changes. Next work must repair held negative-delta/source executable semantics and intent translation conflict, not repackage this as simply needing more epoch or generic loss weight.

## 2026-05-14 r62 Research Data Lake Status
- Fact: r62 adds a local DuckDB + Parquet research data lake in `daily_research.data_lake`, with catalog APIs, build/import CLIs, and train-policy lake-backed reusable dataset lookup.
- Fact: full-universe Bronze/Silver market-feature data is registered for uncapped `learned_all_a`, `000300.SH`, `2010-01-04` to `2026-05-13`: `12,184,830` market rows, `109` feature panels, and `1,328,146,470` feature cells across `3,070` stocks.
- Fact: `1200` was only a prior engineering cap and remains in catalog as comparison evidence; data lake CLI default is now full universe via `--max-universe-size 0`.
- Fact: existing reusable Gold training caches were imported into the lake: `liquid500` and `learned_all_a` strict-train datasets for `2024-01-02` to `2025-12-31`.
- Fact: small real strict/realtime data lake smoke succeeded and explicitly marked realtime tail labels as unobserved.
- Blocker: full-universe Gold training-set construction is not complete; repeated Gold attempts showed `build_training_matrices(...)` is the bottleneck, not DuckDB/Parquet.
- Evidence capsule: `daily_research/brain/references/r62_research_data_lake_status_20260514.md`.
- Current verdict: r62 is research infrastructure. It improves data reuse and auditability, but it is not strategy evidence and must not affect live/default/promotion or the active artifact.

## 2026-05-14 r63 Brain-Skill Operating System Status
- Fact: r63 adds a compact brain operating protocol, task capsule CLI, evidence registry/query CLI, phased brain rules, and repo-canonical `daily-research-brain` project skill.
- Fact: evidence registry rebuild is clean and queryable; `r62` lookup returns market/feature and imported Gold dataset ids.
- Fact: the skill is stored in repo canonical source only; dry-run sync is available, but no global skill install has been claimed.
- Evidence capsule: `daily_research/brain/references/r63_brain_skill_operating_system_status_20260514.md`.
- Current verdict: r63 is brain/workflow infrastructure. It improves agent handoff, retrieval, guard checks, and writeback discipline, but it is not strategy evidence and must not affect live/default/promotion or the active artifact.

## 2026-05-14 r64 Full-Universe Gold Data Lake Status
- Fact: r64 adds resumable sharded Gold construction, lake-backed policy-input loading, portfolio checkpoints across shards, transparent sharded Gold loading, and Gold audit writeback.
- Fact: full-universe 2019H1 smoke passed from explicit Bronze/Silver dataset `policy_input_bundle__0f116a9b78c92ff045a6853d` with uncapped `learned_all_a` (`3,070` symbols).
- Fact: strict smoke dataset `continuous_policy_training_matrices__strict_train__03ce80c27c93371d6b5dcf47` has `78,450` sample rows, `98` daily rows, `5/5` shards, `unobserved_label_rows=0`, and audit `status=ok`.
- Fact: realtime smoke dataset `continuous_policy_training_matrices__realtime_research__acf000b3d5cf52b558706926` has `84,573` sample rows, `118` daily rows, `6/6` shards, `6,202` unobserved tail rows, `is_training_safe=false`, and audit `status=ok`.
- Fact: full-window strict Gold dataset `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1` is registered for `learned_all_a`, `2010-01-04` to `2026-05-13`, strict observed through `2026-04-10`, with `1,863,468` sample rows, `3,949` daily rows, `196/196` shards, `unobserved_label_rows=0`, `is_training_safe=true`, loader check passed, and audit `status=ok`.
- Blocker: full-window realtime Gold is still pending and must not be counted as completed training evidence.
- Evidence capsule: `daily_research/brain/references/r64_full_universe_gold_data_lake_status_20260514.md`.
- Current verdict: r64 has produced reusable full-universe strict Gold training data. It is data infrastructure, not continuous_policy strategy evidence, and must not affect live/default/promotion or the active artifact.

## 2026-05-14 r65 Portfolio-Set V5 Status
- Fact: r65 adds parallel backend `formal_torch_portfolio_set_v5`, artifact type `continuous_policy_torch_portfolio_set_v5`, and active profile `split_heads_portfolio_daily_release_first_portfolio_set_v5_r65` / `alpha_result_value_budget_split_v48`.
- Fact: r65 uses strict Gold dataset `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`, with `is_training_safe=true`, as the default lake-backed training dataset.
- Fact: direct smoke `protocol_r65_portfolio_set_v5_smoke_20260514_01` completed with profile binding, CUDA AMP/pinned-memory/non-blocking diagnostics, latent set attention, and `release_first_allocation_v3_mode=1.0`.
- Fact: dry run `self_opt_study_r65_portfolio_set_v5_dryrun_20260514_01` selected only the r65 v5 profile, kept confirmatory disabled, and used foreground in-process protocol execution.
- Fact: safe protocol `self_opt_study_r65_portfolio_set_v5_screening_safe_20260514_01__trial_01` completed and wrote a parseable protocol summary, but the outer study wrapper was interrupted before final `study_summary.json`.
- Screening protocol verdict: architecture and data wiring work, but behavior remains blocked. Evaluation has `release_first_source_intent_count=0`, `portfolio_daily_source_target_count=0`, `portfolio_daily_receiver_target_count=0`, `intent_translation_conflict_rate=1.0`, and `release_flow_primary_blocker=no_held_source`; shadow remains source/receiver dead.
- Evidence capsule: `daily_research/brain/references/r65_portfolio_set_v5_status_20260514.md`.
- Current verdict: r65 is `research / shadow-only / architecture upgrade`. Do not enter confirmatory, promotion, live/default, or active artifact changes. Next work should repair portfolio-set release/source/receiver target generation and intent translation conflict, not reframe this as simply needing more epochs or a return to MLP core-v4.
