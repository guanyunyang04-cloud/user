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
