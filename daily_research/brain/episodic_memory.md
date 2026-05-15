# Daily Research 过程复盘入口

快照日期：`2026-05-11`

## 当前结论
- `episodic_memory.md` 只保留最新动作后复盘和历史 archive 入口。
- 2026-05-10 之前的大段过程记录已强归档到 `daily_research/brain/references/episodic_memory_archive_20260510.md`。
- 当前最新接管复盘是 2026-05-14 handoff：主脑与 `daily_research` 分脑已按 brain-first 顺序完成接管，守卫通过；r65 仍是 research / shadow-only，下一步只能围绕 held source creation、receiver target realization、target/action translation 与 day-set sampling 做研究修复。
- 脑区平台化与工作流状态机已完成：保留 7 中枢结构，新增 `workflow_registry.json`、`brain_platform.py` 与 `brain_workflow.py`，用于接管胶囊、预检、freshness 和写回计划。

## 证据索引
- 当前状态：`daily_research/brain/state_center.md`。
- 当前规则：`daily_research/brain/knowledge_center.md`。
- 当前操作：`daily_research/brain/operations_center.md`。
- continuous_policy 合同：`daily_research/brain/continuous_policy_design_contract.md`。
- 完整历史快照：`daily_research/brain/references/episodic_memory_archive_20260510.md`。
- 早期证据索引：`daily_research/brain/references/episodic_memory_evidence_index_20260422.md`。

## 历史原文
- 2026-05-10 强归档快照：`daily_research/brain/references/episodic_memory_archive_20260510.md`。
- 早期历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 读取纪律：当前结论以本文件、`state_center.md`、`knowledge_center.md` 和 `operations_center.md` 为准；archive 只作历史证据追溯。

## 2026-05-10 r52 native source-delta closure 修复复盘
- 行动前自检：上一轮 r52 safe screening 已证明 day-set 模型与 safe 资源通道可运行，但两个 completed trials 均 `source_target_count = 0`、`source_realized_sell_rate = 0`；这不能只用 6 epoch 训练资源不足解释。
- 执行动作：训练侧 native projection 的 source 支撑改为当前持仓 delta 口径；simulator 在 native target 有效时由 `target_weight - current_weight` 直接推导 source / receiver target；pipeline 与 study progress 补齐 native-to-simulator 诊断字段。
- 验证证据：`py_compile` 通过；完整合同测试 `90` 项 OK；dry-run `self_opt_study_r52_native_source_delta_closure_dryrun_20260510_02` 通过。
- 筛选结果：safe screening `self_opt_study_r52_native_source_delta_closure_screening_safe_20260510_02` 完成 3 个 trial、0 failed；trial_01 / trial_02 分别恢复 `source_target_count = 51 / 260` 与 `source_realized_sell_rate = 1.0 / 1.0`，trial_03 仍 source dead。
- 行动后判断：source delta 已部分穿透旧 mask，但 native target valid rate、source delta audit threshold、turnover constraint 与 training evidence 仍未闭合；不得进入 confirmatory 或 promotion。

## 2026-05-10 主分脑强归档与 r52 代码模块拆分维护复盘
- 行动前自检：brain / project consistency 均为 OK；`doc_guard.py check` 的主要维护点是 `operations_center.md` 超过结构警戒线；`model_seq_v3.py` 已膨胀到约 12,496 行。
- 执行动作：将 `state_center.md`、`knowledge_center.md`、`operations_center.md` 与旧 `episodic_memory.md` 的归档前完整内容写入 `references/*_archive_20260510.md`，主文件改为当前入口和索引。
- 代码动作：r51/r52 native allocation projection/loss 拆入 `native_allocation.py`；day-set dataset/collate 拆入 `day_set_batching.py`；slot attention 拆入 `day_set_modules.py`；`model_seq_v3.py` 继续 re-export 原符号以保持兼容。
- 边界：本轮不改 r52/r51 行为、不改 loss 权重、不改 profile、不启动 screening/confirmatory、不改 production/live/active artifact。
- 完整验证结果以本轮最终输出和 `git diff` 为准。

## 2026-05-10 脑区平台化与工作流状态机复盘
- 行动前自检：7 中枢结构、强归档、`doc_guard`、`brain_integrity_check` 与 `project_consistency_check` 已通过，问题不在文档数量，而在接管、预检、freshness、写回计划仍缺少统一机器状态。
- 执行动作：新增 workflow registry、共享平台模块与统一 CLI；`brain_bootstrap --json` 兼容原字段并补充 `artifact_freshness`、`workflow_hints` 与 `encoding_report`。
- 边界：workflow JSON 是运行态胶囊，不替代 brain 主文件；默认只读，写回 brain 必须显式触发。

## 2026-05-11 r52b native target validity closure 复盘
- 行动前自检：r52 safe screening `self_opt_study_r52_native_source_delta_closure_screening_safe_20260510_02` 完成 3/3 trial，source delta 在 2/3 trial 中恢复，但 `native_target_valid` 仅约 1%-6%，且 loose `latest_*` 混有 r52 study 与 r34 protocol/audit/ledger。
- 执行动作：新增 explicit study evidence capsule，`status --study-tag <tag>` 可聚合 study / protocol / training / evaluation；simulator 输出 `allocation_layer_native_fallback_used` 与 `native_target_invalid_*` 分原因字段；新增 r52b/v38 validity-first native allocation profile。
- 边界：r52b 仍是 research / shadow 入口，不改 production/live/active artifact，不启用 true solver；是否进入长训取决于后续 dry-run 与 safe screening 的 native target validity 证据。

## 2026-05-11 r52b safe screening 验证复盘
- 行动前自检：`py_compile`、脑区平台测试 `14` 项、continuous_policy 合同测试 `94` 项、brain/doc/project/OpenMP strict 与 `git diff --check` 均通过；无冲突训练进程；screening tag 未存在。
- dry-run gate：`self_opt_study_r52b_native_target_validity_closure_dryrun_safe_20260511_01` 通过，3 个 trial 均为 `alpha_result_value_budget_split_v38`，`native_target_validity_closure_support = true`，`full_universe_train_solver_effective = false`，`confirmatory_enabled = false`，`resource_profile = safe`。
- screening 结果：`self_opt_study_r52b_native_target_validity_closure_screening_safe_20260511_01` 完成 2/3 screening trials、0 failed；resource gate 因 `source_release_dead`、`receiver_deploy_not_clean`、`economic_signal_too_weak` 早停，节省 1 个 trial。
- 关键证据：trial_01 `native_target_valid = 0.0625`、`allocation_layer_native_fallback_used = 0.9375`、`source_target_count = 62`、`source_realized_sell_rate = 1.0`；trial_02 `native_target_valid = 0.0125`、`allocation_layer_native_fallback_used = 0.9875`、`source_target_count = 0`、`source_realized_sell_rate = 0`。
- invalid reason：两个 trial 的主要 simulator invalid reason 均是 unsupported receiver，`native_target_invalid_unsupported_receiver_count = 306 / 346`；sum、turnover、cap、negative weight、sell nonheld 的 simulator invalid count 为 0。
- 行动后判断：r52b 路径可运行，但没有解决 native target validity 主瓶颈，不能进入 confirmatory、promotion 或 22 epoch strict resume；下一轮优先修 receiver executable mask、native target export 与 simulator validity 同口径。
## 2026-05-11 r52c Safe Screening + Evidence Export Closure
- Fact: `self_opt_study_r52c_native_executable_receiver_closure_screening_safe_20260511_01` completed 3/3 safe screening trials (0 failed), confirmatory disabled.
- Fact: native executable receiver closure worked at simulator boundary:
  `native_target_valid=0.975~1.0`, `allocation_layer_native_fallback_used=0~0.025`,
  `native_target_invalid_unsupported_receiver_count=0` across trials.
- Fact: acceptance still failed on closure depth:
  `training_evidence_status=insufficient` for all trials, one trial `source_target_count=2<3`,
  all trials `cash_timing_quality_1d<0`, and exposure utilization stayed near `0.33`.
- Action: patched `run_self_optimizing_study.py` so ranking/summary exports include native validity + fallback + invalid-reason metrics directly in `primary_metrics`.
- Evidence: `brain_workflow status --workflow continuous_policy --study-tag ... --json` now returns coherent trial-level native metrics without relying on stale loose latest files.

## 2026-05-11 全仓维护与 r52d 状态纠偏复盘
- 行动前自检：代码中已有 r52d profile、v40 loss contract 与合同测试，但分脑仍主要写成“后续设计 r52d”。
- 执行动作：`brain_workflow health` 改为并行聚合四项健康检查并输出每项耗时；主分脑写回 r52d code-contract-only 状态。
- 验证补充：`self_opt_study_r52d_validation_closure_dryrun_20260511_01` dry-run 通过，3 个 trials 均为 `alpha_result_value_budget_split_v40`，confirmatory disabled，true solver disabled。
- 运行态发现：既有 `self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01` safe screening 后续已自然结束，产出 `study_summary.json`；结果为 3/3 screening trials completed、0 failed、confirmatory disabled、true solver disabled。
- 边界：本轮不改 `active_execution_strategy.json`，不切换 live/default/promotion，不从 loose `latest_*` 自动落盘。
- 行动后判断：r52d 只能作为 screening-only research result；没有 confirmatory、stable confirm 和 promotion verdict 前，不能写成策略结论。

## 2026-05-11 r52d Explicit Evidence Verdict
- 行动前自检：上一轮只确认 r52d safe screening 完成，并未裁决是否值得进入 confirmatory；loose `latest_*` 仍混有 r52d study 与 r34 protocol/audit/ledger，不能作为自动真源。
- 执行动作：读取 `brain_workflow status --workflow continuous_policy --study-tag self_opt_study_r52d_native_validation_closure_screening_safe_20260511_01 --json`，并固化 reference capsule：`daily_research/brain/references/r52d_native_validation_closure_status_20260511.md`。
- 事实：3/3 screening trials completed、0 failed、confirmatory disabled、true solver disabled；`native_target_valid = 1.0 / 1.0 / 0.987654`，fallback `0 / 0 / 0.012346`。
- 失败证据：3 个 trials 均 `training_evidence_status=insufficient`、`composite_score<0`、`cash_timing_quality_1d<0`、`portfolio_daily_exposure_utilization~0.33`；trial 03 还出现 `source_target_count=0` 与负收益。
- 行动后判断：`r52d screening-only failed confirmatory eligibility`；不得 confirmatory、strict resume、promotion、live/default 或改 active artifact。下一步只应在 research code/objective 层修 deployment、cash timing、exposure utilization 与 training evidence 闭合。

## 2026-05-11 r52e Deployment/Cash/Exposure Closure Implementation
- 行动前自检：r52d 已明确失败在 deployment / cash timing / exposure utilization / training evidence 闭合，不应继续用加长训练或 confirmatory 掩盖逻辑问题。
- 测试先行：新增 focused tests 先复现 actual cash 高、cash reserve signal 为 0、receiver 有支持但 target exposure 低于 stock budget、resource gate 未阻断的失败形态。
- 执行动作：新增 allocation closure diagnostics，补充 native allocation deployable idle cash / stock budget gap loss，导出 actual cash/exposure 指标，并把 scoring/resource gate 改为惩罚实际闲置现金和 `cash_semantics_mismatch`。
- 执行动作：新增 r52e profile `split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e` 与 loss profile `alpha_result_value_budget_split_v41`；不复用 r52d verdict。
- 行动后判断：r52e 目前是代码合同，不是 study verdict；下一步只允许 dry-run / safe screening 验证，confirmatory、strict resume、promotion、live/default 继续阻断。

## 2026-05-12 r52e Safe Screening Failure + Export Fix

## 2026-05-12 r53 Cash-Funded Allocation Core Rebuild
- Action before self-check: r52e was already submitted and failed because deployment/cash/exposure did not close; the next line was allowed to rebuild only the allocation core, not data/training/brain/live frameworks.
- Implementation: added `allocation_core_v2.py`, r53 search profile, v43 loss profile, simulator r53 path, closure diagnostics, r53 cash-first scoring/resource gate, and contract tests.
- First r53 screening finding: old receiver executable candidate gates still limited v2 support; fixed by broadening r53 receiver/source support from score, executability, and headroom.
- Second r53 screening finding: learned gross target could collapse r53 stock budget to about `0.20`; fixed by adding `allocation_core_v2_stock_budget_floor` and honoring it in simulator constraints.
- Third r53 screening result: `self_opt_study_r53_cash_funded_allocation_core_screening_safe_20260512_03` completed 3/3 safe screening trials with confirmatory disabled. Best trial closed cash/exposure materially better but still had insufficient training evidence, negative cash timing, and strongly negative composite score.
- Follow-up fix: r53 resource gate now requires cash-funded deployment only when deployment is actually needed; recomputing the best trial leaves only `cash_timing_bad`.
- Post-action verdict: r53 is a successful code-contract and cash/exposure closure improvement, not a final model. Do not run confirmatory, strict resume, promotion, live/default, or active artifact changes from this evidence.

## 2026-05-12 r54 Semantic Budget Controller Implementation
- 行动前自检：r53 已修现金/敞口闭合，但 audit/scoring 会把预算已闭合且无新增部署需求误判为 receiver/headroom 失败；下一步应修语义和评分，而不是继续 strict resume。
- 测试先行：新增 r53 budget-closed gate 测试、r54 profile / predict-policy / simulator intent 合同测试，以及 deadband-aware intent translation 回归测试。
- 执行动作：新增 r54 `alpha_result_value_budget_split_v44` 和 `split_heads_portfolio_daily_semantic_budget_controller_r54`；预测层导出 target-weight intent，simulator 用 r53 allocator 解最终权重并由 delta 派生 execution action。
- 纠偏动作：`_score_protocol_summary`、resource gate、behavior audit 与 simulator diagnostics 统一 `receiver_activity_required` 口径；intent translation conflict 改为 deadband-aware。
- 验证动作：focused tests 通过；dry-run `self_opt_study_r54_semantic_budget_controller_dryrun_20260512_02` 通过；safe screening `self_opt_study_r54_semantic_budget_controller_screening_safe_20260512_02` 完成，confirmatory disabled。
- 筛选结果：2 个 trial completed，1 个 trial failed；best completed trial 02 `composite_score=4.80437`，actual cash about `0.1885`，actual gross about `0.8115`，target gap about `0.0048`，intent translation conflict `0`，native fallback `0`。
- 行动后判断：r54 改善了评分形态并保持 cash/exposure closure 干净，但仍被 `training_evidence_status=insufficient`、negative cash timing、reduce/exit quality 和 trial 03 abnormal exit 阻断；不得 confirmatory、promotion、live/default 或改 active artifact。

## 2026-05-12 Single-Process Study Runner Correction
- 行动前自检：r54 screening 监控暴露出父进程/子进程口径增加了不必要复杂度；用户明确要求训练就是一个进程。
- 测试先行：新增合同测试，默认 `_run_protocol_with_progress` 必须调用 in-process `protocol_main`，且不得触发 `subprocess.Popen` 或写出 `protocol_runner_command`。
- 执行动作：取消默认 protocol subprocess 路径；resource env、Windows priority/affinity 与 Torch thread limit 改为应用到当前进程；progress 事件固定写 `protocol_runner_mode=in_process`。
- 行动后判断：后续 long-run 监控只看单一 study/training 进程和 progress files，不再按父/子进程拆分判断。

- 行动前自检：r52e dry-run 通过后才启动 safe screening；命令显式设置 `--disable-confirmatory --resource-profile safe --thread-limit 4 --cpu-affinity-count 4 --process-priority below_normal`，未修改 active/live/default。
- 运行结果：`self_opt_study_r52e_deployment_cash_exposure_closure_screening_safe_20260511_01` 完成 1/3 screening trials、0 failed；resource gate 早停并节省 2 个 trial。
- 失败证据：trial 01 `composite_score=-45.938456`、`annual_return=-0.268027`、`cash_timing_quality_1d=-0.053696`、`portfolio_daily_exposure_utilization=0.331584`、`training_evidence_status=insufficient`、`source_target_count=0`。
- 过程发现：原 study summary 未把 `cash_semantics_mismatch` 写入 gate，因为 behavior audit 从聚合后的 `day_merge` 计算 closure，丢失 turnover export 的 stock budget / target sum / receiver target 字段。
- 修复动作：改为从 turnover export 计算 closure，并让 diagnostic 识别 simulator-native 列名；recomputed audit 显示 `deployable_idle_cash_mean=0.540194`、`cash_semantics_mismatch=1.0`、`receiver_candidate_without_target_day_share=0.925926`。
- 行动后判断：r52e 首轮 screening 失败；不得 confirmatory、strict resume、promotion、live/default。下一步应审计 target weight sum 为什么锁在约 0.28，而 stock budget / gross target 仍约 0.83-0.85。

## 2026-05-14 项目接管复盘
- 行动前自检：先读主脑 manifest、identity、state、knowledge、topology、operations、governance，再进入 `daily_research` 分脑；PowerShell 初次中文输出疑似乱码，已用显式 UTF-8 和 `brain_bootstrap.py --json` 复核，确认是终端编码显示问题，不是 brain 文件损坏。
- 事实：当前分支为 `main`，本地相对 `origin/main` 超前 2 个提交；接管时工作区无未提交改动；`daily_research/output/active_execution_strategy.json` 无 diff。
- 事实：当前 live/default 仍由 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active` 承担；continuous_policy 仍是 `research / shadow_only`，r65 portfolio-set v5 只是 architecture upgrade，不是策略有效性证据。
- 事实：守卫通过：`git diff --check` 无输出，`doc_guard.py check` 通过，`brain_integrity_check.py --json` 返回 `status=ok`。
- 风险：`latest_*` 存在来源不一致提醒，latest study tag 与 latest protocol tag 不同；后续必须继续使用 explicit study tag、protocol tag 或 dataset id。
- 推断：当前最合理下一步不是启动 confirmatory 或改 live，而是先修 r65 的 held source creation、receiver target realization、target/action translation 与 day-set sampling。
- 边界：本轮只完成接管、核对和复盘写回；不修改 active artifact，不写 completed strategy evidence，不进入 promotion / live / default 切换。

## 2026-05-15 r73 Lake-Native R71 Collapse Repair 复盘
- 行动前自检：运行 task capsule，确认 `daily_research/output/active_execution_strategy.json` clean；固定证据 tag 为 `protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02`，不使用 loose `latest_*` 作结论真源。
- 事实：r73 实现了 `portfolio_decision_feature_bundle_v1`、lake feature utilization report、r71/lake collapse diagnostics，并修复 r72 lake smoke 中 source/receiver target 为 0 的坍缩。
- 事实：`protocol_r73_lake_native_r71_collapse_repair_smoke_20260515_02` 完整跑通；eval/shadow 均为 `cashflow_decision_contract_valid_rate=1.0`、`intent_translation_conflict_rate=0.0`，且 source/receiver target 均非零。
- 事实：完整回归通过：continuous_policy `248 passed`，data_lake `15 passed`，`git diff --check` clean，`doc_guard.py check` 通过，`brain_integrity_check.py --json` 为 `status=ok`，active artifact diff 为空。
- 推断：当前 blocker 已从 lake provider / source-receiver collapse 转移到 behavior quality 与 training evidence：cash timing、source positive-forward sell、receiver-source spread、training evidence insufficient。
- 边界：r73 是 research / shadow evidence，不是 promotion、confirmatory、live 或 completed strategy verdict；不得把 lake eval 或 realtime/tail label 写成 completed training evidence。
