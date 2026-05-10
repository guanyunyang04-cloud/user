# Daily Research 过程复盘入口

快照日期：`2026-05-10`

## 当前结论
- `episodic_memory.md` 只保留最新动作后复盘和历史 archive 入口。
- 2026-05-10 之前的大段过程记录已强归档到 `daily_research/brain/references/episodic_memory_archive_20260510.md`。
- 当前最新复盘是 r52 native source-delta closure 与本轮主分脑强归档 / r52 代码模块拆分维护。
- 本轮新增脑区平台化与工作流状态机：保留 7 中枢结构，新增 `workflow_registry.json`、`brain_platform.py` 与 `brain_workflow.py`，用于接管胶囊、预检、freshness 和写回计划。

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
