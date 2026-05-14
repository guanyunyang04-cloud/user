# Daily Research 状态中枢

快照日期：`2026-05-14`

## 当前结论
- `daily_research` 是当前正式生产研究与执行主线。
- active 执行物化真源：`daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile：`regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root：`daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前执行权重语义：`research_raw_target_weight`；权重上限语义：`follow_research_raw_no_global_cap`。
- continuous_policy 当前仍是 `research / shadow_only`；未过 formal evidence、v2 gate、stable confirm 与 promotion gate 前，不得替代 active 执行链。
- 当前有效 continuous_policy 研究基线仍是 r39 allocation objective consolidation；r40-r70 均为 research / shadow 升级链或基础设施证据。
- r64 已产出 full-window strict Gold：`continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`，`2010-01-04 -> 2026-04-10` observed strict window，`is_training_safe=true`，audit `ok`。
- r65 已新增 portfolio-set v5 后端，但 safe protocol 行为仍 source/receiver dead；不是策略有效性证据。
- r67 已用论文驱动 DFL-PG v1 替换 portfolio-set v5 内核默认目标、loss、oracle 与 profile default；tiny strict-Gold smoke `protocol_r67_paper_dfl_replace_v5_smoke_20260514_04` 完整跑通，但 training evidence 仍 `insufficient`，promotion gate 仍 `shadow_only`。
- r68 已新增 `portfolio_cashflow_decision_v1` 合同并打通 v5 prediction -> simulator -> release trace -> continuity metrics；tiny strict-Gold smoke `protocol_r68_cashflow_decision_v1_close_v5_smoke_20260514_03` 的 shadow source target=97、receiver target=139、intent conflict=0、cashflow valid=21/21，但仍只是 research / shadow translation-closure evidence。
- r69 已新增显式 `portfolio_set_v5_dfl_pg_v1_r69_value_arbitration` 研究线；traincheck `protocol_r69_value_arbitration_v5_behavior_traincheck_20260514_03` 显示 source target=1185、receiver target=160、wrong-side sell share=0，但 constraint violation 仍高且完整 tiny smoke `_01/_02` 在 evaluate 阶段被 TDX empty batch 中断；r69 尚未通过 behavior acceptance。
- r70 已修复 v5 版本边界与 r69 oracle violation：base 默认回到 `portfolio_set_v5_dfl_pg_v1`，r69 value arbitration 只在显式 profile/loss 或 artifact metadata 下启用；tiny strict-Gold smoke `protocol_r70_v5_version_boundary_oracle_repair_smoke_20260514_01` 完整通过 train/evaluate/shadow/export，oracle violation 近零、cashflow valid=1、intent conflict=0，但 training evidence 仍 `insufficient`，cash timing/reversal/source quality 仍失败，promotion gate 仍 `shadow_only`。

## 当前接管入口
- 默认读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- 首选工具入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<task>" --json`。
- 所有 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- PowerShell 中文显示异常时，先用显式 UTF-8 复读；不得直接判定文档损坏。
- `latest_*` 不得直接当真源；若 latest study/protocol/audit/ledger 不同源，必须使用 explicit study tag / protocol tag / dataset id。

## 当前主问题
- production 执行侧不是当前阻塞点；默认 active 继续由 `short_expert_policy_v5b` 承担。
- continuous_policy 的核心瓶颈是组合日级资金分配：谁是 receiver、谁是 source、留多少 cash、承受多少 turnover / cost / drawdown。
- r53-r55 解决了部分 cash/exposure closure，但 source/reduce/exit 和 cash timing 没闭合。
- r56-r61 推进 release-first / core-v4 接线，证明诊断与部分接线有效，但行为仍未闭合。
- r62-r64 完成通用 data lake 与 full-window strict Gold；数据基础设施已不再是 strict 训练集的主阻塞。
- r65 用 portfolio-set v5 取代 MLP 主线，接线通过但行为失败：`release_first_source_intent_count=0`、`portfolio_daily_source_target_count=0`、`portfolio_daily_receiver_target_count=0`、`intent_translation_conflict_rate=1.0`。
- r67 修复第一层机制闭环：训练 target `source/receiver/cash` 非零且 `target_intent_translation_conflict_count=0`；但评估 source target 与 shadow translation 仍未闭合。
- r68 修复 v5 资金流翻译闭环：cashflow contract 在 `_03` shadow 21/21 天 valid，source/receiver target 均非零，release trace primary blocker 为 `none`。下一 blocker 已转为训练证据不足、source 选择质量、cash timing、drawdown/reversal，而不是 source/receiver translation dead。
- r69 将下一 blocker 进一步定位为：value arbitration 目标覆盖有所改善，但 receiver 覆盖不足、oracle constraint violation 偏高、完整 evaluate/shadow smoke 受 TDX 数据读取阻塞。
- r70 已解除“oracle feasibility / TDX empty batch”这两个机制阻塞；当前 blocker 继续收敛到 receiver 覆盖、cash timing、reversal、source positive-forward sell 与 sufficient training evidence。

## 当前优先级
- P0：冻结 live/default/promotion/active artifact，所有新线先保持 research / shadow-only。
- P1：保持脑区控制面简洁；长历史、完整复盘、长命令进入 `references/`。
- P2：围绕 r70 后的 value arbitration 继续修 receiver/deploy 平衡、cash timing、drawdown/reversal、source quality 与 sufficient training evidence；translation closure 和 oracle feasibility 不再是当前主 blocker。
- P3：继续用 strict Gold dataset id 作为训练数据真源；realtime tail label 只可用于 research/audit。
- P4：保持 study/protocol 单进程研究框架；长任务可用外部后台启动 + 前台轮询，但研究本体仍应可诊断、可恢复。

## 当前边界
- formal、recent、promotion、live 不得混写。
- smoke、dry-run、short-window check、interrupted wrapper、insufficient evidence、failed trial、realtime tail label 都不能升级为正式 verdict。
- safe screening 或代码合同结果不得改 `daily_research/output/active_execution_strategy.json`。
- failed / timeout trial 可写诊断，不得写 completed evidence。
- outer study 未写 `study_summary.json` 时，只能写 protocol-level evidence，不能写 completed study verdict。

## 当前风险
- 主文档继续堆 dated log 会削弱 agent 接管效率；r66 起主脑只保留控制面。
- 只看 target-sum closure 会掩盖 release/source/receiver dead 与 intent translation conflict。
- 只增加 epoch、loss 或模型宽度可能掩盖 target construction 与 allocation semantics 断点。
- realtime Gold 若被误作 training-safe，会污染 completed training evidence。
- 后台 OS 进程轮询能降低交互超时风险，但不能替代 checkpoint、progress、artifact diagnostics。

## 最新证据索引
- r61 core-v4 decision wiring：`daily_research/brain/references/r61_release_first_decision_core_v4_status_20260514.md`。
- r62 DuckDB + Parquet research data lake：`daily_research/brain/references/r62_research_data_lake_status_20260514.md`。
- r63 brain-skill operating system：`daily_research/brain/references/r63_brain_skill_operating_system_status_20260514.md`。
- r64 full-universe strict Gold：`daily_research/brain/references/r64_full_universe_gold_data_lake_status_20260514.md`。
- r65 portfolio-set v5：`daily_research/brain/references/r65_portfolio_set_v5_status_20260514.md`。
- r66 brain maintenance：`daily_research/brain/references/r66_brain_maintenance_status_20260514.md`。
- r67 paper-DFL replace v5：`daily_research/brain/references/r67_paper_dfl_replace_v5_status_20260514.md`。
- r68 cashflow decision v1 close v5：`daily_research/brain/references/r68_cashflow_decision_v1_close_v5_status_20260514.md`。
- r69 value arbitration v5：`daily_research/brain/references/r69_value_arbitration_v5_status_20260514.md`。
- r70 v5 version boundary / oracle repair：`daily_research/brain/references/r70_v5_version_boundary_oracle_repair_status_20260514.md`。
- 机器索引：`daily_research/brain/references/evidence_registry.json`。

## 历史归档入口
- 本文件归档前完整快照：`daily_research/brain/references/state_center_archive_20260510.md`。
- 早期状态原文：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 早期状态索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- r50-r70 详细证据：`daily_research/brain/references/r*_*.md`。
