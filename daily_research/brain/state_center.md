# Daily Research 状态中枢

快照日期：`2026-06-03`

## 当前结论
- `daily_research` 是当前正式生产研究与执行主线；active 执行物化真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 live/default 历史事实标签：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`；effective profile 为 `regoff_k1_20d_ensemble_native_anchor`，执行权重语义为 `research_raw_target_weight` / `follow_research_raw_no_global_cap`。
- 2026-05-31 接管事实：`daily_research/output/` 与 `daily_research/cache/` 曾被误删；用户确认近期 brain 结论仍正确。缺失 payload 只阻塞文件级 replay、active artifact inspection、explicit run evidence lookup 和旧 `project_consistency_check.py` 语义，不自动推翻 brain-confirmed 研究结论。
- 2026-06-01 起执行端处于 `frozen_skeleton_only / awaiting_research_rebuild`：不得恢复 active、重建 production root、生成正式 trade plan、paper/live 或 broker 接线，除非后续有显式恢复/重建授权。
- 当前 path_policy / multi-horizon 新主线为 `daily_research_v2_research_reset`；旧 Stage 2.8 / short_v5b 降级为历史先验和参考 benchmark，不再阻塞 v2 主线。
- 当前 v2 pass-grade 研究基线：`mh_v2_reset_tradeable_mainboard_anchor_20260601_01`，strict pool `policy_pool_view__925e8604a91a9c07a5387fb1`，dataset `policy_input_bundle__45e3d8c059ba718426a9f887`，feature profile `raw_kline_context_v2_tradeable_amount_checked`；该基线不是 promotion-grade，执行端仍冻结。
- stricter traditional-PIT comparison anchor `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01` 为 `near_pass`，不替换当前 v2 strict pass anchor。
- v2 score bridge、candidate review、bad-month attribution、risk overlay、selective throttle、state sizing、local-state input/loss、horizon concentration repair 和 horizon train-contract 已打通为 research-only 证据链；`horizon_30d_soft_penalty_v1` 已三 seed forecast gate pass，但不得直接 promotion 或解冻执行端。
- 2026-06-03 frontier reconciliation 已确认两个未登记 seed7 run 为 research / shadow-only forecast diagnostics，详见 `daily_research/brain/references/daily_research_current_frontier_compaction_20260603.md`。
- continuous_policy 当前仍是 research / shadow-only；r61-r74 已解除若干 translation / oracle / lake collapse blocker，但 training evidence、feature contract health、cash timing、source quality 与 receiver-source spread 仍未闭合。

## 当前接管入口
- 默认读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- 首选工具入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`。
- 当前 frontier freshness：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`。
- 证据查询：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q <tag|dataset_id|r_id> --json`。
- 所有 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- `latest_*` 不得直接当真源；若 latest study/protocol/audit/ledger 不同源，必须使用 explicit run tag / protocol tag / dataset id。
- PowerShell 中文显示异常时，先用显式 UTF-8 复读；不得直接判定文档损坏。

## 当前主问题
- P0：冻结 live/default/promotion/active artifact；所有新线先保持 research / shadow-only。
- P1：执行端只保留骨架、只读诊断和候选评估能力；数据缺口严格阻断，不得回退旧交易日伪装“今日计划”。
- P2：保持脑区控制面简洁；长历史、完整复盘、长命令进入 `references/`。
- P3：v2 下一步是将 `horizon_30d_soft_penalty_v1` declared research candidate 接入 research-only score-backtest bridge，并相对 strict v2 baseline、traditional-PIT strict baseline、local-state input/loss 分支做证据对照。
- P4：之后再推进 per-symbol reversal/volatility bucket sizing、sector/liquidity/volatility/horizon caps、PIT/status 合同二阶段硬化、missing/fill 语义、limit/industry/valuation optional domains。
- P5：continuous_policy 围绕 r71/r74 继续验证 receiver/deploy 平衡、cash timing、drawdown/reversal、source quality、feature contract health 与 sufficient training evidence。

## 当前优先级
- `formal`、`recent`、`promotion`、`live` 不得混写。
- smoke、dry-run、short-window check、interrupted wrapper、insufficient evidence、failed trial、realtime tail label 都不能升级为正式 verdict。
- safe screening 或代码合同结果不得改 `daily_research/output/active_execution_strategy.json`。
- failed / timeout trial 可写诊断，不得写 completed evidence；外层等待窗口耗尽本身也不得写成 failed evidence。
- outer study 未写 `study_summary.json` 时，只能写 protocol-level evidence，不能写 completed study verdict。

## 当前边界
- `daily_research/output/active_execution_strategy.json` 缺失时，守卫应给出可读阻塞，不得 traceback，也不得手工从 brain 文本重造 active artifact。
- 本地 `output/cache` 空目录骨架不等于真实 payload 恢复；guard 通过不等于旧 run 可 file-backed replay。
- cap80、旧 replay 差异、corrected near-pass、单 seed forecast pass 或单一 candidate backtest 只能作为诊断线索，不能替代 v2 gate evidence。
- 长任务必须绑定 PID、stdout/stderr、progress、summary / checkpoint / artifact 路径；`Wait-Process -Id <pid> -Timeout 7200` 只是单轮前台等待窗口，进程仍推进时继续轮询。

## 当前风险
- 文档继续堆 dated log 会削弱接管效率；当前热路径应只保留结论、边界、入口和证据索引。
- 只增加 epoch、loss 或模型宽度可能掩盖 target construction 与 allocation semantics 断点。
- 若低预算、单 seed、best epoch 贴边、缺 learning curve 或缺多 seed聚合，不得写成模型质量结论。
- 只看 target-sum closure 会掩盖 release/source/receiver dead 与 intent translation conflict。

## 最新证据索引
- 当前前沿压缩与未登记 run reconciled：`daily_research/brain/references/daily_research_current_frontier_compaction_20260603.md`。
- v2 research reset 与框架合同：`daily_research/brain/references/daily_research_v2_research_reset_20260601.md`、`daily_research/brain/references/daily_research_v2_research_framework_contract_20260602.md`。
- v2 数据集/score/candidate/risk 证据：`daily_research/brain/references/daily_research_v2_dataset_contract_upgrade_20260602.md`、`daily_research/brain/references/daily_research_v2_score_backtest_bridge_20260602.md`、`daily_research/brain/references/daily_research_v2_candidate_review_matrix_20260602.md`、`daily_research/brain/references/daily_research_v2_selective_throttle_matrix_20260602.md`。
- v2 latest model-side references：`daily_research/brain/references/daily_research_v2_local_state_input_scout_20260602.md`、`daily_research/brain/references/daily_research_v2_local_state_loss_calibration_scout_20260603.md`、`daily_research/brain/references/daily_research_v2_horizon_concentration_repair_scout_20260603.md`、`daily_research/brain/references/daily_research_v2_horizon_concentration_train_contract_scout_20260603.md`。
- continuous_policy r61-r74 证据：见 `daily_research/brain/references/r61_release_first_decision_core_v4_status_20260514.md`、`daily_research/brain/references/r65_portfolio_set_v5_status_20260514.md` 到 `daily_research/brain/references/r74_lake_behavior_quality_status_20260515.md`。
- output/cache 误删恢复边界：`daily_research/brain/references/data_lake_output_cache_loss_recovery_boundary_20260531.md`。
- 机器索引：`daily_research/brain/references/evidence_registry.json`。

## 历史归档入口
- 历史归档：`daily_research/brain/references/state_center_archive_20260510.md`、`daily_research/brain/references/state_center_history_raw_20260424.md`、`daily_research/brain/references/state_center_evidence_index_20260424.md`。
