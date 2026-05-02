# Daily Research 状态中枢

快照日期：`2026-05-02`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- active 执行物化真源为 `daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 为 `regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前执行权重语义固定为 `research_raw_target_weight`，权重上限语义固定为 `follow_research_raw_no_global_cap`。
- continuous_policy 最新有效研究状态为 r39 allocation objective consolidation bounded confirm 之后的 `research / shadow_only`：r31 receiver 语义闭包、r33 source clean-pass、r34 allocation breadth、r35 unified allocation、r36 risk-aware unified allocation、r37 decision-focused allocation、r38 source hard-negative regret 仍保留；r39 已把收益、回撤、现金部署、交易质量、月度质量与 final allocation objective 写入统一训练入口，并把 r39 final objective 回写到 receiver/source/cash 执行评分，但未过 v2 gate / stable confirm。
- 未完成正式判定前，不得 promotion、不得 live、不得改 active artifact。

## 当前接管入口
- 读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- `episodic_memory.md` 只作为过程复盘入口；历史原文、长命令和标题索引默认进入 `daily_research/brain/references/`。
- 运行 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；涉及 PyTorch / MKL / OpenMP 冲突时设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
- PowerShell 出现中文乱码时，先用显式 UTF-8 复读，不得直接判定文档损坏。

## 当前主问题
- production 执行侧不是当前阻塞点；默认 active 继续由 `short_expert_policy_v5b` 承担。
- continuous_policy 的深层瓶颈已经从“单票动作是否能学会”推进到“组合日资金如何分配”。
- r31 已把 receiver 授权闭包前置：所有 direct add/open authorized 必须是 executable receiver candidate 子集；无 headroom held add 必须在语义层降级并审计。
- r33 已把 source 分布防错前置：`portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction` 与 `portfolio_daily_source_distribution_clean_pass` 必须共同约束 source candidate。
- r34 bounded screening 暴露 candidate selection 合同漏洞：旧逻辑会让 `training_evidence_status = insufficient` 的高分 trial 进入 confirm；已修为 confirm 候选优先 evidence sufficient + v2 gate qualified。
- r38 bounded confirm 指向的真实瓶颈仍成立：source 分布方向被明显拉正，`source_positive_forward_sell_share = 0.0`、`source_strong_positive_forward_sell_count = 0`，但收益太弱、source 数量过窄、cash timing / reduce / exit / drawdown 未过线。
- r39 bounded confirm 证明架构性收敛入口已能前台跑通并形成强收益/正 spread：execblend confirm 为 `training_evidence_status = sufficient`、`annual_return = 2.676289`、`sharpe = 3.802282`、`monthly_return_mean = 0.115004`、`receiver_target_count = 6`、`receiver_unrealized_deploy_share = 0.0`、`source_realized_sell_rate = 1.0`、`source_positive_forward_sell_share = 0.0`、`receiver_minus_source_forward_excess_5d = 0.066883`；但 `source_target_count = 1`、`cash_reserve_rate = 0.947020`、`cash_timing_quality_1d = -0.138133`、`max_drawdown = -0.109069`，v2 gate 仍只过 `9/12`，stable confirm 失败于 `confirm_gate_pass` 与 `confirm_source_count_floor`。
- 当前 r31 profile 标记：`split_heads_portfolio_daily_receiver_semantic_closure_r31`；核心审计字段为 `direct_action_authorization_subset_violation_count`、`authorized_add_no_weight_change_share`、`deploy_intent_unrealized_share`。
- 当前 r33 profile 标记：`split_heads_portfolio_daily_source_forward_proxy_r33`；当前 r34 profile 标记：`split_heads_portfolio_daily_allocation_breadth_r34`；当前 r35 profile 标记：`split_heads_portfolio_daily_unified_allocation_r35`；当前 r36 profile 标记：`split_heads_portfolio_daily_risk_aware_unified_allocation_r36`；当前 r37 profile 标记：`split_heads_portfolio_daily_decision_focused_allocation_r37`；当前 r38 profile 标记：`split_heads_portfolio_daily_source_hard_negative_regret_r38`；当前 r39 profile 标记：`split_heads_portfolio_daily_allocation_objective_consolidation_r39`。
- 当前核心 objective / calibration 标记为 `portfolio_daily_ranking_v2_gated` 与 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`；当前核心 loss 标记为 `alpha_result_value_budget_split_v25`；保留 scoring / objective 锚点为 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth`、`portfolio_daily_joint_economic_quality_gate`、`portfolio_daily_unified_allocation_objective`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_receiver_source_spread_reward`；r38 标记为 `portfolio_daily_source_tail_false_sell_penalty`、`portfolio_daily_source_release_preference`、`portfolio_daily_transfer_regret_target`、`portfolio_daily_source_hard_negative_prevalence` 与 `portfolio_daily_source_hard_negative_selected_pressure`；r39 标记为 `portfolio_daily_allocation_trade_quality_target`、`portfolio_daily_allocation_cash_deployment_target`、`portfolio_daily_allocation_risk_adjusted_return_target`、`portfolio_daily_allocation_drawdown_control_target`、`portfolio_daily_allocation_monthly_quality_target` 与 `portfolio_daily_allocation_final_objective`。

## 当前优先级
- P0：保持 live / active artifact 冻结，只在 research / shadow 范围推进。
- P1：维护主分脑入口精炼，避免 `state_center.md` 和 `operations_center.md` 继续变成长日志。
- P2：继续把 receiver 可执行性、source 分布质量、monthly return、exposure utilization、realized deploy、cash timing 与 drawdown 写入 objective / feedback / gate。
- P3：下一轮研究不再优先加单边 source false-sell 或 cash penalty；r39 已说明 final objective 接入后可以恢复收益和正 spread，但仍 source count 过窄、现金过高、cash timing 与 drawdown 不稳。下一优先级应在不放松 source hard-negative 的前提下，提高 clean source breadth 和可卖源分布，并把 cash timing / drawdown 的日级触发从后验 gate 推进到 allocation layer。
- P4：中期正路仍是 source/receiver/cash listwise allocation teacher 向正式 allocation layer 迁移；`solve_semidifferentiable_allocation` 是半可微最终 allocation 层合同，显式约束 cash、turnover、position cap 与 transaction cost；simulator guard 只保留最后安全层。

## 当前边界
- formal、recent、promotion、live 不得混写。
- smoke、dry-run、replay、short-window check、repaired confirm 和 insufficient training evidence 都不能升级为正式 verdict。
- `training_evidence_status = sufficient`、v2 gates 全过、confirm-vs-screening 稳定、`receiver_unrealized_deploy_share = 0`、`source_realized_sell_rate >= 0.35`、`cash_reserve_rate > 0`、收益/月度质量/回撤/source 分布同时达标前，不进入 promotion 讨论。
- `receiver_unrealized_deploy_share = 0` 或 `source_positive_forward_sell_share = 0` 只是必要条件，不是完成态。
- 语义干净但收益弱、收益强但强势误卖、source 休眠、cash dead branch 都只能作为研究证据。

## 当前风险
- 如果继续堆 simulator guard，系统会变成“翻译器补漏洞”，而不是学习组合资金分配。
- 如果放宽 source clean-pass，容易从 source dormant 退回强势误卖。
- 如果只看 release conviction，会放过尾部正 forward source；r33 已证明该风险真实存在。
- 如果只看 receiver headroom，可能得到语义干净但 receiver 数量不足或收益弱的 dead branch。
- 如果入口文档继续按日期堆积，接管会重新变慢，且当前结论会被历史细节淹没。

## 近期研究索引
- r19-r23：组合日频 receiver/source/cash ranking、v2 gated、cash-aware、source execution、receiver execution 与 stable confirmatory 搜索。
- r24-r26：receiver 可买性训练信号、source-release listwise、allocation teacher、funding closure / transfer / dead branch。
- r27-r30：source economic release、forward-strength brake、direct-release relief 与 cash-relief。
- r31：receiver semantic closure，修复 direct add/open 授权绕过 executable receiver gate。
- r33：source forward proxy、release conviction 与 distribution clean-pass，修复强势 source 误卖但尚未恢复足够 clean breadth。
- r34：allocation breadth bounded/evidence-confirm 已完成；工程链路可执行，训练证据可充分，但 fresh confirm 未通过 v2/stability，失败核心是 source 分布与 receiver-source spread，而不是 receiver 可执行性或 GPU/yolos 训练链路。
- r35：unified allocation 已完成修复后 bounded confirm；入口为 `split_heads_portfolio_daily_unified_allocation_r35`、loss 为 `alpha_result_value_budget_split_v21`。`postfix4_bounded_confirm_20260430` 完成 4 个 screening 与 2 个 confirm，`stable_confirmatory_count = 1`，champion 为 `confirm_02`，但 `promotion_status = shadow_only`。
- r36：risk-aware unified allocation 已完成代码接入与 3 轮 screening。入口为 `split_heads_portfolio_daily_risk_aware_unified_allocation_r36`、loss 为 `alpha_result_value_budget_split_v22`。r36c 已封住 unified source 绕过 `source_distribution_clean_pass` 的执行层旁路，并在 `64/48` screening 中达到 `training_evidence_status = sufficient`；但 `promotion_gate.status = shadow_only`，失败项为 `reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`，且真实 source 未来分布仍未过线。
- r37：decision-focused allocation 已完成代码接入与两轮 `64/48` screening。入口为 `split_heads_portfolio_daily_decision_focused_allocation_r37`、loss 为 `alpha_result_value_budget_split_v23`。r37b 已确认推理侧 unified allocation 会消费预测 source penalty heads，训练 diagnostics 为 yolos + CUDA 且 `training_evidence_status = sufficient`；但 `promotion_gate.status = shadow_only`，失败项仍为 `open_win_rate_5d`、`reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`，且 `source_positive_forward_sell_share = 0.833333`、`source_strong_positive_forward_sell_count = 2`、`receiver_minus_source_forward_excess_5d = -0.057937`，说明 source hard-negative 信号已接通但学习强度仍不足。
- r38：source hard-negative regret 已完成代码接入与 bounded screening + fresh confirm。入口为 `split_heads_portfolio_daily_source_hard_negative_regret_r38`、loss 为 `alpha_result_value_budget_split_v24`。本轮 diagnostics 为 yolos + CUDA，`completed_epochs = 64`，`training_evidence_status = sufficient`；confirm 中 `source_positive_forward_sell_share = 0.0`、`source_strong_positive_forward_sell_count = 0`、`receiver_minus_source_forward_excess_5d = 0.037347`、`source_realized_sell_rate = 1.0`、`receiver_unrealized_deploy_share = 0.0`、`cash_reserve_rate = 0.981728`，但 `promotion_gate.status = shadow_only`，失败项为 `reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`，且 `annual_return = 0.009111`、`sharpe = 0.164855`、`source_target_count = 1`，说明 source 防错已过强而收益/广度/时机不足。
- r39：allocation objective consolidation 已完成代码接入、两轮 bounded screening + fresh confirm。入口为 `split_heads_portfolio_daily_allocation_objective_consolidation_r39`、loss 为 `alpha_result_value_budget_split_v25`。第二轮 `execblend` 在预测阶段把 final objective 回写 receiver/source/cash core scores；confirm 为 `training_evidence_status = sufficient`、`annual_return = 2.676289`、`sharpe = 3.802282`、`monthly_return_mean = 0.115004`、`receiver_target_count = 6`、`source_target_count = 1`、`receiver_unrealized_deploy_share = 0.0`、`source_realized_sell_rate = 1.0`、`source_positive_forward_sell_share = 0.0`、`receiver_minus_source_forward_excess_5d = 0.066883`，但 `promotion_status = shadow_only`，v2 gate 只过 `9/12`，失败项为 `exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`，stable confirm 失败于 `confirm_gate_pass` 与 `confirm_source_count_floor`。
- 过程细节与完整实验复盘以 `daily_research/brain/episodic_memory.md` 为准。

## 历史归档入口
- 早期状态原文：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 早期标题索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- continuous_policy 设计合同历史：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- episodic 历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 读取纪律：当前状态以本文件上方章节为准；归档文件只作为历史证据与追溯入口。
