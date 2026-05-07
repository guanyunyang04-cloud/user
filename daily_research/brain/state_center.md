# Daily Research 状态中枢

快照日期：`2026-05-07`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- active 执行物化真源为 `daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 为 `regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前执行权重语义固定为 `research_raw_target_weight`，权重上限语义固定为 `follow_research_raw_no_global_cap`。
- continuous_policy 最新有效研究状态仍为 r39 allocation objective consolidation bounded confirm 之后的 `research / shadow_only`；r40 end-to-end allocation layer clean rerun `self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504` 已自然完成 `3` 个 screening 与 `2` 个 confirmatory，运行通道和持久产物可用，但 `stable_confirmatory_count = 0`，两个 confirmatory 均为 `shadow_only` 且失败于 `cash_timing_quality_1d` / `max_drawdown` 等 gate，因此不能替代 r39 证据基线。r31 receiver 语义闭包、r33 source clean-pass、r34 allocation breadth、r35 unified allocation、r36 risk-aware unified allocation、r37 decision-focused allocation、r38 source hard-negative regret 与 r39 allocation objective consolidation 仍保留；未过 v2 gate / stable confirm 前不得 promotion / live。
- r41 risk-sensitive allocation layer 已完成代码合约入口：新增 uncertainty pressure、tail risk control 与 decision-focused objective 三个 allocation target / heads / loss / predict exports / solver brakes / search profile；当前只代表实现与合同测试通过，尚无训练 verdict，不能替代 r39 证据基线。
- r42 utility-credit allocation 已完成大改入口：新增 net utility、credit closure、resource efficiency 三个目标、v27 loss、模型 heads、solver utility relief / budget multiplier 与 screening resource gate；它取代 r41 成为下一优先 research profile，目标是先用短 screening 和资源闸门排除低信息长训，再决定是否进入 confirmatory。
- r43 primal-dual decision allocation 已完成进一步大改入口：事实是新增 `alpha_result_value_budget_split_v28`、`_portfolio_primal_dual_decision_loss`、r41/r42 target 的 `sample_targets` 显式接线、artifact support flags、`split_heads_portfolio_daily_primal_dual_decision_allocation_r43` 与 dry-run；推断是 r42 仍受“target heads + 半可微 solver + 事后 resource gate”的旧框架限制，r43 把最终日频 receiver/source/cash 决策 regret、false-source、dead-cash 与 risk-cash 防守推进训练目标，取代 r42 成为下一优先 research profile。
- r44 entropic transport allocation 已完成当前更到位的大改入口：事实是新增 `alpha_result_value_budget_split_v29`、`_portfolio_entropic_transport_decision_loss`、Sinkhorn 风格可微资金运输计划、`split_heads_portfolio_daily_entropic_transport_allocation_r44` 与 dry-run；推断是 r43 仍把 receiver/source/cash regret 分开计算，r44 才把 source-to-receiver/cash 资金流放进同一个可反传运输矩阵，取代 r43 成为下一优先 research profile。
- r45 conservative transport allocation 已完成进一步大改入口：事实是新增 `alpha_result_value_budget_split_v30`、`_portfolio_offline_conservative_support_loss`、CQL/OPE 启发的 offline support / OOD action 保守损失、`split_heads_portfolio_daily_conservative_transport_allocation_r45` 与 dry-run；推断是 r44 虽已有 transport surrogate，但仍可能在离线数据支持不足的 receiver/source 动作上过度自信，r45 取代 r44 成为下一优先 research profile。
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
- r40 clean rerun 证明 end-to-end allocation layer 运行通道已可完整产出 study/protocol/model/ranking 证据：`completed_trial_count = 3`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`、模型诊断为 yolos + CUDA；但策略层失败清晰，stable confirm 为空，confirm_01 为 `annual_return = 0.328016`、`sharpe = 1.079482`、`max_drawdown = -0.165923`，confirm_02 为 `annual_return = 0.688870`、`sharpe = 2.448261`、`max_drawdown = -0.106384`，两者均 `source_target_count = 0`、`source_realized_sell_rate = 0`、`cash_timing_quality_1d ≈ -0.114` 且 `promotion_status = shadow_only`。
- 当前 r31 profile 标记：`split_heads_portfolio_daily_receiver_semantic_closure_r31`；核心审计字段为 `direct_action_authorization_subset_violation_count`、`authorized_add_no_weight_change_share`、`deploy_intent_unrealized_share`。
- 当前 r33 profile 标记：`split_heads_portfolio_daily_source_forward_proxy_r33`；当前 r34 profile 标记：`split_heads_portfolio_daily_allocation_breadth_r34`；当前 r35 profile 标记：`split_heads_portfolio_daily_unified_allocation_r35`；当前 r36 profile 标记：`split_heads_portfolio_daily_risk_aware_unified_allocation_r36`；当前 r37 profile 标记：`split_heads_portfolio_daily_decision_focused_allocation_r37`；当前 r38 profile 标记：`split_heads_portfolio_daily_source_hard_negative_regret_r38`；当前 r39 profile 标记：`split_heads_portfolio_daily_allocation_objective_consolidation_r39`；当前 r41 profile 标记：`split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41`；当前 r42 profile 标记：`split_heads_portfolio_daily_utility_credit_allocation_r42`；当前 r43 profile 标记：`split_heads_portfolio_daily_primal_dual_decision_allocation_r43`；当前 r44 profile 标记：`split_heads_portfolio_daily_entropic_transport_allocation_r44`；当前 r45 profile 标记：`split_heads_portfolio_daily_conservative_transport_allocation_r45`。
- 当前有效证据基线仍为 r39 的 `alpha_result_value_budget_split_v25` 与 `portfolio_daily_ranking_v2_gated`；r45 新实现标记为 `alpha_result_value_budget_split_v30`、`end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与 `end_to_end_allocation_layer_v1`，但尚未运行完整 study。保留 scoring / objective 锚点为 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth`、`portfolio_daily_joint_economic_quality_gate`、`portfolio_daily_unified_allocation_objective`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_receiver_source_spread_reward`；r38 标记为 `portfolio_daily_source_tail_false_sell_penalty`、`portfolio_daily_source_release_preference`、`portfolio_daily_transfer_regret_target`、`portfolio_daily_source_hard_negative_prevalence` 与 `portfolio_daily_source_hard_negative_selected_pressure`；r39 标记为 `portfolio_daily_allocation_trade_quality_target`、`portfolio_daily_allocation_cash_deployment_target`、`portfolio_daily_allocation_risk_adjusted_return_target`、`portfolio_daily_allocation_drawdown_control_target`、`portfolio_daily_allocation_monthly_quality_target` 与 `portfolio_daily_allocation_final_objective`；r41 标记为 `portfolio_daily_allocation_uncertainty_pressure_target`、`portfolio_daily_allocation_tail_risk_control_target`、`portfolio_daily_allocation_decision_focused_objective` 与 `portfolio_risk_sensitive_allocation_total`；r42 标记为 `portfolio_daily_allocation_net_utility_target`、`portfolio_daily_allocation_credit_closure_target`、`portfolio_daily_allocation_resource_efficiency_target` 与 `portfolio_utility_credit_closure_total`；r43 标记为 `portfolio_primal_dual_decision_total`、`supports_portfolio_primal_dual_decision_loss` 与 `val_portfolio_primal_dual_decision_loss`；r44 标记为 `portfolio_entropic_transport_decision_total`、`supports_portfolio_entropic_transport_decision_loss` 与 Sinkhorn transport plan；r45 标记为 `portfolio_offline_conservative_support_total`、`supports_portfolio_offline_conservative_support_loss` 与 offline conservative support / OOD action penalty。

## 当前优先级
- P0：保持 live / active artifact 冻结，只在 research / shadow 范围推进。
- P1：维护主分脑入口精炼，避免 `state_center.md` 和 `operations_center.md` 继续变成长日志。
- P2：继续把 receiver 可执行性、source 分布质量、monthly return、exposure utilization、realized deploy、cash timing 与 drawdown 写入 objective / feedback / gate。
- P3：下一轮研究不再优先加单边 source false-sell 或 cash penalty；r39 已说明 final objective 接入后可以恢复收益和正 spread，但仍 source count 过窄、现金过高、cash timing 与 drawdown 不稳。下一优先级应在不放松 source hard-negative 的前提下，提高 clean source breadth 和可卖源分布，并把 cash timing / drawdown 的日级触发从后验 gate 推进到 allocation layer。
- P3a：r41 已把 P3 技术路线落成可测代码入口；后续若进入长训练，必须先做无冲突进程、无同名 tag、dry-run / contract gate 预检，再以前台持久日志和进度文件方式运行，不能把代码测试通过写成策略有效。
- P3b：r42 已把“避免 r41 长训低收益”写入代码主线；后续优先跑 r42 短 screening + resource gate，不优先启动 r41 full clean study。
- P3c：r43 已把“r42 仍偏 target-head / 事后 gate”的结构缺口继续前移到 primal-dual decision loss；后续优先跑 r43 短 screening + resource gate，不再优先跑 r42 或 r41 长训，除非 r43 dry-run / contract / target wiring 失败。
- P3d：r44 已把“r43 仍缺少同一资金运输矩阵”的结构缺口推进到 entropic transport loss；后续优先跑 r44 短 screening + resource gate，不再优先跑 r43/r42/r41 长训，除非 r44 dry-run / contract 失败。
- P3e：r45 已把“r44 仍缺少离线分布支持/保守价值估计”的结构缺口推进到 offline conservative support loss；后续优先跑 r45 短 screening + resource gate，不再优先跑 r44/r43/r42/r41 长训，除非 r45 dry-run / contract 失败。
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
- r40：end-to-end allocation layer clean rerun `self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504` 已完整结束并可读取最终结果；screening 高收益分支因 `training_evidence_status = insufficient` 不可作为 verdict，confirmatory 分支训练证据充分但 stable confirm 为空。主要瓶颈不是前台运行或 GPU/yolos，而是 allocation layer 下 source 释放/资金来源仍休眠、cash timing 为负、drawdown 未过线、order translation drift 仍存在。
- r41：risk-sensitive allocation layer 已完成代码入口和合同测试。入口为 `split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41`、loss 为 `alpha_result_value_budget_split_v26`，目标是把 uncertainty pressure、tail risk control、decision-focused objective、receiver risk brake 与 cash defense 内生到 allocation layer；当前没有 study / protocol / ranking 证据，因此只可作为保留 research profile。
- r42：utility-credit allocation 已完成代码入口和合同测试。入口为 `split_heads_portfolio_daily_utility_credit_allocation_r42`、loss 为 `alpha_result_value_budget_split_v27`，目标是把 net utility、credit closure 与 resource efficiency 写入 allocation layer，并在 screening 后用 resource gate 阻断 source dead / cash timing bad / drawdown bad 的低信息长训；当前只完成 dry-run 产物，不是策略 verdict。
- r43：primal-dual decision allocation 已完成代码入口、合同测试和 dry-run。入口为 `split_heads_portfolio_daily_primal_dual_decision_allocation_r43`、loss 为 `alpha_result_value_budget_split_v28`，目标是让日频 receiver/source/cash 的最终 soft allocation regret、tail false-source、dead cash、risk cash under-defense 与 funding imbalance 进入训练/验证损失；当前只完成 dry-run 产物，不是策略 verdict。
- r44：entropic transport allocation 已完成代码入口、合同测试和 dry-run。入口为 `split_heads_portfolio_daily_entropic_transport_allocation_r44`、loss 为 `alpha_result_value_budget_split_v29`，目标是让 source-to-receiver/cash 的资金运输矩阵、边际匹配、运输 regret、false-source flow、dead cash 与 risk cash under-defense 进入训练/验证损失；当前只完成 dry-run 产物，不是策略 verdict。
- r45：conservative transport allocation 已完成代码入口、合同测试和 dry-run。入口为 `split_heads_portfolio_daily_conservative_transport_allocation_r45`、loss 为 `alpha_result_value_budget_split_v30`，目标是在 r44 transport 基础上加入 offline support / OOD action 保守损失，压制非 executable receiver、非 held/source、强 positive-forward false source 与高风险低 cash 的过度自信；当前只完成 dry-run 产物，不是策略 verdict。
- 过程细节与完整实验复盘以 `daily_research/brain/episodic_memory.md` 为准。

## 历史归档入口
- 早期状态原文：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 早期标题索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- continuous_policy 设计合同历史：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- episodic 历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 读取纪律：当前状态以本文件上方章节为准；归档文件只作为历史证据与追溯入口。

## 2026-05-02 主线停环审计
- 已生成并采纳 `daily_research/output/continuous_policy/analysis/cycle_audits/continuous_policy_cycle_audit_20260502.md`。结论：continuous_policy 自 r19 以来不是完全原地踏步，但 r34-r39 已明确进入同一组矛盾的局部补丁循环。
- 当前停止“r39 后继续追加局部 source/cash/reduce penalty 或 guard”的旧路线；下一主线必须以 `end-to-end allocation layer` 为目标，而不是继续包装旧 action-head / simulator-guard 路径。
- r20-r23、r31、r33 只作为最后安全边界冻结；`portfolio_daily_ranking_v2_gated`、`action_budget_split_v1`、`cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15` 若在 r40+ 继续作为主路径，必须先给出明确架构差异和退出旧路径的证据。
- 当前真正瓶颈：source / receiver / cash 的 credit assignment 尚未在同一个可优化 allocation objective / allocation layer 内闭合；simulator guard 不能再承担主策略逻辑。
## 2026-05-02 r40 end-to-end allocation layer 入口落地
- 已新增 `split_heads_portfolio_daily_end_to_end_allocation_layer_r40`，默认 objective 为 `end_to_end_allocation_layer_v1`，预算语义为 `allocation_layer_v1`，校准为 `end_to_end_allocation_layer_v1`；它不再沿用 `action_budget_split_v1` + v15 receiver-exec guard 作为主路径。
- `portfolio_simulator.py` 已新增 allocation-layer 主模式：在 r40 语义下由 `solve_semidifferentiable_allocation` 直接输出目标权重；direct action 信号降级为非主导辅助，不再伪造 receiver/source 主目标。
- `allocation_optimizer.py` 已要求 `portfolio_daily_receiver_executable_candidate` 与 `portfolio_daily_source_executable_candidate` 作为硬候选掩码；raw score 或 action label 不能绕过 executable candidate。
- r40 当前已完成一次 bounded study，但不是有效训练 verdict；本轮暴露的是超长前台任务 stdout 失效导致的 protocol 阶段异常，而不是 allocation layer 训练有效性通过。因此仍为 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。
## 2026-05-04 r40 bounded study 结果与运行通道修复
- 已前台自然结束 `self_opt_study_r40_end_to_end_allocation_layer_20260502`；进程没有陷入循环，PID 自然消失，最终生成 `study_summary.json`。
- 本轮 screening 只有 `trial_01` 完整完成，`trial_02`、`trial_03` 与 `confirm_01` 在训练或评估产物写出后报 `[Errno 22] Invalid argument`。根因是外层 10h shell 捕获窗口超时返回后，Python 主进程继续运行但 stdout 管道失效，后续 stage 的 JSON 打印把已完成 stage 误标为失败。
- 已加固 continuous_policy 输出路径：train / evaluate / export / protocol / behavior audit / conclusion ledger / self-optimizing study 的末尾 JSON 打印改为 `safe_print_json`；stdout 失效只影响控制台显示，不得再使已持久化的 stage 失败。`progress.py` 同步降级失效 stdout。
- r40 本轮唯一可评估 champion 为 screening `trial_01`：`annual_return = 0.064530`、`sharpe = 0.354676`、`max_drawdown = -0.196677`、`monthly_return_mean = 0.006319`、`training_evidence_status = insufficient`、v2 gate `8/12`，失败项为 `training_evidence_sufficient`、`reduce_success_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`。
- 结论：r40 架构入口成立，但本轮 bounded study 不能用于策略判定；下一次正式 r40 confirm 必须在前台使用持久 stdout/stderr 日志，并按 2h 轮询，不让外层捕获窗口关闭污染主进程。

## 2026-05-07 r40 clean rerun 结果与当前判定
- 已完成并解析 `self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504`：`executed_at = 2026-05-07T04:54:27+08:00`，`completed_trial_count = 3`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`、`portfolio_daily_v2_stable_confirmatory_trials = []`。
- 运行通道判定：原 PID 自然退出，`study_summary.json`、`trial_ranking.csv`、protocol summary 与模型 artifact 均可读取；`foreground.log` 停在旧 stdout 管道失效时间不影响最终持久产物。该 clean_r1 进程启动早于 `study_progress.json/jsonl` 补丁，所以缺少进度文件是预期现象。
- 策略判定：screening `trial_02` 与 `trial_03` 年化收益约 `0.799`、Sharpe 约 `2.34`，但 `training_evidence_status = insufficient`；confirm_01 / confirm_02 为 `training_evidence_status = sufficient`，但均为 `shadow_only`，stable confirm 为空，失败项集中在 `cash_timing_quality_1d`、`max_drawdown`，confirm_01 另失败 `exit_timeliness_rate_5d`。
- 结构诊断：r40 证明新运行通道可用，但没有证明策略强于 r39。confirm_02 虽有 `annual_return = 0.688870`、`sharpe = 2.448261`、`receiver_target_count = 408`、`receiver_realized_deploy_rate = 1.0`，仍有 `source_target_count = 0`、`source_realized_sell_rate = 0`、`cash_timing_quality_1d = -0.114171`、`max_drawdown = -0.106384`、`order_translation_conflict_rate = 0.907379`，说明 source/receiver/cash credit assignment 尚未在 allocation layer 内稳定闭合。
- 决策：r40 clean_r1 继续为 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact；当前有效证据基线仍是 r39。
