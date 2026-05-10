# Daily Research 操作中枢

快照日期：`2026-05-09`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录进入 `daily_research/brain/references/` 或 `episodic_memory.md`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，不得依赖当前 shell Python。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；当前 `yolos` 已完成 OpenMP 原地修复，根治验收命令为 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/openmp_runtime_check.py --strict`，并且不得依赖 `KMP_DUPLICATE_LIB_OK`。
- 脑内文档标题、正文、状态、规则和复盘必须使用简体中文；命令、路径、指标名、tag、模型名等技术标识保留原文。
- 当前 active 执行权重语义固定为 `research_raw_target_weight`，权重上限语义固定为 `follow_research_raw_no_global_cap`。
- 所有训练、评估、审计、bounded study、confirmatory rerun、execution app 任务与交易计划任务默认前台运行，主进程不得后台化规避窗口，也不得中途人为中断。
- 所有项目前台窗口时限统一按 `10` 小时处理；若外层工具有更短硬上限，只允许接续等待或读取自然完成产物，不改变任务本体。
- 对可能超过外层捕获窗口的训练、评估、bounded study 与 confirmatory rerun，仍必须前台运行，但 stdout/stderr 必须同步写入持久日志文件；监控轮询间隔固定为 `2` 小时，若进程提前自然结束则立即解析产物。
- GPU 训练完成后必须核验 `training_diagnostics.json` 中 `device = cuda`、`cuda_available = true`，并确认 `python_executable` 指向 yolos 后再写入正式证据。

## 项目地图
- 当前状态与边界：`daily_research/brain/state_center.md`。
- 稳定事实、规则和教训：`daily_research/brain/knowledge_center.md`。
- continuous_policy 设计合同：`daily_research/brain/continuous_policy_design_contract.md`。
- 过程复盘：`daily_research/brain/episodic_memory.md`。
- 执行与交易计划：`daily_research/execution`。
- 连续策略研究：`daily_research/continuous_policy`。
- 产物、评估、协议、审计：`daily_research/output`。

## 高频命令
- 主脑接管：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_bootstrap.py --child daily_research --json`
- 守卫检查：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/project_consistency_check.py`
- 默认交易计划：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_trade_plan.py --candidate-profile active_execution_strategy --help`
- execution app：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_execution_app.py status`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_execution_app.py tasks`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_execution_app.py doctor`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_execution_app.py web --port 8765`
- continuous_policy：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_continuous_policy_protocol --help`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --help`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/analyze_behavior_gap.py --help`

## 当前 continuous_policy 操作口径
- continuous_policy 当前仍是 `research / shadow_only`，不能切换 live 或 promotion；正式 live 默认执行链仍以 `active_execution_strategy.json` 为真源。
- r31 receiver semantic closure 入口：`split_heads_portfolio_daily_receiver_semantic_closure_r31`，重点读取 `direct_action_authorization_subset_violation_count`、`authorized_add_no_weight_change_share`、`deploy_intent_unrealized_share`、`portfolio_daily_receiver_unrealized_deploy_share`。
- r33 source forward proxy / clean-pass 入口：`split_heads_portfolio_daily_source_forward_proxy_r33`，重点读取 `portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction`、`portfolio_daily_source_distribution_clean_pass`、`portfolio_daily_source_positive_forward_sell_share`、`portfolio_daily_source_strong_positive_forward_sell_count`。
- r34 allocation breadth 入口：`split_heads_portfolio_daily_allocation_breadth_r34`，重点读取 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth`、`portfolio_daily_joint_economic_quality_gate` 与 `allocation_teacher_summary_mean`。
- r35 unified allocation 入口：`split_heads_portfolio_daily_unified_allocation_r35`，重点读取 `unified_allocation_summary_mean`、`portfolio_daily_unified_allocation_objective`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_receiver_source_spread_reward`、`portfolio_daily_unified_constraint_violations` 与 `supports_portfolio_unified_allocation_heads`。
- r36 risk-aware unified allocation 入口：`split_heads_portfolio_daily_risk_aware_unified_allocation_r36`，重点读取 `supports_portfolio_unified_allocation_consistency_loss`、`portfolio_daily_unified_cash_score`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_source_distribution_clean_pass`、`portfolio_daily_source_positive_forward_sell_share`、`portfolio_daily_source_strong_positive_forward_sell_count`、`cash_timing_quality_1d` 与 `max_drawdown`。
- r37 decision-focused allocation 入口：`split_heads_portfolio_daily_decision_focused_allocation_r37`，重点读取 `supports_portfolio_decision_focused_allocation_loss`、`portfolio_daily_source_hard_negative_penalty`、`portfolio_daily_source_strong_false_sell_penalty`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_receiver_source_spread_reward`、`source_positive_forward_sell_share`、`source_strong_positive_forward_sell_count`、`cash_timing_quality_1d` 与 `max_drawdown`。
- r38 source hard-negative regret 入口：`split_heads_portfolio_daily_source_hard_negative_regret_r38`，重点读取 `supports_portfolio_source_hard_negative_tail_loss`、`supports_portfolio_source_listwise_release_loss`、`supports_portfolio_transfer_regret_loss`、`portfolio_daily_source_tail_false_sell_penalty`、`portfolio_daily_source_release_preference`、`portfolio_daily_transfer_regret_target`、`portfolio_daily_source_hard_negative_prevalence`、`portfolio_daily_source_hard_negative_selected_pressure`、`source_positive_forward_sell_share`、`receiver_minus_source_forward_excess_5d`、`cash_timing_quality_1d` 与 `max_drawdown`。
- r39 allocation objective consolidation 入口：`split_heads_portfolio_daily_allocation_objective_consolidation_r39`，重点读取 `supports_portfolio_allocation_objective_consolidation_heads`、`supports_portfolio_allocation_objective_consolidation_loss`、`portfolio_daily_allocation_trade_quality_target`、`portfolio_daily_allocation_cash_deployment_target`、`portfolio_daily_allocation_risk_adjusted_return_target`、`portfolio_daily_allocation_drawdown_control_target`、`portfolio_daily_allocation_monthly_quality_target`、`portfolio_daily_allocation_final_objective`、`source_target_count`、`receiver_target_count`、`cash_timing_quality_1d`、`max_drawdown` 与 `portfolio_daily_v2_confirm_stability_checks`。
- r40 end-to-end allocation layer 入口：`split_heads_portfolio_daily_end_to_end_allocation_layer_r40`，重点读取 `allocation_layer_primary_mode`、`allocation_layer_receiver_target_count`、`allocation_layer_source_target_count`、`budget_semantics = allocation_layer_v1`、`budget_calibration = end_to_end_allocation_layer_v1`、硬 executable candidate 掩码、stdout/stderr 持久日志与 confirm stability。
- r41 risk-sensitive allocation layer 入口：`split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41`，重点读取 `supports_portfolio_risk_sensitive_allocation_heads`、`supports_portfolio_risk_sensitive_allocation_loss`、`portfolio_daily_allocation_uncertainty_pressure_target`、`portfolio_daily_allocation_tail_risk_control_target`、`portfolio_daily_allocation_decision_focused_objective`、receiver risk brake、cash defense、source release、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r42 utility-credit allocation 入口：`split_heads_portfolio_daily_utility_credit_allocation_r42`，重点读取 `supports_portfolio_utility_credit_closure_heads`、`supports_portfolio_utility_credit_closure_loss`、`portfolio_daily_allocation_net_utility_target`、`portfolio_daily_allocation_credit_closure_target`、`portfolio_daily_allocation_resource_efficiency_target`、solver utility relief、utility budget multiplier、`resource_gate`、`source_target_count`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r43 primal-dual decision allocation 入口：`split_heads_portfolio_daily_primal_dual_decision_allocation_r43`，重点读取 `supports_portfolio_primal_dual_decision_loss`、`portfolio_primal_dual_decision_total`、`val_portfolio_primal_dual_decision_loss`、r41/r42 target wiring、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`cash_timing_quality_1d`、`max_drawdown`、tail false-source 与 confirm stability。
- r44 entropic transport allocation 入口：`split_heads_portfolio_daily_entropic_transport_allocation_r44`，重点读取 `supports_portfolio_entropic_transport_decision_loss`、`portfolio_entropic_transport_decision_total`、transport marginal balance、false-source flow、dead cash、risk cash under-defense、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r45 conservative transport allocation 入口：`split_heads_portfolio_daily_conservative_transport_allocation_r45`，重点读取 `supports_portfolio_offline_conservative_support_loss`、`portfolio_offline_conservative_support_total`、offline support / OOD action penalty、executable receiver support、held source support、false-source pressure、risk cash defense、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r46 differentiable convex allocation 入口：`split_heads_portfolio_daily_differentiable_convex_allocation_r46`，重点读取 `supports_portfolio_differentiable_convex_allocation_loss`、`supports_portfolio_differentiable_convex_allocation_diagnostics`、`supports_portfolio_path_risk_loss`、`portfolio_differentiable_convex_allocation_total`、legacy action/duration loss 是否为 0、`portfolio_differentiable_convex_allocation_terms` 内的 KKT / constraint residual、unsupported mass、false-source mass、cash timing、source/receiver shortfall、behavior support、conservative OPE、path risk、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r47 true convex solver allocation 入口：`split_heads_portfolio_daily_true_convex_solver_allocation_r47`，重点读取 `supports_portfolio_cvxpy_convex_allocation_layer`、`portfolio_cvxpy_convex_layer_status`、`supports_portfolio_cvxpy_convex_allocation_diagnostics`、`portfolio_cvxpy_convex_allocation_total`、`portfolio_cvxpy_convex_allocation_terms` 内的 solver regret、solution tracking、gross / turnover / position residual、unsupported mass、false-source mass、cash timing、path risk、solver success rate、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r48 full-universe convex OPE allocation 入口：`split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48`，重点读取 `supports_portfolio_full_universe_convex_allocation_loss`、`supports_portfolio_full_universe_candidate_coverage`、`supports_portfolio_full_universe_ope_diagnostics`、`portfolio_full_universe_convex_allocation_terms` 内的 candidate coverage、universe expansion、liquidity impact、concentration risk、OPE lower-bound、propensity support、doubly-robust gap、solver success rate、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r49 capital-flow closure 入口：`split_heads_portfolio_daily_capital_flow_closure_r49`，重点读取 `supports_portfolio_capital_flow_closure_loss`、`portfolio_capital_flow_closure_terms` 内的 receiver demand、funding shortfall、source dead、over-cash、cash defense、exposure gap、flow conservation、effective source/receiver flow、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`portfolio_daily_exposure_utilization`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r50 integrated convex capital-flow 入口：`split_heads_portfolio_daily_integrated_convex_capital_flow_r50`，重点读取 `supports_portfolio_cvxpy_convex_allocation_layer`、`supports_portfolio_full_universe_convex_allocation_loss`、`portfolio_full_universe_convex_train_solver_effective`、`portfolio_full_universe_convex_allocation_terms`、`portfolio_capital_flow_closure_terms`、`portfolio_cvxpy_convex_allocation_terms`、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`portfolio_daily_exposure_utilization`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。
- r51 native allocation vector 入口：`split_heads_portfolio_daily_native_allocation_vector_r51`，重点读取 `supports_portfolio_native_allocation_vector_heads`、`supports_portfolio_native_allocation_vector_loss`、`portfolio_native_allocation_vector_terms`、`portfolio_capital_flow_closure_terms`、`portfolio_daily_target_weight` / `portfolio_daily_target_delta` 推理字段、`allocation_layer_native_target_used`、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`portfolio_daily_exposure_utilization`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。r51 不启用 cvxpy/diffcp 训练主路径。
- r52 day-set native allocation vector 入口：`split_heads_portfolio_daily_day_set_native_allocation_vector_r52`，重点读取 `supports_portfolio_day_set_native_allocation_vector`、`portfolio_day_set_native_allocation_vector_terms`、`sample_model_type = temporal_day_set`、`day_set_full_day_integrity`、`day_set_batch_size`、`day_set_slot_count`、`portfolio_capital_flow_closure_terms`、`portfolio_daily_target_weight` / `portfolio_daily_target_delta` 推理字段、`allocation_layer_native_target_used`、`resource_gate`、`source_target_count`、`source_realized_sell_rate`、`portfolio_daily_exposure_utilization`、`cash_timing_quality_1d`、`max_drawdown` 与 confirm stability。r52 不启用 cvxpy/diffcp 训练主路径。
- r34 最新证据入口：`cp_v3_portfolio_daily_allocation_breadth_r34_bounded_20260430` 与 `cp_v3_portfolio_daily_allocation_breadth_r34_evidence_confirm_20260430`；读取 `portfolio_daily_v2_confirm_stability_checks`，重点看 `receiver_minus_source_forward_excess_5d`、`source_positive_forward_sell_share`、`source_strong_positive_forward_sell_count` 与 `training_evidence_status`。
- r35 已形成修复后正式 bounded confirm 证据；读取 `cp_v3_portfolio_daily_unified_allocation_r35_postfix4_bounded_confirm_20260430`，重点看 `confirm_02`、`portfolio_daily_v2_stable_confirmatory_trials`、`failed_checks`、source positive distribution、cash timing 与 drawdown。
- r37 screening 证据入口：`cp_v3_portfolio_daily_decision_focused_allocation_r37b_screening64_20260501`；读取 `trial_01`、`training_diagnostics.json`、`unified_allocation_summary_mean` 与 v2 gate report，重点看 predicted source penalty 是否进入 allocation path、source positive distribution、receiver-source spread、cash timing 与 drawdown。
- r38 最新 bounded confirm 证据入口：`self_opt_study_r38_source_hard_negative_regret_20260501`；读取 `study_summary.json`、`protocols/self_opt_study_r38_source_hard_negative_regret_20260501__confirm_01/protocol_summary.json` 与对应 `training_diagnostics.json`，重点看 source false-sell 是否被压住、source/receiver 广度是否足够、cash 是否过度保守、v2 gate failed checks 与 `portfolio_daily_v2_confirm_stability_checks`。
- r39 最新 bounded confirm 证据入口：`self_opt_study_r39_allocation_objective_consolidation_execblend_20260502`；读取 `study_summary.json`、`protocols/self_opt_study_r39_allocation_objective_consolidation_execblend_20260502__confirm_01/protocol_summary.json` 与对应 `training_diagnostics.json`，重点看 final objective 是否真正进入执行评分、source count 是否达标、cash timing / drawdown 是否仍失败、source false-sell 是否仍受控。
- 当前有效证据基线仍是 r39：`alpha_result_value_budget_split_v25` 与 `portfolio_daily_ranking_v2_gated`，对应 simulator calibration 为 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
- 当前已完成 r40 clean rerun：`self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504`，使用 `alpha_result_value_budget_split_v25`、`end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与 `end_to_end_allocation_layer_v1`，已完整生成 study/protocol/model/ranking 证据；但 stable confirm 为空，仍不得替代 r39 证据基线。
- r41-r47 已完成代码入口或 dry-run，但均为保留检查点，不是当前默认长训入口；r48 formal screening + confirmatory 已完成且 stable confirm 为空；r49 / r50 / r51 / r52 仍是 research profile，不是 production 默认。r52 当前是 day-set native allocation vector 主线入口，当前有效证据基线仍是 r39。
- 月度收益评价继续读取 `monthly_returns.csv` / `shadow_monthly_returns.csv`，重点看 `monthly_return_mean`、`monthly_win_rate`、`monthly_worst_return`、`monthly_max_consecutive_loss_months`、`monthly_consistency_score`。
- `analyze_behavior_gap.py` 会写 latest 行为摘要；多条审计必须顺序执行，不得并行抢写。

## 产物读取入口
- study 摘要：`daily_research/output/continuous_policy/studies/<study_tag>/study_summary.json`。
- protocol 摘要：`daily_research/output/continuous_policy/protocols/<run_tag>/protocol_summary.json`。
- 模型诊断：`daily_research/output/continuous_policy/models/<run_tag>__train/training_diagnostics.json`。
- evaluation 摘要：`daily_research/output/continuous_policy/evaluations/<eval_tag>/evaluation_summary.json`。
- behavior audit：`daily_research/output/continuous_policy/analysis/behavior_audits/`。
- v2 gate 报告：`daily_research/output/continuous_policy/analysis/portfolio_daily_ranking_v2_gate_report.*` 或 study 内对应报告。

## 写回路由
- 当前状态、优先级、边界：`state_center.md`。
- 稳定事实、规则、术语：`knowledge_center.md`。
- 新命令口径、环境和流程：`operations_center.md`。
- 设计边界和成功判定：`continuous_policy_design_contract.md`。
- 过程证据、动作后复盘：`episodic_memory.md`。
- 大段历史原文、标题索引和归档说明：`daily_research/brain/references/`。

## 历史归档入口
- 操作中枢早期原文：`daily_research/brain/references/operations_center_history_raw_20260424.md`。
- 操作中枢早期标题索引：`daily_research/brain/references/operations_center_evidence_index_20260424.md`。
- 归档规则：`daily_research/brain/references/archive_rules.md`。
- 读取纪律：当前操作以本文件上方章节为准；旧命令仅作为复现和审计证据，不自动提升为当前推荐命令。

## 2026-05-02 主线循环审计命令
- 审计命令：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/continuous_policy_cycle_audit.py --write --json`
- 审计产物：`daily_research/output/continuous_policy/analysis/cycle_audits/continuous_policy_cycle_audit_20260502.md` 与 `.json`。
- 后续 r40+ 研究立项前必须先读该审计；若新 profile 仍沿用 `action_budget_split_v1`、v15 simulator calibration 和 `portfolio_daily_ranking_v2_gated` 作为主路径，需要先证明不是旧局部补丁循环。

## 2026-05-02 r40 end-to-end allocation layer 操作入口
- r40 profile：`split_heads_portfolio_daily_end_to_end_allocation_layer_r40`。
- r40 objective：`end_to_end_allocation_layer_v1`。
- r40 budget semantics：`allocation_layer_v1`。
- r40 budget calibration：`end_to_end_allocation_layer_v1`。
- r40 合同测试命令：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r40_end_to_end_allocation_layer_profile_exits_action_budget_path daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_unified_allocation_problem_respects_hard_executable_candidate_masks daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_end_to_end_allocation_layer_step_uses_optimizer_targets_without_direct_action`。
- 若涉及 PyTorch / MKL / OpenMP 导入冲突，先运行 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/openmp_runtime_check.py --strict` 定位；当前 yolos 已完成原地修复，前台命令不得默认设置 `KMP_DUPLICATE_LIB_OK`。
- r40 当前只允许作为 research / shadow 入口；`self_opt_study_r40_end_to_end_allocation_layer_20260502` 已自然结束但因外层 stdout 管道失效污染后续 trial/protocol 状态，不能作为有效 bounded verdict，不得 promotion、不得 live、不得写 active artifact。
- r40 或更长 study 的前台运行建议命令形态：
  - `$tag='<tag>'; New-Item -ItemType Directory -Force -Path "daily_research/output/continuous_policy/studies/$tag" | Out-Null; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_end_to_end_allocation_layer_r40 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag $tag *> "daily_research/output/continuous_policy/studies/$tag/foreground.log"`
  - 该命令仍为前台运行；持久日志用于避免外层捕获窗口超时后 stdout 失效导致 `[Errno 22] Invalid argument`。
  - 2026-05-05 后续长训练默认仍可由 Codex 在本机前台执行；用户另行指定时才改为本人执行。不得使用自动化重复监控；由 Codex 执行时采用 `2` 小时轮询，并优先读取 `study_progress.json` / `study_progress.jsonl`。
  - `run_self_optimizing_study` 已新增持久进度入口：`study_progress.json` 保存最新状态，`study_progress.jsonl` 保存事件流；每个 protocol 默认 5 分钟写一次 `protocol_heartbeat`，最终 `study_summary.json` 会记录这两个路径。
  - 进度读取优先级：先看 `daily_research/output/continuous_policy/studies/<study_tag>/study_progress.json`，再看同目录 `study_progress.jsonl` 和 `foreground.log`；最终结论仍以 `study_summary.json`、`trial_ranking.csv` 与 protocol/model JSON 为准。

## 2026-05-07 r40 clean rerun 产物读取口径
- 已完成 study tag：`self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504`。
- 优先读取：
  - `daily_research/output/continuous_policy/studies/self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504/study_summary.json`
  - `daily_research/output/continuous_policy/studies/self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504/trial_ranking.csv`
  - `daily_research/output/continuous_policy/protocols/self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504__<trial_or_confirm>/protocol_summary.json`
  - `daily_research/output/continuous_policy/models/self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504__<trial_or_confirm>__train/training_diagnostics.json`
- 本次 clean_r1 启动早于进度文件补丁，因此 `study_progress.json` / `study_progress.jsonl` 不存在属于预期；后续新 run 才应以进度文件作为轮询主入口。
- 本次 `foreground.log` 仍停在旧 stdout 管道失效时间，不作为最终成败判断；最终判断以 `study_summary.json`、`trial_ranking.csv`、protocol summary 与模型 diagnostics 为准。
- clean_r1 结论只允许写作 `research / shadow_only`：`completed_trial_count = 3`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`，但 stable confirm 为空，confirm_01/confirm_02 均未过 gate。

## 2026-05-07 r41-r47 保留检查点
- r41-r47 已分别覆盖 risk-sensitive、utility-credit、primal-dual decision、entropic transport、conservative transport、differentiable convex allocation 与 true convex solver allocation；当前不作为默认长训入口。若必须回溯验证，只能先 dry-run / contract gate，再以前台持久日志短 screening 执行；任何代码合同、dry-run 或单测通过都不得写成策略有效、promotion、live 或 active artifact 切换依据。

## 2026-05-07 r45-r47 保留检查点索引
- r45-r47 分别对应 conservative transport、differentiable convex allocation 与 true convex solver allocation，当前只作为 research / shadow 保留检查点，不是默认长训入口。
- 若必须回溯验证，只能先跑对应 dry-run / contract gate，再以前台持久日志短 screening 执行；诊断重点仍是 source release、cash timing、drawdown、support conservatism、constraint / solver residual、path risk、receiver deploy 与经济信号是否同步改善。
- 任何代码合同、dry-run 或单测通过都不得写成策略有效、promotion、live 或 active artifact 切换依据；详细命令与历史复盘以 `continuous_policy_design_contract.md`、`episodic_memory.md` 和合同测试为准。

## 2026-05-08 r48 full-universe convex OPE allocation 操作入口
- r48 profile / loss：`split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48` / `alpha_result_value_budget_split_v33`。
- r48 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r48 默认 screening 资源口径：profile 内 `epochs = 8`、`min_epochs = 6`、`batch_size = 256`；命令行未显式传入 `--epochs` / `--min-epochs` 时不得覆盖该默认值。当前源码显式固定 full-universe solver 训练资源口径为 `CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT = 32`、`CVXPY_FULL_UNIVERSE_ALLOCATION_MAX_DAYS_PER_BATCH = 1`、`CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_BATCH_INTERVAL = 2`、`CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED = False`；最终 diagnostics 仍读取 solver terms，不得再把当前源码误读成 48-slot / 3-day 训练长跑口径。
- r48 诊断读取：训练 diagnostics 需优先读取 `portfolio_full_universe_convex_allocation_terms`，其中 `candidate_coverage_loss`、`universe_expansion_loss`、`liquidity_impact_loss`、`concentration_risk_loss`、`ope_lower_bound_loss`、`propensity_support_loss`、`doubly_robust_gap_loss`、`solver_success_rate`、`fallback_surrogate_loss` 与 `total` 是判断是否值得正式 screening 的关键前置信号。
- r48 dry-run 命令：`$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r48 合同测试命令：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r48_full_universe_profile_extends_r47_with_ope_and_larger_slot_bank daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r48_full_universe_candidate_coverage_penalizes_old_mask_blind_spots daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r48_full_universe_ope_and_realistic_cost_terms_respond_to_bad_support`。
- r48 formal run 已完成，tag：`self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3`。优先读取：
  - `daily_research/output/continuous_policy/studies/self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3/study_summary.json`
  - `daily_research/output/continuous_policy/studies/self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3/trial_ranking.csv`
  - `daily_research/output/continuous_policy/protocols/self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3__trial_01/protocol_summary.json`
  - `daily_research/output/continuous_policy/protocols/self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3__confirm_01/protocol_summary.json`
  - `daily_research/output/continuous_policy/models/self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3__confirm_01__train/training_diagnostics.json`
- r48 已知结果：`completed_trial_count = 1`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 1`、`portfolio_daily_v2_stable_confirmatory_trials = []`；confirm_01 为 `annual_return = 0.246043`、`sharpe = 2.728143`、`monthly_return_mean = 0.013886`、`max_drawdown = -0.037729`、`receiver_target_count = 20`、`receiver_realized_deploy_rate = 1.0`、`source_target_count = 0`、`source_realized_sell_rate = 0`、`portfolio_daily_exposure_utilization = 0.331039`、`training_evidence_status = insufficient`。失败项：`training_evidence_sufficient`、`reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`；stable confirm failed checks 为 `source_training_evidence_sufficient`、`confirm_training_evidence_sufficient`、`confirm_gate_pass`、`confirm_source_count_floor`、`confirm_exposure_utilization_floor`。
- r48 当前只允许作为 research / shadow 失败证据；不得 promotion、不得 live、不得改 active artifact。后续若继续研究，不能重复同一 tag 或直接延长训练当作唯一修复，必须先解决 source dead、exposure utilization 低和 training evidence edge。

## 2026-05-09 r49 capital-flow closure 操作入口
- r49 profile / loss：`split_heads_portfolio_daily_capital_flow_closure_r49` / `alpha_result_value_budget_split_v34`。
- r49 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r49 默认 screening 资源口径：profile 内 `epochs = 8`、`min_epochs = 6`、`batch_size = 256`；命令行未显式传入 `--epochs` / `--min-epochs` 时不得覆盖该默认值。
- r49 诊断读取：训练 diagnostics 需优先读取 `portfolio_capital_flow_closure_terms`，其中 `receiver_demand_loss`、`funding_shortfall_loss`、`source_dead_loss`、`false_source_loss`、`over_cash_loss`、`cash_defense_loss`、`cash_timing_loss`、`cash_coherence_loss`、`exposure_gap_loss`、`flow_conservation_loss`、`role_overlap_loss`、`source_breadth_loss`、`desired_receiver_flow_mean`、`effective_receiver_flow_mean`、`effective_source_flow_mean`、`risk_cash_need_mean`、`predicted_gross_mean`、`deploy_pressure_mean`、`risk_pressure_mean` 与 `total` 是判断 source/receiver/cash 是否真正闭合的关键前置信号。
- r49 dry-run 命令：`$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_capital_flow_closure_r49 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r49 合同测试命令：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r49_capital_flow_closure_profile_targets_source_receiver_cash_residual daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r49_capital_flow_loss_penalizes_receiver_deploy_without_source_or_cash_release daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r49_capital_flow_loss_keeps_false_source_protection daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r49_capital_flow_terms_separate_cash_defense_from_dead_cash daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r49_capital_flow_loss_requires_daily_cash_coherence daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r49_capital_flow_loss_blocks_role_overlap_and_narrow_source`。
- r49 若启动正式研究，必须先确认无冲突进程、无同名 tag，并以前台持久日志运行短 screening；若 `resource_gate.triggered = true`，不得进入 confirmatory。代码合同、dry-run 和单测通过不等于策略有效，不得 promotion、不得 live、不得改 active artifact。

## 2026-05-09 r50 integrated convex capital-flow 操作入口
- r50 profile / loss：`split_heads_portfolio_daily_integrated_convex_capital_flow_r50` / `alpha_result_value_budget_split_v35`。
- r50 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r50 默认 screening 资源口径：profile 内 `epochs = 6`、`min_epochs = 5`、`batch_size = 192`；命令行未显式传入 `--epochs` / `--min-epochs` 时不得覆盖该默认值。
- r50 solver 口径：全局 `CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED` 仍为 `False`；只有 `alpha_result_value_budget_split_v35` 通过 `CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_LOSS_PROFILES` 启用训练期 full-universe solver，并继续受 `slot_count = 32`、`max_days_per_batch = 1`、`train_batch_interval = 2` 约束。
- r50 诊断读取：训练 diagnostics 需同时读取 `portfolio_full_universe_convex_train_solver_effective`、`portfolio_full_universe_convex_allocation_terms`、`portfolio_capital_flow_closure_terms` 与 `portfolio_cvxpy_convex_allocation_terms`；若训练期 fallback，则 `solver_success_rate = 0` 且 `fallback_surrogate_loss > 0` 才是正确诊断，不得再写成 solver 成功。
- r50 dry-run 命令：`$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_integrated_convex_capital_flow_r50 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r50 合同测试命令：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r50_integrated_convex_capital_flow_profile_enables_real_solver_training daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r50_full_universe_fallback_terms_do_not_report_fake_solver_success`。
- r50 当前只有代码合同、合同测试与 dry-run；不得 promotion、不得 live、不得改 active artifact。若启动正式研究，仍必须先确认无冲突进程、无同名 tag，并以前台持久日志和进度文件运行短 screening。

## 2026-05-10 r51 native allocation vector 操作入口
- r51 profile / loss：`split_heads_portfolio_daily_native_allocation_vector_r51` / `alpha_result_value_budget_split_v36`。
- r51 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r51 默认 screening 资源口径：profile 内 `epochs = 8`、`min_epochs = 6`、`batch_size = 256`；命令行未显式传入 `--epochs` / `--min-epochs` 时不得覆盖该默认值。
- r51 solver 口径：不启用 `cvxpy` / `diffcp` 训练主路径，不加入 true-solver resource guard；`alpha_result_value_budget_split_v36` 中 `portfolio_cvxpy_convex_allocation_total = 0`、`portfolio_full_universe_convex_allocation_total = 0`，full-universe train solver effective 必须为 false。
- r51 诊断读取：训练 diagnostics 需优先读取 `supports_portfolio_native_allocation_vector_heads`、`supports_portfolio_native_allocation_vector_loss`、`portfolio_native_allocation_vector_terms` 与 `portfolio_capital_flow_closure_terms`；推理与 simulator 需读取 `portfolio_daily_target_weight`、`portfolio_daily_target_delta`、native receiver/source/cash score 和 `allocation_layer_native_target_used`。若 native target 缺失或违规，允许回退到现有 `allocation_layer_v1` safety path。
- r51 dry-run 命令：`$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_native_allocation_vector_r51 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r51 合同测试命令：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r51_native_allocation_vector_profile_uses_torch_only_loss daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r51_native_projection_enforces_allocation_constraints_and_derived_roles daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r51_native_allocation_loss_penalizes_unfunded_receiver_and_underdefended_cash daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r51_native_target_weight_takes_priority_in_allocation_layer`。
- r51 当前只允许作为 research / shadow 入口；不得 promotion、不得 live、不得改 active artifact。若启动正式研究，仍必须先确认无冲突进程、无同名 tag，并以前台持久日志和进度文件运行短 screening。

## 2026-05-10 r52 day-set native allocation vector 操作入口
- r52 profile / loss：`split_heads_portfolio_daily_day_set_native_allocation_vector_r52` / `alpha_result_value_budget_split_v37`。
- r52 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r52 默认 screening 资源口径：profile 内 `epochs = 6`、`min_epochs = 4`、`batch_size = 1`，这里的 `batch_size` 表示 day batch；搜索允许 `batch_size = 1/2`。
- r52 solver 口径：不启用 `cvxpy` / `diffcp` 训练主路径，不加入 true-solver resource guard；`alpha_result_value_budget_split_v37` 中 `portfolio_cvxpy_convex_allocation_total = 0`、`portfolio_full_universe_convex_allocation_total = 0`，full-universe train solver effective 必须为 false。
- r52 诊断读取：训练 diagnostics 需优先读取 `supports_portfolio_day_set_native_allocation_vector`、`portfolio_day_set_native_allocation_vector_terms`、`sample_model_type`、`day_set_full_day_integrity`、`day_set_batch_size` 与 `day_set_slot_count`；推理与 simulator 仍读取 `portfolio_daily_target_weight`、`portfolio_daily_target_delta`、native receiver/source/cash score 和 `allocation_layer_native_target_used`。
- r52 dry-run 命令：`$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_day_set_native_allocation_vector_r52 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r52 当前只允许作为 research / shadow 入口；不得 promotion、不得 live、不得改 active artifact。若启动正式研究，仍必须先确认无冲突进程、无同名 tag，并以前台持久日志和进度文件运行短 screening。

## 2026-05-10 r52 safe screening 操作结果
- 已完成 safe screening-only：`self_opt_study_r52_day_set_native_allocation_vector_screening_safe_20260510_02`。
- 产物入口：study summary `daily_research/output/continuous_policy/studies/self_opt_study_r52_day_set_native_allocation_vector_screening_safe_20260510_02/study_summary.json`，ranking `daily_research/output/continuous_policy/studies/self_opt_study_r52_day_set_native_allocation_vector_screening_safe_20260510_02/trial_ranking.csv`，protocol summaries `daily_research/output/continuous_policy/protocols/self_opt_study_r52_day_set_native_allocation_vector_screening_safe_20260510_02__trial_01/protocol_summary.json` 与 `...__trial_02/protocol_summary.json`，training diagnostics `daily_research/output/continuous_policy/models/self_opt_study_r52_day_set_native_allocation_vector_screening_safe_20260510_02__trial_01__train/training_diagnostics.json` 与 `...__trial_02__train/training_diagnostics.json`。
- 运行通道：`completed_trial_count = 2`、`failed_trial_count = 0`、`confirmatory_enabled = false`、`confirmatory_completed_trial_count = 0`；resource gate 触发，停止第 3 个 screening trial，失败项为 `source_release_dead`、`economic_signal_too_weak`、`exposure_utilization_low`。
- 必读指标：trial_01 `source_target_count = 0`、`source_realized_sell_rate = 0`、`receiver_target_count = 54`、`receiver_unrealized_deploy_share = 0`、`portfolio_daily_exposure_utilization = 0.559323`、`cash_timing_quality_1d = 0.019888`、`annual_return = 0.005293`、`training_evidence_status = insufficient`；trial_02 `source_target_count = 0`、`source_realized_sell_rate = 0`、`receiver_target_count = 64`、`receiver_unrealized_deploy_share = 0`、`portfolio_daily_exposure_utilization = 0.499263`、`cash_timing_quality_1d = 0.038652`、`annual_return = -0.017049`、`training_evidence_status = insufficient`。
- 操作判定：r52 不进入 confirmatory；不得把 safe screening 写成策略有效。后续若继续推进，应先做代码/目标层 source release 学习修复，再重新 dry-run + safe screening。
