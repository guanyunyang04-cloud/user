# Daily Research 操作中枢

快照日期：`2026-05-07`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录进入 `daily_research/brain/references/` 或 `episodic_memory.md`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，不得依赖当前 shell Python。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；涉及 MKL/OpenMP 冲突时设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
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
- r34 最新证据入口：`cp_v3_portfolio_daily_allocation_breadth_r34_bounded_20260430` 与 `cp_v3_portfolio_daily_allocation_breadth_r34_evidence_confirm_20260430`；读取 `portfolio_daily_v2_confirm_stability_checks`，重点看 `receiver_minus_source_forward_excess_5d`、`source_positive_forward_sell_share`、`source_strong_positive_forward_sell_count` 与 `training_evidence_status`。
- r35 已形成修复后正式 bounded confirm 证据；读取 `cp_v3_portfolio_daily_unified_allocation_r35_postfix4_bounded_confirm_20260430`，重点看 `confirm_02`、`portfolio_daily_v2_stable_confirmatory_trials`、`failed_checks`、source positive distribution、cash timing 与 drawdown。
- r37 screening 证据入口：`cp_v3_portfolio_daily_decision_focused_allocation_r37b_screening64_20260501`；读取 `trial_01`、`training_diagnostics.json`、`unified_allocation_summary_mean` 与 v2 gate report，重点看 predicted source penalty 是否进入 allocation path、source positive distribution、receiver-source spread、cash timing 与 drawdown。
- r38 最新 bounded confirm 证据入口：`self_opt_study_r38_source_hard_negative_regret_20260501`；读取 `study_summary.json`、`protocols/self_opt_study_r38_source_hard_negative_regret_20260501__confirm_01/protocol_summary.json` 与对应 `training_diagnostics.json`，重点看 source false-sell 是否被压住、source/receiver 广度是否足够、cash 是否过度保守、v2 gate failed checks 与 `portfolio_daily_v2_confirm_stability_checks`。
- r39 最新 bounded confirm 证据入口：`self_opt_study_r39_allocation_objective_consolidation_execblend_20260502`；读取 `study_summary.json`、`protocols/self_opt_study_r39_allocation_objective_consolidation_execblend_20260502__confirm_01/protocol_summary.json` 与对应 `training_diagnostics.json`，重点看 final objective 是否真正进入执行评分、source count 是否达标、cash timing / drawdown 是否仍失败、source false-sell 是否仍受控。
- 当前有效证据基线仍是 r39：`alpha_result_value_budget_split_v25` 与 `portfolio_daily_ranking_v2_gated`，对应 simulator calibration 为 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
- 当前已完成 r40 clean rerun：`self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504`，使用 `alpha_result_value_budget_split_v25`、`end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与 `end_to_end_allocation_layer_v1`，已完整生成 study/protocol/model/ranking 证据；但 stable confirm 为空，仍不得替代 r39 证据基线。
- r41-r45 已完成代码入口或 dry-run，但均为保留检查点，不是当前默认长训入口；当前优先 research 入口为 r46：`split_heads_portfolio_daily_differentiable_convex_allocation_r46` / `alpha_result_value_budget_split_v31` / `epochs = 14` / `min_epochs = 9` / 更严格 `resource_gate`。r46 已补 component diagnostics、data-driven constraints、behavior/OPE 与 path risk，但仍未生成正式 screening/protocol/ranking 证据，不能作为正式 verdict。
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
- 若涉及 PyTorch / MKL / OpenMP 导入冲突，前台命令显式设置 `KMP_DUPLICATE_LIB_OK=TRUE`；所有训练、测试、审计继续使用 yolos Python。
- r40 当前只允许作为 research / shadow 入口；`self_opt_study_r40_end_to_end_allocation_layer_20260502` 已自然结束但因外层 stdout 管道失效污染后续 trial/protocol 状态，不能作为有效 bounded verdict，不得 promotion、不得 live、不得写 active artifact。
- r40 或更长 study 的前台运行建议命令形态：
  - `$tag='<tag>'; New-Item -ItemType Directory -Force -Path "daily_research/output/continuous_policy/studies/$tag" | Out-Null; $env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_end_to_end_allocation_layer_r40 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag $tag *> "daily_research/output/continuous_policy/studies/$tag/foreground.log"`
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

## 2026-05-07 r41-r45 保留检查点
- r41-r45 已分别覆盖 risk-sensitive、utility-credit、primal-dual decision、entropic transport 与 conservative transport；当前不作为默认长训入口。若必须回溯验证，只能先 dry-run / contract gate，再以前台持久日志短 screening 执行；任何代码合同、dry-run 或单测通过都不得写成策略有效、promotion、live 或 active artifact 切换依据。

## 2026-05-07 r45 conservative transport allocation 操作入口
- r45 profile / loss：`split_heads_portfolio_daily_conservative_transport_allocation_r45` / `alpha_result_value_budget_split_v30`。
- r45 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r45 默认 screening 资源口径：profile 内 `epochs = 16`、`min_epochs = 10`；命令行未显式传入 `--epochs` / `--min-epochs` 时不得覆盖该默认值。
- r45 dry-run 命令：`$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_conservative_transport_allocation_r45 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r45 合同测试命令：`$env:KMP_DUPLICATE_LIB_OK='TRUE'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r45_conservative_transport_profile_adds_offline_support_gate daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r45_offline_conservative_support_loss_penalizes_ood_overconfidence`。
- r45 若启动正式研究，优先只跑短 screening；若 `resource_gate.triggered = true`，不得进入 confirmatory。只有 resource gate 未触发且 screening 同时显示 source release、offline support conservatism、cash timing、drawdown、tail false-source、transport balance、receiver deploy 与经济信号有同步改善时，才允许规划 confirmatory。
- r45 当前只允许作为 research / shadow 入口；代码合同、dry-run 和单测通过不等于策略有效，不得 promotion、不得 live、不得改 active artifact。

## 2026-05-07 r46 differentiable convex allocation 操作入口
- r46 profile / loss：`split_heads_portfolio_daily_differentiable_convex_allocation_r46` / `alpha_result_value_budget_split_v31`。
- r46 objective / budget：`end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1`。
- r46 默认 screening 资源口径：profile 内 `epochs = 14`、`min_epochs = 9`；命令行未显式传入 `--epochs` / `--min-epochs` 时不得覆盖该默认值。
- r46 诊断读取：训练 diagnostics 需优先读取 `portfolio_differentiable_convex_allocation_terms`，其中 `constraint_residual`、`unsupported_mass`、`false_source_mass`、`cash_timing_loss`、`behavior_support_loss`、`conservative_ope_loss`、`path_risk_loss` 与 `total` 是判断是否值得正式 screening 的关键前置信号。
- r46 dry-run 命令：`$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_differentiable_convex_allocation_r46 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --dry-run`。
- r46 合同测试命令：`$env:KMP_DUPLICATE_LIB_OK='TRUE'; C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m unittest daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r46_differentiable_convex_profile_is_solver_first_not_action_reuse daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts.PortfolioDailyStrategyContractsTest.test_r46_differentiable_convex_loss_penalizes_kkt_and_support_violations`。
- r46 若启动正式研究，优先只跑短 screening；若 `resource_gate.triggered = true`，不得进入 confirmatory。只有 resource gate 未触发且 screening 同时显示 source release、cash timing、drawdown、support conservatism、constraint residual、behavior/OPE、path risk、receiver deploy 与经济信号有同步改善时，才允许规划 confirmatory。
- r46 当前只允许作为 research / shadow 入口；代码合同、dry-run 和单测通过不等于策略有效，不得 promotion、不得 live、不得改 active artifact。
