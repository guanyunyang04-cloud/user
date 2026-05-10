# Daily Research 操作中枢

快照日期：`2026-05-10`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录进入 `daily_research/brain/references/`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，不得依赖当前 shell Python。
- Windows 默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- 中文文档正文不得用 PowerShell here-string 直接写入；优先使用 `apply_patch`，批量生成时必须显式 `encoding='utf-8'` 并用 Python 复核。
- 当前 `yolos` 已完成 OpenMP 原地修复；根治验收命令为 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/openmp_runtime_check.py --strict`。
- 当前命令不得默认设置 `KMP_DUPLICATE_LIB_OK`。
- 所有训练、评估、审计、bounded study、confirmatory rerun、execution app 与交易计划任务默认前台运行，不得为规避窗口而后台化。
- 长任务 stdout/stderr 必须写入持久日志；监控轮询间隔固定为 `2` 小时；进程自然结束后立即解析产物。
- GPU 训练完成后必须核验 `training_diagnostics.json` 中 `device = cuda`、`cuda_available = true` 与 `python_executable`。

## 项目地图
- 当前状态与边界：`daily_research/brain/state_center.md`。
- 稳定事实、规则和教训：`daily_research/brain/knowledge_center.md`。
- continuous_policy 设计合同：`daily_research/brain/continuous_policy_design_contract.md`。
- 过程复盘：`daily_research/brain/episodic_memory.md`。
- 执行与交易计划：`daily_research/execution`。
- 连续策略研究：`daily_research/continuous_policy`。
- 产物、评估、协议、审计：`daily_research/output`。

## 高频命令
- 主脑接管：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_bootstrap.py --child daily_research --json`
- 平台化接管胶囊：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow handoff --child daily_research --json`
- 工作流健康检查：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow health --json`
- continuous_policy 状态胶囊：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow status --workflow continuous_policy --json`
- 写回计划预览：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow writeback-plan --source latest --json`
- 守卫检查：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
- 文档守卫：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
- 项目一致性：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/project_consistency_check.py`
- OpenMP strict：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/openmp_runtime_check.py --strict`
- execution app：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_execution_app.py status`
- execution web：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/execution/run_execution_app.py web --port 8765`
- continuous_policy protocol：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_continuous_policy_protocol --help`
- self-optimizing study：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --help`

## 当前 continuous_policy 操作口径
- 当前 live 默认执行链仍以 `active_execution_strategy.json` 为真源；continuous_policy 当前仍是 `research / shadow_only`。
- 当前有效证据基线仍是 r39；r48 是失败 verdict；r49-r52 是 research profile，不是 production 默认。
- r31：`split_heads_portfolio_daily_receiver_semantic_closure_r31`。
- r31 重点看 `direct_action_authorization_subset_violation_count`。
- r33：`split_heads_portfolio_daily_source_forward_proxy_r33`，重点看 `portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction` 与 `portfolio_daily_source_distribution_clean_pass`。
- r34：`split_heads_portfolio_daily_allocation_breadth_r34`，重点看 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth`、`portfolio_daily_joint_economic_quality_gate` 与 `allocation_teacher_summary_mean`。
- r35：`split_heads_portfolio_daily_unified_allocation_r35`，重点看 `unified_allocation_summary_mean`、`portfolio_daily_unified_allocation_objective`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty` 与 `portfolio_daily_receiver_source_spread_reward`。
- r39 当前证据基线仍读 `portfolio_daily_ranking_v2_gated` 与 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
- 月度收益评价读取 `monthly_returns.csv` 与对应 shadow 月度文件。
- r51：`split_heads_portfolio_daily_native_allocation_vector_r51` / `alpha_result_value_budget_split_v36`，不启用 `cvxpy/diffcp` 训练主路径。
- r52：`split_heads_portfolio_daily_day_set_native_allocation_vector_r52` / `alpha_result_value_budget_split_v37`，使用完整交易日 day-set batch，不启用 true solver。
- r52 诊断优先读取：`supports_portfolio_day_set_native_allocation_vector`、`portfolio_day_set_native_allocation_vector_terms`、`sample_model_type`、`day_set_batch_size`、`allocation_layer_native_target_used`、`native_source_delta_alignment_support`、`native_target_valid`、`native_source_target_count`、`native_source_audit_threshold_gap`。
- r52 当前判定：native source-delta closure 已部分修通，但 evidence insufficient，不能进入 confirmatory。

## r52 dry-run 模板
```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_day_set_native_allocation_vector_r52 --objective-profile end_to_end_allocation_layer_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --study-tag <tag> --disable-confirmatory --dry-run
```

## 产物读取入口
- study 摘要：`daily_research/output/continuous_policy/studies/<study_tag>/study_summary.json`。
- protocol 摘要：`daily_research/output/continuous_policy/protocols/<run_tag>/protocol_summary.json`。
- 模型诊断：`daily_research/output/continuous_policy/models/<run_tag>__train/training_diagnostics.json`。
- evaluation 摘要：`daily_research/output/continuous_policy/evaluations/<eval_tag>/evaluation_summary.json`。
- behavior audit：`daily_research/output/continuous_policy/analysis/behavior_audits/`。

## 写回路由
- 当前状态、优先级、边界：`state_center.md`。
- 稳定事实、规则、术语：`knowledge_center.md`。
- 新命令口径、环境和流程：`operations_center.md`。
- 设计边界和成功判定：`continuous_policy_design_contract.md`。
- 过程证据、动作后复盘：`episodic_memory.md`。
- 大段历史原文、标题索引和归档说明：`daily_research/brain/references/`。
- 默认先用 `brain_workflow writeback-plan` 生成路由建议；只有显式确认写回时才修改 brain 主文件。

## 历史归档入口
- 本文件归档前完整快照：`daily_research/brain/references/operations_center_archive_20260510.md`。
- 历史操作原文：`daily_research/brain/references/operations_center_history_raw_20260424.md`。
- 历史操作索引：`daily_research/brain/references/operations_center_evidence_index_20260424.md`。
