# Daily Research 操作中枢

快照日期：`2026-04-29`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录进入 `daily_research/brain/references/` 或 `episodic_memory.md`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，不得依赖当前 shell Python。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；涉及 MKL/OpenMP 冲突时设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 脑内文档标题、正文、状态、规则和复盘必须使用简体中文；命令、路径、指标名、tag、模型名等技术标识保留原文。
- 当前 active 执行权重语义固定为 `research_raw_target_weight`，权重上限语义固定为 `follow_research_raw_no_global_cap`。
- 所有训练、评估、审计、bounded study、confirmatory rerun、execution app 任务与交易计划任务默认前台运行，主进程不得后台化规避窗口，也不得中途人为中断。
- 所有项目前台窗口时限统一按 `10` 小时处理；若外层工具有更短硬上限，只允许接续等待或读取自然完成产物，不改变任务本体。
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
- 当前正式默认仍是 research / shadow，不能切换 live 或 promotion。
- r31 receiver semantic closure 入口：`split_heads_portfolio_daily_receiver_semantic_closure_r31`，重点读取 `direct_action_authorization_subset_violation_count`、`authorized_add_no_weight_change_share`、`deploy_intent_unrealized_share`、`portfolio_daily_receiver_unrealized_deploy_share`。
- r33 source forward proxy / clean-pass 入口：`split_heads_portfolio_daily_source_forward_proxy_r33`，重点读取 `portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction`、`portfolio_daily_source_distribution_clean_pass`、`portfolio_daily_source_positive_forward_sell_share`、`portfolio_daily_source_strong_positive_forward_sell_count`。
- 当前核心 loss / objective：`alpha_result_value_budget_split_v20` 与 `portfolio_daily_ranking_v2_gated`。
- 当前核心 simulator calibration：`cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`，并继承 v13 cash-aware 与 v14 source-exec 语义。
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
