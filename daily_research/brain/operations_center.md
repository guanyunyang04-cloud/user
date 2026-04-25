# Daily Research 操作中枢

快照日期：`2026-04-25`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录已归档到 `daily_research/brain/references/`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；涉及 MKL/OpenMP 冲突时设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 不启动训练、不切换 live、不改写 promotion，除非用户明确要求或状态中枢已有新正式决策。
- `analyze_behavior_gap.py` 会写 latest 行为摘要；需要多条审计时必须顺序执行，不得并行抢写。
- 长训练、评估或审计按阻塞等待完成处理；不做无意义轮询。

## 项目地图
- 当前状态与研究优先级：`daily_research/brain/state_center.md`。
- 稳定规则、术语和教训：`daily_research/brain/knowledge_center.md`。
- continuous_policy 设计合同：`daily_research/brain/continuous_policy_design_contract.md`。
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
- continuous_policy 入口：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_continuous_policy_protocol --help`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --help`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/analyze_behavior_gap.py --help`

## 当前 r12-r18 操作口径
- `split_heads_release_translation_deploy_r12` 使用 `release_translation_deploy_v1`，用于联合检查 release learning、order translation drift 与 deploy executability。
- `split_heads_action_value_unification_r13` 使用 `action_value_unification_v1`，用于检查 open/add/hold/reduce/exit 是否共享同一套多周期未来价值锚。
- `split_heads_direct_action_value_r14` 使用 `direct_daily_policy_v1`，用于检查模型是否能通过直接动作值仲裁日级动作。
- `split_heads_direct_action_translation_r15` 使用 `direct_action_translation_v1` 与 `cash_constraint_direct_action_guard_v8`，用于检查订单/预算层是否保留 direct action intent。
- `split_heads_direct_action_reallocation_r16` 使用 `direct_action_reallocation_v1` 与 `cash_constraint_direct_action_reallocation_guard_v9`，用于检查高置信 add/open 是否能获得显式预算再分配。
- `split_heads_direct_action_pair_reallocation_r17` 使用 `direct_action_pair_reallocation_v1` 与 `cash_constraint_direct_action_pair_reallocation_guard_v10`，用于检查 core deploy target 是否能在满仓/预算受限日通过成对换仓真实成交。
- `split_heads_direct_action_pair_cost_guard_r18` 使用 `direct_action_pair_cost_guard_v1` 与 `cash_constraint_direct_action_pair_cost_guard_v11`，用于检查 pair-source 是否相对 core target 足够弱、机会成本可接受且不会单纯扩大牺牲源数量。
- 月度收益评价已接入通用曲线指标与 study ranking；重点看 `monthly_return_mean`、`monthly_win_rate`、`monthly_worst_return`、`monthly_max_consecutive_loss_months`、`monthly_consistency_score`，并读取 `monthly_returns.csv` 或 `shadow_monthly_returns.csv` 明细。
- r12/r13/r14/r15/r16/r17/r18 全部仍为 research / shadow 证据，不改变 live 默认执行。

## r16-r17 当前证据入口
- r16 dry-run：`daily_research/output/continuous_policy/studies/verify_direct_action_reallocation_r16_dryrun_20260425/study_summary.json`。
- r16 最佳 smoke evaluation：`daily_research/output/continuous_policy/evaluations/verify_direct_action_reallocation_r16_v9_eval_smoke2_20260425/evaluation_summary.json`。
- r16 最佳 smoke audit：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_reallocation_r16_v9_audit_smoke2_20260425.json`。
- r16 held-side detail：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_reallocation_r16_v9_audit_smoke2_20260425__held_side_details.csv`。
- r17 dry-run：`daily_research/output/continuous_policy/studies/verify_direct_action_pair_reallocation_r17_dryrun_20260425/study_summary.json`。
- r17 最佳 smoke evaluation：`daily_research/output/continuous_policy/evaluations/verify_direct_action_pair_reallocation_r17_v10_eval_smoke3_20260425/evaluation_summary.json`。
- r17 最佳 smoke audit：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_pair_reallocation_r17_v10_audit_smoke3_20260425.json`。
- r17 bounded study 原始 summary：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/study_summary.json`。
- r17 repaired confirm 对照：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/manual_confirm_repair_summary.json` 与 `manual_confirm_repair_comparison.csv`。
- r17 pair-source 专项审计：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/pair_source_audit_summary.json` 与 `pair_source_audit_comparison.csv`。
- r17 repaired confirm held-side 审计：`daily_research/output/continuous_policy/analysis/behavior_audits/cp_v3_direct_action_pair_reallocation_r17__study_r1__confirm_01_repair_audit.json`、`...confirm_02_repair_audit.json`。
- r18 dry-run：`daily_research/output/continuous_policy/studies/verify_direct_action_pair_cost_guard_r18_dryrun_20260425/study_summary.json`。
- r18 smoke2 evaluation：`daily_research/output/continuous_policy/evaluations/verify_direct_action_pair_cost_guard_r18_v11_confirm01_smoke2_20260425/evaluation_summary.json`。
- r18 smoke2 audit：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_pair_cost_guard_r18_v11_confirm01_smoke2_audit_20260425.json`。
- r15 formal study summary：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_translation_r15__study_r1/study_summary.json`。
- r15 champion protocol：`daily_research/output/continuous_policy/protocols/cp_v3_direct_action_translation_r15__study_r1__confirm_01/protocol_summary.json`。

## r17-r18 推荐命令
- dry-run：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_direct_action_pair_reallocation_r17 --objective-profile direct_action_pair_reallocation_v1 --trial-count 4 --study-tag verify_direct_action_pair_reallocation_r17_dryrun_20260425 --dry-run`
- r18 dry-run：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_direct_action_pair_cost_guard_r18 --objective-profile direct_action_pair_cost_guard_v1 --trial-count 4 --study-tag verify_direct_action_pair_cost_guard_r18_dryrun_20260425 --dry-run`
- 评估 r15 champion artifact 的 v10 pair reallocation smoke 时，必须显式指定 `cash_constraint_direct_action_pair_reallocation_guard_v10`，并保留独立 tag，避免覆盖正式 r15 evidence。
- 评估 r17 repaired confirm artifact 的 v11 pair cost guard smoke 时，必须显式指定 `cash_constraint_direct_action_pair_cost_guard_v11`，并同时读取 `direct_action_core_minus_pair_forward_excess_5d`、pair-source cost、blocked count、turnover 与月度收益质量。
- 审计 r17 evaluation summary 时使用 `--export-held-side-details`，并顺序运行，避免 latest 摘要竞争。
- 若需要回看 r16 对照，可使用 `split_heads_direct_action_reallocation_r16`、`direct_action_reallocation_v1` 与 `cash_constraint_direct_action_reallocation_guard_v9`，但 r16 当前只作为结构对照基线。
- r17 bounded study 的自动 confirm summary 失败属于 study runner 子进程 `[Errno 22] Invalid argument`；使用 direct protocol rerun / strict resume 产物时，以 `manual_confirm_repair_summary.json` 为 repaired confirm 对照真源。
- 下一轮不得继续把主要算力投入到单独动作 loss 堆叠；优先设计组合级日决策 objective / pair-listwise ranking / 月度收益直接优化入口。

## execution app 运行时
- 统一运行时目录：`daily_research/output/execution_app`。
- 常用流程：`tasks -> doctor -> run --task ... -> status/tail -> resume/unlock`。
- Web 控制台默认本地端口：`127.0.0.1:8765`。
- `agent` 本地验收可用同一 PowerShell 会话 `Start-Job` 后台启动；该约定不改变用户侧公开教程默认。

## 写回路由
- 当前状态、优先级、边界：`state_center.md`。
- 稳定事实、规则、术语：`knowledge_center.md`。
- 新命令口径、环境和流程：`operations_center.md`。
- 设计边界和成功判定：`continuous_policy_design_contract.md`。
- 过程证据、动作后复盘：`episodic_memory.md`。
- 大段历史原文与标题索引：`daily_research/brain/references/`。

## 历史归档入口
- 原 `operations_center.md` 已原样归档：`daily_research/brain/references/operations_center_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/operations_center_evidence_index_20260424.md`。
- 原始行数：`1483`。
- 原始 SHA256：`6ade627c15612feab90eaf9c1389a6be7f24b5573c2c2e62583a816f41c8a2a3`。
- 读取纪律：当前操作以本文件上方章节为准；旧命令仅作为复现和审计证据。
## 2026-04-25 r19 portfolio daily ranking operation path
- Use `split_heads_portfolio_daily_ranking_r19` when the task is to study portfolio-level daily execution rather than another action-loss variant.
- The simulator budget calibration for this path is `cash_constraint_portfolio_daily_ranking_guard_v12`; it ranks capital receivers, capital sources, and cash reserve pressure in one daily portfolio context.
- The study objective is `portfolio_daily_ranking_v1`; promotion discussion is forbidden until formal bounded evidence passes monthly quality, turnover, drawdown, receiver-source spread, source sell realization, and cash timing checks.
- Audit priority: read `portfolio_daily_receiver_minus_source_forward_excess_5d`, `portfolio_daily_source_realized_sell_rate`, `portfolio_daily_cash_reserve_rate`, and monthly returns before interpreting headline annual return.
## 长时训练任务运行纪律
- 长时训练任务默认在后台运行；前台不承担训练主进程，只负责保持监控直到后台进程完成。
- 前台不需要反复轮询进度；除非进程异常、用户要求状态、或需要读取最终产物，否则不要做无意义轮询。
- 监控完成后再一次性读取日志、summary、checkpoint、evaluation 或 audit 产物，并按 `state_center.md` / `episodic_memory.md` 写回复盘。
- 这条规则适用于 formal training、bounded study、confirmatory rerun、长窗口 evaluation 和长耗时 audit。
