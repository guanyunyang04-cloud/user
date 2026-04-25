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

## 当前 r12-r16 操作口径
- `split_heads_release_translation_deploy_r12` 使用 `release_translation_deploy_v1`，用于联合检查 release learning、order translation drift 与 deploy executability。
- `split_heads_action_value_unification_r13` 使用 `action_value_unification_v1`，用于检查 open/add/hold/reduce/exit 是否共享同一套多周期未来价值锚。
- `split_heads_direct_action_value_r14` 使用 `direct_daily_policy_v1`，用于检查模型是否能通过直接动作值仲裁日级动作。
- `split_heads_direct_action_translation_r15` 使用 `direct_action_translation_v1` 与 `cash_constraint_direct_action_guard_v8`，用于检查订单/预算层是否保留 direct action intent。
- `split_heads_direct_action_reallocation_r16` 使用 `direct_action_reallocation_v1` 与 `cash_constraint_direct_action_reallocation_guard_v9`，用于检查高置信 add/open 是否能获得显式预算再分配。
- 月度收益评价已接入通用曲线指标与 study ranking；重点看 `monthly_return_mean`、`monthly_win_rate`、`monthly_worst_return`、`monthly_max_consecutive_loss_months`、`monthly_consistency_score`，并读取 `monthly_returns.csv` 或 `shadow_monthly_returns.csv` 明细。
- r12/r13/r14/r15/r16 全部仍为 research / shadow 证据，不改变 live 默认执行。

## r16 当前证据入口
- r16 dry-run：`daily_research/output/continuous_policy/studies/verify_direct_action_reallocation_r16_dryrun_20260425/study_summary.json`。
- r16 最佳 smoke evaluation：`daily_research/output/continuous_policy/evaluations/verify_direct_action_reallocation_r16_v9_eval_smoke2_20260425/evaluation_summary.json`。
- r16 最佳 smoke audit：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_reallocation_r16_v9_audit_smoke2_20260425.json`。
- r16 held-side detail：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_reallocation_r16_v9_audit_smoke2_20260425__held_side_details.csv`。
- r15 formal study summary：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_translation_r15__study_r1/study_summary.json`。
- r15 champion protocol：`daily_research/output/continuous_policy/protocols/cp_v3_direct_action_translation_r15__study_r1__confirm_01/protocol_summary.json`。

## r16 推荐命令
- dry-run：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_direct_action_reallocation_r16 --objective-profile direct_action_reallocation_v1 --trial-count 4 --study-tag verify_direct_action_reallocation_r16_dryrun_20260425 --dry-run`
- 评估 r15 champion artifact 的 v9 reallocation smoke 时，必须显式指定 `cash_constraint_direct_action_reallocation_guard_v9`，并保留独立 tag，避免覆盖正式 r15 evidence。
- 审计 r16 evaluation summary 时使用 `--export-held-side-details`，并顺序运行，避免 latest 摘要竞争。

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
