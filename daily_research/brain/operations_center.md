# Daily Research 操作中枢

快照日期：`2026-04-24`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录已归档到 `daily_research/brain/references/`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；涉及 MKL/OpenMP 冲突时按既有命令设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 不启动训练、不切换 live、不改写 promotion，除非用户明确要求或状态中枢已有新正式决策。
- `analyze_behavior_gap.py` 会写 latest 行为摘要；需要多条审计时必须顺序执行，不得并行抢写。

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

## 当前 r11/r11b/r12/r13 操作口径
- `split_heads_release_translation_deploy_r12` 是当前最新 study profile，用于联合检查 release learning、order translation drift 与 deploy executability。
- `split_heads_action_value_unification_r13` 是当前最新已落地 research profile，用于检查 open/add/hold/reduce/exit 是否共享同一套多周期未来价值锚。
- `release_translation_deploy_v1` 是 r12 默认 objective；`release_translation_deploy_health_score` 是审计侧主健康分，必须结合组件分解解释。
- `action_value_unification_v1` 是 r13 默认 objective；`action_value_consistency_score`、`action_value_conflict_share`、`sell_against_keep_value_share`、`keep_against_release_value_share` 是 r13 主审计指标。
- 已完成的 r11b 仍全部是 `shadow_only`；r12 也只允许作为 shadow 研究入口，不改变 live 默认执行。
- 已验证 dry-run：`verify_release_translation_deploy_r12_dryrun_20260424` 只生成 study plan，baseline 为 `alpha_result_value_budget_split_v12 + result_value_v9 + cash_constraint_sell_source_guard_v7`。
- 已完成正式 bounded shadow study：`cp_v3_release_translation_deploy_r12__study_r1`，4 个 screening、2 个 confirmatory、0 失败。
- 当前 r12 champion：`confirm_02 = alpha_result_value_budget_split_v12 + result_value_v9`；`promotion_status = shadow_only`，`release_translation_deploy_failure_mode = order_translation_drift`。
- 当前 r13 状态：`cp_v3_action_value_unification_r13__study_r1` 已完成正式 bounded shadow study；4 个 screening、2 个 confirmatory 完成，最终 champion 为 `confirm_01 = alpha_result_value_budget_split_v13 + result_value_v10`，但 `promotion_status = shadow_only`，不得视为 live 证据。
- r13 产物入口：
  - study summary：`daily_research/output/continuous_policy/studies/cp_v3_action_value_unification_r13__study_r1/study_summary.json`
  - trial ranking：`daily_research/output/continuous_policy/studies/cp_v3_action_value_unification_r13__study_r1/trial_ranking.csv`
  - champion protocol：`daily_research/output/continuous_policy/protocols/cp_v3_action_value_unification_r13__study_r1__confirm_01/protocol_summary.json`
- r12 排名与明细入口：
  - study summary：`daily_research/output/continuous_policy/studies/cp_v3_release_translation_deploy_r12__study_r1/study_summary.json`
  - trial ranking：`daily_research/output/continuous_policy/studies/cp_v3_release_translation_deploy_r12__study_r1/trial_ranking.csv`
  - champion protocol：`daily_research/output/continuous_policy/protocols/cp_v3_release_translation_deploy_r12__study_r1__confirm_02/protocol_summary.json`
  - champion held-side detail：`daily_research/output/continuous_policy/analysis/behavior_audits/cp_v3_release_translation_deploy_r12__study_r1__confirm_02__details_v1__held_side_details.csv`
- `--export-held-side-details` 是 held-side 根因分析入口；导出逐仓明细时必须顺序审计。
- 旧的完整 r10/r11/r11b 命令、产物路径和复盘说明已归档；需要复现时先读历史操作索引。

## execution app 运行时
- 统一运行时目录：`daily_research/output/execution_app`。
- 常用流程：`tasks -> doctor -> run --task ... -> status/tail -> resume/unlock`。
- Web 控制台默认本地端口：`127.0.0.1:8765`。
- `agent` 本地验收可用同一 PowerShell 会话 `Start-Job` 后台启动；该约定不改变用户侧公开教程默认。

## 写回路由
- 当前状态、优先级、边界：`state_center.md`。
- 稳定事实、规则、术语：`knowledge_center.md`。
- 新命令口径、环境和流程：`operations_center.md`。
- 过程证据、动作后复盘：`episodic_memory.md`。
- 大段历史原文与标题索引：`daily_research/brain/references/`。

## 历史归档入口
- 原 `operations_center.md` 已原样归档：`daily_research/brain/references/operations_center_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/operations_center_evidence_index_20260424.md`。
- 原始行数：`1483`。
- 原始 SHA256：`6ade627c15612feab90eaf9c1389a6be7f24b5573c2c2e62583a816f41c8a2a3`。
- 读取纪律：当前操作以本文件上方章节为准；旧命令仅作为复现和审计证据。
