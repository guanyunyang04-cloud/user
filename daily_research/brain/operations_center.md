# Daily Research 操作中枢

快照日期：`2026-04-26`

## 默认操作纪律
- 本文件只保留当前高频入口、运行纪律和写回路由；旧命令长记录已归档到 `daily_research/brain/references/`。
- `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- Windows 下默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`；涉及 MKL/OpenMP 冲突时设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 脑内文档当前层标题、正文、状态、规则和复盘写回必须使用简体中文；命令、路径、指标名、tag、模型名等技术标识保留原文。
- 不启动训练、不切换 live、不改写 promotion，除非用户明确要求或状态中枢已有新正式决策。
- `analyze_behavior_gap.py` 会写 latest 行为摘要；需要多条审计时必须顺序执行，不得并行抢写。
- 所有训练、评估、审计、bounded study、confirmatory rerun、execution app 任务与交易计划任务默认前台运行，任务主进程不得后台化规避窗口，也不得中途人为中断。
- 所有项目任务前台窗口时限统一按 `10` 小时处理；若工具单次调用存在更短硬上限，应在同一前台任务语义下接续等待，不改变任务本体。
- continuous_policy / deep_alpha 等 GPU 训练完成后，必须核验 `training_diagnostics.json` 中 `device = cuda` 与 `cuda_available = true`，并确认解释器来自 `yolos` 后再写入正式证据。

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
- r16 dry-run 证据：`daily_research/output/continuous_policy/studies/verify_direct_action_reallocation_r16_dryrun_20260425/study_summary.json`。
- r16 最佳 smoke evaluation：`daily_research/output/continuous_policy/evaluations/verify_direct_action_reallocation_r16_v9_eval_smoke2_20260425/evaluation_summary.json`。
- r16 最佳 smoke audit：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_reallocation_r16_v9_audit_smoke2_20260425.json`。
- r16 held-side 明细：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_reallocation_r16_v9_audit_smoke2_20260425__held_side_details.csv`。
- r17 dry-run 证据：`daily_research/output/continuous_policy/studies/verify_direct_action_pair_reallocation_r17_dryrun_20260425/study_summary.json`。
- r17 最佳 smoke evaluation：`daily_research/output/continuous_policy/evaluations/verify_direct_action_pair_reallocation_r17_v10_eval_smoke3_20260425/evaluation_summary.json`。
- r17 最佳 smoke audit：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_pair_reallocation_r17_v10_audit_smoke3_20260425.json`。
- r17 bounded study 原始 summary：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/study_summary.json`。
- r17 repaired confirm 对照：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/manual_confirm_repair_summary.json` 与 `manual_confirm_repair_comparison.csv`。
- r17 pair-source 专项审计：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/pair_source_audit_summary.json` 与 `pair_source_audit_comparison.csv`。
- r17 repaired confirm held-side 审计：`daily_research/output/continuous_policy/analysis/behavior_audits/cp_v3_direct_action_pair_reallocation_r17__study_r1__confirm_01_repair_audit.json`、`...confirm_02_repair_audit.json`。
- r18 dry-run 证据：`daily_research/output/continuous_policy/studies/verify_direct_action_pair_cost_guard_r18_dryrun_20260425/study_summary.json`。
- r18 smoke2 评估：`daily_research/output/continuous_policy/evaluations/verify_direct_action_pair_cost_guard_r18_v11_confirm01_smoke2_20260425/evaluation_summary.json`。
- r18 smoke2 审计：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_pair_cost_guard_r18_v11_confirm01_smoke2_audit_20260425.json`。
- r15 正式 study 摘要：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_translation_r15__study_r1/study_summary.json`。
- r15 冠军 protocol：`daily_research/output/continuous_policy/protocols/cp_v3_direct_action_translation_r15__study_r1__confirm_01/protocol_summary.json`。

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
## 2026-04-25 r19 组合日频排序操作路径
- 当任务目标是研究组合级日频执行，而不是继续增加动作 loss 变体时，使用 `split_heads_portfolio_daily_ranking_r19`。
- 这一路径的模拟器预算校准为 `cash_constraint_portfolio_daily_ranking_guard_v12`；它在同一个日频组合状态里排序资金接收方、资金释放方和现金保留压力。
- study 目标为 `portfolio_daily_ranking_v1`；在正式 bounded evidence 通过月度质量、换手、回撤、receiver-source spread、source sell realization 和 cash timing 检查前，禁止进入 promotion 讨论。
- 审计优先读取 `portfolio_daily_receiver_minus_source_forward_excess_5d`、`portfolio_daily_source_realized_sell_rate`、`portfolio_daily_cash_reserve_rate` 和月度收益，再解释 headline annual return。
## 2026-04-26 r19 bounded study 证据路径
- bounded study 摘要：`daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r19__study_r1/study_summary.json`。
- trial ranking 表：`daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r19__study_r1/trial_ranking.csv`。
- 表现线 confirm protocol：`daily_research/output/continuous_policy/protocols/cp_v3_portfolio_daily_ranking_r19__study_r1__confirm_01/protocol_summary.json`。
- 综合稳定线 confirm protocol：`daily_research/output/continuous_policy/protocols/cp_v3_portfolio_daily_ranking_r19__study_r1__confirm_02/protocol_summary.json`。
- 训练环境检查：读取每个 `daily_research/output/continuous_policy/models/cp_v3_portfolio_daily_ranking_r19__study_r1*__train/training_diagnostics.json`，要求 `device = cuda`、`cuda_available = true`、`trainer_backend = formal_torch_seq_v3`。
## 长时任务运行纪律
- 长时任务默认前台运行；训练、评估、审计、bounded study、confirmatory rerun、长窗口 evaluation 和长耗时 audit 都不得默认转后台。
- 前台任务不得中途人为中断；若外层工具因自身硬上限断开，只允许接续等待或读取已自然完成的产物，不得停止训练主进程。
- 前台窗口时限统一为 `10` 小时；启动命令必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 判断任务是否仍在运行时，必须核对 PID、CommandLine、run_tag 和产物时间戳；不要在 Python summary/diagnostics 解析命令并行运行时用裸 `Get-Process python` 下结论。
- 完成后一次性读取日志、summary、checkpoint、evaluation 或 audit 产物，并按 `state_center.md` / `episodic_memory.md` 写回复盘。

## 2026-04-26 r20 v2/v13 操作路径
- 当前 r20 默认预算校准标记：`cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13`。
- r19 离线反冠军诊断：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/portfolio_daily_ranking_gate_report.py --study-summary daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r19__study_r1/study_summary.json`
- r20 v2/v13 smoke 推荐命令：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_r19 --objective-profile portfolio_daily_ranking_v2_gated --trial-count 1 --disable-confirmatory --epochs 6 --min-epochs 3 --early-stop-patience 3 --study-tag cp_v3_portfolio_daily_ranking_r20_v2_gated_smoke_20260426_retry2`
- r20 bounded confirmatory 推荐命令：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_r19 --objective-profile portfolio_daily_ranking_v2_gated --trial-count 1 --confirmatory-max-candidates 1 --epochs 6 --min-epochs 3 --early-stop-patience 3 --confirmatory-epochs 8 --confirmatory-min-epochs 4 --study-tag cp_v3_portfolio_daily_ranking_r20_v2_gated_confirm_20260426`
- 读取 r20 结果时，优先看 v2 gate report 中的 `v2_champion`、失败 gate 和 `portfolio_daily_effective_model_action` 后的冲突指标；不要只看旧 `study_summary.json` 的 `champion` 字段，尤其是代码修复前已经生成的历史 summary。
- `portfolio_daily_ranking_v2_gated` 的 champion selection 已修复；后续新 summary 应读取 `champion_selection_policy = portfolio_daily_v2_stable_confirmatory_then_screening_fallback`、`portfolio_daily_v2_confirm_stability_checks` 与 `rejected_confirmatory_trials`。
- 任何 r20 训练证据写成正式结论前，仍必须逐个读取 `training_diagnostics.json`，确认 `device = cuda`、`cuda_available = true`、`trainer_backend = formal_torch_seq_v3`、显式 `yolos` 解释器路径和 strict resume。
- 进程检查继续按语义核验：PID、CommandLine、run_tag、日志/summary 时间戳必须一致；不得把当前检查脚本或短暂 Python 解析进程误判为 r19/r20 训练仍在运行。

## 2026-04-26 r20 stability sweep 操作路径
- r20 稳定性搜索 profile：`split_heads_portfolio_daily_ranking_stability_r20`。
- 已执行 bounded sweep：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_stability_r20 --trial-count 6 --confirmatory-max-candidates 2 --epochs 8 --min-epochs 4 --early-stop-patience 4 --confirmatory-epochs 12 --confirmatory-min-epochs 6 --study-tag cp_v3_portfolio_daily_ranking_r20_stability_sweep_20260426`
- 本轮报告：
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r20_stability_sweep_20260426/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_gate_report.md`
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r20_stability_sweep_20260426/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_rescore.csv`
- 读取该报告时必须区分 `v2_top_ranked` 与 `qualified_v2_champion`；若 `qualified_v2_champion` 为空，说明没有完全通过 v2 gates 的路线。
- 后续训练诊断应包含 `python_executable`、`conda_prefix` 与 `runtime_env`；若历史 diagnostics 缺少解释器字段，只能结合前台命令记录说明其由 `yolos` 启动，不能把缺字段写成 diagnostics 内证据。

## 2026-04-26 r21 source-exec 操作路径
- r21 source execution profile：`split_heads_portfolio_daily_ranking_source_exec_r21`。
- r21 预算校准标记：`cash_constraint_portfolio_daily_ranking_source_exec_guard_v14`。
- r21 默认目标仍为 `portfolio_daily_ranking_v2_gated`，但必须额外读取 `source_not_sold_ceiling`、`portfolio_daily_source_target_not_sold_share`、`portfolio_daily_source_exec_cap_guard_count`、`portfolio_daily_source_realized_reduction_weight` 与 `portfolio_daily_effective_capital_transfer_count`。
- dry-run 验证命令：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_source_exec_r21 --trial-count 1 --study-tag verify_r21_source_exec_profile_dryrun_20260426 --dry-run`
- 已执行 smoke：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_source_exec_r21 --trial-count 1 --disable-confirmatory --epochs 6 --min-epochs 3 --early-stop-patience 3 --study-tag cp_v3_portfolio_daily_ranking_r21_source_exec_smoke_20260426_retry2`
- r21 retry2 gate report：
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r21_source_exec_smoke_20260426_retry2/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_gate_report.md`
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r21_source_exec_smoke_20260426_retry2/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_rescore.csv`
- smoke/正式训练仍必须前台阻塞运行，窗口时限 `10` 小时；任何 r21 结果都先写成 `research / shadow_only`，不得跳过 v2 gate report、GPU diagnostics 和 semantic process check。

## 2026-04-26 r22 receiver-exec 操作路径
- r22 receiver execution profile：`split_heads_portfolio_daily_ranking_receiver_exec_r22`。
- r22 预算校准标记：`cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
- r22 默认目标：`portfolio_daily_ranking_v2_gated`。
- r22 dry-run 元数据检查命令：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_receiver_exec_r22 --trial-count 1 --study-tag verify_r22_receiver_exec_profile_metadata_20260426 --dry-run`
- 已执行 smoke retry2：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_receiver_exec_r22 --trial-count 1 --disable-confirmatory --epochs 6 --min-epochs 3 --early-stop-patience 3 --study-tag cp_v3_portfolio_daily_ranking_r22_receiver_exec_smoke_20260426_retry2`
- r22 retry2 gate report：
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r22_receiver_exec_smoke_20260426_retry2/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_gate_report.md`
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r22_receiver_exec_smoke_20260426_retry2/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_rescore.csv`
- r22 读取重点：
  - `portfolio_daily_receiver_exec_guarded`
  - `portfolio_daily_receiver_exec_guard_count`
  - `portfolio_daily_receiver_exec_guard_reason`
  - `portfolio_daily_receiver_add_headroom`
  - `portfolio_daily_receiver_min_add_delta`
  - `portfolio_daily_receiver_realized_deploy_rate`
  - `portfolio_daily_receiver_unrealized_deploy_share`
- 若后续继续推进，优先跑 bounded confirmatory；不得把 r22 smoke 的 `qualified_v2_champion` 误写成 promotion，因为 `training_evidence_status = insufficient` 且缺少 fresh confirm。

## 2026-04-27 r23 receiver-exec stability 操作路径
- r23 稳定性搜索 profile：`split_heads_portfolio_daily_ranking_receiver_exec_stability_r23`。
- r23 预算校准标记仍为：`cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
- r23 默认目标仍为：`portfolio_daily_ranking_v2_gated`。
- r23 dry-run 元数据检查命令：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_receiver_exec_stability_r23 --trial-count 1 --study-tag verify_r23_receiver_exec_stability_profile_20260426 --dry-run`
- 已执行 r23 首轮前台验证：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_receiver_exec_stability_r23 --trial-count 3 --epochs 40 --min-epochs 32 --early-stop-patience 10 --confirmatory-max-candidates 2 --confirmatory-epochs 48 --confirmatory-min-epochs 40 --study-tag cp_v3_portfolio_daily_ranking_r23_receiver_exec_stability_20260426`
- 已执行同一 tag strict resume：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_ranking_receiver_exec_stability_r23 --trial-count 3 --epochs 64 --min-epochs 56 --early-stop-patience 12 --confirmatory-max-candidates 2 --confirmatory-epochs 64 --confirmatory-min-epochs 56 --study-tag cp_v3_portfolio_daily_ranking_r23_receiver_exec_stability_20260426`
- r23 gate report：
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r23_receiver_exec_stability_20260426/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_gate_report.md`
  - `daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r23_receiver_exec_stability_20260426/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_rescore.csv`
- 读取 r23 时必须先看：
  - `confirm_stable`
  - `confirm_stability_failed`
  - `annual_return_delta_vs_source`
  - `sharpe_delta_vs_source`
  - `monthly_return_mean_delta_vs_source`
  - `receiver_minus_source_5d`
  - `source_realized_sell_rate`
  - `source_target_not_sold_share`
  - `effective_capital_transfer_count`
- r23 已出现 stable confirmatory，但仍不得自动 promotion；下一轮若继续推进，应以 r23 稳定性配置为锚，尝试恢复 r22 的收益上限，同时保持 `confirm_stable = True`。

## 2026-04-27 r24 listwise allocation 操作路径
- r24 profile：`split_heads_portfolio_daily_listwise_allocation_r24`。
- r24 loss：`alpha_result_value_budget_split_v16`；关键新诊断为 `supports_deploy_executability_head` 与 `supports_portfolio_listwise_heads`。
- r24 关键可买性字段：`portfolio_daily_receiver_add_capacity`、`portfolio_daily_receiver_executability`、`portfolio_daily_receiver_score`、`portfolio_daily_source_score`、`portfolio_daily_cash_score`。
- r24 继承校准：`cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`；默认目标仍为 `portfolio_daily_ranking_v2_gated`。
- dry-run 已执行：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_listwise_allocation_r24 --trial-count 1 --study-tag verify_r24_listwise_allocation_profile_20260427 --dry-run`。
- smoke 已执行：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/continuous_policy/run_self_optimizing_study.py --search-profile split_heads_portfolio_daily_listwise_allocation_r24 --trial-count 1 --epochs 8 --min-epochs 4 --early-stop-patience 4 --confirmatory-max-candidates 1 --confirmatory-epochs 10 --confirmatory-min-epochs 5 --study-tag cp_v3_portfolio_daily_listwise_allocation_r24_smoke_20260427`。
- smoke 产物：`daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_listwise_allocation_r24_smoke_20260427/study_summary.json` 与 `trial_ranking.csv`；训练诊断在 `daily_research/output/continuous_policy/models/cp_v3_portfolio_daily_listwise_allocation_r24_smoke_20260427__trial_01__train/training_diagnostics.json` 和 `...__confirm_01__train/training_diagnostics.json`。
- 后续正式验证必须前台运行、10h 窗口、显式 yolos 解释器；未满足 sufficient evidence 与稳定 confirm 前不得 promotion/live。
