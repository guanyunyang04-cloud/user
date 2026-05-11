# Daily Research 状态中枢

快照日期：`2026-05-11`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- active 执行物化真源：`daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile：`regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root：`daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前执行权重语义：`research_raw_target_weight`；权重上限语义：`follow_research_raw_no_global_cap`。
- continuous_policy 当前仍是 `research / shadow_only`；未过 formal evidence、v2 gate、stable confirm 与 promotion gate 前，不得替代 active 执行链。
- 当前有效研究证据基线仍是 r39 allocation objective consolidation 的 `alpha_result_value_budget_split_v25` 与 `portfolio_daily_ranking_v2_gated`。
- r48 formal screening + confirmatory 已完成但 stable confirm 为空；r49-r52 是 research 入口与结构升级，不是策略有效 verdict。
- r52 native source-delta closure 已修通部分 source 通道，但未完成策略闭合：2/3 screening trials 恢复 source target，3/3 仍为 `training_evidence_status = insufficient`。
- 2026-05-11 已新增 explicit study evidence capsule：`brain_workflow status --workflow continuous_policy --study-tag <tag> --json` 可按指定 study 聚合 study / protocol / training / evaluation 证据；loose `latest_*` 仍显示 r52 study 与 r34 protocol/audit/ledger 不同源，不能直接作为真源。
- r52b 结构入口为 `split_heads_portfolio_daily_day_set_native_target_validity_closure_r52b` / `alpha_result_value_budget_split_v38`；它是 r52 的 native target validity 修复入口，不是 formal verdict。
- r52b safe screening `self_opt_study_r52b_native_target_validity_closure_screening_safe_20260511_01` 已完成 2/3 screening trials、0 failed，并因 resource gate 早停；结论是运行通道可用但 validity 未改善，不能进入 confirmatory 或 22 epoch resume。
- r52c 结构入口为 `split_heads_portfolio_daily_day_set_native_executable_receiver_closure_r52c` / `alpha_result_value_budget_split_v39`；safe screening 已完成 3/3 trials、0 failed，receiver executable closure 在 simulator 边界有效，但仍不是 formal verdict。

## 当前接管入口
- 读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- `episodic_memory.md` 只作为过程复盘入口；历史原文、长命令和标题索引默认进入 `daily_research/brain/references/`。
- 运行 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- Windows 默认设置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- OpenMP 根治验收口径：无 `KMP_DUPLICATE_LIB_OK` 运行 `daily_research/tools/openmp_runtime_check.py --strict` 必须通过。
- PowerShell 出现中文乱码时，先用显式 UTF-8 复读，不得直接判定文档损坏。

## 当前主问题
- production 执行侧不是当前阻塞点；默认 active 继续由 `short_expert_policy_v5b` 承担。
- continuous_policy 的深层瓶颈是组合日资金分配：同一天谁是 receiver、谁是 source、留多少 cash、承受多少 turnover / cost / drawdown。
- r31 / r33 / r34-r39 保留为 receiver/source/cash 合同与证据基线；r40-r52 是 allocation layer 升级链。
- r52c 之后当前最新问题不再是 unsupported receiver validity；瓶颈已转为 deployment / cash timing 闭合、exposure utilization 偏低、source depth 不稳与 formal training evidence 不足。
- r52c safe screening 中 `native_target_valid = 0.975~1.0`，`allocation_layer_native_fallback_used = 0~0.025`，`native_target_invalid_unsupported_receiver_count = 0`；但全部 trials 仍为 `training_evidence_status = insufficient`，`cash_timing_quality_1d < 0`，`portfolio_daily_exposure_utilization ~= 0.33`。
- 下一轮若继续 r52 系列，应优先围绕 deployment / cash timing / exposure utilization / source breadth 的 native target feedback 与 resource gate 设计 r52d；不得回到 receiver mask 修补，也不得直接加长训练资源。

## 近期研究索引
- r31：`split_heads_portfolio_daily_receiver_semantic_closure_r31`，保留 receiver executable closure 合同。
- r31 审计 marker：`direct_action_authorization_subset_violation_count`、`authorized_add_no_weight_change_share`、`deploy_intent_unrealized_share`。
- r33：`split_heads_portfolio_daily_source_forward_proxy_r33`，保留 `portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction` 与 `portfolio_daily_source_distribution_clean_pass`。
- r34：`split_heads_portfolio_daily_allocation_breadth_r34`，保留 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth` 与 `portfolio_daily_joint_economic_quality_gate`。
- r35：`split_heads_portfolio_daily_unified_allocation_r35`，保留 `portfolio_daily_unified_allocation_objective`。
- r36-r39：risk-aware / decision-focused / source hard-negative / allocation objective consolidation 是当前有效证据链，保留 `portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty`、`portfolio_daily_receiver_source_spread_reward`。
- r39 当前仍对应 `portfolio_daily_ranking_v2_gated` 与 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15` 证据基线。
- r40-r48：end-to-end allocation layer、convex/OPE 方向可用但未过 stable confirm。
- r49-r52：capital-flow closure、true solver 入口、native allocation vector 与 day-set native allocation vector 均为 research / shadow；`source/receiver/cash listwise allocation teacher` 仍只是 teacher / warm start。

## 当前优先级
- P0：冻结 live / active artifact，只在 research / shadow 范围推进。
- P1：保持主分脑入口精炼，dated log 必须进入 `references/` archive。
- P2：继续把 receiver 可执行性、source 分布质量、monthly return、exposure utilization、realized deploy、cash timing 与 drawdown 写入 objective / feedback / gate。
- P3：不再优先追加单边 source/cash/reduce guard；优先推进统一 allocation objective / native allocation vector 的结构闭合。
- P4：r50 true solver 保留为 research 入口，但因本机负荷过高，不作为当前默认长训路径。

## 当前边界
- formal、recent、promotion、live 不得混写。
- smoke、dry-run、short-window check、repaired confirm、insufficient evidence 都不能升级为正式 verdict。
- `receiver_unrealized_deploy_share = 0`、`source_positive_forward_sell_share = 0` 或 source 通道恢复都只是必要条件，不是完成态。
- 不得用 safe screening 或代码合同结果改 `active_execution_strategy.json`。

## 当前风险
- 继续堆 simulator guard 会让系统退回“翻译器补漏洞”，不是学习组合资金分配。
- 放宽 source clean-pass 容易从 source dormant 退回强势误卖。
- 只加训练资源可能掩盖 native target 约束问题，并再次触发本机资源风险。
- 主文档若继续按日期堆积，接管会被历史细节淹没。

## 历史归档入口
- 本文件归档前完整快照：`daily_research/brain/references/state_center_archive_20260510.md`。
- 早期状态原文：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 早期状态索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- 过程复盘归档：`daily_research/brain/references/episodic_memory_archive_20260510.md`。
- 知识中枢归档：`daily_research/brain/references/knowledge_center_archive_20260510.md`。
- 操作中枢归档：`daily_research/brain/references/operations_center_archive_20260510.md`。
- 早期历史原文与索引仍保留在 `daily_research/brain/references/*_history_raw_*.md` 与 `*_evidence_index_*.md`。
## 2026-05-11 r52c Safe Screening Latest Verdict
- study tag: `self_opt_study_r52c_native_executable_receiver_closure_screening_safe_20260511_01` completed 3/3 trials, 0 failed, confirmatory disabled.
- receiver executable closure is effective in this round: `native_target_valid=0.975~1.0`, `allocation_layer_native_fallback_used=0~0.025`, `native_target_invalid_unsupported_receiver_count=0`.
- remaining blockers are no longer unsupported receiver validity; now the bottleneck is deployment/cash timing closure with weak utilization.
- stop conditions still triggered: `training_evidence_status=insufficient`; one trial has `source_target_count=2<3`; all trials have `cash_timing_quality_1d<0`; `portfolio_daily_exposure_utilization~0.33`.
- decision boundary: do not enter confirmatory and do not start 22-epoch resume on r52c; keep this branch as research/shadow-only.
