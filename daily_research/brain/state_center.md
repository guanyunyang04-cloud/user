# Daily Research 状态中枢

快照日期：`2026-04-24`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- 当前 active 执行物化真源为 `daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 为 `regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前 continuous_policy 研究主线已推进到 r14 直接动作值仲裁，并已完成 bounded shadow study 与 repaired confirm。
- r11/r11b/r12/r13/r14 全部分支仍是 `shadow_only` 或 research 入口；不得把 `result_value_v10`、`alpha_result_value_budget_split_v11/v12/v13/v14` 或任何 confirm 分支直接解释成 promotion / live 切换证据。
- 本轮已完成历史归档压缩、r12/r13/r14 shadow study、月度收益评价接入与 r14 repaired confirm 写回；未切换 live，未改写 promotion gate。

## 当前接管入口
- 默认读取顺序仍为：`identity_layer.md -> state_center.md -> knowledge_center.md -> operations_center.md -> governance_layer.md`。
- `episodic_memory.md` 只作为过程复盘入口；历史原文和长命令默认进入 `daily_research/brain/references/`。
- 运行 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，并设置 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- PowerShell 出现中文乱码时，先用显式 UTF-8 复读，不得直接判定文档损坏。

## 当前研究状态
- strongest-model 主线已收口，不是当前阻塞点；learned-control 的生产执行仍以 `policy_v5b` active artifact 为准。
- continuous_policy 当前已具备训练、评估、shadow continuity、导出、行为审计与 study scoring 闭环。
- r10 证明 deploy intent 可执行性可以被度量和约束，但不能只用 clip reduction 代替 deploy executability。
- r11 证明 sell-source contract 可以进入训练侧，但 `result_value_v10` 本身尚未成为稳定预算目标。
- r11b 证明 `alpha_result_value_budget_split_v11` 能把 funding-sell 推向更少、更干净，但 `deploy_funding_release_consistent_share` 仍为 `0.0`，且 deploy/order translation 仍会互相拉扯。
- 当前真正瓶颈收敛为：`held-side release learning + order translation drift + deploy executability` 三方耦合。
- 已新增 `split_heads_release_translation_deploy_r12`、`release_translation_deploy_v1`、`alpha_result_value_budget_split_v12` 与 `release_translation_deploy_health_score`，用于把上述三方耦合变成可搜索、可审计的 shadow 研究目标。
- `split_heads_action_value_unification_r13` 已完成 `cp_v3_action_value_unification_r13__study_r1` 正式 bounded shadow study：4 个 screening、2 个 confirmatory 均完成；最终 champion 为 `confirm_01 = alpha_result_value_budget_split_v13 + result_value_v10`，但仍是 `shadow_only`，不改变 live / promotion。
- r13 关键事实：`confirm_01` 的 `action_value_consistency_score = 0.92`、`action_value_conflict_share = 0.0`、`sell_against_keep_value_share = 0.0`，说明动作价值统一显著压低了显性动作冲突；但 `annual_return = 0.4522`、`sharpe = 1.6735`、`max_drawdown = -0.0827`、`open_low_action_value_share = 1.0`、`failure_mode = order_translation_drift`，说明尚未达到 promotion 级结果。
- 已新增月度收益评价合同：`monthly_return_mean`、`monthly_win_rate`、`monthly_worst_return`、`monthly_consistency_score` 与月度收益明细 CSV，用于补足年化/日级指标看不清的持续性、最差月和亏损月连贯性问题。
- `split_heads_direct_action_value_r14` 的 `cp_v3_direct_action_value_r14__study_r1` 已完成 4 个 screening；自动 confirm 最初因 protocol 写出阶段 `[Errno 22] Invalid argument` 失败，后用原口径 strict resume 修复 `confirm_01`，并用 65 epoch 严格续跑补齐 `confirm_02` 对照。当前 r14 champion 为 `confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9`，但仍是 `shadow_only`。
- r14 关键事实：`confirm_01` 的 `annual_return = 1.0904`、`sharpe = 2.8082`、`max_drawdown = -0.1326`、`monthly_return_mean = 0.0587`、`monthly_win_rate = 0.75`、`monthly_worst_return = -0.0797`、`monthly_consistency_score = 0.7722`、`direct_action_value_mode_share = 1.0`；但 `direct_action_value_low_margin_share = 0.7372`、`direct_action_order_translation_conflict_rate = 0.2867`、`reduce_success_rate_5d = 0.0`、`failure_mode = order_translation_drift`。
- r14 held-side detail：`confirm_01` 有 135 条 held-side sell 事件，其中 134 条来自 `deploy_funding_rebalance`、1 条来自 `model_release_signal`；18 条为 `protected_hold_conflict`，13 条为 `release_consistent`，主要集中在 `002371.SZ / 000333.SZ / 002028.SZ`。结论是直接动作仲裁显著改善收益，但 release/funding translation 尚未闭合。
- `verify_release_translation_deploy_r12_dryrun_20260424` 已证明 r12 study plan 可生成；该 dry-run 未训练、未生成正式 verdict。
- `cp_v3_release_translation_deploy_r12__study_r1` 已完成：4 个 screening、2 个 confirmatory、0 失败；最终 champion 为 `confirm_02 = alpha_result_value_budget_split_v12 + result_value_v9`，但仍为 `shadow_only`。
- r12 关键结论：收益与 Sharpe 可以被推高，但 `release_translation_deploy_health_score = 0.3235`、`failure_mode = order_translation_drift`、`deploy_funding_release_consistent_share = 0.0`，说明 release/translation/deploy 三方闭环仍未完成。
- held-side detail 显示 `confirm_02` 有 119 条 funding sell，全部来自 `deploy_funding_rebalance`；其中 `002371.SZ / 002256.SZ / 002157.SZ` 最集中，`protected_hold_conflict = 14`。

## 当前优先级
- 冻结 live 默认执行，不做静默切换。
- 继续把 r11/r11b/r12/r13/r14 视为 research / shadow 证据，而不是 production 证据。
- 后续若继续推进 continuous_policy，应优先进入 r15 direct-action-preserving translation / release-funding repair，处理 `order_translation_drift`、direct action 低边际与 budget/action entanglement；单纯继续调高 release loss 已不是最高优先级。
- 不回退到 simulator-only 解释，不把单一 aggregate share 当作结论；需要逐仓明细时读取 held-side detail audit。
- `analyze_behavior_gap.py` 不得并行运行多个会写 `latest_behavior_audit_summary.json` 的实例。

## 当前边界
- formal、recent、promotion、live 不得混写。
- 不足正式证据的 smoke / dry-run / short-window check 不能升级为正式 verdict。
- continuous_policy 只有在正式协议、参考对照、连续 shadow continuity、行为语义和 promotion gate 均稳定后，才允许进入 promotion 讨论。
- 当前任何 r11/r11b/r12/r13/r14 结果都不改变 active execution artifact。
- r12 虽有 `annual_return = 1.7484`、`sharpe = 3.0857` 的 confirm_02，但 `max_drawdown = -0.1756`、`reduce_success_rate_5d = 0.0`、`exit_timeliness_rate_5d = 0.25`、`order_translation_conflict_rate = 0.3862`，不得进入 promotion 讨论。

## 当前风险
- 若身份层再次写入具体 live 默认、最新分数或 winner，属于文档职责漂移。
- 若状态中枢继续堆叠日期段，后续接管会重新退化为长日志扫描。
- 若只看 r11b 自动 confirm，会错过 `confirm_03_semantic_v11v9` 这条重要语义对照线。
- 若并行运行行为审计，仍可能重现 latest 摘要文件竞争。

## 推荐下一步
- 若继续研究：围绕 r15 direct-action-preserving translation / release-funding repair 设计，不再把 release consistency 失败单独归因给 release loss 不够。
- r14 已证明“日级连续决策直接由动作值仲裁”有价值；下一步应让订单/预算翻译层尽量保留 direct action intent，并专项降低 `direct_action_value_low_margin_share`、`direct_action_order_translation_conflict_rate` 与 held-side funding rebalance 过度依赖。
- 若继续维护：优先保持入口文档轻量，把过程证据写入 `episodic_memory.md` 或 `brain/references/`。
- 若需要旧状态细节：按下方索引读取历史原文，不把归档历史自动提升为当前状态。

## 历史归档入口
- 原 `state_center.md` 已原样归档：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- 原始行数：`1863`。
- 原始 SHA256：`1c135d58f6e1951cf60c8bd2234522ccc5357755953f6962d9070ec67b7a1e16`。
- 读取纪律：当前状态以本文件上方章节为准；归档文件只作为历史证据与追溯入口。
