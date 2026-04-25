# Daily Research 状态中枢

快照日期：`2026-04-25`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- 当前 active 执行物化真源为 `daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 为 `regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- continuous_policy 已从 r15 direct-action-preserving translation 推进到 r16 direct-action reallocation repair；r16 入口为 `split_heads_direct_action_reallocation_r16`、`cash_constraint_direct_action_reallocation_guard_v9`、`direct_action_reallocation_v1`。
- r15 正式 bounded study 已完成，champion 为 `confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9 + cash_constraint_direct_action_guard_v8`，但 `promotion_status = shadow_only`，不得切换 live。
- r15 事实：收益尚可但执行语义失败，`annual_return = 0.3816`、`sharpe = 1.0867`、`max_drawdown = -0.1493`、`monthly_consistency_score = 0.6703`、`deploy_intent_realized_rate = 0.1642`、`add_to_hold_conflict_share = 0.8524`、`failure_mode = deploy_not_realized`。
- r16 当前最佳 smoke2 事实：复用 r15 champion artifact 后，`annual_return = 0.5807`、`sharpe = 1.6261`、`max_drawdown = -0.1180`、`monthly_return_mean = 0.0337`、`monthly_consistency_score = 0.7247`，但 `direct_action_deploy_authorized_realized_rate = 0.1557`、`add_to_hold_conflict_share = 0.8483` 仍未解决。
- r11/r11b/r12/r13/r14/r15/r16 全部仍是 `shadow_only` 或 research 证据；不得把 smoke、dry-run、confirm 或 repaired confirm 直接解释成 promotion / live 切换依据。

## 当前接管入口
- 默认读取顺序仍为：`identity_layer.md -> state_center.md -> knowledge_center.md -> operations_center.md -> governance_layer.md`。
- `episodic_memory.md` 只作为过程复盘入口；历史原文和长命令默认进入 `daily_research/brain/references/`。
- 运行 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，并设置 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- 涉及 PyTorch / MKL / OpenMP 的训练、评估或审计命令，按既有口径设置 `KMP_DUPLICATE_LIB_OK=TRUE`。
- PowerShell 出现中文乱码时，先用显式 UTF-8 复读，不得直接判定文档损坏。

## 当前研究状态
- strongest-model 主线已收口，不是当前阻塞点；learned-control 的生产执行仍以 `policy_v5b` active artifact 为准。
- continuous_policy 当前已具备训练、评估、shadow continuity、导出、行为审计与 study scoring 闭环。
- 月度收益评价已接入通用曲线指标与 study ranking；重点看 `monthly_return_mean`、`monthly_win_rate`、`monthly_worst_return`、`monthly_max_consecutive_loss_months`、`monthly_consistency_score`，并读取 `monthly_returns.csv` 或 `shadow_monthly_returns.csv` 明细。
- `split_heads_release_translation_deploy_r12` 与 `release_translation_deploy_v1` 证明 release / translation / deploy 三方必须联合看；r12 champion 仍因 `order_translation_drift` 保持 shadow。
- `split_heads_action_value_unification_r13` 证明统一动作价值能压低显性冲突，但 `open_low_action_value_share` 与 translation drift 未闭合。
- `split_heads_direct_action_value_r14` 证明直接日级动作仲裁有收益价值，但 held-side funding rebalance 仍过度依赖订单/预算翻译。
- `split_heads_direct_action_translation_r15` 短窗 smoke 曾显示 direct action intent 可以被保留，但正式 study 暴露全窗 budget clipping 后 deploy/add 仍被翻译成 hold。
- `split_heads_direct_action_reallocation_r16` 的本质定位是修复 r15 的资金再分配路径：当 direct add/open 有授权时，允许从低 continuation、低 keep advantage 的持仓释放预算，并降低相关 sell-source retention floor。
- r16 smoke2 修复了“没有可用 reallocation source / sell-source floor 锁死当前仓位”的结构问题，`direct_action_reallocation_source_count = 105`；但高置信 add/open 真实成交率仍低，核心执行学习问题没有结束。

## 当前优先级
- 冻结 live 默认执行，不做静默切换。
- 把 r16 作为 `shadow_only` 修复基线，而不是 promotion 候选。
- 若继续研究，应优先做 r16 bounded shadow study，并把 `direct_action_deploy_authorized_realized_rate`、`direct_action_add_authorized_realized_rate`、`add_to_hold_conflict_share`、budget clipping 分层冲突和月度收益质量作为主判据。
- 不回退到 simulator-only 解释，不把单一 aggregate share 当作结论；需要逐仓明细时读取 held-side detail audit。
- `analyze_behavior_gap.py` 不得并行运行多个会写 `latest_behavior_audit_summary.json` 的实例；长任务按阻塞等待完成，不做无意义轮询。

## 当前边界
- formal、recent、promotion、live 不得混写。
- 不足正式证据的 smoke / dry-run / short-window check 不能升级为正式 verdict。
- continuous_policy 只有在正式协议、参考对照、连续 shadow continuity、行为语义和 promotion gate 均稳定后，才允许进入 promotion 讨论。
- 当前任何 r11/r11b/r12/r13/r14/r15/r16 结果都不改变 active execution artifact。
- r16 虽改善收益、Sharpe、回撤和月度一致性，但只要 authorized deploy/add realization 仍低、add -> hold 冲突仍高，就不得进入 promotion 讨论。

## 当前风险
- 若身份层再次写入具体 live 默认、最新分数或 winner，属于文档职责漂移。
- 若状态中枢继续堆叠日期段，后续接管会重新退化为长日志扫描。
- 若只看 r16 smoke2 的收益改善，会掩盖 direct add/open 仍未成交的本质问题。
- 若并行运行行为审计，仍可能重现 latest 摘要文件竞争。

## 推荐下一步
- 若继续研究：以 `split_heads_direct_action_reallocation_r16` 和 `direct_action_reallocation_v1` 做 bounded shadow study，先 dry-run 再 screening/confirm。
- 若继续修复：优先解决“authorized deploy/add 预算落地”而不是单纯提高 release loss；重点检查 budget clipping 日、add->hold 冲突样本和 reallocation source 的逐仓收益/风险质量。
- 若继续维护：保持入口文档轻量，把过程证据写入 `episodic_memory.md` 或 `brain/references/`。
- 若需要旧状态细节：按下方索引读取历史原文，不把归档历史自动提升为当前状态。

## 历史归档入口
- 原 `state_center.md` 已原样归档：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- 原始行数：`1863`。
- 原始 SHA256：`1c135d58f6e1951cf60c8bd2234522ccc5357755953f6962d9070ec67b7a1e16`。
- 读取纪律：当前状态以本文件上方章节为准；归档文件只作为历史证据与追溯入口。
