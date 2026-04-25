# Continuous Policy 设计合同

快照日期：`2026-04-25`

## 北极星
- 构建一个以日为单位进行连续决策的交易执行模型。
- 模型直接从市场全局状态、个股演化路径与持仓上下文中学习 `open / hold / add / reduce / exit / cash`。
- 目标是在尽量少的人为执行桥约束下，综合权衡未来收益、风险、成本与部署可执行性。

## 非目标
- 不把优化固定调仓频率、固定持有周期或人工执行桥作为主目标。
- 不把更像 teacher 当作最终目标；teacher 只是 warm start / auxiliary prior / heuristic scaffold。
- 不用 screening 高收益、短窗 smoke 或单一局部指标替代正式 verdict。

## 当前绑定原则
- 个股动作语义与组合预算语义必须分层：个股层表达生命周期动作，组合层表达 gross / cash / turnover / deployment。
- 执行层只翻译策略语义为权重和订单，不得静默改写策略本体。
- 评价必须使用 execution-aligned 可比口径，禁止混用不同回测语义的 headline 指标。
- `execution_action` 表示模型生命周期语义；`weight_change_action` 表示订单/预算翻译后的真实权重变化。
- `semantic_conflict_rate` 应接近 0；`order_translation_conflict_rate` 允许存在但必须被预算层解释并逐步压低。

## 当前阶段合同
- r10 合同：deploy intent 必须能真实形成正向权重变化，不能用 clip reduction 代替 deploy executability。
- r11 合同：sell-source 必须从隐式 budget-origin sell 改为显式、可审计、可限制的来源链。
- r11b 合同：held-side release/funding 必须同时满足低 funding 污染、非零 release consistency 与可接受 deploy realization。
- r12 合同：`alpha_result_value_budget_split_v12`、`split_heads_release_translation_deploy_r12` 与 `release_translation_deploy_health_score` 必须联合评分 release learning、order translation drift 与 deploy executability。
- r13 合同：`alpha_result_value_budget_split_v13` 与 `action_value_consistency_score` 把 `open/add/hold/reduce/exit` 放进同一个多周期未来价值合同中；高一致性不等于可上线。
- r14 合同：`alpha_result_value_budget_split_v14`、`direct_action_value_mode_share` 和直接动作值可以承担日级动作仲裁，但订单/预算层不得继续把 direct action 隐式改写成其他动作。
- r15 合同：`alpha_result_value_budget_split_v15`、`cash_constraint_direct_action_guard_v8` 与 `direct_action_intent_preserved_share` 必须让 direct action intent 尽量穿透订单/预算翻译层；funding sell 必须有 direct release authorization 或其他可审计证据。
- r16 合同：`cash_constraint_direct_action_reallocation_guard_v9` 只允许在高置信 direct add/open 存在时释放低 continuation、低 keep advantage 的持仓预算；核心成功指标新增 `direct_action_deploy_authorized_realized_rate` 与 `direct_action_add_authorized_realized_rate`。
- r16 边界：可以降低被授权 reallocation source 的 sell-source retention floor，但不得把这种修复解释成 promotion；只要 authorized deploy/add 仍被翻译成 hold，模型仍是 research/shadow。

## 当前已知事实
- 当前已证实：`alpha_result_value_budget_split_v11` 能让 funding-sell 更少、更干净。
- 当前已证实：`alpha_result_value_budget_split_v12 + result_value_v9` 可以在 r12 confirm_02 中拿到更高收益，但不能自动闭合三方语义。
- 当前已证实：`alpha_result_value_budget_split_v13 + result_value_v10` 可以把显性动作价值冲突压到 `action_value_conflict_share = 0.0`，但仍未通过 promotion gate。
- 当前已证实：r14 直接动作价值仲裁能显著改善收益质量，但低边际动作与 held-side funding rebalance 仍未闭合。
- 当前已证实：r15 formal 全窗暴露 `deploy_not_realized`，budget clipping 日的订单翻译冲突显著高于 unclipped 日。
- 当前已证实：r16 smoke2 改善 return/risk 并恢复非零 reallocation source，但 `direct_action_deploy_authorized_realized_rate = 0.1557` 仍明显不足。
- 当前未证实：`result_value_v10` 尚不能升为稳定预算目标，r16 尚不能升为 production 或 promotion 证据。

## 当前禁止事项
- 不得把 r11/r11b/r12/r13/r14/r15/r16 任一分支写成 promotion 或 live 切换依据。
- 不得只看 `budget_origin_sell_share = 0` 就宣布 sell-source 成功。
- 不得只看自动 confirm 排序；必须保留语义对照和 held-side detail 证据。
- 不得并行运行多个会写 latest 行为摘要的审计。
- 不得在 release consistency 或 authorized deploy realization 仍低时声称 release / deploy 已学成。
- 不得把 r16 smoke2 的收益改善解释为可上线；只要 add->hold 冲突和 authorized deploy 未实现仍高，就只能作为修复性 shadow 证据。

## 成功判定
- 结果层：`annual_return / sharpe / max_drawdown / trend_capture_rate_10d`。
- 月度收益层：`monthly_return_mean / monthly_win_rate / monthly_worst_return / monthly_max_consecutive_loss_months / monthly_consistency_score` 必须辅助判断收益质量，避免只看年化或短窗口动作指标。
- 结构层：`reduce_success_rate_5d / exit_timeliness_rate_5d / cash_timing_quality_1d / shadow_reversal_rate_3d`。
- 语义层：`semantic_conflict_rate / order_translation_conflict_rate / deploy_intent_realized_rate / deploy_funding_release_consistent_share`。
- 联合层：`release_translation_deploy_health_score` 必须拆开看 `deploy / release / translation / funding / model_release` 组件。
- 动作价值层：`action_value_consistency_score` 必须拆开看 `action_value_conflict_share`、`sell_against_keep_value_share`、`keep_against_release_value_share`、`open_low_action_value_share`。
- 直接决策层：`direct_action_value_mode_share / direct_action_value_gap_mean / direct_action_value_low_margin_share / direct_action_order_translation_conflict_rate` 必须和月度收益质量一起看。
- 直接翻译层：`direct_action_intent_preserved_share / direct_action_funding_authorized_sell_share / direct_action_funding_protected_sell_share / direct_action_release_advantage_mean` 必须和 `deploy_intent_realized_rate`、held-side detail 及月度收益质量一起看。
- 直接再分配层：`direct_action_deploy_authorized_realized_rate / direct_action_add_authorized_realized_rate / direct_action_reallocation_source_count / add_to_hold_conflict_share` 必须一起看，防止“有授权、有资金源、但真实订单仍不加仓”。
- 责任链：sell-source、funding source、held-side release support、protected-hold conflict 与 reallocation source 必须可审计。

## 下一步方向
- r16 应优先从 smoke 推进到 bounded shadow study，验证直接动作授权、预算再分配和月度收益质量能否稳定共存。
- 后续若调整 loss / objective / simulator，必须随行补跑 source attribution、held-side detail 和 budget-clipped 分层审计。
- 若 r16 bounded study 仍出现高 add->hold 冲突，下一轮应优先改预算层的可成交分配机制，而不是继续叠加动作分类 loss。

## 历史归档入口
- 原 `continuous_policy_design_contract.md` 已原样归档：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
- 原始行数：`581`。
- 原始 SHA256：`b3f83530ec585e284a60e568bd66be4f7c8402d455efec20944af41e58f0afe8`。
- 读取纪律：当前设计合同以本文件上方章节为准；r1-r11b 的完整合同演化只作为历史证据。
