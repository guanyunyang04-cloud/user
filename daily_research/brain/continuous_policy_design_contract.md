# Continuous Policy 设计合同

快照日期：`2026-04-24`

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
- r10 合同：deploy intent 必须能真实形成正向权重变化，不能用 `clip reduction` 代替 deploy executability。
- r11 合同：sell-source 必须从隐式 budget-origin sell 改为显式、可审计、可限制的来源链。
- r11b 合同：held-side release/funding 必须同时满足低 funding 污染、非零 release consistency 与可接受 deploy realization。
- r12 合同：release learning、order translation drift 与 deploy executability 必须联合评分；单点改善不得掩盖三者之间的压力转移。
- 当前已证实：`alpha_result_value_budget_split_v11` 能让 funding-sell 更少、更干净。
- 当前已证实：`alpha_result_value_budget_split_v12 + result_value_v9` 可以在 r12 confirm_02 中拿到更高收益，但不能自动闭合三方语义。
- 当前已证实：`alpha_result_value_budget_split_v13 + result_value_v10` 可以在 r13 confirm_01 中把显性动作价值冲突压到 `action_value_conflict_share = 0.0`、`action_value_consistency_score = 0.92`，但仍未通过 promotion gate。
- 当前未证实：`result_value_v10` 尚不能升为稳定预算目标，`deploy_funding_release_consistent_share` 仍未打通。
- 当前主瓶颈：`order_translation_drift` 正在主导 release / deploy / funding 的压力转移，held-side release learning 仍没有被真实转化为 release consistency。
- 当前已新增：`alpha_result_value_budget_split_v12`、`split_heads_release_translation_deploy_r12`、`release_translation_deploy_v1` 与审计字段 `release_translation_deploy_health_score`。
- 当前已新增：`alpha_result_value_budget_split_v13`、`split_heads_action_value_unification_r13`、`action_value_unification_v1` 与审计字段 `action_value_consistency_score`，把 `open/add/hold/reduce/exit` 放进同一个 `1/3/5/10/20d` 多周期未来价值合同中。

## 当前禁止事项
- 不得把 r11/r11b/r12/r13 任一分支写成 promotion 或 live 切换依据；所有 confirm 仍是 `shadow_only`。
- 不得只看 `budget_origin_sell_share = 0` 就宣布 sell-source 成功。
- 不得只看自动 confirm 排序；`confirm_03_semantic_v11v9` 是当前必须保留的语义对照线。
- 不得并行运行多个会写 latest 行为摘要的审计。
- 不得在 release consistency 仍为 0 时声称 release head 已学成。
- 不得把 r13 的高 `action_value_consistency_score` 解释为可上线；只要 `open_low_action_value_share`、`order_translation_drift` 或 promotion gate 仍失败，就只能作为 shadow 研究证据。

## 成功判定
- 结果层：`annual_return / sharpe / max_drawdown / trend_capture_rate_10d`。
- 结构层：`reduce_success_rate_5d / exit_timeliness_rate_5d / cash_timing_quality_1d / shadow_reversal_rate_3d`。
- 语义层：`semantic_conflict_rate / order_translation_conflict_rate / deploy_intent_realized_rate / deploy_funding_release_consistent_share`。
- 联合层：`release_translation_deploy_health_score` 必须拆开看 `deploy / release / translation / funding / model_release` 组件。
- 动作价值层：`action_value_consistency_score` 必须拆开看 `action_value_conflict_share`、`sell_against_keep_value_share`、`keep_against_release_value_share`、`open_low_action_value_share`，不能只看收益或 Sharpe。
- 责任链：sell-source、funding source、held-side release support 与 protected-hold conflict 必须可审计。

## 下一步方向
- r13 应优先面向 translation / simulator / budget-action 解耦：让 `weight_change_action` 更忠实地表达 `execution_action`，并减少 add -> hold 与 held-side funding trim 的混合污染。
- 继续保留 `cash_constraint_sell_source_guard_v7` 与 held-side detail audit 作为 source attribution 基线。
- 后续若调整 loss / objective / simulator，必须随行补跑 source attribution 与 held-side detail，而不是只看 aggregate metrics。

## 历史归档入口
- 原 `continuous_policy_design_contract.md` 已原样归档：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
- 原始行数：`581`。
- 原始 SHA256：`b3f83530ec585e284a60e568bd66be4f7c8402d455efec20944af41e58f0afe8`。
- 读取纪律：当前设计合同以本文件上方章节为准；r1-r11b 的完整合同演化只作为历史证据。
