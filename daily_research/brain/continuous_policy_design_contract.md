# Continuous Policy 设计合同

快照日期：`2026-04-17`

## 1. 北极星
- 构建一个以日为单位进行连续决策的交易执行模型。
- 该模型直接从市场全局状态、个股演化路径与持仓上下文中学习 `open / hold / add / reduce / exit / cash` 的动态执行。
- 它的目标是在尽量少的人为约束下，综合权衡未来收益、风险与成本，并做出当前条件下最优的动态执行判断。

## 2. 明确非目标
- 不再把“优化固定调仓频率”视为主目标。
- 不再把“优化固定持有周期”视为主目标。
- 不再把“优化人工执行桥接规则”视为主目标。
- 不再把“更像 teacher”视为主目标。

## 3. 绑定原则
- `teacher` 的角色是 `warm start / auxiliary prior / heuristic scaffold`，不是最终老师。
- `cash timing` 不得继续主要依赖 teacher imitation，因为 teacher 自身在该项上并不可靠。
- 个股动作语义与组合预算语义必须分层：
  - 个股层负责 `open / hold / add / reduce / exit` 与生命周期连续性。
  - 组合层负责 `gross / cash / turnover / deployment`。
- 执行层只负责把策略语义翻译成订单与权重，不得改写策略本体语义。
- 任何评价都必须优先使用 execution-aligned 可比口径，禁止混用不同回测语义的 headline 指标。

## 4. 当前已识别的根因
- 当前系统同时在学四件不同层级的事情：
  - 个股是否继续持有
  - 个股是否应小减
  - 个股是否应退出
  - 组合是否应整体提高现金
- 这四件事的监督来源、时间尺度和经济含义并不一致。
- 如果继续把它们混在同一 teacher / gate / budget 脚手架里，模型更容易学到折中和捷径，而不是稳定的收益最优执行。

## 5. 成功判定
- 不只看 `annual_return / sharpe`。
- 不只看 `reduce / exit / cash timing` 是否过局部 gate。
- 必须同时满足三类标准：
  - 结果层：`annual_return_vs_active / sharpe_vs_active / max_drawdown / trend_capture_rate_10d`
  - 结构层：`reduce_success_rate_5d / exit_timeliness_rate_5d / cash_timing_quality_1d / shadow_reversal_rate_3d`
  - 语义层：执行层不再显著改写模型动作，预算层不再系统性吞掉个股动作语义

## 6. 当前阶段的优先顺序
1. 先修 `动作语义 / 预算语义 / 执行翻译` 的分层问题。
2. 再让结果驱动目标接管 `cash timing / drawdown / trend capture`。
3. 最后才扩大模型、搜索面或 universe。

## 7. 当前禁止事项
- 不得再把 screening 上的高收益短期冠军直接当作主线结论。
- 不得在语义冲突仍明显时，把局部结构改善解释成“模型已经学稳”。
- 不得在 action head 与 budget head 仍强耦合时，继续靠扩大搜索面掩盖目标分解错误。

## 8. 执行语义双账本
- `execution_action` 固定表示模型生命周期语义，也就是模型此刻想表达的 `open / hold / add / reduce / exit`。
- `weight_change_action` 固定表示真实权重变化，也就是订单/预算翻译后实际发生的 `open / hold / add / reduce / exit`。
- `semantic_conflict_rate` 衡量执行层是否仍在改写模型策略语义。
- `order_translation_conflict_rate` 衡量真实权重变化是否仍偏离模型动作意图。
- 前者必须尽量接近 0，后者允许存在但必须被预算层解释并逐步压低。

## 9. 预算语义分层口径
- `budget_semantics = legacy_total_candidate` 表示旧口径：候选预算直接裁剪全体目标强度，收益口径当前仍以它为稳定对照。
- `budget_semantics = action_budget_split_v1` 表示分层口径：已有持仓生命周期动作不再被候选预算直接丢弃，候选预算优先约束新开仓名额。
- `budget_calibration = cash_exit_guard_v1` 表示组合层只校准 `gross / candidate / turnover / position_cap`，不得改写个股 `model_action`。
- 当前事实：在 `cp_v3_seq_self_opt_return_recovery_r2__confirm_02` 旧模型同窗评估里，`action_budget_split_v1` 降低了短期反手，但压低了收益和 sell-side 质量，因此它是下一轮训练/ablation 维度，不是当前默认 promotion 口径。
- 当前约束：任何 budget split 升级必须同时报告 `annual_return / sharpe / max_drawdown / reduce_success_rate_5d / exit_timeliness_rate_5d / cash_timing_quality_1d / semantic_conflict_rate / order_translation_conflict_rate`，不得只用语义指标宣布成功。

## 10. 训练级预算分层验收结论
- `cp_v3_budget_layer_ablation_r2` 是当前干净训练级预算分层证据；`cp_v3_budget_layer_ablation_r1` 因用户要求关闭后台训练而含 failed trial，不作为正式结论。
- `action_budget_split_v1 + cash_exit_guard_v1` 在 screening 中可以产生高收益信号，但 64 epoch confirmatory 后仍为 `shadow_only`，因此不得提升为默认。
- `legacy_total_candidate + none` 仍是当前稳定执行锚点，但训练级收益弱，不得被误写为长期最优结构。
- 预算分层的通过条件必须同时满足：
  - `semantic_conflict_rate` 接近 0；
  - `order_translation_conflict_rate` 不显著恶化；
  - `annual_return / sharpe / max_drawdown` 达到 promotion 口径；
  - `exit_timeliness_rate_5d / cash_timing_quality_1d` 不再成为主要失败项。
- 若上述条件继续不满足，下一步应推进结果驱动 `budget/value head` 或接入 `deep_alpha / policy_v5b` alpha prior，而不是继续叠加人工 cash guard。

## 11. 2026-04-18 alpha prior 与 result/value budget r1 合同
- 当前可交付事实：`state_builder.py` 已支持 `alpha_prior_source = none / active_execution_strategy / manifest json / run dir / explicit panel`，并把 active policy_v5b 的 score/target_weight 转成可观测日频状态特征。
- 当前可交付事实：`budget_objective = result_value_v1` 已接入训练目标生成，但保持 daily controller 的既有 5 个输出头不变，避免破坏历史 artifact 兼容性。
- 当前可交付事实：`alpha_result_value_budget_r1` 自优化 profile 已固化为四格对照：无 alpha + teacher、active alpha + teacher、无 alpha + result_value、active alpha + result_value。
- 当前可交付事实：`run_execution_counterfactuals.py` 已提供固定模型下的执行预算语义反事实入口，用来拆分“模型动作能力”和“执行/预算翻译层影响”。
- 推断：下一条最高 ROI 主线不是继续扩大 backbone，也不是继续单独打磨 loss-profile，而是验证 `active_execution_strategy alpha prior + result_value_v1 budget objective` 是否能在语义干净的前提下恢复收益与现金/退出质量。
- 假设：r1 只是让策略学到更好的信用分配入口；是否真正提升收益，仍必须由 10h 正式训练窗口下的 `screening + confirmatory` 结果决定，不能由 smoke 或 dry-run 直接判断。
- 禁止项：不得把 active alpha prior 当成新的人工调仓桥接规则。它只能作为可观测机会先验进入状态/目标，continuous_policy 仍必须学习日频连续执行。
- 正式实证事实：`cp_v3_alpha_result_value_budget_r1` 已完成 4 screening + 2 confirmatory，所有 trial 仍为 `shadow_only`。
- 正式实证事实：screening 冠军是 `active_execution_strategy + result_value_v1`，`annual_return=2.8598`、`sharpe=5.0898`，但 `training_evidence_status=insufficient` 且 `cash_timing_quality_1d=-0.2318`。
- 正式实证事实：confirmatory 中 `active_execution_strategy + result_value_v1` 回落为 `annual_return=-0.0491`、`sharpe=-0.0237`、`cash_timing_quality_1d=-0.3185`，不能 promotion。
- 复盘结论：alpha prior + result/value budget 已证明有 screening 潜力，但当前 r1 没有证明稳定泛化；下一步不得直接升默认，应继续围绕训练证据充分性、cash timing 信用分配和 confirmatory 稳定性推进。
