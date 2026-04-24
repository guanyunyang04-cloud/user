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

## 12. 2026-04-18 split heads 与 result/value budget v2 合同
- 当前可交付事实：`model_seq_v3.py` 已落地 `daily_head_layout = monolithic_v1 / split_v2`；`split_v2` 将 daily controller 拆为 exposure / deployment / lifecycle / signal 四类子头，不再把全部预算与生命周期语义压进单一 5 维头。
- 当前可交付事实：teacher/global daily target 已从原 5 项扩展为 8 项控制目标与 4 项辅助预算信号，新增 `reduce_bias_target / exit_patience_target / reentry_guard_target` 以及 `budget_risk/deploy/cash_timing/alpha_focus_signal_target`。
- 当前可交付事实：`budget_objective = result_value_v2` 已接入 pipeline，并显式把 `cash_timing_score / reentry_guard_score / opportunity_concentration / risk_deploy_gap` 写入训练目标与评估审计。
- 当前可交付事实：新的 study profile `split_heads_cash_timing_r1` 已固化到 `run_self_optimizing_study.py`，用于在 `split_v2` 结构下做 `teacher_imitation / result_value_v2` 与 `alpha prior on/off` 的四格对照。
- 当前可交付事实：评估指标已新增 `budget_model_risk_timing_quality_1d / budget_model_deploy_timing_quality_1d / budget_model_cash_timing_quality_1d / budget_model_risk_deploy_gap_quality_1d`，允许把生命周期学习与预算学习分开审计。
- smoke 事实：`cp_split_heads_cash_timing_smoke_train_r2` 已完成训练，artifact 明确记录 `daily_head_layout=split_v2`、`loss_profile=alpha_result_value_budget_split_v2`、`budget_objective=result_value_v2`、`alpha_prior_source=active_execution_strategy`。
- smoke 事实：`cp_split_heads_cash_timing_smoke_eval_r2` 与 `r3` 已证明 split head 链路可端到端运行，且 `avg_semantic_conflict_rate=0.0`，说明模型级动作/预算拆分没有重新引入旧式语义改写。
- smoke 事实：尽管语义保持干净，`cash_timing_quality_1d` 仍接近 0 或为负，且 `immediate_reversal_rate_3d / reversal_after_reduce_3d_rate / order_translation_conflict_rate` 仍偏高，说明真正瓶颈已前移到现金时机与执行翻译信用分配，而非“是否做了分头”本身。
- 推断：`split_v2 + result_value_v2` 已把“结构化分工”从理念推进到可训练实现，但它只是修好了学习接口，并没有自动解决 cash timing 的长期价值学习。
- 决策：下一条主线可以正式围绕 `split_heads_cash_timing_r1` 开展 10h 前台 study，但在 confirmatory 证据充分前，不得把 split smoke 的轻微收益或语义干净直接解释为机制完成。
- 正式实证事实：`cp_v3_split_heads_cash_timing_r1` 已完成 4 screening + 2 confirmatory，最佳 screening 与 confirmatory 均来自 `active_execution_strategy + result_value_v2 + split_v2`。
- 正式实证事实：冠军 `cp_v3_split_heads_cash_timing_r1__confirm_01` 取得 `annual_return=0.9918`、`sharpe=3.4369`、`max_drawdown=-0.0536`，说明分头结构没有牺牲基础收益能力。
- 正式实证事实：同一冠军仍为 `shadow_only`，失败项只有两个但都关键：`training_evidence_status=insufficient` 与 `cash_timing_quality_1d=-0.1861`。
- 正式实证事实：行为审计显示 `budget_model_cash_timing_quality_1d=-0.1775`，说明 cash timing 的问题已经明确落在 budget/value 学习本体，而不再主要是执行层语义污染。
- 正式实证事实：语义冲突已被压到低位，`semantic_conflict_rate=0.0108`；但 `order_translation_conflict_rate=0.2366` 仍偏高，说明生命周期动作与真实权重翻译仍未完全对齐。
- 复盘结论：`split_v2 + result_value_v2 + active alpha prior` 是当前最合理、最有潜力的主线，但它证明的是“结构修正方向正确”，而不是“cash timing 已经学会”。后续主攻点必须是 budget credit assignment 与 translation drift，而不是回退到 monolithic 或 legacy 收益表象。

## 13. 2026-04-18 translation guard v2 与 result/value budget r3 合同
- 当前可交付事实：`portfolio_simulator.py` 已新增 `budget_calibration = cash_translation_guard_v2`，执行层开始显式约束 `hold` 不再静默变成加减仓，`open/add/reduce/exit` 必须在真实权重变化上留下可审计痕迹。
- 当前可交付事实：`pipeline_utils.py` 已新增 `budget_objective = result_value_v3`，并强化 `cash_timing_score` 的 shaping，使其更直接对齐 `cash_timing_quality_1d` 所代表的风险回避目标。
- 当前可交付事实：`model_seq_v3.py` 已新增 `alpha_result_value_budget_split_v3` loss profile，并在推理阶段对 `result_value_v3` 的 budget signal 做专门融合，避免训练目标被旧融合逻辑稀释。
- 当前可交付事实：新的正式 study `cp_v3_split_heads_cash_timing_r2` 已完成 4 screening + 2 confirmatory；其核心只比较 `cash_exit_guard_v1 / cash_translation_guard_v2` 与 `result_value_v2 / result_value_v3`，固定 `active alpha + split_v2`。
- 正式实证事实：screening 最优是 `cash_translation_guard_v2 + result_value_v3`，`annual_return=1.2183`、`sharpe=3.1359`、`order_translation_conflict_rate=0.0534`，说明 translation guard 明显压低了权重翻译漂移。
- 正式实证事实：但 `result_value_v3` 在 confirmatory 中没有稳住，`cp_v3_split_heads_cash_timing_r2__confirm_01` 退化为 `annual_return=0.1101`、`sharpe=0.5080`、`max_drawdown=-0.1766`，不能作为新主线。
- 正式实证事实：最终 confirmatory 冠军变为 `cash_translation_guard_v2 + result_value_v2`，`annual_return=1.3904`、`sharpe=3.7228`、`max_drawdown=-0.0709`、`order_translation_conflict_rate=0.0801`，相对 r1 的 `0.2366` 有显著改善，但 `cash_timing_quality_1d=-0.1555` 仍未过线。
- 正式实证事实：behavior audit 显示当前新增高优先级瓶颈是 `reduce_too_early_or_wrong_side`，同时 `budget_action_entanglement` 仍存在，即预算/换手约束仍会放大动作层偏离。
- 复盘结论：这一轮真正被证明有效的是 `translation_guard_v2`，不是 `result_value_v3`。也就是说，执行翻译层的语义约束继续往正确方向推进，但 cash timing 的训练目标本体还没有稳定泛化。
- 决策：后续主线应保留 `cash_translation_guard_v2`，并回到 `result_value_v2` 基础上继续做 cash/reduce 的信用分配；不要把 `result_value_v3` 的 screening 表现误写成 confirmatory 成功。
## 14. 2026-04-19 sell attribution 与 result/value budget v4 合同
- 当前可交付事实：`label_builder.py` 已把真实 `forward_benchmark_return_1d/3d/5d/10d/20d` 写入训练 label frame，并把这些列列入 label-only 字段，避免未来信息进入模型特征。
- 当前可交付事实：训练标签已新增 `sell_attribution_score`，用于刻画持仓中“该卖谁”的未来归因，降低把组合风险收缩平均摊到所有持仓上的错误信用分配。
- 当前可交付事实：`pipeline_utils.py` 已新增 `budget_objective = result_value_v4`，并把真实 forward benchmark downside、held sell pressure、sell selection pressure 接入预算目标；行为审计新增 `sell_selection_quality_5d / sell_selection_hit_rate_5d`。
- 当前可交付事实：`portfolio_simulator.py` 已新增 `budget_calibration = cash_translation_sell_guard_v3`，在组合需要收缩时优先按 `sell_attribution_score` 缩减高卖出归因持仓，而不是对持仓做无差别比例压缩。
- 当前可交付事实：`model_seq_v3.py` 已新增 `alpha_result_value_budget_split_v4`，并在推理阶段输出可审计的 `sell_attribution_score`。
- 正式实证事实：`cp_v3_split_heads_cash_timing_r3` 已完成 4 screening + 2 confirmatory，全部仍为 `shadow_only`，`latest_state_restored=true`。
- 正式实证事实：r3 confirmatory 冠军为 `cp_v3_split_heads_cash_timing_r3__confirm_01 = active_execution_strategy + split_v2 + result_value_v2 + cash_translation_sell_guard_v3 + alpha_result_value_budget_split_v4`，指标为 `annual_return=0.5762`、`sharpe=2.2604`、`max_drawdown=-0.1116`、`cash_timing_quality_1d=-0.0945`、`semantic_conflict_rate=0.0000`、`order_translation_conflict_rate=0.0597`。
- 正式实证事实：`result_value_v4` 对应的 confirmatory 分支 `confirm_02` 取得 `annual_return=0.1565`、`sharpe=0.8738`、`max_drawdown=-0.0906`、`cash_timing_quality_1d=-0.0762`、`exit_timeliness_rate_5d=0.5455`，收益与稳定性不足以替代 `result_value_v2`。
- 行为审计事实：r3 冠军 `sell_selection_quality_5d=0.0001`，而 teacher recomputed 为 `0.2531`；`wrong_side_reduce_share=0.5128`，说明模型仍没有稳定学会“该减谁/该退谁”。
- 行为审计事实：r3 冠军 `budget_model_cash_timing_quality_1d=-0.0990`，`cash_timing_quality_1d=-0.0945`，说明 cash timing 的泛化仍未过线。
- 复盘结论：r3 把旧问题从“执行层改写动作语义”进一步推进为“卖出对象选择与现金时机的信用分配失败”。语义层已经基本干净，主要矛盾不再是 `semantic_conflict_rate`，而是 sell-side rank learning、cash/deployment timing 与预算剪裁下的动作偏离。
- 决策：不 promotion `cp_v3_split_heads_cash_timing_r3`；不让 `result_value_v4` 替代 `result_value_v2`；`cash_translation_sell_guard_v3` 与 `sell_attribution_score` 保留为候选机制，但必须通过下一轮更稳的 sell-selection/cash objective 验证后才可升主线。
- 下一轮约束：不要继续单纯扩大模型或延长同一目标训练；优先设计 `result_value_v4b` 或等价目标，把 `sell_selection_quality_5d`、错边 reduce、现金机会成本和部署下限同时纳入训练/审计闭环。
## 15. 2026-04-19 sell attribution head 与 deployment opportunity floor v4b 合同
- 当前可交付事实：`pipeline_utils.py` 已新增 `budget_objective = result_value_v4b`，在 v4 的 sell-side attribution 基础上加入 `forward_benchmark_upside / opportunity_cost_pressure / deployment_floor_pressure`，避免模型把风险控制误学成长期低暴露。
- 当前可交付事实：`model_seq_v3.py` 已新增 `alpha_result_value_budget_split_v4b` loss profile 与独立 `sell_attribution_head`，并让动作 soft target 在高卖出归因样本上显式压低 hold/add、提高 reduce/exit。
- 当前可交付事实：推理阶段已支持 v4b 专用 deployment floor：当 alpha focus、deploy signal 与机会成本较高时，提高 `min_gross_exposure_target / min_candidate_budget / min_position_cap_target`，防止 cash head 过度防守吞掉机会。
- 当前可交付事实：`run_self_optimizing_study.py` 已新增 `split_heads_cash_timing_r4` profile，固定 `active_execution_strategy + split_v2 + alpha_result_value_budget_split_v4b`，只比较 `cash_translation_guard_v2 / cash_translation_sell_guard_v3` 与 `result_value_v2 / result_value_v4b`。
- smoke 事实：`cp_split_heads_cash_timing_smoke_train_r5` 训练成功，artifact 记录 `supports_sell_attribution_head=true`，且 v4b 诊断中 `result_value_opportunity_cost_pressure=0.4621`、`result_value_deployment_floor_pressure=0.5521`。
- smoke 事实：加入推理 deployment floor 后，`cp_split_heads_cash_timing_smoke_eval_r7` 相比修复前把 `avg_gross_exposure` 从 `0.1351` 拉到 `0.1960`，`sell_selection_quality_5d=0.0165`，但 `cash_timing_quality_1d=-0.1155` 仍为负。
- 正式实证事实：`cp_v3_split_heads_cash_timing_r4` 已完成 4 screening + 2 confirmatory，运行约 73 分钟，所有 trial/confirm 仍为 `shadow_only`，`latest_state_restored=true`。
- 正式实证事实：r4 performance champion 为 `cp_v3_split_heads_cash_timing_r4__confirm_01 = result_value_v2 + cash_translation_guard_v2 + alpha_result_value_budget_split_v4b`，指标为 `annual_return=1.1745`、`sharpe=3.0222`、`max_drawdown=-0.0911`、`avg_gross_exposure=0.8057`、`cash_timing_quality_1d=-0.0414`、`semantic_conflict_rate=0.0147`、`order_translation_conflict_rate=0.0772`。
- 正式实证事实：r4 stability/sell-side champion 为 `cp_v3_split_heads_cash_timing_r4__confirm_02 = result_value_v2 + cash_translation_sell_guard_v3 + alpha_result_value_budget_split_v4b`，指标为 `annual_return=0.2190`、`sharpe=1.2970`、`max_drawdown=-0.0690`、`reduce_success_rate_5d=0.5429`、`exit_timeliness_rate_5d=0.6190`、`sell_selection_quality_5d=0.0228`、`cash_timing_quality_1d=-0.0673`、`order_translation_conflict_rate=0.1749`。
- 行为审计事实：r4 confirm_01 仍触发 `sell_selection_not_learned / cash_timing_not_learned / budget_action_entanglement`，`sell_selection_quality_5d=0.0036`、`wrong_side_reduce_share=0.5526`，说明高收益分支没有真正学会“该卖谁”。
- 行为审计事实：r4 confirm_02 未触发 `sell_selection_not_learned`，`sell_selection_hit_rate_5d=0.6757`、`exit_timeliness_rate_5d=0.6190`，但收益显著低且 `order_translation_conflict_rate=0.1749`，说明 sell guard 能改善卖出质量，却会通过预算剪裁/动作漂移牺牲收益。
- 复盘结论：v4b 证明“机会成本 + 部署下限”能修复一部分过度现金化，并把 `cash_timing_quality_1d` 从 r2 的 `-0.1555`、r3 的 `-0.0945` 收窄到 r4 performance champion 的 `-0.0414`；但 direct sell head 只在部分分支生效，尚未成为稳定收益机制。
- 本质判断：当前矛盾不是“模型不够强”或“信号完全没进来”，而是生命周期动作仲裁仍缺失。模型能预测一部分卖出归因，但在 add/hold/reduce/exit 同时竞争、预算剪裁介入时，sell-side ranking 会被部署收益与翻译约束覆盖。
- 决策：不 promotion `cp_v3_split_heads_cash_timing_r4`；不让 `result_value_v4b` 直接替代 `result_value_v2`；r4 保留为 cash timing 进展证据与 sell-side 分支诊断，不覆盖 r2 的当前收益主线。
- 下一轮约束：优先做 lifecycle action arbitration，而不是继续调 v4b 权重。下一步应把 `sell_attribution_score` 升级为持仓内 pairwise/rank 约束，并加入 clipped-intent loss，使模型在训练时看见预算剪裁后的动作漂移成本。

## 16. 2026-04-19 lifecycle arbitration r5 合同
- 当前可交付事实：`label_builder.py` 已把 `sell_attribution_score` 升级为可排序的 `sell_rank_score`，并新增 `lifecycle_sell_gate / clipped_intent_risk / holding_flag_target`，用于把“该卖哪只”和“执行剪裁会不会扭曲意图”写入训练目标。
- 当前可交付事实：`model_seq_v3.py` 已新增 `alpha_result_value_budget_split_v5`、`sell_rank_head`、`lifecycle_sell_gate_head`、`clipped_intent_head`，以及生命周期动作仲裁 loss、held-only pairwise sell-rank loss、clipped-intent loss。
- 当前可交付事实：`portfolio_simulator.py` 已让 `sell_rank_score / lifecycle_sell_gate / clipped_intent_risk` 参与真实执行翻译，避免新 head 只停留在训练诊断层。
- 当前可交付事实：`pipeline_utils.py` 与 `analyze_behavior_gap.py` 已新增排序对齐与剪裁风险审计指标，r5 不再只看传统 `sell_selection_quality_5d`。
- 正式实证事实：`cp_v3_lifecycle_arbitration_r5` 已完成 4 screening + 2 confirmatory，所有 trial/confirm 仍为 `shadow_only`，因此不能 promotion。
- 正式实证事实：final champion `cp_v3_lifecycle_arbitration_r5__confirm_02` 取得 `annual_return=0.1529`、`sharpe=0.6942`、`max_drawdown=-0.0941`、`reduce_success_rate_5d=0.5957`、`exit_timeliness_rate_5d=0.7333`、`cash_timing_quality_1d=-0.1379`、`sell_selection_quality_5d=0.0207`。
- 正式实证事实：r5 的 `sell_rank_forward_alignment_5d=0.1188`、`lifecycle_sell_gate_forward_alignment_5d=0.1335`，说明生命周期卖出排序/门控已经出现可学习证据。
- 正式实证事实：r5 的 `semantic_conflict_rate=0.0163`、`order_translation_conflict_rate=0.0380`，说明相对 r4 sell guard 分支，执行/订单翻译更干净。
- 复盘结论：r5 证明“生命周期动作仲裁结构”是有效诊断与学习方向，但没有证明它已经恢复收益；它把矛盾从“卖出排序是否可学”推进到“alpha/deployment、sell lifecycle、cash/budget 三者如何统一仲裁”。
- 当前禁止项：不得把 `alpha_result_value_budget_split_v5` 直接升为默认主线；不得因 `exit_timeliness_rate_5d=0.7333` 单项改善而忽略 `cash_timing_quality_1d=-0.1379` 与收益/sharpe 退化。
- 当前决策：不 promotion `cp_v3_lifecycle_arbitration_r5`；r5 作为 sell-rank/action-arbitration 结构证据保留；收益主线仍不得覆盖 r2，cash timing 进展证据仍不得覆盖 r4。
- 下一轮合同：优先设计 `cash/deployment/value arbitration`，让 alpha opportunity、sell rank、cash risk 在同一结果驱动价值头或门控混合器里竞争；只有当该三方仲裁闭环后，才考虑更强 backbone、offline RL 或更大搜索。

## r6 value arbitration 实证补充
- 正式实证事实：`cp_v3_value_arbitration_r6` 已完成 4 screening + 2 confirmatory，所有候选仍为 `shadow_only`。
- 正式实证事实：r6 confirm_01 将 `semantic_conflict_rate` 压到 `0.0019`，说明动作语义污染已经基本被控制。
- 正式实证事实：r6 confirm_01 的 `reduce_success_rate_5d=1.0000`、`exit_timeliness_rate_5d=0.7500`、`sell_release_forward_alignment_5d=0.2593`，说明卖出释放价值被模型强烈学到。
- 正式实证事实：r6 confirm_01 的 `annual_return=-0.0913`、`sharpe=-0.4327`、`value_arbitration_forward_alignment_5d=-0.0625`、`alpha_opportunity_forward_alignment_5d=-0.0739`，说明统一单标量价值仲裁没有学稳高 alpha 部署。
- 复盘结论：r6 证明“三方价值仲裁”必须做，但不能继续压成一个单一 scalar；deploy、release、defense 三类价值的学习难度和交易含义不同，必须拆成可竞争的门控结构。
- 当前禁止项：不得把 `alpha_result_value_budget_split_v6` 或 `result_value_v5` 设为默认主线；不得因 sell-side 指标改善而覆盖收益/sharpe 退化；不得通过扩大 backbone 替代价值目标重构。
- 下一轮合同：优先推进 r6b 三值门控，明确区分 `deploy_value`、`release_value`、`defense_value`；预算层只能做容量、风险和成本约束，不得静默吞掉高价值生命周期动作。
- 治理合同：训练证据中的 `teacher_action_rows` 必须使用日频持仓容量自适应有效下限；原始 10000 行阈值只能作为参考，不得再把容量受限的日频执行教师误判为训练证据不足。
## 17. 2026-04-19 r6b 三值门控合同修正
- 已完成实证事实：
  - `cp_v3_three_value_gate_r6b` 已完成 `4 screening + 2 confirmatory`，所有候选仍为 `shadow_only`。
  - r6b 已把 `deploy_value / release_value / defense_value` 与对应 gate 全链路接入 teacher label、budget objective、model heads、推理仲裁、行为审计和 self-optimizing study。
  - `smoke01` 揭示了一个真实目标污染：三值 gate 回退逻辑把合法 0 gate 误当成缺失值；该 bug 已修复。
  - 修复后 `training_samples_preview.csv` 直接证明三值 gate label 归一成立：`gate_sum_mean=0.999998`。
- 正式实证事实：
  - performance champion `cp_v3_three_value_gate_r6b__confirm_01`
    - `annual_return=0.2780`
    - `sharpe=1.0538`
    - `reduce_success_rate_5d=1.0000`
    - `exit_timeliness_rate_5d=0.1111`
    - `cash_timing_quality_1d=-0.1833`
    - `deploy_gate_forward_alignment_5d=-0.0842`
    - `release_gate_forward_alignment_5d=0.0238`
    - `defense_gate_timing_quality_1d=-0.0147`
  - stability champion `cp_v3_three_value_gate_r6b__confirm_02`
    - `annual_return=-0.1480`
    - `sharpe=-0.5523`
    - `reduce_success_rate_5d=1.0000`
    - `exit_timeliness_rate_5d=0.3750`
    - `cash_timing_quality_1d=-0.0978`
    - `release_gate_forward_alignment_5d=0.2826`
    - `defense_gate_timing_quality_1d=-0.0875`
- 合同级结论修正：
  - r6b 证明“把 r6 的单一 scalar 拆成多值目标”是正确方向，但“把 deploy / release / defense 全部放进同一个个股级 gate simplex”仍然不对。
  - `deploy / release` 更接近个股层、持仓层的边际资金去留判断。
  - `defense / cash` 更接近组合层、市场层的全局风险与机会密度判断。
  - 因此 `defense` 不应继续作为与个股 `deploy / release` 同层竞争的 stock-level gate；这会天然诱导模型学成高 deploy 暴露、弱 defense、弱 exit 的折中。
- 新设计合同：
  - 个股层只负责：
    - `open / hold / add / reduce / exit`
    - `deploy_value`
    - `release_value`
    - 持仓内 sell ranking / lifecycle arbitration
  - 组合层只负责：
    - `defense_value / cash regime`
    - `gross / candidate / turnover / exposure floor`
    - 市场机会密度与风险状态
  - 预算层只允许：
    - 做容量、风险、成本与换手约束
    - 显式记录 clipped-intent
    - 绝不静默吞掉高价值生命周期动作
- 禁止项更新：
  - 不得继续把 r6b 当成“再加 epoch、再加 loss、再加 backbone 就会自然跑通”的主线。
  - 不得把 `cash_translation_sell_guard_v3` 的局部收益恢复误判为三值仲裁已经成立。
  - 不得继续把 `defense` 设计成与每只股票 `deploy/release` 同粒度、同竞争层级的门控头。
- 下一轮合同：
  - 优先实现层级化仲裁，而不是继续同层三值门控：
    - stock-level `deploy / release`
    - portfolio-level `defense / cash`
  - 正式 study 仍固定小矩阵，只比较层级仲裁与预算剪裁修复，不扩大 backbone。
  - 只有当层级仲裁同时改善收益、cash timing、exit 与 gate alignment 后，才允许进入更强 backbone 或 offline RL 讨论。
## 18. 2026-04-20 r7 后的层级仲裁修正合同
- 已被正式证明的事实：
  - 同层三值 gate 不对。
  - 初步层级化 r7 比同层三值 gate 更对，因为它已经能把收益与 sharpe 拉回到接近主线强度。
  - 但 r7 仍未完成真正闭环，因为最高收益 confirmatory 里的 `deploy / value arbitration` 对齐仍为负。
- 新合同：
  - `deploy / release` 只属于 stock-level 资本去留判断。
  - `defense / cash regime` 只属于 portfolio-level 风险与机会密度判断。
  - budget layer 只允许做：
    - 容量约束
    - 风险约束
    - 换手约束
    - clipped-intent 记录与惩罚
  - budget layer 不允许再承担：
    - 重写 stock-level 生命周期动作语义
    - 用组合约束静默吞掉高价值 `hold / add / reduce / exit`
- 下一轮验收要求：
  - 不只看 `annual_return / sharpe`。
  - 必须同时改善：
    - `reduce_success_rate_5d`
    - `exit_timeliness_rate_5d`
    - `cash_timing_quality_1d`
    - `value_arbitration_forward_alignment_5d`
    - `deploy_gate_forward_alignment_5d`
    - `order_translation_conflict_rate`
  - 如果收益恢复和结构对齐继续分裂在不同 confirmatory 上，则视为 hierarchy 仍未闭环，不得 promotion。
## 19. 2026-04-21 r8 constraint-only arbitration 合同修正
- 已被正式证明的事实：
  - `defense / cash regime` 继续留在 portfolio-level 是正确方向；它不应该重新回到 held-side `sell_arbitration / keep_arbitration / reduce / exit` 的同层竞争里。
  - `result_value_v8` 已经把这一方向正式写进 `budget_objective`，并让 budget layer 进入 `constraint_only_mode`。
  - 但 r8 正式最优 confirmatory 不是 `alpha_result_value_budget_split_v8`，而是 `alpha_result_value_budget_split_v7 + result_value_v8`。
  - 这说明“新 objective 的方向正确”不等于“新 loss 配比已经正确”，objective 演进与 loss 演进必须分开验收。
- r8 合同级失败定义：
  - 若高收益分支同时出现：
    - `value_arbitration_forward_alignment_5d < 0`
    - `alpha_opportunity_forward_alignment_5d < 0`
    - `deploy_gate_forward_alignment_5d < 0`
  - 即使 `annual_return / sharpe` 回升，也不能视为 continuous policy 已经学成统一资本仲裁。
  - 若结构分支虽把 `value/deploy/release` 对齐拉回正区间，但收益与 sell selection 明显偏弱，也不能视为闭环完成。
- 新的强约束：
  - budget layer 不只要“少改写动作”，还必须“在被 clip 后仍尽量保留高价值 stock-level 意图”。
  - 后续所有 layered arbitration 实验都必须显式审计并优化：
    - `budget_clipped_day_share`
    - `avg_budget_drop_count`
    - `order_translation_conflict_rate`
    - `top_order_translation_conflict_pairs`
    - held-side `sell_selection_quality_5d`
  - 不能再接受“组合层约束是对的，但 stock-level deploy/release 被 clip 掉”的折中解。
- 新的设计合同：
  - stock-level 只负责：
    - `deploy_value`
    - `release_value`
    - `open / hold / add / reduce / exit`
    - held-side sell ranking / lifecycle arbitration
  - portfolio-level 只负责：
    - `defense_value / cash regime`
    - `gross / candidate / turnover / exposure floor`
    - 市场机会密度与风险状态
  - budget layer 只负责：
    - 容量约束
    - 风险约束
    - 换手约束
    - intent-preserving translation
    - clipped-intent audit 与惩罚
- 明确禁止：
  - 不得把 `alpha_result_value_budget_split_v8` 直接升为默认主线。
  - 不得因为 `result_value_v8` 的结构方向更对，就忽略当前正式最优仍依赖旧 `v7` loss 的事实。
  - 不得继续通过“加 epoch / 加 loss / 换更大 backbone”来掩盖 `budget clipping + deploy misalignment` 问题。
- 下一轮验收补充：
  - 除 `annual_return / sharpe / max_drawdown / reduce_success_rate_5d / exit_timeliness_rate_5d / cash_timing_quality_1d` 外，必须同时要求：
    - `deploy_gate_forward_alignment_5d > 0`
    - `value_arbitration_forward_alignment_5d > 0`
    - held-side `sell_selection_quality_5d` 不为负
    - `order_translation_conflict_rate` 不再因约束层切换而显著恶化
  - 若收益恢复与结构对齐继续分裂在不同 confirmatory 上，则视为 r8 之后的 constraint-only hierarchy 仍未闭环。
## 20. 2026-04-21 r9 intent-preserving translation 合同修正
- 已被正式证明的事实：
  - `cash_constraint_intent_guard_v5` 可以显著压低 `budget_clipped_day_share`，说明“budget layer 过度硬剪裁”并非不可修复。
  - 但在 clip 降低后，`order_translation_conflict_rate` 没有同步下降，反而把主冲突集中暴露成 `model_action=add -> realized_weight_change=hold`。
  - 因此“clip 更少”不等于“intent-preserving translation 已成立”。
- 新的合同级结论：
  - 当前主病灶已从 `budget clipping too heavy` 升级为 `deploy intent not executable`。
  - `semantic_conflict_rate` 低并不能说明执行闭环成立；还必须检查 `model_action -> weight_change_action` 是否一致。
  - 后续所有 budget translation / constraint arbitration 实验，必须把 `add -> hold` 视为核心失败模式，而不是把它当作一般性的噪声冲突。
- 新的强约束：
  - 任一新 calibration 只有在同时满足以下条件时，才可被视为有效进展：
    - `budget_clipped_day_share` 明显下降
    - `order_translation_conflict_rate` 不恶化
    - `top_order_translation_conflict_pairs` 不以 `add -> hold` 为主导
    - `deploy_gate_forward_alignment_5d` 不恶化
    - `sell_selection_quality_5d` 不因 translation 修正而继续转负
  - 若只看到 clip 降低，但 `add -> hold` 大幅上升，则判定为“失败模式转移”，不算真正进展。
- 新的设计合同：
  - stock-level 训练目标除了 `deploy_value / release_value` 外，必须新增或强化 deploy executability 审计：
    - `model_action=add` 时，真实权重变化是否仍保留正向 deploy 含义
    - `model_action=open/add` 时，是否被 translation 层静默吃成 `hold`
  - budget layer 仍只允许承担：
    - 容量约束
    - 风险约束
    - 换手约束
    - intent-preserving translation
    - clipped / dropped intent audit
  - budget layer 不允许以“约束合理”为理由，持续把高价值 deploy intent 吃成无动作。
- 明确禁止：
  - 不得把 `cash_constraint_intent_guard_v5` 直接升级为默认 calibration。
  - 不得把“clip 降低”误判成“translation 闭环已经完成”。
  - 不得继续只用 `budget_clipped_day_share` 作为 translation 改进的主验收指标。
- 下一轮验收补充：
  - 除传统收益、回撤、cash timing、reduce/exit 指标外，必须同步验收：
    - `order_translation_conflict_rate`
    - `top_order_translation_conflict_pairs`
    - `add -> hold` 冲突占比
    - `deploy_gate_forward_alignment_5d`
    - `sell_selection_quality_5d`
  - 只有当 clip 降低与 deploy executability 改善同时出现，才允许判断 translation 方向真正闭环。

## 21. 2026-04-22 r10 卖出来源归因合同修正
- 已被新证据明确的事实：
  - 已在 `portfolio_simulator.py` 与 `analyze_behavior_gap.py` 中接入显式卖出来源归因：
    - `sell_execution_origin`
    - `sell_suppression_origin`
    - `budget_origin_sell_share`
    - `sell_intent_realized_rate`
    - `high_cash_budget_origin_sell_share`
  - 在 `cp_v3_deploy_executability_r10__budget_fix_protocol_r1__source_eval` / `source_audit` 中，当前 sell-side 结构被首次拆清：
    - `budget_origin_sell_share = 0.9000`
    - `sell_intent_action_count = 9`
    - `sell_intent_realized_rate = 1.0000`
    - `sell_intent_suppressed_share = 0.0000`
    - `high_cash_budget_origin_sell_share = 0.6667`
    - `sell_execution_origin_counts` 以 `budget_sell_priority / forced_zero / weight_translation` 为主，而非 `model_sell_intent`
  - 这说明当前问题不只是“模型卖出意图被压回 hold”，还包括“预算/翻译层主动制造了大多数真实卖出”。
- 新的合同级结论：
  - 当 `sell_intent_realized_rate` 很高、但 `budget_origin_sell_share` 仍接近主导时，不能判定为“release gate / sell selection 已学成”，只能判定为“模型自身卖出意图没有被压住，但真实卖出仍主要由预算层制造”。
  - 当 `high_cash_budget_origin_sell_share` 明显偏高时，不能把高现金日误读为主动 cash timing；这更可能是预算挤压后的被动收缩。
  - 因此当前 r10 的主矛盾已进一步收敛为：
    - `sell_execution_source_entangled`
    - `cash_timing_still_passive`
    - `sell_selection_not_learned`
    - `unified_value_arbitration_not_aligned`
- 新的强约束：
  - 后续任何 simulator / budget translation 修正，只要触及卖出路径，必须同步输出 sell-source attribution 审计，至少检查：
    - `budget_origin_sell_share`
    - `sell_intent_realized_rate`
    - `sell_intent_suppressed_share`
    - `high_cash_budget_origin_sell_share`
    - `sell_execution_origin_counts`
  - budget layer 可以做容量、风险、换手和 slot 竞争，但不应长期充当隐藏的主要卖出决策器。
  - 若未来 patched eval 只改善收益或 deploy realization，却仍由 budget-origin sell 主导，则只能判定为“工程层暂时拉正”，不能判定为“sell-side 已闭环”。
- 操作纪律：
  - `source_eval` / `source_audit` 属于来源归因复跑，其职责是拆清责任来源，不得覆盖正式 `protocol_summary` 的 training-evidence 或 promotion verdict。
  - formal verdict 仍以训练级 protocol / bounded study 的正式 tag 为准。
- 下一轮验收补充：
  - 除传统收益、回撤、cash timing、reduce/exit 指标外，必须同时要求：
    - `budget_origin_sell_share` 下降
    - `high_cash_budget_origin_sell_share` 下降
    - `sell_selection_quality_5d` 不为负
    - `release_gate_forward_alignment_5d` 不为负
    - `order_translation_conflict_rate` 不因卖出归因修正而重新恶化

## 22. 2026-04-23 r10 sell-source decoupling v7c 合同修正

- 已落地的代码侧事实：
  - 新增 `budget_calibration = cash_constraint_sell_source_guard_v7`。
  - 该 calibration 明确把真实卖出来源拆为：
    - `model_sell_intent`
    - `model_release_signal`
    - `deploy_funding_rebalance`
    - 以及只作审计保留的 budget / translation fallback。
  - 行为审计新增并返回：
    - `model_release_signal_count`
    - `deploy_funding_rebalance_signal_count`
    - `sell_authorized_held_count`
    - `sell_source_floor_guard_count`
    - `model_release_signal_sell_count/share`
    - `deploy_funding_rebalance_sell_count/share`
    - `deploy_funding_rebalance_forward_excess_5d`
- 新合同级结论：
  - budget layer 不能再静默制造主要真实卖出；真实卖出必须尽量归因到模型显式卖出、模型释放信号或显式 deploy funding rebalance。
  - `budget_origin_sell_share = 0` 是必要条件，不是充分条件。第一版 hard guard 已证明，若 deploy 被冻住，来源干净仍然不合格。
  - 高收益也不是充分条件。v7b 已证明，若 deploy funding sell 占比过高且卖出后 forward excess 仍为正，则该分支更像过度资金腾挪，而不是成熟的 value arbitration。
- v7c 当前合同：
  - deploy funding rebalance 只允许在以下条件同时成立时触发：
    - 已持仓且不是模型显式授权卖出。
    - 原模型动作为 `hold/skip`，不得把 `open/add` 直接作为 funding sell 来源。
    - deploy 需求足够强。
    - 弱持仓证据数达到门槛。
    - 受动态 retention floor 约束，不能无底线削仓。
  - 对未授权且不满足 deploy funding 的持仓，translation 层应守住上一期权重底线，避免预算层把轻微权重扰动翻译成生命周期卖出。
- 当前证据判决：
  - v6 对照确认原实现存在 `sell_execution_source_entangled`。
  - 第一版 v7 确认 hard floor 会过度保护旧仓。
  - v7b 确认显式 funding rebalance 有收益潜力，但语义上过激。
  - v7c 是当前最稳妥的代码侧 calibration：来源干净、deploy 未冻结、sell quality 为正、deploy funding 卖出后 forward excess 为负。
- 明确禁止：
  - 不得把 v7c 直接写成 promotion / live 默认执行。
  - 不得用 v7b 的 headline return 覆盖语义风险。
  - 不得只凭 `budget_origin_sell_share = 0` 宣称 sell-side 闭环。
- 下一轮验收补充：
  - `deploy_funding_rebalance_sell_share` 应继续下降，或由训练级 value / release head 提供更强可解释性。
  - `deploy_funding_rebalance_forward_excess_5d` 应保持非正。
  - `sell_source_floor_guard_share` 不宜长期过高，否则说明模型释放信号仍没有学会。
  - cash timing 需要单独闭环，sell-source 解耦不能替代主动择时学习。

## 23. 2026-04-23 r11 sell-source contract 训练侧合同

- 已落地的训练侧事实：
  - `pipeline_utils.py` 已新增 `budget_objective = result_value_v10`。
  - `model_seq_v3.py` 已新增 `loss_profile = alpha_result_value_budget_split_v10`。
  - `model_seq_v3.py` 已新增 `funding_release_discipline_loss`，把 `protected_hold` 与 `funding_release` 的概率约束直接写进训练损失。
  - `run_self_optimizing_study.py` 已新增：
    - `search_profile = split_heads_sell_source_contract_r11`
    - `objective_profile = sell_source_contract_v1`
- 新合同级结论：
  - sell-source contract 不再只属于 simulator；它现在必须同时进入：
    - 预算目标如何生成
    - 模型如何学习 keep / release
    - study 如何打分选优
  - 若只改 `budget_calibration` 而不改训练目标与 study objective，系统会继续把旧目标当成“最优”，导致 contract 漂浮在训练主线之外。
- `result_value_v10` 的合同含义：
  - 不再把 deploy 需求直接翻译成泛化的持仓释放压力。
  - 必须先显式计算：
    - `protected_hold_pressure`
    - `funding_release_pressure`
    - `disciplined_funding_need`
    - `portfolio_release_pressure`
  - 只有当 deploy 机会、可执行性和资金需求同时成立时，才允许预算层朝更高 candidate / turnover 倾斜。
- `alpha_result_value_budget_split_v10` 的合同含义：
  - 模型除了学 `deploy_value / release_value / gate` 外，还必须学会：
    - 什么时候该继续保留高质量旧仓
    - 什么时候该为了更强新机会释放资金
  - `funding_release_discipline_loss` 的目标不是鼓励多卖，而是避免“应该保留的仓位被弱证据 funding sell 吃掉”。
- `sell_source_contract_v1` 的合同含义：
  - 后续 bounded study 不能只奖励收益、Sharpe 和 deploy realization。
  - 还必须显式惩罚：
    - `budget_origin_sell_share`
    - `high_cash_budget_origin_sell_share`
    - 过高的 `deploy_funding_rebalance_sell_share`
    - 正向的 `deploy_funding_rebalance_forward_excess_5d`
    - 过高的 `sell_source_floor_guard_share`
- 当前验证边界：
  - dry-run 与 short-window teacher rollout 已证明训练侧合同链条打通。
  - 但它们还不能回答 `v10` 是否正式优于 `v9`。
  - 因此当前允许的结论只有：
    - `r11` 已具备正式 bounded study 条件
    - `v10` 已显式影响预算目标与选优标准
    - 是否升为正式主线，仍需新 study 决定

## 24. 2026-04-23 r11 正式 bounded study 后的合同修正

- 已被正式证据确认的事实：
  - `cp_v3_sell_source_contract_r11__study_r1` 已完整跑完，且当前最佳可复现分支为：
    - `budget_objective = result_value_v9`
    - `loss_profile = alpha_result_value_budget_split_v10`
    - `budget_calibration = cash_constraint_sell_source_guard_v7`
  - `result_value_v10 + alpha_result_value_budget_split_v10` 虽然在 screening 可行，但 fresh confirmatory 失稳，当前不能升格为默认预算目标。
- 新合同级结论：
  - `budget objective` 演进与 `loss profile` 演进必须分离验收。
    - 若 `loss` 提供了有效归因纪律，而 `objective` 仍不稳，则允许暂时保留旧 objective、吸收新 loss。
    - 不得因为同属 `v10` 家族就把两者绑定升格。
  - `sell_source_contract_v1` 的正式职责，不只是打掉 `budget_origin_sell_share`，还要继续把下列 funding-sell 风险留在主评价面：
    - 过高的 `deploy_funding_rebalance_sell_share`
    - 正向的 `deploy_funding_rebalance_forward_excess_5d`
    - 过高的 `sell_source_floor_guard_share`
  - 当前合同下，`budget_origin_sell_share = 0` 只代表第一层责任解耦已完成，不代表 held-side release/funding 仲裁已经学成。
- 新的默认研究约束：
  - 在 `r11` 家族继续推进时，优先保留：
    - `result_value_v9`
    - `alpha_result_value_budget_split_v10`
    - `cash_constraint_sell_source_guard_v7`
  - 只有当新的 bounded study 能证明 `result_value_v10` 在 confirmatory 中稳定优于上述组合时，才允许改写默认 objective。
- 导出与可观测性合同补充：
  - 连续策略导出链路必须接受“当天无 share-level action”是合法状态。
  - 任一导出入口若消费 `share_actions`，都不得假定结果非空；空表也必须保留稳定 schema，避免研究 run 因展示层 merge 失败而中断。
## 2026-04-23 held-side release/funding 合同补强

- `analyze_behavior_gap.py` 必须把 held-side 释放学习拆成三类证据，而不是只看 aggregate sell share：
  - `protected_hold` 证据：
    - `avg_protected_hold_support`
    - `model_release_against_protected_hold_share`
    - `deploy_funding_against_protected_hold_share`
  - `funding_release` 证据：
    - `avg_funding_release_support`
    - `model_release_release_consistent_share`
    - `deploy_funding_release_consistent_share`
  - `forward outcome` 证据：
    - `model_release_signal_forward_excess_5d`
    - `deploy_funding_rebalance_forward_excess_5d`

- held-side 合同的解释顺序必须固定：
  1. 先看 `budget_origin_sell_share` 是否已归零。
  2. 再看 `deploy_funding_rebalance_sell_share` 是否过高。
  3. 再看 funding sell 是否真的对齐 `release support`，而不是只看它有没有砍到强保护旧仓。
  4. 最后再看 `model_release_signal` 是否已形成稳定、可复现的释放头。

- 解释规则：
  - `deploy_funding_against_protected_hold_share` 很低，但 `deploy_funding_release_consistent_share = 0` 时，正确结论不是“held-side 已学成”，而是“系统没有频繁砍最强保护旧仓，但 release/funding 证据仍几乎为空”。
  - `model_release_signal_sell_count` 很低时，不得把少量 release 样本的好坏过度外推为稳定能力；study 评分只应在样本达到一定数量后再加强惩罚。

- `sell_source_contract_v2` 的设计职责：
  - 延续 `sell_source_contract_v1` 对 `budget_origin_sell_share` 和 `deploy_funding_rebalance_forward_excess_5d` 的硬约束。
  - 新增对 held-side 学习失败的显式惩罚：
    - 过高的 `deploy_funding_rebalance_sell_share`
    - 偏低的 `deploy_funding_release_consistent_share`
    - release 样本达到阈值后仍为正的 `model_release_signal_forward_excess_5d`
    - release 样本达到阈值后仍偏高的 `model_release_against_protected_hold_share`
  - 目标不是追求“完全不 funding sell”，而是让 funding sell 逐步从“部署副作用”变成“被 release/value 证据解释的显式资金来源”。

- `split_heads_sell_source_contract_r11b` 的默认起点合同：
  - `budget_objective = result_value_v9`
  - `loss_profile = alpha_result_value_budget_split_v10`
  - `budget_calibration = cash_constraint_sell_source_guard_v7`
  - `objective_profile = sell_source_contract_v2`
  - 该起点的含义是：先保留已经正式证明有效的 objective/loss 组合，再用更严格的 held-side 合同继续压缩 funding-sell 污染，而不是直接扶正未过 confirm 的 `result_value_v10`。

## 2026-04-23 r11b v11 loss 合同修正

- `alpha_result_value_budget_split_v11` 的设计职责：
  - 不是直接替代 `v10` 成为默认 loss。
  - 而是更强地约束：
    - `sell_release_value`
    - `release_value_target`
    - `release_gate_target`
    - `deploy_executability_target`
    - `funding_release_total`
  - 并在 `funding_release_discipline_loss` 中把 `disciplined_funding_need`、dominant gap 和 margin 惩罚显式加重。

- `v11` 的正式解释规则：
  - 若 `v11 + result_value_v10` 仍出现：
    - 过高的 `deploy_funding_rebalance_sell_share`
    - `deploy_funding_release_consistent_share = 0`
    - 为负的 `cash_timing_quality_1d`
    - `shadow_only`
    则不得把它解释成“更强 loss 已经完成 held-side 学习闭环”。
  - 若 `v11 + result_value_v9` 虽然显著压低 funding share、并把 `deploy_funding_rebalance_forward_excess_5d` 压到非正，但同时把 `deploy_intent_realized_rate` 压低到不可接受区间，则不得把它当作可推广主线，只能视为语义改善证据。

- `r11b` 之后的合同更新：
  - `deploy_funding_release_consistent_share` 仍为 `0.0` 时，不论 funding share 降到多低，都不能宣称 release head 已学成。
  - held-side 新 contract 不只要求“少卖、卖得干净”，还要求：
    - deploy intent 不显著塌陷
    - order translation conflict 不显著恶化
    - hold continuity 不被系统性破坏
  - 因而 future confirm 的最低验收语言必须同时覆盖：
    - `deploy_funding_rebalance_sell_share`
    - `deploy_funding_rebalance_forward_excess_5d`
    - `deploy_funding_release_consistent_share`
    - `deploy_intent_realized_rate`
    - `order_translation_conflict_rate`

- 逐仓 held-side detail export 的合同地位：
  - 当研究目标涉及 held-side release/funding 时，`--export-held-side-details` 产物应被视为标准审计附件，而不是可有可无的调试文件。
  - 原因是 aggregate share 无法告诉我们 funding trim 是：
    - 广泛分散
    - 集中在少数旧仓反复修剪
    - 还是以少量 protected-hold conflict 为主
  - 当前 `001309.SZ`、`002371.SZ`、`002049.SZ` 这类重复 trim 名单，已经证明逐仓明细对合同判读具有实质价值。
