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
