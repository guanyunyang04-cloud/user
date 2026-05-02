# Continuous Policy 设计合同

快照日期：`2026-05-02`

## 北极星
- 构建一个以日为单位进行连续决策的交易执行模型。
- 模型直接从市场全局状态、个股演化路径与持仓上下文中学习 `source / receiver / cash allocation ranking`。
- 目标是在尽量少的人为执行桥约束下，综合权衡未来收益、风险、成本、现金防守与部署可执行性。

## 非目标
- 不把固定调仓频率、固定持有周期或人工执行桥作为最终目标。
- 不把更像 teacher 当作最终目标；teacher 只是 warm start / auxiliary prior / heuristic scaffold。
- 不用 screening 高收益、短窗 smoke、replay 单项改善或单一局部指标替代正式 verdict。

## 当前绑定原则
- 个股动作语义与组合预算语义必须分层：个股层表达生命周期动作，组合层表达资金接收、资金释放、现金保留、gross / turnover / cost。
- 个股未来上涨不等于今天该加仓；持仓仍有正 forward 也不一定永远不能卖，关键是相对 receiver、现金与风险的机会成本。
- 卖出、现金与资金来源是同一条 credit assignment 链；source 必须解释为“当前组合状态下更适合释放资金”，不是简单看跌。
- 执行层只翻译策略语义为权重和订单，不得静默改写策略本体。
- 日频数据不能完整学习盘中冲击、真实滑点、新闻驱动、盘口流动性和开盘跳空过程；所有 promotion 判断必须保留这个盲区。
- 2024-2026 的有效独立 regime 和月度样本有限；高收益分支必须经 confirm-vs-screening 稳定性与月度质量复核。

## 当前阶段合同
- r31 receiver 授权闭包：所有 `direct_action_add_authorized` 与 `direct_action_open_authorized` 必须是 executable `portfolio_daily_receiver_target` 的子集；`direct_action_authorization_subset_violation_count > 0` 时不得通过 v2 gate。
- r31 no-headroom 降级：无 headroom 的 held add 必须在语义层提前降级为 hold/no-op，并计入 `portfolio_daily_receiver_semantic_no_headroom`、`authorized_add_no_weight_change_share` 或 `deploy_intent_unrealized_share`。
- r31 receiver 广度：高现金、低持仓数、gross exposure target 较高时，flat open candidate 的 listwise 排名必须能重新进入 receiver path，不能只反复加已到目标权重的 held 名字。
- r33 source forward proxy：`portfolio_daily_source_forward_proxy_keep_risk` 必须进入 label、训练头、推理融合、source candidate gate、feedback、continuity metrics、study scoring 与 behavior audit；旧 artifact 缺少该 head 时只能使用 fallback proxy。
- r33 release conviction：`portfolio_daily_source_release_conviction` 必须联合 source score、release quality、economic release、executability、release capacity、opportunity cost、forward-strength brake、bad forward spread、economic block 与 proxy keep-risk。
- r33 distribution clean-pass：source candidate 必须同时通过 forward proxy、`portfolio_daily_source_release_conviction_pass` 与 `portfolio_daily_source_distribution_clean_pass`；强势正 forward 误卖不得被均值掩盖。
- r34 allocation breadth scaffold：`split_heads_portfolio_daily_allocation_breadth_r34` 只作为 bounded confirm 搜索入口，必须保留 r31/r33 守门，并通过 `portfolio_daily_receiver_candidate_breadth`、`portfolio_daily_clean_source_candidate_breadth` 与 `portfolio_daily_joint_economic_quality_gate` 把广度和经济质量写入 scoring。
- r34 confirm candidate 合同：v2 gated profile 的 confirm 候选必须优先满足 `training_evidence_status = sufficient` 与 v2 gate qualified；不得让 insufficient screening 高分 trial 绕过 confirm 候选选择。
- r35 unified allocation 合同：`split_heads_portfolio_daily_unified_allocation_r35` 必须把 source、receiver 与 cash 当作同一个 listwise allocation problem；`portfolio_daily_unified_receiver_score`、`portfolio_daily_unified_source_score`、`portfolio_daily_unified_cash_score`、`portfolio_daily_unified_allocation_objective` 是训练 surface，不得作为普通 feature 泄漏。
- r35 经济 credit assignment 合同：`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty` 与 `portfolio_daily_receiver_source_spread_reward` 必须写入训练目标、summary 与 scoring；不能只在 v2 gate 后验拦截正 forward source 误卖或负 receiver-source spread。
- r35 optimizer 合同：`solve_semidifferentiable_allocation` 是半可微最终 allocation 层合同，显式约束 `cash_reserve_target`、`turnover_limit`、`max_position_weight`、transaction cost、slippage 与 sell tax；simulator guard 只做最后安全层，不再承担主策略逻辑。
- r36 risk-aware unified allocation 合同：cash timing、drawdown、market downside、forward benchmark return、source positive forward penalty、source opportunity cost penalty 与 receiver-source spread reward 必须进入 unified allocation target 与 consistency loss；不能只靠 v2 gate 后验拦截。
- r36 source distribution 执行合同：所有 unified source candidate 必须满足 `portfolio_daily_source_distribution_clean_pass`，或满足更严格的低 penalty / 低 opportunity cost / 低 brake-risk / 高 spread relief 条件；普通 source candidate 与 unified source candidate 的 OR 合并不得绕过 source distribution gate。
- r37 source hard-negative 合同：`portfolio_daily_source_strong_false_sell_penalty` 与 `portfolio_daily_source_hard_negative_penalty` 必须作为 source candidate、unified source score、allocation objective、summary、study scoring 与 decision-focused loss 的一等信号；推理阶段没有未来标签时必须消费模型预测的 positive-forward / opportunity-cost penalty heads，不能退回全零惩罚路径。
- r37 decision-focused allocation 合同：`portfolio_decision_regret_total` 必须同时惩罚错误释放强正 forward source、receiver/source ranking regret、dead cash 与 cash/objective regret；它是 source/receiver/cash 同一 allocation problem 的训练反馈，不得被解释为单票 action head 的附属损失。
- r38 source hard-negative regret 合同：`portfolio_daily_source_tail_false_sell_penalty`、`portfolio_daily_source_release_preference` 与 `portfolio_daily_transfer_regret_target` 必须进入训练 heads、sample targets、listwise / pairwise loss、unified allocation summary、study scoring 与 protocol metrics；source 防错不能只靠后验 v2 gate。
- r38 balance 合同：压住强势误卖后，必须同步约束 `source_target_count`、receiver/source breadth、cash deployment、open/reduce/exit 质量、monthly return 与 drawdown；不得把接近全现金的语义安全状态解释为成功。
- r39 allocation objective consolidation 合同：`portfolio_daily_allocation_trade_quality_target`、`portfolio_daily_allocation_cash_deployment_target`、`portfolio_daily_allocation_risk_adjusted_return_target`、`portfolio_daily_allocation_drawdown_control_target`、`portfolio_daily_allocation_monthly_quality_target` 与 `portfolio_daily_allocation_final_objective` 必须进入 label、heads、sample targets、loss、predict policy frame、optimizer objective 与 profile；`alpha_result_value_budget_split_v25` 中 action loss 只能作为辅助，不能继续主导 allocation objective。
- r39 execution blend 合同：当 r39 objective heads 可用时，`portfolio_daily_allocation_final_objective` 必须回写 `portfolio_daily_unified_receiver_score/source_score/cash_score` 和核心 receiver/source/cash score；否则会重回“训练目标已接入、执行仍走旧 score”的半旧路径。
- foundation model 合同：foundation model 只能作为状态表征增强或 encoder prior，不能替代 source/receiver/cash allocation decision layer；最终资金分配必须仍由可审计的 listwise allocation objective 与 optimizer layer 负责。
- listwise allocation teacher：`build_allocation_teacher_summary` 只输出 source/receiver/cash teacher surface 摘要，当前不得替代 simulator 或 active 执行路径。
- repeat release relief 只能用于 clean-pass 已通过、recent sell blocking 明显过强、opportunity cost 低、release capacity 足、economic block 受控的窄场景。
- 当前 gate / scoring 合同继续使用 `portfolio_daily_ranking_v2_gated`，不能用局部 clean-pass 指标替代 v2 gates 与 confirm stability。

## 成功判定
- 训练证据：`training_evidence_status = sufficient`，且 diagnostics 显示 `device = cuda`、`cuda_available = true`、`python_executable` 指向 yolos。
- v2 gates：收益、Sharpe、max drawdown、monthly return、monthly consistency、receiver/source/cash 质量、执行冲突与 source 分布全部过线。
- 稳定性：confirm-vs-screening 不能靠单点偶然，`stable_confirmatory` 必须成立。
- receiver：`receiver_target_count >= 3`、`receiver_unrealized_deploy_share = 0`、direct add/open authorization subset 无违规。
- source：`source_target_count >= 3`、`source_realized_sell_rate >= 0.35`、positive forward sell share / strong false sell / max forward 受控。
- cash：`cash_reserve_rate > 0`，但不能退化为高现金 dead branch。
- 经济质量：`monthly_return_mean`、`monthly_consistency_score`、`portfolio_daily_exposure_utilization`、receiver realized deploy、receiver-source spread、drawdown 必须联合判断。

## 当前已知事实
- r31 已证明 receiver 语义旁路可以前置到 receiver target 层；bounded confirm 中 authorization subset、authorized add no weight、deploy unrealized 与 receiver unrealized 均可压到 0。
- r31 仍未证明 promotion readiness：training evidence、source activity、confirm gate 与收益质量仍不足。
- r33 trainable proxy 可以防错，但会让 source dormant；source-soft 可以恢复 source，却会强势误卖。
- r33 release conviction 可以恢复收益和 source 数量，但单独不足以控制尾部正 forward source。
- r33 clean-pass + repeat relief 可以清掉强势误卖，但 clean-pass bounded 仍出现 source/receiver 广度不足和 receiver-source spread 不稳。
- r34 已补上 receiver/source breadth scoring、joint economic quality penalty 与 allocation teacher summary，并完成 bounded/evidence-confirm；最新 fresh confirm 训练证据充分且 receiver/cash/source 执行率达标，但因 `receiver-source spread` 转负、`source_positive_forward_sell_share` 与 `source_strong_positive_forward_sell_count` 失控，仍不是 verdict。
- r35 已完成 unified allocation 修复后 bounded confirm：`postfix4_bounded_confirm_20260430` 完成 4 个 screening、2 个 confirm，`stable_confirmatory_count = 1`，champion `confirm_02` 达到 `training_evidence_status = sufficient`、`annual_return = 1.409793`、`sharpe = 2.846432`、`receiver_unrealized_deploy_share = 0`、`source_realized_sell_rate = 1.0`、`cash_reserve_rate = 0.806071`、`receiver_minus_source_forward_excess_5d = 0.206611`，但仍 `promotion_status = shadow_only`。
- 当前瓶颈已经从“能不能卖”推进到“能否在 unified allocation 下同时控制 cash timing、drawdown、reduce/exit 质量和 source positive distribution”。
- r36c screening 证明：执行层 source distribution 旁路已被封住，source target 行全部为 `portfolio_daily_source_distribution_clean_pass = true`；但真实未来分布仍失败，`source_positive_forward_sell_share = 0.666667`、`source_strong_positive_forward_sell_count = 2`、`receiver_minus_source_forward_excess_5d = -0.029367`。剩余根因不是 simulator OR 旁路，而是模型对 source 正向前景与机会成本的低估。
- r37b screening 证明：source hard-negative 与 decision-focused allocation loss 的工程路径已接通，推理侧也会消费预测 penalty heads；但模型学出的 penalty 仍太弱，`source_positive_forward_sell_share = 0.833333`、`source_strong_positive_forward_sell_count = 2`、`receiver_minus_source_forward_excess_5d = -0.057937`。当前问题不是字段未接或 guard 漏洞，而是 source hard-negative 信号的学习强度和信用分配仍不足。
- r38 bounded confirm 证明：source hard-negative tail、release preference 与 transfer regret 能把 positive source false sell 压住，confirm 中 `source_positive_forward_sell_share = 0.0`、`source_strong_positive_forward_sell_count = 0`、`receiver_minus_source_forward_excess_5d = 0.037347`；但 `annual_return = 0.009111`、`sharpe = 0.164855`、`source_target_count = 1`、`cash_reserve_rate = 0.981728`，说明当前失败已转为收益弱、交易广度不足和现金过度保守。
- r39 bounded confirm 证明：统一 allocation objective 和 execution blend 能恢复收益、正 spread 与 receiver 广度，execblend confirm 达到 `annual_return = 2.676289`、`sharpe = 3.802282`、`monthly_return_mean = 0.115004`、`receiver_target_count = 6`、`receiver_minus_source_forward_excess_5d = 0.066883`、`source_positive_forward_sell_share = 0.0`；但仍未稳定，`source_target_count = 1`、`cash_reserve_rate = 0.947020`、`cash_timing_quality_1d = -0.138133`、`max_drawdown = -0.109069`，v2 gate 只过 `9/12`。

## 当前禁止事项
- 不得把 r31/r33 任一 replay、smoke、bounded 或 insufficient run 写成 promotion / live / active artifact 切换依据。
- 不得为了恢复 source count 粗暴放宽 source forward proxy、release conviction 或 distribution clean-pass。
- 不得把 `receiver_unrealized_deploy_share = 0`、`source_positive_forward_sell_share = 0` 或 `add_to_hold_conflict_share = 0` 单独解释为成功。
- 不得让 simulator guard 继续承担主要策略翻译职责；guard 只能是最后防线。

## 下一步方向
- 短期：不要继续单边加重 source false-sell 或 cash penalty；先在 r39 objective consolidation 基础上提高 clean source breadth、降低过高 cash reserve，并把 cash timing / drawdown 的日级触发直接并入 allocation layer。
- 中期：把 monthly return、exposure utilization、receiver realized deploy、source realized sell、positive spread distribution、cash timing 和 drawdown 更深地写进 objective / feedback。
- 长期：推进真正的 listwise 组合日决策，让模型直接输出当日 source/receiver/cash allocation ranking。

## 阶段索引
- r1-r11b：alpha prior、split heads、translation guard、sell attribution、value arbitration、sell-source contract，详见 `daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- r12-r18：release/translation/deploy、action-value、direct action、pair reallocation 与 pair-source cost，详见 `daily_research/brain/episodic_memory.md`。
- r19-r23：portfolio daily ranking、v2 gated、cash-aware、source/receiver execution 与 stable confirmatory。
- r24-r26：listwise allocation、source-release listwise、allocation teacher。
- r27-r30：source economic release、forward-strength brake、direct-release relief 与 cash-relief。
- r31/r33：当前合同核心，分别约束 receiver 语义闭包与 source distribution clean-pass。
- r34：allocation breadth 入口，聚焦 receiver/source breadth、经济质量内生化与 source/receiver/cash teacher surface 摘要。
- r35：unified allocation 入口，聚焦 unified allocation surface、半可微 optimizer layer、经济 credit assignment 与 simulator guard 降级为最后安全层。
- r36：risk-aware unified allocation 入口，聚焦 risk-aware cash/source distribution objective、unified allocation consistency loss 与 source distribution 执行旁路封闭。
- r37：当前执行入口，聚焦 source hard-negative、strong false sell、预测 penalty 推理消费与 decision-focused allocation regret。
- r38：source hard-negative regret 入口，聚焦 source hard-negative tail、source release preference、transfer-level allocation regret，以及防错后恢复收益、广度和资金时机。
- r39：当前执行入口，聚焦 allocation objective consolidation、final objective execution blend、action loss 降级为辅助，以及 cash timing / drawdown / clean source breadth 的统一收敛。

## 历史归档入口
- 早期设计合同原文：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 早期标题索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
- 过程复盘入口：`daily_research/brain/episodic_memory.md`。
- 读取纪律：当前合同以本文件上方章节为准；历史合同只作为证据与演化追溯。
