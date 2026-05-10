# Continuous Policy 设计合同

快照日期：`2026-05-09`

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
- r41 risk-sensitive allocation layer 合同：`portfolio_daily_allocation_uncertainty_pressure_target`、`portfolio_daily_allocation_tail_risk_control_target` 与 `portfolio_daily_allocation_decision_focused_objective` 必须进入 label、heads、sample targets、loss、predict policy frame、optimizer objective 与 profile；solver 必须在高 uncertainty / tail risk 下压制 receiver deploy 并提高 cash defense，同时不能杀死 clean source release。
- r41 推理融合合同：当 r41 heads 可用时，预测的 uncertainty / tail / decision objective 必须参与 unified receiver/source/cash score 和 `solve_semidifferentiable_allocation`，不能只停留在训练 loss；旧 artifact 缺少 r41 heads 时只能走可解释 fallback target，不能默认为策略成功。
- r42 utility-credit allocation 合同：`portfolio_daily_allocation_net_utility_target`、`portfolio_daily_allocation_credit_closure_target` 与 `portfolio_daily_allocation_resource_efficiency_target` 必须进入 label、heads、sample targets、loss、predict policy frame、optimizer objective、summary 与 profile；目标不是继续加 risk brake，而是直接约束 source/receiver/cash 的同日资金信用闭合、净效用和单位训练资源效率。
- r42 resource gate 合同：`split_heads_portfolio_daily_utility_credit_allocation_r42` 默认必须用短 screening 验证，profile 内默认 `epochs = 24`、`min_epochs = 16`，不得被命令行默认值静默覆盖为长训；screening 后若出现 source dead、cash timing bad、drawdown bad 或 receiver deploy 不干净且经济信号弱，必须阻断 confirmatory，不能继续消耗长训练资源。
- r43 primal-dual decision allocation 合同：`alpha_result_value_budget_split_v28` 必须让 `portfolio_primal_dual_decision_total` 强于单票 action / duration 辅助损失，并把同一交易日内的 receiver/source/cash soft allocation regret、tail false-source、dead cash、risk cash under-defense 与 funding imbalance 写入训练和验证损失；r41/r42 目标列必须进入 `sample_targets`，不得只存在于 label/head/export 代码里。
- r43 profile 合同：`split_heads_portfolio_daily_primal_dual_decision_allocation_r43` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与短 screening resource gate；profile 默认 `epochs = 20`、`min_epochs = 14`，用于先排除低信息方向，不得静默升级为长训。
- r44 entropic transport allocation 合同：`alpha_result_value_budget_split_v29` 必须让 `portfolio_entropic_transport_decision_total` 强于 r43 primal-dual 辅助损失，并用 Sinkhorn 风格可微运输计划同时约束 source supply、receiver demand、cash source、cash sink、运输 regret、false-source flow、dead cash 与 risk cash under-defense；不得退回分离 source/receiver/cash ranking 的旧损失。
- r44 profile 合同：`split_heads_portfolio_daily_entropic_transport_allocation_r44` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与短 screening resource gate；profile 默认 `epochs = 18`、`min_epochs = 12`，用于先排除低信息方向，不得静默升级为长训。
- r45 conservative transport allocation 合同：`alpha_result_value_budget_split_v30` 必须在 r44 entropic transport 之上加入 `portfolio_offline_conservative_support_total`，用 offline support / OOD action 保守损失压制非 executable receiver、非 held/source、强 false-source 与高风险低 cash 的过度自信；不得把离线数据未支持的动作高分解释为泛化突破。
- r45 profile 合同：`split_heads_portfolio_daily_conservative_transport_allocation_r45` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与更严格的短 screening resource gate；profile 默认 `epochs = 16`、`min_epochs = 10`，用于先排除低信息方向，不得静默升级为长训。
- r46 differentiable convex allocation 合同：`alpha_result_value_budget_split_v31` 必须让 `portfolio_differentiable_convex_allocation_total` 成为主导 allocation loss，并把 legacy `action_total` / `duration_total` 降为 `0`；torch 图内必须同时惩罚 KKT / budget / turnover / position / gross exposure residual、unsupported receiver/source mass、false-source mass、cash timing、source/receiver shortfall、candidate breadth、behavior-support / conservative OPE、路径级 CVaR / drawdown / OCE 风险与 oracle regret。
- r46 diagnostics / target 合同：r46 loss 不得继续写死 position cap、turnover、gross exposure 或 cash timing；训练 sample targets 必须从日级 `gross_exposure_target`、`candidate_budget`、`turnover_budget`、`max_position_weight_target`、`budget_cash_timing_signal_target` 映射到个股样本，并在 diagnostics 中输出 `portfolio_differentiable_convex_allocation_terms`，至少包含 constraint、support、OPE、path risk 与 total。
- r46 profile 合同：`split_heads_portfolio_daily_differentiable_convex_allocation_r46` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与更严格短 screening resource gate；profile 默认 `epochs = 14`、`min_epochs = 9`，用于先验证真正可微信用闭合，不得静默升级为长训。
- r47 true convex solver allocation 合同：`alpha_result_value_budget_split_v32` 必须让 `portfolio_cvxpy_convex_allocation_total` 成为主导 allocation loss，并让 `portfolio_differentiable_convex_allocation_total` 退为 fallback / auxiliary；r47 必须通过真实 `cvxpy` / `cvxpylayers` / `diffcp` 的 DPP-compliant convex solver layer 求解 fixed-slot 候选组合，不能只复用 r46 torch surrogate 或半可微 numpy solver。
- r47 fixed-slot solver 合同：训练时按 date 构造 receiver/source/held support candidate bank，选择固定 `slot_count` 的可执行候选，使用同一 `CvxpyLayer` 分别求解 predicted utility 与 oracle utility 下的组合权重；loss 必须惩罚 solver regret、solution tracking、gross / turnover / position residual、unsupported mass、false-source mass、cash timing、path risk 与 solver success rate。
- r47 dependency / diagnostics 合同：yolos 环境必须安装 `cvxpy`、`cvxpylayers`、`diffcp` 与至少 `SCS` / `DIFFCP` solver；训练 diagnostics 必须输出 `supports_portfolio_cvxpy_convex_allocation_layer`、`portfolio_cvxpy_convex_layer_status` 与 `portfolio_cvxpy_convex_allocation_terms`。若 solver 不可用，只能显式 fallback 到 r46 surrogate 并记录 status，不能默认为 r47 策略成功。
- r47 profile 合同：`split_heads_portfolio_daily_true_convex_solver_allocation_r47` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与更严格短 screening resource gate；profile 默认 `epochs = 10`、`min_epochs = 7`，用于先验证真实 solver layer 的训练成本、梯度稳定性、source release、cash timing、drawdown 和 support 诊断，不得静默升级为长训。
- r48 full-universe convex OPE 合同：`alpha_result_value_budget_split_v33` 必须让 `portfolio_full_universe_convex_allocation_total` 成为主导 allocation loss，r47 fixed-slot loss 不得继续主导；r48 必须把 solver 候选银行扩大到 full-universe aware resource-safe slot bank，并用 `candidate_coverage_loss` 惩罚旧 receiver/source mask 漏掉的高 oracle 机会，不能让旧 mask 决定全部梯度入口。当前训练口径显式固定为 `slot_count = 32`、`max_days_per_batch = 1`、`train_batch_interval = 2`、`train_solver_enabled = false`，最终 diagnostics 仍运行 solver terms；不得再把 r48 误写成未受资源约束的 48-slot / 3-day 长训口径。
- r48 realistic objective 合同：r48 solver / loss 必须显式输出并训练 `liquidity_impact_loss`、`concentration_risk_loss`、`universe_expansion_loss`，将 liquidity support、impact cost、factor concentration proxy、dynamic cost、turnover、risk pressure 与 path risk 纳入真实 solver 诊断；不得只用固定 cost / position cap / turnover slack 解释为现实可交易约束。
- r48 OPE 合同：r48 必须输出 `ope_lower_bound_loss`、`propensity_support_loss` 与 `doubly_robust_gap_loss`，用 behavior propensity、当前持仓行为收益 proxy、policy lower bound 与 DR gap 对离线分布外动作保持保守；dry-run / 单测只能证明接线，不能证明离线策略安全。
- r48 profile 合同：`split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与更严格短 screening resource gate；profile 默认 `epochs = 8`、`min_epochs = 6`、`batch_size = 256`，用于先验证 full-universe coverage、OPE、成本风险和 solver 成本，不得静默升级为长训。
- r49 capital-flow closure 合同：`alpha_result_value_budget_split_v34` 必须让 `portfolio_capital_flow_closure_total` 成为强于 full-universe convex auxiliary 的主导资金闭合损失；loss 必须输出 `portfolio_capital_flow_closure_terms`，至少包含 `receiver_demand_loss`、`funding_shortfall_loss`、`source_dead_loss`、`false_source_loss`、`over_cash_loss`、`cash_defense_loss`、`cash_timing_loss`、`cash_coherence_loss`、`exposure_gap_loss`、`flow_conservation_loss`、`role_overlap_loss`、`source_breadth_loss`、`desired_receiver_flow_mean`、`effective_receiver_flow_mean`、`effective_source_flow_mean`、`risk_cash_need_mean`、`predicted_gross_mean`、`deploy_pressure_mean`、`risk_pressure_mean` 与 `total`。r49 的目标不是再加单边 source/cash penalty，而是让 receiver demand 必须被 clean source supply 或 cash release 解释，并能区分低风险死现金与高风险现金防守不足。
- r49 旧失败阻断合同：当 receiver demand 高、clean source target 存在但 predicted source 近零，或 cash score 在低风险高部署压力下过高，或同一持仓同时被模型高分标成 receiver 与 source，或 source 释放集中在过窄候选上，或 avg gross exposure target 高而 exposure utilization 低时，训练 loss 和 resource gate 必须显式失败；不得把 `receiver_realized_deploy_rate = 1.0` 但 `source_target_count = 0` / `portfolio_daily_exposure_utilization` 低解释为结构成功。
- r49 profile 合同：`split_heads_portfolio_daily_capital_flow_closure_r49` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1` 与短 screening resource gate；profile 默认 `epochs = 8`、`min_epochs = 6`、`batch_size = 256`。resource gate 必须至少约束 source count、source realized sell rate、cash timing、drawdown、monthly / annual return、receiver unrealized deploy 与 exposure utilization。
- r50 integrated convex capital-flow 合同：`alpha_result_value_budget_split_v35` 必须同时保留 r49 capital-flow closure、r48 full-universe/OPE、r47 true convex solver、r41 risk-sensitive 与 r45 offline support 的核心训练信号；`portfolio_cvxpy_convex_allocation_total` 必须大于 `0`，`portfolio_full_universe_convex_allocation_total` 与 `portfolio_capital_flow_closure_total` 必须继续保持主导地位，legacy `action_total` / `duration_total` 必须为 `0`。
- r50 真实 solver 训练合同：全局 full-universe train solver 默认保持关闭，只有 `alpha_result_value_budget_split_v35` 通过 `CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_LOSS_PROFILES` 受控启用；训练期 solver 必须继续受 `slot_count = 32`、`max_days_per_batch = 1`、`train_batch_interval = 2` 约束，不得静默升级为无界全 A 股大规模 solver 长训。
- r50 diagnostics 合同：训练 diagnostics 必须输出 `portfolio_full_universe_convex_train_solver_effective` 与 train-solver loss profile 列表；当 full-universe loss 走 surrogate fallback 时，`solver_success_rate` 必须为 `0` 且 `fallback_surrogate_loss` 必须反映 surrogate total，不得再把 fallback 写成 solver 成功。
- r50 profile 合同：`split_heads_portfolio_daily_integrated_convex_capital_flow_r50` 必须继续使用 `end_to_end_allocation_layer_v1`、`allocation_layer_v1`、`end_to_end_allocation_layer_v1` 与短 screening resource gate；profile 默认 `epochs = 6`、`min_epochs = 5`、`batch_size = 192`，用于先验证真实 solver 训练参与、资金闭合与资源消耗，不得静默升级为长训。
- r51 native allocation vector 合同：`alpha_result_value_budget_split_v36` 必须让 `portfolio_native_allocation_vector_total` 成为主导 allocation loss，并把 legacy `action_total` / `duration_total` 置为 `0`；`portfolio_cvxpy_convex_allocation_total` 与 `portfolio_full_universe_convex_allocation_total` 必须为 `0`，不得调用 `cvxpy` / `diffcp` 训练主路径。r51 的目标不是继续让 source、receiver、cash 多个独立 head 事后对账，而是让模型直接输出日级目标仓位向量和现金比例，再从 `target_delta` 自然推导 source / receiver / cash。
- r51 projection / diagnostics 合同：`_project_native_allocation_vector` 必须在 torch 内按 `date_code` 聚合，输出 `portfolio_daily_target_weight`、`portfolio_daily_target_delta`、`portfolio_daily_target_cash_weight`、`portfolio_daily_target_turnover`、`portfolio_daily_native_receiver_score`、`portfolio_daily_native_source_score` 与 `portfolio_daily_native_cash_score`；投影后必须保证不可交易 receiver 权重为 0、非持仓不能产生卖出、单票上限、现金预算、换手预算和资金流守恒。`_portfolio_native_allocation_vector_loss` 必须输出 allocation sum、cash reserve、position cap、turnover、unsupported receiver、sell nonheld、funding shortfall、cash timing、decision utility、risk cost、source breadth、exposure utilization 与 total 诊断。
- r51 profile 合同：`split_heads_portfolio_daily_native_allocation_vector_r51` 必须继续使用 `holdcash_v3`、`budget_v3`、`end_to_end_allocation_layer_v1`、`allocation_layer_v1` 与短 screening resource gate；profile 默认 `epochs = 8`、`min_epochs = 6`、`batch_size = 256`。r51 不加入 true-solver resource guard，不作为 production / live / active artifact，dry-run / 单测只能证明接线和合同正确，不能证明策略有效。
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
- r40 clean rerun 证明：end-to-end allocation layer 的运行通道、持久产物与 confirmatory 流程已经可完整跑通，`completed_trial_count = 3`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`、模型 diagnostics 为 yolos + CUDA；但 stable confirm 为空，confirm_01 / confirm_02 均为 `shadow_only`，且 source 释放为 0、cash timing 为负、drawdown 未过线，因此 r40 不是 promotion verdict。
- r41 代码合同证明：risk-sensitive / uncertainty-aware allocation layer 已完成实现和合同测试，覆盖目标构造、solver 风险刹车、v26 loss profile、risk-sensitive loss、模型 heads、推理导出与 profile 注册；但尚未运行 bounded study，因此不能作为策略有效性事实。
- r42 代码合同证明：utility-credit allocation 已完成实现、合同测试与 dry-run，覆盖 net utility / credit closure / resource efficiency 三个目标、v27 loss profile、utility credit closure loss、模型 heads、推理融合、optimizer utility relief / budget multiplier 与 study resource gate；但尚未运行正式 screening，因此不能作为策略有效性事实。
- r43 代码合同证明：primal-dual decision allocation 已完成实现、合同测试与 dry-run，覆盖 v28 loss profile、date-grouped primal-dual decision loss、r41/r42 目标列 sample target 接线、artifact support flags、r43 search profile 与 resource gate；但尚未运行正式 screening，因此不能作为策略有效性事实。
- r44 代码合同证明：entropic transport allocation 已完成实现、合同测试与 dry-run，覆盖 v29 loss profile、Sinkhorn 风格可微资金运输损失、transport-first search profile 与更严格 resource gate；但尚未运行正式 screening，因此不能作为策略有效性事实。
- r45 代码合同证明：conservative transport allocation 已完成实现、合同测试与 dry-run，覆盖 v30 loss profile、offline support / OOD action 保守损失、training/validation 接线、diagnostics support flag、r45 search profile 与更严格 resource gate；但尚未运行正式 screening，因此不能作为策略有效性事实。
- r46 代码合同证明：differentiable convex allocation 已完成实现与合同测试，覆盖 v31 loss profile、torch 图内 allocation surrogate、KKT/constraint residual、data-driven 日级约束 target、executable support targets、behavior-support / conservative OPE、路径级 CVaR / drawdown / OCE 风险、training/validation 接线、component diagnostics support flag、r46 search profile 与更严格 resource gate；但尚未运行正式 screening，因此不能作为策略有效性事实。
- r47 代码合同证明：true convex solver allocation 已完成实现与合同测试，覆盖 v32 loss profile、真实 `cvxpy` / `cvxpylayers` fixed-slot solver layer、DPP 合同检查、solver regret / solution tracking、gross / turnover / position / support / cash timing / path risk 诊断、training/validation 接线、diagnostics support flag、r47 search profile 与更严格 resource gate；但尚未运行正式 screening，因此不能作为策略有效性事实。
- r48 正式研究事实：full-universe convex OPE allocation 已完成实现、合同测试、dry-run、formal screening 与 confirmatory，覆盖 v33 loss profile、resource-safe full-universe aware candidate bank、candidate coverage loss、liquidity / impact / concentration risk、propensity support、OPE lower-bound、doubly-robust gap、training/validation 接线、diagnostics support flags、r48 search profile 与更严格 resource gate。tag `self_opt_study_r48_full_universe_convex_ope_allocation_screening_20260508_p0p5_r3` 完成 `1` 个 screening 与 `1` 个 confirmatory，但 stable confirm 为空；confirm_01 为 `training_evidence_status = insufficient`、`source_target_count = 0`、`source_realized_sell_rate = 0`、`portfolio_daily_exposure_utilization = 0.331039`。因此 r48 是失败 verdict，不是策略有效性事实。
- r49 代码合同事实：capital-flow closure 已完成实现、合同测试与 dry-run，覆盖 v34 loss profile、`_portfolio_capital_flow_closure_loss`、training/validation 接线、diagnostics support flags、r49 search profile、source/exposure/cash resource gate，以及旧失败模式测试；后续加固已把 `cash_defense_loss`、有效 source/receiver flow、风险现金需求、预测暴露、部署压力和风险压力纳入 diagnostics。dry-run `self_opt_study_r49_capital_flow_closure_dry_run_20260509` 只证明接线和配置正确，尚未证明策略有效。

## 当前禁止事项
- 不得把 r31/r33 任一 replay、smoke、bounded 或 insufficient run 写成 promotion / live / active artifact 切换依据。
- 不得为了恢复 source count 粗暴放宽 source forward proxy、release conviction 或 distribution clean-pass。
- 不得把 `receiver_unrealized_deploy_share = 0`、`source_positive_forward_sell_share = 0` 或 `add_to_hold_conflict_share = 0` 单独解释为成功。
- 不得让 simulator guard 继续承担主要策略翻译职责；guard 只能是最后防线。

## 下一步方向
- 短期：r48 已完成正式 screening + confirmatory 但未过 stable confirm；r49 已把 source release dead、exposure utilization floor、cash timing 与资金流守恒写入代码合同；r50 已验证真实 solver 训练入口但计算负荷过高，不作为当前默认推进路径；r51 改走 native allocation vector，优先以轻量 dry-run / 短 screening 验证 `portfolio_native_allocation_vector_terms`、`portfolio_capital_flow_closure_terms` 与 v2 / stability gate。
- 中期：把 monthly return、exposure utilization、receiver realized deploy、source realized sell、positive spread distribution、cash timing 和 drawdown 更深地写进 native target-weight feedback，而不是回到 source / receiver / cash 独立 head 对账。
- 长期：若 r51 formal evidence 有效，再考虑 day-set encoder / set transformer 级别的日级组合模型；simulator guard 只作为最后安全裁剪。

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
- r37：decision-focused allocation 入口，聚焦 source hard-negative、strong false sell、预测 penalty 推理消费与 decision-focused allocation regret。
- r38：source hard-negative regret 入口，聚焦 source hard-negative tail、source release preference、transfer-level allocation regret，以及防错后恢复收益、广度和资金时机。
- r39：当前有效证据基线，聚焦 allocation objective consolidation、final objective execution blend、action loss 降级为辅助，以及 cash timing / drawdown / clean source breadth 的统一收敛。
- r40：当前已完成 clean rerun 的架构入口，聚焦 end-to-end allocation layer、硬 executable candidate 掩码、半可微 allocation optimizer 主路径、持久产物读取，以及 source/receiver/cash credit assignment 未稳定闭合的结构诊断。
- r41-r45：已完成代码合同或 dry-run 的保留检查点，分别聚焦 risk-sensitive、utility-credit、primal-dual decision、entropic transport 与 conservative transport；尚无 study verdict，不再作为默认长训入口。
- r46：已完成代码合同的保留检查点，聚焦 differentiable convex allocation、torch 图内 KKT/constraint residual、data-driven 日级约束 target、executable support、false-source pressure、cash timing、source/receiver shortfall、behavior-support / OPE、路径级 CVaR / drawdown / OCE、legacy action loss 归零与更严格短筛 resource gate；尚无 study verdict。
- r47：已完成代码合同的保留检查点，聚焦真实 `cvxpy` / `cvxpylayers` fixed-slot convex solver layer、DPP 合同、predicted/oracle solver regret、solution tracking、gross / turnover / position / support / cash timing / path risk 诊断、r46 surrogate fallback 与更严格短筛 resource gate；尚无 study verdict。
- r48：当前已完成代码合同与正式研究的失败 profile，聚焦 full-universe aware resource-safe solver、候选覆盖、旧 mask 盲区修复、liquidity / impact / concentration risk、propensity support、OPE lower-bound、doubly-robust gap 与更严格短筛 resource gate；study verdict 为 `research / shadow_only` 且不可 promotion。
- r49：当前已完成代码合同与 dry-run 的最新 research profile，聚焦 capital-flow closure、receiver demand、clean source supply、cash release / defense、exposure gap、flow conservation、source dead 阻断与 false-source 保护；尚无 formal study verdict。
- r50：当前已完成代码合同与 dry-run 的最新 integrated research profile，聚焦真实 convex solver 训练参与、full-universe/OPE、capital-flow closure、risk-sensitive/offline support、fallback diagnostics 修正与短筛 resource gate；尚无 formal study verdict。
- r51：当前最新轻量 research profile，聚焦 native allocation vector、日级目标仓位/现金输出、torch-only projection、从 target delta 自然派生 source / receiver / cash、r49 capital-flow closure 辅助诊断与 r50 solver 高负荷规避；尚无 formal study verdict。

## 历史归档入口
- 早期设计合同原文：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 早期标题索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
- 过程复盘入口：`daily_research/brain/episodic_memory.md`。
- 读取纪律：当前合同以本文件上方章节为准；历史合同只作为证据与演化追溯。

## 2026-05-02 r39 后停环合同
- 主线循环审计产物：`daily_research/output/continuous_policy/analysis/cycle_audits/continuous_policy_cycle_audit_20260502.md` 与同名 JSON。审计结论为存在重复循环，必须停止把 r39 后续工作继续包装成局部 penalty / guard 修补。
- 禁止路线：新增单边 source false-sell penalty、cash penalty、reduce/exit rescue、receiver hard guard，只要仍依赖 `action_budget_split_v1` + `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15` + `portfolio_daily_ranking_v2_gated` 作为主路径，就视为旧方案换皮。
- 保留路线：r20-r23 execution、r31 receiver executable subset、r33 source distribution clean-pass 继续保留为最后安全边界；它们不能再承担主策略收益、资金释放、现金时机的主逻辑。
- 新主线定义：下一阶段必须以 `end-to-end allocation layer` 为名称和目标，让 allocation objective / optimizer 同时决定 source、receiver、cash、turnover、position cap、transaction cost、drawdown 与 monthly quality；simulator guard 只允许做最终安全裁剪。
## 2026-05-02 r40 end-to-end allocation layer 合同
- r40 新入口为 `split_heads_portfolio_daily_end_to_end_allocation_layer_r40`；其 objective 必须是 `end_to_end_allocation_layer_v1`，预算语义必须是 `allocation_layer_v1`，预算校准必须是 `end_to_end_allocation_layer_v1`。该入口不得重新回落到 `action_budget_split_v1` 或 v15 receiver-exec guard 作为主路径。
- r40 执行合同：`solve_semidifferentiable_allocation` 是 source/receiver/cash 的主 allocation 层，显式约束 cash reserve、turnover、position cap、transaction cost、slippage 与 sell tax；`portfolio_simulator.py` 只在其后做安全裁剪、状态更新和诊断记录。
- r40 候选合同：所有 receiver/source 目标必须先通过硬 executable candidate 掩码；`portfolio_daily_receiver_executable_candidate = 0` 或 `portfolio_daily_source_executable_candidate = 0` 时，raw score、action label、unified score 均不得绕过进入最终 allocation target。
- r40 语义合同：direct action open/add/reduce/exit 在 r40 中只能作为辅助表征或兼容输入，不能再生成主策略目标；诊断项 `allocation_layer_primary_mode` 必须明确标记新路径，`direct_action_open_signal_count` 等旧信号不得被伪造为 r40 主目标。
- r40 判定边界：`self_opt_study_r40_end_to_end_allocation_layer_20260502` 仍因外层 stdout 管道失效污染而不能作为有效策略依据；后续 clean rerun `self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504` 已完整产出 3 个 screening 与 2 个 confirmatory，但 stable confirm 为空，confirmatory 均为 `shadow_only`，因此同样不能作为 promotion 依据。
- r40 运行通道合同：超过外层捕获窗口的前台任务必须把 stdout/stderr 写入持久日志；stage 的持久化 JSON 是正式合同，控制台 JSON 只能是诊断输出，不得因 stdout 失效而否定已写出的 train/evaluate/protocol 产物。

## 2026-05-07 r40 clean rerun 合同复盘
- 事实：`self_opt_study_r40_end_to_end_allocation_layer_clean_r1_20260504` 自然完成，`executed_at = 2026-05-07T04:54:27+08:00`，`completed_trial_count = 3`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`，confirmatory diagnostics 显示 `device = cuda`、`cuda_available = true`、`runtime_env = yolos`、`trainer_backend = formal_torch_seq_v3`。
- 事实：screening `trial_02` / `trial_03` 高收益但 `training_evidence_status = insufficient`；confirm_01 / confirm_02 为 `training_evidence_status = sufficient`，但 `portfolio_daily_v2_stable_confirmatory_trials = []`，且均未过 promotion gate。
- 推断：r40 的工程通道已经从 stdout 管道污染中恢复，最终训练结果可读；策略失败不是“结果无法读取”，而是 confirm 层经济质量和稳定性不足。
- 结构约束：后续不得把 screening 高收益但 evidence insufficient 的分支作为 champion，不得把 receiver deploy 数量单独解释为成功；必须同时满足 source 释放、cash timing、drawdown、v2 gate 与 stable confirm。
- 边界：r40 clean_r1 仍是 `research / shadow_only`，当前有效证据基线仍是 r39。
