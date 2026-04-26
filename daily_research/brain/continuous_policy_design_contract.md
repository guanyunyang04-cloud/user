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
- 个股未来上涨不等于今天该加仓；组合级目标必须判断“谁获得资金、谁释放资金、释放多少、是否保留现金”，不能只做个股动作分类。
- 卖出、现金与资金来源是同一条 credit assignment 链：释放旧仓不是因为它一定差，而是因为在当前组合状态下继续持有的机会成本可能高于把资金给 core target。
- 日频数据只能学习趋势、强弱、回撤和量价结构；对盘中冲击、真实滑点、新闻驱动、盘口流动性和开盘跳空过程必须保留盲区假设。
- 2024-2026 这类训练/验证切分的有效独立 regime 数量有限；月度样本更少，任何高收益分支都必须防止学到单段市场风格。
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
- r17 合同：`cash_constraint_direct_action_pair_reallocation_guard_v10` 必须把 direct deploy signal 与 core executable target 分开；在预算饱和日，只允许少数高排序 add/open 成为 core target，并允许弱排序 add 持仓作为显式 pair reallocation source。
- r17 边界：pair reallocation source 是独立可审计的资金来源，不得混入 budget-origin sell；`direct_action_core_deploy_target_realized_rate`、`direct_action_pair_reallocation_source_count`、pair-source 机会成本、换手和月度收益质量必须联合判断。
- r18 合同：`cash_constraint_direct_action_pair_cost_guard_v11` 必须把 pair-source 从“可释放资金”升级为“相对 core target 机会成本可接受”；`direct_action_core_minus_pair_forward_excess_5d`、`direct_action_pair_source_opportunity_cost`、blocked count、换手和月度收益质量必须联合判断。
- r18 边界：`split_heads_direct_action_pair_cost_guard_r18` 是从规则桥走向组合级日决策的过渡层，不得被解释为最终模型；下一阶段目标应转为 pair/listwise portfolio ranking 与多日/月度组合收益直接优化。

## 当前已知事实
- 当前已证实：`alpha_result_value_budget_split_v11` 能让 funding-sell 更少、更干净。
- 当前已证实：`alpha_result_value_budget_split_v12 + result_value_v9` 可以在 r12 confirm_02 中拿到更高收益，但不能自动闭合三方语义。
- 当前已证实：`alpha_result_value_budget_split_v13 + result_value_v10` 可以把显性动作价值冲突压到 `action_value_conflict_share = 0.0`，但仍未通过 promotion gate。
- 当前已证实：r14 直接动作价值仲裁能显著改善收益质量，但低边际动作与 held-side funding rebalance 仍未闭合。
- 当前已证实：r15 formal 全窗暴露 `deploy_not_realized`，budget clipping 日的订单翻译冲突显著高于 unclipped 日。
- 当前已证实：r16 smoke2 改善 return/risk 并恢复非零 reallocation source，但 `direct_action_deploy_authorized_realized_rate = 0.1557` 仍明显不足。
- 当前已证实：r17 smoke3 在同一 r15 champion artifact 上把 `direct_action_core_deploy_target_realized_rate` 提升到 `0.9921`，把 `add_to_hold_conflict_share` 压到 `0.0`，并改善收益与月度一致性。
- 当前已证实：r17 bounded study 的 screening champion `trial_03` 与 repaired confirm champion `confirm_01` 均保持高 core target 成交率，说明 v10 成对换仓机制不是单次 smoke 偶然。
- 当前已证实：r17 pair-source 机会成本并非单调稳定；`trial_03 / confirm_01 / confirm_02` 的 core-minus-pair 5 日超额均值为正，但 `trial_01 / trial_02` 不成立或接近 0。
- 当前已证实：r18 v11 可以把 `direct_action_core_minus_pair_forward_excess_5d` 修为正值并显著降低换手，但绝对收益低于 r17 repaired confirm，说明机会成本守门有效但组合级目标仍未端到端闭合。
- 当前未证实：`result_value_v10` 尚不能升为稳定预算目标，r18 尚不能升为 production 或 promotion 证据；cash timing、卖出责任链和 portfolio-level objective 仍需后续修复。

## 当前禁止事项
- 不得把 r11/r11b/r12/r13/r14/r15/r16/r17 任一分支写成 promotion 或 live 切换依据。
- 不得只看 `budget_origin_sell_share = 0` 就宣布 sell-source 成功。
- 不得只看自动 confirm 排序；必须保留语义对照和 held-side detail 证据。
- 不得并行运行多个会写 latest 行为摘要的审计。
- 不得在 release consistency 或 authorized deploy realization 仍低时声称 release / deploy 已学成。
- 不得把 r16 smoke2 的收益改善解释为可上线；只要 add->hold 冲突和 authorized deploy 未实现仍高，就只能作为修复性 shadow 证据。
- 不得把 r17 smoke3 或 r17 screening champion 的高收益和高成交率解释为可上线；repaired confirm 仍是 `shadow_only`，且 pair-source 机会成本、换手、cash timing 与长窗稳定性仍未闭合。
- 不得继续把主要矛盾简化成堆动作 loss；open/add/hold/reduce/exit 分开学会强化局部目标冲突，必须向组合级日决策目标收敛。

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
- 成对再分配层：`direct_action_deploy_signal_count / direct_action_core_deploy_target_count / direct_action_core_deploy_target_realized_rate / direct_action_pair_reallocation_source_count / direct_action_pair_reallocation_sell_share` 必须和 pair-source 逐仓 forward excess、换手和月度收益质量一起看。
- 成对机会成本层：`direct_action_pair_cost_guard_pass_rate / direct_action_pair_source_spread_mean / direct_action_pair_source_cost_mean / direct_action_core_minus_pair_forward_excess_5d` 必须和真实收益、换手、cash timing 与月度收益质量一起看。
- 责任链：sell-source、funding source、held-side release support、protected-hold conflict 与 reallocation source 必须可审计。

## 下一步方向
- r18 已把 pair-source 成本守门落地；下一步应转向组合级日决策模型，而不是继续在固定动作集合里做局部桥接。
- 后续若调整 loss / objective / simulator，必须随行补跑 source attribution、held-side detail 和 budget-clipped 分层审计。
- 后续模型应直接学习“当前组合状态下的仓位调整集合”，通过 pair/listwise ranking、多日收益风险成本与月度收益质量联合优化，显式解决资金获得、资金释放和现金保留。

## 历史归档入口
- 原 `continuous_policy_design_contract.md` 已原样归档：`daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/continuous_policy_design_contract_evidence_index_20260424.md`。
- 原始行数：`581`。
- 原始 SHA256：`b3f83530ec585e284a60e568bd66be4f7c8402d455efec20944af41e58f0afe8`。
- 读取纪律：当前设计合同以本文件上方章节为准；r1-r11b 的完整合同演化只作为历史证据。
## 2026-04-25 r19 组合日频排序合同
- r19 把学习/评估问题从孤立的 `open/add/hold/reduce/exit` 标签改为日频组合分配：哪些标的接收资金，哪些标的释放资金，以及是否应保留现金。
- 必要成功证据包括正向 `portfolio_daily_receiver_minus_source_forward_excess_5d`、非平凡 `portfolio_daily_source_realized_sell_rate`、受控 `portfolio_daily_cash_reserve_rate`、可接受换手，以及不扩大回撤前提下更好的月度一致性。
- source stock 仍可能是好股票；只有在同一组合状态下它的机会成本低于被选 receiver 时，它才是有效资金来源。
- r19 是通向 listwise/pairwise 组合决策学习的 research 桥，不是 live execution profile。

## 2026-04-26 r20 v2/v13 合同
- `portfolio_daily_ranking_v2_gated` 必须 gate-first：`annual_return`、`sharpe`、`monthly_return_mean` 必须为正，`max_drawdown` 不得低于 `-0.18`，`monthly_consistency_score` 不得低于 `0.45`，执行冲突、add-to-hold 和现金行为必须达标后，receiver-source spread 才能获得主要奖励。
- `cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13` 必须让现金保留成为真实竞争分支；若 portfolio daily ranking 已观测但 `portfolio_daily_cash_reserve_rate = 0.0`，不得把该分支写成组合级完成态。
- 组合排序模式下的冲突指标必须优先使用 `portfolio_daily_effective_model_action`；未被选为 core receiver 的原始 `add/open` 只是候选意图，不得直接当成最终 add/open 失败。
- v2 champion selection 必须拒绝失败 confirmatory：只有通过 v2 gate 且通过 confirm-vs-screening 稳定性检查的 confirmatory 才能优先成为 champion；否则必须回退，并把失败 confirm 写入 `rejected_confirmatory_trials`。
- `stable_confirmatory` 必须同时满足 v2 gates 与 confirm-vs-screening 稳定性检查；source target 必须真实释放资金，`source_realized_sell_floor` 要求当 `portfolio_daily_source_target_count >= 5` 时，`portfolio_daily_source_realized_sell_rate` 不得低于 `0.35`；否则 receiver-source spread 不能被视为完成态。
- r20 成功判定不是单次 smoke 过 gate，而是在 bounded/fresh confirm 下同时保持正收益、受控回撤、正向月度收益、非零且合理现金保留、正向 receiver-source spread、足够 source realized sell、低 order translation conflict 和低 add-to-hold conflict。

## 2026-04-26 r21 source-exec 合同
- `cash_constraint_portfolio_daily_ranking_source_exec_guard_v14` 必须把 source target 的真实 reduce/exit 放进执行生成层：降低 source target 保留底线、提高 source sell priority、压低 source deploy priority，并避免原始 add/hold translation floor 把 source 锁回持有。
- v14 必须继承 v13 的 cash-aware competition：使用弱 receiver、弱市场宽度、高暴露、高换手、稀疏 receiver、真实回撤和 defense gate 共同激活现金保留；不得因修 source execution 让 cash branch 再次死亡。
- `split_heads_portfolio_daily_ranking_source_exec_r21` 使用 `portfolio_daily_ranking_v2_gated`，但新增 source execution 证据：`portfolio_daily_source_target_not_sold_share`、`portfolio_daily_source_exec_cap_guard_count`、`portfolio_daily_source_realized_reduction_weight`、`portfolio_daily_receiver_realized_deploy_count` 与 `portfolio_daily_effective_capital_transfer_count`。
- `source_not_sold_ceiling` 是 `source_realized_sell_floor` 的配套约束：当 `portfolio_daily_source_target_count >= 5` 时，source target 未真实卖出的占比不得高于 `0.65`。
- r21 成功判定必须同时看 source 真实释放、receiver 真实成交和 effective capital transfer；只改善 source realized sell 但牺牲收益、回撤、月度一致性或现金行为，仍不能 promotion。
