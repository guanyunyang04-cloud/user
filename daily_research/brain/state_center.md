# Daily Research 状态中枢

快照日期：`2026-04-25`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- 当前 active 执行物化真源为 `daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 为 `regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- continuous_policy 已推进到 r18 direct-action pair-source cost guard repair；当前 research 入口为 `split_heads_direct_action_pair_cost_guard_r18`、`cash_constraint_direct_action_pair_cost_guard_v11`、`direct_action_pair_cost_guard_v1`。
- r15 正式 bounded study 已完成，champion 为 `confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9 + cash_constraint_direct_action_guard_v8`，但 `promotion_status = shadow_only`，不得切换 live。
- r15 事实：收益尚可但执行语义失败，`annual_return = 0.3816`、`sharpe = 1.0867`、`max_drawdown = -0.1493`、`monthly_consistency_score = 0.6703`、`deploy_intent_realized_rate = 0.1642`、`add_to_hold_conflict_share = 0.8524`、`failure_mode = deploy_not_realized`。
- r16 当前最佳 smoke2 事实：复用 r15 champion artifact 后，`annual_return = 0.5807`、`sharpe = 1.6261`、`max_drawdown = -0.1180`、`monthly_return_mean = 0.0337`、`monthly_consistency_score = 0.7247`，但 `direct_action_deploy_authorized_realized_rate = 0.1557`、`add_to_hold_conflict_share = 0.8483` 仍未解决。
- r17 当前最佳 smoke3 事实：`annual_return = 1.0177`、`sharpe = 2.3094`、`max_drawdown = -0.1180`、`monthly_return_mean = 0.0516`、`monthly_consistency_score = 0.7684`、`direct_action_core_deploy_target_realized_rate = 0.9921`、`direct_action_pair_reallocation_source_count = 65`、`add_to_hold_conflict_share = 0.0`；代价是 `avg_turnover = 0.0752` 且出现成对换仓源的 `add -> reduce` 翻译冲突。
- r17 bounded study `cp_v3_direct_action_pair_reallocation_r17__study_r1` 已完成 4 个 screening；自动 confirm 因 `[Errno 22] Invalid argument` 未写出 summary，已用 direct protocol rerun / strict resume 方式补齐 repaired confirm 对照。
- r17 正式对照事实：screening champion 为 `trial_03 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 0.9682`、`sharpe = 2.2180`、`max_drawdown = -0.1183`、`monthly_consistency_score = 0.7703`；repaired confirm champion 为 `confirm_01`，`annual_return = 0.6225`、`sharpe = 2.0071`、`max_drawdown = -0.1128`、`monthly_consistency_score = 0.7374`，仍为 `shadow_only`。
- r18 smoke2 事实：复用 r17 `confirm_01` artifact 后，`annual_return = 0.4891`、`sharpe = 1.9492`、`max_drawdown = -0.1060`、`monthly_return_mean = 0.0322`、`monthly_consistency_score = 0.6693`、`avg_turnover = 0.0286`、`direct_action_pair_reallocation_source_count = 31`、`direct_action_pair_cost_guard_blocked_count = 327`、`direct_action_core_minus_pair_forward_excess_5d = 0.0060`。
- r18 当前结论：v11 已把 pair-source 选择从“能释放资金”推进到“相对 core target 是否值得释放”，但绝对收益低于 r17 repaired confirm，说明它是机会成本守门与审计修复，不是最终 portfolio-level 日决策模型。
- 当前五个根因卡点已固定：个股动作不等于组合决策；局部动作分类会互相打架；卖出/现金/资金来源 credit assignment 最难；日频数据对盘中冲击、滑点、新闻与盘口有盲区；有效独立市场 regime 样本很小。
- 根因排序：第一是目标函数与组合级决策不完全一致；第二是卖出、现金与资金来源的责任分配困难；第三才是数据粒度和 regime 样本不足。
- r11/r11b/r12/r13/r14/r15/r16/r17/r18 全部仍是 `shadow_only` 或 research 证据；不得把 smoke、dry-run、confirm 或 repaired confirm 直接解释成 promotion / live 切换依据。

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
- `split_heads_direct_action_pair_reallocation_r17` 的本质定位是修复 r16 的“所有持仓都想 add 时无人释放预算”问题：先把 direct deploy signal 收敛为少数 core executable target，再允许弱排序 add 持仓作为成对资金来源。
- r17 smoke3 已把 `direct_action_deploy_authorized_realized_rate` 从 r16 的 `0.1557` 修到 `0.9921`，把 `add_to_hold_conflict_share` 从 `0.8483` 修到 `0.0`；但 pair-source 仍需正式 study 验证其长期风险、换手成本和相对机会成本。
- r17 bounded study 进一步证实 core target 成交机制稳定：`trial_03 direct_action_core_deploy_target_realized_rate = 0.9844`，`confirm_01 = 0.9933`；但 formal confirm 收益低于 smoke，且 pair-source 机会成本并非单调稳定。
- `split_heads_direct_action_pair_cost_guard_r18` 的本质定位是修复 r17 的“被减仓股票也可能是好票，只是弱于 core target”的机会成本归因问题：只允许相对 core target 有足够 spread、且继续持有成本可接受的 pair-source 释放资金。
- r18 smoke2 证明 `direct_action_core_minus_pair_forward_excess_5d` 可转正并纳入评分，但 `deploy_intent_not_executable`、`budget_action_entanglement`、`sell_execution_source_entangled` 仍被审计判为高优先级瓶颈。

## 当前优先级
- 冻结 live 默认执行，不做静默切换。
- 把 r17 bounded study 作为当前最佳 `shadow_only` 修复证据，而不是 promotion 候选。
- 若继续研究，应以 r18 为桥梁转向 portfolio-level daily decision：主判据为 pair-source 相对 core target 的 forward excess spread、`avg_turnover`、`cash_timing_quality_1d`、`budget_origin_sell_share`、月度收益质量、listwise/pairwise portfolio ranking 与 promotion gate。
- 不回退到 simulator-only 解释，不把单一 aggregate share 当作结论；需要逐仓明细时读取 held-side detail audit。
- `analyze_behavior_gap.py` 不得并行运行多个会写 `latest_behavior_audit_summary.json` 的实例；长任务按阻塞等待完成，不做无意义轮询。

## 当前边界
- formal、recent、promotion、live 不得混写。
- 不足正式证据的 smoke / dry-run / short-window check 不能升级为正式 verdict。
- continuous_policy 只有在正式协议、参考对照、连续 shadow continuity、行为语义和 promotion gate 均稳定后，才允许进入 promotion 讨论。
- 当前任何 r11/r11b/r12/r13/r14/r15/r16/r17 结果都不改变 active execution artifact。
- r17 虽显著改善 direct add/open 成交率并通过 bounded study 验证机制有效，但 repaired confirm 仍为 `shadow_only`，且 pair-source 机会成本、cash timing 与卖出责任链未闭合，仍不得进入 promotion 讨论。
- r18 虽修复了 pair-source 机会成本审计与部分守门逻辑，但仍不得进入 promotion 讨论；下一阶段不宜继续堆动作 loss，应转向“当前组合中谁获得资金、谁释放资金、释放多少、是否保留现金”的组合级日决策目标。

## 当前风险
- 若身份层再次写入具体 live 默认、最新分数或 winner，属于文档职责漂移。
- 若状态中枢继续堆叠日期段，后续接管会重新退化为长日志扫描。
- 若只看 r17 smoke3 的收益改善，会掩盖 pair-source `add -> reduce` 冲突、换手上升和长期机会成本尚未正式验证的问题。
- 若只看 r17 screening champion，会掩盖 repaired confirm 收益回落、`cash_timing_quality_1d` 仍弱和 pair-source 排序不稳定的问题。
- 若只看 r18 的 `core-minus-pair` 转正，会掩盖绝对收益回落和组合级目标仍未端到端学习的问题。
- 若并行运行行为审计，仍可能重现 latest 摘要文件竞争。

## 推荐下一步
- 若继续研究：以 r18 产物为基线，设计组合级 pair/listwise daily decision 目标，直接学习资金获得者、资金释放者、释放幅度与现金保留，而不是继续拆开训练 open/add/hold/reduce/exit。
- 若继续修复：优先把卖出、现金与资金来源 credit assignment 接入训练和审计，并用月度收益质量验证，而不是继续扩大 pair-source 数量或堆动作 loss。
- 若继续维护：保持入口文档轻量，把过程证据写入 `episodic_memory.md` 或 `brain/references/`。
- 若需要旧状态细节：按下方索引读取历史原文，不把归档历史自动提升为当前状态。

## 历史归档入口
- 原 `state_center.md` 已原样归档：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- 原始行数：`1863`。
- 原始 SHA256：`1c135d58f6e1951cf60c8bd2234522ccc5357755953f6962d9070ec67b7a1e16`。
- 读取纪律：当前状态以本文件上方章节为准；归档文件只作为历史证据与追溯入口。
## 2026-04-25 r19 组合日频排序状态
- 事实：新增 research entry 为 `split_heads_portfolio_daily_ranking_r19` + `cash_constraint_portfolio_daily_ranking_guard_v12` + `portfolio_daily_ranking_v1`。
- 事实：r19 把 r18 pair-source guard 转成组合日频 receiver/source/cash ranking；资金接收方、资金来源和现金保留会同时进入模拟器、pipeline metrics、behavior gap analysis 和 study scoring 审计。
- 事实：r19 仍只属于 research/shadow 证据；没有 live 或 promotion 含义。
- 决策：下一条正式证据必须比较 receiver vs source forward excess、source realized sell rate、现金保留行为、月度质量、换手和回撤。
## 2026-04-25 r19 核验状态
- 事实：`verify_portfolio_daily_ranking_r19_v12_confirm01_smoke_20260425` 已完成，结果为 `annual_return = 0.843131`、`sharpe = 2.072912`、`max_drawdown = -0.122117`、`monthly_return_mean = 0.051062`、`monthly_consistency_score = 0.723549`、`avg_turnover = 0.048518`。
- 事实：r19 smoke 产生 `portfolio_daily_receiver_target_count = 392`、`portfolio_daily_source_target_count = 308`、`portfolio_daily_source_realized_sell_rate = 0.551948`、`portfolio_daily_receiver_minus_source_forward_excess_5d = 0.002383`。
- 风险：`portfolio_daily_source_forward_excess_5d = 0.010149` 仍为正，因此 sell/opportunity-cost attribution 已经足够进入审计，但还没有解决到可 promotion 的程度。
## 2026-04-26 r19 bounded study 状态
- 事实：`cp_v3_portfolio_daily_ranking_r19__study_r1` 完成 `4` 个 screening trial 和 `2` 个 confirmatory trial，`failed_trial_count = 0`；latest state 已恢复到 `cp_v3_direct_action_pair_reallocation_r17__study_r1__confirm_02`。
- 事实：所有 r19 训练诊断均使用 `formal_torch_seq_v3`、`device = cuda`、`cuda_available = true`、strict resume 和显式 `yolos` 解释器路径。
- 事实：表现线 confirm 为 `confirm_01 = alpha_result_value_budget_split_v15 + result_value_v9`，指标为 `annual_return = 0.967208`、`sharpe = 2.394526`、`max_drawdown = -0.154689`、`monthly_return_mean = 0.053307`、`monthly_consistency_score = 0.772378`、`avg_turnover = 0.035627`。
- 事实：`confirm_01` 仍有 `portfolio_daily_receiver_minus_source_forward_excess_5d = -0.008111`、`cash_timing_quality_1d = -0.053872`、`add_to_hold_conflict_share = 0.410628`、`order_translation_conflict_rate = 0.305556`。
- 事实：综合稳定线 confirm 为 `confirm_02 = alpha_result_value_budget_split_v14 + result_value_v9`，receiver-source separation 更强（`portfolio_daily_receiver_minus_source_forward_excess_5d = 0.031289`、`portfolio_daily_source_realized_sell_rate = 0.692308`），但收益为负（`annual_return = -0.221432`、`sharpe = -0.565395`、`max_drawdown = -0.187207`）。
- 决策：r19 bounded evidence 证明组合级框架有价值，但 `portfolio_daily_ranking_v1` 目前相对真实收益/回撤过度奖励 receiver-source spread；r19 仍是 `research / shadow_only`，不能进入 promotion 讨论。
- 已更新操作规则：所有 `daily_research` 任务必须在 `yolos` 下前台运行，不得中断，使用 `10h` 前台窗口；GPU 训练证据写成正式证据前必须从 `training_diagnostics.json` 核验。

## 2026-04-26 r20 v2/v13 当前状态
- 事实：`split_heads_portfolio_daily_ranking_r19` 的默认目标已从 `portfolio_daily_ranking_v1` 升级为 `portfolio_daily_ranking_v2_gated`；默认预算校准已从 v12 升级为 `cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13`。
- 事实：新增反冠军诊断工具 `daily_research/tools/portfolio_daily_ranking_gate_report.py`，可离线重算 v1/v2 分数、列出 gate 失败项，并生成 `portfolio_daily_ranking_v2_gate_report.md/json` 与 `portfolio_daily_ranking_v2_rescore.csv`。
- 事实：r19 离线重排显示，v1 champion `confirm_02` 因 `annual_return = -0.221432`、`sharpe = -0.565395` 被 v2 降为负分；v2 champion 变为 `confirm_01`，但它仍有 `order_translation_conflict_rate = 0.305556`、`add_to_hold_conflict_share = 0.410628`、`cash_reserve_rate = 0.0`，不能进入 promotion。
- 事实：r20 retry2 smoke 通过全部 v2 gate，关键指标为 `annual_return = 0.316303`、`sharpe = 1.152146`、`max_drawdown = -0.129243`、`monthly_return_mean = 0.020376`、`monthly_consistency_score = 0.727527`、`portfolio_daily_receiver_minus_source_forward_excess_5d = 0.004380`、`portfolio_daily_source_realized_sell_rate = 0.8`、`portfolio_daily_cash_reserve_rate = 0.009479`、`order_translation_conflict_rate = 0.033175`、`add_to_hold_conflict_share = 0.072464`。
- 事实：bounded confirmatory 暴露 fresh 8 epoch confirm 失稳，`annual_return = -0.253312`、`sharpe = -0.587612`、`max_drawdown = -0.220648`、`monthly_return_mean = -0.015989`、`add_to_hold_conflict_share = 0.578947`；该 confirm 只能作为失败证据，不能被 champion selector 自动扶正。
- 已修正：v2 champion selection 改为“stable confirmatory 先过 v2 gate 和 confirm-vs-screening 稳定性检查才可优先；否则回退”，并记录 `champion_selection_policy`、`portfolio_daily_v2_confirm_stability_checks` 与 `rejected_confirmatory_trials`。
- 决策：r20/v2/v13 是当前最有效突破口，已修复 v1 奖励错位、现金死分支和候选动作误判；但 fresh confirm 稳定性未过，仍保持 `research / shadow_only`，不进入 live 或 promotion 讨论。

## 2026-04-26 r20 stability sweep 状态
- 事实：新增 `split_heads_portfolio_daily_ranking_stability_r20`，用于 v2/v13 稳定性搜索；baseline 使用较低学习率 `0.001`、较高 `dropout = 0.16`、`daily_dropout = 0.10`，候选覆盖 v14/v15、v9/v10、低学习率和高 dropout 组合。
- 事实：`cp_v3_portfolio_daily_ranking_r20_stability_sweep_20260426` 已完成 `6` 个 screening trial 和 `2` 个 confirmatory trial，`failed_trial_count = 0`，latest state 已恢复到 `cp_v3_direct_action_pair_reallocation_r17__study_r1__confirm_02`。
- 事实：8 个训练目录均为 `trainer_backend = formal_torch_seq_v3`、`device = cuda`、`cuda_available = true`、strict resume；本轮命令由 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` 前台启动并自然结束。历史 diagnostics 尚未写入解释器路径，代码已修复未来训练会写入 `python_executable`、`conda_prefix` 与 `runtime_env`。
- 事实：现金分支已经转活，本轮 gate report 中没有路线触发 `cash_branch_alive`；`cash_score >= 阈值` 天数合计 `139`，`cash reserve` 天数合计 `139`。
- 事实：加入 `source_realized_sell_floor` 后，本轮没有合格 v2 champion。按分数 top-ranked 的 `confirm_01` 为 `annual_return = 0.570476`、`sharpe = 1.448580`、`max_drawdown = -0.182756`，但触发 `max_drawdown_floor`；v1 top `confirm_02` 为 `annual_return = 0.465136`、`sharpe = 1.419674`、`max_drawdown = -0.171690`，但 `source_realized_sell_rate = 0.088889`，触发 `source_realized_sell_floor`。
- 决策：r20 的瓶颈已从“现金死分支”推进到“source 被选中但没有真实释放资金”和“fresh confirm 回撤边界”；当前仍为 `research / shadow_only`，不得 promotion 或 live。

## 2026-04-26 r21 source-exec guard 当前状态
- 事实：新增预算校准 `cash_constraint_portfolio_daily_ranking_source_exec_guard_v14` 与 search profile `split_heads_portfolio_daily_ranking_source_exec_r21`，目标是把 source target 从“被选中”推进到“真实 reduce/exit 释放资金”。
- 事实：v14 在模拟器执行链路中降低 source target 的保留底线、提高 source 的 sell reduction priority、压低其 deploy priority，并在 translation guard 中绕过原先 add/hold 保护导致 source 被锁回持有的路径。
- 事实：审计新增 `portfolio_daily_source_target_not_sold_share`、`portfolio_daily_source_exec_cap_guard_count`、`portfolio_daily_source_realized_reduction_weight`、`portfolio_daily_receiver_realized_deploy_count` 与 `portfolio_daily_effective_capital_transfer_count`；gate report 新增 `source_not_sold_ceiling`。
- 事实：第一次 r21 smoke 暴露 v14 未继承 v13 cash-aware 分支，已修复为 v14 同样使用 cash-aware competition score、`0.24` cash threshold 和 `0.72` receiver pressure ceiling。
- 事实：`cp_v3_portfolio_daily_ranking_r21_source_exec_smoke_20260426_retry2` 已前台自然完成，GPU 诊断为 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、strict resume。
- 事实：retry2 关键指标为 `annual_return = 0.126142`、`sharpe = 0.677610`、`max_drawdown = -0.112456`、`monthly_return_mean = 0.008781`、`monthly_consistency_score = 0.564085`、`portfolio_daily_receiver_target_count = 53`、`portfolio_daily_source_target_count = 55`、`portfolio_daily_source_realized_sell_rate = 1.0`、`portfolio_daily_source_target_not_sold_share = 0.0`、`portfolio_daily_effective_capital_transfer_count = 17`、`portfolio_daily_cash_reserve_rate = 0.097872`。
- 事实：retry2 仍无合格 v2 champion，失败 gate 为 `order_translation_conflict_ceiling` 与 `add_to_hold_conflict_ceiling`；对应 `order_translation_conflict_rate = 0.289362`、`add_to_hold_conflict_share = 0.454545`。
- 决策：r21 已把第一瓶颈从 source execution 推进到 order translation / add-to-hold 冲突；当前仍是 `research / shadow_only`，不得 promotion 或 live。
