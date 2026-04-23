# 研究日志

## 当前结论

- 本文件现在是 `daily_research` 分脑的轻量 episodic 入口，不再承载全部历史原文。
- 当前三层结构固定为：
  - 当前结论：本文件保留最新可执行结论、当前证据判断和后续写回纪律。
  - 证据索引：`daily_research/brain/references/episodic_memory_evidence_index_20260422.md`。
  - 历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 当前接管默认仍然先读 `identity_layer.md -> state_center.md -> knowledge_center.md -> operations_center.md`；只有需要完整过程证据时才进入本文件和历史原文。
- 当前生产 / 执行结论仍以 `state_center.md` 为准；本文件只记录过程证据、归档索引和动作后复盘。
- 当前默认执行链仍是 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`，默认 production root 仍是 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前默认执行分数语义已修为 selected composite decision score，对外字段使用 `model_decision_score` / `模型综合决策分`；`learned_score` 只作为 sub-head / debug 信号。
- 当前 continuous_policy 研究主矛盾仍是 `deploy intent not executable`：r9 已说明 `clip reduction != deploy executability`，后续不得把减少 budget clip 误当作执行意图闭环。
- 当前 r7 / r8 / r9 的证据关系：
  - r7：hierarchy 方向正确，但信用分配没有闭环。
  - r8：constraint-only portfolio defense 方向正确，但 budget clipping 与 deploy alignment 仍未闭环。
  - r9：`cash_constraint_intent_guard_v5` 能压低 clip，但主要冲突暴露为 `add -> hold`。
- 当前 r10 已把 `deploy intent not executable` 从经验判断推进为可执行工程闭环：
  - 新增 `deploy_executability_target` 派生信号。
  - 新增 `result_value_v9`、`alpha_result_value_budget_split_v9`、`cash_constraint_deploy_guard_v6`。
  - 新增 `split_heads_deploy_executability_r10` 和 `deploy_executability_v1` search scoring。
  - 行为审计现在显式输出 `deploy_intent_realized_rate`、`open_add_positive_weight_change_rate`、`add_to_hold_conflict_share` 与 `deploy_intent_dropped_share`。
- 当前 r10 bounded self-opt 已完成，当前 deploy-executability objective 冠军为 `cp_v3_deploy_executability_r10__study_r1__confirm_02`；但结论仍是 `shadow_only`，不能作为 promotion 或 live 切换依据。
- 当前默认纪律：不启动训练、不切换 live 默认执行、不改写 promotion 结论，除非用户明确要求或 `state_center.md` 已更新为新的正式决策。
- 当前剩余维护风险：`daily_research/output` 与 `daily_research/cache` 仍是大体量热产物区；没有明确保留策略前继续不裁剪。

## 证据索引

- 详细标题索引：`daily_research/brain/references/episodic_memory_evidence_index_20260422.md`。
- 历史原文：`daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 归档前原文件规模：`861319` UTF-8 bytes，约 `16573` 行，`302` 个二级章节。

| 证据层 | 入口 | 使用场景 |
| --- | --- | --- |
| 当前结论 | `daily_research/brain/episodic_memory.md` | 接管时快速确认最新过程结论和索引位置 |
| 证据索引 | `daily_research/brain/references/episodic_memory_evidence_index_20260422.md` | 需要按日期、阶段或标题定位旧证据 |
| 历史原文 | `daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md` | 需要完整复盘、核查指标、追溯原始实验叙述 |

## 历史原文

- 历史原文已原样归档到 `daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
- 该归档保存 2026-03-17 至 2026-04-22 的长过程记录，包括 baseline、advanced_ml、deep_alpha、execution、short_expert、continuous_policy、默认执行维护和主分脑治理。
- 历史原文只作为证据，不自动覆盖当前状态；若与 `state_center.md`、`knowledge_center.md` 或最新正式验证冲突，以当前中心文档和最新验证为准。
- 后续新增 episodic 记录默认先写入本文件；当记录再次膨胀时，再按同样方式生成新的证据索引和历史原文归档。

## 2026-04-22 episodic_memory 历史归档瘦身

- 动作前自检：
  - 用户明确要求专门做一次 `episodic_memory.md` 历史归档瘦身，并分层为“当前结论 / 证据索引 / 历史原文”。
  - 本轮只处理脑文档结构，不启动训练、不停止训练、不切换 live 默认执行、不改写 promotion 结论。
  - 归档原则是保全优先：旧正文先原样归档，再生成索引，最后重写轻量入口。
- 已完成实现：
  - 将归档前完整 `episodic_memory.md` 原样保存为 `daily_research/brain/references/episodic_memory_history_raw_20260317_20260422.md`。
  - 生成 `daily_research/brain/references/episodic_memory_evidence_index_20260422.md`，按阶段和二级标题行号索引历史原文。
  - 将当前 `episodic_memory.md` 精简为三层入口：当前结论、证据索引、历史原文。
- 动作后复盘：
  - 事实：历史证据没有删除，只从默认接管入口下沉到 `brain/references/`。
  - 推断：后续接管成本会明显下降，因为 agent 不再需要默认加载 1.6 万行长日志。
  - 决策：未来只有需要完整证据链时才打开历史原文；日常接管以当前结论和索引为入口。

## 2026-04-22 r10 部署可执行性闭环落地

- 动作前自检：
  - 事实：r9 暴露的主冲突是 `add -> hold`，且历史经验已经说明 `clip reduction != deploy executability`。
  - 事实：默认生产链仍是 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`，本轮不得静默切换 live。
  - 假设：最有效的下一步不是扩 backbone 或扩大 universe，而是让系统先能度量、约束并搜索“部署意图是否真实形成正向权重变化”。
- 已完成实现：
  - 在 label / policy output / budget objective / simulator / behavior audit / continuity metrics 中接入 `deploy_executability_target` 和部署可执行性指标。
  - 新增 `result_value_v9`、`alpha_result_value_budget_split_v9`、`cash_constraint_deploy_guard_v6`。
  - 新增 self-opt 搜索入口 `split_heads_deploy_executability_r10`，默认 scoring 为 `deploy_executability_v1`。
  - 未新增神经网络权重头，避免破坏旧 artifact 的加载兼容性。
- 验证：
  - `py_compile` 通过。
  - `run_continuous_policy_protocol --help` 已暴露 `result_value_v9 / alpha_result_value_budget_split_v9 / cash_constraint_deploy_guard_v6`。
  - `run_self_optimizing_study --help` 已暴露 `split_heads_deploy_executability_r10`。
  - r10 dry-run `cp_v3_deploy_executability_r10__dry_run` 正确选择 r10 baseline。
  - r10 smoke `cp_v3_deploy_executability_r10__smoke01` 已跑通完整 protocol。
- smoke 结果：
  - `promotion_gate.status = shadow_only`。
  - `training_evidence.status = insufficient`，因为本轮只跑 `1` epoch。
  - 行为审计中 `deploy_intent_action_count = 4`、`deploy_intent_realized_rate = 1.0`、`open_add_positive_weight_change_rate = 1.0`、`add_to_hold_conflict_share = 0.0`、`deploy_intent_dropped_share = 0.0`、`avg_deploy_intent_candidate_budget_drop_share = 0.0714285714`。
- 动作后复盘：
  - 事实：r10 工程闭环已可执行，且能把 `add -> hold` 从泛化订单冲突中单独量化出来。
  - 推断：下一轮正式研究应使用 r10 profile 做 bounded self-opt / full formal；不能把本次 `1` epoch smoke 当作模型效果结论。
  - 决策：live 默认执行、active production root 与 promotion 判断保持不变；r10 只进入 continuous_policy research backlog。

## 2026-04-22 r10 bounded self-opt 前台完整执行

- 动作前自检：
  - 事实：r10 smoke 只证明链路可执行，不足以回答“哪组 loss / calibration / objective 组合最值得继续 formal 推进”。
  - 事实：r10 搜索空间只有 `8` 个 screening 组合，属于应该一次扫清的 bounded study，而不是继续凭经验猜测。
  - 事实：用户明确要求前台跑、训练不要中断、时限 `10h`。
  - 假设：在当前信息下，最有效动作是完整跑完 `split_heads_deploy_executability_r10` 的 `8` 个 screening trial 与 `2` 个 confirmatory trial，而不是先缩回更小预算。
- 已完成执行：
  - 前台完整执行 `cp_v3_deploy_executability_r10__study_r1`，命令为 `run_self_optimizing_study --search-profile split_heads_deploy_executability_r10 --trial-count 8 --confirmatory-max-candidates 2 --study-tag cp_v3_deploy_executability_r10__study_r1`。
  - Study 在约 `3.45h` 内完成，`completed_trial_count = 8`、`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`。
  - confirmatory 使用了独立 train run，而不是复用 screening 目录；两条冠军线都在更长预算下复现回原 screening 最优 checkpoint。
- 关键结果：
  - deploy-executability objective 冠军：`cp_v3_deploy_executability_r10__study_r1__confirm_02`
    - `loss_profile = alpha_result_value_budget_split_v9`
    - `budget_calibration = cash_constraint_deploy_guard_v6`
    - `budget_objective = result_value_v9`
    - `composite_score = 8.588851`
    - `training_evidence.status = sufficient`
    - `promotion_gate.status = shadow_only`
  - performance champion：`cp_v3_deploy_executability_r10__study_r1__confirm_01`
    - `budget_objective = result_value_v8`
    - raw `annual_return / sharpe` 更高，但 `cash_timing_quality_1d` 与 `max_drawdown` 仍不过 gate
  - screening 的结构性分层很清楚：
    - `cash_constraint_deploy_guard_v6` 平均 `deploy_intent_realized_rate = 0.8195`，远高于 `cash_constraint_intent_guard_v5 = 0.3235`
    - `cash_constraint_deploy_guard_v6` 平均 `add_to_hold_conflict_share = 0.0503`，远低于 `cash_constraint_intent_guard_v5 = 0.7594`
    - 说明 `v5` 在 r10 目标下已经退化成“把 deploy intent 压回 hold”的分支，不应继续作为主线
- 与 r9 confirmatory champion 的同层对照：
  - 改善：
    - `reduce_success_rate_5d: 0.5 -> 0.8571`
    - `exit_timeliness_rate_5d: 0.4 -> 0.6667`
    - `cash_timing_quality_1d: -0.0367 -> -0.0156`
    - `semantic_conflict_rate: 0.0108 -> 0.0019`
  - 未改善：
    - `annual_return: 0.7203 -> 0.2918`
    - `sharpe: 2.3841 -> 1.1732`
    - `order_translation_conflict_rate: 0.2950 -> 0.3070`
- 动作后复盘：
  - 事实：r10 方向已经从“工程上可执行”推进到“结构性优胜组合已跑清”，主胜线明确落在 `v6`，不是 `v5`。
  - 推断：当前真正的主瓶颈已经不是 `add -> hold` 本身，而是 `cash_timing_quality_1d`、`deploy_intent_candidate_budget_drop_share` 与 `budget_action_entanglement`。
  - 决策：后续若继续推进 r10，应固定 `cash_constraint_deploy_guard_v6`，在其上继续拆预算头与动作头，而不是回退到 `cash_constraint_intent_guard_v5`。

## 2026-04-22 r10 预算/动作解耦补丁验证

- 动作前自检：
  - 事实：r10 正式 study 已经收口，但核心瓶颈仍是 `hold/add` 被微幅负向 `delta` 扭成 `reduce`，以及“旧仓一律先占坑”导致的 candidate budget 饱和。
  - 事实：当前最有效动作不是再猜下一组 profile，而是先验证代码层预算/动作解耦补丁能否直接改善现有冠军模型的执行表现。
  - 假设：如果 patched simulator 能在不改模型权重的前提下明显改善收益与 deploy realization，则说明当前存在可直接修复的工程性漂移。
- 已完成实现：
  - 在 `portfolio_simulator.py` 中补强了 `hold/add` 的微幅负向 `delta` 守卫。
  - 在 `action_budget_split_v1` 下新增弱持仓 slot 竞争，并补充 `budget_reclaimable_held_count` / `budget_released_held_count` / `budget_released_from_hold` 诊断字段。
  - 用现有冠军模型 `cp_v3_deploy_executability_r10__study_r1__confirm_02` 重跑 patched 评估与行为审计。
- 结果：
  - patched eval：`cp_v3_deploy_executability_r10__study_r1__confirm_02__budget_fix_eval`
  - patched audit：`cp_v3_deploy_executability_r10__study_r1__confirm_02__budget_fix_audit`
  - 关键指标改善：
    - `annual_return: 0.2918 -> 0.9022`
    - `sharpe: 1.1732 -> 3.5963`
    - `max_drawdown: -0.0826 -> -0.0683`
    - `avg_order_translation_conflict_rate: 0.3149 -> 0.2787`
    - `deploy_intent_realized_rate: 0.9050 -> 0.9312`
    - `add_to_hold_conflict_share: 0.0219 -> 0.0000`
    - `cash_timing_quality_1d: -0.0156 -> -0.0010`
  - 同时暴露出的恶化项：
    - `reduce_success_rate_5d: 0.8571 -> 0.2500`
    - `exit_timeliness_rate_5d: 0.6667 -> 0.1429`
    - `sell_selection_quality_5d: 0.0826 -> -0.1009`
- 动作后复盘：
  - 事实：这次补丁已经证明，r10 当前有一部分损失来自预算/动作翻译层本身，而不是只能靠重新训练解决。
  - 推断：deploy side 的工程漂移被压住以后，sell-side / release-side 的真实学习短板会更清楚地暴露出来。
  - 决策：维持 `shadow_only`；后续若继续 formal 推进，必须在 patched simulator 上重跑新的训练级 protocol / study，而不是把 patched eval 直接当成 promotion 证据。

## 2026-04-22 r10 卖出来源归因复跑

- 动作前自检：
  - 事实：上一轮 patched eval 已证明 deploy-side 工程漂移存在，但还不能回答“当前真实卖出主要是谁造成的”。
  - 事实：若不把模型主动卖出与 budget-origin 被动卖出拆开，后续继续修 sell-side 会反复把预算层问题误判成 release gate 问题。
  - 假设：最有效动作不是再开一轮大搜索，而是先对当前 patched protocol 做来源归因复跑，把 sell-side 责任链拆清。
- 已完成实现：
  - 在 `portfolio_simulator.py` 中为 action outcome 新增：
    - `sell_execution_origin`
    - `sell_suppression_origin`
    - `translation_floor_guarded`
    - `translation_cap_guarded`
    - `sell_priority_guarded`
  - 在 `analyze_behavior_gap.py` 中新增 sell-source attribution 统计、诊断与瓶颈：
    - `sell_execution_source_entangled`
    - `cash_timing_still_passive`
  - 已生成：
    - `cp_v3_deploy_executability_r10__budget_fix_protocol_r1__source_eval`
    - `cp_v3_deploy_executability_r10__budget_fix_protocol_r1__source_audit`
- 动作后复盘：
  - 事实：
    - `source_eval` 收益弱于正式 protocol：`annual_return = -0.1140`、`sharpe = -0.3284`、`max_drawdown = -0.2088`
    - `source_audit` 显示：
      - `budget_origin_sell_share = 0.9000`
      - `sell_intent_realized_rate = 1.0000`
      - `sell_intent_suppressed_share = 0.0000`
      - `high_cash_budget_origin_sell_share = 0.6667`
  - 推断：
    - 当前更深层的问题不是“模型卖出意图被压掉”，而是“预算/翻译层仍主动制造了大多数真实卖出”。
    - 高现金日当前仍偏被动形成，cash timing 还没有真正学成主动择时。
  - 决策：
    - 正式 verdict 仍保持 `cp_v3_deploy_executability_r10__budget_fix_protocol_r1` 的 `shadow_only`。
    - 后续凡是触及 simulator 卖出路径的修正，都应默认补跑 source attribution audit。

## 2026-04-23 r10 卖出来源解耦 v7c 复盘

- 动作前自检：
  - 事实：v6 source audit 已显示 `budget_origin_sell_share = 0.9000`、`high_cash_budget_origin_sell_share = 0.6667`，说明真实卖出主要由预算/翻译层制造。
  - 事实：用户要求前台推进、训练不要中断；本轮因此只做代码侧评估、审计与文档写回，未启动训练、未停止训练、未做 live/promotion 切换。
  - 推断：当前底层思路不是整体错误，而是实现层把“模型显式卖出 / 模型释放信号 / 组合部署资金来源”混成了一个预算副作用。
  - 假设：若把卖出来源显式解耦，并只允许弱证据旧仓为强 deploy 候选提供有限资金，就能同时避免 budget-origin 隐性卖出和 hard floor 冻结部署。
- 已完成实现：
  - 在 `portfolio_simulator.py` 中新增 `cash_constraint_sell_source_guard_v7` 及别名。
  - 增加模型释放信号、卖出授权、deploy funding rebalance、卖出来源底线守卫与对应诊断字段。
  - 在 `analyze_behavior_gap.py` 中补齐 deploy funding rebalance 的计数、占比、forward excess 与瓶颈判断。
  - 前台顺序完成 v6 对照、v7 hard guard、v7b 宽松 funding、v7c 克制 funding 的评估与行为审计。
- 关键证据：
  - v6 对照：
    - `annual_return = -0.1140`
    - `sharpe = -0.3284`
    - `budget_origin_sell_share = 0.9000`
    - `high_cash_budget_origin_sell_share = 0.6667`
  - 第一版 v7：
    - `budget_origin_sell_share = 0.0`
    - `deploy_intent_realized_rate = 0.2605`
    - `add_to_hold_conflict_share = 0.7727`
    - 结论：过度保护旧仓，部署被冻结。
  - v7b：
    - `annual_return = 0.7562`
    - `sharpe = 2.4132`
    - `deploy_funding_rebalance_sell_share = 0.8945`
    - `deploy_funding_rebalance_forward_excess_5d = 0.0058`
    - 结论：证明收益上限，但 funding sell 过多且卖出后仍平均跑赢，不宜作为语义最终落点。
  - v7c：
    - `annual_return = 0.2716`
    - `sharpe = 1.0760`
    - `max_drawdown = -0.1162`
    - `avg_order_translation_conflict_rate = 0.2376`
    - `deploy_intent_realized_rate = 0.7321`
    - `sell_selection_quality_5d = 0.0538`
    - `budget_origin_sell_share = 0.0`
    - `high_cash_budget_origin_sell_share = 0.0`
    - `deploy_funding_rebalance_sell_share = 0.7250`
    - `deploy_funding_rebalance_forward_excess_5d = -0.0179`
- 动作后复盘：
  - 事实：v7c 已把卖出责任从隐藏预算副作用改成显式、可审计、可限制的来源链。
  - 推断：当前不是“不该有组合资金腾挪”，而是资金腾挪必须显式标注为 `deploy_funding_rebalance`，并接受 forward excess 与占比约束。
  - 决策：当前代码采用 v7c，而不是 headline return 更高的 v7b；原因是 v7c 更符合来源合同和稳健推进纪律。
  - 边界：本轮仍是 shadow / research 证据，不构成 promotion；正式训练级结论仍需新的 protocol / bounded study 支撑。
- 遗留问题：
  - `deploy_funding_rebalance_sell_share = 0.7250` 仍偏高，说明模型 release/value head 还没有充分承担资金来源判断。
  - `sell_source_floor_guard_share = 0.7661` 仍偏高，说明大量持仓仍需要 simulator 底线保护。
  - `deploy_intent_candidate_budget_drop_share`、`budget_action_entanglement`、`order_translation_drift` 与 `cash_timing_quality_1d` 仍是下一阶段主瓶颈。

## 2026-04-23 r11 卖出来源契约训练侧接通

- 动作前自检：
  - 事实：v7c 已把 sell-source 从代码侧拆清，但当前正式训练主线仍主要沿 `result_value_v9 / alpha_result_value_budget_split_v9 / deploy_executability_v1` 选优。
  - 事实：若只继续在 simulator 层加规则，study 仍会按旧目标打分，后续极易重复“代码已修、训练没学、study 还在奖旧行为”的无效路径。
  - 推断：当前最高 ROI 不是马上再开长训练，而是先把卖出来源合同同步接到 `budget objective + loss profile + study objective`。
  - 假设：若训练侧合同真正接通，短窗 teacher rollout 至少会显示出更克制的 candidate / turnover / release 偏置变化，即使 headline return 不一定立刻最好。
- 已完成实现：
  - 在 `pipeline_utils.py` 中新增 `result_value_v10`，引入：
    - `protected_hold_pressure`
    - `funding_release_pressure`
    - `disciplined_funding_need`
    - `portfolio_release_pressure`
  - 在 `model_seq_v3.py` 中新增 `alpha_result_value_budget_split_v10` 与 `funding_release_discipline_loss`。
  - 在 `run_self_optimizing_study.py` 中新增：
    - `split_heads_sell_source_contract_r11`
    - `sell_source_contract_v1`
  - 新增可复用核验脚本：`daily_research/continuous_policy/check_budget_objective_contract.py`
- 已完成验证：
  - `r11 dry-run` 已跑通，`study_plan.json` 已生成。
  - 短窗 contract check 已生成：
    - `daily_research/output/continuous_policy/analysis/budget_objective_checks/sell_source_contract_v10_short_window_20251008_20251231.json`
  - 同窗 `v10 - v9` 关键变化：
    - `candidate_budget = -0.75`
    - `turnover_budget = -0.0269`
    - `reduce_bias_target = -0.0219`
    - `exit_patience_target = +0.0071`
    - `result_value_executable_deploy_pressure = +0.0525`
    - `result_value_portfolio_release_pressure = -0.0530`
    - `teacher total_return = 4.1852 -> 4.1273`
  - `sell_source_contract_v1` 已能对真实 `protocol_summary.json` 正常评分：
    - `composite_score = 4.566913`
    - `budget_origin_sell_penalty = 0`
    - `deploy_funding_sell_share_penalty = 0`
- 动作后复盘：
  - 事实：训练侧链路已经接通，当前不再只有 simulator 知道 sell-source contract。
  - 事实：`v10` 的直接效果是更克制地压低候选预算、换手和 reduce 偏置，而不是完全推翻 deploy。
  - 推断：先前看到的巨大 `annual_return` 差值主要受短窗年化放大影响；真实应优先看 `total_return` 与目标变化，不应误判为“思路整体错误”。
  - 决策：本轮不直接把 dry-run 或 short-window check 写成正式训练 verdict；下一轮若继续正式推进，应直接用 `r11` 做 bounded study，而不是回退到只靠 `v9` 的旧主线。

## 2026-04-23 默认执行池外持仓显式动作修复

- 动作前自检：
  - 事实：`daily_research/execution/current_positions.csv` 当前包含 `001202.SZ`、`1400` 股、成本价 `17.03`。
  - 事实：`001202.SZ` 存在于全市场识别缓存中，但不在 `2026-04-23` 当前默认执行价格宇宙与目标权重/分数面板中。
  - 事实：旧逻辑按 `latest_price.index` 过滤持仓，导致池外持仓从动作生成和持仓概览中被静默丢失，并被文本误报成“当前无持仓/无明确调仓动作”。
  - 推断：这不是股票代码无法识别，而是“已识别但不在当前默认执行自动估值宇宙”。
- 已完成实现：
  - 修改 `daily_research/baseline/generate_daily_trade_plan.py`：
    - 池外持仓不再被静默过滤。
    - `target_weight = 0` 时生成显式 `卖出`。
    - `target_weight > 0` 时生成显式 `保留`。
    - 为 summary 新增：
      - `priced_position_count`
      - `unpriced_position_count`
      - `unpriced_action_suggestions`
      - `blocked_buy_candidate_count`
      - `positions_source_mtime`
    - 文本输出新增：
      - 持仓识别统计
      - 池外持仓提醒
      - 一手/现金约束导致的 blocked buy 解释
- 关键证据：
  - 前台重生成默认执行计划后，`latest_trade_plan.txt` 已显示：
    - `1. 卖出 001202.SZ | 估算价格基准 待核对 | 原因: 池外持仓，默认执行建议退出`
  - `actions_today.csv` 已包含：
    - `001202.SZ, 卖出, 1400, price=NaN, est_value=NaN`
  - `holdings_snapshot.csv` 已显示：
    - `状态 = 池外持仓，默认执行建议退出`
- 动作后复盘：
  - 事实：默认执行现在可以对池外持仓给出明确卖出/保留建议，而不是只给模糊备注。
  - 事实：当前 `cash = 1071.57`、`lot_size = 100`，同时存在 `15` 个正目标由于一手约束未形成可执行买入，因此“无新增买入动作”也是独立事实，不应与池外持仓问题混淆。
  - 决策：默认执行后续要继续把“识别得到但无法自动估值”的持仓视为显式决策对象，而不是显示层例外。

## 2026-04-23 默认执行 21 日节奏回滚

- 动作前自检：
  - 事实：`short_expert_policy_v5b` 的 formal 主结论来自 `liquid500 + rolling_pool_rebalance_days = 21 + execution_first + regoff_k1_20d_ensemble_native_anchor`。
  - 事实：production 层后来被改成了 `rolling_pool_rebalance_days = 1` 与 `trading_day_interval = 10`，已经偏离 formal 主结论的制度节奏。
  - 推断：如果不回滚，后续收益变化会继续混入“股票池刷新制度变化”和“模型本体变化”，归因会失真。
- 已完成实现：
  - 修改 `daily_research/execution/update_default_candidate_production.py`
    - 默认滚动股票池重建周期改回 `21` 个交易日
    - 默认自动重训周期改回 `21` 个交易日
    - `warn_after_trading_days = 21`
    - `block_after_trading_days = 42`
    - `daily_pool_refresh_enabled = false`
    - `daily_pool_refresh_policy = rolling_liquidity_pool_every_21_trading_days`
  - 同步改写当前生效产物：
    - `daily_research/output/active_execution_strategy.json`
    - `daily_research/output/short_expert_policy_v5b_execalign_production_default/metrics.json`
    - `daily_research/output/short_expert_policy_v5b_execalign_production_default/production_retrain_manifest.json`
    - `daily_research/output/short_expert_policy_v5b_execalign_production_default/production_retrain_summary.md`
- 动作后复盘：
  - 事实：本轮没有启动、停止或中断任何训练。
  - 结论：默认执行当前重新对齐为 `21` 日股票池节奏 + `21` 日自动重训节奏，与 formal 主结论保持一致。

## 2026-04-23 r11 正式 bounded study 收口与 export 守护

- 动作前自检：
  - 事实：`r11` 的训练侧合同链路已经通过 dry-run 与 short-window contract check 打通，但这还不是正式 study 证据。
  - 事实：用户明确要求继续下一轮并一次性交付全部可完成内容，因此本轮不能停在建议层，而应直接完成正式 bounded study、必要复核、写回和验证。
  - 推断：最有效路径不是继续猜哪条 profile 可能更强，而是先把 `cp_v3_sell_source_contract_r11__study_r1` 跑完，再针对 confirmatory 候选空缺补做 runner-up 复核。
- 已完成实现：
  - 前台执行正式 bounded study：
    - `run_self_optimizing_study --search-profile split_heads_sell_source_contract_r11 --trial-count 4 --confirmatory-max-candidates 2 --study-tag cp_v3_sell_source_contract_r11__study_r1`
  - 在默认 confirmatory 之外，手动补做了同口径 runner-up confirm：
    - `cp_v3_sell_source_contract_r11__study_r1__confirm_03_runnerup_alla`
  - 本轮还修复了一个真实导出 bug：
    - `translate_target_weights_to_share_actions(...)` 为空时，原返回表缺少固定列，导致 export merge 因缺少 `stock` 列崩溃
    - 现已在 `pipeline_utils.py` 与 `export_action_panel.py` 中为零行结果保留稳定 schema
  - 过程中曾误跑过一次默认 `liquid500` 的 runner-up rerun；该 run 不作为正式证据，只作为触发 export bug 的上下文。
- 关键事实：
  - 正式 study 完成：
    - `completed_trial_count = 4`
    - `failed_trial_count = 0`
    - `confirmatory_completed_trial_count = 2`
    - `objective_profile = sell_source_contract_v1`
  - `confirm_01`（当前正式最佳）：
    - `loss_profile = alpha_result_value_budget_split_v10`
    - `budget_objective = result_value_v9`
    - `annual_return = 0.8743`
    - `sharpe = 2.3028`
    - `sell_selection_quality_5d = 0.0223`
    - `budget_origin_sell_share = 0.0`
    - `deploy_funding_rebalance_sell_share = 0.8974`
    - `promotion_status = shadow_only`
  - `confirm_02`（`v10/v10` fresh confirmatory）：
    - `annual_return = -0.3206`
    - `sharpe = -1.7629`
    - `sell_selection_quality_5d = -0.0816`
    - `deploy_funding_rebalance_sell_share = 0.9185`
    - `promotion_status = shadow_only`
  - `confirm_03_runnerup_alla`（补充 runner-up confirm）：
    - `annual_return = 0.2267`
    - `sharpe = 0.9590`
    - `cash_timing_quality_1d = 0.0001`
    - `release_gate_forward_alignment_5d = 0.1496`
    - `deploy_funding_rebalance_sell_share = 0.9449`
    - `deploy_funding_rebalance_forward_excess_5d = 0.0439`
    - 未超过 `confirm_01`
- 动作后复盘：
  - 事实：`v10 objective` 本轮没有通过正式 confirmatory 验收；当前可复用胜出结论是“保留 `result_value_v9`，吸收 `alpha_result_value_budget_split_v10`”。
  - 事实：所有正式分支的 `budget_origin_sell_share` 都已归零，但 funding-sell 占比仍普遍偏高，说明主瓶颈已经从隐藏 budget 卖出转移到显式 funding rebalance 的过度依赖。
  - 推断：当前思路不是底层原理错了，而是前一阶段只把 sell-source 修到了 simulator，尚未完全把 held-side release/funding 仲裁学稳。
  - 决策：当前 r11 家族继续推进时，应优先沿 `result_value_v9 + alpha_result_value_budget_split_v10 + cash_constraint_sell_source_guard_v7` 做后续对照，不回退到只修 simulator，也不直接扶正 `result_value_v10`。
  - 边界：本轮未切换 live 默认执行，未改写 promotion gate，未中断任何正式训练。
