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
- 当前 active 执行口径的物化真源为 `daily_research/output/active_execution_strategy.json`；`identity_layer.md` 不再承载可变 live 默认、最新分数或实验指标。
- 当前文档守卫已显式检查 active 执行口径与 `state_center.md` / `knowledge_center.md` 的一致性，并禁止 mutable live 默认回流到 `identity_layer.md`。
- 当前 `state_center.md`、`operations_center.md` 与 `continuous_policy_design_contract.md` 已完成历史归档压缩；长原文与标题索引统一下沉到 `daily_research/brain/references/`。
- 当前已新增 r12 shadow 研究闭环：`alpha_result_value_budget_split_v12`、`split_heads_release_translation_deploy_r12`、`release_translation_deploy_v1` 与 `release_translation_deploy_health_score`，用于联合审计 release learning、order translation drift 与 deploy executability。
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
- 表现冠军：`cp_v3_deploy_executability_r10__study_r1__confirm_01`
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
- 修补后评估：`cp_v3_deploy_executability_r10__study_r1__confirm_02__budget_fix_eval`
- 修补后审计：`cp_v3_deploy_executability_r10__study_r1__confirm_02__budget_fix_audit`
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
## 2026-04-23 r11 held-side release/funding 合同收紧与重审

- 动作前自检：
  - 事实：
    - `r11` 正式 bounded study 已经完成，当前最优 formal 分支仍是 `confirm_01 = result_value_v9 + alpha_result_value_budget_split_v10 + cash_constraint_sell_source_guard_v7`。
    - 旧的 `sell_source_contract_v1` 已经能压住 `budget_origin_sell_share`，但还不能直接说明 held-side release/funding 已学成。
    - `analyze_behavior_gap.py` 已经具备 sell-source 诊断框架，最有效的下一步不是再堆 simulator 规则，而是把 held-side 失败模式显式量化并接进 study 评分。
  - 推断：
    - 当前主问题更可能是“release/funding 证据没学出来”，而不是“还在大面积砍到最强保护旧仓”。
  - 假设：
    - 若把 `protected_hold / funding_release / release_signal` 三类指标同时接进 audit 和 study objective，就能把原先停留在口头分析里的 held-side 问题转成正式可比较证据。

- 已完成实现：
  - 在 `daily_research/continuous_policy/analyze_behavior_gap.py` 中补齐 held-side 指标：
    - `avg_protected_hold_support`
    - `avg_funding_release_support`
    - `model_release_signal_forward_excess_5d`
    - `model_release_against_protected_hold_share`
    - `model_release_release_consistent_share`
    - `deploy_funding_against_protected_hold_share`
    - `deploy_funding_release_consistent_share`
  - 增加新瓶颈：
    - `held_funding_release_arbitration_not_learned`
    - `release_signal_not_selective`
  - 在 `daily_research/continuous_policy/run_self_optimizing_study.py` 中新增：
    - `objective_profile = sell_source_contract_v2`
    - `search_profile = split_heads_sell_source_contract_r11b`
  - `sell_source_contract_v2` 继续惩罚：
    - 过高的 `deploy_funding_rebalance_sell_share`
    - 为正的 `deploy_funding_rebalance_forward_excess_5d`
    - 偏低的 `deploy_funding_release_consistent_share`
    - release 样本达到阈值后仍不干净的 `model_release_signal_forward_excess_5d / against_protected_hold`

- 关键执行与证据：
  - 用新审计字段串行重跑：
    - `cp_v3_sell_source_contract_r11__study_r1__confirm_01__reaudit_v2`
    - `cp_v3_sell_source_contract_r11__study_r1__confirm_02__reaudit_v2`
    - `cp_v3_sell_source_contract_r11__study_r1__confirm_03_runnerup_alla__reaudit_v2`
  - 关键事实：
    - `confirm_01`
      - `deploy_funding_rebalance_sell_share = 0.8974`
      - `deploy_funding_rebalance_forward_excess_5d = 0.0031`
      - `deploy_funding_against_protected_hold_share = 0.0143`
      - `deploy_funding_release_consistent_share = 0.0`
      - `model_release_signal_sell_count = 2`
      - `model_release_signal_keep_support = 0.3272`
      - `model_release_signal_release_support = 0.0`
    - `confirm_02`
      - `deploy_funding_rebalance_sell_share = 0.9185`
      - `deploy_funding_rebalance_forward_excess_5d = -0.0371`
      - `deploy_funding_against_protected_hold_share = 0.0242`
      - `deploy_funding_release_consistent_share = 0.0`
    - `confirm_03_runnerup_alla`
      - `deploy_funding_rebalance_sell_share = 0.9449`
      - `deploy_funding_rebalance_forward_excess_5d = 0.0439`
      - `deploy_funding_against_protected_hold_share = 0.0167`
      - `deploy_funding_release_consistent_share = 0.0`
  - 新对比产物：
    - `daily_research/output/continuous_policy/analysis/protocol_contract_comparisons/r11_sell_source_contract_v2_compare_20260423.json`
  - `sell_source_contract_v2` 下排序：
    - `confirm_01 = 0.752571`
    - `confirm_03_runnerup_alla = -2.039523`
    - `confirm_02 = -7.803599`
  - `r11b` dry-run 已完成：
    - `daily_research/output/continuous_policy/studies/cp_v3_sell_source_contract_r11b__dryrun_20260423/study_plan.json`

- 动作后复盘：
  - 事实：
    - 新指标证明当前 held-side 问题不主要表现为“经常卖到最强保护旧仓”，因为 `deploy_funding_against_protected_hold_share` 实际很低。
    - 真正更深的问题是：`deploy_funding_release_consistent_share` 约等于 `0`，而 `model_release_signal_release_support` 也基本为 `0`。
  - 推断：
    - 这不是“funding sell 只是卖错了一点点仓位”，而是“release/funding 价值仲裁几乎还没学出来”；系统现在更多是在用显式 funding 卖出顶替过去的隐式 budget 卖出。
    - 当前 `sell_source_contract_v2` 的价值在于把这个 held-side 学习缺口从口头结论变成正式 study 评分约束。
  - 自纠偏：
    - 中途曾并行触发过两次 `analyze_behavior_gap`，考虑到它会写 latest 摘要，这种做法有潜在竞态风险。
    - 已立即按同一 tag 顺序覆盖重跑三份审计，消除了 latest 指针歧义；单独审计 JSON 以串行结果为准。

- 当前结论：
  - 当前思路不是整体错了。
  - 真正错的是早期把 held-side release/funding 混成 budget 副作用；这部分现在已经被显式拆开，并继续接进正式研究合同。
  - 但 held-side 学习闭环还没有完成，`r11b` 之后真正该验证的是：能否在不牺牲正式 confirm 稳定性的前提下，让 `deploy_funding_release_consistent_share` 从接近 `0` 提升到可解释区间。

## 2026-04-23 r11b v11 正式 study 与 held-side 逐仓复盘

- 动作前自检：
  - `r11b dry-run` 已打通，本轮真正不确定点不再是“能不能开跑”，而是 `v11` 到底会带来真实 held-side 改善，还是只会把执行链拉塌。
  - 为避免重复“只看 aggregate share 就下结论”的旧错误，这轮先把 held-side detail export 接进审计，再跑正式 bounded study，并保留手动语义 confirm 作为对照。
  - 约束保持不变：前台跑，不中断既有训练，不回退到 simulator-only 解释。

- 实施：
  - 在 `daily_research/continuous_policy/analyze_behavior_gap.py` 中新增：
    - `--export-held-side-details`
    - `--held-side-detail-limit`
    - `disciplined_funding_need`
    - `held_side_support_gap`
    - `held_side_release_consistent`
    - `held_side_against_protected_hold`
  - 在 `daily_research/continuous_policy/model_seq_v3.py` 中新增：
    - `alpha_result_value_budget_split_v11`
    - `funding_release_discipline_loss(variant='v11')`
  - 在 `daily_research/continuous_policy/run_self_optimizing_study.py` 中把 `split_heads_sell_source_contract_r11b` 扩成 `v10 / v11` 与 `v9 / v10` 联合搜索。
  - 正式执行：
    - `cp_v3_sell_source_contract_r11b__study_r1`
    - 自动 confirm：`confirm_01`、`confirm_02`
    - 手动补 confirm：`cp_v3_sell_source_contract_r11b__study_r1__confirm_03_semantic_v11v9`
  - 顺序导出逐仓明细：
    - `confirm_01__details_v1`
    - `confirm_03_semantic_v11v9__details_v1`

- 动作后复盘：
  - 事实：
    - `confirm_01 = v11 + v10`：
      - `annual_return = 0.5505`
      - `sharpe = 1.5937`
      - `deploy_funding_rebalance_sell_share = 0.9699`
      - `deploy_funding_rebalance_forward_excess_5d = 0.0097`
      - `deploy_funding_release_consistent_share = 0.0`
      - `deploy_intent_realized_rate = 0.8708`
      - `promotion = shadow_only`
    - `confirm_02 = v10 + v10` 仍为负收益，说明 `result_value_v10` 旧问题没有被自动消除。
    - `confirm_03_semantic_v11v9`：
      - `annual_return = 0.4653`
      - `sharpe = 1.4377`
      - `deploy_funding_rebalance_sell_share = 0.6957`
      - `deploy_funding_rebalance_forward_excess_5d = -0.0121`
      - `deploy_funding_release_consistent_share = 0.0`
      - `deploy_intent_realized_rate = 0.2627`
      - `order_translation_conflict_rate = 0.3869`
      - `promotion = shadow_only`
    - held-side 明细显示：
      - 自动 confirm 有 `129` 次 funding trim，且全部来自 `deploy_funding_rebalance`
      - trim 主要集中在 `002371.SZ`、`001309.SZ`、`002049.SZ`
      - 语义线只有 `16` 次 funding trim，集中在 `002049.SZ`、`002157.SZ`
      - 两条线都仍存在 protected-hold conflict，且 `deploy_funding_release_consistent_share` 仍为 `0.0`
  - 推断：
    - `v11` 不是无效，它确实能把 funding-sell 往更少、更干净的方向推。
    - 但当前真正没打通的是“held-side release 学习”和“deploy/order translation 执行闭环”的兼容性；这两者现在仍然互相拉扯。
    - `result_value_v10` 即使叠加更强的 `v11 loss`，也还没有形成可 promotion 的预算目标。
  - 自纠偏：
    - `trial_01 = v11 + v9` 在 screening composite 中很差，若机械依赖自动 confirm，会错过一条重要的语义样本。
    - 本轮补做 `confirm_03_semantic_v11v9`，避免把“可以更干净但还不可执行”的半成品误记成不存在。
    - 这次没有再并行跑 `analyze_behavior_gap.py`；逐仓 detail audit 全部按顺序执行，避免 latest 摘要竞态。

- 当前结论：
  - 当前思路仍然不是整体错了。
  - `alpha_result_value_budget_split_v11` 已证明 held-side funding 语义还能继续往正确方向推进。
  - 但 promotion 级别的真正瓶颈已进一步收敛到：release learning、order translation drift 与 deploy executability 的三方耦合还没被同时解决。

## 2026-04-24 接管复核与守卫验证

- 动作前自检：
  - 事实：用户要求新 agent 在继承已有目标、规则、记忆、经验和计划的前提下继续推进，并要求区分事实、推断、假设。
  - 事实：当前项目已有主脑/分脑接管合同，默认应先读主脑，再读 `daily_research` 分脑，不得先盲扫 body。
  - 约束：本轮只是接管与复核，不启动训练、不改 live、不改 promotion、不并行运行会写 latest 摘要的 `analyze_behavior_gap.py`。
- 已完成动作：
  - 运行 `brain_bootstrap.py --child daily_research --json`，确认 `daily_research` 分脑状态为 `attached_to_main_brain`。
  - 用 UTF-8 方式重读主脑与 `daily_research` 的 `identity / state / knowledge / operations / governance` 关键入口，避免把 PowerShell 乱码误判为文件损坏。
  - 阅读当前未提交差异，确认代码改动集中在 held-side 逐仓审计、`alpha_result_value_budget_split_v11` 与 `split_heads_sell_source_contract_r11b` 搜索口径。
  - 运行守卫与轻量验证：`brain_integrity_check.py --json`、`doc_guard.py check`、三份修改脚本 `py_compile`、`git diff --check`，均通过。
- 动作后复盘：
  - 事实：当时复核的工作树为 `main...origin/main [ahead 33]`，未提交改动记录为 `9` 个既有文件；该轮复核未引入新的可见 Git 变更范围之外的产物。
  - 推断：brain 写回与代码差异目前一致；当前最新正式结论仍是 r11b/v11 证明方向可继续推进，但所有 confirm 分支仍为 `shadow_only`。
  - 决策：下一步若继续研究，应把 `release learning / order translation drift / deploy executability` 作为联合问题处理；不得把 `deploy_funding_release_consistent_share = 0.0` 的分支解释为 release head 已学成。

## 2026-04-24 主分脑维护与 active 执行口径纠偏

- 动作前自检：
  - 事实：用户要求系统审阅、整理、维护、简练并修复主分脑；本轮目标是文档职责、事实口径和守卫一致性，不是重开实验。
  - 事实：`daily_research/output/active_execution_strategy.json` 中的 active label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`，production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
  - 事实：`identity_layer.md` 仍残留旧的可变 live 默认口径，属于身份层职责漂移。
  - 假设：把可变执行事实统一路由到 `state_center.md`、`knowledge_center.md` 与 active artifact，可以降低后续接管误读概率。
- 已完成实现：
  - `identity_layer.md` 改为只保留目标、边界、禁区和事实入口，不再记录 mutable live 默认、最新分数或实验指标。
  - `state_center.md` 与 `knowledge_center.md` 的 active 执行口径统一为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
  - `doc_guard.py` 新增 identity-layer 禁止模式，并新增 active artifact 与 brain 状态/知识中心的对齐检查。
  - 主脑 `brain/state_center.md` 只记录维护摘要，具体 production 与 continuous_policy 状态继续由 `daily_research` 分脑承载。
- 动作后复盘：
  - 事实：本轮未启动训练，未运行会写 `latest_behavior_audit_summary.json` 的新行为审计，未切换 live 默认执行，未改写 promotion gate。
  - 事实：`doc_guard.py check` 与 `brain_integrity_check.py --json` 已在修正后通过；最终验证需继续包含 `project_consistency_check.py`、`py_compile` 与 `git diff --check`。
  - 推断：原问题不是执行策略本身切换，而是 mutable live 事实被放进了身份层，导致状态中枢、知识中枢和 artifact 之间存在误读风险。
  - 决策：后续若身份层再次出现具体 live 默认、实验指标或最新 winner，应先视作文档职责漂移修复，再继续任何重动作。

## 2026-04-24 state / operations / design contract 历史归档压缩

- 动作前自检：
  - 事实：用户要求继续做历史归档压缩；上轮遗留风险是 `state_center.md` 与 `operations_center.md` 仍承载大量历史日期段。
  - 事实：`continuous_policy_design_contract.md` 也已膨胀为 r1-r11b 的合同演化长文，适合同步归档。
  - 约束：本轮只压缩 brain 文档，不启动训练、不运行行为审计、不切换 live、不改写 promotion。
  - 假设：先原样归档 raw history，再生成标题索引，最后重写当前入口，可以降低丢证据与误删关键命令的风险。
- 已完成实现：
  - 原样归档 `state_center.md` 到 `daily_research/brain/references/state_center_history_raw_20260424.md`，并生成 `state_center_evidence_index_20260424.md`。
  - 原样归档 `operations_center.md` 到 `daily_research/brain/references/operations_center_history_raw_20260424.md`，并生成 `operations_center_evidence_index_20260424.md`。
  - 原样归档 `continuous_policy_design_contract.md` 到 `daily_research/brain/references/continuous_policy_design_contract_history_raw_20260424.md`，并生成 `continuous_policy_design_contract_evidence_index_20260424.md`。
  - 三个入口文件已改写为当前结论、当前纪律、当前合同与历史归档入口。
  - `doc_guard.py` 已新增这三个入口文件的行数上限与归档引用守卫。
- 动作后复盘：
  - 事实：归档前原文均已保全，并记录原始行数与 SHA256；入口文档不再承载长历史日志。
  - 推断：后续接管成本会下降，且需要旧证据时仍可通过索引精确回到原文标题。
  - 决策：后续若这三个入口再次膨胀，应优先追加到 `episodic_memory.md` 或 `brain/references/`，不要把历史过程重新堆回当前入口。

## 2026-04-24 r12 release / translation / deploy 联合闭环落地

- 动作前自检：
  - 事实：r10 已证明 deploy intent executability 可测；r11/r11b 已证明 sell-source 与 held-side funding 可进入训练侧和审计侧。
  - 事实：r11b 仍未打通 `deploy_funding_release_consistent_share`，且 deploy/order translation 仍互相拉扯。
  - 推断：当前根因不是主线方向错误，而是 release learning、order translation drift 与 deploy executability 三者没有被同一个目标函数同时约束。
  - 约束：本轮只落地代码、审计、守卫和文档，不启动训练、不切换 live、不改 promotion gate、不运行会改写 latest 行为摘要的新审计。
- 已完成实现：
  - `model_seq_v3.py` 新增 `alpha_result_value_budget_split_v12`，复用 v11 funding-release discipline 变体并提高 release / deploy / protected-hold 相关权重。
  - `run_self_optimizing_study.py` 新增 `split_heads_release_translation_deploy_r12` 与默认 objective `release_translation_deploy_v1`。
  - `analyze_behavior_gap.py` 新增 `release_translation_deploy_health_score`、组件分解、failure mode 与低分 bottleneck，同时修复两处历史编码残留诊断句。
  - `project_consistency_check.py` 与 `doc_guard.py` 已补入 r12 合同守卫，防止 profile / objective / audit / brain 文档再次漂移。
- 验证：
  - `py_compile` 通过：`model_seq_v3.py`、`run_self_optimizing_study.py`、`analyze_behavior_gap.py`、`project_consistency_check.py`、`doc_guard.py`。
  - `doc_guard.py check` 通过。
  - `brain_integrity_check.py --json` 通过。
  - `project_consistency_check.py` 通过。
  - `git diff --check` 通过。
  - `run_self_optimizing_study --help` 已暴露 `split_heads_release_translation_deploy_r12`。
  - r12 dry-run `verify_release_translation_deploy_r12_dryrun_20260424` 已生成 study plan，默认 `objective_profile = release_translation_deploy_v1`，baseline 使用 `alpha_result_value_budget_split_v12 + result_value_v9 + cash_constraint_sell_source_guard_v7`。
- 动作后复盘：
  - 事实：本轮没有训练产物、没有新的正式 verdict、没有 active execution artifact 变更。
  - 推断：r12 把“该不该为新部署释放旧仓、释放谁、订单翻译是否保留意图”变成了同一张审计表和同一个 self-opt scoring 目标。
  - 决策：后续若继续投入正式算力，应先跑 r12 shadow study；任何结果必须同时看 `release_translation_deploy_health_score` 与组件分解，不能只看 annual return、deploy realized 或 budget-origin sell。

## 2026-04-24 r12 bounded shadow study 完整执行

- 动作前自检：
  - 事实：r12 dry-run、scoring smoke、守卫均已通过，profile 可执行。
  - 事实：用户要求直接执行完整方案；本轮允许启动 r12 研究训练，但仍不得切 live、不得改 active artifact、不得改 promotion gate。
  - 假设：最有效验证不是继续加代码，而是跑满 r12 的 4 个 screening 组合与 2 个 confirmatory 复核。
- 已完成执行：
  - 前台执行 `cp_v3_release_translation_deploy_r12__study_r1`。
  - 搜索空间：`alpha_result_value_budget_split_v11/v12` × `result_value_v9/v10`，固定 `cash_constraint_sell_source_guard_v7`。
  - 完成 `4` 个 screening、`2` 个 confirmatory，失败数为 `0`；latest state 在 study 结束后按 runner 机制恢复，随后为了逐仓根因补做 detail audit，并把 latest 行为摘要刷新到 `confirm_02__details_v1`。
  - 顺序导出 held-side detail：`confirm_02__details_v1`、`confirm_01__details_v1`、`trial_04__details_v1`；未并行运行审计。
- 关键结果：
  - 最终 champion：`cp_v3_release_translation_deploy_r12__study_r1__confirm_02`
    - `loss_profile = alpha_result_value_budget_split_v12`
    - `budget_objective = result_value_v9`
    - `annual_return = 1.7484`
    - `sharpe = 3.0857`
    - `max_drawdown = -0.1756`
    - `composite_score = 2.4736`
    - `promotion_status = shadow_only`
    - `release_translation_deploy_health_score = 0.3235`
    - `release_translation_deploy_failure_mode = order_translation_drift`
    - `deploy_intent_realized_rate = 0.6703`
    - `deploy_funding_release_consistent_share = 0.0`
    - `order_translation_conflict_rate = 0.3862`
    - `reduce_success_rate_5d = 0.0`
    - `exit_timeliness_rate_5d = 0.25`
- 筛选冠军：`trial_04 = alpha_result_value_budget_split_v11 + result_value_v10`
    - `composite_score = 3.0082`
    - `annual_return = 0.9111`
    - `deploy_intent_realized_rate = 0.9130`
    - `release_translation_deploy_health_score = 0.4352`
    - `deploy_funding_release_consistent_share = 0.0`
    - `failure_mode = order_translation_drift`
- 确认项 confirm_01：`alpha_result_value_budget_split_v11 + result_value_v10`
    - `annual_return = 0.5505`
    - `release_translation_deploy_health_score = 0.4093`
    - `deploy_intent_realized_rate = 0.8708`
    - `deploy_funding_release_consistent_share = 0.0`
    - `failure_mode = order_translation_drift`
- held-side detail 复盘：
  - `confirm_02`：`119` 条 funding sell，全部来自 `deploy_funding_rebalance`；`105` 条 ambiguous，`14` 条 protected-hold conflict；主要集中在 `002371.SZ`、`002256.SZ`、`002157.SZ`。
  - `confirm_01`：`129` 条 funding sell，全部来自 `deploy_funding_rebalance`；`126` 条 ambiguous，`3` 条 protected-hold conflict；主要集中在 `002371.SZ`、`001309.SZ`、`002049.SZ`。
  - `trial_04`：`101` 条 held-side sell，其中 `98` 条来自 `deploy_funding_rebalance`、`3` 条来自 `model_release_signal`；但 release consistency 仍为 `0.0`。
- 动作后复盘：
  - 事实：r12 能提高收益与 Sharpe，但没有闭合三方语义；所有 confirm 仍为 `shadow_only`。
  - 事实：`v12 + result_value_v9` 在 confirm 中比 `v11 + result_value_v10` 更强，但它通过更激进收益换来更高 drawdown 和更重 order translation drift。
  - 推断：当前不是单纯 release loss 不够，而是预算/订单翻译层仍在把 deploy 与 funding 的压力互相转嫁；`weight_change_action` 仍不能稳定保留 `execution_action` 语义。
  - 决策：下一轮最高价值不是 v13 继续加 release 权重，而是面向 translation / simulator / budget-action 解耦，降低 add -> hold、deploy candidate budget drop 与 held-side funding trim 的相互污染。

## 2026-04-24 r13 动作价值统一入口落地

- 行动前自检：
  - 事实：r12 最终 failure mode 是 `order_translation_drift`，并且用户明确指出选股、建仓、加仓、减仓、清仓之间可能各学各的。
  - 推断：继续单独加重 release loss 不能直接解决动作互斥；需要把所有动作放在同一个多周期未来价值坐标上。
  - 边界：本轮只落地 research / shadow 能力，不切换 live，不改 promotion gate，不把 r13 当成已验证 verdict。
- 已完成执行：
  - `label_builder.py` 新增 `multi_horizon_forward_value / risk / path_value`，基于 `1/3/5/10/20d` 未来路径生成统一价值锚。
  - `label_builder.py` 新增 `open_action_value / add_action_value / hold_action_value / reduce_action_value / exit_action_value / action_value_consistency_target`。
  - `model_seq_v3.py` 新增 `alpha_result_value_budget_split_v13`、动作价值 heads、`_action_value_consistency_loss` 与老 artifact 兼容加载标记。
  - `pipeline_utils.py`、`portfolio_simulator.py`、`analyze_behavior_gap.py` 贯通动作价值字段，并新增 `action_value_consistency_score`、`action_value_conflict_share`、`sell_against_keep_value_share`、`keep_against_release_value_share` 等审计指标。
  - `run_self_optimizing_study.py` 新增 `split_heads_action_value_unification_r13` 与 `action_value_unification_v1`。
  - `doc_guard.py` 与 `project_consistency_check.py` 已加入 r13 合同守卫。
- 动作后复盘：
  - 本轮完成的是 r13 可运行入口，不是正式研究结论。
  - 后续正式 study 必须同时看收益、Sharpe、drawdown、`release_translation_deploy_health_score` 与 `action_value_consistency_score`。
  - 如果 r13 仍失败，优先判断是动作价值标签本身不清、订单翻译层仍改写动作，还是预算层继续把个股动作头吞掉。

## 2026-04-24 r13 action-value bounded study 完整执行

- 行动前自检：
  - 事实：r13 入口、dry-run、标签/矩阵烟测与静态守卫已通过；用户要求直接完成可执行部分。
  - 事实：第一次正式 study 暴露出 `pipeline_utils.py` 中 `open_low_value_mask` 缺少括号，导致 pandas 把 `"open" & Series` 解析为 `rand_` 布尔错误。
  - 决策：先修复指标路径并用真实 rollout 复现通过，再用同一 study tag 做 strict resume，避免留下失败 study 作为最新事实。
- 已完成执行：
  - 修复 `open_low_value_mask = (model_action_lookup == "open") & (...)`。
  - 用 r13 trial_01 artifact 跑真实 rollout metrics smoke，确认 `action_value_consistency_score` 等连续性指标可产出。
  - 执行 `cp_v3_action_value_unification_r13__study_r1`：4 个 screening 均完成，2 个 confirmatory 均完成；screening 从 40 epoch strict resume 到 56 epoch，confirmatory 使用 64 epoch 预算。
- 关键结果：
  - 最终 champion：`cp_v3_action_value_unification_r13__study_r1__confirm_01 = alpha_result_value_budget_split_v13 + result_value_v10`。
  - `composite_score = 6.1955`，`annual_return = 0.4522`，`sharpe = 1.6735`，`max_drawdown = -0.0827`，`promotion_status = shadow_only`。
  - 动作统一指标：`action_value_consistency_score = 0.92`，`action_value_conflict_share = 0.0`，`sell_against_keep_value_share = 0.0`，但 `open_low_action_value_share = 1.0`。
  - 执行语义指标：`deploy_intent_realized_rate = 0.8684`，`order_translation_conflict_rate = 0.2378`，`release_translation_deploy_health_score = 0.4701`，`failure_mode = order_translation_drift`。
  - confirm_02：`alpha_result_value_budget_split_v13 + result_value_v9` 收益更高但 drawdown 更差，`annual_return = 0.7877`，`max_drawdown = -0.2213`，`failure_mode = release_not_learned_despite_deploy`。
- 动作后复盘：
  - 事实：r13 方向不是无效；它显著压低了显性 action-value 冲突。
  - 事实：r13 仍未达到 promotion，主要被 drawdown、卖出时机、cash timing、相对 active Sharpe 和 order translation drift 拦住。
  - 推断：用户提出的“统一学习未来多日涨跌/价值”是正确方向，但仅靠统一 action value 不足以解决 open 低价值入场和订单/预算翻译层改写动作的问题。
  - 决策：r13 保持 `shadow_only`；下一轮若继续，优先处理 `open_low_action_value_share`、`order_translation_drift` 与 release/deploy 翻译层错配，不要把高 `action_value_consistency_score` 误写成 live 证据。

## 2026-04-24 月度收益评价接入

- 行动前自检：
  - 事实：用户指出“评价收益以月为单位更好，更能体现模型的优势和问题”。
  - 推断：日级收益、年化收益和 5 日动作质量各自有用，但不足以暴露收益是否按月持续、是否集中在少数短窗口、是否存在最差月/连续亏损月风险。
  - 决策：把月度评价放进通用曲线指标和 study ranking，而不是单独做一次性报告。
- 已完成执行：
  - `pipeline_utils.py` 新增 `build_monthly_return_frame`，从 daily returns 聚合 `monthly_returns.csv` 明细。
  - `compute_curve_metrics` 新增 `monthly_return_mean`、`monthly_return_median`、`monthly_return_std`、`monthly_sharpe`、`monthly_win_rate`、`monthly_best_return`、`monthly_worst_return`、`monthly_max_consecutive_loss_months`、`monthly_intramonth_max_drawdown`、`monthly_consistency_score`。
  - `evaluate_policy.py` 与 `run_continuous_policy_protocol.py` 分别导出 `monthly_returns.csv` 与 `shadow_monthly_returns.csv`。
  - `run_self_optimizing_study.py` 已把月度指标纳入 primary metrics、trial ranking 和所有 objective 的轻量 scoring bonus/penalty。
- 动作后复盘：
  - 月度指标不替代动作语义审计；它回答“收益质量是否稳定”，动作审计回答“为什么会这样”。
  - 后续比较 r12/r13 或新分支时，应同时看年化/Sharpe/drawdown、月度持续性和 action/release/translation 指标。

## 2026-04-24 r14 直接日级动作价值仲裁入口落地

- 行动前自检：
  - 事实：r13 已把动作价值放到统一多周期坐标中，但推理端仍存在较长规则桥接，`order_translation_drift` 仍是关键 failure mode。
  - 事实：用户明确要求构建“以日为单位进行连续决策”的交易执行模型，不依赖固定调仓频率、固定持有周期或人工执行桥接规则。
  - 推断：下一步最高价值不是继续增加单个动作 loss，而是让统一动作价值成为日级动作仲裁的直接共同货币，同时保留最小合法性与风险边界。
  - 边界：本轮只落地 research / shadow 能力，不切换 live，不改 active artifact，不改 promotion gate，不把 r14 写成已验证结论。
- 已完成执行：
  - `model_seq_v3.py` 新增 `alpha_result_value_budget_split_v14` 与 `direct_action_value_v1`，在 v14 或带有对应 train summary 的 artifact 上启用直接动作价值仲裁。
  - `predict_policy_v3` 新增 `skip/open/hold/add/reduce/exit` 六类直接动作 utility；持仓态只允许 `hold/add/reduce/exit`，空仓态只允许 `skip/open`，并只保留最小合法性纠偏。
  - `portfolio_simulator.py`、`pipeline_utils.py` 与 `analyze_behavior_gap.py` 贯通 `direct_action_value_*` 字段，新增 direct mode 覆盖、标签匹配、边际、低边际与订单翻译冲突指标。
  - `run_self_optimizing_study.py` 新增 `split_heads_direct_action_value_r14` 与 `direct_daily_policy_v1`，把月度收益质量、直接动作边际、动作价值一致性和订单翻译漂移放入同一个 scoring 目标。
  - `doc_guard.py` 与 `project_consistency_check.py` 已补入 r14 合同守卫，主脑与分脑状态中心同步记录 r14 仅为研究入口。
- 验证：
  - `py_compile` 通过：`model_seq_v3.py`、`portfolio_simulator.py`、`pipeline_utils.py`、`analyze_behavior_gap.py`、`run_self_optimizing_study.py`、`project_consistency_check.py`、`doc_guard.py`。
  - r14 dry-run `verify_direct_action_value_r14_dryrun_20260424` 已生成 `split_heads_direct_action_value_r14` 研究计划，默认 objective 为 `direct_daily_policy_v1`，baseline 为 `alpha_result_value_budget_split_v14 + result_value_v10`。
  - 伪 artifact 推理烟测已进入 `direct_action_value_v1`，并产出 `direct_action_value_label`、`direct_action_value_applied`、`direct_action_value_gap` 与各动作 utility 列。
  - `doc_guard.py check`、`brain_integrity_check.py --json`、`project_consistency_check.py` 与 `git diff --check` 均通过。
- 动作后复盘：
  - 事实：r14 完成的是直接日级仲裁链路，不是正式 study verdict；后续仍需 bounded shadow study 才能判断是否优于 r13。
  - 推断：若 r14 改善 `direct_action_value_mode_share` 和月度一致性但仍有高 `direct_action_order_translation_conflict_rate`，根因将继续指向订单/预算层改写，而不是动作价值学习本身。
  - 决策：后续评估 r14 必须同时看 `monthly_consistency_score`、`monthly_worst_return`、`direct_action_value_gap_mean`、`open_low_action_value_share` 与 `order_translation_drift`，不能只看年化收益。

## 2026-04-24 r14 direct-action bounded study 与 repaired confirm

- 行动前自检：
  - 事实：r14 入口、dry-run、伪 artifact 推理烟测和静态守卫已通过，但尚缺正式 bounded shadow 证据。
  - 事实：用户目标是以日为单位进行连续动态决策，减少固定调仓、固定持有周期和人工桥接规则依赖。
  - 边界：本轮允许启动 r14 shadow study，但不得切换 live、不得改 active artifact、不得改 promotion gate。
  - 假设：最有效验证是一次跑清 r14 的 4 个 screening，并用 confirm / repaired confirm 对照 r13，而不是继续只做架构讨论。
- 已完成执行：
  - 执行 `cp_v3_direct_action_value_r14__study_r1`：4 个 screening 完成、0 个 screening 失败。
  - 自动 confirm 训练产物生成后在 protocol 写出阶段触发 `[Errno 22] Invalid argument`，未形成完整 `protocol_summary.json`；随后用原 study 口径和 `--max-universe-size 1200` strict resume 修复 `confirm_01`。
  - `confirm_02` 的 64 epoch checkpoint 已完成，因此用 65 epoch strict resume 补齐对照；该结果只作为严格续跑 sanity check，不等同于原 64 epoch 自动 confirm 行。
  - 曾误启动一条缺少 `--max-universe-size 1200` 的 manual confirm；因 universe / sample 口径不一致，未纳入正式证据。
  - 生成 repaired confirm 对比：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_value_r14__study_r1/manual_confirm_repair_summary.json` 与 `manual_confirm_repair_comparison.csv`。
  - 顺序导出 `confirm_01` held-side detail，未并行运行会写 latest 行为摘要的审计。
- 关键结果：
- 筛选冠军：`trial_02 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 0.3631`、`sharpe = 1.1531`、`max_drawdown = -0.1201`、`monthly_return_mean = 0.0234`、`monthly_win_rate = 0.75`、`direct_action_value_mode_share = 1.0`、`direct_action_order_translation_conflict_rate = 0.2842`，`promotion_status = shadow_only`。
- 修复后冠军：`confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 1.0904`、`sharpe = 2.8082`、`max_drawdown = -0.1326`、`monthly_return_mean = 0.0587`、`monthly_win_rate = 0.75`、`monthly_worst_return = -0.0797`、`monthly_consistency_score = 0.7722`、`direct_action_value_mode_share = 1.0`、`direct_action_value_gap_mean = 0.0351`。
  - r14 的主要失败项仍是 `reduce_success_rate_5d = 0.0`、`exit_timeliness_rate_5d = 0.25`、`cash_timing_quality_1d`、`max_drawdown` 与 `failure_mode = order_translation_drift`；`promotion_status` 仍为 `shadow_only`。
  - held-side detail：135 条 held-side sell 中 134 条来自 `deploy_funding_rebalance`，1 条来自 `model_release_signal`；104 条 ambiguous、18 条 protected-hold conflict、13 条 release-consistent，集中在 `002371.SZ / 000333.SZ / 002028.SZ`。
- 动作后复盘：
  - 事实：r14 直接动作值仲裁显著改善了 repaired confirm 的收益、Sharpe 与月度收益质量，也证明用户提出的“按未来收益风险直接学习日级动作”方向有实证价值。
  - 事实：direct action 已经进入推理主路径，但低边际占比仍高，且订单/预算翻译冲突与 held-side funding rebalance 过度依赖没有闭合。
  - 推断：当前主瓶颈已从“是否应该直接学习日级动作”转为“如何让订单、预算、release/funding 层保留 direct action intent 并承担真实退出责任”。
  - 决策：r14 保持 `shadow_only`；下一步应推进 r15 direct-action-preserving translation / release-funding repair，而不是把 r14 高收益 confirm 直接升级为 production 证据。

## 2026-04-24 r15 direct-action-preserving translation / release-funding repair 落地

- 行动前自检：
  - 事实：r14 repaired confirm 证明直接动作值仲裁有收益价值，但 `direct_action_order_translation_conflict_rate = 0.2867`、`direct_action_value_low_margin_share = 0.7372`，且 held-side sell 主要来自 `deploy_funding_rebalance`。
  - 推断：当前最高价值不是继续单点加 release loss，而是让订单/预算层尽量保留 direct action intent，并把 funding sell 绑定到可审计 release 授权。
  - 假设：先用 r14 champion artifact 做短窗 v8 smoke，可以验证新执行层和审计字段是否真实贯通；正式优劣仍需后续 bounded study。
  - 边界：本轮不切换 live，不改 active artifact，不改 promotion gate，不把 smoke 写成正式 verdict。
- 已完成执行：
  - `portfolio_simulator.py` 新增 `cash_constraint_direct_action_guard_v8`，引入 direct action release authorization / protected hold 证据，并把 funding sell 的模型释放责任显式化。
  - `model_seq_v3.py` 新增 `alpha_result_value_budget_split_v15`，加强 direct action margin、release/funding 责任、deploy/cash timing 与多周期价值训练权重。
  - `pipeline_utils.py` 与 `analyze_behavior_gap.py` 新增 `direct_action_intent_preserved_share`、`direct_action_funding_authorized_sell_share`、`direct_action_funding_protected_sell_share`、`direct_action_release_advantage_mean` 等审计字段。
  - `run_self_optimizing_study.py` 新增 `split_heads_direct_action_translation_r15` 与 `direct_action_translation_v1`，把月度收益、直接动作保真、funding 授权和低边际惩罚纳入 study scoring。
  - `doc_guard.py`、`project_consistency_check.py`、主脑状态中枢与分脑状态/操作/设计合同已写回 r15 research 入口。
- 验证与证据：
  - dry-run：`verify_direct_action_translation_r15_dryrun_20260424` 生成 4 条计划，baseline 为 `alpha_result_value_budget_split_v15 + result_value_v9 + cash_constraint_direct_action_guard_v8`。
- smoke 评估：`daily_research/output/continuous_policy/evaluations/verify_direct_action_translation_r15_v8_eval_smoke_20260424/evaluation_summary.json`。
- smoke 审计：`daily_research/output/continuous_policy/analysis/behavior_audits/verify_direct_action_translation_r15_v8_audit_smoke_20260424.json`。
  - 短窗事实：`direct_action_intent_preserved_share = 0.9191`、`direct_action_order_translation_conflict_rate = 0.0515`、`direct_action_value_low_margin_share = 0.8676`、`deploy_intent_realized_rate = 0.7083`，短窗未产生 deploy funding sell。
- 动作后复盘：
  - 事实：r15 代码链路、指标链路、dry-run 与短窗 smoke 均已贯通；它修的是“动作意图如何穿透执行翻译层”，不是新的 live 策略。
  - 事实：低边际 direct action 仍偏高，add -> hold / hold -> add 的小额翻译漂移仍在；审计继续把 `direct_action_intent_not_preserved` 与 `deploy_intent_not_executable` 判为高优先级。
  - 推断：r15 已把根因从“预算层暗改动作”推进到“模型动作边际不够锋利、deploy 预算落地仍受约束”的更窄问题。
  - 决策：r15 保持 research / smoke 状态；下一步若继续，应启动正式 bounded shadow study，并以 `direct_action_translation_v1` 对比 v14/v15 与 result_value_v9/v10。

## 2026-04-25 r15 formal 失败复盘与 r16 direct-action reallocation repair

- 行动前自检：
  - 事实：r15 正式 bounded study 已完成，champion 为 `confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9 + cash_constraint_direct_action_guard_v8`，但 `promotion_status = shadow_only`。
  - 事实：r15 formal 的主要失败不是收益完全失效，而是 `deploy_intent_realized_rate = 0.1642`、`add_to_hold_conflict_share = 0.8524`、`release_translation_deploy_health_score = 0`，failure mode 为 `deploy_not_realized`。
  - 事实：budget clipped 日的翻译冲突远高于 unclipped 日，说明 direct add/open 的失败主要卡在预算/订单翻译，而不是动作值路径完全缺失。
  - 推断：下一轮最高价值不是继续泛化加 loss，而是给高置信 direct add/open 建立显式可成交预算再分配，并让释放资金的持仓必须有可审计来源。
  - 假设：在不切换 live、不改 promotion gate 的边界内，可以先用 r15 champion artifact 评估 v9 reallocation guard，验证执行层结构修复是否有效。
- 已完成执行：
  - `portfolio_simulator.py` 新增 `cash_constraint_direct_action_reallocation_guard_v9`，为 direct add/open 授权、deploy advantage、reallocation source 与 retention floor 建立显式链路。
  - `pipeline_utils.py`、`analyze_behavior_gap.py` 新增 `direct_action_deploy_authorized_realized_rate`、`direct_action_add_authorized_realized_rate`、`direct_action_reallocation_source_count` 等 r16 审计字段。
  - `run_self_optimizing_study.py` 新增 `split_heads_direct_action_reallocation_r16` 与 `direct_action_reallocation_v1`，把授权成交率、再分配 source、月度收益质量和 direct action 语义一起纳入 scoring。
  - `project_consistency_check.py` 与 `doc_guard.py` 已加入 r16 合同守卫；状态、操作、设计合同同步写回 r16 事实边界。
- 验证与证据：
  - `py_compile` 通过本轮修改的 continuous_policy 与工具文件。
  - r16 dry-run `verify_direct_action_reallocation_r16_dryrun_20260425` 通过，4 条 trial 均使用 v9 reallocation guard。
- r16 第二次 smoke `verify_direct_action_reallocation_r16_v9_eval_smoke2_20260425`：`annual_return = 0.5807`、`sharpe = 1.6261`、`max_drawdown = -0.1180`、`monthly_consistency_score = 0.7247`、`direct_action_reallocation_source_count = 105`。
  - r16 smoke2 同时暴露：`direct_action_deploy_authorized_realized_rate = 0.1557`、`direct_action_add_authorized_realized_rate = 0.1439`、`add_to_hold_conflict_share = 0.8483`，说明核心 deploy/add 落地仍未完成。
  - r16 smoke3 曾尝试强制压低 source cap 并收窄授权阈值，结果显著恶化收益、回撤和成交率；该路径已回退，不作为当前实现。
- 动作后复盘：
  - 事实：r16 修复了“reallocation source 选不出来 / sell-source floor 锁死当前仓位”的结构问题，并在同一模型窗口显著改善 return/risk。
  - 事实：r16 没有解决 direct add/open 高授权低成交的本质问题，订单/预算层仍会把大量 add 翻译成 hold。
  - 推断：下一轮若继续，应优先研究 budget-clipped 日的可成交分配机制和 source/target 同步约束，而不是只继续提升动作分类置信度。
  - 决策：r16 保持 `shadow_only` / research baseline；不切换 live，不改 active artifact，不改 promotion gate。

## 2026-04-25 r17 direct-action pair reallocation / core target 执行修复

- 行动前自检：
  - 事实：r16 已恢复非零 reallocation source，但在高 add/open 压力日仍有大量授权 deploy/add 被翻译成 hold，`direct_action_deploy_authorized_realized_rate = 0.1557`、`add_to_hold_conflict_share = 0.8483`。
  - 事实：逐日分析显示，高 add 日常常是所有持仓都想 add，r16 只从 hold/skip 或弱 release source 找资金，因此满仓状态下没有足够可释放预算。
  - 推断：本质问题不是再加一个 add loss，而是缺少“在同一日把目标和资金来源配对排序”的组合预算机制。
  - 假设：在不切换 live、不改 promotion gate 的边界内，可以先用 r15 champion artifact 验证 v10 pair reallocation guard 是否能修复可成交性。
- 已完成执行：
  - `portfolio_simulator.py` 新增 `cash_constraint_direct_action_pair_reallocation_guard_v10`：把 direct deploy signal 收敛为 core deploy target，并允许弱排序 add 持仓作为 `direct_action_pair_reallocation_source`。
  - `pipeline_utils.py` 与 `analyze_behavior_gap.py` 新增 `direct_action_deploy_signal_count`、`direct_action_core_deploy_target_count`、`direct_action_core_deploy_target_realized_rate`、`direct_action_pair_reallocation_source_count`、`direct_action_pair_reallocation_sell_share` 等审计字段。
  - `run_self_optimizing_study.py` 新增 `split_heads_direct_action_pair_reallocation_r17` 与 `direct_action_pair_reallocation_v1`，把 core target 成交率、pair-source 数量、月度收益质量和 direct action 保真纳入 scoring。
  - `project_consistency_check.py` 与 `doc_guard.py` 已加入 r17 合同守卫；状态、操作、设计合同同步写回 r17 research 边界。
- 验证与证据：
  - `py_compile` 通过本轮修改的 continuous_policy 与工具文件。
  - r17 dry-run `verify_direct_action_pair_reallocation_r17_dryrun_20260425` 通过，4 条 trial 均使用 v10 pair reallocation guard。
- r17 第三次 smoke `verify_direct_action_pair_reallocation_r17_v10_eval_smoke3_20260425`：`annual_return = 1.0177`、`sharpe = 2.3094`、`max_drawdown = -0.1180`、`monthly_return_mean = 0.0516`、`monthly_consistency_score = 0.7684`、`avg_turnover = 0.0752`。
  - r17 smoke3 执行语义：`direct_action_core_deploy_target_realized_rate = 0.9921`、`direct_action_add_authorized_realized_rate = 0.9917`、`direct_action_pair_reallocation_source_count = 65`、`deploy_intent_realized_rate = 0.7143`、`add_to_hold_conflict_share = 0.0`、`direct_action_order_translation_conflict_rate = 0.1764`。
  - r17 audit smoke3：`sell_execution_origin_counts = {budget_sell_priority: 9, direct_action_pair_reallocation: 65, model_release_signal: 1, model_sell_intent: 6, weight_translation: 19}`，修正后 `budget_origin_sell_share = 0.28`，pair-source 不再混入 budget-origin sell。
- 动作后复盘：
  - 事实：r17 直接修复了 r16 的核心成交瓶颈，direct add/open 不再大面积被预算层压成 hold。
  - 事实：pair-source 引入了新的权衡：`direct_action_pair_reallocation_sell_share = 0.65`，主要冲突从 `add -> hold` 转为可审计的 `add -> reduce`，换手也上升。
  - 推断：这说明模型开始具备“今日从较弱 add 切到更强 core target”的组合级日决策雏形，但 pair-source 的长期机会成本尚不能用单次 smoke 证明。
  - 决策：r17 是当前最佳 research 修复证据，但仍保持 `shadow_only`；下一步应做 bounded shadow study，并重点审计 pair-source 相对收益、换手成本和月度稳定性。

## 2026-04-25 r17 bounded study / repaired confirm / pair-source audit 收口

- 行动前自检：
  - 事实：用户要求按“高瞻远瞩的策略规划者”方案直接执行，核心任务是把 r17 从 smoke 推进到 bounded study，并审计 pair-source 机会成本。
  - 事实：r17 smoke 已证明可成交性，但 smoke 不能替代 formal / bounded evidence；promotion 和 live 仍冻结。
  - 推断：本轮最高价值不是继续写 r18 代码，而是先验证 v10 成对换仓机制是否稳定，并把新的代价量化清楚。
  - 假设：`cp_v3_direct_action_pair_reallocation_r17__study_r1` 的 4 trial screening + repaired confirm 足以作为当前信息下的正式 shadow 对照。
- 已完成执行：
  - 启动并阻塞等待 `split_heads_direct_action_pair_reallocation_r17 + direct_action_pair_reallocation_v1` bounded study；4 个 screening 全部完成，0 个 screening failed。
  - 自动 confirm 阶段触发 `[Errno 22] Invalid argument`，但训练进程已推进到 confirm 目录；随后用 direct protocol rerun 修复 `confirm_01`，用 65 epoch strict resume 修复已完成 64 epoch 但缺 summary 的 `confirm_02`。
  - 生成 repaired confirm 对照：`daily_research/output/continuous_policy/studies/cp_v3_direct_action_pair_reallocation_r17__study_r1/manual_confirm_repair_summary.json` 与 `manual_confirm_repair_comparison.csv`。
  - 生成 pair-source 专项审计：`pair_source_audit_summary.json` 与 `pair_source_audit_comparison.csv`，并顺序导出 `confirm_01 / confirm_02` held-side detail，避免 latest 审计竞争。
- 关键结果：
- 筛选冠军：`trial_03 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 0.9682`、`sharpe = 2.2180`、`max_drawdown = -0.1183`、`monthly_return_mean = 0.0519`、`monthly_consistency_score = 0.7703`、`direct_action_core_deploy_target_realized_rate = 0.9844`、`add_to_hold_conflict_share = 0.0090`、`avg_turnover = 0.0686`，`promotion_status = shadow_only`。
- 修复后 confirm 冠军：`confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 0.6225`、`sharpe = 2.0071`、`max_drawdown = -0.1128`、`monthly_return_mean = 0.0369`、`monthly_consistency_score = 0.7374`、`direct_action_core_deploy_target_realized_rate = 0.9933`、`add_to_hold_conflict_share = 0.0`、`avg_turnover = 0.0890`，`promotion_status = shadow_only`。
  - repaired confirm_02：`result_value_v10` 方向明显弱，`annual_return = 0.0805`、`sharpe = 0.4082`、`max_drawdown = -0.1698`，继续不能升为默认预算目标。
  - pair-source 审计：`trial_03` core-minus-pair 5 日超额均值 `+0.00936`，`confirm_01` 为 `+0.00710`，`confirm_02` 为 `+0.02734`；但 `trial_01` 为 `-0.00908`、`trial_02` 约 `-0.00046`，说明 pair-source 排序有价值但不稳定。
  - held-side 审计：`confirm_01` 的 `direct_action_pair_reallocation_sell_count = 87`、`direct_action_pair_reallocation_sell_share = 0.6744`、`budget_origin_sell_share = 0.2713`；`confirm_02` 的 `budget_origin_sell_share = 0.5495`，卖出责任链仍未闭合。
- 动作后复盘：
  - 事实：r17 bounded study 证明 v10 机制能稳定修复 core target 成交，不再是单次 smoke 偶然。
  - 事实：正式 confirm 收益低于 smoke，且 promotion gate 仍被 `cash_timing_quality_1d`、`max_drawdown`、reduce/exit 质量等项拦住。
  - 推断：主矛盾已经从“add/open 能不能成交”转为“pair-source 是否长期值得牺牲、cash/release 是否主动、卖出责任链是否干净”。
  - 决策：r17 作为当前最佳 shadow 修复基线收口；下一轮不应继续无约束扩大 pair-source，而应推进 r18 pair-source cost guard、主动 cash timing 和卖出责任链修复。

## 2026-04-25 r18 pair-source opportunity cost guard 与根因写回

- 行动前自检：
  - 事实：用户明确要求记住五个卡点，并基于“目标函数与组合级决策不一致”这一第一根因继续解决问题。
  - 事实：r17 已修复 core deploy target 成交，但暴露 pair-source 可能也是好票，只是相对 core target 没那么强；这说明卖出源选择必须做相对机会成本归因。
  - 推断：继续堆 open/add/hold/reduce/exit 动作 loss 会强化局部目标冲突，最高价值动作应是把 pair-source、cash、release 与月度收益质量纳入组合级目标。
  - 边界：本轮仍不切 live、不改 active artifact、不改 promotion gate；r18 只能作为 research / shadow 修复证据。
- 已完成实现：
  - `portfolio_simulator.py` 新增 `cash_constraint_direct_action_pair_cost_guard_v11`，把 pair-source 从“可释放资金”升级为“相对 core target 有足够 spread 且机会成本可接受”。
  - `pipeline_utils.py` 与 `analyze_behavior_gap.py` 新增 `direct_action_pair_opportunity_spread`、`direct_action_pair_source_opportunity_cost`、`direct_action_pair_source_release_score`、`direct_action_core_minus_pair_forward_excess_5d` 等审计字段。
  - `run_self_optimizing_study.py` 新增 `split_heads_direct_action_pair_cost_guard_r18` 与 `direct_action_pair_cost_guard_v1`，把 pair-source spread、source cost、core-minus-pair forward excess、换手和月度收益质量纳入 scoring。
  - `doc_guard.py` 与 `project_consistency_check.py` 新增 r18 标记守卫，防止 profile / objective / simulator / audit / brain 文档再次漂移。
  - 主脑、分脑状态中枢、知识中枢、操作中枢与设计合同已写回五个卡点、根因排序和 r18 当前边界。
- 验证与证据：
  - `py_compile` 已通过本轮修改的 continuous_policy 与工具文件。
  - r18 dry-run `verify_direct_action_pair_cost_guard_r18_dryrun_20260425` 已生成 4 条 trial 计划。
- r18 第二次 smoke `verify_direct_action_pair_cost_guard_r18_v11_confirm01_smoke2_20260425`：`annual_return = 0.4891`、`sharpe = 1.9492`、`max_drawdown = -0.1060`、`monthly_return_mean = 0.0322`、`monthly_consistency_score = 0.6693`、`avg_turnover = 0.0286`。
  - r18 smoke2 pair-source 事实：`direct_action_pair_reallocation_source_count = 31`、`direct_action_pair_cost_guard_blocked_count = 327`、`direct_action_pair_source_spread_mean = 0.1490`、`direct_action_pair_source_cost_mean = 0.6716`、`direct_action_core_minus_pair_forward_excess_5d = 0.0060`。
  - r18 smoke2 audit 仍把 `direct_action_intent_not_preserved`、`deploy_intent_not_executable`、`budget_action_entanglement` 与 `sell_execution_source_entangled` 判为高优先级瓶颈。
- 行动后复盘：
  - 事实：r18 修复了“pair-source 是否相对 core target 值得牺牲”的量化与守门问题，且减少了无约束换手。
  - 事实：r18 绝对收益低于 r17 repaired confirm，说明它不是最终提升来源，只是把错误的 source credit assignment 明确化、可审计化。
  - 推断：真正的大提升仍需组合级日决策模型，直接学习资金获得者、资金释放者、释放幅度与现金保留，而不是继续学习单只股票该 add 还是 hold。
  - 决策：下一轮主线应转向 portfolio-level pair/listwise ranking、直接优化多日/月度组合收益、卖出与 cash timing 联合 credit assignment；不再把动作 loss 堆叠作为主方向。
## 2026-04-25 r19 实现过程
- 已实现组合级日频排序路径，把 simulator execution 中的 receiver/source/cash 信号贯穿到 metrics、behavior audit、study scoring 和 guard checks。
- 下一阶段研究目标已重构为资金接收方、资金来源、释放金额和现金保留，而不是继续围绕独立 action heads。
- r19 保持为 research/shadow-only 证据；没有改变 live default、promotion status 或 active artifact。
## 2026-04-25 r19 核验过程
- 已核验 compile、dry-run profile expansion、v12 smoke evaluation、behavior audit、直接 `portfolio_daily_ranking_v1` scoring、doc guard、brain integrity、project consistency 和 `git diff --check`。
- smoke 证据：annual return 0.843131、Sharpe 2.072912、max drawdown -0.122117、monthly consistency 0.723549、receiver-source forward excess 0.002383、source realized sell rate 0.551948。
- 残余风险：source forward excess 仍为正，数值为 0.010149，所以下一个根问题仍是 sell/source credit assignment，而不是继续堆叠 action-loss。
## 2026-04-26 r19 bounded study 与运行规则纠偏
- 自检前动作：完整脑接管后，r19 只有 dry-run 和 smoke 证据；下一条正式 shadow 证据缺口是 bounded `split_heads_portfolio_daily_ranking_r19 + portfolio_daily_ranking_v1` study。
- 执行：用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`、`KMP_DUPLICATE_LIB_OK=TRUE` 启动 `cp_v3_portfolio_daily_ranking_r19__study_r1`，包含 `4` 个 screening trial 和 `2` 个 confirmatory trial；进程自然完成并写出 `study_summary.json`。
- 核验：六个训练目录均报告 `trainer_backend = formal_torch_seq_v3`、`device = cuda`、`cuda_available = true`、strict resume，且训练诊断完整。
- 结果：screening 完成 `4/4`，confirmatory 完成 `2/2`，failed trial count 为 `0`，latest state 已恢复到 `cp_v3_direct_action_pair_reallocation_r17__study_r1__confirm_02`。
- 证据：`confirm_01` 是表现线（`annual_return = 0.967208`、`sharpe = 2.394526`、`monthly_consistency_score = 0.772378`），但 `portfolio_daily_receiver_minus_source_forward_excess_5d = -0.008111`。
- 证据：`confirm_02` 是综合稳定线，receiver-source spread 更好（`0.031289`）且 source sell realization 为 `0.692308`，但真实收益为负（`annual_return = -0.221432`、`sharpe = -0.565395`）。
- 决策：r19 仍是 `research / shadow_only`；有用教训是组合级 ranking 是正确问题框架，但 `portfolio_daily_ranking_v1` 当前需要 scoring alignment repair，之后才能讨论 promotion。
- 运行规则纠偏：脑规则已更新为所有项目任务前台运行、不中断、使用 `10h` 窗口，且 `daily_research` 任务必须在 `yolos` 下运行；GPU 训练证据写回前必须从 diagnostics 核验。

## 2026-04-26 r20 v2 gated 与 v13 cash-aware 执行复盘
- 行动前自检：
  - 事实：r19 已自然完成且 GPU 诊断完整，但 v1 objective 把 `confirm_02` 的 receiver-source spread 奖励到高位，同时忽略其负收益和超回撤。
  - 事实：r19 全部路线 `cash_reserve_rate = 0.0`，说明现金分支在组合日频排序里没有真正参与竞争。
  - 推断：最高价值动作不是继续跑更多 v1 trial，而是重写 scoring gate、修复现金竞争、校正组合有效意图，并防止失败 confirm 被自动选为 champion。
- 已完成实现：
  - `run_self_optimizing_study.py` 新增并默认启用 `portfolio_daily_ranking_v2_gated`，对收益、Sharpe、月度收益、回撤、月度一致性、执行冲突、add-to-hold 和现金保留进行 gate-first scoring。
  - `portfolio_simulator.py` 新增 `cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13`，把现金保留从近零死分支变成由弱 receiver、弱市场宽度、暴露、换手、稀疏 receiver 和回撤共同驱动的竞争项。
  - `portfolio_simulator.py` 与 `analyze_behavior_gap.py` 新增 `portfolio_daily_effective_model_action`，避免把未被组合排序选中的原始 `add/open` 候选误算成最终执行冲突。
  - 新增 `daily_research/tools/portfolio_daily_ranking_gate_report.py`，用于离线 v1/v2 重排、反冠军诊断和 gate 报告。
  - 修正 v2 champion selector：confirmatory 必须过 v2 gate 才可优先，否则回退到最佳 completed screening，并记录被拒 confirm。
- 执行证据：
  - 第一次 smoke 因 `current_gross` 局部变量作用域失败，已修复后重跑；该失败保留为真实工程证据。
  - retry1 smoke 让现金分支变活，`cash_reserve_rate = 0.009479`，但因原始 `add/open` 候选被误计为执行意图，`order_translation_conflict_rate = 0.398104`、`add_to_hold_conflict_share = 0.5`。
  - retry2 smoke 通过全部 v2 gate：`annual_return = 0.316303`、`sharpe = 1.152146`、`max_drawdown = -0.129243`、`monthly_return_mean = 0.020376`、`monthly_consistency_score = 0.727527`、`receiver_minus_source_5d = 0.004380`、`source_realized_sell_rate = 0.8`、`order_translation_conflict_rate = 0.033175`、`add_to_hold_conflict_share = 0.072464`。
  - bounded confirmatory 暴露 fresh confirm 失败：`annual_return = -0.253312`、`sharpe = -0.587612`、`max_drawdown = -0.220648`、`monthly_return_mean = -0.015989`、`add_to_hold_conflict_share = 0.578947`。
- 行动后复盘：
  - 事实：v2/v13 可以同时修复 v1 奖励错位、现金死分支和候选动作误判，retry2 已给出一条完整过 gate 的短窗证据。
  - 事实：fresh confirm 明确失败，说明短窗过 gate 不等于稳定模型，champion selector 必须拒绝失败 confirm。
  - 推断：当前瓶颈已经从“能否形成组合日频排序”推进到“v2/v13 能否跨 confirmatory 长窗稳定保持收益、回撤、现金和执行一致性”。
  - 决策：r20 保持 `research / shadow_only`；下一轮应在 v2/v13 下扩大 bounded evidence，并把失败 confirm 当作稳定性约束，而不是当作可 promotion 的候选。

## 2026-04-26 r20 stability sweep 与 source realization gate 复盘
- 行动前自检：
  - 事实：上轮 v2/v13 已通过 smoke，但 fresh confirm 失败；下一条最高价值证据是多候选 stability sweep，而不是继续追加单候选 smoke。
  - 事实：当前 diagnostics 能证明 CUDA，但历史 `training_diagnostics.json` 缺少解释器路径字段，证据链对 `yolos` 的表达不够硬。
  - 推断：本轮应同时做稳定性 profile、confirm-vs-screening 稳定性检查、source 真实释放门槛和诊断字段修复。
- 已完成实现：
  - 新增 `split_heads_portfolio_daily_ranking_stability_r20`，以低学习率和更高 dropout 作为稳定性 baseline，并扩展候选组合。
  - v2 champion selection 改为 `portfolio_daily_v2_stable_confirmatory_then_screening_fallback`，新增 `portfolio_daily_v2_confirm_stability_checks`。
  - `portfolio_daily_ranking_v2_gated` 新增 `source_realized_sell_floor`，当 source target 足够多时要求 `portfolio_daily_source_realized_sell_rate >= 0.35`。
  - `model_seq_v3.py` 未来会把 `python_executable`、`conda_prefix` 与 `runtime_env` 写入 `training_diagnostics.json`。
  - gate report 改为区分 `v2_top_ranked` 与 `qualified_v2_champion`，避免把未过 gate 的分数第一误读成合格冠军。
- 执行证据：
  - `cp_v3_portfolio_daily_ranking_r20_stability_sweep_20260426` 前台自然完成，包含 `6` 个 screening 与 `2` 个 confirmatory，`failed_trial_count = 0`。
  - 全部 8 个训练诊断均为 `device = cuda`、`cuda_available = true`、`trainer_backend = formal_torch_seq_v3`、strict resume；命令由 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` 启动。
  - 现金分支已转活：报告中无 `cash_branch_alive` 失败，`cash reserve` 天数合计 `139`。
  - 加入 source realization gate 后，`qualified_v2_champion` 为空。
  - `confirm_01` 是经济冠军，`annual_return = 0.570476`、`sharpe = 1.448580`，但 `max_drawdown = -0.182756`，触发 `max_drawdown_floor`。
  - `confirm_02` 是旧 v1 top-ranked，`annual_return = 0.465136`、`sharpe = 1.419674`、`max_drawdown = -0.171690`，但 `source_realized_sell_rate = 0.088889`，触发 `source_realized_sell_floor`。
- 行动后复盘：
  - 事实：v13 已经解决现金死分支，但 v2/v13 仍会产生“source 排得漂亮、资金却没真正释放”的假突破。
  - 事实：回撤边界非常敏感，`confirm_01` 只差约 `0.002756` 就过 `-0.18`，但仍必须按硬门槛拒绝。
  - 推断：下一轮真正要攻的是 source target 到 reduce/exit 的执行耦合，以及回撤边界下的收益保留，而不是继续提高 receiver-source spread。
  - 决策：本轮无合格 champion；r20 继续 `research / shadow_only`，不进入 promotion 或 live。

## 2026-04-26 r21 source-exec guard 实施复盘
- 触发原因：r20 stability sweep 显示现金分支已转活，但 `confirm_02` 的 `source_realized_sell_rate = 0.088889` 暴露 source 被选中后没有真实释放资金。
- 本轮动作：新增 `cash_constraint_portfolio_daily_ranking_source_exec_guard_v14` 与 `split_heads_portfolio_daily_ranking_source_exec_r21`，把 source target 的真实减仓从事后评分推进到模拟器执行链路。
- 关键实现：v14 对 source target 降低 retention floor、提高 sell reduction priority、压低 deploy priority，并在 translation guard 中绕开原始 add/hold 保护导致 source 被锁回持有的路径。
- 新审计：新增 `portfolio_daily_source_target_not_sold_share`、`portfolio_daily_source_target_not_sold_reason`、`portfolio_daily_source_exec_cap_guard_count`、`portfolio_daily_source_realized_reduction_weight`、`portfolio_daily_receiver_realized_deploy_count` 与 `portfolio_daily_effective_capital_transfer_count`。
- 新 gate：`source_not_sold_ceiling` 与 `source_realized_sell_floor` 配套；后续报告必须区分“source 被选中”“source 被减仓”和“receiver/source 真实形成资金转移”。
- 执行证据：
  - 第一次 r21 smoke 前台自然完成，证明 source 执行链路有效（source target 全部真实卖出），但暴露 v14 未继承 v13 cash-aware 分支，导致 `cash_branch_alive` 失败。
  - 已修复 v14 cash-aware 条件；retry2 前台自然完成，`device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、strict resume。
  - retry2 指标：`annual_return = 0.126142`、`sharpe = 0.677610`、`max_drawdown = -0.112456`、`monthly_return_mean = 0.008781`、`monthly_consistency_score = 0.564085`、`source_realized_sell_rate = 1.0`、`source_not_sold_share = 0.0`、`effective_capital_transfer_count = 17`、`cash_reserve_rate = 0.097872`。
  - retry2 gate report 仍无合格 v2 champion，失败 gate 为 `order_translation_conflict_ceiling` 与 `add_to_hold_conflict_ceiling`。
- 状态：本轮为代码侧执行耦合修复，不改变 live，不改变 active execution artifact；第一瓶颈已从 source execution 推进到 order translation / add-to-hold 冲突。

## 2026-04-26 r22 receiver-exec guard 实施复盘
- 触发原因：r21 retry2 已让 `source_realized_sell_rate = 1.0`、`source_not_sold_share = 0.0`，但仍失败于 `order_translation_conflict_rate = 0.289362` 与 `add_to_hold_conflict_share = 0.454545`。
- 根因分析：
  - r21 中 source 已经释放资金，残余冲突集中在 receiver/add 侧。
  - `portfolio_daily_receiver_target` 把已持仓且没有足够加仓空间的标的选成 core receiver，导致执行层无法再加仓，只能把 add 翻译成 hold/reduce。
  - 这不是“项目思路错了”，而是组合日频目标还少了一条执行前合同：receiver 不仅要值得买，还必须能买。
- 本轮动作：
  - 新增 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
  - 新增 `split_heads_portfolio_daily_ranking_receiver_exec_r22`。
  - 在模拟器中新增 `portfolio_daily_receiver_add_headroom`、`portfolio_daily_receiver_min_add_delta`、`portfolio_daily_receiver_exec_guarded` 与 `portfolio_daily_receiver_exec_guard_reason`。
  - 在 behavior audit、continuity metrics、study scoring 和 gate report 中接入 receiver guard 指标。
  - 修复 `run_self_optimizing_study.py` 的 study 顶层元数据，避免 search profile trial 明明使用 v15，但 study summary 顶层仍显示 `budget_calibration = none`。
- 执行证据：
  - dry-run `verify_r22_receiver_exec_profile_metadata_20260426` 确认 `budget_semantics = action_budget_split_v1`、`budget_calibration = cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`。
  - smoke retry2 `cp_v3_portfolio_daily_ranking_r22_receiver_exec_smoke_20260426_retry2` 前台自然结束。
  - 训练诊断为 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、`completed_epochs = 6`、`best_epoch = 6`、strict resume。
  - gate report 产物：`daily_research/output/continuous_policy/studies/cp_v3_portfolio_daily_ranking_r22_receiver_exec_smoke_20260426_retry2/portfolio_daily_ranking_v2_gate_report/portfolio_daily_ranking_v2_gate_report.md`。
- 关键结果：
  - r22 retry2 生成合格 v2 champion：`cp_v3_portfolio_daily_ranking_r22_receiver_exec_smoke_20260426_retry2__trial_01`。
  - `annual_return = 0.126142`、`sharpe = 0.677610`、`max_drawdown = -0.112456`、`monthly_return_mean = 0.008781`。
  - `portfolio_daily_receiver_target_count = 16`、`portfolio_daily_receiver_exec_guard_count = 37`、`portfolio_daily_receiver_realized_deploy_rate = 1.0`、`portfolio_daily_receiver_unrealized_deploy_share = 0.0`。
  - `portfolio_daily_source_target_count = 55`、`portfolio_daily_source_realized_sell_rate = 1.0`、`portfolio_daily_source_target_not_sold_share = 0.0`、`portfolio_daily_effective_capital_transfer_count = 16`、`portfolio_daily_cash_reserve_rate = 0.097872`。
  - `order_translation_conflict_rate = 0.136170`、`add_to_hold_conflict_share = 0.0`。
  - receiver guard 原因分布：`no_position_cap_headroom = 36`、`insufficient_min_add_headroom = 1`。
- 行动后复盘：
  - 事实：v15 在不破坏 source/cash 的前提下，把 r21 的 receiver unrealized deploy 直接压到 `0`。
- 事实：本轮仍是 `6` epoch smoke，`training_evidence_status = insufficient`，没有 confirmatory 稳定性证据。
- 推断：当前最高价值下一步是 bounded confirmatory 或更长 budget 的 r22 稳定性验证，而不是继续堆单点 guard。
- 决策：r22 仍为 `research / shadow_only`，不改变 live、不改变 active execution artifact、不进入 promotion。

## 2026-04-26 r22 formal 48 epoch 执行复盘
- 行动前自检：
  - 用户要求一次性完成优先级行动方案；上一轮收敛的 P0 是补齐 r22/v15 的正式训练证据与 confirmatory 稳定性。
  - 当前规则为所有任务前台运行、不转后台、不打断，窗口时限 10h；`daily_research` 必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
  - 启动前语义进程检查没有发现遗留 `daily_research` Python 训练进程。
- 执行：
  - 前台运行 `cp_v3_portfolio_daily_ranking_r22_receiver_exec_formal_20260426`，先用 `trial-count = 2`、screening `24/16` epoch、confirmatory `32/24` epoch 完成一轮 formal。
  - 首轮 formal 自然结束并生成 gate report；`confirm_01` 自身过 v2 gates，但训练证据仍因 `best_epoch_not_at_edge` 不足。
  - 按训练证据建议沿同一 run_dir strict resume 到 screening `48/40` epoch 与 confirmatory `48/40` epoch。
- 结果：
  - strict resume 后 `trial_01`、`trial_02`、`confirm_01` 均 `training_evidence_status = sufficient`，GPU/yolos 证据均为 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`。
  - gate report 合格 v2 champion 为 `cp_v3_portfolio_daily_ranking_r22_receiver_exec_formal_20260426__trial_01`，`annual_return = 2.002426`、`sharpe = 3.868514`、`max_drawdown = -0.111321`、`monthly_return_mean = 0.084873`。
  - 执行指标为 `order_translation_conflict_rate = 0.110211`、`add_to_hold_conflict_share = 0.0`、`receiver_realized_deploy_rate = 1.0`、`receiver_unrealized_deploy_share = 0.0`、`source_realized_sell_rate = 0.9875`、`effective_capital_transfer_count = 39`。
  - `confirm_01` 自身过 v2 gates 且训练证据充分，但相对 screening 衰减，失败稳定性项为 `annual_return_decay_limit`、`sharpe_decay_limit`、`monthly_return_decay_limit`。
- 动作后复盘：
  - r22/v15 已从 smoke 机制证明升级为正式训练证据充分的 research 主线。
  - 但 stable confirmatory 仍未成立；这不是终止 bug，也不是脚本循环问题，而是模型在确认阶段收益复现能力不足。
  - 本轮命令均自然退出；最终语义进程检查没有发现遗留 `daily_research` Python 进程。
  - 代码编译、brain integrity、doc guard、project consistency 与 `git diff --check` 均通过。
- 决策：
  - r22/v15 继续 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。
  - 下一轮优先级应转向降低 screening/confirmatory 衰减：收紧高收益 trial 的稳健约束、加入 confirm-stability-aware search 或在 r22/v15 基础上做小范围稳定性搜索，而不是继续扩大 receiver guard。

## 2026-04-27 r23 receiver-exec stability 执行复盘
- 行动前自检：
  - r22/v15 已证明执行合同正确，但 stable confirmatory 为空；本轮 P0 是把收益衰减问题转成稳定性优先搜索。
  - 启动前语义进程检查未发现遗留 `daily_research` 训练进程；所有命令继续使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` 前台运行，窗口按 10h 处理。
- 本轮动作：
  - 新增 `split_heads_portfolio_daily_ranking_receiver_exec_stability_r23`。
  - r23 继承 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15`、`alpha_result_value_budget_split_v15` 与 `portfolio_daily_ranking_v2_gated`。
  - r23 收窄到低学习率、高 dropout、单一 `result_value_v9`，减少高收益 screening 的不稳定自由度。
- 执行：
  - dry-run `verify_r23_receiver_exec_stability_profile_20260426` 通过，确认 study plan 元数据正确。
  - 首轮前台训练 `cp_v3_portfolio_daily_ranking_r23_receiver_exec_stability_20260426` 自然结束，包含 `3` 个 screening 与 `2` 个 confirmatory；但 diagnostics 显示多个 best epoch 贴边。
  - 沿同一 tag strict resume 到 `64/56` epoch，任务自然结束。
- 结果：
  - 最终 5 条训练诊断均为 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、`completed_epochs = 64`、strict resume，且 `resumed_from_checkpoint` 非空。
  - gate report 的 `qualified_v2_champion` 为 `cp_v3_portfolio_daily_ranking_r23_receiver_exec_stability_20260426__confirm_01`，且 `confirm_stable = True`。
  - `confirm_01` 指标：`annual_return = 0.351246`、`sharpe = 1.191887`、`max_drawdown = -0.136122`、`monthly_return_mean = 0.023111`、`receiver_minus_source_5d = 0.025178`、`source_realized_sell_rate = 0.979798`、`order_translation_conflict_rate = 0.134128`、`add_to_hold_conflict_share = 0.0`。
  - `confirm_02` 也是 stable confirmatory，且是经济冠军：`annual_return = 1.337017`、`sharpe = 2.936905`、`max_drawdown = -0.154638`、`monthly_return_mean = 0.065722`。
- 行动后复盘：
  - r23 证明稳定性优先搜索有效解决了 r22 的核心残差：stable confirmatory 不再为空。
  - r23 同时暴露新的取舍：稳定 v2 champion 的收益低于 r22 高峰；经济冠军收益更高，但结构质量不是第一。
  - 当前最优策略不是上线，而是在 r23 稳定性配置基础上恢复收益上限，或引入 confirm-stability-aware 的二阶段筛选。
- 决策：
  - r23 仍为 `research / shadow_only`，不切 live、不改 active artifact、不进入 promotion。
  - 后续若继续推进，必须保留 v15 执行合同和 stable confirmatory gate，不得回到只追求 screening 年化。

## 2026-04-27 r24 listwise allocation 执行复盘
- 行动前判断：r23 已解决 stable confirmatory 为空的问题，但仍是 guard/simulator 主导；用户要求先做 P1 和 P4，因此本轮目标是让 receiver 可买性与组合日分配进入训练目标，而不是继续堆后置过滤。
- 已完成实现：`label_builder.py` 生成 receiver headroom/min_add_delta/capacity/executability/receiver_score/source_score/cash_score 与候选 mask；`pipeline_utils.py` 把执行反馈写回这些标签并新增 continuity metrics；`model_seq_v3.py` 新增 `alpha_result_value_budget_split_v16`、`deploy_executability_head`、portfolio listwise heads、pairwise/cash-margin loss 和推理侧融合；`portfolio_simulator.py` 读取模型 listwise 输出参与 receiver/source/cash 排序；`run_self_optimizing_study.py` 新增 `split_heads_portfolio_daily_listwise_allocation_r24`。
- 验证：`py_compile` 通过；r24 dry-run 通过；r24 smoke 在前台自然结束，trial 8 epoch 与 confirm 10 epoch 均为 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、`trainer_backend = formal_torch_seq_v3`、`resume_mode = strict`。
- 新头证据：trial 与 confirm diagnostics 均显示 `supports_deploy_executability_head = true`、`supports_portfolio_listwise_heads = true`。
- 结果边界：smoke 仍为 `shadow_only`，`training_evidence_status = insufficient`；confirm_01 年化 `0.027326`、Sharpe `0.239028`、最大回撤 `-0.152518`、月均 `0.003561`，稳定性失败项为 `confirm_annual_return_floor`、`confirm_sharpe_floor`、`confirm_monthly_return_floor`。
- 结构观察：confirm_01 的 `receiver_unrealized_deploy_share = 0.0`、`cash_reserve_rate = 0.110526`，但 `source_target_count = 0`、`effective_capital_transfer_count = 0`，说明 r24 链路已能学 receiver/cash 侧，但资金释放/source 侧仍未形成有效日分配闭环。
- 决策：r24 是 P1/P4 机制落地，不是 promotion 证据；下一轮若继续，应做 bounded confirmatory 的 v16 参数稳定性搜索，而不是把本次 smoke 当成正式策略结论。
## 2026-04-27 r25 source-release listwise 实施记录
- 行动前判断：r24 已完成 P1/P4 的第一阶段，但 confirm 中 source target 消失，说明长期正确方向不是继续堆 receiver guard，而是补齐 source 释放资金的可学习信号。
- 已执行实现：新增 source release capacity/executability 标签、模型头、v17 loss、r25 profile、模拟器排序字段、continuity metrics、source 消失审计工具，并把 guard/doc/project consistency 同步到 r25。
- 待验证事实：需要用 yolos 前台完成 r25 dry-run、r25 smoke、r24 confirm source audit、GPU/yolos diagnostics、doc guard、project consistency 与语义进程检查。
- 当前边界：r25 只是 research/shadow 修复路线；任何 smoke 结果都不能替代 bounded/fresh confirm 与 promotion gate。
## 2026-04-27 r25 confirmfix + release-quality 复盘
- 起因：用户要求先修 confirm 阶段 TQ 初始化失败，让 r25 能完成 bounded confirm；再针对负的 receiver-source forward spread 调整 source opportunity cost / release label，使“能卖”进一步变成“卖得对”。
- 执行：数据层修复 TQ session 唯一化、重试、失败日志与 universe/date fallback；策略层新增 source release-quality 标签、模型头、loss 权重、推理融合、模拟器排序和 audit 工具。
- 验证：`cp_v3_portfolio_daily_source_release_listwise_r25_confirmfix_release_quality6_20260427` 前台自然完成 trial + confirm，确认 TQ confirm 初始化失败不再阻断 bounded confirm。
- 关键发现：release-quality 生效后，confirm 中 source 被抑制为 0，负 spread 不再出现；审计显示评估窗口内的 held/context rows 多为正 forward excess，release-quality 很低，因此不卖是合理防错。
- 本质瓶颈：当前不是“脚本会不会结束”的工程问题，而是 source release 正样本稀疏、短训练预算泛化不足、best epoch 仍贴边导致 insufficient training evidence；模型学会了不要乱卖，但还没有证明能稳定主动卖出正确 source。
- 决策：本轮不进入 promotion/live，不改 active artifact；下一轮若继续推进，应在 r25 release-quality 机制上增加 confirm 训练预算或做小规模稳定性搜索，目标是非零 source target + 正 receiver-source spread + 正收益/月度质量 + sufficient training 同时成立。
## 2026-04-27 r25 quality6 审计补记
- 已补跑 `portfolio_daily_listwise_audit.py`，trial 与 confirm 均生成 summary/daily 审计文件。
- trial 审计显示 `top_source_release_quality = 0.035386`、`source_target_count = 0`、`source_realized_sell_rate = 0`，诊断为 funding/keep protection 阻断。
- confirm 审计显示 `top_source_release_quality = 0.035386`、`source_target_count = 0`、`source_realized_sell_rate = 0`，同样诊断为 funding/keep protection 阻断。
- 解释：当前机制已经避免把仍有持有价值的 source 硬卖掉；下一阶段要证明的是在真实低质量 source 出现时能够非零卖出，并保持正 spread 与正收益，而不是简单降低 protection。
## 2026-04-27 r25 P0/P1 执行复盘
- 本轮先按计划完成 P0 strict-resume，确认 TQ/confirm 链路已修好，但长预算暴露出新的真实瓶颈：source 会恢复 active，却卖掉正 forward source；receiver 也会生成大量不能兑现的 target。
- 第一轮修正只压住 source 误卖，未充分压住 receiver unrealized；第二轮加入 receiver funding-context slot cap 和更强 source observable/keep-value pass 后，短预算 confirm 恢复干净语义。
- 最新 v2 smoke confirm：`annual_return = -0.028102`、`sharpe = -0.275511`、`max_drawdown = -0.049638`、`receiver_realized_deploy_rate = 1.0`、`receiver_unrealized_deploy_share = 0.0`、`source_target_count = 0`、`receiver_forward_excess_5d = 0.031391`。
- 本轮最重要的本质结论：r25 的问题不再是“脚本能不能跑完”，也不再只是“source 能不能卖”，而是“模型内生分配目标在长预算下会过度自信”。继续推进前，必须先让双守门在长预算下稳定，再谈 P1 参数搜索。

## 2026-04-28 r25 v3 长预算执行复盘
- 本轮先按铁律用 `yolos` 前台完成 v2 长预算复核，结果 confirm `receiver_unrealized_deploy_share = 0.8`，证明上一轮 v2 smoke 的干净性只是短预算现象。
- 随后实现 v3：约束模型 receiver capacity 不能覆盖可观测 headroom，高仓位且无 funding context 时 receiver slot 可为 0，并加入 final-exec receiver guard，把最终没有真实正向 delta 的 receiver 从 target 中移除。
- v3 smoke confirm 恢复干净：`receiver_target_count = 3`、`receiver_realized_deploy_rate = 1.0`、`receiver_unrealized_deploy_share = 0.0`，但收益仍弱。
- v3 长预算 confirm 更重要：`receiver_target_count = 0`、`source_target_count = 0`、`cash_reserve_rate = 0.989510`、`annual_return = -0.386116`、`sharpe = -1.172900`。这不是可推广成功，而是 r25 在严格守门后退化为 dead allocation branch。
- 已把 dead-branch 惩罚写进 objective/gate。下一轮最有效路径不是继续放宽 r25 guard，而是回到 r23 稳定主线恢复收益上限，或升级 teacher/listwise allocation，让模型直接学 source/receiver/cash 的组合资金分配闭环。

## 2026-04-28 r26 allocation-teacher 执行复盘
- 行动前判断：r25 v3 已证明“清掉错误 receiver”不等于“学会组合分配”，因此本轮优先把 funding coverage、closure、transfer 与 dead-branch risk 做成模型可学习目标，而不是继续放宽 guard。
- 已执行实现：`label_builder.py` 新增四个 allocation teacher 标签并把 receiver demand 反推到 source release teacher；`model_seq_v3.py` 新增 v18 loss 与四个输出头；`pipeline_utils.py` 接入反馈、continuity metrics 与输出指标；`portfolio_simulator.py` 用新头参与 receiver/source/cash 评分；`run_self_optimizing_study.py` 新增 r26 profile 与 scoring penalty。
- 验证过程：dry-run 通过；第一次 smoke 暴露 `current_gross` 初始化顺序问题并已修复；smoke2/smoke3 证明 receiver unrealized 被压到 0 但 source 仍塌缩；smoke4 完整完成 screening + confirm，GPU/yolos 诊断成立。
- smoke4 关键结果：confirm `annual_return = 0.135025`、`sharpe = 0.570791`、`max_drawdown = -0.153265`、`monthly_return_mean = 0.010839`、`receiver_target_count = 2`、`receiver_realized_deploy_rate = 1.0`、`source_target_count = 0`、`cash_reserve_rate = 0.107143`、`training_evidence_status = insufficient`。
- 反证实验：尝试让 protected source override 穿透下单路径后，source target 能恢复且 sell rate 可达 1.0，但 receiver-source spread 变负、回撤变差，说明“能卖”不等于“卖得对”。最终只保留 protected release 审计字段，不让它影响下单。
- 本质结论：当前瓶颈是 listwise day allocation teacher 尚未学会稳定选择 funding source；硬打穿 `direct_action_funding_protected` 会把问题从“source 消失”变成“卖错 source”。r26 仍是 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。
## 2026-04-29 r27-r30 source economic release 与 cash-relief 复盘
- 本轮目标是继续把“能卖”推进成“卖得对”。r27 首先加入 source forward spread、bad-forward risk、economic release/block 四个经济释放信号与 `alpha_result_value_budget_split_v19`，但 bounded confirm 暴露出单点假阳性：`source_target_count = 1`，该 source 5 日超额为 `+0.125484`，`receiver_minus_source_forward_excess_5d = -0.143557`，说明模型把仍有前瞻强度的持仓误判为可释放资金源。
- r28 新增 `portfolio_daily_source_forward_strength_brake_risk`、模型头、`alpha_result_value_budget_split_v20`、simulator brake 与 v2 gate 检查；验证结果消除了 r27 的错误卖出，但 confirm `source_target_count = 0`，问题从“卖错”转为“卖不出来”。
- r29 增加低机会成本 direct-release relief 通道，只允许 `direct_action_pair_source_release_score` 高、`direct_action_pair_source_opportunity_cost` 很低、brake 低、且不是 open/add 意图的持仓进入救援候选。confirm 恢复为 `source_target_count = 1`，且该 source 5 日超额为 `-0.143821`，`receiver_minus_source_forward_excess_5d = +0.250541`，方向正确但数量不足。
- r30 允许合格 direct-release relief source 在没有当天 receiver 的情况下独立释放为现金。confirm 达到 `source_target_count = 5`、`source_realized_sell_rate = 1.0`、`source_target_not_sold_share = 0.0`、`source_forward_excess_5d = -0.009351`、`receiver_minus_source_forward_excess_5d = +0.021760`、`cash_reserve_rate = 0.020833`，说明 source 侧“卖得对”的结构已有进展。
- r30 仍未通过 promotion 级别验证：`receiver_target_count = 2` 低于 v2 gate 下限，且 confirm 失败项仍包含 `confirm_annual_return_floor`、`confirm_sharpe_floor`、`confirm_monthly_return_floor`。本轮只证明 source 释放机制更干净，不证明组合收益质量可上线。
- 全部任务均使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` 前台自然运行；训练 diagnostics 显示 `device = cuda`、`cuda_available = true`、`runtime_env = yolos`。本轮没有改 live、promotion 或 active execution artifact，所有结果保持 `research / shadow_only`。

## 2026-04-29 r31 receiver semantic closure 执行复盘
- 行动前判断：r30 已把 source 从“硬卖好票”推进到低机会成本 cash-relief，但 receiver 侧暴露新的最高优先级问题：`direct_action_add_authorized/open_authorized` 仍可能通过语义旁路停留在不可执行 held add 上，尤其是无 headroom 的已持仓标的。
- 已执行实现：在 `portfolio_simulator.py` 中把无 headroom held add 提前降级，强制 receiver 授权闭包为 executable receiver target 子集，并让 `portfolio_daily_effective_model_action` 反映最终 receiver 语义；新增 flat open breadth candidate，使高现金、低持仓、目标暴露较高的场景重新寻找新 receiver。
- 已执行审计与评分：`pipeline_utils.py`、`analyze_behavior_gap.py` 与 `run_self_optimizing_study.py` 已接入 receiver semantic no-headroom、authorized add no weight change、deploy unrealized、authorization subset violation、source positive forward sell share、strong false sell、exposure utilization 等指标，并写入 objective、v2 gate 与 confirm stability。
- 工程修复：第一次 bounded 运行暴露 `budget_deploy_score` 初始化顺序问题，已改为使用已存在的 daily head/flat 信号；第二次运行暴露旧 turnover 文件缺少 `gross_exposure`，已按 `1 - cash_weight` 提供兼容回退；`analyze_behavior_gap.py` 同步补齐旧 CSV 兼容。
- 验证：`py_compile` 与 r31 dry-run 通过；GPU/yolos 诊断通过，`device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`。bounded fix2 前台自然结束，`completed_trial_count = 1`、`confirmatory_completed_trial_count = 1`、`failed_trial_count = 0`。
- 关键结果：bounded confirm 中 `direct_action_authorization_subset_violation_count = 0`、`authorized_add_no_weight_change_share = 0.0`、`deploy_intent_unrealized_share = 0.0`、`receiver_unrealized_deploy_share = 0.0`，说明 receiver 语义旁路已被当前短预算验证压住；`portfolio_daily_receiver_open_breadth_candidate_count = 1`，说明 flat receiver 广度链路已重新接上。
- 失败边界：bounded confirm 仍为 `training_evidence_status = insufficient`，`stable_confirmatory_count = 0`，失败项包含 `confirm_gate_pass`、`confirm_annual_return_floor`、`confirm_source_count_floor`；`annual_return = 0.100831` 低于 confirm floor，`source_target_count = 0`，source 分布惩罚尚未被真实 source target 检验。
- 行动后复盘：r31 证明“语义闭包”可以被前置到 receiver target 层，而不是继续依赖 simulator guard 补漏洞；但当前最深瓶颈已经转为 source active selection 与收益质量。下一轮不应放宽 r30/r31 守门，而应在足够 confirm 预算下验证 r31 training evidence，并让 source/receiver/cash listwise allocation 学会真正的资金分配。
- 决策：r31 仍是 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。

## 2026-04-29 r33 source forward proxy / clean-pass 执行复盘
- 行动前判断：r31 已压住 receiver 语义旁路，但 source 侧仍存在两种相反失败模式：过严会退化为 `source_target_count = 0`，过松会卖出仍有强正 forward 的 source。因此本轮目标不是继续放宽 source，而是把“卖得对”的分布条件前置到 label、simulator 和 gate。
- 已执行实现：`portfolio_simulator.py` 新增 `portfolio_daily_source_forward_proxy_keep_risk`、`portfolio_daily_source_release_conviction`、`portfolio_daily_source_release_conviction_pass` 与 `portfolio_daily_source_distribution_clean_pass`；source candidate 现在必须通过 forward proxy、release conviction 与 distribution clean-pass。repeat release relief 仅保留窄口径 clean-pass 通道。
- 已执行训练信号：`label_builder.py` 生成 source forward proxy keep-risk、release conviction 与 distribution clean-pass 标签，并把它们纳入 source candidate mask；`model_seq_v3.py` 新增可训练 forward proxy head、v20 loss 权重、旧 artifact 兼容逻辑与 source-aware soft target，使 source candidate 倾向 reduce/exit 而非 hold/add。
- 已执行反馈与审计：`pipeline_utils.py`、`run_self_optimizing_study.py` 与 `analyze_behavior_gap.py` 已接入 proxy risk、release conviction、distribution clean blocked、positive forward sell share、strong positive false sell、max/p75 source forward 等指标，并把它们写入 scoring、v2 gate 与 confirm stability。
- 训练/验证事实 1：`cp_v3_portfolio_daily_source_forward_proxy_r33_trainable_proxy_bounded_20260429` 达到 `training_evidence_status = sufficient`，但 source dormant，`source_target_count = 0`、`receiver_target_count = 1`。这证明 proxy 本身会防错，但不能单独产生主动资金释放。
- 训练/验证事实 2：`cp_v3_portfolio_daily_source_forward_proxy_r33_source_soft_bounded_20260429` 恢复 source，confirm 中 `receiver_target_count = 5`、`source_target_count = 3`、`source_realized_sell_rate = 1.0`、`receiver_unrealized_deploy_share = 0.0`；但 `source_forward_excess_5d = +0.189710`、`source_positive_forward_sell_share = 0.6667`、`source_strong_positive_forward_sell_count = 2`，说明 soft target 会恢复“能卖”，但仍会“卖错”。
- 训练/验证事实 3：`cp_v3_portfolio_daily_source_forward_proxy_r33_release_conviction_bounded_20260429` confirm 收益与执行很强，`annual_return = 1.439900`、`sharpe = 3.022400`、`receiver_target_count = 8`、`source_target_count = 6`、`source_realized_sell_rate = 1.0`，但仍有 `source_positive_forward_sell_share = 0.6667`、`source_strong_positive_forward_sell_count = 2`、`source_max_forward_excess_5d = 0.140600`。release conviction 不能单独替代分布门槛。
- 训练/验证事实 4：clean-pass + repeat relief full replay 中 `source_target_count = 1`、`source_forward_excess_5d = -0.040850`、`source_positive_forward_sell_share = 0.0`、`source_strong_positive_forward_sell_count = 0`、`receiver_minus_source_forward_excess_5d = +0.039088`，说明强势误卖被挡住，方向正确。
- 训练/验证事实 5：`cp_v3_portfolio_daily_source_forward_proxy_r33_cleanpass_bounded_20260429` fresh bounded confirm 仍未通过，`training_evidence_status = insufficient`、`receiver_target_count = 2`、`source_target_count = 1`、`receiver_minus_source_forward_excess_5d = -0.018510`；同时 source positive/strong false sell 已为 0。当前失败已经从“误卖强势 source”转为“clean source 与 receiver 广度不足”。
- GPU/yolos 证据：本轮训练/评估均用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` 前台自然结束；关键 bounded run 的 diagnostics 显示 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`。
- 行动后复盘：r33 的价值是把 source 尾部误卖从后验 gate 推进到语义层，但它还没有解决组合日资金分配本体。后续不应把 clean-pass 重新放宽，而应扩大 clean source breadth、恢复 receiver breadth，并把 monthly return、exposure utilization、receiver realized deploy 与 positive spread distribution 更深地写进 objective/feedback。
- 决策：r33 当前仍是 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。

## 2026-04-30 r34 allocation breadth bounded / evidence-confirm 复盘
- 行动前判断：r34 已把 receiver/source breadth、joint economic quality 与 allocation teacher summary 接入 scoring / summary，但必须用 bounded screening + fresh confirm 验证是否稳定，而不能把 scaffold 当 verdict。
- 执行：用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`、`KMP_DUPLICATE_LIB_OK=TRUE` 前台运行 `cp_v3_portfolio_daily_allocation_breadth_r34_bounded_20260430`，完成 4 个 screening、2 个 confirmatory。用户中断发生在 orchestrator 写最终 study summary 前；检查无残留 Python 进程后，补齐 `confirm_02` strict resume，并重建 `study_summary.json` / `trial_ranking.csv`。
- 工程修复：发现旧 confirm candidate 选择会让 `training_evidence_status = insufficient` 的高分 screening trial 进入 confirm；已修为 v2 gated profile 候选优先 `training_evidence_status = sufficient` + v2 gate qualified，并把 confirm stability 显式检查 source/confirm training evidence。
- 追加验证：基于只含 screening 的 seed `cp_v3_portfolio_daily_allocation_breadth_r34_screening_seed_20260430`，运行 `cp_v3_portfolio_daily_allocation_breadth_r34_evidence_confirm_20260430`，只 fresh confirm evidence-sufficient 的 `trial_03`。
- GPU/yolos 证据：evidence confirm 的 diagnostics 为 `trainer_backend = formal_torch_seq_v3`、`device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、`completed_epochs = 61`、`best_epoch = 51`、`resume_mode = fresh`。
- 结果：evidence confirm 达到 `training_evidence_status = sufficient`、`receiver_unrealized_deploy_share = 0`、`source_realized_sell_rate = 1.0`、`cash_reserve_rate = 0.668333`、`annual_return = 0.808193`、`monthly_return_mean = 0.047187`，但 `stable_confirmatory = false`。
- 失败本质：核心失败不是 receiver 可执行性，也不是脚本/GPU 链路，而是 source 分布与资金去向相对价值：`receiver_minus_source_forward_excess_5d = -0.126380`、`source_forward_excess_5d = +0.085735`、`source_positive_forward_sell_share = 0.70`、`source_strong_positive_forward_sell_count = 4`、`source_max_forward_excess_5d = 0.548741`。
- 决策：r34 仍是 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。下一轮不应扩大搜索面，而应把 source opportunity cost / release label / distribution feedback 加强到能硬性压住正 forward source 误卖与负 receiver-source spread。

## 2026-04-30 r35 unified allocation 工程合同执行复盘
- 行动前判断：r34 已证明 receiver 可执行性、source 卖出执行率与现金保留可以同时达标，但仍会卖出强正 forward source 并造成负 receiver-source spread。因此下一步不能继续把 simulator guard 当主策略，而要把 source、receiver、cash 改成同一个 listwise allocation problem。
- 已执行实现：新增 `daily_research/continuous_policy/allocation_optimizer.py`，提供 `AllocationOptimizerConstraints`、`build_unified_allocation_problem`、`attach_unified_allocation_targets`、`build_unified_allocation_summary` 与 `solve_semidifferentiable_allocation`。solver 用连续 score + capped projection 显式约束 cash、turnover、position cap、transaction cost、slippage 与 sell tax。
- 已执行训练目标接入：`pipeline_utils.py` 在 label frame 阶段接入 unified allocation surface，并把核心 `portfolio_daily_receiver_score`、`portfolio_daily_source_score`、`portfolio_daily_cash_score` 替换为统一分配后的可训练目标；原始分数保留为 `portfolio_daily_pre_unified_*`，新增字段登记为 label/non-feature，避免未来信息泄漏成输入特征。
- 已执行模型与 study 接入：`model_seq_v3.py` 新增 `alpha_result_value_budget_split_v21` 与 unified allocation heads；`run_self_optimizing_study.py` 新增 `split_heads_portfolio_daily_unified_allocation_r35`，并把 `portfolio_daily_unified_allocation_objective`、`portfolio_daily_source_positive_forward_penalty`、`portfolio_daily_source_opportunity_cost_penalty` 与 `portfolio_daily_receiver_source_spread_reward` 写入 v2 scoring。
- 已执行协议与守卫接入：`run_continuous_policy_protocol.py` 将 `teacher_summary` 透传到 protocol summary，便于 study scoring 读取 `unified_allocation_summary_mean`；`project_consistency_check.py` 与 `doc_guard.py` 已加入 r35 合同标记。
- 当前验证范围：本轮完成 r35 单测、编译、dry-run 与文档/一致性守卫，不包含正式 bounded training/confirm，因此没有新增 GPU training diagnostics verdict。当前 r35 仍是工程合同与研究入口，不是 promotion 证据。
- 决策：r35 仍为 `research / shadow_only`；不得 promotion、不得 live、不得改 active artifact。下一步正式研究应跑 r35 bounded screening + fresh confirm，重点观察 `training_evidence_status`、v2 gates、confirm-vs-screening 稳定性、source positive forward penalty、receiver-source spread、monthly return 与 drawdown。

## 2026-05-01 r35 unified allocation 修复后 bounded confirm 复盘
- 根因定位：首轮 r35 bounded confirm 证明 training evidence 可充分，但 action panel 中 `portfolio_daily_unified_receiver_score/source_score/cash_score` 全为 0，断点不是 TQ、GPU 或训练目标，而是 `model_seq_v3.predict_policy_v3` 没有把 unified heads 传播到 policy frame；同时 simulator 旧候选逻辑仍过度依赖 direct action label。
- 已执行修复：`portfolio_simulator.py` 接入 unified receiver/source/cash heads，让 receiver/source candidate 可以由 unified allocation score 独立触发，并把 source positive forward penalty、opportunity cost penalty、receiver-source spread reward 写入 source 评分和 action row；`model_seq_v3.py` 将 unified heads 预测结果导出到 policy frame；`pipeline_utils.py` 聚合 unified objective / score / penalty / reward 到 evaluation 与 continuity metrics。
- 测试证据：新增回归测试 `test_predict_policy_exports_unified_allocation_heads_to_policy_frame`，先确认缺列失败，再修复为通过；`test_portfolio_daily_strategy_contracts` 当前 14 项通过，`py_compile` 覆盖 `model_seq_v3.py`、`portfolio_simulator.py`、`pipeline_utils.py` 与测试文件。
- 正式 bounded confirm：前台运行 `cp_v3_portfolio_daily_unified_allocation_r35_postfix4_bounded_confirm_20260430`，完成 4 个 screening 与 2 个 confirmatory，`failed_trial_count = 0`、`confirmatory_completed_trial_count = 2`、`stable_confirmatory_count = 1`，champion 为 `confirm_02`。
- GPU/yolos 证据：训练 diagnostics 显示 `trainer_backend = formal_torch_seq_v3`、`device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`，任务前台自然完成。
- champion 关键结果：`confirm_02` 为 `training_evidence_status = sufficient`、`annual_return = 1.409793`、`sharpe = 2.846432`、`max_drawdown = -0.127331`、`monthly_return_mean = 0.074407`、`receiver_target_count = 15`、`source_target_count = 9`、`receiver_unrealized_deploy_share = 0`、`source_realized_sell_rate = 1.0`、`cash_reserve_rate = 0.806071`、`receiver_minus_source_forward_excess_5d = 0.206611`。
- 仍未通过原因：`promotion_status = shadow_only`，失败项仍含 `reduce_success_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`；筛选分支中仍出现 strong positive source false sell，说明 r35 解决了 unified heads 旁路与正 spread 机制验证，但没有完成 promotion 级 cash/reduce/drawdown/source distribution 稳定性。
- 决策：r35 继续 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。下一轮不应再只补 simulator guard，应把 cash timing、drawdown、reduce/exit 质量和 source positive distribution 写进更强 allocation objective / feedback。

## 2026-05-01 r36 risk-aware unified allocation 与 source distribution 旁路修复复盘
- 行动前判断：r35 已证明 source/receiver/cash unified allocation 可以形成正 spread、非零现金和真实 source sell，但仍失败于 cash timing、drawdown、reduce/exit 与 source positive distribution。最高价值动作不是重跑 r35，而是把 cash timing、drawdown、source opportunity cost 与 positive source distribution 写进训练目标和执行约束。
- 已执行实现：`allocation_optimizer.py` 新增 risk-aware unified allocation target，接入 `market_downside_pressure`、`cash_regime_pressure`、`portfolio_drawdown_20d`、`forward_benchmark_return_1d/3d`、`cash_timing_target`、`source_distribution_quality` 与更强的 positive-forward/opportunity-cost penalty；`model_seq_v3.py` 新增 `alpha_result_value_budget_split_v22` 与 `_unified_allocation_consistency_loss`；`run_self_optimizing_study.py` 新增 `split_heads_portfolio_daily_risk_aware_unified_allocation_r36`。
- 已执行测试：新增并通过 r36 profile 注册、risk-aware cash/source consistency loss、forward benchmark cash timing target、source distribution bypass regression 等契约测试；`daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts` 当前 19 项通过，相关文件 `py_compile` 通过。
- r36 初始 screening：`cp_v3_portfolio_daily_risk_aware_unified_allocation_r36_screening_20260501` 证明训练链路可跑通，但 `training_evidence_status = insufficient`、`reduce_success_rate_5d = 0`、`cash_timing_quality_1d = -0.124542`、`max_drawdown = -0.143042`。
- r36 64/48 screening：`cp_v3_portfolio_daily_risk_aware_unified_allocation_r36_screening64_20260501` 达到 `training_evidence_status = sufficient`，并把 `reduce_success_rate_5d = 1.0`、`source_realized_sell_rate = 1.0`、`receiver_unrealized_deploy_share = 0` 拉正；但仍失败于 `exit_timeliness_rate_5d`、`cash_timing_quality_1d` 与 `max_drawdown`。
- r36b cash timing patch：加入 forward benchmark cash timing target 后，`annual_return = 1.877440`、`sharpe = 5.078928`、`max_drawdown = -0.053202`，但 `training_evidence_status = insufficient`，且 source 分布明显退化：`receiver_minus_source_forward_excess_5d = -0.053013`、`source_positive_forward_sell_share = 0.533333`、`source_strong_positive_forward_sell_count = 7`。
- r36b 根因定位：action panel 显示 source target 中 `source_distribution_clean_pass = false` 的持仓仍能通过 `portfolio_daily_unified_source_candidate` 进入最终 source candidate；普通 source candidate 已要求 clean-pass，但 unified source 通过 OR 合并绕过该 gate。这是 r31/r33 精神下必须修的语义旁路。
- 已执行旁路修复：`portfolio_simulator.py` 要求 unified source candidate 满足 `portfolio_daily_source_distribution_clean_pass`，或同时满足高 spread、低 positive penalty、低 opportunity cost、低 brake/proxy/economic block 的严格 relief 条件；新增 `test_unified_source_head_cannot_bypass_source_distribution_gate` 锁定该契约。
- 已执行 target 加强：`allocation_optimizer.py` 将 strong positive forward penalty 作为一等惩罚，降低 positive source 的 spread reward 对 source score/objective 的抵消能力，避免 `receiver_source_spread_reward` 单独压过正向误卖风险。
- r36c 验证：前台运行 `cp_v3_portfolio_daily_risk_aware_unified_allocation_r36c_source_distribution_gate_screening64_20260501`，训练 diagnostics 显示 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`，任务自然完成，`training_evidence_status = sufficient`。
- r36c 关键结果：`annual_return = 0.452007`、`sharpe = 1.411792`、`max_drawdown = -0.166775`、`monthly_return_mean = 0.034079`、`receiver_target_count = 6`、`source_target_count = 3`、`source_realized_sell_rate = 1.0`、`receiver_unrealized_deploy_share = 0`、`cash_reserve_rate = 0.797578`。promotion gate 仍为 `shadow_only`，失败项为 `reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`。
- r36c source 结论：执行层旁路已封住，source target 行全部 `portfolio_daily_source_distribution_clean_pass = true`；但真实未来分布仍失败，`source_positive_forward_sell_share = 0.666667`、`source_strong_positive_forward_sell_count = 2`、`source_max_forward_excess_5d = 0.128745`、`receiver_minus_source_forward_excess_5d = -0.029367`。这说明剩余根因是模型低估 source 正向前景/机会成本，而不是 simulator OR 旁路。
- 决策：r36 继续 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact；在 source distribution、cash timing、drawdown 与 reduce/exit gate 同时过线前，不运行 bounded confirm 作为正式候选结论。

## 2026-05-01 r37 decision-focused allocation 执行复盘
- 行动前判断：r36 已封住 unified source 绕过 source distribution gate 的执行层旁路，但 r36c 仍卖出真实正 forward source。最高优先级因此不是继续堆 simulator guard，而是把 source hard-negative、strong false sell、opportunity cost 与 allocation regret 写回训练目标，并确认推理侧实际消费这些预测信号。
- 已执行实现：`allocation_optimizer.py` 新增 `portfolio_daily_source_strong_false_sell_penalty` 与 `portfolio_daily_source_hard_negative_penalty`，并让 future label 与模型预测的 positive-forward / opportunity-cost penalty 都能进入 unified source score、candidate、summary 与 objective；`model_seq_v3.py` 新增 `alpha_result_value_budget_split_v23` 与 `_decision_focused_allocation_regret_loss`；`run_self_optimizing_study.py` 新增 `split_heads_portfolio_daily_decision_focused_allocation_r37`，并把 hard-negative penalty 写入 scoring。
- 已执行审计修复：`analyze_behavior_gap.py` 新增 `_compute_exposure_utilization_from_turnover`，当旧 turnover 文件的 `gross_exposure` 全零或缺失时回退到 `1 - cash_weight`，避免把高实际暴露误读成 0。
- 已执行测试：新增 source hard-negative、预测 penalty 推理消费、decision-focused allocation regret、r37 profile 注册和 exposure utilization 兼容测试；`daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts` 当前 24 项通过。
- r37 初始 screening：`cp_v3_portfolio_daily_decision_focused_allocation_r37_screening64_20260501` 使用 yolos + CUDA 前台自然完成，`training_evidence_status = sufficient`；但 `promotion_gate.status = shadow_only`，失败项为 `open_win_rate_5d`、`reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`。同时 `source_positive_forward_sell_share = 0.833333`、`source_strong_positive_forward_sell_count = 2`、`receiver_minus_source_forward_excess_5d = -0.057937`。
- 关键工程发现：初始 r37 后确认训练目标已存在，但推理 allocation path 在没有未来标签时没有充分消费模型预测 penalty heads；这会让 source hard-negative 在真实推理时弱化。已新增回归测试并修复。
- r37b 复核：`cp_v3_portfolio_daily_decision_focused_allocation_r37b_screening64_20260501` 再次使用 yolos + CUDA 前台自然完成，`training_evidence_status = sufficient`、`receiver_unrealized_deploy_share = 0.0`、`source_realized_sell_rate = 1.0`、`cash_reserve_rate = 0.718153`；但核心失败几乎未变，`source_positive_forward_sell_share = 0.833333`、`source_strong_positive_forward_sell_count = 2`、`receiver_minus_source_forward_excess_5d = -0.057937`。
- 本质结论：r37 已把 source hard-negative 路径接通，残余瓶颈不是字段未接、脚本中止、GPU/yolos 或 receiver 可执行性，而是模型学出的 positive-forward / opportunity-cost penalty 强度不足。下一轮应强化 source hard-negative 样本权重、tail distribution loss、pairwise source regret 与 optimizer 约束，而不是继续放宽 source 或追加后置 guard。
- 决策：r37 继续 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact；在 source positive distribution、cash timing、drawdown、reduce/exit 与 open quality 同时过线前，不进入 bounded confirm 或 promotion 讨论。

## 2026-05-01 r38 source hard-negative regret bounded confirm 执行复盘
- 行动前判断：r37 已接通 source hard-negative 与 decision-focused allocation loss，但 source positive forward sell 仍明显失控。最高优先级不是继续补 simulator guard，而是把 source hard-negative 尾部样本、release preference 和 transfer-level allocation regret 写进训练目标，让“能卖”进一步变成“卖得对”。
- 已执行实现：`allocation_optimizer.py` 新增 `portfolio_daily_source_tail_false_sell_penalty`、`portfolio_daily_source_release_preference` 与 `portfolio_daily_transfer_regret_target`；`model_seq_v3.py` 新增 `alpha_result_value_budget_split_v24`、对应 heads / sample targets / inference exports，并接入 `_source_hard_negative_tail_loss`、`_source_listwise_release_regret_loss` 与 `_transfer_level_allocation_regret_loss`；`run_self_optimizing_study.py` 新增 `split_heads_portfolio_daily_source_hard_negative_regret_r38`，并把 hard-negative / transfer regret 指标写入 scoring 与 primary metrics。
- 已执行测试：先按 TDD 让缺失 `_source_hard_negative_tail_loss` 的单测红灯失败，再补齐实现；`daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts` 当前 29 项通过，相关文件 `py_compile` 通过。
- bounded confirm 执行：用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`、`KMP_DUPLICATE_LIB_OK=TRUE`、`PYTHONUTF8=1` 前台运行 `self_opt_study_r38_source_hard_negative_regret_20260501`，完成 1 个 screening 与 1 个 fresh confirm，任务自然结束，耗时约 2 小时 27 分钟。
- GPU/yolos 证据：screening 与 confirm 的 `training_diagnostics.json` 均显示 `device = cuda`、`cuda_available = true`、`python_executable = C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`runtime_env = yolos`、`completed_epochs = 64`、`loss_profile = alpha_result_value_budget_split_v24`，且 r38 support flags 全为 true。
- 关键结果：confirm 为 `training_evidence_status = sufficient`，`source_positive_forward_sell_share = 0.0`、`source_strong_positive_forward_sell_count = 0`、`source_forward_excess_5d = -0.026266`、`receiver_forward_excess_5d = 0.011082`、`receiver_minus_source_forward_excess_5d = 0.037347`、`source_realized_sell_rate = 1.0`、`receiver_unrealized_deploy_share = 0.0`、`cash_reserve_rate = 0.981728`。
- 未通过原因：`promotion_gate.status = shadow_only`，v2 gate 只过 `8/12`，失败项为 `reduce_success_rate_5d`、`exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`；`portfolio_daily_v2_stable_confirmatory_trials` 为空，stability 失败项含 `confirm_gate_pass`、`confirm_annual_return_floor`、`confirm_sharpe_floor` 与 `confirm_source_count_floor`。经济质量很弱：`annual_return = 0.009111`、`sharpe = 0.164855`、`monthly_return_mean = 0.006829`、`source_target_count = 1`。
- 行动后复盘：r38 证明 source hard-negative tail / release / transfer regret 能把强势误卖压住，但也暴露单边防错会导致过度保守、source 广度不足和收益弱。下一轮不应放松 source 防错来换收益，而应在同一 allocation objective 中恢复 source/receiver breadth、现金部署时机、open/reduce/exit 质量、monthly return 与 drawdown。
- 决策：r38 继续 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。`study_summary.json` 与 `protocol_summary.json` 需要显式 UTF-8 读取；PowerShell 默认编码显示乱码不构成 JSON 或 brain 损坏证据。

## 2026-05-02 r39 allocation objective consolidation 执行复盘
- 行动前判断：r38 已把强势 source 误卖压住，但系统退化为收益弱、source 数量窄和现金过度保守；继续追加局部 penalty/guard 会回到“卖错”和“太保守”的跷跷板。因此本轮不命名为局部 r39 修补，而是做 `split_heads_portfolio_daily_allocation_objective_consolidation_r39`，目标是把收益、回撤、现金、换手、source opportunity cost、receiver realized deploy、monthly consistency 收敛为统一 allocation objective。
- 已执行审计：确认 r35-r38 已有 unified allocation surface，但 rollout 仍大量沿用旧 action head / direct-action / simulator guard 路径；`portfolio_daily_unified_*` 进入了训练，`solve_semidifferentiable_allocation` 也存在，但 final allocation objective 没有完全接管执行评分。
- 已执行实现：`allocation_optimizer.py` 新增 `portfolio_daily_allocation_trade_quality_target`、`portfolio_daily_allocation_cash_deployment_target`、`portfolio_daily_allocation_risk_adjusted_return_target`、`portfolio_daily_allocation_drawdown_control_target`、`portfolio_daily_allocation_monthly_quality_target` 与 `portfolio_daily_allocation_final_objective`；`solve_semidifferentiable_allocation` 改为用 final objective / trade quality / risk-adjusted return 参与 source/receiver 分配评分。
- 已执行模型接入：`model_seq_v3.py` 新增 `alpha_result_value_budget_split_v25`，将 `action_total` 降为辅助权重 `0.06`，新增 r39 heads、sample targets、`_allocation_objective_consolidation_loss` 与 support flags；推理阶段新增 execution blend，当 r39 heads 可用时把 final objective 回写到 unified receiver/source/cash 与核心 receiver/source/cash score。
- 已执行 study 接入：`run_self_optimizing_study.py` 新增 `split_heads_portfolio_daily_allocation_objective_consolidation_r39`，默认 objective 仍为 `portfolio_daily_ranking_v2_gated`，calibration 仍保留 `cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15` 作为最后安全层。
- 已执行测试：按 TDD 先让缺失 `_allocation_objective_consolidation_loss` / v25 profile 的测试红灯，再补齐实现；最终 `daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts` 33 项通过，`compileall` 通过。CUDA 环境检查为 `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`、`cuda_available = True`、GPU 为 `NVIDIA GeForce RTX 2060`。
- 第一轮 bounded confirm：`self_opt_study_r39_allocation_objective_consolidation_20260502` 前台自然完成，1 个 screening + 1 个 fresh confirm，`training_evidence_status = sufficient`，但执行评分尚未回写 final objective，confirm v2 gate 只过 `9/12`，失败项为 `cash_timing_quality_1d`、`hold_share`、`max_drawdown`，stable confirm 失败于 `confirm_gate_pass` 与 `confirm_source_count_floor`。
- 第二轮 execblend bounded confirm：`self_opt_study_r39_allocation_objective_consolidation_execblend_20260502` 前台自然完成，1 个 screening + 1 个 fresh confirm，训练 diagnostics 为 yolos + CUDA，任务后检查无残留 Python 进程。confirm 关键指标：`training_evidence_status = sufficient`、`annual_return = 2.676289`、`sharpe = 3.802282`、`monthly_return_mean = 0.115004`、`receiver_target_count = 6`、`receiver_unrealized_deploy_share = 0.0`、`source_realized_sell_rate = 1.0`、`source_positive_forward_sell_share = 0.0`、`source_strong_positive_forward_sell_count = 0`、`receiver_minus_source_forward_excess_5d = 0.066883`。
- 未通过原因：execblend confirm 仍为 `promotion_status = shadow_only`，v2 gate 只过 `9/12`，失败项为 `exit_timeliness_rate_5d`、`cash_timing_quality_1d`、`max_drawdown`；stable confirm 失败项为 `confirm_gate_pass` 与 `confirm_source_count_floor`。主要残余指标为 `source_target_count = 1`、`cash_reserve_rate = 0.947020`、`cash_timing_quality_1d = -0.138133`、`max_drawdown = -0.109069`。
- 本质结论：r39 证明“统一 allocation objective + execution blend”比 r38 单边防错更接近正路，能恢复收益、receiver 广度和正 receiver-source spread，同时继续压住 source positive false sell；但它还没有完成真正稳定的资金来源分布和现金时机控制。下一轮最有效方向不是回到局部 guard，而是在 r39 基础上提高 clean source breadth、让 cash timing / drawdown 的日级触发进入 allocation layer，并降低过高 cash reserve。
- 决策：r39 继续 `research / shadow_only`，不得 promotion、不得 live、不得改 active artifact。

## 2026-05-02 continuous_policy 主线循环审计与停环
- 触发：用户要求先检阅本主线自成立以来所有研究、方案、代码、讨论和失败记录，判断是否一直在相同问题之间兜圈，并在发现循环后停止旧思路、重组任务。
- 执行：新增 `daily_research/tools/continuous_policy_cycle_audit.py`，用 yolos 前台运行生成 `daily_research/output/continuous_policy/analysis/cycle_audits/continuous_policy_cycle_audit_20260502.md` 与 `.json`。
- 结论：存在结构性重复循环。反复问题为 source 卖错、receiver 买不了或买不宽、cash 死分支或过度保守、收益/回撤/月度质量不过线、confirm 不稳定、旧 action translation / simulator guard 主路径残留。
- 停止路线：不再把 r39 后续包装成新增 source/cash/reduce 局部 penalty 或 guard；不再用单项 gate 清零、短窗 smoke 高分或 insufficient evidence confirm 作为阶段成功；不再让 simulator guard 承担主策略逻辑。
- 新主线：冻结 r20-r23、r31、r33 为最后安全边界；下一阶段应命名并执行为 `end-to-end allocation layer`，让 allocation objective / optimizer 同时决定 source、receiver、cash、成本、换手、仓位上限、回撤和月度质量。
