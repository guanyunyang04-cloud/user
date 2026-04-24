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
  - screening champion：`trial_04 = alpha_result_value_budget_split_v11 + result_value_v10`
    - `composite_score = 3.0082`
    - `annual_return = 0.9111`
    - `deploy_intent_realized_rate = 0.9130`
    - `release_translation_deploy_health_score = 0.4352`
    - `deploy_funding_release_consistent_share = 0.0`
    - `failure_mode = order_translation_drift`
  - confirm_01：`alpha_result_value_budget_split_v11 + result_value_v10`
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
  - screening champion：`trial_02 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 0.3631`、`sharpe = 1.1531`、`max_drawdown = -0.1201`、`monthly_return_mean = 0.0234`、`monthly_win_rate = 0.75`、`direct_action_value_mode_share = 1.0`、`direct_action_order_translation_conflict_rate = 0.2842`，`promotion_status = shadow_only`。
  - repaired champion：`confirm_01 = alpha_result_value_budget_split_v14 + result_value_v9`，`annual_return = 1.0904`、`sharpe = 2.8082`、`max_drawdown = -0.1326`、`monthly_return_mean = 0.0587`、`monthly_win_rate = 0.75`、`monthly_worst_return = -0.0797`、`monthly_consistency_score = 0.7722`、`direct_action_value_mode_share = 1.0`、`direct_action_value_gap_mean = 0.0351`。
  - r14 的主要失败项仍是 `reduce_success_rate_5d = 0.0`、`exit_timeliness_rate_5d = 0.25`、`cash_timing_quality_1d`、`max_drawdown` 与 `failure_mode = order_translation_drift`；`promotion_status` 仍为 `shadow_only`。
  - held-side detail：135 条 held-side sell 中 134 条来自 `deploy_funding_rebalance`，1 条来自 `model_release_signal`；104 条 ambiguous、18 条 protected-hold conflict、13 条 release-consistent，集中在 `002371.SZ / 000333.SZ / 002028.SZ`。
- 动作后复盘：
  - 事实：r14 直接动作值仲裁显著改善了 repaired confirm 的收益、Sharpe 与月度收益质量，也证明用户提出的“按未来收益风险直接学习日级动作”方向有实证价值。
  - 事实：direct action 已经进入推理主路径，但低边际占比仍高，且订单/预算翻译冲突与 held-side funding rebalance 过度依赖没有闭合。
  - 推断：当前主瓶颈已从“是否应该直接学习日级动作”转为“如何让订单、预算、release/funding 层保留 direct action intent 并承担真实退出责任”。
  - 决策：r14 保持 `shadow_only`；下一步应推进 r15 direct-action-preserving translation / release-funding repair，而不是把 r14 高收益 confirm 直接升级为 production 证据。
