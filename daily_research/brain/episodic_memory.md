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
