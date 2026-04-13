# Daily Research 稳定语义

快照日期：`2026-04-13`

## 1. 项目身份
- `daily_research` 维护一条面向主板 A 股、以执行后净收益最大化为主目标的研究-执行统一链路。
- 当前项目级接管原则是：
  - `Agent 无状态，项目大脑有状态。`
- 历史过程与原始证据写入 `episodic_memory.md`。
- 当前判决写入 `working_memory.md`。
- 高频入口与固定命令写入 `action_system.md`。

## 2. 固定边界
- 市场范围固定为主板 A 股：
  - 上证 A 股
  - 深证 A 股
  - 默认剔除创业板、科创板、ST
- 成交假设固定为 `next_open`。
- 正式研究、训练、回测、执行默认解释器固定为 `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`。
- brain 文档默认使用简体中文。
- shell 运行时输出与日志默认使用英文。

## 3. 双模型协议
- formal 验证采用滚动窗口协议。
- 每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型。
- formal 单窗评估段保持独立 holdout，不允许把该窗评估期并回训练集。
- recent 验证是研究闭环必备伴随证据，不能只报 formal 不报 recent。
- `production full-fit` 的职责是把当前默认执行物化成最新 live panel 和默认执行，不替代 formal / recent 研究裁决。
- `formal = 滚动 formal 评估窗`
- `recent = 截至当前评估时点的最近一年 12 个月窗口`
- `live = 当前生产执行语义与面板`

## 4. current recent 语义
- strongest-model 当前只承认 `independent_recent_model_as_of_recent_start`。
- 当前 recent root 是：
  - `short_alpha_recent_model_protocol_20260412_r1`
- 当前 requested recent cutoff 是：
  - `20260410`
- 当前 effective validation window 是：
  - `2025-04-11 -> 2026-03-31`
- `requested_recent_end_date` 与 `effective validation end` 必须分开叙述。

## 5. 当前默认执行语义
- 当前 active execution strategy 为 `deep_alpha_short_alpha_execalign_production_default`。
- 当前 active candidate label 为 `short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`。
- 当前 active manifest 真源为 `daily_research/output/active_execution_strategy.json`。
- 当前默认执行入口为 `daily_research/execution/run_trade_plan.py`。
- 当前默认 production root 为 `daily_research/output/deep_alpha_short_alpha_execalign_production_default`。
- 当前统一权重语义为 `research_raw_target_weight`。
- 当前统一上限语义为 `follow_research_raw_no_global_cap`。
- 当前 live 默认执行 profile 为 `regoff_k2_5d_ensemble_native_anchor`。
- 研究最强模型默认可以直接作为执行默认。
- 但最终写入默认执行的产物，仍必须先做 latest-data `production full-fit + highest family budget`。

## 6. strongest-model 当前稳定结论
- strongest-model 当前 formal winner 是 `short_expert_monthly_v1`。
- strongest-model 当前 recent winner 也是 `short_expert_monthly_v1`。
- strongest-model 当前 promotable winner 仍是 `short_expert_monthly_v1`。
- strongest-model 当前 recent readout：
  - `short_expert_monthly_v1 recent monthly_robust_score = 0.0633`
  - `recent_excess_annual_return = 72.32%`
- `state_liquidity_listwise_v1` 仍是当前 short-alpha 主线最强 stable base model。
- `baseline_current` 现在只保留为其他 family / recent 对照参考，不再表述成 strongest-model 的 current recent winner。

## 7. learned-control 当前稳定结论
- 当前 learned-control 最强 deployable candidate 已前移到：
  - `short_expert_policy_v5b__k1_20d = 0.1177`
- 当前 learned-control current recent winner 也已前移到：
  - `short_expert_policy_v5b = 0.1200`
- `short_expert_policy_v5a = 0.1112` 是当前 v5 recent runner-up。
- `short_expert_policy_v5a__k1_20d = 0.1018` 与 `short_expert_policy_v5c__k1_20d = 0.1010` 说明 v5 family 的 constrained formal 整体有效。
- 当前 active v5 family fresh formal best 是：
  - `short_expert_policy_v5b = 0.0839`
- historical cross-family learned-control fresh formal best 仍是：
  - `short_expert_policy_v4b = 0.1008`
- `policy_v5b bridge sensitivity audit` 已确认：
  - deployable anchor 仍是 `k1_20d = 0.1177`
  - fast bridge `k1_3d` constrained 只有 `0.0248`
  - external cap wrapper 对当前 `k1_5d` 变体是 no-op
- `policy_v5d / policy_v5e` successor 首轮没有取代 `policy_v5b`：
  - `policy_v5d = formal 0.0676 / constrained 0.0925 / recent 0.1071`
  - `policy_v5e = formal 0.0812 / constrained 0.0867 / recent 0.0934`
- 因此 learned-control 当前不能再用 `policy_v2b / policy_v4b` 作为默认标准答案。
- 因此 learned-control 当前也不能把 `policy_v5d / policy_v5e` 写成新的 current answer。

## 8. 结果解释边界
- short-alpha 默认执行 winner 与 learned-control research frontier 必须分开叙述。
- 不同股票池、不同成本口径、不同 gate 协议下的结果，不得直接混成单一“全项目最高”结论。
- 不得拿 `production full-fit` 的最新训练截止去反推 formal 协议被破坏。
- 不得把 recent/live 回放结果直接表述成 formal 研究证据。
- 如果用户明确要“当前全项目最高净收益”，必须先进入同一成本引擎、同一 execution-policy audit、同一 `global_deployable_non_capacity_adjusted_v1`，然后再做跨 universe 排名。

## 9. 文档分工语义
- `identity_layer.md` 只保留项目使命、北极星、硬约束与禁区。
- `handoff_packet.md` 只保留标准交接包与当前最小完备状态。
- `semantic_memory.md` 只保留稳定事实与长期边界。
- `rule_memory.md` 只保留高优先级规则。
- `lesson_memory.md` 只保留可复用教训。
- `temporal_state.md` 只保留 `Past Ledger / Present State / Future Map`。
- `project_map.md` 只保留项目结构、主线地图与决策闭环。
- `working_memory.md` 只保留当前判决、优先级与下一步。
- `procedural_memory.md` 只保留可复用方法学规则。
- `action_system.md` 只保留高频操作入口。
- `episodic_memory.md` 只保留历史过程与原始证据。
