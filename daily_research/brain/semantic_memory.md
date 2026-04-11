# Daily Research 稳定语义

快照日期：`2026-04-12`

## 1. 项目身份
- `daily_research` 维护一条面向主板 A 股、以执行后净收益最大化为主目标的研究-执行统一链路。
- 当前项目级接管原则已升级为：
  - `Agent 无状态，项目大脑有状态。`
- 历史过程与时序证据写入 `episodic_memory.md`。
- 当前判决写入 `working_memory.md`。
- 日常入口与固定命令写入 `action_system.md`。

## 2. 固定边界
- 市场范围固定为主板 A 股：
  - 上证 A 股
  - 深证 A 股
  - 默认剔除创业板、科创板、ST
- 成交假设固定为 `next_open`。
- 正式研究、训练、回测、执行默认解释器固定为 `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`。
- brain 文档默认使用简体中文。
- shell 运行时输出、终端日志、进度条文本默认使用英文。
- 日常生成计划不允许无条件静默重训模型；默认 production 仅按 `Retrain Monthly` 自动重训。

## 3. 双模型协议
- formal 验证采用滚动窗口协议。
- 每个 formal 窗口都必须使用该窗口起点前最新可标注数据训练当时最新模型，不允许拿更旧的冻结模型替代。
- formal 单窗的评估段仍保持独立 holdout，不允许把该窗评估期并回该窗训练集。
- recent 验证是研究闭环必备伴随证据，不能只报 formal 不报 recent。
- `production full-fit` 的职责是把当前默认执行物化成最新 live panel 和默认执行，不是替代 formal / recent 研究裁决。
- `formal = 滚动 formal 评估窗`。
- `recent = 截至当前评估时点的最近一年 12 个月窗口`。
- `live = 当前生产执行语义与面板`。

## 4. 研究与执行统一目标
- 当前统一目标不是“raw holdout 指标最大”，而是“执行后净收益最大”。
- 默认研究口径固定为：
  - `research_objective_mode = execution_first`
  - `execution_alignment_mode = train_eval_auto`
  - `execution_alignment_objective = robust_composite`
  - realistic cost = `3 / 7 / 10 bps`
- execution policy 本身属于研究对象，不再是固定后置适配壳。
- formal + recent 一起负责研究判决。
- production full-fit 负责把研究 winner 物化为默认执行。
- production full-fit 结果不得回填为 formal 研究证据。
- 在用户已明确改写目标的前提下，当前默认裁决顺序为“月度收益优先、模型偏短线”；年化与 Sharpe 保留为辅助指标。

## 5. 当前默认执行语义
- 当前 active execution strategy 为 `deep_alpha_short_alpha_execalign_production_default`。
- 当前 active candidate label 为 `short_expert_monthly_v1__regoff_k2_5d_ensemble_native_anchor__active`。
- 当前 active manifest 真源为 `daily_research/output/active_execution_strategy.json`。
- 当前默认执行入口为 `daily_research/execution/run_trade_plan.py`。
- 当前默认 production root 为 `daily_research/output/deep_alpha_short_alpha_execalign_production_default`。
- 当前统一权重语义为 `research_raw_target_weight`。
- 当前统一上限语义为 `follow_research_raw_no_global_cap`。
- 当前默认 target-weight 来源为 production root 下的 `execution_aligned_daily_live_target_weight_panel.csv`。
- 当前 live 默认执行 profile 为 `regoff_k2_5d_ensemble_native_anchor`。
- 当前 static fallback 仍保留在 production root 里作为安全基线，但不再是默认 target-weight 来源。
- active manifest 必须显式保存当前 live 执行态，包括：
  - `effective_live_target_weight_mode`
  - `effective_live_execution_profile`
  - `effective_live_execution_bridge_meta`
  - `effective_live_weight_generation_note`
  - `monthly_first_*` 裁决字段

## 6. 当前稳定研究与执行结论
- 在 `liquid500 + execution_first + formal 3 windows + primary_monthly_robust_score + window_count=3` strongest-model gate 下，`short_expert_monthly_v1` 是当前 strongest research model。
- `baseline_current` 是当前 strongest-model recent 一年 `12` 个月窗口里的 recent winner，用于回答最近一年兑现质量。
- `state_liquidity_listwise_v1` 仍是当前 short-alpha 主线最强 stable base model。
- 按当前新协议，研究最强模型默认可以直接作为执行默认，不再额外设置独立 promotion 哲学阻塞层。
- 但最终写入默认执行的产物，仍必须先用最新可标注数据做一次 `production full-fit`，并默认使用该 family 当前最高预算；不允许为了省算力沿用旧模型、低预算 probe 或未重训产物。
- 这一步已经在 `2026-04-09` 物化到 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor` 默认执行。
- execution-side 当前稳定执行桥主线仍是 `raw + regoff_k2_5d_ensemble_native_anchor`。
- `trend_up_low_vol|expand|stable -> topk3_1d_regoff` 只保留为历史 targeted repair 候选与观察分支，不再作为当前默认 repair 叙事。
- broad execution-policy sweep 已停止；后续执行侧升级默认只沿 `month-start / first-week / signal-to-weight / month-trigger / execution` 下钻。
- 任何 simple regime-conditioned repair，只有在同窗、同成本、同协议、月度优先口径下打赢当前 `k2` 静态基线后，才允许再次进入 promotion 讨论。
- `dynamic_graph_no_priors` 是当前 rolling liquid800 / mainboard monthly execution-first formal winner。
- 当前默认不再把 industry/style priors 当作稳定增益。
- `dynamic_graph_no_priors` 已补 liquid500 同宇宙 challenger formal，但当前仍不超过 short-alpha 主线，因此不是当前 active-default candidate。
- `structure_context_only` 保留为 raw 架构 challenger，不是当前 execution-upgrade 答案。
- `encoder_transformer_v1` 仍是 high-upside but unstable 的主要 architecture 候选。
- `graph_off_plain` 已做预算补齐复核，但仍未通过 liquid500 challenger gate，只保留为 monitored architecture branch。

## 7. 结果解释边界
- short-alpha 默认执行 winner 与 liquid800 / mainboard 研究 winner 仍需分开叙述。
- 不同股票池、不同成本口径、不同 gate 协议下的结果，不得直接混成单一“全项目最高”结论。
- 不得拿 `production full-fit` 的最新训练截止去反推 formal 协议被破坏。
- 不得把 recent/live 回放结果直接表述成 formal 研究证据。
- 如果 recent/live 变弱，默认先查市场状态、execution bridge 和 live 约束，不允许先把原因偷换成“研究训练数据不够新”。
- 如果用户明确要“当前全项目最高净收益”，必须先进入同一成本引擎、同一 execution-policy audit、同一 `global_deployable_non_capacity_adjusted_v1`，然后再做跨 universe 排名。

## 8. 文档分工语义
- `identity_layer.md` 只保留项目使命、北极星、硬约束与禁区。
- `handoff_packet.md` 只保留标准交接包与当前最小完备状态。
- `semantic_memory.md` 只保留稳定事实与长期边界。
- `rule_memory.md` 只保留高优先级规则。
- `lesson_memory.md` 只保留可复用教训。
- `temporal_state.md` 只保留 `Past Ledger / Present State / Future Map`。
- `project_map.md` 只保留项目结构、主线地图与决策闭环。
- `working_memory.md` 只保留当前判决、优先级与下一步。
- `procedural_memory.md` 只保留可复用方法学规则。
- `governance_layer.md` 只保留自检、反偏移、修复和写回闭环。
- `environment_model.md` 只保留环境、解释器、工具与编码口径。
- `action_system.md` 只保留高频操作入口。
- `episodic_memory.md` 只保留历史过程与原始证据。

## 9. 当前协议补充
- strongest-model 的 recent 层现在只承认 `independent_recent_model_as_of_recent_start`，不再复用 formal run_dir 做 replay recent。
- 当前 strongest research model 仍是 `short_expert_monthly_v1`。
- 当前 strongest-model 的 recent winner 是 `baseline_current`，对应 recent 一年 `monthly_robust_score = 0.0982`。
- 当前 strongest-model 的 promotable winner 仍是 `short_expert_monthly_v1`；formal / recent / promotable 三层语义必须分开叙述。
- `state_liquidity_listwise_v1` 仍是当前 short-alpha 主线最强 stable base model，但不再表述成 strongest-model 的 recent winner。
- learned-control 分支里，当前要分两层讲：
  - constrained formal front-runner 已前移到 `short_expert_policy_v2b__k1_20d = 0.1104`
  - corrected recent winner 已前移到 `short_expert_policy_v4b = 0.1387`
- `short_expert_policy_v4b` 也是当前 direct formal family 的 fresh best，formal `0.1008` 接近 mainline `0.1012`；但它的 constrained best 只有 `0.0687`，还不能直接讲成 promotable winner。
- `short_expert_policy_v2c = 0.1046` 与 `short_expert_policy_v2b = 0.1020` 仍是上一轮 `policy_v2 family` 的 recent 前沿；`policy_v3` 目前没有打赢这些 learned-control 前沿。
- 当前 learned-control 的主问题已经从“补 `v2b / v2c` constrained formal 缺口”切到“如何保住 `policy_v4b` 的 recent 强度并把 deployable constrained formal 拉回 `policy_v2b` 水平以上”。
- 旧 replay-based 机制根统一由 `daily_research/archive/output/replay_based_reference_index.md` 管理；主脑正文不再散落直引这些路径。
- 本机正式训练纪律补充为：`yolos` + 前台执行 + `num_workers = 0` + `pin_memory = false`；未经用户明确允许，不重新启用 CPU 并行供数。
