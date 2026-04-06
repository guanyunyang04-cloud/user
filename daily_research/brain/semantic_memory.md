# Daily Research 稳定语义

快照日期：`2026-04-06`

## 1. 项目身份
- `daily_research` 维护一条面向主板 A 股、以执行后净收益最大为唯一主目标的研究-执行统一链路。
- 历史过程与时序证据写入 `episodic_memory.md`。
- 当前判决写入 `working_memory.md`。
- 日常入口与固定命令写入 `action_system.md`。

## 2. 固定边界
- 市场范围固定为主板 A 股：
  - 上证 A 股
  - 深证 A 股
  - 默认剔除创业板、科创板、ST
- 成交假设固定为 `next_open`。
- 正式研究、训练、回测、执行默认解释器固定为：
  - `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`
- brain 文档默认使用简体中文。
- shell 运行时输出、终端日志、进度条文本默认使用英文。
- 日常生成计划不允许无条件静默重训模型；默认 production 仅按 `Retrain Monthly` 自动重训。

## 3. 研究与执行统一目标
- 当前统一目标不是“raw holdout 指标最大”，而是“执行后净收益最大”。
- 默认研究口径固定为：
  - `research_objective_mode = execution_first`
  - `execution_alignment_mode = train_eval_auto`
  - `execution_alignment_objective = robust_composite`
  - realistic cost = `3 / 7 / 10 bps`
- execution policy 本身属于研究对象，不再是固定后置适配壳。
- formal holdout 负责研究判决。
- production full-fit 负责默认执行。
- production full-fit 结果不得回填为 formal 研究证据。

## 4. 当前默认执行语义
- 当前 active execution strategy 为：
  - `state_liquidity_listwise_v1_execfirst_profitmax_global_winner`
- 当前 active manifest 真源为：
  - `daily_research/output/active_execution_strategy.json`
- 当前默认执行入口为：
  - `daily_research/execution/run_trade_plan.py`
- 当前默认 production root 为：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- 当前默认 execution policy 为：
  - `regoff_k1_5d_ensemble_native_anchor`
- 当前 active default 通过 `panel_mode = raw` + 精确 bridge spec 执行，不依赖预先导出的 `execution_aligned` panel。
- active manifest 显式保存：
  - `liquidity_pool_name`
  - `liquidity_pool_size`
  - `execution_policy_label`
  - `execution_alignment_selected_profile_spec`
- 当前“全项目最高”的唯一正式口径为：
  - `global_deployable_non_capacity_adjusted_v1`
  - 即同一成本引擎、同一 execution-policy audit 搜索空间下的跨 universe deployable leaderboard

## 5. 当前稳定研究结论
- `state_liquidity_listwise_v1` 是当前 liquid500 active default。
- liquid500 short-alpha 主线默认 checkpoint objective 仍是：
  - `primary_annual_return`
- `primary_monthly_robust_score` 当前只保留为 fresh-run challenger objective，不作为 liquid500 主线默认值。
- `weak_month_repair_v1` 扩展静态 score-to-weight / bridge 搜索没有翻掉：
  - `regoff_k1_5d_ensemble_native_anchor`
  - 因此 liquid500 当前剩余修复方向是 targeted weak-month repair，而不是继续扩大静态桥接集合。
- simple regime-conditioned execution policy 当前不成立：
  - leave-window-out formal review 对静态 `regoff_k1_5d_ensemble_native_anchor` 为 `0/3` 全败。
- 显式按 `regoff_k1_5d_ensemble_native_anchor` 做的 fresh production refresh 当前不成立：
  - 同一 policy 下打不赢当前 production root。
- `dynamic_graph_no_priors` 是当前 rolling liquid800 / mainboard monthly execution-first formal winner。
- 当前默认不再把 industry/style priors 当作稳定增益。
- `dynamic_graph_no_priors` 已补 liquid500 同宇宙 challenger formal，但当前仍不超过 liquid500 short-alpha 主线，因此不是 liquid500 active-default candidate。
- `structure_context_only` 保留为 raw 架构 challenger，不是当前 execution-upgrade 答案。
- `encoder_transformer_v1` 是当前 high-upside but unstable 的主要 architecture 候选。
- `graph_off_plain` 已做预算补齐复核，但仍未通过 liquid500 challenger gate，只保留为 monitored architecture branch。

## 6. 训练预算与月度协议语义
- family epoch budget 的唯一真源为：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- 当前冻结预算为：
  - `baseline -> 4`
  - `structure -> 12`
  - `short_alpha -> 24`
  - `dynamic_graph -> 16`
- `deep_alpha` 当前主研究时间单位为 `calendar_months`。
- 月度协议作用于：
  - train/valid 切窗
  - train-side eval window
  - adaptive task window
  - monthly summary artifacts
- 月度分析是当前研究判读的第一视角。
- 单次 run 现在会额外落盘：
  - `primary_research_monthly_objectives.json`

## 7. 结果解释边界
- liquid500 默认执行 winner 与 liquid800 / mainboard 研究 winner 仍需分开叙述。
- 不同股票池、不同成本口径、不同 gate 协议下的结果，不得直接混成单一“全项目最高”结论。
- 如果用户明确要“当前全项目最高净收益”，必须先进入：
  - 同一成本引擎
  - 同一 execution-policy audit
  - 同一 `global_deployable_non_capacity_adjusted_v1`
  然后再做跨 universe 排名。

## 8. 文档分工语义
- `semantic_memory.md` 只保留稳定事实与长期边界。
- `project_map.md` 只保留项目结构、主线地图与决策闭环。
- `working_memory.md` 只保留当前判决、优先级与下一步。
- `procedural_memory.md` 只保留可复用方法学规则。
- `environment_model.md` 只保留环境、解释器、工具与编码口径。
- `action_system.md` 只保留高频操作入口。
