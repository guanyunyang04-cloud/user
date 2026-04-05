# Daily Research 稳定语义

快照日期：`2026-04-05`

## 1. 项目身份
- `daily_research` 维护一条可运行、可复核、可回退的 A 股日频研究与执行链路。
- 历史实验与时间序列证据写入 `episodic_memory.md`。
- 当前判断写入 `working_memory.md`。
- 日常命令与固定入口写入 `action_system.md`。

## 2. 固定边界
- 市场范围固定为主板 A 股：
  - 上证 A 股
  - 深证 A 股
  - 默认剔除创业板、科创板、ST
- 成交假设固定为 `next_open`。
- 默认研究与执行解释器固定为 `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`。
- 日常生成计划不允许无条件静默重训模型；默认 production 候选仅按 `Retrain Monthly` 规则自动重训。

## 3. 研究与执行统一目标
- 当前统一目标不是“raw holdout 指标最大”，而是“执行后净收益最大”。
- 当前默认研究口径固定为：
  - `research_objective_mode = execution_first`
  - `execution_alignment_mode = train_eval_auto`
  - `execution_alignment_objective = robust_composite`
  - realistic cost = `3 / 7 / 10 bps`
- formal holdout 负责研究判决。
- production full-fit 负责默认执行。
- production full-fit 结果不得回填为 formal 研究证据。

## 4. 当前默认执行语义
- 当前 active execution strategy 为：
  - `state_liquidity_listwise_v1_execfirst_winner`
- 当前 active manifest 真源为：
  - `daily_research/output/active_execution_strategy.json`
- 当前默认执行入口为：
  - `daily_research/execution/run_trade_plan.py`
- 当前默认 production root 为：
  - `daily_research/output/deep_alpha_short_alpha_execalign_production_default`
- 当前默认 execution profile 为：
  - `regoff_k2_10d_ensemble_native_anchor`

## 5. 当前稳定研究结论
- `structure_context_only` 的稳定语义：
  - 它仍可作为 raw 架构挑战者保留。
  - 它不是当前 execution-upgrade 答案。
- `state_liquidity_listwise_v1` 的稳定语义：
  - 它已通过 budget-normalized monthly execution-first formal H2H。
  - 它已通过 recent realistic replay gate。
  - 它已通过 production full-fit replay gate。
  - 它是当前 liquid500 active default。
  - 它当前仍有 production recipe 的预算压力，下一步应继续做 epoch extension，而不是回退到旧 baseline。
- `dynamic_graph_no_priors` 的稳定语义：
  - 它是当前 rolling liquid800 / mainboard monthly execution-first formal winner。
  - 当前默认不再把 industry/style priors 当作稳定增益。

## 6. 家族级训练预算语义
- family epoch budget 的唯一真源为：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- 当前冻结预算为：
  - `baseline -> 4`
  - `structure -> 12`
  - `short_alpha -> 24`
  - `dynamic_graph -> 16`
- formal rich experiment 默认读取这份 manifest，而不是手写统一 `--epochs`。

## 7. 月度协议语义
- `deep_alpha` 当前主研究时间单位为 `calendar_months`。
- 月度协议作用于：
  - train/valid 切窗
  - train-side eval window
  - adaptive task window
  - monthly summary artifacts
- 月度分析是当前研究判读的第一视角：
  - 先看 `primary_research_monthly_diagnostics`
  - 再看 `Monthly Priority Summary`
  - 最后才看整窗 mean annual / Sharpe
- 若已存在更晚的 budget-normalized monthly landscape review，则更早的 `monthly_r1` 结论只保留为阶段性诊断，不再作为当前排序结论。
- 当前月度重点字段固定包括：
  - positive-month ratio
  - median monthly excess return
  - worst monthly excess return
  - top3 positive-month share
  - longest negative streak
- 月度协议不改变底层数据频率；底层仍然是日频样本与 `next_open` 回测。

## 8. 文档分工语义
- `semantic_memory.md` 只保留稳定事实与当前长期有效边界。
- `working_memory.md` 只保留当前判决、优先级与下一步。
- `action_system.md` 只保留高频操作入口。
- `procedural_memory.md` 只保留可复用方法学规则。
- `project_map.md` 只保留项目脉络与主线地图。
