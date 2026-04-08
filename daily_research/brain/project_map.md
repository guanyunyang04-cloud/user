# Daily Research 项目地图

快照日期：`2026-04-08`

## 1. 项目定义
- `daily_research` 是一条面向 A 股、以执行后净收益最大化为唯一主目标的研究-执行统一链路。
- 当前默认主战场是 `liquid500`，默认研究目标是 `execution_first`，默认日常执行真源是 `daily_research/output/active_execution_strategy.json`。
- 当前研究/执行统一语义是：
  - `research_raw_target_weight`
  - `follow_research_raw_no_global_cap`
  - 也就是研究怎么给 raw target weight，执行就怎么落，不再在 external target-weight 路径上另套一层通用仓位上限
- 项目治理上，用户最新要求拥有最高优先级；任何新要求一旦改变当前口径，必须同步收口到代码、入口、manifest 和 brain 文档。

## 2. 立项以来的主线演进
1. 基础设施期
- 统一了研究、训练、回测、执行入口。
- 把默认执行从脚本隐式参数改成 manifest-driven。
- 把 family epoch budget、monthly-first 判读、execution policy audit 变成正式协议。

2. 主线确立期
- `state_liquidity_listwise_v1 + regoff_k1_5d_ensemble_native_anchor` 成为 liquid500 当前默认执行线。
- 这条线同时是当前全局 deployable winner。

3. 问题收敛期
- broad execution-policy sweep、扩静态 bridge/profile、simple regime-conditioned policy 都已验证无效或不优。
- 项目主问题收敛为 targeted weak-month repair，而不是继续横向扩 execution policy。

4. 当前优化期
- 执行侧已经把可解释的 first-week repair candidate 物化成 active default。
- 执行侧随后又补齐成“两层管线”：heavy research refresh 负责 formal 证据，live-only refresh 负责日常 trade-plan 更新。
- `2026-04-08` 已补做 latest-model production full-fit refresh，但保留 execution candidate 作为 active manifest。
- 模型侧已经证明“大 bundle 自动更强”不成立，后续只应做窄而有效的修复。

## 3. 当前项目结构
- `baseline`
  - 数据、回测、账户、交易计划、旧模型链路。
- `execution`
  - active strategy、production promotion、daily trade plan、股票池维护。
- `deep_alpha`
  - liquid500 主研究链、formal runner、execution audit、challenger review。
- `tools`
  - `doc_guard`、维护工具、辅助脚本。
- `brain`
  - 当前状态、方法、操作入口、历史实验记录。
- `cache / output / archive`
  - 缓存、产物、归档。

## 4. 当前正式主线
- active default
  - 模型：`state_liquidity_listwise_v1`
  - execution：`trend_up_low_vol|expand|stable -> topk3_1d_regoff` single-mapping candidate pipeline
  - universe：`liquid500`
  - panel mode：`raw`
  - source：`daily_research/output/active_execution_strategy.json`
- 全局 deployable 排名
  - rank 1：liquid500 当前主线
  - rank 2：`dynamic_graph_no_priors` on `liquid800 / mainboard`

## 5. 当前两条高 ROI 分支
- 执行侧
  - 当前最强 candidate：`trend_up_low_vol|expand|stable -> topk3_1d_regoff`
  - current pipeline root：`daily_research/output/short_alpha_execution_single_mapping_candidate_pipeline_20260408_r2`
  - formal full-period H2H：candidate `51.42% / 2.253` vs static `39.85% / 1.738`
  - trigger coverage：`3/36 = 8.33%`
  - triggered mean monthly delta：`+8.50%`
  - 最新 live 月份未触发，因此当前 active panel 等价于静态默认，但未来触发时会自动切到 candidate 逻辑
- 模型侧
  - 当前没有任何 narrow split 干净打赢 `state_liquidity_listwise_v1`
  - best restart point：`short_expert_penalty_only_monthly_v1`
  - strongest split challenger：`short_expert_penalty_only_light_monthly_v1`
  - `heavy 48 -> 64` strict resume 已证伪，不再继续

## 6. 当前明确停止的方向
- 不再重开 broad execution-policy sweep。
- 不再把 simple regime-conditioned policy 当作默认解法。
- 不再把 `short_expert_monthly_v2` 或 `heavy penalty` 当作当前主修复线。
- 不再用 stale model、低预算模型或 CPU 训练结果给 execution / monthly refresh 下结论。

## 7. 决策闭环
1. formal holdout
- 负责研究 winner 判定。

2. recent realistic gate
- 负责执行层可落地性判定。

3. production full-fit
- 负责默认执行候选物化。

4. global deployable leaderboard
- 负责跨 universe 的统一口径比较。

5. active strategy manifest
- 负责日常执行真源。

## 8. 下一阶段
1. 执行侧继续沿已转正的 first-week repair candidate 收敛。
- 重点不是找更多 policy，而是跟踪真实触发、积累 live 样本、验证触发稀疏性是否可接受，并确保月更后用最新模型重刷 candidate pipeline。

2. 模型侧如重新开线，只从 `penalty-only` 窄修复继续。
- 目标是修月度中位数，不是再堆更大的 bundle。

3. 月更与 production 链继续执行最严格训练纪律。
- 最新模型。
- 最高 family 预算。
- 默认起训 `32`，不足再 strict resume continuation。
- GPU only。
- 同模型扩预算必须 strict resume。
- 口径改动收尾必须通过 `daily_research/tools/project_consistency_check.py`。
