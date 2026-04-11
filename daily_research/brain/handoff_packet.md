# Daily Research 标准交接包

快照日期：`2026-04-12`

## 1. 项目当前状态摘要
- 项目性质：
  - 当前工作区正式生产研究与执行主线。
- 当前 live 默认：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
- 当前 strongest-model 三层答案：
  - formal winner = `short_expert_monthly_v1`
  - recent winner = `baseline_current`
  - promotable winner = `short_expert_monthly_v1`
- 当前 learned-control 三层答案：
  - constrained formal front-runner = `short_expert_policy_v2b__k1_20d = 0.1104`
  - corrected recent winner = `short_expert_policy_v4b = 0.1387`
  - fresh formal family best = `short_expert_policy_v4b = 0.1008`

## 2. 当前正在处理的问题
- 当前真正的主问题不是“谁是 recent 最大值”，而是：
  - 如何保住 `policy_v4b` 的 recent 强度，同时把 constrained formal 拉回 `policy_v2b` 水平以上。

## 3. 已完成动作及证据
- `policy_v2 family` constrained formal 已补齐：
  - [summary.json](H:/new_tdx64/PYPlugins/user/daily_research/output/short_alpha_policy_v2_family_constrained_execution_review_20260411_r2/summary.json)
- `policy family` formal loss breakdown 已补齐：
  - [summary.md](H:/new_tdx64/PYPlugins/user/daily_research/output/short_alpha_policy_family_formal_loss_breakdown_20260411_r1/summary.md)
- `policy_v4 family` 第一轮 formal / constrained / recent 已跑完：
  - [formal summary](H:/new_tdx64/PYPlugins/user/daily_research/output/short_alpha_policy_v4_family_formal_review_20260411_r1/summary.json)
  - [constrained summary](H:/new_tdx64/PYPlugins/user/daily_research/output/short_alpha_policy_v4_family_constrained_execution_review_20260411_r1/summary.json)
  - [recent summary](H:/new_tdx64/PYPlugins/user/daily_research/output/short_alpha_policy_v4_family_recent_eval_20260411_r1/summary.json)

## 4. 未完成动作
- 还没有一条新 learned-control 分支同时兼顾：
  - `v4b` 的 recent 强度
  - `v2b` 的 constrained deployability
- 下一轮窄迭代还未开始。

## 5. 已知风险
- `policy_v4b` 现在很容易被误读成 promotable winner。
- 只看 fresh formal 会高估 `policy_v4b` 的部署质量。
- 若不持续写回 brain，后续接管者容易退回旧叙事：把 `v2c / v2b` 当 recent frontier，把 `policy_v2` 当 deployable front-runner。

## 6. 最近失败教训
- recent frontier 不等于 promotion frontier。
- family pipeline 变宽后，constrained runner 也必须同步变宽。
- detached 长实验会破坏接管质量。
- selected profile 和 constrained best 分裂时，必须先拆 drift，再谈升级。

## 7. 当前必须遵守的规则
- 正式实验必须 `strict resume`。
- 长实验只允许前台执行。
- 默认超时预算 `10` 小时。
- 默认追求最有效，不追求最小改动。
- formal / recent / live / promotion 不得混写。

## 8. 推荐下一步
- 用 `policy_v2b` 作为 deployable 锚点，围绕 `policy_v4b` 做窄修复分支。
- 新分支必须从一开始就同时回答：
  - corrected recent 能不能保住
  - constrained formal 会不会再塌

## 9. 禁止重复尝试的无效路径
- 不要再开宽 hand-crafted repair sweep。
- 不要再把 broad backbone / Mamba / TSFM / RL 当当前主线。
- 不要在没有新 constrained formal 证据前讨论 live promotion。

## 10. 置信度与不确定性说明
- 高置信度：
  - `policy_v2b` 是当前最强 deployable learned-control candidate。
  - `policy_v4b` 是当前最强 recent frontier。
- 中置信度：
  - `policy_v4b` 的 constrained 崩塌主要来自 concentration / bridge / control 稳定性问题。
