# Daily Research 标准交接包

快照日期：`2026-04-12`

## 1. 角色定位
- 这是当前工作区的正式生产研究与执行主线交接包。
- 它服务于 strongest-model、learned-control 和 live 默认执行三条主线的统一接管。

## 2. 当前状态摘要
- 当前 live 默认：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
- strongest-model 当前三层答案：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- strongest-model 当前 recent 协议：
  - `recent_model_root_tag = short_alpha_recent_model_protocol_20260412_r1`
  - `requested_recent_end_date = 20260410`
  - `effective recent validation window = 2025-04-11 -> 2026-03-31`
- learned-control 当前关键答案：
  - `deployable winner = short_expert_policy_v5b__k1_20d = 0.1177`
  - `recent winner = short_expert_policy_v5b = 0.1200`
  - `active v5 family fresh formal best = short_expert_policy_v5b = 0.0839`
  - `historical cross-family fresh formal best = short_expert_policy_v4b = 0.1008`

## 3. 当前正在处理的问题
- 当前真正的主问题已经不是“谁是 strongest-model recent winner”。
- 当前主问题是：
  - 如何保住 `policy_v5b` 已经建立的 recent / constrained 优势，
  - 同时缩小它对 current overall formal mainline `short_expert_monthly_v1 = 0.1012` 的 fresh-formal gap。

## 4. 已完成动作及证据
- strongest-model 已按新 recent 协议重刷：
  - `daily_research/output/short_alpha_strongest_model_verdict_20260412_r1/summary.json`
- recent protocol 已修正 requested cutoff 与 effective validation end 的分离：
  - `daily_research/tools/recent_model_protocol.py`
  - `daily_research/output/short_alpha_recent_model_protocol_20260412_r1`
- `policy_v5 family` formal / constrained / recent / breakdown 已完整跑完：
  - `daily_research/output/short_alpha_policy_v5_family_formal_review_20260412_r1/summary.json`
  - `daily_research/output/short_alpha_policy_v5_family_constrained_execution_review_20260412_r1/summary.json`
  - `daily_research/output/short_alpha_policy_v5_family_recent_eval_20260412_r1/summary.json`
  - `daily_research/output/short_alpha_policy_v5_family_formal_loss_breakdown_20260412_r1/summary.md`
  - `daily_research/output/short_alpha_policy_v5_family_pipeline_20260412_r1_status.json`

## 5. 未完成动作
- 还没有一条新 learned-control 分支同时做到：
  - 保住 `policy_v5b` 的 recent `0.1200`
  - 保住 `policy_v5b__k1_20d` 的 constrained formal `0.1177`
  - 同时把 fresh formal 从 `0.0839` 拉近甚至追平 overall mainline formal `0.1012`
- 新一轮窄迭代还未开始。

## 6. 关键风险
- `policy_v5b` 容易被误读成已经应当替换 live 默认。
- 如果只看 constrained formal，会低估 `policy_v5b` 的 fresh-formal gap。
- 如果只看 fresh formal，又会错过 `policy_v5b` 已经建立的 deployable 优势。
- 如果不持续写回 brain，接管者仍可能沿用旧阶段叙事。

## 7. 必守规则
- 正式实验必须 `strict resume`。
- 长实验只允许前台执行。
- 默认超时预算 `10` 小时。
- 默认追求最有效，不追求最小改动。
- formal / recent / live / promotion 不得混写。

## 8. 禁止重复尝试的无效路径
- 不再回到 broad hand-crafted repair sweep 作为当前主线。
- 不再把 single-window recent 胜利直接讲成 live promotion 结论。
- 不再混写 `requested_recent_end_date` 与 `effective recent validation window`。

## 9. 推荐下一步
- 以 `policy_v5b` 为当前 learned-control 主研究锚点。
- 开窄版 `policy_v5` 后继分支，只围绕三件事：
  - execution-stability regularization
  - candidate-count / concentration regularization
  - 在必要时保留 gross-teacher distillation

## 10. 置信度与不确定性
- 高置信度：
  - strongest-model 三层已经重新对齐到 `short_expert_monthly_v1`
  - `policy_v5b` 是当前最强 deployable learned-control candidate
  - `policy_v5b` 也是当前 learned-control recent winner
- 中置信度：
  - `policy_v5b` 的下一步主要瓶颈是 fresh-formal gap，而不是 constrained formal
