# Daily Research 标准交接包

快照日期：`2026-04-13`

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
- `policy_v5b bridge sensitivity audit` 当前结论：
  - deployable anchor 仍是 `k1_20d`
  - `k1_3d` 是主要 fast-bridge tail-risk 来源
  - external cap wrapper 对当前 `k1_5d` 变体没有产生修复作用
- `policy_v5 successor` 首轮当前结论：
  - `policy_v5d` 与 `policy_v5e` 都没有取代 `policy_v5b`
  - 当前 learned-control 主研究锚点继续保持在 `policy_v5b`

## 3. 当前正在处理的问题
- 当前真正的主问题已经不是“谁是 strongest-model recent winner”。
- 当前主问题是：
  - `policy_v5b` 的 bridge-speed fragility 已经被看清，
  - 首轮 successor `policy_v5d / policy_v5e` 又没能同时守住 formal / constrained / recent，
  - 所以下一步要设计的是不同于本轮温和平滑思路的新窄假设，而不是重复同一路径。

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
- `policy_v5b bridge sensitivity audit` 已完成：
  - `daily_research/output/short_alpha_policy_v5b_bridge_sensitivity_audit_20260412_r1/summary.md`
- `policy_v5 successor` formal / constrained / recent / breakdown 已完整跑完：
  - `daily_research/output/short_alpha_policy_v5_successor_formal_review_20260412_r1/summary.json`
  - `daily_research/output/short_alpha_policy_v5_successor_constrained_execution_review_20260412_r1/summary.json`
  - `daily_research/output/short_alpha_policy_v5_successor_recent_eval_20260412_r1/summary.json`
  - `daily_research/output/short_alpha_policy_v5_successor_formal_loss_breakdown_20260412_r1/summary.md`

## 5. 未完成动作
- 还没有一条新 learned-control 分支同时做到：
  - 保住 `policy_v5b` 的 recent `0.1200`
  - 保住 `policy_v5b__k1_20d` 的 constrained formal `0.1177`
  - 同时把 fresh formal 从 `0.0839` 拉近甚至追平 overall mainline formal `0.1012`
- `policy_v5d / policy_v5e` 已经跑完并被否决，因此下一条 successor 假设还未定义、也还未启动。

## 6. 关键风险
- `policy_v5b` 容易被误读成已经应当替换 live 默认。
- 如果只看 constrained formal，会低估 `policy_v5b` 的 fresh-formal gap。
- 如果只看 fresh formal，又会错过 `policy_v5b` 已经建立的 deployable 优势。
- 如果继续沿 `policy_v5d / policy_v5e` 这种温和平滑路径追加试错，最可能重复无效路径。
- 如果把 external cap/gross wrapper 当成主修复手段，会再次落入 no-op 修补。
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
- 不再把 `policy_v5d / policy_v5e` 这轮温和平滑 successor 当作默认续跑方向。
- 不再把 external cap wrapper 当作 `policy_v5b` 当前主修复路径。

## 9. 推荐下一步
- 以 `policy_v5b` 为当前 learned-control 主研究锚点。
- 冻结 `policy_v5d / policy_v5e`，保留它们作为失败对照而不是当前候选。
- 下一轮只开 `1-2` 个新 successor，但要显式区别于本轮：
  - 优先针对 bridge-speed sensitivity / slow-fast consistency
  - 优先内生化 execution semantics，而不是再叠外部 cap
- 每个新分支都必须同时跑：
  - formal
  - constrained formal
  - recent
- keep gate 继续使用：
  - `constrained formal >= 0.110`
  - `recent >= 0.110`
  - `fresh formal > 0.0839`

## 10. 置信度与不确定性
- 高置信度：
  - strongest-model 三层已经重新对齐到 `short_expert_monthly_v1`
  - `policy_v5b` 仍是当前最强 deployable learned-control candidate
  - `policy_v5b` 也是当前 learned-control recent winner
  - `policy_v5d / policy_v5e` 已经在三层评估中失败
- 中置信度：
  - `policy_v5b` 的下一步主要瓶颈更像 bridge-speed sensitivity 与 execution semantics 内生化，而不是 gross-teacher 不足
