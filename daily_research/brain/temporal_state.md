# Daily Research 时态状态模型

快照日期：`2026-04-12`

## 1. Past Ledger

### 1.1 已验证事实
- `2026-04-12` recent protocol 已修正：
  - `requested_recent_end_date = 20260410`
  - `effective validation end = 2026-03-31`
  - recent root = `short_alpha_recent_model_protocol_20260412_r1`
- `2026-04-12` strongest-model verdict 已按新协议重刷：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- `2026-04-12` `policy_v5 family` 已完整跑完：
  - `policy_v5b` recent `0.1200`
  - `policy_v5b__k1_20d` constrained formal `0.1177`
  - `policy_v5b` active family fresh formal `0.0839`
- `2026-04-12` `policy_v5a__k1_20d = 0.1018`、`policy_v5c__k1_20d = 0.1010`
  - 说明 v5 家族不只是单点偶然值，slow-bridge deployable 能力整体成立。

### 1.2 已沉淀经验
- requested recent cutoff 与 effective validation end 必须分离。
- strongest-model recent 结论会随 recent 协议修正而改变，不能沿用旧 replay 叙事。
- learned-control 的 current best 要分 `fresh formal / constrained formal / recent` 三层。
- 接管质量高度依赖标准交接包、规则记忆和 lesson memory。

## 2. Present State

### 2.1 已验证事实
- 当前 live 默认执行仍是 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`。
- strongest-model 三层当前已经重新对齐到 `short_expert_monthly_v1`。
- 当前 learned-control 最强 deployable candidate 是 `short_expert_policy_v5b__k1_20d`。
- 当前 learned-control recent winner 也是 `short_expert_policy_v5b`。
- historical cross-family learned-control fresh-formal best 仍是 `short_expert_policy_v4b = 0.1008`。

### 2.2 当前推断
- 当前主问题已经不再是“如何把 `v4b` recent 强度变成 deployable constrained formal”。
- `policy_v5b` 已经比 `v4b` 更好地回答了 recent + constrained 这两层。
- 当前更像是：
  - `v5b` 已经拿到 deployable 主导权
  - 但 fresh-formal 还没打穿 overall formal mainline

### 2.3 待验证假设
- `policy_v5b` 的主要剩余问题更像：
  - fresh-formal 稳定性不足
  - candidate-count / concentration 还有继续收敛空间
  - execution-stability regularization 可能仍不够强

### 2.4 当前最高优先级
- 在不动 live 默认值的前提下，围绕 `policy_v5b` 做下一轮窄迭代：
  - 保住 recent `0.1200`
  - 保住 constrained formal `0.1177`
  - 把 fresh formal 从 `0.0839` 往 `0.1012` 收敛

## 3. Future Map

### 3.1 最近里程碑
- 里程碑 A：
  - 维持当前 live 默认链稳定，不发生口径漂移。
- 里程碑 B：
  - 把 `policy_v5b` 固化成 learned-control 当前主研究分支。
- 里程碑 C：
  - 如果新分支能同时稳住 recent / constrained，并缩小 fresh-formal gap，再进入 promotion 讨论。

### 3.2 最近任务包
- 任务 1：
  - 围绕 `policy_v5b` 开窄版 `v5` 后继分支，不再广扫。
- 任务 2：
  - 保留 `policy_v5a`、`policy_v5c` 作为稳定性与 gross-teacher 参考对照。
- 任务 3：
  - 每轮结果都同步写回 handoff packet、temporal state、lesson memory、working memory。

### 3.3 主要风险
- 若只追 recent，可能重新造出“强 recent 但 formal 不稳”的新 frontier。
- 若只追 fresh formal，可能丢掉 `policy_v5b` 已经建立的 deployable 优势。
- 若不持续写回 brain，新 agent 接管会继续沿用 `baseline_current / policy_v2b / policy_v4b` 的旧口径。
