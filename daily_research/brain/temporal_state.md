# Daily Research 时态状态模型

快照日期：`2026-04-12`

## 1. Past Ledger

### 1.1 已验证事实
- `2026-04-09` strongest-model 结论已经收口：
  - formal winner = `short_expert_monthly_v1`
  - recent winner = `baseline_current`
  - promotable winner = `short_expert_monthly_v1`
- `2026-04-10` current-default hand-crafted repair 线证明：
  - gross-control / cash-sizing 是真实主因之一
  - 但手工修补整体接近天花板
- `2026-04-11` `policy_v2 family` fresh formal / recent 已补齐：
  - direct formal family best 仍是老 `policy_v2 = 0.0830`
  - corrected recent winner 前移到 `policy_v2c = 0.1046`
- `2026-04-12` `policy_v2 family` constrained formal 已补齐：
  - `policy_v2b__k1_20d = 0.1104`
  - 已高于 current mainline formal `0.1012`
- `2026-04-12` `policy_v4 family` 第一轮已完整跑完：
  - `policy_v4b` recent `0.1387`
  - `policy_v4b` fresh formal `0.1008`
  - `policy_v4b` constrained best `0.0687`

### 1.2 已沉淀经验
- recent frontier 与 deployable frontier 往往不是同一个模型。
- execution profile 漂移必须显式测量，而不是事后解释。
- 接管质量高度依赖标准交接包、规则记忆和 lesson memory。

## 2. Present State

### 2.1 已验证事实
- 当前 live 默认执行仍是 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`。
- 当前 strongest-model recent winner 仍是 `baseline_current`。
- 当前 learned-control 最强 deployable candidate 是 `short_expert_policy_v2b__k1_20d`。
- 当前 learned-control recent winner 是 `short_expert_policy_v4b`。

### 2.2 当前推断
- 当前主问题已经不是“learned-control 行不行”，而是“怎么把 `v4b` 的 recent 强度变成不塌的 constrained formal”。
- `policy_v4b` 更像捕捉到了强 alpha / control 组合，但部署形态不稳。
- `policy_v2b` 更像稳定 slow-bridge deployable winner，但 recent 锋利度不如 `v4b`。

### 2.3 待验证假设
- `v4b` 的 constrained 崩塌主要来自：
  - concentration / candidate count 仍不够稳
  - bridge 依赖过强
  - learned gross / hold control 还没有稳住 deployable 形态

### 2.4 当前最高优先级
- 在不动 live 默认值的前提下，设计一轮窄新分支，把 `policy_v4b` 的 recent 强势转成接近 `policy_v2b` 的 constrained formal。

## 3. Future Map

### 3.1 最近里程碑
- 里程碑 A：
  - 维持当前 live 默认链稳定，不发生口径漂移。
- 里程碑 B：
  - 设计并验证 `policy_v4b` 的 deployable 修复分支。
- 里程碑 C：
  - 若新分支能同时过 recent 和 constrained formal，再进入 formal / promotion 讨论。

### 3.2 最近任务包
- 任务 1：
  - 窄迭代 `policy_v4` 后继分支，不再广扫。
- 任务 2：
  - 保留 `policy_v2b` 作为 deployable 对照锚点。
- 任务 3：
  - 每轮结果都同步写回 handoff packet、temporal state、lesson memory。

### 3.3 主要风险
- 若只追 recent，容易再造一个不可部署 frontier。
- 若只追 constrained formal，容易丢掉 `v4b` 的 alpha 强度。
- 若不持续写回 brain，新 agent 接管会重新陷入“谁是 current best”的口径混乱。
