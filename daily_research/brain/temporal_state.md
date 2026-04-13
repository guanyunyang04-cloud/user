# Daily Research 时态状态模型

快照日期：`2026-04-13`

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
- `2026-04-13` `policy_v5b bridge sensitivity audit` 已完成：
  - `k1_20d` 仍是 constrained deployable anchor `0.1177`
  - `k1_3d` constrained 只有 `0.0248`
  - `k1_5d = 0.0834`
  - `cap6_k1_5d` 与 `cap4_g092_098_k1_5d` 都是 no-op
- `2026-04-13` `policy_v5 successor` 首轮已完整跑完：
  - `policy_v5d = formal 0.0676 / constrained 0.0925 / recent 0.1071`
  - `policy_v5e = formal 0.0812 / constrained 0.0867 / recent 0.0934`
  - 二者都未取代 `policy_v5b`

### 1.2 已沉淀经验
- requested recent cutoff 与 effective validation end 必须分离。
- strongest-model recent 结论会随 recent 协议修正而改变，不能沿用旧 replay 叙事。
- learned-control 的 current best 要分 `fresh formal / constrained formal / recent` 三层。
- 接管质量高度依赖标准交接包、规则记忆和 lesson memory。
- `policy_v5b` 的主要脆弱性已经不只是“fresh formal 略低”，而是 bridge-speed sensitivity 明显存在。
- external cap wrapper 与温和平滑 successor 都不足以修复当前 `policy_v5b` 的快桥问题。

## 2. Present State

### 2.1 已验证事实
- 当前 live 默认执行仍是 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`。
- strongest-model 三层当前已经重新对齐到 `short_expert_monthly_v1`。
- 当前 learned-control 最强 deployable candidate 仍是 `short_expert_policy_v5b__k1_20d`。
- 当前 learned-control recent winner 仍是 `short_expert_policy_v5b`。
- historical cross-family learned-control fresh-formal best 仍是 `short_expert_policy_v4b = 0.1008`。
- `policy_v5d / policy_v5e` 都没有通过当前三层 keep gate。

### 2.2 当前推断
- 当前主问题已经不再是“如何把 `v4b` recent 强度变成 deployable constrained formal”。
- `policy_v5b` 已经比 `v4b` 更好地回答了 recent + constrained 这两层。
- 当前更像是：
  - `v5b` 已经拿到 deployable 主导权
  - 但 `v5d / v5e` 证明简单平滑方案还不足以抬起 fresh-formal 并守住三层
  - 下一轮瓶颈更像 bridge-speed sensitivity 与 execution semantics 内生化

### 2.3 待验证假设
- 下一条 successor 更可能需要：
  - 降低 slow-bridge 与 fast-bridge 的表现分裂
  - 直接约束 fast-bridge tail risk，而不是只做温和 concentration smoothing
  - 在模型内部收敛执行语义，而不是继续追加外部 cap/gross wrapper

### 2.4 当前最高优先级
- 在不动 live 默认值的前提下，围绕 `policy_v5b` 定义下一条不同于 `v5d / v5e` 的新窄分支：
  - 保住 recent `0.1200`
  - 保住 constrained formal `0.1177`
  - 把 fresh formal 从 `0.0839` 往 `0.1012` 收敛

## 3. Future Map

### 3.1 最近里程碑
- 里程碑 A：
  - 维持当前 live 默认链稳定，不发生口径漂移。
- 里程碑 B：
  - 完成 `policy_v5b` bridge sensitivity audit，确认快桥脆弱性与 external cap no-op。
- 里程碑 C：
  - 完成 `policy_v5d / policy_v5e` successor 首轮验证，并明确否决。
- 里程碑 D：
  - 定义下一条不同于本轮平滑思路的新 successor，再进入下一轮验证。

### 3.2 最近任务包
- 任务 1：
  - 以 `policy_v5b` 为锚点设计新 successor，避免重复 `v5d / v5e` 路径。
- 任务 2：
  - 保持 `formal / constrained formal / recent` 三层联动评估与 keep gate。
- 任务 3：
  - 每轮结果都同步写回 handoff packet、temporal state、lesson memory、working memory、action system。

### 3.3 主要风险
- 若只追 recent，可能重新造出“强 recent 但 constrained 不稳”的新 frontier。
- 若只追 fresh formal，可能丢掉 `policy_v5b` 已经建立的 deployable 优势。
- 若继续沿 `v5d / v5e` 这类温和平滑路径试错，会重复已经证伪的无效路线。
- 若重新依赖 external cap wrapper，会再次得到 no-op 修补。
- 若不持续写回 brain，新 agent 接管会继续沿用 `baseline_current / policy_v2b / policy_v4b` 的旧口径。
