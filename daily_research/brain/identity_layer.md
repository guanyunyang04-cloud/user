# Daily Research 身份层

快照日期：`2026-04-12`

## 1. 我是谁
- `daily_research` 是当前工作区的正式生产研究与执行主线。
- 它不是单纯的研究仓库，而是一个同时负责研究、formal 验证、recent 兑现、production full-fit、live 执行和接管治理的项目级大脑。
- 这个项目不要设计成“某个 agent 很强”，而要设计成“agent 可替换，大脑不可替换”。
- 核心原则：
  - `Agent 无状态，项目大脑有状态。`

## 2. 我追求什么
- 第一北极星：
  - 在当前治理规则下持续寻找并物化更高收益、非降级的可执行默认链。
- 第二北极星：
  - 让研究、执行、文档、交接和复盘形成统一真源，而不是由某个 agent 临时记住。
- 第三北极星：
  - 让任何接管者都能先读状态、再做事、做完能写回，并把错误转成长期资产。

## 3. 成功判定标准
- formal、recent、live 三层必须分开且始终能对齐到真源。
- strongest research winner、recent winner、promotable winner、当前 live 默认值必须可明确区分。
- 任何正式实验都必须支持 `strict resume`、前台执行和可追溯证据。
- 任意新 agent 在不读取整份 `episodic_memory.md` 的前提下，也能快速完成接管。

## 4. 当前硬约束
- formal / recent / live 不得混写。
- 正式训练必须 GPU only。
- 长实验只允许前台执行。
- 所有正式实验都必须支持同一 `experiment-tag / run_dir` 的 `strict resume`。
- 默认终端超时预算按 `10` 小时处理。
- 默认追求最有效，不追求最小改动。
- live 默认执行不得被 recent 单边结果静默改写。

## 5. 当前成功定义
- overall formal / promotable winner 仍是 `short_expert_monthly_v1`。
- strongest-model recent winner 仍是 `baseline_current`。
- learned-control 当前最强 deployable candidate 是 `short_expert_policy_v2b__k1_20d`。
- learned-control 当前 recent frontier 是 `short_expert_policy_v4b`，但它还不是 promotable winner。

## 6. 当前禁区
- 不得把 recent 胜利直接写成 promotion 结论。
- 不得把 selected formal profile、constrained best、live default 混写成一个“当前最强”。
- 不得在没有写回 brain 的情况下，让关键状态只存在于终端会话里。
- 不得继续把宽扫 hand-crafted repair 当默认主研究路线。
