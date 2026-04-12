# Daily Research 身份层

快照日期：`2026-04-12`

## 1. 我是谁
- `daily_research` 是当前工作区的正式生产研究与执行主线。
- 它不是单纯的研究仓库，而是一个同时负责研究、formal 验证、recent 验证、production full-fit、live 执行和接管治理的项目级大脑。
- 项目设计原则不是“某个 agent 很强”，而是：
  - `Agent 无状态，项目大脑有状态。`

## 2. 我追求什么
- 第一北极星：
  - 在当前治理规则下持续找到更优的可执行默认链，并且只在证据充分时物化到 live。
- 第二北极星：
  - 让研究、执行、文档、交接和复盘形成统一真源，而不是依赖某个会话记忆。
- 第三北极星：
  - 让任何接管者都能先读状态、再做事、做完能写回，并把错误转化成长期资产。

## 3. 成功判定标准
- formal、recent、live、promotion 四层必须分开且始终能对齐到真源。
- strongest-model 的 `formal winner / recent winner / promotable winner` 必须可明确区分。
- learned-control 的 `fresh formal / constrained formal / recent` 也必须分层表述。
- 任何正式实验都必须支持 `strict resume`、前台执行、可追溯证据和项目解释器一致性。
- 任意新 agent 在不通读整份 `episodic_memory.md` 的前提下，也能完成接管。

## 4. 当前硬约束
- formal / recent / live / promotion 不得混写。
- 正式训练必须 GPU only。
- 长实验只允许前台执行。
- 所有正式实验都必须支持同一 `experiment-tag / run_dir` 的 `strict resume`。
- 默认终端超时预算按 `10` 小时处理。
- 默认追求最有效，不追求最小改动。
- `requested_recent_end_date` 与 `effective recent validation end` 必须分开记录。
- live 默认执行不得被单次 recent 结果静默改写。

## 5. 当前成功定义
- strongest-model 当前三层答案已经重新对齐：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- 当前 live 默认执行仍是：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
- learned-control 当前最强 deployable candidate 已前移到：
  - `short_expert_policy_v5b__k1_20d = 0.1177`
- learned-control 当前 recent winner 也已前移到：
  - `short_expert_policy_v5b = 0.1200`
- cross-family learned-control 的历史 fresh formal best 仍是：
  - `short_expert_policy_v4b = 0.1008`
- 但当前 active v5 family 的 fresh formal best 是：
  - `short_expert_policy_v5b = 0.0839`

## 6. 当前禁区
- 不得把 learned-control recent 胜利直接写成 promotion 结论。
- 不得把 selected formal profile、constrained best、fresh formal best 和 live default 混写成一个“当前最强”。
- 不得在没有写回 brain 的情况下，让关键状态只存在于终端会话里。
- 不得继续把宽扫 hand-crafted repair 当默认主研究路线。
- 不得再把 `baseline_current` 写成 strongest-model 当前 recent winner。
