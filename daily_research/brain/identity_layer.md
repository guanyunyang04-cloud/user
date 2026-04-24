# Daily Research 身份层

快照日期：`2026-04-24`

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
- 第四北极星：
  - 构建一个以日为单位进行连续决策的交易执行模型，而不是继续围绕固定调仓频率、固定持有周期或人工执行桥做局部优化。
  - 这个模型应直接从市场全局状态、个股演化路径与持仓上下文中学习 `open / hold / add / reduce / exit / cash` 的动态最优执行。
  - 它追求的是在尽量少的人为约束下，综合权衡未来收益、风险与成本，并做出当前条件下最优的动态执行判断。

## 3. 成功判定标准
- formal、recent、live、promotion 四层必须分开且始终能对齐到真源。
- strongest-model 的 `formal winner / recent winner / promotable winner` 必须可明确区分。
- learned-control 的 `fresh formal / constrained formal / recent` 也必须分层表述。
- 任何正式实验都必须支持可追溯证据、前台执行、项目解释器一致性，并明确声明自己属于 `epoch formal candidate` 还是 `non-epoch shadow prototype`。
- 任意新 agent 在不通读整份 `episodic_memory.md` 的前提下，也能完成接管。

## 4. 当前硬约束
- formal / recent / live / promotion 不得混写。
- `deep_alpha` 与可 promotion 的 `continuous_policy formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4` 都属于 `epoch formal candidate`：必须 GPU only。
- `epoch formal candidate` 至少从 `32` epoch 起步；不够就沿同一 `experiment-tag / run_dir` 做 `strict resume` 续训。
- `continuous_policy prototype_gbdt_v1` 明确属于 `non-epoch shadow prototype`：只允许 shadow / teacher / ablation，不计入 formal 完整判决，不得直接 promotion。
- `continuous_policy formal_torch_seq_v3` 是 stronger temporal sequence branch：只在 `v2` 仍受 `hold / reduce / cash` 行为瓶颈约束时进入正式主计划，但一旦启用，仍必须遵守 `GPU only + >=32 epoch + strict resume`。
- `continuous_policy formal_torch_hier_v4` 是 market / portfolio / cross-section interaction 的分层时序分支：一旦启用，同样必须遵守 `GPU only + >=32 epoch + strict resume`，并先以 `shadow_only` 方式验证。
- 长实验只允许前台执行。
- 所有 `epoch formal candidate` 都必须支持同一 `experiment-tag / run_dir` 的 `strict resume`。
- 默认终端超时预算按 `10` 小时处理。
- 默认追求最高效、最合理，不追求最小改动。
- 某设定在较弱模型上失效，不等于在更强模型上永久淘汰；是否重开验证，取决于预期信息增益是否足够高。
- `requested_recent_end_date` 与 `effective recent validation end` 必须分开记录。
- live 默认执行不得被单次 recent 结果静默改写。

## 5. 当前事实入口
- 身份层只保留目标、边界和硬约束，不再承载可变的 live 默认、winner 数值或阶段指标。
- 当前状态、当前优先级、当前 live 默认解释：
  - `daily_research/brain/state_center.md`
- 稳定事实、硬规则与长期教训：
  - `daily_research/brain/knowledge_center.md`
- 当前 active 执行物化真源：
  - `daily_research/output/active_execution_strategy.json`
- 如果身份层与上述真源冲突，以状态中枢和 active artifact 为准，并立即回写纠偏。

## 6. 当前禁区
- 不得把 learned-control recent 胜利直接写成 promotion 结论。
- 不得把 selected formal profile、constrained best、fresh formal best 和 live default 混写成一个“当前最强”。
- 不得在没有写回 brain 的情况下，让关键状态只存在于终端会话里。
- 不得把可变 live 默认、最新分数或实验指标长期写在 `identity_layer.md`。
- 不得继续把宽扫 hand-crafted repair 当默认主研究路线。
- 不得再把 `baseline_current` 写成 strongest-model 当前 recent winner。
