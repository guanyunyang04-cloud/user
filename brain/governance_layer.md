# 主脑治理层

快照日期：`2026-05-23`

## 1. 治理目标
- 保证主脑是唯一 agent 接管入口。
- 保证任务先经主脑路由，再进入明确分脑。
- 保证关键状态、规则、入口和风险始终外显。
- 在结构漂移、状态过期、接管失真时先纠偏，再执行。

## 2. 标准闭环
1. `Wake`：运行 `tools.brain.workflow capsule`，读取主脑上下文。
2. `Route`：判断任务属于 workspace governance 还是某个分脑。
3. `Choose`：选当前最该做的一件事。
4. `Preflight`：过目标、规则、教训、依赖四检。
5. `Act`：执行动作并保留证据。
6. `Reflect`：把结果沉淀回状态、知识、操作、治理或 reference。
7. `Replan`：如结果改写路径，立刻更新 state 与 next step。

## 2.1 Agent Meta Protocol
- 元能力属于 agent；brain 是持久化载体，负责保存协议、证据、守卫和写回路径；capsule、audit 和 guard 只是传感器。
- 闭环为 `Brain primes agent -> Agent observes -> Detect -> Classify -> Route -> Propose -> Verify -> Brain updates -> Future agent reuses`。
- 第一版权限为 `propose_only`：agent 可自动发现、分级、提示、审计和生成 proposal；核心 brain docs、workflow、skill、guard、tests 写入仍需用户授权。
- capsule 若返回 `agent_meta.review.status != clear`，agent 必须在最终答复或后续计划中说明信号、目标层、writeback route 和验证要求。
- 用户说“这应该学会 / 为什么没提示 / 以后都要”，或发现低预算证据污染模型质量结论时，优先创建 agent learning proposal 或运行 `brain_runtime.py agent-meta-audit --cwd . --mode compact`。
- agent 必须主动提示待决 agent learning proposal：若队列中存在 `proposed` 或 `approved`，在下一次实质进展更新或最终答复中列出需用户批准或跟进的学习项；若已检查且没有待决项，也要简短说明当前无待批准 proposal。
- daily_research 的低预算实验纪律由分脑知识中枢维护；主脑只保存 agent 元能力边界和路由责任。

## 3. 守卫
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- 结构升级后必须保证：
  - manifest 可解析。
  - 主脑 workflow registry 可加载。
  - 分脑 workflow registry 只承载项目工作流。
  - 读取顺序可执行。
  - 写回路由可追踪。
  - 主脑 child 引用与分脑 manifest 一致。
  - brain 中不存在平行真源。

## 4. 写回纪律
- 工作区级当前状态写回 `brain/state_center.md`。
- 工作区级长期知识写回 `brain/knowledge_center.md`。
- 工作区级拓扑写回 `brain/master_brain.md`。
- 工作区级操作与环境写回 `brain/operations_center.md`。
- 工作区级治理写回 `brain/governance_layer.md`。
- 项目事实写回主脑路由后的分脑，不写入主脑。
