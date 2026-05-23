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
