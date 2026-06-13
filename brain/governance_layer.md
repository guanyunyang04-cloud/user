# 主脑治理层

快照日期：`2026-06-12`

## 1. 治理目标
- 保证 agent 以用户目标为中心，必要时读取主脑和分脑记忆。
- 保证关键状态、少数硬边界和写回位置始终外显。
- 在结构漂移、状态过期、接管失真时先纠偏，再执行。
- 治理服务个人研究效率；默认不制造团队式审批、兼容和测试负担。

## 2. 标准闭环
1. `Understand`：先理解用户目标、当前 dirty 状态和最可能相关的事实位置。
2. `Inspect`：按需要读取主脑、分脑、代码、registry、manifest 或产物；route / capsule 只是可选诊断。
3. `Decide`：agent 自主判断工作范围、事实归属和验证强度。
4. `Preflight`：只确认少数硬边界和当前可用入口。
5. `Act`：执行动作并保留证据。
6. `Complete`：用足够支撑结论的轻量证据收尾；需要保存的代码变更按项目范围提交。
7. `Reflect`：把结果沉淀回状态、知识、操作、治理或 reference。
8. `Replan`：如结果改写路径，立刻更新 state 与 next step。

## 2.1 Agent Meta Protocol
- 元能力属于 agent；brain 是持久化载体，负责保存协议、证据、守卫和写回路径；capsule、audit 和 guard 只是传感器。
- 默认闭环为 `Brain primes agent -> Agent acts -> Evidence captured -> Brain updates when useful`。
- `before_final` 不是固定 checklist；只在发现框架、权威排序、评价机制或长期记忆偏差时触发。
- 研究 / 模型结论闭合前只做必要的证据层级对齐：不要把诊断、低预算、未完成、样本太窄的结果说成高等级结论。
- 低风险脑区语义修正、旧流程降级和文档收敛可在用户明确“就这么办”后直接实施；不再强制 proposal 队列。
- 高风险项才需要先停下来：改变 live/default、active artifact、不可重建数据、外部服务状态、密钥、PIT/no-leakage 规则或跨项目所有权。
- daily_research 的实验纪律由分脑维护；主脑只保存少数共享事实、硬边界和个人研究者默认风格。

## 2.2 Brain Burden Governance
- 规则分级为 `hard_safety`、`operating_default`、`deep_dive`、`deprecated`；安全硬规则不可自动绕过，流程默认可被 agent 临时压缩但必须说明理由。
- 当规则冲突、文档读取成本超过任务收益、兼容入口造成歧义、测试锁住旧设计或 agent 被迫执行无关流程时，记录 `brain_rule_obstruction`，目标层为 `brain_burden_governance`。
- 默认热路径预算由 `brain/brain_manifest.json#brain_burden_contract` 管理；超预算内容迁入 references 或 runtime help，不继续堆进 skill / state / operations。
- 兼容入口必须有 `owner`、`usage_evidence`、`delete_by`；没有证据的兼容入口直接删除。
- 重流程 skill、旧测试、旧 wrapper 和旧数据入口不得因“曾经存在”自动保留；当前主线不需要就清理或降级为 explicit-only。
- 审计入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`。

## 2.3 项目协作与并行工作
- 工作区允许多个项目并行推进；agent 根据用户目标、路径、事实归属和风险判断读取与修改范围。
- 跨项目读取默认允许，前提是服务当前目标且不把无关 output、loose latest、并行 dirty work 写成证据。
- QDP 是 canonical 数据基底、registry、policy bundle、memmap 和数据清理的默认事实来源；daily_research 是研究、模型、回测、执行候选和 active artifact 的默认事实来源。
- 写入、删除、清理、重建、进程管理和提交要按真实风险收敛到相关路径；遇到 active artifact、唯一数据、PIT/no-leakage、secrets 或外部服务状态时先停下来确认。
- 临时产物优先写入相关项目或明确 task/run 目录；不要把 workspace 根或无关项目 output 当成杂物区。
- 提交助手、agent_run、resource lease 都是可选辅助；需要跨回合进程或共享资源时可用，不作为日常短任务门禁。

## 3. 守卫
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json` 可作为诊断入口，不是默认门禁。
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed` for ordinary changed-surface checks; bare `doc_guard check` is full/final maintenance.
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
- 项目事实写回 agent 判断的相关分脑，不写入主脑。
