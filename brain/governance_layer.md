# 主脑治理层

快照日期：`2026-06-12`

## 1. 治理目标
- 保证主脑是唯一 agent 接管入口。
- 保证任务先经主脑路由，再进入明确分脑。
- 保证关键状态、入口、硬边界和写回位置始终外显。
- 在结构漂移、状态过期、接管失真时先纠偏，再执行。
- 治理服务个人研究效率；默认不制造团队式审批、兼容和测试负担。

## 2. 标准闭环
1. `Wake`：运行 `tools.brain.workflow capsule`，读取主脑上下文。
2. `Route`：判断任务属于 workspace governance 还是某个分脑。
3. `Select`：agent 结合 route evidence、path evidence、manifest 与用户意图写下最终 `agent_selected_brain_id`。
4. `Preflight`：只确认目标、项目归属、少数硬边界和当前可用入口。
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
- daily_research 的实验纪律由分脑维护；主脑只保存路由、硬边界和个人研究者默认风格。

## 2.2 Brain Burden Governance
- 规则分级为 `hard_safety`、`operating_default`、`deep_dive`、`deprecated`；安全硬规则不可自动绕过，流程默认可被 agent 临时压缩但必须说明理由。
- 当规则冲突、文档读取成本超过任务收益、兼容入口造成歧义、测试锁住旧设计或 agent 被迫执行无关流程时，记录 `brain_rule_obstruction`，目标层为 `brain_burden_governance`。
- 默认热路径预算由 `brain/brain_manifest.json#brain_burden_contract` 管理；超预算内容迁入 references 或 runtime help，不继续堆进 skill / state / operations。
- 兼容入口必须有 `owner`、`usage_evidence`、`delete_by`；没有证据的兼容入口直接删除。
- 重流程 skill、旧测试、旧 wrapper 和旧数据入口不得因“曾经存在”自动保留；当前主线不需要就清理或降级为 explicit-only。
- 审计入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`。

## 2.3 项目任务命名空间与并行所有权隔离
- 工作区允许多个项目由不同 agent 并行开发；每次任务必须先绑定一个明确 `project_id` / task namespace，可变更范围由主脑路由结果和用户显式授权共同决定。
- 被路由任务默认只拥有目标项目范围、该项目 brain write routes、该项目 output/cache/tmp/reports/test artifacts，以及用户明确纳入的 workspace 共享文件；不得把其他项目的工作树变更自动视为当前任务的一部分。
- 普通读写、短脚本、一次性诊断、测试、报告、临时 JSON、日志、截图、提交和进程管理都受同一项目任务命名空间约束；需要轮询或异步观察的任务只是其中一种需要可观察 handle 的场景，不是唯一需要隔离的任务。
- 路由范围外的项目 dirty paths、进程和输出默认视为外部并行工作；agent 可在接管摘要中报告其存在，但不得下钻内容、回滚、修复、暂存、提交、合并、停止、等待、复用，或把这些对象解释为当前任务证据，除非用户明确扩展任务范围。
- 主脑、workflow 工具、根配置、跨项目 registry 等 workspace 共享文件不自动归属任何单一项目；修改前必须单独评估影响面。
- 若外部项目变更与当前路由任务或共享文件产生真实冲突，先报告冲突和边界，再等待用户决定是否扩展任务范围。
- 每个分脑 manifest 必须能声明或继承 `guard_profile`、`verification_profile`、`commit_policy`、`process_namespace`、`cross_project_policy`；daily active artifact/freshness/project consistency 只属于 daily profile。
- 项目 agent 完成一次任务后默认本地提交；提交助手只把该项目 profile 允许路径纳入 stage/commit pathspec，外部项目 dirty paths 仅作为 `ignored_external_paths` 报告，不阻塞当前项目提交。
- 已验证 mutation 的最终答复前必须至少跑提交助手 dry-run 或实际提交；dry-run 要带 `--expect-paths` 覆盖本轮目标文件，避免目标文件被误判为外部并行改动。
- 若当前提交候选路径本身越过 profile 范围，或与接管前 baseline dirty 发生目标范围内 overlap，返回 `project_commit_scope_conflict`；不得把多个项目混成一个提交。
- 短命令与普通测试默认在被选项目的工作目录、验证 profile 和 changed-surface 范围内执行；跨项目测试、共享工具验证或下钻其它项目状态/产物必须由真实依赖、共享文件修改或用户授权触发。
- 临时产物必须优先写入当前项目或当前 task/run 命名空间；不得把散落在 workspace 根、其它项目 output 或 loose latest 的文件当成当前任务证据。
- 需要跨回合运行、后台驻留或保留 PID / 日志 / progress / summary 的轮询任务，优先通过 `tools.brain.agent_run` 绑定到项目 `process_namespace`；短同步命令可直接等待，轮询间隔和观察窗口由 agent 按信号密度、资源成本和风险自适应决定。
- 轮询 / 异步任务必须携带项目归属与可观察 handle；没有显式 active cross-project lease 时，其他项目 agent 不得 wait、tail/read 日志产物或管理该进程。
- 共享 GPU、端口、数据 provider 等资源租约写入 `brain/output/resource_leases/`；租约不存在时按互不干扰处理。

## 3. 守卫
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
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
- 项目事实写回主脑路由后的分脑，不写入主脑。
