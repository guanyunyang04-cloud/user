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
3. `Select`：agent 结合 route evidence、path evidence、manifest 与用户意图写下最终 `agent_selected_brain_id`。
4. `Preflight`：按 selected project profile 过目标、规则、教训、依赖四检。
5. `Act`：执行动作并保留证据。
6. `Complete`：验证通过后按项目范围提交；workspace 治理提交使用 `workspace-brain:` 前缀。
7. `Reflect`：把结果沉淀回状态、知识、操作、治理或 reference。
8. `Replan`：如结果改写路径，立刻更新 state 与 next step。

## 2.1 Agent Meta Protocol
- 元能力属于 agent；brain 是持久化载体，负责保存协议、证据、守卫和写回路径；capsule、audit 和 guard 只是传感器。
- 闭环为 `Brain primes agent -> Agent observes -> Detect -> Classify -> Route -> Propose -> User approves implementation -> Verify -> Brain updates -> Future agent reuses`。
- `before_final` 是闭合边界的元问题发现，不是固定 checklist：对象层任务准备交付但 final answer 尚未发出时，agent 低噪声判断是否暴露了框架、标准、方法、权威排序、评价机制或学习显著性问题。
- 研究 / 模型结论闭合前必须做 `scope-grade alignment`：结论层级必须匹配 evidence scope/grade，至少核对 `universe_scope`、dataset/pool 绑定、seed count、budget class、gate result、diagnostic-vs-evidence-grade status；不匹配时先降级措辞，再决定是否提出协议演化。
- 若工具 / audit 显示 clear，但人类反馈或任务事实说明闭合判断不成立，clear 只能作为传感器信号；agent 必须优先说明元问题，并对低风险项直接生成 `proposed` agent-learning proposal。
- 第一版权限为 `propose_only`：agent 可自动发现、分级、提示、审计和生成低风险 proposal；核心 brain docs、workflow、skill、guard、tests 写入仍需用户授权。高风险、不清晰或会改变核心行为的 proposal 创建前先询问。
- capsule 若返回 `agent_meta.review.status != clear`，agent 必须在最终答复或后续计划中说明信号、目标层、writeback route 和验证要求。
- 用户说“这应该学会 / 为什么没提示 / 以后都要”，或发现低预算证据污染模型质量结论时，优先创建 agent learning proposal 或运行 `brain_runtime.py agent-meta-audit --cwd . --mode compact`。
- agent 必须主动提示待决 agent learning proposal：若队列中存在 `proposed` 或 `approved`，在下一次实质进展更新或最终答复中列出需用户批准或跟进的学习项；若已检查且没有待决项，也要简短说明当前无待批准 proposal。
- daily_research 的低预算实验纪律由分脑知识中枢维护；主脑只保存 agent 元能力边界和路由责任。

## 2.2 Brain Burden Governance
- 规则分级为 `hard_safety`、`operating_default`、`deep_dive`、`deprecated`；安全硬规则不可自动绕过，流程默认可被 agent 临时压缩但必须说明理由。
- 当规则冲突、文档读取成本超过任务收益、兼容入口造成歧义、测试锁住旧设计或 agent 被迫执行无关流程时，记录 `brain_rule_obstruction`，目标层为 `brain_burden_governance`。
- 默认热路径预算由 `brain/brain_manifest.json#brain_burden_contract` 管理；超预算内容迁入 references 或 runtime help，不继续堆进 skill / state / operations。
- 兼容入口必须有 `owner`、`usage_evidence`、`delete_by`；没有证据的兼容入口直接删除。
- 审计入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`。

## 2.3 并行项目所有权隔离
- 工作区允许多个项目由不同 agent 并行开发；每次任务的可变更范围由主脑路由结果和用户显式授权共同决定。
- 被路由任务默认只拥有目标项目范围、该项目 brain write routes，以及用户明确纳入的 workspace 共享文件；不得把其他项目的工作树变更自动视为当前任务的一部分。
- 路由范围外的项目 dirty paths 默认视为外部并行工作；agent 不得回滚、修复、暂存、提交、合并或解释这些路径，除非用户明确扩展任务范围。
- 主脑、workflow 工具、根配置、跨项目 registry 等 workspace 共享文件不自动归属任何单一项目；修改前必须单独评估影响面。
- 若外部项目变更与当前路由任务或共享文件产生真实冲突，先报告冲突和边界，再等待用户决定是否扩展任务范围。
- 每个分脑 manifest 必须能声明或继承 `guard_profile`、`verification_profile`、`commit_policy`、`process_namespace`、`cross_project_policy`；daily active artifact/freshness/project consistency 只属于 daily profile。
- 项目 agent 完成一次任务后默认本地提交；提交助手只把该项目 profile 允许路径纳入 stage/commit pathspec，外部项目 dirty paths 仅作为 `ignored_external_paths` 报告，不阻塞当前项目提交。
- 若当前提交候选路径本身越过 profile 范围，或与接管前 baseline dirty 发生目标范围内 overlap，返回 `project_commit_scope_conflict`；不得把多个项目混成一个提交。
- 长任务必须通过 `tools.brain.agent_run` 绑定到项目 `process_namespace`，PID、日志、progress 和 summary 默认位于 `<project>/output/agent_runs/<run_id>/`。
- `long_task_monitor` 必须携带 `--project-id` 与 `--run-id`；没有显式 active cross-project lease 时，其他项目 agent 不得 wait/read/管理该进程。
- 共享 GPU、端口、数据 provider 等资源租约写入 `brain/output/resource_leases/`；租约不存在时按互不干扰处理。

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
