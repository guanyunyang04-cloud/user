# 主脑操作中枢

## 1. 默认接管入口
- 当前工作区根目录：`H:\quant_project`。
- 旧路径 `H:\new_tdx64\PYPlugins\user` 已退出本项目主链路，不得作为接管根目录。
- schema v4 capsule：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
  - 读取 `target_kind` 与 `workflow_domain` 区分 workspace governance 和 child brain body work。
  - 读取 `agent_meta` 与 `agent_review` 执行 agent 元能力检查；brain 只提供协议和传感器结果。
- 任务路由：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json`
  - workspace 治理任务返回 `target.id=workspace`、`target.kind=workspace`、`target.domain=workspace_governance`。
  - route 是 advisory sensor；治理词不得替 agent 覆盖唯一 hard child evidence。
- 主脑或分脑 bootstrap：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json`
  - `workspace_governance` 是 workspace bootstrap alias，不是分脑 id；读取 `project_profile` 决定 guard、verification、commit 和 process namespace。

## 2. Skills / Brain / Tools 调用顺序
- 先运行主脑 capsule，确认事实、推断、假设、权威层级、分支纪律、active artifact 禁区和目标分脑边界。
- capsule 输出 `agent_selected_brain_id`、`selection_reason`、`routing_evidence` 与 `project_profile`；`needs_agent_decision` 是提示，不是 preflight blocker。
- 再调用适用的本机 skill：TDD、debugging、planning、verification、文档、前端、安全和部署等通用操作流程以 `C:/Users/ASUS/.codex/skills` 为准。
- 最后进入被主脑路由选中的分脑，读取项目事实、项目命令、证据边界和验证矩阵。
- 冲突时先服从项目安全边界：如果通用 skill 默认要求 worktree、commit、写 spec 或扩大执行，而 brain 明确要求 `main`、不提交、不触碰 active artifact，则以 brain 约束为准。

## 3. Mutation 前预检
- 先确认工作区和分支：
  - `git status --short --branch --untracked-files=all`
  - `git branch --show-current`
- 如果当前分支不是 `main`，任何会修改 repo-tracked 文件的任务都必须先纠偏到 `main`，或由用户显式撤销 `main-branch-only` 规则。
- 分支异常是 preflight blocker；不得写成研究证据、promotion 证据或分脑当前结论。
- mutation 前将 dirty paths 分为三类：
  - `target-scope`：当前路由项目或用户明确纳入的路径；相关变更需要按任务风险读取并协同处理。
  - `workspace-shared`：主脑、workflow 工具、根配置、跨项目 registry 等共享路径；修改前必须单独评估影响面。
  - `external-project`：路由范围外的项目路径；默认视为外部并行工作，只在有助于说明边界时报告，不作为 blocker，也不得回滚、修复、暂存、提交或混入当前任务。
- 如果 `external-project` 变更与 `target-scope` 或 `workspace-shared` 变更发生真实冲突，先停止扩大操作并说明冲突点，由用户决定是否扩展任务范围。

## 4. 结构变更顺序
- 先改 `brain/brain_architecture.md`。
- 再改 `brain/brain_manifest.json`。
- 再改目标分脑 manifest、workflow registry 与区域特化。
- 最后改具体中枢正文、skill 入口和守卫。

## 5. 守卫入口
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow health --brain <brain_id|workspace> --json`
- `daily_research/tools/project_consistency_check.py` 与 OpenMP strict 只属于 daily profile，不是 workspace 或 traditional 默认守卫。
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`
- Agent Meta Protocol 守卫确认 agent 是元能力执行者，brain 是持久化载体，tools 是传感器。
- `agent-meta-audit` 若返回 `agent_learning.pending_approval_count > 0`，下一次实质进展更新或最终答复必须主动提示待批准 / 待跟进 proposal；若为 `0`，可简短说明当前没有待批准 proposal。
- `brain-burden-audit` 检查热路径预算、skill 体量、冗余兼容和非源缓存；blocked 项必须先处理再继续脑区治理写回。
- `doc_guard` 已包含主分脑完整性检查；结构变更后仍建议单独跑一次 `integrity_check` 便于快速定位。
- 项目验证从 `project_profile.verification_profile.always_commands` 读取；`selective_verification.py --paths <paths>` 必须按路径推断项目，不得默认注入 daily active guard。
- 默认开发验证采用 changed-surface-only：先运行 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`，以 `blocking_commands` 作为本次必须通过的小验证包；未修改且未受影响区域由 `skipped_reason_by_area` 显式说明。
- `always_commands` 保留为兼容和收尾守卫入口，不代表每次小改都要全量执行；`deferred_commands` / `deferred_long_commands` 只用于慢速、研究、维护或最终确认批次。
- 普通 docs-only 只需要 `git diff --check`；brain 文档变更再加 `tools.brain.doc_guard check`。单模块 Python 变更只跑对应测试或 nodeid；shared helper、protocol、schema、config、active/execution 边界变更必须扩大测试半径或进入 manual review。
- 测试编写保持最小 fixture、最小断言面、无真实网络、无真实长训练；单测只证明数据、label、loss、bridge、gate、guard 合约，不用单测证明模型收益强。慢测必须带 `slow` / `research` / `guard` / `external` 等 marker 和明确触发条件。
- `tools.brain.workflow health` 聚合、`doc_guard check`、`integrity_check`、`audit-brain --scope all` 属于维护/收尾守卫；不得作为每次小改默认测试包。`daily_research/output/active_execution_strategy.json` 一旦有 diff 是 critical blocker，不进入普通测试推荐。
- 个人研究者写代码默认采用 direct-change：内部研究代码、脑区工具和项目脚本优先改当前真实入口；旧入口、旧测试、旧 helper 若无真实调用证据或证据价值，直接删除或改到新合约，不为抽象兼容留壳。
- direct-change 的硬边界：不得静默改 active artifact、live/paper/broker 行为、PIT/no-leakage/OOS 证据边界、不可重建研究证据或路由外并行 dirty work；这些仍按项目 profile 和 manual review 处理。

## 5.1 项目任务命名空间纪律
- 每次任务在 mutation 前必须绑定一个明确 `project_id` / task namespace；workspace 共享脑区、workflow 工具、skill 或根配置改动使用 `workspace` / `workspace-brain` 命名空间，不挂靠任一子项目。
- 默认读写、短脚本、一次性诊断、测试、临时报告、日志、截图、JSON、cache/output 和提交候选都只属于当前 project profile 允许范围，加上用户明确纳入的路径。
- 当前任务的临时产物优先写入当前项目的 output/cache/tmp/reports 或 `<project>/output/agent_runs/<run_id>/`；不得把其它项目 output、进程、loose latest 或 dirty paths 当作当前任务证据。
- 其它项目正在变化的源码、研究日志、输出、进程和测试结果默认是外部并行工作；可在接管摘要中报告其存在，但不下钻内容、不等待、不停止、不清理、不复用、不提交，也不写成本任务证据，除非用户明确扩展任务范围或存在已声明 cross-project lease。
- changed-surface 验证也受项目命名空间约束：普通项目改动只跑本项目影响面；shared tooling / schema / workflow / project profile 改动才扩大到依赖项目或 workspace 守卫。

## 5.2 项目提交闭环
- 完成项目任务且验证通过后使用项目提交助手：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.project_commit --project-id <project|workspace-brain> --task-summary "<summary>" --verified <commands> --json`。
- 提交格式为 `<project_id>: <summary>`，trailer 包含 `Project:`、`Agent-Task:`、`Verified:`。
- 提交助手只把 profile 允许范围内的当前 dirty paths 纳入 candidate pathspec；路由外项目 dirty paths 输出为 `ignored_external_paths`，不作为 blocker，也不得被 stage/commit。
- 只有当前候选路径出现 baseline dirty overlap、未验证、无项目内变更或明确范围冲突时才阻塞；不得把多个项目混成一个提交。
- 用户明确授权的跨分脑脑区治理改动，可作为 `workspace-brain` 变更提交；提交前必须显式列出 pathspec，只纳入 brain / tools/brain / child brain 文档，不纳入子项目 body dirty work。

## 6. 写回路由
- 工作区级当前状态写回 `brain/state_center.md`。
- 工作区级固定规则与教训写回 `brain/knowledge_center.md`。
- 工作区级拓扑写回 `brain/master_brain.md`。
- 工作区级治理写回 `brain/governance_layer.md`。
- 工作区级项目特例、环境与守卫入口写回 `brain/operations_center.md`。
- 项目事实、实验状态、rXX 证据和项目命令写回被路由选中的分脑。

## 7. 长时任务运行纪律
- 长任务继承项目任务命名空间纪律；它不是唯一需要隔离的任务，只是需要额外 PID、日志、progress、summary 和资源租约。
- 项目长任务必须通过 `tools.brain.agent_run paths/register/launch/status --project-id <project> --run-id <run>` 建立或登记，PID、stdout/stderr、progress、summary、run tag 和产物路径写入该项目 `process_namespace`。
- `long_task_monitor status/wait-once/trace-poll` 必须显式传 `--project-id` 与 `--run-id`；匿名 PID/日志监控直接阻塞，避免误读或误等其它项目进程。
- 已由外部命令启动的进程必须先用 `agent_run register` 绑定到当前项目 run；新启动的长任务优先用 `agent_run launch`，不再手工拼写跨项目日志路径。
- 资源租约写入 `brain/output/resource_leases/`；GPU、端口、数据 provider 或跨项目进程读取必须先声明 active lease，`--allow-cross-project` 只有带有效 `--lease-id` 才放行。
- 轮询模板：启动后先短查 PID、日志和 GPU；一旦确认 GPU 进程已正式工作，默认使用前台 `Wait-Process -Id <pid> -Timeout 7200` 轮询，除非再次出现系统崩溃或宿主不可用。
- `7200` 秒窗口结束后，先检查 PID、exit code、日志、progress、summary / checkpoint / artifact 时间戳；若进程仍在推进且没有明确代码错误、资源危险或用户停止指令，继续下一轮 `Wait-Process -Id <pid> -Timeout 7200`，不得杀进程或把窗口耗尽写成任务失败。
- 每轮状态必须计算已用时间和预计剩余时间；状态来源优先使用 progress、PID、日志尾部、GPU/内存与最新产物时间戳。

## 8. 工作区迁移纪律
- 当前唯一工作区根：`H:\quant_project`。
- `H:\new_tdx64\PYPlugins\user` 只属于通达信插件用户目录，不再保存本项目代码、brain、output、cache 或 archive。
- 历史 reference 中的旧路径保留为历史事实；新命令、新文档、新产物路径必须使用 `H:\quant_project`。
- 若需要从旧备份恢复冷数据，先复制到 `H:\quant_project`，校验后再删除旧备份；不得把旧备份目录当成当前项目根。
