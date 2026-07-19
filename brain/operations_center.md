# 主脑操作中枢

## 1. 对象方法入口
操作中枢记录对象可用方法，不规定每次任务都要完整执行。agent 先识别当前任务触碰的对象，再选择对应方法和验证。

### `workspace_memory.inspect`
- 当前工作区根目录：`H:\quant_project`。
- 旧路径 `H:\new_tdx64\PYPlugins\user` 已退出本项目主链路；新命令、新文档、新产物路径使用 `H:\quant_project`。
- `workspace-brain` skill 是入口提示器，用于校准对象归属、当前状态和可能的受保护对象。
- capsule / route / bootstrap 是可选传感器：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow closure-check --task "<task>" --paths <changed_paths> --json`
- `brain/object_registry.json` 是对象归属、路由词、路径前缀、写回目标和验证入口的机器可读表；不要在 route / closure 工具里继续散落同类硬编码。
- 传感器职责只保留一条链路：`object_registry` 识别对象；`route` 选 primary/supporting brain；`capsule` 读启动上下文；`verify-plan` 选代码测试；`closure-check` 选收尾写回和 guard；`writeback-plan` 只服务证据写回，不承担通用闭环判断。

### `workspace_git_surface.inspect`
- repo-tracked mutation、提交、清理或迁移前查看：
  - `git status --short --branch --untracked-files=all`
  - `git branch --show-current`
- 纯只读分析不需要激活这个对象。
- 发现无关 dirty paths 时不回滚、不混提交；共享文件、数据资产或 active artifact 先判断对象风险。

### `brain_doc.validate`
- 普通 docs-only：`git diff --check`
- brain 文档小改：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --files <paths>`
  - 或 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`
- 结构改动：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.brain_sync_audit --json`
- 全量维护时才使用裸 `doc_guard check` 或 full health。

### `brain_health.inspect`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-structure-audit --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py multi-paradigm-lint --cwd . --scope attached`
- `Agent Meta Protocol` 状态由 agent-meta-audit 暴露；它是经验写回对象，不是每次任务的固定 checklist。

### `project_test.select`
- 项目验证由当前对象、改动面和风险决定。
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json` 可辅助推荐。
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow closure-check --paths <changed_paths> --json` 可辅助判断 brain writeback、active artifact guard 和对象级验证。
- 单模块 Python 变更跑对应测试；shared helper、schema、registry、PIT、active/execution 边界扩大验证。
- 测试只证明合约和边界，不用单测证明模型收益强。

## 2. 写回方法
- `workspace_state.update` -> `brain/state_center.md`
- `workspace_knowledge.update` -> `brain/knowledge_center.md`
- `workspace_topology.update` -> `brain/master_brain.md`
- `workspace_architecture.update` -> `brain/brain_architecture.md`
- `workspace_operations.update` -> `brain/operations_center.md`
- `workspace_governance.update` -> `brain/governance_layer.md`
- `project_memory.update` -> agent 判断的相关分脑
- `long_evidence.archive` -> 对应 `brain/references/`

route 不决定写回权；对象归属和证据内容决定写回位置。

## 3. 受保护方法
- `qdp_active_data_base.switch_pointer / cleanup / rebuild`：先识别 replacement pointer、dry-run 或抽样验证。
- `daily_research_active_artifact.update / restore / activate`：先读取 active artifact 和 promotion 边界，并需要显式授权。
- `secret_or_external_state.update`：不把密钥、账号或外部服务状态当普通文本改。
- `cross_project_process.manage`：没有 active lease 时只报告外部存在，不下钻、不等待、不管理。

## 4. 轮询方法
- 同步短命令直接等待结果。
- 需要后台进程、外部 job、服务端口、下载、回填、训练、评估、浏览器或部署状态时，绑定最可靠的 handle：PID、job id、run id、stdout/stderr、progress、summary、artifact mtime、端口或 API status。
- 轮询间隔按信号密度、阶段、成本和风险自适应调整；观察窗口耗尽不是失败证据。

## 5. 项目提交方法
- 提交助手是可选辅助，不是最终答复门禁：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.project_commit --project-id <project|workspace-brain> --task-summary "<summary>" --verified <commands> --json`
- 用户同时要求提交和推送时，显式列出本次任务路径并使用单一可重试入口：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.project_commit --project-id <project|workspace-brain> --task-summary "<summary>" --verified <commands> --expect-paths <task_paths> --push --json`
- 该入口必须检查未跟踪文件、远端领先/分叉、精确暂存范围，并在成功后验证本地与远端提交一致；不得以 `git add .`、force push 或静默 rebase 代替。
- 需要提交时明确 pathspec、验证证据和提交说明；不需要提交时报告改动与验证即可。

## 6. 工作区迁移对象
- `H:\quant_project` 是当前唯一工作区根。
- `H:\new_tdx64\PYPlugins\user` 只属于通达信插件用户目录，不保存本项目代码、brain、output、cache 或 archive。
- 历史 reference 中的旧路径保留为历史事实；若需要从旧备份恢复冷数据，先复制到 `H:\quant_project` 并校验。
