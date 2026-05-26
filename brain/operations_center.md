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
- 主脑或分脑 bootstrap：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json`
  - `workspace_governance` 是 workspace bootstrap alias，不是分脑 id。

## 2. Skills / Brain / Tools 调用顺序
- 先运行主脑 capsule，确认事实、推断、假设、权威层级、分支纪律、active artifact 禁区和目标分脑边界。
- 再调用适用的本机 skill：TDD、debugging、planning、verification、文档、前端、安全和部署等通用操作流程以 `C:/Users/ASUS/.codex/skills` 为准。
- 最后进入被主脑路由选中的分脑，读取项目事实、项目命令、证据边界和验证矩阵。
- 冲突时先服从项目安全边界：如果通用 skill 默认要求 worktree、commit、写 spec 或扩大执行，而 brain 明确要求 `main`、不提交、不触碰 active artifact，则以 brain 约束为准。

## 3. Mutation 前预检
- 先确认工作区和分支：
  - `git status --short --branch --untracked-files=all`
  - `git branch --show-current`
- 如果当前分支不是 `main`，任何会修改 repo-tracked 文件的任务都必须先纠偏到 `main`，或由用户显式撤销 `main-branch-only` 规则。
- 分支异常是 preflight blocker；不得写成研究证据、promotion 证据或分脑当前结论。

## 4. 结构变更顺序
- 先改 `brain/brain_architecture.md`。
- 再改 `brain/brain_manifest.json`。
- 再改目标分脑 manifest、workflow registry 与区域特化。
- 最后改具体中枢正文、skill 入口和守卫。

## 5. 守卫入口
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/project_consistency_check.py`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`
- Agent Meta Protocol 守卫确认 agent 是元能力执行者，brain 是持久化载体，tools 是传感器。
- `agent-meta-audit` 若返回 `agent_learning.pending_approval_count > 0`，下一次实质进展更新或最终答复必须主动提示待批准 / 待跟进 proposal；若为 `0`，可简短说明当前没有待批准 proposal。
- `brain-burden-audit` 检查热路径预算、skill 体量、冗余兼容和非源缓存；blocked 项必须先处理再继续脑区治理写回。
- `doc_guard` 已包含主分脑完整性检查；结构变更后仍建议单独跑一次 `integrity_check` 便于快速定位。

## 6. 写回路由
- 工作区级当前状态写回 `brain/state_center.md`。
- 工作区级固定规则与教训写回 `brain/knowledge_center.md`。
- 工作区级拓扑写回 `brain/master_brain.md`。
- 工作区级治理写回 `brain/governance_layer.md`。
- 工作区级项目特例、环境与守卫入口写回 `brain/operations_center.md`。
- 项目事实、实验状态、rXX 证据和项目命令写回被路由选中的分脑。

## 7. 长时任务运行纪律
- 项目长任务必须受监管运行，任务主进程不得脱离 PID、日志、run tag 或产物路径追踪。
- 启动模板：用 `Start-Process -PassThru` 启动目标命令，记录 PID、stdout/stderr 日志路径、run tag、预期 summary / progress / checkpoint 路径。
- 轮询模板：默认用 `Wait-Process -Id <pid> -Timeout 7200` 等待；`7200` 秒是长任务单轮前台等待窗口，进程提前自然结束时必须立即返回并解析产物。
- 单轮等待窗口耗尽后，先检查 PID、exit code、日志、progress、summary / checkpoint / artifact 时间戳；若进程仍在推进且没有明确代码错误、资源危险或用户停止指令，继续下一轮 `Wait-Process -Id <pid> -Timeout 7200`，不得杀进程或把窗口耗尽写成任务失败。
- 每轮状态必须计算已用时间和预计剩余时间；状态来源优先使用 progress、PID、日志尾部、GPU/内存与最新产物时间戳。

## 8. 工作区迁移纪律
- 当前唯一工作区根：`H:\quant_project`。
- `H:\new_tdx64\PYPlugins\user` 只属于通达信插件用户目录，不再保存本项目代码、brain、output、cache 或 archive。
- 历史 reference 中的旧路径保留为历史事实；新命令、新文档、新产物路径必须使用 `H:\quant_project`。
- 若需要从旧备份恢复冷数据，先复制到 `H:\quant_project`，校验后再删除旧备份；不得把旧备份目录当成当前项目根。
