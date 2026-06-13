# 主脑操作中枢

## 1. 默认接管入口
- 当前工作区根目录：`H:\quant_project`。
- 旧路径 `H:\new_tdx64\PYPlugins\user` 已退出本项目主链路，不得作为接管根目录。
- capsule / route / bootstrap 都是可选诊断工具，不是默认流程门禁：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json`
- 默认接管由 agent 直接根据用户目标、路径、文件内容、git diff、少数硬边界和当前记忆判断；route 输出只作提示，不替 agent 决策。
- 稳定事实锚点：QDP 负责 canonical 数据基底、registry、policy bundle、memmap 和数据清理；daily_research 负责研究、模型、回测、执行候选和 active artifact 边界。

## 2. Skills / Brain / Tools 调用顺序
- 先理解用户目标；需要时读最相关的 brain / code / output，而不是为了流程完整读取所有中心。
- 本机 skill、capsule、route、bootstrap、health、audit 都是工具，不是上级流程。个人研究者任务优先直接做、直接改、直接清理。
- 已禁用或降级为 explicit-only 的重流程 skill 不进入默认路径：`subagent-driven-development`、`requesting-code-review`、`finishing-a-development-branch`、`using-git-worktrees`、`verification-before-completion`、`test-driven-development`、`testing-strategies`。
- 可默认使用的轻量 skill 只在任务真实匹配时触发：`workspace-brain`、`executing-plans`、`systematic-debugging`、`doc` / `technical-writing`、前端 / 部署 / 安全等领域 skill。
- 分脑读取由 agent 判断：读哪里取决于目标和事实归属，不取决于 route 是否选中。
- 冲突时服从当前用户目标和脑区少数硬边界；如果通用 skill 要求 worktree、TDD、全量测试、PR、code review 或兼容层，而当前个人研究任务不需要，则跳过。

## 3. Mutation 前预检
- 先确认工作区和分支：
  - `git status --short --branch --untracked-files=all`
  - `git branch --show-current`
- 如果当前分支不是 `main`，任何会修改 repo-tracked 文件的任务都必须先纠偏到 `main`，或由用户显式撤销 `main-branch-only` 规则。
- 分支异常是 preflight blocker；不得写成研究证据、promotion 证据或分脑当前结论。
- mutation 前按常识查看 dirty paths：本轮要改的文件要读清楚；明显无关的并行改动不回滚、不混提交；共享文件和数据/active artifact 先确认风险。
- 如果 `external-project` 变更与 `target-scope` 或 `workspace-shared` 变更发生真实冲突，先停止扩大操作并说明冲突点，由用户决定是否扩展任务范围。
- mutation 预检只为防止误改、误删和混提交；不得扩展成默认审查仪式。

## 4. 结构变更顺序
- 先改 `brain/brain_architecture.md`。
- 再改 `brain/brain_manifest.json`。
- 再改目标分脑 manifest、workflow registry 与区域特化。
- 最后改具体中枢正文、skill 入口和守卫。
- 新增根目录正式项目时，用 `brain_runtime.py init/register` 创建并注册分脑；除非明确要拆出外部仓库，否则不在子目录初始化独立 `.git`。

## 5. 守卫入口
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`；裸 `doc_guard check` 只作为全量收尾/维护守卫。
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow health --brain <brain_id|workspace> --mode compact --json`；`--mode full` 只用于完整维护。
- `daily_research/tools/project_consistency_check.py --mode research` 是 daily 研究态轻量守卫；`--mode execution/full` 与 OpenMP strict 只属于执行/完整维护，不是 workspace 或 traditional 默认守卫。
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`
- Agent Meta Protocol 守卫确认 agent 是元能力执行者，brain 是持久化载体，tools 是传感器。
- `agent-meta-audit` 若返回 `agent_learning.pending_approval_count > 0`，下一次实质进展更新或最终答复必须主动提示待批准 / 待跟进 proposal；若为 `0`，可简短说明当前没有待批准 proposal。
- `brain-burden-audit` 检查热路径预算、skill 体量、冗余兼容和非源缓存；blocked 项必须先处理再继续脑区治理写回。
- `doc_guard check` 裸命令是全量收尾守卫；日常局部检查优先用 `doc_guard check --files <paths>` 或 `doc_guard check --scope changed`，且这两种轻量模式默认不跑 layout、active、large-file、integrity 全局检查。结构变更后仍建议单独跑一次 `integrity_check` 便于快速定位。
- 项目验证由 agent 根据改动面和风险选择；`selective_verification.py --paths <paths>` 可辅助推荐，不得默认注入 daily active guard。
- 默认开发验证采用 changed-surface-only：可运行 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json` 获取小验证包；agent 可按当前目标增减检查。
- 工作区测试治理与减负策略的 canonical 正文见 `brain/references/testing_governance.md`；不再保留 `docs/testing_governance.md` 外部入口。默认测试轻量化服务研究速度；canonical、PIT/no-leakage、清理边界、active artifact 和可回滚性仍属硬边界。
- `always_commands` 保留为兼容和收尾守卫入口，不代表每次小改都要全量执行；`deferred_commands` / `deferred_long_commands` 只用于慢速、研究、维护或最终确认批次。
- 普通 docs-only 只需要 `git diff --check`；brain 文档变更再加 `tools.brain.doc_guard check --files <paths>` 或 `--scope changed`。单模块 Python 变更只跑对应测试或 nodeid；shared helper、protocol、schema、config、active/execution 边界变更必须扩大测试半径或进入 manual review。
- 测试编写保持最小 fixture、最小断言面、无真实网络、无真实长训练；单测只证明数据、label、loss、bridge、gate、guard 合约，不用单测证明模型收益强。慢测必须带 `slow` / `research` / `guard` / `external` 等 marker 和明确触发条件。
- 主脑工具轻量 lane：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest tools/brain/tests -m "smoke and not slow and not external and not benchmark" -q`；完整 `tools/brain/tests` 保留为维护 / 收尾守卫，不作为普通小改默认包。
- `tools.brain.workflow health --mode full`、裸 `doc_guard check`、`audit-brain --scope all` 属于维护/收尾守卫；不得作为每次小改默认测试包。`integrity_check` 可作为结构改动的轻量定位守卫。`daily_research/output/active_execution_strategy.json` 一旦有 diff 是 critical blocker，不进入普通测试推荐。
- 默认代码改动采用 objective-first direct-change：先判断当前目标需要什么形态，再决定小改、重构、删除或重写；不为了保持旧结构、旧测试或旧入口而增加复杂度。
- 若小补丁能干净完成目标，就小改；若小补丁会引入新分支、新 fallback、新兼容层、新配置开关或重复真源，优先收敛到单一路径并删除过时路径。
- direct-change 的硬边界：不得静默改 active artifact、live/default/paper/broker 行为、promotion gate、PIT/no-leakage/OOS 证据边界、不可重建研究证据、secrets、外部服务状态或路由外并行 dirty work；这些仍按 project profile、manual review 和显式授权处理。
- 结论验证采用“足够支撑当前说法”的证据原则：能 smoke 就不全量，能抽样就不长跑，能靠文件/manifest 证明就不重新训练；不得把轻量验证伪装成强结论。

## 5.1 项目协作纪律
- 不再强制每次任务绑定 `project_id`；agent 根据用户目标、路径、事实归属和风险直接判断工作范围。
- 跨项目读取默认允许，只要服务当前目标且不把无关 output / loose latest / 并行 dirty work 误写成证据。
- 写入、删除、清理、重建、提交和进程管理要按真实风险收敛到相关路径；QDP 数据资产、daily active artifact、secrets、外部服务状态和不可重建证据仍需特别小心。
- 临时产物优先放到相关项目或明确 run/task 目录；不要把 workspace 根变成杂物堆。

## 5.2 项目提交闭环
- 提交助手是可选辅助，不是最终答复前门禁：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.project_commit --project-id <project|workspace-brain> --task-summary "<summary>" --verified <commands> --json`。
- 需要提交时由 agent 明确 pathspec、验证证据和提交说明；不需要提交时报告改动与验证即可。
- 不混提交无关 dirty paths；跨多个真实目标的改动可以一起说明，也可以拆分，取决于清晰度和回滚便利。

## 6. 写回路由
- 工作区级当前状态写回 `brain/state_center.md`。
- 工作区级固定规则与教训写回 `brain/knowledge_center.md`。
- 工作区级拓扑写回 `brain/master_brain.md`。
- 工作区级治理写回 `brain/governance_layer.md`。
- 工作区级项目特例、环境与守卫入口写回 `brain/operations_center.md`。
- 项目事实、实验状态、rXX 证据和项目命令写回 agent 判断的相关分脑；route 不决定写回权。

## 7. 轮询任务运行纪律
- 任何需要重复观察、等待外部状态或跨多轮完成的任务都继承项目任务命名空间纪律；“长任务”只是其中一种，不再作为单独治理类别。
- 同步短命令可以直接等待结果；一旦任务需要后台进程、外部 job、服务端口、下载 / 回填、训练 / 评估、浏览器 / 部署状态或跨回合观察，就使用当前项目内最可靠的可观察 handle：PID、job id、run id、stdout/stderr、progress、summary、artifact mtime、端口 / API status 或资源状态。
- 需要跨回合保留 PID、日志、progress 或 summary 时，优先用 `tools.brain.agent_run paths/register/launch/status --project-id <project> --run-id <run>` 绑定到当前项目 `process_namespace`；已有外部进程可登记，不要求为所有等待任务强行套同一 launcher。
- 轮询间隔由 agent 按信号密度、阶段、成本和风险自适应调整：启动期、故障排查或快速变化时缩短；稳定慢进展、昂贵检查或外部限流时拉长；不得用固定 sleep、固定窗口或历史专用模板替代判断。
- 每轮状态优先报告与本任务有关的最强信号；elapsed / ETA 只在有意义时给出，不能为了形式报告而制造不可靠估计。
- 等待窗口耗尽不是失败；先检查 exit/status、日志、progress、summary、artifact、端口 / API 或资源信号。仍在推进且无明确错误、资源危险、失败产物或用户停止时继续自适应轮询；有明确失败或不安全状态时及时停止、降级或请求决策。
- 跨项目进程、日志、端口、GPU、数据 provider 或其它共享资源读取仍需 active lease；没有 lease 时只报告外部存在，不下钻、不等待、不管理。

## 8. 工作区迁移纪律
- 当前唯一工作区根：`H:\quant_project`。
- `H:\new_tdx64\PYPlugins\user` 只属于通达信插件用户目录，不再保存本项目代码、brain、output、cache 或 archive。
- 根目录身份当前分为：主脑基础设施（`brain/`、`tools/`）、已注册分脑项目、过渡资产（`canonical_data/`）和本地缓存依赖（如未跟踪的 `node_modules/`、`.pytest_cache/`）。
- 历史 reference 中的旧路径保留为历史事实；新命令、新文档、新产物路径必须使用 `H:\quant_project`。
- 若需要从旧备份恢复冷数据，先复制到 `H:\quant_project`，校验后再删除旧备份；不得把旧备份目录当成当前项目根。
