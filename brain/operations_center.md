# 主脑操作中枢

## 1. 默认接管入口
- 只接主脑：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_bootstrap.py`
- 接主脑并进入目标分脑：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_bootstrap.py --child <brain_id>`

## 1.0 Skills / Brain / Tools 调用顺序
- 先读 brain 约束：确认事实、推断、假设、权威层级、分支纪律、active artifact 禁区和目标分脑边界。
- 再调用适用的本机 skill：TDD、debugging、planning、verification、文档、前端、安全和部署等通用操作流程以 `C:/Users/ASUS/.codex/skills` 为准。
- 最后跑项目守卫：项目特有预检、验证矩阵、证据查询和写回路线由对应分脑与 `daily_research/tools/` 提供。
- 冲突时先服从项目安全边界：如果通用 skill 默认要求 worktree、commit、写 spec 或扩大执行，而 brain 明确要求 `main`、不提交、不触碰 active artifact，则以 brain 约束为准。
- brain 只记录项目特例和必要命令索引；不得在主脑正文复制本机 skill 的完整方法正文。

## 1.1 Mutation 前预检
- 先确认工作区和分支：
  - `git status --short --branch --untracked-files=all`
  - `git branch --show-current`
- 如果当前分支不是 `main`，任何会修改 repo-tracked 文件的任务都必须先纠偏到 `main`，或由用户显式撤销 `main-branch-only` 规则。
- 分支异常是 preflight blocker；不得写成研究证据、promotion 证据或分脑当前结论。

## 2. 结构变更顺序
- 先改 `brain/brain_architecture.md`
- 再改 `brain/brain_manifest.json`
- 再改目标分脑 manifest 与区域特化
- 最后改具体中枢正文和兼容别名

## 2.1 文档收口顺序
- 先判断文档内容属于哪个脑：
  - 工作区共性规则写入 `brain/`
  - 项目事实、入口、流程写入对应 `<child>/brain/`
  - 过长但仍需保留的参考材料写入 `<child>/brain/references/`
- 再把 body 顶层 README / 兼容文档改成简体中文索引，明确指向 brain 真源
- 最后运行 `brain_integrity_check.py --json` 与 `doc_guard.py check`

## 3. 写回路由
- 工作区级当前状态写回 `brain/state_center.md`
- 工作区级固定规则与教训写回 `brain/knowledge_center.md`
- 工作区级拓扑写回 `brain/master_brain.md`
- 工作区级治理写回 `brain/governance_layer.md`
- 工作区级项目特例、环境与守卫入口写回 `brain/operations_center.md`
- 通用操作技能的详细方法不写入 brain；需要时使用本机 `C:/Users/ASUS/.codex/skills`。
## 4. 守卫入口
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/project_consistency_check.py`
- `doc_guard.py check` 已包含主分脑完整性检查；结构变更后仍建议单独跑一次 `brain_integrity_check.py --json` 便于快速定位
## 5. 环境基线
- brain 文档统一使用 UTF-8
- brain 当前层正文、标题、复盘和规则写回遵循 `brain/language_policy.md`：中文语义 + 英文工程标识。
- 命令、路径、JSON key、workflow id、dataset id、tag、模型名等技术标识保留原文。
- 工具调用优先走显式 Python 路径或显式脚本入口；当前 `daily_research` 相关 Python 命令统一使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`
- 不依赖“当前 shell 恰好已经处于正确环境”的隐性状态

## 长时任务运行纪律
- 工作区级规则：项目任务默认前台运行，任务主进程不得被转入后台规避窗口，也不得中途人为中断。
- 所有训练、评估、审计、bounded study、confirmatory rerun 和执行任务的前台窗口时限统一扩为 `10` 小时。
- 任务启动必须使用目标分脑声明的显式运行环境；当前 `daily_research` 全部 Python 任务必须使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- 涉及 GPU 训练的任务必须在完成后核验训练诊断中的 `device = cuda` 与 `cuda_available = true`，不得把未核验环境的结果写成正式证据。
- 进程存活判断必须绑定 PID、CommandLine、run_tag 或最新产物时间戳；不得用裸 `Get-Process python` 把检查脚本自身或其他短暂 Python 误判为目标任务。
- 可能超过外层捕获窗口的长任务必须同步写入持久 stdout/stderr 日志；默认使用 `Start-Process -PassThru` 记录目标 PID，再用 `Wait-Process -Id <pid> -Timeout 7200` 轮询。
- `7200` 秒是长任务单轮默认最大等待上限；进程提前自然结束时必须立即返回并解析产物，不得用固定 sleep 取代 PID 绑定等待。
- 首个 progress 尚未生成、正在排障、用户要求更密集状态或资源风险较高时，可以临时缩短单轮 timeout；每轮状态优先读取 progress、PID、日志、GPU/内存与最新产物时间戳。
- 任务完成后一次性读取日志、summary、checkpoint、evaluation 或 audit 产物，并按目标分脑写回；除 7200s 长任务监控、异常与用户询问外，不做无意义进度轮询。
