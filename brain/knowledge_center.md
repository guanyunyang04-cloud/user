# 主脑知识中枢

## 1. 固定规则
- `brain-first`
  - 先接主脑，再接分脑，再进 body
- `common-in-main`
  - 共享结构、共享顺序、共享治理只在主脑定义一次
- `local-in-child`
  - 分脑只维护项目事实、当前状态和 body 入口
- `brain-as-doc-hub`
  - 权威治理文档、接管文档和长文参考默认只留在 `brain/` 或 `brain/references/`
- `authority-matrix`
  - 主脑 `brain/` 只维护跨项目规则、分脑拓扑、默认接管顺序、全局分支纪律和执行纪律；分脑只维护项目事实、项目状态、项目命令和项目验证矩阵；body 顶层 README、AGENTS、CLAUDE、SKILL 只作为公开指南或兼容入口，不能覆盖 brain
- `docs-into-brain`
  - README、教程、审计、迁移说明等文档内容必须先整合进对应主脑或分脑；body 顶层文档只保留简体中文索引、公开指南或兼容入口，不能成为平行真源
- `simplified-chinese-docs`
  - 工作区内面向人读的项目治理与接管文档默认使用简体中文；英文只保留在代码标识、命令、第三方专名、链接或产品必须的多语言公开文档中
- `main-branch-only`
  - 所有代码、文档与实验工作默认在 `main` 分支展开；不得自行创建、切换或继续使用非 `main` 分支
  - 实际分支不为 `main` 时，任何 repo-tracked mutation 都必须先纠偏到 `main`，或取得用户对本次任务使用分支 / worktree 例外的明确授权
- `skills-home-rule`
  - 通用操作技能归本机 `C:/Users/ASUS/.codex/skills` 维护；brain 不复制 skill 正文，不把 TDD、调试、计划、验证、前端、安全或部署方法写成平行技能库。
- `brain-cognitive-rule`
  - brain 只保留事实、推断、假设、权威层级、项目状态、风险边界、证据索引、写回路由和必要命令入口；脑区不是通用操作技能仓库。
- `workspace-fallback-rule`
  - API 或 no-plugin 会话无法加载本机 skills 时，可以使用主脑 `tools.brain.workflow capsule/workflow-guide` 作为本地 fallback；fallback 只承载主脑路由、项目约束、预检、证据边界和验证调度，不宣称替代本机 skills。
- `skills-brain-tools-matrix`
  - 本机 skills 管通用操作能力；主脑和分脑管认知治理与项目真相；项目 `tools/` 管可执行守卫。项目安全边界高于通用 skill 默认行为。
## 2. 已验证教训
- 如果主脑和分脑维护两套平行接管顺序，后续 agent 很快会漂移
- 如果当前状态只写聊天或终端，不写 brain，接管可靠性会明显下降
- 如果把 `episodic_memory` 当默认入口，接管速度和质量都会恶化
- 如果长文继续散落在 body 顶层，brain 的中枢地位会被稀释
- 如果 README 里保留了 brain 未收录的接管规则、命令入口或稳定结论，后续接管会重新绕过大脑
- 如果接管入口、命令入口或写回路由已经漂移，先纠偏再重开实验，通常比直接推进更能降低误操作风险
- 如果控制台显示疑似中文乱码，先用 UTF-8 读取工具确认真实文件内容，不能把终端编码错觉当作文件损坏来修
- 脑内文档铁律：当前层标题、正文、规则、状态和复盘写回必须使用简体中文；命令、路径、指标名、tag、模型名等技术标识保留原文
- 主分脑结构变更后必须跑 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`，确认父子附着、读序、写回路由、body 映射和编码合同仍一致
- 主脑 `state_center.md` 只承载当前路由和跨项目边界，不再追加日期型实验日志；分脑高频入口也必须优先保留当前结论，历史细节下沉到 `episodic_memory.md` 或 `brain/references/`
- 长任务运行纪律是受监管独立进程，不是脱管后台化：训练、评估、审计、bounded study、confirmatory rerun 与执行任务可以用 `Start-Process -PassThru` 启动，但必须记录 PID、持久 stdout/stderr、run tag 或产物路径
- 长任务默认用 `Wait-Process -Id <pid> -Timeout 7200` 轮询；`7200` 秒是单轮最大等待上限，进程提前自然结束时必须立即返回并解析 progress、日志、summary、checkpoint 或评估产物
- 固定 sleep 式轮询容易错过提前完成和混淆进程状态；长任务轮询必须绑定 PID、progress、日志、GPU/内存或最新产物时间戳，每轮状态必须计算已用时间和预计剩余时间，首个 progress 未出现或排障时才临时缩短 timeout
- 当前 `daily_research` 任务必须显式使用 `yolos` 环境；GPU 训练任务完成后必须核验 `training_diagnostics.json` 中 `device = cuda`、`cuda_available = true` 与 `python_executable` 指向 yolos
- 如果本机 skill 要求建 worktree、写 spec、提交或执行默认流程，但项目脑区要求 `main`、不提交、不触碰 active artifact，则先服从项目脑区安全边界。
- 如果脑区和本机 skill 对“怎么做 TDD、调试、计划或验证”有重复描述，以本机 skill 为通用操作真源；脑区只记录本工作区和项目特例。

## 3. 当前长期边界
- 主脑不是分脑事实库
- 分脑不是跨项目规则库
- `daily_research` 负责正式生产研究与执行主线
- `t0_project` 负责盘中实验与 RL 原型，不直接替代正式主线
- `daily_stock_analysis-main` 是独立产品分脑，不改写 `daily_research` 默认执行

## 4. daily_research continuous_policy 路由边界
- `daily_research` 的 continuous_policy 细节只写入分脑；主脑只保留跨项目边界：该主线在未过正式 gate 与 stable confirm 前始终是 `research / shadow_only`。
- rXX references、trial 指标、长 tag、局部命令、Path20 / continuous_policy / deep_alpha 结论均属于 `daily_research` 分脑事实；不得在主脑展开或更新。
- 全局教训：promotion / live / active artifact 切换不能由局部研究证据、单项 smoke、局部 guard 清零或短窗高分直接触发；必须回到目标分脑的正式 gate 与 active 边界。
