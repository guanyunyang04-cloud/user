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
- `simplified-chinese-user-communication`
  - 与用户沟通默认使用简体中文；只有用户明确要求其他语言、引用外部原文、或代码 / 命令 / 专有名词需要保留原文时才切换或混用
- `main-branch-only`
  - 所有代码、文档与实验工作默认在 `main` 分支展开；不得自行创建、切换或继续使用非 `main` 分支
  - 实际分支不为 `main` 时，任何 repo-tracked mutation 都必须先纠偏到 `main`，或取得用户对本次任务使用分支 / worktree 例外的明确授权
- `skills-home-rule`
  - 通用操作技能归本机 `C:/Users/ASUS/.codex/skills` 维护；brain 不复制 skill 正文，不把 TDD、调试、计划、验证、前端、安全或部署方法写成平行技能库。
- `brain-cognitive-rule`
  - brain 只保留事实、推断、假设、权威层级、项目状态、风险边界、证据索引、写回路由和必要命令入口；脑区不是通用操作技能仓库。
- `decisive-cleanup-rule`
  - 对确定性收益、低事实损失、可测试验证的清理，默认彻底移除旧路径，不保留兼容层；保守兼容只有在仍有真实外部调用者或不可替代证据价值时才成立。
- `solo-owner-objective-first-engineering`
  - 本工作区由个人独立掌控，内部研究代码、脑区工具和项目脚本默认以当前目标闭合、系统简洁、可验证、可回退为准；不按多人协作项目的保守兼容、迁移周期、形式小 diff 或历史包袱约束内部开发。
  - 改动大小不是风险判断依据；是否更接近目标、是否降低长期复杂度、是否保护真实证据、是否可验证和可回退，才是判断依据。小补丁会制造 wrapper、fallback、alias、legacy mode、重复入口或额外配置时，优先直接重构、合并或删除旧路径。
  - 兼容层、旧入口、旧测试、旧 helper、旧 adapter 只有在存在真实调用证据、不可替代证据价值或明确外部接口责任时才保留；否则默认清理，测试跟随当前真实合约。
- `natural-actionable-boundaries`
  - 对用户沟通和脑区写回默认少写防御性免责声明；优先写清事实、判断、行动、验证和真实边界。风险必须可定位、可执行、可验证，不把泛化保守语气当成安全。
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
- 轮询任务纪律不再按“长任务 / 长训练 / 重任务”分类；凡需要等待外部状态、跨多轮观察、后台进程、服务启动、数据下载、训练、评估、审计或异步 job 的任务，都按可归属、可观察、可解释、可停止处理。
- 轮询方式和间隔由 agent 根据任务类型、信号密度、资源成本和风险自适应选择；优先使用已有 handle（PID / job id / run id、日志、progress、artifact mtime、端口 / API status、summary），必要时用 `tools.brain.agent_run` 绑定到项目命名空间；不得把固定 sleep、固定窗口或历史固定模板当作通用规则。
- 等待窗口耗尽只表示观察窗口结束，不是失败证据；若可观察信号仍推进且没有明确代码错误、资源危险、停止指令或失败状态，继续自适应轮询；只有明确错误、退出状态、产物失败或用户停止才能写成 failed evidence。
- 当前 `daily_research` 任务必须显式使用 `yolos` 环境；GPU 训练任务完成后必须核验 `training_diagnostics.json` 中 `device = cuda`、`cuda_available = true` 与 `python_executable` 指向 yolos
- 如果本机 skill 要求建 worktree、写 spec、提交或执行默认流程，但项目脑区要求 `main`、不提交、不触碰 active artifact，则先服从项目脑区安全边界。
- 如果脑区和本机 skill 对“怎么做 TDD、调试、计划或验证”有重复描述，以本机 skill 为通用操作真源；脑区只记录本工作区和项目特例。
- agent-mediated 自进化闭环：脑区不是思考主体；agent 负责观察、判断、提出学习机会和执行授权写回，brain 负责保存协议、证据、守卫和复用入口。遇到 timeout、入口失败、残留进程、验证误选或其他可复现异常时，先定位根因并区分“时间没给足 / 慢 / 卡 / 失败 / 入口问题”；若证据表明只是时间没给足，应移除或绕开该限制并持续轮询；凡可修问题必须形成“agent 记录经验 -> 工程修复 -> 防复发测试或验证调度 -> brain 持久化”的闭环，不能只写聊天复盘，也不能把 timeout 直接当失败结论。
- 可执行问题优先工程化：脑区记录原则和项目特例，代码或工具负责消除可重复踩坑的入口、验证选择和守卫缺口；若只能操作手动命令，必须写明安全匹配边界，避免误伤其他任务。

## 3. 当前长期边界
- 主脑不是分脑事实库
- 分脑不是跨项目规则库
- `daily_research` 负责正式生产研究与执行主线
- `t0_project` 负责盘中实验与 RL 原型，不直接替代正式主线
- `daily_stock_analysis-main` 是独立产品分脑，不改写 `daily_research` 默认执行

## 4. daily_research continuous_policy 路由边界
- `daily_research` 的 continuous_policy 细节只写入分脑；主脑只保留跨项目边界：该主线在未过正式 gate 与 stable confirm 前始终是 `research / shadow_only`。
- rXX references、trial 指标、长 tag、局部命令、Path20 历史线 / multi_horizon_utility 当前主线 / continuous_policy / deep_alpha 结论均属于 `daily_research` 分脑事实；不得在主脑展开或更新。
- 全局教训：promotion / live / active artifact 切换不能由局部研究证据、单项 smoke、局部 guard 清零或短窗高分直接触发；必须回到目标分脑的正式 gate 与 active 边界。
