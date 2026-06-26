# 主脑知识中枢

## 1. 对象化原则
脑区采用面向对象的自然语言模型。规则、习惯、工具和边界都挂在对象上：

- 对象先说明自己是什么、不是什么。
- 属性说明归属、消费者、当前状态和证据入口。
- 方法说明 inspect、update、validate、cleanup 等操作如何触发。
- 保护语义只在任务触碰相关对象或方法时激活。

这样可以保留硬边界，又避免在无关任务里机械复述无关风险。

## 2. 核心对象

### `workspace_memory`
- 定义：`H:\quant_project` 的个人研究者长期记忆。
- 归属：主脑 `brain/`。
- 方法：identify、handoff、route_by_object、writeback。
- 经验：先理解用户目标，再读取最小相关脑区、代码、registry、manifest、output 或 reference；route / capsule / bootstrap 是可选传感器。

### `workspace_git_surface`
- 定义：本工作区默认连续工作面。
- 当前状态：默认分支为 `main`，不默认使用额外 git worktree。
- 激活条件：repo-tracked mutation、提交、清理、迁移、分支或 worktree 操作。
- 保护语义：激活时先看当前分支和 dirty paths；纯只读分析无需复述该边界。

### `workspace_brain_skill`
- 定义：脑区接管的入口提示器。
- 不是：审批系统、固定读取顺序或完整流程模板。
- 方法：校准目标归属、当前对象、dirty paths 和可能的受保护对象。
- 经验：脑区管辖项目的 repo-tracked mutation 前使用它；普通分析可直接读取相关对象。

### `qdp_canonical_data`
- 定义：工作区共享数据基底对象，包括 canonical lake、registry、policy bundle、memmap、coverage audit 和数据清理。
- 归属：`quant_data_platform`。
- 消费者：`daily_research`、`traditional_quant_research` 和后续研究项目。
- 保护语义：registry pointer、canonical rebuild、memmap cleanup、唯一数据删除等方法需要 replacement pointer、dry-run 或抽样验证。

### `daily_research_active_artifact`
- 定义：`daily_research` 执行状态和 active artifact 对象。
- 何时相关：执行、paper/live、active/default、broker、交易计划或 promotion。
- 何时无关：数据源评估、QDP provider、普通研究计划、脑区文档整理。
- 保护语义：update/restore/activate 需要显式授权和分脑 promotion 边界；inspect 不需要。

### `evidence_grade`
- 定义：研究证据可信等级。
- 适用：模型、策略、数据集、训练、回测、实验结论。
- 语义：smoke 证明接线，scout 生成方向，evidence-grade 支撑比较，promotion-grade 才能触及执行边界。
- 保护语义：不要把低预算、单 seed、短窗口、timeout 或 incomplete run 写成正式结论。

### `language_and_encoding`
- 定义：脑区人读语义和编码对象。
- 当前策略：中文语义 + 英文工程标识。
- 方法：写脑区文档时使用简体中文；路径、tag、dataset id、model id、CLI key 保留英文。
- 保护语义：疑似中文乱码先用 UTF-8 读取复核，不把终端显示问题当文件损坏。

## 3. 方法对象

### `changed_surface_verification`
- 来源：个人研究工作区不适合每次跑全量测试、长训练、全量 memmap 或历史回归。
- 默认：用最小检查支撑当前结论。
- 升级条件：shared helper、schema、registry、canonical、PIT/no-leakage、active artifact、执行边界或清理删除发生变化。

### `polling_or_async_work`
- 定义：需要等待外部状态、后台进程、下载、训练、评估、服务或跨回合观察的任务。
- 方法：优先绑定 PID、job id、run id、stdout/stderr、progress、summary、artifact mtime、端口或 API status。
- 经验：观察窗口耗尽只表示本轮观察结束；仍有进展且无明确失败时继续自适应轮询。

### `docs_into_brain`
- 定义：长期阅读材料、审计、迁移说明、研究日志和项目结论的归档方法。
- 默认：对应主脑或分脑 `references/` 保存长证据；body 顶层 README、AGENTS、CLAUDE、SKILL 只保留薄入口或外部接口责任。

### `decisive_cleanup`
- 来源：个人项目比团队项目更重视简单当前架构和可回滚。
- 默认：旧 wrapper、fallback、compatibility path、旧测试或旧入口若无真实调用证据、不可替代证据价值或外部接口责任，可直接清理或降级。

## 4. 长期事实
- `daily_research` 是正式生产研究与执行主线。
- `quant_data_platform` 是共享数据平台、canonical 数据基底和 memmap 治理 owner。
- `t0_project` 是盘中实验与 RL 原型，不替代正式主线。
- `daily_stock_analysis-main` 是独立产品分脑，不改写 `daily_research` 默认执行。
- `traditional_quant_research` 是传统量化方法研究分脑。
- `daily_research` 任务默认使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`；GPU 训练结论需要核验训练诊断里的设备与解释器。
- 主脑保存跨项目对象和共享边界；分脑保存项目事实；reference 保存长历史。

## 5. 已验证教训
- 如果主脑和分脑维护两套平行对象定义，后续 agent 很快会漂移。
- 如果当前状态只写聊天或终端，不写 brain，接管可靠性会明显下降。
- 如果把 `episodic_memory` 当默认入口，接管速度和质量都会恶化。
- 如果长文继续散落在 body 顶层或旧外部 docs/research_log，brain 的中枢地位会被稀释。
- 如果 README 里保留 brain 未收录的接管规则、命令入口或稳定结论，后续接管会重新绕过大脑。
- 如果脑区和通用 skill 对 TDD、调试、计划或验证有重复描述，以通用 skill 为操作能力真源；脑区只记录本工作区对象和项目特例。

## 6. continuous_policy 边界对象
- `daily_research` 的 continuous_policy 细节属于分脑对象；主脑只保留跨项目语义：未过正式 gate 与 stable confirm 前始终是 `research / shadow_only`。
- rXX references、trial 指标、长 tag、局部命令、Path20 历史线、multi_horizon_utility 当前主线、continuous_policy 和 deep_alpha 结论均属于 `daily_research` 分脑事实。
- promotion / live / active artifact 切换只能由相关分脑对象和正式 evidence-grade/promotion-grade 证据激活，不能由局部 smoke、单项 guard 清零或短窗高分触发。
