# 主脑架构

## 1. 架构定位
`brain/` 是 `H:\quant_project` 的主脑：它保存跨项目对象、长期记忆、项目关系和少数保护语义。

脑区是一套多范式的自然语言程序：对象式描述“是什么”，过程式描述“怎么做”，函数式描述“如何从输入推导判断”。三者都用自然语言表达，但要像程序一样有清晰接口、作用域和副作用边界。

## 2. 多范式模型

### 对象式：Object Layer
对象式负责项目、数据资产、执行物、证据集合和工具入口这些“名词”。对象默认使用这组属性描述：

- `定义`：对象是什么，不是什么。
- `归属`：由哪个主脑或分脑维护。
- `消费者`：哪些项目或流程读取它。
- `当前状态`：接管时真正需要知道的当前事实。
- `方法`：常见操作，如 inspect、ingest、update、validate、cleanup。
- `激活条件`：哪些任务会让这个对象相关。
- `保护语义`：触碰该对象时需要保留的不变量。
- `证据入口`：长报告、registry、manifest、run tag 或 reference。

保护语义挂在对象上，不作为每次任务的全局口号。无关对象不激活，无关边界不复述。

### 过程式：Procedure Layer
过程式负责可执行步骤。它不作为全局仪式存在，只作为对象方法的实现说明，例如 `shortline_scorer_baseline_procedure`、`qdp_provider_ingest_procedure`、`active_artifact_change_procedure`。

过程式写法应包含：

- `输入对象`
- `前置条件`
- `步骤`
- `验证`
- `写回`
- `副作用`

### 函数式：Function Layer
函数式负责纯判断和变换，不直接改文件或状态。它把任务、对象和证据变成结论，例如：

- `select_relevant_objects(task) -> objects`
- `activate_boundaries(objects, method) -> boundary_set`
- `classify_evidence(run) -> smoke|scout|evidence|promotion`
- `derive_next_action(state, evidence) -> next_method`

函数式判断应可复述、可测试、无隐藏副作用；它帮助 agent 灵活判断，而不是机械执行规则。

## 3. 7 个记忆区域
主脑和分脑仍共享 7 模块核；它们是记忆区域，由任务相关性决定读取范围：

- `identity_layer.md`：对象身份、目标函数和核心角色。
- `state_center.md`：当前对象实例、当前可调用方法和下一步函数输出。
- `knowledge_center.md`：对象类定义、长期事实、经验来源和稳定语义。
- `brain_architecture.md`：对象模型、模块边界和扩展方式。
- `operations_center.md`：过程式方法入口、环境基线和验证方式。
- `governance_layer.md`：受保护对象的不变量和激活逻辑。
- `episodic_memory.md`：时间顺序记忆，按需回溯。

## 4. 主脑与分脑
- 主脑维护共享对象、项目拓扑、跨项目关系和全局保护对象类型。
- 分脑维护本项目对象、当前状态、项目方法、证据索引和区域特化。
- 项目事实写在对应分脑；共享事实写在主脑；长证据写入 `references/`。
- `brain/brain_manifest.json` 是 child list、shared contract 和注册拓扑的结构源。

## 5. 扩展原则
- 新项目先初始化分脑对象，再注册到主脑 manifest 和 catalog。
- 新内容优先归入现有对象或现有记忆区域；确有新对象再命名。
- 长命令、长审计、完整实验过程和历史复盘下沉到 `references/`。
- 低风险语义整理在用户确认方向后可直接收敛；改变 active/live/canonical/PIT/secrets/external service 等受保护对象时，需要显式授权和对象级验证。

## 6. 工具地位
route、capsule、bootstrap、health、guard 和 skill 都是传感器或方法入口，不是上级流程。它们可以帮助识别对象和风险，但最终由 agent 根据用户目标、文件证据和对象语义判断。
