# Daily Research 分脑架构

## 1. 结构目标
`daily_research/brain/` 不是传统 README，而是项目分脑。它负责把项目知识拆成稳定事实、当前判决、方法规则、操作入口和时间证据，避免同一结论散落在多处重复维护。

当前升级目标是把它从“有记忆的分脑”继续升级成“更像人脑工作流的项目认知中枢”：

- 身份层负责回答“我是谁、追求什么、不能做什么”
- 记忆层负责规则、事实、教训与时间证据
- 认知层负责 `过去 / 当下 / 未来`
- 决策层负责当前最值得做什么
- 执行层负责统一接管协议
- 治理层负责自检、反偏移、修复和写回

## 2. 推荐读取顺序
1. `brain_manifest.json`
   - 机器可读入口、读写路由、模块清单
2. `identity_layer.md`
   - 项目身份、北极星、成功标准、禁区
3. `handoff_packet.md`
   - 当前标准交接包，先给接管者最小完备状态
4. `semantic_memory.md`
   - 稳定事实与长期边界，尤其是研究模型与执行模型的硬协议
5. `rule_memory.md`
   - 宪法级 / 策略级 / 经验级规则
6. `lesson_memory.md`
   - 已沉淀教训与预防资产
7. `temporal_state.md`
   - `Past Ledger / Present State / Future Map`
8. `brain_architecture.md`
   - 文档结构、职责和接管顺序
9. `project_map.md`
   - 双环闭环、主线地图、当前瓶颈
10. `working_memory.md`
   - 当前窗口、当前默认值、当前优先级
11. `procedural_memory.md`
   - 可复用方法学与写回纪律
12. `handoff_rules.md`
   - 接管纪律、前台 / 续训 / 超时预算
13. `governance_layer.md`
   - 自检、漂移检测、执行前四检、修复分级
14. `environment_model.md`
   - 环境、解释器、编码口径
15. `action_system.md`
   - 高频命令与引用顺序
16. `episodic_memory.md`
   - 历史过程与原始证据

## 3. 模块职责
- `identity_layer.md`
  - 只写项目使命、目标、成功标准、硬约束与禁区
- `semantic_memory.md`
  - 只写稳定事实、固定边界、长期有效语义
  - 尤其负责锁定 formal / recent / live 的定义，以及“formal 每窗最新模型 + recent 必报 + strongest model 可直达执行默认”的硬协议
- `rule_memory.md`
  - 只写高优先级规则，不混入单次实验叙事
- `lesson_memory.md`
  - 只写可复用教训，避免经验埋在 episodic 里
- `temporal_state.md`
  - 统一记录 `Past Ledger / Present State / Future Map`
- `project_map.md`
  - 只写项目结构、主线地图、决策闭环、当前瓶颈
  - 尤其负责把研究环和执行环的关系讲清楚
- `working_memory.md`
  - 只写当前判决、当前默认值、当前窗口、当前优先级
- `procedural_memory.md`
  - 只写“怎么做”的规则，不写单次结论
- `environment_model.md`
  - 只写运行环境、解释器、编码与工具基线
- `action_system.md`
  - 只写可直接执行的高频入口和引用顺序
- `episodic_memory.md`
  - 只写按时间顺序保存的过程证据
- `handoff_packet.md`
  - 只写标准交接包，不展开完整历史
- `governance_layer.md`
  - 只写自检、反偏移、修复和执行前后闭环

## 4. 接管最小流程
- 第一步：先在 `identity_layer.md` 和 `handoff_packet.md` 确认项目身份与当前状态。
- 第二步：在 `semantic_memory.md`、`rule_memory.md`、`lesson_memory.md` 确认硬协议、关键规则和最近教训。
- 第三步：在 `temporal_state.md` 和 `working_memory.md` 确认当前讨论的是 formal、recent、live 还是 promotion，以及当前最重要任务。
- 第四步：如果需要操作，再到 `action_system.md` 找入口；如果需要改规则，再到 `procedural_memory.md` 与 `governance_layer.md`。
- 第五步：如果需要追溯原因或核对旧结论，再去 `episodic_memory.md`。

## 5. 写入路由
- 新的项目身份、禁区与成功标准先进入 `identity_layer.md`。
- 新证据先进入 `working_memory.md`。
- 证据稳定后再沉淀进 `semantic_memory.md`。
- 新规则先进入 `rule_memory.md`，再同步方法学。
- 新教训先进入 `lesson_memory.md`，再按需要升级到规则或 SOP。
- 当前 `过去 / 当下 / 未来` 统一写进 `temporal_state.md`。
- 当前交接摘要写进 `handoff_packet.md`。
- 单次实验细节、路径、时间序列过程进入 `episodic_memory.md`。
- 命令口径变化进入 `action_system.md`。
- 结构变化先改本文件，再改 `brain_manifest.json`。

## 6. 维护原则
- 同一件事只保留一个真源。
- `working_memory.md` 可以快，但不能和 `semantic_memory.md` 冲突。
- `project_map.md` 必须反映当前主线，不能停留在旧执行法或旧瓶颈。
- `action_system.md` 只保留仍在使用的入口，不堆历史命令。
- `episodic_memory.md` 允许保留旧结论，但必须以时间顺序呈现，不得冒充当前默认值。
- 任何 agent 接管时，都必须先分清研究环与执行环；如果混报 formal 证据和 production 证据，视为分脑使用错误。
- 当前默认接管不再从 `episodic_memory.md` 开始，而是从 `handoff_packet.md` 与 `temporal_state.md` 开始。
