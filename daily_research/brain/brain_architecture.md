# Daily Research 分脑架构

## 1. 结构目标
`daily_research/brain/` 不是传统 README，而是项目分脑。它负责把项目知识拆成稳定事实、当前判决、方法规则、操作入口和时间证据，避免同一结论散落在多处重复维护。

## 2. 推荐读取顺序
1. `brain_manifest.json`
   - 机器可读入口、读写路由、模块清单
2. `semantic_memory.md`
   - 稳定事实与长期边界，尤其是研究模型与执行模型的硬协议
3. `brain_architecture.md`
   - 文档结构、职责和接管顺序
4. `project_map.md`
   - 双环闭环、主线地图、当前瓶颈
5. `working_memory.md`
   - 当前窗口、当前默认值、当前优先级
6. `procedural_memory.md`
   - 可复用方法学与写回纪律
7. `environment_model.md`
   - 环境、解释器、编码口径
8. `action_system.md`
   - 高频命令与引用顺序
9. `episodic_memory.md`
   - 历史过程与原始证据

## 3. 模块职责
- `semantic_memory.md`
  - 只写稳定事实、固定边界、长期有效语义
  - 尤其负责锁定 formal / recent / live 的定义，以及研究模型与执行模型的边界
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

## 4. 接管最小流程
- 第一步：先在 `semantic_memory.md` 确认硬协议，不要直接根据当前产物猜规则。
- 第二步：在 `working_memory.md` 确认当前讨论的是 formal、recent 还是 live。
- 第三步：如果需要操作，再到 `action_system.md` 找入口；如果需要改规则，再到 `procedural_memory.md`。
- 第四步：如果需要追溯原因或核对旧结论，再去 `episodic_memory.md`。

## 5. 写入路由
- 新证据先进入 `working_memory.md`。
- 证据稳定后再沉淀进 `semantic_memory.md`。
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
