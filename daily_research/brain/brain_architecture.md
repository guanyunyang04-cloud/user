# Daily Research Brain Architecture

## 1. 结构目标
`daily_research/brain/` 不是传统 README，而是项目分脑。它负责把项目知识拆成稳定事实、当前判决、方法规则、操作入口和时间证据，避免同一结论散落在多处重复维护。

## 2. 推荐读取顺序
1. `brain_manifest.json`
   - 机器可读入口、读写路由、模块清单
2. `semantic_memory.md`
   - 稳定事实与长期边界
3. `brain_architecture.md`
   - 文档结构与职责
4. `project_map.md`
   - 项目结构、主线地图、决策闭环
5. `working_memory.md`
   - 当前默认值、优先级、边界
6. `procedural_memory.md`
   - 可复用方法学
7. `environment_model.md`
   - 环境、解释器、编码口径
8. `action_system.md`
   - 高频命令与执行入口
9. `episodic_memory.md`
   - 历史过程与原始证据

## 3. 模块职责
- `semantic_memory.md`
  - 只写稳定事实、固定边界、长期有效语义
- `project_map.md`
  - 只写项目结构、主线地图、决策闭环、当前瓶颈
- `working_memory.md`
  - 只写当前判决、当前默认值、当前优先级
- `procedural_memory.md`
  - 只写“怎么做”的规则，不写单次结论
- `environment_model.md`
  - 只写运行环境、解释器、编码与工具基线
- `action_system.md`
  - 只写可直接执行的高频入口
- `episodic_memory.md`
  - 只写按时间顺序保存的过程证据

## 4. 写入路由
- 新证据先进入 `working_memory.md`。
- 证据稳定后再沉淀进 `semantic_memory.md`。
- 单次实验细节、路径、时间序列过程进入 `episodic_memory.md`。
- 命令口径变化进入 `action_system.md`。
- 结构变化先改本文件，再改 `brain_manifest.json`。

## 5. 维护原则
- 同一件事只保留一个真源。
- `working_memory.md` 可以快，但不能和 `semantic_memory.md` 冲突。
- `project_map.md` 必须反映当前主线，不能停留在旧执行法或旧瓶颈。
- `action_system.md` 只保留仍在使用的入口，不堆历史命令。
- `episodic_memory.md` 允许保留旧结论，但必须以时间顺序呈现，不得冒充当前默认值。
