# Main Brain Architecture

## 1. 结构目标
主脑负责把整个工作区组织成“可治理的脑网络”，而不是零散的人类文档。

核心原则：

- 主脑掌管分脑
- 分脑只维护本项目内部认知
- 行动系统、环境、方法学、时间记忆分层保存
- 不再把同一信息写进多个入口文档
- 每个项目都必须形成“body <-> brain”镜像关系
- 后续 agent 应通过 brain 接管项目，而不是重新从 body 里盲扫

## 2. 层级结构
### 2.1 主脑
- `brain/master_brain.md`
  - 工作区身份、分脑拓扑、全局边界
- `brain/working_memory.md`
  - 当前跨项目优先级与治理决策
- `brain/procedural_memory.md`
  - 跨项目方法学、Gemini 协同、写入规则
- `brain/environment_model.md`
  - 工作区运行环境、共享工具、根级命令口径
- `brain/brain_manifest.json`
  - 机器可读拓扑与控制关系

### 2.2 `daily_research` 分脑
- 语义记忆：`daily_research/brain/semantic_memory.md`
- 项目地图：`daily_research/brain/project_map.md`
- 工作记忆：`daily_research/brain/working_memory.md`
- 程序记忆：`daily_research/brain/procedural_memory.md`
- 环境模型：`daily_research/brain/environment_model.md`
- 情景记忆：`daily_research/brain/episodic_memory.md`
- 行动系统：`daily_research/brain/action_system.md`

### 2.3 `t0_project` 分脑
- 语义记忆：`t0_project/brain/semantic_memory.md`
- 工作记忆：`t0_project/brain/working_memory.md`
- 程序记忆：`t0_project/brain/procedural_memory.md`
- 环境模型：`t0_project/brain/environment_model.md`
- 情景记忆：`t0_project/brain/episodic_memory.md`
- 行动系统：`t0_project/brain/action_system.md`

### 2.4 `daily_stock_analysis-main` 分脑
- 语义记忆：`daily_stock_analysis-main/brain/semantic_memory.md`
- 工作记忆：`daily_stock_analysis-main/brain/working_memory.md`
- 程序记忆：`daily_stock_analysis-main/brain/procedural_memory.md`
- 环境模型：`daily_stock_analysis-main/brain/environment_model.md`
- 情景记忆：`daily_stock_analysis-main/brain/episodic_memory.md`
- 行动系统：`daily_stock_analysis-main/brain/action_system.md`

## 3. 控制关系
- 主脑可以规定分脑结构、路由、命名与协作方式
- 分脑不能越权改写其它分脑的默认结论
- 涉及跨项目边界的结论，先写主脑，再写对应分脑
- 主脑要求每个分脑提供 body_map 与 handoff_contract

## 4. 写入路由
- 工作区拓扑、跨项目边界：
  - `brain/master_brain.md`
- 当前全局优先级：
  - `brain/working_memory.md`
- 跨项目方法学与协作技能：
  - `brain/procedural_memory.md`
- 具体项目内部结论：
  - 写进对应分脑

## 5. Agent 接脑协议
- agent 接手项目时，默认读取顺序：
  - `brain/brain_manifest.json`
  - 对应子项目 `brain/brain_manifest.json`
  - 子项目 `semantic_memory.md`
  - 子项目 `working_memory.md`
  - 子项目 `procedural_memory.md`
  - 子项目 `environment_model.md`
  - 子项目 `action_system.md`
- 只有在 brain 指向具体 body 模块后，才进入源码

## 6. 去冗余规则
- 不再维护根 `README.md`
- 不再维护项目级 `README.md`
- 不再维护子目录 `execution/README.md`
- 行动细节统一收口到各项目 brain 的 `action_system.md`
