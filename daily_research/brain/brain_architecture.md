# Daily Research Brain Architecture

## 1. 结构目标
`daily_research/brain/` 是 `daily_research` 分脑，不再承担传统 README 职责，而是按脑模块分层保存认知。

它由上级主脑治理：

- `brain/master_brain.md`
- `brain/brain_manifest.json`

## 2. 分脑分层
### 2.1 语义记忆
- `daily_research/brain/semantic_memory.md`
  - 当前稳定主线、项目身份、脑模块目录

### 2.2 项目地图
- `daily_research/brain/project_map.md`
  - 项目背景、主线演化、瓶颈、未来方向

### 2.3 工作记忆
- `daily_research/brain/working_memory.md`
  - 当前默认值、升级 gate、优先级、停止规则

### 2.4 程序记忆
- `daily_research/brain/procedural_memory.md`
  - 已验证方法学、写入路由、以及 Gemini 协作尝试的停用结论

### 2.5 环境模型
- `daily_research/brain/environment_model.md`
  - 解释器、依赖、命令与工具入口

### 2.6 行动系统
- `daily_research/brain/action_system.md`
  - 盘后执行流程、模型更新、计划生成、执行边界

### 2.7 情景记忆
- `daily_research/brain/episodic_memory.md`
  - 按时间顺序保存实验、产物、证据与结论

### 2.8 机器索引
- `daily_research/brain/brain_manifest.json`
  - 机器可读读写路由与父子脑关系

## 3. 写入路由
- 当前稳定状态：
  - `semantic_memory.md`
- 项目演化与瓶颈：
  - `project_map.md`
- 当前默认决策与优先级：
  - `working_memory.md`
- 可复用方法学：
  - `procedural_memory.md`
- 环境与命令口径：
  - `environment_model.md`
- 执行流程：
  - `action_system.md`
- 单轮实验与时间证据：
  - `episodic_memory.md`

## 4. 去冗余规则
- 不再维护 `daily_research/README.md`
- 不再维护 `daily_research/execution/README.md`
- 不再把执行口径、环境、默认决策重复写进多个入口
- 结构变更先改本文件，再改 `brain_manifest.json`
