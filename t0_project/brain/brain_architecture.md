# T0 Project Brain Architecture

## 1. 结构目标
`t0_project/brain/` 是 `t0_project` 的分脑，由主脑治理，用来保存盘中实验项目的长期认知和安全边界。

## 2. 分脑模块
- `semantic_memory.md`
  - 项目身份、稳定边界、body_map 摘要
- `working_memory.md`
  - 当前优先级、实验边界、隔离要求
- `procedural_memory.md`
  - 实验纪律、写入路由、升级约束
- `environment_model.md`
  - 运行环境、维护命令
- `action_system.md`
  - 执行抽象层、运行模式、接入顺序
- `episodic_memory.md`
  - 时间顺序实验记录
- `brain_manifest.json`
  - 父子脑关系、读写路由、body_map、handoff_contract

## 3. 写入路由
- 稳定认知：
  - `semantic_memory.md`
- 当前优先级：
  - `working_memory.md`
- 可复用方法：
  - `procedural_memory.md`
- 环境与命令：
  - `environment_model.md`
- 执行抽象与操作链路：
  - `action_system.md`
- 单轮实验：
  - `episodic_memory.md`

## 4. 去冗余规则
- 不再维护 `t0_project/README.md` 作为 AI 入口
- 不再维护 `t0_project/execution/README.md` 作为 AI 入口
- 行动细节统一收口到 `action_system.md`
