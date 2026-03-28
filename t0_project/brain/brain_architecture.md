# T0 Project Brain Architecture

## 1. 结构目标
`t0_project/brain/` 是 `t0_project` 分脑，由主脑治理，用来保存盘中实验项目的长期认知。

上级主脑位于：

- `brain/master_brain.md`

## 2. 分脑模块
- `t0_project/brain/semantic_memory.md`
  - 项目身份、边界、入口
- `t0_project/brain/working_memory.md`
  - 当前优先级和安全边界
- `t0_project/brain/procedural_memory.md`
  - 可复用操作规则与实验纪律
- `t0_project/brain/environment_model.md`
  - 运行环境和维护命令
- `t0_project/brain/action_system.md`
  - 执行抽象层与交易接口设计
- `t0_project/brain/episodic_memory.md`
  - 时间顺序实验记录
- `t0_project/brain/brain_manifest.json`
  - 机器可读索引

## 3. 去冗余规则
- 不再维护 `t0_project/README.md`
- 不再维护 `t0_project/execution/README.md`
- 行动细节统一收口到 `action_system.md`
