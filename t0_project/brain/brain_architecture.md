# T0 Project Brain Architecture

## 1. 分脑定位
`t0_project/brain/` 是 `t0_project` 的实验型分脑，由主脑治理。

它在整个脑网络中的角色是：

- 保存盘中实验、执行抽象和 RL 原型的长期认知
- 与 `daily_research` 的正式生产主线保持清晰隔离
- 为后续接管者提供实验边界、安全边界和 body 入口

## 2. 继承的标准合同
`t0_project` 继承主脑定义的标准附着分脑合同，统一结构与读写顺序以 [brain/brain_architecture.md](H:/new_tdx64/PYPlugins/user/brain/brain_architecture.md) 为准。

因此本文件不再重复展开整套通用模块定义，只保留本项目相对标准合同的特有强调。

## 3. 本项目的特有强调
- `working_memory.md`
  - 必须始终强调与 `daily_research` 的隔离边界
- `procedural_memory.md`
  - 重点沉淀实验纪律、RL 原型边界和接入限制
- `action_system.md`
  - 重点保留 `strategy / execution / rl / gateway` 的进入顺序

## 4. body 进入顺序
本项目的 body 主要从以下区域进入：

- `t0_project/strategy`
- `t0_project/execution`
- `t0_project/rl`
- `t0_project/gateway`

默认原则：

- 先确认实验边界和隔离规则
- 再进入具体代码目录

## 5. 去冗余原则
- 通用分脑结构只在主脑 `brain_architecture.md` 定义一次。
- 本文件只保留 `t0_project` 的实验边界和 body 差异。
