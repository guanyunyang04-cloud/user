# T0 Project Brain Architecture

## 1. 区域定位
`t0_project/brain/` 是实验型分脑。

它负责：

- 盘中执行实验
- 执行抽象
- RL 原型

## 2. 内部中枢
- `identity_layer.md`
- `state_center.md`
- `knowledge_center.md`
- `brain_architecture.md`
- `operations_center.md`
- `governance_layer.md`
- `episodic_memory.md`

## 3. 区域特化
- `state_center`
  - 当前实验定位、优先级和风险
- `knowledge_center`
  - 实验边界、硬规则、长期教训
- `operations_center`
  - 真实 body 入口、命令和写回路由
