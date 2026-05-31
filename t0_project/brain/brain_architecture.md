# T0 项目脑架构

## 1. 区域定位
`t0_project/brain/` 是实验型分脑，负责盘中执行实验、执行抽象和 RL 原型。

## 2. 共享脑核
本分脑采用主脑 manifest 定义的 7 模块核：identity、state、knowledge、architecture、operations、governance、episodic。

读取顺序、写回路由、routing hints 和 body map 由 `t0_project/brain/brain_manifest.json` 声明；本文件只解释区域特化。

## 3. 区域特化
- `state_center`
  - 当前实验定位、优先级和风险
- `knowledge_center`
  - 实验边界、硬规则、长期教训
- `operations_center`
  - 真实 body 入口、命令和写回路由

## 4. 扩展原则
新增实验线优先写入现有 7 模块或 `references/`；只有无法归入当前模块时才提案新增结构。
