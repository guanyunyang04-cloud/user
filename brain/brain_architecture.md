# 主脑架构

## 1. 目标
整个项目的 brain 不再追求“文档细分完整”，而追求“接管路径最短、职责边界最清、长期维护最稳”。

核心原则：

- `Agent 无状态，项目大脑有状态`
- 主脑定义共享脑核
- 分脑只保留区域特化
- 同一种功能只保留一个权威中枢

## 2. 精炼脑核
现在每个分脑统一收敛为 `7` 个核心模块：

1. `identity_layer.md`
   - 我是谁、追求什么、不能做什么
2. `state_center.md`
   - 当前状态、当前问题、当前优先级、当前时态、当前 handoff
3. `knowledge_center.md`
   - 稳定事实、硬规则、长期教训
4. `brain_architecture.md`
   - 结构合同、模块边界、读取原则
5. `operations_center.md`
   - body 地图、环境基线、命令入口、流程与写回路由
6. `governance_layer.md`
   - 治理闭环、接管纪律、反偏移机制
7. `episodic_memory.md`
   - 时间顺序证据库，按需下钻

## 3. 标准读取顺序
附着分脑默认读取顺序统一为：

1. `brain_manifest.json`
2. `identity_layer.md`
3. `state_center.md`
4. `knowledge_center.md`
5. `brain_architecture.md`
6. `operations_center.md`
7. `governance_layer.md`
8. `episodic_memory.md`

读取原则：

- 默认只读到能完成接管为止
- `episodic_memory.md` 不是默认入口
- 不再为了“兼容旧拆分”维持平行文档

## 4. 主脑与分脑分工
- 主脑：
  - 只维护共享脑核、跨项目边界、拓扑和统一治理
- 分脑：
  - 只维护本项目状态、知识、入口和区域差异
- 任何共享结构只在主脑写一次
- 任何项目事实只在对应分脑写一次

## 5. 去冗余规则
- 不再保留 `handoff / working / temporal` 三份平行现状文档
- 不再保留 `semantic / rule / lesson` 三份平行长期记忆文档
- 不再保留 `project_map / procedural / environment / action / handoff_rules` 五份平行操作文档
- 新文档默认先判断能否并入现有中枢
- 只有当信息类型无法归入现有中枢时，才允许新增模块
