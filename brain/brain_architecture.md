# 主脑架构

## 1. 真源边界
`brain/brain_manifest.json` 是 boot、read、write、attach、shared contract、child list 的单一权威源。

本文件只解释结构原则，不维护读取顺序、子脑清单或入口命令；这些由 manifest、catalog 和 runtime helper 生成。

## 2. 统一 7 模块核
主脑和附着子脑共享同一 7 模块核：

- `identity_layer.md`：身份、目标、禁区。
- `state_center.md`：当前状态、优先级、handoff。
- `knowledge_center.md`：稳定事实、硬规则、长期教训。
- `brain_architecture.md`：结构解释、模块边界、扩展原则。
- `operations_center.md`：body 地图、环境基线、命令入口、写回路由。
- `governance_layer.md`：治理闭环、接管纪律、反偏移机制。
- `episodic_memory.md`：时间顺序证据库，按需下钻。

## 3. 主脑与子脑
- 主脑维护共享脑核、跨项目边界、拓扑、注册和治理规则。
- 子脑维护本项目状态、知识、入口、证据和区域特化。
- 共享结构只在主脑写一次；项目事实只在对应子脑写一次。
- `daily_research/brain/brain_operating_protocol.md` 是可选补充协议，不属于核心模块。

## 4. 扩展原则
- 新项目先生成 7 模块 skeleton，再通过 runtime `register` 写入主脑 child list 和 catalog。
- 新模块默认不新增；先判断能否并入现有 7 模块。
- 长证据、命令 transcript、dated review 放入 `references/`，核心中枢只保留当前索引和稳定结论。
- 自进化保持 proposal-only：低风险观察可进入 proposal 队列，协议或行为变更必须用户批准后实施。

## 5. 运行边界
- routing target 只允许 `workspace`、已附着 child brain 或 `ambiguous`。
- `workspace_governance` 是 workspace workflow domain 和 bootstrap alias，不是 child brain id。
- workflow JSON / CLI 输出是运行态传感器，不替代 Markdown 与 manifest 的权威写回。
