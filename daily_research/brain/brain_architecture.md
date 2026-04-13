# Daily Research 分脑架构

## 1. 区域定位
`daily_research/brain/` 是当前工作区最重的生产型分脑。

它负责：

- strongest-model
- learned-control
- live 默认执行
- 研究到执行的正式闭环

## 2. 内部中枢
本分脑只保留以下权威中枢：

- `identity_layer.md`
- `state_center.md`
- `knowledge_center.md`
- `brain_architecture.md`
- `operations_center.md`
- `governance_layer.md`
- `episodic_memory.md`

## 3. 当前读取原则
- 默认先读 `identity -> state -> knowledge -> operations`
- 遇到流程或纪律问题再读 `governance`
- 需要历史证据时才下钻 `episodic`

## 4. 区域特化
- `state_center`
  - strongest-model、learned-control、live 当前答案与当前主问题
- `knowledge_center`
  - 正式协议、硬规则、长期教训
- `operations_center`
  - body 地图、环境口径、高频命令、写回路由

## 5. 去冗余原则
- 不再把“当前状态”拆成 handoff / temporal / working 三份
- 不再把“长期知识”拆成 semantic / rule / lesson 三份
- 不再把“执行入口”拆成 project_map / procedural / action / environment / handoff_rules 五份
