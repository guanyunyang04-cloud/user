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
- `episodic_memory.md` 是轻量入口；完整历史原文与标题索引放在 `brain/references/`

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
- 主文件只放当前状态、稳定规则、操作入口和最新复盘摘要；dated log、长命令、历史实验细节必须进入 `daily_research/brain/references/` 归档。
- 归档不等于删除：主文件保留索引，完整历史快照保存在 `*_archive_*.md`、`*_history_raw_*.md` 或 `*_evidence_index_*.md`。

## 6. 平台化边界
- `daily_research/brain/*.md` 仍是权威项目事实层。
- `daily_research/brain/workflows/` 只声明 daily_research 项目工作流，不新增第 8 个中枢。
- `tools.brain.workflow` 输出的是主脑接管胶囊、预检状态、产物 freshness 和写回计划。
- 主脑 workflow 输出默认落在 `brain/output/brain_workflow/`，不得被当作 brain 主文件的替代事实。
