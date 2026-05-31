# Daily Research 分脑架构

## 1. 区域定位
`daily_research/brain/` 是生产研究与执行分脑，负责 strongest-model、learned-control、live 默认执行和研究到执行闭环。

## 2. 共享脑核
本分脑采用主脑 manifest 定义的 7 模块核：identity、state、knowledge、architecture、operations、governance、episodic。

读取顺序、写回路由、routing hints 和可选模块由 `daily_research/brain/brain_manifest.json` 声明；本文件只解释区域特化。

## 3. 区域特化
- `state_center`：strongest-model、learned-control、live 当前答案与当前主问题。
- `knowledge_center`：正式协议、硬规则、长期教训。
- `operations_center`：body 地图、环境口径、高频命令、写回路由。
- `episodic_memory` / `references/`：历史证据、dated review、长命令 transcript。

## 4. 可选补充
`brain_operating_protocol.md` 是 daily_research 的可选操作补充协议，不属于核心 7 模块，也不改变主脑共享契约。

## 5. 平台边界
`daily_research/brain/workflows/` 只声明 daily_research 项目 workflow；workspace governance workflow 位于 `brain/workflows/`。

`tools.brain.workflow` 输出运行态 capsule、预检和 freshness 传感器，不替代本分脑 Markdown 真源。
