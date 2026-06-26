# Daily Research 分脑架构

## 1. 区域定位
`daily_research/brain/` 是生产研究与执行分脑，负责研究对象、数据消费对象、执行对象、证据对象和接管记忆。

本分脑继承主脑的多范式自然语言程序模型：对象式描述项目和产物是什么，过程式描述对象方法怎么运行，函数式描述如何从任务和证据推导判断。

## 2. 共享脑核
本分脑采用主脑 manifest 定义的 7 模块核：identity、state、knowledge、architecture、operations、governance、episodic。

读取顺序、写回路由、routing hints 和可选模块由 `daily_research/brain/brain_manifest.json` 声明；本文件只解释区域特化。

## 3. 区域特化
- `identity_layer`：分脑身份对象、目标函数和长期不变量。
- `state_center`：当前对象实例、可调用方法、纯函数输出和下一步指针。
- `knowledge_center`：对象类定义、稳定事实、长期教训和研究线索引。
- `operations_center`：过程式方法入口、环境基线、高频命令和验证方式。
- `governance_layer`：对象级不变量、激活逻辑、guard 入口和写回路由。
- `episodic_memory` / `references/`：历史证据、dated review、长命令 transcript。

## 4. 可选补充
`brain_operating_protocol.md` 是 daily_research 的可选操作补充协议，不属于核心 7 模块，也不改变主脑共享契约。

## 5. 平台边界
`daily_research/brain/workflows/` 只声明 daily_research 项目 workflow；workspace governance workflow 位于 `brain/workflows/`。

`tools.brain.workflow` 输出运行态 capsule、预检和 freshness 传感器，不替代本分脑 Markdown 真源。
