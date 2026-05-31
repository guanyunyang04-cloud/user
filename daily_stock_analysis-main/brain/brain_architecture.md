# 每日股票分析脑架构

## 1. 区域定位
`daily_stock_analysis-main/brain/` 是独立产品型分脑，负责多市场 AI 股票分析产品的稳定认知、产品代码入口和仓库级 AI 接管真源。

## 2. 共享脑核
本分脑采用主脑 manifest 定义的 7 模块核：identity、state、knowledge、architecture、operations、governance、episodic。

读取顺序、写回路由、routing hints 和 body map 由 `daily_stock_analysis-main/brain/brain_manifest.json` 声明；本文件只解释区域特化。

## 3. 区域特化
- `state_center`
  - 当前产品边界、优先级和 handoff
- `knowledge_center`
  - 产品稳定事实、硬规则和教训
- `operations_center`
  - 目录地图、验证入口和写回路由

## 4. 扩展原则
新产品面或外部集成优先进入 body map、operations 或 references；结构变更保持 proposal-only。
