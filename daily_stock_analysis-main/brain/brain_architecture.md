# 每日股票分析脑架构

`daily_stock_analysis-main/brain/` 继承主脑多范式自然语言程序模型：对象描述产品面和兼容入口，过程描述验证/同步/发布步骤，函数描述任务属于哪个产品 surface。

## Object Layer
- `daily_stock_analysis_product`: multi-market product brain.
- `ai_handoff_surface`: repository-native AI compatibility entries.
- `public_docs_surface`: README and public documentation entry.
- `product_surface`: backend, API, Web, Desktop, Bot, Agent, data provider, workflow.

## Procedure Layer
- `enter_product_surface`
- `changed_surface_validation`
- `ai_asset_sync`
- `public_docs_sync`
- `web_desktop_validation`

## Function Layer
- `select_body_surface(task)`
- `requires_ai_asset_sync(change)`
- `requires_public_docs_reference(change)`
- `select_validation(surface)`

## Expansion
New product surfaces first extend body map, procedure entries or references; structure changes stay aligned with main brain manifest.
