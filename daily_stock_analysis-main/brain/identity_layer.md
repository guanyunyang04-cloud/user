# Daily Stock Analysis 身份对象

## object `daily_stock_analysis_product`
`type`: multi_market_product_brain
`definition`: 独立的多市场 AI 股票分析产品分脑，覆盖 CLI、FastAPI、Web、Desktop、Bot、Agent 和多数据源。
`not`: `daily_research` 的正式执行默认值或交易执行主线。
`principle`: Agent 无状态，项目大脑有状态。
`north_star`: 让复杂产品仓库在多入口、多模块条件下仍可稳定接管。
`methods`: `inspect_product_state()`；`select_body_surface(task)`；`run_product_validation(surface)`；`sync_ai_compat_entries()`。

## object `ai_handoff_surface`
`type`: compatibility_entry_surface
`entries`: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md`, `SKILL.md`
`semantics`: repository-native AI entries point to this brain; they are thin compatibility surfaces, not parallel truth sources.

## Pure Functions
- `select_body_surface(task) -> backend|api|web|desktop|bot|agent|data_provider|workflow|docs`
- `is_public_docs_change(task) -> bool`
- `requires_ai_asset_sync(change) -> bool`

## Routing
- Current product state: `state_center.md`
- Stable product facts and lessons: `knowledge_center.md`
- Body map and commands: `operations_center.md`
- Compatibility and product invariants: `governance_layer.md`
