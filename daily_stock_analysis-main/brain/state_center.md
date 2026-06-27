# Daily Stock Analysis 状态程序

## Object Instances
### object `daily_stock_analysis_product`
`type`: multi_market_product_instance
`state`: 独立产品分脑；不接管 `daily_research` 正式执行默认值。
`current_truth_source`: this brain for AI handoff; main brain for workspace-level topology and cross-project governance.

### object `ai_compat_entries`
`type`: compatibility_surface
`state`: `AGENTS.md / CLAUDE.md / SKILL.md / .github instructions` keep thin entry roles and point to brain.
`sync_status`: 2026-05-11 AI asset check passed; no new parallel truth source found.

### object `public_docs_surface`
`type`: public_user_entry
`state`: README keeps thin user entry, install entry and disclaimer; durable product docs live in `brain/references/public_docs/`.
`activation`: public docs, README, installation, deployment, user-facing feature explanation.

### object `validation_20260511`
`type`: validation_evidence
`state`: backend syntax, critical flake8, deterministic checks, offline pytest passed; Web lint/test/build/smoke passed; backend PyInstaller build passed.
`open_issue`: Windows desktop build still depends on Developer Mode / electron-builder symlink and network download conditions.

## Pure Functions
- `select_product_surface(task)`: maps task to backend/API/web/desktop/bot/data/docs/AI-compat.
- `derive_next_action(state)`: keep product brain and AI compatibility entries synchronized; desktop build should be retested in Developer Mode or equivalent CI.
- `classify_validation(result)`: separates product code validation, web validation, build environment issue and release evidence.

## Procedures
### procedure `product_change`
`input`: task and selected product surface
`steps`: inspect body map；change scoped files；run surface validation；write durable product facts when behavior or entrypoints change.

### procedure `public_docs_change`
`input`: README or user-facing docs change
`steps`: update canonical public docs in `brain/references/public_docs/`；keep README as thin entry；run docs/AI guard as needed.
