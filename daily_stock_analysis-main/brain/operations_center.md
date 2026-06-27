# Daily Stock Analysis 过程目录

## Body Map Objects
- `backend_surface`: `daily_stock_analysis-main/src`, `daily_stock_analysis-main/api`
- `frontend_desktop_surface`: `daily_stock_analysis-main/apps`
- `bot_agent_surface`: `daily_stock_analysis-main/bot`, `daily_stock_analysis-main/src/agent`
- `data_provider_surface`: `daily_stock_analysis-main/data_provider`
- `ops_surface`: `daily_stock_analysis-main/scripts`, `daily_stock_analysis-main/.github/workflows`
- `tests_surface`: `daily_stock_analysis-main/tests`
- `public_docs_surface`: `daily_stock_analysis-main/README.md`, canonical body in `brain/references/public_docs/`
- `ai_compat_surface`: `AGENTS.md`, `CLAUDE.md`, `.github/*instructions*`, `SKILL.md`

## Procedure Entries
### procedure `enter_product_surface`
`input`: task
`steps`: select body surface；inspect corresponding code/docs；change scoped files；run surface validation；write durable facts when public behavior or AI entry changes.

### procedure `changed_surface_validation`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`
`semantics`: use returned blocking commands or focused manual validation.

### procedure `backend_validation`
`commands`: `python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`; `python -m pytest -m "not network"`

### procedure `web_desktop_validation`
`commands`: `npm ci`; `npm run lint`; `npm run test`; `npm run build`; `npm run test:smoke`; `powershell -ExecutionPolicy Bypass -File scripts/build-backend.ps1`; `powershell -ExecutionPolicy Bypass -File scripts/build-desktop.ps1`
`note`: desktop build may require Windows Developer Mode / electron-builder symlink support and network availability.

### procedure `ai_asset_sync`
`commands`: `python scripts/check_ai_assets.py`; `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`
`activation`: AI compatibility entry changes or stable AI handoff semantics change.

### procedure `public_docs_sync`
`input`: README, docs, deployment, user-visible config or feature text.
`steps`: write durable body to `brain/references/public_docs/`；keep README as user entry；update `.env.example` when config fields change.

## Writeback Routes
- Current product objects: `state_center.md`
- Stable product facts and lessons: `knowledge_center.md`
- Body map, commands and validation: `operations_center.md`
- Compatibility invariants: `governance_layer.md`
- Chronological changes: `episodic_memory.md`
