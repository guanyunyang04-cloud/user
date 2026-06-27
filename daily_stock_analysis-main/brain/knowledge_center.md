# Daily Stock Analysis 知识对象

## Object Classes
### class `multi_market_ai_stock_product`
`definition`: A/HK/US/US-index AI watchlist and analysis system.
`capabilities`: decision dashboard, buy/sell levels, checklist, market review, multi-channel notification, history, backtesting, holdings, Agent question answering, strategy library.
`entrypoints`: Web, Bot, API, Desktop and CLI.

### class `model_provider_surface`
`providers`: AIHubMix, Gemini, OpenAI-compatible, DeepSeek, Qwen, Claude, Ollama and LiteLLM-style configurations.
`invariant`: no secrets, account-specific endpoints or environment-specific branch logic in brain or public docs.

### class `market_data_surface`
`providers`: AkShare, Tushare, Pytdx, Baostock, YFinance.
`semantics`: non-critical third-party failures degrade gracefully when the main analysis chain can still run.

### class `notification_surface`
`providers`: WeCom, Feishu, Telegram, Discord, Slack, DingTalk, email, Pushover, PushPlus, Server Chan and custom webhook.

### class `ai_compatibility_entry`
`entries`: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md`, `SKILL.md`.
`semantics`: stable rules, validation matrix and product boundaries are stored in this brain; compatibility entries point here.

## Long-Term Lessons
- If AI compatibility assets do not point to this brain, future agents read the wrong entry.
- If the module map drifts from real directories, handoff and maintenance slow down.
- README is a user entrance; it should not become the AI handoff truth source.
- User-visible behavior, CLI/API, deployment, notification or report structure changes create durable product facts.

## Pure Functions
- `classify_docs_content(change) -> stable_fact|user_entry|release_note|temporary_note`
- `select_validation(surface) -> backend|web|desktop|ai_assets|docs|changed_surface`
- `requires_env_example_update(change) -> bool`
