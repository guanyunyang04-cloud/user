# Daily Stock Analysis 治理对象

## Governed Objects
### object `product_vs_execution_boundary`
`scope`: product analysis workflows vs `daily_research` production execution.
`invariant`: product analysis outputs do not rewrite `daily_research` default execution.

### object `ai_compatibility_entries`
`scope`: `AGENTS.md`, `CLAUDE.md`, `.github` instructions and `SKILL.md`.
`invariant`: compatibility entries point to this brain and do not carry a separate truth system.

### object `public_user_docs`
`scope`: README, public docs, deployment and user-facing configuration text.
`invariant`: durable product facts and stable instructions live in brain/reference first; public files stay entry-oriented.

### object `secret_or_config_surface`
`scope`: API keys, account identifiers, model endpoints, notification credentials, ports and env-specific branches.
`invariant`: secrets and environment-specific private values stay out of brain and public docs; config fields update `.env.example`.

## Pure Functions
- `select_governed_object(task) -> product_boundary|ai_compat|public_docs|secret_config`
- `requires_ai_asset_check(change) -> bool`
- `requires_public_docs_reference(change) -> bool`

## Procedures
### procedure `ai_compat_change`
`input`: compatibility entry diff
`steps`: update brain semantics；update thin entry；run `scripts/check_ai_assets.py` and doc guard.

### procedure `public_docs_change`
`input`: user-facing docs change
`steps`: write stable content to `references/public_docs`；sync README entry；validate docs and impacted product surface.
