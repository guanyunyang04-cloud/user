# Quant Data Platform 治理对象

## Governed Objects
### object `raw_market_data`
`scope`: external provider raw data before/after lake archive.
`invariant`: preserve original OHLCV units and adjustment state; derived adjusted fields are sidecars.

### object `historical_semantics`
`scope`: historical universe, industry, status, valuation, index constituents and disclosure availability.
`invariant`: current snapshots do not backfill historical semantics without PIT/available-time proof.

### object `coverage_status`
`scope`: domain coverage, missing, blocked, partial and completed status.
`invariant`: required blocked domains are visible and do not register as complete research-consumable datasets.

### object `registry_pointer`
`scope`: active canonical, policy bundle, memmap and training pack pointers.
`invariant`: pointer mutation records source bundle/signature, replacement pointer and sample validation.

### object `data_cleanup`
`scope`: old parquet, old bundle, old `.dat`, old training pack and duplicate assets.
`invariant`: unique data or assets without replacement remain protected; duplicates can be cleaned after dry-run and validation.

### object `slow_disclosure_domain`
`scope`: financial quarters, forecasts/express reports, announcements and PDFs.
`invariant`: model-facing use waits for `publish_date / available_date` or equivalent PIT-safe audit.

## Pure Functions
- `select_governed_object(task) -> object`
- `requires_replacement_pointer(object, method) -> bool`
- `classify_domain_status(domain) -> complete|partial|blocked|exploratory`
- `can_register_for_research(bundle) -> bool`

## Procedures
### procedure `registry_pointer_change`
`input`: new pointer, source bundle/signature, validation sample.
`steps`: inspect current pointer；compare source signature；validate sample；write registry and state summary.

### procedure `provider_domain_promotion`
`input`: provider probe result and target domain.
`steps`: audit unit/PIT/coverage/stability；archive raw；define canonical transform；register domain status.

### procedure `cleanup_governed_asset`
`input`: cleanup target.
`steps`: dry-run；replacement pointer check；sample validation；delete/archive；write audit result.

## Writeback
- Project facts: this brain.
- Cross-project topology: main brain.
- Long audits, provider reports and cleanup reports: `references/`, `data/audits`, or `data/provider_eval`.
