# Quant Data Platform 治理对象
快照日期：`2026-07-01`

## Governed Objects
### object `active_manifest`
`scope`: `quant_data_platform/data/qdp_v2/active/active.json`
`invariant`: active switch is atomic；must point only to readable dataset manifests.

### object `dataset_manifest`
`scope`: `quant_data_platform/data/qdp_v2/datasets/<domain>/<dataset_id>/dataset.json`
`invariant`: schema, primary key, date range, row count, shard paths and quality evidence describe the parquet facts.

### object `raw_market_data`
`scope`: `market_daily_raw`、`market_intraday_1m` and provider raw/staging data.
`invariant`: preserve original OHLCV units and adjustment state; derived adjusted fields or features are separate outputs.

### object `pit_historical_semantics`
`scope`: trading calendar, universe, security status, valuation, index constituents, industry/concept and event availability.
`invariant`: current snapshots do not backfill historical semantics without PIT/available-time proof.

### object `rebuildable_cache`
`scope`: 5m bars, daily panel, intraday daily features, limit intraday features.
`invariant`: cache/derived tables must be reproducible from active raw facts or an explicit source dataset.

### object `data_cleanup`
`scope`: unreferenced dataset dirs, archived repair outputs, staging outputs.
`invariant`: unique data or assets without replacement remain protected; deletion follows active graph dry-run and explicit `--delete --yes`.

### object `archived_v1_surface`
`scope`: old lake/catalog/canonical/memmap/daily-update/repair commands.
`invariant`: archived code can be referenced for history but should not re-enter public CLI without explicit redesign.

## Pure Functions
- `select_governed_object(task) -> object`
- `requires_replacement_pointer(object, method) -> bool`
- `classify_domain_status(domain) -> active|cache|derived|archived|optional`
- `can_activate_dataset(manifest) -> bool`
- `can_delete_dataset_dir(path) -> bool`

## Procedures
### procedure `active_pointer_change`
`input`: candidate dataset manifests and validation result.
`steps`: inspect current active；validate candidate manifests and shard paths；write atomic active JSON；run status/check.

### procedure `provider_domain_promotion`
`input`: provider result and target domain.
`steps`: audit unit/PIT/coverage/stability；archive raw/staging；normalize；write dataset manifest；activate only after check.

### procedure `cleanup_governed_asset`
`input`: cleanup target.
`steps`: `qdp gc --dry-run`；replacement/active graph check；delete only with explicit confirmation；write summary if durable.

## Writeback
- Project facts: this brain.
- Cross-project topology: main brain.
- Long audits, provider reports and cleanup reports: `references/`, `data/audits`, or run manifests.
