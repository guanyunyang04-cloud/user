# Quant Data Platform 知识对象

## Object Classes
### class `canonical_dataset`
`definition`: 可审计、可按日使用的共享数据基底；完整承载数据域，研究 profile 决定字段子集。
`owner`: `quant_data_platform`
`consumers`: research projects declare lineage instead of owning duplicated lake/catalog.
`invariant`: raw OHLCV remains raw; adjusted prices, adjusted returns and intraday aggregate features are sidecars.

### class `provider_object`
`properties`: domain、endpoint、frequency、unit、adjustment semantics、coverage depth、server/source stability、PIT/available-time status。
`methods`: `probe()`；`audit_units()`；`archive_raw()`；`canonicalize()`；`register_status()`。

### class `mootdx_online`
`domain`: fast market bars, quotes, index, transaction/minute raw candidates, xdxr auxiliary.
`known_semantics`: `bars(frequency=9)` volume needs about `100x` normalization before canonical use.
`production_condition`: endpoint/frequency-level unit, adjustment, server and depth audit.

### class `baostock_online`
`domain`: trading calendar, universe/listing status, security status, industry/concept, index constituents, turnover/valuation and other daily structural fields.
`bar_role`: audit/backfill/fallback only unless a specific task selects it.

### class `cninfo_online`
`domain`: announcement, report, PDF, disclosure date and event verification.
`production_condition`: PIT-safe `available_time` / disclosure audit.

## Long-Term Lessons
- 数据基底和 memmap belong in QDP; experiments should generate light sample indexes, normalization manifests and model artifacts.
- Full-market single-process memmap is fragile on low-memory machines; sharded feature/label stores are the default shape.
- Canonical bundle identity and memmap source signature must match before reuse.
- Sharded memmap unit is `year + symbol block`; top registry aggregates schema and status.
- 1m data is cold archive; prepared 5m data can be reused after equivalence audit.

## Pure Functions
- `classify_provider(source) -> market|structure|disclosure|exploration|local_qdp`
- `is_training_safe(dataset) -> bool`
- `requires_pit_available_time(domain) -> bool`
- `can_reuse_memmap(bundle_id, signature) -> bool`
