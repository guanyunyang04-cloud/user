# Quant Data Platform 状态程序
快照日期：`2026-07-01`

## Module Interface
`exports`: QDP v2 active data base status、active table list、dataset manifest summaries、quality/audit entrypoints。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances
### object `qdp_project`
`type`: shared_manifest_first_data_base_instance
`state`: v2 已成为当前 active 数据基底；旧 lake/ingest/memmap/canonical 命令已从公开 CLI 归档。
`public_cli`: `qdp status/list/describe/check/rebuild/gc/update`
`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`

### object `qdp_v2_active_data_base`
`type`: active_manifest_first_data_base
`state`: `status=ok`
`root`: `quant_data_platform/data/qdp_v2`
`active_manifest`: `quant_data_platform/data/qdp_v2/active/active.json`
`scope`: `2011-11-22..2026-06-26`
`universe`: 沪深 A 股主板，剔除创业板、科创板、ST、退市股；当前名称含 `退市` 的股票也排除，即使 provider 的 `is_delisted` 标志未同步。
`symbol_start_overrides`: `600036.SH -> 2016-07-25`
`active_symbol_count`: `3037`
`active_table_count`: `17`
`evidence`: `qdp status --json`、`qdp check --quick --json`、`qdp check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json`
`meta_quality_state`: PIT/metadata/factor/index proof passed with zero findings on 2026-07-01.

### object `active_tables`
`type`: current_table_set
`raw_facts`: `market_daily_raw`、`market_intraday_1m`、`trading_calendar`、`universe_snapshot`、`security_status`、`valuation`、`adjust_factor`、`industry_concept`、`index_constituents`、`corporate_actions`、`share_capital`、`name_change`
`caches`: `market_intraday_5m`、`market_daily_panel`
`derived_features`: `intraday_daily_features`、`limit_intraday_features`
`derived_events`: `limit_status`
`boundary`: 1m and daily raw are source facts; 5m/panel/features are reproducible caches or derived tables.

### object `meta_domain_quality_proof`
`type`: active_quality_evidence
`state`: passed
`domains`: `trading_calendar`、`universe_snapshot`、`security_status`、`adjust_factor`、`industry_concept`、`index_constituents`
`audit_path`: `quant_data_platform/data/qdp_v2/audits/meta_domain_quality_20260701T140834+0000.json`
`adjust_factor`: active dataset `adjust_factor__4e0e31d3c1fd34bfcfa7a7dd`; one row per `market_daily_raw` key; `adjust_factor` uses positive `back_adjust_factor`; `default_factor_rows=1`; `ffilled_rows=42474`.
`industry_concept`: active dataset `industry_concept__9da8e8e69bbebafa996aed5f`; one row per universe key; blank/UNKNOWN industry rows are now `0`; `001399.SZ` was filled from AkShare/CNInfo profile industry; concept tags remain blank because reliable historical concept tags are not present.
`security_status`: historical ST rows remain PIT flags; active scope has `delisted_rows=0`.

### object `scope_active_rebuild_20260701`
`type`: active_scope_correction
`state`: completed
`scope_name`: `mainboard_hs_a_ex_current_st_name_delisted_v2`
`excluded_current_name_contains`: `退市`
`excluded_symbols`: `600193.SH`、`600608.SH`、`600636.SH`、`600696.SH`、`605081.SH`
`active_symbol_count_after`: `3037`
`effect`: all active symbol-bearing datasets were rebuilt under the same scope; `trading_calendar` unchanged.

### object `share_valuation_completion_20260701`
`type`: active_quality_completion
`state`: completed
`share_capital`: active dataset `share_capital__2ef0e66db12236b868b8c626`; `total_share_null_rows=0`、`float_share_null_rows=0`、`restricted_share_null_rows=0`.
`valuation`: active dataset `valuation__1e4c0d84334ff43e9f4e8fba`; `total_mv_null_rows=0`、`circ_mv_null_rows=0`; `pb_null_rows=229` and `turnover_rate_null_rows=224390` remain source/metric availability boundaries, not key/coverage failures.

### object `archived_v1_workflows`
`type`: archived_process_surface
`state`: old v1 data-lake workflows, downstream training-artifact workflows and one-off repair commands are not public QDP commands.
`notes`: `tools/archive_v1/README.md`、`tools/archive_repair/README.md`

### object `provider_runtime`
`type`: upstream_ingest_runtime
`state`: production Python environment is `yolos`; update workflow may use mootdx/BaoStock/CNInfo according to domain.
`boundary`: provider staging and update outputs must pass manifest/audit gates before active pointer changes.

## Pure Functions
- `active_table(domain)`: read `active.json.datasets[domain]` and then the referenced `dataset.json`.
- `describe_table(domain, full=false)`: summary by default; full manifest only with `--full --json`.
- `derive_next_qdp_action(state)`: if active coverage is current, prefer quality check, rebuild cache/feature only when raw facts change, and keep update dry-run separate from memmap work.
- `classify_cleanup(path)`: requires active manifest graph traversal and dry-run before deletion.

## Procedures
### procedure `inspect_qdp_status`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json`
`output`: active scope and table summaries, without requiring a catalog.

### procedure `quick_quality_check`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --no-write --json`
`output`: fast manifest/contract/path check.

### procedure `meta_quality_check`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json`
`output`: global exact primary-key and coverage/alignment proof for PIT/meta/factor/index domains.

### procedure `update_dry_run`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --dry-run --runtime fast --json`
`output`: coverage and gap plan; does not run providers.

## Evidence Entrypoints
- Current data-base document: `quant_data_platform/DATA_BASE.md`
- v1 archive notes: `tools/archive_v1/README.md`
- repair archive notes: `tools/archive_repair/README.md`
