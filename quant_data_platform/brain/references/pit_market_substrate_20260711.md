# PIT daily market substrate and research dataset view

Date: `2026-07-11`

## Contract

`quant_data_platform.qdp_v2.pit_market_substrate` builds five content-addressed,
non-active QDP datasets from one or more traditional BaoStock PIT sources:

- `market_daily_raw`
- `security_status`
- `limit_status`
- `adjust_factor`
- `pit_signal_universe`

The sources may be complete traditional snapshots (`daily_universe.parquet` plus
`daily_bars.parquet`) or a recovery cache (`cache/daily_stock_lists/year=*.parquet`
plus `cache/security_master.parquet`) combined with independent QDP market-daily
backfill parquet/shard/data-lake manifests. Snapshot overlap is deterministic:
latest `created_at`, then resolved path, wins each `(trade_date, symbol)` key.

Output view schema: `schema_version=1`, `kind=qdp_v2_research_dataset_view`,
`base_active_manifest`, a complete `datasets` map, the five-item `overrides` map,
scope/source identity, and quality statistics. Views are written to
`quant_data_platform/data/qdp_v2/views/<view_id>.json`. The builder has no active
write path and verifies byte-for-byte unchanged `active/active.json` before return.

## Factor and limit semantics

Factors combine active QDP dense factors and repeatable external BaoStock factor
event inputs. Precedence is active exact key, active same-symbol as-of row, then
external same-symbol as-of event. Every market key must have a positive
`back_adjust_factor` and source date. Missing or nonpositive factors hard-block the
view. A factor of `1` is allowed only through an explicit audited
`--allow-no-event-factor-symbol` exception and remains identified in provenance.

The current-day raw reference for a limit is:

`prior_valid_adjusted_close / current_back_adjust_factor`.

The builder then applies the current date-local ST rate (5%, otherwise 10%) and
rounds half-up to the `0.01` tick. This prevents ex-right factor jumps from turning
the prior raw close into a false limit price. Suspended/no-bar rows remain in status
and PIT universe, but not in `market_daily_raw`; `eligible_for_signal` requires a
valid market bar and Gate 0 proves that every eligible row has one.

## CLI and downstream consumption

Build:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild pit-market-substrate `
  --snapshot-root <snapshot-or-recovery-cache-root> `
  --snapshot-root <another-snapshot-root> `
  --market-bars-source <optional-qdp-market-daily-manifest> `
  --factor-events <factor-shard-manifest> `
  --factor-events <another-factor-shard-manifest>
```

Verify:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli verify pit-market-view --view <view.json>
```

Seq100 consumes the full suite atomically with `--dataset-view <view.json>`. A
partial override is rejected, preventing a PIT universe from being combined with
canonical active price/factor/limit/status tables.

## Real-scale probe

The 2016-01-04 through 2026-06-01 probe produced view
`seq100_pit_2016_2026_probe__58498361379f14f30e77cf90` without activating it:

- `3,393` historical symbols;
- `7,440,688` valid market/factor/limit keys;
- `7,451,610` PIT universe/status keys;
- `7,031,085` signal-eligible rows and zero Gate-0 missing market rows;
- zero duplicate primary keys;
- zero missing/nonpositive factor rows;
- active manifest byte-for-byte unchanged.

This is an integration proof, not the final 2012-2025 research view. The latter
must wait for the independent 2012-2015 market-daily recovery to complete and pass
the same gates.
