# QDP Alpha V2 Replacement Data Base Activation 20260626

## Verdict

- Status: `replacement_data_base_activated / qdp_canonical_current / research_data_only`.
- Scope: activate the repaired QDP alpha_v2 data substrate after rebuilding a tradeable-mainboard, pool-filtered sharded memmap from the repaired policy bundle.
- Active execution impact: unchanged. This does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker, or promotion state.
- Model quality impact: none by itself. This only changes the data substrate for future research/training.

## Activated Canonical Data

```text
canonical_dataset_id:
  policy_input_bundle__2c27a13a7c7ae85417c36340

active sharded memmap:
  H:\quant_project\quant_data_platform\data\memmap\sharded\tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_2010_2026_20260626_01\sharded_memmap_manifest.json

feature profile:
  style_structural_alpha_v2

label schema:
  path20_basic_v2

static context:
  exchange, industry
  symbol_id deliberately excluded
```

Supporting side artifacts:

```text
v2 status sidecar:
  data_platform_v2_status_sidecar__97bf843dbc56cbb4e1603640

tradeable-mainboard pool view:
  policy_pool_view__7aca6133c8076d3c6c0cebf6

coverage audit:
  H:\quant_project\quant_data_platform\data\audits\qdp_alpha_v2_tradeable_mainboard_pool_filtered_coverage_20260626_01\qdp_alpha_v2_tradeable_mainboard_pool_filtered_coverage_20260626_01\coverage_audit_report.json
```

## Build Result

```text
planned_shard_count:   187
processed_shard_count: 187
stored_shard_count:    173
empty_shard_count:     14
failed_shard_count:    0
feature_count:         307
symbol_count:          3025
years:                 2010-2026
```

Validation:

```text
qdp validate-memmap --manifest <replacement_manifest> --json
status: ok
blockers: []
```

QDP status after activation:

```text
canonical_dataset_id:                    policy_input_bundle__2c27a13a7c7ae85417c36340
active_memmap_source_market_dataset_id:  policy_input_bundle__2c27a13a7c7ae85417c36340
active_memmap_matches_canonical_bundle:  true
active_memmap_state:                     current
```

## Coverage Result

Coverage audit scanned:

```text
scanned_shards: 173
scanned_rows:   9,552,984
feature_count:  307
```

Key repaired fields:

```text
turn finite_rate:
  2010 0.8851
  2015 0.8464
  2024 0.9926
  2025 0.9896

turn_z20 finite_rate:
  2010 0.8809
  2015 0.8446
  2024 0.9925
  2025 0.9896

intraday_last_5m_ret finite_rate:
  2010 0.8966
  2015 0.8883
  2024 0.9926
  2025 0.9903

cs_z_intraday_last_5m_ret finite_rate:
  2010 0.8810
  2015 0.8875
  2024 0.9926
  2025 0.9903
```

Interpretation:

```text
The old 2010-2015 turnover_context zero-coverage problem is repaired in the replacement memmap.
The old intraday last_5m_ret collapsed-signal issue is repaired in the replacement sidecar/memmap.
The remaining suspicious coverage is concentrated in valuation_context.
```

Known valuation limitations:

```text
valuation_psTTM and valuation_pcfNcfTTM are structurally missing in 2010-2015 legacy source periods.
valuation_peTTM remains lower coverage than pb/turn in some years.
These are treated as PIT-safe missing values, not heuristic-filled facts.
```

## Important Boundary

The market and intraday daily data extend to 2026-06-10, but `security_status` and `universe_snapshot` currently end at 2026-06-05. The tradeable-mainboard pool view is intentionally capped at 2026-06-05 to avoid silently marking missing ST/status rows as clean tradeable rows.

## Code/Contract Fixes Included

```text
policy_input_loader.py:
  legacy valuation aliases are coalesced, not ignored when canonical target columns exist but are NaN.

v2_status_sidecar.py / pool_views.py:
  domain/status readers support both single silver_domain_data files and shard_manifest references.

quant_data_platform memmap builder:
  supports explicit --canonical-dataset-id, --force-years, and batched parallel year-input cache.

quant_data_platform registry status:
  active sharded source id is read from the active sharded manifest/status, not stale monolithic memmap registry fields.
```

## Next Allowed Actions

## Replacement Training Pack

Status after activation: `completed / loader_smoke_passed`.

```text
manifest:
  H:\quant_project\quant_data_platform\data\memmap\training_pack\tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_training_pack_20260626_01\qdp_training_pack_manifest.json

sample_count:
  total       6,992,361
  train       5,696,015
  validation    643,708
  test          652,638

feature_panel_shape:
  [2976, 3989, 307]

feature_dtype:
  float16

static_context:
  exchange, industry
```

Loader smoke:

```text
load_forecast_memmap_dataset(..., max_samples_per_role=32)
row_count: 96
input_dim: 307
lookback_days: 252
roles: train 32 / validation 32 / test 32
static_context_shape: [96, 2]
label arrays opened successfully
```

During this build, a path-semantics defect was found and repaired: `build_qdp_training_pack` accepted `tag` but ignored it when `output_root` was passed. It now resolves `output_root/tag` unless the provided output root is already the tag directory. This prevents untagged overwrites of the generic training_pack root.

## Next Allowed Actions

1. Use the replacement training pack for future alpha_v2 or shortline model/scorer research when repaired turnover/intraday semantics matter.
2. Future alpha_v2 or shortline research should prefer the replacement pack rather than any deleted stale acceleration artifact.
3. Do not interpret this data activation or training-pack build as model improvement, strategy promotion, or execution thaw.

## Stale Artifact Cleanup

Status after activation validation: `completed`.

Deleted stale acceleration artifacts:

```text
H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_2010_2026_20260624_01
H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01
H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01
H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_date_slate_pack_20260621_01
H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01
H:\quant_project\quant_data_platform\data\memmap\training_pack\labels
H:\quant_project\quant_data_platform\data\memmap\training_pack\feature_panel_stock_date_feature.float16.dat
H:\quant_project\quant_data_platform\data\memmap\training_pack\qdp_training_pack_manifest.json
H:\quant_project\quant_data_platform\data\memmap\training_pack\qdp_training_pack_progress.json
H:\quant_project\quant_data_platform\data\memmap\training_pack\sample_index.parquet
H:\quant_project\quant_data_platform\data\memmap\training_pack\static_context_ids.int32.dat
```

Approximate space reclaimed from the dry-run inventory: `91.8 GiB`.

Registry update:

```text
quant_data_platform/registry/sharded_memmap_registry.json:
  removed entries pointing to the two deleted style_structural_alpha_v2 sharded manifests.
  removed one missing qdp_smoke_2022_10x1 validation_partial registry reference whose manifest was already absent.
  active_manifest_json remains the 20260626 tradeable-mainboard replacement memmap.
```

Explicitly kept:

```text
active replacement memmap:
  tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_2010_2026_20260626_01

active replacement training pack:
  tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_training_pack_20260626_01

non-alpha_v2/frozen lineage artifacts:
  canonical_short_horizon_core_v1_full

source data:
  canonical policy bundle, data lake, sidecars, and pool view
```

Boundary:

```text
This cleanup removes obsolete generated acceleration artifacts only.
It does not delete canonical source data, active/default/live execution artifacts, promotion evidence, paper/live state, or broker state.
```
