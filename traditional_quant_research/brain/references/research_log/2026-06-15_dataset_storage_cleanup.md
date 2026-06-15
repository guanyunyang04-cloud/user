# 2026-06-15 Dataset Storage Cleanup

## Summary

- Goal: retire redundant internal legacy data after QDP canonical/sharded datasets became available, while keeping traditional research reproducible.
- Kept: latest v2 PIT snapshot `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`, root `latest_manifest.json`, QDP active/frozen sharded manifests, experiment summaries/audits/config-level outputs.
- Removed: old v2 build snapshots, raw build cache, old build logs, `tq_daily_mainboard_v1`, generalized strong-event cache, and large regenerable experiment detail files.

## Deleted Categories

- `traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/`: removed historical full/smoke/audit/build snapshots, raw `cache/`, and build/fetch logs; retained latest full PIT snapshot and root `latest_manifest.json`.
- `traditional_quant_research/data/raw/tq_daily_mainboard_v1`: removed obsolete v1 residue.
- `traditional_quant_research/cache/generalized_strong_events`: removed regenerable event cache.
- `traditional_quant_research/output/experiments`: removed large regenerable detail files matching prediction/panel/trade-detail patterns:
  - `ml_signal_predictions.csv`
  - `ml_signal_predictions_by_year/eval_year=*.csv`
  - `model_zoo_predictions.csv`
  - `ml_predictions.csv`
  - `event_feature_panel*.csv`
  - `selected_trades.csv` when larger than 50 MB

## Space Impact

- Raw v2/v1 cleanup: approximately `2.2089 GB` reclaimed.
- Generalized event cache cleanup: approximately `0.9670 GB` reclaimed.
- Experiment detail cleanup: approximately `15.3505 GB` reclaimed.
- Total reclaimed: approximately `18.5264 GB`.

Post-cleanup observed sizes:

- `traditional_quant_research/data`: `0.47 GB` (`481.8 MB`, 10 files).
- `traditional_quant_research/cache`: `0.00 GB` (0 files).
- `traditional_quant_research/output`: `1.28 GB` (`1314.4 MB`).
- `traditional_quant_research/output/experiments`: `1.28 GB` (`1314.3 MB`).
- `quant_data_platform/data`: `56.15 GB` (`57499.8 MB`), not modified by this cleanup.

## Verification

- `traditional_quant_research.dataset_v2.load_pit_manifest()` still resolves latest snapshot:
  - `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- Existing latest PIT files remain present:
  - `traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/latest_manifest.json`
  - `traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603/manifest.json`
- QDP active/frozen sharded manifest files remain present:
  - `quant_data_platform/data/memmap/sharded/canonical_short_horizon_core_v1_full/sharded_memmap_manifest.json`
  - `quant_data_platform/data/memmap/sharded/mainboard_style_structural_1y_path20_v1_full_2010_2026/sharded_memmap_manifest.json`

## Interpretation

The traditional project now keeps a small compatibility PIT snapshot locally, while QDP remains the canonical large data substrate. Future cleanup should focus on migrating old traditional loaders to QDP-backed inputs before deleting the final local v2 PIT compatibility snapshot.
