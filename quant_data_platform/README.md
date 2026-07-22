# Quant Data Platform

QDP is the workspace's single mutable research-data store. The source of truth
is:

1. `data/qdp_v2/active/active.json`;
2. each referenced `datasets/<domain>/<dataset_id>/dataset.json`;
3. the Parquet shards referenced by those manifests.

`qdp_v2` is a protected historical directory name, not a compatibility API.
There are no generation, candidate, publish, rollback, legacy lake, memmap, or
1-minute rebuild surfaces.

## Current boundary

- Active date: `2026-07-21`; 14 active domains; quick check passes.
- Daily, 5-minute, factor, calendar, status, universe, industry, and index tails
  reach July 21.
- Share capital, valuation, and corporate actions remain through July 16 because
  strict PIT confirmation rejected unconfirmed changes. Do not weaken that
  guard to make freshness look complete.
- BaoStock and mootdx are the normal current-update providers. A stock-day is
  never stitched across providers and missing bars are not interpolated.
- The store is a user-selected current-survivor research store; it does not
  claim survivorship-bias-free delisting research semantics.

## Commands

```powershell
$env:PYTHONPATH='H:\quant_project\quant_data_platform\src;H:\quant_project'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date YYYY-MM-DD --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli compact --dry-run --json
```

Training packs, models, and backtests belong to `daily_research`, not QDP.
Current machine facts are in `brain/state.md`.
