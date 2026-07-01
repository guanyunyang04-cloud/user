# QDP v1 Archived Workflow

This directory documents the archived catalog-first QDP workflow.

The active local data base no longer uses these concepts as public commands:

- catalog-first lake commands
- canonical policy bundle commands
- memmap build/freeze/compose commands
- v1 refresh/daily-update DAGs
- legacy ingest/import commands
- old policy-input/gold-training builders

Current daily commands are intentionally limited to:

```bash
qdp status
qdp list
qdp describe <table>
qdp check --quick
qdp check --full
qdp rebuild <cache-or-feature>
qdp gc --dry-run
qdp update
```

The current source of truth is:

1. `quant_data_platform/data/qdp_v2/active/active.json`
2. `quant_data_platform/data/qdp_v2/datasets/<domain>/<dataset_id>/dataset.json`
3. parquet shards referenced by each dataset manifest

Old v1 implementation modules may still remain under `src/quant_data_platform/` when a small helper or test fixture depends on them, but they are not part of the active QDP command surface.
