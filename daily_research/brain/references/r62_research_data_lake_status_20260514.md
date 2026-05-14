# r62 Research Data Lake Status - 2026-05-14

## Summary
- r62 implements a local research data lake using DuckDB catalog metadata plus Parquet storage.
- Scope completed: reusable catalog APIs, Bronze/Silver market-feature storage, Gold training dataset registration, train-policy lake-backed reuse with legacy pickle fallback, and CLI tooling.
- Boundary: this is research infrastructure only. It does not change live/default/promotion behavior and does not modify `daily_research/output/active_execution_strategy.json`.

## Implemented Interfaces
- Package: `daily_research.data_lake`.
- Main API: `ResearchDataLake`, `list_datasets()`, `describe_dataset()`, `load_training_dataset()`, `query()`, `write_catalog_manifest()`.
- Build CLI: `python -m daily_research.data_lake.build_research_database`.
- Legacy import CLI: `python -m daily_research.data_lake.import_legacy_training_caches`.
- `train_policy.py` now defaults to `--training-dataset-store lake` while preserving legacy pickle fallback.
- Environment: `daily_research/environment.yml` now includes `duckdb>=1.0,<2`; local `yolos` env was updated with `duckdb 1.5.2`.

## Data Evidence
- Full Bronze/Silver data lake built under `daily_research/output/research_data_lake`.
- Catalog manifest: `daily_research/output/research_data_lake/manifest_latest.json`.
- Full market dataset:
  - dataset_id: `policy_input_bundle__4db1a32ab6e7d77ac7b8671c`
  - universe: `learned_all_a`, max universe size `1200`, benchmark `000300.SH`
  - date window: `2018-05-14` to `2026-05-13`
  - rows/cells: `2,328,000` Bronze market rows, `1,940` benchmark rows, `1,940` membership rows, `109` feature panels, `253,752,000` feature cells.
- Gold training datasets registered from existing reusable caches:
  - `continuous_policy_training_matrices__strict_train__a6e4d2c45a8f1c41d8165651`: `liquid500`, `2024-01-02` to `2025-12-31`, `232,500` sample rows, `465` daily rows, observed-label ratio `1.0`.
  - `continuous_policy_training_matrices__strict_train__22298f27aeb8e5a6bfe113a4`: `learned_all_a`, `2024-01-02` to `2025-12-31`, `558,000` sample rows, `465` daily rows, observed-label ratio `1.0`.
- Small real data lake smoke succeeded for `learned_all_a/20`, `2025-01-02` to `2025-03-31`:
  - strict zone: `740` sample rows, `37` daily rows, observed-label ratio `1.0`.
  - realtime zone: `1,140` sample rows, `57` daily rows, `400` unobserved tail-label rows explicitly marked.

## Blocker
- Full Gold construction for `2018-05-14` to `2026-05-13`, `learned_all_a/1200`, is not complete.
- Attempt 1: full strict+realtime build wrote Bronze/Silver, then was stopped before Gold because memory pressure rose and free physical memory fell to about `2.4GB`.
- Attempt 2: strict-only full Gold build was stopped after about `31` CPU minutes and about `6.2GB` working set, with no Gold files written.
- Attempt 3: 2024-2026 Gold rebuild was also stopped after a long no-output training-matrix phase; existing 2024-2025 reusable caches were imported instead.
- Diagnosis: the bottleneck is not DuckDB/Parquet. It is `build_training_matrices(...)`, which constructs large continuous-policy training surfaces in memory without chunk progress or checkpointed shards.

## Verification
- Focused tests passed:
  - `daily_research/data_lake/tests/test_research_data_lake.py`
  - `daily_research/continuous_policy/tests/test_training_dataset_cache.py`
- r61+r62 regression passed: `151 passed, 24 warnings`.
- Active artifact guard remained clean during implementation.

## Next
- r63 should split Gold training construction into resumable date shards, with explicit progress events and per-shard catalog registration.
- Do not describe full 8-year Gold as completed until a parseable catalog entry exists with strict/realtime label completeness.
- The current reusable database is valid for market/feature audit and existing Gold cache reuse, but not yet for full-window continuous-policy Gold training.
