# r64 Full-Universe Gold Data Lake Status - 2026-05-14

## Summary
- Fact: r64 adds a resumable, sharded Gold training dataset builder for the DuckDB + Parquet research data lake.
- Fact: the builder uses the existing full-universe Bronze/Silver dataset `policy_input_bundle__0f116a9b78c92ff045a6853d` instead of refetching from TQ.
- Fact: the first full-universe lake-backed smoke passed for uncapped `learned_all_a` with `3,070` symbols over `2019-01-02` to `2019-06-30`.
- Fact: full-window strict Gold now exists for uncapped `learned_all_a`, sourced from `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Current verdict: r64 infrastructure and full-window `strict_train` Gold are complete and audit-clean. Full-window `realtime_research` Gold is still pending.
- Boundary: this is research data infrastructure only. It is not strategy evidence and does not change live/default/promotion behavior.
- Production anchor: `daily_research/output/active_execution_strategy.json` remains unchanged.

## Implemented Facts
- Added `daily_research.data_lake.gold_training_builder` with `GoldBuildSpec` and `build_sharded_gold_training_dataset(...)`.
- Added CLI `python -m daily_research.data_lake.build_gold_training_dataset`.
- Added CLI/API `daily_research.data_lake.audit_gold_dataset`.
- Added `daily_research.data_lake.policy_input_loader.load_policy_inputs_from_lake(...)`.
- Extended `ResearchDataLake` with:
  - sharded Gold dataset identity construction,
  - `save_sharded_training_dataset(...)`,
  - transparent sharded `load_training_dataset(...)`,
  - audit report writeback.
- Shards write reusable artifacts under fingerprinted Gold dataset roots:
  - `sample_frame/<date_start>_<date_end>.parquet`
  - `daily_frame/<date_start>_<date_end>.parquet`
  - `teacher_summary/<date_start>_<date_end>.json`
  - `portfolio_checkpoint/<date_end>.json`
  - `shard_manifest.json`
  - `build_manifest.json`
  - `audit_report.json`
- `PortfolioState.snapshot()` / `PortfolioState.from_snapshot()` are used across shards so holdings, hold days, recent turnover, and recent returns are not reset at shard boundaries.
- Strict and realtime zones are explicitly separated:
  - `strict_train` trims to the max forward horizon and must have `unobserved_label_rows=0`.
  - `realtime_research` keeps tail rows, marks `is_observed=false`, and is never training-safe.

## Smoke Evidence
- Source Bronze/Silver dataset:
  - dataset_id: `policy_input_bundle__0f116a9b78c92ff045a6853d`
  - universe: `learned_all_a`, uncapped, `3,070` symbols
  - full source window: `2010-01-04` to `2026-05-13`
- Smoke command:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_lake.build_gold_training_dataset --universe learned_all_a --max-universe-size 0 --benchmark 000300.SH --start-date 2019-01-02 --end-date 2019-06-30 --zones strict_train,realtime_research --source-market-dataset-id policy_input_bundle__0f116a9b78c92ff045a6853d --shard-frequency month --resume --progress-jsonl daily_research/output/research_data_lake/r64_smoke_progress_2019H1_full_lake.jsonl`
- Explicit progress log:
  `daily_research/output/research_data_lake/r64_smoke_progress_2019H1_full_lake.jsonl`

## Full Strict Gold Dataset
- dataset_id: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`
- dataset root:
  `daily_research/output/research_data_lake/parquet/gold/continuous_policy_training_matrices/strict_train/36c234208d5f375ea1cccfc1`
- explicit build manifest:
  `daily_research/output/research_data_lake/parquet/gold/continuous_policy_training_matrices/strict_train/36c234208d5f375ea1cccfc1/build_manifest.json`
- source Bronze/Silver dataset: `policy_input_bundle__0f116a9b78c92ff045a6853d`
- requested full window: `2010-01-04` to `2026-05-13`
- strict observed window: `2010-01-04` to `2026-04-10`
- strict end date: `2026-04-10`
- universe: `learned_all_a`, uncapped, `3,070` symbols
- shard count: `196`; completed shard count: `196`
- sample rows: `1,863,468`
- daily rows: `3,949`
- unobserved label rows: `0`
- training safe: `true`
- audit status: `ok`, `error_count=0`, `warning_count=0`
- transparent sharded loader check passed:
  `sample_rows=1,863,468`, `daily_rows=3,949`, sample date range `2010-01-04 -> 2026-04-10`, `unobserved=0`, `training_safe=true`.

## Strict Smoke Dataset
- dataset_id: `continuous_policy_training_matrices__strict_train__03ce80c27c93371d6b5dcf47`
- dataset root:
  `daily_research/output/research_data_lake/parquet/gold/continuous_policy_training_matrices/strict_train/03ce80c27c93371d6b5dcf47`
- explicit build manifest:
  `daily_research/output/research_data_lake/parquet/gold/continuous_policy_training_matrices/strict_train/03ce80c27c93371d6b5dcf47/build_manifest.json`
- window: requested `2019-01-02` to `2019-06-30`; prepared market data ended on trading day `2019-06-28`.
- strict end date: `2019-05-30`
- shard count: `5`; completed shard count: `5`
- sample rows: `78,450`
- daily rows: `98`
- unobserved label rows: `0`
- training safe: `true`
- audit status: `ok`, `error_count=0`, `warning_count=0`

## Realtime Smoke Dataset
- dataset_id: `continuous_policy_training_matrices__realtime_research__acf000b3d5cf52b558706926`
- dataset root:
  `daily_research/output/research_data_lake/parquet/gold/continuous_policy_training_matrices/realtime_research/acf000b3d5cf52b558706926`
- explicit build manifest:
  `daily_research/output/research_data_lake/parquet/gold/continuous_policy_training_matrices/realtime_research/acf000b3d5cf52b558706926/build_manifest.json`
- window: requested `2019-01-02` to `2019-06-30`; prepared market data ended on trading day `2019-06-28`.
- strict end date: `2019-05-30`
- shard count: `6`; completed shard count: `6`
- sample rows: `84,573`
- daily rows: `118`
- observed label rows: `78,371`
- unobserved label rows: `6,202`
- training safe: `false`
- audit status: `ok`, `error_count=0`, `warning_count=0`

## Verification
- Focused regression passed:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/data_lake/tests/test_research_data_lake.py daily_research/data_lake/tests/test_gold_training_builder.py daily_research/continuous_policy/tests/test_training_dataset_cache.py -q`
- Result: `18 passed`.
- `git diff -- daily_research/output/active_execution_strategy.json` produced no output.
- `git diff --check` passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check` passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json` returned `status=ok`.

## Blocker
- Full-window `strict_train` Gold is no longer blocked.
- Full-window `realtime_research` Gold for uncapped `learned_all_a`, `2010-01-04` to `2026-05-13`, is still pending.
- The previous r62 blocker was one-shot in-memory `build_training_matrices(...)`; r64 removes that architectural blocker by adding resumable shards and checkpoints.
- Do not write realtime Gold as training-ready; realtime Gold must remain research/audit data only. Its unobserved tail labels must not count as completed training evidence.

## Next
- Next allowed realtime build:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.data_lake.build_gold_training_dataset --universe learned_all_a --max-universe-size 0 --benchmark 000300.SH --start-date 2010-01-04 --end-date 2026-05-13 --zones realtime_research --source-market-dataset-id policy_input_bundle__0f116a9b78c92ff045a6853d --shard-frequency month --resume --progress-jsonl daily_research/output/research_data_lake/r64_full_realtime_progress_20100104_20260513.jsonl`
- If a realtime full build is interrupted, record the last completed shard, failure reason, and exact resume command. Do not call realtime complete until audit passes.
