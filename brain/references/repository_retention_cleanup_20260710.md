# Repository Retention Cleanup — 2026-07-10

- Status: `in_progress / authorized cleanup / guarded physical deletion`.
- Authorization: the user approved the repository/brain optimization plan and explicitly waived per-step confirmations.
- Rollback anchor for tracked files: Git HEAD `1fa2ae82a918b1458c99898ded97ad649c2d0b83`.
- Active execution impact: none; this cleanup does not create or modify `daily_research/output/active_execution_strategy.json`.

## Phase 1 — High-confidence cleanup

### Failed rebuild payloads

- Root: `daily_research/cache/rebuild_quarantine/20260531_seed7_failed_attempts/`.
- Two failed `forecast_feature_store.dat` files, each `2,891,616,448` bytes; total `5,783,232,896` bytes.
- Failure evidence: the non-empty `launch_stderr.log` records the failed rebuild; all log files are retained.
- Action: delete only the two rebuildable `.dat` payloads, not the diagnostic logs.

### Exact duplicate path-value shards

- Superseded incomplete root: `daily_research/output/path_policy/path_value_predictability/qdp_v2_path_value_predictability_full_stream_20260704_223040/`.
- Retained complete replacement: `daily_research/output/path_policy/path_value_predictability/qdp_v2_path_value_predictability_full_stream_v2_20260704_223929/`.
- Compared files: 14 yearly parquet shards, 2012 through 2025.
- Duplicate bytes: `2,601,248,171`.
- Verification: every corresponding file had identical size and BLAKE2b digest.
- Aggregate proof over `name:digest` rows: `69a9f1d30ab1931c8a9db46f7bddee318deb45cb8f99dd82536003ff3f06172c8a11a5c9e5e5d9b55e7315b2ec2b338c497b7540f99895a18dbfc7794cb27f1c`.
- The superseded root contains no completed summary; the retained replacement contains summary, metrics, report, predictions and charts.

### Git interrupted temporary objects

- Six `.git/objects/??/tmp_obj_*` files from the interrupted June 29 object write.
- Total bytes: `439,567,158`.
- Paths: `.git/objects/29/tmp_obj_hvcYHF`, `.git/objects/4d/tmp_obj_NYIGym`, `.git/objects/b2/tmp_obj_MRciGZ`, `.git/objects/d7/tmp_obj_6wBdeh`, `.git/objects/e4/tmp_obj_Hujgzl`, `.git/objects/f6/tmp_obj_i8eLH0`.
- Action boundary: remove only these files after confirming no Git process is running; retain the recent cruft pack and the May 24 stash.

## Phase 2 — QDP replacement proof

- Dry-run inventory: 43 dataset directories; 17 active/referenced and 26 unreferenced.
- Planned reclaim: `44,891,842,512` bytes.
- Every unreferenced dataset has a same-domain active replacement whose manifest covers the old start/end range.
- All active manifests were resolved; all active shard targets exist; no active shard target is a symlink; no unsafe hard-link state was found.
- Active 1m, 5m and intraday-daily-feature replacements have 7,861 / 7,861 / 7,827 materialized shards respectively. Their `source_path` fields are provenance only; active `path` targets are independent files.
- First/middle/last shard checks were run for those three large domains. Across nine sampled shards, active rows were an exact subset of the old source on 13, 10 and 61 common non-source columns respectively; `active_rows_not_in_source=0` for every sample.
- Pre-delete `qdp check --quick --runtime fast --json` returned `status=ok`, wrote no audit file, and completed in about 2.5 seconds after quick-check optimization.

### Deletion result

- Manifest-aware `qdp gc --delete --yes` removed all 26 unreferenced dataset directories.
- Reclaimed bytes: `44,891,842,512`.
- Post-delete inventory: 17 dataset directories, all 17 referenced, zero unreferenced.
- Post-delete `qdp status --verify-files --json` verified every active shard path and returned `status=ok` with zero missing manifests/shards.
- Post-delete `qdp check --quick --runtime fast --json` returned `status=ok` with all 17 required domains and no findings; no audit file was written.
- The same nine first/middle/last active shards remained readable after source deletion.

## Pending phases

- Research-output prediction trimming and active-view retention graph.
- Repository hygiene and control-plane compaction.
