# Repository Retention Cleanup — 2026-07-10

- Status: `completed / cleanup, evidence writeback, guards and full health verified`.
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

## Phase 3 — Research-store reachability and output retention

- Added `daily_research/brain/research_store_retention_policy.json` with two active today-close views and explicit cold-component tombstones.
- Rebuilt `research_store_index.json` as a schema-v2 pointer-only index with relative view paths and SHA-256 hashes; it no longer inlines panel/label manifests.
- Active dependency graph retained exactly six component IDs: the two active views, panel store, today-close label store, and normal/rollforward sample indexes.
- Archived manifest hashes, shapes, provenance and inventory digests to `daily_research/brain/references/research_store_cold_assets_archive_20260710_105437.{md,json}` before deleting 9 unreachable components.
- Cold-component reclaim: `61,582,271,039` bytes (`57.353 GiB`); post-delete active-view verification returned `ok` with no inactive disk views.
- Prediction trim removed 86 large prediction CSV/parquet/feather files across 43 summarized runs: `17,583,163,833` bytes (`16.376 GiB`), with zero skips. Summary JSON, metrics, reports and checkpoints were retained and marked with trim manifests.
- Directory GC then removed 63 unreferenced partial/smoke directories: `3,270,410,936` bytes (`3.046 GiB`), with zero skips.
- Post-cleanup scan reported zero safe-directory, prediction-trim and cold-component candidates. Both active views loaded a real test sample with `input_shape=[100,32]`, `path_shape=[60,4]` and `price_anchor=today_close`.

## Aggregate reclaim

- Phase 1: `8,824,048,307` bytes.
- Phase 2: `44,891,842,512` bytes.
- Phase 3: `82,435,845,808` bytes.
- Total: `136,151,736,627` bytes (`126.82 GiB`, `136.15 GB`).

## Phase 4 — Repository and control-plane hygiene

- Removed the orphan root `package.json` / `package-lock.json` and root `node_modules` dependency tree (about 82 MB, not included in the data-reclaim total above); no root JavaScript source or build entry depended on them.
- Removed 10 tracked `.playwright-cli` logs plus empty `.tmp` / `.agents` directories. Root `.gitignore` now covers `.pytest_cache/`, `.tmp/` and `.playwright-cli/` in addition to `node_modules/`.
- Archived the only durable facts from the output-only `a_stock_daily_selection/` directory to `brain/references/a_stock_daily_selection_archive_20260710.md`, then removed its 8 tracked output files and stale brain-catalog exception.
- Preserved `daily_stock_analysis-main/.github/workflows` as upstream standalone templates. They are not active GitHub Actions in this monorepo, but remain part of that embedded project's public standalone contract.
- Slimmed the workspace workflow CLI by removing the dead/redundant `handoff`, `preflight`, `workflow-guide`, `select-workflow` and `audit-brain` command surfaces. Their useful implementation functions remain internal to capsule/platform code. `status` remains public because it uniquely returns the complete explicit-`run_tag` evidence packet and is still a current `daily_research` playbook dependency.

### Final verification

- Workflow/capsule/platform focused suite: 82 passed.
- Full `tools/brain` suite: 314 passed.
- Research-store and seq100 focused suite: 35 passed; QDP focused suite: 13 passed.
- Research-store dry-runs: zero safe-delete, prediction-trim and cold-component candidates; both active views verify `ok` and each loads a real test sample with input `[100,32]`, path `[60,4]`, anchor `today_close`.
- QDP final state: 17/17 dataset directories referenced, zero unreferenced bytes, zero missing manifests/shards, `status=ok`, and quick check `status=ok` without a persisted audit.
- Repository hygiene paths are absent and `daily_research/output/active_execution_strategy.json` is unchanged.
- Evidence registry rebuild: `status=ok`, 160 records, zero duplicate IDs and zero missing paths.
- Full doc guard, integrity, brain-sync, skill-sync, brain-structure, agent-meta and multi-paradigm checks all returned `ok`; integrity/sync/lint reported zero errors and zero warnings.
- Workspace-brain full health returned `status=ok`; final `git diff --check` passed after removing one trailing blank line from `.gitignore`.

## Closeout result

- No known safe-delete, prediction-trim, cold-component, QDP GC, repository-hygiene or brain-guard work remains from this cleanup plan.
- The resulting tracked changes are intentionally left unstaged and uncommitted for user review; no commit or push was requested in this closeout.
