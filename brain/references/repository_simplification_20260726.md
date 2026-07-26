# Repository simplification 2026-07-26

Owner-initiated over-engineering audit and retirement. No protected dataset,
research pack, or registered checkpoint was modified.

## What was retired and why

### QDP

- `qdp_v2/current_store_repair.py` (2,383) plus its test and
  `configs/qdp_current_corrections.json`. A dated one-shot 2026-07-17 migration
  that had already run; its runtime dir and reason string were hard-coded to
  that date. Only its own test imported it. Its effect survives as
  `repair_mutate_replace_*` shard filenames and
  `quality.scope_rewrite: qdp_current_store_repair_20260717`.
- `qdp_v2/permanent_exclusions.py` (798) plus its test. The
  `permanent_st_delisting_exclusion_v1` policy was never applied: `active.json`
  has no `scope.permanent_exclusion_policy`, `status.py` reported
  `retired_not_applied`, and `pit_history.py` actively stripped the marker.
  Call sites in `status.py`, `database_audit.py`, and `pit_history.py` were
  removed. The 37 KB registry under protected `data/qdp_v2/` was left in place.
- `qdp_v2/runtime.py` (81). Zero importers. `--runtime` is separately a bare
  thread-count dict in `database_audit.py`.
- `progress.py` `StageProgress` machinery. Never instantiated, so the stack was
  always empty and `create_progress` always took the `_SimpleProgress` branch.
- `gc.py` pin registry. `data/qdp_v2/pins/` never existed and nothing wrote a pin.
- `manifest.py` `qdp_snapshot_payload` / `qdp_snapshot_sha256`. Test-only.
- `repair.py` `replace_active_shards` / `replace_active_shard_from_parquet` /
  `replace_active_shards_from_parquet`. Test-only; superseded by
  `mutate_active_shards_from_parquet`.
- `dataset.py` `validate_dataset` (unreachable from the CLI) and `_domain_role`
  prose for four domains that do not exist.
- `core/json_io.py` `write_json`, and the six unread `QdpPaths` fields.
- Four `registry/*.json` and two `configs/*.json` with zero code readers.

Deduplicated: one `_sql_literal`, one `_sha256_file`, one `EXPECTED_BAR_TIMES`,
one shard mutation-id implementation, one valuation required-column set.
Dropped the `critical` severity tier (never emitted) and three dead branches
(`check.py` `deep`, `status.py` legacy active layout, `audit.py` pending
contract states).

Kept deliberately: the manifest CAS commit path, commit-then-delete ordering,
mutation-id replay, the DuckDB memory watchdog, `compact.py` fingerprint
equality, and all PIT/no-future invariants. These are justified by real provider
errors and by single-machine crash recovery over multi-GB shards.

### Daily Research

- `path_policy/seq100_path_relevance.py` (4,208) plus its test and
  `studies/seq100_pit_path_relevance_v1.json`. The abandoned predecessor fork of
  `seq100_signal_quality.py`: 89 shared top-level names, 74 byte-identical
  functions over 2,196 lines, 9 near-identical over 829 more. Zero importers.
- Three finished or superseded study contracts whose conclusions are already
  indexed research records: `signal_close_path_value_2x2_v1`,
  `pit_l35v2_survivorship_frozen_audit_v1`, `signal_close_capital_speed_v4`.

## Metadata drift corrected

- `studies/seq100_pit_signal_quality_v1.json` status `planned` -> `active`.
  The field sits outside the hashed `contract` block, so `contract_sha256`
  remains `f15d1af2...561b2`.
- `research_records/seq100/index.json` `active_studies` was an empty array while
  live contracts existed; it now lists the three remaining contracts.
- `seq100_development.py` `ACTIVE_STUDIES` pointed at a deleted completed
  contract; it now points at the active signal-quality study.
- `quant_data_platform/brain/state.md` claimed a `qdp exclude` command that
  `cli.py` never dispatched.
- `tmp/` (114 tracked process artifacts: 76 parquet, 29 png, 7 json, 2 pdf) is
  now gitignored and untracked. Files remain on disk.

## Scale

Working tree: 43 files changed, 80 insertions, 10,992 deletions.
Index: 114 `tmp/` files untracked.

## Verification

- `daily_research/path_policy/tests`: 202 passed
- `quant_data_platform/tests` + `tools`: 85 passed
- `tools.brain.integrity_check`: ok, 0 errors, `active_study_count: 3`
- `seq100_development verify`: ok, 15 bundles, 16 records
- `qdp check --quick`: ok, 14 datasets, 0 findings
- `seq100_signal_quality validate-research`: validated, QDP and model-registry
  protection hashes unchanged
- `git diff --check`: clean

## Known remaining friction (not addressed)

The owner still has no supported path to evolve a dataset schema or make a
one-cell correction: `repair.py` is library-only with no CLI, schema authority is
a live parquet footer rather than a declaration, and downstream packs bind
`manifest_sha256` with no rebind command. `seq100_fold_contract.py` includes
`mtime_ns` in its provenance digest, so copying a pack invalidates every fold
contract. Study `contract_sha256` is a self-hash with no tool that computes it.
