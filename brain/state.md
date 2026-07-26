# Current workspace state

Updated: `2026-07-26`

- QDP is usable with `active_as_of_date=2026-07-21`. PIT main-board daily history
  includes historical ST, long suspensions, and delisted securities.
- Daily domains are normalized to `symbol_history` effective ticker intervals
  for both code-change identities. The 5-minute dataset was not rewritten.
- Protected QDP datasets, research packs, and registered checkpoints remain in
  place. The stable model registry still contains 15 bundles.
- Daily Research's frozen registered baseline is Structured 180x35 V2 (`L35V2`),
  but the complete-PIT frozen audit invalidated its legacy selection evidence:
  survivorship-complete 2023-2025 ending equity was 73.52% below the legacy-pool
  replay, and the six-year PIT durability account was nearly wiped out by an
  omitted delisting.
- The deterministic signal-close 2x2 study is complete. `V2C-P0` won with
  `Top1 / 1 slot / fixed D44`: 2023-2025 liquidated equity was CNY 9.10m,
  annualized log growth was 0.76563, and all three annual log-growth values were
  positive. The formal model registry and active execution remain unchanged.
- All 12 successful checkpoints, full-candidate predictions, 4,148 account jobs,
  reports, and study runners remain under the ignored study output for later
  reproduction. Probability, OOF-tree, and 2026 work require separate contracts
  and have not started.
- The complete PIT 180x35 pack and six 2020-2025 fold views are built with a
  60-trading-day purge. No long training task is currently running.
- The active `seq100_pit_signal_quality_v1` study has frozen `pareto_ordinal_v1`
  and the F1 profile, and model screening is COMPLETE:
  `model_screen/attempt_001/formal_matrix.json` is frozen with
  `replacement_allowed: false`. Formal fold training is 3 of 18 cells done
  (LightGBM, all three folds, seed 7); one TabM cell is interrupted and
  resumable; there is no `evaluate` and no winner. Neural runtime v2 is
  implemented and hardware-qualified. See
  `daily_research/brain/references/seq100_signal_quality_runtime_optimization_20260725.md`.
- Formal training is HELD because the frozen target is under review. A read-only
  diagnosis over the completed LightGBM folds showed the frozen design ranks path
  robustness rather than growth, and its perfect-foresight ceiling is about 23%
  of the return-only ceiling. An Oracle review accepted the diagnosis but ruled
  the current evidence insufficient to unfreeze, and ruled that any retarget
  needs a new study contract rather than an `attempt_002`. The owner has not
  chosen a path. Authoritative detail lives in `daily_research/brain/state.md`
  and `daily_research/brain/references/seq100_signal_quality_target_diagnosis_20260726.md`.
- The true-batch-1024 comparison remains paused and may resume only on explicit
  request. There is no active frontend or execution system.
- Repository simplification evidence remains
  `brain/references/repository_simplification_20260722.md`; the portable
  `workspace-brain/v1` manifests remain the takeover and protection map.
- The 2026-07-26 over-engineering retirement is complete, verified, and committed
  together with the follow-up gap work: 48 files changed, 1,880 insertions,
  11,121 deletions, plus 114 `tmp/` process artifacts untracked and gitignored
  with files left on disk. No protected dataset, research pack, or registered
  checkpoint was modified. Evidence is
  `brain/references/repository_simplification_20260726.md`.
- Both designed gaps are now implemented, verified, and committed. QDP has
  `qdp repair patch|mutate|replace-table` inside the existing `repair.py`,
  dry-run by default and requiring `--apply`, `--reason`, and
  `--expected-manifest-sha256`, with append-only in-manifest receipts carrying
  reason, previous manifest hash, affected date range, changed columns, both
  schema hashes, and a UTC timestamp. `dataset.json` `schema`/`schema_hash` is now
  authoritative and `qdp check --quick` verifies footer schema and column order.
  Three of the 14 active manifests carry legacy declarations
  (`security_identity` and `symbol_history` have `dtype` but no `type`;
  `trading_calendar` uses lowercase `string`/`bool`), so comparison is normalized
  through `canonical_manifest_schema()` and untyped declarations degrade to a
  name-and-order check plus a `legacy_untyped_manifest_schema` warning. Current
  result: status ok, 0 errors, 2 warnings, 14 datasets.
- Backfilling those legacy declarations is blocked by sequencing, not by design.
  `seq100_pit_signal_quality_v1` binds `manifest_sha256` and `schema_hash` for all
  14 domains and `seq100_signal_quality.py:554/559` raises on any change, with
  `protection.failure: stop_immediately_on_any_hash_change`. All 14 schema hashes
  currently match. Normalization becomes legal only after the rebind channel
  exists.
- Step 3 remains unstarted and needs owner sign-off: a study-scoped `rebind-qdp`
  in `seq100_signal_quality.py` that walks the receipt chain, appends a
  reason-bearing transition under `qdp_binding_amendments`, has
  `validate-research` verify every transition, and refuses when a correction
  intersects the dates a pack consumed. No `--force` escape.
- That audit left two structural gaps, now designed and scheduled. First, QDP has
  no supported schema-evolution or one-cell-correction path: `repair.py` is
  library-only with no CLI, which is what caused dated one-shot migration modules
  to accumulate. The fix is `qdp repair patch/mutate/replace-table` inside the
  existing `repair.py`, in-manifest repair receipts, authoritative
  `dataset.json` `schema`/`schema_hash` with footer drift detection in
  `qdp check --quick`, and a study-scoped `rebind-qdp` that refuses when a
  correction intersects the dates a pack consumed. Second,
  `seq100_fold_contract.py` puts `mtime_ns` and absolute paths in its backing-file
  provenance digest, so copying or restoring a pack invalidates every fold
  contract; identity becomes pack-relative path plus size plus content SHA-256 at
  `source_view_provenance.schema_version = 2`, with a narrow v1 reader so the six
  recorded `l35v2_sixfold_2020_2025` contracts and protected training summaries
  keep their registered hashes.
