# Current workspace state

Updated: `2026-07-25`

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
- The active `seq100_pit_signal_quality_v1` study has frozen
  `pareto_ordinal_v1` and the F1 snapshot profile. Model screening is paused:
  LightGBM and TabM prescreens are complete, PatchTST has no completed
  checkpoint, and no formal model matrix or formal fold training exists yet.
  The resumable process evidence is under the ignored study output. Neural
  runtime v2 is now implemented and hardware-qualified, but screening has not
  resumed: the old TabM prescreen is intentionally invalid under exact-batch
  semantics, while LightGBM remains reusable. The concise evidence pointer is
  `daily_research/brain/references/seq100_signal_quality_runtime_optimization_20260725.md`.
- The true-batch-1024 comparison remains paused and may resume only on explicit
  request. There is no active frontend or execution system.
- Repository simplification evidence remains
  `brain/references/repository_simplification_20260722.md`; the portable
  `workspace-brain/v1` manifests remain the takeover and protection map.
