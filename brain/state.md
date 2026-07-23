# Current workspace state

Updated: `2026-07-24`

- QDP is usable with `active_as_of_date=2026-07-21`; quick and full validation
  pass. PIT main-board daily history includes historical ST, long suspensions,
  and delisted securities. Historical 5-minute gaps remain an accepted
  non-blocking warning.
- Daily domains are normalized to `symbol_history` effective ticker intervals
  for both code-change identities. The 5-minute dataset was not rewritten.
- Protected QDP datasets, research packs, and registered checkpoints remain in
  place. The stable model registry still contains 15 bundles.
- Daily Research's frozen registered baseline is Structured 180x35 V2 (`L35V2`),
  but the complete-PIT frozen audit invalidated its legacy selection evidence:
  survivorship-complete 2023-2025 ending equity was 73.52% below the legacy-pool
  replay, and the six-year PIT durability account was nearly wiped out by an
  omitted delisting.
- The current research route is the deterministic signal-close semantic 2x2 in
  `daily_research/studies/signal_close_path_value_2x2_v1.json`. It compares V2C
  and V4 with and without rank under one PIT pack and one execution contract.
  Probability models and OOF trees remain gated behind a deterministic winner.
- The complete PIT 180x35 pack and six 2020-2025 fold views are built with a
  60-trading-day purge. No long training task is currently running.
- The true-batch-1024 comparison remains paused and may resume only on explicit
  request. There is no active frontend or execution system.
- Repository simplification evidence remains
  `brain/references/repository_simplification_20260722.md`; the portable
  `workspace-brain/v1` manifests remain the takeover and protection map.
