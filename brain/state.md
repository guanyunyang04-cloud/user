# Current workspace state

Updated: `2026-07-22`

- QDP is usable with `active_as_of_date=2026-07-21`; quick validation passes.
- Protected QDP and research datasets remain in place.
- The stable model registry contains 12 bundles: Legal flat, Structured 100×32,
  and Structured 180×35 V2 for 2023–2026.
- The selected research baseline is Structured 180×35 V2 (`L35V2`). Selection is
  based on 2023–2025 evidence; 2026 is confirmation, not model selection.
- Two studies remain active but paused: L35V2 folds for 2020–2022 followed by a
  six-fold summary, and the later batch-1024 comparison.
- Global-tail, intraday-feature, Capital Speed V3, and bounded-training-window
  routes are retired. Their compact evidence is retained; their runners are not
  part of the current architecture.
- There is no active frontend or execution system. Both will be redesigned from
  scratch if needed.
- Repository simplification completed on 2026-07-22: four active top-level
  systems, nine Seq100 core modules, 12 registered model bundles, 12 compact
  research records, and two paused study contracts remain. The retained core
  regression set has 210 passing tests.
- The full cleanup evidence is
  `brain/references/repository_simplification_20260722.md`.
