# Current workspace state

Updated: `2026-07-23`

- QDP is usable with `active_as_of_date=2026-07-21`; quick and full validation
  pass. PIT main-board daily history now includes historical ST, long
  suspensions, and delisted securities; historical 5-minute gaps remain an
  explicit non-blocking warning.
- Daily domains are normalized to `symbol_history` effective ticker intervals
  for both code-change identities. The 5-minute dataset was not rewritten.
- Protected QDP and research datasets remain in place.
- The stable model registry contains 15 bundles: Legal flat and Structured
  100×32 for 2023–2026, plus Structured 180×35 V2 for 2020–2026.
- The selected research baseline is Structured 180×35 V2 (`L35V2`). Selection is
  based on 2023–2025 evidence; 2026 is confirmation, not model selection.
- The L35V2 2020–2025 six-fold durability study is complete. Rank IC is positive
  in all six folds; the pure-growth account winner is Top1/1 + D14, while the
  highest-growth all-years-positive strategy is Top3/3 + D42. The later true
  batch-1024 comparison is the only paused study.
- Global-tail, intraday-feature, Capital Speed V3, and bounded-training-window
  routes are retired. Their compact evidence is retained; their runners are not
  part of the current architecture.
- There is no active frontend or execution system. Both will be redesigned from
  scratch if needed.
- Repository simplification completed on 2026-07-22: four active top-level
  systems, nine Seq100 core modules, 15 registered model bundles, 13 compact
  research records, and one paused study contract remain. The retained core
  retained regression set has 214 passing tests (136 research/path-policy and
  78 QDP/brain integrity tests).
- The full cleanup evidence is
  `brain/references/repository_simplification_20260722.md`.
