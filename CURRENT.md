# Current project state

Updated: 2026-08-21

## Objective

Evaluate repeatable A-share cross-sectional ranking signals under point-in-time
information timing and legal after-cost execution. Results are probability
evidence, not a promise of profit. No live trading is active and research
never reads outcome data after the cutoff declared in the active research
manifest.

## Active paths

- Code: `src/quantlab/data` and `src/quantlab/research`.
- QDP data: `data/qdp/qdp_v2/active/active.json`.
- Research contract: `data/research/daily/manifest.json`.
- Model outputs: `runs/daily/`.
- Durable evidence: `research/records/` and `research/studies/`.

The former `daily_research` and `quant_data_platform` trees were removed from
the active repository. Their tracked source history remains in Git. The large
ignored experiment artifacts previously staged at
`H:\quant_project_archive\20260820` were purged on 2026-08-21; only small
audit, migration, and legacy-provenance files remain there.

The unified package passes the full test suite and static checks. The QDP
physical contract checks are green; the deep check reports one expected
medium-coverage warning because historical 5-minute bars are unavailable for
some restored daily stock-days. That gap is explicit in the QDP contract and
is not used as a hidden eligibility filter.

## Refactor status

The active package now keeps provider protocol code separate from deterministic
normalization code. `data/provider_symbols.py` owns provider symbol
conversions, `data/identifiers.py` owns stable security identity mappings,
`data/qdp_v2/normalization.py` owns auxiliary payload normalization, and
`data/qdp_v2/pit_normalization.py` owns PIT history transformations. The
original modules retain compatibility names, so no data update or research
entry point changed during the split.

The remaining large provider and QDP repair modules still contain network,
retry, multiprocessing, and DuckDB orchestration by design. They are not part
of the current research import path, and further splitting them should wait
for a behavior-level need rather than create another layer of wrappers.
The current refactor is validated by the complete test suite (`163 passed`),
Ruff, bytecode compilation, and the physical QDP/research checks. The selected
semantic invariants pass. Five specialty repair records contain explicit
machine-checkable checks; three older records (research reports, quarterly
statements, and the legacy Tushare backfill) contain provenance and workflow
statistics but no explicit checks, so the semantic command now reports
warnings and marks them `provenance_only` without blocking the technical path.
The active store also contains domains outside the specialty pass; covered and uncertified domain
lists are recorded in the audit output. This is an evidence boundary, not a
failure of the 158-field technical research matrix.

The research cutoff is now read from `data/research/daily/manifest.json` rather
than encoded as a calendar year in the active model code. New run files emit
only the generic `cutoff_violation_count`; the former year-specific field is
accepted only when reading historical results. Auxiliary updates use the
free-source tail policy by default (BaoStock, MootDX detection and CNInfo
confirmation).
The historical Tushare-dependent repair path is isolated behind the explicit
`--legacy-tushare` option.

## Data and execution contract

- 4,191,476 unique stock-days, with 2010-2011 used only as warm-up and a
  manifest-declared maximum outcome date.
- 158 fields are stable daily price-volume and market-state inputs. 183 adds
  25 same-day 5-minute summaries and remains a challenger, not the default.
- Five expanding forward folds cover the configured validation window with a
  30-day purge.
- The primary target is executable D10 net return ranking. Replay uses next-open
  entry, T+1, suspension/limit restrictions, 100-share lots, finite cash,
  delayed legal exits, fees, and double-slippage stress.
- Missing, unfilled, or unaffordable selections remain cash; they are not
  replaced after looking at future outcomes.

## Established findings

- The corrected 158-field tree and raw-60-day sequence both carry weak positive
  out-of-fold ranking information. Their errors are complementary, so a frozen
  50/50 rank ensemble is the current research benchmark.
- The 183-field view has not shown stable incremental economic value over 158;
  minute fields receive little tree gain and may be more useful for shorter
  execution horizons than for D10 selection.
- The benchmark's raw return still depends materially on a small number of
  right-tail winners. Winner-capped and seed-stability results are the gating
  evidence before any paper-trading candidate is considered.
- Financial, announcement, and revision fields are not default inputs. They
  must be PIT- and age-aware and prove residual value over the technical
  baseline.

## Operational status

- H: filesystem repair is complete. `chkdsk H: /f` verified 543,565 files,
  found no problems and no bad sectors; the volume now reports
  `HealthStatus=Healthy`, `OperationalStatus=OK`, and `fsutil dirty query H:`
  returns `NOT Dirty`. iQuant is currently stopped.
- iQuant's own download log records a completed task message for 2,170,416
  one-minute requests and 50,280 daily requests. The platform data directory
  contains about 4.8 GiB of `.DAT` files, but the A-share minute cache currently
  has only 276 SH files and 5 SZ files (the roughly 630-file total also includes
  other markets). The external `xtdata` probe still returns empty frames because
  its RPC data directory is unset/defaulted to `userdata_mini`. This is useful
  evidence for a recent-data adapter, not a replacement for the project's
  complete PIT history. No trade or order API has been called by this project.
- QDP contains optional historical repair modules with frozen legacy contracts
  and Tushare provenance. They are outside the current technical research
  import path; the default tail updater now uses free sources, while replacing
  old provenance-only repair code is a separate migration, not evidence that
  the 158-field research matrix is invalid.

## Immediate next action

1. Copy a small iQuant sample to a project-owned
   staging area and compare it with the existing daily/5-minute bars. Do not
   treat the platform cache as a complete historical source until symbol,
   date, adjustment and unit coverage are measured.
2. Rerun the predeclared raw-60 seed check and paired ensemble increment using
   the unified package. Do not search blend weights, TopK, exit grids, minute
   subsets, or LightGBM parameter grids on the already-used folds. Build a
   historical/live parity path only after winner-cap and seed-stability checks
   pass.
