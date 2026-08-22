# Current project state

Updated: 2026-08-22

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

The repository-wide refactor is complete to the agreed standard: legacy large
modules were not merely moved; their mixed internals were divided into focused
packages and helpers. Provider transport, normalization, retry, preparation,
installation, auditing, workflow state, model evaluation, and account replay
now have explicit owners. Former module import paths remain available through
small package facades, so callers did not need a migration.

Shared SHA-256 and atomic file installation live in `quantlab.core.io`.
Production code has no mixed-responsibility function of 100 lines or more. The
only function above that threshold is `_prepared_sql`, a single named DuckDB
query builder whose complete relational statement is intentionally kept
together. Large files such as PIT preparation and tree research now consist of
small cohesive helpers rather than monolithic control flow.

The project `brain/` directory and the user-level `workspace-brain` skill were
deleted. This file is the sole concise cross-session handoff; detailed evidence
belongs in `research/records/` and code behavior belongs in tests.

The refactor is validated by the complete suite (`169 passed`), whole-repository
Ruff checks, bytecode compilation, and the physical QDP/research checks. The
selected semantic invariants pass. Five specialty repair records contain explicit
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

## Local minute research (2026-08-22)

- Two downloaded archives are preserved outside Git under
  `H:\BaiduNetdiskDownload\量化数据`: the 2000-2025 1-minute archive is about
  71.94 GB compressed (about 377.55 GiB unpacked), and the 2026 multi-period
  archive is about 7.46 GB compressed (about 29.52 GiB unpacked). They are not
  copied into the iQuant cache.
- `quantlab.data.minute_archive` selectively reads ZIP members and writes
  standard Parquet without unpacking the archives. The active research extract
  contains 256 fixed symbols and only `09:31`-`10:10`: 14,752,840 rows for
  2020-2025. A separate 1,341,400-row 2026 extract is staged for a future
  research window but is not read by the formal run. The standalone `09:30`
  auction row is excluded and the manifest declares raw prices.
- The auction row is retained in a separate parity extract. For two symbols and
  485 dates (46,560 five-minute bars), folding `09:30` into the `09:35` bucket
  reproduces the active QDP intraday archive exactly for OHLC and volume; amount
  differs only by floating-point rounding. The iQuant minute cache starts at
  `09:31`, so it is a different first-bucket convention and must not be mixed
  without an explicit transformation. Evidence is in
  `research/records/minute_local_parity_probe_20260822.json`.
- The executable minute contract is now explicit: use information through
  10:00, rank candidates before seeing the fill window, enter at the
  10:01-10:10 VWAP, and begin exits in the same window on the next market day.
  Prices are adjusted across days with the QDP factor. Entries and each partial
  exit are capped at 1% of window volume; ordinary-share T+1, ST/suspension
  state, fees, slippage, one-price windows, finite cash, overlap, and delayed
  exits are simulated. One-price entries are rejected; a one-price-down exit is
  delayed while a one-price-up exit remains sellable.
- The final expanding walk-forward test uses 368,821 stock-days and predicts
  each year from 2021 through 2025 using only prior years. Its contract filters
  source bars at `2025-12-31`; the last complete T+1 signal is therefore
  2025-12-30. Across 304,852
  observed OOF labels, the seven-feature morning LightGBM has Rank IC
  `-0.00967`; four of five years are negative. Top-5 mean action gross return is
  `-0.00463%` per signal day even though the conditional universe mean is
  `+0.05168%`. The base-cost account loses `90.24%` with a `-91.26%` maximum
  drawdown; double-slippage stress loses `98.51%`. Every evaluation year loses
  money under both scenarios. Artifacts are under
  `runs/minute_walk_forward_256_2021_2025`.
- This rejects the current combination of seven morning summaries, fixed
  10:00 decision, and one-night target. It does not show that minute data are
  useless. The sample is not the full market, the model has no prior-day trend
  context or raw minute sequence, and it does not model Level-2 queue position.
  The failed baseline is not a paper- or live-trading candidate and remains
  separate from the 158/raw-60 daily benchmark.

## Operational status

- H: filesystem repair is complete. `chkdsk H: /f` verified 543,565 files,
  found no problems and no bad sectors; the volume now reports
  `HealthStatus=Healthy`, `OperationalStatus=OK`, and `fsutil dirty query H:`
  returns `NOT Dirty`.
- At the frozen parity snapshot, iQuant was running and actively writing. The installed
  `xtdata.get_local_data` is an empty compatibility stub, so the project now has
  a tested read-only DAT adapter. It decodes the fixed 64-byte K-line records as
  Asia/Shanghai right-edge bars, price `/1000`, volume lots `*100`, amount in
  currency units, and raw/unadjusted prices.
- The frozen 2026-08-21 16:05 parity sample compared 15,930 common daily rows:
  open/high matched exactly, low matched 99.987%, close matched 99.962%, and
  volume/amount p95 relative errors were 0.00070%/0.000081%. QDP alone contained
  one additional day. The 1-minute-to-5-minute test compared 864/864 common
  buckets across two stocks and nine dates; every key aligned and all 18 stock-
  days produced 48 bars. In the 288 buckets from 2025 onward, OHLC matched 100%,
  volume p95 relative error was 0.0248%, and amount p95 was below 0.000006%.
- The cache was not complete at that snapshot: 867 one-minute files occupied
  2.16 GB, only 60 matched the 3,416 symbols with QDP daily bars (1.76%), and the
  latest file write occurred 3.5 seconds before the scan ended. Older minute
  sources also show small OHLC differences and larger relative flow differences
  in thin buckets. iQuant is therefore approved for recent/live supplementation
  and independent validation, but not for replacing or selectively filling the
  historical QDP minute table. Evidence is in
  `research/records/iquant_cache_parity_20260821/result.json`; no trade or order
  API was called.
- QDP contains optional historical repair modules with frozen legacy contracts
  and Tushare provenance. They are outside the current technical research
  import path; the default tail updater now uses free sources, while replacing
  old provenance-only repair code is a separate migration, not evidence that
  the 158-field research matrix is invalid.

## Immediate next action

1. Freeze one paired minute increment: compare a prior-close daily technical
   score, the existing morning-only score, and a hybrid using the prior-close
   daily structure plus information available by 10:00. Keep the same 256
   symbols, annual expanding folds, execution contract, Top-5 rule, and costs.
   This tests whether minute state improves a known daily signal without an
   exit-time or parameter search.
2. Only if the hybrid adds stable OOF and account value, test a raw 30-minute
   sequence encoder and predeclared D1/D3 targets. Do not tune decision time,
   TopK, holding period, or feature subsets on these same folds.
3. After iQuant account/API access and a stable download are available, rerun
   both parity commands against the complete cache. Use iQuant for current-day
   supplementation and execution; keep the project Parquet store as the
   reproducible historical source unless whole-cache coverage proves otherwise.
