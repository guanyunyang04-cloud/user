# Current project state

Updated: 2026-08-23

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

The refactor is validated by the complete suite (`165 passed`), whole-repository Ruff checks,
bytecode compilation, and the physical QDP/research checks. The canonical
curation record contains explicit machine-checkable assertions for industry,
share capital, valuation, source archives and retired domains. Older specialty
records that contain provenance but no explicit assertions are reported as
`provenance_only`; they do not certify unrelated domains and do not block the
technical path. Covered and uncertified domain lists remain explicit in the
semantic audit.

The research cutoff is now read from `data/research/daily/manifest.json` rather
than encoded as a calendar year in the active model code. New run files emit
only the generic `cutoff_violation_count`; the former year-specific field is
accepted only when reading historical results. Auxiliary updates use the
free-source tail policy by default (BaoStock, MootDX detection and CNInfo
confirmation). The unmaintainable Tushare factor, black-box money-flow and
analyst-forecast domains have been removed from the active store and their
dedicated producer code has been retired.

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

## Local minute research (2026-08-23)

- The immutable archives now live inside QDP source management at
  `data/qdp/source_archives/minute/`. The 2000-2025 one-minute ZIP is
  71,942,482,275 bytes and the 2026 multi-period ZIP is 7,464,393,924 bytes.
  They are streamed directly and are neither unpacked into a 400-GiB CSV tree
  nor copied into the iQuant private cache.
- The formal append is complete for 2010-2025 plus the available 2026 tail. The
  active store covers 2010-01-04 through 2026-08-21 with 3,301,495,126
  continuous bars and 13,756,395 separate 09:30 rows. Each domain has 200
  year/month shards and one schema; Parquet footer totals, manifest totals,
  file sizes and active-manifest bindings all agree. The compressed canonical
  bars occupy 35.03 GiB and the annual evidence occupies another 0.56 GiB.
- Every annual primary-key audit reports zero duplicates and zero unexplained
  incomplete active sessions. The importer repaired 194,394 all-zero,
  zero-flow price placeholders using a deterministic neighboring close and
  recorded every changed row. The 2026 archive also required 46,177 explicit
  `13:00` to `11:30` label repairs. Its 125 one-row suspension placeholders are
  retained but excluded from session features. No missing minute is silently
  interpolated.
- The older 2000-2025 ZIP is not uniformly exchange-grade for intraminute
  extremes. Against the trusted QDP daily bars, 1,587,530 stock-days exceed the
  one-cent OHLC tolerance, predominantly because the minute aggregate high is
  below the daily high or its low is above the daily low. Median maximum error
  among flagged rows is about 0.03 currency units; a much smaller tail contains
  obvious decimal or extreme omissions. Open, close, volume and amount are
  materially more consistent. These rows remain immutable and are listed in
  annual `minute_feature_exclusions.parquet`; the model must not treat their
  high/low-derived features as trusted observations.
- Recent source quality is much stronger. In 2025 only 57 stock-days exceed the
  one-cent price tolerance; in the available 2026 tail only five do. The 2026
  comparison covers every QDP daily reference row, with open/high/low/close
  exact rates of 99.9995%/100%/99.9998%/99.9976%. Its volume and amount p95
  relative errors are approximately 0.00108% and 0.000077%.
- A zero-volume 09:30 source row is a pre-open placeholder, not proof of an
  auction transaction. Daily parity therefore uses first/high/low/last
  trade-bearing bars with an all-zero-day fallback. The opening row stays in a
  separate domain so a model can choose its treatment explicitly.
- Earlier 64/256-symbol staging Parquets and parity extracts were deleted after
  the formal import; they are reproducible from the ZIP and no active manifest
  referenced them. The failed model result and compact research record remain
  as evidence, not as a second market-data source.
- The one-minute table is the only canonical intraday fact table. A model may
  resample 5/15/30/60-minute views in a query or data loader, but QDP will not
  retain four duplicate long-term truths. The importer performs one ZIP member
  pass for all requested years, uses PyArrow parsing and monthly buffered
  Parquet writes, and now sizes/flushes those buffers from live available RAM.
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
- QDP source archives now own BaoStock evidence, both minute ZIPs, and the
  Shenwan industry/index files. The active manifest no longer exposes the
  obsolete factor, black-box money-flow or Tushare analyst-forecast datasets.
  Retained historical research-report metadata is separate from the deleted
  forecast panel.

## Immediate next action

1. Build the first full-market hybrid research view directly from canonical 1m:
   prior-close daily context plus the minute sequence available at each decision
   time. Use 2010-2011 only for warm-up and begin formal samples in 2012.
2. Treat daily-parity failures as field-level quality masks, not as a reason to
   delete a stock-day or redefine the historical universe. Start with the more
   reliable close/volume/amount path; use minute high/low only where certified,
   with trusted daily range available as prior-day context.
3. Resample multi-scale views on demand rather than materializing permanent
   5/15/30/60-minute copies. First establish a simple sequence baseline and an
   executable T+1 target, then test whether intraday timing adds value over the
   current daily score. iQuant remains the eventual current-day/ordering layer;
   QDP remains the reproducible historical store.
