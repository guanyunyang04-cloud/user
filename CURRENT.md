# Current project state

Updated: 2026-08-24

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

The source-tree consolidation and large-module split are complete, but the QDP
storage refactor is not. Legacy large modules were divided into focused
packages and helpers, and provider transport, normalization, preparation,
installation, auditing, workflow state, model evaluation, and account replay
now have explicit owners. QDP still uses the `qdp_v2` Python namespace and
physical root, an `active.json` generation pointer, hashed dataset directories,
and some cross-generation shard references.

This is architecture debt, not evidence of damaged data. The current 26 active
domains pass verified physical status and the quick database check with zero
errors and warnings. The migration must therefore preserve current files and
PIT behavior while simplifying the layout. The intended end state is one
canonical directory per domain and one compact per-domain contract, with
atomic temporary installation retained for crash safety. Removing every
manifest or renaming directories without a transactional migration would be a
regression, not simplification.

The latest inventory found 42 physical dataset-generation directories for 26
active domains, including 16 inactive generations. The active 5-minute
manifest still references 81 shards stored in an earlier generation. Active
metadata contains 150 absolute path strings; 111 no longer exist and 110 point
at the removed `quant_data_platform` tree. These stale prepared-source paths
do not invalidate active Parquet shards, but they must be removed or converted
to durable workspace-relative provenance during the QDP migration.

Shared SHA-256 and atomic file installation live in `quantlab.core.io`.
Production code has no mixed-responsibility function of 100 lines or more. The
only function above that threshold is `_prepared_sql`, a single named DuckDB
query builder whose complete relational statement is intentionally kept
together. Large files such as PIT preparation and tree research now consist of
small cohesive helpers rather than monolithic control flow.

The project `brain/` directory and the user-level `workspace-brain` skill were
deleted. This file is the sole concise cross-session handoff; detailed evidence
belongs in `research/records/` and code behavior belongs in tests.

The completed source refactor and current minute-quality policy are validated
by the complete suite (`166 passed`),
whole-repository Ruff checks, bytecode compilation, and the physical
QDP/research checks. The canonical
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
  prices. After three evidence-gated repairs, 427,727 stock-days exceed the
  agreed absolute tolerance of 0.05 currency units. Of these, 247,714 stay as
  warnings because the maximum error is at most 0.5% of the maximum absolute
  daily OHLC; 180,013 are field-level unreliable observations, including
  56,122 above 1%. The former one-cent exclusion rule is retired.
- Price evidence is field-level rather than a stock-day deletion. Among the
  180,013 unreliable rows, 126,805 affect only high/low, 46,863 affect only
  open/close (almost entirely open), and 6,345 affect both groups. Annual
  `minute_feature_exclusions.parquet` files now carry explicit
  `exclude_open/high/low/close` masks. Volume and amount retain independent
  relative-error audits. A full-session parity outcome is not causal before
  the close and must never be supplied to a same-day intraday model.
- Recent source quality is much stronger. Under the new policy, 2025 has five
  rows above 0.05 (three field-level unreliable), and the available 2026 tail
  has two (one unreliable). The 2026
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
  returns `NOT Dirty`. The 2026-08-23 takeover check found 784.73 GiB free.
  H: is exFAT, so the QDP migration cannot rely on NTFS hard links or metadata
  journaling. It needs a resumable move journal and per-domain validation before
  each commit point, especially because no full duplicate data backup is planned.
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

## Temporary Tushare-compatible repair source (2026-08-24)

- The private provider profile at
  `data/qdp/qdp_private/provider_profiles.json` now points to the restored
  `ts.gyzcloud.top/api` source. The tier-15000 weekly entitlement expires at
  2026-08-30 16:46 Asia/Shanghai and advertises 150 requests/minute. The source
  separately enforces 20,000 `stk_mins` calls per calendar day. The direct-HTTP
  client defaults to 96 requests/minute with three workers and one shared
  lock-serialized limiter; targeted repair may explicitly use 140. HTTP 429 now
  stops the queued run immediately instead of retrying every pending batch. No
  active code or tracked evidence contains the token.
- This provider is approved only as a candidate repair and cross-validation
  source. Its public client bundle exposes an administrative control described
  as randomly polluting a marked user's API responses, as well as throttling
  controls. Repeated hashes, seven historical daily references and the recent
  tail reconciliation found no evidence that the current account is affected,
  but every formal install still requires schema/grain, repeatability,
  endpoint-freshness and independent-source/QDP gates.
- Read-only probes succeeded for daily prices, adjustment factors, daily basic,
  stock status, limits, four financial-statement/indicator families, forecast,
  margin, research reports, per-symbol name history and dividends. Margin,
  margin-detail and research-report data were populated through 2026-08-20 but
  empty for 2026-08-21, so freshness is endpoint-specific. A bulk name-history
  request returned exactly 10,000 rows and historical records outside the
  requested window; the existing per-symbol implementation must be retained.
- The earlier inference that tier 15000 excluded stock minutes was wrong. The
  current token actually returns `stk_mins` for A-share stocks at
  1/5/15/30/60-minute frequencies: a normal day returned 241/48/16/8/4 rows,
  both 2009 and 2010 historical one-minute probes returned 241 rows, and a long
  query stopped at the documented 8,000-row limit. Both `pro.stk_mins` and
  `ts.pro_bar(api=pro, freq="1min")` work through the proxy. Actual calls, not
  the public tier label, are authoritative for this vendor's bundled weekly
  permission.
- The permission is specifically A-share stock history, not every minute
  family. `etf_mins`, `idx_mins`, `ft_mins` and `opt_mins` returned HTTP 403;
  opening/closing auction calls returned permission code 40101, and `anns_d`
  returned HTTP 403. Continue using the existing CNInfo announcement path and
  the QDP auction domain unless a separate entitlement is bought.
- Stock-minute access makes targeted repair possible, but not safe as a bulk
  replacement. Six known bad stock-days across 2010-2026 were compared against
  the old ZIP and trusted daily bars. The proxy repaired the absurd 2010
  `600000.SH` high of 189.10 back to 19.17, but its volume for that day was far
  below the daily reference. On the other five samples the proxy and ZIP shared
  the same missing open/high/low extremum. Repairs must therefore be accepted
  per field only when the proxy improves daily reconciliation without
  regressing another price or flow field; existing quality masks stay active.
- The repair workflow is now implemented and has formally installed three
  price-only batches: 45 priority stock-days, 653 open errors above 2%, and
  4,350 open errors in `(1%, 2%]`. In total it changed 5,048 minute rows and
  removed 4,878 stock-days from the unreliable set. Every accepted row has a
  captured provider response and SHA-256, a complete 241-row session, daily
  parity evidence, a pre-install backup and an annual re-audit. Volume and
  amount were retained from QDP. Continuous and auction row counts remain
  3,301,495,126 and 13,756,395, with 200 shards in each domain; verified status
  and the quick QDP check are both `ok`.
- Five explicit fallback runs on 2026-08-25 then evaluated 32,518 unique
  stock-days from the purchased archive's unresolved queue. Four installable
  runs changed another 1,607 price rows; 1,578 stock-days fully left the
  current quality mask and 29 retained a different flagged price field. The
  cumulative Tushare-compatible repair count was then 6,655 minute rows.
  The useful hits were almost entirely open repairs: a deferred 19,254-day
  high/low-era batch accepted only 19 open changes, while a severe 874-day
  sample accepted none. Do not spend another daily quota on a bulk high/low
  retry against this provider.
- On 2026-08-26 the remaining 19,049 unattempted open stock-days were queried
  with a five-trading-day range cap instead of 33. The run captured 17,124 new
  response batches and 4,725,768 raw rows, versus about 20.3 million rows under
  the old 33-day plan. It reused 723 prior captures, accepted and installed
  6,631 open rows, and rejected 12,415 same-source errors plus three incomplete
  provider sessions. It fully cleared 6,625 stock-days; six accepted rows retain
  another field mask. The cumulative Tushare-compatible repair count is now
  13,286 minute rows. An interrupted Windows/DuckDB annual audit was safely
  resumed without another download; the recovery path now creates untouched
  years' missing pre-audit backups and has a focused regression test.
- The 2026-07-22 through 2026-08-21 cross-check joined 73,364 common main-board
  stock-days. All had 241 source rows; 73,359 matched daily OHLC exactly, four
  stayed inside the normal absolute tolerance, one open was warning-only, and
  none were unreliable or severe. No volume or amount row exceeded 0.1%
  relative error. This independently confirms that the recent 2026 minute ZIP
  is high quality and that the much older minute anomalies are genuine source
  defects rather than bad daily references.
- Reuse the client, normalizers and PIT contracts, not the old one-time command
  dates. The active daily cutoff is still 2026-07-21, so the auxiliary planner
  correctly rejects a 2026-08-21 target until market daily is advanced first.
  Several statement, research and margin workflows still freeze 2025 dates;
  they need explicit as-of and observed-cutoff inputs before a 2026 incremental
  run. Provider capability evidence remains in
  `research/records/tushare_compatible_provider_probe_20260823/result.json`;
  installed minute-repair evidence is under
  `data/qdp/source_archives/tushare_compatible/minute_repair/`.

## Purchased local historical minute repair (2026-08-25)

- `H:\BaiduNetdiskDownload\量化数据\stock_1min` contains 5,826 canonical
  per-symbol Parquet files. The 43 parenthesized `(1)` files are duplicate
  copies and are explicitly excluded. The new adapter is read-only against
  that directory and reuses the existing minute-repair evaluator and
  installer; it does not create a second mutation path.
- The `(0.5%, 1%]` open run checked 34,642 audited stock-days. Exact session
  validation accepted 6,172 and installed 6,172 row repairs. The high/low run
  skipped 2019-01-01 through 2021-03-31 for later fallback, checked 113,630
  stock-days, accepted 107,467 stock-days and installed 108,161 row-level
  repairs. Only prices were overlaid; QDP volume and amount were retained.
- High/low installation was gated by timing evidence rather than daily extrema
  alone. The local source reproduced the trusted daily high/low on 261/263
  stratified samples. BaoStock supplied 83 complete five-minute sessions; on
  the 53 sessions where BaoStock itself reproduced the trusted daily extreme,
  52 used the same five-minute bucket as the local source. The combined gate
  passed.
- The two local runs removed 109,935 stock-days from the unreliable set. The
  immediate post-local audit had 70,078 price-unreliable days and 25,867
  severe days; including 125 session exclusions, the total minute-feature
  exclusion count is 70,203. Continuous and auction domains remain at
  3,301,495,126 and
  13,756,395 rows with 200 shards each. Verified status and the quick QDP check
  are both `ok`.
- The original all-target scan is retained with the fallback evidence: 41
  stock-days are absent from the local archive and 14 contain an invalid price
  row. A consolidated 53,121-stock-day queue is ready for later targeted
  `stk_mins` repair; the other 16,957 unresolved days retain their current
  masks. Evidence and hashes are under
  `data/qdp/source_archives/external_quant_data/minute_repair/`. This purchased
  archive is minute-only for QDP purposes. It was not used to extend or repair
  the daily, factor, status, limit or valuation domains.
- The 2026-08-25/26 Tushare fallback reduced the consolidated queue from
  53,121 to 44,918 current stock-days and reduced the global
  price-unreliable count to 61,875. Severe remains 25,867; with 125 session
  exclusions, the total minute-feature exclusion count is 62,000. Of the
  remaining queue, 1,554 high/low-only stock-days are unattempted, 43,329 were
  tried without an accepted improvement, and 35 accepted partial repairs
  retain another field mask.
  Resumable queue files and run accounting are under
  `data/qdp/source_archives/external_quant_data/minute_repair/tushare_fallback_after_local_20260825/`.

## Immediate next action

1. Do not resume the slow bulk `open_d38bfb1ac3abe97a` download and do not issue
   another bulk Tushare minute run. Every remaining unattempted fallback row is
   high/low-only, where this compatible source's measured hit rate was
   effectively zero. Keep the 44,918 unresolved queue masks until a genuinely
   independent historical source is available. Daily/factor/basic and PIT
   auxiliary tails remain separate and must continue through the existing
   local/free-source update path.
2. Finish the QDP storage migration before starting another large model run.
   Rename the code namespace to `quantlab.data.qdp`, move the physical QDP root
   up one level without copying the 42-GiB store, flatten each active domain to
   one canonical directory, and remove generation pointers and cross-generation
   shard references. Keep a compact domain manifest and atomic file replacement
   so PIT contracts, row counts, schemas, source identity, and crash safety are
   not lost.
3. During that migration, convert durable provenance to workspace-relative
   paths, discard stale prepared-runtime paths, classify rather than blindly
   delete old Tushare-derived historical facts, and retire the permanent 5m
   fact table only after its remaining consumers use deterministic 1m
   resampling. Runtime and archive directories require a reference/provenance
   audit before deletion.
4. After the migrated store passes physical, PIT, and research-contract checks,
   build the first full-market hybrid view from prior-close daily context plus
   the causal minute sequence available at each decision time. Use 2010-2011
   only for warm-up, begin formal samples in 2012, apply the new field-level
   quality masks causally, and keep iQuant as the current-day execution layer.
