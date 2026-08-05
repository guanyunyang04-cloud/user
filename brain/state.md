# Current state

Updated: 2026-08-05

## Objective

Build a daily A-share main-board selector that keeps upside opportunity, path
quality, and pre-peak adversity separate. Entry output structure is frozen,
the supervised post-entry update challenge is finished, and the first
continuous executable-account monetization audit plus a true-label economic
ceiling and candidate-aligned prediction-to-oracle loss audits are complete.
The strict T+1 direct-return heads and their bounded low-position execution
audit are also complete. No production deployment policy, stop, deep model, or
reinforcement-learning policy has been selected.

## Current QDP and pre-training state (2026-08-02)

The pre-training data repair, storage compaction, canonical freeze chain,
descriptive feature audit, and takeover semantic corrections are complete. The
first core-only retrospective rolling model study is also complete and audited.
The quality-liquidity
preparation workflow now has one current study ID and output directory per
stage; superseded numbered preparation artifacts were removed after the
canonical chain passed evaluation. QDP content-addressed dataset IDs remain as
an internal data-integrity mechanism. Cleanup removed nine superseded study
directories (1,283 files, 11.882 GiB logical) and nine zero-reference QDP
dataset versions (699,722,187 bytes); the post-cleanup QDP GC dry-run is empty.

The active QDP catalog contains 27 immutable domains. The current quick audit
and selected-invariant semantic audit v2 both have `status=ok` and no errors.
The last balanced full database audit predates the two semantic-only successors;
it passed with only the known medium-severity
`historical_5m_unavailable_for_restored_daily_symbols`. The successors preserve
rows and all out-of-scope fields and passed focused invariants, so the full
audit was not repeated merely for targeted state/boolean changes. Quick/full
audit scope is not a claim that every active domain has complete semantic
certification.

- Daily-task-closed `report_rc` v2 contains 1,274,918 reports and 2,764,977
  forecast rows for 2010-2025. Every natural-day task has a terminal state,
  2021 is restored, full pages are continued by offset, and confirmed empty
  days require two consistent responses. Reports become visible on the next
  exchange-open day.
- `balance_sheet_quarterly__c97d2e51362266b0b097f3bb` retains its 179,220
  PIT rows and adds separately named total other-receivable/payable fields,
  contract liabilities, a customer-liability combination, and explicit field
  states. It does not overwrite component fields or turn not-applicable and
  unknown values into zero. Its extension-conflict flag compares only the four
  extension numeric values: 314 events conflict. The earlier broad flag found
  2,723 whole-balance-sheet conflicts and is retained in the superseded version.
- `share_capital__dd8dd0446ecc5d2343606d14` has 10,212,710 rows. Missing
  `total_share` values were repaired only from same-day provider evidence after
  converting ten-thousand-share units to shares. Existing non-null values were
  preserved. `valuation__b94bd77f7ef9b261baf80c5c` was rebuilt so total and
  float market value equal unadjusted close times the corresponding share
  count within tolerance.
- `margin_eligibility__17297a42fd20f688b435b218` has 9,710,637 exchange-date
  states with `eligible_observed`, `known_ineligible`, and
  `source_unavailable` semantics. `margin_detail__2ed8aba21121548425f56b7c`
  has 3,927,235 rows after evidence-preserving repair. SSE detail presence
  proves eligibility, but absence is unknown rather than a negative eligibility
  list; SZSE keeps its explicit positive/negative eligibility semantics.
- The external archive repair added 129,116 independently validated stock-days
  (6,197,568 bars) without overwriting existing keys. Active
  `market_intraday_5m__2c071f88738b5cdea4695f16` has 472,515,360 rows. The
  quality pool gained 35,450 complete days and retains 152 explicit minute
  gaps; only three are in 2023-2025. All accepted days have the standard 48
  bars and pass daily price plus volume/amount reconciliation.
- Runtime archive tooling sealed 151 workflow-domain-year units containing
  73,357 expanded files. Hash verification and a full-year restore check
  preceded explicit deletion. The archive is about 21.23 GiB and the cleanup
  receipt estimates 71.66 GiB of exFAT allocation reclaimed. Stable `tmp`
  artifacts and old large research stores remain untouched because they are
  still referenced or are not small-file-allocation problems. Future seals
  preserve a content-addressed redacted state snapshot and richer request-ledger
  fields; existing verified archives are not rewritten.

The canonical freeze chain is now the source of truth for new research:

- `seq100_quality_liquidity_data_prep` materializes both pools without
  changing the pool thresholds. The formal 2011-2025 daily pool has 4,477,003
  rows; the complete daily/minute support has 4,476,851 rows. 2010 is burn-in
  only and 2026 reads/writes are zero.
- `seq100_quality_liquidity_research_scope` pins repaired dataset IDs and
  both row spines. Expanding training counts for the 2023/2024/2025 folds are
  3,242,301, 3,652,008, and 4,064,304.
- `seq100_quality_liquidity_training_ready` is
  `ready_with_documented_optional_gaps`: 518 existing features, 106 new formal
  candidates, 83 diagnostic-only fields, and four availability-gated numeric
  fields. QFQ, failed-HFQ, and `net_mf_*` remain diagnostic-only. Availability
  metadata is separate from economic zero values. It has 13,863 stock-days with
  exact extension conflicts and zero non-null gated numeric cells; correcting
  the earlier broad conflict flag released 97,945 stock-days from unnecessary
  gating.
- Historical model experiments such as `seq100_entry_contract_oos_v4` remain
  readable and are not part of this preparation-chain cleanup. Future model
  reports using knowledge from all 2011-2025 descriptive research are called
  retrospective rolling OOS rather than untouched confirmatory OOS.

`seq100_quality_liquidity_descriptive_feature_audit` completed on every one
of the 4,476,851 formal rows and 628 numeric candidates, with overall, annual,
and three-period views. It used daily cross-sectional Top-1%/Top-5% true MFE
labels, separate risk/state coordinates, explicit source-state semantics, and
market-wide time-series deciles where daily cross-sectional deciles would be
degenerate. Redundancy was the only sampled calculation (4,975 deterministic
rows); the relationship tables use the full population.

- Eleven families remain first-model ablation candidates under the corrected
  historical-reversal baseline and independent-representative gate:
  announcements, daily cross-sectional technical, daily price/volume
  technical, financial statements, industry context, market state, same-day
  5-minute, size/liquidity/status, structured financial summary, traditional
  moneyflow, and traditional technical indicators.
- Financial-statement extensions, margin detail, margin market, and research
  reports are `availability_gated_family`; no family is automatically frozen
  into a model.
- High 10-day ATR has about 2.48x Top-5 tail lift but materially worse MAE;
  the highest turnover quintile has 9.73% 10-day Top-5 incidence versus 2.15%
  in the lowest quintile, again with worse adversity. Smaller eligible
  capitalization bands show more true tails, but this remains descriptive and
  confounded rather than evidence to loosen or tighten the frozen pool.
- The 152 minute-excluded quality rows have higher true-tail incidence and
  worse MAE, confirming non-random missingness historically; their tiny recent
  count means the 2023-2025 common-support distortion is negligible.

The descriptive audit is retained as machine-readable Parquet and JSON evidence.
Its findings are explained directly in the working conversation; no HTML report
is part of the canonical workflow.

## Quality-liquidity core model (2026-08-04)

`seq100_quality_liquidity_model` completed the 2023-2025 retrospective rolling
study on the 4,476,851-row complete support. The canonical input is one
self-contained row-major `compact_core` cache with 557 fields. It starts from
the 587-field core, removes 14 frozen-support constants and 11 conservative
semantic duplicates, removes five coverage-unstable financial fields, and
replaces three component financial fields with their more complete statement
semantics. The removed coverage-unstable fields are `income_ebitda`,
`cashflow_net_profit`, `income_research_development_expense`,
`income_rd_intensity`, and `income_continuing_net_income`. QDP retains their raw
evidence. The model input includes daily/minute technical,
financial statement, market-state, industry, size/liquidity, announcement,
traditional technical, and traditional moneyflow information. Margin and
research-report fields and availability metadata are not model inputs.

There are exactly 24 formal tasks: nine time-consistent capacity tasks, six
MFE rolling tasks, and nine risk/state rolling tasks. All completed and the
model audit has `status=ok`. The prior 56 variant/gated task directories, old
selection directory, and obsolete extra/availability caches were removed;
6,415,905,694 logical bytes were reclaimed. The canonical output contains no
preserved diagnostic task.

MFE has material but incomplete opportunity learnability:

- D10 Rank IC is `0.1049/0.0979/0.1718` and daily Top-5 true-tail capture is
  `18.63%/14.83%/17.31%` in 2023/2024/2025, or `3.73x/2.97x/3.46x` random.
  Top-1 capture is `10.03%/7.22%/7.23%`.
- D20 Rank IC is `0.1193/0.1353/0.1914`; Top-5 capture is
  `16.12%/13.52%/13.55%`, and Top-1 capture is `6.51%/5.64%/4.67%`.
- Predicted D10 Top-5 names realize mean MFE of `6.52%/7.93%/8.71%` versus
  the full evaluation population's `3.39%/5.33%/4.94%`. D20 is
  `9.80%/13.50%/13.44%` versus `5.52%/9.15%/8.56%`.

MFE is maximum future upside excursion. These results establish useful upside
opportunity selection, not sustained trend, legal exit timing, executable
return, or economic monetization. Predicted Top-5 D10 paths still have median
pre-peak adversity around `2.45%/2.97%/2.65%`; risk and state must remain
separate coordinates.

Risk D10 Rank IC is `0.2087/0.1928/0.2422`; D20 is
`0.2089/0.1889/0.2409`. Deep-adverse PR-AUC is `0.328-0.388` against a 20%
daily event prevalence. State ordinal IC is only `0.0648/0.0648/0.0414`, while
selecting the highest predicted state probabilities raises high-state incidence
by `10.61/7.10/11.99` percentage points. State is useful as an auxiliary
coordinate but remains much weaker than risk or MFE.

The matched 562-to-557 retraining is not a lossless-equivalence result. Removing
the five fields improved MFE Rank IC in 2023-2024, D10 risk ranking and
deep-adverse PR-AUC in all three years, and both state ranking and high-state
Top-5 lift in all three years. However, D20 predicted-Top-5 mean MFE fell
slightly in every year, recent D10 tail strength weakened in 2024-2025, and
D20 risk failed its prior safety gate because 2025 MAE rose by 2.07% while
deep-adverse PR-AUC declined slightly in all three years. The 557-field input is
the current owner-requested canonical set, but the experiment does not prove
that every removed field was useless. `income_rd_intensity` is the leading
targeted add-back candidate: it has roughly 88%-89% coverage in 2021-2025 and
distinct recent univariate opportunity information. No add-back has been
selected or trained yet.

Gain attribution is dominated by market state plus daily price/volume:
roughly 86%-96% across the five heads. Financial statements contribute more to
D20 MFE than D10 (about `5.6%` versus `2.3%` mean gain share); same-day 5-minute
features contribute about `1.0%-2.0%`; traditional technical about
`0.6%-1.5%`; traditional moneyflow is below `0.5%`. These are conditional model
gain shares, not causal feature-family ablations, because correlated fields can
substitute for one another.

Input and output hashes pass, prediction candidate IDs match each evaluation
year, horizon purges pass, the corrected listing-age field is used, 2010 is not
formal, and 2026 rows are zero. No economic replay, position policy, exit
policy, ensemble, or deployable strategy was selected in this study.

## Quality-liquidity execution research (2026-08-04)

`seq100_quality_liquidity_execution` completed a 72-task continuous-account
replay of the canonical 557-field model over 2023-2025. All five prediction
heads use the exact same ordered 557-field contract. The six signal families
were tested at 6/12/24 slots, two replacement buffers, and base/stress costs.
Signals were formed at the close and executed at the next open with T+1,
100-share lots, sell-before-buy, failed-order handling, no leverage, and the
existing pack cost schedule. New trading stopped after the 2025-12-03 signal;
positions were marked through 2025-12-31 with terminal exit costs accrued.

The audit is `ok`: 72/72 tasks, 727 signal dates, 1,234,550 candidate rows,
cash/position conservation, actual order cutoff, output dates, hashes, no HTML,
and zero 2026 rows all pass. The quality-pool equal-weight benchmark returned
`-4.00%/+5.40%/+26.30%` by year and about `+27.80%` cumulatively.

Formal conclusion:

`compact_model_not_monetized_by_simple_execution_surface`

No cell passed the absolute-return and benchmark-excess requirements under
both base and stress costs, and there was no isolated or adjacent passing
region. The best base-cost cell was dual MFE plus risk veto, K=6, buffer=0.5:
terminal return `-40.99%`, relative excess `-53.83%`, gross same-sequence return
`-3.34%`, turnover `279.4x` starting cash, and maximum drawdown `-73.36%`.
Its annual net returns were `-60.19%/-11.23%/+67.22%`; the 2025 improvement does
not repair the severe 2023 failure. The corresponding best stress result was
`-60.51%`.

Risk prediction is economically useful but not sufficient. Relative to plain
dual-MFE cells, the risk veto improved median base terminal return by about
29.3 percentage points and median gross return by 42.4 points. State-only veto
improved them by about 7.7 and 12.3 points. Adding state to risk was mixed at
K=6 and helpful at K=12/24, matching the model evidence that state is the
weaker auxiliary coordinate. The 0.5 buffer usually reduced turnover and
improved return, especially for the risk-veto family, but remained far too
small to create a profitable region.

Tail-path diagnostics show an objective/execution mismatch rather than absent
opportunity. Overall predicted Top-1% D10/D20 mean MFE is about `9.03%/14.23%`,
roughly double the pool baseline, but its D10/D20 fixed-endpoint mean return is
`-0.66%/-0.82%` and median return is `-3.05%/-4.40%`. Typical predicted-tail
peaks occur around D5 for D10 and D8-D9 for D20. The account's best cells hold
positions only about 2-3 days because daily reranking is unstable; holding all
the way to the target endpoint is also too late after the opportunity fades.
Base costs then widen the best gross-to-net gap by roughly 37.6 percentage
points. Fill and capacity constraints are not the primary cause at CNY 1
million: the best cell's buy/sell failure rates are about `0.74%/1.36%`, and
only about 7.8% of filled orders exceed 0.1% of trailing median amount.

This result does not justify changing the 557-feature input or retraining. The
next bounded research should keep the frozen predictions and test a small,
horizon-aligned retention/exit surface around dual-MFE plus risk veto. That
bounded test is now recorded below. It should not reopen broad feature search,
claim a production policy, or consume 2026.

## Quality-liquidity profit-timeout research (2026-08-04)

`seq100_quality_liquidity_profit_timeout` reused the audited 557-field signal
book without retraining or QDP reads. It ran 32 continuous-account tasks over
2023-2025: dual-MFE plus risk veto, 6/12 slots, D10/D20 vertical exits,
take-profit levels of none/5%/8%/10%, and base/stress costs. Entries use the
next open; a standing take-profit becomes active on D2 for A-share T+1; an
unhit position exits at the precommitted D10/D20 close; blocked exits defer to
the next sellable open. Candidate selection does not inspect next-open
buyability.

Formal research conclusion:

`profit_timeout_has_robust_research_candidate`

The only robust region is 8% take-profit plus D20 timeout at both slot counts:

- K=6 base/stress terminal return is `+70.02%/+54.94%`, with relative excess
  `+33.03%/+21.23%`; base annual return is `+1.26%/+13.19%/+48.33%`.
- K=12 base/stress terminal return is `+81.48%/+65.73%`, with relative excess
  `+42.00%/+29.67%`; base annual return is `-6.64%/+12.24%/+73.20%`.
- The same-timeout D20 baseline returns `+42.96%` at K=6 and `-3.48%` at K=12.
  The 8% barrier improves terminal return by `+27.06/+84.96` percentage points
  under base costs and by `+18.02/+75.19` points under stress. Its paired
  annual return improves in at least two years for both K values and costs.

This is not a production selection. Maximum drawdown remains
`-42.00%/-42.79%` at K=6/12 and the common peak-to-trough period runs from
June 2023 to 2024-02-07, recovering only in October 2024. The barrier succeeds
about 66.5% of completed positions with a median D4 hit, but D20 timeout exits
average about `-13.12%` gross and cluster in weak months. K=6/12 complete
407/821 round trips; mean holding is 9.58/9.45 trading-index days, turnover is
162.6x/155.0x starting cash, and base costs consume CNY 212k/202k. Top-five
single-symbol absolute PnL concentration is only 6.3%/4.1%, so the result is
not carried by a few names.

The audit is `ok`: all 32 tasks, hashes, source 557-field contract, T+1,
timeout boundaries, accounting conservation, output dates, no HTML, absent
fixed stop loss, and zero 2026 rows pass. Fixed 5%-20% stop-loss diagnostics
remain rejected as the primary realization rule because they materially cut
later winners. The next question is whether timeout-loser clustering and the
roughly 43% account drawdown can be reduced without destroying the barrier's
paired advantage; do not retrain the model merely because this execution rule
worked retrospectively.

## Prior QDP data state (superseded by the 2026-08-02 state above)

The 2026-07-31 incremental QDP repair changed only defective metadata/tables or
missing rows. The active catalog now contains 26 domains and passes the full
deep audit with `status=ok` and zero blocking errors. The only finding is the
expected medium-severity warning for explicit historical 5-minute gaps;
missing intraday history is not an eligibility rule.

- `market_intraday_5m`: 466,317,792 rows. The BaoStock historical repair
  accepted 214,694 independently validated stock-days (10,305,312 bars) from
  the 215,028 requested 2020-2025 gaps; 334 remain explicitly missing. The
  residual rejection counts are 134 price mismatches, 184 volume/amount
  mismatches, 7 incomplete days, 2 invalid numeric/OHLC days, and 7 provider
  empty days. Canonical positive-daily coverage is now 95.1288% through
  2026-07-21: 9,714,954 complete stock-days and 497,463 explicit missing
  stock-days. The manifest and full physical audit now use the same counters.
- The three dated PIT index histories (`000016.SH`, `000300.SH`, `000905.SH`)
  were rebuilt from 200 snapshots and now contain 3,159,561 rows, including
  historical members that later delisted or changed ticker. Latest independent
  snapshot Jaccard is 1.0 for all three.
- `industry_concept` retains raw historic labels and adds normalized taxonomy
  fields. It has 84 modern coded industries, 18 section codes; the metadata
  repair resolved 507 dated rows and correctly retains 252 unresolved
  `Unknown/unavailable` rows out of 10,212,710.
- Optional PIT event domains were added: `financial_quarterly` has 175,885
  rows and `performance_forecast` has 77,711 rows. Event time is announcement
  date; feature use must lag publication by one day.
- The canonical announcement domain contains 4,796,992 rows for 2010-2025.
  CNINFO succeeded for every requested symbol; Eastmoney remains a whole-symbol
  fallback and was not used in the active dataset.
- The report domains contain 820,246 report identities and 1,515,126 forecast
  details. Tushare-compatible `report_rc` is the forecast source and Eastmoney
  adds report metadata only. Forecast coverage is complete-span for 2010-2016,
  2022, and 2025; 2017-2020 and 2023-2024 are partial spans, while 2021 is
  explicitly source-unavailable. Missing source coverage must not be treated as
  evidence that no report existed.
- PIT income, balance-sheet, and cash-flow statement domains contain 181,900,
  179,220, and 181,047 rows. They use actual announcement dates and become
  available on the next exchange-open day.
- Existing valuation already contains PE, PB, market value, float market
  value, and turnover. The new financial domain adds ROE, margins, growth,
  EPS, leverage, liquidity, turnover, and operating-cash-flow-per-share
  fields. Absolute revenue/profit was left null where no safely aligned
  statement source was fetched.
- The active Tushare-compatible credential is now intentionally persisted only
  in the Git-ignored QDP private provider profile. The profile also records the
  compatible API URL, gzip transport, SDK override, MCP URL template, plan,
  expiry, and provider rate limit. The active local profile takes precedence
  over legacy environment variables; `QDP_TUSHARE_PREFER_ENV=1` is required
  for an explicit process-local override. Credentials remain prohibited from
  datasets, manifests, runtime state, logs, tracked code, and Brain.

The extended Tushare-compatible backfill is complete for 2010-2025 (2010 is
burn-in only): `stk_factor_pro_raw` has 9,795,055 rows, `margin_market` 8,361,
`margin_detail` 3,926,502, and `moneyflow_raw` 9,790,824. `margin_secs` is
explicitly source-unavailable and remains unknown rather than being filled as
false. Stock-level endpoints use date batching; legacy truncated annual caches
remain evidence only and are excluded from the prepared inventory. The active
domains pass primary-key, PIT-date, nonnegative-field, cross-year schema,
2026-zero-read, and credential-isolation audits. All 261 technical fields
are preserved; sampled HFQ consistency failed and HFQ/QFQ-dependent fields are
not formal candidates. Moneyflow `net_mf_vol` and `net_mf_amount` retain their
provider aggregate values as diagnostic-only because they are not stably
reconstructible from the buy/sell buckets.

The 2026-08-01 storage cleanup retained the active catalog, raw provider
evidence, and frozen research inputs while removing only verified redundant
copies. QDP GC now treats dataset IDs found in durable study specifications,
research records, and frozen manifests as live roots before following manifest
shard dependencies. It protected 32 research-pinned versions and removed nine
zero-reference dataset versions (9.8334 GiB). The extended-backfill prepared
cache was removed only after all 64 annual Parquet shards matched the four
active installed datasets by dataset ID, year, recorded SHA-256, and size;
this recovered another 10,433,623,907 bytes while preserving the per-request
raw cache. Its receipt is
`quant_data_platform/data/qdp_v2/audits/tushare_extended_backfill_v1_prepared_cleanup_2026-08-01T105744+0000.json`.
Generic runtime deletion now requires a separate explicit purge flag; the
workflow-specific verified cleanup is the default path.

Git cruft contained 1,084 unreachable blobs (22.5866 GiB inflated, 8.8088 GiB
packed) plus obsolete WIP/temp snapshots, including a tracked DuckDB temporary
file. `git gc --prune=now` reduced the object store to one 229.07 MiB pack with
zero garbage. Across data and Git cleanup, verified logical removal was about
28.3 GiB; H-drive free space increased from 566,759,522,304 to
601,169,592,320 bytes during the cleanup window. The remaining raw request
cache and stable artifacts under `tmp` were deliberately not deleted; future
small-file compaction must preserve exact raw evidence and first pin every
durable `tmp` dependency.

The pool coverage audit through 2025 reports complete 48-bar 5-minute coverage
of 92.79% for all positive daily rows, 93.77% for the same-day
non-ST/non-suspended/non-delisted pool, and 94.66% for the dated union of
SSE50/CSI300/CSI500. In 2023/2024/2025 the three-index union reaches
98.98%/99.48%/99.90%. This supports a PIT index-pool research challenger, but
minute availability itself must never filter the universe.

Retained audit artifacts:

- `quant_data_platform/data/qdp_v2/audits/database_audit_20260730T194231+0000.json`
- `quant_data_platform/data/qdp_v2/audits/pool_coverage_audit_v1.json`

The local PIT metadata repair filled 8,602,631 empty `universe_snapshot.list_date`
values from the canonical identity mapping without changing non-empty values,
rows, keys, or schema. Future listing dates remain zero.

## Active entry contract

`seq100_entry_contract_oos_v4` is the active candidate-aligned 2020-2025
contract. It contains 4,337,640 rows and five logical output blocks:
`mfe_10`, `mfe_20`, three-class `state_10`, `pre_peak_mae_10`, and
`pre_peak_mae_20`.

- `mfe_10`: base plus `turnover_cost_proxy`, fixed 512 boosting rounds.
- `mfe_20`: base plus `breakout_retest_levels`, fixed 512 boosting rounds.
- MFE is rank-first. Raw predictions are retained but are not literal expected
  returns suitable for direct cost subtraction.
- Raw state outputs remain relative state scores, not stable literal
  probabilities.
- Candidate rows are exactly equal to v3. Only the two MFE columns changed;
  state/risk raw values and date ranks are byte-for-byte equal to v3.
- v2 and v3 remain preserved historical contracts.

The fixed-capacity entry audit trained 24 boosters. All 18 newly trained
adaptive prefixes exactly reproduced v3: maximum absolute error `0` and minimum
daily Spearman `1.0`. Fixed 256 failed the safety gate for all three auxiliary
heads:

- `state_10`: worst 2023-2025 ordinal-IC delta `-0.00811`; median Brier harm
  `0.303%`.
- `risk_10`: worst Rank-IC delta `-0.00433`; median MAE harm `1.150%`.
- `risk_20`: Rank IC improved slightly, but median MAE harm was `0.954%` and
  absolute bias worsened in all three years.

Therefore v4 uses fixed 512 only for MFE; state/risk retain their strict,
time-consistent adaptive capacities.

Retained results:

- `daily_research/research_records/seq100/seq100_entry_fixed_capacity_audit_v1/result.json`
- `daily_research/research_records/seq100/seq100_entry_contract_oos_v4/result.json`

## Post-entry target capacities

`seq100_post_entry_capacity_audit_v1` trained 30 long A-only boosters. Capacity
selection used only:

- 2021: train 2020 and evaluate 2021.
- 2022: train 2020-2021 and evaluate 2022.

The maximum selection outcome date was 2022-12-30; no 2023-2025 evidence
selected capacity. One capacity is shared across D1/D3/D5 for each target:

- `remaining_mfe_10 = 128` rounds.
- `remaining_mfe_20 = 128` rounds.
- `remaining_pre_peak_mae_10 = 32` rounds.
- `remaining_pre_peak_mae_20 = 32` rounds.
- `remaining_state_10 = 64` rounds.

The 32-round state candidate was rejected after category collapse in three
selection units. The target-specific selection result is retained at:

- `daily_research/research_records/seq100/seq100_post_entry_capacity_audit_v1/result.json`

## Complete v4 post-entry A/B result

`seq100_post_entry_ab_v2` rebuilt v4-aligned D1/D3/D5 landmarks while reusing
only v1 stable row keys and realized price/turnover paths. It recalculated all
v4 entry/current ranks, six rank changes, entry-top-5% flags, and five targets.

- D1/D3/D5 cohorts: 4,320,911 / 4,314,805 / 4,308,702 filled entries.
- No survivor filter.
- Each target keeps its own common support; D20 completeness never removes a
  legal D10/state row.
- D1 uses only the four nonduplicated realized path fields.
- All 45 target-age-year A/B splits have identical training and evaluation row
  hashes, parameters, capacity, and date weights.
- 90/90 formal boosters completed; no 2026 row or label was read.

No target passed at any age:

- `remaining_mfe_10`: B-A Rank-IC was negative in all nine age-year cells.
  D1 deltas were `-0.00254/-0.00639/-0.00818`; D3
  `-0.00281/-0.01008/-0.01177`; D5
  `-0.00294/-0.01136/-0.01415`.
- `remaining_mfe_20`: B-A Rank-IC was also negative in all nine cells.
  D1 deltas were `-0.00209/-0.00607/-0.00690`; D3
  `-0.00174/-0.01003/-0.00901`; D5
  `-0.00137/-0.01055/-0.01086`.
- MFE B generally improved MAE but worsened ranking, Top-5 remaining MFE, and
  tail hit. This means realized-path inputs regularized predictions toward
  average magnitude while damaging the strong-opportunity role.
- `risk_10/risk_20`: localized 2024 improvements did not persist into 2025;
  no age had three positive Rank-IC years or FDR-supported stability.
- `state_10`: Brier and log loss improved in every age-year cell, but ordinal
  IC declined in every cell. D1 worst delta was `-0.01370`, D3 `-0.02375`, and
  D5 `-0.02057`. Better class-frequency fit did not improve state ordering.

The formal conclusion is:

`daily_recomputation_v4_sufficient_within_tested_supervised_LightGBM_scope`

This is a scoped rejection of the tested independent post-entry LightGBM
updates, not proof that realized paths can never help a different model class.
There is no opportunity, risk, state, or age-specific update head to carry into
a hold-versus-switch value study.

Retained result:

- `daily_research/research_records/seq100/seq100_post_entry_ab_v2/result.json`

## v4 economic realizability result

`seq100_v4_economic_realizability_v1` ran a continuous account from the first
2023 v4 signal through 2025-12-31. It used only rolling-OOS v4 ranks and PIT
execution data; 2020-2022 outcomes did not select policy parameters and no 2026
row, price, label, or liquidation was read.

- Signal book: 727 dates and 2,239,539 candidate rows.
- Surface: 7 families including deterministic noise, 2 exposure modes, 6 slot
  counts, 4 rank-width buffers, and 2 cost scenarios; 672/672 tasks completed.
- All accounts were continuous across years. Raw-open execution, T+1, board
  lots, minimum commission, the 2023-08-28 stamp-tax change, limit/suspension
  failures, adjusted-ratio total-return marks, and pack terminal recovery were
  enforced.
- Maximum daily cash/position conservation error was `3.05e-08`; the negative
  control did not pass; 2026 reads were zero.

Formal conclusion:

`v4_not_monetized_by_preregistered_policy_surface`

No family/exposure pair formed the required contiguous `3 slots × 2 buffers`
economic rectangle. Median daily excess was negative for all 12 formal
hypotheses; no HAC/BH/bootstrap gate passed. There were three isolated economic
cells, which are evidence of parameter fragility rather than deployable
winners:

- dual MFE + state/risk veto, target full, K=3, buffer=2:
  base/stress terminal return `98.75%/68.31%`, excess `37.06%/16.07%`,
  maximum drawdown `-33.87%`.
- the same family at K=6, buffer=2:
  `88.70%/57.97%`, excess `30.13%/8.94%`, drawdown `-33.97%`.
- dual MFE + state veto, strong-candidate cash, K=12, buffer=1:
  `57.41%/47.40%`, excess `8.55%/1.65%`, drawdown `-36.93%`.

Ungated MFE10/MFE20 and dual-MFE policies did not convert learnable MFE into
stable executable returns. State/risk vetoes materially improved the tested
portfolio paths, but only in isolated neighborhoods. Lifecycle/exit blockage
is economically material: 600 of 672 tasks encountered at least one terminal
recovery, reinforcing that MFE opportunity is not itself realizable profit.

Retained result:

- `daily_research/research_records/seq100/seq100_v4_economic_realizability_v1/result.json`

## True-label economic ceiling

`seq100_true_label_economic_ceiling_v1` deliberately replaced v4 predictions
with realized future labels to answer whether the research targets themselves
contain economically realizable opportunity. This is a hindsight ceiling, not
an OOS strategy result.

- Oracle book: 727 signal/mark dates, 2,239,539 candidates, 2,199,108 complete
  D10 labels and 2,168,585 complete D20 labels.
- Stored MFE labels were independently reconstructed from PIT future paths;
  maximum reconstruction error was `0`.
- It ran 864 finite-capital account tasks: true-label daily reranking, sale at
  the first legal true-peak close, and sale at the first legal open after that
  peak. All modes used the same costs, T+1, board lots, minimum commission,
  trading masks, lifecycle handling, and continuous 2023-2025 account as the
  v4 economic audit.
- All 432 paired-cost configurations passed the pre-registered economic gates.
  Every family/mode/exposure pair formed every possible stable slot or
  slot-buffer region, and all 36 pairs also passed HAC/BH/bootstrap
  confirmation.
- Maximum relative cash/position conservation error was
  `1.477e-15`; 2026 reads were zero.
- Actual-label Top-5% mean MFE was `22.35%/28.21%/29.58%` for D10 and
  `34.39%/44.15%/47.51%` for D20 in 2023/2024/2025. All six groups had a
  positive approximate round-trip return after base costs.
- Even the worst pre-registered configuration compounded positively under
  stress costs. A relaxed exact single-slot interval ceiling produced
  D10/D20 base-cost wealth multipliers around `1.68e31/4.08e30`.
- True pre-peak-risk vetoes improved the economic ceiling in the daily mapping;
  the true K3 state-low veto alone generally reduced it. This is evidence for
  the risk coordinate's economic role, not proof that current risk predictions
  are good enough.

Formal conclusion:

`mfe_direction_has_strong_executable_true_label_ceiling`

The prior v4 conclusion remains unchanged: predicted v4 signals were not
robustly monetized. The gap is therefore prediction/selection/realization, not
an absence of profitable opportunity in the MFE labels. The astronomical
hindsight compounding is not deployable: most oracle orders exceed the
reported participation thresholds, and the frozen slippage model has no
endogenous market impact. It establishes direction and loss budget, not
capacity or live expected return.

Retained result:

- `daily_research/research_records/seq100/seq100_true_label_economic_ceiling_v1/result.json`

## Candidate-aligned prediction-to-oracle gap

`seq100_prediction_oracle_gap_audit_v1` aligned every predicted and true
coordinate on the same label-complete candidate support, disabled profit
reinvestment, and capped each position at the initial CNY 1 million divided by
the slot count. It ran 1,200 continuous 2023-2025 accounts: 816 daily-rerank,
192 true-peak-close, and 192 first-legal-open-after-peak tasks.

The predicted Top-5% captured only a minority of the oracle opportunity:

- D10 Top-5% overlap was `18.30%/14.65%/15.32%` and mean true-MFE capture was
  `28.83%/26.27%/27.51%` in 2023/2024/2025.
- D20 overlap was `15.65%/14.41%/12.06%` and capture was
  `30.08%/31.08%/27.88%`.
- Broad MFE Rank IC improved through time while extreme overlap did not. The
  binding prediction problem is strong-tail identification, not merely broad
  cross-sectional ordering.

Matched base-cost substitutions measured terminal-return change relative to
the initial CNY 1 million; a delta of `1.0` is CNY 1 million or 100 percentage
points:

- perfect D10 MFE selection: median `+39.62`;
- perfect D20 MFE selection: `+22.67`;
- perfect dual-MFE selection: `+35.59`;
- true post-peak-next-open exit with predicted dual entry: `+7.77`;
- true risk conditional on true MFE: `+7.64`;
- true state conditional on true MFE: `+1.24`;
- true peak close versus first legal next open for true dual MFE: `+2.62`.

The label-space economic direction is therefore strong even without
reinvestment. Median base-cost dual-true-MFE daily reranking returned
`+3,074%` with `231.5%` CAGR; adding true risk returned `+3,703%` with
`252.9%` CAGR. Both had 36/36 positive cross-configuration-median months.
Predicted dual entry combined with hindsight post-peak-next-open exit returned
a median `+695%`; true dual entry with the same exit returned `+4,770%`.
Stress costs did not change the oracle conclusion.

This audit does not supersede the formal v4 policy-surface rejection. On the
matched fixed-notional support, predicted state/risk veto variants often had
positive absolute terminal returns, but this audit did not apply the prior
benchmark-excess neighborhood gates and did not select a policy. Its purpose
is loss attribution.

Capacity remains material. The true strongest opportunities are less liquid
than predicted top names. For daily true-dual-MFE accounts, the median share of
filled orders above 0.1% of trailing median turnover fell from about `95.8%`
at K=1 to `25.3%` at K=24 and `7.3%` at K=48. Oracle results are ceilings, not
scalable live return estimates.

All 1,200 accounts reconciled annual and monthly CNY P&L to terminal P&L.
Maximum relative conservation error was `1.21e-14`; 2026 reads were zero.

Retained result:

- `daily_research/research_records/seq100/seq100_prediction_oracle_gap_audit_v1/result.json`

## Prior evidence retained

- The bounded feature-union audit found no eligible union. Keep D10 turnover
  and D20 breakout heads only.
- Original Huber remains the MFE objective. LambdaRank improves broad Rank IC
  but loses high-MFE Top 5%; tail-weighted Huber is biased and path-riskier.
- Rolling state atlases passed stability gates, but K3 remains a stable
  representation rather than proof of exactly three natural market states.
- `seq100_post_entry_ab_v1` is retained as v2-only historical evidence. Its
  risk/state rejection is now superseded by the broader v4 five-target result.

## Prior v4 decision (retained)

Use v4 as a frozen research signal contract and recompute its five coordinates
daily. Do not create the rejected independent supervised LightGBM holding
update heads. Do not select an oracle account as a policy, reinterpret its
hindsight return as deployable performance, select any of the three isolated
v4 profitable cells, or consume 2026.

The largest measured loss is strong-tail MFE entry selection. Peak/exit
realization and risk prediction are meaningful secondary losses; state
prediction is a much smaller conditional loss. The next model research should
therefore target candidate-aligned Top-1%/Top-5% opportunity capture rather
than another broad-IC or calibration exercise, and keep a separate
peak-realization challenge. A new model class is justified only if it is
tested against frozen v4 on these matched tail and economic diagnostics. Do
not reopen completed LightGBM capacity, feature-union, or five-target A/B
searches without new evidence.

There is no active training process.

## PIT stock-pool audit

`seq100_v4_pit_stock_pool_audit_v1` was prepared and evaluated without model
training or 2026 reads. It contains the all-market baseline, dated CSI300 union
CSI500, and a PIT quality/liquidity pool. Both global-rank-then-filter and
filter-then-pool-rerank books were built; all-pit books are numerically
equivalent to the frozen economic signal book. The quality pool uses only
signal-date status, listing age, trailing liquidity, float market value, and
announcement-lagged financial quality. Minute availability is not a membership
condition.

- Candidate rows: 2,239,539 over 727 signal dates.
- CSI800: about 628-662 candidates/day, 20.5-21.6% retention; full-market
  true-Top-5 retention is about 11-13%, so it fails the 50% opportunity gate.
- Quality/liquidity: about 1,695-1,697 candidates/day, 55.1-55.4% retention;
  full-market true-Top-5 retention is about 48-55% and falls below the gate in
  2024.
- All four new books completed 672/672 account tasks (2,688 total). Each was
  evaluated with the existing 672-task policy surface, annual/monthly ledgers,
  HAC, contiguous-rectangle, BH, and moving-block bootstrap diagnostics.
- CSI800 produced no robust net rectangle. The quality pool has isolated/gross
  economic cells but no robust net rectangle. No pool is adopted; `all_pit`
  remains the default baseline. Negative controls and accounting checks pass.

Retained result:

- `daily_research/research_records/seq100/seq100_v4_pit_stock_pool_audit_v1/result.json`

## Original D1 T+1 execution audit

The original `g_1` return model was replayed without retraining under a
continuous CNY 1 million account. The signal is ranked at the signal-date
close; buys occur at the next open (or the validated first 5-minute VWAP), and
sales begin at the following open so every filled sale satisfies A-share T+1.
The account keeps a selected symbol when it remains in the target Top-K and
retries blocked sales without filling a replacement. The final five trading
days of 2025 are liquidation-only, so no 2026 data is read.

The 140-cell surface covered K=`1/2/3/5/10`, replacement buffers equivalent to
Top `0/1/2/5/10/20/50 K`, open/VWAP entry, and base/stress costs. All tasks
passed the execution audit (T+1, position limits, cash conservation, terminal
flatness, and zero 2026 reads), but no cell passed the preregistered stability
gate. The best stress terminal return was approximately `-56.5%` (open, K=10,
buffer=20 K); the best low-position alternatives were materially worse.

The loss decomposition is structural: predicted Top-K D1 open-to-close returns
are positive on average, while the same names lose roughly `1.0%` to `1.5%`
from that close to the next open. T+1 prevents realizing the label's close
exit, and daily turnover adds about `13.3 bps` of realized base cost per traded
notional. First-5-minute VWAP entry and wider rank buffers reduce turnover but
do not produce a stable positive net account. Treat `g_1` as a predictive
diagnostic, not a production execution signal; a future executable model must
train an open-to-open or otherwise T+1-compatible target.

Retained result:

- `daily_research/output/path_policy/studies/seq100_quality_liquidity_model/direct_returns/d1_execution/manifest.json`
- `daily_research/output/path_policy/studies/seq100_quality_liquidity_model/direct_returns/d1_execution/audit.json`

## Strict T+1 open-to-open target contract

`seq100_quality_liquidity_t1_targets` now defines the next bounded model target
without changing or overwriting `g_1`. The signal is still formed at the signal
day close, entry is the next trading-day open, and the target exit is the
following trading-day open. The primary back-adjusted return is accompanied by
separate entry-open-to-close and close-to-exit-open components. Their stored
identity reconciles to within `2.36e-08`.

The contract covers all 4,476,851 formal complete-support rows. It has 4,456,500
valid price labels; the maximum signal date read is 2025-12-29, the maximum
outcome date is 2025-12-31, and 2026 reads are zero. Price observation defines
label validity. Future entry buyability and exit-open sellability are separate
state coordinates: 8,270 valid-label rows are not entry-buyable and 4,844 are
not exit-open-sellable, proving that future fill state is not a supervision or
candidate-universe gate. Missing execution-state evidence remains distinct from
a known blocked state. The full target audit is `ok`.

The frozen original `g_1` predictions were then evaluated against the new target
without retraining. Open-to-open Rank IC is `0.0033/0.0335/0.0306` in
2023/2024/2025. Daily Top-5 excess is slightly negative in 2023 and positive in
2024-2025. This passes the bounded preregistered gate for a direct-target
baseline, but it is not evidence of an executable strategy.

## Strict T+1 direct-return model experiment

`seq100_quality_liquidity_t1_return_models` completed the bounded follow-up.
It trained exactly six retrospective rolling LightGBM heads: the audited
open-to-open target and an open-to-D2-close target for 2023, 2024, and 2025.
Both targets have the same 4,456,500 valid rows and the same normalized support.
The D2-close values were reconstructed exactly from the source panels; the
maximum stored-value error is `0`, the open/open plus exit-day intraday identity
reconciles within `2.83e-08`, and 2026 reads are zero.

All models use the fixed 557-field `compact_core`, L2 regression, 512 rounds,
daily 1%/99% winsorization and z-scoring, date-equal weights, and expanding
history. Both labels depend on two future trading days, so every fold limits
its maximum training signal index to OOS start minus three. The six maximum
training signal dates are 2022-12-28, 2023-12-27, and 2024-12-27 for the two
targets. Candidate alignment, task hashes, fixed rounds, purge, outputs, and
the absence of future-fill label gates all pass the final audit.

On identical target support, direct open-to-open Rank IC versus frozen `g_1`
is `0.0420/0.0742/0.0774` versus `0.0033/0.0335/0.0306` in 2023/2024/2025.
Daily Top-5 excess is `0.0497%/0.3196%/0.2178%` versus
`-0.0014%/0.1643%/0.1041%`. The direct target therefore improves broad rank
and Top-5 excess in all three years.

Direct open-to-D2-close Rank IC is `0.0254/0.0435/0.0563` versus frozen `g_1`
at `0.0007/0.0236/0.0209`. Daily Top-5 excess is
`0.1510%/0.3994%/0.2543%` versus `0.1586%/0.3711%/0.2216%`, improving in two
of three years. All six direct annual heads have positive Rank IC and positive
Top-5 excess.

The improvement is not uniformly a strong-tail improvement. Direct Top-5
capture is only `8.20%/8.84%/7.85%` for open-to-open and
`9.74%/10.02%/8.29%` for open-to-D2-close, below frozen `g_1` in every year.
Top-1 capture also declines in every comparison. The L2 heads produce better
broad ordering and realized selected-basket means while overlapping less with
the ex-post extreme-return set. This is a material tail-selection tradeoff,
not an audit failure.

Formal conclusion:

`direct_t1_targets_improve_rank_and_selected_returns_but_not_tail_overlap`

Open-to-open was the stronger predictive candidate because it improved both
Rank IC and Top-5 excess in all three years. The bounded T+1 account comparison
recorded below has now completed without changing the six trained heads. It
does not turn either predictive result into deployable return evidence.

Retained result:

- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_targets/manifest.json`
- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_targets/audit.json`
- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/manifest.json`
- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/audit.json`

## Strict T+1 direct-return execution research (2026-08-05)

`seq100_quality_liquidity_t1_execution` reused the six frozen direct-return
heads without retraining. It aligned 1,234,550 candidate predictions over
2023-2025 and ran 3,456 continuous-account tasks, or 1,728 base/stress pairs.
The surface covered both direct scores, three rank blends, a rank-agreement
score, open and first-5-minute-VWAP entry, target-aligned and rolling/renewable
exits, causal confidence gates, rank-retention bands, staggered D2 cohorts, and
K=`1/2/3`. Accounts start with CNY 1 million, use 100-share lots, sell before
buying, obey T+1, retry blocked exits, do not use future fill state for
selection, and stop at 2025-12-31 without reading 2026.

Formal conclusion:

`no_stable_profitable_execution_region_found`

No configuration has positive cumulative return under either base or stress
costs, no stress configuration is positive in all three years, and neither the
strict nor relaxed stability gate has a passing cell. Only seven base-cost
same-sequence gross returns are positive; all use the D2 score, and none is
gross-positive in all three years. Fusion and score agreement do not create a
profitable region. No production policy is selected.

The least-loss stress cell uses the D2 score, next-open entry, renewable D2
close exit, K=2 staggered cohorts, the prior-only Q80 confidence gate, and a
10K retention band. Its base/stress terminal returns are `-18.35%/-22.83%`,
stress annual returns are `+32.52%/-20.51%/-26.67%`, maximum drawdown is
`-44.89%`, and average position count is `0.143`. This is mostly a cash policy,
not a profitable signal: average cash is about 92.95%, its same-sequence gross
return is negative, and concentrated active-day orders have material liquidity
participation.

For the unextended target contracts, the best open-to-open cell is open entry,
fixed next-open exit, Q80, and K=2; it loses `-46.97%/-53.56%` under base/stress
costs and is negative in every stress year. The best D2 cell is open entry,
fixed D2-close exit, staggered K=2, and Q50. It earns `+11.27%` gross but loses
`-13.36%/-24.59%` net under base/stress costs; its base annual gross returns are
`+2.69%/+34.70%/-19.55%`, so cost is not the only instability.

Strictly paired diagnostics favor open over first-5-minute VWAP entry in 97.2%
of stress-return pairs. Q50 and Q80 confidence gates reduce turnover and median
loss, with Q80 improving median stress return by about 16.7 percentage points
versus always trading. Staggering D2 cohorts materially improves return and
drawdown versus full-cohort rotation. Renewal and wider retention bands save
some trades but add only small, inconsistent return improvements. K=1 has the
worst concentration outcome; K=2 is the smallest defensible research size,
while K=3 adds turnover without producing a stable region.

The final audit is `ok`: all task and base/stress pairs are complete, T+1 and
position limits hold, cash/position conservation error stays below `6e-9`,
close sellability is independent evidence, and 2026 rows are zero. There is no
active training or execution process. Do not continue tuning this same
2023-2025 surface into a production rule; further work requires a new model or
objective and an independent confirmation design.

Retained result:

- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/execution/manifest.json`
- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/execution/audit.json`
- `daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models/execution/decision.json`
