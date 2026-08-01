# Current state

Updated: 2026-08-01

## Objective

Build a daily A-share main-board selector that keeps upside opportunity, path
quality, and pre-peak adversity separate. Entry output structure is frozen,
the supervised post-entry update challenge is finished, and the first
continuous executable-account monetization audit plus a true-label economic
ceiling and candidate-aligned prediction-to-oracle loss audits are complete.
No score fusion, deployment policy, holding period, stop, deep model, or
reinforcement-learning policy has been selected.

## Current QDP data state

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

## Current decision

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

## Quality-liquidity pre-training dataset

`seq100_quality_liquidity_data_prep_v1` is complete and no model was trained.
It provides the physical 2010-2025 PIT sample for all future daily, minute,
fundamental, announcement, and report feature comparisons:

- `quality_liquidity_complete_pit` has 4,487,912 physical stock-days. Membership uses
  only contemporaneously available status, listing age, liquidity, float market
  value, and announcement-lagged financial quality.
- Every retained stock-day has exactly 48 current-day five-minute bars. Missing
  minute data removes only that stock-day, never the whole security, and there
  is no daily-feature fallback inside this common sample.
- Listing age uses exchange calendar history before 2010, so early-2010 support
  is no longer incorrectly empty.
- The formal 2011-2025 complete common support contains 4,441,395 unique
  stock-days. The existing 518-feature atlas is unchanged. The training-ready
  package preserves 106 new formal technical/margin/moneyflow candidates and
  83 diagnostic-only fields; no feature set or model variant has been selected.
- The atlas is descriptive only. No feature set or model variant has been
  selected. Future rolling OOS model evaluation is limited to 2023-2025 and
  every variant must use the same common-support identity.
- The owner selected this common sample for the next model research. The old
  all-market v4 contract remains the frozen comparison baseline; the earlier
  stock-pool audit's default decision does not redefine the new training pool.

`seq100_quality_liquidity_research_scope_v1` freezes the formal research
boundary without rebuilding those physical partitions. Raw PIT history still
starts in 2010 so causal lagged features may use it, but 2010 is burn-in only:
it is excluded from training rows, evaluation rows, labels, and feature-atlas
statistics. The formal 2011-2025 common support contains 4,441,395 unique
stock-days and 518 features. Rolling OOS evaluation remains 2023-2025; the
pre-purge expanding training counts are 3,206,854, 3,616,558, and 4,028,849.
No model or feature set was selected while freezing this scope.

`seq100_quality_liquidity_training_ready_v1` is completed with status
`ready_with_documented_optional_gaps`. Daily and minute models share the exact
same row spine and missing-minute action is stock-day deletion only. It records
the 2023-2025 OOS boundaries and expanding-window training counts, keeps 2026
at zero rows, and explicitly records `training_performed=false` and
`feature_set_selected=false`. The next research step is descriptive
candidate/target and tail-opportunity analysis before any new model training.

The training-ready boundary was hardened after a takeover audit found that its
source common-support files physically carried five inherited pack metadata
columns: `entry_trade_date`, `entry_filled`, `label_valid`,
`price_label_valid`, and `va_aux_valid`. These columns never defined pool
membership or formal feature eligibility, but 1,696 rows on 2025-12-31 had an
`entry_trade_date` of 2026-01-05, so wildcard projection was unsafe. The
remediated row-spine contract is version 2 and contains exactly
`candidate_id, year, trade_date, date_idx, symbol_idx, symbol, security_id`.
All three new physical feature blocks repeat that explicit identity projection,
feature loading is registry-whitelist only, and label validity must come only
from the cutoff memmaps and label flags. Symbol-history mapping is cut off at
2025-12-31; 16 later history rows are explicitly excluded. The rebuilt package
preserves all 4,441,395 rows and the frozen common-support hash, documents the
1,696 source-only future-date rows, and physically verifies zero consumed 2026
dependencies across every row-spine and feature partition.

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
