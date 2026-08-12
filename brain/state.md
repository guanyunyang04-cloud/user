# Current state

Updated: 2026-08-12

## Active objective

Find a repeatable A-share policy from the 2012-2025 point-in-time data that is
causally observable, legally executable after costs, and historically useful.
The five fixed forward folds may be reused for early stopping, model choice and
historical evaluation under the user's convention. Results are retrospective
evidence, never a guaranteed-profit claim. No real trading is active and no
2026 outcome may be read. Treat the market as a partially observable,
non-stationary probability system: optimize stable conditional rankings,
net-return distributions and action value rather than deterministic direction.

## Data and execution contract

- Signal universe is `quality_liquidity_pit`.
- The certified causal matrix has 4,191,476 stock-days and 557 fields through
  2025-12-31; 2010-2011 is burn-in only.
- Five expanding folds cover 2020-2025 with purge and no future-fold training.
- Decisions use signal-close information, next-open fills, T+1, suspension and
  limit constraints, 100-share lots, fees, stamp tax, transfer fee, double
  slippage, finite cash, overlapping cohorts, delayed legal sales and no rank
  substitution after an unfilled order.
- Formal economics mark adjusted total return but calculate fills and costs at
  raw exchange prices.
- Every formal payoff ranking now orders the complete finite-prediction slate
  before consulting future fill/outcome status. A selected unfilled or
  unaffordable order is cash; it is never replaced by a lower-ranked stock.

## Durable negative conclusions

- The strict causal Chan parser, literal buy points, small structural-rule
  grids, fixed exits, financing-balance rules and KAMA crosses did not produce
  stable legal net value. Do not reopen those small grids.
- A common-high-rank abstention gate was tested diagnostically on the current
  tree/sequence selections; removing low-agreement names reduced Top10 stress
  performance. Model disagreement is useful complementarity, not a reason to
  require unanimous ranks.
- Lookback-32 sequence Fold 1 was trained as a resource pilot. Rank IC was
  0.12585 versus 0.12601 for lookback-16 and Top10 stress mean was about
  1.189% versus about 1.317%; the roughly 17-minute fold showed no clear gain.
  Do not spend four more folds on lookback-32 without a new reason.
- The attachment's literal seven-family 25/20/15/20/10/5/5 main-board
  Quality x Value x Earnings Revision x Confirmation score was frozen and
  audited. It underperforms the existing 30/30/25/10/5 value-growth score on
  D20/D60/D120 cohort economics and on the matched risk-budget account. Do not
  tune its weights or rating thresholds around this consumed history. After a
  positive-trend hard screen, the extra price-confirmation subscore has
  negative D120 Rank IC and should not be rewarded more aggressively without
  a new independently specified test.
- A single frozen Framework 2.0 risk-overlay challenger kept the old policy's
  exact industry-capped Top30 scan universe and reranked it with expanding
  revision-reversal, multiple-compression and early-spike/fade heads. The
  heads genuinely predict their events (full-history OOF AUC 0.613/0.607/0.587),
  but risk-only reranking is rejected: paired D120 return falls 1.42 percentage
  points with a negative HAC upper bound, while the matched account falls from
  CNY2.863m to CNY2.647m. Drawdown improves only about 0.50 points and winner
  concentration worsens. Do not tune risk weights, thresholds, band width or
  a manual core/risk blend on this consumed result. Retain the heads only as
  diagnostics or auxiliary targets for a separately frozen action-value model.

## Existing interpretable candidates

- The exact main-board 30/30/25/10/5 value-growth policy remains a strong
  interpretable historical candidate: about CNY1m to CNY2.863m in 2012-2025,
  8.02% annualized log growth, 29.22% maximum drawdown and 11/14 positive
  years.
- The literal D60 Top3 concentrated version at 50% gross grew about CNY1m to
  CNY3.427m, with about 9.99% compound growth, 22.89% drawdown and 10/14
  positive years. Both are historical candidates, not stable-profit claims.

## Seven-family framework audit

The supplied main-board framework was implemented with PIT proxies for
normalized valuation, actual earnings improvement, forecast revision, company
quality, earnings quality/cash flow, price confirmation and governance risk.
It selects monthly Top10 names with at most two per industry. The score is a
retrospective quantification of the attachment, not a replacement for its
unavailable qualitative analyst judgements.

- The audit covers 168 month ends and 1,676 selections from 33,468 eligible
  observations. Selected medians are PE 16.40x, market cap CNY21.63bn, net
  profit growth 52.40%, revenue growth 26.34%, ROE 9.96%, CFO/net profit 1.17x,
  debt/assets 39.76% and 90-day net-profit forecast revision +4.39%.
- Full-history D20/D60/D120 Rank IC is 0.04728/0.06602/0.07576, with positive
  HAC lower bounds. Top10 D120 net return averages 6.77%, but excess over the
  eligible pool is only 1.88 percentage points and its HAC lower bound is
  slightly negative. The 2020-2022 D120 eligible-pool excess is -2.37%.
- The score mainly avoids the weakest tail: D120 quintile means rise from
  2.75% in Q1 to 5.80% in Q4, then plateau at 5.71% in Q5. The rating buckets
  are not calibrated and are not monotonic in return.
- Next-fiscal-year selected net-profit forecasts have median error -12.44%,
  median absolute error 26.64%, only 41.23% within 20%, 64.69% overforecast
  and 58.75% growth-direction accuracy. EPS realization is weaker.
- The most common D120 fundamental outcome is earnings delivery with multiple
  compression (37.71%, mean -2.77%). Revision reversal occurs in 39.32% of
  selections; delayed D60-loss/D120-gain realization occurs in 12.65%.
- New and old Top10 selections overlap 67.60%. Framework-only selections favor
  current growth and quality but have weaker valuation/cash quality and D120
  mean 5.13%; old-only selections have D120 mean 8.59%.
- The matched frozen risk-budget account grows CNY1m to CNY2.717m, with 7.62%
  annualized log growth, -29.15% drawdown and 11/14 positive years. The old
  score reaches CNY2.863m, 8.02% and -29.22% under the same execution contract.
  These are consumed historical results and do not permit a stable-profit
  claim.

## Ranking and execution correctness audit

- A real look-ahead selection bug was found and fixed in tree, sequence,
  downside-fusion, tree/sequence-ensemble and legacy D5 market-regime payoff
  paths. They previously removed rows using future target validity before
  ranking, allowing a lower-scored stock to replace a selected unfilled stock.
- All corrected paths rank finite predictions first, book known unfilled or
  unaffordable selections as zero-return cash, exclude right-censored selected
  dates, and never substitute a lower rank. Correcting the established 557-
  field three-model account reduced Top10 stress from +151.65% to +148.70%; the
  bug was genuine but not the source of the strategy's headline result.
- The corrected legacy D5 market gate moved consensus Top10 stress mean only
  from about 0.1090% to 0.1078%; its HAC lower bound remains negative and it is
  still rejected.
- Same-symbol cohort pyramiding is now explicit and stress-tested. The corrected
  557-field three-model account has +148.70%, -28.08% drawdown and 6/6 positive
  years when repeated signals may stack, versus +55.77%, -9.02% drawdown and
  6/6 positive years when an already-held symbol cannot be bought again. The
  signal is not solely pyramiding, but stacking materially amplifies both gain
  and risk.

## Fixed feature-view competition

Three feature views were frozen before comparison with the same
`strong_127`, exact D10 target, five folds, early stopping and account contract.

- Full `all_causal_557`: best iterations 1/3/12/4/1, OOF Rank IC 0.07739,
  Top10 stress +100.02%, -24.33% drawdown and 5/6 positive years. No-overlap
  return is +41.88%; 10% winner cap is -13.70%.
- `price_path_context_364` (cross-sectional technical, price/volume,
  traditional technical, market state and same-day 5m): best iterations
  96/75/84/14/2, OOF Rank IC 0.08531, Top10 stress +92.79% and no-overlap
  +35.43%. The 10% winner cap is -22.57%.
- `price_path_core_183` (daily price/volume, market state and same-day 5m):
  best iterations 18/4/48/21/32, OOF Rank IC 0.08687, Top10 stress +137.17%,
  -28.74% drawdown and 5/6 positive years. No-overlap is +49.47%, -8.37%
  drawdown and 6/6 positive years; 10% winner cap is only +3.21%.
- The reduced view both trains deeper and improves ranking/economics, so the
  full-field tree's early stopping at a few rounds is not a hard-coded model
  simplification. Extra fields currently add noise, missingness and temporal
  drift under this flat LightGBM representation. This does not prove that
  fundamentals, revisions or announcements are useless; they should return
  through staleness-aware normalization or a separately validated branch, not
  be mixed indiscriminately into the flat core.
- A controlled Fold-1 early-stop audit found that monitoring NDCG@100 alone
  changes the full-field best iteration from 1 to 2 and improves exact Rank IC
  from 0.06115 to 0.06534, but worsens Top10 stress mean from +0.041% to
  -0.100%. Metric choice matters, but no hidden large gain was exposed.
- Known cash outcomes excluded from exact-net training weights are only about
  0.19%-0.43% of each validation fold. A cash-aware action-value label remains
  a clean future challenger, but this mismatch cannot explain the main result.

## Rejected D10 target variants

Three pre-specified challengers were completed against the same exact D10
executable payoff and finite-account contract. None improves on the current
three-model benchmark, so do not tune weights or thresholds around them.

- A `market_excess_endpoint_return_d10_rank` `strong_127` tree contains useful
  ordering information (OOF Rank IC 0.07876), but Top10 stress has a negative
  event-level HAC lower bound. Its exact account gains 54.01% with -23.09%
  drawdown and 5/6 positive years; the 10% winner cap loses 25.71% and the 5%
  cap loses 64.40%. Reject it as a standalone or fusion component.
- The `exact_net_return_d10_q10` lower-quantile head reaches OOF Rank IC
  0.08846, but Top10 stress gains only 5.29%, draws down 37.45% and has 3/6
  positive years. Its best uncapped breadth is Top1 at +19.01%, also with only
  3/6 positive years. The downside head is not independently tradeable.
- The frozen 50/50 average of within-date return-rank and q10-rank reaches OOF
  Rank IC 0.09133, but its Top10 stress account gains only 26.01%, draws down
  26.40% and has 3/6 positive years; 2020-2022 all lose. The 10% cap is -0.44%
  and the 5% cap is -32.07%. It neither improves economics nor removes right-
  tail dependence.

## Current best historical challengers

The primary short-horizon benchmark is now the equal within-date rank average
of `sequence lookback-16 + strong_127 price_path_core_183`, always on and held
to the legal D10 exit.

- Five OOF Rank IC folds are 0.11362/0.07928/0.11524/0.10021/0.12697;
  combined Rank IC is 0.10693.
- Exact Top10 stress grows CNY1m to CNY2.897m (+189.65%), with -30.15%
  drawdown, 6/6 positive years and daily HAC lower +0.01760 percentage points.
  All five account fold compound returns are positive.
- Forbidding repeated open cohorts of the same symbol leaves +64.90%, -11.70%
  drawdown, 6/6 positive years and a positive HAC lower bound.
- The 10% winner cap leaves only +9.42%, 4/6 positive years and a negative HAC
  lower bound; the 5% cap is -59.64%. Right-tail dependence remains the main
  unresolved weakness and no stable-profit claim is allowed.

A three-component robustness alternative adds `qlib_capacity_210` trained on
the same 183 fields. Its two tree ranks correlate about 0.789, while their
correlations with the sequence rank are only about 0.370-0.389.

- Combined Rank IC is 0.10508; Top10 stress is +188.11%, -29.77% drawdown,
  6/6 positive years and HAC lower +0.01586 percentage points.
- No-overlap return is slightly better than the two-component version at
  +66.15%, with -11.51% drawdown; no-overlap 10% cap is +11.96%.
- `qlib_capacity_210` therefore diversifies some constrained-account outcomes
  but slightly dilutes raw IC and uncapped economics. Keep it as a robustness
  alternative, not an automatic third component.

Both component choices were made after reviewing reusable OOF evidence. They
are adaptive retrospective challengers, not independent confirmation.

## Authoritative artifacts

- Current two-component core OOF:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_ensemble_evaluation/horizon_10/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__tree_features_price_path_core_183/manifest.json`
- Current two-component core account replay:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_ensemble_account_replay/horizon_10/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__tree_features_price_path_core_183/manifest.json`
- Three-component core robustness OOF/account use the same paths with
  `__tree_profiles_strong_127_qlib_capacity_210__tree_features_price_path_core_183`.
- Corrected legacy 557-field three-model mean OOF:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_ensemble_evaluation/horizon_10/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__tree_profiles_strong_127_qlib_capacity_210/manifest.json`
- Corrected legacy 557-field three-model mean account replay:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_ensemble_account_replay/horizon_10/lookback_16/sequence_outer_early_stop__tree_outer_early_stop__tree_profiles_strong_127_qlib_capacity_210/manifest.json`
- Three-model minimum control uses the same paths with
  `__fusion_minimum__tree_profiles_strong_127_qlib_capacity_210`.
- Single-tree and sequence manifests are under the corresponding
  `payoff_evaluation`, `payoff_account_replay` and
  `sequence_payoff_evaluation` directories in the same study root.
- Lookback-32 pilot:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/sequence_challenger/horizon_10/outer_early_stop/lookback_32/fold_1/manifest.json`
- Rejected market-relative tree evaluation/account:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_evaluation/market_excess_endpoint_return_d10_rank__strong_127/manifest.json`
  and the matching path under `payoff_account_replay`.
- Rejected q10 tree evaluation/account:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_evaluation/exact_net_return_d10_q10__strong_127/manifest.json`
  and the matching path under `payoff_account_replay`.
- Rejected frozen return/q10 fusion evaluation/account:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/payoff_score_fusion_evaluation/horizon_10/strong_127__outer_early_stop/manifest.json`
  and the matching path under `payoff_score_fusion_account_replay`.
- Seven-family framework frozen study:
  `daily_research/studies/seq100_qver_confirmation_effect_v1.json`.
- Seven-family framework authoritative summary and Parquet evidence:
  `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/summary.json`
  and the sibling files in that study root.
- Seven-family framework durable research record:
  `daily_research/research_records/seq100/seq100_qver_confirmation_effect_v1_20260811/research_record.md`.
- Frozen QVER risk-overlay validation:
  `daily_research/studies/seq100_qver_risk_overlay_validation_v1.json`.
- Risk-overlay authoritative summary and record:
  `daily_research/output/path_policy/studies/seq100_qver_risk_overlay_validation_v1/summary.json`
  and
  `daily_research/research_records/seq100/seq100_qver_risk_overlay_validation_v1_20260811/research_record.md`.

## Code and verification

- `daily_research/path_policy/seq100_full_market_multitask_forecast.py` now
  shares `PAYOFF_TOP_KS = (1, 3, 5, 10)` across payoff OOF and account tasks.
- `daily_research/path_policy/seq100_full_market_sequence_challenger.py` now
  supports horizon-isolated payoff evaluation, multi-tree profile rank fusion,
  explicit tree feature views, independent artifact namespaces and the matching
  account replay. Audited account caches reject legacy manifests without an
  explicit zero 2026-read count.
- `daily_research/path_policy/seq100_qver_confirmation_effect.py` implements
  the frozen attachment audit, post-selection outcome reads, forecast
  realization, scenario decomposition and matched legal account replay.
- `daily_research/path_policy/seq100_qver_risk_overlay_validation.py` implements
  the frozen old-Top30 expanding three-head risk rerank, strict D120 label
  availability, paired cohort inference and exactly matched account replay.
- Focused and full joint tests pass: 42 tests. Ruff, `py_compile` and
  `git diff --check` pass. All new target-variant manifests plus the refreshed
  sequence and three-model mean account manifests explicitly report zero
  forbidden 2026 reads.
- The seven-family audit's five focused tests pass; its summary schema, file
  hashes, score reconstruction, selection uniqueness, industry cap, account
  endpoint and zero forbidden 2026 reads were independently rechecked.
- The risk-overlay module's six tests and the five base QVER tests pass. Its
  4,982 pre-outcome band identities, industry cap, probability bounds, strict
  training-label timing, exact old-account endpoint, artifact hashes and zero
  forbidden 2026 reads were independently rechecked.

## Next step

Do not label the current challenger as stable or deploy it. Preserve the
two-component 183-field mean as the current short-horizon historical benchmark,
the three-component core as its constrained-account robustness alternative, and the old
30/30/25/10/5 score as the stronger interpretable D60 baseline. Use the new
seven-family framework as a candidate-pool and scenario-audit lens, especially
for revision reversal, multiple compression and early-spike fade risk. The
equal-risk Top30 rerank has now been rejected even though its individual heads
work. The next clean competition should use a pre-specified cash-aware
action-value or distributional target on the 183-field path core, then admit
fundamental/revision data only through a staleness-aware separate branch. Use
the same five folds, corrected full-slate ranking, no-overlap account stress and
winner caps. Avoid ad hoc fusion weights, thresholds or small rule grids.
