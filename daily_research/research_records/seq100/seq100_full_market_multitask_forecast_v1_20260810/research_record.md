# Full-market short-path forecasting, ablation, and capacity audit

## Status

This is adaptive retrospective development evidence over 2012-2025. The five
forward validation folds cover 2020-2025 and every prediction uses only earlier
signal dates with a 30-trading-day purge. An additional earlier-only inner
validation window chooses the tree count; the outer fold is never used for
early stopping. No 2026 outcome is read.

The first upside/safety study found repeatable information about next-close
direction, reachable upside, and short-horizon downside risk, but did **not**
find a robust profitable stock-ranking strategy. A subsequent payoff-aligned
extension changed the learning target, added a compact raw-path sequence model,
and found the first executable historical candidate that passed its candidate
gate. It remains adaptive development evidence, not a stable-profit result.

## Data and execution contract

- Full `quality_liquidity_pit` population, with no financing, Chan, or KAMA
  candidate gate.
- 4,191,476 stock-days and 557 causal fields from 2012-2025.
- Five forward development/validation folds over 2020-2025; all earlier history
  is available for each fold's training set.
- Signal at close, next-open entry, T+1, suspensions, price limits, 100-share
  lots, minimum commission, transfer fee, date-dependent stamp tax, and base or
  double slippage.
- D2 planned-close sale request with legal-sale deferral.
- Finite CNY 1 million account, no leverage, at most two overlapping D2 cohorts,
  and 50% of prior equity per cohort.
- A selected but unfilled order remains cash; no future-known rank substitution.
- Resource planner uses 16 CPU threads and reserves exactly 1 GiB for the system.

## What the models can learn

The frozen 557-field, 127-leaf models produced the following five-fold ranges:

| Target | Five-fold result | Interpretation |
|---|---|---|
| Next close up | AUC about 0.536-0.559 | weak repeated market-direction information |
| D2 legal MFE at least 1% | AUC about 0.570-0.599; daily Rank IC about 0.10-0.15 | reachable upside is learnable |
| D2 exposure MAE above -3% | AUC about 0.672-0.748; daily Rank IC about 0.32-0.42 | downside/volatility state is strongly learnable |
| D2 legal final profit above 0.3% | AUC about 0.507-0.546 | final payoff location is much less learnable |

Gain importance is diagnostic rather than causal ablation. Market state dominates
the direction and MFE heads; market state plus daily price/volume dominates the
safety head. Same-day five-minute summaries add smaller information. Final return
ranking relies on a wider mix but remains weak.

Fixed take-profit execution at 1%-4% lost heavily; 5% was approximately flat
before stress and negative under double slippage. Predicting that a stock can
touch a profit is useful state information, but mechanically capping every winner
at that threshold destroys the positive tail.

## First promising but fragile Top1 path

The adaptive score ranks every signal-date stock by the minimum of its within-day
upside-probability rank and safety-probability rank. It trades only when the daily
mean next-close probability exceeds 0.5, buys Top1 at the next open, and requests
the D2 close exit.

For the 557-field, 127-leaf version:

- base costs: +63.67%, 16.21% maximum drawdown, 333 trades, 6/6 positive years;
- double slippage: +29.63%, 19.32% maximum drawdown, 5/6 positive years, with
  2023 approximately -0.03%;
- HAC 95% lower bounds for mean event return cross zero;
- removing the best 1% of dates makes mean return negative;
- capping every positive gross trade at 10% leaves +44.65% base and +14.59%
  stress, but a 5% cap leaves only +3.20% base and -18.17% stress.

This is promising development evidence, not a stable-profit result.

## True feature ablation

All variants were retrained with the same nested causal folds and targets.

| Input | Base Top1 | Stress Top1 | Positive base years | Max drawdown | Decision |
|---|---:|---:|---:|---:|---|
| all 557 fields | +63.67% | +29.63% | 6/6 | 16.21% | strongest path, still fragile |
| 364 price/path/context fields | +74.18% | +38.89% | 4/6 | 22.71% | higher tail return, worse stability |
| 183 market + daily price/volume + 5m fields | -22.13% | -37.84% | 3/6 | 40.46% | rejected |

The 183-field failure shows that broad inputs are not automatically useless.
The 105 same-date cross-sectional coordinates and 76 traditional indicators
materially change stock selection even though many have low individual gain.
The additional 193 size, industry, financial, announcement, and money-flow fields
also improve this historical path's year-to-year stability relative to 364
fields, but that difference is not independent confirmation.

## Capacity and ensemble audit

The 210-leaf, depth-8, minimum-leaf-1000 profile was trained on all 557 fields.
Its target AUC and Rank IC were usually close to the 127-leaf model and sometimes
slightly better, but its account result collapsed:

| Model | Base Top1 | Stress Top1 | Positive base years | Max drawdown |
|---|---:|---:|---:|---:|
| 127 leaves | +63.67% | +29.63% | 6/6 | 16.21% |
| 210 leaves | +3.81% | -17.33% | 4/6 | 32.33% |
| mean probability ensemble | +6.20% | -15.44% | 4/6 | 25.61% |

On common gated dates, the 127- and 210-leaf models selected the same Top1 stock
only about 10% of the time. Their probability scores are similar, but the extreme
rank is unstable. More capacity therefore does not solve the problem.

## Breadth and monotonicity audit

Holding more names did not rescue the signal under stress:

| Version | Top3 base/stress | Top10 base/stress | Top30 base/stress |
|---|---:|---:|---:|
| 557 / 127 leaves | +4.84% / -16.82% | -4.49% / -23.89% | +4.16% / -17.82% |
| 364 / 127 leaves | +27.08% / +1.26% | +23.74% / -1.02% | +14.13% / -9.51% |
| 557 / 210 leaves | -0.16% / -20.40% | +8.81% / -13.25% | +14.76% / -8.59% |
| 557 ensemble | +6.60% / -14.87% | +5.47% / -15.65% | +11.06% / -11.89% |

The combined upside/safety score has only about 0.007-0.010 mean daily Rank IC
with final D2 legal return. On market-gated dates the highest score decile is not
better than the middle deciles. The models genuinely rank path amplitude and
downside state, but that ranking does not transfer monotonically to final legal
profit. The exceptional 557/127 Top1 path is therefore dominated by extreme-rank
and positive-tail luck.

## Payoff-aligned extension

Exact CNY100,000-lot net-return targets were constructed for D2, D3, D5 and
D10 with raw fill prices, total-return economics, minimum commission, taxes,
base/double slippage, price limits, suspensions and legal-sale deferral. The D5
LambdaRank head produced mean daily Rank IC about 0.0546 across the five folds;
all folds were positive and the combined return deciles were monotonic.

A lookback-8 sequence challenger reads all 557 current-day fields plus 32 daily
raw/path coordinates and a 54-field market branch. It uses a GRU path encoder
and joint stock-rank, market-return and market-direction losses. Its fold Rank
IC values were approximately 0.0795, 0.0540, 0.0626, 0.0523 and 0.1124. The
model is small enough for the RTX 2060, while still training on the full sample
and feature surface.

The frozen stock score is the equal average of the within-date percentile ranks
from the D5 tree and sequence heads. Ranking is performed on the complete
signal-day universe before future fill is known. An unfilled or economically
unaffordable requested order stays cash; it cannot be replaced with the next
ranked stock. This future-fill correction was applied before any final account
conclusion.

The first tree-only market gate left a real approximately 20%-21% drawdown in
the January-February 2024 small-cap crash. A separately identified dual gate
requires both the existing Ridge/logistic market consensus and the neural
return/probability consensus. This gate was proposed after seeing that drawdown,
so every result below is explicitly adaptive.

## Frozen D3 Top10 historical candidate

Fixed D2, D3, D5 and D10 legal exits were compared under the unchanged dual
market gate and stock score. D3 Top10 was the strongest breadth/risk compromise.
With CNY1 million, double slippage, no leverage and one third of previous equity
per signal cohort, it produced:

- CNY1.849 million ending equity and about 11.02% annualized net return;
- 15.50% maximum drawdown and daily Sharpe about 1.16;
- six of six positive calendar years;
- a positive daily HAC lower bound;
- 43.14% total return and a positive HAC lower bound when fold 1 was removed;
- 42.36% total return and a slightly positive HAC lower bound when every winner
  above 10% was credited as only 10%.

It failed the small-gain robustness check. Crediting every winner above 5% as
only 5%, while leaving every loss unchanged, produced about -2.05% total return
and a negative HAC lower bound. The policy therefore depends on retaining a
right tail of larger winners; it is not a stream of many smooth small gains.

Two final bounded challengers were rejected. A direct D3 LambdaRank model had
mean daily Rank IC about 0.0392 and double-slippage Top10 mean about -0.0895% per
date, so “D5 selection, D3 realization” remains preferable. A D3 conditional
10% tail model was informative only weakly: vetoing its bottom predicted decile
reduced annualized return from about 11.02% to 10.55% and improved drawdown only
from 15.50% to 14.90%. That marginal change does not justify another live head.

## Rebuilt-source availability stress (2026-08-10)

The earlier source-availability robustness replay had a scheduling defect: its
D2 and D5 variants reused the source panel's D3 payoff and fill-day columns.
That run is superseded. The corrected replay reloads the matching causal legal
exit return and actual legal fill day for each horizon (schema
`seq100_full_market_rebuildable_core_account_robustness/2`).

This is a fixed robustness audit, not a new search. It masks 243 fields whose
current source metadata says they are unavailable, while rebuilding the 105
cross-sectional ranks and the industry/index market fields from active QDP
snapshots. The retained model surface is 314 fields. Its Top10 selections
overlap the original full-field selections by about 45% on the stress-active
dates, so it is a source-outage challenger rather than a pointwise replacement
of the original model.

Under the same next-open, legal-sale and double-slippage account contract:

| Fixed replay | Ending equity | Annualized | Max drawdown | Positive years | HAC lower |
|---|---:|---:|---:|---:|---:|
| D2 / Top10 | CNY1.450m | 6.53% | -19.82% | 4/6 | -0.0072% |
| D3 / Top10 | CNY1.573m | 8.02% | -16.15% | 6/6 | +0.0036% |
| D5 / Top10 | CNY1.756m | 10.05% | -14.26% | 6/6 | +0.0112% |
| D3 / Top3 | CNY1.445m | 6.47% | -15.89% | 5/6 | -0.0047% |
| D3 / Top1 | CNY1.382m | 5.66% | -20.45% | 5/6 | -0.0163% |
| D3 / Top10, 3x slippage | CNY1.486m | 6.97% | -17.56% | 5/6 | +0.0001% |
| D3 / Top10, 10% gross cap | CNY1.254m | 3.92% | -20.12% | 4/6 | -0.0087% |
| D3 / Top10, 5% gross cap | CNY0.878m | -2.18% | -29.77% | 2/6 | -0.0314% |
| D3 / Top10, folds 2-5 only | CNY1.297m | 6.17% | -16.01% | 4/5 | -0.0083% |

The predeclared robustness gate passes because the primary D3 variant is
positive in all six years with a positive HAC lower bound and every required
neighbor has positive total wealth. This is a deliberately weaker condition
than requiring every neighbor's HAC bound and every year to be positive. D2 is
materially weaker, D5 is stronger in this source-stress sample, and the 5% cap
turns negative. The audit supports further historical value in the signal but
does not establish stable profitability or a uniquely correct D3 exit.

Authoritative corrected output:
`daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/rebuildable_core_account_robustness/manifest.json`.

## Durable decision and next boundary

The original upside/safety Top1 rule, fixed take-profits and its breadth variants
remain rejected. The payoff-aligned D3 Top10 dual-gate policy is now frozen as
the first promising short-horizon historical candidate. It is not independently
confirmed: the dual gate and D3 exit were chosen after inspecting 2020-2025,
and all 2012-2025 outcomes are consumed.

Do not continue an unbounded threshold, holding-day, score-weight or risk-veto
grid on the same outcomes. The current user priority is historical strategy
research, so outcome-blind live scoring is deferred. The next historical work
must keep a new, explicit adaptive-development contract: restore temporarily
unavailable Tushare-compatible source families when the provider is available,
rebuild their PIT features without changing the execution contract, and compare
the full-source and 314-field variants without choosing a new winner from a
large result grid. No 2012-2025 result can be relabeled independent confirmation.

## Authoritative artifacts

- Contract: `daily_research/studies/seq100_full_market_multitask_forecast_v1.json`
- Implementation:
  `daily_research/path_policy/seq100_full_market_multitask_forecast.py`
- Tests:
  `daily_research/path_policy/tests/test_seq100_full_market_multitask_forecast.py`
- Recomputable outputs:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/`
- Cross-sectional diagnostic:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/robustness_ablation/cross_sectional_monotonicity.json`
- Frozen candidate contract:
  `daily_research/studies/seq100_full_market_dual_gate_d3_candidate_v1.json`
- Payoff sequence implementation:
  `daily_research/path_policy/seq100_full_market_sequence_challenger.py`
- Candidate and horizon evidence:
  `daily_research/output/path_policy/studies/seq100_full_market_multitask_forecast_v1/dual_market_horizon_challenge/manifest.json`
