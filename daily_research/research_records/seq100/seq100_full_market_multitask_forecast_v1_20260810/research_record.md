# Full-market short-path forecasting, ablation, and capacity audit

## Status

This is adaptive retrospective development evidence over 2012-2025. The five
forward validation folds cover 2020-2025 and every prediction uses only earlier
signal dates with a 30-trading-day purge. An additional earlier-only inner
validation window chooses the tree count; the outer fold is never used for
early stopping. No 2026 outcome is read.

The study found repeatable information about next-close direction, reachable
upside, and short-horizon downside risk. It did **not** find a robust profitable
stock-ranking strategy. One strong Top1 account path is rejected as insufficient
evidence because nearby feature, capacity, ensemble, and breadth variants do not
preserve it.

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

## Durable decision and next experiment

Do not promote the current Top1 path, fixed take-profit rule, Top3, Top10, or
Top30 variant to shadow production as a claimed profitable strategy. Do not keep
tuning the same upside/safety score on consumed outcomes.

The next experiment should change the learning problem rather than the threshold:

1. train direct cross-sectional rank and distributional heads for executable
   legal net return at D2, D3, D5, and D10;
2. evaluate monotonic deciles and Top1/Top3/Top10 before any account optimization;
3. model opportunity, downside, and final payoff jointly, but choose stocks by
   expected legal utility rather than by an ad-hoc minimum of two ranks;
4. retain all 557 fields initially, then repeat frozen family ablation only after
   a payoff-aligned head shows monotonic information;
5. require neighboring capacities and simple ensembles to preserve the result.

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

