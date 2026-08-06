# Seq100 Stock Distribution V1 Research Record

Status: retrospective development research, no model or portfolio selected,
2026-08-06

Audience: maintainers of Daily Research and future stock-model experiments.
This is an internal research guide, not a deployable strategy or a profit
claim.

## Decision Enabled

This record decides what the first corrected-data stock experiment actually
supports and what the next model must learn. It does not decide which stocks to
buy. The current evidence promotes conditional risk scale to a required
baseline, keeps conditional return location unproved, retains same-day
five-minute summaries as a small incremental candidate, and rejects naive
concatenation of all PIT fields under the current linear estimator.

## Evidence Contract

- Signal information is frozen at the signal-date close.
- Hypothetical entry is the next adjusted open.
- Planned exits are the adjusted close at D5, D10, or D20; D10 and D20 are
  primary horizons and D5 is diagnostic.
- A blocked planned exit is retried for at most 20 trading dates.
- No future entry or exit state is used to select the candidate universe.
- Formal label support is 2012-2025. The maximum source date is 2025-12-31 and
  2026 reads are forbidden.
- Input is the certified 4,191,476-row, 557-field corrected Seq100 panel with
  fingerprint
  `a2bea9ed174b0a04ea2e97af475e9b4f1da79d739288a0cd92114998e31784ae`.
- Label fingerprint is
  `c93cf15be7f58642b34e086c0c25e00b56a4f46de514d01a35b3132f2e0bc65d`.
- Development evaluation is 1,459 trading dates from 2017 through 2022.
  It contains 1,946,848 candidate rows, 1,940,085 valid D10 executable returns,
  and 1,938,254 valid D20 executable returns.
- Expanding folds purge `H + 20` trading dates before each evaluation year.
- Each training date has total weight one and contributes at most 64
  deterministically sampled stocks to the bounded first baseline.
- The three frozen feature blocks contain 289 `core`, 314 `core_minute`, and
  465 `core_minute_pit` fields.
- The 2022 preflight and the full 2017-2022 development matrix are development
  evidence. The previously inspected 2023-2025 period is not an untouched
  confirmation sample and was not used in this run.

## Mathematical Forecast Object

Let `F_t` be the point-in-time information available at the signal close. For
stock `i` and horizon `H`, define

`A_buy(i,t) = 1`

when the next-open entry state is known and legally buyable. Define
`A_sell(i,t,H)` as one when a legal sellable close occurs from `H` through
`H+20`, and let `tau_sell(i,t,H)` be the first such delay. A value of 21 is the
right-censoring code when the bounded retry window is known but never sellable.

With adjusted entry open `O(i,t+1)` and adjusted close `C(i,t+H)`, the shadow
return is

`X_shadow(i,t,H) = log(C(i,t+H) / O(i,t+1))`.

If both execution states are filled, the executable return is

`X_exe(i,t,H) = log(C(i,t+H+tau_sell) / O(i,t+1))`.

MFE is the maximum nonnegative log return over legal sellable closes from D2
through DH. MAE is the minimum nonpositive path log return from D1 through DH
under the pack's explicit suspended-price carry semantics. These are separate
path coordinates; neither is being treated as realized account profit.

The stock target in this first experiment removes common future movement only
inside the label. For each signal date, `M(-i)` is the leave-one-out market mean
of shadow returns and `I(-i)` is the leave-one-out PIT-industry mean minus
`M(-i)`. The modeled residual is

`Y(i,t,H) = X_exe(i,t,H) - M(-i,t,H) - I(-i,t,H)`.

A singleton or unknown industry falls back to the market component. This
leave-one-out construction prevents a stock from mechanically subtracting its
own outcome. The future factors are target transformations, not features in
`F_t`. Consequently, a residual forecast alone is not an absolute-return
forecast; a later complete model must also forecast the common factor law.

The intended full law factors schematically as

`P(A_buy | F_t) * P(A_sell,tau_sell | A_buy,F_t)`

times a conditional law for executable return and path coordinates. V1 has
prepared all labels, but the first baseline fits only the marginal law of `Y`
and Bernoulli heads for `A_buy` and `A_sell`. It does not yet fit a survival or
hazard law for `tau_sell`, a joint D10/D20 path, or MFE/MAE heads.

## Baseline Learning Rules

Four transparent Student-t models form a two-by-two comparison:

| Location | Scale | Model |
|---|---|---|
| zero | constant | `zero_mean_student_t` |
| zero | feature-dependent | `zero_mean_feature_scale_student_t` |
| ridge mean | constant | `ridge_student_t` |
| ridge mean | feature-dependent | `ridge_hetero_student_t` |

The ridge location is `mu(x) = beta_0 + x' beta`. The scale head regresses
`log(residual^2 + epsilon)` on the same standardized features, exponentiates
half of the fitted value, and calibrates a zero-location Student-t law on the
standardized training residual. With sample weights summing to the number of
training dates, `alpha = penalty * effective_dates` keeps the normalized ridge
penalty fixed as the expanding sample grows. This run fixes `penalty=1e-3`; it
does not claim that this value is optimal.

Entry and sellability baselines are a long-run date-equal probability, a
trailing-252-date probability, and ridge logistic regression. The state heads
are rare-event diagnostics, not a complete execution model.

Primary scores are date-level log score and CRPS. RMSE, pinball losses,
coverage, PIT, and Brier score are diagnostics. Comparisons use date-level
Bartlett/Newey-West HAC with lag 20, moving blocks of length 20, and a White
max-statistic Reality Check over the declared challenger family. Millions of
stock rows are not treated as millions of independent observations.

## Verification

- Prepared labels read back as a `4,191,476 x 21` continuous matrix with no
  infinite value. Missing outcomes remain `NaN` by contract.
- State values are exactly `-1/0/1`; delay values are in `-1..21`.
- All 36 development folds completed: 6 years by 2 horizons by 3 feature
  blocks.
- Every output summary carries the current input and label fingerprints.
- The run reports no model selection, portfolio selection, execution search,
  or profitability authorization.
- Focused pure-function tests pass, including leave-one-out factors, bounded
  date sampling, Student-t CRPS, HAC behavior, and the location/scale Shapley
  identity.

## Continuous-Distribution Results

The table pools date-level scores over all 1,459 development dates. Higher log
score and lower CRPS/RMSE are better.

| H | Feature/model | Log score | Gain vs static | CRPS | RMSE | 90% coverage |
|---:|---|---:|---:|---:|---:|---:|
| 10 | static zero-mean Student-t | 1.331030 | 0 | 0.036305 | 0.069748 | 0.8842 |
| 10 | `core_minute`, zero mean + scale | 1.392069 | +0.061039 | 0.035538 | 0.069748 | 0.8869 |
| 10 | `core_minute`, ridge mean + scale | 1.386538 | +0.055508 | 0.035574 | 0.069658 | 0.8851 |
| 20 | static zero-mean Student-t | 0.989664 | 0 | 0.050846 | 0.096797 | 0.8816 |
| 20 | `core_minute`, zero mean + scale | 1.041269 | +0.051604 | 0.049938 | 0.096797 | 0.8840 |
| 20 | `core_minute`, ridge mean + scale | 1.036120 | +0.046456 | 0.049908 | 0.096526 | 0.8799 |

The zero-mean feature-scale gain is positive in every development year for
both horizons. A pooled 5,000-repetition Reality Check across all three feature
blocks and learned model variants selects `core_minute` zero-mean feature
scale at both horizons with `p=0.00020`. CRPS falls about 2.1% at D10 and 1.8%
at D20. Coverage remains imperfect, so this is evidence for conditional scale,
not a final calibrated law.

### Location versus scale

For `core_minute`, adding the ridge location to an already feature-scaled law
changes mean log score by `-0.005531` at D10 and `-0.005149` at D20. The HAC
95% intervals are `[-0.008335,-0.002726]` and
`[-0.009212,-0.001085]`; moving-block intervals have the same sign. Thus the
current linear mean pipeline degrades the conditional distribution once scale
is controlled.

The pooled two-by-two Shapley decomposition is descriptive because the scale
head is refit from each location residual. At D10 it assigns about `-0.00229`
to the location pipeline and `+0.05780` to the scale pipeline; at D20 the
corresponding values are about `-0.00083` and `+0.04729`. This is not a
structural causal decomposition, but its accounting identity is exact for the
four fitted pipelines.

The ridge mean does reduce squared error slightly with `core_minute`: pooled
OOS R-squared versus a zero mean is `0.259%` at D10 and `0.561%` at D20.
However, the two-sided HAC and block intervals for the MSE gain both cross
zero. A one-sided family Reality Check gives `p=0.124` at D10 and a marginal
`p=0.049` at D20. That is weak D20 evidence worth challenging, not a stable
directional-alpha result. The current data therefore support predictable risk
more strongly than predictable residual mean.

## Feature-Block Results

Adding the 25 same-day five-minute summaries to `core` produces small pooled
gains. For the zero-mean scale model, the log-score increment is `+0.000523`
at D10 with HAC interval `[+0.000272,+0.000774]`, and `+0.000474` at D20 with
interval `[+0.000187,+0.000760]`. The ridge mean and ridge heteroskedastic
variants also have positive pooled increments. A family Reality Check over the
three learned variants gives `p=0.00020` at each horizon.

The effect is small and not positive in every year for every model. Its
mathematical interpretation is narrow: same-day five-minute summaries contain
some incremental information about daily residual distribution, especially
scale. It does not support intraday trading, and it does not establish a
standalone five-minute alpha.

Naively adding all 151 PIT financial and announcement fields to
`core_minute` is not supported under the fixed ridge estimator. Even the best
paired candidate, the zero-mean scale model, changes log score by `-0.001436`
at D10 and `-0.000973` at D20, with intervals crossing zero. Ridge location
and heteroskedastic variants deteriorate more strongly. This rejects this
estimator/block combination, not PIT information itself. The likely
alternatives are stronger group shrinkage, sparse selection, or late fusion
evaluated inside nested chronological folds.

## Execution-State Results

The `core_minute` entry logistic head beats the trailing-252-date baseline in
all six years. Its pooled log-score gain is `+0.000921` at D10 and `+0.000910`
at D20, with positive HAC and block lower bounds. This is usable evidence that
next-open buyability is not purely a constant rate, although the blocked event
is rare: only 4,880 of 1,946,848 development candidate rows are not filled.

For bounded sellability, the recent-frequency baseline is better than the
logistic head in pooled D10 results by `0.000418` log-score points. The D20
difference is smaller and its interval crosses zero. Event support changes
abruptly: D10 censored counts are 1,167, 853, 0, 20, 0, and 10 from 2017 to
2022; D20 counts are 2,213, 1,592, 0, 35, 0, and 10. A sellability covariate
model cannot be promoted from this unstable rare-event evidence.

Delay is economically real even though it is uncommon. Across all formal
years there are 38,843 delayed-but-filled D10 exits and 8,620 D10 censored
exits; D20 has 41,163 delayed fills and 15,557 censored exits. In the
development period, delayed fills number 12,116 at D10 and 12,009 at D20.
The current Bernoulli sell head discards this timing information, so a discrete
hazard or survival head remains required before claiming a complete executable
law.

## Interpretation for the Project

The result changes the modeling priority. The first corrected-data stock
baseline finds a reproducible conditional-scale signal and little robust
linear conditional-mean signal. A risk forecast can improve uncertainty,
position sizing, cash decisions, and avoidance of adverse states. It cannot by
itself create positive expected return. A long-only profit claim still needs a
learnable net conditional mean, tail-ranking advantage, or another explicit
source of expected utility after costs.

This also explains part of the historical tree-model frustration. A model can
learn broad path or volatility structure while failing at the exact
next-open-entry, T+1-compatible, cost-bearing decision. The failure was not
evidence that all data are useless; it was evidence that target, filtration,
execution state, selection metric, and portfolio action had been conflated.
The current experiment separates them, but it has not yet closed the alpha
gap.

## Conclusions and Boundaries

1. Promote `zero_mean_feature_scale_student_t` with `core_minute` to the
   mandatory transparent risk baseline for the next development comparison.
   Do not promote it to a trading model.
2. Keep a conditional-mean challenger, but do not claim directional alpha from
   the fixed linear ridge head. D20's small MSE result is a hypothesis for
   nested testing.
3. Retain same-day five-minute summaries as a small distributional increment.
   They are daily signal-date state variables, not a five-minute strategy.
4. Do not concatenate the full PIT block into the next baseline without a
   different regularization or fusion design.
5. Do not use the sell logistic head as a selected model. Preserve recent-rate
   and explicit censoring baselines until a hazard model survives regime
   checks.
6. No direct, joint, or iterated path class has been selected at the stock
   level. The single-step kernel remains a candidate, not a theorem.
7. No transaction-cost account, expected-log-utility optimization, CVaR
   policy, deep model, or production selector has been run under this contract.
   Profitability remains unproved.

## Required Next Experiment

The next frozen comparison should proceed in this order:

1. Complete the probabilistic object with a discrete `tau_sell` hazard and
   joint D10/D20 return/path outputs. Keep common market/industry factors and
   idiosyncratic residuals explicit.
2. Compare the transparent scale baseline with a direct LightGBM quantile law,
   a joint multi-horizon law, an iterated one-step kernel, and a HAR/rough or
   skew/jump scale-tail challenger. A deep architecture enters only if it has
   a defined conditional-law advantage and enough effective dates to justify
   its capacity.
3. Select penalties, feature blocks, and model variants only inside nested
   chronological folds. Treat PIT group shrinkage or late fusion as a separate
   declared family.
4. Evaluate log score, CRPS, tail calibration, D10/D20 dependence, conditional
   mean MSE, and capacity-aligned tail ranking at the date level. Apply HAC,
   blocks, and selection-adjusted family tests.
5. Only if a forecasting gate passes, map the joint law to a cash-permitted
   expected-log-wealth/CVaR decision with state-dependent costs and capacity.
   A positive net-utility lower confidence bound must be established on data
   unused by model and policy selection.

## Source Files

- Contract: `daily_research/studies/seq100_stock_distribution_v1.json`
- Implementation:
  `daily_research/path_policy/seq100_stock_distribution.py`
- Focused tests:
  `daily_research/path_policy/tests/test_seq100_stock_distribution.py`
- Labels and machine-readable experiments:
  `daily_research/output/path_policy/studies/seq100_stock_distribution_v1/`
- Preflight run: `experiments/preflight_2022_h10_core/`
- Full development run:
  `experiments/development_2017_2022_h10_h20_feature_blocks/`
