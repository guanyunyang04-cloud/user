# A-share Quant Research: Mathematical Foundations Audit (2011-2025)

> **Superseded for corrected-data model decisions.** This historical report
> uses the pre-status-repair support and is retained as background evidence
> only. Use `mathematical_market_model_20260806/research_record.md` for the
> corrected-input, candidate-neutral market-model study.

Status: technical research report, generated 2026-08-05. The analysis is
read-only: it does not train a model, mutate QDP, or consume 2026 data.

## Technical summary

The central result is negative but useful: the project has not yet defined a
mathematically privileged trading label. A fixed 8% barrier is a reporting
threshold, not a theorem. Under a first-passage model its probability depends
on the dimensionless barrier `a/(sigma*sqrt(H))` and dimensionless drift
`mu*sqrt(H)/sigma`; in the current data the 8% log barrier has a median
standardized distance of `0.731`, with the 1%-99% range `0.269-2.300`. The
observed row-weighted annual hit rate therefore moves from `32.58%` to
`70.99%`.

The broad D20 price path contains a strongly skewed opportunity distribution,
but not strong evidence of positive growth for a date-level decision. Among
`4,385,062` complete paths over `3,624` signal dates, the arithmetic endpoint
mean is `+0.775%` and the median is `-0.437%`. The row-weighted mean log return
is only `+0.011%`; the date-equal mean log return is `-0.0317%`, with HAC t
statistics `-0.067` and `-0.064` at 20/60 day bandwidths. Moving-block 95%
intervals for the date-equal log mean are `[-0.973%, +0.839%]` and
`[-1.042%, +0.950%]`. This is not evidence that the unconditional pool has a
positive growth edge.

The right model object is a conditional distribution of execution-relevant
outcomes, not a binary 8% label. The first model should forecast multiple
open/open and open/close horizons, path excursion and adversity, first-passage
time, and execution-state censoring with strictly proper scores. A low number
of holdings belongs in a later decision problem (`||w||_0 <= K` with cash
allowed), not in the prediction target. A model is "strong" only if it beats
simple distributional baselines on date-clustered out-of-sample scores,
calibrates its tails, and remains superior after the entire model family has
been corrected for selection.

## The problem is statistical decision-making, not a privileged formula

At the signal close, the observable information is a filtration `F_t`; the
unknown object is a conditional law `Q_t = P(Z_t in dz | F_t)`. A forecast is
an estimate of `Q_t`. A later policy maps that estimate and a stated utility
to an action, and realized net wealth supplies the reward. These are three
different layers: measurement, prediction, and decision. Collapsing them into
one backtested label makes it impossible to identify which layer improved.

Mathematics cannot choose the investor's utility, assert stationarity, or make
the data-generating process identifiable. Those are economic and empirical
assumptions. Its role is to expose them, enforce information timing, define a
loss that is proper for the forecast object, and quantify uncertainty under
the actual dependence structure. There is therefore no universally correct
label independent of horizon, state, utility, and data quality.

## What was actually measured

The formal complete-support input has `4,476,851` rows, `3,644` signal dates,
`3,005` securities in the model row index, and `3,419` historical symbols in
the pinned pack. `2010` is burn-in only; all formal outcomes are bounded by
2025-12-31 and the audit recorded zero 2026 reads. A D20 pure-price path is
defined from the next open through the D20 close; the take-profit excursion
starts on D2 to respect A-share T+1. It does not require a future buy or sell
to be legally executable, so it is a price-path diagnostic rather than an
account result.

The primary arithmetic endpoint is
`R_H = (C_{t+H}/O_{t+1}) - 1`. The corresponding wealth quantity is
`g_H = log(1 + R_H)`. MFE is the maximum high relative to the entry open over
the legal sale window; MAE is the minimum low over the same window. These are
different random variables and must not be collapsed into one "success" bit.

## Annotation 1: why 8% is not a mathematical constant

Let log-price displacement after entry be

`X_s = mu*s + sigma*W_s`, `tau_a = inf{s: X_s >= a}`, `a = log(1.08)`.

For a continuous Brownian approximation, the finite-horizon first-passage
probability is

`P(tau_a <= H) = Phi((mu*H-a)/(sigma*sqrt(H)))
                 + exp(2*mu*a/sigma^2)
                   * Phi((-mu*H-a)/(sigma*sqrt(H)))`.

At zero drift this reduces to

`P(tau_a <= H) = 2*Phi(-a/(sigma*sqrt(H)))`.

Thus the scale-free Brownian inputs are the standardized barrier
`k = a/(sigma*sqrt(H))` and standardized drift
`delta = mu*sqrt(H)/sigma`; at zero drift, `k` alone remains. A fixed 8%
barrier implicitly assigns a different event probability to every volatility
regime even before drift and jumps are considered. With the observed median
20-day volatility (`2.354%` per day), the median standardized distance is
`0.731`, whose zero-drift Brownian probability is about `46.5%`; it is not the
same as the `31.73%` probability for `k=1`.

The data confirm the dimensional argument:

- Fixed 8% hit rate by year ranges from `32.58%` to `70.99%` (standard deviation
  `9.06` percentage points).
- A volatility-scaled `k=1` barrier ranges from `22.25%` to `41.68%` (standard
  deviation `5.27` points), still far from constant.
- Across volatility deciles, fixed-hit probability rises from `26.35%` to
  `62.77%`, while the volatility-scaled hit probability falls from `46.46%` to
  `19.32%`. The highest-volatility decile has arithmetic endpoint mean only
  `+0.31%`, mean log return `-1.36%`, and ES5 `-34.32%`.

The remaining variation is expected: the Brownian formula assumes continuous
paths, a stable diffusion coefficient, and a specified drift. A-share prices
have jumps, daily price limits, suspensions, market-wide factors, changing
cross-sectional composition, and an estimated rather than known volatility.
The scaled experiment is therefore a falsification of the idea that volatility
alone is a sufficient state variable, not a claim that the exact Brownian
model is the correct replacement.

An 8% number may still be used as one business-readable slice of a continuous
forecast. It must not be used as the training oracle, a supposedly natural
class boundary, or evidence that a neighboring 10% level is inferior without a
pre-specified decision utility and an out-of-sample threshold-selection test.

## Annotation 2: what the earlier 8% experiment really established

The earlier K12 D20 account comparison is a valid retrospective diagnostic of
one policy surface, not a proof of an 8% law. Pairwise HAC and block-bootstrap
results show:

- K6: 8% versus 5%, 10%, and no take-profit has intervals crossing zero and
  HAC t below about `0.95` under both cost scenarios.
- K12: 8% is clearly better than 5% and timeout-only on this reused sample,
  but versus 10% the HAC t is only `1.03-1.08` and the interval crosses zero.
- The paired annual log-return difference of the new hit-plus-timeout label
  versus the old MFE-plus-risk label is about `+4.57%/+5.46%` for K6 and
  `-0.61%/+0.28%` for K12 (base/stress); all intervals are wide and cross zero.

The hit/miss decomposition explains why the experiment looked attractive:

| construction | hit rate | hit mean | miss mean | gross mean | loss < -10% | ES5 |
|---|---:|---:|---:|---:|---:|---:|
| old MFE + risk | 66.50% | +8.68% | -13.12% | +1.38% | 19.61% | -29.80% |
| hit + finite timeout | 43.62% | +8.46% | -3.63% | +1.64% | 6.21% | -15.50% |

For a two-component approximation, the break-even hit rate is
`p* = -m/(h-m)`, where `h` and `m` are conditional hit and miss means. This
gives `60.18%` for the old construction and `30.03%` for the new one. The
lower break-even rate is a real improvement in the failure distribution, but
it does not identify 8% as optimal and did not produce a statistically stable
paired account improvement. The likely lesson is to model the left tail and
the time-to-resolution, not to hard-code a particular percentage.

## The unconditional path distribution is not a growth strategy

For all complete D20 paths:

- arithmetic mean `+0.775%`, standard deviation `12.85%`, median `-0.437%`;
- skewness `1.62`, excess kurtosis `12.47`;
- ES5 `-22.76%`, ES1 `-33.23%`.

The positive arithmetic mean is dominated by a right tail. Since repeated
capital compounds through `sum(log(1+R))`, the relevant quantity is the log
mean. The identity `log(1+E[R]) >= E[log(1+R)]` (strict unless returns are
constant) is the volatility-drag reason that a positive average trade is not
automatically positive wealth growth. The discrepancy is especially severe in
the highest-volatility decile.

The binary hit event is also insufficient. Among paths that hit 8%, `19.97%`
finish D20 below zero, `4.68%` finish below `-10%`, and the mean peak-to-end
retrace is `10.35%`. Among misses, the endpoint mean is `-5.74%` and ES5 is
`-25.04%`. A useful predictor must estimate the full conditional distribution
of endpoint, peak, retrace, and left-tail loss.

## Dependence changes the effective sample size

The `4.385M` rows share date-level market shocks and overlapping 20-day paths.
Date fixed effects explain `29.81%` of expanded endpoint variance. The lag-1
autocorrelation of daily cross-sectional endpoint means is `0.951` (lag 5:
`0.748`; lag 20: `-0.101`). Treating rows as independent gives a standard error
of only `0.0061` percentage points; equal-weighting dates gives `0.1274` points
before serial correction and `0.463-0.493` points after HAC. The HAC-equivalent
effective date count is only about `274` or `242`, versus `3,624` observed dates.

The correct inferential object for a cross-sectional forecast is usually a
date-level loss or return difference. Define improvement with an unambiguous
sign as `d_t = mean_i [L_baseline(i,t) - L_model(i,t)]`, followed by HAC or a block
bootstrap over dates. This does not make dates independent; it makes the
sampling unit explicit. For overlapping labels, the purge length must cover
the maximum outcome dependence, and a policy label with an unbounded delayed
exit has no finite purge length until its contract is repaired.

## Data-quality findings that block strong claims

### 1. Policy labels are not finite-horizon labels

The policy target implementation retries a blocked exit at the next sellable
open without a horizon-specific retry cap. D10 has `31,931` delayed opens,
`5,744` delayed by more than 20 trading days, and `2,960` by more than 60; D20
has `29,039`, `9,030`, and `4,352`, respectively. The maximum delay is `1,006`
trading days. `000520.SZ` on 2013-12-25 receives a policy return of `+587.02%`
with a fill roughly two years later.

This contaminates `timeout_return` and all downstream quantities that depend on
policy return. It does not invalidate every direct endpoint or every event
indicator: the direct `endpoint_return` is a separate finite-date quantity,
and `tp_hit` primarily records whether the path reached the event before the
timeout. Each target needs an explicit `complete`, `right_censored`,
`unresolved`, or `invalid` state.

### 2. The ST state has a reproducible splice anomaly

The pinned raw unadjusted OHLC diagnostic finds a market ST rate of `7.2068%`
on 2011-11-21 and `1.7875%` on 2011-11-22, a one-day fall of `5.4193` points.
The underlying status table changes from `145` ST rows (80 SH, 65 SZ) to `36`
(23 SH, 13 SZ) while row coverage remains about 2,012-2,014. The exact date is
also where the historical path tooling's archive scope starts, which is
consistent with a provider/name-history splice rather than a plausible market
transition. The current market ST-rate feature agrees with the `is_st` mask to
within `3.71e-9`; it is therefore internally consistent but may be consistently
wrong.

The raw flat-price test is a diagnostic, not official truth: it finds `6,982`
marked and `4,398` unmarked stock-days at an exact +/-5% tick. Corporate-action
boundaries and rounding can create false positives. The minimum repair is to
rebuild dated exchange-name/ST intervals, compare them with raw `isST` and
price-limit geometry, then run drop-versus-rebuilt feature ablations. Tree gain
is conditional split usage, not causal contribution; the suspect feature has
positive gain in all `63/63` audited models, so the issue cannot be ignored.

### 3. Validity is horizon-specific

For rows with a complete D60 observation window, `57,969` have a false D60
`price_label_valid` flag. The complete D20 pure-price paths have zero detected
continuity breaks. Therefore D60 invalidity cannot be used as proof that D20 is
invalid; every horizon needs its own observation and censoring contract.

## A mathematically coherent model specification

Let `F_t` be the sigma-field generated by information available at the signal
close. For stock `i`, forecast the conditional law

`F_theta(z | X_i,t, M_t) = P(Z_i,t in dz | F_t)`,

where `Z` is a vector containing:

1. entry gap and open-to-open/open-to-close returns at D1, D2, D5, D10, and
   D20;
2. MFE, MAE, peak time, and peak-to-end retrace at the same horizons;
3. first passage time to a grid of standardized barriers `k` rather than one
   fixed percentage;
4. buyability, sellability, suspension, and right-censoring states;
5. a date-level market factor and a stock-relative residual, so common risk is
   not mistaken for idiosyncratic alpha.

The first implementation should be small and auditable:

- an unconditional/date-prior baseline and a ridge/L2 conditional mean;
- separate LightGBM quantile heads for the chosen horizons and quantiles,
  followed by a non-crossing constraint or monotone rearrangement;
- a discrete-time hazard head for first passage with a survival likelihood;
- a joint VaR/ES head only after the marginal tails are calibrated.

For a quantile `q_tau`, use the pinball loss
`rho_tau(y-q) = (tau - 1{y<q})(y-q)`. For a full univariate distribution `F`,
use CRPS
`CRPS(F,y) = integral (F(u)-1{y<=u})^2 du`, equivalently an integral of
pinball losses over all quantiles. For a path vector, add marginal/path
functionals or a dependence-sensitive score; a single energy score can hide
bad dependence even when marginal forecasts look good. For VaR and ES use a
Fissler-Ziegel joint strictly consistent score, not an isolated ES regression.

Right-censored observations contribute the survival term rather than an
invented negative return. For discrete hazard `h_{j,d}(x)` and event type `j`,

`P(T=d,J=j | x) = h_{j,d}(x) * product_{u<d}
 (1 - sum_k h_{k,u}(x))`.

An unresolved blocked sale is not a price event; it is either a competing
execution state or a censoring indicator. This distinction prevents the model
from learning the data-collection retry policy instead of the market process.

## Prediction and decision must be separate layers

Prediction estimates `P(Z|F_t)` using a proper forecast score. A later decision
layer maps scenarios to weights. A robust low-position formulation is

`max_w E[log(1 + w'R_net)] - lambda*CVaR_alpha(-w'R_net)`

subject to `w_i >= 0`, `sum(w_i) <= 1`, `||w||_0 <= K`, turnover and capacity
limits, and a cash variable. Here

`CVaR_alpha(L) = min_eta { eta + E[(L-eta)_+]/(1-alpha) }`.

The cash inequality is essential: if the lower confidence bound of the
conditional utility is not positive after costs, the mathematically correct
action is zero risky weight. Small `K` is a combinatorial selection constraint,
not a reason to distort the label or to assume the top-ranked name is safer.
Top-1 selection amplifies order-statistic error and winner's curse. Fractional
Kelly or a distributionally robust uncertainty set is a later risk control;
full Kelly on estimated tails is not justified by a good point forecast.

Decision-focused "predict-then-optimize" losses can be tested later, but only
after the optimization and execution model are frozen. Otherwise the loss
silently changes whenever the execution study changes, making it impossible to
tell whether a forecast improved or the policy was overfit.

## Validation protocol that can support a strong claim

1. **Repair and freeze.** Rebuild ST intervals and finite-horizon labels first;
   record dataset hashes, feature list, target definition, horizon, cost model,
   candidate count, and hyperparameter budget.
2. **Nested chronological evaluation.** Use an outer expanding/rolling split;
   choose model class, quantiles, and capacity only inside an inner purged
   rolling split. Purge at least the largest finite label horizon and embargo
   any publication or execution lag. Do not call 2023-2025 an untouched holdout:
   it has already been repeatedly inspected.
3. **Date-clustered forecast scoring.** Report CRPS/pinball/log score, tail
   coverage, VaR/ES joint score, calibration curves, and effect sizes relative
   to the unconditional and L2 baselines. Aggregate losses by date before HAC
   or block bootstrap. Report annual and regime slices without selecting on
   them.
4. **Selection correction.** Pre-register the full candidate family and use a
   White Reality Check or Hansen SPA for many model/target comparisons. Use DSR
   only as a supplementary selected-Sharpe correction. Use PBO only if the
   number of approximately independent blocks is adequate; the current D20
   history has only about `181` non-overlapping blocks.
5. **Economic separation.** After the forecast contract is frozen, run one
   separately specified account replay with costs, T+1, capacity, cash, and a
   low-`K` constraint. Do not tune thresholds and execution on the same OOS
   path. A positive account without forecast-score improvement is suspicious;
   forecast skill without account profit is a realization problem, not proof
   that the model is useless.
6. **Confirmation.** Only a future period not used for any design choice can
   provide confirmatory evidence. Until then, conclusions are retrospective
   rolling OOS and should be treated as research estimates.

## What success should mean

Do not replace 8% with another arbitrary pass rule. Before fitting, define
`d_t = mean_i[L_baseline-L_model]`, a minimum relevant score gain
`delta_score`, pre-specified market regimes `r`, and a tolerated regime loss
`delta_regime`. A strong *forecast* requires the selection-adjusted lower
confidence bound for `E[d_t]` to exceed `delta_score`, while simultaneous
regime intervals support `min_r E[d_t|r] > -delta_regime`. Quantile exceedance
and VaR/ES calibration errors must also lie inside pre-registered tolerances
after date dependence and multiplicity are accounted for. The tolerances come
from forecast resolution, costs, and risk budget before results are observed;
they are not chosen from the winning curve.

This establishes predictive value, not a profitable trading system. Calling
the model economically useful additionally requires a separately frozen
low-`K`, cash-permitted decision layer whose net-utility improvement survives
its own uncertainty analysis. Both claims require no unresolved material
data-quality blocker.

## Recommended next steps

1. Reconstruct historical ST/name intervals and run a drop-versus-rebuilt
   ablation for `market_all__st_rate` and every derived limit/state feature.
2. Replace policy retries with finite-horizon `complete/right_censored/
   unresolved/invalid` labels and horizon-specific validity masks.
3. Freeze a small distributional target contract and benchmark: unconditional
   date prior, L2 mean, quantile LightGBM, and a hazard model on standardized
   barrier levels.
4. Run the nested purged rolling evaluation with date-level proper scores and
   pre-registered family-level SPA/Reality Check. Do not add execution tuning
   until this forecast gate passes.
5. After a passing forecast, map its scenarios to a cash-permitted robust
   log-growth/CVaR optimizer with `K` as a constraint, then reserve a new future
   confirmation period.

## Further questions

- Which features forecast the conditional *residual* after removing date-level
  market factors, rather than simply identifying volatile market days?
- Can a calibrated first-passage/hazard distribution improve peak realization
  without using future sellability as a label gate?
- What is the smallest effect size that remains positive after the audited
  commission, stamp-tax, spread, and participation costs?
- How much of any selected-tail improvement survives after correcting for the
  full number of target, horizon, feature, threshold, and execution trials?

## Evidence paths

- Reproducible audit code: `daily_research/research_records/seq100/mathematical_foundations_20260805/audit.py`
- Machine-readable audit summary: `daily_research/research_records/seq100/mathematical_foundations_20260805/summary.json`
- Daily dependence evidence: `daily_research/research_records/seq100/mathematical_foundations_20260805/endpoint_by_date.csv`
- Data-quality findings: `daily_research/research_records/seq100/mathematical_foundations_20260805/data_quality_findings.csv`
- External literature evidence: `daily_research/research_records/seq100/mathematical_foundations_20260805/literature_evidence.md`
