# Seq100 executable bad-tail study v1

Date: 2026-08-06

## Research question

The corrected stock-level object is factorized as

\[
 p(A_{buy},A_{sell},\tau,R^{exe}_H\mid F_t)
 =p(A_{buy}\mid F_t)
  p(A_{sell},\tau\mid A_{buy},F_t)
  p(R^{exe}_H\mid A_{buy}=A_{sell}=1,F_t).
\]

The primary economic continuous target is the absolute gross executable
log-return `executable_log_return_H`.  The leave-one-out residual return is a
stock-selection diagnostic only.  Costs are deliberately not hidden in the
label: after the gross law is forecast, a decision layer can compare its
quantiles with a candidate-specific cost hurdle.

The bad-tail coordinate is

\[
 q_-(F_t)=P(R^{exe}_H\leq0\mid F_t),
\]

and is evaluated separately from the score used to rank potential winners.
Sell delay is treated as a discrete survival process.  For a filled duration
`d`, the likelihood contribution is `log S(d-1)+log h(d)`; a right-censored
duration contributes `log S(20)`.  This avoids treating unresolved exits as
ordinary negative returns.

## Frozen design

- Primary horizon D20; D10 is diagnostic.
- Expanding chronological folds for 2017-2022, with an `H+20` purge and at
  most 64 deterministic training rows per date.  This is 1,459 signal dates
  and 12 horizon-year folds.
- Student-t feature-scale baseline on the existing 314-field `core_minute`
  block.
- A low-capacity additive rank GAM over the five predeclared structural
  features: `log_turnover_pct_1d`, `ma_distance_60d`, `return_20d`, `atr_60d`,
  and `volatility_60d`.  Ranks are computed over the full date cross-section
  before any target-validity mask.
- A fixed-capacity LightGBM challenger on the 314-field `core_minute` block.
  Mean and six quantile heads use 128 rounds, depth 4, 15 leaves, minimum leaf
  size 512 and L2 penalty 1.  Binary heads use the same declared capacity.
  Independent quantiles are pointwise monotonically rearranged after fitting;
  raw crossing rates are retained for audit.
- Entry and sell states use causal date-equal logistic baselines and the two
  declared challenger families.  `tau_sell` uses an empirical hazard and an
  additive rank-GAM hazard with delay indicators.
- Top-48 gross and cost-proxy returns are diagnostics only.  The two fixed
  cost proxies are 30bp and 60bp per round trip; neither is an account cost
  assumption.

No PIT concatenation, portfolio search, account optimization, or 2023-2026
read was performed.

## Mechanical result

The run is recorded at
`daily_research/output/path_policy/studies/seq100_stock_bad_tail_v1/experiments/development_2017_2022/`.

All 12 folds have `audit.status=ok`.  Full candidate rows are retained before
observed-return masking; all purges pass; all post-rearrangement quantiles are
monotone; the input and label fingerprints match the corrected pack; and the
forbidden-2026 read count is zero.  A separate risk-decile audit is stored in
`risk_decile_audit.json`, with daily, summary and inference Parquet files.

The mechanical audit passes.  The predictive forecast gate does not.

## Forecast evidence

### Absolute executable-return law (primary)

For D20, pooled mean-pinball improvement over the Student-t scale baseline is:

- LightGBM direct quantiles: `-0.001653`, HAC 95% interval
  `[-0.002774,-0.000532]`, moving-block interval
  `[-0.002871,-0.000640]`.
- Rank GAM distribution: `+0.000058`, HAC interval
  `[-0.000187,0.000302]`, moving-block interval
  `[-0.000181,0.000356]`.

The same conclusion holds for the quantile-CRPS approximation.  The D10
LightGBM result is also negative; the D10 GAM difference is small and its
interval crosses zero.  Thus the direct absolute law is not improved by the
fixed nonlinear challenger family.

### Residual stock-selection law (diagnostic)

After removing the leave-one-out market and industry components, LightGBM
direct quantiles improve D20 mean pinball by `+0.000131` with HAC interval
`[+0.000057,+0.000205]`, block interval
`[+0.000059,+0.000209]`, and family Reality Check `p=0.0040`.  D10 has the
same sign.  The rank GAM does not improve this residual law.

This is strong evidence that the current feature set contains cross-sectional
idiosyncratic information, but it is not evidence that the common market move
is forecast well enough for a long-only absolute-profit rule.

### Bad-tail probability versus bad-tail ranking

Proper probability scores fail to support the current hurdle.  At D20,
LightGBM binary loss probability is worse than the recent-252-day probability
baseline by `-0.040761` log-score units (HAC lower bound `-0.070388`, block
lower bound `-0.071211`).  The GAM logistic improvement is positive in point
estimate but its confidence intervals cross zero.  The LightGBM probabilities
also show severe prior/regime miscalibration: annual predicted loss rates lag
the observed 2018-2022 changes.

The ordering contains useful risk information despite the failed probability
calibration.  The separately persisted risk-decile audit ranks every full
date cross-section first and only then masks unavailable outcomes.  For D20,
the highest predicted-loss decile minus the lowest predicted-loss decile has
an average loss-rate gap of `+0.0895`; HAC and block lower bounds are `+0.0571`
and `+0.0552`, respectively, and the sign is positive in all six years.  The
high-risk decile's simple return is about `-1.65%` relative to the middle 80%
of the universe, with HAC/block lower bounds about `-2.31%/-2.26%`.

Therefore the defensible statement is **bad-tail ranking supports a veto
coordinate, while the current probability head is not calibrated enough for a
probability-based utility calculation**.  No veto fraction has been selected
as a policy.

### Winner ranking and absolute cost

The best broad selector is the LightGBM residual conditional-mean score.  Its
D20 Top-48 excess over the same-day full-universe mean is `+0.837%` per cohort;
HAC and block lower bounds are `+0.383%` and `+0.432%`, with family Reality
Check `p=0.0020`.  This is a relative cross-sectional result.

At the same time, the absolute Top-48 60bp cost proxy is only `+0.679%` in
point estimate, with HAC lower bound `-0.516%`, block lower bound `-0.457%`,
and Reality Check `p=0.204`.  The 30bp proxy likewise has a negative lower
bound.  The D10 pattern is similar.  The selector ranks better stocks relative
to their contemporaneous universe; it has not established positive absolute
net return after a conservative cost hurdle.

Regime tables show the relative edge is strongest in high-volatility regimes
and often negative in low-volatility regimes.  This is a candidate interaction
for a future common-factor/cash gate, not permission to tune a regime policy on
the same development sample.

### Execution states and sell delay

LightGBM improves entry-state log score with D20 HAC lower bound about `+0.0010`.
Sell-state improvements are small because sell failures are rare in recent
years.  The rank-GAM sell-delay hazard is decisively worse than the empirical
hazard (`-0.0544` D20 log likelihood, HAC lower bound `-0.0621`).  The current
law should retain a simple empirical hazard until a new survival challenger is
designed; no learned delay policy is justified.

## Formal decision

`forecast_gate_failed_absolute_law_but_risk_veto_and_residual_relative_signal_supported`

The result does **not** say that the data contain no useful information.  It
says the information separates into two coordinates:

1. A robust negative-tail ranking that can identify names to avoid.
2. A robust residual ranking that improves cross-sectional ordering.

Neither coordinate, by itself, supplies the missing common-factor forecast
needed to turn a relative edge into a long-only, cost-positive absolute rule.
Probability calibration and sell-delay hazard are not yet complete enough for
an expected-utility account layer.

Consequences:

- Do not read 2023-2025, consume 2026, or optimize K, veto thresholds, holding
  periods, or account cash on this failed forecast gate.
- Preserve the LightGBM residual score and the separate high-risk veto as
  research coordinates only.
- The next model must be a hierarchical factor law: forecast market/industry
  common return and residual stock return separately, then recombine their
  distributions coherently.  The cash decision should depend on the resulting
  absolute left tail, not on a post-hoc Top-K return search.
- Add a causal prior-shift calibration layer for the loss probability (using
  only training-period prevalence) and compare it against the current proper
  score.  This is a new declared challenger, not a retroactive repair of this
  result.
- Keep the empirical sell-delay hazard as the baseline; a future survival
  challenger must beat it under censoring-aware log likelihood before it enters
  any utility calculation.

The reserved 2023-2025 retrospective matrix and final unused confirmation
period remain untouched.
