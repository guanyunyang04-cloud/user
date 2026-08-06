# Directional Structure and Strategy Feasibility Follow-up

Status: post-hoc exploratory research, no strategy selected, 2026-08-06

Audience: maintainers of the corrected Seq100 stock-distribution research.
This record refines the earlier "risk is more predictable than location"
conclusion. It does not turn inspected data into a confirmation sample.

## Decision Enabled

The corrected data contain directional cross-sectional structure, but most of
it separates a severe negative tail from the rest of the universe. The signal
is much better at identifying crowded, extended, volatile stocks to avoid than
at distinguishing the best winners among the remaining stocks. The next model
should therefore use a bad-tail hurdle or veto plus a separate conditional
return model. A standalone extreme Top-K contrarian rule is not selected.

## Evidence Boundary

- Every feature is available at the signal-date close.
- Entry is the next raw open; planned exit is the D10 or D20 raw close and a
  blocked sale is retried for 20 trading dates.
- The descriptive audit uses corrected, fingerprint-bound labels and inputs.
- Feature-decile results through 2024 use the full inspected history and are
  hypothesis generation, not OOS evidence.
- The five-feature score was constructed after inspecting the current ridge
  coefficients and univariate relationships. Both 2017-2022 and 2023-2025 are
  therefore retrospective checks, not independent confirmation.
- All inference is per trading date with Bartlett/Newey-West lag 20.
- The execution audit uses isolated CNY 1 million daily cohorts. It is a
  cost-bearing signal diagnostic, not a continuous overlapping-capital account.
- The execution engine applies next-open fill state, T+1, price-limit sale
  retries, 100-share lots, minimum commission, transfer fee, date-dependent
  stamp tax, and base/stress slippage. Candidate selection does not use future
  fill or sell state.
- No 2026 outcome is read. Signals without a complete `H+20` window by
  2025-12-31 are excluded before ranking.

## Why the Previous MFE Direction Was Misleading

For a stylized zero-drift path `X_s = sigma W_s`, the expected continuous-time
maximum satisfies

`E[max(0 <= s <= H) X_s] = sigma * sqrt(2 H / pi)`.

The exact discrete value differs, but the structural point remains: expected
MFE rises mechanically with volatility even when expected terminal return is
zero. A model trained on MFE can score well by learning `sigma`, without
learning a positive drift or an executable exit rule.

The project's earlier descriptive audit found that high ATR and high turnover
greatly increase Top-5 MFE incidence while worsening adversity. The corrected
fixed-horizon audit below completes the picture: the same high-ATR,
high-turnover, price-extended stocks have materially worse D10/D20 executable
returns. The old MFE tree direction therefore mixed learnable volatility with
unavailable oracle peak timing. This is a target-design failure, not evidence
that tree algorithms or all stock data are useless.

## Univariate Directional Structure

For each signal date from 2012 through 2024, each feature is ranked among
stocks with a valid target. The table reports the mean return of the high
feature decile minus the low feature decile. Negative values mean the high
feature group subsequently performs worse.

| Feature | D10 executable spread | D20 executable spread | Interpretation |
|---|---:|---:|---|
| signal-day log turnover | `-1.322%` | `-2.477%` | crowded/high-turnover names reverse |
| distance above 60-day mean | `-1.292%` | `-2.093%` | price extension mean-reverts |
| prior 20-day return | `-1.099%` | `-1.856%` | medium-horizon reversal |
| 60-day ATR | `-1.032%` | `-2.057%` | high path scale has poor endpoint return |
| 60-day volatility | `-0.990%` | `-1.978%` | volatile names carry negative endpoint asymmetry |

All five HAC intervals exclude zero. Shadow and market/industry-residual
returns have the same direction, so the result is not created solely by the
future sellability filter or by the common market factor. These features are
correlated and this is a multiple-inspected family; the table is structural
description, not five independent anomalies.

## Exploratory Nonlinear Score

Within each date, define percentile ranks for signal-day log turnover,
60-day moving-average distance, prior 20-day return, 60-day ATR, and 60-day
volatility. The exploratory score is

`S = 1 - mean(the five percentile ranks)`.

A high value means low crowding, low price extension, low prior return, and low
volatility. Equal weighting is deliberately transparent but was not
pre-registered. It also double-represents scale through ATR and volatility, so
it is a diagnostic rather than a final feature formula.

### Decile shape

The date-equal executable-return curves, ordered from low to high score, are:

- D10, 2017-2022:
  `-1.251/-0.499/-0.157/-0.063/+0.051/+0.114/+0.106/+0.076/+0.105/+0.051%`.
- D10, 2023-2025:
  `-1.479/-0.236/+0.104/+0.198/+0.256/+0.318/+0.346/+0.403/+0.364/+0.372%`.
- D20, 2017-2022:
  `-2.308/-0.944/-0.432/-0.138/+0.068/+0.191/+0.159/+0.160/+0.237/+0.184%`.
- D20, 2023-2025:
  `-2.223/-0.408/+0.112/+0.282/+0.408/+0.579/+0.662/+0.680/+0.699/+0.737%`.

The curve is asymmetric. Most of the information is the collapse in the
lowest one or two score deciles. Returns flatten through the upper half, and
the extreme high-score group is not consistently best in development. This is
why concentrating into Top-3 names does not inherit the full top-minus-bottom
spread. The score is a more defensible veto than a winner selector.

### Directional diagnostics

| Horizon/period | Rank IC | Positive-IC dates | Top-decile up rate | Universe up rate | Up-rate increment |
|---|---:|---:|---:|---:|---:|
| D10, 2017-2022 | `0.0765` | `67.4%` | `49.69%` | `47.74%` | `+1.95 pp` |
| D10, 2023-2025 | `0.0955` | `68.3%` | `50.20%` | `47.30%` | `+2.90 pp` |
| D20, 2017-2022 | `0.0967` | `71.9%` | `49.12%` | `46.61%` | `+2.51 pp` |
| D20, 2023-2025 | `0.1105` | `70.6%` | `51.60%` | `47.27%` | `+4.33 pp` |

The Rank-IC HAC intervals are positive. The D10 development and D20 period
up-rate increments have positive HAC lower bounds, except the 2023-2025 D10
lower bound is slightly below zero. This demonstrates modest relative
directional information. It does not mean the model can state with confidence
that an individual stock will rise: the selected up rate remains close to
50%, and most returns are driven by noisy tails.

## Cost-Bearing Top-K Audit

The same post-hoc score is evaluated as an extreme Top-K entry rule. The table
shows date-equal stress-cost net return for K=48; smaller K values are generally
worse and more capacity-concentrated.

| Horizon/period | Stress net mean per cohort | HAC 95% interval | Conclusion |
|---|---:|---:|---|
| D10, 2017-2022 | `-0.467%` | `[-0.999%,+0.066%]` | negative development mean |
| D20, 2017-2022 | `-0.476%` | `[-1.440%,+0.489%]` | negative development mean |
| D10, 2023-2025 | `-0.011%` | `[-0.666%,+0.645%]` | approximately zero, unsupported |
| D20, 2023-2025 | `+0.370%` | `[-0.840%,+1.580%]` | positive point estimate, unsupported |

The 2023-2025 D20 result is not stable by year: mean stress cohort returns are
`-0.876%/+1.358%/+0.677%` in 2023/2024/2025. The interval crosses zero and the
score was selected after inspecting these years. It cannot justify a strategy.

Capacity also argues against an extreme low-turnover Top-K rule. In
2017-2022, K=3 has median signal-day order participation around `0.432%`, and
about `85.7%` of selected orders exceed `0.1%` of signal-day amount. At K=48,
the median falls to `0.029%` and about `5.4%` exceed `0.1%`. The fixed stress
slippage does not model endogenous impact, so concentrated results would be
more fragile than this audit reports.

## What Is and Is Not Predictable

The earlier statement "only risk is predictable" is too broad. The corrected
interpretation is:

1. Conditional scale is the strongest and cleanest signal.
2. There is also nonlinear directional information, concentrated in the
   probability of a severe negative endpoint after crowding, extension, and
   high volatility.
3. Once that bad tail is removed, the remaining stocks are difficult to order;
   the upper score deciles have similar means.
4. Therefore the data currently identify "what not to buy" better than "the
   next winner".
5. A relative spread can be large while absolute selected return is too small
   for costs. Profit depends on the top leg, not on an untradeable short leg in
   a long-only account.

## Required Model Change

Let `B=1` denote an economically derived bad-return event, such as net return
below a cost-and-risk threshold. A useful decomposition is

`E[R_net | F] = P(B|F) E[R_net|B,F] + (1-P(B|F)) E[R_net|not B,F]`.

The current evidence says much of the learnable structure may live in
`P(B|F)`. The next challenger should therefore contain:

1. a lower-tail hurdle or quantile head for the probability and size of the bad
   executable-return state;
2. a separate conditional location/ranking model on the non-vetoed universe;
3. the previously required feature-scale, skew/tail, and sell-delay hazard;
4. cash as an admissible action when the lower confidence bound of net utility
   is non-positive.

A low-capacity generalized additive/monotone-bin baseline is appropriate before
another large tree or deep model, because the decile shape is strongly
nonlinear while the effective sample is trading dates, not stock rows. A
LightGBM quantile/hurdle challenger can follow, with all binning, feature
selection, and regularization performed inside nested chronological folds.

The most defensible strategy hypothesis is not "buy the lowest-volatility three
stocks." It is:

- veto the crowded/extended/high-scale negative tail;
- rank the remaining universe with a separately learned net-return signal;
- use broad K=24-48 and staggered D20 cohorts to control concentration and
  turnover;
- trade only when expected net utility clears costs with a positive lower
  confidence bound; otherwise hold cash.

This is a research contract for the next experiment, not a production policy.
Final confirmation requires future data not used to construct the score or the
policy.

## Engineering Finding

The execution audit exposed and fixed a date-normalization defect:
`evaluate_candidate_execution` accepted a NumPy `date_values` array by contract
but `pandas.Timestamp` rejected its `numpy.str_` elements. `_normalize_date`
now converts `numpy.str_` to native `str`, and the global-date T+1 regression
test uses the native pack representation.

## Sources

- Main study record: `research_record.md`
- Machine-readable follow-up: `directional_structure_followup.json`
- Distribution implementation:
  `daily_research/path_policy/seq100_stock_distribution.py`
- Execution implementation:
  `daily_research/path_policy/seq100_candidate_execution.py`
- Execution regression tests:
  `daily_research/path_policy/tests/test_seq100_candidate_execution.py`

