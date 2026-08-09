# Margin Top10 Adaptive/Down-Day Entry Study (2026-08-09)

Status: completed adaptive retrospective study. The requested rule failed its
entry gate. No finite-account replay or production use is authorized.

## Question and causal translation

The user asked whether stocks with a consecutively increasing financing
balance and a daily Top10 absolute financing-balance increment become
profitable when bought after either:

1. yesterday's close just crossed above the smoothed adaptive line and the
   current low remains above that line; or
2. the current stock-day is down.

The study recomputes the selection from the point-in-time `margin_detail`
source. A margin observation is joined only on its recorded
`feature_available_date`; the source dates must be consecutive trading dates,
and at least three successive increases are required. The daily Top10 ranks
the absolute CNY change in `rzye`, descending, inside `quality_liquidity_pit`.
`rzye` is an outstanding balance, not new smart-money buying flow.

The adaptive line implements the supplied ER10/fast2/slow30 DMA followed by
EMA2 smoothing on back-adjusted valid stock bars. The primary down-day
definition is close below the previous valid close; close below open is kept
as a sensitivity. All conditions are known after the signal-date close, so
entry is the next legal raw-price open. Because A-share T+1 prevents an
entry-day sale, the first legal one-day close exit is signal D2, with a blocked
sale deferred to the first sellable close. The alternative exit reacts to the
first causally available negative `rzye` increment, requests the following
session's close, and times out at D60.

Returns use raw execution prices for lots and fees, the frozen adjustment
factor for corporate-action economics, CNY 100,000 per diagnostic order, and
the existing double-slippage cost contract. No 2026 price, factor, margin, or
outcome is read.

## Coverage

- Formal period: 2012-2025; 2010-2011 only supplies indicator burn-in.
- 33,598 daily margin-Top10 candidates on 3,388 dates and 1,616 symbols.
- Signal counts: 1,067 cross-support, 17,310 down-close, and 18,003 primary
  union candidates. The two conditions overlap only 374 times.
- Exact legal one-day returns are available for 33,192 candidates; the small
  remainder is retained as cash rather than removed before daily ranking.
- A financing-balance decrease is detected within D60 for 99.92% of cases.
  Median detection is signal D2 and median actual exit is D3, so the selected
  increase streak usually ends almost immediately.

## Primary result

Percentages below are per candidate unless explicitly called a daily result.

| Rule | Period | Legal D2 exact net mean | Median | Win rate | Daily cash mean | HAC 95% interval | Positive years |
|---|---:|---:|---:|---:|---:|---:|---:|
| All margin Top10 | 2012-2025 | -0.351% | -0.548% | 42.43% | -0.347% | [-0.470%, -0.225%] | 1/14 |
| Cross-support | 2012-2025 | -0.176% | -0.440% | 45.46% | -0.236% | [-0.513%, +0.041%] | 3/14 |
| Down-close | 2012-2025 | -0.279% | -0.441% | 43.25% | -0.339% | [-0.468%, -0.210%] | 1/14 |
| User union | 2012-2025 | -0.285% | -0.441% | 43.29% | -0.340% | [-0.468%, -0.213%] | 0/14 |
| User union | 2023-2025 | -0.281% | -0.632% | 41.05% | -0.353% | [-0.583%, -0.123%] | 0/3 |

Every predeclared entry-gate check failed. The intersection of cross-support
and down-close illustrates why a small historical pocket cannot be promoted:
its full-history candidate mean was +0.306%, but the late-period mean was
-0.627%, the late daily mean was -0.442%, and only one of the three late years
was positive.

The financing-decrease exit did not rescue the union. Its full-history exact
net mean was -0.289%, median -0.500%, daily cash mean -0.349%, and only 4/14
years were positive. In 2023-2025 its candidate mean was -0.247%, daily cash
mean -0.306%, and only 1/3 years was positive. A later decrease observation is
mostly confirmation that a short balance streak has already ended, not a
profitable exit signal.

## Why the rule loses

The union's full-history gross return from the executable D1 open to the D2
close averaged +0.170%, but exact double-slippage net return averaged -0.285%.
The roughly 0.46 percentage-point implementation drag is larger than the weak
gross edge. In 2023-2025 the comparable gross mean was +0.135% and exact net
mean -0.281%.

More importantly, the gross distribution is strongly right-skewed. The
full-history D2 gross median is -0.037%, and the late-period median is -0.251%.
The illegal entry-day close return averages about +0.19%, but those newly
bought shares cannot legally be sold that day. By D5 the full-history mean is
still only +0.141% and the median is -0.250%. By D60 the mean is +1.12% while
the median is -2.34%. Rare large rebounds lift the mean while the typical path
loses.

In the late period, about 33.9% of union paths are persistent losses, 29.3%
are immediate continuations, and 19.3% are late recoveries. Only about 27.1%
of losing legal-D2 trades recover above the entry by D5. The representative
charts confirm the mechanism:

- the largest winners are abrupt low-open reversals and successive large-up
  bars, often followed by substantial fading;
- the largest losers are peak failures, gap/limit-down cascades, or apparent
  adaptive-line support that breaks immediately;
- the adaptive line is a slow state description and does not bound gap or
  discontinuity risk.

Charts:

- `representative_kline_cases.png`
- `aggregate_path_diagnostics.png`

## Legal higher-price opportunity and take-profit diagnostic

After seeing the primary close-exit failure, a clearly labeled adaptive
sensitivity placed a resting take-profit order on D2, the first sellable day.
If D2 opens above the target it uses the open; if the D2 high reaches the
target it fills at the target; otherwise it retains the legal close/deferred
exit. Gross targets of 0.5%, 1%, 2%, 3%, and 5% were examined. This is a
diagnostic, not independent confirmation.

For the union, the legal D2 high exceeded the entry in 73.9% of full-history
cases and reached +0.5%, +1%, +2%, +3%, and +5% in about 65.6%, 57.6%, 43.7%,
32.9%, and 18.9% of cases. Thus a higher legal sale price often exists. It is
not enough: every target had a negative full-history and late-period mean.
The union's full-history candidate means ranged from about -0.365% at the 0.5%
target to -0.295% at 5%; all 14 annual daily means were negative even at 5%.
The late-period 5% target still had a -0.288% candidate mean and -0.346% daily
mean, with 0/3 positive years.

Fixed targets fail because they cap the rare large winners that support the
mean, while trades that never touch the target keep their entire loss. A daily
high is an opportunity bound, not evidence that the opportunity can be
selected ex ante.

## Fixed continuation models

Two shallow fixed LightGBM models use development 2012-2019, Platt calibration
2020-2022, and a one-shot 2023-2025 test. The inputs are a compact causal
market/stock/minute panel plus margin rank/streak, balance-change coordinates,
and adaptive-line state. Test candidates are ranked before future fill or
outcome availability is known.

- The next-close-up model achieved test AUC 0.543. Daily Top1 had a 50.34%
  continuation fraction but -0.294% daily cash return; Top3 had 47.55% and
  -0.405%. Its highest test decile continued only 53.9% of the time and still
  lost about 0.230% per candidate after costs.
- The exact-legal-net model achieved test AUC 0.527. Top3 improved on all
  margin-Top10 candidates by about +0.187 percentage point per day, with a
  positive paired HAC lower bound, but its absolute daily return remained
  -0.197%, its absolute confidence interval crossed zero, and only 1/3 years
  was positive. This is loss reduction, not a profitable strategy.
- No calibrated probability is genuinely high: the next-up maximum is about
  0.633 and the legal-net maximum about 0.543. The models do not support a
  claim that a subset has a high probability of one more profitable day.

Market state and signal-day VWAP stretch dominate model importance more than
the financing variables. Margin rank itself is not monotonic: late-period
ranks 1-10 are all net negative. Very long balance streaks are not safer; the
late-period 8+ streak bucket averaged about -0.91% net.

## Decision

The literal requested strategy is rejected on the consumed 2012-2025 history.
Neither legal D2 close, first financing decrease, fixed legal-day take-profit,
nor the fixed continuation filters supplies stable positive net value. The
result does not prove all margin information is useless; it says that absolute
`rzye`-increase Top10 plus these price states mostly locates crowded,
high-dispersion names whose small average gross movement cannot pay execution
costs.

No account replay was run because the entry gate failed. Any future margin
hypothesis should change the economic variable rather than tune this rule:
separate new financing minus repayment, normalize by free-float value and
turnover, distinguish coverage/list changes, and test whether a causal
five-minute exit policy can control the loss tail without clipping the rare
right tail. Such work must receive a new explicitly adaptive contract; the
present outputs cannot be relabeled as untouched confirmation.

## Reproducibility

- Frozen primary/adaptive-diagnostic contract:
  `daily_research/studies/seq100_margin_top10_adaptive_entry_v1.json`
- Implementation:
  `daily_research/path_policy/seq100_margin_top10_adaptive_entry.py`
- Focused tests:
  `daily_research/path_policy/tests/test_seq100_margin_top10_adaptive_entry.py`
- Outputs:
  `daily_research/output/path_policy/studies/seq100_margin_top10_adaptive_entry_v1/`
- Primary output contains 33,598 candidate rows, rule/day summaries, adaptive
  take-profit summaries, model scores/importances, representative cases, and
  validated source/output hashes.

