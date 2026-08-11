# Seven-family QVER framework effect audit

Date: 2026-08-11

## Question

What kinds of main-board A-share stocks are selected by the supplied
`Quality × Value × Earnings Revision × Confirmation` framework, what happens
to them afterward, and do the realized earnings, valuation and returns match
the framework's intended predictions?

## Frozen quantitative translation

The attachment's seven weights were translated literally before outcomes were
read:

| Family | Weight | PIT quantitative proxy |
|---|---:|---|
| Normalized valuation | 25% | next-FY earnings yield, lower positive PE/PB, TTM FCF yield, five-year cycle-normalized earnings yield |
| Actual earnings improvement | 20% | net-profit YoY, revenue YoY, performance-forecast midpoint change |
| Earnings revision | 15% | 30/90-day next-FY net-profit and EPS consensus revisions |
| Company quality | 20% | ROE, net/gross margin, asset turnover, lower debt and borrowing ratios |
| Earnings quality/cash flow | 10% | CFO/net profit, CFO/revenue, FCF/revenue, lower receivables, inventory and goodwill ratios |
| Price confirmation | 5% | industry-relative return, trend slope, relative turnover and volume ratio |
| Governance/risk | 5% | recent penalty/litigation/delisting announcements and statement-source conflicts |

Every component is a signal-date industry-relative percentile. The score is a
cross-sectional priority score, not a predicted percentage return. The hard
filter requires positive PE/PB, positive next-FY consensus net profit from at
least three institutions, covered 30/90-day revisions, non-negative relative
return and trend, adequate listing age, and no recent delisting-risk
announcement.

Signals are the last `quality_liquidity_pit` date of each calendar month. The
portfolio selects up to ten names, with at most two from one PIT industry. The
formal history is 2012-01-01 through 2025-12-31. Selection identities are
materialized before market or financial-statement outcomes are read, and no
2026 row is used.

The historical proxy does not fabricate qualitative moat, management quality,
ROIC, discretionary fair-PE ranges, analyst upgrade/downgrade institution
counts or manual high-frequency operating evidence. It therefore tests the
quantifiable core, not every part of a human research report.

## Population and selected-stock profile

The monthly main-board file contains 207,444 candidate observations. The hard
filter and complete seven-family score leave 33,468 eligible observations. The
industry-capped Top10 produces 1,676 decisions across 168 months; one early
month has only six names.

Selected stocks are not simply the smallest or cheapest names. Medians at the
signal date were:

| Characteristic | Selected Top10 | Eligible pool | Monthly main-board pool |
|---|---:|---:|---:|
| Current PE | 16.40x | 26.63x | 27.97x |
| Current PB | 2.57x | 2.90x | 2.38x |
| Market cap | CNY21.63bn | CNY19.07bn | CNY10.97bn |
| Net-profit YoY | 52.40% | 18.31% | 15.24% |
| Revenue YoY | 26.33% | 13.20% | 10.50% |
| ROE | 9.96% | 6.27% | 4.34% |
| CFO / net profit | 1.17x | 0.84x | 0.78x |
| Debt / assets | 39.75% | 45.66% | 46.15% |
| 90-day net-profit revision | +4.39% | -0.23% | -0.63% |
| 20-day industry-relative return | +7.05% | +6.56% | -0.57% |

Market-cap exposure is broad but tilts modestly upward: 21.2%, 22.8%, 25.5%
and 30.5% of selections fall in the eligible pool's small-to-large quartiles.
The largest cumulative industries are computer/communication/electronics
(5.67%), non-metallic minerals (5.61%), chemicals (4.89%), electrical
machinery (4.77%) and special equipment (4.18%). The two-name monthly cap is
always respected.

Mechanical attachment ratings do not behave like calibrated analyst ratings:
only 3 selections score A+ and 54 score A; 66.4% are B+. Qualitative fields are
missing and industry-percentile components compress the scale, so these grade
labels should not be presented as full research ratings.

## Return-ranking evidence

Returns use a signal-close decision, next-open entry, the first legal sellable
close in the D20/D60/D120 plus 20-session retry window, and a 0.6% round-trip
cost proxy.

| Horizon | Mean Top10 net return | HAC 95% lower | Mean excess vs eligible | Mean industry residual | Mean monthly Rank IC |
|---|---:|---:|---:|---:|---:|
| D20 | 0.50% | -0.59% | 0.18% | 0.17% | 0.0473 |
| D60 | 3.53% | +0.61% | 1.29% | 0.83% | 0.0660 |
| D120 | 6.77% | +1.94% | 1.88% | 1.32% | 0.0758 |

The full-history Rank IC HAC lower bounds are positive at every horizon:
0.0289, 0.0412 and 0.0470 for D20/D60/D120. This is strong evidence that the
score contains a weak, repeatable cross-sectional ordering signal.

The economic top-tail evidence is less decisive. The D120 eligible-excess HAC
lower bound is -0.03 percentage points and the industry-residual lower bound is
-0.17 points, although the corresponding three-month block lower bounds are
slightly positive/near zero. D120 is positive in 11/14 calendar years and
remains +5.11% after excluding the best year, but the 2020-2022 period has a
-2.37-point eligible excess and negative industry residual. The late-period
point mean recovers, with large uncertainty.

Score quintiles improve broadly from low to high, but the top is not monotonic:
D120 means are 2.75%, 4.45%, 4.71%, 5.80% and 5.71% from Q1 to Q5. The score is
useful for avoiding the low tail; it is less successful at distinguishing the
very best names from the next-highest group.

## Which families carry the signal

These are univariate, collinear Rank IC diagnostics, not causal contribution
estimates. At D120:

| Family | Mean Rank IC | HAC 95% lower |
|---|---:|---:|
| Normalized valuation | 0.0581 | +0.0076 |
| Earnings revision | 0.0506 | +0.0300 |
| Earnings quality/cash flow | 0.0495 | +0.0235 |
| Company quality | 0.0219 | -0.0054 |
| Actual earnings improvement | 0.0159 | -0.0101 |
| Price confirmation | -0.0271 | -0.0455 |
| Governance proxy | -0.0037 | -0.0170 |

Valuation, revision and cash-flow quality provide the clearest historical
ordering. Once the hard filter has already required non-negative relative
strength and trend, rewarding still stronger turnover/momentum is negatively
associated with subsequent returns. The governance score has little variation
after vetoes and should be treated as a risk convention rather than alpha.

## Comparison with the nearby frozen policy

The previous independent 30/30/25/10/5 policy and the literal seven-family
policy overlap on 67.6% of selected decisions. The prior score has a slightly
higher D120 Rank IC (0.0798 versus 0.0758) and its Top10 is stronger:

| Top10 policy | D60 mean | D120 mean | D120 excess vs eligible | Positive D120 years |
|---|---:|---:|---:|---:|
| Literal seven-family weights | 3.53% | 6.77% | 1.88% | 11/14 |
| Frozen 30/30/25/10/5 | 4.32% | 7.85% | 2.96% | 12/14 |

The 543 framework-only decisions emphasize actual improvement and company
quality but carry weaker normalized value. Their D120 mean is 5.13%. The 543
old-only decisions have stronger valuation and cash-flow quality and return
8.59%. This explains the loss from the literal weights; it does not authorize
re-optimizing the weights on the same history.

## Was the earnings prediction close?

The next-fiscal-year consensus parent-net-profit forecast is compared with the
latest annual result available by 2025-12-31. The selection-decision sample has
1,436 evaluable forecasts (85.7% coverage):

- median actual-versus-forecast error: -12.44%;
- median absolute error: 26.64%;
- only 41.23% fall within ±20%;
- 64.69% overpredict realized parent net profit;
- forecast/actual growth directions agree 58.75% of the time;
- realized next-year profit growth is positive in 57.14% of evaluable cases.

Deduplicating repeated monthly decisions to 1,069 symbol-fiscal-year pairs
makes the picture slightly worse: median error -17.26%, median absolute error
28.04%, 38.60% within ±20%, and 69.41% overprediction.

Reported basic EPS is even less close: for unique symbol-years, median error is
-27.47%, median absolute error 35.08%, only 31.14% are within ±20%, 77.41% are
overpredicted, and the growth-direction match is 50.88%. EPS comparison is
also more sensitive to share-count and corporate-action changes than parent
net profit, so the net-profit forecast is the cleaner primary audit.

The framework improves forecast closeness relative to the eligible pool, but
does not make one-year forecasts precise. “Revision” should be interpreted as
a probability/ranking signal, not as a reliable point estimate.

## Subsequent outcome states

At the latest monthly PIT snapshot on or before the legal D120 exit, market-cap
change is decomposed into TTM parent-earnings change and an implied valuation-
multiple change. The resulting scenarios are:

| Fundamental/valuation state | Share | Mean D120 net return | D120 positive rate |
|---|---:|---:|---:|
| Earnings and multiple both improve | 12.65% | +46.22% | 96.2% |
| Earnings improve, multiple compresses | 37.71% | -2.77% | 41.0% |
| Earnings weaken, multiple supports price | 8.47% | +7.92% | 56.3% |
| Earnings and multiple both weaken | 3.64% | -25.49% | 0.0% |
| Mixed/stable | 30.73% | +6.32% | 55.5% |
| Snapshot missing/non-positive earnings | 6.80% | +0.96% | incomplete |

The most common outcome is not “EPS × PE double hit”; it is earnings delivery
with multiple compression. This directly validates the framework's warning
that a correct earnings call can still produce a poor stock return. Conversely,
8.5% are economically supported by multiple expansion despite weaker
earnings, so a rising price does not prove that the earnings thesis was right.

Revision turns negative again by the D120 snapshot in 39.32% of selections;
that group's mean D120 net return is -1.13%. A new penalty, litigation or
delisting keyword appears by the snapshot in 2.03% of selections, but this
small proxy is not a complete governance-event database.

Observed price paths are also heterogeneous:

| Price-path state | Share | Mean D120 net return |
|---|---:|---:|
| Early spike then fade | 20.88% | -7.20% |
| Persistent loss | 19.21% | -11.02% |
| Dip then recover | 14.74% | +22.28% |
| Persistent trend | 12.35% | +31.64% |
| Late breakout | 4.06% | +22.68% |
| Mixed/range | 28.76% | +8.40% |

Another 12.65% lose at D60 but are positive by D120; their mean moves from
-7.18% to +15.30%. A short holding period will falsely label some delayed
fundamental realizations as failures, while early-spike/fade cases show why
waiting longer is not a universal cure.

## Exact finite-account replay

The literal score was also run through the existing CNY1m legal account:
monthly Top10 orders, 30 slots, next legal raw-price open, D60 legal sale,
double slippage, fees/taxes, 100-share lots, finite cash, no pyramiding,
corporate-action-equivalent marking and a 0.5% signal-day amount cap.

The full-exposure account ends at CNY3.861m after liquidation (+286.10%), but
draws down 39.62%, so full exposure is not acceptable under the existing risk
boundary. A single 2012-2019 volatility budget freezes gross exposure at
71.63%. The risk-budget account:

- ends at CNY2.717m (+171.69%);
- annualized log growth: 7.62%;
- annualized volatility: 14.20%;
- maximum drawdown: 29.15%;
- 11/14 positive years;
- 1,351 closed trades and 50.48% winning trades;
- CNY236,693 fees/slippage;
- maximum observed signal-day participation: 0.302%, below the 0.5% cap.

A fresh 2023 restart gains 16.27% after liquidation, with a 12.39% drawdown
and 2/3 positive years; 2023 loses 10.16%. The prior 30/30/25/10/5 risk account
ends at CNY2.863m, has 8.02% annualized log growth and a similar 29.22%
drawdown. The literal seven-family weights do not improve the nearby historical
benchmark.

Performance is not driven by one isolated winner: the largest 1% of positive
D120 observations supply 6.05% of summed positive returns, and the top ten
supply 6.56%. In the risk account, the largest 1% of winning trades supply
7.32% of positive P&L and the top ten supply 9.72%. This is healthier than a
single-lottery-ticket result, but it does not remove regime dependence.

## Conclusion

The framework has genuine historical screening value. It selects profitable,
lower-leverage, cash-generative, upward-revised, moderately valued companies,
and its score ranks subsequent D60/D120 returns better than chance. Its most
useful quantitative families are normalized value, revision and cash-flow
quality.

It does not deliver precise EPS forecasts, monotonic top-tail returns or a
stable-profit conclusion. About two fifths of revisions reverse within six
months; the most frequent realized state is earnings improvement accompanied
by PE compression; and 2020-2022 plus the 2023 account restart expose material
regime weakness. The literal weights are also weaker than the already-frozen
nearby policy, mainly because they exchange valuation strength for current
growth/quality signals whose marginal Rank IC is weak in this filtered pool.

The correct use is a disciplined shortlist and scenario/falsification engine,
followed by qualitative company research and forward shadow tracking. It
should not be marketed as a target-price oracle, a mechanical A+/A rating, or
a proven live strategy.

## Artifacts

- Frozen study: `daily_research/studies/seq100_qver_confirmation_effect_v1.json`
- Implementation: `daily_research/path_policy/seq100_qver_confirmation_effect.py`
- Authoritative output: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/`
- Summary: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/summary.json`
- Selected outcomes: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/selections.parquet`
- Cohort returns: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/cohort_returns.parquet`
- Component diagnostics: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/component_rank_ic.parquet`
- Forecast audit: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/forecast_realization.parquet`
- Scenario summary: `daily_research/output/path_policy/studies/seq100_qver_confirmation_effect_v1/scenario_summary.parquet`
- Exact account files: matching `account_*` Parquet files in the output root.
