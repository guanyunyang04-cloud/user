# Exact value-growth post-selection paths, entry, exit, and intraday T

Date: 2026-08-09

## Question

For the exact main-board 30/30/25/10/5 value-growth Top10 policy:

1. What paths do selected stocks follow after the signal?
2. Can signal-day path or feature fields provide a better legal entry?
3. Is there a repeatable five-minute T strategy while positions are held?
4. Is the inherited D60 exit economically appropriate?

This is a retrospective diagnostic. All 2012-2025 outcomes are consumed. The
D120 account challenge was opened only after the first frozen path diagnostic
identified D120, so it is explicitly adaptive and is not an untouched
confirmation.

## Data and execution contract

- Selector: frozen `exact_full_top10`, monthly, `quality_liquidity_pit` main
  board, no outcome-aware reselection.
- Selected signals: 1,676 stocks across 168 month ends from 2012-2025.
- Entry: next-session legal open, unless a declared entry challenger says to
  wait; unfilled names remain cash.
- Daily path: back-adjusted OHLC, with separate authoritative buyability and
  sellability masks.
- Cohort cost: 0.6% round trip.
- Five-minute T source: the actual risk-budget account's holdings, strictly
  after entry and before exit. All 77,783 matched stock-days had the required
  48 bars; no 2026 row was admitted.
- T execution: a close signal fills at the next five-minute open; an unpaired
  leg is forced at the last close; one 20% tranche and at most one round trip
  per stock-day.
- Account: raw legal fills, total-return factor economics, T+1, limits,
  suspension, legal-sale deferral, lots, minimum fees, stamp tax, double
  slippage, 0.5% amount capacity, no pyramiding, 30 slots, and the unchanged
  68.6089% development-derived gross-risk budget.

## Complete path distribution

The path is broad and strongly right-skewed. At D60, 1,620 selections had an
exact observed close:

| Horizon | Mean close return | Median close return | Mean MFE | Mean MAE | Any legal net-profit close |
|---:|---:|---:|---:|---:|---:|
| D5 | 0.89% | 0.59% | 5.53% | -3.95% | 69.6% |
| D10 | 1.32% | 0.82% | 7.72% | -5.27% | 78.2% |
| D20 | 1.67% | 0.44% | 10.91% | -7.50% | 84.0% |
| D40 | 3.29% | 1.47% | 16.18% | -9.82% | 88.9% |
| D60 | 5.05% | 1.89% | 20.85% | -11.65% | 90.9% |
| D120 | 8.33% | 2.79% | 31.56% | -15.31% | 93.7% |

“Any legal net-profit close” is an opportunity oracle: it asks whether a
sellable close above entry plus 0.6% occurred at least once. It does not tell a
causal trader in advance which occurrence is the final good exit. The median
first such day was D2, yet mechanically taking that first opportunity destroys
much of the positive tail, as shown below.

The frozen mutually exclusive D60 path grammar gives:

| Primary path | Share | Median D60 | Median D60 MFE | Median D60 MAE |
|---|---:|---:|---:|---:|
| Mixed/range | 28.3% | +3.40% | +11.87% | -6.77% |
| Early spike then fade | 21.0% | -4.68% | +15.49% | -11.85% |
| Persistent loss | 17.8% | -11.93% | +2.85% | -19.87% |
| Dip then recover | 15.7% | +13.53% | +24.51% | -9.65% |
| Persistent trend | 13.0% | +23.41% | +32.79% | -2.21% |
| Late breakout | 4.2% | +17.33% | +21.79% | -7.68% |

These shares and their signs were similar in development (2012-2019), middle
(2020-2022), and late (2023-2025) periods. The selector therefore repeatedly
finds both persistent winners and superficially similar failures. Its wealth
edge comes from the winners' larger positive tail, not from a very high fixed-
day hit rate.

## Entry timing

Every challenger used the same signal-D60 legal exit and retained cash when no
entry occurred. The next open remained best among the frozen simple rules:

| Entry | Fill/trigger rate | Mean monthly cash-denominator net | Paired change vs next open |
|---|---:|---:|---:|
| Next open (D1) | 99.7% | 4.34% | — |
| D2 open | 99.7% | 4.04% | -0.30 pp |
| D3 open | 99.1% | 3.91% | -0.43 pp |
| D5 open | 98.9% | 3.46% | -0.88 pp |
| Wait for -1% close, then next open | 53.5% | 2.48% | -1.85 pp |
| Wait for -2% close, then next open | 41.5% | 2.05% | -2.29 pp |
| Wait for -3% close, then next open | 32.4% | 1.58% | -2.75 pp |
| Avoid >2% opening gap and wait for +1% or lower | 96.8% | 4.12% | -0.21 pp |

No challenger passed the paired HAC/block and cross-period gate. D5 and all
three pullback waits were significantly worse over the full history. Waiting
for a cheaper-looking price often avoids a loss, but it also remains in cash
for the rare stocks that immediately become the portfolio's large winners.

## Signal-day fields

Thirty-four causal fields were split into upper and lower halves within each
month. This is a diagnostic screen within already consumed history, not a new
validated selector.

- Only one field survived the return-oriented screen: lower signal-day
  `minute_vwap_close_deviation` beat its upper half by about 2.48 percentage
  points at D60. In plain language, names that finished less stretched above
  VWAP did better than otherwise similar selected names. The full-history FDR
  q-value was about 0.093; all three period point estimates agreed, but middle
  and late period intervals separately crossed zero.
- Lower signal-day five-minute realized volatility raised the probability of
  finding a legal net-profit close by D20 by about 5.36 percentage points.
- Lower volatility, ATR, amount expansion, recent trend/momentum, and several
  related fields consistently reduced D10 drawdown. Higher earnings/FCF yield
  also tended to reduce early drawdown. These fields are highly correlated and
  mostly changed risk, not D60 return.
- No tested fundamental, traditional technical, market, or industry field
  besides VWAP stretch robustly separated D60 return in this small Top10
  cross-section.

Thus a low-VWAP-stretch ranking arm is worth freezing prospectively, but the
present history cannot establish it as a better strategy after it was found by
screening.

## Profit targets and the meaning of a legal profitable exit

The first legal close reaching a net target was taken; otherwise the position
fell back to D60. This produces a direct trade-off between win rate and wealth:

| Exit | Candidate win rate | Target hit rate | Mean monthly net | Positive years | Paired change vs D60 |
|---|---:|---:|---:|---:|---:|
| Fixed D60 | 53.6% | — | 4.34% | 12/14 | — |
| First net break-even, else D60 | 90.8% | 90.7% | 0.71% | 11/14 | -3.63 pp |
| First +3% net, else D60 | 81.7% | 80.6% | 1.51% | 11/14 | -2.82 pp |
| First +5% net, else D60 | 74.8% | 72.1% | 1.74% | 10/14 | -2.60 pp |
| First +10% net, else D60 | 65.8% | 56.3% | 2.67% | 11/14 | -1.67 pp |

All four target-timeout policies had positive full-history lower bounds, but
all materially underperformed D60. A right-boundary take-profit can make the
ledger look much smoother and raise the percentage of profitable exits; it
does so by selling the stocks responsible for the strategy's rare large gains.
It is not a free improvement in account wealth.

## Five-minute T result

The simple causal reversal rules had essentially no gross edge before cost:

| Rule | Trigger rate | Gross return per triggered tranche | Net return after 0.6% | Positive years |
|---|---:|---:|---:|---:|
| Drop 1%, then rebound 1% | 48.9% | +0.034% | -0.566% | 0/14 |
| Drop 2%, then rebound 1% | 24.7% | +0.059% | -0.541% | 0/14 |
| Rise 1%, then fade 1% | 50.6% | -0.033% | -0.633% | 0/14 |
| Rise 2%, then fade 1% | 28.5% | -0.021% | -0.621% | 0/14 |

Every rule's HAC and block upper bounds for daily equal-position contribution
were below zero in every broad period. Approximate historical incremental P&L
was negative CNY 0.82-1.93 million depending on the rule. Buy-first cash
feasibility was not enforced, making the drop rules optimistic; they still
failed.

A chronological hindsight oracle could find an average 3.10% same-day gross
open-to-open spread and about 2.50% after the same cost. This confirms that
intraday motion exists, but the declared close-to-next-open rules could not
predict which direction and reversal to trade. “There was enough fluctuation”
is not the same as “a causal T rule could harvest it.”

## D60 versus D120

At the cohort level, D120 averaged 7.86% net versus roughly 4.2%-4.3% for D60
on common months. The paired D120-minus-D60 mean was +3.61 percentage points;
its HAC and moving-block lower bounds were +0.62 and +0.85 points. All three
broad periods had positive point differences. D80 was positive but its paired
lower bounds crossed zero.

Because D120 was identified from this diagnostic, it was then replayed as an
explicitly adaptive account challenger on the common 162 signal months from
2012-01-31 through 2025-06-30. All non-exit rules were identical:

| Exposure | Exit | Ending equity | Annualized log growth | Volatility | Max drawdown | Positive years | Trades | Fees/slippage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Risk budget 68.61% | D60 | CNY 2.807m | 7.98% | 13.94% | 29.22% | 11/14 | 1,333 | CNY 227.9k |
| Risk budget 68.61% | D120 | CNY 3.140m | 8.84% | 15.07% | 30.63% | 11/14 | 759 | CNY 138.6k |
| Full exposure | D60 | CNY 4.185m | 11.07% | 20.52% | 40.95% | 11/14 | 1,342 | CNY 421.1k |
| Full exposure | D120 | CNY 5.120m | 12.63% | 22.21% | 43.11% | 11/14 | 759 | CNY 263.8k |

D120 did not improve the winning-trade rate (both were about 53%). It improved
wealth by holding the positive tail longer and cutting turnover, while using
more slots for longer and accepting slightly more risk. The risk-budget arm's
ending equity was 11.8% above D60, annualized log growth was 0.87 percentage
points higher, maximum drawdown was 1.41 points worse, and volatility was 1.14
points higher.

The accurate conclusion is therefore not “D60 is optimal.” D60 is too short
for this particular value-growth selector in the consumed history, and D120 is
the strongest current exit challenger. But D120 still had three losing years,
about 31% drawdown, and was chosen after looking at the data. It cannot be
silently installed as if it had independent confirmation.

## Decision and next boundary

1. Keep next-open entry; the frozen simple delay and pullback rules are
   rejected.
2. Reject the four tested daily T rules. Their gross edge is below realistic
   friction and every historical year was net negative.
3. Do not optimize for win rate with a fixed price/right-boundary take-profit;
   it clips the wealth-producing tail.
4. Retain D60 in the original forward arm so its contract is not rewritten.
5. D120 has now been frozen as a parallel, explicitly new forward challenger
   before the 2026-08-10 first fill, using the same candidates, entry, sizing,
   and risk budget as the original D60 arm. Do not tune another holding-day
   grid on 2012-2025.
6. A separate prospective low-VWAP-stretch arm is defensible, but it must begin
   after this screen and cannot be called historically confirmed.

The current evidence improves the proposed execution policy, but it still
does not establish annual stable profit or guarantee a profitable exit for
every selected stock.

## Artifacts

- Frozen path/T study:
  `daily_research/studies/seq100_exact_value_growth_path_exit_v1.json`
- Path/T implementation:
  `daily_research/path_policy/seq100_exact_value_growth_path_exit.py`
- Path/T output:
  `daily_research/output/path_policy/studies/seq100_exact_value_growth_path_exit_v1/`
- Adaptive exit challenge:
  `daily_research/studies/seq100_exact_value_growth_exit_challenge_v1.json`
- Adaptive implementation:
  `daily_research/path_policy/seq100_exact_value_growth_exit_challenge.py`
- Adaptive output:
  `daily_research/output/path_policy/studies/seq100_exact_value_growth_exit_challenge_v1/`
- Parallel D120 forward contract and output:
  `daily_research/studies/seq100_exact_value_growth_2026_d120_shadow_v1.json`
  and
  `daily_research/output/path_policy/studies/seq100_exact_value_growth_2026_d120_shadow_v1/`
