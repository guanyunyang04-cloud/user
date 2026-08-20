# Exact value-growth Top3 portfolio and risk frontier

## Status and interpretation

This is an adaptive retrospective study requested after the 2012-2025 Top10,
path, D60 and D120 results had already been read. It is useful for choosing a
simple executable Top3 shadow challenger, but it is not an untouched
confirmation and does not authorize a stable-profit or production claim. No
2026 outcome was read, and neither frozen 2026 shadow arm was changed.

## Question and frozen meanings

“Hold only the top three” has two materially different meanings. Both were
tested without using rank 4 or lower as a replacement:

1. A literal account with at most three stocks at any time. It either rotates
   at each monthly signal or holds a cohort to fixed D20, D60 or D120 and skips
   new monthly orders while all three slots are occupied.
2. Buying every month's Top3 as overlapping cohorts. This needs up to 6, 12 or
   21 slots for D20, D60 or D120 and is therefore only a diagnostic, not an
   answer to the literal three-stock constraint.

All accounts use next-legal-open entry, legal-close exit deferral, T+1,
stops/suspensions/price limits, 100-share lots, minimum commission, taxes,
double slippage, a 0.5% signal-day-amount capacity cap, raw execution prices,
and back-adjust-factor total-return accounting. Pyramiding is forbidden.

## Rank and cohort evidence

Complete monthly cohorts show the following stress-net mean returns:

| Horizon | Top3 | Ranks 4-10 | Top3 minus ranks 4-10 | Interpretation |
|---|---:|---:|---:|---|
| D20 | 1.56% | 1.01% | 0.44 pp | full-history paired lower bounds cross zero |
| D60 | 5.52% | 3.84% | 1.58 pp | full-history lower bound is approximately zero; post-2019 delta is only 0.09 pp |
| D120 | 10.70% | 6.71% | 4.09 pp | positive full-history lower bounds, but post-2019 lower bounds cross zero |

Ranks 1 and 2 were materially stronger than rank 3. At D60 their individual
means were about 6.36%, 6.60% and 3.46%; at D120 they were about 12.24%, 12.24%
and 7.75%. Unequal weights were not optimized because doing so after reading
these differences would add another outcome-driven degree of freedom and
reduce diversification.

## First Top3 account comparison

Each strategy initially received its own 15% target-volatility gross fraction
derived only from its 2012-2019 full-exposure path.

| Strategy | Gross | Annual compound return from log growth | Volatility | Max drawdown | Positive years | Gate |
|---|---:|---:|---:|---:|---:|---|
| Monthly rebalance, max 3 | 51.00% | 6.74% | 15.58% | 24.35% | 9/14 | fail |
| Fixed D20, max 3 | 56.92% | 5.90% | 15.40% | 31.10% | 9/14 | fail |
| Fixed D60, max 3 | 57.71% | 10.97% | 16.66% | 25.95% | 10/14 | fail only on volatility |
| Fixed D120, max 3 | 49.41% | 9.00% | 15.65% | 33.83% | 11/14 | fail on volatility and drawdown |
| Fixed D60, overlapping monthly Top3 | 84.84% | 11.66% | 14.25% | 26.33% | 12/14 | pass, but holds up to 12 stocks |

The overlapping D60 result is the strongest diagnostic and shows that
diversification, not a magical exit, is what makes repeated Top3 cohorts more
stable. It cannot be described as a three-stock portfolio.

## Frozen round-number risk frontier

After the first comparison, a separately named posthoc study froze D60 and
D120 and gross fractions 40%, 45%, 50% and 55% before reading any grid output.

| Policy | Gross | Annual compound return | Volatility | Max drawdown | Positive years | Worst year | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| D60 | 40% | 8.33% | 11.89% | 18.65% | 11/14 | -9.85% | pass |
| D60 | 45% | 9.22% | 13.30% | 20.83% | 11/14 | -11.12% | pass |
| D60 | 50% | 9.99% | 14.65% | 22.89% | 10/14 | -12.36% | pass |
| D60 | 55% | 10.65% | 15.96% | 24.97% | 10/14 | -13.52% | fail volatility |
| D120 | 40% | 7.46% | 12.75% | 28.42% | 11/14 | -12.54% | pass |
| D120 | 45% | 8.30% | 14.32% | 31.34% | 11/14 | -14.07% | fail drawdown |
| D120 | 50% | 9.10% | 15.85% | 34.17% | 11/14 | -15.66% | fail |
| D120 | 55% | 9.86% | 17.34% | 36.85% | 11/14 | -17.16% | fail |

The frozen choice rule selects fixed D60 at 50% gross: it is the highest-growth
grid point passing every predefined stability gate. D60 at 45% is the more
conservative neighboring implementation. Refining the boundary to 51%, 52%,
or another hindsight-optimal fraction is not justified.

## Exact meaning of the selected historical challenger

- Run the already-frozen exact 30/30/25/10/5 main-board ranking monthly.
- Buy only ranks 1-3; never substitute rank 4 or lower.
- Hold at most three stocks and do not pyramid.
- At 50% gross, each filled slot receives at most one sixth of current account
  equity. Unfilled or unavailable slots remain cash; unused cash is not
  redistributed to the other names.
- Enter at the next legal open.
- Request sale at signal D60 close and defer to the first legal sellable close
  if necessary.
- While all slots are occupied, skip later monthly lists. After exits, wait for
  the next monthly signal. Thus this is a roughly quarterly cohort strategy,
  not monthly replacement of all holdings.
- Do not add the rejected fixed profit targets or five-minute T rules.

The 50% account closed 143 trades, had a 56.64% winning-trade rate, a 1.71%
median trade return, a 6.09% mean trade return, and mean occupancy of 60.46
sessions. It grew CNY 1 million to about CNY 3.427 million after modeled costs.
Mean signal-period utilization was 42.87%, below the 50% ceiling because of
unfilled/blocked orders and time spent waiting for the next monthly list.

Annual returns were approximately +28.2%, +16.3%, +25.7%, +16.1%, -0.04%,
+16.9%, -12.4%, +16.1%, -2.3%, +18.7%, -8.4%, +0.5%, +23.5%, and +0.4% from
2012 through 2025. Four loss years and dependence on a positive right tail make
clear that “passed historical stability gates” is not “stable profit every
year.”

## Durable decision

If the user insists on a literal maximum-three-stock account, fixed D60 with
equal slots and a 45-50% gross ceiling is the current evidence-backed range;
50% maximizes return among the simple passing grid points, while 45% provides a
larger risk buffer. The stronger overlapping D60 result is evidence in favor of
diversification, not a reason to claim that three names are inherently stable.
The Top3 challenger needs an isolated forward ledger before any production
claim.

## Authoritative artifacts

- Contracts:
  `daily_research/studies/seq100_exact_value_growth_top3_portfolio_v1.json` and
  `daily_research/studies/seq100_exact_value_growth_top3_risk_frontier_v1.json`
- Implementations:
  `daily_research/path_policy/seq100_exact_value_growth_top3_portfolio.py` and
  `daily_research/path_policy/seq100_exact_value_growth_top3_risk_frontier.py`
- Recomputable outputs:
  `daily_research/output/path_policy/studies/seq100_exact_value_growth_top3_portfolio_v1/`
  and
  `daily_research/output/path_policy/studies/seq100_exact_value_growth_top3_risk_frontier_v1/`
