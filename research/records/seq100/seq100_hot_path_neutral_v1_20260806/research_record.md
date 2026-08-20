# Hot-Path Neutral Audit V1

The complete-panel hot-path study and its rolling pair audit found the same
candidate in every 2019-2025 year: attention quintile 3 combined with the
lowest same-date absolute-amount quintile. This follow-up freezes that rule and
uses date-equal coarsened exact-matching ATT estimates to separate activation,
low amount, point-in-time industry, float-market-cap, and turnover exposure.

## Main result

After matching on date, industry, and same-date float-market-cap decile, the
increment from attention quintile 3 relative to other stocks in the same low-
amount quintile is only `+0.0362%` at D20. Its HAC 95% interval is
`[-0.0585%, +0.1309%]`, only four of seven rolling years are positive, and the
pre-registered activation gate fails. The earlier pair result therefore cannot
be credited to a persistent 20-day activation or "funds entering" effect.

There is a smaller short-horizon diagnostic. Using an independently valid D5
sample, the matched activation increment is `+0.0545%`, with HAC 95% interval
`[+0.0116%, +0.0974%]`, positive in six of seven years. The selected matched
cohort itself earns only `+0.1030%` after the frozen 60 bp round-trip proxy;
the implied break-even cost is about `70.3 bp`. This is conditional-information
evidence with an approximately 10 bp cost margin, not an executable profit
result.

## What drove the old pair result

The low-amount coordinate, not activation, supplies most of the apparent D20
return. After industry and size-decile matching it has `+0.8445%` D20 contrast,
but it also has a `-2.23 pp` change in the probability of reaching +10% before
-5% and `-1.72 pp` MFE, while MAE improves by `+1.99 pp`. That geometry is not
a leader or main-rise signature; it is closer to low-turnover and lower-
adversity exposure.

The selected pair has median daily amount around CNY 24.6 million and median
float market value around CNY 2.39 billion; about 54.1% of the cohort lies in
the market's smallest size quintile. When turnover-proxy matching is added,
common support falls to 25%-40%, the D20 low-amount estimate shrinks to roughly
`+0.03%` to `+0.24%`, and its lower bound changes sign across reasonable
coarsenings. The low-amount effect therefore cannot be separated reliably from
liquidity using these observational cells.

## Audit and boundary

The prepared neutral panel contains 8,438,151 unique symbol/date rows across
2012-2025. It has no invalid float-market-cap rows and reads no 2026 data. D5
and D20 use independent validity and matching weights. Primary HAC estimates
were recomputed from the retained daily contrast file and match exactly.

This is not a continuous account, a ranking inside the gate, an exit rule, or a
capacity claim. The next bounded experiment is a fixed D5 execution and cost
audit using the already selected signal. It must not retune the attention or
amount thresholds on 2019-2025. The signal is viable for further research only
if legal T+1 execution, state-dependent costs, liquidity participation, and a
cash-permitted account preserve the narrow D5 margin.

Detailed record:
`daily_research/output/path_policy/studies/seq100_hot_path_neutral_v1/research_record.md`

