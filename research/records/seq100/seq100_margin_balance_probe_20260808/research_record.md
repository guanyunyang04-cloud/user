# Exploratory Margin-Balance Probe (2026-08-08)

Status: exploratory causal evidence only. No promotion, account replay, or
production-policy selection is authorized.

## Contract

The probe tests whether point-in-time margin financing information adds a
short-horizon trading edge inside the `quality_liquidity_pit` pool. The margin
source date is aligned to its recorded `feature_available_date`; the value is
never joined to the signal date as if it were known before publication. The
formal market support is 2012-2025 and the next valid stock open is used for
entry. Outcomes are gross next-open-to-close returns and proportional base and
double-slippage round-trip approximations over 1/3/5/10/20 sessions. Date-wise
deciles, paired daily contrasts, and Newey-West (20-lag) summaries are used.

The causal join contains 2,661,196 observed margin stock-days, 3,349 dates,
and 1,955 symbols. The margin value agrees exactly with the feature value
after source-date alignment (maximum relative discrepancy 0); no 2026 row is
read. A prior quick query that joined the raw source trade date directly to the
signal date was discarded as one-day look-ahead.

## Findings

- Five-observation log growth in `rzye` is not bullish on its own. The lowest
  and highest date-wise deciles have gross D5 means of about `+0.382%` and
  `+0.236%`; proportional base net means are `+0.087%` and `-0.058%`, while
  double-slippage means are `-0.053%` and `-0.198%`. The high-growth decile
  also has lower D5 positive frequency and worse five-session adverse tails.
- The high-margin-growth/low-price-growth quadrant has a gross D5 mean of
  `+0.434%`, versus `+0.397%` for low-margin-growth/low-price-growth. The
  paired increment is only `+0.038` percentage point, HAC 95% interval about
  `[-0.026,+0.102]`, positive in 9 of 14 years. It is not a cost-robust
  incremental financing effect.
- Holding margin growth high, low recent price performance beats high recent
  price performance by about `+0.272` percentage point at D5, positive in
  13/14 years. This large contrast is primarily short-term price reversal,
  not evidence that financing growth identifies a winner.
- A date-wise regression controlling for price, size, volatility, turnover,
  and amount gives a small negative margin-growth coefficient (about
  `-0.074` percentage point across the rank range) and a positive margin/low-
  price interaction (about `+0.337` percentage point). The interaction is a
  conditional hypothesis, not a standalone executable rule.
- Three consecutive increases in margin balance while the five-day price
  return is non-positive has no reliable incremental result: its paired D5
  contrast is about `+0.033` percentage point with a confidence interval that
  crosses zero.

## Interpretation and limits

`rzye` is outstanding yuan balance, not new buying volume. It can rise because
of new leveraged purchases, mark-to-market price changes, refinancing, or a
change in the eligible pool. `rzmre-rzche` (reported new financing minus
repayment) and `rzye`/float-market-value should therefore be studied separately
from the raw level. The observed high-growth deterioration is consistent with
crowding and forced-deleveraging risk, but does not identify a causal mechanism.

The probe does not include exact lot/minimum-commission effects, capacity,
overlapping portfolio slots, or an online learned policy. The positive gross
contrast is too small to authorize account replay. A formal frozen study must
independently recompute the causal join, add state/industry matching, and test
annual and regime stability before any strategy claim.

Detailed raw inputs are the QDP `margin_detail` dataset, the training-ready
margin feature partitions, the quality-pool `row_index.parquet`, and the dense
daily path atlas.
