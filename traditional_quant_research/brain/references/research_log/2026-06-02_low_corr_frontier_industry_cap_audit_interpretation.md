# Low-Corr Frontier Industry Cap Interpretation

- Date: 2026-06-02
- Run: `low_corr_frontier_neutralization_audit_20260602_231532`
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_month_start_20260602`
- Protocol: `horizon=20`, `monthly`, `top_n=200`, `buffer=3.0`, `fee_bps=30`, execution constraints enabled.
- Industry source: Baostock `query_stock_industry`, `month-start` forward-filled PIT industry table.
- Portfolio constraint: `group_col=industry`, `max_group_weight=0.10`.

## Result

The frontier remains strong under a 10% single-industry target cap.

| Base signal | Variant | Mean annualized | Worst year | Delta vs original |
|---|---|---:|---:|---:|
| `multifactor_rolling_ic_weighted_score` | industry-neutral + industry cap | `0.371601` | `0.094033` | `+0.022723` |
| `multifactor_rolling_ic_weighted_score` | original + industry cap | `0.348878` | `0.061373` | `0.000000` |
| `multifactor_ic_weighted_score` | industry-neutral + industry cap | `0.293380` | `0.145490` | `+0.008784` |
| `multifactor_ic_weighted_score` | original + industry cap | `0.284596` | `0.116312` | `0.000000` |
| `multifactor_low_corr_rank_score` | industry-neutral + industry cap | `0.271272` | `0.123628` | `+0.010141` |
| `multifactor_low_corr_rank_score` | original + industry cap | `0.261131` | `0.079667` | `0.000000` |

The proxy-neutral variants remain weaker:

- rolling IC proxy-neutral + cap: `0.214006`.
- IC-weighted proxy-neutral + cap: `0.261709`.
- low-corr proxy-neutral + cap: `0.199215`.

## Industry Control

The cap worked as a target basket constraint. For `top_n=200` and `max_group_weight=0.10`, the selector allows at most about `20` names per industry before execution constraints.

Realized selected weights can slightly exceed `10%` because execution constraints may block or delay some names after selection. In this run, the maximum realized selected industry weight was about `0.102564`.

The cap reduced the largest industry concentration materially, but it is still not full industry neutrality:

- It caps overweights, but does not force the portfolio to match every universe industry weight.
- Large universe industries can still be underweight.
- Execution constraints can shift realized weights after the target basket is chosen.

Observed max absolute active industry weights remained around `0.057250` for the industry-neutral variants, lower than the largest unconstrained/proxy-neutral active weights but not zero.

## Interpretation

This is stronger evidence than signal-level industry demeaning alone. The frontier protocol is not obviously dependent on large single-industry concentration, and the best industry-neutral rolling IC variant still leads after a 10% industry cap.

This still does not promote a formal strategy candidate. Remaining blockers:

- Monthly non-overlapping sample is still small, especially 2026.
- The industry table is `month-start` forward-filled, not exact daily industry-change tracking.
- The cap is a simple max-weight rule, not an optimizer-level industry-neutral portfolio.
- True market cap, float cap, share base and turnover controls remain missing.
- Capacity/slippage evidence is still based on amount/volume proxies, not full execution modeling.

Strategy candidate count remains `0`.
