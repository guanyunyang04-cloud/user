# 2026-06-04 All Limit-Up T+1 Event Study

## Scope

- Data snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`.
- Event definition: `pctChg >= 9.5` and `close >= high * 0.999`.
- Sample: 115,779 limit-up events, 3,348 stocks, 2,524 trade dates, from 2016-01-04 to 2026-05-28.
- Execution convention: signal on T, buy at T+1 open, earliest executable sell is T+2. T+1 intraday profit/loss is only confirmation information, not an executable sell point.

## Overall T+1 Path

| Metric | Mean |
|---|---:|
| T+1 close float return, not sellable | -0.34% |
| T+2 open executable return | -0.04% |
| T+2 close executable return | 0.26% |
| T+2..T+4 close return | 0.77% |
| T+2..T+6 close return | 1.01% |
| T+2..T+11 close return | 1.50% |
| T+2..T+21 close return | 1.20% |
| T+2..T+21 max high | 21.55% |
| T+2..T+21 min low | -13.74% |
| T+2..T+21 hit +20% | 33.4% |
| T+2..T+21 hit -10% | 61.5% |

Core interpretation: raw limit-up events have large upside excursions, but close-to-close holding expectancy is weak unless the T+1 path confirms continuation. A high payoff ratio depends on filtering and management, not simply buying every limit-up stock.

## T+1 Opening Gap Buckets

| T+1 open bucket | n | T+1 limit-up rate | T+1 one-word rate | T+2 close | T+2..T+4 close | T+2..T+21 close | T+2..T+21 max high | hit +20 | hit -10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| near_limit_up_open | 19,520 | 83.1% | 96.5% | 3.35% | 8.87% | 19.68% | 49.84% | 59.1% | 55.3% |
| gap_ge_6 | 7,712 | 45.7% | 0.0% | -1.90% | -2.58% | -5.47% | 16.97% | 31.7% | 76.9% |
| gap_3_to_6 | 18,552 | 28.7% | 0.0% | -0.56% | -1.07% | -3.03% | 16.86% | 30.6% | 67.4% |
| gap_0_to_3 | 35,022 | 12.1% | 0.0% | -0.15% | -0.48% | -1.62% | 15.68% | 27.6% | 59.1% |
| flat | 6,425 | 8.8% | 0.0% | -0.27% | -0.64% | -1.86% | 15.51% | 27.3% | 59.0% |
| gap_minus3_to_0 | 21,117 | 6.4% | 0.0% | -0.13% | -0.58% | -2.13% | 15.15% | 26.5% | 58.5% |
| gap_le_minus3 | 7,431 | 6.8% | 0.0% | -0.04% | -1.51% | -4.95% | 16.06% | 29.4% | 70.0% |

The strongest path is near-limit or one-word open, but it is often unfilled. The worst path is high open between 6% and 9.5% that does not become a hard continuation board.

## Board Stage

| Board stage | n | T+1 limit-up rate | T+1 one-word rate | T+2 close | T+2..T+4 close | T+2..T+21 close | T+2..T+21 max high | hit +20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first_board | 84,077 | 18.6% | 8.2% | -0.10% | -0.12% | -0.65% | 17.24% | 28.7% |
| second_board | 15,652 | 38.4% | 24.7% | 0.05% | 0.35% | 0.87% | 24.69% | 38.3% |
| third_board | 6,009 | 53.2% | 37.7% | 0.89% | 2.01% | 4.50% | 32.96% | 46.1% |
| fourth_plus_board | 10,041 | 68.2% | 57.8% | 3.23% | 8.15% | 14.97% | 47.01% | 59.3% |

Higher board stages have much stronger convexity. This is useful for payoff ratio, but only if the execution can avoid buying failed high-open boards.

## Entry Access

| Entry access | n | T+1 limit-up rate | T+2 close | T+2..T+4 close | T+2..T+21 close | T+2..T+21 max high | hit +20 | hit -10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_or_unfilled_one_word | 18,829 | 83.7% | 3.53% | 9.30% | 20.59% | 50.98% | 59.9% | 54.5% |
| near_limit_but_traded | 691 | 65.3% | -1.62% | -2.90% | -5.20% | 18.67% | 36.0% | 78.4% |
| normal | 96,259 | 16.1% | -0.36% | -0.87% | -2.59% | 15.91% | 28.4% | 62.8% |

This is the central execution finding: one-word strength is real, but frequently not buyable. Near-limit but traded is dangerous because it looks strong while often failing continuation.

## Continuation Confirmation

| Condition | n | T+2 close | T+2..T+4 close | T+2..T+21 close | T+2..T+21 max high | hit +20 | hit -10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T+1 not limit-up | 84,067 | -2.34% | -2.86% | -3.94% | 12.55% | 22.3% | 65.8% |
| T+1 limit-up | 31,712 | 7.15% | 10.39% | 14.72% | 45.42% | 63.0% | 50.3% |
| signal one-word + T+1 limit-up | 11,405 | 7.21% | 15.85% | 32.06% | 65.64% | 72.7% | 41.3% |
| signal one-word + T+1 not limit-up | 4,078 | -6.64% | -8.39% | -9.97% | 7.20% | 17.0% | 83.6% |

The single most important post-entry condition is whether T+1 itself continues limit-up. Under T+1, buy day cannot be sold, so T+1 should be treated as confirmation. If it fails continuation, the default should be T+2 defensive exit.

## Management Grid Highlights

Daily bar management is approximate because intraday stop/target order is unknown. When both stop and target occur inside a window, the study uses conservative handling.

| Filter | Window | Stop | Target | n | Mean | Win rate | Avg win | Avg loss | Payoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| near_limit_entry + signal_one_word | 3 | 5 | 30 | 11,025 | 14.96% | 66.6% | 24.88% | -4.85% | 5.13 |
| third_plus_board + T+1 limit-up | 3 | 5 | 30 | 10,038 | 14.72% | 67.4% | 24.15% | -4.81% | 5.02 |
| normal_entry + T+1 limit-up | 1 | 10 | 15 | 15,497 | 8.79% | 81.8% | 11.55% | -3.69% | 3.13 |
| first_board + T+1 limit-up | 3 | 5 | 30 | 15,643 | 8.34% | 59.8% | 16.95% | -4.50% | 3.76 |
| second_board + T+1 limit-up | 3 | 5 | 30 | 6,007 | 9.92% | 56.5% | 21.23% | -4.76% | 4.46 |
| normal_entry + T+1 not limit-up | 1 | 5 | 10 | 80,762 | -1.35% | 31.7% | 4.55% | -4.13% | 1.10 |
| gap >= 6 and not hard continuation | 1 | 5 | 10 | 7,712 | -0.11% | 34.3% | 8.78% | -4.77% | 1.84 |

## Actionable Hypothesis

1. Do not treat all limit-up stocks as buy candidates. Treat T limit-up as a watchlist generator.
2. Avoid buying 6% to 9.5% high-open stocks unless they quickly become strong continuation boards. This bucket has poor mean close returns and high drawdown frequency.
3. If T+1 is normal/tradable and then closes limit-up, T+2 becomes the first real sell/hold decision. This is the most practical path for a personal account.
4. If T+1 is one-word or near one-word and unfilled, record it as strength but not as a filled trade. The actionable plan is to wait for first open-board/reseal behavior, which requires intraday data to validate.
5. If T+1 fails to continue and closes weak, the default should be T+2 open/early exit. Holding weak failed boards destroys payoff ratio.
6. Let confirmed winners run for 1 to 3 executable days with a hard loss cap around 5% to 7% and a target band around 20% to 30%. This creates high payoff mainly by cutting failed boards quickly and not capping true continuation boards too early.

## Next Research Step

Build a tradable limit-up strategy layer:

- Signal layer: T limit-up plus KAMA/ATR upper-band breakout variants if needed.
- Entry layer: T+1 open gap bucket, board stage, prior one-word status, amount/turnover/volume rank, and whether the board reseals.
- T+1 confirmation layer: since sell is impossible on buy day, classify T+1 as continuation, failed high-open, normal weak, or open-board reseal.
- T+2 management layer: defensive exit for failed confirmation, short hold for normal continuation, trend hold for one-word/third-plus continuation.
- Required upgrade: intraday or minute data for open-board/reseal/封板 strength, because daily Baostock bars cannot distinguish real board quality from simple high/low ranges.
