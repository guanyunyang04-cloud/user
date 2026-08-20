# Margin Top10 Same-Day Adaptive-Line Cross Study (2026-08-09)

Status: completed adaptive retrospective challenger. Same-day crossing does
not robustly raise next-day continuation probability and remains net negative
after legal execution. No account replay or production use is authorized.

## Question and corrected causal contract

This challenger implements the user's corrected literal question:

1. require financing balance to increase on at least two consecutive source
   trading days;
2. rank the source day's absolute financing-balance increase Top10 inside
   `quality_liquidity_pit`;
3. on that same source day's K-line, require adjusted close to cross from at or
   below the smoothed adaptive line to above it;
4. after the source observation is published before the next open, buy that
   next legal raw open;
5. measure both next-close continuation and the first executable T+1-compliant
   close exit on D2.

The sensitivity also requires the adaptive line itself to be rising. A
prior-day-cross/current-low-support rule is retained only as a comparison.
Every candidate satisfies source date = K-line date and availability date =
next-open entry date. No 2026 outcome is read.

## Coverage

- 33,877 min-two Top10 candidates on 3,396 dates and 1,489 symbols.
- 2,481 same-day crosses, of which 1,505 also have a rising adaptive line.
- 1,469 prior-cross/support controls.
- Formal history is 2012-2025, split into development 2012-2019,
  calibration 2020-2022 and late 2023-2025.

## Does the cross increase next-day-up probability?

The answer is no on the consumed history.

| Rule | Period | Candidate next-close-up | Equal-date next-close-up | Paired change vs all Top10 | Paired HAC 95% interval |
|---|---:|---:|---:|---:|---:|
| All min-two Top10 | 2012-2025 | 46.37% | 46.40% | baseline | — |
| Same-day cross | 2012-2025 | 45.82% | 45.63% | -1.01 pp | [-3.10, +1.08] pp |
| Cross + rising line | 2012-2025 | 47.30% | 46.23% | -0.78 pp | [-3.28, +1.71] pp |
| All min-two Top10 | 2023-2025 | 45.72% | 45.73% | baseline | — |
| Same-day cross | 2023-2025 | 46.58% | 46.92% | +0.48 pp | [-3.80, +4.77] pp |
| Cross + rising line | 2023-2025 | 46.67% | 46.23% | -0.97 pp | [-7.04, +5.09] pp |

The rising-line candidate-level percentage looks slightly better than the raw
baseline, but candidates cluster on particular dates. Equal-date pairing
removes that composition effect and the estimate becomes negative with a wide
interval. None of the variants reaches a robust probability above 50%.

## Executable returns

| Rule | Period | Entry-day gross mean | Legal D2 gross mean | Candidate exact net | Equal-date cash net | HAC 95% interval | Positive years |
|---|---:|---:|---:|---:|---:|---:|---:|
| All min-two Top10 | 2012-2025 | +0.149% | +0.049% | -0.377% | -0.366% | [-0.487%, -0.246%] | 1/14 |
| Same-day cross | 2012-2025 | +0.193% | +0.136% | -0.292% | -0.375% | [-0.610%, -0.141%] | 2/14 |
| Cross + rising line | 2012-2025 | +0.267% | +0.214% | -0.215% | -0.357% | [-0.644%, -0.069%] | 4/14 |
| All min-two Top10 | 2023-2025 | +0.144% | -0.049% | -0.441% | -0.430% | [-0.661%, -0.200%] | 0/3 |
| Same-day cross | 2023-2025 | +0.294% | +0.116% | -0.285% | -0.275% | [-0.770%, +0.220%] | 1/3 |
| Cross + rising line | 2023-2025 | +0.341% | +0.070% | -0.331% | -0.271% | [-0.816%, +0.275%] | 1/3 |

There is a small entry-day gross tendency, but shares bought at that open
cannot legally be sold that day. By the first legal close the gross movement
is too small to cover costs, and the equal-date net estimates remain negative.

## Path explanation

The same-day cross is not a neutral turn detector. It selects stocks that have
already risen about 5.08% on the signal day, versus 1.18% for all min-two Top10
candidates; the rising-line subset has already risen about 5.31%. It therefore
mixes genuine breakouts with late chasing after a large bar.

In 2023-2025, the same-day-cross mean path from entry is +0.29% at D1,
+0.12% at D2, +0.06% at D5, +0.35% at D10, -0.21% at D20 and -0.76% at D60.
Its medians are 0.00%, -0.12%, -0.51%, -0.85%, -2.08% and -5.97%. Requiring a
rising line produces even weaker late D5-D60 means and a -7.32% D60 median.
Rare strong continuations lift means, while the typical path fades.

KAMA is an adaptive moving average: the efficiency ratio changes how quickly
the recursion follows price. It is a useful causal state coordinate, but it is
still calculated from past prices and is not an observed order-book supply or
cost boundary. A crossing therefore cannot be assumed to be a strong pressure
break merely because the line is smooth. References:

- TradingView KAMA description and formula:
  <https://www.tradingview.com/support/solutions/43000773012-kaufman-s-adaptive-moving-average-kama/>
- TA-Lib KAMA reference: <https://ta-lib.org/functions/kama.html>

## Decision

The hypothesis `min-two financing increases + Top10 increment + current-day
adaptive-line cross` is not a stable continuation or net-profit rule on
2012-2025. The cross does modestly improve some gross point estimates, but not
the equal-date continuation probability or executable net value. This is an
adaptive study on already consumed history; further threshold tuning cannot be
called confirmation. A future follow-up should test economically different
margin variables or freeze a genuinely forward filter rather than optimize
another KAMA parameter grid.

## Reproducibility

- Contract: `daily_research/studies/seq100_margin_top10_current_cross_v1.json`
- Implementation: `daily_research/path_policy/seq100_margin_top10_current_cross.py`
- Tests: `daily_research/path_policy/tests/test_seq100_margin_top10_current_cross.py`
- Output: `daily_research/output/path_policy/studies/seq100_margin_top10_current_cross_v1/`
- Comparison chart: `current_cross_comparison.png`

