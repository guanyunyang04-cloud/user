# Limit-up volume pattern study (2020-2025)

This is a descriptive event study over active point-in-time QDP main-board daily data. It is not a portfolio backtest.

## Scope and definitions

- Signal window: 2020-01-01 through 2025-12-31; ordinary main-board and ST rows are reported separately.
- Upper limit: 10% for ordinary rows and 5% for ST rows, rounded to two decimals with a half-tick tolerance of 0.0051.
- The first 5 observed trading rows of each symbol are excluded from limit-up classification to avoid IPO no-limit windows.
- Pattern A: first board volume is greater than the previous trading day and greater than 1.5 times its preceding ten-observation average; second board volume is greater than first board volume; the two boards must be consecutive market days.
- Pattern B: first board meets the same 1.5-times ten-observation volume threshold, followed immediately by at least one consecutive one-word limit-up day whose volume is below both the first-board volume and that day's preceding ten-observation average. The shrink days do not need to be monotonically decreasing.
- Pattern B re-expansion: the first later observed day within five market days whose volume exceeds 1.5 times its preceding ten-observation average; `immediate` means the first market day directly after the shrink run.
- Premium is measured from the event-day close. Open, close, and high returns are reported for the next 1, 2, 3, and 5 market days. Missing outcomes remain missing.
- Resonance is same-day same-industry breadth on the event date, excluding the event stock. Concept-level resonance is not claimed.
- Industry codes are used when present; a unique same-day code is used for uncoded historical labels, otherwise a `NAME:` fallback key is retained. The fill method is reported separately.

## Event counts

- Pattern A events: 3,725; Pattern B re-expansion events: 519.
- Pattern B immediate re-expansion events: 479.

## Headline outcomes

| Group | Events | D+1 close mean | D+1 close positive | D+5 close mean | D+5 close positive | A: 3rd-board rate |
|---|---:|---:|---:|---:|---:|---:|
| A all | 3,725 | +0.96% | 50.57% | -0.45% | 37.15% | 28.27% |
| A ordinary | 3,432 | +0.99% | 50.41% | -0.63% | 35.96% | 28.21% |
| B all | 519 | -0.53% | 46.33% | -3.21% | 33.53% | n/a |
| B ordinary | 445 | -0.75% | 44.94% | -3.80% | 31.91% | n/a |
| B immediate | 479 | -0.70% | 46.03% | -3.36% | 33.40% | n/a |

Pattern A's success rate is the proportion whose next market day is still a valid third board. Pattern B rows start on the first qualifying re-expansion day, not on the shrink-board day.

## Industry resonance (ordinary main board)

The bin is the count of other same-industry stocks that closed limit-up on the event date.

### Pattern A

| Other same-industry limit-ups | Events | 3rd-board rate | D+1 close mean | D+1 close positive | D+5 close mean | D+5 close positive |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 884 | 29.17% | +1.12% | 52.15% | -0.66% | 35.86% |
| 1 | 647 | 30.34% | +1.17% | 51.47% | -0.43% | 35.39% |
| 2 | 467 | 29.25% | +0.73% | 47.11% | -0.59% | 35.76% |
| 3+ | 1,434 | 26.33% | +0.92% | 49.65% | -0.71% | 36.12% |

### Pattern B immediate re-expansion

| Other same-industry limit-ups | Events | 3rd-board rate | D+1 close mean | D+1 close positive | D+5 close mean | D+5 close positive |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 131 | n/a | -1.02% | 43.51% | -2.50% | 35.88% |
| 1 | 94 | n/a | -1.18% | 45.74% | -5.85% | 23.40% |
| 2 | 48 | n/a | -0.29% | 45.83% | -2.03% | 33.33% |
| 3+ | 139 | n/a | -1.00% | 43.17% | -4.97% | 32.37% |

## Industry coverage audit

- Event rows without a same-day industry aggregate after the fallback logic: 0.
- Source industry-code rows: 4,648,850; same-day name-to-code fills: 7,175; name fallback rows: 473; unresolved rows: 0.
- The `name_fallback` key is retained for auditability and should not be interpreted as a verified standard industry code.

## Interpretation boundaries

- These are event-level descriptive statistics, not a portfolio backtest. Same-day events can be correlated, and a close-to-close premium is not automatically executable.
- Pattern A's aggregate positive mean can be dominated by the minority that reaches a third board; inspect medians, probabilities, annual splits, and costs before treating it as an edge.
- Pattern B's re-expansion day is often an opening/disagreement day; a negative average is evidence against treating the event as an unconditional buy signal, not proof that every instance should be sold.
- Industry breadth is a descriptive filter. QDP does not provide a reliable historical concept/主题 taxonomy for this run.

The CSV files contain the complete grouped statistics; `event_records.parquet` contains the underlying event rows and all forward outcomes. A positive conditional mean is not treated as evidence of a tradable edge without checking event count, date-level consistency, missing outcomes, costs, and account constraints.

## Output files

- `pattern_summary.csv`: overall and ordinary/ST summaries.
- `annual_summary.csv`: year-by-year summaries.
- `resonance_summary.csv`: industry and market resonance bins.
- `industry_resonance_summary.csv`: industry resonance bins without a market-bin cross join.
- `market_resonance_summary.csv`: market resonance bins without an industry-bin cross join.
- `run_length_summary.csv`: Pattern B by shrink one-word run length and re-expansion gap.
- `outcome_by_success.csv`: Pattern A outcomes split by third-board success/failure.
- `headline_summary.csv`: compact overall, ordinary, and immediate-re-expansion rows.
- `data_audit.csv`: QDP coverage and event audit.
- `industry_fill_audit.csv`: source-code and fallback coverage for the industry join.
