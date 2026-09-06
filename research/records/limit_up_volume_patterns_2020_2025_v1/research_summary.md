# Limit-up volume pattern study (2020-2025)

This is a descriptive event study over active point-in-time QDP main-board daily data. It is not a portfolio backtest.

## Scope and definitions

- Signal window: 2020-01-01 through 2025-12-31; ordinary main-board and ST rows are reported separately.
- Upper limit: 10% for ordinary rows and 5% for ST rows, rounded to two decimals with a half-tick tolerance of 0.0051.
- The first 5 observed trading rows of each symbol are excluded from limit-up classification to avoid IPO no-limit windows.
- Pattern A: first board volume is greater than the previous trading day and greater than 1.5 times its preceding ten-observation average; second board volume is greater than first board volume; the two boards must be consecutive market days.
- Pattern B: first board meets the same 1.5-times ten-observation volume threshold, followed immediately by at least one consecutive one-word limit-up day whose volume is below both the first-board volume and that day's preceding ten-observation average. The shrink days do not need to be monotonically decreasing.
- Pattern B re-expansion: the first later observed day within five market days whose volume exceeds 1.5 times its preceding ten-observation average; `immediate` means it is the day directly after the shrink run.
- Premium is measured from the event-day close. Open, close, and high returns are reported for the next 1, 2, 3, and 5 market days. Missing outcomes remain missing.
- Resonance is same-day same-industry breadth, excluding the event stock. QDP provides a dated industry taxonomy here; concept-level resonance is not claimed.

## Event counts

- Pattern A events: 3,725; Pattern B re-expansion events: 499.
- Pattern B immediate re-expansion events: 438.

The CSV files contain the complete grouped statistics; `event_records.parquet` contains the underlying event rows and all forward outcomes. A positive conditional mean is not treated as evidence of a tradable edge without checking event count, date-level consistency, missing outcomes, costs, and account constraints.

## Output files

- `pattern_summary.csv`: overall and ordinary/ST summaries.
- `annual_summary.csv`: year-by-year summaries.
- `resonance_summary.csv`: industry and market resonance bins.
- `run_length_summary.csv`: Pattern B by shrink one-word run length and re-expansion gap.
- `data_audit.csv`: QDP coverage and event audit.
