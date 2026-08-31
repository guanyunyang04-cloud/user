# Minute-MA 5-minute coarse-screen probe

This record measures whether the active 5-minute archive can safely reduce the
exact 1-minute MA event workload.  It is a single-date implementation probe on
2022-06-16, using all 3,009 point-in-time main-board symbols and the six causal
MA periods.  The exact reference is the memory-bounded 1-minute event run; 2025
was not read.

The candidate key is `(symbol, sixty_minute_bucket, ma_period)`.  A 5-minute
key passes when any adjusted 5-minute high/low range crosses the causal MA
intersection or comes within the configured distance.  Random-time,
strong-without-MA, and cross-sectional control rows are excluded from recall
because they are not MA-event candidates.

On the full April 2022 month (19 trading dates), 25 bps retained an average of
15.82% of possible keys but missed 368 exact keys across all dates.  At 200 bps
the average candidate fraction rose to 36.21%, yet 10 dates still had misses
(31 keys in total).  The single-date probe happened to reach 100% recall at
200 bps, but the month audit shows that was not stable.  The development runner
therefore does not enable the coarse screen.  The month-level counts are in
`april_2022_summary.json`; further market states must be audited before any
exact 1-minute work is filtered.
