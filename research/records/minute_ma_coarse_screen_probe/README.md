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

At 25 bps the screen retained 25.14% of possible keys but missed 17 exact MA
signal keys.  At 200 bps it retained 56.81% and recalled all exact MA signal
keys on this date.  One date is not enough to make 200 bps a production
contract, so the development runner does not enable the coarse screen yet.
Additional representative dates must show stable recall before it can replace
any exact 1-minute work.

