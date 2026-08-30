# Minute-MA rule study pilot

This record contains the representative event-study run for the finite
minute-MA rule registry. It uses the same twelve representative dates and eight
fixed symbols as minute_ma_v1, with the required historical warm-up for MA240.

The run is deliberately a development-chain check. It does not establish
full-universe profitability, does not use 2025, and does not simulate finite
cash, lots, order queues, or portfolio overlap. Entries are the next available
minute open after a causal signal; minute horizons count from that fill bar, and
returns use the percentage cost assumptions stored in the JSON record.

The pool is a declared implementation-check universe, not a survivorship-free
production universe. `s0_random_matched` is a same-stock random-time control.
The cross-sectional `s0_liquidity_matched` control is generated only for
reference signals with a prior-20-session turnover match; unmatched references
are reported explicitly and are not assigned zero returns.

Files:

- strategy_catalog.json: every attempted rule version, including unavailable
  S3 gates.
- event_study_pilot.json: source scope, signal/outcome counts, overall and
  year/category summaries, and matched-control comparisons.
