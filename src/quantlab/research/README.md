# Technical research mainline

This package is the small active path for daily technical research. It does not
contain live order generation.

The separate `minute_ma` module provides causal 60-minute MA10/20/40/60/120/240
states and diagnostic event summaries for the minute-strategy work. Its contract
and representative real-data checks are recorded in
`research/records/minute_ma_v1/`; those checks do not establish profitability.

## Minute-MA rule study

`minute_ma_strategies` contains the finite S0-S4 rule registry. S0 controls,
causal S1 entry variants, and the currently available S2/S4 filters can be
generated from the causal state table with `build_strategy_signals`. The
`s0_random_matched` entry is a same-stock timing control; the separate
`s0_liquidity_matched` builder selects a different stock at the same minute
using a prior-turnover band and pairs it through `reference_signal_id`. The
`s1_posthoc_catchup_diagnostic` entry is deliberately non-executable. S3 market
and sector gates remain listed in the registry but are unavailable until their
point-in-time source fields are joined.

`minute_ma_event_study` resolves every signal at the next minute open and
computes 5/15/30/60-minute, 1/2/3/5-trading-day, same-day MFE/MAE, and T+1
outcomes. Minute horizons start at the fill bar, so the 1-minute result is
that bar's close. It reports observed counts, win rates, profit factors, and
means after removing the largest positive 1% winners. Missing or blocked bars
remain missing; they are never scored as zero returns. This is an event study
with percentage costs, not an inventory or capacity backtest.

The first representative-date run is recorded under
`research/records/minute_ma_v2_pilot/`. It is an implementation and hypothesis
screening check over eight fixed symbols, not a full-universe performance claim;
the eight-symbol pool is also too small to stand in for a production liquidity
match universe.

For a larger development run, `quantlab minute-strategy-study` now evaluates
one date at a time and one symbol chunk at a time.  Month-level causal MA
inputs are cached once for the target dates, the forward outcome loader uses
the same bounded path, and repeated strategy signals at one symbol/minute
share a single price-path calculation.  All configured MA periods are built in
one rolling pass per symbol chunk.  The runner reserves a hard 0.5 GiB
machine-wide RAM floor and stops new heavy chunks below its 1 GiB soft floor;
DuckDB keeps a separate 2 GiB internal reserve by default.  These settings are
execution controls, not statistical assumptions, and are written into the run
manifest for reproducibility.
For very large pooled output sets, the winner-trimmed mean may use a positive
99th-percentile approximation; manageable monthly/grouped summaries retain
the exact top-1% removal definition.

The active-5-minute coarse-screen audits are recorded under
`research/records/minute_ma_coarse_screen_probe/`.  The full April 2022 audit
still missed exact 1-minute MA signal keys at 200 bps, so the screen is not
enabled in the development runner until recall is stable across additional
market states.

## Minute-MA portfolio replay

`minute_strategy_portfolio` replays executable event rows with finite cash,
100-share lots, fees, slippage, no duplicate open symbol, at most five open
symbols, and A-share T+1.  Stops and targets are observed on causal minute bars
and execute at the next legal minute open; locked, suspended, delisted, unknown
and missing sessions remain delayed or unresolved rather than being credited
as fills.  Positions opened during a session are marked from that session's
observed close.  Signals within one symbol/hour are deduplicated across MA
periods, then same-time orders share the available account cash equally.
Earlier signal minutes have priority; same-minute candidates use a declared
fixed-seed hash so Parquet row order and stock-code order cannot choose fills.

The first full-universe comparison is recorded under
`research/records/minute_ma_portfolio_development_v1/`.  It covers April 2022
and September 2023 only and is a development diagnostic, not evidence of a
profitable or frozen strategy.  Use `quantlab research
minute-strategy-portfolio` to reproduce the account layer from existing event
outputs.

The follow-up under `research/records/minute_ma_portfolio_seed_stability_v1/`
replays seeds 0-9 in one shared scan. Every strategy/exit variant has a
negative cross-seed mean and median, so the arbitrary same-minute tie-break is
rejected. A causal cross-sectional rank is required before expanding this
portfolio path.

## Data contract

The current repaired matrix contains 4,191,476 rows from 2012-01-04 through
the date declared by the active research manifest. The first two years of the
underlying pack are available only as lookback/training warm-up. Outcome rows
after the declared cutoff are rejected; changing the research window only
requires changing the manifest, not the code.

- `158`: 104 daily price-volume features and 54 market-state features.
- `183`: the same 158 fields plus 25 features calculated from the signal day's
  5-minute bars.
- `raw60`: normalized raw open, high, low, close, volume and amount over the
  latest 60 trading days, a valid-bar mask, and 14 whole-market state fields.

The 158-field set is a validated prefix of the 183-field matrix. There is no
separate 158 data copy and no v2/v3 data namespace.

## Research contract

- Five expanding forward folds evaluate the configured validation window with a
  30-trading-day purge.
- The common target is executable D10 base-cost net return; portfolio stress
  replay applies double slippage.
- Models rank the complete finite-score slate before future fill/outcome status
  is consulted. An unfilled or unaffordable selected order remains cash and is
  never replaced by a lower-ranked stock.
- Account replay uses next-open entry, T+1, legal delayed exits, 100-share lots,
  finite cash, commission, transfer fee, stamp tax and slippage.
- The tree/sequence ensemble is a fixed 50/50 average of daily percentile
  ranks. Its weight is not searched on OOF data.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research minute-strategy-portfolio --signal-root runs/minute_ma_month_2022_04 --signal-root runs/minute_ma_month_2023_09 --output-root runs/minute_strategy_portfolio_apr_sep --selection-seed 7
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research verify
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research train-tree --features 158
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research evaluate-tree --features 158
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research train-tree --features 183
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research evaluate-tree --features 183
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research compare-tree
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research train-sequence
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research evaluate-sequence
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research evaluate-ensemble
```

`verify` performs the current lightweight quality gate, including complete-file
hashing, unique row keys, date bounds, finite/range checks and exact equality of
the 183 projection with its 557-field source matrix.

## Current interpretation

The 183-field tree does not provide a statistically reliable paired improvement
over the 158-field tree. Same-day 5-minute data remains an optional challenger,
not a required source. The raw sequence model adds a genuinely different rank
signal. Their fixed ensemble has the best current OOF Rank IC and uncapped
no-overlap account point estimates. Its IC increment is statistically clear
against the tree, but not against the raw sequence model; direct TopK and account
increments generally include zero. Winner-capped results are still weak. It is
a development benchmark, not evidence of stable profitability.
