# Minute-MA selection-seed stability

This record tests whether the first finite-cash minute-MA results survive the
arbitrary choice among many stocks signaling in the same minute. It uses the
same April 2022 and September 2023 full-universe event outputs as the initial
portfolio record. It does not read 2025 or change any strategy or exit rule.

## Contract

The fixed seeds `0` through `9` were declared before reading the results. All
ten replays share one signal and minute-bar scan. Earlier signal minutes retain
priority; within one strategy and minute, each seed applies the same
deterministic hash tie-break to the causal candidate set. Cash, five-position
capacity, fees, slippage, T+1 and exit behavior are otherwise identical.

The study contains 22 strategies, seven exit policies, 154 account variants
and 1,540 seed/account results over 39 signal dates. Seed 7 exactly reproduces
the preceding single-seed run, confirming that the shared scan did not change
its account semantics.

## Result

The random tie-break is not stable enough to support strategy selection:

- All 154 variants have negative mean and negative median return across seeds.
- No variant is profitable in at least half of the seeds. The highest positive
  count is 3/10.
- The best median is `ng_r1_auction_reclaim + next_open_1d`: median -3.54%,
  mean -2.94%, range -8.64% to +3.01%, with all ten accounts fully resolved.
- This least-negative variant loses in every April 2022 replay, while eight of
  ten September 2023 replays are positive. It is regime-dependent rather than
  a stable combined strategy.
- The seed-7 leader `s1_touch_reclaim + next_open_3d` is positive only at one
  seed. Its ten-seed median is -12.44%, mean -12.64%, and range -24.27% to
  +8.57%.
- The previously observed two-segment-positive marked account,
  `s4_reclaim_volume_normal + fast_failure_2pct_target4pct_2d`, is positive at
  only one seed; that seed is unresolved. Its ten-seed median is -17.46%.
- Across each complete seed replay, only 4 to 15 of 154 variants are positive,
  and the cross-variant median return ranges from -12.51% to -16.70%.

Fifteen variants happen to be positive in both sample periods for at least one
fully resolved seed, but none does so for more than two of ten seeds. Searching
for the favorable seed would therefore be another form of parameter fitting.

The conclusion is narrow: the current event rules cannot be evaluated as a
five-position portfolio using an arbitrary same-minute choice. It does not
prove that every MA event has no predictive information. The next required
component is a causal cross-sectional ranking rule, evaluated across more
2022-2024 market regimes before 2025 is opened.

## Reproduction

Detailed local results are under
`runs/minute_strategy_portfolio_seed_stability/`, including
`selection_seed_summary.json`, the 1,540 account rows, equity paths and trades.
The shared run completed in 2,396.797 seconds without a memory-floor exception.

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research minute-strategy-portfolio `
  --workspace-root H:\quant_project `
  --signal-root runs/minute_ma_month_2022_04 `
  --signal-root runs/minute_ma_month_2023_09 `
  --output-root runs/minute_strategy_portfolio_seed_stability `
  --starting-cash 100000 --max-positions 5 `
  --selection-seed 0 --selection-seed 1 --selection-seed 2 `
  --selection-seed 3 --selection-seed 4 --selection-seed 5 `
  --selection-seed 6 --selection-seed 7 --selection-seed 8 `
  --selection-seed 9 `
  --memory-floor-gib 0.5 --soft-memory-floor-gib 1.0 `
  --duckdb-threads 2 --duckdb-memory-floor-gib 2.0
```
