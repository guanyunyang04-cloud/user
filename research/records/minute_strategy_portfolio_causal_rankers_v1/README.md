# Minute-MA causal cross-sectional rankers

This record evaluates causal rules for choosing among several executable
signals that arrive in the same minute. It follows the rejection of the
unranked fixed-seed portfolio path and uses the same finite-cash account
contract. It is a development study only: no rule is frozen, 2025 is not read,
and the result is not evidence for live or QMT trading.

## Scope

- Signal outputs: the full point-in-time main-board studies for April 2022 and
  September 2023.
- Trading dates: 39 (19 plus 20).
- Strategies: 21 executable strategies. The old
  `s4_reclaim_vwap_support` strategy is excluded because its retained signal
  output was produced with the pre-correction VWAP field; the corrected data
  contract is not retroactively mixed into this replay.
- Exit rules: seven catalog policies.
- Starting cash: CNY 100,000; maximum five simultaneous symbols; 100-share
  lots; finite fees, slippage, T+1 and legal delayed exits.
- Rankers: `sector_leader`, `trend_structure`, and `flow_quality`.
- Seeds: fixed seeds 0 through 4. A seed is only a deterministic tie-break
  among equal rank scores; it is not a fitted parameter.

The replay processed all rankers and seeds in one shared minute-bar scan. A
signal is ranked using fields available at its signal minute. Earlier signal
minutes retain priority, and an unaffordable or unfilled selected order is not
replaced by a lower-ranked name.

## Ranker contract

`sector_leader` averages the clipped point-in-time sector strength percentile,
sector breadth, and the same-minute candidate percentile of
`leader_relative_return`.

`trend_structure` averages the causal flags `recent_high_breakout`,
`prior_acceleration`, and `daily_trend_positive`.

`flow_quality` averages `volume_normal`, `not_repeated_cross`,
`auction_confirmed`, and one positive-flow flag (positive
`amount_curve_surprise` or `volume_acceleration_5_20`). Missing numeric values
score low and missing boolean flags score false. The deterministic hash is
used only to break equal scores, followed by `signal_id` for reproducibility.

These fields are calculated from current or prior bars and point-in-time daily
context. Forward returns, exit status, affordability, and any future label are
not inputs to a rank score. The score is retained in the trade and equity
artifacts for audit.

## Results

Each ranker has 147 strategy/exit account groups (21 x 7), five seed replays,
and 735 account results. The fair random comparison uses the existing
unranked replay, restricted to the same 21 strategies and seeds 0 through 4.
The comparison is made per strategy/exit group using the median return across
seeds, then summarized across the 147 groups.

| Selection rule | Result mean | Result median | Median of group seed medians | Groups improved vs random | Groups worse vs random | Groups positive in >=3/5 seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_hash` baseline | -14.10% | -14.16% | -13.93% | -- | -- | 0/147 |
| `sector_leader` | -20.26% | -21.04% | -21.04% | 43/147 | 104/147 | 4/147 |
| `trend_structure` | -15.15% | -15.87% | -16.01% | 66/147 | 81/147 | 1/147 |
| `flow_quality` | -13.49% | -13.86% | -13.90% | 68/147 | 61/147 | 1/147 |

The rankers therefore do not provide a general profitable increment in these
two sample months. `flow_quality` is closest to the random baseline overall;
`sector_leader` has the largest number of losing groups despite producing one
promising candidate.

The best observed candidate is:

`sector_leader + s4_auction_confirmed_reclaim + next_open_2d`

- combined return: **+7.80%**;
- 2022 segment: **+6.33%**;
- 2023 segment: **+1.38%**;
- maximum drawdown: **-9.25%**;
- 100 filled and 100 closed trades;
- nine delayed exit attempts, no unresolved position;
- identical result for all five seeds.

For the same strategy and exit rule, the random baseline's five-seed median is
-10.05% (mean -9.56%). The candidate was found after comparing 441 account
groups (three rankers x 21 strategies x seven exits), so it is a hypothesis
for more development data, not a selected or frozen strategy.

## Decision

The causal ranking implementation is accepted as an auditable portfolio
component. The two-month result is **not sufficient to select a ranker or
strategy**. Keep all three rankers and the random baseline as controls, expand
the same fixed contract across additional 2022--2024 months and market
regimes, and record every comparison. Do not open 2025 until the development
window and rule family are frozen.

## Reproduction

Generated account rows and minute paths are retained outside Git under
`runs/minute_strategy_portfolio_causal_rankers/`. The source manifest in that
directory records the exact strategy and policy lists, data configuration,
rankers, seeds, and elapsed time (`2953.473` seconds). A representative command
is:

```powershell
$env:PYTHONPATH='H:\\quant_project\\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research minute-strategy-portfolio `
  --workspace-root H:\\quant_project `
  --signal-root runs/minute_ma_month_2022_04 `
  --signal-root runs/minute_ma_month_2023_09 `
  --output-root runs/minute_strategy_portfolio_causal_rankers `
  --selection-ranker sector_leader --selection-ranker trend_structure `
  --selection-ranker flow_quality `
  --selection-seed 0 --selection-seed 1 --selection-seed 2 `
  --selection-seed 3 --selection-seed 4 `
  --starting-cash 100000 --max-positions 5 `
  --memory-floor-gib 0.5 --soft-memory-floor-gib 1.0 `
  --duckdb-threads 2 --duckdb-memory-floor-gib 2.0
```

The command must also pass the 21 explicit `--strategy-id` values listed in
`run_manifest.json` when exact reproduction is required; omitting them allows
the CLI to infer the older excluded VWAP strategy from the input files.

