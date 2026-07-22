# Low-Corr Frontier Impact Stress Interpretation

## Context

- Run: `low_corr_frontier_impact_stress_20260602_174946`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Protocol: `horizon=20 / monthly / top_n=200 / buffer=3.0 / execution_constraints=True`
- Evaluation years: `2024, 2025, 2026`; final end date `2026-06-01`
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_ic_weighted_score`, `multifactor_low_corr_rank_score`
- Fixed fee: `30 bps`
- Capital stress: `10m, 50m, 100m`
- Participation impact stress: `0, 1, 2, 5, 10 bps per 1 pct participation`

## Key Results

At `100m` capital and the heaviest tested impact setting, `10 bps per 1 pct participation`:

| signal | mean annualized | min annualized | positive years | worst max drawdown | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multifactor_rolling_ic_weighted_score` | `0.307570` | `0.033603` | `3/3` | `-0.102373` | `16` |
| `multifactor_ic_weighted_score` | `0.244139` | `0.081845` | `3/3` | `-0.091689` | `18` |
| `multifactor_low_corr_rank_score` | `0.214914` | `0.045160` | `3/3` | `-0.131485` | `17` |

The ranking remains unchanged under impact stress:

1. `rolling_ic`
2. `ic_weighted`
3. `low_corr`

The impact drag is material but not thesis-breaking at this basket size and protocol:

- `rolling_ic`: 100m / impact `10` reduces mean annualized return by about `-0.042607`.
- `ic_weighted`: 100m / impact `10` reduces mean annualized return by about `-0.036841`.
- `low_corr`: 100m / impact `10` reduces mean annualized return by about `-0.047831`.

Liquidity/capacity proxy remains mixed:

- 2024 selected baskets include weaker liquidity tails: minimum selected-stock amount can be near `3.36m`, and worst max participation at `100m` reaches about `14.90%` for rolling/IC-weighted.
- 2025/2026 look less stressed by this proxy, with `100m` worst max participation mostly near `4% to 6%`.
- All three signals still carry negative average `log_amount_mean_20d_z` and negative `momentum_20d_z` exposure in selected baskets.

## Interpretation

This experiment strengthens the current candidate-frontier evidence: the frontier protocol does not collapse when adding a participation-based impact model on top of `30 bps` fixed fees.

It does not promote any signal to formal strategy candidate. The main reasons are unchanged:

- The monthly non-overlapping sample is still small: only `16-18` completed trades per signal across `2024-2026`.
- 2026 has only `2` completed monthly trades, so the apparent 2026 robustness is thin.
- The protocol was discovered after prior horizon/cost/grid exploration, so selection bias remains.
- The baskets still lean toward lower liquidity and weak momentum exposures.
- The impact model is still a proxy. It does not model queue priority, partial fills, intraday volume curve, order slicing, market impact decay, or live borrow/cash drag.

## Decision

- Status: `candidate-frontier/backtest_only`
- Formal strategy candidate count: `0`
- Current frontier set remains:
  - `multifactor_rolling_ic_weighted_score`
  - `multifactor_ic_weighted_score`
  - `multifactor_low_corr_rank_score`

## Next Gate

The next priority should be industry/size neutralization and exposure-stability audit under the same fixed protocol. Impact stress did not reject the frontier set; the remaining risk is that the apparent alpha is still a compensated or unstable exposure bundle rather than a robust stock-selection signal.
