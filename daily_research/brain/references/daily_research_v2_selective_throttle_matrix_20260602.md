# Daily Research V2 Selective Throttle Matrix - 2026-06-02

## Summary

- Status: `v2_selective_throttle_matrix / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_selective_throttle_matrix`.
- Matrix run tag: `v2_selective_throttle_matrix_narrow_20260602_01`.
- Probe run tag: `v2_selective_throttle_matrix_probe_20260602_01`.
- Source bridge: `v2_score_backtest_bridge_20260602_01`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_selective_throttle_matrix_probe_20260602_01`.
- `v2_selective_throttle_matrix_narrow_20260602_01`.
- `v2_score_backtest_bridge_20260602_01`.

## Implementation

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_selective_throttle_matrix --json`
  - `python -m daily_research.path_policy.v2_selective_throttle_matrix --run-backtests --json`
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_selective_throttle_matrix.py`.
- Mechanism:
  - read the v2 ensemble score panel from `v2_score_backtest_bridge_20260602_01`;
  - load explicit lake Close panel from `policy_input_bundle__45e3d8c059ba718426a9f887`;
  - compute historical close-derived recent return and volatility cross-sectional percentiles;
  - subtract a fixed score penalty from names in selected adverse state buckets;
  - submit the adjusted score panel to the existing shared external score backtest engine.
- No future labels are used by the throttle.
- No production target-weight panel is produced.

## Probe Evidence

- Run tag: `v2_selective_throttle_matrix_probe_20260602_01`.
- Variant:
  - `h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80`.
- Completed shared backtests: `1`.
- Promotion-review eligible variants: `0`.
- Metrics:
  - annual return `0.729223`;
  - excess annual return `0.418955`;
  - excess Sharpe `1.703522`;
  - max drawdown `-0.199176`;
  - positive month ratio `0.727273`;
  - negative months `3`;
  - worst monthly return `-0.032376`;
  - issue flags `[]`.
- Interpretation:
  - selective throttle materially improves Sharpe and drawdown versus the first score bridge / candidate matrix;
  - it still fails monthly positive ratio and negative month count, so it is not promotion-review eligible.

## Narrow Matrix Evidence

- Run tag: `v2_selective_throttle_matrix_narrow_20260602_01`.
- Matrix scope:
  - holding count `20`;
  - max weight `0.08`;
  - rebalance `5d`, all offsets;
  - costs transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`;
  - throttle modes `recent_runup`, `high_volatility`, `recent_runup_or_high_volatility`;
  - penalties `0.03`, `0.05`, `0.08`;
  - recent return window `5`;
  - volatility window `20`;
  - risk bucket threshold `0.80`.
- Variant count: `9`.
- Completed shared backtests: `9`.
- Promotion-review eligible variants: `0`.

## Best By Excess Sharpe

| variant | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | worst_month | issue_flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `h20_mw080_rb5d_all_c3_7_10_thr_high_volatility_p0p05_rrw5_vw20_rq80_vq80` | `1.953831` | `0.472816` | `-0.195026` | `0.636364` | `4` | `-0.028111` | `concentrated_positive_months` |
| `h20_mw080_rb5d_all_c3_7_10_thr_high_volatility_p0p08_rrw5_vw20_rq80_vq80` | `1.856103` | `0.446106` | `-0.191648` | `0.636364` | `4` | `-0.028167` | none |
| `h20_mw080_rb5d_all_c3_7_10_thr_high_volatility_p0p03_rrw5_vw20_rq80_vq80` | `1.751459` | `0.441541` | `-0.233438` | `0.636364` | `4` | `-0.059911` | `deep_bad_month`, `concentrated_positive_months` |
| `h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80` | `1.703522` | `0.418955` | `-0.199176` | `0.727273` | `3` | `-0.032376` | none |

## Interpretation

- Selective historical state throttle is more promising than the coarse market-regime overlay.
- `high_volatility` throttle produces the best excess Sharpe and drawdown profile, but monthly positive ratio remains `0.636364` and negative months remain `4`.
- `recent_runup_or_high_volatility` with penalty `0.05` is the best monthly-stability compromise: it removes deep bad-month flags and raises positive month ratio to `0.727273`, but still misses the `>=0.75` and `<=2` monthly gates.
- `recent_runup` alone underperforms and keeps deep bad-month / concentrated-positive-month flags.
- Current blocker has narrowed again: the candidate signal and selective risk-state adjustment have strong excess-return evidence, but the execution candidate still needs a month-state or sizing layer that repairs one additional negative month without destroying Sharpe.

## Boundaries

- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Do not treat adjusted score panels as production target-weight panels.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Next Allowed Actions

- Add a second-stage month-state or drawdown-state sizing layer on top of selective throttle.
- Test volatility throttle with lower max weight `0.06` and/or state-specific max weight only when portfolio drawdown or market breadth is adverse.
- Add post-throttle bad-month attribution for:
  - `high_volatility_p0p05`;
  - `recent_runup_or_high_volatility_p0p05`.
- Extend attribution dimensions to volatility bucket, recent-return bucket, sector, liquidity, and predicted horizon.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_selective_throttle_matrix.py -q`: `3 passed`.
- `python -m daily_research.path_policy.v2_selective_throttle_matrix --run-tag v2_selective_throttle_matrix_probe_20260602_01 --max-weights 0.08 --penalties 0.05 --recent-return-windows 5 --volatility-windows 20 --recent-return-quantiles 0.80 --volatility-quantiles 0.80 --run-backtests --json`: completed, `1` backtest completed, `0` promotion-review eligible variants.
- `python -m daily_research.path_policy.v2_selective_throttle_matrix --run-tag v2_selective_throttle_matrix_narrow_20260602_01 --max-weights 0.08 --throttle-modes recent_runup,high_volatility,recent_runup_or_high_volatility --penalties 0.03,0.05,0.08 --recent-return-windows 5 --volatility-windows 20 --recent-return-quantiles 0.80 --volatility-quantiles 0.80 --run-backtests --json`: completed, `9` backtests completed, `0` promotion-review eligible variants.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
