# Daily Research V2 State Sizing Matrix - 2026-06-02

## Summary

- Status: `v2_state_sizing_matrix / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_state_sizing_matrix`.
- Narrow matrix run tag: `v2_state_sizing_matrix_narrow_20260602_01`.
- Light probe run tag: `v2_state_sizing_matrix_light_probe_20260602_01`.
- Source matrix: `v2_selective_throttle_matrix_narrow_20260602_01`.
- Source variant: `h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_state_sizing_matrix_dryrun_20260602_01`.
- `v2_state_sizing_matrix_narrow_20260602_01`.
- `v2_state_sizing_matrix_light_probe_20260602_01`.
- `v2_selective_throttle_matrix_narrow_20260602_01`.

## Implementation

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_state_sizing_matrix --json`
  - `python -m daily_research.path_policy.v2_state_sizing_matrix --run-backtests --json`
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_state_sizing_matrix.py`.
- Mechanism:
  - read an already completed candidate backtest's `aligned_daily_target_weight_panel.csv`;
  - read the same candidate's `aligned_daily_score_panel.csv` as score context;
  - read `equity_curve.csv` and compute path state from past `excess_equity`;
  - scale gross target weights under drawdown / previous-month-loss states;
  - submit scaled direct target-weight panels to the shared backtest engine.
- Boundary:
  - this is a second-stage candidate overlay diagnostic;
  - it is not production promotion;
  - it does not write active execution artifacts.

## Narrow Matrix Evidence

- Run tag: `v2_state_sizing_matrix_narrow_20260602_01`.
- Source variant: `recent_runup_or_high_volatility p0.05`, the best monthly-stability compromise from selective throttle.
- Matrix scope:
  - scale modes `drawdown`, `prev_month_loss`, `drawdown_or_prev_month_loss`;
  - stress scales `0.50`, `0.70`;
  - drawdown thresholds `0.03`, `0.05`;
  - monthly loss threshold `0.01`;
  - holding count `20`, max weight `0.08`, rebalance `5d` all offsets;
  - costs transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`.
- Variant count: `12`.
- Completed shared backtests: `12`.
- Promotion-review eligible variants: `0`.

## Best Narrow Matrix Variants

| variant | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | worst_month | issue_flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `state_drawdown_s0p7_dd0p05_ml0p01_h20_mw080_rb5d_all` | `1.321736` | `0.260825` | `-0.218515` | `0.727273` | `3` | `-0.042780` | `concentrated_positive_months` |
| `state_prev_month_loss_s0p7_dd0p05_ml0p01_h20_mw080_rb5d_all` | `1.109632` | `0.247501` | `-0.224203` | `0.727273` | `3` | `-0.106054` | `deep_bad_month`, `concentrated_positive_months` |
| `state_drawdown_s0p5_dd0p05_ml0p01_h20_mw080_rb5d_all` | `0.984565` | `0.180053` | `-0.215749` | `0.636364` | `4` | `-0.043038` | `concentrated_positive_months` |

## Light Probe Evidence

- Run tag: `v2_state_sizing_matrix_light_probe_20260602_01`.
- Scope:
  - scale modes `drawdown`, `drawdown_or_prev_month_loss`;
  - stress scales `0.85`, `0.90`;
  - drawdown thresholds `0.05`, `0.08`;
  - monthly loss threshold `0.01`.
- Variant count: `8`.
- Completed shared backtests: `8`.
- Promotion-review eligible variants: `0`.
- Best by excess Sharpe:
  - variant `state_drawdown_s0p9_dd0p08_ml0p01_h20_mw080_rb5d_all`;
  - annual return `0.653285`;
  - excess annual return `0.356642`;
  - excess Sharpe `1.614474`;
  - max drawdown `-0.228743`;
  - positive month ratio `0.727273`;
  - negative months `3`;
  - worst monthly return `-0.042424`;
  - issue flags `concentrated_positive_months`.

## Interpretation

- Global path-state sizing is not the missing piece for the current v2 candidate.
- Aggressive drawdown / previous-month-loss scaling reduces exposure and cost but usually destroys Sharpe and can worsen monthly stability.
- Light drawdown scaling preserves much of the selective-throttle quality, but it still does not repair the hard monthly gate: positive month ratio remains `0.727273` and negative months remain `3`.
- The key blocker is now narrower: the model score and selective volatility/runup throttle are useful, but the remaining negative month is not solved by whole-portfolio lagged exposure control.
- Next work should move from global path-state sizing to local/state-specific risk control:
  - per-symbol reversal / volatility bucket sizing;
  - sector / liquidity / volatility concentration caps;
  - predicted-horizon bucket caps;
  - post-throttle bad-month attribution on the remaining negative months;
  - weight redistribution rather than simple gross exposure shrinkage.

## Boundaries

- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Do not treat scaled target-weight panels as production target-weight panels.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_state_sizing_matrix.py -q`: `4 passed`.
- `python -m daily_research.path_policy.v2_state_sizing_matrix --run-tag v2_state_sizing_matrix_dryrun_20260602_01 --scale-modes drawdown --stress-scales 0.5 --drawdown-thresholds 0.03 --monthly-loss-thresholds 0.01 --json`: completed dry-run.
- `python -m daily_research.path_policy.v2_state_sizing_matrix --run-tag v2_state_sizing_matrix_narrow_20260602_01 --source-matrix-run-tag v2_selective_throttle_matrix_narrow_20260602_01 --source-variant-id h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80 --scale-modes drawdown,prev_month_loss,drawdown_or_prev_month_loss --stress-scales 0.5,0.7 --drawdown-thresholds 0.03,0.05 --monthly-loss-thresholds 0.01 --run-backtests --json`: completed, `12` backtests completed, `0` promotion-review eligible variants.
- `python -m daily_research.path_policy.v2_state_sizing_matrix --run-tag v2_state_sizing_matrix_light_probe_20260602_01 --source-matrix-run-tag v2_selective_throttle_matrix_narrow_20260602_01 --source-variant-id h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80 --scale-modes drawdown,drawdown_or_prev_month_loss --stress-scales 0.85,0.90 --drawdown-thresholds 0.05,0.08 --monthly-loss-thresholds 0.01 --run-backtests --json`: completed, `8` backtests completed, `0` promotion-review eligible variants.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
