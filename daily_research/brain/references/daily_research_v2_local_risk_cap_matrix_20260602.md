# Daily Research V2 Local Risk Cap Matrix - 2026-06-02

## Summary

- Status: `v2_local_risk_cap_matrix / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_local_risk_cap_matrix`.
- Dry-run tag: `v2_local_risk_cap_matrix_dryrun_20260602_01`.
- Narrow matrix run tag: `v2_local_risk_cap_matrix_narrow_20260602_01`.
- Source matrix: `v2_selective_throttle_matrix_narrow_20260602_01`.
- Source variant: `h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_local_risk_cap_matrix_dryrun_20260602_01`.
- `v2_local_risk_cap_matrix_narrow_20260602_01`.
- `v2_selective_throttle_matrix_narrow_20260602_01`.

## Implementation

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_local_risk_cap_matrix --json`.
  - `python -m daily_research.path_policy.v2_local_risk_cap_matrix --run-backtests --json`.
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_local_risk_cap_matrix.py`.
- Mechanism:
  - read source candidate `aligned_daily_target_weight_panel.csv`;
  - read source candidate `aligned_daily_score_panel.csv`;
  - load explicit lake Close panel for the same dates and symbols;
  - compute historical local state from previous close-derived recent return, volatility, and reversal;
  - reduce selected held names under adverse local score/state buckets;
  - redistribute released weight to same-day selected names that did not trigger the local risk mask, respecting `max_weight`;
  - submit adjusted direct target-weight panels to the shared backtest engine.
- Boundary:
  - this is a candidate overlay diagnostic;
  - it is not production promotion;
  - it does not write active execution artifacts.

## Dry Run Evidence

- Run tag: `v2_local_risk_cap_matrix_dryrun_20260602_01`.
- Variant:
  - `local_score_volatility_or_reversal_s0p6_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_source_weight_h20_mw080`.
- Status: completed.
- Active held-cell risk mask ratio: `0.240969`.
- Mean source gross: `0.900474`.
- Mean adjusted gross: `0.900474`.
- Mean released weight: `0.085519`.
- Max adjusted weight: `0.080000`.
- Interpretation: local cap and redistribution work on real v2 artifacts without silently shrinking gross exposure or violating max-weight.

## Narrow Matrix Evidence

- Run tag: `v2_local_risk_cap_matrix_narrow_20260602_01`.
- Matrix scope:
  - cap modes `score_volatility_or_reversal`, `score_runup_or_volatility_or_reversal`;
  - stress scales `0.40`, `0.60`;
  - score / recent-return / volatility / reversal quantiles all `0.80`;
  - recent return window `5`;
  - volatility window `20`;
  - redistribution mode `source_weight`;
  - holding count `20`, max weight `0.08`, rebalance `5d` all offsets;
  - costs transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`.
- Variant count: `4`.
- Completed shared backtests: `4`.
- Promotion-review eligible variants: `0`.

## Best Variants

| variant | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | worst_month | issue_flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `local_score_volatility_or_reversal_s0p6_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_source_weight_h20_mw080` | `1.711932` | `0.384959` | `-0.217768` | `0.727273` | `3` | `-0.039201` | `concentrated_positive_months` |
| `local_score_volatility_or_reversal_s0p4_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_source_weight_h20_mw080` | `1.693032` | `0.379861` | `-0.215009` | `0.727273` | `3` | `-0.038107` | `concentrated_positive_months` |
| `local_score_runup_or_volatility_or_reversal_s0p6_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_source_weight_h20_mw080` | `1.686561` | `0.379055` | `-0.220016` | `0.727273` | `3` | `-0.042464` | `concentrated_positive_months` |

## Interpretation

- Local cap is a useful but insufficient repair.
- It preserves most of the selective-throttle candidate quality:
  - excess Sharpe stays around `1.65-1.71`;
  - max drawdown stays around `-0.215` to `-0.220`;
  - deep bad month remains absent.
- It still fails the promotion-review gate:
  - positive month ratio remains `0.727273`;
  - negative months remain `3`;
  - concentrated positive month flag remains.
- Compared with the source monthly compromise candidate, local caps do not close the final monthly-stability gap.
- Compared with the best high-vol-only selective throttle candidate, local caps improve robustness shape but do not reach the same excess Sharpe.
- Current conclusion:
  - local state control should remain in the toolbox;
  - the missing piece is likely conditional/local by month state or feature/model-side improvement, not another simple overlay.

## Next Allowed Actions

- Run a targeted second local matrix with:
  - lower score quantile threshold `0.70`;
  - separate `score_reversal` and `score_volatility` instead of combined masks;
  - `redistribution_mode=score` versus `source_weight`;
  - optional month-state conditional cap only after previous-month weak state;
  - post-local-cap attribution on the remaining negative months.
- Consider returning to model-input work:
  - add local reversal / volatility / recent-return state as explicit model input;
  - inspect whether loss/output can penalize high-confidence adverse-local-state failures.
- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_local_risk_cap_matrix.py -q`: `3 passed`.
- `python -m daily_research.path_policy.v2_local_risk_cap_matrix --run-tag v2_local_risk_cap_matrix_dryrun_20260602_01 --cap-modes score_volatility_or_reversal --stress-scales 0.6 --score-quantiles 0.80 --recent-return-quantiles 0.80 --volatility-quantiles 0.80 --reversal-quantiles 0.80 --recent-return-windows 5 --volatility-windows 20 --redistribution-modes source_weight --json`: completed.
- `python -m daily_research.path_policy.v2_local_risk_cap_matrix --run-tag v2_local_risk_cap_matrix_narrow_20260602_01 --cap-modes score_volatility_or_reversal,score_runup_or_volatility_or_reversal --stress-scales 0.4,0.6 --score-quantiles 0.80 --recent-return-quantiles 0.80 --volatility-quantiles 0.80 --reversal-quantiles 0.80 --recent-return-windows 5 --volatility-windows 20 --redistribution-modes source_weight --run-backtests --json`: completed, `4` backtests completed, `0` promotion-review eligible variants.
