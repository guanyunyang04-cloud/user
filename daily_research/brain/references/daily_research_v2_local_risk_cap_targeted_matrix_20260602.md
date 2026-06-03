# Daily Research V2 Local Risk Cap Targeted Matrix - 2026-06-02

## Summary

- Status: `v2_local_risk_cap_targeted_matrix / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_local_risk_cap_matrix`.
- Targeted matrix run tag: `v2_local_risk_cap_matrix_targeted_20260602_01`.
- Source matrix: `v2_selective_throttle_matrix_narrow_20260602_01`.
- Source variant: `h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_local_risk_cap_matrix_targeted_20260602_01`.
- `v2_local_risk_cap_matrix_narrow_20260602_01`.
- `v2_selective_throttle_matrix_narrow_20260602_01`.

## Matrix Scope

- Cap modes:
  - `score_reversal`;
  - `score_volatility`.
- Stress scales:
  - `0.40`;
  - `0.60`.
- Score quantiles:
  - `0.70`;
  - `0.80`.
- Recent-return / volatility / reversal quantiles:
  - `0.80`.
- Recent return window:
  - `5`.
- Volatility window:
  - `20`.
- Redistribution modes:
  - `source_weight`;
  - `score`.
- Holding count `20`, max weight `0.08`, rebalance `5d` all offsets.
- Costs: transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`.

## Evidence

- Run tag: `v2_local_risk_cap_matrix_targeted_20260602_01`.
- Variant count: `16`.
- Completed shared backtests: `16`.
- Promotion-review eligible variants: `0`.
- All variants remain research-only and execution-frozen.

## Best By Excess Sharpe

| variant | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | worst_month | issue_flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `local_score_volatility_s0p4_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_score_h20_mw080` | `1.765417` | `0.395852` | `-0.215306` | `0.727273` | `3` | `-0.040946` | `concentrated_positive_months` |
| `local_score_volatility_s0p4_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_source_weight_h20_mw080` | `1.762332` | `0.396164` | `-0.215232` | `0.727273` | `3` | `-0.040262` | `concentrated_positive_months` |
| `local_score_volatility_s0p4_sq70_rq80_vq80_rvq80_rrw5_vw20_redist_score_h20_mw080` | `1.760900` | `0.394619` | `-0.215905` | `0.727273` | `3` | `-0.042201` | `concentrated_positive_months` |

## Best Worst-Month Shape

- Best worst-month variant:
  - `local_score_volatility_s0p4_sq80_rq80_vq80_rvq80_rrw5_vw20_redist_source_weight_h20_mw080`;
  - worst monthly excess return `-0.040262`;
  - excess Sharpe `1.762332`;
  - max drawdown `-0.215232`;
  - positive month ratio `0.727273`;
  - negative months `3`.

## Mode Comparison

- `score_volatility` dominates `score_reversal` in this targeted matrix:
  - best `score_volatility` excess Sharpe `1.765417`;
  - best `score_reversal` excess Sharpe `1.675362`;
  - best `score_volatility` worst month `-0.040262`;
  - best `score_reversal` worst month `-0.043450`.
- `score` redistribution slightly improves best excess Sharpe.
- `source_weight` redistribution gives the best worst-month shape.
- Lowering score threshold from `0.80` to `0.70` does not repair month stability and often weakens the candidate.

## Interpretation

- The second local matrix confirms the first local cap finding: simple overlay can improve risk-adjusted return shape but cannot close the hard monthly gate.
- The dominant local blocker is volatility-conditioned high-score exposure, not reversal-only exposure.
- The best targeted variant improves excess Sharpe versus the previous local-cap narrow matrix best (`1.765417` vs `1.711932`) and remains close to the strongest high-volatility selective-throttle branch, but the promotion-review blockers persist:
  - positive month ratio remains `0.727273`;
  - negative months remain `3`;
  - concentrated positive month flag remains.
- This is now strong evidence that target-weight overlays alone are approaching their ceiling for the current score panel.
- The next meaningful research step should move local volatility / adverse-state information upstream:
  - model input feature profile;
  - loss / output regularization against high-confidence adverse-local-state failures;
  - or a new score calibration layer trained/evaluated as part of research, not just an execution overlay.

## Next Allowed Actions

- Prefer model-side or score-calibration work over more simple target-weight overlays.
- Candidate next experiments:
  - add explicit local volatility / reversal / recent-return state to a v2 feature profile;
  - train a small score-calibration diagnostic that penalizes high-score high-local-volatility mistakes on validation windows;
  - post-targeted-cap attribution on remaining negative months to identify whether the last negative month is now benchmark/timing dominated.
- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Verification

- `python -m daily_research.path_policy.v2_local_risk_cap_matrix --run-tag v2_local_risk_cap_matrix_targeted_20260602_01 --cap-modes score_reversal,score_volatility --stress-scales 0.4,0.6 --score-quantiles 0.70,0.80 --recent-return-quantiles 0.80 --volatility-quantiles 0.80 --reversal-quantiles 0.80 --recent-return-windows 5 --volatility-windows 20 --redistribution-modes source_weight,score --run-backtests --json`: completed, `16` backtests completed, `0` promotion-review eligible variants.
