# Daily Research V2 Score State Calibration Matrix - 2026-06-02

## Summary

- Status: `v2_score_state_calibration_matrix / validation_selected_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_score_state_calibration_matrix`.
- Dry-run tag: `v2_score_state_calibration_matrix_dryrun_20260602_01`.
- Probe tag: `v2_score_state_calibration_matrix_probe_20260602_01`.
- Source test bridge: `v2_score_backtest_bridge_20260602_01`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_amount_checked`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Implementation

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_score_state_calibration_matrix`.
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_score_state_calibration_matrix.py`.
- Mechanism:
  - build/read v2 validation and test ensemble score panels;
  - load explicit lake Close panel;
  - compute previous-close-derived local recent return, local volatility, and reversal state;
  - select score penalty variants on validation labels;
  - apply the selected variants to the test score panel;
  - submit selected adjusted test panels to the shared candidate backtest engine when requested.
- Boundary:
  - validation labels are used only for selecting calibration variants;
  - test backtests remain research candidates;
  - no production target-weight panel is produced.

## Dry-Run Evidence

- Run tag: `v2_score_state_calibration_matrix_dryrun_20260602_01`.
- Variant count: `4`.
- Selected test variants: `2`.
- Test backtests: `0`.
- Best validation-selected variants:
  - `cal_score_volatility_p0p02_sq80_rq80_vq80_rvq80_rrw5_vw20_h20_mw080`;
  - `cal_score_volatility_or_reversal_p0p02_sq80_rq80_vq80_rvq80_rrw5_vw20_h20_mw080`.
- Validation selection favored small penalty `0.02` over `0.05`.
- `score_volatility` was slightly better than `score_volatility_or_reversal` by validation selection score.

## Probe Evidence

- Run tag: `v2_score_state_calibration_matrix_probe_20260602_01`.
- Variant count: `4`.
- Completed shared backtests: `2`.
- Promotion-review eligible variants: `0`.

| variant | validation_score | excess_sharpe | excess_annual_return | max_drawdown | positive_month_ratio | negative_months | issue_flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cal_score_volatility_p0p02_sq80_rq80_vq80_rvq80_rrw5_vw20_h20_mw080` | `0.263656` | `1.847919` | `0.476215` | `-0.243706` | `0.636364` | `4` | `deep_bad_month`, `concentrated_positive_months` |
| `cal_score_volatility_or_reversal_p0p02_sq80_rq80_vq80_rvq80_rrw5_vw20_h20_mw080` | `0.262645` | `1.571568` | `0.402930` | `-0.251771` | `0.636364` | `4` | `deep_bad_month`, `concentrated_positive_months` |

## Interpretation

- Validation-selected local-state score calibration is a cleaner experimental protocol than directly tuning overlays on test, and the CLI now provides that protocol.
- The first real probe does not close the current promotion-review blocker:
  - positive month ratio remains `0.636364`;
  - negative months remain `4`;
  - `deep_bad_month` and `concentrated_positive_months` flags remain.
- The best probe has strong excess Sharpe (`1.847919`) and stays within the max drawdown gate (`-0.243706`), but it is worse than the best simple high-volatility selective throttle by excess Sharpe (`1.953831`) and worse than the best monthly-stability compromise by month count (`4` vs `3` negative months).
- The result reinforces the targeted local-cap conclusion: simple post-hoc or shallow score penalties are not enough. Local volatility / adverse-state information should move further upstream into:
  - explicit model input features;
  - score calibration trained with a richer validation objective;
  - loss/output regularization against high-score adverse-state failures.

## Next Allowed Actions

- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Prefer the next experiment to be model-side or loss/output-side rather than another simple fixed-penalty overlay.
- Candidate next steps:
  - add local volatility / reversal / recent-return features to a v2 input profile;
  - train a lightweight calibration head or calibration model with validation-only selection and no test tuning;
  - add loss regularization for high-score adverse-local-state false positives;
  - run post-calibration bad-month attribution if a richer calibration candidate improves monthly stability.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_score_state_calibration_matrix.py -q`: `3 passed`.
- `python -m daily_research.path_policy.v2_score_state_calibration_matrix --run-tag v2_score_state_calibration_matrix_dryrun_20260602_01 --calibration-modes score_volatility,score_volatility_or_reversal --penalties 0.02,0.05 --top-n-test-backtests 2`: completed.
- `python -m daily_research.path_policy.v2_score_state_calibration_matrix --run-tag v2_score_state_calibration_matrix_probe_20260602_01 --calibration-modes score_volatility,score_volatility_or_reversal --penalties 0.02,0.05 --top-n-test-backtests 2 --run-backtests`: completed, `2` backtests completed, `0` promotion-review eligible variants.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
