# Alpha Multi-Horizon Target Function Shadow Seed7 - 2026-05-24

## Scope
- Status: `research / shadow-only / target-function control / single-seed`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool view: `policy_pool_view__c11400fa72ad263f3d1eecfa` (`rolling_liquid500`).
- Sector/board view: `policy_sector_board_view__ed15b2873f544e9e9b24aae5`.
- Feature profile: `raw_kline_context_no_alpha_prior_v1`.
- Model family: `gru_sequence_static_context`; seed `7`.
- Role years: train `2019-2022`, validation `2023`, test `2024`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Artifacts
- `daily_research/output/path_policy/studies/mh_short_utility_1_3_5d_v1/`
- `daily_research/output/path_policy/studies/mh_mid_utility_5_10_20d_v1/`
- `daily_research/output/path_policy/studies/mh_long_utility_15_20_30d_v1/`
- `daily_research/output/path_policy/studies/mh_target_function_shadow_seed7_comparison_20260524_01/target_function_shadow_seed7_comparison.json`
- `daily_research/output/path_policy/studies/mh_target_function_shadow_seed7_comparison_20260524_01/target_function_shadow_seed7_comparison.csv`

## Verdict
- Fact: all three shadow-only seed7 target-function runs completed and produced study summaries, validation/test predictions, target calibration audits, and decision score diagnostics.
- Fact: `mh_short_utility_1_3_5d_v1` is weak. It stopped at epoch `8`, best epoch `1`; validation decision rank IC `0.013986`, spread `0.005116`; test decision rank IC `0.029569`, spread `0.005598`, but test decision utility profile failed and hit lift was negative.
- Fact: `mh_mid_utility_5_10_20d_v1` is test-confirmed. It stopped at epoch `8`, best epoch `2`; validation decision rank IC `0.086660`, spread `0.010785`; test decision rank IC `0.036099`, spread `0.010266`; test decision utility profile passed.
- Fact: `mh_long_utility_15_20_30d_v1` is the strongest single-seed candidate. It stopped at epoch `9`, best epoch `3`; validation decision rank IC `0.094242`, spread `0.025083`; test decision rank IC `0.095089`, spread `0.041296`; validation/test decision utility profiles passed.
- Fact: diagnostics differ by target family: short and mid report `no_selection_only_fix_found`; long reports `selection_calibration_candidate_found` with `forecast_20d_mu`.
- Inference: this result supports `long_continue` as the primary next research branch and `mid_continue` as a secondary candidate. It does not support continuing the current short target without a separate short-horizon data/input redesign.
- Boundary: this single seed does not authorize production retrain, live/default, active manifest edits, production root edits, allocator/replay, trade-plan bridge, paper account bridge, or broker actions.

## Key Evidence
- Short target:
  - Validation monthly positive rate `91.7%`, negative months `2023-01`, but spread magnitude is small.
  - Test monthly positive rate was not enough to overcome failed decision utility profile and negative hit lift.
  - Predicted horizon distribution is dominated by `1d`, while future best horizons remain distributed across `1/3/5d`.
- Mid target:
  - Validation horizons strengthen with length: `5d` rank IC `0.089328`, `10d` `0.115538`, `20d` `0.141368`.
  - Test horizons remain positive: `5d` rank IC `0.044457`, `10d` `0.064485`, `20d` `0.081251`.
  - Validation monthly positive rate `72.7%`, with negative months `2023-01`, `2023-03`, `2023-09`.
- Long target:
  - Validation: `15d` rank IC `0.093867`, `20d` `0.104622`, `30d` `0.107512`.
  - Test: `15d` rank IC `0.081138`, `20d` `0.085284`, `30d` `0.100722`.
  - Test decision spread `0.041296` is materially higher than mid `0.010266` and short `0.005598`.
  - Validation monthly positive rate `81.8%`, negative months `2023-03`, `2023-09`.

## Interpretation
- Fact: current daily input and GRU static-context setup can learn much stronger `5-30d` utility than `1-5d` utility.
- Inference: short horizon is not disproven forever, but the current target/input combination is not the right route for a robust short-line edge.
- Inference: mid target is viable but less stable than long on this seed because validation has three negative months and no selection-only diagnostic candidate.
- Inference: long target currently best matches the observed root-cause audit: real long-horizon edge plus remaining selection-calibration work.
- Inference: because best epochs are early (`1/2/3`), the next step should include multi-seed stability before any architecture expansion.

## Next Allowed Actions
- Run multi-seed confirmation for the long target first: `mh_long_utility_15_20_30d_v1` with seeds `7,11,19`.
- Optionally keep mid as a secondary multi-seed candidate if compute budget allows.
- Do not run architecture comparison until at least one target family passes multi-seed stability.
- Do not connect these artifacts to execution, trade plan, paper account, live/default, or production root.

## Validation
- `target_calibration_audit.py` completed for all three tags.
- `decision_score_diagnostics.py` completed for all three tags.
- `daily_research/output/active_execution_strategy.json` remained unchanged during this run.
