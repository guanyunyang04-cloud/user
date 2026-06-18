# Path Policy Alpha V2 Time-Efficient TopK Training Contract Implementation 20260618

## Verdict
- Status: `code_contract_implemented / focused_tests_passed / research_only / execution_frozen`.
- Date: `2026-06-18`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `qdp_alpha_v2_time_efficient_topk_research`.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` was not touched.
- This is code-level training-contract enablement, not a full alpha_v2 model run, not multi-seed evidence, not score-backtest bridge, not candidate matrix, and not execution-candidate evidence.

## Implemented
The path_policy forecast training stack now supports a clean time-efficient personal top-K objective:

```text
personal_time_efficient_topk_v1
future_score = max_h((future_cumulative_excess_return_h - cost - path_risk_penalty_h) / h)
horizons = existing cumulative horizons, typically 1/3/5/10/20
```

Code changes:
- Added `personal_time_efficient_topk_v1` to `FORECAST_LOSS_PROFILES` and decision-output loss contracts.
- Added `validation_loss` to `FORECAST_SELECTION_PROFILES`.
- Added a dedicated torch target helper and numpy target helper for time-efficient utility.
- Added a dedicated batch top1/top3/top5 alignment proxy against `future_time_eff_score`; it does not reuse the old 20d high-return proxy.
- Reused the existing `decision_aux` output layout, but changed its semantic contract for this profile:
  - first block = predicted time-efficient utility by horizon;
  - second block = hit logits by horizon;
  - third block = best-horizon logits.
- Added prediction columns:
  - `pred_time_eff_utility_{h}d`
  - `future_time_eff_utility_{h}d`
  - `future_time_eff_net_utility_{h}d`
  - `pred_time_eff_score`
  - `future_time_eff_score`
  - `pred_time_eff_best_horizon`
  - `future_time_eff_best_horizon`
- For this profile, `pred_decision_score`, `future_decision_score`, and `trade_utility_score` intentionally point to the time-efficient score so existing diagnostics remain readable.
- Added forecast metrics for:
  - `time_eff_score_rank_ic`
  - `time_eff_score_top_bottom_spread`
  - `time_eff_best_horizon_accuracy`
  - `time_eff_hit_lift_top20_mean`
- Fixed an existing validation-frame mismatch: epoch-level validation prediction frames now pass `loss_profile`, so profile-specific score calibration / semantics match the active training contract.
- Added validation-loss checkpoint selection support:
  - epoch best uses `score = -validation_loss` when `selection_profile=validation_loss`;
  - family / seed selection also uses `validation_loss_score`;
  - summary selection rule becomes `lowest_validation_loss_for_training_loss_profile_then_seed_score`.
- Protocol parser accepts:

```text
--forecast-loss-profile personal_time_efficient_topk_v1
--forecast-selection-profile validation_loss
```

and auto-selects `output_profile=decision_utility_v1` from the loss contract.

## Semantics
The new contract directly addresses the user's objection that cumulative-return utility can over-prefer longer horizons:

```text
old approximate utility:
  cumulative_excess_return_h - cost - risk_penalty

new v1 utility:
  (cumulative_excess_return_h - cost - risk_penalty) / h
```

This means:
- `1d +1%` and `20d +20%` are comparable on capital-time efficiency before risk/cost details.
- Fixed cost is subtracted before `/h`, so short horizon churn is naturally penalized.
- Long horizon can still win if its cost/risk-adjusted per-day utility is stronger.
- Checkpoint selection can now follow the declared training objective through validation loss.

## Verification
Commands passed:

```text
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_forecast_training.py::test_auxiliary_decision_loss_profiles_record_weight_contract_and_finite_loss daily_research/path_policy/tests/test_forecast_training.py::test_time_efficient_utility_targets_normalize_by_capital_time_and_cost daily_research/path_policy/tests/test_forecast_training.py::test_time_efficient_topk_profile_writes_prediction_columns_and_scores_metrics daily_research/path_policy/tests/test_forecast_training.py::test_train_forecast_models_selects_time_efficient_topk_by_validation_loss -q
4 passed

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_validation_loss_time_efficient_forecast_contract daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_auxiliary_decision_loss_profiles_and_sets_decision_output -q
2 passed

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_forecast_training.py -q
29 passed

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_decision_utility_forecast_contract daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_validation_loss_time_efficient_forecast_contract daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_auxiliary_decision_loss_profiles_and_sets_decision_output -q
3 passed

git diff --check
ok

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
ok
```

## Boundaries
- No alpha_v2 full training run was launched.
- No old checkpoint validation-loss audit was launched yet.
- No personal_topK checkpoint reselection was run.
- No multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default change.
- Component weights and time-efficient diagnostics are recorded; a full separate per-component validation-loss breakdown is still a useful hardening step before long full runs if detailed loss attribution becomes necessary.

## Next Actions
1. Run the old-model validation-loss checkpoint audit if saved epoch checkpoints are available.
2. Run a small alpha_v2 smoke using:

```text
loss_profile=personal_time_efficient_topk_v1
selection_profile=validation_loss
output_profile=decision_utility_v1
```

3. If smoke passes, run the fixed h256 seed7 full experiment from the plan.
4. Interpret broad diagnostics as diagnostics only:
   - `personal_topk_v1`
   - rank/spread by horizon
   - same-candidate test
   - time-efficient score diagnostics
5. Keep the current anchor `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` until a new-contract model beats it under the clean rule.
