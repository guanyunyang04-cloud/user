# alpha_path20 Decision Output Stability Gate 2026-05-21

## Verdict
- Status: `forecast decision-output stability gate / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Decision: do not run liquid800 and do not run multi-seed yet; liquid500 decision output candidates remain aggregate-positive but month-level stability is insufficient.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Inputs
- Audited candidates: `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01` and `path20_decision_output_liquid500_du_cost20_hit20_dd000_20260521_01`.
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`; pool `policy_pool_view__c11400fa72ad263f3d1eecfa`; sector view `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`.
- Audit artifacts: `path20_decision_output_liquid500_stability_audit_20260521.json` and `path20_decision_output_score_selection_stability_20260521.json`.
- Acceptance rule: aggregate test `decision_score_rank_ic`, `decision_score_top_bottom_spread`, and `decision_hit_lift_top20_mean` must be positive, and test monthly spread positive rate must be at least `0.60`.

## Stability Gate Results
| Candidate | Test rank IC | Test spread | Test hit lift | Test spread positive months | Gate | Negative test months |
|---|---:|---:|---:|---:|---|---|
| `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01` | 0.040376 | 0.009600 | 0.015789 | 0.500 | failed | 2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12 |
| `path20_decision_output_liquid500_du_cost20_hit20_dd000_20260521_01` | 0.021039 | 0.006395 | 0.014051 | 0.583 | failed | 2024-01, 2024-06, 2024-10, 2024-11, 2024-12 |

## Score / Selection Stability Check
- Validation selected the current `max_pred_utility` score for both candidates; no tested score variant met the `0.60` test monthly spread positive-rate gate.
- `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01`:
  - `max_pred_utility`: test spread `0.009600`, monthly spread positive rate `0.500`, negative months `2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12`.
  - `horizon_selected_utility`: test spread `0.010502`, monthly spread positive rate `0.500`, negative months `2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12`.
  - `forecast_5d_mu`: test spread `0.008193`, monthly spread positive rate `0.500`, negative months `2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12`.
  - `decision_forecast_blend`: test spread `0.003925`, monthly spread positive rate `0.417`, negative months `2024-01, 2024-05, 2024-06, 2024-08, 2024-10, 2024-11, 2024-12`.
- `path20_decision_output_liquid500_du_cost20_hit20_dd000_20260521_01`:
  - `max_pred_utility`: test spread `0.006395`, monthly spread positive rate `0.583`, negative months `2024-01, 2024-06, 2024-10, 2024-11, 2024-12`.
  - `horizon_selected_utility`: test spread `0.007200`, monthly spread positive rate `0.500`, negative months `2024-01, 2024-06, 2024-08, 2024-10, 2024-11, 2024-12`.
  - `forecast_5d_mu`: test spread `0.006178`, monthly spread positive rate `0.500`, negative months `2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12`.
  - `decision_forecast_blend`: test spread `-0.000664`, monthly spread positive rate `0.500`, negative months `2024-01, 2024-06, 2024-08, 2024-10, 2024-11, 2024-12`.

## Interpretation
- The best aggregate candidate remains `cost20/hit10/dd0.10`, but it failed the stricter monthly stability gate with positive spread in only `50%` of 2024 months.
- `cost20/hit20/dd0.00` was closer to the gate at `58.3%`, but its aggregate decision rank IC and spread were weaker than `cost20/hit10/dd0.10`.
- Negative months cluster around `2024-01`, `2024-06`, and `2024-10` to `2024-12`, which suggests regime sensitivity rather than a single score aggregation bug.
- Because simple score alternatives did not fix the monthly instability, liquid800 expansion would test a not-yet-stable signal and is intentionally skipped.

## Boundaries
- This is forecast-stage stability evidence only; no allocator, replay, live/default, or promotion claim is made.
- No `daily_research/output/active_execution_strategy.json` change is authorized.
- Do not treat aggregate-positive decision utility as sufficient without month/regime stability.

## Next Allowed Actions
- Do not run liquid800 decision-output expansion until a liquid500 candidate passes monthly stability.
- Next research should use liquid500 input/selection ablations, prioritizing `raw_kline_context_no_alpha_prior_v1` and `raw_kline_context_v1 + gru_sequence`, to determine whether alpha prior or sector/static context is amplifying regime sensitivity.
- Inspect negative months by market regime and predicted horizon distribution before changing target parameters again.
