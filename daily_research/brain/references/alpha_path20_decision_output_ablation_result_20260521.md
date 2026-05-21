# alpha_path20 Decision Output Ablation Result 2026-05-21

## Verdict
- Status: `forecast decision-output target ablation / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Scope: liquid500 single-family/single-seed target-parameter ablation for `decision_utility_v1`.
- Evidence result: two ablations passed validation and test decision utility gates; no liquid800 expansion is authorized by this reference.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Inputs
- Lake dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool view id: `policy_pool_view__c11400fa72ad263f3d1eecfa` (`rolling_liquid500`).
- Sector/board view id: `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`.
- Feature profile: `raw_kline_context_sector_v1`.
- Model family: `gru_sequence_static_context`; seed `7`; device `cuda` with AMP.
- Output/loss/selection: `decision_utility_v1` / `decision_utility_v1` / `decision_utility`.
- Reused memmap manifest: `path20_arch_gain_liquid500_sector_v1_20260520_01/forecast_dataset_manifest.json`.

## Offline Score Audit
- Source study: `path20_decision_output_liquid500_sector_v1_20260521_01`.
- Audit artifact: `daily_research/output/path_policy/studies/path20_decision_output_liquid500_sector_v1_20260521_01/decision_score_audit_20260521.json`.
- Tested score variants: `max_pred_utility`, `horizon_selected_utility`, `hit_weighted_utility`, `short_horizon_blend`.
- None made test `decision_score_top_bottom_spread` positive; best tested alternative was `horizon_selected_utility` at `-0.002043`.
- Interpretation: the failure was not fixed by a simple score aggregation swap, so target-parameter ablation was justified.

## Ablation Matrix
| Label | Study tag | Cost bps | Hit bps | Drawdown penalty | Verdict | Val rank IC | Val spread | Val hit lift | Test rank IC | Test spread | Test hit lift |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `baseline` | `path20_decision_output_liquid500_sector_v1_20260521_01` | 20 | 20 | 0.25 | failed | 0.093516 | 0.012156 | 0.066078 | 0.020946 | -0.004432 | 0.011445 |
| `cost10_hit10_dd025` | `path20_decision_output_liquid500_du_cost10_hit10_dd025_20260521_01` | 10 | 10 | 0.25 | failed | 0.088918 | 0.011849 | 0.056438 | 0.013103 | -0.007138 | 0.011634 |
| `cost20_hit10_dd010` | `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01` | 20 | 10 | 0.10 | passed | 0.088119 | 0.016299 | 0.034048 | 0.040376 | 0.009600 | 0.015789 |
| `cost20_hit20_dd000` | `path20_decision_output_liquid500_du_cost20_hit20_dd000_20260521_01` | 20 | 20 | 0.00 | passed | 0.085699 | 0.016953 | 0.044667 | 0.021039 | 0.006395 | 0.014051 |

## Interpretation
- Baseline `cost20/hit20/dd0.25` learned a validation-positive decision signal but failed on test spread (`-0.004432`).
- `cost10/hit10/dd0.25` also failed test spread (`-0.007138`), so merely lowering cost and hit threshold without relaxing drawdown pressure did not solve transfer.
- `cost20/hit10/dd0.10` passed all candidate gates: test `decision_score_rank_ic=0.040376`, test spread `0.009600`, test hit lift `0.015789`.
- `cost20/hit20/dd0.00` also passed all candidate gates: test `decision_score_rank_ic=0.021039`, test spread `0.006395`, test hit lift `0.014051`.
- The strongest current candidate is `cost20/hit10/dd0.10` because it has the best test decision rank IC and spread among ablations, despite weaker 20d forecast rank IC than the original baseline.
- The practical signal is that the original drawdown penalty was likely too hard for test-period transfer; relaxing drawdown pressure helped decision utility generalize more than changing score aggregation.

## Candidate Gate
- Candidate pass requires validation and test `decision_score_rank_ic > 0`, `decision_score_top_bottom_spread > 0`, and `decision_hit_lift_top20_mean > 0`.
- Passed: `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01`, `path20_decision_output_liquid500_du_cost20_hit20_dd000_20260521_01`.
- Failed: baseline `path20_decision_output_liquid500_sector_v1_20260521_01`, `path20_decision_output_liquid500_du_cost10_hit10_dd025_20260521_01`.

## Boundaries
- This is forecast-stage target-output evidence only.
- No allocator, replay, target-weight head, differentiable optimizer, live/default, or promotion claim is made.
- Do not modify `daily_research/output/active_execution_strategy.json` based on this result.
- Do not run liquid800 automatically; liquid800 requires a separate plan and should start from the best liquid500 candidate only after freshness and guard checks.

## Next Allowed Actions
- Prefer `cost20/hit10/dd0.10` as the next decision-output candidate for a separately approved robustness run.
- Before liquid800, inspect score/selection stability by year/month and compare against the forecast-path 20d anchor to avoid overfitting to one decision metric.
- If continuing research, keep `decision_utility_v1` shadow-only and compare it against `forecast_path_v1` in an execution-simulation layer before any live/default discussion.
