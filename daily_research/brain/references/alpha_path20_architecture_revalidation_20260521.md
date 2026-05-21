# alpha_path20 Architecture Revalidation 2026-05-21

## Verdict
- Status: `forecast decision-output architecture revalidation / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Scope: liquid500 single-family/single-seed architecture check under the same `decision_utility_v1` contract.
- Decision: no non-GRU architecture passed the full decision candidate gate; do not run liquid800 or promotion work from this result.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Inputs
- Lake dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool view id: `policy_pool_view__c11400fa72ad263f3d1eecfa` (`rolling_liquid500`).
- Sector/board view id: `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`.
- Feature profile: `raw_kline_context_sector_v1`.
- Reused memmap manifest: `path20_arch_gain_liquid500_sector_v1_20260520_01/forecast_dataset_manifest.json`.
- Output/loss/selection: `decision_utility_v1` / `decision_utility_v1` / `decision_utility`.
- Decision parameters: cost `20` bps, hit threshold `10` bps, drawdown penalty `0.10`.
- Seed: `7`; epochs `16`; min epochs `8`; patience `6`; batch size `512`; device `cuda` with AMP.

## Source Audits
- `daily_research/output/path_policy/studies/path20_architecture_revalidation_audit_20260521.json`.
- `daily_research/output/path_policy/studies/path20_decision_architecture_gate_20260521.json`.
- Source forecast-path studies:
  - `path20_arch_gain_liquid500_sector_v1_20260520_01`.
  - `path20_arch_gain_liquid800_sector_v1_20260520_01`.

## Forecast-Path Architecture Recheck
| Study | Validation-selected family | Best test-spread family | Test spread | Selection mismatch |
|---|---|---|---:|---|
| `path20_arch_gain_liquid500_sector_v1_20260520_01` | `gru_sequence_static_context` | `gru_sequence_static_context` | 0.019350 | no |
| `path20_arch_gain_liquid800_sector_v1_20260520_01` | `gru_sequence_static_context` | `stock_mixer_sequence` | 0.022239 | yes |

## Decision-Output Architecture Gate
Candidate pass requires validation and test `decision_score_rank_ic > 0`, `decision_score_top_bottom_spread > 0`, `decision_hit_lift_top20_mean > 0`, plus test monthly spread positive rate `>= 0.60`.

| Family | Study tag | Best epoch | Test rank IC | Test spread | Test hit lift | Test spread positive months | Gate | Negative test months |
|---|---|---:|---:|---:|---:|---:|---|---|
| `gru_sequence_static_context` | `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01` | 4 | 0.040376 | 0.009600 | 0.015789 | 0.500 | failed | 2024-01, 2024-05, 2024-06, 2024-10, 2024-11, 2024-12 |
| `patch_transformer_static_context` | `path20_decision_arch_patch_liquid500_du_cost20_hit10_dd010_20260521_01` | 4 | 0.002701 | -0.007474 | 0.014528 | 0.333 | failed | 2024-01, 2024-05, 2024-07, 2024-08, 2024-09, 2024-10, 2024-11, 2024-12 |
| `stock_mixer_sequence` | `path20_decision_arch_stock_mixer_liquid500_du_cost20_hit10_dd010_20260521_01` | 1 | 0.067616 | 0.033849 | -0.001243 | 0.833 | failed | 2024-01, 2024-06 |
| `sector_slot_mixer_sequence` | `path20_decision_arch_sector_slot_liquid500_du_cost20_hit10_dd010_20260521_01` | 2 | 0.002860 | -0.007961 | 0.002363 | 0.500 | failed | 2024-01, 2024-07, 2024-09, 2024-10, 2024-11, 2024-12 |

## Interpretation
- The prior anchor remains valid: `gru_sequence_static_context` is still the best liquid500 forecast-path anchor.
- The stronger claim that other architectures have no doubt is not supported. Liquid800 forecast-path evidence shows validation selection can miss the best test-spread family.
- Under `decision_utility_v1`, simply replacing GRU with patch transformer or sector-slot mixer did not fix decision-score transfer; both failed test decision spread.
- `stock_mixer_sequence` is the useful near miss: test decision rank IC and spread were positive and monthly spread positive rate reached `0.833`, but hit lift was slightly negative and forecast 20d/5d metrics were negative. This suggests a score/ranking signal that is not yet aligned with the hit-threshold head.
- Early best epoch remains a cross-family symptom, not a GRU-only failure. This supports the current diagnosis that target/selection/regime stability is the main bottleneck.

## Boundaries
- This is forecast-stage evidence only.
- No allocator, replay, live/default, target-weight head, differentiable optimizer, or promotion claim is made.
- Do not modify `daily_research/output/active_execution_strategy.json` based on this result.
- Do not run liquid800 decision-output expansion from this result; no liquid500 non-GRU candidate passed the full gate.

## Next Allowed Actions
- Keep `gru_sequence_static_context` as the liquid500 research anchor for now.
- Do not continue by adding larger architectures. First inspect why `stock_mixer_sequence` has positive decision rank/spread but negative hit lift.
- Compare hit-threshold calibration, top20 hit composition, and score-to-hit monotonicity for GRU versus stock mixer.
- Continue with input/selection stability ablations before any pool expansion: `raw_kline_context_no_alpha_prior_v1` and `raw_kline_context_v1 + gru_sequence`.
- Treat recurring negative months, especially `2024-01` and `2024-06`, as regime diagnostics rather than architecture-only failures.
