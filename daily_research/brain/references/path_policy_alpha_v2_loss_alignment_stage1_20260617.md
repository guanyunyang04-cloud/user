# Path Policy Alpha V2 Loss Alignment Stage 1 20260617

## Verdict
- Status: `stage1_loss_alignment_completed / no_anchor_beater / research_only`.
- Date: `2026-06-17` start, completed `2026-06-18` local time.
- Study family: `qdp_alpha_v2_strong_model_research`.
- Active plan: `daily_research/brain/references/path_policy_alpha_v2_strong_model_research_plan_20260617.md`.
- Anchor to beat: `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` remains absent and was not created or modified.
- Interpretation: Stage 1 tooling is now usable, but none of the three clean loss-only candidates beat the alpha_v2 h256 anchor under validation-selected same-candidate `personal_topk_v1`. They should not graduate to multi-seed, score-backtest bridge, candidate matrix, or execution-candidate review.

## Code And Contract Changes
- Added loss profile `decision_score_topk_alignment_v1` in `daily_research/path_policy/forecast_training.py`.
- The new loss keeps output profile `decision_utility_v1` and selection profile `decision_utility`.
- It adds a narrow batch-level small top-K proxy on 20d future cumulative excess return for requested K `1/3/5`, plus pairwise rank alignment on `pred_decision_score`.
- Semantics caveat: this is still a stock-sequence hybrid batch proxy, not a same-date full-market top-K loss. It is meant to test whether `pred_decision_score` aligns better with `personal_topk_v1`, not to turn hybrid into a date-level cross-sectional model.
- Added `daily_research/path_policy/qdp_alpha_v2_loss_alignment_scout.py` as a research-only fixed h256/T4/GRU2/b512/seed7 task generator and runner for alpha_v2 loss comparisons.
- Added tests in `daily_research/path_policy/tests/test_qdp_alpha_v2_loss_alignment_scout.py`.
- Important correction: scout commands must match the anchor decision-utility semantics: cost `20bps`, hit threshold `20bps`, drawdown penalty `0.25`. A first relaunch accidentally used `20/10/0.10`; it was stopped and moved aside.

## Misconfigured Run Boundary
- Stopped run id: `qdp_alpha_v2_topk_align_full_seed7_20260617_02`.
- Misconfigured tag: `qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_01`.
- Moved directory: `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_misconfigured_utility_20260617_01`.
- Reason stopped: decision utility parameters differed from the anchor (`20/10/0.10` instead of `20/20/0.25`), so it was not a clean loss-only comparison.
- Evidence status: invalid for model-quality comparison; keep only as a process trace.

## Completed Full Scout
- Agent run id: `qdp_alpha_v2_topk_align_full_seed7_20260617_03`.
- Valid run tag: `qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02`.
- Output root: `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02`.
- Task list: `daily_research/output/path_policy/studies/qdp_alpha_v2_loss_alignment_scout_20260617_02/qdp_alpha_v2_loss_alignment_task_list.json`.
- Training pack manifest: `H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01\qdp_training_pack_manifest.json`.
- Model/loss: `hybrid_expert_fusion_static_context`, h256, transformer layers `4`, heads `8`, GRU layers `2`, batch `512`, seed `7`, output `decision_utility_v1`, loss `decision_score_topk_alignment_v1`, selection `decision_utility`.
- Split: train `2012-2023`, validation `2024`, test `2025`; cumulative horizons `1,3,5,10,20`; forecast horizon `20`.
- Decision utility semantics: cost `20bps`, hit threshold `20bps`, drawdown penalty `0.25`.
- Status: completed with returncode `0`; early-stopped after epoch `4`; best epoch `1`.
- Best checkpoint: `forecast_model_hybrid_expert_fusion_static_context_seed7_best.pt`.
- Total elapsed seconds: `27763.281`; epoch seconds: `6857.203`, `6608.766`, `6633.094`, `6424.578`.

## Learning Curve Summary
| epoch | train loss | val selection | val rank_ic20 | val spread20 | val decision IC | is best |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `5.728880` | `1000.705755` | `0.060526` | `0.014092` | `0.066855` | yes |
| 2 | `5.451871` | `1000.589553` | `0.052768` | `0.011395` | `0.056032` | no |
| 3 | `5.332365` | `1000.527597` | `0.047877` | `0.011906` | `0.050300` | no |
| 4 | `5.242312` | `1000.503617` | `0.059726` | `0.015441` | `0.048165` | no |

Interpretation:
- The training loss kept falling, but validation selection and decision-score IC weakened after epoch 1.
- Compared with the anchor epoch2 validation metrics, the selected topK-align checkpoint is weaker on validation rank_ic20 (`0.060526` vs `0.076656`), spread20 (`0.014092` vs `0.018850`), and decision hit lift (`0.030224` vs `0.037445`).
- Test rank_ic20 is close (`0.111326` vs anchor `0.115449`) and test spread20 is slightly higher (`0.024501` vs `0.023697`), but this does not rescue the candidate because validation personal-topK selection failed.

## Personal Top-K Result
Validation-selected same-candidate comparison after all three Stage 1 candidates:

| model | validation selected candidate | val personal score | val net | test same-candidate net | val rank_ic20 | test rank_ic20 | val spread20 | test spread20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| alpha_v2 h256 anchor | `pred_decision_score top1 h10` | `276.548917` | `0.027220` | `0.027329` | `0.076656` | `0.115449` | `0.018850` | `0.023697` |
| topK-align loss | `pred_decision_score top1 h5` | `52.367663` | `0.012976` | `0.010897` | `0.060526` | `0.111326` | `0.014092` | `0.024501` |
| monthly-robust loss | `pred_cum_mu_5d top1 h5` | `11.221993` | `0.011953` | `0.007224` | `0.081773` | `0.117310` | `0.018807` | `0.025441` |
| h30soft loss | `pred_decision_score top1 h5` | `203.001090` | `0.019615` | `0.008833` | `0.096840` | `0.123863` | `0.024875` | `0.025812` |

Additional diagnostics:
- Validation net leaderboard had strong-looking `pred_cum_mu_20d top1 h20` net `0.033724`, but it was not selected by `personal_topk_v1` because its bad-month penalty made the selection score worse.
- Test-only reselection chose `pred_cum_mu_5d top1 h5` with score `291.526319` and net `0.021419`; this is diagnostic only and not valid same-candidate generalization evidence.
- The formal validation-selected test net fell from anchor `2.73%` to `1.09%`.
- `score_monthly_robust_v1` looked healthy on rank/spread and even slightly exceeded the anchor on test rank_ic20/spread20, but the validation-selected personal top-K candidate was weak and transferred to only `0.72%` same-candidate test net. This is not a personal top-K improvement.
- `horizon_30d_soft_penalty_v1` was the strongest forecast-quality candidate by rank/spread and had very strong test-only top-K diagnostics, but the formal validation-selected same-candidate test net was only `0.88%`. This is not an anchor beat under the current rule.

## Completed Monthly-Robust Scout
- Agent run id: `qdp_alpha_v2_monthly_robust_full_seed7_20260618_01`.
- Run tag: `qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02`.
- Output root: `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02`.
- Task list: `daily_research/output/path_policy/studies/qdp_alpha_v2_loss_alignment_scout_monthly_robust_20260618_01/qdp_alpha_v2_loss_alignment_task_list.json`.
- Model/loss: `hybrid_expert_fusion_static_context`, h256, transformer layers `4`, heads `8`, GRU layers `2`, batch `512`, seed `7`, output `decision_utility_v1`, loss `score_monthly_robust_v1`, selection `decision_utility`.
- Decision utility semantics: cost `20bps`, hit threshold `20bps`, drawdown penalty `0.25`.
- Status: completed with returncode `0`; early-stopped after epoch `5`; best epoch `2`.
- Progress: `forecast_progress.json` status `completed`, elapsed seconds `32318.906`, average epoch seconds `6232.406`, final throughput about `960` samples/sec.

## Monthly-Robust Learning Curve Summary
| epoch | train loss | val selection | val rank_ic20 | val spread20 | val decision IC | is best |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `6.359694` | `1000.623910` | `0.081691` | `0.018679` | `0.059197` | yes |
| 2 | `6.018849` | `1000.636001` | `0.081773` | `0.018807` | `0.060498` | yes |
| 3 | `5.877261` | `1000.622163` | `0.084245` | `0.019759` | `0.059525` | no |
| 4 | `5.771058` | `1000.510504` | `0.086270` | `0.020660` | `0.048820` | no |
| 5 | `5.680210` | `1000.407849` | `0.075068` | `0.018581` | `0.039195` | no |

Interpretation:
- The rank/spread curve was not the problem: epoch3/4 improved validation rank/spread, and the selected epoch2 test rank_ic20/spread20 reached `0.117310 / 0.025441`.
- The weakness was score semantics for personal top-K: validation `personal_topk_v1` selected `pred_cum_mu_5d top1 h5`, not `pred_decision_score`, with score only `11.22`, validation net `1.20%`, and same-candidate test net `0.72%`.
- This reinforces the anchor as the current single-seed alpha_v2 h256 reference: `topn_excess_rank_v1` still gives the best validation-selected same-candidate personal top-K evidence.

## Completed H30Soft Scout
- Agent run id: `qdp_alpha_v2_h30soft_full_seed7_20260618_01`.
- Run tag: `qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02`.
- Output root: `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02`.
- Task list: `daily_research/output/path_policy/studies/qdp_alpha_v2_loss_alignment_scout_h30soft_20260618_01/qdp_alpha_v2_loss_alignment_task_list.json`.
- Model/loss: `hybrid_expert_fusion_static_context`, h256, transformer layers `4`, heads `8`, GRU layers `2`, batch `512`, seed `7`, output `decision_utility_v1`, loss `horizon_30d_soft_penalty_v1`, selection `decision_utility`.
- Decision utility semantics: cost `20bps`, hit threshold `20bps`, drawdown penalty `0.25`.
- Status: completed with returncode `0`; early-stopped after epoch `4`; best epoch `1`.
- Progress: `forecast_progress.json` status `completed`, elapsed seconds `26931.203`, average epoch seconds `6429.953`, final throughput about `947` samples/sec.

## H30Soft Learning Curve Summary
| epoch | train loss | val selection | val rank_ic20 | val spread20 | val decision IC | is best |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `6.759631` | `1000.826428` | `0.096840` | `0.024875` | `0.078404` | yes |
| 2 | `6.392293` | `1000.723794` | `0.103859` | `0.025520` | `0.067654` | no |
| 3 | `6.239230` | `1000.659398` | `0.102129` | `0.026014` | `0.062059` | no |
| 4 | `6.126097` | `1000.651167` | `0.099009` | `0.026177` | `0.065817` | no |

Interpretation:
- `horizon_30d_soft_penalty_v1` clearly improved forecast rank/spread over the anchor on the selected checkpoint: validation rank_ic20/spread20 `0.096840 / 0.024875` and test rank_ic20/spread20 `0.123863 / 0.025812`.
- Test-only reselection was very strong: test `personal_topk_v1` selected `pred_cum_mu_1d top1 h20`, score `640.31`, net `4.74%`, hit rate `62.16%`, worst month about `-1.97%`. This is diagnostic only.
- Formal validation-selected comparison did not transfer: validation selected `pred_decision_score top1 h5`, score `203.00`, validation net `1.96%`, but same-candidate test net was only `0.88%` with positive-month rate `58.33%`.
- The likely bottleneck is no longer basic forecast signal. It is score/head/checkpoint-selection alignment for personal top-K, especially the mismatch between strong 20d test signals and validation selecting a 5d decision-score candidate.

## Decision
- `decision_score_topk_alignment_v1` does not beat the current alpha_v2 h256 anchor.
- `score_monthly_robust_v1` also does not beat the current alpha_v2 h256 anchor under the same validation-selected same-candidate rule.
- `horizon_30d_soft_penalty_v1` improves rank/spread but also does not beat the current alpha_v2 h256 anchor under the same validation-selected same-candidate rule.
- Do not run multi-seed for any of these three losses.
- Do not run score-backtest bridge, candidate matrix, or execution review for any of these three losses.
- Keep `decision_score_topk_alignment_v1` available as a documented negative result and possible component for later ablations, but do not treat it as the next strong-model route.
- Keep `score_monthly_robust_v1` as a signal-preserving but personal-topK-weak negative result for alpha_v2 h256.
- Keep `horizon_30d_soft_penalty_v1` as a strong forecast-signal diagnostic and possible architecture/head-ablation baseline, but not as the current anchor.

## Handles
- Status:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.agent_run status --project-id daily_research --run-id qdp_alpha_v2_topk_align_full_seed7_20260617_03 --json`
- Progress:
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02/forecast_progress.json`
- Agent logs:
  `daily_research/output/agent_runs/qdp_alpha_v2_topk_align_full_seed7_20260617_03/stdout.log`
  `daily_research/output/agent_runs/qdp_alpha_v2_topk_align_full_seed7_20260617_03/stderr.log`
- Scout logs:
  `daily_research/output/path_policy/studies/qdp_alpha_v2_loss_alignment_scout_20260617_02/logs/`

## Required Next Steps
1. Keep `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` as the current alpha_v2 h256 seed7 anchor.
2. Pause Stage 1 loss-only scouting. It has not produced an anchor-beating personal top-K candidate.
3. Move to Stage 2 hybrid architecture/head/router ablation, or first add checkpoint/output diagnostics that can explain why h30soft has strong forecast/test-only signals but weak validation-selected transfer.
4. Continue to use validation-selected same-candidate test reporting; test-only reselection remains diagnostic only.

## Verification Run
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_qdp_alpha_v2_loss_alignment_scout.py daily_research/path_policy/tests/test_forecast_training.py::test_auxiliary_decision_loss_profiles_record_weight_contract_and_finite_loss daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_parser_accepts_auxiliary_decision_loss_profiles_and_sets_decision_output -q`
- Result: `10 passed in 8.81s`.
- `git diff --check`: passed.
- `personal_topk_v1` validation/test diagnostics completed under:
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02/personal_topk_v1_validation`
  and
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02/personal_topk_v1_test`.
- `score_monthly_robust_v1` validation/test diagnostics completed under:
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02/personal_topk_v1_validation`
  and
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02/personal_topk_v1_test`.
- `horizon_30d_soft_penalty_v1` validation/test diagnostics completed under:
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02/personal_topk_v1_validation`
  and
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02/personal_topk_v1_test`.
