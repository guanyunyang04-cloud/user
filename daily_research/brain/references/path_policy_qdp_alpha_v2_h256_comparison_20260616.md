# Path Policy QDP Alpha V2 H256 Comparison 20260616

## Verdict
- Status: `completed / forecast_test_confirmed / research_only`.
- Run tag: `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
- Purpose: first same-mouth strong-model comparison on the new QDP `style_structural_alpha_v2_label_v2` pack, using the old `style_structural_v1` h256/topn/hybrid setup as the closest default control.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` was not created or modified.
- Interpretation: alpha_v2 is not just a consumption-smoke substrate anymore. This single-seed h256 run is evidence-grade supervised forecasting evidence, but it is still research/shadow-only and not promotion, paper/live, broker, or active-artifact evidence.

## Run Contract
- Training pack manifest: `H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01\qdp_training_pack_manifest.json`.
- Feature profile/count/hash: `style_structural_alpha_v2 / 307 / 38a98a4a96637df6bc465837`.
- Label schema: `path20_basic_v2` version `2`; execution mode `next_open`; lookback `252`.
- Split: train `2012-2023`, validation `2024`, test `2025`; samples `6,050,268 / 681,979 / 689,467`.
- Model/loss: `hybrid_expert_fusion_static_context`, hidden `256`, transformer layers `4`, heads `8`, GRU layers `2`, batch `512`, seed `7`, output profile `decision_utility_v1`, loss profile `topn_excess_rank_v1`, selection profile `decision_utility`.
- Early stop: ran epochs `1-5`; training system selected epoch `2` by validation selection score.
- Output root: `H:\quant_project\daily_research\output\path_policy\studies\qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.

## Learning Curve
| epoch | train loss | val selection | val rank_ic20 | val spread20 | val upside_ic20 | val decision IC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.544646` | `1000.756235` | `0.062528` | `0.013367` | `0.121012` | `0.071485` |
| 2 | `5.252182` | `1000.772821` | `0.076656` | `0.018850` | `0.136011` | `0.072948` |
| 3 | `5.128743` | `1000.675966` | `0.085958` | `0.023303` | `0.112695` | `0.064389` |
| 4 | `5.036705` | `1000.657861` | `0.090550` | `0.024551` | `0.104084` | `0.062771` |
| 5 | `4.955862` | `1000.662654` | `0.087616` | `0.023061` | `0.098563` | `0.063399` |

Notes:
- Training-system best is epoch `2`.
- Pure rank/spread best is epoch `4`.
- This confirms the earlier concern that training selection score, rank/spread, and personal small-capital top-K can diverge.

## Default Best Metrics
- Selected checkpoint: `forecast_model_hybrid_expert_fusion_static_context_seed7_best.pt` = epoch `2`.
- Validation metrics: rank_ic20 `0.0766560165`, spread20 `0.0188504279`, upside_ic20 `0.1360109685`, decision_score_rank_ic `0.0729477865`.
- Test metrics: rank_ic20 `0.1154490901`, spread20 `0.0236970004`, upside_ic20 `0.1736912561`, decision_score_rank_ic `0.0783070896`.
- Prediction CSVs:
  - `H:\quant_project\daily_research\output\path_policy\studies\qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01\forecast_predictions_validation.csv`
  - `H:\quant_project\daily_research\output\path_policy\studies\qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01\forecast_predictions_test.csv`

## Personal Top-K Default Best
Default epoch2 validation `personal_topk_v1` selected:
- Candidate: `pred_decision_score`, `top1`, horizon `10`.
- Validation personal score: `276.548917`; validation net mean `0.027220`; hit `0.583710`; positive-month rate `0.75`.
- Same validation-selected candidate on test: net mean `0.027329`; hit `0.590090`; positive-month rate `0.666667`.

Test-only reselection, for diagnostic only and not as formal generalization evidence, selected:
- `pred_cum_mu_1d`, `top1`, horizon `20`.
- Test personal score `440.941311`; net mean `0.038366`; hit `0.585586`; positive-month rate `0.75`.

## Checkpoint Reselection
Validation checkpoint reselection on epochs `1-5`:

| epoch | personal score | selected score | topK | horizon | val net | val rank_ic20 | val spread20 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | `276.548912` | `pred_decision_score` | 1 | 10 | `0.027220` | `0.076656` | `0.018850` |
| 3 | `246.916808` | `pred_decision_score` | 1 | 5 | `0.019566` | `0.085958` | `0.023303` |
| 1 | `71.220338` | `pred_cum_mu_20d` | 1 | 5 | `0.021158` | `0.062528` | `0.013367` |
| 5 | `-29.738528` | `pred_cum_mu_1d` | 1 | 20 | `0.040435` | `0.087616` | `0.023061` |
| 4 | `-66.991457` | `pred_cum_mu_20d` | 1 | 5 | `0.008641` | `0.090550` | `0.024551` |

Interpretation:
- Epoch4 has the strongest validation rank/spread, but personal-topK selection penalizes it due bad-month/instability behavior.
- Epoch2 is the best current personal small-capital validation checkpoint under `personal_topk_v1`.
- This is healthier than the old v1 h256 default result, where the default training-selected epoch1 had weak personal score and checkpoint reselection preferred epoch3.

## Same-Candidate Comparison Against Old V1 Defaults
Semantics: validation selects score/topK/horizon; test row reports that same validation-selected candidate, without test reselection.

| model | val selected candidate | val personal score | val net | test same-candidate net | val rank_ic20 | test rank_ic20 | val spread20 | test spread20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old h128 v1 | `pred_aux_upside_20d top3 h5` | `-70.033654` | `0.014115` | `0.015046` | `0.096024` | `0.117689` | `0.022176` | `0.026563` |
| old h192 v1 | `pred_cum_mu_20d top1 h5` | `165.930826` | `0.020668` | `0.013956` | `0.086927` | `0.123313` | `0.019456` | `0.027825` |
| old h256 v1 default | `pred_decision_score top5 h5` | `3.119396` | `0.013112` | `0.013593` | `0.065314` | `0.081305` | `0.013848` | `0.015489` |
| alpha_v2 h256 default | `pred_decision_score top1 h10` | `276.548917` | `0.027220` | `0.027329` | `0.076656` | `0.115449` | `0.018850` | `0.023697` |

Comparison artifact:
- `H:\quant_project\daily_research\output\path_policy\studies\qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01\comparison_v1_defaults\alpha_v2_vs_v1_default_personal_topk_comparison.json`
- `H:\quant_project\daily_research\output\path_policy\studies\qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01\comparison_v1_defaults\alpha_v2_vs_v1_default_personal_topk_comparison.csv`

## Speed Observation
- This run consumed the full alpha_v2 label_v2 pack with no loader error.
- Per-epoch total seconds including validation ranged roughly `6304-6538`.
- Observed train progress was mostly around `1050-1080 samples/s`; final learning curve reports `925-960 train_samples_per_second` including broader epoch accounting.
- This is not fast, but it is faster than the old v1 h256 full run's first epoch and acceptable for a first same-mouth strong-model comparison. No code-level speed change was made during the run because the measured path was healthy and changing batch/input dtype would weaken comparability.

## Boundaries
- `single_seed_only=true`: this is strong-model scout evidence, not final model-quality proof.
- `not_a_backtest=true`: personal top-K diagnostics use forward labels from prediction CSVs; they are not a score-backtest bridge, candidate matrix, or execution simulation.
- `promotion_allowed=false`; `shadow_only=true`; no active/default/live/paper/broker implication.
- Old v1 training pack manifest is no longer present locally, so old v1 epoch3 checkpoint test same-candidate replay was not rerun. The comparison above uses existing old v1 default prediction CSVs and personal top-K outputs.

## Next Suggested Research Move
- Since the user currently prioritizes finding a strong model, postpone multi-seed and score-backtest bridge/candidate matrix.
- Next alpha_v2 model work should tune the strength route rather than execution:
  - try a variant that aligns decision_score with the epoch4 rank/spread signal without creating bad-month instability;
  - compare h192/h256 or h256 loss/output-head variants on alpha_v2;
  - keep `personal_topk_v1` validation selection and same-candidate test reporting as mandatory diagnostics.
- Do not claim execution-candidate, live/default unfreeze, or production readiness until multi-seed stability and score-backtest bridge/candidate matrix are explicitly run later.
