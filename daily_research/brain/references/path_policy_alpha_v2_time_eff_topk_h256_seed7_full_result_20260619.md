# Path Policy Alpha V2 Time-Efficient TopK H256 Seed7 Full Result 20260619

## Verdict
- Status: `completed / first_new_contract_full_run / research_only / execution_frozen`.
- Run tag: `qdp_alpha_v2_hybrid_time_eff_topk_h256_t4_b512_seed7_20260618_01`.
- Study root: `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_time_eff_topk_h256_t4_b512_seed7_20260618_01`.
- Completion artifact timestamp: `2026-06-19T07:56:55+08:00`.
- Evidence verdict in study summary: `forecast_promising`.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` remained absent and was not recreated or modified.
- Interpretation: the new `personal_time_efficient_topk_v1` training contract completed a full alpha_v2 h256 seed7 run and checkpoint selection followed validation loss correctly. It is single-seed research evidence only and did not beat the old-contract alpha_v2 h256 anchor under the current validation-selected personal top-K diagnostic.

## Contract
```text
training pack:
  quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json

canonical dataset:
  policy_input_bundle__f926a496f69c61f6b92b5faf

pool:
  sh_sz_tradeable_mainboard_all_a_v1 / policy_pool_view__fc63d6b996abc7abde88bae7

feature profile:
  style_structural_alpha_v2 / 307 features / hash 38a98a4a96637df6bc465837

label schema:
  path20_basic_v2 version 2

split:
  train 2012-2023 / validation 2024 / test 2025
  rows 6,050,268 / 681,979 / 689,467

model:
  hybrid_expert_fusion_static_context
  hidden_dim 256 / transformer_layers 4 / transformer_heads 8 / gru_layers 2
  batch_size 512 / seed 7

output/loss/selection:
  output_profile = decision_utility_v1
  loss_profile = personal_time_efficient_topk_v1
  selection_profile = validation_loss
  selection_rule = lowest_validation_loss_for_training_loss_profile_then_seed_score
```

The training objective is:

```text
future_score = max_h((future_cumulative_excess_return_h - cost - path_risk_penalty_h) / h)
```

This preserves the user's intended capital-time semantics: a shorter holding period can beat a longer one if its cost/risk-adjusted return per day is better.

## Learning Curve
| epoch | train loss | validation loss | best? | patience | epoch seconds | train samples/s |
|---:|---:|---:|---|---:|---:|---:|
| 1 | `3.401109` | `4.415852` | yes | 0 | `7273.55` | `831.82` |
| 2 | `3.232907` | `4.459621` | no | 1 | `6923.89` | `873.82` |
| 3 | `3.160479` | `4.460868` | no | 2 | `6741.98` | `897.40` |
| 4 | `3.113372` | `4.491095` | no | 3 | `6736.42` | `898.14` |

Result:
- Best epoch: `1`.
- Best validation loss: `4.4158515878980635`.
- Early stop: `early_stopping_patience_exhausted` after epoch `4`.
- Important interpretation: train loss kept falling while validation loss worsened, so validation-loss checkpoint selection correctly chose epoch `1` and treated later epochs as overfit under this objective.

## Speed Change During Monitoring
The full run exposed that each epoch could pay for both exact validation loss and full validation prediction-frame metrics. A fast path was added:

```text
--no-forecast-per-epoch-prediction-metrics
```

Code-level contract:
- exact `_evaluate_loss` remains the checkpoint selector when `selection_profile=validation_loss`;
- full validation/test predictions and metrics are still generated for the final best checkpoint;
- `per_epoch_prediction_metrics=False` is guarded to only work with `selection_profile=validation_loss`;
- epochs with `validation_loss_only` in the learning curve intentionally have zero placeholder rank/spread metric columns, not zero model skill.

Focused tests passed for validation-loss selection and deferred per-epoch prediction metrics.

## Forecast Metrics
Best-checkpoint validation metrics:

```text
rank_ic_1d   = 0.048951
rank_ic_3d   = 0.063316
rank_ic_5d   = 0.065482
rank_ic_10d  = 0.055682
rank_ic_20d  = 0.069275
time_eff_score_rank_ic = 0.074451
```

Best-checkpoint test metrics:

```text
rank_ic_1d   = 0.053918
rank_ic_3d   = 0.065194
rank_ic_5d   = 0.073936
rank_ic_10d  = 0.098280
rank_ic_20d  = 0.106003
time_eff_score_rank_ic = 0.060200
```

These are usable supervised forecast diagnostics, but weaker than the old-contract alpha_v2 h256 anchor on the main rank/spread comparison currently used in the brain.

## Personal Top-K Diagnostics
`personal_topk_v1` remains an evaluation view, not the checkpoint selector for this run.

Validation-selected candidate:

```text
score_column = pred_decision_score
top_k        = 3
horizon      = 3
personal_selection_score = 41.153386
validation net_mean = 0.006001
validation excess_vs_all_mean = 0.008590
validation hit_rate = 0.558069
validation positive_month_rate = 0.666667
```

Same validation-selected candidate on test:

```text
score_column = pred_decision_score
top_k        = 3
horizon      = 3
test personal_selection_score = -14.400881
test net_mean = 0.005722
test excess_vs_all_mean = 0.006139
test hit_rate = 0.509009
test positive_month_rate = 0.750000
```

Test-only reselection, diagnostic only:

```text
score_column = pred_cum_mu_5d
top_k        = 1
horizon      = 20
test personal_selection_score = 311.677723
test net_mean = 0.042286
test excess_vs_all_mean = 0.035285
test hit_rate = 0.563063
test positive_month_rate = 0.583333
```

Interpretation:
- The validation-selected candidate generalizes to a small positive test net, but the personal score becomes negative due to the diagnostic penalty mix.
- The attractive test-only selection is not a valid formal conclusion because it is selected on test.
- The old-contract alpha_v2 h256 anchor remains stronger on validation-selected same-candidate personal top-K evidence: anchor test same-candidate net was about `2.73%`, while this new-contract run is about `0.57%`.

## Conclusion
This run is valuable because it validates the clean training/checkpoint contract on the full pack:

```text
train objective = personal_time_efficient_topk_v1
checkpoint selection = lowest validation loss for that same objective
personal_topk_v1 / rank-spread / same-candidate test = diagnostics only
```

Model-quality conclusion:
- `personal_time_efficient_topk_v1` as currently weighted did not beat the old alpha_v2 h256 `topn_excess_rank_v1` anchor in this single seed.
- Later epochs should not be rescued by rank/spread or test-only top-K selection, because validation loss worsened after epoch `1`.
- This does not prove the time-efficient idea is bad; it proves the first full h256 weighting is not yet a better strong-model candidate.

## Boundaries
- `single_seed_only=true`.
- `not_a_backtest=true`.
- `promotion_allowed=false`; `shadow_only=true`.
- No multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default artifact change.

## Next Allowed Actions
1. Keep `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` as the old-contract comparison anchor.
2. Do not enter multi-seed, bridge, candidate matrix, or execution review from this new-contract seed7 run.
3. Before launching another long full run, inspect why validation loss overfits after epoch `1` and whether `personal_time_efficient_topk_v1` component weights over-constrain the hybrid model.
4. Candidate next research should be a controlled, cheaper diagnostic rather than random loss-only scouting:
   - per-component validation-loss attribution if available or easy to add;
   - h192 or stronger regularization under the same contract;
   - lighter `time_eff_topk_alignment` / `decision_rank_aux` weighting;
   - then rerun one controlled h256 or h192 full candidate only if the diagnostic supports it.
