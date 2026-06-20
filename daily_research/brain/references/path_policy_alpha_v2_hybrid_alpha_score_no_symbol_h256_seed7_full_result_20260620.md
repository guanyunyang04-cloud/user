# Path Policy Alpha V2 Hybrid Alpha-Score No-Symbol H256 Seed7 Full Result 20260620

## Verdict
- Status: `completed / no_symbol_alpha_score_full_run / research_only / execution_frozen`.
- Run tag: `qdp_alpha_v2_hybrid_alpha_score_no_symbol_h256_t4_b512_seed7_20260619_01`.
- Study root: `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_alpha_score_no_symbol_h256_t4_b512_seed7_20260619_01`.
- Completion artifact timestamp: `2026-06-20T09:11:10+08:00`.
- Evidence verdict in summary: `forecast_promising`.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` was not modified.
- Interpretation: this run validates the combined user hypothesis operationally: remove `symbol_id` from hybrid static context and train hybrid as a market-fact alpha score generator instead of an execution-utility learner. It is still single-seed research evidence and does not replace the old alpha_v2 anchor.

## Contract
```text
training pack:
  quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json

feature profile:
  style_structural_alpha_v2 / 307 features

label schema:
  path20_basic_v2 version 2

split:
  train 2012-2023 / validation 2024 / test 2025
  rows 6,050,268 / 681,979 / 689,467

model:
  hybrid_expert_fusion_static_context
  hidden_dim 256 / transformer_layers 4 / transformer_heads 8 / gru_layers 2
  batch_size 512 / seed 7

static context:
  source pack fields = symbol,exchange,industry
  training effective fields = exchange,industry
  symbol_id excluded from the model input

output/loss/selection:
  output_profile = forecast_path_v1
  loss_profile = hybrid_alpha_score_v1
  selection_profile = validation_loss
  per_epoch_prediction_metrics = false
```

The `hybrid_alpha_score_v1` loss is prediction-first:

```text
path_daily                 0.70
quantile                   0.25
path_aux                   0.35
rank_aux                   1.35
alpha_efficiency_rank_aux  0.35
risk_aux                   0.08
minor risk/direction rank  0.05 total
decision_utility           0.00
decision_rank_aux          0.00
time_eff_topk_alignment    0.00
hit/horizon classification 0.00
```

This means hybrid is trained as a stable alpha-score generator. Cost, risk preference, horizon choice, and top-K selection remain external scorer/execution-layer responsibilities.

## Learning Curve
| epoch | train loss | validation loss | best? | patience | epoch seconds | train samples/s |
|---:|---:|---:|---|---:|---:|---:|
| 1 | `2.598986` | `3.437678` | yes | 0 | `6707.75` | `901.98` |
| 2 | `2.471708` | `3.413702` | yes | 0 | `6684.38` | `905.14` |
| 3 | `2.416181` | `3.425408` | no | 1 | `6719.58` | `900.39` |
| 4 | `2.371028` | `3.431220` | no | 2 | `6597.03` | `917.12` |
| 5 | `2.330885` | `3.422741` | no | 3 | `6705.64` | `902.27` |
| 6 | `2.294944` | `3.422156` | no | 4 | `7023.69` | `861.41` |
| 7 | `2.263373` | `3.432582` | no | 5 | `7175.36` | `843.20` |
| 8 | `2.237390` | `3.439301` | no | 6 | `7108.30` | `851.16` |
| 9 | `2.215131` | `3.438731` | no | 7 | `6761.55` | `894.81` |
| 10 | `2.196752` | `3.457431` | no | 8 | `6697.97` | `903.30` |
| 11 | `2.181156` | `3.493673` | no | 9 | `6677.45` | `906.07` |
| 12 | `2.167131` | `3.517288` | no | 10 | `6675.20` | `906.38` |

Result:
- Best epoch by validation loss: `2`.
- Best validation loss: `3.413702074622337`.
- Stopped reason: `max_epochs_reached`.
- Train-to-plateau audit result: train loss kept falling through epoch `12`, but validation loss did not improve after epoch `2` and deteriorated materially by epoch `12`.
- Interpretation: under this cleaner no-symbol alpha-score contract, the early validation optimum remains real. Continuing only because train loss falls would select a more train-fitted checkpoint, not a better validation checkpoint.

## Forecast Metrics
Best-checkpoint validation metrics:

```text
rank_ic_1d   = 0.045725
rank_ic_3d   = 0.063501
rank_ic_5d   = 0.079266
rank_ic_10d  = 0.070320
rank_ic_20d  = 0.112704
spread_20d   = 0.031486
upside_ic20  = 0.119316
```

Best-checkpoint test metrics:

```text
rank_ic_1d   = 0.070504
rank_ic_3d   = 0.084203
rank_ic_5d   = 0.081944
rank_ic_10d  = 0.085306
rank_ic_20d  = 0.116119
spread_20d   = 0.025071
upside_ic20  = 0.172557
```

Compared with the old-contract alpha_v2 anchor `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`, this run has stronger validation rank/spread diagnostics, while test rank/spread is roughly similar. That is promising as a predictor, but it is not yet a stronger formal model candidate because the personal top-K same-candidate result remains weaker.

## Seen/Unseen Symbol Diagnostics
Because `symbol_id` was excluded, this run directly tests the cleaner generalization hypothesis.

```text
validation unseen rows = 8,566
validation unseen rank_ic20 = 0.154391
validation seen   rank_ic20 = 0.113534

test unseen rows = 15,979
test unseen rank_ic20 = 0.110278
test seen   rank_ic20 = 0.115920
```

Interpretation:
- Removing `symbol_id` did not break unseen-stock forecasting.
- Validation unseen-stock metrics are strong, though the unseen sample is much smaller than seen-stock.
- Test unseen-stock rank_ic20 is close to seen-stock rank_ic20. This supports keeping `symbol_id` disabled or at least heavily controlled in future hybrid alpha-score experiments.

## Personal Top-K Diagnostics
Because `forecast_path_v1` has no `pred_decision_score`, this run's personal top-K diagnostics must use alpha prediction columns such as `pred_cum_mu_*`, `pred_aux_cum_*`, and `pred_aux_upside_*`. Do not compare it as if it emitted decision-score columns.

Validation-selected alpha-score candidate:

```text
score_column = pred_aux_cum_20d
top_k        = 5
horizon      = 1
personal_selection_score = 32.931257
validation net_mean = 0.001479
validation excess_vs_all_mean = 0.003665
validation hit_rate = 0.530317
validation positive_month_rate = 0.666667
```

Same validation-selected candidate on test:

```text
score_column = pred_aux_cum_20d
top_k        = 5
horizon      = 1
test personal_selection_score = 79.856022
test net_mean = 0.001766
test excess_vs_all_mean = 0.003285
test hit_rate = 0.521622
test positive_month_rate = 0.833333
```

Test-only reselection, diagnostic only:

```text
score_column = pred_aux_cum_5d
top_k        = 3
horizon      = 20
test personal_selection_score = 459.765966
test net_mean = 0.042704
test excess_vs_all_mean = 0.035703
test hit_rate = 0.614114
test positive_month_rate = 0.833333
```

Interpretation:
- The validation-selected alpha-score candidate is positive on test, but very small in net return.
- The attractive test-only candidate is not a formal conclusion because it is selected on test.
- This strengthens the diagnosis that the predictor has useful long-horizon information, but the current validation selection/scorer is still not extracting a strong personal small-capital candidate.

## Conclusion
This run is valuable and cleaner than the first time-efficient contract run:
- `symbol_id` was removed without collapsing forecast quality.
- `hybrid_alpha_score_v1` aligns better with the idea that hybrid should predict market facts and alpha strength, not learn execution cost/risk preferences directly.
- Train-to-plateau showed no late validation recovery; best remains epoch `2`.
- Forecast rank/spread metrics are promising, especially validation rank_ic20 and test upside_ic20.

But it does not yet replace the current old-contract anchor:
- The same-candidate personal top-K result is too small: validation-selected `pred_aux_cum_20d top5 h1` gives test net about `0.18%`.
- It is single-seed only.
- It is not a backtest, not a bridge result, not a candidate matrix, and not execution evidence.

## Boundaries
- `single_seed_only=true`.
- `not_a_backtest=true`.
- `promotion_allowed=false`; `shadow_only=true`.
- No multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default artifact change.

## Next Allowed Actions
1. Keep `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` as the comparison anchor for now.
2. Treat this run as evidence that `no_symbol + prediction-first alpha-score loss` is a worthwhile direction, but the external scorer/selection needs diagnosis.
3. Next best diagnostic is not another long full run by default. First compare validation-selected versus test-only candidate structure, especially why `pred_aux_cum_5d top3 h20` is very strong on test but not selected on validation.
4. Consider a controlled follow-up that keeps no-symbol and alpha-score loss but adjusts the external personal top-K selection profile or runs rolling/multi-year validation selection before any multi-seed or bridge.
