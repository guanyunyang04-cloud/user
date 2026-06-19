# Path Policy Alpha V2 Old-Contract Validation-Loss Audit 20260618

## Verdict
- Status: `stage0_audit_completed / old_contract_only / research_only / execution_frozen`.
- Date: `2026-06-18`.
- Research program: `qdp_alpha_v2_time_efficient_topk_research`.
- Scope: audit whether old alpha_v2 h256 hybrid runs were materially misread by old checkpoint-selection semantics before starting the new `personal_time_efficient_topk_v1` full run.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` is absent and was not touched.

## Why This Audit Exists
The user clarified the training contract rule:

```text
training objective defines what the model learns;
checkpoint selection should follow that exact objective on validation;
evaluation can be broad, but must not replace checkpoint selection.
```

Old alpha_v2 runs used mixed `decision_utility_v1`-family objectives and mostly selected checkpoints by old `decision_utility` selection score. This audit checks whether the old runs contain an obvious alternate epoch that should change interpretation before moving to the new objective.

Important boundary:

```text
old validation_loss = old mixed-contract validation loss
```

It is useful for interpreting old training, but it is not the new `personal_time_efficient_topk_v1` validation loss.

## Source Artifacts
- `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01/forecast_learning_curve.csv`
- `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02/forecast_learning_curve.csv`
- `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02/forecast_learning_curve.csv`
- `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02/forecast_learning_curve.csv`
- Existing `personal_topk_v1_validation` and `personal_topk_v1_test` reports under the same study roots.
- Anchor-only existing `personal_checkpoint_reselection_validation/checkpoint_personal_topk_reselection_validation.csv`.

No new model training, prediction generation, score-backtest bridge, candidate matrix, execution-candidate review, paper/live, broker, or active/default artifact action was performed.

## Compact Table

| run | best validation-loss epoch | old selection best epoch | best rank IC20 epoch | best spread20 epoch | personal-topK validation selection | test-only diagnostic selection |
|---|---:|---:|---:|---:|---|---|
| `anchor_topn` | `3` (`7.5573`) | `2` (`1000.7728`) | `4` (`0.09055`) | `4` (`0.02455`) | `pred_decision_score top1 h10`, score `276.55`, net `2.72%` | `pred_cum_mu_1d top1 h20`, score `440.94`, net `3.84%` |
| `topk_align` | `1` (`7.8797`) | `1` (`1000.7058`) | `1` (`0.06053`) | `4` (`0.01544`) | `pred_decision_score top1 h5`, score `52.37`, net `1.30%` | `pred_cum_mu_5d top1 h5`, score `291.53`, net `2.14%` |
| `monthly_robust` | `1` (`8.8013`) | `2` (`1000.6360`) | `4` (`0.08627`) | `4` (`0.02066`) | `pred_cum_mu_5d top1 h5`, score `11.22`, net `1.20%` | `pred_aux_upside_20d top1 h20`, score `280.52`, net `3.23%` |
| `h30soft` | `1` (`9.6313`) | `1` (`1000.8264`) | `2` (`0.10386`) | `4` (`0.02618`) | `pred_decision_score top1 h5`, score `203.00`, net `1.96%` | `pred_cum_mu_1d top1 h20`, score `640.31`, net `4.74%` |

## Interpretation
The old runs show three different notions of "best":

```text
lowest old mixed validation loss
highest old decision_utility selection score
best forecast diagnostic such as rank_ic20/spread20
```

They do not consistently select the same epoch. This validates the user's concern that model training, checkpoint selection, and evaluation views must be semantically separated.

The anchor remains the old-contract comparison baseline because:

- it still has the strongest validation-selected `personal_topk_v1` same-candidate evidence among old alpha_v2 runs;
- its existing epoch-level personal checkpoint reselection selects epoch `2`, matching the old selected checkpoint;
- the validation-loss best epoch `3` improves old mixed validation loss but has lower epoch-level personal topK score than epoch `2`;
- later epochs improve rank/spread diagnostics but do not improve the old checkpoint contract enough to replace epoch `2`.

The other Stage 1 scouts do not produce an obvious reason to replace the anchor:

- `topk_align`: validation-loss best and old selection best both choose epoch `1`; final personal topK is weak.
- `monthly_robust`: validation-loss best epoch `1` and old selection epoch `2` differ, but both are old mixed-contract evidence and the run-level personal topK result is weak.
- `h30soft`: forecast diagnostics are strong, and test-only selection is attractive, but validation-loss and old selection both choose epoch `1`; validation-selected same-candidate test remains weak versus anchor.

## Conclusion
Stage 0 does not reveal a higher-priority old-checkpoint issue that should block the new objective.

The correct next step remains:

```text
run one fixed h256 seed7 full experiment with:
loss_profile = personal_time_efficient_topk_v1
selection_profile = validation_loss
```

Old runs should remain as diagnostic and comparison evidence only. They should not be promoted, bridged, entered into candidate matrix, or used to alter active/default artifacts.

## Next Allowed Actions
- Start the first full alpha_v2 h256 seed7 run under `personal_time_efficient_topk_v1` and `validation_loss`.
- After completion, write a compact result reference with validation loss, component interpretation where available, time-efficient metrics, standard forecast diagnostics, and `personal_topk_v1` as an evaluation view.
- Keep multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, and active/default changes deferred.
