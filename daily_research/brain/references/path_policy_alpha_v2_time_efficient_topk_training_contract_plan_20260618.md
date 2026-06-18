# Path Policy Alpha V2 Time-Efficient TopK Training Contract Plan 20260618

## Verdict
- Status: `plan_accepted / training_contract_redesign / research_only / execution_frozen`.
- Date: `2026-06-18`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `qdp_alpha_v2_time_efficient_topk_research`.
- Supersedes hot-path interpretation in `path_policy_alpha_v2_strong_model_research_plan_20260617.md` where Stage 1 treated `personal_topk_v1` as the primary selection lens for loss scouts.
- Current anchor remains `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` until a new contract produces stronger evidence.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` must remain absent or unchanged. No live/default, paper, broker, promotion, score-backtest bridge, candidate matrix, or execution-candidate action is allowed from this plan.

## Core Correction
The next path_policy alpha_v2 work must separate three concepts:

- Training objective: what the model is asked to learn.
- Checkpoint selection: which epoch best completed that objective on validation.
- Evaluation views: multiple diagnostics used to understand the trained model.

The user preference is now explicit: checkpoint selection should follow the training objective, not an external evaluation view. Evaluation can remain broad, but it must not silently replace the training contract.

For the next contract, the intended objective is:

```text
For each signal date, identify a small number of stocks whose future net utility is high after costs, path risk, and capital-time usage are considered.
```

Practical interpretation:

```text
personal small-capital, high-efficiency, high-return, top-K alpha
```

## Why The Old Contract Is Not Enough
Existing `decision_utility_v1` losses train a mixed objective:

```text
path_daily
quantile
path_aux
risk_aux
rank_aux
decision_utility
hit_aux
horizon_classification
decision_rank_aux
optional topN / bad-state / router terms
```

This is not wrong as a multi-task forecaster, but it is not a clean personal top-K training contract. In particular:

- `decision_utility` is only one component of the total loss.
- Current utility is approximately `cumulative_excess_return - cost - drawdown_penalty * downside`.
- Without time normalization, longer horizons can dominate because cumulative return scale usually grows with horizon length.
- `rank_ic20` and `spread20` are evaluation metrics, not the full training loss. They can show useful signal but cannot alone prove the model completed the full task.
- `personal_topk_v1` is useful as an evaluation view, but using it as the only model verdict while training another objective creates semantic drift.

The next contract must make validation loss meaningful: if validation loss improves under the new contract, it should mean the model is better at the declared personal time-efficient top-K task.

## Existing Evidence To Preserve
Current alpha_v2 artifacts remain useful and should be reinterpreted, not discarded:

- Anchor: `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
  - Current training-system best: epoch `2`.
  - Validation-loss best from learning curve: epoch `3`.
  - Anchor remains the comparison baseline until a new contract beats it under a clean rule.
- `horizon_30d_soft_penalty_v1` scout:
  - Strongest forecast signal among Stage 1 scouts.
  - Best selection and validation-loss epoch both `1`.
  - Useful diagnostic that forecast signal exists, but not a current anchor.
- `score_monthly_robust_v1` scout:
  - Best selection epoch `2`, validation-loss best epoch `1`.
  - Useful to audit checkpoint-selection sensitivity.
- `decision_score_topk_alignment_v1` scout:
  - Best selection and validation-loss epoch both `1`.
  - Keep as negative evidence for old batch-level topK alignment.

Do not rewrite old outcomes as promotion, execution, or final model-quality evidence. Use them as audit inputs.

## Stage 0: Old Model Validation-Loss Audit
Purpose: check whether old models were under- or over-stated by decision-utility checkpoint selection.

Scope:
- No retraining.
- No active artifact change.
- No score-backtest bridge.
- No candidate matrix.

Tasks:
1. Discover saved epoch checkpoints for:
   - `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`
   - `qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02`
   - `qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02`
   - `qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02`
2. For each available epoch checkpoint, compute or collect:
   - `train_loss`
   - `validation_loss`
   - old `validation_selection_score`
   - rank/spread by horizon
   - decision-score IC / hit lift
   - `personal_topk_v1` validation and same-candidate test
   - horizon distribution of `pred_best_horizon`
3. Produce a compact audit table:

```text
run_tag
epoch
validation_loss_rank
old_selection_rank
personal_topk_rank
test_same_candidate_rank
horizon_bias_summary
```

Done condition:
- We know whether any old epoch deserves to be kept as an alternate diagnostic checkpoint.
- We do not replace the current anchor unless the user explicitly accepts a new old-contract selection rule.

Important boundary:
- Old `validation_loss` is the loss of the old mixed contract. It can audit old training, but it does not yet represent the new personal time-efficient objective.

## Stage 1: Define `personal_time_efficient_topk_v1`
Purpose: create a clean objective where validation loss maps to the user goal.

Target horizons:

```text
horizons = [1, 3, 5, 10, 20]
```

Base future net utility per horizon:

```text
net_utility_h =
    future_cumulative_excess_return_h
    - round_trip_cost
    - drawdown_penalty * max(0, -path_min_return_h)
    - worst_day_penalty * max(0, -worst_1d_h)
```

Time-efficiency transform:

```text
efficient_utility_h = net_utility_h / h
```

Rationale:
- This treats `1d +1%` and `20d +20%` as comparable on capital-time efficiency.
- Short horizons are not automatically preferred, because fixed trading cost is divided by fewer days and therefore penalizes short-term churn naturally.
- Longer horizons can still win if their risk-adjusted per-day utility is better.

Optional later ablation:

```text
efficient_utility_h = net_utility_h / sqrt(h)
```

This is not the default. Use it only if `/h` proves too short-horizon aggressive.

Primary target:

```text
future_score = max_h efficient_utility_h
future_best_horizon = argmax_h efficient_utility_h
```

Prediction contract:

```text
pred_eff_utility_h for each h
pred_score = max_h pred_eff_utility_h
pred_best_horizon = argmax_h pred_eff_utility_h
```

Required semantics:
- Inputs remain PIT and use `style_structural_alpha_v2` features.
- Labels remain from `path20_basic_v2`; no new QDP full rebuild is required for v1 if existing labels include cumulative excess, drawdown, worst day, and upside by horizon.
- The loss must not consume test labels or future data beyond the training label horizon.

## Stage 2: Loss Design
Purpose: make one primary objective with small auxiliaries, not another ambiguous mixed bundle.

Proposed loss profile:

```text
personal_time_efficient_topk_v1
```

Primary components:

```text
efficient_utility_huber:
  Huber(pred_eff_utility_h, future_eff_utility_h)

score_rank:
  pairwise_rank_loss(pred_score, future_score)

small_topk_proxy:
  batch-level top1/top3/top5 surrogate on future_score
```

Secondary components:

```text
best_horizon_classification:
  cross_entropy(pred_best_horizon_logits, future_best_horizon)

hit_aux:
  predict future_score > threshold_per_day

risk_aux:
  preserve path risk awareness
```

Small auxiliary components:

```text
path_daily
path_aux
quantile
```

These should remain small and diagnostic. They help representation learning but must not dominate the objective.

Initial weight intent:

```text
efficient_utility_huber: high
score_rank: high
small_topk_proxy: medium
best_horizon_classification: low-medium
hit_aux: low-medium
risk_aux: low
path_daily/path_aux/quantile: low
```

Implementation should record component losses separately. A future reader must be able to tell whether validation loss fell because the primary efficient top-K task improved or because an auxiliary got easier.

## Stage 3: Checkpoint Selection Contract
Purpose: make best checkpoint selection follow the training contract.

For `personal_time_efficient_topk_v1`:

```text
best_checkpoint = lowest validation_loss for this exact loss profile
```

Tie-breakers only when validation losses are effectively equal:

```text
1. lower primary validation component loss
2. higher validation future_score rank IC
3. better validation topK proxy metric
```

Rules:
- Do not select by test.
- Do not select by `personal_topk_v1` unless that exact objective is promoted into the training loss contract.
- Do not select by `rank_ic20/spread20` alone.
- Always save enough per-epoch checkpoints to audit selection.

Done condition:
- If validation loss improves, the statement means: the model better completed the declared personal time-efficient top-K task on validation.

## Stage 4: Evaluation Views
Purpose: keep broad diagnostics without letting them replace the objective.

Required evaluation after each full run:

```text
training contract:
  train loss, validation loss, component losses, best epoch

time-efficient diagnostics:
  future_score IC, pred_score IC, top1/top3/top5 efficient utility
  horizon distribution of predicted best horizon

standard forecast diagnostics:
  rank IC and spread for 1/3/5/10/20d
  daily path / quantile coverage
  risk auxiliary quality

personal topK diagnostics:
  personal_topk_v1 validation-selected same-candidate test
  treated as economic diagnostic, not checkpoint selector

runtime diagnostics:
  epoch seconds, validation seconds, throughput, GPU memory if available
```

Interpretation:
- Short horizon is not assumed better.
- Long horizon is not assumed better.
- The winner is the horizon with highest cost- and risk-adjusted utility per unit capital-time.

## Stage 5: Implementation Plan
Likely files:

```text
daily_research/path_policy/forecast_training.py
daily_research/path_policy/forecast_checkpoint_topk_reselection.py
daily_research/path_policy/personal_topk_diagnostics.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_qdp_alpha_v2_loss_alignment_scout.py
new or updated alpha_v2 scout runner
```

Implementation steps:
1. Add target helper for time-efficient utility using existing label tensors.
2. Add loss profile contract metadata for `personal_time_efficient_topk_v1`.
3. Add prediction columns:
   - `pred_time_eff_utility_{h}d`
   - `future_time_eff_utility_{h}d`
   - `pred_time_eff_score`
   - `future_time_eff_score`
   - `pred_time_eff_best_horizon`
   - `future_time_eff_best_horizon`
4. Add loss calculation with separate component logging.
5. Add selection profile or selection mode that chooses lowest validation loss for this profile.
6. Add tests:
   - finite loss and backward pass
   - horizon normalization makes `1d 1%` comparable to `20d 20%`
   - costs penalize short horizons appropriately
   - best checkpoint selection can use validation loss
   - prediction frame contains required columns
7. Add a smoke run on limited samples.
8. Add a full seed7 h256 run only after smoke passes.

## Stage 6: First Full Experiment
Fixed settings for first fair comparison:

```text
data: QDP alpha_v2 label_v2 training pack
feature_profile: style_structural_alpha_v2
label_schema: path20_basic_v2 version 2
model: hybrid_expert_fusion_static_context
hidden_dim: 256
transformer_layers: 4
transformer_heads: 8
gru_layers: 2
batch_size: 512
seed: 7
horizons: 1,3,5,10,20
selection: validation_loss under personal_time_efficient_topk_v1
```

Do not change architecture in the first new-contract run. The only intended variable is the target/loss/selection contract.

Compare against:
- Current alpha_v2 anchor.
- H30Soft diagnostic run.
- Old validation-loss audit results.

Graduation from single seed requires:
- Lower validation loss under the new contract with a healthy component breakdown.
- Better or at least plausible same-candidate test diagnostics.
- No collapse in 1/3/5/10/20 horizon metrics.
- Horizon distribution not trivially stuck on 20d unless justified by efficient utility.
- Runtime acceptable for follow-up.

## Stage 7: Architecture Work After Contract
Only after the new training contract exists:

```text
head depth
fusion depth
GRU depth
router diagnostics
expert ablation
hidden dim
transformer layers
```

Do not run these before the objective is clean. Architecture ablation without a clean objective only multiplies ambiguity.

## Stage 8: Deferred Work
Still deferred by user preference:

```text
multi-seed
score-backtest bridge
candidate matrix
execution-candidate review
paper/live/broker
active/default artifacts
```

When a strong single-seed new-contract model exists, revisit multi-seed before any execution bridge.

## Risks And Mitigations
- Risk: `/h` over-favors one-day noise.
  - Mitigation: cost per day naturally penalizes high turnover; inspect horizon distribution and add `/sqrt(h)` ablation only if needed.
- Risk: batch-level topK proxy is not true same-date market topK.
  - Mitigation: keep it a proxy for hybrid stock-sequence training; later use date-level reranker if needed.
- Risk: validation loss improves by auxiliary losses.
  - Mitigation: log component validation losses and make primary components dominant.
- Risk: old conclusions become confusing.
  - Mitigation: mark old Stage 1 as old-contract evidence, not invalid evidence.
- Risk: test-only attractive results tempt reselection.
  - Mitigation: test-only remains diagnostic. Selection stays validation-only.

## Immediate Next Actions
1. Run Stage 0 old checkpoint audit if enough epoch checkpoints are present.
2. Implement `personal_time_efficient_topk_v1` target helper, loss contract, prediction columns, and validation-loss selection support.
3. Add focused tests for horizon efficiency and finite loss.
4. Run a small smoke on alpha_v2 pack.
5. If smoke is clean, run one h256 seed7 full experiment.
6. Write a compact result reference and update `state_center.md` after each completed full experiment.

## References
- `daily_research/brain/references/path_policy_alpha_v2_strong_model_research_plan_20260617.md`
- `daily_research/brain/references/path_policy_alpha_v2_loss_alignment_stage1_20260617.md`
- `daily_research/brain/references/path_policy_qdp_alpha_v2_h256_comparison_20260616.md`
- `daily_research/path_policy/forecast_training.py`
- `daily_research/path_policy/personal_topk_diagnostics.py`
- `daily_research/path_policy/forecast_checkpoint_topk_reselection.py`
