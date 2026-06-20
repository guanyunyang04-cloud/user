# Path Policy Alpha V2 Generalization Repair Plan 20260620

## Verdict
- Status: `plan_accepted / superseding_hot_path_plan / research_only / execution_frozen`.
- Date: `2026-06-20`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Scope: alpha_v2 hybrid forecast model generalization, not execution promotion.
- Supersedes hot-path next steps from:
  - `path_policy_alpha_v2_strong_model_research_plan_20260617.md`
  - `path_policy_alpha_v2_time_efficient_topk_training_contract_plan_20260618.md`
  - `path_policy_alpha_v2_hybrid_alpha_score_no_symbol_h256_seed7_full_result_20260620.md` next-action section.
- Active artifact impact: none. Do not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, production root, score-backtest bridge, candidate matrix, or execution-candidate review from this plan.

## Current Evidence

The latest clean diagnostic run is:

```text
run_tag:
  qdp_alpha_v2_hybrid_alpha_score_no_symbol_h256_t4_b512_seed7_20260619_01

data:
  QDP style_structural_alpha_v2_label_v2 full training pack
  train 2012-2023 / validation 2024 / test 2025
  train rows 6,050,268
  validation rows 681,979
  test rows 689,467

model:
  hybrid_expert_fusion_static_context
  hidden_dim 256
  GRU layers 2
  Patch Transformer layers 4
  Transformer heads 8
  DLinear branch enabled
  static fields exchange,industry only
  symbol_id disabled

loss/output:
  output_profile forecast_path_v1
  loss_profile hybrid_alpha_score_v1
  selection_profile validation_loss
```

Key loss facts:

```text
epoch  train_loss  validation_loss  test_loss
1      2.598986    3.437678         2.732956
2      2.471708    3.413702         2.746245
3      2.416181    3.425408         2.770642
4      2.371028    3.431220         2.794706
5      2.330885    3.422741         2.817378
6      2.294944    3.422156         2.852535
7      2.263373    3.432582         2.910806
8      2.237390    3.439301         2.925683
9      2.215131    3.438731         2.954119
10     2.196752    3.457431         3.004223
11     2.181156    3.493673         3.042005
12     2.167131    3.517288         3.067003
```

Interpretation:
- Train loss keeps improving, but test loss is best at epoch `1` and then degrades almost monotonically.
- Validation loss is best at epoch `2`; validation and test disagree on the exact best early epoch but agree that later training hurts generalization.
- This is stronger evidence for an early generalization failure than the previous narrower conclusion that only epoch2+ overfits.

Full train top-K diagnostic:

```text
epoch12 full train dates: 2,894
full train rows: 6,050,268
personal_time_efficiency_v2 train-selected:
  pred_aux_cum_1d top1 h1
  net/day 2.67%
  excess/day 2.86%
  hit_rate 78.33%
  positive_month 98.60%
```

Same train-selected candidate does not transfer well:

```text
validation:
  net/day 0.449%
  score -9.27

test:
  net/day 0.111%
  positive_month 41.67%
  score -104.17
```

Interpretation:
- The model can fit the training domain very strongly.
- Direct full-train posterior candidate selection is not a valid future rule.
- A train-internal rolling selection protocol is required before using train-domain candidate choices.

Feature composition fact:

```text
feature count: 307
rough intraday / price-volume-like group: about 159 / 307
lookback: 252 days
```

The pack does not feed raw 5-minute bars, but it contains many intraday summary features and their cross-sectional ranks/z-scores. This is a high-priority generalization suspect, not proof of a bug.

## Working Diagnosis

The primary blocker is now `generalization_under_high_overlap_and_high_frequency_features`, not loss/output semantics alone.

Current likely contributors:

1. Highly overlapping 252-day windows:
   - Adjacent samples share 251/252 days of input.
   - Labels can differ materially because future returns shift by one day and market events are noisy.
   - Treating all adjacent stock-date windows as independent overstates effective sample size.

2. Intraday feature granularity:
   - Intraday summaries may capture useful short-term alpha.
   - They may also encode regime-specific microstructure, opening/closing behavior, or transient liquidity states.
   - Current experts all consume the same full feature panel, so intraday detail can influence every branch freely.

3. Capacity and regularization:
   - h256/T4/GRU2 is not objectively too large, but it is large enough to fit low-signal training details.
   - Dropout `0.15` and weight decay `1e-4` may be too weak for this data.
   - The correct question is not "large model bad" but whether current capacity has the right inductive bias and regularization.

4. Validation protocol:
   - Single-year validation is useful but insufficient for model-design decisions.
   - Rolling train-internal validation with purge/embargo is needed to select checkpoint, feature groups, and scorer rules without touching test.

5. Architecture:
   - Current hybrid is late-fusion multi-expert over the same input.
   - It lacks feature-group bottlenecks and explicit constraints separating stable daily structure from noisy intraday summaries.
   - Architecture changes should target inductive bias and input routing, not blind depth/width increases.

## Updated Objective

Build a stronger alpha_v2 hybrid predictor by reducing overfit and improving generalization before any multi-seed, bridge, candidate matrix, or execution discussion.

Primary target:

```text
Reduce train/validation/test loss divergence under hybrid_alpha_score_v1 or successor
while preserving or improving validation/test rank IC, spread, and same-candidate personal top-K diagnostics.
```

Non-targets:

```text
Do not optimize train loss alone.
Do not select scorer/horizon/topK on full training data without train-internal rolling validation.
Do not start raw 5-minute K-line modeling in this stage.
Do not treat test-only candidate selection as evidence.
```

## Evaluation Contract

Every diagnostic experiment must report:

```text
train loss curve
validation loss curve
test loss curve for saved checkpoints or selected checkpoints
validation-selected personal_topK same-candidate test result
test-only personal_topK only as diagnostic
rank_ic by 1/3/5/10/20d
top-bottom spread by 1/5/20d
epoch of best validation loss
epoch of best test loss, diagnostic only
train-domain topK if relevant, clearly labeled posterior/train-only
runtime and throughput
```

Use test loss only for audit, not checkpoint selection. Checkpoint selection for future runs should use train-internal rolling validation or 2024 validation according to the experiment contract; test remains final diagnostic.

## Stage 0: Tooling And Audit Baseline

Purpose: make the observed generalization failure reproducible and comparable.

Required artifacts:
- Existing `forecast_learning_curve.csv`.
- Existing all-epoch test-loss audit:
  `daily_research/output/path_policy/studies/qdp_alpha_v2_hybrid_alpha_score_no_symbol_h256_t4_b512_seed7_20260619_01/test_loss_audit_hybrid_alpha_score_v1_all_epochs.json`
- Existing full-train time-efficiency diagnostic:
  `role_date_stream_train_full_time_eff_v2_last_b1024`.

Actions:
1. Add or keep a reusable all-epoch loss-audit script if repeated often.
2. Ensure it can evaluate `train`, `validation`, and `test` roles by checkpoint with incremental JSON output.
3. Record exact loss profile and checkpoint static fields in every audit output.

Done condition:
- Future agents can reproduce the table showing test loss best at epoch1 and validation loss best at epoch2.

## Stage 1: Overlap-Aware Sampling Diagnostics

Purpose: test whether highly overlapping 252-day windows are a primary overfit driver.

Experiments:

```text
A1: train_date_stride=3
A2: train_date_stride=5
A3: optional block/month sampling if A1/A2 improve
```

Expected implementation options:
- Add a training-pack sample view or sampler option that selects training dates with stride while keeping validation/test full.
- Do not rebuild the QDP pack unless required; prefer runtime sample filtering.
- Keep no-symbol static context.
- Keep model/loss fixed initially:
  `hybrid_alpha_score_v1`, h256/T4/GRU2 or h192 if Stage 3 becomes first.

Success signal:
- Validation/test loss stops degrading immediately after epoch1/2, or degrades materially slower.
- Rank/spread and same-candidate top-K do not collapse.

Failure signal:
- Train loss rises but validation/test loss remains similar or worse.
- Then overlap is not the dominant issue, move priority to feature groups and regularization.

## Stage 2: Intraday Feature Group Diagnostics

Purpose: test whether intraday summary features are useful alpha or overfit noise.

Experiments:

```text
B1: no_intraday
  remove intraday_* plus their cs_rank/cs_z derivatives from model input.

B2: coarse_intraday
  keep a small curated intraday summary set, remove overly granular first/last 5m detail and duplicated rank/z pairs.

B3: intraday_group_dropout
  keep current features but randomly drop or zero the intraday group during training.
```

Do not start raw 5-minute K-line modeling here.

Reason:
- Raw 5m sequence modeling would increase noise and capacity, and may shift the model toward intraday trading / T+0 style behavior rather than daily alpha selection.
- Current task is daily or multi-day alpha ranking, not execution microstructure prediction.

Preferred later architecture if intraday helps but overfits:

```text
daily/market/industry/valuation features -> main temporal experts
intraday summaries -> small bottleneck branch, e.g. 16/32 dims
late fusion -> forecast heads
```

Success signal:
- Removing or compressing intraday features improves validation/test loss or same-candidate transfer.
- If no_intraday hurts but group_dropout helps, intraday is useful but needs regularization.

## Stage 3: Capacity And Regularization Grid

Purpose: determine whether current h256/T4/GRU2 is over-capacity for alpha_v2 under clean no-symbol input.

Controlled experiments:

```text
C1: h192, dropout 0.25, weight_decay 3e-4
C2: h128, dropout 0.30, weight_decay 3e-4
C3: h192, Transformer layers 2, heads 4, dropout 0.30
C4: keep h256 but dropout 0.30 and weight_decay 1e-3
```

Keep data and feature groups fixed when testing C variants unless combined only after Stage 1/2 evidence.

Success signal:
- Later epochs no longer monotonically worsen test loss.
- Best validation epoch no longer occurs immediately at epoch1/2.
- Validation/test rank IC and spread remain competitive.

Interpretation:
- If smaller/regularized models improve, current problem is partly capacity/regularization.
- If not, the main issue is likely feature semantics, sampling, target noise, or validation regime.

## Stage 4: Architecture With Better Inductive Bias

Purpose: improve structure only after Stage 1-3 identify the main overfit source.

Candidate architecture changes:

1. Feature-group routing:
   - Stable daily features to GRU/Patch/DLinear.
   - Intraday summaries to a small bottleneck branch.
   - Static context remains exchange,industry only.

2. Feature-group dropout:
   - Drop intraday group, valuation group, or rank/z group stochastically during training.

3. Late fusion improvement:
   - Consider 2-layer fusion transformer only after bottleneck/routing is in place.
   - Do not deepen Patch Transformer first by default.

4. Expert-specific inputs:
   - GRU branch: smoother daily trend/volume/market features.
   - Patch branch: multi-scale daily structure.
   - DLinear: low-frequency trend-like features.
   - Intraday branch: compressed same-day microstructure summary.

5. Output/loss:
   - Keep prediction-first market facts.
   - Use robust loss or Huber/winsorized targets if label noise dominates.
   - Keep cost/risk/execution preferences out of the primary loss for now.

Boundary:
- Do not implement a large raw 5m model until these cheaper diagnostics are complete.
- Do not add model depth/width without a specific diagnostic reason.

## Stage 5: Rolling Validation And Selection Protocol

Purpose: make selection happen inside the training domain without using the final test year.

Recommended protocol:

```text
fold1: train <= 2020, validate 2021
fold2: train <= 2021, validate 2022
fold3: train <= 2022, validate 2023
fold4: train <= 2023, validate 2024
test:  2025, final diagnostic only
```

Use purge/embargo:

```text
embargo at least 20 trading days
because label horizon is 20d
```

What this selects:
- checkpoint epoch,
- scorer family,
- score column/topK/horizon rule,
- feature group setting,
- regularization setting.

What it must not do:
- Select anything on 2025 test.
- Use full training posterior topK as if it were a future rule.

## Stage 6: External Scorer After Generalization Improves

Purpose: revisit personal small-capital top-K selection only after the predictor generalizes better.

Allowed:
- `personal_time_efficiency_v2` as diagnostic.
- Validation-selected same-candidate reporting.
- Train-internal rolling selection of score column / topK / horizon.

Not allowed yet:
- score-backtest bridge,
- candidate matrix,
- execution-candidate review,
- live/default/paper/broker,
- active artifact changes.

Reason:
- Current weak same-candidate transfer may be partly model generalization, not only scorer mismatch.
- Fixing scorer first risks fitting selection rules to unstable predictions.

## Stage 7: Graduation Criteria

A single-seed model can become the next strong-model candidate only if:

```text
validation/test loss no longer show immediate monotonic degradation after epoch1
validation-selected same-candidate test topK improves over current no-symbol run
rank_ic/spread remain competitive with or better than current anchor
test-only candidates are treated only as diagnostics
runtime remains acceptable
no active/live/default artifact touched
```

Only after this may the user decide whether to enter:

```text
multi-seed
score-backtest bridge
candidate matrix
execution-candidate review
```

Current default: do not enter these stages.

## Immediate Next Actions

Recommended order:

1. Implement or expose overlap-aware training sampling:
   - `train_date_stride=3`
   - `train_date_stride=5`
   - validation/test full unchanged.

2. Run the smallest meaningful full-pack scouts:
   - no-symbol,
   - `hybrid_alpha_score_v1`,
   - h192 or h256 fixed per experiment,
   - save per-epoch checkpoints,
   - compute validation and test loss curves.

3. In parallel or next, implement intraday feature group selection:
   - no_intraday,
   - coarse_intraday,
   - intraday_group_dropout.

4. Run capacity/regularization controls:
   - h192/dropout0.25/wd3e-4,
   - h128/dropout0.30/wd3e-4,
   - h192/T2/H4/dropout0.30.

5. Only after Stage 1-3 evidence, decide whether to implement feature-group routed hybrid with intraday bottleneck.

## Explicit Boundary Summary

- Current alpha_v2 anchor remains `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` for old-contract comparison.
- Current clean predictor evidence remains `qdp_alpha_v2_hybrid_alpha_score_no_symbol_h256_t4_b512_seed7_20260619_01`.
- New hot path is generalization repair, not more loss random walks.
- Do not claim the model is solved because train loss or full-train topK is strong.
- Do not claim 2024 validation alone is the issue; 2025 test loss also worsens with later epochs.
- Do not use raw 5m K-line modeling before cheaper feature-group and bottleneck diagnostics.
