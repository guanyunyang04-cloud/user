# Path Policy Alpha V2 Date-Slate Incremental Implementation 20260621

## Verdict
- Status: `code_data_contract_implemented / real_pack_smoked / research_only / execution_frozen`.
- Date: `2026-06-21`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Implemented plan: `daily_research/brain/references/path_policy_alpha_v2_date_slate_incremental_architecture_plan_20260621.md`.
- Active artifact impact: none. No `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP registry active pointer was changed.

## Implemented Code Contract

Files changed:

```text
daily_research/path_policy/models.py
daily_research/path_policy/forecast_dataset.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/run_alpha_path20_protocol.py
daily_research/path_policy/tests/test_models.py
daily_research/path_policy/tests/test_forecast_memmap_dataset.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

Implemented:

```text
forecast_incremental_path_v2:
  daily future excess-return increments
  cumulative aux derived from daily_mu cumsum at 1/3/5/10/20
  risk aux remains explicit

date_slate_alpha_fusion_v1:
  symbol static context forbidden
  static context requires exchange,industry
  group-contiguous alpha_v2 features preserved
  local TCN + recency patch Transformer + multi-EWMA experts
  router used as residual gate/pooling bias, not hard pre-fusion suppression
  intraday branch is context-gated daily-increment residual

date_grouped_alpha_score_v1:
  prediction-first market-fact alpha loss
  no execution cost/risk utility
  no batch topK execution alignment
  rank loss only within same prediction-date slate
  unit-time alpha rank included

finite guard:
  input/prediction/loss/gradient/parameter finite checks
  bad-batch JSON dump
  fail-fast only; no clipping, winsorization, canonical mutation, or tensor replacement
```

Protocol wiring added:

```text
--forecast-build-date-slate-pack
--forecast-date-slate-output-root
--forecast-date-slate-tag
--forecast-date-slate-feature-dtype
--forecast-date-slate-stock-chunk-size
--forecast-dates-per-batch
--forecast-stocks-per-date
--forecast-rank-min-group-size
--forecast-rank-max-pairs-per-date
--forecast-finite-guard
--forecast-bad-batch-dump-dir
```

The protocol rejects incompatible date-slate contracts:

```text
symbol in --forecast-train-static-fields
output_profile != forecast_incremental_path_v2
loss_profile != date_grouped_alpha_score_v1
non-memmap dataset mode
```

## Real Date-Slate Pack

Source structured sidecar:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01/qdp_structured_alpha_v2_training_pack_manifest.json
```

New date-slate pack:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_date_slate_pack_20260621_01/qdp_date_slate_training_pack_manifest.json
```

Pack facts:

```text
artifact_type: qdp_date_slate_training_pack_v1
status: completed
feature layout: date_stock_feature
date_major_feature_panel_shape: [3989, 3150, 307]
date_major_feature_dtype: float16
feature_panel_date_stock_feature.float16.dat: 7,715,124,900 bytes
date_slate_index.parquet: 43,671,308 bytes
row_count after load: 7,421,714
static context: exchange,industry after training override
```

This is a sidecar data/training layout. It does not change QDP canonical data or active registry pointers.

## Real Smoke

Loader/model/loss smoke on the real date-slate pack:

```text
input_shape: [16, 252, 307]
static_shape_after_filter: [16, 2]
date_group_ids: [0]
mu_shape: [16, 20]
aux_shape: [16, 20]
derived_cum_mu_shape: [16, 5]
router_weights_shape: [16, 3]
loss: 3.3881771564483643
loss_isfinite: true
prediction_all_finite: true
```

Real training-contract smoke:

```text
run_tag: qdp_alpha_v2_date_slate_alpha_fusion_contract_smoke_20260621_01
study_root: daily_research/output/path_policy/studies/qdp_alpha_v2_date_slate_alpha_fusion_contract_smoke_20260621_01
manifest: qdp_date_slate_training_pack_v1
role cap: 16 / 16 / 16
model_family: date_slate_alpha_fusion_v1
output_profile: forecast_incremental_path_v2
loss_profile: date_grouped_alpha_score_v1
selection_profile: validation_loss
static context: exchange,industry
date_slate_dates_per_batch: 1
date_slate_stocks_per_date: 16
rank_min_group_size: 8
rank_max_pairs_per_date: 128
finite_guard: enabled
status: completed
best_epoch: 1
best_validation_loss: 2.1607880126684904
```

Produced artifacts:

```text
forecast_model_date_slate_alpha_fusion_v1_seed7_best.pt
forecast_model_date_slate_alpha_fusion_v1_seed7_last.pt
forecast_learning_curve.csv
forecast_predictions_validation.csv
forecast_predictions_test.csv
forecast_training_summary.json
forecast_progress.json
```

Checkpoint finite check:

```text
best checkpoint tensor_count: 176
nonfinite_count: 0
```

No bad-batch dump was produced in this smoke.

## Verification

Passed:

```text
py_compile changed implementation/test files
git diff --check
pytest daily_research/path_policy/tests/test_models.py
pytest daily_research/path_policy/tests/test_forecast_memmap_dataset.py
pytest daily_research/path_policy/tests/test_forecast_training.py
pytest daily_research/path_policy/tests/test_rl_protocol.py -x
focused 23-test date-slate/model/loss/protocol regression group
json.tool qdp_date_slate_training_pack_manifest.json
json.tool forecast_training_summary.json
checkpoint finite tensor scan
tools.brain.integrity_check --json
```

Notes:

```text
test_rl_protocol.py full run needs about 7.5 minutes on this machine; the first 5-minute attempt timed out, then the rerun with -x passed 59/59.
test_forecast_training.py emits ConstantInputWarning in tiny fixtures; this is existing fixture behavior and not a contract failure.
```

## Interpretation

This completes the code/data contract and real smoke portions of the superseding date-slate incremental plan. It proves:

```text
date-slate pack can be built from the structured alpha_v2 sidecar
date-major loader returns same-date flattened slates
model forward is finite on real alpha_v2 windows
date-grouped loss is finite and same-date scoped
training loop can save/restore date-slate checkpoints
validation_loss selection uses the same date-grouped loss contract
```

It does not prove:

```text
model quality
formal single-seed superiority
multi-seed stability
score-backtest bridge quality
candidate matrix eligibility
execution-candidate readiness
paper/live/broker suitability
```

## Next Allowed Actions

Recommended next step:

```text
run a real CUDA throughput scout on qdp_date_slate_training_pack_v1:
  hidden_dim=128
  model_family=date_slate_alpha_fusion_v1
  output_profile=forecast_incremental_path_v2
  loss_profile=date_grouped_alpha_score_v1
  static_context=exchange,industry
  finite_guard enabled
  compare dates_per_batch/stocks_per_date shapes such as 1x256, 1x512, 2x256 if memory permits
```

Only after throughput and finite stability are acceptable:

```text
launch first formal research run, likely 3-6 epochs, validation_loss selection, no per-epoch prediction metrics, final validation/test predictions enabled.
```

Still not allowed from this evidence:

```text
multi-seed
score-backtest bridge
candidate matrix
execution-candidate review
paper/live/broker
active/default artifact changes
```
