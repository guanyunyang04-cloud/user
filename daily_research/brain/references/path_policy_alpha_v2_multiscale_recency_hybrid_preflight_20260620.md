# Path Policy Alpha V2 Multiscale Recency Hybrid Preflight 20260620

## Verdict
- Status: `preflight_completed / code_contract_validated / throughput_profiled / research_only / execution_frozen`.
- Date: `2026-06-20`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Scope: prepare the multi-scale recency-aware hybrid experiment before any long formal run.
- Active artifact impact: none. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, or execution-candidate state.

## Architecture Contract

New model family:

```text
hybrid_multiscale_recency_aware_v1
```

Implemented as a single forecast model, not an ensemble of separately trained models.

Input contract:

```text
QDP style_structural_alpha_v2_label_v2 training pack
feature_count: 307
static context used for training: exchange,industry
symbol_id: disabled through --forecast-train-static-fields exchange,industry
output_profile: forecast_path_v1
loss_profile: hybrid_alpha_score_v1
selection_profile: validation_loss
```

Expert routing:

```text
main non-intraday features -> GRU main branch
main non-intraday features -> recency-aware patch Transformer branch
main non-intraday features -> multi half-life EWMA trend branch
intraday_* / cs_rank_intraday_* / cs_z_intraday_* -> short-half-life bottleneck branch
router + 1-layer fusion Transformer -> forecast path heads
```

Confirmed real-pack config from smoke:

```text
intraday_feature_count: 126
main_feature_count: 181
static fields: exchange,industry
intraday examples:
  intraday_first_5m_ret
  cs_rank_intraday_first_5m_ret
  cs_z_intraday_first_5m_ret
```

This directly implements the current generalization-repair idea:
- keep `symbol_id` out to reduce identity memorization risk;
- stop feeding intraday summaries freely into all temporal experts;
- let daily/market/industry/valuation context drive the main sequence encoders;
- let intraday detail enter through a compressed, recency-biased branch.

## Code Changes

Files changed:

```text
daily_research/path_policy/models.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/forecast_checkpoint_topk_reselection.py
daily_research/path_policy/tests/test_models.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

Key implementation points:
- Added `HybridMultiScaleRecencyAwarePath20Forecaster`.
- Added `MultiScaleEWMATrendEncoder`.
- Added `RecencyAwarePatchTransformerPath20Encoder`.
- Added `_intraday_feature_indices(...)` with prefixes:

```text
intraday_
cs_rank_intraday_
cs_z_intraday_
```

- Stored model config fields:

```text
fusion_version
expert_families
recency_half_lives
patch_recency_halflife
intraday_recency_halflife
intraday_bottleneck_dim
intraday_feature_indices
intraday_feature_count
intraday_feature_columns
main_feature_count
```

- Updated checkpoint top-K reselection reconstruction so selected checkpoints can restore the new family and intraday feature indices.

## Speed Optimization

Applied a semantics-preserving EWMA branch optimization:

```text
old: loop over each half-life and compute weighted summary one by one
new: build all half-life weights as one matrix and compute summaries with one batched matmul
```

This keeps the same half-life summaries:

```text
half_lives: 5,10,20,60,120
```

but reduces repeated tensor operations in the EWMA branch.

## Real Throughput Results

All throughput runs used:

```text
model_family: hybrid_multiscale_recency_aware_v1
hidden_dim: 256
gru_layers: 2
transformer_layers: 4
transformer_heads: 8
patch_sizes: 4,20
loss_profile: hybrid_alpha_score_v1
selection_profile: validation_loss
static fields: exchange,industry
device: cuda
per_epoch_prediction_metrics: disabled
```

Short 4096-row train-role tests:

```text
b512 workers=0 before EWMA optimization:
  epoch_seconds: 10.140
  samples_per_second: 403.945

b512 workers=0 after EWMA optimization:
  epoch_seconds: 8.344
  samples_per_second: 490.892

b256 workers=0:
  epoch_seconds: 12.230
  samples_per_second: 334.810

b768 workers=0:
  epoch_seconds: 13.688
  samples_per_second: 299.240

b1024 workers=0:
  epoch_seconds: 33.690
  samples_per_second: 121.590

b512 workers=2:
  epoch_seconds: 25.609
  samples_per_second: 159.944
```

Representative 32768-row train-role tests:

```text
b512 workers=0 before EWMA optimization:
  epoch_seconds: 60.547
  samples_per_second: 541.199
  total_elapsed_seconds: 159.656

b512 workers=0 after EWMA optimization:
  epoch_seconds: 55.953
  samples_per_second: 585.634
  total_elapsed_seconds: 150.922
```

Interpretation:
- Best measured training config is `batch_size=512`, `dataloader_num_workers=0`.
- Windows `num_workers=2` is materially slower for this memmap training path.
- Larger batches are slower on the current RTX 2060 6GB setup.
- EWMA batched optimization improved representative training throughput by about `8.2%` on the 32k run and about `21.5%` on the 4096 run.
- New architecture remains heavier than the old no-symbol hybrid: previous no-symbol full runs reported about `900 samples/s`, while this architecture is about `586 samples/s` after optimization on the 32k representative test.

Estimated full training time:

```text
train rows: 6,050,268
representative throughput: 585.634 samples/s
estimated train-only time per epoch: about 2.87 hours
```

This excludes validation loss, checkpoint writing, and final validation/test prediction output. A long full run should be treated as a major foreground/background training job, not a quick scout.

## Validation

Focused tests:

```text
4 passed in 10.97s
```

Covered:
- new model emits forecast_path_v1 contract and router diagnostics;
- all forecast model families still emit path20 sequence contract;
- training summary/checkpoint records the multiscale recency hybrid contract;
- protocol accepts `hybrid_multiscale_recency_aware_v1`.

Additional checks:

```text
py_compile:
  models.py
  forecast_training.py
  forecast_checkpoint_topk_reselection.py
  run_alpha_path20_protocol.py
  passed

git diff --check:
  passed
```

## Recommended Formal Command

Do not launch a 12-epoch full run blindly. The current recommended next experiment is a smaller but meaningful stride scout, because the active hot path is still generalization repair:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol `
  --stage forecast-walkforward-study `
  --tag qdp_alpha_v2_multiscale_recency_hybrid_stride3_h256_t4_b512_seed7_20260620_01 `
  --data-source lake `
  --lake-dataset-id policy_input_bundle__f926a496f69c61f6b92b5faf `
  --execution-mode next_open `
  --forecast-dataset-mode memmap `
  --forecast-memmap-manifest quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json `
  --forecast-train-start-year 2012 `
  --forecast-train-end-year 2023 `
  --forecast-validation-year 2024 `
  --forecast-test-year 2025 `
  --forecast-model-families hybrid_multiscale_recency_aware_v1 `
  --forecast-include-static-context `
  --forecast-train-static-fields exchange,industry `
  --forecast-output-profile forecast_path_v1 `
  --forecast-loss-profile hybrid_alpha_score_v1 `
  --forecast-selection-profile validation_loss `
  --forecast-cumulative-horizons 1,3,5,10,20 `
  --forecast-horizon 20 `
  --forecast-epochs 6 `
  --forecast-min-epochs 2 `
  --forecast-early-stop-patience 3 `
  --forecast-checkpoint-every-n-epochs 1 `
  --forecast-batch-size 512 `
  --forecast-hidden-dim 256 `
  --forecast-gru-layers 2 `
  --forecast-transformer-layers 4 `
  --forecast-transformer-heads 8 `
  --forecast-patch-sizes 4,20 `
  --forecast-train-date-stride 3 `
  --forecast-device cuda `
  --forecast-dataloader-num-workers 0 `
  --forecast-prefetch-factor 2 `
  --no-forecast-per-epoch-prediction-metrics
```

Rationale:
- `train_date_stride=3` directly tests the high-overlap hypothesis from the current hot-path plan.
- Validation/test remain full.
- `epochs=6` is enough to observe whether validation/test loss immediately degrades after epoch1/2 without committing to a 12-epoch full run.
- Per-epoch prediction metrics stay disabled; final best checkpoint still produces validation/test predictions.

Only if this scout shows better validation/test loss dynamics should we consider:

```text
train_date_stride=5
or a no-stride full h256/T4 run with this architecture
or capacity/regularization variants
```

## Boundaries

- This reference is not model-quality evidence.
- Throughput/smoke runs do not replace full validation/test evidence.
- The old alpha_v2 anchor remains `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
- Do not enter multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default artifact changes from this preflight.
- Do not interpret test-only candidate selection from smoke/throughput runs as a model conclusion.

