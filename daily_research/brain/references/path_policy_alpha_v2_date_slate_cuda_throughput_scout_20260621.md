# Path Policy Alpha V2 Date-Slate CUDA Throughput Scout 20260621

## Verdict
- Status: `cuda_throughput_scout_completed / stability_constraints_found / research_only / execution_frozen`.
- Date: `2026-06-21`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Model family: `date_slate_alpha_fusion_v1`.
- Output profile: `forecast_incremental_path_v2`.
- Loss profile: `date_grouped_alpha_score_v1`.
- Active artifact impact: none. No `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, QDP canonical data, or QDP registry active pointer was changed.

## Fixed Contract

All successful/diagnostic commands used:

```text
manifest: quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_date_slate_pack_20260621_01/qdp_date_slate_training_pack_manifest.json
train years: 2012-2023
validation year: 2024
test year: 2025
hidden_dim: 128
transformer_layers: 4
transformer_heads: 4
static context: exchange,industry
finite_guard: enabled
per_epoch_prediction_metrics: disabled
device: cuda / RTX 2060 6GB
```

## Runs

| Run tag | Shape | AMP | Prediction batch | Sampling | Status | Key result |
|---|---:|---:|---:|---|---|---|
| `qdp_alpha_v2_date_slate_cuda_throughput_d128_1x256_20260621_01` | `1x256` | on | `512` | sparse `max_samples_per_role=32768` | failed | finite guard stopped at epoch1 step579: `nonfinite_grad.base_head.5.weight`; `amp_scale=65536`; sparse cap was not representative for slate throughput |
| `qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_1x256_20260621_01` | `1x256` | off | `512` | dense per-date cap 256, train_date_stride 20 | partial | training epoch completed, but final validation prediction failed with CUDA `invalid configuration argument` from large prediction batch |
| `qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_1x256_b64_20260621_01` | `1x256` | off | `64` | dense per-date cap 256, train_date_stride 20 | completed | stable full train + final validation/test prediction |
| `qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_1x512_b64_20260621_01` | `1x512` | off | `64` | dense per-date cap 512, train_date_stride 20 | failed | CUDA `invalid configuration argument` in group-mixer attention during train forward |
| `qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_2x256_b64_20260621_01` | `2x256` | off | `64` | dense per-date cap 256, train_date_stride 20 | failed | same CUDA `invalid configuration argument`; effective batch 512 is unsafe |
| `qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_1x384_b64_20260621_01` | `1x384` | off | `64` | dense per-date cap 384, train_date_stride 20 | failed | same CUDA `invalid configuration argument`; effective batch 384 is unsafe |

## Completed Configuration

Completed run:

```text
run_tag: qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_1x256_b64_20260621_01
status: completed
device: cuda
amp_enabled: false
effective_batch_size: 256
date_slate_dates_per_batch: 1
date_slate_stocks_per_date: 256
forecast_batch_size: 64
train_date_stride: 20
max_samples_per_date_per_role: 256
train_rows: 37,120
validation_rows: 56,576
test_rows: 56,832
epoch_seconds: 310.328
train_samples_per_second: 119.615
total_elapsed_seconds: 533.703
best_epoch: 1
best_validation_loss: 2.594799554725578
finite_guard: enabled
bad_batch_dump: none
```

Validation/test metrics from this scout are diagnostic only because train_date_stride=20 and this is a 1-epoch throughput run:

```text
validation rank_ic20: 0.041554867897557704
validation spread20: 0.0017373022495346991
test rank_ic20: 0.08106624639886126
test spread20: 0.01013667347645803
```

## Findings

1. `--forecast-max-samples-per-role` is not appropriate for representative date-slate throughput. It makes each date sparse and produced only about 13.44 samples/s in the failed AMP run because the loader still iterated many dates with very few stocks per date.

2. AMP is not currently safe for this contract on RTX 2060. The first CUDA AMP run tripped finite guard on a nonfinite gradient with `amp_scale=65536`. This is useful evidence that finite guard is working, but AMP should not be used for the next formal run unless scaler/normalization stability is addressed first.

3. Effective batch size above 256 is currently unsafe. `1x384`, `1x512`, and `2x256` all fail in `DateSlateAlphaFusionForecaster._group_sequence`, specifically at `group_mixer(group_tokens.reshape(batch * steps, groups, hidden_dim))`, where the flattened `batch * steps` dimension becomes too large for the current CUDA attention kernel path.

4. Final prediction also needs a smaller batch than training slate size. `forecast_batch_size=512` failed during validation prediction; `forecast_batch_size=64` completed.

## Current Safe Configuration

For the next run under current code, use:

```text
--forecast-device cuda
--no-forecast-amp
--forecast-hidden-dim 128
--forecast-transformer-heads 4
--forecast-transformer-layers 4
--forecast-dates-per-batch 1
--forecast-stocks-per-date 256
--forecast-batch-size 64
--forecast-rank-max-pairs-per-date 1024
--forecast-finite-guard
--no-forecast-per-epoch-prediction-metrics
```

If using dense per-date capped training for a controlled first formal run:

```text
--forecast-max-samples-per-date-per-role 256
```

Expected train-only runtime for all train dates with this per-date cap is roughly:

```text
740,864 rows / 119.615 samples/s = about 1.72 hours per epoch
```

Without per-date cap, runtime may be far higher because the model would process many more stock rows per date.

## Recommended Next Actions

1. Do not launch formal training with AMP yet.
2. Do not use effective slate batch 384 or 512 until the group-mixer path is chunked or replaced.
3. Decide whether the first formal run should use per-date cap 256 as a practical research run, or whether to first implement group-mixer chunking and/or a date-stock sampling policy that keeps full-date semantics cleaner.
4. If launching immediately under current code, use `1x256 / no AMP / forecast_batch_size=64 / finite_guard`.
5. Treat all metrics here as throughput/stability evidence only, not model-quality evidence, multi-seed evidence, bridge evidence, candidate matrix evidence, or execution evidence.
