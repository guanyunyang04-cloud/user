# Path Policy Alpha V2 Structured Alpha V2 Preflight 20260621

## Verdict
- Status: `preflight_completed / code_contract_validated / real_pack_smoked / throughput_profiled / research_only / execution_frozen`.
- Date: `2026-06-21`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- New model family: `hybrid_structured_alpha_v2`.
- Active artifact impact: none. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, or execution-candidate state.

## Architecture Contract

`hybrid_structured_alpha_v2` replaces the earlier v1 scout as the next hot-path architecture. It is a prediction-first daily alpha model, not an execution model and not a raw 5-minute sequence model.

Key semantic decisions:
- Hidden dimension default remains `192`, per user request.
- `symbol_id` is forbidden. Training must use `--forecast-train-static-fields exchange,industry`.
- Output is restricted to `forecast_path_v1`.
- Execution losses are forbidden for this family. Use prediction-first losses such as `hybrid_alpha_score_v1` or `forecast_path_v1_baseline`.
- Intraday features are not a peer expert vote. They enter as a decayed short-horizon residual correction.
- Static/regime context conditions the sequence and experts through FiLM-style gates instead of only being appended to the final head.

Implemented hierarchy:

```text
307 alpha_v2 features
  -> deterministic feature groups
  -> per-group encoders
  -> group mixer and group weights
  -> context token from market/regime + industry/peer + event/quality + static exchange/industry
  -> FiLM-conditioned sequence
  -> four temporal experts:
       GRU continuity
       recency-biased multiscale Patch Transformer
       multi-half-life EWMA trend
       local dilated TCN
  -> router + 2-layer fusion Transformer
  -> structured forecast_path_v1 heads
  -> intraday residual correction with horizon-decay mask
```

Real-pack feature group counts from `style_structural_alpha_v2_label_v2`:

```text
daily_price_volume:    61
cross_section:         14
market_regime:         21
industry_peer:         19
valuation_liquidity:   47
event_quality:         19
intraday:             126
total:                307
```

## Code Changes

Files changed:

```text
daily_research/path_policy/models.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/forecast_checkpoint_topk_reselection.py
daily_research/path_policy/run_alpha_path20_protocol.py
daily_research/path_policy/tests/test_models.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

Key implementation points:
- Added `HybridStructuredAlphaV2Forecaster`.
- Added deterministic `_structured_alpha_v2_feature_groups(...)`.
- Added structured feature group contract into checkpoint `model_config` and `resume_contract`.
- Checkpoint top-K reselection now restores `feature_group_indices`.
- Protocol rejects `hybrid_structured_alpha_v2` runs that include `symbol`, omit `exchange/industry`, request `decision_utility_v1`, or use execution-style loss such as `personal_time_efficient_topk_v1`.

## Real Pack Smoke

Final smoke tag:

```text
qdp_alpha_v2_structured_alpha_v2_contract_smoke_20260621_03
```

Command shape:

```text
model_family: hybrid_structured_alpha_v2
training pack: mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01
static fields: exchange,industry
output_profile: forecast_path_v1
loss_profile: hybrid_alpha_score_v1
selection_profile: validation_loss
max_samples_per_role: 32
device: cpu
hidden_dim: 24
```

Result:

```text
status: completed
selected_model_family: hybrid_structured_alpha_v2
selected_seed: 7
best_epoch: 1
best_validation_loss: 3.972476363182068
```

This smoke proves code/data compatibility only. It is not model-quality evidence.

## Throughput Preflight

Throughput tags:

```text
qdp_alpha_v2_structured_alpha_v2_throughput_h192_b128_1k_20260621_01
qdp_alpha_v2_structured_alpha_v2_throughput_h192_b256_1k_20260621_01
```

Config:

```text
hidden_dim: 192
gru_layers: 2
transformer_layers: 4
transformer_heads: 6
patch_sizes: 3,5,20
device: cuda
dataloader_num_workers: 0
max_samples_per_role: 1024
per_epoch_prediction_metrics: disabled
```

Measured results on RTX 2060 6GB:

```text
batch_size=128:
  epoch_seconds: 8.078
  samples_per_second: 126.764
  total_elapsed_seconds: 50.828

batch_size=256:
  epoch_seconds: 20.453
  samples_per_second: 50.066
  total_elapsed_seconds: 65.891
```

Interpretation:
- `batch_size=128` is the current safe throughput config.
- `batch_size=256` is materially slower, likely memory pressure / kernel inefficiency on this GPU.
- The complete v2 architecture is much heavier than `hybrid_multiscale_recency_aware_v1`; this is expected because v2 adds group encoders, group mixer, context conditioning, TCN, 2-layer fusion, and structured heads.

Rough train-only full epoch estimate using b128:

```text
train rows: 6,050,268
throughput: 126.764 samples/s
train-only epoch estimate: about 13.25 hours
```

That excludes validation loss, checkpoint I/O, and final validation/test prediction output. A full no-stride run is not recommended as the next step.

## Validation

Focused tests:

```text
18 passed in 11.53s
```

Covered:
- v2 forecast output contract and diagnostics;
- symbol static context rejection;
- all forecast families still emit path20 sequence contract;
- v2 training/checkpoint contract and checkpoint reload;
- protocol accepts correct v2 command and rejects symbol / execution-loss misuse.

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

active_execution_strategy diff:
  empty
```

## Recommended Next Experiment

Because h192 full no-stride is too slow, the next useful experiment should combine complete v2 with overlap-aware sampling:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol `
  --stage forecast-walkforward-study `
  --tag qdp_alpha_v2_structured_alpha_v2_stride3_h192_b128_seed7_20260621_01 `
  --data-source lake `
  --lake-dataset-id policy_input_bundle__f926a496f69c61f6b92b5faf `
  --execution-mode next_open `
  --forecast-dataset-mode memmap `
  --forecast-memmap-manifest quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json `
  --forecast-train-start-year 2012 `
  --forecast-train-end-year 2023 `
  --forecast-validation-year 2024 `
  --forecast-test-year 2025 `
  --forecast-model-families hybrid_structured_alpha_v2 `
  --forecast-include-static-context `
  --forecast-train-static-fields exchange,industry `
  --forecast-output-profile forecast_path_v1 `
  --forecast-loss-profile hybrid_alpha_score_v1 `
  --forecast-selection-profile validation_loss `
  --forecast-cumulative-horizons 1,3,5,10,20 `
  --forecast-horizon 20 `
  --forecast-epochs 4 `
  --forecast-min-epochs 1 `
  --forecast-early-stop-patience 2 `
  --forecast-checkpoint-every-n-epochs 1 `
  --forecast-batch-size 128 `
  --forecast-hidden-dim 192 `
  --forecast-gru-layers 2 `
  --forecast-transformer-layers 4 `
  --forecast-transformer-heads 6 `
  --forecast-patch-sizes 3,5,20 `
  --forecast-train-date-stride 3 `
  --forecast-device cuda `
  --forecast-dataloader-num-workers 0 `
  --forecast-prefetch-factor 2 `
  --no-forecast-per-epoch-prediction-metrics
```

Rationale:
- Tests the complete user-requested architecture.
- Keeps validation/test full.
- Keeps `symbol_id` out.
- Keeps model prediction-first.
- Uses overlap-aware sampling to reduce near-duplicate 252d-window overfitting and control runtime.

Do not launch a blind 12-epoch full no-stride h192 run unless this stride3 scout shows useful validation/test dynamics or the user explicitly accepts the runtime cost.

## Boundaries

- This is not model-quality evidence.
- The old alpha_v2 anchor remains `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
- `hybrid_multiscale_recency_aware_v1` remains available as a lighter scout, but the active next hot-path architecture is now `hybrid_structured_alpha_v2`.
- Do not enter multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default artifact changes from this preflight.
