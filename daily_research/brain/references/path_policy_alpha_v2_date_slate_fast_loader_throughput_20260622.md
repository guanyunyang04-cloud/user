# Path Policy Alpha V2 Date-Slate Fast Loader Throughput 20260622

## Verdict
- Status: `code_contract_validated / throughput_scout_completed / research_only / execution_frozen`.
- Date: `2026-06-22`.
- Scope: first fast date-slate loader/index implementation and CUDA throughput scout for `date_slate_cross_stock_alpha_fusion_v1`.
- Active artifact impact: none. No `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, promotion gate, QDP canonical data, QDP registry pointer, or active artifact was changed.

## Implemented Loader And Scout Tooling

Implemented without TB-scale rolling-window materialization:

```text
ForecastDateSlateTorchDataset fast index:
  date -> row_idx / global_date_pos / global_stock_pos chunks
  __getitem__ -> date_input_windows_from_positions(...)
  avoids per-batch pandas iloc/factorize

ForecastTrainingPackDataset:
  raw_date_input_windows_from_positions(...)
  date_input_windows_from_positions(...)
```

Added throughput-only training controls:

```text
--forecast-train-only
--forecast-max-train-steps-per-epoch
--forecast-skip-validation-loss
```

These are diagnostic controls. Formal training defaults are unchanged.

Also fixed an AMP dtype bug in the cross-stock slate mixer:

```text
DateSlateCrossStockAlphaFusionV1Forecaster._slate_context(...)
```

Root cause: autocast could create fp16 destination `context` while the slate mixer returned fp32; assignment now casts the mixed context to destination dtype/device.

## Throughput Scout Setup

Common setup:

```text
manifest: quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_date_slate_pack_20260621_01/qdp_date_slate_training_pack_manifest.json
source_market_dataset_id: policy_input_bundle__f926a496f69c61f6b92b5faf
model_family: date_slate_cross_stock_alpha_fusion_v1
output_profile: forecast_incremental_path_v2
loss_profile: date_listwise_alpha_score_v2
static context: exchange,industry
hidden_dim: 128
transformer_layers: 4
transformer_heads: 4
dates_per_batch: 1
train_date_stride: 60
device: cuda / RTX 2060 6GB
AMP: enabled for bounded probes
train_only: true
skip_validation_loss: true
```

Important semantic note:

```text
stocks_per_date is a per-date chunk cap, not a guarantee that every step has exactly that many rows.
With train_date_stride=60, sampled train date sizes ranged from 803 to 3053 rows.
For cap=4096 this is effectively full sampled date; for cap=2048 large dates split into a 2048 chunk plus a remainder.
```

## Results

Bounded AMP probes:

| Probe | Result | Actual rows processed | Time | Throughput | Interpretation |
|---|---:|---:|---:|---:|---|
| `1x512`, 1 step | completed | 512 | 11.516s | 44.460 samples/s | single step feasible |
| `1x512`, 8 steps | completed | 3724 | 97.422s | 38.225 samples/s | short continuous probe feasible |
| `1x1024`, 1 step | completed | 574 | 15.234s | 37.679 samples/s | first sampled chunk feasible but not a full 1024-row stress step |
| `1x1024`, 2 steps | OOM | none completed | n/a | n/a | not sustainable at this config |
| `1x2048`, 1 step | OOM | none completed | n/a | n/a | first-step infeasible |
| `1x4096`, 1 step | OOM | none completed | n/a | n/a | first-step infeasible |

Longer non-bounded attempts:

```text
1x512 FP32:
  OOM at step 94/224
  last progress: 41751 rows, 28.367 samples/s

1x512 AMP:
  OOM at step 20/224 before dtype fix was fully useful for long run
  last progress: 9371 rows, 49.163 samples/s
```

## Conclusion

The fast date-slate loader/index is valid and faster/cleaner semantically, but the current h128/t4 cross-stock model is memory-bound on RTX 2060 6GB:

```text
usable bounded diagnostic: 1x512
not safe for long formal training: 1x512 h128/t4
not sustainable: 1x1024
first-step infeasible: 1x2048 / 1x4096
```

Therefore full-date/full-market slate training is not feasible on the current GPU with this architecture and capacity. Any immediate training must be labeled sampled-chunk context, not full-market context.

Next engineering direction:

```text
1. Keep fast date-slate loader.
2. Do not run formal long training at 1x1024+ on RTX 2060 6GB.
3. If continuing cross-stock training now, use 1x256 or bounded 1x512 diagnostics only, with explicit sampled-chunk semantics.
4. For semantically cleaner sampled chunks, implement/compare sampled_chunk_full_self_attention vs sampled_chunk_low_rank_slot.
5. For true full-date semantics, reduce architecture memory materially or move to a larger GPU; do not label current sampled chunks as full-market.
```

## Validation

Passed:

```text
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_forecast_memmap_dataset.py daily_research/path_policy/tests/test_forecast_training.py -q
69 passed, 96 warnings

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_models.py daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_accepts_date_slate_cross_stock_alpha_fusion_v1_contract -q
20 passed

C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m py_compile daily_research/path_policy/models.py daily_research/path_policy/forecast_dataset.py daily_research/path_policy/forecast_training.py daily_research/path_policy/run_alpha_path20_protocol.py daily_research/path_policy/tests/test_models.py daily_research/path_policy/tests/test_forecast_memmap_dataset.py daily_research/path_policy/tests/test_forecast_training.py
pass

git diff --check
pass
```

Warnings are existing tiny synthetic constant-input metric warnings in tests, not contract failures.

## Boundaries

This is not model-quality evidence and not an execution candidate.

Not authorized by this reference:

```text
multi-seed
score-backtest bridge
candidate matrix
execution-candidate review
paper/live/broker
active/default artifact changes
QDP registry pointer changes
old data deletion
```
