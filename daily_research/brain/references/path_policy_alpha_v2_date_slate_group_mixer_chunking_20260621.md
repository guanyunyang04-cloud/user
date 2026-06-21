# Path Policy Alpha V2 Date-Slate Group Mixer Chunking 20260621

## Verdict
- Status: `throughput_blocker_repaired / large_slate_shape_unblocked / throughput_not_improved / research_only / execution_frozen`.
- Scope: `date_slate_alpha_fusion_v1` and `hybrid_structured_alpha_v2` group-mixer forward path.
- Active artifact impact: none. No `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, QDP canonical data, or QDP registry active pointer was changed.

## Code Change

Files changed:

```text
daily_research/path_policy/models.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_models.py
```

Implementation:

```text
DEFAULT_GROUP_MIXER_CHUNK_SIZE = 32768
_chunked_sequence_module(module, x, chunk_size)
HybridStructuredAlphaV2Forecaster(..., group_mixer_chunk_size=...)
DateSlateAlphaFusionV1Forecaster(..., group_mixer_chunk_size=...)
```

The group mixer previously ran:

```text
group_mixer(group_tokens.reshape(batch * steps, groups, hidden_dim))
```

The repair chunks only the flattened `batch * steps` dimension before applying the same `group_mixer`, then concatenates outputs and reshapes back. This is semantically equivalent because `group_mixer` attends only across feature-group tokens inside each independent stock-date-step row. It does not mix across dates, stocks, or time rows. Loss, outputs, labels, active artifacts, and data semantics are unchanged.

`forecast_training.py` now records `group_mixer_chunk_size` in model config for new checkpoints. Resume compatibility treats older checkpoints without this field as default-equivalent when the expected value is the default.

## Verification

Focused tests:

```text
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest \
  daily_research/path_policy/tests/test_models.py::test_structured_alpha_v2_group_mixer_chunking_is_equivalent \
  daily_research/path_policy/tests/test_models.py::test_date_slate_alpha_fusion_v1_group_mixer_chunking_is_equivalent_and_large_batch_safe \
  daily_research/path_policy/tests/test_forecast_training.py::test_make_forecast_model_registers_date_slate_alpha_fusion_v1 -q

3 passed in 10.43s
```

Additional checks:

```text
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m py_compile daily_research/path_policy/models.py daily_research/path_policy/forecast_training.py
git diff --check
```

Both passed.

## CUDA Scout Results

### 1x512 After Chunking

Run:

```text
qdp_alpha_v2_date_slate_cuda_chunked_throughput_d128_noamp_dense_stride20_1x512_b64_20260621_01
```

Result:

```text
status: completed
phase: completed
shape: 1 date x 512 stocks
chunk_size: 32768
amp: off
forecast_batch_size: 64
train_date_stride: 20
max_samples_per_date_per_role: 512
train_sample_count: 74,240
epoch_seconds: 2377.469
train_samples_per_second: 31.226
best_validation_loss: 2.562649487909688
validation rank_ic20: 0.041676510640682105
test rank_ic20: 0.08008681514309897
```

Interpretation:

```text
CUDA invalid configuration argument is repaired for 1x512.
Throughput is worse than the prior 1x256 safe baseline, so 1x512 is not the next formal training recommendation.
```

### 2x256 After Chunking

Run:

```text
qdp_alpha_v2_date_slate_cuda_chunked32768_throughput_d128_noamp_stride60_2x256_b64_20260621_01
```

Result:

```text
status: completed
phase: completed
shape: 2 dates x 256 stocks
chunk_size: 32768
amp: off
forecast_batch_size: 64
train_date_stride: 60
max_samples_per_date_per_role: 256
train_sample_count: 12,544
epoch_seconds: 532.985
train_samples_per_second: 23.535
best_validation_loss: 2.6296334762918465
validation rank_ic20: 0.033697868820011155
test rank_ic20: 0.07616615882312175
```

Interpretation:

```text
CUDA invalid configuration argument is repaired for 2x256.
Throughput is also worse than 1x256. The date-slate formal run should not switch to 2x256 for speed.
```

### Rejected Larger Chunk

Diagnostic attempt:

```text
DEFAULT_GROUP_MIXER_CHUNK_SIZE = 65000
qdp_alpha_v2_date_slate_cuda_chunked65000_throughput_d128_noamp_stride60_1x512_b64_20260621_01
```

Result:

```text
failed with CUDA out of memory during backward
Tried to allocate about 3.91 GiB on RTX 2060 6GB
```

Interpretation:

```text
65000 is not safe as the default chunk size on the current GPU. The default was reverted to 32768.
```

## Current Decision

The chunking repair is useful because it removes the shape crash and makes larger slate diagnostics possible. It is not a throughput win on RTX 2060 6GB under the current model.

Current fastest confirmed formal-training candidate remains the earlier safe configuration:

```text
D=128
1 date x 256 stocks
no AMP
forecast_batch_size=64
finite_guard enabled
group_mixer_chunk_size=32768
```

Known baseline from `path_policy_alpha_v2_date_slate_cuda_throughput_scout_20260621.md`:

```text
qdp_alpha_v2_date_slate_cuda_throughput_d128_noamp_dense_stride20_1x256_b64_20260621_01
train_samples_per_second: 119.615
epoch_seconds: 310.328 for 37,120 train rows
```

## Next Allowed Actions

1. For immediate formal research training, use `1x256 / no AMP / forecast_batch_size=64 / finite_guard / group_mixer_chunk_size=32768`.
2. Do not use `1x512`, `2x256`, or `group_mixer_chunk_size=65000` as speed settings on RTX 2060 6GB.
3. If more speed is required before formal training, optimize outside the group-mixer chunk size first: date-slate loader locality, validation-loss evaluation batching, full-date sampling policy, or a lighter group mixer.
4. AMP remains disabled until scaler/normalization stability is repaired and finite-guard scout passes.
5. This is throughput/stability evidence only, not model-quality evidence, multi-seed evidence, bridge evidence, candidate matrix evidence, or execution evidence.

