# Minute-strategy numeric-kernel benchmark

This record isolates the element-wise arithmetic used after causal hourly
history has already been prepared. It is a feasibility measurement for a
future GPU path, not a replacement for the CPU/ DuckDB state pipeline.

## Scope

The kernel computes live-MA arithmetic, distance-to-intersection features, and
two boolean masks. It deliberately excludes Parquet reads, rolling/grouped
operations, event state transitions, string/date joins, and portfolio logic.
Those excluded operations dominate the end-to-end study and are not moved to
the GPU by this experiment.

The reference and optional Torch implementations are in
`src/quantlab/research/minute_strategy_numeric.py`. Torch is imported lazily;
the production runner is unchanged and does not require Torch.

## Environment and method

- CPU: AMD Ryzen 7 4800H, 8 physical / 16 logical cores.
- Memory: 15.42 GiB; the 0.5 GiB machine reserve was checked before and after
  each batch.
- GPU: NVIDIA GeForce RTX 2060, 6 GiB, compute capability 7.5.
- Torch: 2.5.1 with CUDA 12.4.
- NumPy and Torch used `float32` arrays; each size had two warm-ups and five
  timed repetitions.
- `gpu_cuda_median_seconds` keeps inputs on the device and measures arithmetic
  plus synchronization. `gpu_cuda_full_transfer_median_seconds` includes both
  host-to-device input copies and device-to-host output copies.

## Results

| rows | NumPy median (s) | CUDA arithmetic (s) | CUDA full transfer (s) | arithmetic speedup | full-transfer speedup | GPU allocated (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262,144 | 0.008455 | 0.000669 | 0.007245 | 12.65x | 1.17x | 15.5 |
| 1,048,576 | 0.040264 | 0.001215 | 0.017388 | 33.14x | 2.31x | 62.0 |
| 4,194,304 | 0.136433 | 0.004448 | 0.051554 | 30.67x | 2.65x | 248.0 |

The Torch results matched the NumPy reference on a 4,096-row prefix (maximum
absolute numeric error `0.0`; boolean masks identical). The largest batch
left more than 4.9 GiB of system memory available.

## Decision

The isolated arithmetic is GPU-friendly. Once realistic host/device transfers
are included, the largest tested batch was about 2.65x faster than NumPy (the
kernel alone was about 30x faster); small batches were close to break-even.
That is useful for a future batched numeric stage, but not sufficient reason to
move the minute strategy runner wholesale to CUDA. Rolling history, event
state, and Parquet/data-frame work remain CPU-side. No GPU dependency or
production path was enabled by this benchmark.
