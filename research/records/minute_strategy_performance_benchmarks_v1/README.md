# Minute-strategy performance benchmarks

These are bounded engineering benchmarks for the causal minute-strategy
pipeline. They measure execution cost and memory headroom only; they do not
change the research sample or establish a trading result. The machine-wide
hard reserve remains 0.5 GiB.

## Environment

- CPU: AMD Ryzen 7 4800H, 8 physical / 16 logical cores.
- Memory: 15.42 GiB total.
- GPU: NVIDIA RTX 2060, 6 GiB; no GPU path was used for these benchmarks.
- Source data: local QDP Parquet on `H:` (USB SSD).
- Python: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`.

## Thread and chunk smoke matrix

Each run used one target date (`2022-06-16`), 150 point-in-time symbols, all
six MA periods, no normalized-artifact write, and a 0.5 GiB memory floor. Wall
time includes process startup and final summary; the runner's month elapsed
field is the useful internal comparison.

| DuckDB threads | Symbol chunk | Internal seconds | Wall seconds | Minimum available GiB | Outcome rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 49.24 | 82.477 | 4.757 | 19,355 |
| 1 | 128 | 41.403 | 74.960 | 4.782 | 19,355 |
| 2 | 32 | 39.923 | 71.192 | 4.781 | 19,355 |
| 2 | 128 | 32.402 | 63.429 | 4.808 | 19,355 |
| 4 | 64 | 30.811 | 61.924 | 4.816 | 19,355 |
| 4 | 128 | **28.025** | **59.176** | 4.792 | 19,355 |

Within this bounded workload, four DuckDB threads and a 128-symbol chunk were
the fastest tested setting. The result is not a claim that four threads is
optimal for a full three-year run; available RAM and concurrent system load
will change the safe point.

## Date parallelism smoke test

Two independent one-date runner processes used 100 symbols, four MA periods,
two DuckDB threads and a 128-symbol chunk:

- sequential: 107.568 seconds, minimum available memory 6.195 GiB;
- two concurrent processes: 60.181 seconds, minimum available memory
  3.791 GiB;
- observed speedup: 1.79x.

This demonstrates useful I/O/CPU overlap, but it is not yet enabled in the main
runner. A full date worker would duplicate month-level context/cache memory,
complicate checkpoint ordering, and has not been validated at the 3,000-symbol
universe. It should be introduced only with an explicit worker cap and a
machine-memory admission check.

## Normalized artifact conversion and reads

The existing April 2022 and September 2023 wide outputs were converted
date-at-a-time into `events`, `paths`, and `references` tables. The conversion
took 97.751 seconds and 136.334 seconds respectively. It reduced on-disk size
to about 88.0% and 85.7% of the corresponding wide outputs, and reduced the
number of stored forward price paths from 6,113,466 to 2,511,217 in April and
from 8,864,129 to 3,180,006 in September.

The normalized reader joins the event table to the unique path table. On one
April date with all strategies, median read times were:

- legacy wide Parquet: 0.427 seconds, 31 columns;
- normalized event/path join: 1.157 seconds, 34 columns.

The join is therefore not a single-read speed optimization. Its value is that
forward outcomes are computed and stored once per path, while multiple rules,
rankers, seeds and exit analyses can reference the same path. A portfolio
parity replay over a 30-symbol subset produced identical account results from
wide and normalized inputs.

Moving DuckDB's temporary directory from the default `H:` workspace spill path
to an internal NVMe temporary directory did not materially change this
workload: the same 150-symbol/4-thread/128-symbol-chunk smoke took 28.668
internal seconds on `H:` versus 28.492 seconds on the NVMe path. Keeping the
default is therefore acceptable until a workload actually spills heavily.

## Decision

Use four DuckDB threads and a 128-symbol chunk as the current benchmark
starting point, subject to the existing memory guard. Keep date parallelism as
an audited option rather than enabling it by default; the parent-side admission
policy is documented in
`research/records/minute_strategy_date_parallel_admission_v1/`. Keep
normalized artifacts enabled for new development runs and use the
compatibility-wide files for old runs until their three tables are complete.
The isolated numeric kernel is GPU-friendly, but the full-transfer benchmark
shows only about 2.65x at a large batch and does not justify moving the whole
pipeline to CUDA; details are in
`research/records/minute_strategy_numeric_kernel_v1/`.

## Reproduction material

The disposable benchmark outputs are under the ignored `tmp/` directory:

- `tmp/perf_matrix/results.json`;
- `tmp/perf_date_parallel/results.json`.
- `tmp/perf_tempdir/results.json`.

The persistent normalized artifacts used for the read/parity checks are under
the ignored `runs/minute_ma_month_2022_04/normalized/` and
`runs/minute_ma_month_2023_09/normalized/` directories.

## Full-month follow-up

After the initial four-thread full-universe attempt exposed a DuckDB history
aggregation OOM, the loader was changed to remove its redundant global SQL
sort and aggregate historical minutes in bounded symbol chunks. A complete
April 2022 run with four threads and a 128-symbol chunk then finished in
7,005.635 wall seconds (19 trading dates, 6,091,670 outcome rows). The memory
curve bottomed at 3.552 GiB available, with a 3.399 GiB peak process-tree RSS.
Details and the raw telemetry paths are recorded in
`research/records/minute_strategy_full_month_benchmark_v1/`.

The same-source one-date replay with two threads and a 512-symbol chunk
matched the four-thread/128-symbol run on all 70 output columns for all
393,275 rows. This is a capacity and semantic check, not a profitability
result.
