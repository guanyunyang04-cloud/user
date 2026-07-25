# Seq100 signal-quality runtime optimization — 2026-07-25

## Scope

- Study: `seq100_pit_signal_quality_v1`.
- This work optimized and hardened the neural-model training runtime only.
- `screen-models` was not resumed, no formal matrix or formal fold was created,
  and no QDP dataset, PIT pack, registered model, or active execution object was
  modified.
- Full process evidence is under the ignored output at
  `daily_research/output/path_policy/studies/seq100_pit_signal_quality_v1/runtime_optimization/summary.json`.

## Execution semantics

- TabM, PatchTST, and DeepHit now use deterministic cross-date optimizer-batch
  plans. Every optimizer update is bound to the frozen effective batch, except
  the unavoidable final short batch.
- Quality, path, terminal, and fill components use the complete optimizer-batch
  denominator. Changing the resource micro-batch no longer changes those loss
  weights.
- DeepHit ranking is computed over the complete frozen effective batch rather
  than the incidental loader micro-batch.
- Deep Sets keeps each signal-date cross-section atomic and packs only complete
  dates toward the frozen candidate target.
- NeuralNDCG keeps deterministic complete 256-item lists.
- Each neural adapter has an execution-semantics version in its resolved config
  and resume key. The old completed TabM prescreen is therefore invalid and must
  rerun. The LightGBM preflight and prescreen remain hash-valid and reusable.

## Runtime implementation

- PatchTST uses the audited explicit Student-t NLL formula, fused AdamW, exact
  global normalization, cross-date F0 loading, and two-batch CPU prefetch.
- Non-relational prediction and calibration batches may cross date boundaries;
  Deep Sets remains complete-date atomic.
- Real-hardware preflight autotunes only legal resource batch sizes, records
  actual sustained throughput and peak CUDA memory, and requires 15% memory
  headroom.
- Training writes atomic `progress.json` heartbeats at most 30 seconds apart,
  appends `training_events.jsonl`, reports samples/s, ETA, loss, epoch, patience,
  CPU/RAM/GPU diagnostics, and emits event-only console updates.
- `last_checkpoint.pt` is saved every 10 minutes and at every epoch boundary. It
  contains model, optimizer, scaler, RNG, best state, patience, epoch, exact step
  cursor, and resume binding. A `pause.request` sentinel pauses only after a
  durable optimizer-boundary checkpoint.

## RTX 2060 selections

| Model | Train batch semantics | Validation | Prediction | Measured train throughput |
|---|---:|---:|---:|---:|
| PatchTST Student-t | micro 128 / effective 512 | 1024 | 1024 | 2,256.90/s autotune |
| TabM | micro 2048 / effective 4096 | 2048 | 2048 | 136,085.16/s |
| DeepHit | micro 4096 / effective 4096 | 4096 | 8192 | 44,673.76/s |
| Market/industry Deep Sets | complete dates / target 2048 | complete date | complete date | 58,059.96/s |
| NeuralNDCG | complete list 256 | 4096 calibration | 4096 | 18,639.57/s |

The stricter PatchTST 100k gate completed three epochs after a deliberate safe
pause and resume. Its last and median observed training rates were 2,188.86/s
and 2,348.90/s, the maximum heartbeat gap was 30 seconds, both final and rolling
checkpoints existed, and the resume key matched.

## Validation and protection

- Focused tests: `40 passed` across the signal-quality and reusable
  path-relevance suites.
- All five neural adapters passed real-pack preflight and complete tiny-fit
  checkpoint tests; PatchTST passed the 100k sustained pause/resume gate.
- Protected hashes after all tests:
  - QDP active manifest:
    `ff39ae379d6edb09695430e7b0093c19d8ee2471b6b9bf1b5dd8492e5a23c8ff`
  - Registered model registry:
    `54c4ac4617d440a238a40087cf07682c8a5dda7d895b782459ae22daadbf3475`

## Resume rule

Do not automatically start model screening. On explicit resume, reuse the
existing LightGBM result, invalidate and rerun old TabM under v2 semantics, run
new autotuned neural preflights inside the model-screen attempt, and preserve all
older interrupted output directories.
