# Path Policy Alpha V2 Train Date Stride Tooling 20260620

## Verdict
- Status: `tooling_completed / no_training_run / research_only / execution_frozen`.
- Date: `2026-06-20`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Stage: `Stage 1 overlap-aware sampling diagnostics`.
- Active artifact impact: none. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, or execution-candidate state.

## What Changed

Implemented a runtime training sampler switch:

```text
--forecast-train-date-stride N
```

Semantics:
- Applies only to the `train` role.
- Keeps every `N`th train trading date.
- Keeps all stocks for each kept train date.
- Leaves `validation` and `test` full.
- Does not rebuild or mutate QDP memmap/training-pack data.
- Records sampling metadata into `forecast_training_summary.json`, family summaries, checkpoints, and resume contracts.

Python API:

```text
train_forecast_models(..., train_date_stride=3)
```

CLI:

```text
run_alpha_path20_protocol.py ... --forecast-train-date-stride 3
```

## Files Changed

```text
daily_research/path_policy/forecast_training.py
daily_research/path_policy/run_alpha_path20_protocol.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

## Resume / Reproducibility Contract

New checkpoints include:

```text
training_config.sampling_config
resume_contract.sampling_config
```

Default `train_date_stride=1` remains compatible with old checkpoints that do not have `sampling_config`. Non-default stride is treated as part of the resume contract so a stride experiment cannot silently resume as a different sampling regime.

## Verification

Focused tests:

```text
5 passed in 12.63s
```

Covered:
- `train_date_stride=3` filters tiny fixture train rows from 24 to 8 while leaving validation/test at 8/8.
- Sampling metadata is written to training summary and checkpoint resume contract.
- CLI parser accepts `--forecast-train-date-stride 3`.
- Existing no-symbol hybrid alpha score smoke remains valid.
- Existing static-context override and deferred per-epoch metrics smokes remain valid.

Static check:

```text
git diff --check
```

Result: passed.

## Current Interpretation

This is tooling only. It does not prove that overlap-aware sampling improves the model. It enables the planned A1/A2 scouts:

```text
A1: no-symbol hybrid_alpha_score_v1 train_date_stride=3
A2: no-symbol hybrid_alpha_score_v1 train_date_stride=5
```

Validation/test must remain full for both scouts. Report train/validation/test loss curves, validation-selected same-candidate test diagnostics, rank IC, spreads, runtime, and throughput.

## Next Allowed Actions

1. Run the smallest meaningful full-pack Stage 1 scout with:
   - no symbol static context,
   - `hybrid_alpha_score_v1`,
   - `selection_profile=validation_loss`,
   - `--forecast-train-date-stride 3`,
   - per-epoch checkpoints/loss audit as needed.
2. If stride 3 improves generalization, run stride 5.
3. If both fail, move priority to intraday feature-group diagnostics and regularization grid.

## Boundaries

- Do not start multi-seed yet.
- Do not start score-backtest bridge or candidate matrix.
- Do not promote or edit execution artifacts.
- Do not claim model quality improvement from this reference alone.
