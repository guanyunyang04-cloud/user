# alpha_path20 Neural Forecaster Stage 1 Contract 2026-05-17

## Verdict
- Status: `implemented supervised forecast contract / research / shadow-only / no promotion`.
- Current Path20 research mainline pointer remains `alpha_path20_neural_policy_v1`.
- Stage 1 scope is only forecast validation: past stock state sequence -> future 20 trading-day excess-return path.
- This is not allocator, oracle, replay, or live/default strategy evidence.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Facts
- Added forecast sequence data contract under `daily_research/path_policy/forecast_dataset.py`.
- Added forecast training/evaluation contract under `daily_research/path_policy/forecast_training.py`.
- Extended `daily_research/path_policy/run_alpha_path20_protocol.py` with stages:
  - `forecast-dataset`
  - `forecast-train`
  - `forecast-walkforward-study`
- Default forecast window is train `2019-2022`, validation `2023`, test `2024`, lookback `252`, horizon `20`.
- 2026-05-17 data-lake repair supersedes the old default input:
  - old blocked bundle: `policy_input_bundle__0f116a9b78c92ff045a6853d`
  - repaired full bundle: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`
  - repaired full bundle passed strict audit for `2019-01-01 -> 2024-12-31` with benchmark open required.
- Model families are `linear_last_day`, `mlp_last_day`, `gru_sequence`, and `patch_transformer`.
- 2026-05-17 aggressive upgrade before full training:
  - `linear_last_day` remains a pure linear last-day baseline.
  - `mlp_last_day` is added as a nonlinear last-day baseline.
  - `gru_sequence` uses configurable multi-layer GRU state encoding plus attention pooling over the lookback sequence.
  - `patch_transformer` uses learned positional embeddings, a CLS token, and multi-scale patch sizes, default `4,20`.
  - Training is device-aware with `auto|cpu|cuda`, optional AMP, mini-batch DataLoader, gradient clipping, weight decay, early stopping, best checkpoint selection, and learning-curve output.
  - Multi-seed training is supported through `--forecast-seeds`; default smoke CLI remains one seed, while full-run recommendations can explicitly use `7,11,19`.
- Training uses `target_scale=100.0`; persisted prediction CSV values are restored to true return units.
- Best checkpoints are written as `forecast_model_<family>_seed<seed>_best.pt`; selected predictions are written only for the selected family/seed unless `--forecast-write-all-predictions` is set.
- Forecast verdicts are validation-first:
  - `insufficient_or_incomplete` for missing/incomplete data, severe target issues, or interrupted training.
  - `forecast_failed` when validation rank/spread/calibration gates fail.
  - `forecast_promising` when validation gates pass.
  - `forecast_test_confirmed` only when validation has passed and test rank/spread are also positive.

## Inferences
- This contract matches the forecast-then-allocation framing, but explicitly avoids claiming portfolio success from forecast loss alone.
- Validation `rank_ic_20d`, `top_bottom_spread_20d`, and quantile coverage are more decision-relevant than daily MSE alone because allocation will amplify forecast errors.
- Stronger architectures and longer training improve the information value of a negative or positive result, but only if validation-first selection, seed stability, calibration gates, and shadow-only boundaries are preserved.
- Test-year metrics are interpretable only after validation passes, preventing a validation-failed but test-good narrative from becoming false progress.

## Assumptions
- Past one-year input means the signal date plus the previous `lookback_days-1` trading days.
- Future 20d labels use `next_open` semantics unless explicitly changed.
- Because existing `next_open` labels enter on next open and exit on the future open, role-tail purge is implemented as `horizon + 1` trading days to guarantee labels do not cross roles.
- Full real lake training is not started by this contract; it requires a separate explicit user approval.
- Full-universe Stage 1 training also requires memory-safe sequence loading; capped pilots may use the repaired full bundle with `--max-universe-size`.
- Smoke defaults remain intentionally small; an evidence-grade local RTX 2060 run should explicitly set longer training controls such as `--forecast-epochs 120 --forecast-min-epochs 20 --forecast-early-stop-patience 12 --forecast-seeds 7,11,19`.

## Boundaries
- Shadow-only: no live/default promotion.
- No writes to `daily_research/output/active_execution_strategy.json`.
- No allocator/replay/oracle second-stage claim from Stage 1.
- No loose latest/default dataset ids.
- Mainline remains switchable by explicit future user/project decision.

## Next Allowed Actions
- Run fixture and guard tests for the new forecast dataset, training, and protocol contracts.
- After explicit approval, run:
  `forecast-walkforward-study --data-source lake --lake-dataset-id policy_input_bundle__7c8f58d851bce8179e1e9e2d --max-universe-size 80 --forecast-train-start-year 2019 --forecast-train-end-year 2022 --forecast-validation-year 2023 --forecast-test-year 2024 --forecast-epochs 120 --forecast-min-epochs 20 --forecast-early-stop-patience 12 --forecast-seeds 7,11,19`
- Only if validation becomes `forecast_promising` or stronger, design Stage 2 allocator/oracle/replay experiments.
