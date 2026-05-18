# alpha_path20 Stage 1 Formal Cap80 Result 2026-05-18

## Verdict
- Status: `completed capped formal forecast evidence / forecast_test_confirmed / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Study tag: `path20_stage1_formal_cap80_repaired_20260518_01`.
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Feature profile: `raw_kline_context_v1`, capped at 192 features.
- Universe cap: `max_universe_size=80`; this is not full-universe evidence.
- Selected family/seed: `gru_sequence`, seed `19`.
- Selected signal profile: `multiscale`.
- Evidence verdict: `forecast_test_confirmed`.
- `shadow_only=true`, `promotion_allowed=false`, `active_execution_strategy_expected_diff=none`.

## Facts
- Preflight tag `path20_stage1_preflight_cap80_repaired_20260518_01` completed on the repaired bundle with:
  - train/validation/test samples present;
  - `role_purge_trading_days=21`;
  - `feature_profile=raw_kline_context_v1`;
  - raw K-line, market context, and peer context feature counts all greater than zero;
  - benchmark open loaded from `silver_benchmark.open`.
- Pilot attempts `_01` and `_02` hit CUDA runtime availability problems before a clean training result.
- Pilot `_03` exposed a metric dtype bug: AMP/CUDA predictions could reach Pandas as `float16`, and `nlargest/nsmallest` failed during top-bottom spread computation.
- Fixed the metric path by casting prediction and target columns to `float64` in rank-IC and top-bottom-spread metric routines.
- Added regression coverage: `test_forecast_prediction_metrics_accepts_amp_float16_predictions`.
- Pilot tag `path20_stage1_pilot_cap80_repaired_20260518_04` completed after the metric fix, wrote all expected artifacts, and produced `forecast_test_confirmed` with selected `gru_sequence` seed `7`.
- Formal tag `path20_stage1_formal_cap80_repaired_20260518_01` completed with:
  - train/validation/test rows: `8000/8000/8000`;
  - model families: `linear_last_day`, `mlp_last_day`, `gru_sequence`, `patch_transformer`;
  - seeds: `7,11,19`;
  - `forecast-epochs=120`, `forecast-min-epochs=20`, `forecast-early-stop-patience=12`;
  - `forecast-device=cuda`, AMP enabled;
  - 12 best checkpoints written;
  - validation/test prediction CSVs written;
  - learning curve and walkforward summaries written.

## Formal Validation Metrics
- Selected GRU seed19 validation metrics:
  - `rank_ic_1d=0.054034`
  - `rank_ic_3d=0.076536`
  - `rank_ic_5d=0.087974`
  - `rank_ic_10d=0.088521`
  - `rank_ic_20d=0.051951`
  - `top_bottom_spread_1d=0.001181`
  - `top_bottom_spread_3d=0.002272`
  - `top_bottom_spread_5d=0.001612`
  - `top_bottom_spread_10d=0.000575`
  - `top_bottom_spread_20d=0.003094`
  - `rank_ic_upside_20d=0.090375`
  - `top_bottom_spread_upside_20d=0.020895`
  - `q10_coverage_mean=0.934275`
  - `q90_coverage_mean=0.890644`
- Selected GRU seed19 test metrics are interpretable because validation passed:
  - `rank_ic_1d=0.044175`
  - `rank_ic_3d=0.069478`
  - `rank_ic_5d=0.102321`
  - `rank_ic_10d=0.150255`
  - `rank_ic_20d=0.191404`
  - `top_bottom_spread_1d=0.001949`
  - `top_bottom_spread_3d=0.007582`
  - `top_bottom_spread_5d=0.012847`
  - `top_bottom_spread_10d=0.031651`
  - `top_bottom_spread_20d=0.058073`
  - `rank_ic_upside_20d=0.147662`
  - `top_bottom_spread_upside_20d=0.028474`
  - `q10_coverage_mean=0.858450`
  - `q90_coverage_mean=0.868075`

## Family Summary
- `gru_sequence`: validation multiscale score mean `1000.729170`, rank-IC 20d mean `0.057584`, 20d spread mean `0.000925`, 20d rank-IC positive seed rate `1.0`, 20d spread positive seed rate `0.666667`; signal profiles were 2 multiscale and 1 short-burst.
- `patch_transformer`: validation multiscale score mean `1000.687404`, rank-IC 20d mean `0.077252`, 20d spread mean `0.012649`, 20d rank-IC positive seed rate `1.0`, 20d spread positive seed rate `1.0`; all 3 seeds were multiscale.
- `mlp_last_day`: validation multiscale score mean `1000.439150`, rank-IC 20d mean `-0.003606`, 20d spread mean `-0.007050`; all 3 seeds were short-burst.
- `linear_last_day`: validation multiscale score mean `0.139409`, rank-IC 20d mean `-0.000592`, 20d spread mean `-0.001029`; profiles were mixed and calibration coverage was weak.

## Inferences
- Stage 1 capped evidence is positive: validation shows stable multi-horizon rank signal and positive top-bottom spread for the selected run.
- The selected GRU is chosen by the implemented family-level selection rule; this does not mean it dominates every metric.
- Patch Transformer is especially notable as a family because all three seeds are multiscale and 20d spread/rank stability are stronger than GRU in the family summary.
- MLP last-day learning short-burst signal but failing 20d spread suggests sequence context matters for 20d trend/path evidence.
- Linear last-day is useful as a sanity baseline and is materially weaker than nonlinear/sequence models.
- This result is prediction-task evidence only. It does not prove a tradable allocator, portfolio replay, turnover/cash behavior, or live execution success.
- Because this run is capped at 80 symbols and uses an eager dataset, full-universe claims remain blocked until memory-safe sequence loading exists.

## Assumptions
- The repaired bundle remains the correct fixed input bundle for near-term Path20 capped experiments.
- The 2023 validation year is the primary model-selection year; 2024 test metrics are interpreted only because validation gates passed.
- Coverage gate semantics remain those implemented in the forecast training summary.

## Boundaries
- No active execution file change.
- No allocator, oracle, replay, or live/default claim.
- No full-universe claim.
- No loose latest/default dataset id claim.

## Next Allowed Actions
- Use completed feature ablation evidence in `daily_research/brain/references/alpha_path20_stage1_feature_ablation_result_20260518.md`.
- Inspect why the family-level selector preferred GRU while Patch Transformer had stronger 20d family metrics; decide whether selection should prefer family mean, best seed, or a robustness-weighted hybrid before larger experiments.
- Design Stage 2 allocator/oracle/replay only after documenting ablation results and preserving validation-first interpretation.
- Before full-universe Stage 1, implement memory-safe sequence loading and batched prediction writing.

## Guards
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_forecast_features.py daily_research/path_policy/tests/test_forecast_dataset.py daily_research/path_policy/tests/test_forecast_training.py daily_research/path_policy/tests/test_models.py daily_research/path_policy/tests/test_rl_protocol.py -q`: `45 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/tools/tests/test_brain_evidence_registry.py daily_research/tools/tests/test_brain_capsule.py -q`: `8 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`: passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`: status `ok`.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.
- `git diff --check`: passed.
