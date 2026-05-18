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
- 2026-05-17 multi-horizon opportunity upgrade changes the Stage 1 forecast target from mostly 20d-centered supervision to multi-scale opportunity supervision:
  - cumulative auxiliary horizons are now `1,3,5,10,20` instead of only `5,10,20`;
  - forecaster `aux` output is now 8-dimensional: five cumulative excess-return heads plus downside floor, worst 1d, and upside opportunity heads;
  - forecast datasets persist `y_rank_by_horizon` for `1,3,5,10,20` while retaining `y_rank_20d` for compatibility;
  - persisted prediction CSVs include `pred_aux_cum_1d`, `pred_aux_cum_3d`, and matching `future_cum_excess_return_1d/3d` columns.
- Multi-horizon evaluation now records `rank_ic`, `top_bottom_spread`, and direction accuracy for `1d/3d/5d/10d/20d`, plus `rank_ic_upside_20d` and `top_bottom_spread_upside_20d`.
- `--forecast-selection-profile` controls validation-first model selection and defaults to `multiscale`; supported values are `multiscale`, `trend20`, and `short_burst`.
- Training summaries include `selected_signal_profile`, one of `multiscale`, `trend_20d`, `short_burst`, or `failed`.
- 2026-05-17 input feature profile upgrade changes Stage 1 inputs from a single implicit state-column cap into an explicit, auditable feature profile contract:
  - `state_v1` preserves the old state-feature selection path for ablation.
  - `raw_kline_v1` adds raw OHLCV shape channels derived only from signal-day and historical data.
  - `raw_kline_context_v1` is the new default and adds raw K-line shape, cross-sectional market breadth, benchmark context, and low-cost peer bucket context.
  - `raw_kline_context_no_alpha_prior_v1` removes alpha-prior and old alpha-score dependencies to test whether the forecaster is merely replaying legacy alpha.
- Default forecast input controls are now `--forecast-feature-profile raw_kline_context_v1` and `--forecast-max-feature-columns 192`.
- Forecast dataset manifests now persist `feature_profile`, `feature_manifest`, feature group counts, cap-before/cap-after counts, and raw/market/peer/alpha-prior feature counts.
- Forecast training summaries and best checkpoints persist the same feature profile metadata so metrics cannot be interpreted without their input contract.
- The Stage 1 model is still a per-stock sequence forecaster: one sample remains `(signal_date, stock)` with tensor shape `[lookback, feature_dim]`.
- Stock-to-stock interaction is represented in v1 through features, not true cross-attention: cross-sectional ranks, market breadth, benchmark state, and peer bucket context are repeated into each stock's sequence.
- 2026-05-17 aggressive upgrade before full training:
  - `linear_last_day` remains a pure linear last-day baseline.
  - `mlp_last_day` is added as a nonlinear last-day baseline.
  - `gru_sequence` uses configurable multi-layer GRU state encoding plus attention pooling over the lookback sequence.
  - `patch_transformer` uses learned positional embeddings, a CLS token, and multi-scale patch sizes, default `4,20`.
  - Training is device-aware with `auto|cpu|cuda`, optional AMP, mini-batch DataLoader, gradient clipping, weight decay, early stopping, best checkpoint selection, and learning-curve output.
  - Multi-seed training is supported through `--forecast-seeds`; default smoke CLI remains one seed, while full-run recommendations can explicitly use `7,11,19`.
- Training uses `target_scale=100.0`; persisted prediction CSV values are restored to true return units.
- Forecast cumulative excess-return auxiliary targets are aligned to the daily excess-return path by summing daily excess labels over the requested horizon; this keeps `pred_cum_mu_<h>d` comparable to the supervised cumulative target.
- Best checkpoints are written as `forecast_model_<family>_seed<seed>_best.pt`; selected predictions are written only for the selected family/seed unless `--forecast-write-all-predictions` is set.
- 2026-05-18 capped formal Stage 1 evidence is recorded in `daily_research/brain/references/alpha_path20_stage1_formal_cap80_result_20260518.md`.
- Formal capped tag `path20_stage1_formal_cap80_repaired_20260518_01` completed on `policy_input_bundle__7c8f58d851bce8179e1e9e2d` with `raw_kline_context_v1`, `max_universe_size=80`, `8000` samples per role, four model families, and seeds `7,11,19`.
- Formal capped verdict is `forecast_test_confirmed`; selected family/seed is `gru_sequence` seed `19`; selected signal profile is `multiscale`.
- Selected validation metrics include positive `rank_ic` and positive `top_bottom_spread` for `1d/3d/5d/10d/20d`; selected validation `rank_ic_20d=0.051951`, `top_bottom_spread_20d=0.003094`, `rank_ic_upside_20d=0.090375`, and `top_bottom_spread_upside_20d=0.020895`.
- Patch Transformer family evidence is also strong: all three seeds are `multiscale`, with family validation `rank_ic_20d_mean=0.077252` and `top_bottom_spread_20d_mean=0.012649`; selection nuance should be reviewed before larger runs.
- 2026-05-18 feature ablation evidence is recorded in `daily_research/brain/references/alpha_path20_stage1_feature_ablation_result_20260518.md`.
- Feature ablation completed profiles `state_v1`, `raw_kline_v1`, `raw_kline_context_v1`, and `raw_kline_context_no_alpha_prior_v1` on the repaired bundle with cap80, train `2019-2022`, validation `2023`, test `2024`, samples `8000` per role, seeds `7,11`, and CUDA/AMP.
- All four ablation profiles produced `forecast_test_confirmed`, but with different signal shapes:
  - `state_v1` selected `trend_20d` and had weak/negative short-spread and upside metrics.
  - `raw_kline_v1` selected `multiscale` and materially improved 3d/5d/10d/20d validation rank/spread.
  - `raw_kline_context_v1` selected `multiscale` and was the only selected run with strongly positive upside opportunity metrics.
  - `raw_kline_context_no_alpha_prior_v1` selected `multiscale` and remained positive across 1d/3d/5d/10d/20d rank/spread after removing alpha-prior fields.
- Forecast verdicts are validation-first:
  - `insufficient_or_incomplete` for missing/incomplete data, severe target issues, or interrupted training.
  - `forecast_failed` when validation rank/spread/calibration gates fail.
  - `forecast_promising` when validation gates pass.
  - `forecast_test_confirmed` only when validation has passed and test rank/spread are also positive.

## Inferences
- This contract matches the forecast-then-allocation framing, but explicitly avoids claiming portfolio success from forecast loss alone.
- Validation rank/spread now needs to be read as a profile: 20d trend, short burst, multiscale, or failed. This avoids losing stocks whose real opportunity is concentrated in the first few days of the 20d window.
- Input evidence now also needs ablation: `state_v1` vs `raw_kline_v1` vs `raw_kline_context_v1` vs `raw_kline_context_no_alpha_prior_v1`.
- Raw K-line data is useful but not sufficient by itself; technical/state features, market regime, and peer context provide lower-noise summaries and interaction proxies.
- True stock-to-stock attention may be valuable later, but it requires memory-safe day-grouped datasets and is intentionally outside this input-profile upgrade.
- Validation `rank_ic_20d`, `top_bottom_spread_20d`, short-horizon rank/spread, upside capture rank/spread, and quantile coverage are more decision-relevant than daily MSE alone because allocation will amplify forecast errors.
- Stronger architectures and longer training improve the information value of a negative or positive result, but only if validation-first selection, seed stability, calibration gates, and shadow-only boundaries are preserved.
- Test-year metrics are interpretable only after validation passes, preventing a validation-failed but test-good narrative from becoming false progress.
- The first capped formal result is promising prediction evidence, but the selector nuance matters: GRU won by the implemented family-level selection rule, while Patch Transformer has stronger 20d family stability.
- Feature ablation supports the input-upgrade hypothesis: raw K-line features add material signal beyond old state features, and no-alpha-prior context remains predictive, so the result is not simply legacy alpha-prior replay.
- Alpha-prior/context may still matter for upside opportunity capture because the no-alpha-prior selected run lost upside opportunity signal while retaining 1d/3d/5d/10d/20d rank/spread signal.

## Assumptions
- Past one-year input means the signal date plus the previous `lookback_days-1` trading days.
- Future 20d labels use `next_open` semantics unless explicitly changed.
- Because existing `next_open` labels enter on next open and exit on the future open, role-tail purge is implemented as `horizon + 1` trading days to guarantee labels do not cross roles.
- `cum1/cum3` are intended to help the forecaster learn short burst opportunities, but 1d evidence is kept low-weighted so daily noise cannot dominate model selection.
- Full real lake training is not started by this contract; it requires a separate explicit user approval.
- Full-universe Stage 1 training also requires memory-safe sequence loading; capped pilots may use the repaired full bundle with `--max-universe-size`.
- Feature profiles must never include future returns, future labels, oracle outputs, or allocator/replay outcomes as inputs.
- Industry/concept context is not a hard dependency in this version; liquidity/price/vol peer buckets are used as point-in-time-safe proxies until a validated point-in-time industry map exists.
- Smoke defaults remain intentionally small; an evidence-grade local RTX 2060 run should explicitly set longer training controls such as `--forecast-epochs 120 --forecast-min-epochs 20 --forecast-early-stop-patience 12 --forecast-seeds 7,11,19`.

## Boundaries
- Shadow-only: no live/default promotion.
- No writes to `daily_research/output/active_execution_strategy.json`.
- No allocator/replay/oracle second-stage claim from Stage 1.
- No loose latest/default dataset ids.
- Mainline remains switchable by explicit future user/project decision.

## Next Allowed Actions
- Inspect selection policy before larger experiments: compare family mean, best seed, 20d robustness, multiscale score, and coverage gates.
- Design Stage 2 allocator/oracle/replay experiments only as research/shadow follow-up to capped prediction and feature-ablation evidence.
- Implement memory-safe sequence loading before any full-universe `max_universe_size=0` Stage 1 run.
