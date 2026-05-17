# alpha_path20_sequence_policy_v4 Validation Repair Contract 2026-05-17

## Purpose
- Upgrade path20 sequence/RL from "train then inspect replay" to validation-first repair.
- Keep `alpha_path20_sequence_policy_v1` as the policy version; v4 is the evidence, checkpoint-selection, and repair contract.
- Use 2022 exact validation replay to choose checkpoints. 2024 test replay is interpretable only after validation passes.
- Preserve `shadow_only=true`; no live/default mutation, no promotion, and no `daily_research/output/active_execution_strategy.json` write are allowed.

## Fixed Split
- Main train years: `2019`, `2020`.
- Validation/model selection year: `2022`.
- Final shadow test year: `2024`.
- Optional data-size ablation may use `2019`, `2020`, `2021`, but that is not the default v4 evidence split.
- Train-year replay is diagnostic only and must not be used as the success verdict.

## Stage And Defaults
- Main stage: `rl-v4-validation-repair-study`.
- Main repair model: `sequence_gru`.
- `decision_transformer` remains a diagnostic comparator, not the default repair target.
- Default seeds: `7,11,19`.
- Default projection penalty grid: `0.05,0.20,0.50`.
- Default source lake id remains `policy_input_bundle__0f116a9b78c92ff045a6853d`.

## Checkpoint Selection
- Every epoch writes a checkpoint and immediately runs exact 2022 replay through `PortfolioState.step`.
- Selection reads validation metrics only:
  - prefer `validation_total_return > 0`;
  - then prefer `validation_sharpe > 0`;
  - then prefer lower projection L1 and lower turnover.
- The selection record must include `test_metrics_used_for_selection=false`.
- 2024 replay may be produced for the selected checkpoint, but `test_interpretable=false` unless validation first passes.

## Baseline Suite
- Required baselines:
  - `cash_no_trade`
  - `liquidity_equal_top30`
  - `score_blend_top30`
  - `alpha_prior_target_weight`
  - `v3_final_checkpoint`
- All baselines use the same exact replay, costs, projection constraints, validation year, and test year.
- `v3_final_checkpoint` must load an explicit `--v3-checkpoint-model-pt`; if no path is provided, it is skipped rather than faked.
- A model can be `validation_promising` only if 2022 validation return and sharpe are positive and it beats the validation baselines.

## Diagnostics
- v4 records raw intent diagnostics: tail mass outside top-k, raw target count, raw/projected top-k overlap, raw/projected rank correlation, raw gross, over-cap count, and entropy.
- Training accepts `projection_penalty_weight`, `tail_mass_penalty_weight`, and `turnover_penalty_weight`.
- The first repair ablation changes projection penalty only; reward/loss structure and model architecture should not be expanded until diagnostics justify it.
- Negative validation after more epochs or extra training years must be interpreted as evidence against "training volume alone" as the root cause.

## Verdict Rules
- `insufficient_or_incomplete`: artifact, data completeness, or projection parity is broken.
- `contract_passed`: pipeline ran but validation did not prove usefulness.
- `diagnosed_failure`: pipeline ran and selected checkpoint exists, but validation is negative or fails baseline comparison.
- `validation_promising`: exact 2022 replay is positive, sharpe is positive, and baselines are beaten.
- `test_promising`: only allowed after `validation_promising`; 2024 alone cannot create success evidence.
- All verdicts keep `promotion_allowed=false`.

## Boundaries
- Sequence/RL inputs must reject oracle, future, `path_q*`, and `path_mu_*` label fields.
- Oracle is diagnostic upper-bound only, not a teacher.
- Do not create loose `latest_*` evidence aliases.
- Do not run expensive full-year v4 studies unless explicitly approved for that run.
