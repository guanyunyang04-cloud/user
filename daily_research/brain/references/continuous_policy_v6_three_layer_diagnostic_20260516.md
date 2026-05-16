# continuous_policy v6 three-layer diagnostic 2026-05-16

## Scope

This note records the pivot after the v6 aggressive replacement sprint. The purpose is not promotion and not more gate tweaking. It documents a three-layer diagnosis:

1. Target upper bound: replay v6 teacher/target directly.
2. Model imitation gap: compare learned v6 rollout with target replay.
3. Simulator/execution gap: verify whether v6 adapter and simulator semantics are still breaking execution.

All evidence is `research / shadow-only`. `daily_research/output/active_execution_strategy.json` was not changed.

## Facts

- Added and fixed the diagnostic runner:
  - `daily_research/continuous_policy/diagnose_decision_core_v6_layers.py`.
  - It now defaults to `budget_semantics=allocation_layer_v1` and `budget_calibration=end_to_end_allocation_layer_v1`.
  - It rejects non-v6 execution paths unless `--allow-non-v6-execution-path` is explicitly used for counterfactual work.
  - It writes target/model summaries plus execution-path validity diagnostics.
- Added tests:
  - `daily_research/continuous_policy/tests/test_decision_core_v6_layer_diagnostics.py`.
  - Targeted test command passed: `17 passed`.
- The first 2024Q1 smoke run was misleading because default legacy budget semantics bypassed the v6 cashflow/native target path. After forcing the v6 allocation path, `cashflow_decision_used_mean=1.0` and source/receiver target counts became meaningful.
- A single 2019-2024 full-window diagnostic exceeded the fixed `3600s` timeout and produced only partial CSV files without `layer_diagnostic_summary.json`. The orphan diagnostic process was stopped and the incomplete directory was removed.
- Completed multi-segment diagnostic summary:
  - `daily_research/output/continuous_policy/studies/decision_core_v6_layer_diagnostic_multisegment_20260516_01/layer_diagnostic_multisegment_summary.json`.
- Completed segment summaries:
  - `decision_core_v6_layer_diagnostic_2019q1_20260516_01`.
  - `decision_core_v6_layer_diagnostic_2020q1_20260516_01`.
  - `decision_core_v6_layer_diagnostic_2022q1_20260516_01`.
  - `decision_core_v6_layer_diagnostic_2024q1_20260516_02`.
- Fixed dataset and artifact references:
  - strict Gold dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
  - lake bundle: `policy_input_bundle__0f116a9b78c92ff045a6853d`.
  - model artifact: `daily_research/output/continuous_policy/models/decision_core_v6_forward_spread_smoke_20260516_01__train/continuous_policy_decision_core_v6_artifact.pt`.

## Segment Results

| Segment | Target return | Target Sharpe | Target contract valid | Target fail-closed | Model return | Model gap | Match | Target source positive sell share | Target receiver-source 5d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019Q1 | 0.420348 | 5.574 | 0.661 | 0.293 | 0.338146 | -0.082202 | 0.961 | 0.250 | 0.0813 |
| 2020Q1 | -0.022892 | -0.065 | 0.853 | 0.121 | -0.109508 | -0.086615 | 0.930 | 0.182 | 0.0952 |
| 2022Q1 | -0.008871 | -0.107 | 0.727 | 0.224 | -0.107607 | -0.098736 | 0.908 | 0.341 | 0.0587 |
| 2024Q1 | 0.144893 | 4.482 | 0.905 | 0.086 | 0.120776 | -0.024116 | 0.945 | 0.500 | 0.0247 |

## Layer Diagnosis

### Target upper bound

- Fact: target replay is highly profitable in 2019Q1 and 2024Q1.
- Fact: target replay is not profitable in 2020Q1 and 2022Q1.
- Fact: target cashflow contract is not clean in any segment. Invalid reasons are concentrated in `receiver_direction_conflict|cash_conservation_gap`, with additional `source_direction_conflict|cash_conservation_gap` in 2022Q1.
- Fact: target receiver-minus-source forward spread is positive in all four segments.
- Fact: target source positive forward sell share fails the `<=0.10` gate in all four segments.

### Model imitation gap

- Fact: model rollout underperforms target replay in all four segments.
- Fact: model action match is not catastrophic, but the return gap is material:
  - about -8.2 points in 2019Q1.
  - about -8.7 points in 2020Q1.
  - about -9.9 points in 2022Q1.
  - about -2.4 points in 2024Q1.
- Inference: model capacity, loss shape, training amount, and dataset quality remain real issues, but they are not the first blocker while target replay itself is inconsistent and sometimes unprofitable.

### Simulator/execution gap

- Fact: model rollout uses the v6 execution path cleanly in these diagnostics: cashflow contract valid rate is 1.0 and fail-closed mean is 0.0 across tested segments.
- Fact: direct target replay exposes cashflow contract failures. This is not a legacy-budget bypass after the diagnostic fix; it occurs inside the v6 allocation path.
- Inference: the simulator can consume v6 model outputs, but the direct teacher/target frame and cashflow contract are not semantically aligned. The likely issue is target/oracle/objective consistency, not only adapter plumbing.

## Inferences

- The user's concern is valid: continuing to tune hard gates would likely hide the underlying problem instead of solving it.
- The v6 line did not simply fail because the model is too weak or undertrained. The upper-bound layer shows a deeper problem:
  - in some market segments the teacher/target itself does not make money;
  - in all tested segments the direct target violates the cashflow contract on part of the run;
  - source release still sells too many future-positive names.
- A stronger model trained longer may improve imitation, but it cannot fix an inconsistent or sometimes unprofitable target.
- The old `deep_alpha` evidence still matters: prediction signal can exist while this v6 source/receiver/cash target objective is the wrong abstraction for maximizing portfolio return.

## Assumptions

- The lake bundle and strict Gold dataset are the right references for this diagnostic.
- Q1 windows are sufficient to expose the main failure pattern, but not sufficient to certify a replacement strategy.
- The diagnostic model artifact is adequate for measuring an imitation gap, not for promotion.
- Full 2019-2024 can be rerun later only after memory/runtime improvements or streaming summary writes are added.

## Boundaries

- Do not promote v6.
- Do not change live/default strategy.
- Do not modify `active_execution_strategy.json`.
- Do not run more long v6 training on the current objective before target/oracle consistency is repaired.
- Do not interpret passing hard contract gates on model rollout as proof that target design is correct.

## Recommended Next Direction

1. Repair target/oracle/cashflow semantics first:
   - make source/receiver supply exactly consistent with `target_delta`;
   - align deadbands between v6 target generation and simulator cashflow normalization;
   - eliminate `receiver_direction_conflict|cash_conservation_gap` and `source_direction_conflict|cash_conservation_gap` in direct target replay.
2. Replace source/receiver imitation as the primary objective with a return-aware allocator objective:
   - use `deep_alpha` or equivalent predictive signal as the predictive backbone;
   - train allocation/position sizing against realized portfolio utility or differentiable proxy;
   - keep source/receiver/cash as explanations or constraints, not the main learned target.
3. After target replay is profitable and contract-clean across segments, then revisit model capacity, training duration, loss weights, and larger strict Gold training.

