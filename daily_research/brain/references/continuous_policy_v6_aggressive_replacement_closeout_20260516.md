# continuous_policy v6 aggressive replacement closeout 2026-05-16

## Scope

This note closes the current `continuous_policy` v6 aggressive replacement sprint as research / shadow-only evidence work. It does not promote any artifact, does not modify live/default execution, and does not change `daily_research/output/active_execution_strategy.json`.

## Facts

- A new v6 decision core was implemented in `daily_research/continuous_policy/decision_core_v6.py`.
- v6 was wired into training, prediction, protocol, study, simulator, and continuity metrics paths.
- Legacy r69/r71/r74 research profile and loss-profile entry points are rejected for new protocol/training runs.
- Targeted tests passed:
  - `daily_research/continuous_policy/tests/test_decision_core_v6.py`: 14 passed.
  - `daily_research/continuous_policy/tests/test_protocol_profile_binding.py` plus `test_research_registry_simplification.py`: 20 passed.
  - Earlier broader continuous-policy contract tests passed: 125 passed.
- `active_execution_strategy.json` had no diff during closeout checks.
- Strict Gold dataset used for v6 work:
  - `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Lake policy bundle used for v6 protocol work:
  - `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Strict Gold offline v6 target sanity on 2026-05-16:
  - raw sample rows: 1,863,468.
  - selected train rows: 163,059.
  - selected train days: 256.
  - v6 source rows: 68.
  - v6 receiver rows: 466.
  - v6 action rows: 534.
  - max oracle constraint violation: about `1.86e-9`.
  - feature contract blocker rate: 0.0.
  - feature contract degraded rate: 0.0.

## Completed Evidence

### `decision_core_v6_longrun_candidate_20260515_01__trial_01`

- Training evidence status: sufficient.
- Completed epochs: 32.
- Best epoch: 27.
- Hard contract metrics passed:
  - evaluation/shadow cashflow contract valid rate: 1.0.
  - evaluation/shadow intent conflict rate: 0.0.
  - evaluation/shadow oracle constraint violation mean: about `1e-11`.
  - evaluation feature blocker/degraded rate: 0.0/0.0.
  - shadow feature blocker/degraded rate: 0.0/about 0.003.
- Behavior gates failed:
  - evaluation source positive forward sell share: about 0.462.
  - shadow source positive forward sell share: about 0.495.
  - evaluation immediate reversal rate 3d: about 0.044.
  - shadow immediate reversal rate 3d: about 0.028.
  - evaluation receiver-minus-source forward excess 5d: about -0.061.
  - shadow receiver-minus-source forward excess 5d: about -0.058.
  - evaluation cash timing quality 1d: about -0.053.
  - shadow cash timing quality 1d: about -0.052.

### `decision_core_v6_objective_shape_smoke_20260516_01`

- Training evidence status: insufficient.
- Completed epochs: 32.
- Best epoch: 32, which violates the best-epoch edge gate.
- Hard contract metrics still passed:
  - cashflow contract valid rate: 1.0.
  - intent conflict rate: 0.0.
  - oracle constraint violation mean below `1e-6`.
  - feature blocker rate: 0.0.
  - feature degraded rate below 0.25.
- Behavior improved in some places but still failed:
  - evaluation source positive forward sell share: about 0.464.
  - shadow source positive forward sell share: about 0.463.
  - evaluation immediate reversal rate 3d: about 0.0073.
  - shadow immediate reversal rate 3d: about 0.0060.
  - evaluation receiver-minus-source forward excess 5d: about -0.013.
  - shadow receiver-minus-source forward excess 5d: about -0.0094.
  - evaluation cash timing quality 1d: about 0.0379.
  - shadow cash timing quality 1d: about 0.0326.

## Partial Evidence

### `decision_core_v6_forward_spread_smoke_20260516_01`

- Train/evaluate/shadow artifacts were produced, but `protocol_summary.json` was not written.
- Progress stopped after export stage start. No active training/protocol process remained at closeout.
- Train completed 32 epochs with best epoch 28.
- Train diagnostics reported v6 source target count 54 and receiver target count 427.
- Evaluation hard contract metrics passed:
  - cashflow contract valid rate: 1.0.
  - intent conflict rate: 0.0.
  - oracle violation mean: 0.0.
  - feature blocker/degraded rate: 0.0/0.0.
- Evaluation behavior still failed:
  - source positive forward sell share: about 0.569.
  - receiver-minus-source forward excess 5d: about -0.0058.
  - immediate reversal rate 3d: 0.0.
  - cash timing quality 1d: about 0.0054.
- Shadow metrics showed no source/receiver spread evidence, so the shadow behavior readout is not sufficient to rescue the run.

## Inferences

- The v6 work largely solved legality and semantic consistency problems: cashflow, intent translation, oracle feasibility, and feature contract metrics became stable.
- The same work did not solve economic behavior: source release quality, receiver-source spread, and long-horizon allocation quality remained below the acceptance gates.
- Further hardening source release gates is unlikely to be the right primary fix. A previous stricter release direction improved some safety metrics but collapsed source coverage in shadow.
- The current v6 training target is sparse relative to the training matrix size, so longer imitation training alone may not address the behavioral failure.
- The central issue is likely objective definition rather than only model capacity, dataset size, or epoch count. The model is learning a shaped `DecisionFrameV6` imitation target, not directly optimizing long-horizon portfolio utility.

## Assumptions

- The original `deep_alpha` line contains useful predictive signal and should not be discarded.
- Production/live/default execution boundaries must remain untouched until there is stronger evidence.
- v6 artifacts and protocol outputs remain useful for audit and diagnostics even if the current direction is not promoted.
- Existing v6 code changes can be retained as a research branch, but should not be treated as a completed longrun candidate.

## Boundaries

- Do not promote v6.
- Do not modify `active_execution_strategy.json`.
- Do not claim v6 longrun candidate success.
- Do not keep spending longrun budget on the current source/receiver imitation objective unless a new upper-bound test shows that the v6 teacher/target itself can beat the acceptance gates.
- Keep v6 conclusions labeled as `research / shadow-only / incomplete evidence`.

## Recommended Next Direction

Before more long training, add a target upper-bound test:

1. Replay v6 teacher/target directly through the simulator.
2. Measure whether the target itself passes source positive forward sell share, immediate reversal, receiver-minus-source forward excess, cash timing, and portfolio return gates.
3. If target replay fails, stop tuning model capacity and fix the objective/label.
4. If target replay passes but learned rollout fails, then investigate model architecture, loss weighting, training duration, and imitation gap.
5. If learned rollout passes but simulated/executed output fails, then investigate adapter and simulator semantics.

The likely replacement research line is `deep_alpha` strong prediction plus a return-aware learned allocator, where source/receiver fields are explanation outputs rather than the primary training objective.
