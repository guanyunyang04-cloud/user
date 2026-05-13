# r59 Core V4 Release-First Status 2026-05-13

## Scope
- Status: research / shadow-only / core-v4 infrastructure.
- Goal: add a parallel `formal_torch_core_v4` backend for the r56 release-first research line without extending the `model_seq_v3.py` monolith.
- Boundary: no live/default/promotion change, no true-solver default change, and no write to `daily_research/output/active_execution_strategy.json`.

## Implementation Facts
- Added `daily_research/continuous_policy/model_core_v4.py`.
- Registered trainer backend `formal_torch_core_v4` in the training contract.
- Core-v4 contract is epoch-based, resume-capable, and GPU-required, but `promotable=False`.
- `train_policy.py` routes core-v4 training to `fit_policy_models_core_v4(...)` and writes `continuous_policy_core_v4_artifact.pt`.
- `model.py` can load and predict `continuous_policy_torch_core_v4` artifacts.
- Core-v4 loss resolver only accepts `alpha_result_value_budget_split_v46` and alias `core_v4_release_first_v1`; legacy v1-v45 losses are rejected for core-v4 new training.
- Core-v4 prediction emits r56 release-first fields, including `portfolio_daily_target_weight_intent`, `portfolio_daily_target_delta_intent`, release/source support fields, and global target `release_first_allocation_v3_mode=1.0`.
- New active profile `split_heads_portfolio_daily_release_first_core_v4_r59` is registered with `trainer_backend=formal_torch_core_v4`, `loss_profile=alpha_result_value_budget_split_v46`, `budget_semantics=allocation_layer_v1`, `budget_calibration=end_to_end_allocation_layer_v1`, `budget_objective=result_value_v10`, and `alpha_prior_source=active_execution_strategy`.
- Active new-study registry is now limited to `focused_seq_v1`, r56 v3 reference, and r59 core-v4.

## Test Evidence
- Command:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_core_v4_training_contract.py daily_research/continuous_policy/tests/test_core_v4_artifact_contract.py daily_research/continuous_policy/tests/test_core_v4_release_first_loss.py daily_research/continuous_policy/tests/test_research_registry_simplification.py daily_research/continuous_policy/tests/test_training_runtime_acceleration.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py daily_research/continuous_policy/tests/test_semantic_budget_intent.py -q`
- Result: `136 passed, 24 warnings`.
- Warnings are existing cvxpylayers / NumPy deprecation warnings in portfolio strategy contract tests.

## Runtime Evidence
- Direct smoke `_01` failed during training because AMP autocast rejected `binary_cross_entropy`; this exposed an unsafe loss implementation.
- Fix: core-v4 now uses logits with `binary_cross_entropy_with_logits`.
- Direct smoke `_02` completed training but failed during evaluation because `derive_release_first_intent(...)` treated missing optional risk columns as floats and then called `.fillna(...)`.
- Fix: optional release-first intent inputs now use a series helper with stable defaults.
- Direct smoke `_03` completed successfully:
  - Tag: `protocol_r59_core_v4_direct_smoke_20260513_03`.
  - Protocol summary: `daily_research/output/continuous_policy/protocols/protocol_r59_core_v4_direct_smoke_20260513_03/protocol_summary.json`.
  - Model artifact: `daily_research/output/continuous_policy/models/protocol_r59_core_v4_direct_smoke_20260513_03__train/continuous_policy_core_v4_artifact.pt`.
  - Trainer backend: `formal_torch_core_v4`.
  - Contract: `promotable=False`, `gpu_required=True`.
  - Diagnostics: CUDA AMP enabled, pinned DataLoader memory enabled, non-blocking transfer enabled, release-first allocation v3 support enabled.
  - Training evidence: `insufficient`, with `best_epoch=1` and `completed_epochs=1`.
  - Promotion gate: `shadow_only`.
- Study dry run completed:
  - Tag: `self_opt_study_r59_core_v4_release_first_dryrun_20260513_01`.
  - Search profile: `split_heads_portfolio_daily_release_first_core_v4_r59`.
  - Selected trial backend: `formal_torch_core_v4`.
  - Selected trial loss: `alpha_result_value_budget_split_v46`.
  - Budget semantics/objective/prior: `allocation_layer_v1`, `result_value_v10`, `active_execution_strategy`.
  - Resource gate includes `core_v4_shadow_only=true`.

## Guard Evidence
- `git diff -- daily_research/output/active_execution_strategy.json`: no output.

## Interpretation
- Fact: r59 proves the core-v4 training, artifact, prediction, registry, and study dry-run wiring exists and is test-covered.
- Inference: core-v4 reduces future r56/r59 work pressure on `model_seq_v3.py` by moving new release-first semantics into a small parallel backend.
- Limitation: r59 has no safe screening evidence and no strategy-effectiveness verdict.
- Rule: do not use r59 smoke metrics as completed evidence, confirmatory evidence, promotion evidence, or live/default support.

## Next Gate
- A future safe screening can be considered only after direct smoke and dry-run remain clean, active artifact remains unchanged, and the run uses explicit r59 tags.
- If r59 later fails behaviorally, the conclusion must distinguish "core-v4 infrastructure works" from "release-first strategy semantics still fail".
