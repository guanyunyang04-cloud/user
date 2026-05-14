# r65 Portfolio-Set V5 Status - 2026-05-14

## Summary
- Status: `research / shadow-only / architecture upgrade`.
- Production anchor: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Active live strategy remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- r65 adds the parallel backend `formal_torch_portfolio_set_v5` to replace MLP-style core-v4 as the next research mainline, without replacing v3/v4 artifact loading.
- r65 uses full-universe strict Gold dataset `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1` as the default training-safe dataset.
- This note uses explicit tags only and does not infer conclusions from loose `latest_*` pointers.

## Implemented Facts
- Added `daily_research/continuous_policy/model_portfolio_set_v5.py`.
- Added artifact type `continuous_policy_torch_portfolio_set_v5`.
- Added artifact file name `continuous_policy_portfolio_set_v5_artifact.pt`.
- Added trainer backend `formal_torch_portfolio_set_v5` to training contracts.
- Added training and prediction routing in `train_policy.py` and `model.py`.
- Added protocol and study routing for portfolio-set v5 artifacts.
- Added r65 active profile `split_heads_portfolio_daily_release_first_portfolio_set_v5_r65`.
- Added canonical v5 loss `portfolio_set_release_first_decision_v1` with alias `alpha_result_value_budget_split_v48`.
- r65 keeps `promotable=False`, uses CUDA AMP / pinned memory / non-blocking transfer diagnostics, and keeps true solver disabled.
- r65 active new-study registry is now `focused_seq_v1`, `split_heads_portfolio_daily_release_first_constrained_decoder_r56`, and `split_heads_portfolio_daily_release_first_portfolio_set_v5_r65`; r61 core-v4 is legacy-compatible baseline evidence.

## Architecture Facts
- v5 uses a per-symbol temporal encoder with day-set batching.
- v5 adds a portfolio state token for cash, gross exposure, turnover, drawdown, held count, recent sell/reduce/exit, and market context.
- v5 uses a latent set encoder instead of full cross-sectional self-attention; diagnostics show `portfolio_set_v5_uses_latent_attention=true` and `portfolio_set_v5_full_self_attention=false`.
- v5 release-first decoder outputs source supply, receiver demand, cash buffer, target weight, target delta, release intent, reduce quality, and exit hazard surfaces.
- `action_total=0.0` and `duration_total=0.0` remain part of the research contract; action labels are hints/reporting surfaces, not the main strategy objective.

## Verification
- Focused r65 regression passed before writeback:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_set_v5_training_contract.py daily_research/continuous_policy/tests/test_portfolio_set_v5_day_set_dataset.py daily_research/continuous_policy/tests/test_portfolio_set_v5_artifact_contract.py daily_research/continuous_policy/tests/test_portfolio_set_v5_release_first_loss.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py daily_research/continuous_policy/tests/test_research_registry_simplification.py daily_research/continuous_policy/tests/test_training_runtime_acceleration.py -q`
- Result: `138 passed, 24 warnings`.
- `git diff -- daily_research/output/active_execution_strategy.json` produced no output.
- `git diff --check` passed, with only CRLF warnings on Windows-touched files.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check` passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json` returned `status=ok`.

## Direct Smoke Evidence
- Direct smoke tag: `protocol_r65_portfolio_set_v5_smoke_20260514_01`.
- Protocol summary: `daily_research/output/continuous_policy/protocols/protocol_r65_portfolio_set_v5_smoke_20260514_01/protocol_summary.json`.
- Profile binding: `profile_applied=true`, active profile true.
- Backend/loss: `formal_torch_portfolio_set_v5` / `alpha_result_value_budget_split_v48`.
- Dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Dataset label safety: `is_training_safe=true`, `unobserved_label_rows=0`.
- Training diagnostics: CUDA AMP enabled, pinned memory enabled, non-blocking transfers enabled.
- Training evidence: `insufficient`.
- Promotion gate: `shadow_only`.
- Smoke evaluation behavior:
  - `release_first_source_intent_count=0.0`
  - `portfolio_daily_source_target_count=0.0`
  - `portfolio_daily_receiver_target_count=0.0`
  - `portfolio_daily_target_sum_gap=0.008661513452751781`
  - `intent_translation_conflict_rate=1.0`
  - `portfolio_daily_actual_cash_weight_mean=0.5529311205813372`
  - `cash_timing_quality_1d=-0.08504017541511455`
  - `release_flow_primary_blocker=no_held_negative_delta`

## Dry Run Evidence
- Study dry-run tag: `self_opt_study_r65_portfolio_set_v5_dryrun_20260514_01`.
- Study plan: `daily_research/output/continuous_policy/studies/self_opt_study_r65_portfolio_set_v5_dryrun_20260514_01/study_plan.json`.
- Dry run selected only `split_heads_portfolio_daily_release_first_portfolio_set_v5_r65`.
- Dry run selected loss `alpha_result_value_budget_split_v48`.
- Confirmatory remained disabled.
- Protocol runner mode remained `in_process`.
- Resource gate included `portfolio_set_v5_shadow_only=true`, `release_first_source_intent_floor=1.0`, `target_sum_gap_cap=0.25`, and `intent_translation_conflict_cap=0.3`.
- No strategy-effectiveness conclusion is drawn from the dry run.

## Safe Screening Evidence
- Screening study tag attempted: `self_opt_study_r65_portfolio_set_v5_screening_safe_20260514_01`.
- Protocol tag completed: `self_opt_study_r65_portfolio_set_v5_screening_safe_20260514_01__trial_01`.
- Protocol summary: `daily_research/output/continuous_policy/protocols/self_opt_study_r65_portfolio_set_v5_screening_safe_20260514_01__trial_01/protocol_summary.json`.
- Important nuance: the protocol wrote a parseable `protocol_summary.json`, but the outer study wrapper was interrupted before final `study_summary.json`. Therefore this is parseable protocol-level evidence, not a completed study-summary verdict.
- Backend/loss: `formal_torch_portfolio_set_v5` / `alpha_result_value_budget_split_v48`.
- Dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`, training safe true.
- Training diagnostics: `completed_epochs=12`, `best_epoch=5`, `train_day_count=256`, raw train day count `3949`, day cap `256`.
- Training evidence: `insufficient`, failed because `teacher_action_rows=0`.
- Promotion gate: `shadow_only`.
- Safe protocol evaluation behavior:
  - `release_first_source_intent_count=0.0`
  - `portfolio_daily_source_target_count=0.0`
  - `portfolio_daily_receiver_target_count=0.0`
  - `portfolio_daily_target_sum_gap=0.008653925893132733`
  - `intent_translation_conflict_rate=1.0`
  - `portfolio_daily_actual_cash_weight_mean=0.5536105950575143`
  - `cash_timing_quality_1d=-0.085068551447407`
  - `release_flow_primary_blocker=no_held_source`
  - receiver executable support was present, but not converted to receiver targets.
- Safe protocol shadow behavior:
  - `release_first_source_intent_count=0.0`
  - `portfolio_daily_source_target_count=0.0`
  - `portfolio_daily_receiver_target_count=0.0`
  - `portfolio_daily_target_sum_gap=0.025612951319466188`
  - `intent_translation_conflict_rate=1.0`
  - `portfolio_daily_actual_cash_weight_mean=0.5692092749543123`
  - `cash_timing_quality_1d=-0.4385459624151429`
  - `release_flow_primary_blocker=no_held_negative_delta`

## Interpretation
- Fact: r65 successfully introduces a non-MLP portfolio-set backend, strict Gold dataset loading, latent set attention, day-set batching, v48 loss registration, and release-first prediction contracts.
- Fact: r65 did not meet the minimum behavior gate. Source intent, source target, receiver target, reduce, and exit remain dead in the completed protocol.
- Fact: target-sum gap is small, but intent translation conflict is `1.0`; clean closure alone does not prove release-first behavior is connected.
- Inference: the next blocker is target/action semantics inside the portfolio-set decoder and teacher target surface, especially held source creation and receiver target realization. This should not be packaged as a generic "more epoch" or "more loss weight" issue.
- Inference: the full-universe strict Gold dataset is usable by the v5 training path, but the current implementation capped training days at `256` for first-pass safety. Future r66 work should decide whether to expand day coverage, rebalance day-set sampling, or repair target construction first.

## Boundary
- r65 remains `research / shadow-only / architecture upgrade`.
- Do not enter confirmatory from this evidence.
- Do not promote, change live/default behavior, or modify `daily_research/output/active_execution_strategy.json`.
- Do not count the interrupted outer safe study as a completed study-summary verdict.
- Do not count realtime Gold or unobserved labels as completed training evidence.
- If continuing from r65, first repair release/source/receiver target generation and intent translation conflict in portfolio-set v5; do not revert to MLP core-v4 as the main research line unless used as a baseline or ablation.
