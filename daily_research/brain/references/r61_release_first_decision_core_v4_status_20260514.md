# r61 Release-First Decision Core V4 Status - 2026-05-14

## Scope
- Status: `research / shadow-only / core-v4 decision wiring`.
- Production anchor: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Active strategy label remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- This note uses explicit tags only. It does not use `latest_*` pointers as conclusion sources.

## Implemented Facts
- Added release-flow trace diagnostics in `release_flow_trace.py` and surfaced trace fields through the release-first simulator path and protocol continuity metrics.
- Added reusable core-v4 target construction in `core_v4_release_targets.py`, including held release targets, receiver support targets, keep-risk / block-risk suppression, and target weight / delta coherence.
- Added reusable training dataset cache in `training_dataset_cache.py`. Constructed training surfaces are saved and reloaded by fingerprint: `sample_frame.pkl`, `daily_frame.pkl`, `teacher_summary.json`, and `metadata.json`.
- `train_policy.py` now supports `--training-dataset-cache-mode {auto,refresh,off}` with default `auto`.
- `run_continuous_policy_protocol.py` now exposes `train.training_dataset_cache` in `protocol_summary.json`.
- `formal_torch_core_v4` now accepts `alpha_result_value_budget_split_v47` / `core_v4_release_first_decision_v2` and rejects legacy v1-v45 losses for new core-v4 training.
- Active new-study profiles are now `focused_seq_v1`, `split_heads_portfolio_daily_release_first_constrained_decoder_r56`, and `split_heads_portfolio_daily_release_first_decision_focused_core_v4_r61`. r59 remains readable as legacy-compatible evidence, not an active new-study entrypoint.
- `run_self_optimizing_study.py` now resolves core-v4 loss profiles with the core-v4 resolver instead of the seq-v3 resolver.

## Verification
- Focused regression passed: `154 passed, 24 warnings`.
- `git diff -- daily_research/output/active_execution_strategy.json` produced no output.
- `git diff --check` passed with only CRLF normalization warnings for existing Windows-touched files.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check` passed.

## Smoke Evidence
- Direct smoke tag: `protocol_r61_release_first_decision_core_v4_smoke_20260514_03`.
- Profile binding: `profile_applied=true`, active profile true.
- Backend/loss: `formal_torch_core_v4` / `alpha_result_value_budget_split_v47`.
- Training dataset cache: hit, key `0d732414aba60a1b0eeef865`, `sample_rows=232500`, `daily_rows=465`.
- Training diagnostics: CUDA enabled, AMP enabled, pinned memory enabled, non-blocking transfers enabled.
- Stratified core-v4 sample diagnostics: `core_v4_train_release_rows=1223`, `core_v4_train_receiver_rows=1857`, `core_v4_train_held_rows=2901`.
- Smoke interpretation: wiring improved. Evaluation produced nonzero release/source counts, but shadow still had `release_first_source_intent_count=0` and `release_flow_primary_blocker=source_executable_dead`. Smoke is wiring evidence only, not strategy evidence.

## Dry Run Evidence
- Study dry-run tag: `self_opt_study_r61_release_first_decision_core_v4_dryrun_20260514_02`.
- Dry run selected only the r61 active profile with `formal_torch_core_v4`, `alpha_result_value_budget_split_v47`, `allocation_layer_v1`, `end_to_end_allocation_layer_v1`, `result_value_v10`, and `active_execution_strategy`.
- Protocol runner mode remained `in_process`.
- Confirmatory remained disabled.

## Safe Screening Evidence
- Screening study tag attempted: `self_opt_study_r61_release_first_decision_core_v4_screening_safe_20260514_01`.
- Protocol tag completed: `self_opt_study_r61_release_first_decision_core_v4_screening_safe_20260514_01__trial_01`.
- Important nuance: the protocol wrote a parseable `protocol_summary.json`, but the outer study wrapper was interrupted before writing final `study_summary.json`. Therefore this is parseable protocol-level safe-screening evidence, not a completed study-summary verdict.
- Profile binding: `profile_applied=true`, active profile true.
- Backend/loss: `formal_torch_core_v4` / `alpha_result_value_budget_split_v47`.
- Training dataset cache: stored, key `7dc0c3e7b12db4f05a8d6924`, `sample_rows=558000`, `daily_rows=465`.
- Training diagnostics: `completed_epochs=12`, `best_epoch=12`, CUDA enabled, AMP enabled, pinned memory enabled, non-blocking transfers enabled.
- Stratified core-v4 sample diagnostics: `core_v4_train_release_rows=1144`, `core_v4_train_receiver_rows=2121`, `core_v4_train_held_rows=2728`.
- Training evidence: insufficient because `best_epoch` is still at the edge.

## Behavior Evidence
- Evaluation continuity:
  - `release_first_source_intent_count=0.024096385542168676`
  - `portfolio_daily_source_target_count=0.024096385542168676`
  - `portfolio_daily_source_realized_sell_rate=1.0`
  - `portfolio_daily_receiver_target_count=1.0`
  - `portfolio_daily_target_sum_gap=0.0025976089301360085`
  - `intent_translation_conflict_rate=0.9999799196787149`
  - `portfolio_daily_actual_cash_weight_mean=0.237426640376997`
  - `cash_timing_quality_1d=-0.16635911035836506`
  - `release_flow_primary_blocker=no_held_negative_delta`
- Shadow continuity:
  - `release_first_source_intent_count=0.0`
  - `portfolio_daily_source_target_count=0.0`
  - `portfolio_daily_receiver_target_count=0.0`
  - `portfolio_daily_target_sum_gap=0.0032894928414897735`
  - `intent_translation_conflict_rate=1.0`
  - `portfolio_daily_actual_cash_weight_mean=0.2365824768698479`
  - `cash_timing_quality_1d=0.2125687950211981`
  - `release_flow_primary_blocker=source_executable_dead`
- Behavior bottleneck report blockers: `cash_timing_negative`, `sell_intent_dead`, `release_flow_disconnected`, `exit_timeliness_weak`, `training_evidence_insufficient`, and `best_epoch_at_edge`.

## Interpretation
- Fact: r61 fixed several infrastructure and wiring blockers from r60. Profile binding, core-v4 v47 loss resolution, reusable dataset cache, release-flow trace, and protocol diagnostics all work.
- Fact: release/source is no longer fully absent in evaluation, but it is far below the r61 floor of `>= 1.0`; shadow remains fully source-dead.
- Inference: the release/source/receiver/target-delta loop is still not behavior-closed. The current blocker is not a generic need for more loss weight. It is that held negative delta and executable source semantics are still not produced reliably enough for the allocator.
- Inference: the very high intent translation conflict rate means target intent and final execution semantics remain misaligned even when target-sum closure is clean.

## Boundary
- r61 remains `research / shadow-only`.
- Do not enter confirmatory from this evidence.
- Do not promote, change live/default behavior, or modify `active_execution_strategy.json`.
- Do not count the interrupted outer study wrapper as a completed study-summary verdict.
- Future r62 work should target held negative-delta creation, source executable candidate semantics, and intent translation conflict before adding epochs or generic loss weight.
