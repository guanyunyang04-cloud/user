# alpha_path20 Frontier Outputs 2026-05-20

## Verdict
- Status: `frontier output freshness writeback / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Scope: compact registration for recent Path20 memmap datasets and GRU probes from 2026-05-18/19.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Completed Forecast Evidence
- `path20_pool_liquid500_sector_gru_seed7_probe_20260519_01`: completed `forecast_walkforward_study`, verdict `forecast_test_confirmed`, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__c11400fa72ad263f3d1eecfa`, sector/board view `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`, feature profile `raw_kline_context_sector_v1`, universe size `2430`, model family `gru_sequence`, seed `7`.
- `path20_pool_liquid800_gru_seed7_probe_20260519_01`: completed `forecast_walkforward_study`, verdict `forecast_test_confirmed`, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__8161f6879815d3a22dd3b3f2`, feature profile `raw_kline_context_v1`, universe size `2813`, model family `gru_sequence`, seed `7`.
- `path20_pool_liquid500_gru_seed7_probe_20260519_01`: completed `forecast_walkforward_study`, verdict `forecast_test_confirmed`, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__c11400fa72ad263f3d1eecfa`, feature profile `raw_kline_context_v1`, universe size `2430`, model family `gru_sequence`, seed `7`.
- `path20_pool_liquid300_gru_seed7_probe_20260519_01`: completed `forecast_walkforward_study`, verdict `forecast_test_confirmed`, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__ef6e9054e28cf5a71c6996e4`, feature profile `raw_kline_context_v1`, model family `gru_sequence`, seed `7`.
- `path20_memmap_training_smoke_full_gru_reuse_20260518_01`: completed full-universe memmap GRU smoke, verdict `forecast_test_confirmed`, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, universe size `3070`, model family `gru_sequence`, seed `7`.
- `path20_memmap_training_smoke_cap80_gru_patch_20260518_01`: completed capped memmap GRU/Patch smoke, verdict `forecast_test_confirmed`, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, model families `gru_sequence` and `patch_transformer`.

## Failed / Diagnostic Artifacts
- `path20_memmap_training_smoke_cap80_20260518_02`: completed run with verdict `forecast_failed`; diagnostic only and not completed positive evidence.

## Dataset-Only Artifacts
- `path20_memmap_dataset_liquid500_sector_probe_20260519_01`: completed memmap dataset build, verdict `insufficient_or_incomplete` because it is dataset-only, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__c11400fa72ad263f3d1eecfa`, sector/board view `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`, feature profile `raw_kline_context_sector_v1`, universe size `2430`.
- `path20_memmap_dataset_liquid800_probe_20260519_01`: completed memmap dataset build, verdict `insufficient_or_incomplete` because it is dataset-only, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__8161f6879815d3a22dd3b3f2`, feature profile `raw_kline_context_v1`, universe size `2813`.
- `path20_memmap_dataset_liquid500_probe_20260519_01`: completed memmap dataset build, verdict `insufficient_or_incomplete` because it is dataset-only, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__c11400fa72ad263f3d1eecfa`, feature profile `raw_kline_context_v1`.
- `path20_memmap_dataset_liquid300_probe_20260519_01`: completed memmap dataset build, verdict `insufficient_or_incomplete` because it is dataset-only, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, pool view `policy_pool_view__ef6e9054e28cf5a71c6996e4`, feature profile `raw_kline_context_v1`.
- `path20_memmap_dataset_full_repaired_20260518_01`: completed full repaired memmap dataset build, verdict `insufficient_or_incomplete` because it is dataset-only, dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, universe size `3070`.

## Inferences
- Repaired-bundle memmap infrastructure is ready for controlled Path20 architecture experiments.
- The liquid500 sector GRU probe is the correct local baseline for the next architecture-gain study.
- Dataset-only artifacts are input-readiness evidence, not completed forecast evidence.

## Boundaries
- No allocator, replay, oracle, live/default, or active execution claim.
- No production promotion is authorized.
- No `daily_research/output/active_execution_strategy.json` change is expected.

## Next Allowed Actions
- Run shadow-only liquid500 sector architecture-gain training with `gru_sequence_static_context`, `patch_transformer_static_context`, `stock_mixer_sequence`, and `sector_slot_mixer_sequence`.
- Gate any liquid800 expansion on validation/test-positive Stage 1 evidence.
