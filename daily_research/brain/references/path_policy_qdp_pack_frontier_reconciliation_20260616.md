# Path Policy QDP Pack Frontier Reconciliation 20260616

## Verdict
- Status: `frontier_reconciled / qdp_pack_runs_registered / alpha_v2_training_pack_consumption_smoke_completed`.
- Scope: reconcile 8 latest QDP-pack / QDP-sharded runs from 2026-06-13 to 2026-06-14 that were newer than the brain references, and register the latest QDP `style_structural_alpha_v2_label_v2` data state.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` remains unchanged; the file is still absent in the frozen execution skeleton and was not recreated.
- Interpretation: the 2026-06-13/14 runs are `style_structural_v1` QDP consumption evidence; the 2026-06-16 smoke proves the daily protocol can consume the new `style_structural_alpha_v2_label_v2` training pack, feature schema, static context, label v2, and role split. It is not model-quality or promotion evidence.

## Reconciled Run Tags
- `qdp_pack_hybrid_topn_h256_t4_b512_full_e12_20260614_01`
- `qdp_pack_hybrid_topn_h256_t4_b512_throughput_20260614_01`
- `qdp_pack_hybrid_topn_h192_b768_pilot_20260614_01`
- `qdp_training_pack_hybrid_full_b1024_e12_20260613_02`
- `qdp_training_pack_progress_smoke_20260613_01`
- `qdp_training_pack_hybrid_pilot_b512_20260613_01`
- `qdp_sharded_hybrid_static_smoke_20260613`
- `qdp_sharded_linear_smoke_20260613`
- New smoke run: `qdp_alpha_v2_label_v2_pack_consumption_smoke_20260616_01`

## QDP Data State
- QDP root status: `canonical_bundle_ready_full_sharded_memmap_ready`.
- Canonical dataset id: `policy_input_bundle__f926a496f69c61f6b92b5faf`.
- Active sharded memmap registry status: `ready`; active/latest manifest is `H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01\sharded_memmap_manifest.json`.
- Active sharded profile: `style_structural_alpha_v2`; feature count `307`; feature schema hash `38a98a4a96637df6bc465837`; label schema `path20_basic_v2` version `2`; static context enabled with fields `symbol, exchange, industry`.
- Active sharded scope/status: `full_canonical_candidate / completed`; planned/processed/stored/empty/failed shards: `187/187/183/4/0`; years `2010-2026`; lookback `252`.
- Frozen short-horizon base remains `canonical_short_horizon_core_v1_full_2010_2026`, profile `short_horizon_core_v1`, status `frozen_base_ready`, manifest `H:\quant_project\quant_data_platform\data\memmap\sharded\canonical_short_horizon_core_v1_full\sharded_memmap_manifest.json`.

## Alpha V2 Training Pack
- Manifest: `H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01\qdp_training_pack_manifest.json`.
- Artifact/status: `qdp_training_pack_v1 / completed`; scope `full_canonical_candidate`; canonical dataset id `policy_input_bundle__f926a496f69c61f6b92b5faf`; pool view `policy_pool_view__fc63d6b996abc7abde88bae7`.
- Feature profile/count/hash: `style_structural_alpha_v2 / 307 / 38a98a4a96637df6bc465837`.
- Label schema: `path20_basic_v2` version `2`; execution mode `next_open`; lookback `252`.
- Full sample count: `7,421,714`; split train/validation/test `6,050,268 / 681,979 / 689,467`.
- Role years: train `2012-2023`, validation `2024`, test `2025`.
- Static context enabled with fields `symbol, exchange, industry`; date-major feature panel layout is `date_stock_feature`.

## Recent QDP Consumption Evidence
- `qdp_pack_hybrid_topn_h256_t4_b512_full_e12_20260614_01`: completed, `forecast_test_confirmed`, `style_structural_v1`, full pack `7,421,735` samples, hybrid static context, validation rank_ic20 `0.0653138594`, validation spread20 `0.0138480476`, validation decision IC `0.0939137578`, test rank_ic20 `0.0813052259`, test spread20 `0.0154890155`, test decision IC `0.0525392327`.
- `qdp_pack_hybrid_topn_h256_t4_b512_throughput_20260614_01`: completed, `forecast_test_confirmed`, `style_structural_v1`, throughput cap `900,000` samples, hybrid static context, validation rank_ic20 `0.0790512026`, validation spread20 `0.0156659907`, validation decision IC `0.0489710153`, test rank_ic20 `0.1023188709`, test spread20 `0.0224407061`, test decision IC `0.1072112426`.
- `qdp_pack_hybrid_topn_h192_b768_pilot_20260614_01`: completed, `forecast_test_confirmed`, `style_structural_v1`, full pack `7,421,735` samples, hybrid static context, validation rank_ic20 `0.0869268709`, validation spread20 `0.0194559166`, validation decision IC `0.0915726437`, test rank_ic20 `0.1233131686`, test spread20 `0.0278248650`, test decision IC `0.0915843382`.
- `qdp_training_pack_hybrid_full_b1024_e12_20260613_02`: completed, `forecast_test_confirmed`, `style_structural_v1`, full pack `7,421,735` samples, hybrid static context, validation rank_ic20 `0.0960240056`, validation spread20 `0.0221756604`, test rank_ic20 `0.1176893796`, test spread20 `0.0265627053`; decision utility status was `not_available`.
- `qdp_training_pack_progress_smoke_20260613_01`: completed training/protocol smoke on `style_structural_v1`, `384` samples; verdict `forecast_failed` because tiny smoke is not model-quality evidence.
- `qdp_training_pack_hybrid_pilot_b512_20260613_01`: completed pilot on `style_structural_v1`, `24,576` samples; verdict `forecast_failed`.
- `qdp_sharded_hybrid_static_smoke_20260613`: completed sharded memmap hybrid static smoke on `style_structural_v1`, `48` samples; verdict `forecast_failed`.
- `qdp_sharded_linear_smoke_20260613`: completed sharded memmap linear smoke on `style_structural_v1`, `96` samples; verdict `forecast_failed`.

## Alpha V2 Consumption Smoke
- Command evidence: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol --stage forecast-walkforward-study --tag qdp_alpha_v2_label_v2_pack_consumption_smoke_20260616_01 --data-source lake --lake-dataset-id policy_input_bundle__f926a496f69c61f6b92b5faf --execution-mode next_open --forecast-dataset-mode memmap --forecast-memmap-manifest quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json --forecast-train-start-year 2012 --forecast-train-end-year 2023 --forecast-validation-year 2024 --forecast-test-year 2025 --forecast-model-families gru_sequence_static_context --forecast-epochs 1 --forecast-min-epochs 1 --forecast-early-stop-patience 1 --forecast-batch-size 32 --forecast-hidden-dim 32 --forecast-gru-layers 1 --forecast-transformer-layers 1 --forecast-transformer-heads 1 --forecast-max-samples-per-role 64 --forecast-max-samples-per-date-per-role 16 --forecast-device cpu --no-forecast-amp --no-forecast-save-last --forecast-dataloader-num-workers 0`.
- Summary path: `daily_research/output/path_policy/studies/qdp_alpha_v2_label_v2_pack_consumption_smoke_20260616_01/study_summary.json`.
- Protocol status: `completed`; `forecast_memmap_manifest_fast_path=true`; `training_summary.status=completed`; selected model family `gru_sequence_static_context`; selected checkpoint exists.
- Consumed sample split: train/validation/test `64/64/64`; role years `2012-2023 / 2024 / 2025`.
- Confirmed contract: artifact `qdp_training_pack_v1`, profile `style_structural_alpha_v2`, feature count `307`, schema hash `38a98a4a96637df6bc465837`, label schema `path20_basic_v2` version `2`, static context enabled fields `symbol, exchange, industry`, date-major layout `date_stock_feature`.
- Evidence verdict: `forecast_failed`, expected for `smoke_only` with 1 epoch and 64 samples per role; this does not negate the consumption contract.
- Boundaries retained: `shadow_only=true`, `promotion_allowed=false`, `active_execution_strategy_expected_diff=none`.

## Current Semantics
- QDP owns canonical data and registries; daily_research consumes explicit QDP manifests for research/model/backtest evidence.
- The current daily data substrate is QDP canonical full sharded memmap plus explicit QDP training packs. Old daily lake bundle ids and copied memmap manifests are lineage/history unless an explicit run consumes them.
- `style_structural_v1` runs are no longer the latest data state; they are prior QDP-pack model evidence.
- `style_structural_alpha_v2_label_v2` is the latest data state and has passed consumption smoke, but it has not yet produced evidence-grade model comparison.
- Capsule/current-frontier are optional diagnostics and freshness guards, not a hard prerequisite to enter the daily_research brain when the user has already provided context and files can be read directly.

## Next Allowed Actions
- Rebuild `daily_research/brain/references/evidence_registry.json` after this reference is added.
- Run `current-frontier --json`; expected outcome is no stale 2026-06-13/14 QDP-pack runs, with only genuinely newer output needing reconciliation.
- Next model work should run same-config `style_structural_v1` versus `style_structural_alpha_v2_label_v2` comparisons before making any quality claim.
- If alpha_v2 comparison is promising, move to evidence-grade multi-seed runs; do not use smoke/pilot/datecap evidence for promotion, paper/live, broker, or active artifact changes.
