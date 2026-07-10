# Daily Research 过程目录
快照日期：`2026-07-10`

本文件保存可调用过程、环境基线、命令入口和验证选择。它描述“怎么做”，不承担当前事实长卷；当前对象实例见 `state_center.md`。

## Runtime Objects
### object `workspace_runtime`
`cwd`: `H:\quant_project`
`branch_default`: `main`
`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`
`deprecated_path`: `H:\new_tdx64\PYPlugins\user` 只作历史 reference，不作当前入口。
`env_note`: `KMP_DUPLICATE_LIB_OK` 不作为默认方案。

### object `qdp_dependency`
`owner`: `quant_data_platform`
`daily_research_role`: consumer of QDP v2 active tables, dataset manifests and explicit downstream packs.
`provider_route`: online provider, CSV or external source work enters through QDP update/rebuild/check flows.
`current_data_base`: see `state_center.md` object `qdp_consumption`.

### object `research_store`
`owner`: `daily_research`
`preferred_root`: `daily_research/data/research_store`
`compat_roots`: none active; old `quant_data_platform/data/qdp_v2/research/` artifacts were physically migrated or deleted on 2026-07-07.
`rule`: model-ready data uses shared `panel_store`、`label_store`、`sample_index` and lightweight `views`; QDP active data remains read-only source unless a task explicitly changes active datasets.

### object `seq100_mainline`
`owner`: `daily_research`
`contract`: `daily_research/brain/references/path_policy_seq100_mainline_slimming_contract_20260707.md`
`cli`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline`
`default_view`: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
`rollforward_2025_view`: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json`
`rule`: default work uses the today-close daily-only summary_v2 profile with low-weight VA auxiliary supervision; legacy all-channel base, old pure daily-only summary_v2, richer, residual, symbol, OHLCVA-unified and rank-heavy variants are comparison/archived surfaces unless explicitly requested.

### object `execution_runtime`
`state`: frozen skeleton / read-only diagnostics / candidate wrappers.
`active_artifact`: `daily_research/output/active_execution_strategy.json`; missing or ignored-untracked is not a clean execution state.
`app_entry`: `conda run -n yolos python daily_research/execution/run_execution_app.py web --port 8765`
`manual_flow`: refresh data/signals -> generate trade plan -> simulate account posting -> review status.
`activation`: only execution tasks select this object.

## Procedure Entries
### procedure `brain_maintenance`
`input`: changed brain docs, registry, workflow, skill, or governance files.
`steps`: keep hot-path docs compact；move long history to `references/`；run doc guard, integrity check, structure audit；sync skill when skill files changed.
`validation`: `doc_guard changed`；`integrity_check --json`；`brain-structure-audit --mode compact`。

### procedure `shortline_research_work`
`input`: QDP pack / manifest, feature diagnostics, shortline condition priors.
`steps`: read explicit QDP artifacts；build or inspect scorer / feature / candidate evidence；classify evidence；write current summary to state and durable details to reference.
`validation`: score decile, top-k / top-decile expectation, validation-selected same-candidate test, cost-aware backtest where applicable.

### procedure `seq100_path_value_research_work`
`input`: QDP v2 active tables or existing compatibility sequence pack, model/loss/value-function change, evaluation request.
`steps`: route task as primary `daily_research` with supporting read-only `quant_data_platform` when QDP data is referenced；start from `seq100_mainline` unless the user explicitly asks for a comparison branch；prefer `--store-view daily_research/data/research_store/views/<view>.json` over self-contained full packs；train/evaluate models；write compact current conclusion to `state_center.md` and dated evidence to `references/`.
`validation`: no PIT leakage, manifest/schema consistency, validation/test predictions, rank IC, topK realized/path value spread, comparison only against named baseline surfaces.

### procedure `seq100_mainline_default`
`input`: train/evaluate/summarize the current seq100 path-value mainline.
`train_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --json`
`contract_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline contract --json`
`dry_run_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --dry-run --json`
`summary_v2_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2 --json`
`summary_v2_no60_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-no60 --json`
`summary_v2_price_delta_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-price-delta --json`
`summary_v2_ohlcva_aux_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux --json`
`summary_v2_ohlcva_aux_low_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux-low --json`
`summary_v2_ohlcva_aux_low_price_delta_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux-low-price-delta --json`
`summary_v2_ohlcva_path_equal_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-path-equal --json`
`legacy_all_channels_base_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-legacy-all-channels-base --json`
`daily_only_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only --json`
`daily_only_summary_v2_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2 --json`
`no_intraday_summary_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-intraday-summary --json`
`no_limit_structure_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-limit-structure --json`
`direct_value_5d_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-5d --json`
`direct_value_10d_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-10d --json`
`direct_value_60d_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-60d --json`
`daily_only_summary_v2_price_delta_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-price-delta --json`
`daily_only_summary_v2_ohlcva_aux_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux --json`
`daily_only_summary_v2_ohlcva_aux_low_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low --json`
`daily_only_summary_v2_ohlcva_aux_low_price_delta_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low-price-delta --json`
`daily_only_summary_v2_ohlcva_path_equal_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-path-equal --json`
`summary_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline summarize --run-dir <run_dir> --json`
`fixed_profile`: `store_view=seq100_path60_todayclose_ohlcva`；`model_type=gru_ohlcva_aux_path_value`；`input_channel_profile=daily_only`；`summary_loss_profile=multi_horizon_ohlc`；`loss=path0.45/summary0.20/value0.20/rank0.15/va_level0.02/va_delta0.01`；`top_k=1,3,5,10,20,50,100`；`prediction_mode=compact`。
`legacy_broad_topk_baseline`: `all_channels_base_summary` uses `input_channel_profile=all` and `summary_loss_profile=base`; keep it as explicit broad-TopK comparison, not the default research entrance.
`comparison_profile`: `summary_v2_multi_horizon_ohlc` keeps `model_type=gru_path_value` and `path_dim=4`, but changes `summary_loss_profile=multi_horizon_ohlc` to constrain OHLC-derived 5/10/20/40/60-day summaries; the narrow CLI default uses `early_stopping_patience=2`.
`all_channel_summary_v2_combo_profile`: `train-summary-v2-price-delta`、`train-summary-v2-ohlcva-aux`、`train-summary-v2-ohlcva-aux-low`、`train-summary-v2-ohlcva-aux-low-price-delta` and `train-summary-v2-ohlcva-path-equal` keep `input_channel_profile=all` and `summary_loss_profile=multi_horizon_ohlc`, then add the requested price-delta, VA auxiliary or equal OHLCVA path-loss constraint. The 2026-07-10 result says none replaces the default `daily_only_summary_v2_ohlcva_aux_low`; retain them as all-channel comparison branches.
`summary_control_profile`: `summary_v2_no60` keeps the same OHLC-derived summary family and loss weight as `summary_v2_multi_horizon_ohlc`, but changes `summary_loss_profile=multi_horizon_ohlc_no60`, using windows `5/10/20/40` only. It is a single-factor control and must not include new shape summaries.
`input_ablation_profile`: `daily_only_no_minute` keeps labels/loss/splits fixed but uses `input_channel_profile=daily_only`, removing `intraday_summary` and `limit_structure`; old `daily_only_summary_v2` combines `input_channel_profile=daily_only` with `summary_loss_profile=multi_horizon_ohlc` and remains a pure-OHLC comparison; `no_intraday_summary` keeps `daily_raw + daily_state + limit_structure`; `no_limit_structure` keeps `daily_raw + daily_state + intraday_summary`; the narrow CLIs default to `early_stopping_patience=2`.
`price_delta_profile`: `daily_only_summary_v2_price_delta` keeps `model_type=gru_path_value`, `input_channel_profile=daily_only`, `summary_loss_profile=multi_horizon_ohlc`, and adds `price_delta0.03` auxiliary SmoothL1 on first differences of anchored close log returns.
`ohlcva_aux_profile`: default `daily_only_summary_v2_ohlcva_aux_low` uses `model_type=gru_ohlcva_aux_path_value`, keeps `input_channel_profile=daily_only` and `summary_loss_profile=multi_horizon_ohlc`, keeps price-only `path_trade_value_v2` for summary/value/rank, and adds low `va_level0.02/va_delta0.01` auxiliary losses plus split metrics `va_level_mae/va_delta_mae`; high-weight `daily_only_summary_v2_ohlcva_aux` uses `va_level0.05/va_delta0.02` and remains comparison only; `daily_only_summary_v2_ohlcva_aux_low_price_delta` combines the low VA weights with `price_delta0.03`.
`ohlcva_path_equal_profile`: `daily_only_summary_v2_ohlcva_path_equal` uses `model_type=gru_ohlcva_aux_path_value`, sets `path_loss_profile=ohlcva_equal`, averages SmoothL1 over OHLCVA six fields inside main `path_loss`, keeps `future_path` as 4D OHLC for summary/value/rank, and sets `price_delta/va_level/va_delta` auxiliary weights to zero.
`primary_evaluation_rule`: compare seq100 profiles mainly on `2024 validation` plus `2025 train-through-2024 forward test`; use 2024 validation for profile selection/training stability, and use 2025 forward test for the recent production-like out-of-sample check. Do not treat the old 2012-2023-trained 2025 test as the primary test口径.
`rollforward_2025_profile`: use `--store-view daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json --epochs 1 --early-stopping-patience 0`; the view labels 2012-2024 as train and duplicates 2025 as validation/test only to satisfy the trainer, so formal interpretation uses the `test` split.
`direct_value_profile`: `direct_value_rank_5d/10d/60d` uses `model_type=gru_direct_value`, outputs only `score`, uses `loss=value0.50/rank0.50`, sets path/summary/richer loss to zero, and derives the supervised target from true future OHLC labels at the requested horizon; it is a comparison profile, not the default mainline.
`implementation_note`: multi-horizon OHLC summary-loss derivation is vectorized in training while preserving the old per-horizon equal-weight loss semantics.
`side_effects`: research artifacts only；does not activate execution surface.

### procedure `research_store_gc`
`input`: H: space pressure, repeated sequence packs, smoke/partial packs, large prediction outputs, or research artifact cleanup request.
`steps`: run dry-run scan；review safe directory candidates and prediction trim candidates；never delete QDP active datasets from this procedure；only execute directory deletion or prediction-output trim with explicit confirmation token after the dry-run report is reviewed.
`commands`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc scan --write-report --json`
`migration_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc migrate-legacy-packs --write-report --json`
`trim_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc trim-predictions --write-report --json`
`view_build_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view build-from-legacy-packs --json`
`view_verify_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view verify --json`
`delete_guard`: deletion requires `--delete --confirm-delete DELETE_RESEARCH_ARTIFACTS`; safe directory deletion is limited to unreferenced smoke/partial/interrupted artifacts under configured research roots.
`trim_guard`: prediction trim requires `--delete --confirm-trim TRIM_PREDICTION_OUTPUTS`; it deletes only large prediction CSV/parquet/feather files under the studies root and retains summary JSON, metrics CSV, reports and checkpoints.
`prevention_rule`: sequence path training defaults to `--prediction-mode compact`; full path-level prediction CSVs require explicit `--prediction-mode full --allow-large-predictions`. Flat LightGBM does not write predictions unless `--write-predictions` is provided.
`validation`: inspect JSON/Markdown report；confirm active QDP paths are absent from delete candidates；run `git diff --check` after code/doc changes.

### procedure `qdp_data_request`
`input`: missing field, stale dataset, provider coverage gap, canonical requirement.
`steps`: express requirement as QDP table/update/rebuild work；run or request `python -m quant_data_platform.cli ...`；return explicit table/domain, manifest or downstream pack to daily research.
`validation`: QDP `status`, `describe`, `check`, source conflict report when relevant.

### procedure `evidence_query`
`input`: claim, run tag, dataset id, r-number, status question.
`steps`: query registry；open explicit summary/reference；separate fact / inference / missing evidence；answer with evidence entrypoints.
`commands`: `python -m tools.brain.workflow query --q <tag|dataset_id|r_id> --json`

### procedure `long_run_observation`
`input`: background PID, stdout/stderr, progress, summary, artifact path.
`steps`: register or inspect run；observe logs and artifacts；classify timeout by process health and artifact movement；write evidence grade after outcome is observed.
`commands`: `python -m tools.brain.agent_run paths/register/launch/status --project-id daily_research --run-id <run>`

### procedure `execution_readonly_or_change`
`input`: execution task, explicit authorization when change is requested.
`steps`: select `execution_runtime`；inspect active artifact and daily verdicts；for change, call governance procedure `execution_change`.
`validation`: `project_consistency_check.py --mode execution` or `--mode full` only when execution surface is active.

## Command Palette
- Task capsule sensor: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- Current frontier sensor: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`
- Seq100 mainline contract: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline contract --json`
- Seq100 mainline dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --dry-run --json`
- Seq100 legacy all-channel base dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-legacy-all-channels-base --dry-run --json`
- Seq100 summary_v2 dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2 --dry-run --json`
- Seq100 summary_v2 no60 dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-no60 --dry-run --json`
- Seq100 all-channel summary_v2 price-delta dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-price-delta --dry-run --json`
- Seq100 all-channel summary_v2 OHLCVA aux dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux --dry-run --json`
- Seq100 all-channel summary_v2 OHLCVA aux low dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux-low --dry-run --json`
- Seq100 all-channel summary_v2 OHLCVA aux low + price-delta dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux-low-price-delta --dry-run --json`
- Seq100 all-channel summary_v2 OHLCVA equal path-loss dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-path-equal --dry-run --json`
- Seq100 daily-only dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only --dry-run --json`
- Seq100 daily-only summary_v2 dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2 --dry-run --json`
- Seq100 no-intraday-summary dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-intraday-summary --dry-run --json`
- Seq100 no-limit-structure dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-limit-structure --dry-run --json`
- Seq100 direct-value 5d dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-5d --dry-run --json`
- Seq100 direct-value 10d dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-10d --dry-run --json`
- Seq100 direct-value 60d dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-60d --dry-run --json`
- Seq100 daily-only summary_v2 price-delta dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-price-delta --dry-run --json`
- Seq100 daily-only summary_v2 OHLCVA aux low dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low --dry-run --json`
- Seq100 daily-only summary_v2 OHLCVA aux low + price-delta dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low-price-delta --dry-run --json`
- Seq100 daily-only summary_v2 OHLCVA equal path-loss dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-path-equal --dry-run --json`
- Evidence registry rebuild: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json`
- Research artifact GC dry-run: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc scan --write-report --json`
- Selective verification: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`
- Brain doc guard: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`
- Brain integrity: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- Brain structure: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-structure-audit --cwd . --mode compact`

## Validation Selection Function
- `brain_docs_changed -> doc_guard + integrity_check + burden_audit`
- `qdp_artifact_changed -> QDP status / describe / check`
- `research_code_changed -> selective_verification blocking commands`
- `execution_surface_active -> project_consistency execution/full + active artifact diff inspection`
- `small_doc_or_analysis -> git diff --check + targeted guard`

## Writeback Routes
- Current object instances and next method: `daily_research/brain/state_center.md`
- Stable object classes and long-term lessons: `daily_research/brain/knowledge_center.md`
- Procedure entries and commands: `daily_research/brain/operations_center.md`
- Object invariants and guard logic: `daily_research/brain/governance_layer.md`
- Durable evidence, long commands, full reviews: `daily_research/brain/references/`
- Machine evidence index: `daily_research/brain/references/evidence_registry.json`
