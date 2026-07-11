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
`retention_policy`: `daily_research/brain/research_store_retention_policy.json`；model-ready data protects the source view, four formal purged folds, and the deprecated 2025 view retained for historical reproducibility; QDP active data remains read-only unless explicitly changed.

### object `seq100_mainline`
`owner`: `daily_research`
`contract`: `daily_research/brain/references/path_policy_seq100_mainline_slimming_contract_20260707.md`
`cli`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline`
`default_view`: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
`formal_walkforward_views`: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_purged_oos{2022,2023,2024,2025}.json`
`deprecated_rollforward_2025_view`: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json`；historical reproduction only, not a new verdict input.
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
`validation`: recompute `max(train.date_idx + forward_days) < min(oos.date_idx)`；verify manifest/schema and normalization cutoff；fixed OOS must not select checkpoints；compare paired daily IC/TopK on identical universe hashes；keep complete-case and execution boundaries explicit.

### procedure `seq100_mainline_default`
`input`: train, compare, or summarize the current seq100 today-close path-value line.
`profile_registry`: `daily_research.path_policy.seq100_mainline.PROFILE_SPECS` is the single source for command names, run tags, input/loss/model overrides and parser defaults.
`help_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline --help`
`contract_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline contract --json`
`single_model_train_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --json`；compatibility/diagnostic entry only, not a formal verdict by itself.
`dry_run_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline <train-command> --dry-run --json`
`summary_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline summarize --run-dir <run_dir> --json`
`fixed_profile`: `store_view=seq100_path60_todayclose_ohlcva`；`model_type=gru_ohlcva_aux_path_value`；`input=daily_only`；`summary=multi_horizon_ohlc`；`loss=path0.45/summary0.20/value0.20/rank0.15/va0.02/0.01`；`prediction_mode=compact`。
`comparison_rule`: every non-default command shown by `--help` is comparison-only unless a dated evidence reference changes the default; all-channel, price-delta, equal-OHLCVA, input-ablation and direct-value families remain available through the registry.
`primary_evaluation_rule`: use purged expanding outer OOS folds with `train.label_end_trade_date < oos_start_trade_date`; freeze profile/seed/epoch/primary metric/checkpoint policy before OOS; do not reuse OOS for early stopping or best-epoch selection.
`walkforward_cli`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_walkforward`
`walkforward_build_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_walkforward build-folds --oos-years 2022-2025 --json`
`walkforward_run_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_walkforward run-study --oos-years 2022-2025 --profiles summary_v2_all_channels,daily_only_summary_v2_ohlcva_aux_low --seed 7 --epochs 1 --device cuda --bootstrap-replications 10000 --json`
`walkforward_summary_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_walkforward summarize-study --bootstrap-replications 10000 --block-length 60 --json`
`fixed_oos_contract`: only `train/oos`; `evaluation_mode=fixed_oos`；`early_stopping_patience=0`；save final checkpoint before evaluating OOS exactly once；retain daily TopK and compact Top100 candidates even when `prediction_mode=none`.
`side_effects`: research artifacts only；does not activate execution surface.

### procedure `research_store_gc`
`input`: space pressure, cold shared components, partial/smoke runs, or large prediction outputs.
`retention_policy`: `daily_research/brain/research_store_retention_policy.json`; only components unreachable from its active views and explicitly allowlisted cold may be pruned.
`scan_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc scan --write-report --json`
`trim_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc trim-predictions --json`
`cold_prune_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc prune-cold-store --json`
`index_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view refresh-index --json`
`view_verify_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view verify --json`
`guards`: directory GC uses `DELETE_RESEARCH_ARTIFACTS`; prediction trim uses `TRIM_PREDICTION_OUTPUTS`; cold-store prune uses `DELETE_COLD_RESEARCH_STORE_COMPONENTS` and archives manifest/hash/provenance before deletion.
`prevention_rule`: default prediction mode is compact; full row payloads require explicit opt-in. QDP active datasets are outside this procedure.
`validation`: post-scan candidate sets empty；all six protected views verify；formal folds have zero label overlap；all-channel `[100,84]` and daily-only `[100,32]` sample loads succeed.

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
- Task capsule: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- Frontier: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`
- Seq100 profiles: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline --help`
- Seq100 contract / default dry-run: `... seq100_mainline contract --json` / `... seq100_mainline train --dry-run --json`
- Research-store scan / verify: `... research_store_gc scan --json` / `... research_store_view verify --json`
- Evidence rebuild: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json`
- Selective verification: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`
- Brain guards: `tools.brain.doc_guard check --scope changed`；`tools.brain.integrity_check --json`；`brain_runtime.py brain-structure-audit --cwd . --mode compact`

## Validation Selection Function
- `brain_docs_changed -> doc_guard + integrity_check + brain_structure_audit`
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
