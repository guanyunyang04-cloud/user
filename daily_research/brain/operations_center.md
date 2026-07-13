# Daily Research 过程目录
快照日期：`2026-07-12`

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
`retention_policy`: `daily_research/brain/research_store_retention_policy.json`；显式保护 candidate-complete v7、四个 development folds、v1→v2→v3 证据链和兼容 replay views；QDP active data remains read-only unless explicitly changed.

### object `seq100_development`
`owner`: `daily_research`
`current_cli`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development`
`compatibility_engine`: `daily_research.path_policy.seq100_research_generation`；保留冻结 registry 的复现和旧 screen/confirmation/fixed-OOS 命令，不是新研究热路径。
`source_pack`: `daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_candidate_complete_2012_2025_v7/manifest.json`
`formal_fold_views`: registry 所绑定的 `seq100_path60_todayclose_ohlcva_development_{2022,2023,2024,2025}.json`。
`rule`: 当前无 default/champion profile；baseline 仅作 control，hard-ST 已否决，global-tail 暂停，直到一个 execution-aligned soft-exit 候选通过四年资格门。

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
`input`: candidate-complete v7 pack、exit-policy audit、single model/loss/value-function change、2022-2025 development evaluation request.
`steps`: first audit fixed/predicted/oracle executable exits on frozen candidates and costs；change one core mechanism at a time；register the full four-fold matrix through `seq100_development`；run every fold on all eligible rows；select by the frozen gates；write compact current conclusion to `state_center.md` and dated evidence to `references/`. If the user terminates the direction or a frozen gate becomes mathematically unreachable, stop all related processes, mark unfinished entries cancelled/result-invalid, keep `winner=null`, and write an explicit early-termination reference instead of silently leaving a runnable registry.
`validation`: require `max_label_dependency_date_idx < development_start_date_idx`；normalization only uses feature dates before development；checkpoint selection uses only `development_total_loss`；Top1/3/5/10 cost-adjusted realized-plan metrics decide candidate eligibility；keep opportunity score、realized-plan return and live PnL separate.

### procedure `seq100_corrected_source_pack`
`source_view`: `quant_data_platform/data/qdp_v2/views/seq100_pit_2012_2025_formal__9d6feb2a5ba7f15a7635b42f.json`.
`manifest`: `daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_candidate_complete_2012_2025_v7/manifest.json`.
`semantics`: lookback100/forward60、today-close、back-adjusted OHLC、candidate/supervision 双索引、tick-rounded open-below-limit entry、signal-day PIT pool、carry-adjusted-close suspension valuation、unfilled retained、separate price/VA/availability masks.
`split_boundary`: source index uses 2012-2025 as one source role and contains no fixed validation/test；development builders own the four train/development roles and fold-specific normalization.
`validation`: run `qdp_v2_sequence_path_pack validate --manifest <manifest> --json` and require `status=ok`, PIT price missing=0, exact five-domain view binding and active manifest unchanged.

### procedure `seq100_development_workflow`
`input`: explicit study root、fold store root、candidate profile set；baseline is control, not an implied champion.
`contract_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development contract`
`register_command`: `... seq100_development register --root <study-root> --store-root <fold-store-root> --profiles <profiles>`
`run_command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe tools/memory_guard.py --min-available-gb 1.0 -- C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development run --root <study-root> --registry <study-root>/development_registry.json`
`select_command`: `... seq100_development select --registry <study-root>/development_registry.json --ledger <study-root>/result_ledger.json`
`freeze_command`: `... seq100_development freeze --root <study-root> --registry <registry> --ledger <ledger> --champion <profile>`；only after `winner != null`，otherwise blocked by code.
`fixed_contract`: 2022-2025 purged expanding `train/development`、seed 7、all eligible rows、minimum 1 / maximum 10 complete epochs、`development_total_loss` patience 2、restore best checkpoint；无历史 test/outer audit。
`metric_roles`: checkpoint=`development_total_loss`；candidate gate=Top1/3/5/10 cost-adjusted realized-plan alpha/stress/coverage/year stability；diagnosis=opportunity alpha/oracle regret/rank IC/path MAE/fill rate。
`historical_compatibility`: `seq100_mainline` keeps profile aliases and single-run diagnostics；`seq100_research_generation` keeps retired screen/confirmation/fixed-OOS replay. Neither route may bypass a new development registry to form a current verdict.
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
`validation`: post-scan safe-delete candidates empty；candidate-complete v7 and v1→v2→v3 evidence hashes unchanged；current development folds have zero label-dependency overlap；daily-only `[100,32]` sample loads succeed.

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
- Seq100 current workflow/contract: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development --help` / `... seq100_development contract`
- Seq100 historical profiles: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline --help`
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
