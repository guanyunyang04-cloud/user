# Quant Data Platform 过程目录
快照日期：`2026-07-01`

## Runtime Objects
### object `qdp_body_map`
`code`: `quant_data_platform/src`
`data`: `quant_data_platform/data/qdp_v2`
`references`: `quant_data_platform/brain/references`
`tests`: `quant_data_platform/tests`
`archives`: `tools/archive_v1`、`tools/archive_repair`

### object `qdp_python_env`
`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`
`pythonpath`: `H:/quant_project/quant_data_platform/src;H:/quant_project`
`note`: prefer direct `yolos/python.exe` for brain tools when `conda run` stdout encoding is noisy.

## Current Command Palette
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_1m
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_1m --full --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --no-write
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --full --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild 5m --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild daily-panel --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild valuation --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild limit-intraday --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli gc --dry-run --with-size --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --dry-run --runtime fast --json
```

## Procedure Entries
### procedure `describe_active_table`
`input`: table/domain name
`steps`: run `qdp describe <table>` for human summary；run `qdp describe <table> --full --json` only for manifest internals.
`side_effects`: none.

### procedure `check_active_data_base`
`input`: quick or full mode
`steps`: quick for manifest/footer/coverage；full for deep row/cross-frequency audit.
`side_effects`: optional audit files only.

### procedure `rebuild_cache_or_feature`
`input`: target `5m|daily-panel|valuation|limit-intraday`
`steps`: rebuild from active raw facts or explicit input dataset；validate new manifest；activate only after audit.
`side_effects`: new dataset manifest and optional active pointer update.

### procedure `garbage_collect_data_base`
`input`: dry-run or explicit delete
`steps`: traverse active manifest graph；list unreferenced dataset dirs；delete only with explicit `--delete --yes`.
`side_effects`: none in dry-run.

### procedure `brain_sync_after_qdp_change`
`input`: changed QDP CLI/data scope/table semantics/quality conclusion
`steps`: update QDP hot-path brain docs；update consumers if their data dependency wording changed；run brain sync audit and doc/integrity checks.

## Validation Selection
- `qdp_cli_changed -> py_compile + qdp --help + focused qdp tests`
- `active_manifest_or_dataset_changed -> qdp status + qdp check --quick + qdp gc --dry-run`
- `raw_data_or_cache_changed -> qdp check --full when feasible + targeted cross-frequency audit`
- `brain_docs_changed -> brain_sync_audit + doc_guard changed + integrity_check`

## Writeback Routes
- Current data pointers and table status: `state_center.md`
- Stable source semantics and lessons: `knowledge_center.md`
- Commands and process entries: `operations_center.md`
- Protected data invariants: `governance_layer.md`
- Long audits/provider reports: `references/` or `data/audits`
