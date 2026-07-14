# Quant Data Platform 过程目录
快照日期：`2026-07-14`

## Runtime Objects
### object `qdp_body_map`
`code`: `quant_data_platform/src`
`data`: active `quant_data_platform/data/qdp_v2`；rebuild/candidate `quant_data_platform/data/qdp_v3`
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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild 5m --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild daily-panel --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild valuation --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild scope-active --runtime fast --workers 4 --activate --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild pit-signal-universe --runtime fast --threads 4 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild pit-market-substrate --snapshot-root <snapshot-or-recovery-root> --market-bars-source <optional-manifest> --factor-events <manifest> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli verify pit-market-view --view <view.json>
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild industry-concept-filled --runtime fast --activate --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild share-capital-daily --runtime fast --activate --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild valuation-market-cap --runtime fast --activate --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild limit-intraday --runtime fast --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli gc --dry-run --with-size --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --dry-run --runtime fast --json
```

QDP v3 重建入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --smoke --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 ingest --provider baostock --mode date-snapshot --start-date <date> --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 ingest --provider baostock --mode date-events --start-date <date> --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 build candidate --start-date 2010-01-01 --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 audit --candidate <id> --semantic --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 diff --candidate <id> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 publish --candidate <id> --expect-active-sha <sha> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 gc --dry-run --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 update --as-of-date <date> --dry-run --bootstrap --start-date 2010-01-01 --json
```

`compatibility run` 的当前 full report 已为 `status=passed, scope=full, issue_count=0`；任何后续报告不满足该条件时，live date-snapshot/date-events 和 canonical staging 都会阻断。首次 rebuild 仍受 M0 v2 lineage freeze 阻断；不要用 `--no-publish` 绕过 raw/identity/factor 质量门来制造“可用”结论。

## Procedure Entries
### procedure `describe_active_table`
`input`: table/domain name
`steps`: run `qdp describe <table>` for human summary；run `qdp describe <table> --full --json` only for manifest internals.
`side_effects`: none.

### procedure `check_active_data_base`
`input`: quick or full mode
`steps`: quick for manifest/coverage；`check meta --writeback` for PIT/meta/factor/index exact proof；use targeted PK/cross-frequency scans for 1m/5m and daily/intraday evidence instead of routinely running monolithic `check --full`.
`side_effects`: optional audit files only.

### procedure `rebuild_cache_or_feature`
`input`: target `5m|daily-panel|valuation|scope-active|pit-signal-universe|pit-market-substrate|industry-concept-filled|share-capital-daily|valuation-market-cap|limit-intraday`
`steps`: rebuild from active raw facts or explicit input dataset；validate new manifest；activate only after audit.
`side_effects`: new dataset manifest and optional active pointer update.

### procedure `verify_pit_research_view`
`input`: immutable five-domain research view JSON.
`steps`: run `verify pit-market-view`；require atomic overrides for market/status/limit/factor/PIT；check independent research-eligible non-suspended anti-join, status/PIT row-count alignment, factor positivity/provenance, and active manifest hash.
`formal_view`: `quant_data_platform/data/qdp_v2/views/seq100_pit_2012_2025_formal__9d6feb2a5ba7f15a7635b42f.json`.
`side_effects`: none.

### procedure `garbage_collect_data_base`
`input`: dry-run or explicit delete
`steps`: traverse active manifest graph；list unreferenced dataset dirs；delete only with explicit `--delete --yes`.
`side_effects`: none in dry-run.

### procedure `brain_sync_after_qdp_change`
`input`: changed QDP CLI/data scope/table semantics/quality conclusion
`steps`: update QDP hot-path brain docs；update consumers if their data dependency wording changed；run brain sync audit and doc/integrity checks.

### procedure `qdp_v3_initial_rebuild`
`input`: full BaoStock compatibility proof、已修复的 v2 M0 freeze、start/end date。
`steps`: calendar；security master；由旧到新 date-snapshot 与 date-events；stable identity/symbol history；legacy factor history + xdxr/official arbitration；主板双源 5m；次级 PIT；build candidate；semantic audit；diff；CAS publish。
`resume`: 每日、每证券和每 symbol-month 都写 job state；不可变 raw 内容相同则复用 hash，不重复写。5m 的恢复单位仍是 symbol-month，但网络单位是每证券完整目标区间，下载后本地切月；不得恢复旧的逐月远端回翻。
`download_runtime`: BaoStock 网络并发固定为 1 并复用任务内 login；日期任务可做两日期本地预取。mootdx 启动时并行做 TCP/协议健康选择，失败负缓存 300 秒；不可用时立即走 BaoStock。财务季报按证券生命周期加 550 日前置缓冲裁剪。
`hard_stop`: full compatibility 未通过、M0 lineage 缺失、raw partition quarantined、未映射主板 identity、factor disputed/baseline unproven、strict 5m 覆盖不足或 semantic audit 非 passed。

### procedure `qdp_v3_incremental_update`
`input`: existing v3 active and as-of date。
`steps`: 最近 10 个交易日日线重取；最近 60 个交易日因子事件重取；当日 mootdx 5m；BaoStock 缺口修补/抽检；PIT 与衍生表重建；candidate semantic audit；CAS publish。
`boundary`: active CAS 失败时 active byte content 不变；训练包/memmap 不属于此 DAG。

## Validation Selection
- `qdp_cli_changed -> py_compile + qdp --help + focused qdp tests`
- `qdp_v3_provider_or_contract_changed -> compileall + tests/data_platform + compatibility smoke/full proof`
- `qdp_v3_candidate_or_release_changed -> candidate quick/full/semantic audit + active SHA comparison + v3 GC dry-run`
- `active_manifest_or_dataset_changed -> qdp status + qdp check --quick + qdp gc --dry-run`
- `raw_data_or_cache_changed -> qdp check --quick + qdp check meta + targeted PK/cross-frequency audit`
- `brain_docs_changed -> brain_sync_audit + doc_guard changed + integrity_check`

## Writeback Routes
- Current data pointers and table status: `state_center.md`
- Stable source semantics and lessons: `knowledge_center.md`
- Commands and process entries: `operations_center.md`
- Protected data invariants: `governance_layer.md`
- Long audits/provider reports: `references/` or `data/audits`
