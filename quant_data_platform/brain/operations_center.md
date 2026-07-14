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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m --full --json
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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --provider mootdx --as-of-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --provider tushare-proxy --as-of-date 2026-07-13 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 ingest --provider baostock --mode date-snapshot --start-date <date> --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 ingest --provider baostock --mode date-events --start-date <date> --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 ingest --provider tushare-proxy --mode historical --domain intraday --start-date 2010-01-01 --end-date 2026-07-13 --resume --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 build candidate --start-date 2010-01-01 --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 audit --candidate <id> --semantic --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 diff --candidate <id> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 publish --candidate <id> --expect-active-sha <sha> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 gc --dry-run --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 update --bootstrap --historical-provider tushare-proxy --start-date 2010-01-01 --as-of-date 2026-07-13 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli retire-v2-intraday --expect-active-sha <v3-active-sha> --delete --yes --json
```

BaoStock full compatibility、Tushare 5m-only compatibility 与 mootdx 5m 协议 gate 当前均通过；Tushare gate 还必须证明 8,000 行反向分页、2010 早期覆盖和分钟股/元单位。任何后续报告失败时，对应 raw 不得进入 canonical。不要用 `--no-publish` 绕过 raw/identity/factor/5m 质量门来制造“可用”结论。

## Procedure Entries
### procedure `describe_active_table`
`input`: table/domain name
`steps`: run `qdp describe <table>` for human summary；run `qdp describe <table> --full --json` only for manifest internals.
`side_effects`: none.

### procedure `check_active_data_base`
`input`: quick or full mode
`steps`: quick for manifest/coverage；`check meta --writeback` for PIT/meta/factor/index exact proof；当前 v2 可做 1m/5m 旧合同扫描，v3 只做 5m 与 daily 的 PK/聚合证据，不再要求 1m。避免日常运行 monolithic `check --full`。
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
`input`: BaoStock/Tushare/mootdx compatibility proof、已授权的 v2 M0 freeze、`2010-01-01..2026-07-13` 固定历史范围。
`steps`: Tushare reference domains；BaoStock date-snapshot/date-events；stable identity/symbol history；Tushare 两波历史 5m；legacy factor history + xdxr/official arbitration；次级 PIT；5m-only candidate；semantic audit；diff；CAS publish。
`resume`: 低频按日/证券写 job；Tushare 5m 每页写 staging/cursor，每证券完整后合并成一个 immutable raw 并清 staging。内容 hash 相同则复用；不得恢复旧的逐月远端回翻，也不得生成 1m job。
`download_runtime`: Tushare 3 workers、最多 3 in-flight、135 次/分钟目标、共享 429 冷却、低于 200 GiB 暂停；BaoStock 网络并发 1；mootdx 并行 TCP 排序后以真实 `frequency=0` 48-bar 请求选最快 3 个健康节点，失败负缓存 300 秒；财务按生命周期裁剪。
`hard_stop`: 任一 compatibility 失败、M0 未授权、raw quarantined、未映射主板 identity、factor disputed/baseline unproven、2010 起 strict 5m 覆盖低于 99.95%、5m watermark 不等于 daily 或 semantic audit 非 passed。

### procedure `qdp_v3_incremental_update`
`input`: existing v3 active and as-of date。
`steps`: 最近 10 个交易日日线重取；最近 60 个交易日因子事件重取；mootdx 最近 10 日 5m；BaoStock 完整日 fallback/抽检；PIT 与衍生表重建；candidate semantic audit；CAS publish。
`boundary`: 5m watermark 不得落后或超前于日线；没有 1m watermark/lag。active CAS 失败时 active byte content 不变；训练包/memmap 不属于此 DAG。

### procedure `retire_v2_intraday_after_v3_publish`
`input`: 已发布 v3 active SHA 与显式 `--delete --yes`。
`steps`: 验证 v3 不含 1m、5m 覆盖至少 99.95%、watermark 等于日线、semantic audit/diff/hash 全通过、下游已切换且无运行 job/消费者；写 immutable retirement manifest；只删除 v2 `market_intraday_1m`、`market_intraday_5m`、`intraday_daily_features`、`limit_intraday_features` parquet；再跑 v3 status/check/path scan。
`boundary`: v3 尚未发布或任一前置失败时命令必须拒绝删除；保留旧 manifest、文件清单、字节数和替代 dataset id。

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
