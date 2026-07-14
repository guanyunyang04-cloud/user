# Quant Data Platform 过程目录
快照日期：`2026-07-15`

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
`note`: 运行模块、pytest 和脑区工具均优先使用固定 yolos 解释器。

### object `qdp_storage_roots`
`data_root`: `QDP_DATA_ROOT=H:\quant_project\quant_data_platform\data`
`runtime_root`: `QDP_RUNTIME_ROOT=C:\Users\ASUS\AppData\Local\QDP\runtime`
`layout`: H 保存 canonical、manifest 与 compact raw：reference 为 undated x 1，低频为 year x 1，只有 5m 为 year x 16；C 保存 SQLite WAL、job、cursor、heartbeat 和 staging。
`disk_state`: H 已于 2026-07-15 完成 `chkdsk H: /f`；dirty bit=false、health=Healthy、operational=OK、bad sectors=0。用户取消 F 盘备份，F 不属于当前流程。
`shell_boundary`: 新写入的用户级环境变量不会自动进入已经启动的 Codex 进程；在当前进程运行生产命令时显式设置 `$env:QDP_DATA_ROOT`、`$env:QDP_RUNTIME_ROOT` 与 provider secret，但绝不回显 Token。

## Current Command Palette

公共读取与维护：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli gc --dry-run --json
```

read 命令通过 `validate_published_active()` 选择 generation：只有具备 manifest v4、真实候选/哈希、完整九域和 `published_at` 的 v3 active 才能切换默认读取，否则回到 v2。2026-07-15 的空占位已删除，当前默认 status 正确返回 v2。管理命令属于 v3；任何 legacy v2 写命令必须显式给出 `--generation v2`。

QDP v3 管理入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --provider tushare-proxy --as-of-date 2026-07-13 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compatibility run --provider mootdx --as-of-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compact --raw-domain <raw-domain> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 compact --raw-domain <raw-domain> --delete-source --yes --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 ingest --provider tushare-proxy --mode historical --domain intraday --start-date 2010-01-01 --end-date 2026-07-13 --resume --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 build candidate --start-date 2010-01-01 --end-date <date> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 audit --candidate <id> --semantic --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 diff --candidate <id> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 publish --candidate <id> --expect-active-sha <sha> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v3 update --bootstrap --historical-provider tushare-proxy --start-date 2010-01-01 --as-of-date 2026-07-13 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli retire-v2-intraday --expect-active-sha <v3-active-sha> --delete --yes --json
```

## Procedure Entries

### procedure `inspect_qdp`
`input`: generation、active/candidate target、quick/full/semantic mode。
`steps`: `status/list/describe/check` 默认读取有效 generation；v3 未发布而要检查 candidate 时使用 `--generation v3 check --candidate <id> --quick|--full|--semantic`。零 datasets/coverage 的 v3 manifest 不得视为有效 active；检查不等于发布，正式 active 只能通过合格 candidate 的 CAS publish 改变。
`side_effects`: `check` 默认可写 audit report；加 `--no-write-report` 时只读。

### procedure `repair_and_guard_h_volume`
`state`: 已完成，不再是当前 blocker。
`evidence`: `C:\Users\ASUS\AppData\Local\QDP\maintenance\chkdsk_H_20260715.log`
`boundary`: 只有 dirty bit=false、HealthStatus=Healthy、OperationalStatus=OK、bad sectors=0 才允许恢复 H 写入。当前四项均满足；本流程没有 F 盘备份或格式化步骤。

### procedure `compact_legacy_raw`
`input`: 一个或多个 legacy raw domain。
`steps`: 先不删源运行 `compact`；验证源 content hash、行数、schema、bundle SHA、row-group mapping、`raw_index.parquet` 与 `raw_receipts.parquet` sidecar；再对同一小域运行 `--delete-source --yes` 并验证 bundle-aware read/audit/revision；小域通过后才可逐域全量 compact。
`layout`: reference raw 为 undated x 1，低频时序 raw 为 year x 1，只有 5m raw 为 year x 16 stable buckets；zstd，目标 part 约 384 MiB、最大 1 GiB。成功空响应只写 SQLite receipt/index。
`hard_stop`: index export 缺失、hash/row count/schema/mapping 不一致或读取回归失败时不得删除 legacy raw。该流程与 v2 退休互相独立。

### procedure `qdp_v3_initial_rebuild`
`input`: 合同 `qdp_v3_20260715_trusted_source_5m`、已授权 M0 gap、固定历史范围 `2010-01-01..2026-07-13`。
`steps`: 复用已完成 Tushare stock-basic/calendar/daily/daily-basic/identity raw；compact 旧 raw；优先用分钟额度回灌 Tushare 全历史 5m；额度耗尽后只补 `suspend_d/factor`；建立稳定 identity/PIT/corrections；构建九个核心域；quick/full/semantic audit；diff；CAS publish。
`trusted_route`: 历史 canonical 只使用 Tushare，不启动 BaoStock/mootdx 全量逐行验证。已存在 BaoStock 历史 raw 只紧凑归档，不进入 historical canonical。
`runtime`: Tushare 三个 HTTP worker 各自 Session，但共享一个 `96 rpm / burst 1` limiter、quota state 和 429 cooldown；每证券串行 8,000 行反向分页，不同证券并行；30 秒 heartbeat，超过 5 分钟且 OS lock 不存在才重置 stale task。
`release_domains`: `trading_calendar`、`security_identity`、`symbol_history`、`market_daily_raw`、`security_status_daily`、`adjust_factor_daily`、`eligible_signal_D`、`tradable_open_D1`、`market_intraday_5m`。
`hard_stop`: secret 泄漏、schema/unit/PK/identity/PIT/correction/bundle/index 失败、未解释股票日、5m strict coverage 低于 98%、5m/daily watermark 不等或 semantic audit 非 passed。

### procedure `qdp_v3_incremental_update`
`input`: 已发布 v3 active 与 cutoff 后 as-of date。
`steps`: BaoStock 重取最近 10 个交易日日线/状态及最近 60 个交易日因子事件；mootdx 获取最近窗口的完整 5m 股票日，只有不完整时才以 BaoStock 完整股票日 fallback；重建 PIT/candidate；semantic audit；CAS publish。
`boundary`: 不做常态跨源数值仲裁，不拼接或插值不完整分钟日；5m watermark 必须等于 daily watermark。第一次 bootstrap publish 之后，必须至少成功完成一次 `watermark > 2026-07-13` 的增量 publish，才可能退休 v2。

### procedure `publish_and_retire_v2`
`input`: bootstrap candidate、后续增量 candidate、当前 active SHA。
`steps`: 首次 CAS publish 后只写 `awaiting_post_cutoff_incremental` gate，不删除 v2；完成 cutoff 后增量 CAS publish 后验证新 active SHA/candidate、daily/5m watermark、5m coverage、semantic audit、diff、全部 shard hash、下游切换及无消费者/job；写 immutable retirement manifest；再物理删除 v2 `market_intraday_1m`、`market_intraday_5m`、`intraday_daily_features`、`limit_intraday_features` parquet；最后复跑 v3 status/check/path scan。
`boundary`: 任一前置失败返回 `published_cleanup_blocked` 或拒绝删除；保留旧 manifest、文件清单、字节数和替代 dataset id。首次发布本身绝不触发删除。

### procedure `brain_sync_after_qdp_change`
`input`: QDP CLI、contract、storage、provider route、active pointer、quality threshold 或 retirement boundary 变化。
`steps`: 更新 QDP hot-path brain；跨项目依赖变化时更新主脑；运行 brain sync audit、doc guard、integrity check 和 `git diff --check`。

## Validation Selection
- `qdp_cli_or_contract_changed -> compileall + tests/data_platform + qdp --help`
- `raw_compaction_changed -> compact focused tests + source/bundle/index round-trip + secret scan`
- `candidate_or_release_changed -> candidate quick/full/semantic + active SHA comparison + GC dry-run`
- `active_or_dataset_changed -> status + check quick/semantic + shard/hash validation`
- `brain_docs_changed -> brain_sync_audit + doc_guard changed + integrity_check + git diff --check`

## Writeback Routes
- Current data pointers and task status: `state_center.md`
- Stable source/storage semantics: `knowledge_center.md`
- Commands and procedures: `operations_center.md`
- Protected invariants: `governance_layer.md`
- Long evidence/history: `references/` or `data/audits`
