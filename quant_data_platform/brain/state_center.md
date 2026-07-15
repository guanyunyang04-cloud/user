# Quant Data Platform 状态程序
快照日期：`2026-07-15`

## Module Interface
`exports`: QDP v2 active data base status、QDP v3 rebuild/candidate status、dataset manifest summaries、quality/audit entrypoints。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances
### object `qdp_project`
`type`: shared_manifest_first_data_base_instance
`state`: v2 仍是唯一有效、可研究的 active 数据基底；v3 已收敛为可信来源、5m-only、紧凑 raw bundle 合同。旧 Tushare 全市场分钟任务与全量 archive 任务均已停止并保留断点，当前路线先把 v2 的合法完整 5m 日迁移为 v3 自有 raw，再只补最终残差；正式 candidate 和 active 切换均未完成。2026-07-15 发现的空 v3 active 占位已确认无候选/无数据并删除；默认 status 已恢复到 v2。
`public_cli`: 公共面为 `status/list/describe/check/update/gc`，管理面为 `ingest/compact/build/audit/diff/publish/rollback/retire-v2-intraday`。read routing 只接受通过 `validate_published_active()` 的 v3，否则默认 v2；`check` 也可显式指定 candidate，任何 v2 写命令必须显式给出 `--generation v2`。
`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`

### object `qdp_v3_rebuild`
`type/state`: immutable_manifest_v3_candidate_pipeline / interrupted_recoverable_waiting_for_compaction_and_bootstrap
`root`: `quant_data_platform/data/qdp_v3`
`contract`: `qdp_v3_20260715_trusted_source_5m` / schema `3.3.0` / manifest `4`；正式起点 `2010-01-01`；canonical 主键使用稳定 `security_id + 时间键`。新生产数据只发布 `strict/quarantined`，旧 `provisional` 仅可读；v3 不采集、构建、发布或更新 1m。
`providers`: BaoStock `0.9.3` wheel SHA-256 `acbd19403285bc4e254cee8297cf0e2646ae2276e5af7e549deed3988ab02293`；mootdx `0.11.7`；Tushare-compatible proxy 负责低频历史基底与历史 5m 最终残差，`upstream_provenance=not_exposed`，Token 只从环境变量读取。
`implemented_domains`: 首次发布只要求 calendar、identity/symbol history、daily/status、adjust-factor daily、PIT signal/open 与 5m；估值、事件、财务、分红、行业、指数和股本均为非阻塞 enrichment。v3 1m 路由与转换代码已删除。
`compatibility`: 0.9.1 full golden 已捕获；BaoStock 0.9.3 full gate 为 `passed`（34 anchors、0 issues、普通多页 5537 rows、wheel hash 匹配）。该报告仍约束 cutoff 后 BaoStock 增量，但不再要求 BaoStock 证明 Tushare historical canonical。
`legacy_baostock_history`: 2010/2011/2012 已有 `729/729` 个日线、all-stock 和日期因子事件分区，日线 `1,578,095` rows、因子事件 `5,192` rows。它们作为可追溯 legacy evidence 紧凑归档，不进入 Tushare 独占的 historical canonical，也不继续 2013+ 历史复刻。
`proxy_bootstrap`: Tushare `stock_basic`、`trade_cal`、4,011 个全市场 `daily`、4,011 个 `daily_basic` 与 5,864 个 identity 任务已有 raw。分钟代理只接收本地 archive、v2 迁移和 2020+ 免费源之后的最终残差；额度耗尽后只补 `status/factor`。三个 worker 共享单一 `96 rpm / burst 1` limiter；成功但生命周期内为空的分钟响应写 index-only `quarantined` 事实。
`download_runtime`: v2 active 5m 只读叶包含 `402,849,072` 行、`8,392,689` 个 48-bar 股票日、`3,037` 个证券和 `7,861` 个 shard；全部证券可映射到稳定 identity。迁移器以 DuckDB 单次 fan-out、8 个本地 worker 和 0.5 GiB/5 秒资源闸门写成独立 v3 raw，不伪称补回缺失祖先。66 个已完成的本地直接 archive 分区保留并具有更高优先级；其余 archive、2020+ 免费源和 Tushare 只处理残差。不同来源的日线对照仅作诊断，只有 Tushare 5m 与同源日线核对是硬门禁。
`storage_state`: `QDP_DATA_ROOT=H:\quant_project\quant_data_platform\data`；`QDP_RUNTIME_ROOT=H:\quant_project\quant_data_platform\data\qdp_runtime`。数据、job、SQLite、staging 和 DuckDB temp 全部位于本仓库；C 盘旧 runtime 的 161 个文件已逐文件 SHA256 校验迁移后删除。reference raw 为 undated x 1，低频时序 raw 为 year x 1，只有 5m raw 为 year x 16；zstd，目标约 384 MiB/最大 1 GiB。
`disk_health`: H 已于 2026-07-15 完成 `chkdsk H: /f`，dirty bit=false、HealthStatus=Healthy、OperationalStatus=OK、bad sectors=0；日志为 `C:\Users\ASUS\AppData\Local\QDP\maintenance\chkdsk_H_20260715.log`。用户取消 F 盘备份，当前没有 F 盘备份/fallback 前置。
`rejected_parallel_route`: 两个独立 BaoStock login 的 live probe 出现 `10001001 用户未登录`，因此禁止生产双 session；历史低错误率不得自动解锁第二个网络槽。每个 ingest job 有 OS advisory single-writer lock，进程异常退出后自动释放。
`adapter_smoke_boundary`: ETF adapter `1574` rows provisional；mootdx 的 `600000.SH/000001.SZ/600076.SH` 最近完整日均为 48 bars 且止于 15:00。Tushare 5m 的 `vol/amount` 已用沪深、早期/近期固定样本证明为股/元，canonical scale 均为 1.0。
`pit_name_boundary`: stock-basic 当前名称禁止反填历史；Tushare `namechange` 与日期级 snapshot 是合法 PIT 名称证据。代码/名称例外通过区间化 `qdp_v3_corrections.json` 修正，官方文档 hash 不再是发布前置。
`identity_regression`: 官方代码历史固定为 `000022.SZ -> 001872.SZ @ 2018-12-26`、`000043.SZ -> 001914.SZ @ 2019-12-16`、`300114.SZ -> 302132.SZ @ 2025-02-17`；raw code-set 对账按 PIT symbol interval 计算，禁止把新代码的原上市日期反推为历史预期代码。
`candidate_state`: 历史 candidate `candidate__5060cee5169a45e4162200ad` 属于旧合同，已 superseded；它只保留为 2010—2012 链路证据，不能发布。新的 trusted-source candidate 尚未构建；v2 迁移代码与残差调度已通过测试，但生产迁移、compact、candidate、audit 和 publish 尚未完成。
`local_build_performance`: 三年 candidate 本地流式构建约 42.6 分钟，明显慢于已优化的下载；这是后续独立的 canonical build 单核/向量化优化项，不应通过降低审计或发布门解决。
`release_boundary`: v3 没有 active manifest，未执行有效 publish 或 active 切换。首次 candidate 只要求九域：calendar、identity/symbol history、daily/status、adjust-factor daily、eligible signal、tradable open 与 5m。全历史核心域、identity/PIT、2010 起完整 5m 与 bundle/index/hash gate 未完成前不得发布。5m strict 覆盖低于 98% 阻断、98%—99% 警告、至少 99% 正常。首次发布后还必须完成一次 watermark 晚于 `2026-07-13` 的增量发布，才允许退休 v2；v2 active SHA-256 仍为 `e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e`。
`evidence`: `quant_data_platform/brain/references/qdp_v3_rebuild_20260713.md`

### object `qdp_v2_active_data_base`
`type`: active_manifest_first_data_base
`state`: `status=ok`
`root`: `quant_data_platform/data/qdp_v2`
`active_manifest`: `quant_data_platform/data/qdp_v2/active/active.json`
`scope`: `2011-11-22..2026-06-26`
`universe`: 沪深 A 股主板，剔除创业板、科创板、ST、退市股；当前名称含 `退市` 的股票也排除，即使 provider 的 `is_delisted` 标志未同步。
`symbol_start_overrides`: `600036.SH -> 2016-07-25`
`active_symbol_count`: `3037`
`active_table_count`: `17`
`scope_boundary`: 17 个 core symbol tables 仍是 2026-06-26 当前存续 3037 股的静态范围；历史不按当前存续筛选时必须使用独立 `pit_mainboard_non_st_v1` 研究范围，不能把 core `universe_snapshot` 误称为无幸存者偏差股票池。
`evidence`: `qdp status --json`、`qdp check --quick --json`、`qdp check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json`
`meta_quality_state`: 2026-07-01 的 PIT/metadata/factor/index 检查只证明结构、键覆盖和正值；2026-07-13 的语义审计证明旧因子可在无公司行动时逐日跳变，因此旧因子及其调整收益结论均降为 provisional。
`lineage_health`: 当前 17 个 active dataset 的 parquet/manifest 仍可读，但各引用一个已经不存在的 source ancestor，共 17 个缺失 dataset id。全盘搜索未恢复副本；用户明确授权后形成不伪称恢复的替代冻结证明，M0 状态为 `complete_with_authorized_lineage_gap`。39 个仍存在 dataset、23,738 shards、23,777 files 已逐文件 hash；v3 发布与发布后 5m/lineage/hash/diff/downstream 校验完成前禁止删除，之后仅按受保护 retirement 命令清理四条旧分钟链。

### object `active_tables_v2_legacy`
`type`: current_table_set
`raw_facts`: `market_daily_raw`、`market_intraday_1m`、`trading_calendar`、`universe_snapshot`、`security_status`、`valuation`、`adjust_factor`、`industry_concept`、`index_constituents`、`corporate_actions`、`share_capital`、`name_change`
`caches`: `market_intraday_5m`、`market_daily_panel`
`derived_features`: `intraday_daily_features`、`limit_intraday_features`
`derived_events`: `limit_status`
`candidate_research_scope_tables`: `pit_signal_universe`、`pit_signal_universe_daily`（已构建、未写入当前 `active.json`）
`boundary`: 这是当前 v2 active 的历史合同：1m/daily 为事实，5m 为 1m 派生缓存。它不得外推为 v3 语义；v3 以 5m 为唯一分钟事实且没有 1m。

### object `v2_nonactive_pit_research_evidence`
`type/state/datasets`: immutable_research_views / audited_not_active；`pit_signal_universe__7c8d1f2986a646a5cb8a93a6`、`pit_signal_universe_daily__7c8d1f2986a646a5cb8a93a6`；formal view `seq100_pit_2012_2025_formal__9d6feb2a5ba7f15a7635b42f`。
`boundary`: 它们证明 date-local universe 与五表原子 view 的结构，但不在 v2 `active.json`，也不能证明旧 dense factor 语义正确；下游必须显式选择。完整范围、行数和 gate 证据见 `references/pit_signal_universe_20260711.md` 与 `references/pit_market_substrate_20260711.md`。

### object `v2_quality_history`
`type/state`: active_structural_proof / semantically_superseded_by_v3；2026-07-01 scope rebuild 将 v2 active 收敛到 3037 个当前主板非 ST/非退市 symbol；share-capital/market-cap 关键字段已补齐，industry 无 blank/UNKNOWN。旧 factor 仍只证明键覆盖、正值和 provenance，`600076.SH/2024` 是语义反例。
`evidence`: `data/qdp_v2/audits/meta_domain_quality_20260701T140834+0000.json`；这些历史质量明细不再展开在 hot path。

### object `archived_v1_workflows`
`type`: archived_process_surface
`state`: old v1 data-lake workflows, downstream training-artifact workflows and one-off repair commands are not public QDP commands.
`notes`: `tools/archive_v1/README.md`、`tools/archive_repair/README.md`

### object `downstream_research_artifact_boundary`
`type`: ownership_boundary
`state`: downstream model-ready artifacts are no longer stored under `quant_data_platform/data/qdp_v2/research/`; that legacy research root was physically migrated or deleted on 2026-07-07.
`owner`: `daily_research`
`qdp_role`: provide source active tables, dataset manifests, quality proofs and stable data scope; do not own model loss, path value, normalization, sample index, memmap or training result semantics.
`artifact_root`: `daily_research/data/research_store/<artifact_id>/`

### object `provider_runtime`
`type`: upstream_ingest_runtime
`state`: production Python environment is `yolos`；历史分钟路由为 direct archive、v2 migrated raw、免费完整日、Tushare residual，截止日后由 BaoStock/mootdx 维护增量。raw payload 存大 Parquet bundle，运行 job/cursor/index 位于仓库内 `QDP_RUNTIME_ROOT`。
`boundary`: provider 不再互相证明数值正确，但 schema、单位、identity、PIT、48-bar 完整性、lineage、secret scan 和 CAS 仍必须通过；不完整股票日 quarantine，不拼接、不插值。

### object `qdp_storage_retention`
`state`: on `2026-07-10`, manifest-aware GC removed 26 unreferenced dataset directories and reclaimed `44,891,842,512` bytes; 17 active dataset directories remain.
`proof`: every removed dataset had a same-domain active replacement covering its date range; all active shard targets were materialized and non-symlink; pre/post quick check and post-delete `status --verify-files` passed.
`runtime`: metadata-only `qdp status` completes in about 0.6s; read-only `qdp check --quick --runtime fast` completes in about 2.5s on the current store.
`invariant`: future deletion still requires active-manifest traversal, dry-run, replacement evidence for unique-data boundaries, and post-delete file verification.
`m0_correction`: v3 freeze 审计发现 17 个 active input ancestor 已缺失；当前所有 39 个仍存在的 v2 dataset 及 17 个缺失祖先 id 均被 pin。用户授权的替代缺口证明只允许重建继续，不代表祖先已恢复；v3 发布、5m 覆盖/hash 复核、v2/v3 diff、下游切换和 retirement manifest 全部完成前禁止删除旧分钟 parquet。满足全部条件后只退休 v2 的 1m、旧 5m 与两条分钟特征链。
`raw_compaction`: v3 legacy raw 的小域演练、全域 bundle 写入和 `--delete-source --yes` 已完成；删除前后均验证 source content hash、catalog、bundle SHA、row-group、精确历史 revision、SQLite catalog 重建与 secret scan。H 的 1 MiB allocation unit 是此前高占用的直接原因；v2 删除仍只由发布后的 retirement gate 授权。
`retirement_sequence`: v3 首次发布只建立 `awaiting_post_cutoff_incremental` gate；完成一次 cutoff 后增量发布、watermark/coverage/hash/audit/diff/consumer 检查后才写 retirement manifest 并删除四域 parquet。任一失败保持 v2 数据不变。

## Pure Functions
- `active_table(domain)`: read `active.json.datasets[domain]` and then the referenced `dataset.json`.
- `describe_table(domain, full=false)`: summary by default; full manifest only with `--full --json`.
- `derive_next_qdp_action(state)`: if active coverage is current, prefer quality check, rebuild cache/feature only when raw facts change, and keep update dry-run separate from memmap work.
- `classify_cleanup(path)`: requires active manifest graph traversal and dry-run before deletion.

## Procedures
### procedure `inspect_qdp_status`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v2 status --json`
`output`: active scope and table summaries, without requiring a catalog.

### procedure `quick_quality_check`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v2 check --quick --json`；默认只读，只有显式 `--write-audit` 才写审计文件。
`output`: fast manifest/contract/path check.

### procedure `meta_quality_check`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --generation v2 check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json`
`output`: global exact primary-key and coverage/alignment proof for PIT/meta/factor/index domains.

### procedure `update_dry_run`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --dry-run --runtime fast --json`
`output`: coverage and gap plan; does not run providers.

## Evidence Entrypoints
- Current data-base document: `quant_data_platform/DATA_BASE.md`
- v1 archive notes: `tools/archive_v1/README.md`
- repair archive notes: `tools/archive_repair/README.md`
