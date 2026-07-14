# Quant Data Platform 状态程序
快照日期：`2026-07-14`

## Module Interface
`exports`: QDP v2 active data base status、QDP v3 rebuild/candidate status、dataset manifest summaries、quality/audit entrypoints。
`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances
### object `qdp_project`
`type`: shared_manifest_first_data_base_instance
`state`: v2 仍是唯一 active 数据基底；v3 已收敛为 5m-only，provider、raw、identity、build、audit、CAS release、retirement 与 GC 代码面已建立。BaoStock 2010—2012 日线/日期因子事件 raw 已 strict 完成，Tushare 代理历史 bootstrap 正在可恢复运行；因子仲裁、全历史 5m、candidate 和 active 切换尚未完成。
`public_cli`: 共用 `status/list/describe/check/gc/update`；v3 新增 `ingest/build candidate/audit/diff/publish/rollback/compatibility`；首次 bootstrap 发布后会重跑 semantic/hash 审计并在全部前置通过时自动执行受保护的 `retire-v2-intraday`，也可单独 dry-run/执行该命令。v2 legacy rebuild/verify 在退休前仍通过 `--generation v2` 只读保留。
`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`

### object `qdp_v3_rebuild`
`type/state`: immutable_manifest_v3_candidate_pipeline / historical_backfill_in_progress_with_authorized_m0_gap
`root`: `quant_data_platform/data/qdp_v3`
`contract`: `qdp_v3_20260714_bootstrap_5m` / schema `3.2.0` / manifest `3`；正式起点 `2010-01-01`；日期分区 raw 不可变修订链；canonical 主键使用稳定 `security_id + 时间键`；历史代码由有证据的 `symbol_history` 恢复；quality tier 为 `strict/provisional/quarantined`。v3 不采集、构建、发布或更新 1m，`market_intraday_5m` 是唯一分钟事实域。
`providers`: BaoStock `0.9.3` wheel SHA-256 `acbd19403285bc4e254cee8297cf0e2646ae2276e5af7e549deed3988ab02293`；mootdx `0.11.7`；Tushare-compatible proxy 仅作 2010-01-01..2026-07-13 历史启动源，`upstream_provenance=not_exposed`，Token 只从环境变量读取。
`implemented_domains`: 全 A 股批量日线/状态/估值、批量因子事件、逐证券因子史、security identity/symbol history、PIT signal/open、5m 唯一分钟事实域、xdxr/股本、财务/业绩/行业/指数次级域；v3 1m 路由与转换代码已删除。
`compatibility`: 0.9.1 full golden 已捕获；0.9.3 full gate 为 `passed`（34 anchors、0 issues、普通多页 5537 rows、wheel hash 匹配）。报告不是 `passed/full` 时仍禁止 live date-partition ingest。
`historical_backfill`: 2010/2011/2012 分别有 `242/244/243` 个交易日；日线、`query_all_stock` 和日期因子事件均为 `729/729` strict 分区。日线共 `1,578,095` rows；因子事件共 `5,192` rows、135 个合法零事件日；三年均无缺日、结构 blocker 或 20,000 行截断。
`proxy_bootstrap`: Tushare `stock_basic`、`trade_cal`、4,011 个全市场 `daily` 与 4,011 个 `daily_basic` 日期任务已完成；5,864 个全历史 L/D/P identity 任务正在原 job 推进，之后按 status/factor/dividend/financial 顺序执行。历史 5m 固定反向 8,000 行分页、3 workers/最多 3 in-flight；live 429 证据将默认 limiter 修正为 96 次/分钟、burst 1，并保留共享冷却和最低 90 的自适应降速。重启后的前 1,053 个真实请求为 `1053/1053` 成功、0 网络/协议/限流错误，已通过低于 0.5% 的生产闸门。低于 200 GiB 自动暂停；全历史预计约 65,613 个分钟请求，额度耗尽必须进入 `paused_quota` 而非空数据。
`download_runtime`: BaoStock 2013+ 日线/因子日期回灌在独立线程与 Tushare 参考域并行，但 BaoStock 内部仍复用单一 login 且网络在途数为 1；Tushare 历史 5m 每证券串行翻页并在完成后合并为一个 immutable raw 分区。mootdx 候选由 last-good、tdxpy 与 mootdx 内置列表合并；最新真实闸门冷启动 `1.43s`、热启动 `0.07s`，三只固定证券均为 48 bars 且止于 15:00，保存最快 3 个健康节点并用 300 秒负缓存快速 fallback。
`rejected_parallel_route`: 两个独立 BaoStock login 的 live probe 出现 `10001001 用户未登录`，因此禁止生产双 session；历史低错误率不得自动解锁第二个网络槽。每个 ingest job 有 OS advisory single-writer lock，进程异常退出后自动释放。
`adapter_smoke_boundary`: ETF adapter `1574` rows provisional；mootdx 的 `600000.SH/000001.SZ/600076.SH` 最近完整日均为 48 bars 且止于 15:00。Tushare 5m 的 `vol/amount` 已用沪深、早期/近期固定样本证明为股/元，canonical scale 均为 1.0。
`pit_name_boundary`: stock-basic 当前名称禁止反填历史；`name_on_date` 只能来自日期级 all-stock snapshot 或带 hash 的官方证据，semantic audit 回查 raw。
`identity_regression`: 官方代码历史固定为 `000022.SZ -> 001872.SZ @ 2018-12-26`、`000043.SZ -> 001914.SZ @ 2019-12-16`、`300114.SZ -> 302132.SZ @ 2025-02-17`；raw code-set 对账按 PIT symbol interval 计算，禁止把新代码的原上市日期反推为历史预期代码。
`candidate_state`: 历史 candidate `candidate__5060cee5169a45e4162200ad` 仅证明 2010—2012 的日线/identity/PIT/manifest 链路；它缺少新的全历史 5m-only 合同且 factor event/daily 仍被 `factor_event_not_verified_or_arbitrated` 与 `factor_reference_price_proof_missing` 阻断，不能发布。
`local_build_performance`: 三年 candidate 本地流式构建约 42.6 分钟，明显慢于已优化的下载；这是后续独立的 canonical build 单核/向量化优化项，不应通过降低审计或发布门解决。
`release_boundary`: v3 没有 active manifest；未执行 publish 或 active 切换。全历史参考域、身份、逐证券因子史/xdxr/官方仲裁、2010 baseline 与 2010 起 5m strict gate 未完成前不得发布；5m watermark 必须与日线完全相等。v2 active SHA-256 仍为 `e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e`。
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

### object `pit_mainboard_non_st_v1`
`type/state`: date_local_research_universe / candidate_ready_not_active
`datasets`: eligibility `pit_signal_universe__7c8d1f2986a646a5cb8a93a6`；daily audit `pit_signal_universe_daily__7c8d1f2986a646a5cb8a93a6`
`scope`: 逐交易日已上市沪深主板普通 A 股；研究池剔除当日 ST/已退市，信号池再剔除当日停牌与无 bar。
`coverage/evidence`: `2016-01-04..2026-06-01`、2526 日、3393 个历史证券、7,451,610 个唯一证券日；375 个历史曾合格证券不在源末日合格成员中。
`pit_boundary`: source 历史名称存在回填行为，名称和未来 `out_date` 均不输出为模型输入；逐日 eligibility 只使用当日上市/板块/ST/停牌/bar 事实。
`channel_boundary`: core OHLCV/分钟/limit/辅助表仍为当前 3037 股范围；该对象修复股票池定义，但尚未单独恢复额外历史证券的全通道数据。
`activation_boundary`: 当前 `active.json` 保持原 17 表；在额外历史证券的价格/特征通道与下游 pack 通过质量门前，不激活这两个候选 pointer。
`evidence`: `quant_data_platform/brain/references/pit_signal_universe_20260711.md`

### object `pit_market_substrate_v2`
`type/state`: immutable_research_dataset_view / formal_2012_2025_research_window_ready_not_active
`atomic_overrides`: `market_daily_raw`、`security_status`、`limit_status`、`adjust_factor`、`pit_signal_universe`；seq100 `--dataset-view` 拒绝任何不完整覆盖，避免 PIT universe 与 active survivor-scope 价格/因子混用。
`inputs`: 可重复 traditional PIT snapshot；也可直接读取 recovery cache 的 `daily_stock_lists + security_master` 并拼接独立 QDP market-daily backfill manifest；factor events 可重复输入，active dense factor 只作为只读来源。
`gates`: 五表主键唯一；`eligible_for_research && !is_suspended` 对 market 做独立 anti-join，且校验预期 market 起止日期；security-status 与 PIT row-count 对齐；每个 market key 必须有正 `back_adjust_factor` 和 provenance；任何缺口阻止 view 生成。
`limit_semantics`: `prior_valid_adjusted_close / current_back_adjust_factor` 转回当日 raw reference，再按当日 ST 5%/普通 10% 和 0.01 tick half-up；避免除权日沿用上一 raw close。
`probe`: non-active view `seq100_pit_2016_2026_probe__58498361379f14f30e77cf90`；3393 symbols、7,440,688 market/factor/limit keys、7,451,610 PIT/status keys、7,031,085 signal-eligible rows、Gate-0 missing=0、factor missing/nonpositive=0、PK duplicate=0；`active.json` byte-for-byte unchanged。
`formal_view`: `seq100_pit_2012_2025_formal__9d6feb2a5ba7f15a7635b42f`；底层覆盖 `2012-01-04..2026-06-01` 以支持 2025 年末 forward-60 标签；3409 symbols、3496 dates、9,530,656 market/factor/limit keys、9,542,426 PIT/status keys、8,805,538 signal-eligible rows；所有增强 Gate-0/factor/PK checks 为 0 findings。
`activation_boundary`: formal view 与五个 dataset 均保持 non-active；`active.json` SHA-256 在构建前后均为 `E56F72A6CBA8BCF86055817F6A0EC5E7391271FB3C27B4D628C3ABC62944051E`。
`evidence`: `quant_data_platform/brain/references/pit_market_substrate_20260711.md`

### object `meta_domain_quality_proof`
`type`: historical_structural_quality_evidence
`state`: structurally_passed_semantically_superseded
`domains`: `trading_calendar`、`universe_snapshot`、`security_status`、`adjust_factor`、`industry_concept`、`index_constituents`
`audit_path`: `quant_data_platform/data/qdp_v2/audits/meta_domain_quality_20260701T140834+0000.json`
`adjust_factor`: active dataset `adjust_factor__4e0e31d3c1fd34bfcfa7a7dd`; one row per `market_daily_raw` key、正值与 provenance 仍成立，但不再构成语义正确证明；`600076.SH/2024` 无事件日异常跳变是固定反例。
`industry_concept`: one row per universe key; blank/UNKNOWN industry rows are `0`; `001399.SZ` was filled from AkShare/CNInfo profile industry; concept tags are not part of the active short-line data base because no reliable PIT concept-tag source is active.
`security_status`: historical ST rows remain PIT flags; active scope has `delisted_rows=0`.

### object `scope_active_rebuild_20260701`
`type`: active_scope_correction
`state`: completed
`scope_name`: `mainboard_hs_a_ex_current_st_name_delisted_v2`
`excluded_current_name_contains`: `退市`
`excluded_symbols`: `600193.SH`、`600608.SH`、`600636.SH`、`600696.SH`、`605081.SH`
`active_symbol_count_after`: `3037`
`effect`: all active symbol-bearing datasets were rebuilt under the same scope; `trading_calendar` unchanged.

### object `share_valuation_completion_20260701`
`type`: active_quality_completion
`state`: completed
`share_capital`: active dataset `share_capital__2ef0e66db12236b868b8c626`; `total_share_null_rows=0`、`float_share_null_rows=0`、`restricted_share_null_rows=0`.
`valuation`: active dataset `valuation__1e4c0d84334ff43e9f4e8fba`; `total_mv_null_rows=0`、`circ_mv_null_rows=0`; `pb_null_rows=229` and `turnover_rate_null_rows=224390` remain source/metric availability boundaries, not key/coverage failures.

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
`state`: production Python environment is `yolos`；Tushare proxy 只负责固定截止日历史启动，BaoStock 长期维护日线/状态/因子事件并作 5m fallback，mootdx 负责截止日后的优先 5m。BaoStock 长任务使用任务内持久单 login；mootdx 先做真实 5m 协议健康检查并负缓存不可用状态。
`boundary`: BaoStock 与 Tushare 兼容闸门、mootdx 5m 健康证明、provider staging、identity、factor、PIT、5m、lineage 与自动密钥扫描均通过 semantic audit 后才允许 CAS active pointer change；发布后再次执行 semantic/full-shard hash 复核，任意完整来源冲突不得用优先级强行覆盖。

### object `qdp_storage_retention`
`state`: on `2026-07-10`, manifest-aware GC removed 26 unreferenced dataset directories and reclaimed `44,891,842,512` bytes; 17 active dataset directories remain.
`proof`: every removed dataset had a same-domain active replacement covering its date range; all active shard targets were materialized and non-symlink; pre/post quick check and post-delete `status --verify-files` passed.
`runtime`: metadata-only `qdp status` completes in about 0.6s; read-only `qdp check --quick --runtime fast` completes in about 2.5s on the current store.
`invariant`: future deletion still requires active-manifest traversal, dry-run, replacement evidence for unique-data boundaries, and post-delete file verification.
`m0_correction`: v3 freeze 审计发现 17 个 active input ancestor 已缺失；当前所有 39 个仍存在的 v2 dataset 及 17 个缺失祖先 id 均被 pin。用户授权的替代缺口证明只允许重建继续，不代表祖先已恢复；v3 发布、5m 覆盖/hash 复核、v2/v3 diff、下游切换和 retirement manifest 全部完成前禁止删除旧分钟 parquet。满足全部条件后只退休 v2 的 1m、旧 5m 与两条分钟特征链。

## Pure Functions
- `active_table(domain)`: read `active.json.datasets[domain]` and then the referenced `dataset.json`.
- `describe_table(domain, full=false)`: summary by default; full manifest only with `--full --json`.
- `derive_next_qdp_action(state)`: if active coverage is current, prefer quality check, rebuild cache/feature only when raw facts change, and keep update dry-run separate from memmap work.
- `classify_cleanup(path)`: requires active manifest graph traversal and dry-run before deletion.

## Procedures
### procedure `inspect_qdp_status`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json`
`output`: active scope and table summaries, without requiring a catalog.

### procedure `quick_quality_check`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json`；默认只读，只有显式 `--write-audit` 才写审计文件。
`output`: fast manifest/contract/path check.

### procedure `meta_quality_check`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check meta --runtime fast --duckdb-memory-limit 12GB --threads 4 --writeback --json`
`output`: global exact primary-key and coverage/alignment proof for PIT/meta/factor/index domains.

### procedure `update_dry_run`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli update --as-of-date <date> --dry-run --runtime fast --json`
`output`: coverage and gap plan; does not run providers.

## Evidence Entrypoints
- Current data-base document: `quant_data_platform/DATA_BASE.md`
- v1 archive notes: `tools/archive_v1/README.md`
- repair archive notes: `tools/archive_repair/README.md`
