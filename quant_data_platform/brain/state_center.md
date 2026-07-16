# Quant Data Platform 状态程序

快照日期：`2026-07-16`

## Module Interface

`exports`: 单一可变 QDP 数据仓库、当前表摘要、轻量质量检查和近期更新。

`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances

### object `qdp_current_store`

`type`: mutable_manifest_indexed_parquet_store

`root`: `quant_data_platform/data/qdp_v2`；`v2` 只保留为历史目录名，不再表示 generation。

`state`: `status=usable`；正式范围 `2010-01-01..2026-07-16`；价格域为沪深主板 A 股并保留历史证券，不做当前名称幸存者过滤；每个 domain 只有一个当前 dataset directory。`active.json` 只是表名索引，不承担 candidate/publish 语义。

`domains`: 当前 14 表为 calendar、identity/symbol history、daily、status、universe、factor、5m，以及 retained corporate/name/industry/index/share-capital/valuation 通用低频表。

`core_rows`: `trading_calendar=6,041`、`market_daily_raw=10,203,195`、`universe_snapshot=10,551,539`、`security_status=10,551,539`、`adjust_factor=10,203,195`、`security_identity=5,534`、`symbol_history=7,232`、`market_intraday_5m=472,590,576`。

`five_minute`: 唯一分钟表；共 `9,845,637` 个完整股票日，按自然年压缩为 17 shards、约 5.21 GiB。新写入只接受合法 OHLCV 和精确 48 根股票日，不跨源拼接、不插值。当前完整 5m 日缺 daily 为 0；daily 缺完整 5m 为 357,558，其中 357,550 位于 2010—2019 早期来源边界，2020 年以后只剩 8 个已证实上游均无完整日的例外。

`identity`: security_id 主键重复 0、symbol_history 主键重复 0、孤儿 ID 0、有效区间重叠 0；`000022/001872`、`000043/001914`、`300114/302132` 均归并到稳定身份。

`factor`: daily/factor 键差 0、非正或空值 0；`600076.SH` 2024 年 242 个交易日因子只有一个合法值，不再出现旧库逐日跳变污染。

### object `provider_policy`

`history`: 本地购买的 mootdx 系 5m 是早期主体；Tushare 只做一次性早期/退市补缺。其有效结果已经并入当前表，Tushare raw、runtime、provider 路由和 Token 配置均已退休。

`incremental`: BaoStock 是免费低频主源和完整日 fallback，mootdx 是近期 5m 高速下载器。2026-07-14..16 增量由 4 worker 的 mootdx 一次补齐 9,572 个股票日、459,456 行，BaoStock fallback 任务为 0、provider error 为 0；低频 daily/factor/status 同步到 2026-07-16。

`trust_boundary`: provider 数值默认可信；只检查 schema、主键、身份、OHLC 合法性、非负成交量/成交额和 48 根完整日。不同来源不做逐行价格仲裁。

`memory`: DuckDB 与下载器按系统剩余内存动态使用，不设 2 GiB 固定上限；可用物理内存低于 0.5 GiB 持续 2 秒即中断并从断点恢复。

### object `storage_state`

`layout`: `data/qdp_v2/active/active.json` + `data/qdp_v2/datasets/<domain>/<current_id>/dataset.json|shards/`。

`cleanup`: 已删除 1m、两条分钟特征链、daily panel、limit status、18 个旧 generation、全部 qdp_v3 data、Tushare/raw/provider 路由；5m 又从 8,269 个旧 shards 压缩为 17 个年度文件，逻辑体积约 6.77 GiB→5.21 GiB。`qdp_runtime` 是可随任务产生的仓库内临时目录，确认没有活动任务后可用 `qdp gc --runtime --delete --yes` 清理。研究产物 `agent_runs` 与 `event_packs` 不属于 canonical，未作为重复数据删除。

`disk`: H 盘已通过 chkdsk/dirty-bit/health 检查；数据和 runtime 均只使用本仓库，不使用 C/F 盘。

### object `public_surface`

`commands`: `status`、`list`、`describe`、`check`、`update`、`compact`、`gc`。

`retired`: public generation、v3、candidate、publish、rollback、Tushare bootstrap、1m rebuild 和一次性 archive planners。

`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`

## Pure Functions

- `current_table(domain)`: 从 `active.json.datasets[domain]` 定位唯一当前 `dataset.json`。
- `accept_daily_row(row)`: 主键唯一、OHLC 合法、volume/amount 非负。
- `accept_5m_day(rows)`: `accept_daily_row` 且精确 48 个标准 bar_time。
- `choose_provider(period)`: 近期先 mootdx 加速，剩余完整日由 BaoStock；历史不再主动调用 Tushare。

## Procedures

### procedure `inspect`

`commands`: `qdp status --json`；`qdp list`；`qdp describe <domain>`。

### procedure `check`

`command`: `qdp check --quick --json`；需要时才运行 full，不为普通更新做跨源验证。

### procedure `update_recent_5m`

`command`: `python -m quant_data_platform.cli update --as-of-date <date> --workers 4 --workspace-root H:\quant_project`

`effect`: mootdx 快速补入后，4 个 BaoStock 连接只补当前表仍缺的完整股票日；临时 bundle 提交后可删除 runtime。

### procedure `compact_5m`

`command`: `python -m quant_data_platform.cli compact --workspace-root H:\quant_project`

`effect`: 把唯一当前 5m 表流式重写为每自然年一个 Parquet；全行指纹一致后原位切换 manifest，再删除旧 shards，不创建第二个 generation。

## Ownership Boundary

QDP 只拥有通用数据表。训练包、memmap、标签、模型、回测和执行产物属于 `daily_research` 或对应研究项目。
