# Quant Data Platform 状态程序

快照日期：`2026-07-16`

## Module Interface

`exports`: 单一可变 QDP 数据仓库、当前表摘要、轻量质量检查和近期更新。

`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances

### object `qdp_current_store`

`type`: mutable_manifest_indexed_parquet_store

`root`: `quant_data_platform/data/qdp_v2`；`v2` 只保留为历史目录名，不再表示 generation。

`state`: `status=usable`；正式范围 `2010-01-01..2026-07-16`；价格域为 3,192 只当前上市沪深主板 A 股。当前 ST 保留，证券正式退市后回溯清除其各域历史。这是用户明确选择的 survivor store，不用于研究退市概率或无幸存者偏差的历史全市场。每个 domain 只有一个当前 dataset directory；`active.json` 只是表名索引，不承担 candidate/publish 语义。

`domains`: 当前 14 表为 calendar、identity/symbol history、daily、status、universe、factor、5m，以及 retained corporate/name/industry/index/share-capital/valuation 通用低频表。

`core_rows`: `trading_calendar=6,041`、`market_daily_raw=9,644,664`、`universe_snapshot=9,927,671`、`security_status=9,927,671`、`adjust_factor=9,644,664`、`security_identity=3,192`、`symbol_history=4,677`、`market_intraday_5m=462,928,800`。

`five_minute`: 唯一分钟表；共 `9,644,350` 个完整股票日，按自然年压缩为 17 shards、约 5.09 GiB。新写入只接受合法 OHLCV 和精确 48 根股票日，不跨源拼接、不插值。完整 5m 日缺 daily 为 0；9,644,351 个正成交 daily 中只缺 `600568.SH/2011-11-21` 一个 5m 日，覆盖率 `99.99998963123594%`。本地 1m/5m、BaoStock、mootdx、原 Tushare 与 2026-07-17 当前代理复核均无数据。最新日 `2026-07-16` 为 `3,191/3,191`。

`identity`: security_id 主键重复 0、symbol_history 主键重复 0、孤儿 ID 0、有效区间重叠 0；当前研究证券中的代码变更 `000022/001872`、`000043/001914` 均归并到稳定身份。

`factor`: daily/factor 键差 0、非正或空值 0、短间隔污染 0、年度过度变化证券数 0；`600076.SH` 2024 年因子变化数为 0。16 个跨 53—990 天停牌/重组后的 raw 价格跳变只记为连续训练窗口警告，不判为因子污染。

`status`: 旧表曾把部分 ST 日误标为停牌，并漏标 27 只长期停牌股。已原位修正 15,528 行：10,338 行改为停牌、5,190 行改为可交易。现定义为“无 daily 或 daily.volume<=0 才是整日停牌”，`is_st` 独立；9,927,671 行全历史语义矛盾 0。

### object `provider_policy`

`history`: 本地购买的 mootdx 系 5m 是早期主体；已接受的 Tushare 定点补缺直接并入当前表，退市证券及其历史现已按用户范围删除。Tushare 只在本地与免费源都无覆盖时作为当前研究证券的显式历史缺口 fallback，凭据只从进程环境读取，不保留第二套 raw。

`incremental`: BaoStock 是免费低频主源和完整日 fallback，mootdx 是近期 5m 高速下载器。2026-07-14..16 增量由 4 worker 的 mootdx 一次补齐 9,572 个股票日、459,456 行，BaoStock fallback 任务为 0、provider error 为 0；低频 daily/factor/status 同步到 2026-07-16。

`trust_boundary`: provider 数值默认可信；只检查 schema、主键、身份、OHLC 合法性、非负成交量/成交额和 48 根完整日。不同来源不做逐行价格仲裁。

`memory`: DuckDB 与下载器按系统剩余内存动态使用，不设 2 GiB 固定上限；可用物理内存低于 0.5 GiB 持续 2 秒即中断并从断点恢复。

`last_full_audit`: `data/qdp_v2/audits/database_audit_20260717T092836+0000.json`；`status=ok`、阻断错误 0、警告 2（唯一历史 5m 缺口与 16 个长停牌 raw 跳变）。

### object `storage_state`

`layout`: `data/qdp_v2/active/active.json` + `data/qdp_v2/datasets/<domain>/<current_id>/dataset.json|shards/`。

`cleanup`: 已删除 1m、两条分钟特征链、daily panel、limit status、18 个旧 generation、全部 qdp_v3 data 和过时 raw/bootstrap；5m 为 17 个年度文件、约 5.09 GiB。`qdp_runtime` 是可随任务产生的仓库内临时目录，确认没有活动任务后可用 `qdp gc --runtime --delete --yes` 清理。研究产物 `agent_runs` 与 `event_packs` 不属于 canonical，未作为重复数据删除。

`disk`: H 盘已通过 chkdsk/dirty-bit/health 检查；数据和 runtime 均只使用本仓库，不使用 C/F 盘。

### object `public_surface`

`commands`: `status`、`list`、`describe`、`check`、`update`、`compact`、`gc`。

`retired`: public generation、v3、candidate、publish、rollback、Tushare 全量 bootstrap、1m rebuild 和一次性 archive planners。Tushare 仅保留非公开、显式触发的定点历史缺口修复能力。

`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`

## Pure Functions

- `current_table(domain)`: 从 `active.json.datasets[domain]` 定位唯一当前 `dataset.json`。
- `accept_daily_row(row)`: 主键唯一、OHLC 合法、volume/amount 非负。
- `accept_5m_day(rows)`: `accept_daily_row` 且精确 48 个标准 bar_time。
- `choose_provider(period)`: 近期先 mootdx 加速，剩余完整日由 BaoStock；只有当前研究证券的历史缺口同时超出本地、mootdx 与 BaoStock 能力时，才显式定点调用 Tushare。

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
