# Quant Data Platform 状态程序

快照日期：`2026-07-16`

## Module Interface

`exports`: 单一可变 QDP 数据仓库、当前表摘要、轻量质量检查和近期更新。

`consumers`: `daily_research`、`traditional_quant_research`、`t0_project`。

## Object Instances

### object `qdp_current_store`

`type`: mutable_manifest_indexed_parquet_store

`root`: `quant_data_platform/data/qdp_v2`；`v2` 只保留为历史目录名，不再表示 generation。

`state`: `status=usable`；正式范围 `2010-01-01..2026-07-13`；每个 domain 只有一个当前 dataset directory。`active.json` 只是表名索引，不承担 candidate/publish/CAS 语义。

`domains`: 当前 14 表为 calendar、identity/symbol history、daily、status、universe、factor、5m，以及 retained corporate/name/industry/index/share-capital/valuation 通用低频表。

`core_rows`: `trading_calendar=6,038`、`market_daily_raw=10,193,252`、`universe_snapshot=10,541,959`、`security_status=10,541,959`、`adjust_factor=10,190,067`、`security_identity=5,534`、`symbol_history=7,232`、`market_intraday_5m=472,107,360`。

`five_minute`: 唯一分钟表；当前 8,237 shards。新写入只接受合法 OHLCV 和精确 48 根股票日，不跨源拼接、不插值。2020—2026 的 109,886 个待补日中已补 109,878 个；剩余 8 个上游均无完整日，研究完整窗口自然排除，比例约 `0.0073%`。

`identity`: security_id 主键重复 0、symbol_history 主键重复 0、孤儿 ID 0、有效区间重叠 0；`000022/001872`、`000043/001914`、`300114/302132` 均归并到稳定身份。

`factor`: daily/factor 键差 0、非正或空值 0；`600076.SH` 2024 年 242 个交易日因子只有一个合法值，不再出现旧库逐日跳变污染。

### object `provider_policy`

`history`: 本地购买的 mootdx 系 5m 是早期主体；Tushare 只做一次性早期/退市补缺。其有效结果已经并入当前表，Tushare raw、runtime、provider 路由和 Token 配置均已退休。

`incremental`: mootdx 是近期高速下载器；BaoStock 是免费主源和完整日 fallback。BaoStock 已实测 4 个独立进程/登录稳定完成 109,877 个股票日、provider error 0，最低可用内存约 6.72 GiB。

`trust_boundary`: provider 数值默认可信；只检查 schema、主键、身份、OHLC 合法性、非负成交量/成交额和 48 根完整日。不同来源不做逐行价格仲裁。

`memory`: DuckDB 与下载器按系统剩余内存动态使用，不设 2 GiB 固定上限；仅当可用物理内存低于 0.5 GiB 持续 5 秒才中断并从断点恢复。

### object `storage_state`

`layout`: `data/qdp_v2/active/active.json` + `data/qdp_v2/datasets/<domain>/<current_id>/dataset.json|shards/`。

`cleanup`: 已删除 1m、两条分钟特征链、daily panel、limit status、18 个旧 generation、全部 qdp_v3 data、Tushare/raw/runtime/repair/audit/process 目录；累计实际释放约 52.2 GiB。研究产物 `agent_runs` 与 `event_packs` 不属于 canonical，未作为重复数据删除。

`disk`: H 盘已通过 chkdsk/dirty-bit/health 检查；数据和 runtime 均只使用本仓库，不使用 C/F 盘。

### object `public_surface`

`commands`: `status`、`list`、`describe`、`check`、`update`、`gc`。

`retired`: public generation、v3、candidate、publish、rollback、compact、Tushare bootstrap、1m rebuild 和一次性 archive planners。

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

`command`: `python -m quant_data_platform.qdp_v2.recent_market_repair all --start-date <date> --end-date <date> --workers 4 --workspace-root H:\quant_project`

`effect`: mootdx 快速补入后，4 个 BaoStock 连接只补当前表仍缺的完整股票日；临时 bundle 提交后可删除 runtime。

## Ownership Boundary

QDP 只拥有通用数据表。训练包、memmap、标签、模型、回测和执行产物属于 `daily_research` 或对应研究项目。
