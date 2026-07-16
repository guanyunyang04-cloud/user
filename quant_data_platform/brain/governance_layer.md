# Quant Data Platform 最小治理对象

快照日期：`2026-07-16`

## Governed Objects

### object `current_table_index`

`scope`: `data/qdp_v2/active/active.json`。

`invariant`: 每个 domain 指向一个存在的 `dataset.json`；每个被引用 shard 必须存在。dataset_id/hash 不具有发布审批含义。

### object `unique_data`

`invariant`: 删除旧目录前确认当前表已包含需保留的唯一数据；确认后可直接物理删除，不要求备份或 candidate。

### object `market_row`

`invariant`: 主键唯一、日期合法、OHLC 合法、成交量/额非负；5m 股票日精确 48 根。

### object `stable_identity`

`invariant`: security_id 唯一；symbol_history 无孤儿、无区间重叠；代码变化按有效日期解析。

### object `provider_input`

`invariant`: provider 数值默认可信；不跨源拼股票日，不插值。不完整或明显非法数据不写当前表。

### object `resource_guard`

`invariant`: 所有写入在 H 仓库内；可用内存低于 0.5 GiB 持续 5 秒才停止；Token 或 secret 不写文件/日志。

### object `research_boundary`

`invariant`: QDP GC 不删除 `daily_research`、`agent_runs` 或 `event_packs` 中仍有研究价值的产物；这些也不属于 QDP canonical。

## Procedures

### procedure `mutate_current_table`

`steps`: 读取当前 manifest；验证目标增量/替换；原子安装 Parquet；更新 manifest；删除被替代文件；复核引用。

### procedure `cleanup_redundant_data`

`steps`: 保留 active 指向目录；删除其他 generation、1m/派生链、raw、runtime 和一次性过程文件；运行 quick check。

## Writeback

稳定状态写 `state_center.md`；长历史只保留在 `references/`，不得重新进入当前命令或数据路径。
