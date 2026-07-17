# Quant Data Platform 身份对象

快照日期：`2026-07-16`

## object `quant_data_platform_project`

`type`: shared_mutable_research_data_store

`definition`: 主脑管辖下的通用量化数据仓库；保存研究项目反复使用的当前数据，支持直接添加、覆盖和删除。

`north_star`: 数据随时可用、结构简单、缺什么补什么、错什么改什么。

`truth_source`: `parquet + dataset.json + active.json`。`active.json` 是轻量表索引，不是发布系统。

`not`: 数据湖发布平台、candidate registry、下游训练产物仓库或实时交易系统。

`public_methods`: `status()`、`list()`、`describe(table)`、`check(mode)`、`update()`、`compact()`、`gc()`。

`consumers`: `daily_research`、`traditional_quant_research`、`t0_project` 和未来研究项目。

## object `qdp_current_store`

`root`: `quant_data_platform/data/qdp_v2`；目录名 `v2` 仅为兼容历史路径。

`scope`: 3,192 只当前上市沪深主板 A 股的可获得记录，正式研究起点 `2010-01-01`；当前 ST 保留，正式退市后回溯清除该证券历史。稳定身份与代码历史可早于正式研究起点。

`minute_domain`: 仅 `market_intraday_5m`，完整股票日为 48 根。

`invariant`: 每个 domain 只保留一个当前 dataset directory；不保留第二套 raw/candidate/generation。

## object `provider_surface`

`history`: 本地购买数据提供早期 5m 主体；已接受的 Tushare 定点补缺已经并入当前表。Tushare 仅作为当前研究证券超出免费源窗口的显式历史缺口 fallback。

`incremental`: mootdx 优先快速获取近期市场数据，BaoStock 用 4 个隔离连接补剩余完整日并承担免费持续更新。

`trust`: provider 数值默认正确；平台只防止主键、类型、身份、单位、OHLC 和缺 bar 等工程错误。

## object `research_artifact_boundary`

`owner`: 训练包、memmap、标签、模型、回测和执行产物由 `daily_research` 或相应研究项目管理，不写入 QDP active。

## Routing

- 当前事实：`state_center.md`
- 稳定语义：`knowledge_center.md`
- 命令与维护：`operations_center.md`
- 最小数据不变量：`governance_layer.md`
- 历史过程：`references/`
