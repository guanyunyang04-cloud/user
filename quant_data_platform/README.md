# Quant Data Platform

Canonical brain source: `quant_data_platform/brain/state_center.md`.

QDP 是个人量化研究使用的单一可变数据仓库。当前数据位于
`data/qdp_v2`；`v2` 只是历史目录名，不再表示 generation 或发布流程。

## 当前边界

- 每个 domain 只保留一套当前 Parquet 与一个 `dataset.json`；`active.json`
  只是表名到目录的轻量索引。
- 当前价格研究范围是沪深主板 A 股，并保留历史证券；不按当前名称做
  幸存者过滤。创业板、科创板和北交所不会由增量器误接入。
- 正式研究范围从 `2010-01-01` 开始，唯一分钟表是
  `market_intraday_5m`，每个可用股票日必须是完整 48 根。
- 已物理删除 1m、1m 派生特征、旧 generation、raw/candidate/publish
  流程和 Tushare 下载实现。
- 历史 5m 由本地购买数据构成主体；Tushare 的一次性补缺结果已经并入
  当前表，原始下载与运行状态不再保留。
- 后续更新以 BaoStock 为免费主源，默认使用 4 个独立连接；mootdx 先做
  近期快速下载，BaoStock 再补 mootdx 未获得的完整股票日。不拼接、不插值。
- 数据源默认可信，只保留主键、类型、OHLC 合法性、非负成交量和 48 根
  完整日等必要检查，不做跨源逐行价格仲裁。
- 所有数据、临时文件和运行状态都在 `H:\quant_project`；内存保护线是
  可用物理内存低于 0.5 GiB 持续 2 秒。

## 命令

```text
qdp status
qdp list
qdp describe <table>
qdp check --quick|--full
qdp update
qdp compact
qdp gc
# 仅在没有运行中的 QDP 任务时清理仓库内临时文件：
qdp gc --runtime --delete --yes
```

没有 public generation、candidate、publish、rollback 或 1m rebuild 命令。
训练包、memmap、模型与回测属于 `daily_research`，不属于 QDP 数据仓库。

项目事实与维护边界见 `quant_data_platform/brain/state_center.md`。
