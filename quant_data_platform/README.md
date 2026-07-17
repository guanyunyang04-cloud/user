# Quant Data Platform

Canonical brain source: `quant_data_platform/brain/state_center.md`.

QDP 是个人量化研究使用的单一可变数据仓库。当前数据位于
`data/qdp_v2`；`v2` 只是历史目录名，不再表示 generation 或发布流程。

## 当前边界

- 每个 domain 只保留一套当前 Parquet 与一个 `dataset.json`；`active.json`
  只是表名到目录的轻量索引。
- 当前价格研究范围是 3,192 只当前上市沪深主板 A 股。当前 ST 保留；证券
  正式退市后，从 QDP 回溯删除该证券各域历史。这是用户明确选择的可变
  survivor store，不宣称适合研究退市概率或构造无幸存者偏差的历史全市场。
- 正式研究范围从 `2010-01-01` 开始，唯一分钟表是
  `market_intraday_5m`，每个可用股票日必须是完整 48 根。
- 截至 `2026-07-16`，日线与因子各 9,644,664 行，5m 为
  462,928,800 行（9,644,350 个完整股票日）。
- 历史 5m 由本地购买的 mootdx 系数据构成主体；已接受的 Tushare 定点补缺
  已直接并入当前表，不保留第二套 raw。Tushare 只在本地与免费源均无覆盖
  时用于当前研究证券的定点历史补缺，不是常规更新源。
- 后续更新由 BaoStock 负责日线、状态和因子；mootdx 先做近期 5m 快速下载，
  BaoStock 再补 mootdx 未获得的完整股票日，默认 4 个隔离连接。不跨来源
  拼接一个股票日，也不插值。
- 数据源数值默认可信，只保留主键、类型、身份、OHLC、非负量额、因子语义
  和 48 根完整日等转换边界检查，不做跨源逐行价格仲裁。
- 所有数据、临时文件和运行状态都在 `H:\quant_project`；内存保护线是
  可用物理内存低于 0.5 GiB 持续 2 秒。

## 当前质量结论

- `2026-07-17` full audit 为 `ok`，阻断错误 0；最新交易日 3,191 个应交易
  股票日全部有完整 5m。
- 全历史只有 `600568.SH / 2011-11-21` 一个正成交日缺 5m，覆盖率
  `99.9999896%`。本地 1m/5m、BaoStock、mootdx、原 Tushare 与当前代理
  均无完整分钟响应，因此该日保持 daily-only，完整窗口查询会自然排除。
- 因子与日线键完全一致，非正因子、短间隔污染和年度过度变化均为 0；
  `600076.SH` 的 2024 年回归检查不再出现逐日跳变。
- `security_status` 已把 ST 与停牌彻底分开：15,528 个旧错标已修正；全历史
  9,927,671 个状态日中，停牌标志与日线存在性/正成交量的语义矛盾为 0。
- 另有 16 个跨 53—990 天停牌/重组后的 raw 价格跳变，仅作为连续训练窗口
  警告，不是复权因子污染。

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
