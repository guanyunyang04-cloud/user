# Quant Data Platform 知识对象

快照日期：`2026-07-16`

## Object Classes

### class `mutable_current_table`

`definition`: 每个 domain 只有一套当前 Parquet；`dataset.json` 描述 schema、主键、范围、行数与 shards，`active.json` 负责查找。

`rule`: 数据可直接增删改，不需要 candidate、publish、rollback 或 lineage 图。

### class `stable_security_identity`

`tables`: `security_identity`、`symbol_history`。

`rule`: 代码变化证券共享稳定 security_id；日期查询恢复当日 symbol。当前研究范围内两组固定变更为 `000022→001872`、`000043→001914`。

### class `daily_market_fact`

`table`: `market_daily_raw`。

`rule`: 一证券一交易日一行；OHLC 有限且为正、high/low 合法、volume/amount 非负；停牌不制造零价行情。

### class `intraday_5m_fact`

`table`: `market_intraday_5m`，唯一分钟域。

`rule`: 一个被接纳的股票日必须恰好 48 个标准右闭合 bar；不完整日不拼接、不插值，研究连续窗口自动排除。

### class `adjust_factor`

`table`: 与 daily key 对齐的正因子。历史异常已经直接修正；执行价格始终使用 raw，因子只用于研究复权。

### class `security_status`

`rule`: `is_st` 与 `is_suspended` 独立；只有当前交易日无 daily 行或 daily.volume<=0 才视为整日停牌。有正成交日线时不得标记停牌。

### class `trusted_provider`

`rule`: 本地数据、mootdx、BaoStock 和已物化/定点取得的 Tushare 结果默认可信。跨源差异不触发逐行仲裁；只有明显结构错误或少数已知特例才修正。

`roles`: 本地数据负责早期主体；mootdx 负责近期速度；BaoStock 负责免费持续更新和完整日 fallback；Tushare 只补当前研究证券在前三者窗口外的显式历史缺口，不参与普通增量。

### class `storage`

`rule`: canonical 只在 H 盘；大表使用 Parquet 压缩是为了减少 I/O 和 H 盘占用，不是额外数据副本。Pandas/Arrow 校验只在写入边界处理小批数据，不对全库反复复制。

## Long-Term Lessons

- 对个人量化项目，研究可用性优先于发布工程；一套可变通用数据比多代 immutable pipeline 更合适。
- 数据源默认可信仍不等于转换永远正确；身份、时间、单位、主键和 48 根完整性必须在写入边界检查。
- 5m 可以直接作为事实表，不需要保留 1m 或由 1m 重建。
- 免费源之间应按覆盖能力互补，而不是互相证明价格正确。
- 更多 BaoStock 登录只有在隔离进程实测稳定时使用；当前 4 连接已通过本次全量补缺。
- 当前 QDP 是用户明确选择的 survivor store：当前 ST 仍视为研究证券，正式退市后才回溯删除；因此不宣称支持退市概率或无幸存者偏差研究。
- raw/provider staging、runtime 和旧 generation 完成合并后应立即删除，避免 H 盘 1 MiB allocation unit 放大小文件占用。

## Pure Functions

- `current_table(domain) -> manifest + shards`
- `valid_daily(row) -> bool`
- `valid_5m_day(rows) -> bool`
- `provider_order(recent) -> mootdx_then_baostock`
- `is_qdp_asset(path) -> canonical|runtime|research_output|redundant`
