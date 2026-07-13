# QDP v3 数据基底重建实施与审计记录

日期：`2026-07-13..2026-07-14`

## 结论

QDP v3 已从方案进入可执行代码：BaoStock 0.9.3 日期批量协议、不可变 raw、稳定证券身份、PIT 代码历史、因子事件双路径、双源 5m、次级 PIT、manifest v3、candidate 审计、CAS 发布/回滚和递归 GC 均已落地并有自动化测试。

但“代码可执行”不等于“v3 数据已经可用”。当前没有执行 2010 年以来全量回灌，没有发布 v3 active，也没有改动 v2 active pointer。M0 反而发现 v2 现有 lineage 已不完整，因此首次 v3 rebuild 在入口处硬阻断。

## 固定运行时

- Python：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe`
- BaoStock：`0.9.3`
- 官方 wheel SHA-256：`acbd19403285bc4e254cee8297cf0e2646ae2276e5af7e549deed3988ab02293`
- mootdx：`0.11.7`
- BaoStock 版本检查使用 `importlib.metadata.version("baostock")`；不使用异常的模块字符串 `00.9.30`。
- 依赖锁：`configs/china_free_sources.lock.json`

## 主要代码面

- `domains/contracts.py`：`DatePartitionFetchRequest` 与日期 snapshot/events 协议。
- `providers.py`：BaoStock 0.9.3 bulk parser、retry/timeout/global limiter、日期日线/ETF/因子接口、mootdx xdxr、次级域 adapter。
- `qdp_v3/storage.py`：content-addressed immutable raw version、receipt、revision chain。
- `qdp_v3/identity.py`：稳定 `security_id`、官方证据驱动 `symbol_history`、provider/current code 与 PIT code 分离。
- `qdp_v3/quality.py`：raw/canonical/factor/5m 结构和语义闸门、每日代码集合与生命周期分类。
- `qdp_v3/build.py`：按自然年/自然月流式构建 canonical datasets 和 candidate。
- `qdp_v3/corporate_actions.py`：mootdx xdxr lossless raw、官方证据仲裁、参考价一 tick 证明、股本事件。
- `qdp_v3/secondary.py`：财务季报、预告、快报、行业和指数成分；日期级披露从下一交易日可用。
- `qdp_v3/intraday.py`：mootdx/BaoStock 5m 单源完整日选择、quality tier、抽检任务。
- `qdp_v3/manifest.py`、`datasets.py`、`audit.py`、`release.py`、`gc.py`：manifest v3、递归 lineage、semantic audit、CAS、rollback 和 two-stage trash。
- `qdp_v3/update.py`：采集—构建—审计—发布 DAG；M0 freeze 不完整时立即停止。
- `qdp_v3/cli.py` 与公共 `cli.py`：generation routing 和 v3 命令；训练包/memmap 已从 QDP CLI 移除。

## BaoStock bulk 安全协议

日期批量响应不经过普通 `ResultData.next()`：解析器直接读取单次响应的 `fields/data`，强制 `per_page_count == 20000`，字段宽度必须一致，行数达到 20,000 视为潜在截断。provider error、超时、解压/CRC/消息体异常都抛出错误，不能解释成合法空数据。

所有 BaoStock endpoint 共享进程级 limiter；初始并发 1，累计 1,000 请求且网络错误率低于 0.5% 才允许升到 2。批量 endpoint timeout 120 秒，退避 2/5/15 秒。

日线 raw 必须满足日期一致、`(date, code)` 唯一、`adjustflag=3`、0 < rows < 20,000，并与 `query_all_stock`、security master 和相邻日对账。缺口被逐项分为 `not_yet_listed`、`already_delisted`、`suspended`、官方代码重述、identity mapping problem 或 provider gap；后两类和任何未解释大跳变阻断。

## 稳定证券身份

canonical 事实不再以 symbol 为长期主键。provider 返回值保留为 `provider_symbol`，当时真实代码恢复为 `symbol_on_date`，主键使用内部稳定 `security_id`。

`query_stock_basic()` 的名称是当前快照，只能写入 identity 当前属性，禁止反填历史 `name_on_date`。历史名称由对应日期的 `query_all_stock()` 快照产生，或由带官方文档 hash 的事件证据产生；semantic audit 会回查每个名称区间起点的 raw 快照。provider 的 `board=1` 也不直接作为板块枚举，板块按当日 symbol 规则恢复；`689` CDR 归 STAR。`outDate` 是首个无效日，历史区间截止到其前一日。

固定回归：中航电测/中航成飞映射到 `QDP-CN-SZSE-AVICCAC-20100827`。官方公告证据 SHA-256 为 `dd68049c48df826848f361fd9e7b23dd20b6805144a2e5bc36e54db638611488`；`300114.SZ` 有效到 `2025-02-16`，`302132.SZ` 从 `2025-02-17` 生效。即使 BaoStock 在 2016 snapshot 返回 `302132`，canonical `symbol_on_date` 也必须为 `300114.SZ`。

## 因子与公司行动

`query_daily_adjust_factor(date)` 被实现为当日事件接口，`dividOperateDate` 必须等于查询日。字段只显式兼容 `adjustFacto`、`adjustFactor`、`adjust_factor`。

首次重建要求 batch date-events 与每证券 `query_adjust_factor()` 完整事件史做键和值的双路径比较。仅一侧出现或数值不一致的事件进入 disputed；mootdx xdxr 仍只是辅助事件源，必须结合带官方 URL/hash 的证据才能唯一仲裁。无法证明 2010 初始状态的证券标记 `baseline_unproven`，不静默填 1.0。

旧 v2 因子检查只证明键覆盖、正值和 provenance。`600076.SH/2024` 的非事件日密集跳变证明语义不合格；所有依赖旧 factor 的调整收益、标签和研究结论保持 provisional。

## 5 分钟边界

strict 股票日只能选一个完整来源：完整 mootdx 优先，完整 BaoStock fallback；不完整来源禁止拼接。标准 bar_end 为 `09:35..11:30`、`13:05..15:00`，共 48 根。2011-2019 旧 TDX 只能 provisional；2020 年以来由逐证券覆盖和双源冲突审计决定 strict，发布覆盖率至少 99.95%。1 分钟退出 v3 active，但没有删除。

## M0 freeze 真实结果

v2 active SHA-256 仍为：

`e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e`

freeze 保存并 pin 了 39 个仍存在的 v2 dataset，同时 pin 了 17 个 active lineage 中已经缺失的 ancestor id。缺失项为：

- `adjust_factor__dbdfa4e10aa2ef7662014013`
- `corporate_actions__f709a1aa9e476479d683314c`
- `data_platform_trading_calendar__03298281bc6712b72d0b5a6b`
- `index_constituents__57292fa9a393f017b793ad5c`
- `industry_concept__9da8e8e69bbebafa996aed5f`
- `intraday_daily_features__2c8f5a390bf281df3b63584d`
- `limit_intraday_features__5657dd1e33720bac7798fe0e`
- `limit_status__ed4f73356ed9a07b249f90df`
- `market_daily_panel__cb3a5c69017d796af4e705a6`
- `market_daily_raw__10f9bea742d3f8c61c4f2d24`
- `market_intraday_1m__07219ab3cb8d547804bab514`
- `market_intraday_5m__64cfacca8eccb00d6cfe7cb2`
- `name_change__410e040978feb32f90861734`
- `security_status__fe2b31ad5bd3fce5374e75bb`
- `share_capital__63c69110652118d022f2b71d`
- `universe_snapshot__a468bc540bf34f07fa93a1d0`
- `valuation__5005fb2d8755d7f0c2f3aa4c`

状态为 `blocked_missing_lineage`。搜索未找到副本；不能假定一定由哪一次 GC 删除。v2 GC 现在读取 freeze pin，dry-run 不再把仍存在的数据列为可删。发布函数还要求 `status=complete` 且 full shard hash proof，因此无法绕过。

## 兼容性证据

升级前已用 0.9.1 捕获 2010 至 2026 每年首尾交易日共 34 个 anchor、每 anchor 最多 64 个分层证券的 golden fixture，并校验内容 hash 后升级到 0.9.3。

0.9.3 full gate：

- `scope=full`
- anchors：2010 至 2026 每年首尾共 34 个日期
- `issue_count=0`
- 普通 `query_stock_basic()` 多页证明：5537 rows
- wheel SHA-256 与锁定值完全一致

synthetic bulk rows 的 1999/2000/2001 正常、20000 阻断由自动化测试覆盖。最终报告为 `data/qdp_v3/metadata/baostock_0_9_3_compatibility_gate.json`；该报告今后不是 `passed/full` 时，live date partition ingest 仍会拒绝执行。

## Live adapter smoke

- 全 A 股日线：`2010-01-04=1699`、`2016-01-04=2811`、`2026-06-26=5207`，三个 raw 分区均为 strict。
- 因子事件：`2024-07-18=27`，raw 字段实际为 `adjustFacto`，分区为 strict。
- ETF：`2026-06-26=1574`，适配器成功；因为本阶段不做 ETF cross-check/full backfill，保持 provisional。
- 5m：`600000.SH/2020-01-02` 的 mootdx 为空，BaoStock 单源完整 48 根并被选为 strict；没有跨源拼接。
- security master：5537 rows；交易日历 `2010-01-01..2026-06-26` 共 6021 calendar rows。

live 2016 snapshot 同时出现旧/新别名，说明 canonical 必须先映射到稳定 identity，再恢复 `symbol_on_date`；不能把批量返回 code 当历史主键。

## 验证与未执行事项

截至本记录：

- `tests/data_platform`：210 passed，1 个既有 pandas FutureWarning。
- `compileall`：通过。
- 单日 `2010-01-04` candidate smoke 已完成；symbol history、日线、状态、估值与 PIT 研究视图为 strict。semantic audit 按预期被因子参考价证明和 2020+ 5m 覆盖阻断，没有 identity/name/raw hash/manifest lineage 错误。
- `qdp diff --against active` 在首次 v3 发布前会自动以 v2 active 为基线；smoke 已正确识别 v2 active SHA、跨版本主键变化和 schema 增删。该 diff 已进入 update DAG 的 semantic-audit 与 CAS-publish 之间。
- v3 GC dry-run：11 个 candidate datasets 全部 reachable，16 个 raw versions 因 candidate 引用或未满 30 天均不可清理。
- v2 active SHA 未改变。
- 没有全历史 backfill。
- 没有 v3 publish 或 active pointer change。
- 没有删除 v2、旧 1m、旧 5m。

下一安全动作是：恢复 M0 缺失 lineage，或由用户明确批准并定义可替代的完整冻结证明；随后才启动 2010 至今的可恢复全量日期回灌。full compatibility 与少量 live adapter smoke 已完成，不再是当前阻塞点。
