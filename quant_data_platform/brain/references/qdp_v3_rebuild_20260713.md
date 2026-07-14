# QDP v3 数据基底重建实施与审计记录

日期：`2026-07-13..2026-07-15`

> 当前合同（2026-07-15）：本文按时间记录了多轮设计与执行，前文中的双源历史验证、99.95% 覆盖、因子官方仲裁、BaoStock 2013+ 历史复刻和首次发布即退休均已被最后一节取代。当前规范以 `qdp_v3_20260715_trusted_source_5m`、schema `3.3.0`、manifest `4` 及 QDP hot-path brain 为准。

## 结论

QDP v3 已从方案进入可执行和历史回灌阶段：BaoStock 0.9.3 日期批量协议、不可变 raw、稳定证券身份、PIT 代码历史、因子事件双路径、双源 5m、次级 PIT、manifest v3、candidate 审计、CAS 发布/回滚和递归 GC 均已落地并有自动化测试；2010—2012 日线与日期因子事件 raw 已完成。

但“raw 回灌 strict”不等于“v3 因子和全部数据已经可用”。当前没有发布 v3 active，也没有改动 v2 active pointer。M0 发现的 17 个缺失 v2 ancestor 由用户明确授权的替代冻结证明闭合入口，但该证明不声称祖先已恢复，删除型 GC 继续禁止。因子双路径仲裁、2010 baseline、后续年份和 5m 全量仍未完成。

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

所有 BaoStock endpoint 共享进程级 limiter，生产网络并发固定为 1。两个独立 login 的 live probe 即使在历史错误率为 0 时仍出现 `10001001 用户未登录`，因此取消按 1,000 请求错误率自动升为双 session 的路线。批量 endpoint timeout 120 秒，退避 2/5/15 秒。

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

初次状态为 `blocked_missing_lineage`。搜索未找到副本；不能假定一定由哪一次 GC 删除。用户随后明确授权接受不可恢复 lineage gap 的替代证明：39 个仍存在 dataset、23,738 shards、23,777 files 已全量 hash，缺失 id 和引用证据继续保留，最终状态为 `complete_with_authorized_lineage_gap`。发布验证只能显式接受该授权合同，不能把它伪装为 `lineage_complete=true`；后续 5m-only 授权将删除边界更新为“v3 发布与发布后全部校验通过，再由专用 retirement 命令只清理四条旧分钟链”。

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

## 2010—2012 历史回灌

日期 raw 分区结果：

| 年份 | 交易日 | 日线 rows | all-stock rows | 因子事件 rows | 零事件日 | 质量 |
|---|---:|---:|---:|---:|---:|---|
| 2010 | 242 | 453,319 | 453,236 | 1,495 | 38 | 全部 strict |
| 2011 | 244 | 537,338 | 537,094 | 1,762 | 45 | 全部 strict |
| 2012 | 243 | 587,438 | 587,196 | 1,935 | 52 | 全部 strict |

合计 729 个交易日、1,578,095 行日线和 5,192 条因子事件。逐年复核均无缺日、日期错位、重复键、非法 adjustflag、20,000 行截断或结构 blocker。日线与 all-stock 行数差异来自已显式仲裁的 provider 当前代码重述，不代表漏数。

最新 candidate 为 `candidate__5060cee5169a45e4162200ad`：覆盖 2010—2012 的 729/729 日、2,483 个 securities。日线、状态、估值、identity、symbol history、PIT signal/open 与 calendar 为 strict；factor event/daily 仍因 `factor_event_not_verified_or_arbitrated` 和 `factor_reference_price_proof_missing` quarantined。semantic audit 报告 `data/qdp_v3/audits/candidate__5060cee5169a45e4162200ad/20260714T052337+0000__semantic.json` 只出现这两个唯一 blocker code，未出现 identity、PIT、manifest lineage 或不适用的 pre-2020 5m blocker。

三年 candidate 本地构建约 42.6 分钟，CPU 基本单核持续推进，峰值工作集约 1.56GB。该耗时是 canonical build 的独立性能债，不是网络下载瓶颈；后续应优化向量化/分片增量构建，不能用跳过 identity、PIT 或 factor gate 换速度。

## 下载提速与失败路线

最终生产路线：

1. 一个隔离 spawn 子进程只登录 BaoStock 一次，跨日期复用 session。
2. 同一交易日的 batch daily 与 `query_all_stock` 在该 session 内作为一个原子命令执行。
3. `max_workers=2` 只让 CPU/存储处理与下一日期网络请求流水化；网络 session 数和同时在途请求数均为 1。
4. 覆盖范围更大的 strict calendar 可直接复用；前一交易日映射和 raw refs 一次建索引，去掉逐日全目录扫描的 O(N²) 路径。
5. 每个 job id 使用 OS advisory single-writer lock；重复进程立即报错，进程崩溃后由 OS 自动释放，不依赖可过期 PID 文件。

实测相近年度：

- 日线：2011 旧路径约 `2932.7s`，2012 新路径约 `1239.7s`，约 `2.4x`。
- 因子事件：2011 旧路径约 `937.4s`，2012 新路径约 `79.6s`，约 `11.8x`。
- 2012 因子新取 235 日、复用 8 日探针，0 失败、0 网络错误。

曾测试两个独立 BaoStock login。因子小探针成功，但日线探针在 `2012-01-06` 返回 `10001001 用户未登录`，说明第二次登录可使另一 session 失效；该路线被永久否决，现有 revoke 证据保留在 `data/qdp_v3/metadata/baostock_parallel_health_gate.json`。历史错误率再低也不自动开放第二个网络槽。

## 新增官方代码历史

2012 审计曾把 `001872.SZ`、`001914.SZ`、`302132.SZ` 误判为 provider 缺口，原因是当前 security master 给新代码继承了原证券上市日。现改为按官方 symbol interval 生成当日预期集合：

- `000022.SZ` 至 `2018-12-25`，`001872.SZ` 自 `2018-12-26`；CNInfo PDF SHA-256 `bf406b1759dd7fda45cc46927e36316840b0b1152bb40677a7b52e11568d4350`。
- `000043.SZ` 至 `2019-12-15`，`001914.SZ` 自 `2019-12-16`；CNInfo PDF SHA-256 `46160431c51df23c30b3be802eb140492255128fb166d103ed735a62f303ab05`。
- `300114.SZ` 至 `2025-02-16`，`302132.SZ` 自 `2025-02-17`；既有官方证据继续有效。

修复后 `2012-09-10` 的 expected/raw/all-stock 均为 2,459 codes，missing/extra 为 0，分区由 quarantined 转为 strict；内容 hash 未被伪造或替换，只新增了正确的质量评估。

## 2026-07-14 全域下载提速

在继续历史回灌前，对 BaoStock、mootdx、5m 分片和财务查询做了实际调用链审计并完成以下改造：

1. BaoStock 日期批量之外，逐证券因子史、财务/业绩域和 5m symbol-range 也可在各自长任务内复用单一隔离子进程/login。响应按 symbol 隔离错误，只重试失败 symbol，成功 symbol 不重复下载；通用 provider 默认仍保持原隔离调用合同，避免改变既有调用方语义。
2. mootdx 对内置服务器先并行 TCP 排序，再用两行 `600000` 日线做真实协议探针；只缓存协议健康节点，失败节点进入 300 秒 circuit/negative cache。当前机器所有公共候选均未通过协议 bars 探针：旧首次失败约 `23.138s`，新首次全源判定约 `6.435s`，同一失败窗口后续约 `0.008s`，随后可立即转 BaoStock。这只是 2026-07-14 的外部节点状态。
3. mootdx 历史分页从最新日期向后回翻，因此 5m 不再按 symbol-month 重复请求远端；每只证券一次获取完整目标区间，规范化后切成不可变月分区。对 2020-01 至 2026-06 的 78 个月，估算页面上界从约 4,280 降为 105（`40.76x`），顶层请求从 78 降为 1。
4. BaoStock 5m fallback 在首次需要某证券时也一次获取完整目标区间并本地切月。`600000.SH/2020-01-02..2026-06-26` live probe 返回 75,312 rows、1,569 dates，每日均为 48 根，耗时约 `57.795s`；同 session 月度探针外推约慢 `1.6x`。因此只能把 78 倍表述为逻辑请求数减少，不能表述为实测吞吐提升。
5. symbol-month 恢复改为确定键直接查 raw partition，消除分区增长时反复遍历整个 raw domain 的 O(N²) 文件系统开销。
6. 完整财务季报按 security master 生命周期裁剪：起点为上市日前 550 日所在季度，保留约 6 个前置季度作初始 TTM/PIT 证据，退市后不再查询；业绩预告和快报保持全局范围。当前 5,537 个证券估算五类季度 endpoint calls 从 1,799,525 降为 1,256,305，减少 543,220（`30.19%`）。

warm-session 小样本显示第二批逐证券因子史约从 `1.87s` 降至 `0.33s`（约 `5.7x`），第二个月 BaoStock 5m 约从 `1.85s` 降至 `0.20s`（约 `9.2x`）；首次持久会话仍包含约 `2.56s` 登录成本。旧/新调用的逐证券因子史 67 rows 与 5m 912 rows 均逐值等价，质量选择与禁止跨源拼接的闸门未放宽。

## 当前验证与下一动作

- 新下载调用链的 provider/identity/intraday 定向回归：46 passed。
- `quant_data_platform/tests` 完整回归：340 passed，只有 1 个既有 pandas FutureWarning。
- v2 active SHA 未改变；没有 v3 publish 或 active pointer change。
- 没有删除 v2、旧 1m、旧 5m；删除型 GC 继续禁止。
- 三个仍期望已归档 v1 CLI 可执行的旧测试已改为验证 archive boundary，防止训练 pack/memmap/provider-eval 被重新接回 QDP 日常 CLI。

下一安全动作是继续 2013 以后日期回灌，同时启动逐证券完整因子史、mootdx xdxr 和官方公告仲裁；只有 2010 baseline、参考价一 tick 证明、因子双路径和 5m 发布门全部通过后，才构建全历史 release candidate。full compatibility、M0 入口和 2010—2012 raw 已不再是当前阻塞点。

## 2026-07-14 5m-only 合同收敛与 Tushare bootstrap

用户随后明确取消 v3 1m，QDP v3 合同更新为 `qdp_v3_20260714_bootstrap_5m`、schema `3.2.0`、manifest `3`。`market_intraday_5m` 成为唯一分钟 canonical 事实域，正式起点为 2010-01-01；旧 2020 strict 分界已删除，质量按每个股票日的结构、日线、身份与来源证据决定。v2 当前 active 的 1m/5m 合同保持原样，直到 v3 正式发布与发布后校验全部通过。

代码面删除了 v3 `intraday_1m.py`、`mootdx_1m.py`、`--include-1m`、`--frequency 1m|5m`、1m watermark/lag、241→240 和 1m→5m 生产转换。`HistoryPageFetchRequest` 固定为 5m，Tushare provider 内部固定 `stk_mins(freq=5min)`；canonical 5m 以 `security_id, trade_date, bar_end` 为主键，按自然年和稳定 64 个 security bucket 分片。候选发布要求 5m watermark 与 daily 完全相等，不只是“不落后”。

Tushare-compatible proxy 被限定为 `2010-01-01..2026-07-13` 的一次性历史启动源，`upstream_provenance=not_exposed`，不使用 SDK。Token 只从 `QDP_TUSHARE_PROXY_TOKEN` 读取；exact-token 全工作区文本扫描为 0。历史页每次最多 8,000 行、时间倒序，页级 staging/cursor 可恢复；单证券完成后合并为一个 immutable raw。三个 worker 共享进程级 limiter，最多 3 个在途请求；429 会对全部 worker 施加共享冷却。实测 135/121/108 次每分钟均持续触发网关 429，而独立成功吞吐约为 94 次/分钟；因此生产默认收敛为 96 次/分钟、burst 1，保留 3 workers 覆盖慢响应，并在 429 后按冷却窗口自适应降至最低 90。把在途数固定降为 2 的实测成功吞吐仅约 80 次/分钟，不作为默认路线。

分钟单位用固定样本与同源日线聚合证明：

- `600000.SH/2010-01-04`：48 bars，volume ratio `0.99999943`，amount ratio `1.00049729`。
- `000001.SZ/2010-01-04`：48 bars，volume ratio `0.99999686`，amount ratio `1.00007160`。
- `302132.SZ/2016-01-06`：48 bars，volume ratio `0.99999352`，amount ratio `1.00005662`；旧代码 `300114.SZ` 返回空，再次证明 provider code 不能作历史身份。
- `600000.SH/2026-07-13`：48 bars，volume ratio `1.00000050`，amount ratio `1.00000000`。

唯一稳定换算是分钟 raw `vol=share`、`amount=CNY`，两者 canonical scale 均为 1.0；兼容闸门现会在 `(1,100)` 与 `(1,1000)` 候选中证明唯一尺度，manifest 显式记录 raw units 和 scale。

mootdx 节点不再依赖硬编码 IP。候选是持久化 last-good、`tdxpy.constants.hq_hosts`、mootdx `HQ_HOSTS/SERVER.HQ` 的去重并集；先并发 TCP 排序，再对最快 24 个中的最多 8 个做真实 `frequency=0` 5m 请求。`600000.SH`、`000001.SZ`、`600076.SH` 最近完整日均为 48 bars 且止于 15:00；冷启动 9.6877 秒、热启动 0.0784 秒，保存最快 3 个协议健康节点。全源失败才打开 300 秒负缓存并交给 BaoStock。

新增 `retire-v2-intraday` 只在 v3 active SHA、semantic audit、5m 99.95% 覆盖、daily/5m watermark、全部 shard hash、v2/v3 diff、下游切换和无运行 job/消费者全部通过后执行；先写 immutable retirement manifest，再只删除 v2 `market_intraday_1m`、`market_intraday_5m`、`intraday_daily_features`、`limit_intraday_features` parquet。当前 v3 未发布，因此该命令会拒绝删除，v2 active SHA 未改变。

验证结果：5m-only/provider/identity 目标测试 76 passed；加入单位与 pre-2020 evidence-driven 回归后相关测试 36 passed；完整 `tests/data_platform` 为 239 passed、1 个既有 pandas FutureWarning；`compileall` 与 `git diff --check` 通过。后台 bootstrap 已从原 job 断点恢复，`stock_basic` 与 `trade_cal` 完成，`daily` 正在推进；日志无协议/网络错误，额度或外部下载未完成前不构造 candidate、不发布、不退休 v2。

## 2026-07-14 5m-only 实施闭环与生产恢复

进一步审计修复了三个会影响正式 canonical 的边界：同一 `security_id` 的新旧 provider code 只要历史值有任何差异就整体进入 quarantine，不能因 PIT code 恰好存在而忽略另一个别名；完整 mootdx 与不完整 BaoStock 并存时仍选择完整 mootdx，只有两个完整来源超阈值冲突才隔离；两波历史捕获即使都是有官方代码重述证据的合法空分区也能稳定合并，不要求伪造时间列。对应回归已加入 identity/intraday 与 proxy 测试。

Tushare 参考域的实际执行顺序收敛为 `daily -> daily_basic -> identity -> status -> factor -> dividend -> financial`。BaoStock 2013+ 日期日线/因子回灌在单独线程与 Tushare 并行，但 BaoStock 内部继续保持一个持久 login 和一个网络在途；Tushare quota/disk 暂停时先记录 BaoStock 结果，再以可恢复状态退出。2026-07-14 22:43 本地证据快照中，全市场 `daily` 与 `daily_basic` 均完成 `4011/4011`；DAG 已自动切换到 5,864 个 L/D/P 全历史 identity 任务并完成 `3072/5864`。修正 limiter 后的新进程前 1,053 个真实请求为 `1053/1053` 成功，网络、协议、429 均为 0；旧进程的 7% 级 429 重试不再被最终任务成功掩盖。BaoStock job 同时为 731 个已有日期复用加 182 个新日期完成并继续向 2013+ 推进。所有 job/cursor 均为原子 JSON 或 immutable raw，v2 active 未改变。

mootdx 当前代码重新运行真实闸门，`600000.SH`、`000001.SZ`、`600076.SH` 在 `2026-07-13` 均为 48 根且最后一根 15:00；冷启动 `1.4336s`、last-good 热启动 `0.0692s`。报告为 `data/qdp_v3/compatibility/mootdx_5m__edc251d8b9a04640fab6f93b.json`，没有硬编码探针 IP。

semantic audit 新增自动 secret gate：控制目录、仓库生产文本、candidate 实际 dataset manifest/quality proof 以及已有 lineage 审计单次读取的 raw receipt 都检查当前 Token 精确值、未脱敏 token URL 和 credential 字段；内容 hash 形式的兼容缓存 token 明确不误判。当前运行产物扫描为零泄露。首次 bootstrap CAS 发布后会再跑一次 semantic/full-shard hash 审计；只有复核通过才自动调用 `retire-v2-intraday --delete --yes`，任一前置失败状态为 `published_cleanup_blocked` 且不删除旧 parquet。

接管复核又修正了两个发布证据边界：已有 5m cutoff proof 通过时必须返回 sidecar 中的 48 根证据，不能把 immutable legacy 1m 锁中的 241 根回写到 run 状态；canonical 5m 构建和 full audit 统一为自然年加稳定 64 个 `security_id` bucket，每个 year/bucket 只生成一个可包含多证券的 shard，并逐证券复核稳定桶归属。

最终代码验证为 `255 passed`、1 个既有 pandas FutureWarning；`compileall`、`git diff --check`、`brain_sync_audit`、`doc_guard` 与 `integrity_check` 均通过。CLI 会拒绝 `--include-1m` 和 `--frequency 1m`；v3 生产路由中的 1m 引用为零，剩余 `market_intraday_1m` 字样只存在于 v2 冻结与受保护退休清单。当前 H 盘空闲仍高于 200 GiB 暂停阈值。全量参考域、历史 5m、identity/factor/PIT 仲裁、candidate 与发布仍是进行中任务，不能把“代码已实现”误写成“数据基底已发布”。

## 2026-07-15 可信单源、紧凑存储与两阶段退休

用户明确将个人研究的数据合同进一步简化为可信单源：Tushare-compatible proxy 独占 `2010-01-01..2026-07-13` historical canonical；截止日后 BaoStock 更新日线、状态和因子事件，mootdx 优先更新完整 5m 股票日，只有 mootdx 不完整时才使用 BaoStock 完整日 fallback。历史 BaoStock/Tushare 全量逐行验证、2013+ BaoStock 历史复刻和“两个完整来源冲突仲裁”退出生产 DAG。指定来源仍必须通过 schema、单位、分页、主键、stable identity、PIT 和完整性闸门；不完整数据 quarantine，不拼接、不插值。

合同更新为 `qdp_v3_20260715_trusted_source_5m`、schema `3.3.0`、manifest `4`。新 active 只产生 strict/quarantined，legacy provisional 只读。首次正式 candidate 只要求九域：`trading_calendar`、`security_identity`、`symbol_history`、`market_daily_raw`、`security_status_daily`、`adjust_factor_daily`、`eligible_signal_D`、`tradable_open_D1`、`market_intraday_5m`。估值、factor events、财务、分红、行业、指数和股本降为非阻塞 enrichment。5m strict 覆盖低于 98% 阻断，98%—99% 告警，至少 99% 正常；5m/daily watermark 必须相等。

因子合同同步简化：每只证券首个 2010+ Tushare `adj_factor` 归一为 1，保留相邻因子比率；cutoff 后只把 BaoStock 新事件比率续接到已有序列，避免重复乘入历史变化。pre-2010 官方 baseline、三路径全集一致、xdxr/dividend/公告仲裁和逐事件参考价不再是首次发布条件。`300114/302132`、`000022/001872`、`000043/001914`、`600076` 等已知例外由 `configs/qdp_v3_corrections.json` 在 normalize 后、canonical 写入前处理；raw 永不改写，correction 重复、区间冲突或 old value 不匹配会阻断构建。

H 盘不再先复制到 F。用户取消备份后直接运行了提升权限的 `chkdsk H: /f`；结果为 dirty bit=false、HealthStatus=Healthy、OperationalStatus=OK、bad sectors=0，日志保存在 `C:\Users\ASUS\AppData\Local\QDP\maintenance\chkdsk_H_20260715.log`。数据根固定为 `H:\quant_project\quant_data_platform\data`，runtime root 为 `C:\Users\ASUS\AppData\Local\QDP\runtime`；H 只保存大数据和少量 manifest，C 保存 SQLite WAL、job、cursor、heartbeat 与 staging。

紧凑存储代码已实现并修正分区策略：reference raw 固定为 undated x 1，低频时序 raw 为 year x 1，只有 5m raw 为 year x 16 stable security buckets；均使用 zstd，目标 part 约 384 MiB、最大 1 GiB。SQLite 记录 receipt 与 bundle row-group 映射，并导出带 SHA256 sidecar 的 `raw_index.parquet`、`raw_receipts.parquet`，因此发布 manifest 不依赖 C 盘数据库。bundle-aware read 支持 latest 与精确历史 revision；SQLite 丢失时可从校验过的 export 恢复。删除有双重保护：必须先在小域完成不删源 round-trip，再显式 `compact --delete-source --yes`；当前尚未全量 compact 或删除 legacy raw。

现有 Tushare stock-basic、calendar、4,011 个 daily、4,011 个 daily-basic 和 5,864 个 identity raw 继续复用；旧 `stk_limit` 过程任务已中断，新 status 路由只保留 `suspend_d`。所有陈旧 provider job 已接管为 `interrupted_recoverable`，生产下载尚未恢复；下一生产顺序是先完成 compact 验证，再运行 5m-only compatibility 和全历史 5m，额度耗尽后补 status/factor，最后构建九域 candidate。v2 active 未改变。H 上 2026-07-15 00:52 出现的空 v3 active（SHA-256 `4215123b7f0d64c855a2f2a5c414930dfd41c25289b4cf53ae9ccd2c6fc0d19d`）已确认无候选/无数据并删除；public read 现通过 manifest v4、候选/哈希、完整九域与 published-at 验证 active，pytest 也自动清除 production roots/credentials。默认 status 已恢复为 v2 `status=ok`、as-of `2026-06-26`、17 datasets。

退休被改为严格两阶段：bootstrap CAS publish 只写 `awaiting_post_cutoff_incremental` gate，不删除 v2；必须再完成一次 watermark 晚于 `2026-07-13` 的增量 CAS publish，并通过 active SHA/candidate、daily/5m watermark、coverage、semantic audit、diff、全部 shard hash、下游切换和无消费者/job 检查，才写 retirement manifest 并物理删除 v2 的 1m、旧 5m 与两条分钟特征链。首次发布即删除的旧描述已失效。
