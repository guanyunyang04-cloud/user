# Traditional Quant Research 情景记忆

- 时间顺序证据和长复盘写入这里或 `brain/references/`。

## 2026-06-01

- 创建 `traditional_quant_research` 项目，用于传统方向量化方法研究。
- 初始化并注册项目脑区，建立因子、组合、指标、回测、实验、日志和数据契约骨架。

## 2026-06-03

- 执行免费优先 size 与晋级门禁补强：新增 free size scout、AKShare/CNInfo 重建 size 标准化、proxy-only 诊断源、daily_size 审计 source grade/current cross-check 门禁、weak-year rebuild 和 trial ledger。实际复跑结果保持保守：latest snapshot 无 `daily_size.parquet`，promotion gate 未通过，策略候选数量仍为 `0`。
- 继续补强免费 size gate 证据链：新增 CNInfo share-event audit 与 free current cross-check artifact，修正 `已流通股份` -> `float_share` 解析；正式复跑显示 CNInfo 事件 4/4 样本可诊断重建，但 current quote cross-check 不可用、正式 `daily_size` cache 未写入，promotion gate 继续失败于 `size_gate/return_gate/year_gate/style_exposure_gate`。
- 生成北极星结构化证伪总报告 `frontier_structured_falsification_report_20260603_202448`：最新门禁仍无 out-of-sample supported 策略，阻塞为 size、弱年 return/year 和 style exposure；报告写出 failure matrix、next minimum actions、evidence manifest 和 research log。
