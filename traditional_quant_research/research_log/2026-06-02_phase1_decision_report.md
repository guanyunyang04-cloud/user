# 2026-06-02 Phase 1 Decision Report

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_phase1_decision_report.md`.


## Decision

第一阶段结论：可以进入第二阶段的多因子诊断与多因子打分基线，但不建议直接进入传统 ML 建模阶段。

当前 v2 PIT 快照、标签审计、全周期单因子诊断和基础回测协议已经足以支持下一步做可解释的多因子研究。它们还不足以支持直接把 Ridge、LightGBM、随机森林等传统 ML 模型作为主线，因为长 horizon 极端标签、复权/除权核查、执行约束、因子中性化和样本外协议仍未补强。

建议的阶段门禁是：

- `go`: 多因子 diagnostic baseline。
- `hold`: 传统 ML production-style modeling。
- `next decision`: 多因子基线完成后，再判断是否进入传统 ML。

## Evidence Used

本报告引用以下第一阶段产物：

- 数据/标签审计：`research_log/2026-06-02_v2_data_label_audit.md`。
- 全周期单因子和多 horizon 诊断：`research_log/2026-06-02_full_cycle_factor_diagnostics.md`。
- 基础回测协议升级：`research_log/2026-06-02_backtest_protocol_upgrade.md`。
- 第一条闭环基线：`research_log/2026-06-02_first_loop_baseline.md`。

数据范围固定为 v2 PIT 快照 `baostock_v2_pit_20160101_20260601_stockbasic_fixed`，样本从 `2016-01-04` 到 `2026-06-01`，tradeable panel 为 `7,031,085` 行、`2,526` 个交易日、`3,392` 只证券。

## Findings

### 1. 数据集可用于第一轮传统量化研究

v2 已经不再依赖当前股票列表回看历史，而是基于 Baostock 点时股票池、历史 ST、停牌和上市/退市状态生成每日 universe。研究侧默认读取 `dataset_v2.py` loader，避免实验代码直接访问 Baostock 或 TDX。

标签可用率较高：

- `fwd_ret_1d`: `0.999518`。
- `fwd_ret_5d`: `0.997588`。
- `fwd_ret_20d`: `0.990360`。

主要风险是极端长 horizon 标签仍需核查：`fwd_ret_20d` 中 `abs(ret)>0.5` 有 `57,163` 行。这不必然说明数据不可用，但在进入传统 ML 之前必须明确这些样本来自真实行情、复权/除权、停复牌跳变还是数据异常。

### 2. 单因子里有稳定方向，但等权 baseline_score 较弱

全周期诊断显示，部分传统因子具有稳定方向：

- `reversal_5d_z`: 在 `1d/5d/20d` horizon 上均有正向 RankIC，年度正向率较高。
- `neg_volatility_20d_z`: 在 `5d/20d` 上稳定性较强，年度正向率为 `1.0`。
- `neg_amplitude_20d_z`: 在 `5d/20d` 上稳定性较强，年度正向率为 `1.0`。

但当前 `baseline_score` 是简单等权合成，信号强度很弱：

- `1d` 年均 RankIC 约 `0.003105`。
- `5d` 年均 RankIC 约 `0.002836`。
- `20d` 年均 RankIC 约 `0.001083`。

这说明第二阶段应该先研究因子选择、方向、权重、相关性和中性化，而不是把弱的等权分数直接交给更复杂模型。

### 3. 回测协议已经能比较成本和调仓频率，但还不是生产级

基础回测协议已覆盖 Top-N、手续费和日/周/月调仓频率。在 `2026-01-01` 到 `2026-06-01` 的 smoke run 中，`baseline_score` 的日频 Top-N 表现较好，手续费会显著压低收益但没有完全消除信号。

不过该结果仍只属于 `diagnostic/backtest_only` 证据：

- 样本窗口短。
- 尚未加入涨跌停不可成交。
- 尚未加入真实滑点或冲击成本。
- 长 horizon 尚未处理重叠持仓。
- 仍需补充更长区间的成本和调仓频率矩阵。

## Decision Rationale

进入多因子阶段的理由：

- v2 PIT 数据和 loader 已能支持可复验研究。
- 标签覆盖率足以做 `1d/5d/20d` 诊断。
- 单因子中存在稳定方向，不是完全无信号。
- 回测协议已能初步观察成本、换手和调仓频率的影响。

暂缓传统 ML 的理由：

- 传统 ML 会放大标签异常、样本选择和执行约束缺失的影响。
- 当前因子池还很小，且缺少中性化、相关性控制和滚动训练协议。
- 等权 `baseline_score` 本身较弱，复杂模型可能只是学习噪声或隐含暴露。
- 还没有建立足够强的可解释多因子对照组，无法判断 ML 的真实增益。

## Recommended Phase 2

第二阶段目标应是“多因子研究基线”，不是“直接训练传统 ML 模型”。

优先工作：

1. 扩展传统因子池：反转、动量、波动率、振幅、流动性、成交额、均线结构和量价关系。
2. 增加收益标签版本：原始收益、市场中性收益、分组中性收益和多 horizon 标签。
3. 做因子质量诊断：覆盖率、缺失率、年度 RankIC、分位数组合收益、因子相关性和方向稳定性。
4. 构造多因子打分：等权 rank、IC 加权、滚动 IC 加权、低相关因子组合。
5. 升级回测协议：更长样本、手续费网格、调仓频率网格、换手、涨跌停不可成交和极端行情切片。
6. 复核极端标签：重点核查 `5d/20d` 的大幅收益样本和除权/复权影响。

## Entry Criteria For Traditional ML

只有同时满足以下条件，才建议进入传统 ML：

- 多因子 rank baseline 在全周期和年度切片中优于单因子。
- 多因子组合在成本和调仓频率压力测试下仍有稳健表现。
- 极端标签问题有明确处理规则。
- 训练/验证/测试切分、滚动训练和防未来函数协议已固定。
- 已有强 baseline，可衡量 ML 是否真的带来增益。

## Final Stage Classification

第一阶段成果分类：

- 数据集：`diagnostic_ready`，可用于传统因子研究；仍需补充复权/极端标签核查。
- 单因子：`diagnostic`，存在可继续研究的稳定方向。
- baseline_score：`diagnostic`，不应作为最终策略。
- 回测协议：`backtest_only`，可用于比较研究方案；还不是生产级执行模型。
- 下一阶段：`go_to_multifactor_diagnostics`。
