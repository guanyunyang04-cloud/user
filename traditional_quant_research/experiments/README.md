# Experiments

实验目录用于保存传统量化研究的可复现实验入口。

每个实验建议包含：

- `config.*`: 数据范围、样本切分、参数和交易约束。
- `run.*`: 可重复执行的脚本或 notebook 入口。
- `summary.md`: 结论摘要、失败原因、下一步。
- `artifacts/`: 小型可提交结果；大型结果放入 ignored `output/` 或外部存储。

默认先建立简单对照组，再引入更复杂规则。

## First Loop Baseline

第一条研究闭环入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.baseline_first_loop --start-date 2026-01-01 --end-date 2026-06-01 --split-date 2026-04-01 --horizon 1 --top-n 100 --fee-bps 10 --write-research-log
```

该实验从 v2 PIT 快照读取 `tradeable panel`，构造基础传统价量因子、未来收益标签、横截面 z-score、`baseline_score`，输出全样本与 IS/OOS 的 IC、分组收益和 Top-N 等权多头基线。产物写入 ignored 的 `traditional_quant_research/output/experiments/baseline_first_loop/`，可选研究日志写入 `research_log/`。

## V2 Data Label Audit

数据/标签审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.audit_v2_data_labels --horizons 1,5,20 --write-research-log
```

该实验审计 v2 PIT 快照的 universe/status、OHLCV 异常、`1d/5d/20d` 未来收益标签可用率和极端收益样本。产物写入 ignored 的 `traditional_quant_research/output/experiments/v2_data_label_audit/`，摘要写入 `research_log/`。

## V2.1 Daily Metrics Audit

日频指标缺失审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_daily_metrics_audit --root traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/<snapshot_id> --start-date 2026-05-25 --end-date 2026-06-01 --write-research-log
```

该实验以 `daily_universe` 的 `date, code` 为期望键，左连接可选 `daily_metrics.parquet`，分别报告全 universe 和 `is_tradeable=True` 样本中 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM` 的覆盖率、有限值数量、零值数量和年度覆盖率。它只判断 metrics 表是否具备进入研究输入的覆盖率基础，不验证估值字段 PIT timing，也不解决真实市值/流通市值来源问题。

## V2.1 Metrics Semantics Audit

日频指标语义与时点风险审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_metrics_semantics_audit --root traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/<snapshot_id> --start-date 2016-01-01 --end-date 2026-06-01 --write-research-log
```

该实验读取含 metrics 的 v2.1 tradeable panel，比较 `pctChg` 与逐股 `close-to-close` 收益的差异，输出差异分布和极端样本；同时统计 `peTTM/pbMRQ/psTTM/pcfNcfTTM` 的覆盖、负值、零值、日变动率以及与收盘收益的相关性。它用于证明字段语义和质量风险，不证明估值字段 PIT 发布时间；在完成独立时点验证前，估值字段应至少滞后一日使用，且不能单独支持策略候选晋级。

## V2.1 Metrics Exposure Diagnostics

frontier 信号 metrics 暴露诊断入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_metrics_exposure_diagnostics --years 2024,2025,2026 --final-end-date 2026-06-01 --write-research-log
```

该实验在当前 `20d/monthly/top_n=200/buffer=3.0` candidate-frontier 研究上下文中，重建 rolling IC、IC-weighted 和 low-corr 三条 frontier 信号，并把 `turn/pctChg`、流动性/动量/低波动 proxy 和一日滞后估值字段加入暴露诊断。输出包括 `daily_signal_metric_correlation.csv`、`yearly_signal_metric_correlation.csv`、`basket_metric_exposure.csv`、`basket_metric_exposure_summary.csv`、`year_meta.csv` 和 `summary.md`。当前 run `v2_metrics_exposure_diagnostics_20260603_064734` 显示最大平均绝对 signal-metric 相关 `0.760402`、最大平均绝对篮子指标主动暴露 `1.170983`；高暴露集中在低波动、低流动性/小成交额、动量和换手结构。该实验是候选晋级门禁，不产生策略候选。

## Full Cycle Factor Diagnostics

全周期单因子诊断入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.full_cycle_factor_diagnostics --horizons 1,5,20 --top-n 100 --top-n-signals baseline_score --fee-bps 10 --skip-top-n --write-research-log
```

该实验基于 v2 PIT tradeable panel 构造基础价量因子和 `baseline_score`，输出 `1d/5d/20d` 全周期 IC、年度 IC 稳定性和年度分组收益。`--skip-top-n` 用于先完成轻量稳定性诊断；Top-N/成本/调仓频率矩阵留给基础回测协议升级实验。

## Backtest Protocol Upgrade

基础回测协议升级入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.backtest_protocol_upgrade --horizon 1 --top-n 50,100 --fee-bps 0,10 --frequencies daily,weekly,monthly
```

该实验在 v2 PIT panel 上，对 `baseline_score` 做 Top-N、成本和调仓频率矩阵比较，先覆盖日/周/月三种调仓频率，再比较手续费敏感性。输出用于判断当前信号是否值得进入更复杂的多因子或传统 ML 阶段。

## Phase 1 Decision Report

阶段决策报告：

```text
traditional_quant_research/research_log/2026-06-02_phase1_decision_report.md
```

结论：第一阶段支持进入多因子诊断与多因子打分基线，但不建议直接进入传统 ML 生产式建模。传统 ML 需要等待多因子 baseline、极端标签核查、执行约束和滚动验证协议补强后再启动。

## Multifactor Baseline

第二阶段多因子基线入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.multifactor_baseline --start-date 2024-01-01 --end-date 2026-06-01 --horizon 5 --label-mode xsec-excess --rolling-window 252 --rolling-min-periods 60 --max-factor-corr 0.75 --neutralize-by log_amount_mean_20d_z --top-n 50,100,200 --fee-bps 0,10,20 --frequencies daily,weekly,monthly --write-research-log
```

该实验读取 v2 PIT `tradeable panel`，构造第一阶段价量因子，按单因子 RankIC 方向校正后生成等权 rank 多因子分数、全样本 IC 加权 rank 多因子分数、rolling IC 加权 rank 多因子分数、低相关因子 rank 分数和 proxy-neutralized rank 分数，并输出因子覆盖率、日度横截面相关性、单因子 IC、多因子 IC、分位组合收益、分位 high-minus-low spread、rolling IC 权重表、低相关因子列表、neutralized 因子列表和 Top-N 成本/调仓频率回测矩阵。`--label-mode raw` 使用原始 forward return；`--label-mode xsec-excess` 使用同日横截面均值剥离后的 forward return，更适合作为选股诊断入口，但其 Top-N 指标是 alpha-style 排序诊断，不是独立可投资组合收益。rolling IC 权重只使用评分日前的历史日期；历史不足 `--rolling-min-periods` 时回退为等权方向。低相关因子选择按绝对 RankIC 从强到弱贪心筛选，并剔除与已选因子平均日度相关性超过 `--max-factor-corr` 的冗余因子。`--neutralize-by` 默认使用 `log_amount_mean_20d_z` 作为流动性/规模 proxy 做按日横截面残差化；这不是正式行业/市值中性化。

若要评估较短样本但让 rolling IC 权重拥有更长历史，可使用 `--history-start-date`。例如只评估 2026，但用 2025 作为权重暖场：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.multifactor_baseline --history-start-date 2025-01-01 --start-date 2026-01-01 --end-date 2026-06-01 --horizon 5 --label-mode raw --rolling-window 252 --rolling-min-periods 60 --max-factor-corr 0.75 --neutralize-by log_amount_mean_20d_z --top-n 100 --fee-bps 0,10,30 --frequencies weekly --horizon-backtest --horizon-buffer-multipliers 1.0,1.5,2.0 --write-research-log
```

启用 `--horizon-backtest` 时，除全样本和年度汇总外，还会输出 `horizon_backtest_period_summary.csv`，按月和季度切分严格持有期收益；同时输出 `basket_factor_exposure.csv`，按月和季度审计实际入选 Top-N 篮子的因子暴露、同日 universe 均值和 active exposure。实验始终输出 `rolling_ic_weight_audit.csv`，按月和季度审计 rolling IC 权重、方向、fallback rate 和历史样本数。

聚焦审计可用 `--signals` 限定候选输入，避免无关信号拖慢执行约束实验。启用 `--execution-constraints --limit-threshold 0.095` 时，严格持有期回测会近似处理：下一全市场交易日涨停或不可交易则不买入，未成交资金保留为现金；目标退出日跌停或不可交易则延迟到下一可卖收盘。该口径仍不包含排队优先级、部分成交、滑点和冲击成本。

组合暴露约束可用 `--selection-filters`，例如：

```powershell
--selection-filters "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8,neg_volatility_20d_z<=1.0,neg_amplitude_20d_z<=1.2"
```

该过滤只让不合格的信号日候选不可选，不删除未来 entry/exit 价格路径；因此 horizon 回测仍使用完整 tradeable panel 做成交和退出模拟。

## Phase 2 Interpretation

阶段解释报告：

```text
traditional_quant_research/research_log/2026-06-02_phase2_multifactor_validation_interpretation.md
```

结论：多因子 rank baseline 已经显著强于第一阶段 `baseline_score`，可以进入 raw-return、年度稳定性、成本压力和执行约束更强的候选策略筛选基础设施阶段；但当前多因子结果仍属于 `diagnostic/backtest_only`，不能直接声明为策略候选，也不建议立即进入传统 ML 生产式建模。

## Raw-Return Validation

原始收益验证解释：

```text
traditional_quant_research/research_log/2026-06-02_phase3_raw_return_validation_interpretation.md
```

结论：`raw` IC 和分位结果继续支持 `multifactor_low_corr_rank_score` 与 `multifactor_rolling_ic_weighted_score` 作为候选输入；但 `fwd_ret_5d` 搭配 daily Top-N 暴露了持有期/调仓频率错配风险，下一步必须实现 horizon-aligned portfolio simulator 后再评价真实策略收益。

## Horizon-Aligned Backtest Foundation

严格持有期回测基础模块：

```text
traditional_quant_research/horizon_backtest.py
```

核心接口：

```python
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest
```

该模块按 `research_panel.build_factor_label_panel` 的标签语义执行：t 日收盘后产生信号，下一可交易日开盘入场，持有到 `horizon` 对应可交易日收盘。`non_overlapping=True` 时会跳过仍在持有期内的 rebalance date，避免把 `5d/20d` 未来收益在 daily Top-N 中重复计入。当前它是候选策略筛选的协议基础；后续还需要继续加入涨跌停不可成交、停牌持仓卖出约束、滑点/冲击成本和年度切片报告。

多因子实验可通过以下参数额外输出严格持有期矩阵：

```powershell
--horizon-backtest
```

当前解释报告：

```text
traditional_quant_research/research_log/2026-06-02_horizon_aligned_multifactor_validation_interpretation.md
```

结论：`multifactor_rolling_ic_weighted_score` 是当前最强候选输入；`multifactor_low_corr_rank_score` 继续作为排序诊断对照。二者仍未升级为策略候选，下一步需要年度稳定性和执行约束压力测试。

年度稳定性解释：

```text
traditional_quant_research/research_log/2026-06-02_horizon_aligned_yearly_stability_interpretation.md
```

结论：rolling IC 在 2024/2025 强，但 2026 半年明显降温且成本后转负；年度稳定性和成本鲁棒性尚未通过，下一步应优先做换手/成本压缩、月度/季度切片和执行约束。

buffer 换手压缩解释：

```text
traditional_quant_research/research_log/2026-06-02_buffered_horizon_turnover_validation_interpretation.md
```

结论：`buffer_multiplier=2.0` 能把 rolling IC weekly Top-100 的 mean turnover 从约 `1.3349` 降到约 `1.0027`，并改善 30 bps 下的净收益和 Sharpe；但 2026 半年仍未通过，下一步需要做 2026 月度/季度切片和 rolling IC 权重审计。

2026 月度/季度切片与权重审计解释：

```text
traditional_quant_research/research_log/2026-06-02_2026_period_weight_audit_interpretation.md
```

结论：用 `2025-01-01` 暖场后，2026 rolling IC 失效主要集中在 March 和 May，且并非单纯成本问题；下一步应加入涨跌停/停牌执行约束和选中篮子暴露审计，再判断是否继续推进模型复杂度。

2026 选中篮子暴露审计解释：

```text
traditional_quant_research/research_log/2026-06-02_2026_basket_exposure_audit_interpretation.md
```

结论：March/May 弱表现对应的实际入选篮子持续偏小成交额、弱动量、低波动/低振幅防御和短期反转暴露；问题更像当前传统因子族的 regime/exposure 失效，而不是单纯 rolling 权重或交易成本问题。

2026 执行约束审计解释：

```text
traditional_quant_research/research_log/2026-06-02_2026_execution_constrained_audit_interpretation.md
```

结论：约束版 weekly Top-100 对 rolling IC 和 low-corr 两个候选输入都进一步恶化；本次没有涨停买入阻塞，主要是跌停/不可卖导致延迟退出。执行约束后两个候选输入更明确不能升级为策略候选。

2026 执行约束 buffer 网格解释：

```text
traditional_quant_research/research_log/2026-06-02_2026_constrained_buffer_grid_interpretation.md
```

结论：buffer 能降低换手，但不能修复 constrained gross alpha。low-corr + buffer `2.0` 是本次相对最优设置，但 0 bps 与 30 bps 仍为负，不能升级为策略候选。下一步应转向暴露上限或 regime filter。

2026 暴露上限审计解释：

```text
traditional_quant_research/research_log/2026-06-02_2026_exposure_cap_audit_interpretation.md
```

结论：第一版暴露上限能压低小成交额、弱动量、低波动/低振幅极端暴露，并改善 low-corr，但仍未转正；rolling IC 在该约束下明显恶化。`multifactor_baseline_20260602_135636` 因错误删除未来价格路径而废弃，不作为证据。

## Low-Corr Exposure Grid

低相关多因子暴露控制网格入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_exposure_grid --history-start-date 2025-01-01 --fit-start-date 2025-01-01 --fit-end-date 2025-12-31 --start-date 2026-01-01 --end-date 2026-06-01 --horizon 5 --top-n 100 --fee-bps 0,30 --frequencies weekly --buffer-multipliers 2.0 --execution-constraints --write-research-log
```

该实验专门服务于当前 `multifactor_low_corr_rank_score` 暴露控制研究：先用指定 fit window 选择低相关因子并确定方向，再在 evaluation window 中批量测试流动性、动量、低波动和低振幅暴露阈值。selection filter 只让不合格信号日候选不可选，不删除未来 entry/exit 价格路径；因此严格 horizon 回测仍能模拟涨跌停、停牌和延迟退出。产物写入 ignored 的 `traditional_quant_research/output/experiments/low_corr_exposure_grid/`，包括 `grid_summary.csv`、`grid_selection_reports.csv`、`grid_period_summary.csv`、`grid_basket_exposure.csv` 和 `summary.md`。该实验输出仍是 `backtest_only` 证据，不能单独升级为策略候选。

## Low-Corr Regime Filter

低相关多因子 regime filter 入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_regime_filter --history-start-date 2025-01-01 --fit-start-date 2025-01-01 --fit-end-date 2025-12-31 --start-date 2026-01-01 --end-date 2026-06-01 --horizon 5 --selection-filters "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8" --top-n 100 --fee-bps 0,30 --frequencies weekly --buffer-multipliers 2.0 --execution-constraints --write-research-log
```

该实验在当前 low-corr watchlist filter 之上测试市场状态过滤。regime 指标来自信号日已知的同日 tradeable universe 汇总，例如 `market_ret_20d_mean`、`breadth_20d_positive_rate`、`market_volatility_20d_mean` 和 `market_amplitude_20d_mean`。关键口径：调仓日先由完整 evaluation calendar 固定，再判断该调仓日 regime 是否允许开仓；不允许的调仓日会跳过，不会提前顺延到同周或同月的前一个好日子。产物写入 ignored 的 `traditional_quant_research/output/experiments/low_corr_regime_filter/`，包括 `market_regime.csv`、`regime_summary.csv`、`regime_filter_reports.csv`、`regime_period_summary.csv`、`regime_basket_exposure.csv` 和 `summary.md`。该实验输出仍是 `backtest_only` 证据。

## Low-Corr Regime Yearly Validation

低相关多因子 regime 多年份验证入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_regime_yearly_validation --years 2024,2025,2026 --final-end-date 2026-06-01 --selection-filters "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8" --regime-filter-specs "market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45" --top-n 100 --fee-bps 0,30 --frequencies weekly --buffer-multipliers 2.0 --execution-constraints --write-research-log
```

该实验按评估年份自动使用前一年作为 fit window，例如 `2026` 评估使用 `2025-01-01` 到 `2025-12-31` 选择 low-corr 因子和方向。它调用 `low_corr_regime_filter` 子实验并汇总 `yearly_validation_summary.csv`、`yearly_validation_regime_reports.csv`、`yearly_validation_period_summary.csv` 和 `yearly_validation_aggregate.csv`。默认可用 `--market-ret-thresholds` 与 `--breadth-thresholds` 跑阈值网格；也可用 `--regime-filter-specs` 指定少量规则做快速复验。该实验输出用于检查多年份稳定性和成本门禁，不能单独升级为策略候选。

## Low-Corr Regime Capital Scaling

低相关多因子动态仓位缩放入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_regime_capital_scaling --history-start-date 2025-01-01 --fit-start-date 2025-01-01 --fit-end-date 2025-12-31 --start-date 2026-01-01 --end-date 2026-06-01 --selection-filters "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8" --top-n 100 --fee-bps 0,30 --frequencies weekly --buffer-multipliers 2.0 --execution-constraints --write-research-log
```

该实验不跳过调仓日，而是在 market regime 上生成同日 `capital_scale`，再传入 horizon-aligned 回测的 `capital_col`。默认测试 `breadth_soft`、`breadth_floor60` 和 `ret_breadth_soft` 三类软仓位计划，并保留 `baseline_full_capital` 对照。产物写入 ignored 的 `traditional_quant_research/output/experiments/low_corr_regime_capital_scaling/`，包括 `capital_scaling_summary.csv`、`capital_plan_reports.csv`、`capital_scaling_period_summary.csv`、`capital_scaling_basket_exposure.csv` 和 `summary.md`。该实验输出仍是 `backtest_only` 证据。

## Low-Corr Regime Capital Yearly Validation

低相关多因子动态仓位多年份验证入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_regime_capital_yearly_validation --years 2024,2025,2026 --final-end-date 2026-06-01 --selection-filters "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8" --top-n 100 --fee-bps 0,30 --frequencies weekly --buffer-multipliers 2.0 --execution-constraints --write-research-log
```

该实验按评估年份自动使用前一年作为 fit window，逐年调用 `low_corr_regime_capital_scaling`，并汇总 `capital_yearly_validation_summary.csv`、`capital_yearly_validation_reports.csv`、`capital_yearly_validation_period_summary.csv` 和 `capital_yearly_validation_aggregate.csv`。当前用途是验证软仓位是否能在弱年降损且少伤强年；只有多年份、30 bps、执行约束后的收益和回撤同时过门禁，才允许升级为策略候选。

## Low-Corr Horizon Cost Grid

低相关多因子持有期/成本网格入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_horizon_cost_grid --history-start-date 2025-01-01 --fit-start-date 2025-01-01 --fit-end-date 2025-12-31 --start-date 2026-01-01 --end-date 2026-06-01 --horizons 5,10,20 --filter-specs "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8" --top-n 100,200 --fee-bps 0,30 --frequencies weekly,monthly --buffer-multipliers 2.0,3.0 --execution-constraints --write-research-log
```

该实验逐 horizon 调用 `low_corr_exposure_grid`，汇总不同持有期、调仓频率、Top-N、buffer 和成本设置下的 `horizon_cost_summary.csv`。它用于检查当前信号是否只是 5 日/周频协议过度成本敏感。输出仍是 `backtest_only` 证据，不能单独升级为策略候选。

## Low-Corr Horizon Cost Yearly Validation

低相关多因子持有期/成本多年份验证入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_horizon_cost_yearly_validation --years 2024,2025,2026 --final-end-date 2026-06-01 --horizons 20 --filter-specs "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8" --top-n 100,200 --fee-bps 0,30 --frequencies weekly,monthly --buffer-multipliers 2.0,3.0 --execution-constraints --write-research-log
```

该实验按评估年份使用前一年 fit，复验 20 日低换手协议是否跨年稳定。当前最强 `baseline_no_filter / horizon=20 / monthly / top_n=200 / buffer=3.0 / 30 bps` 行属于 candidate-frontier，而非正式策略候选；后续必须补充暴露、容量、滑点冲击和更长样本审计。

## Low-Corr Candidate-Frontier Audit

低相关多因子候选前审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_candidate_frontier_audit --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 0,30,60,100 --execution-constraints --capital-amounts 10000000,50000000,100000000 --write-research-log
```

该实验固定当前 candidate-frontier 协议 `low_corr / horizon=20 / monthly / top_n=200 / buffer=3.0 / baseline_no_filter`，逐年以前一年 fit 选择低相关因子和方向，然后输出 fee stress、交易明细、选中持仓、篮子因子暴露、流动性/容量 proxy、因子覆盖率和 low-corr 元数据。当前 run `low_corr_candidate_frontier_audit_20260602_170249` 显示 30/60 bps 多年份表现仍强，但交易样本偏少、2026 证据薄、100 bps 最差年转负，且持仓持续偏低流动性和弱动量；因此仍是 `candidate-frontier/backtest_only`，策略候选数量为 `0`。

## Low-Corr Candidate Signal Comparison

同协议候选信号对照入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_candidate_signal_comparison --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 0,30,60,100 --execution-constraints --rolling-window 252 --rolling-min-periods 60 --write-research-log
```

该实验固定 `20d/monthly/top_n=200/buffer=3.0/execution_constraints` 协议，只替换信号列，比较 `baseline_score`、`multifactor_equal_rank_score`、`multifactor_ic_weighted_score`、`multifactor_rolling_ic_weighted_score` 和 `multifactor_low_corr_rank_score`。输出包括 `signal_protocol_summary.csv`、`signal_protocol_aggregate.csv`、交易明细、年度/月度/季度切片、篮子暴露、信号覆盖率、rolling IC 权重审计和元数据。当前 run `low_corr_candidate_signal_comparison_20260602_171522` 显示 rolling IC 是 30 bps 均值冠军，IC-weighted 是高成本鲁棒性最好的一条，low-corr 排第三；因此 frontier 集合扩展为 rolling IC、IC-weighted、low-corr 三条信号，但正式策略候选数量仍为 `0`。

## Low-Corr Frontier Impact Stress

同协议候选信号冲击成本压力测试入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_impact_stress --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 30 --capital-amounts 10000000,50000000,100000000 --impact-bps-per-1pct 0,1,2,5,10 --execution-constraints --rolling-window 252 --rolling-min-periods 60 --write-research-log
```

该实验固定 `20d/monthly/top_n=200/buffer=3.0/execution_constraints` 协议，对 rolling IC、IC-weighted 和 low-corr 三条 frontier 信号加入基于选中篮子 signal-date 流动性的参与率冲击成本。冲击成本使用每笔交易 p95 participation：`impact_rate = participation_p95 * impact_bps_per_1pct / 100`，再乘以换手得到额外成本。输出包括 `impact_stress_summary.csv`、`impact_stress_aggregate.csv`、冲击调整交易表、逐笔流动性、年度流动性摘要、元数据和 `summary.md`。当前 run `low_corr_frontier_impact_stress_20260602_174946` 显示三条 frontier 在 `30 bps / 100m / 10 bps per 1 pct participation` 下仍三年为正，但月度非重叠交易样本仍薄、2026 仅 `2` 笔，且低流动性/弱动量暴露仍在；因此结果只强化 `candidate-frontier/backtest_only`，策略候选数量仍为 `0`。

## Low-Corr Frontier Neutralization Audit

同协议候选信号 proxy/industry neutralization 审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_neutralization_audit --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 30 --neutralize-by log_amount_mean_20d_z --execution-constraints --rolling-window 252 --rolling-min-periods 60 --include-industry --industry-neutralize --industry-column industry --write-research-log
```

该实验固定 `20d/monthly/top_n=200/buffer=3.0/execution_constraints` 协议，对 rolling IC、IC-weighted 和 low-corr 三条 frontier 信号做两类控制：按日横截面对 `log_amount_mean_20d_z` 残差化，作为规模/流动性 proxy；按同日同行业对信号去均值，作为 Baostock `month-start` 行业暴露初审。输出包括 `neutralization_summary.csv`、`neutralization_aggregate.csv`、交易表、篮子因子暴露、篮子行业暴露、信号覆盖率、信号-neutralizer 相关性、信号行业暴露、元数据和 `summary.md`。当前 run `low_corr_frontier_neutralization_audit_20260602_224153` 显示三条 industry-neutral 信号在 30 bps 下仍为 `3/3` 正收益年，rolling IC 均值年化 `0.366028`、IC-weighted `0.302858`、low-corr `0.284148`；但 Top-N/buffer/execution 后仍存在行业 active weight。该结果仍是 `candidate-frontier/backtest_only`，不能替代组合层行业中性和真实市值/流通市值控制。

组合层行业 cap 可通过同一入口追加：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_neutralization_audit --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 30 --neutralize-by log_amount_mean_20d_z --execution-constraints --rolling-window 252 --rolling-min-periods 60 --include-industry --industry-neutralize --industry-column industry --group-col industry --max-group-weight 0.10 --write-research-log
```

该 cap 在 Top-N/buffer 选股阶段限制单行业目标持仓上限。当前 run `low_corr_frontier_neutralization_audit_20260602_231532` 显示 `max_group_weight=0.10` 后，rolling IC industry-neutral 均值年化 `0.371601`、最差年 `0.094033`，IC-weighted industry-neutral `0.293380`，low-corr industry-neutral `0.271272`；但该规则只是单行业最高权重门禁，不是完整行业中性优化器，且执行约束后实际成交 holdings 少于 `top_n` 时 realized industry weight 可能略高于 cap。

metrics 多指标中性化可通过同一入口追加 `--include-metrics`，并把 metrics 衍生暴露列加入 `--neutralize-by` 与 `--exposure-columns`：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_neutralization_audit --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 30 --neutralize-by log_amount_mean_20d_z,momentum_20d_z,neg_volatility_20d_z,turn_xsec_z --exposure-columns log_amount_mean_20d_z,momentum_20d_z,neg_volatility_20d_z,turn_xsec_z,pctChg_xsec_z --include-metrics --execution-constraints --write-research-log
```

该模式输出 `neutralization_basket_exposure_summary.csv`，用于比较中性化前后实际 Top-N 篮子的指标 active exposure。当前 run `low_corr_frontier_neutralization_audit_20260603_073116` 显示硬多指标中性化能把信号层相关性降到数值零，但 30 bps 下 rolling IC / IC-weighted / low-corr 均值年化分别从 `0.350177/0.280980/0.262745` 降到 `0.173130/0.172495/0.135335`，并提高换手、恶化回撤；篮子层 `log_amount_mean_20d_z` 和 `neg_volatility_20d_z` active exposure 只部分降低，`turn_xsec_z` active exposure 反而上升。结论：该结果是暴露诊断门禁，不是策略候选晋级证据。

组合层连续暴露惩罚可通过同一入口追加：

```powershell
--portfolio-exposure-penalty-cols log_amount_mean_20d_z,neg_volatility_20d_z,momentum_20d_z,turn_xsec_z --portfolio-exposure-penalty-strength 0.25
```

该参数传入 `horizon_aligned_top_n_backtest()`，在 Top-N/buffer 选股阶段按加入候选后的篮子平均暴露偏离扣分。当前 run `low_corr_frontier_neutralization_audit_20260603_081810` 显示 `strength=0.25` 基本不伤原始 frontier，rolling IC / IC-weighted / low-corr 的 30 bps 均值年化为 `0.353015/0.277791/0.260064`；但原始篮子 active exposure 只小幅下降，尚不足以作为候选晋级门禁。strength grid 入口见下节。

## Low-Corr Frontier Exposure Penalty Grid

同协议组合层暴露惩罚强度网格入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_exposure_penalty_grid --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 30 --exposure-penalty-strengths 0,0.25,0.5,1.0 --execution-constraints --include-metrics --write-research-log
```

该实验固定 `20d/monthly/top_n=200/buffer=3.0/execution_constraints` 协议，对 rolling IC、IC-weighted 和 low-corr 三条 frontier 信号测试多档 `exposure_penalty_strength`。输出包括 `exposure_penalty_summary.csv`、`exposure_penalty_aggregate.csv`、交易明细、篮子暴露、篮子暴露汇总、元数据和 `summary.md`。当前 run `low_corr_frontier_exposure_penalty_grid_20260603_085614` 显示 rolling IC 最优为 `strength=0.25`，IC-weighted 最优是不加惩罚，low-corr 最优为 `strength=1.0`；惩罚能温和压低篮子 active exposure，但不能完成暴露中性门禁。该实验输出仍是 `candidate-frontier/backtest_only` 证据，策略候选数量为 `0`。

## Low-Corr Frontier Combined Constraint Audit

同协议组合约束联动门禁入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_combined_constraint_audit --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 30 --capital-amounts 100000000 --impact-bps-per-1pct 0,10 --signal-penalty-strengths multifactor_rolling_ic_weighted_score=0.25,multifactor_ic_weighted_score=0.0,multifactor_low_corr_rank_score=1.0 --group-col industry --max-group-weight 0.10 --execution-constraints --include-metrics --include-industry --write-research-log
```

该实验把当前 frontier 的信号特定 exposure penalty、组合层行业 cap、固定费率和参与率冲击成本放在同一门禁下评估。输出包括 `combined_constraint_summary.csv`、`combined_constraint_aggregate.csv`、交易表、流动性/参与率、篮子风格暴露、行业暴露和元数据。当前 run `low_corr_frontier_combined_constraint_audit_20260603_094139` 显示三条 frontier 在 `30 bps / 100m / 10 bps per 1 pct participation` 下仍三年为正，但交易样本仍薄，且低成交额/低波动/弱动量/换手 active exposure 仍明显。该实验强化 `candidate-frontier/backtest_only`，不产生策略候选。

扩展样本门禁可用同一入口跑 2017-2026：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_frontier_combined_constraint_audit --years 2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 --final-end-date 2026-06-01 --write-research-log --research-log-path traditional_quant_research/research_log/2026-06-03_low_corr_frontier_extended_combined_constraint_audit.md
```

扩展 run `low_corr_frontier_combined_constraint_audit_20260603_122047` 显示样本数已不再是主问题：10 bps impact 下 rolling IC / low-corr / IC-weighted 的 total periods 为 `60/64/64`，但均值年化降为 `0.096737/0.056183/0.023878`，正收益年份均为 `0.6`，最差年均为负。因此该 frontier 不能按 2024-2026 强表现升级为样本外支持。

## Frontier Promotion Gate

frontier 候选晋级门禁入口，默认是 Baostock-only 研究模式：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_promotion_gate --write-research-log
```

该门禁读取当前 `low_corr_frontier_combined_constraint_audit` 的结构化输出和 `v2_daily_size_audit` 的 `summary.json`，不重跑回测。默认 `--research-mode baostock_only` 要求 `30 bps / 100m / 10 bps per 1 pct participation` 下收益、年度稳定、回撤、最少 `24` 个非重叠 periods、以及月度篮子风格 active exposure 均通过；真实 `daily_size` readiness 会作为 `true_size_gate` 披露，但不阻塞 Baostock-only 研究审查。若这些非 true-size gate 通过，promotion level 只能是 `candidate-frontier/baostock_only`，不能计为 `strategy_candidate` 或 `out_of_sample_supported`。

true-size 策略候选审查必须显式运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_promotion_gate --research-mode true_size --write-research-log
```

`--research-mode true_size` 仍要求 `daily_size_ready_for_research=True`，且该证据必须来自认证 PIT `daily_size` 来源；`amount`、`volume`、`turn`、`log_amount_mean_20d_z`、CNInfo/AkShare/efinance/current quote 都不能替代 true-size gate。历史真实 run `frontier_promotion_gate_20260603_125120` 读取 2017-2026 扩展 combined constraint 输出后，三条 frontier 均通过 `sample_gate` 和 `drawdown_gate`，但失败于 `size_gate`、`return_gate`、`year_gate` 和 `style_exposure_gate`；best mean annualized return 为 rolling IC 的 `0.096737`，仍不能升级为策略候选。当前 `strategy_candidate_count=0`。

## Frontier Personal Candidate Gate

当前 North Star 的个人小资金候选门禁入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_personal_candidate_gate --write-research-log
```

该门禁读取当前 `low_corr_frontier_combined_constraint_audit` 的结构化输出，不重跑回测。它面向 `Baostock-only personal quant strategy research`，不要求真实总市值/流通市值，也不输出机构级 `strategy_candidate`。默认检查：Baostock snapshot、2017-2026 walk-forward meta、`30 bps / 100m / 10 bps per 1 pct participation` 压力行覆盖个人 `1m` 小资金、执行约束启用、均值年化至少 `5%`、正收益年份率至少 `0.6`、最差年不低于 `-35%`、worst drawdown 不低于 `-25%`、非重叠 periods 至少 `50`、proxy 风格暴露不过度极端且无 optimizer fallback。通过后分级为 `personal_backtest_candidate`，`paper_tracking_recommendation=user_discretion`，decision 为 `personal_strategy_candidates_selected`；失败则保持 `personal_research/backtest_only`。当前边界是：agent 只负责筛出好的模型/策略候选，后续风险、记录、paper/live tracking 和执行决策由用户自行判断。

## Frontier Personal Protocol Grid

个人小资金 Top-N 协议网格入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_personal_protocol_grid --top-n-values 20,50,100 --years 2017,2018,2019,2020,2021,2022,2023,2024,2025,2026 --write-research-log
```

该实验按 `top_n` 网格逐次调用 `low_corr_frontier_combined_constraint_audit`，随后对每个 combined run 调用 `frontier_personal_candidate_gate`，最后汇总为 `personal_protocol_grid_ledger.csv` 和 `personal_protocol_grid_top_n_summary.csv`。它用于把当前 `20d/monthly/buffer=3.0` frontier 从原来的 `top_n=200` 研究协议，系统化比较到更贴近个人小资金的 `top_n=20/50/100` 协议。输出只允许产生 `personal_backtest_candidate` 或 `personal_research/backtest_only`；通过行只表示模型/策略候选已选出，不能直接称为 `personal_paper_candidate`、`strategy_candidate` 或生产候选。后续是否记录、跟踪、实盘或做风险裁量不再由本流程判断。

## Frontier Personal Paper Tracking Bootstrap

个人候选 paper tracking 准备包入口已取消：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_personal_paper_tracking_bootstrap --personal-gate-run-dir traditional_quant_research/output/experiments/frontier_personal_candidate_gate/<run_id> --write-research-log
```

该入口保留为历史兼容模块，但当前运行会直接抛出 `RuntimeError`。本项目不再由 agent 生成 `paper_tracking_candidates.csv`、`paper_tracking_protocol.csv`、日志模板或审查规则；`personal_backtest_candidate` 已是 agent 工作边界内的候选选择结果。后续记录、风险审查、paper/live tracking 和执行判断都由用户自行处理。

## Frontier Personal Paper Tracking Review

个人 paper tracking 日志审查入口已取消：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_personal_paper_tracking_review --bootstrap-run-dir traditional_quant_research/output/experiments/frontier_personal_paper_tracking_bootstrap/<run_id> --tracking-log-path <filled_paper_tracking_log.csv> --write-research-log
```

该入口保留为历史兼容模块，但当前运行会直接抛出 `RuntimeError`。本项目不再由 agent 审查 paper log、输出 `personal_paper_candidate`、降级建议或继续跟踪建议；agent 只交付候选选择证据，后续记录和判断由用户处理。

## Frontier Failure Attribution

frontier 扩展样本失败归因入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_failure_attribution --combined-run-dir traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit/low_corr_frontier_combined_constraint_audit_20260603_122047 --write-research-log
```

该实验只读取 combined constraint 产物，不重跑回测，用于把 promotion gate 的失败拆成年度收益、成本冲击、执行阻塞、流动性和篮子 active exposure。真实 run `frontier_failure_attribution_20260603_132813` 显示三条 frontier 的 weak years 均为 `2017,2018,2022,2023`，total weak signal-years 为 `12`；rolling IC 是均值最强信号，但 positive year rate 仍为 `0.6`。结论：当前 frontier 需要重建跨阶段稳健性，不能继续按近三年强窗口微调后晋级。

## Frontier Weak-Year Regime Attribution

frontier 弱年份市场状态归因入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_weak_year_regime_attribution --failure-run-dir traditional_quant_research/output/experiments/frontier_failure_attribution/frontier_failure_attribution_20260603_132813 --write-research-log
```

该实验读取 failure attribution 的 `yearly_failure_attribution.csv`，再从 latest v2 PIT `tradeable panel` 重建 broad market regime，不重跑 frontier 回测。输出包括 `market_regime_daily.csv`、`yearly_market_regime.csv`、`weak_year_regime_profile.csv`、`signal_year_regime_attribution.csv`、`weak_vs_positive_regime_summary.csv` 和 `summary.md`。真实 run `frontier_weak_year_regime_attribution_20260603_134922` 确认共同弱年仍为 `2017,2018,2022,2023`；相对正收益年，弱年 `breadth_20d_positive_rate` 低约 `0.069451`，`market_ret_20d_mean` 低约 `0.025886`，`breadth_5d_positive_rate` 低约 `0.032508`。结论：frontier 失败更像广度/20日市场强度不足下的全信号共振失效，而不是单一信号、成本或样本数问题。任何由此产生的 regime rule 仍需 fit/eval 分离并重跑 full combined-constraint gate。

## V2 Industry/Size Source Audit

v2 PIT 快照行业/市值字段来源审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_industry_size_source_audit --write-research-log
```

该实验默认只检查当前 v2 snapshot schema 和本地 Baostock 客户端能力，不做长任务网络拉取。输出包括 `snapshot_schema.csv`、`source_field_audit.csv`、`baostock_client_capabilities.csv`、`summary.json` 和 `summary.md`。

若要短窗口 live probe Baostock 日线扩展字段，可运行：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_industry_size_source_audit --live-history-probe --live-stock-basic-probe --history-probe-code sh.600000 --history-probe-start-date 2026-05-25 --history-probe-end-date 2026-06-01 --write-research-log
```

当前 run `v2_industry_size_source_audit_20260603_063037` 显示 latest v2.1 snapshot 已有真实行业表和 `turn` 换手率字段，但仍没有真实市值、流通市值、股本或 share-base 字段；Baostock 日线支持 `turn`、`pctChg`、`peTTM`、`pbMRQ`、`psTTM`、`pcfNcfTTM`，不支持 `turnover`/`turnover_rate`，也不接受 `totalShare/liqaShare/total_mv/float_mv/market_cap/float_market_cap`；`query_stock_basic` 只返回 `code/code_name/ipoDate/outDate/type/status`。市值/流通市值需新增外部 PIT cap/float-cap 来源，或明确保持 proxy-only。

## V2.2 External Size Source Scout

外部 PIT 市值/流通市值来源侦察入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_external_size_source_scout --write-research-log
```

该实验生成 `local_package_capabilities.csv`、`external_size_source_candidates.csv`、`summary.json` 和 `summary.md`，把 Tushare、JoinQuant、RQData、AkShare、efinance 与 Baostock 放入同一张可复验来源矩阵。当前 run `v2_external_size_source_scout_20260603_115009` 显示本地已安装 `tushare=1.4.29`、`akshare=1.18.63`、`efinance=0.5.8`、`baostock=0.9.1`，但没有 `jqdatasdk/rqdatac`，也没有已认证并 live-probe 通过的 PIT daily cap/float-cap 来源。v2.2 首选验证对象是 Tushare `daily_basic` 的 `total_mv/circ_mv/total_share/float_share/free_share`；JoinQuant/RQData 是有订阅时的机构级备选；AkShare/efinance 只作为公开端点交叉检查，不能默认作为 PIT 主源。策略候选数量仍为 `0`。

Tushare `daily_basic` 双票 live probe 入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_tushare_size_probe --symbols 600000.SH,000001.SZ --trade-dates 20260525,20260601 --write-research-log
```

该 probe 输出 `tushare_daily_basic_probe.csv`、标准化 `daily_size_probe.csv/parquet`、`failures.csv`、`summary.json` 和 `summary.md`，验证 `total_mv/circ_mv/total_share/float_share/free_share` 的字段存在、非空率、双票双日期覆盖和单位假设。标准化产物由 `size_source.standardize_tushare_daily_basic_size()` 映射到项目 `daily_size` schema，可被 `dataset_v2.load_pit_daily_size()` 读取；当前 run `v2_tushare_size_probe_20260603_114552` 为 `skipped/auth_missing`，说明本地 Tushare 已安装但未配置 `TUSHARE_TOKEN/TS_TOKEN`，live probe 未实际执行；配置 token 后可直接重跑。

v2.2 `daily_size.parquet` 覆盖率审计入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.v2_daily_size_audit --write-research-log
```

该审计以 `daily_universe` 的 `date, code` 为期望键，左连接可选 `daily_size.parquet`，分别报告全 universe 和 `is_tradeable=True` 样本中 `total_market_cap/float_market_cap/total_share/float_share/free_share` 的覆盖率、正值数量、年度覆盖率以及 `market_cap_unit/share_unit/source` 组合。当前 run `v2_daily_size_audit_20260603_103517` 对 latest snapshot 返回 `daily_size_absent`：`universe_rows=7451610`、`tradeable_rows=7031085`、`size_rows=0`、`min_tradeable_coverage=0.0`、`daily_size_ready_for_research=False`。这说明 v2.2 size 表尚未生成，不产生策略候选。
