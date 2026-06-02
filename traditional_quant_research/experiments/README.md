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
