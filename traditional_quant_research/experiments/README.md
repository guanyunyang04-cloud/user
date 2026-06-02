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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.baseline_first_loop --start-date 2026-01-01 --end-date 2026-06-01 --horizon 1 --top-n 100 --fee-bps 10 --write-research-log
```

该实验从 v2 PIT 快照读取 `tradeable panel`，构造基础传统价量因子、未来收益标签、横截面 z-score、`baseline_score`，输出 IC、分组收益和 Top-N 等权多头基线。产物写入 ignored 的 `traditional_quant_research/output/experiments/baseline_first_loop/`，可选研究日志写入 `research_log/`。
