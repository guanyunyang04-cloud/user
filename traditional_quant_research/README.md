# Traditional Quant Research

Canonical brain source: `traditional_quant_research/brain/identity_layer.md`.

本项目用于传统量化方法研究。本文只保留项目入口；研究框架、数据契约、实验日志和当前状态以 `traditional_quant_research/brain/` 为准。

## Entry

- 项目身份：`traditional_quant_research/brain/identity_layer.md`
- 当前状态：`traditional_quant_research/brain/state_center.md`
- 操作规则：`traditional_quant_research/brain/operations_center.md`
- 研究框架：`traditional_quant_research/brain/references/research_framework.md`
- 实验日志：`traditional_quant_research/brain/references/research_log/`

## Body

- `backtest.py`、`factors.py`、`metrics.py`、`portfolio.py`：可复用研究函数和轻量基线模块。
- `experiments/`：实验脚本；新实验日志默认写入脑区 references。
- `tests/`：稳定性与基础行为测试。
- `data/`、`cache/`、`output/`：本地数据、缓存和实验产物，不作为阅读文档真源。

## Quick Check

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests
```
