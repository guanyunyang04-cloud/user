# 2026-06-02 Horizon-Aligned Backtest Foundation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_horizon_aligned_backtest_foundation.md`.


## Decision

已新增 horizon-aligned Top-N 回测基础模块，用于修正 `fwd_ret_5d/20d` 在 daily Top-N 中被重复年化的问题。该模块是第三阶段“多因子候选策略筛选基础设施”的协议基础，不改变旧 `backtest_protocol.py` 的诊断用途。

当前状态：

- `go`: 使用 `horizon_backtest.horizon_aligned_top_n_backtest` 作为多日 horizon 候选筛选的默认回测入口。
- `hold`: 继续把旧 daily Top-N 的 `fwd_ret_5d/20d` 年化收益解释为真实 PnL。
- `next`: 将 `multifactor_low_corr_rank_score` 与 `multifactor_rolling_ic_weighted_score` 接入该模拟器，输出年度切片和成本压力矩阵。

## Implemented Module

核心文件：

- `horizon_backtest.py`
- `tests/test_horizon_backtest.py`

核心接口：

```python
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest
```

默认语义：

- t 日收盘后生成信号。
- 下一可交易日开盘入场。
- 持有到 `horizon` 对应可交易日收盘。
- `non_overlapping=True` 时，仍在上一篮子持有期内的 rebalance date 会被跳过。
- 使用等权 Top-N。
- 使用目标持仓变化计算换手和交易成本。
- 摘要按持有期频率年化，而不是把多日标签按日年化。

## Smoke Evidence

真实 v2 小窗口 smoke：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe - <<script>
from traditional_quant_research.research_panel import load_baseline_factor_panel
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest

panel = load_baseline_factor_panel(start_date="2026-01-01", end_date="2026-06-01", horizons=(5,))
result = horizon_aligned_top_n_backtest(panel, "baseline_score", horizon=5, top_n=100, fee_bps=10, rebalance_frequency="daily")
print(result.summary)
</script>
```

摘要结果：

- horizon: `5`
- rebalance frequency: `daily`
- non-overlapping periods: `16`
- periods per year: `50.4`
- annualized return: `0.162110`
- Sharpe: `1.119832`
- max drawdown: `-0.040444`
- mean turnover: `0.806250`
- mean net return per holding period: `0.003189`

解释：该 smoke 只验证新协议可以在真实 v2 panel 上工作，并展示非重叠口径会把 5 日 daily 诊断压回 16 个持有期。它不是策略候选证据，因为使用的是第一阶段 `baseline_score`，区间只有 2026 半年，且尚未加入涨跌停/停牌持仓卖出约束。

## Remaining Gaps

进入策略候选前仍需补齐：

1. 多因子信号接入：至少跑 `multifactor_low_corr_rank_score` 与 `multifactor_rolling_ic_weighted_score`。
2. 年度切片：逐年收益、Sharpe、最大回撤、换手、成本敏感性。
3. 执行约束：涨停不可买入、跌停不可卖出、停牌持仓处理。
4. 成本模型：手续费、滑点、可选冲击成本。
5. 样本外协议：rolling walk-forward 或明确 train/validation/test 切分。

## Classification

- horizon-aligned simulator: `protocol_foundation`
- smoke result: `implementation_smoke`
- strategy candidate count: `0`
