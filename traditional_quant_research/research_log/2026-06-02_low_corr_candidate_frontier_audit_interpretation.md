# Low-Corr Candidate-Frontier Audit Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_candidate_frontier_audit_interpretation.md`.


## Question

`multifactor_low_corr_rank_score / horizon=20 / monthly / top_n=200 / buffer=3.0 / baseline_no_filter` 是当前第一条 candidate-frontier 协议。本次审计问题是：它能否在执行约束、成本压力、选中篮子暴露和容量 proxy 下升级为正式策略候选。

## Run

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_candidate_frontier_audit --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 0,30,60,100 --execution-constraints --capital-amounts 10000000,50000000,100000000 --write-research-log
```

- Run ID: `low_corr_candidate_frontier_audit_20260602_170249`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Artifacts: `traditional_quant_research/output/experiments/low_corr_candidate_frontier_audit/low_corr_candidate_frontier_audit_20260602_170249`

## Findings

- Fee stress remains strong at normal assumptions: 30 bps mean annualized return `0.262745`, min annualized return `0.076594`, positive years `3/3`, mean Sharpe `2.144099`, worst max drawdown `-0.114807`.
- 60 bps still remains positive in all three evaluation years: mean annualized return `0.213437`, min annualized return `0.032992`.
- 100 bps stress breaks the weakest year: mean annualized return `0.150434`, min annualized return `-0.022631`, positive years `2/3`.
- Evidence count is thin: only `17` completed monthly trades across 2024, 2025, and 2026; 2026 has only `2` trades through `2026-06-01`.
- Execution constraints are not empty: 2024 has `7` delayed exits and `5` limit-down delayed exits; 2025 has `4` delayed exits and `2` limit-down delayed exits.
- Capacity proxy is acceptable for moderate capital but not enough for promotion: at `100m` capital, worst single-name participation proxy is `11.8058%` in 2024, `5.5611%` in 2025, and `4.3539%` in 2026, measured against signal-date amount.
- Selected baskets are consistently exposed to lower liquidity and weaker momentum: yearly mean `log_amount_mean_20d_z` is about `-0.97` to `-1.08`, and yearly mean `momentum_20d_z` is about `-0.56` to `-0.62`.
- The protocol's alpha appears tied to a defensive/reversal structure, not a neutral broad-market selection effect. Monthly basket exposures show persistent positive `neg_amplitude_20d_z`, `neg_volatility_20d_z`, and `reversal_5d_z` active exposure.

## Decision

Do not promote to strategy candidate yet. Classification remains `candidate-frontier/backtest_only`; formal strategy candidate count remains `0`.

The protocol is meaningfully stronger than previous 5-day weekly low-corr variants, but the evidence is not yet enough for `out_of_sample_supported`:

- It was originally found after observing the 2026 horizon/cost grid, so selection bias remains.
- The number of non-overlapping monthly trades is small.
- 2026 evidence is especially thin.
- No same-protocol comparison against rolling IC and equal-rank signals has been completed.
- Capacity is only proxied by same-day amount; no slippage or impact model has been applied.
- Industry, market-cap, and style neutralization are still missing.

## Next

1. Run a same-protocol signal comparison: low-corr vs rolling IC vs equal-rank under `horizon=20/monthly/top_n=200/buffer=3.0/execution_constraints`.
2. Add a simple slippage/impact stress layer tied to participation proxy, separate from fixed fee bps.
3. Add industry/size proxy audit or neutralization before considering promotion.
4. Keep traditional ML paused until the traditional candidate-frontier has passed these gates.
