# Low-Corr Candidate Signal Comparison Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_candidate_signal_comparison_interpretation.md`.


## Question

此前 `multifactor_low_corr_rank_score / horizon=20 / monthly / top_n=200 / buffer=3.0 / baseline_no_filter` 是第一条 candidate-frontier 协议。本次问题是：在完全相同的组合执行协议下，low-corr 是否确实优于 baseline、equal-rank、IC-weighted 和 rolling-IC 信号。

## Run

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.low_corr_candidate_signal_comparison --years 2024,2025,2026 --final-end-date 2026-06-01 --horizon 20 --top-n 200 --frequency monthly --buffer-multiplier 3.0 --fee-bps 0,30,60,100 --execution-constraints --rolling-window 252 --rolling-min-periods 60 --write-research-log
```

- Run ID: `low_corr_candidate_signal_comparison_20260602_171522`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Artifacts: `traditional_quant_research/output/experiments/low_corr_candidate_signal_comparison/low_corr_candidate_signal_comparison_20260602_171522`

## 30 bps Result

| rank | signal | mean annualized | min annualized | positive years | mean Sharpe | worst drawdown | mean turnover | mean delta vs low-corr |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `multifactor_rolling_ic_weighted_score` | `0.350177` | `0.063879` | `3/3` | `2.357263` | `-0.093640` | `1.010595` | `+0.087432` |
| 2 | `multifactor_ic_weighted_score` | `0.280980` | `0.113754` | `3/3` | `2.220153` | `-0.085134` | `1.001458` | `+0.018235` |
| 3 | `multifactor_low_corr_rank_score` | `0.262745` | `0.076594` | `3/3` | `2.144099` | `-0.114807` | `1.125774` | `0.000000` |
| 4 | `multifactor_equal_rank_score` | `0.225184` | `0.007685` | `3/3` | `1.385990` | `-0.122953` | `1.171429` | `-0.037561` |
| 5 | `baseline_score` | `0.154593` | `-0.003412` | `2/3` | `3.314154` | `-0.090777` | `0.748810` | `-0.108152` |

## Key Findings

- Low-corr is not the same-protocol champion. Rolling IC wins by mean annualized return at every fee level from 0 to 100 bps.
- IC-weighted is less spectacular than rolling IC, but has the strongest robustness profile at high cost: at 100 bps it remains positive in all three years, with min annualized return `0.013505`; rolling IC turns negative in the weakest year at `-0.029625`.
- At 30 bps, rolling IC beats low-corr in 2024 and 2025, but low-corr is slightly better in 2026: `0.076594` vs rolling IC `0.063879`.
- Baseline has low turnover and high mean Sharpe, but 2024 is already negative at 30 bps and it has many blocked entries, so it is not a promotion candidate.
- The strong frontier result is therefore not only a low-corr factor-selection effect. The `20d/monthly/top_n=200/buffer=3.0` protocol itself is important, and prior-year IC weighting may be a better signal construction path.
- Rolling IC fallback rate is not zero: `0.132231` in 2024, `0.131959` in 2025, and `0.188791` in 2026. This must be considered before treating rolling IC as a stable trained weighting rule.

## Decision

Do not promote any signal to formal strategy candidate.

Update candidate-frontier set:

- Keep `multifactor_low_corr_rank_score` as a candidate-frontier protocol because it remains robust at 30/60 bps and performed slightly better than rolling IC in 2026.
- Add `multifactor_rolling_ic_weighted_score` as a candidate-frontier signal because it is the same-protocol mean-return leader.
- Add `multifactor_ic_weighted_score` as a candidate-frontier signal because it is the high-cost robustness leader.

Formal strategy candidate count remains `0`.

## Next

1. Add slippage/impact stress based on participation proxy and rerun the three frontier signals.
2. Add industry/size proxy exposure audit for rolling IC and IC-weighted, not only low-corr.
3. Investigate why rolling IC wins 2024/2025 but low-corr edges it in 2026.
4. Keep traditional ML paused until these traditional protocol gates are settled.
