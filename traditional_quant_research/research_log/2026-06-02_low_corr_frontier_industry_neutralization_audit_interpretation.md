# Low-Corr Frontier Industry Neutralization Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_frontier_industry_neutralization_audit_interpretation.md`.


- Date: 2026-06-02
- Run: `low_corr_frontier_neutralization_audit_20260602_224153`
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_month_start_20260602`
- Protocol: `horizon=20`, `monthly`, `top_n=200`, `buffer=3.0`, `fee_bps=30`, execution constraints enabled.
- Industry source: Baostock `query_stock_industry`, `month-start` forward-filled PIT industry table.

## Result

The frontier signals remain positive after same-date industry demeaning.

| Base signal | Variant | Mean annualized | Worst year | Delta vs original |
|---|---|---:|---:|---:|
| `multifactor_rolling_ic_weighted_score` | industry-neutral | `0.366028` | `0.093565` | `+0.015851` |
| `multifactor_rolling_ic_weighted_score` | original | `0.350177` | `0.063879` | `0.000000` |
| `multifactor_ic_weighted_score` | industry-neutral | `0.302858` | `0.164678` | `+0.021878` |
| `multifactor_low_corr_rank_score` | industry-neutral | `0.284148` | `0.183613` | `+0.021404` |

Proxy-neutralization against `log_amount_mean_20d_z` remains more damaging than industry demeaning, especially for rolling IC:

- rolling IC proxy-neutral mean annualized: `0.207259`, delta `-0.142918`.
- IC-weighted proxy-neutral mean annualized: `0.262882`, delta `-0.018098`.
- low-corr proxy-neutral mean annualized: `0.219815`, delta `-0.042930`.

## Industry Exposure

Signal-level industry demeaning worked mechanically: mean industry active signal for the three industry-neutral variants was approximately `2.7e-17`, while the original signals had mean absolute industry active score around `0.077` to `0.082`.

However, selected baskets are still not portfolio industry-neutral. The Top-N/buffer/execution selection layer can recreate active industry weights even if the signal is industry-demeaned.

Examples from rolling IC:

- Original rolling IC was materially underweight `C39计算机、通信和其他电子设备制造业` across several quarters, with active weights around `-0.08`.
- Industry-neutral rolling IC still showed active industry weights, including 2026Q2 overweight `C39计算机、通信和其他电子设备制造业` around `+0.072337`, `C26化学原料和化学制品制造业` around `+0.058585`, and `C38电气机械和器材制造业` around `+0.058263`.

## Interpretation

This removes one important blocker: the candidate-frontier evidence is not obviously explained by Baostock industry group means. Industry demeaning did not destroy the result, and in this short audit it slightly improved the 30 bps multi-year mean for all three frontier signals.

It does not promote a strategy candidate. The result is still `candidate-frontier/backtest_only` because:

- The industry table is `month-start` forward-filled, not exact daily industry-change tracking.
- The monthly non-overlapping trade sample remains small, especially 2026.
- Signal-level industry demeaning is not the same as portfolio industry neutrality.
- Market cap, float cap, share base and turnover are still missing as true controls.
- Candidate promotion still requires portfolio-level industry/size/capacity controls and stronger sample-out validation.

Strategy candidate count remains `0`.
