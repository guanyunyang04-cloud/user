# 2026-06-02 Low-Corr Regime Capital Yearly Validation Interpretation

## Context

- Objective: validate dynamic regime-based capital scaling across 2024, 2025, and 2026 using each prior year as the fit window.
- Snapshot: v2 PIT tradeable panel loaded through `dataset_v2.py`.
- Evaluation years: `2024`, `2025`, `2026`, with `2026` ending on `2026-06-01`.
- Stock filter: `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`.
- Backtest protocol: horizon-aligned weekly Top-100, 5 tradeable-day holding period, `buffer_multiplier=2.0`, execution constraints enabled, fees `0/30` bps.
- Artifacts: `traditional_quant_research/output/experiments/low_corr_regime_capital_yearly_validation/low_corr_regime_capital_yearly_validation_20260602_153754`.

## Aggregate Result

Best aggregate plan was still `breadth_soft`:

```text
breadth_20d_positive_rate>=0.50 @ 1.0
breadth_20d_positive_rate>=0.45 @ 0.6
default @ 0.3
```

| plan | fee_bps | mean_ann | min_ann | positive_year_rate | mean_delta_vs_baseline | worst_drawdown | mean_capital_scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| breadth_soft | 0 | 0.056632 | -0.004522 | 0.666667 | 0.006182 | -0.092562 | 0.669206 |
| baseline_full_capital | 0 | 0.050451 | -0.100812 | 0.333333 | 0.000000 | -0.216177 | 1.000000 |
| breadth_soft | 30 | -0.077398 | -0.129649 | 0.000000 | 0.057700 | -0.141405 | 0.669206 |
| baseline_full_capital | 30 | -0.135098 | -0.256237 | 0.333333 | 0.000000 | -0.302099 | 1.000000 |

30 bps yearly rows for `breadth_soft`:

| year | baseline_ann | breadth_soft_ann | delta | breadth_soft_drawdown |
|---:|---:|---:|---:|---:|
| 2024 | -0.256237 | -0.076004 | +0.180233 | -0.141405 |
| 2025 | 0.105040 | -0.026541 | -0.131581 | -0.101466 |
| 2026 | -0.254097 | -0.129649 | +0.124449 | -0.062797 |

## Interpretation

- Dynamic capital scaling improved weak years clearly: 2024 and 2026 both had large positive deltas versus full capital, and worst drawdown improved from `-0.302099` to `-0.141405` at 30 bps.
- It still damages the strong 2025 year. The damage is smaller than the previously tested static all-or-nothing regime skip, but it remains enough to keep the 30 bps aggregate negative.
- At 0 bps, `breadth_soft` slightly beats full capital and improves the minimum year. This suggests the rule has real risk-control value, but not enough alpha density after transaction costs.
- At 30 bps, all dynamic plans have negative mean annualized return. The best plan improves the baseline by about `+0.057700` annualized, but its mean annualized return remains `-0.077398`.

## Decision

- Candidate count remains `0`.
- Dynamic capital scaling is useful as a risk-control diagnostic, not a strategy candidate.
- Stop tuning small static or soft breadth thresholds for now.
- Next research path should prioritize one of:
  - lower-turnover construction under the same low-corr signal,
  - broader traditional factor-family expansion,
  - cost-aware rebalance/horizon variants,
  - formal industry/size/liquidity neutralization.
