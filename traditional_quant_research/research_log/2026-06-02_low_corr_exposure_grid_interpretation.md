# 2026-06-02 Low-Corr Exposure Grid Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_exposure_grid_interpretation.md`.


## Context

- Objective: test whether first-pass liquidity, momentum, volatility, and amplitude exposure caps improve the current `multifactor_low_corr_rank_score` under stricter fit/evaluation separation.
- Run: `traditional_quant_research/output/experiments/low_corr_exposure_grid/low_corr_exposure_grid_20260602_141729`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Fit window: `2025-01-01` to `2025-12-31`
- Evaluation window: `2026-01-01` to `2026-06-01`
- Protocol: 5-day horizon, weekly Top-100, buffer `2.0`, execution constraints enabled, fees `0/30` bps.

## Tested Rule

```text
log_amount_mean_20d_z>=-0.8,
momentum_20d_z>=-0.8,
neg_volatility_20d_z<=1.0,
neg_amplitude_20d_z<=1.2
```

The rule kept `175567 / 292483` evaluation rows, or `60.03%`, and retained all `96` evaluation dates.

## Results

Baseline low-corr with no exposure filter:

- 0 bps annualized return: `-0.175190`; Sharpe: `-1.063846`
- 30 bps annualized return: `-0.314117`; Sharpe: `-2.183667`
- Mean turnover: `1.214667`
- Exit delayed count: `3`

Exposure-filtered low-corr:

- 0 bps annualized return: `-0.255563`; Sharpe: `-1.612759`
- 30 bps annualized return: `-0.399511`; Sharpe: `-2.851432`
- Mean turnover: `1.410000`
- Blocked entry count: `1`
- Exit delayed count: `2`

## Interpretation

The first-pass exposure cap did not improve the fit/evaluation-separated 2026 low-corr result. It reduced the eligible universe to about 60%, but the selected portfolio became worse than the no-filter baseline at both 0 bps and 30 bps. Turnover also rose, so the rule did not act as a cost-control tool in this stricter protocol.

This is materially different from the earlier same-window exposure-cap audit, where a similar rule improved low-corr but remained negative. The stricter test suggests that same-window improvement is not enough evidence for promotion. Exposure controls should now be treated as a research direction, not as a validated fix.

Candidate count remains `0`.

## Next Step

- Run a compact grid with 2025 fit / 2026 evaluation, not same-window fitting.
- Compare looser liquidity and momentum thresholds before tightening defensive caps.
- Add a regime filter experiment before moving to traditional ML.
- Keep all promotion decisions tied to execution-constrained horizon backtests, not IC or overlapping-label Top-N diagnostics.
