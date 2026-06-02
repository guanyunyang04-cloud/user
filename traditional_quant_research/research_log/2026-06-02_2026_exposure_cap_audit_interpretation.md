# 2026-06-02 2026 Exposure Cap Audit Interpretation

## Question

Can simple exposure caps repair the 2026 execution-constrained failure enough to produce a strategy candidate?

## Evidence

- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- History warm-up: `2025-01-01`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_low_corr_rank_score`.
- Portfolio: weekly Top-100, 5-day horizon, non-overlapping.
- Buffer grid: `1.0, 2.0`.
- Fee grid: `0, 30` bps.
- Execution approximation: `limit_threshold=0.095`.
- Selection filters:
  - `log_amount_mean_20d_z >= -0.8`
  - `momentum_20d_z >= -0.8`
  - `neg_volatility_20d_z <= 1.0`
  - `neg_amplitude_20d_z <= 1.2`
- Experiment output: `traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_140109`.
- Auto research log: `traditional_quant_research/research_log/2026-06-02_multifactor_2026_exposure_cap_audit.md`.

Implementation note: selection filters only make ineligible signal-date rows unselectable. They do not remove future price-path rows, so entry/exit simulation remains on the full tradeable panel. Earlier run `multifactor_baseline_20260602_135636` used filtered price paths and is invalid evidence.

## Selection Impact

The filter kept `175567 / 292483` candidate rows, or `60.03%`, while preserving all `96` evaluation dates.

The low-corr buffer `2.0` monthly active exposures moved to a less extreme profile:

| Period | log_amount | momentum | neg_vol | neg_amp | reversal |
|---|---:|---:|---:|---:|---:|
| 2026-01 | -0.3871 | -0.5110 | 0.6372 | 0.5706 | 0.6709 |
| 2026-02 | -0.4696 | -0.5083 | 0.6051 | 0.5174 | 0.6226 |
| 2026-03 | -0.4948 | -0.4847 | 0.5627 | 0.5457 | 0.6580 |
| 2026-04 | -0.3725 | -0.4504 | 0.5136 | 0.3991 | 0.8180 |
| 2026-05 | -0.4715 | -0.5494 | 0.4780 | 0.6135 | 0.6010 |

This confirms the filter is doing what it is supposed to do: reducing the previous small/liquidity-constrained and defensive extreme.

## Result

| Signal | Buffer | 0 bps AnnRet | 0 bps Sharpe | 30 bps AnnRet | 30 bps Sharpe |
|---|---:|---:|---:|---:|---:|
| `multifactor_low_corr_rank_score` | 1.0 | -0.081884 | -0.344859 | -0.287570 | -1.655595 |
| `multifactor_low_corr_rank_score` | 2.0 | -0.074079 | -0.303403 | -0.262020 | -1.488158 |
| `multifactor_rolling_ic_weighted_score` | 1.0 | -0.267976 | -1.778049 | -0.416903 | -3.129874 |
| `multifactor_rolling_ic_weighted_score` | 2.0 | -0.256090 | -1.670878 | -0.391764 | -2.867157 |

Compared with the unconstrained-exposure buffer grid:

- Low-corr improved meaningfully: buffer `2.0` annualized return improved by about `+0.0543` at 0 bps and `+0.0260` at 30 bps.
- Rolling IC worsened materially: the same filter removed names that were needed by the rolling IC portfolio.
- March and May remain the main negative months.

## Decision

Exposure caps are useful as a risk-control direction, but this first rule does not pass promotion gates.

Current status remains `candidate_input_watchlist`.

Strategy candidate count remains `0`.

Next research step:

- Continue with low-corr, not rolling IC, for exposure-control experiments.
- Test a narrower low-corr-only cap grid around liquidity and defensive exposure thresholds.
- Add a regime filter or market-state gate before considering traditional ML.
