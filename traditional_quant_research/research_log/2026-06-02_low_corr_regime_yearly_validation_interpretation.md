# 2026-06-02 Low-Corr Regime Yearly Validation Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_regime_yearly_validation_interpretation.md`.


## Context

- Objective: validate whether the current best low-corr regime rule is stable across multiple evaluation years.
- Run: `traditional_quant_research/output/experiments/low_corr_regime_yearly_validation/low_corr_regime_yearly_validation_20260602_145716`
- Evaluation years: `2024`, `2025`, `2026`
- Fit protocol: each evaluation year uses the prior calendar year as the fit window.
- Stock filter: `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`
- Regime rule: `market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45`
- Protocol: 5-day horizon, weekly Top-100, buffer `2.0`, execution constraints enabled, fees `0/30` bps.

## Aggregate Results

No-regime baseline:

- 0 bps mean annualized return: `0.050451`
- 0 bps positive-year rate: `0.333333`
- 30 bps mean annualized return: `-0.135098`
- 30 bps positive-year rate: `0.333333`
- 30 bps worst max drawdown: `-0.302099`

Regime-filtered:

- 0 bps mean annualized return: `0.102988`
- 0 bps positive-year rate: `1.000000`
- 30 bps mean annualized return: `-0.095656`
- 30 bps positive-year rate: `0.333333`
- 30 bps worst max drawdown: `-0.116810`

## Yearly Details

At 30 bps:

- `2024`: baseline `-0.256237`; regime `0.001956`; delta `+0.258193`
- `2025`: baseline `0.105040`; regime `-0.124281`; delta `-0.229321`
- `2026`: baseline `-0.254097`; regime `-0.164644`; delta `+0.089454`

The rule allowed:

- `2024`: `25 / 52` scheduled rebalances
- `2025`: `34 / 53` scheduled rebalances
- `2026`: `11 / 21` scheduled rebalances

## Interpretation

The regime rule is useful as drawdown and bad-year control, but it is not a robust strategy-candidate gate. It helps materially in weak years (`2024`, `2026`) and improves the multi-year worst drawdown, but it damages the strong `2025` baseline enough that 30 bps mean annualized return remains negative.

The 0 bps evidence is stronger: the regime rule is positive in all three evaluation years and improves mean annualized return. But the requested promotion gate requires cost robustness, and that gate is not met.

This suggests the next problem is not simply finding a stricter static regime threshold. A static rule can avoid bad markets, but may also skip profitable recovery or trend-continuation periods. The next experiment should compare a threshold grid and inspect whether any rule preserves `2025` while still improving `2024/2026`.

Candidate count remains `0`.

## Next Step

- Run the compact threshold grid:
  - `market_ret_20d_mean >= -0.03/-0.02/-0.01`
  - `breadth_20d_positive_rate >= 0.40/0.45/0.50`
- Promote only if 30 bps mean annualized return becomes positive, positive-year rate improves, and no single year has unacceptable drawdown.
- Keep this line in `watchlist`, not candidate.
