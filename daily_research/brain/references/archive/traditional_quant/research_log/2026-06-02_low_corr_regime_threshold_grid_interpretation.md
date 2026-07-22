# 2026-06-02 Low-Corr Regime Threshold Grid Interpretation

## Context

- Objective: test whether a static market-return and breadth threshold grid can preserve strong-year performance while improving weak years.
- Run: `traditional_quant_research/output/experiments/low_corr_regime_yearly_validation/low_corr_regime_yearly_validation_20260602_150401`
- Evaluation years: `2024`, `2025`, `2026`
- Fit protocol: each evaluation year uses the prior calendar year as the fit window.
- Stock filter: `log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8`
- Grid:
  - `market_ret_20d_mean >= -0.03/-0.02/-0.01`
  - `breadth_20d_positive_rate >= 0.40/0.45/0.50`
- Protocol: 5-day horizon, weekly Top-100, buffer `2.0`, execution constraints enabled, fees `0/30` bps.

## Best 30 Bps Rule

The best 30 bps rows are equivalent across the three market-return thresholds:

```text
breadth_20d_positive_rate >= 0.50
```

with any of:

```text
market_ret_20d_mean >= -0.03
market_ret_20d_mean >= -0.02
market_ret_20d_mean >= -0.01
```

Aggregate at 30 bps:

- Mean annualized return: `-0.027940`
- Minimum annualized return: `-0.194002`
- Positive-year rate: `0.333333`
- Mean annualized delta vs no-regime baseline: `+0.107158`
- Positive-delta-year rate: `0.666667`
- Worst max drawdown: `-0.107982`

Year details for the best 30 bps rule:

- `2024`: baseline `-0.256237`; regime `0.213586`; delta `+0.469823`
- `2025`: baseline `0.105040`; regime `-0.103404`; delta `-0.208444`
- `2026`: baseline `-0.254097`; regime `-0.194002`; delta `+0.060096`

## Best 0 Bps Rule

The best 0 bps mean annualized return also comes from `breadth_20d_positive_rate >= 0.50`:

- Mean annualized return: `0.187198`
- Minimum annualized return: `-0.020595`
- Positive-year rate: `0.666667`
- Mean annualized delta vs no-regime baseline: `+0.136748`

The more lenient `breadth_20d_positive_rate >= 0.45` rules are positive in all three years at 0 bps, but their 30 bps annualized returns remain negative with mean `-0.095656`.

## Interpretation

The static threshold grid confirms that market breadth is the useful regime dimension in this setup. Market-return thresholds add little once breadth is enforced. Higher breadth thresholds improve weak-year control and drawdown, but they still damage `2025` enough that 30 bps cost robustness is not achieved.

This is a clear upper bound for the current static regime-filter approach:

- It can improve the baseline and reduce drawdown.
- It can rescue `2024` strongly.
- It partially improves `2026`.
- It does not preserve `2025`.
- It does not produce positive 30 bps multi-year mean return.

Candidate count remains `0`.

## Decision

Static weekly regime filters stay in the watchlist as a risk-control idea, not a strategy candidate. The next research move should not be another small threshold tweak. The evidence now points to either:

- dynamic capital scaling instead of all-or-nothing regime skipping, or
- factor-family expansion so the selected basket has alpha in both defensive and strong regimes.

Traditional ML remains premature until one of those paths produces a positive 30 bps, multi-year, execution-constrained candidate.
