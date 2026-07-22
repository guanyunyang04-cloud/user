# Low-Corr Frontier Metrics Neutralization Audit Interpretation

## Summary

This audit tested whether the current `20d/monthly/top_n=200/buffer=3.0` frontier protocol survives hard same-day cross-sectional neutralization against the main observed metrics/style exposures.

Run:

- `run_id`: `low_corr_frontier_neutralization_audit_20260603_073116`
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- Years: `2024,2025,2026`, with each evaluation year using the prior year as the fit window.
- Protocol: `horizon=20`, `monthly`, `top_n=200`, `buffer=3.0`, `fee_bps=30`, execution constraints enabled.
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_ic_weighted_score`, `multifactor_low_corr_rank_score`.
- Neutralizers: `log_amount_mean_20d_z`, `momentum_20d_z`, `neg_volatility_20d_z`, `turn_xsec_z`.
- Metrics included: `True`.
- Artifact directory: `traditional_quant_research/output/experiments/low_corr_frontier_neutralization_audit/low_corr_frontier_neutralization_audit_20260603_073116`.

Decision:

- No formal strategy candidate is promoted.
- Candidate count remains `0`.
- Original frontier signals remain ahead of hard metrics-neutral variants.
- Hard multi-neutralization is useful as a diagnostic gate, but it is too blunt to become the current promotion path.

## Aggregate Results

At `30 bps`, original frontier signals remain stronger:

| signal | variant | mean annualized return | min annualized return | worst max drawdown | mean turnover | delta vs original |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling IC | original | `0.350177` | `0.063879` | `-0.093640` | `1.010595` | `0.000000` |
| rolling IC | metrics-neutral | `0.173130` | `0.092264` | `-0.160550` | `1.306607` | `-0.177047` |
| IC-weighted | original | `0.280980` | `0.113754` | `-0.085134` | `1.001458` | `0.000000` |
| IC-weighted | metrics-neutral | `0.172495` | `0.129108` | `-0.155154` | `1.220595` | `-0.108485` |
| low-corr | original | `0.262745` | `0.076594` | `-0.114807` | `1.125774` | `0.000000` |
| low-corr | metrics-neutral | `0.135335` | `0.049654` | `-0.169559` | `1.322024` | `-0.127409` |

The neutralized variants are still positive across the three evaluated years, but they reduce mean return, worsen drawdown, and increase turnover. This is not a clean promotion result.

## Year-Level Read

The neutralized variants improve weak 2026, but 2026 has only `2` completed monthly periods in this protocol. The stronger evidence is the 2024/2025 degradation:

- Rolling IC drops from `0.472122` to `0.092264` in 2024 and from `0.514530` to `0.189751` in 2025.
- IC-weighted drops from `0.408102` to `0.129108` in 2024 and from `0.321084` to `0.154893` in 2025.
- Low-corr drops from `0.282025` to `0.049654` in 2024 and from `0.429615` to `0.154421` in 2025.

Interpretation: the hard neutralization appears to remove meaningful alpha components alongside unwanted exposure. The 2026 improvement is too thin to justify sacrificing 2024/2025.

## Exposure Read

Signal-level linear exposure was removed successfully:

- Maximum original mean absolute signal-neutralizer correlation: `0.760402`.
- Maximum neutralized mean absolute signal-neutralizer correlation: approximately `3.3e-15`.
- Maximum neutralized absolute correlation across daily checks: approximately `6.1e-14`.

But portfolio-level exposure was only partly reduced. Monthly basket active exposure changed as follows:

- Rolling IC `log_amount_mean_20d_z`: `1.001891` to `0.352700`.
- Rolling IC `neg_volatility_20d_z`: `0.800373` to `0.570413`.
- Rolling IC `turn_xsec_z`: `0.469745` to `0.523184`.
- IC-weighted `log_amount_mean_20d_z`: `0.986651` to `0.474946`.
- IC-weighted `neg_volatility_20d_z`: `0.846279` to `0.569205`.
- IC-weighted `turn_xsec_z`: `0.482477` to `0.518614`.
- Low-corr `log_amount_mean_20d_z`: `1.019779` to `0.323317`.
- Low-corr `neg_volatility_20d_z`: `0.665956` to `0.493123`.
- Low-corr `turn_xsec_z`: `0.438705` to `0.489453`.

Signal residualization is therefore not equivalent to portfolio exposure neutrality. Top-N selection, buffer retention, execution constraints, and unavailable trades can reintroduce active exposure.

## Research Judgment

This audit strengthens the central frontier diagnosis:

- The current alpha evidence is intertwined with liquidity/size proxy, low-volatility, momentum, and turnover structure.
- Hard same-day residualization is too blunt: it removes the measurable linear exposure, but also removes too much useful signal and raises turnover.
- A better next step is portfolio-level exposure control, soft penalties, constrained selection, or optimizer-style construction.
- External PIT market-cap/float-cap data remains important because `log_amount_mean_20d_z` and `turn_xsec_z` are proxies, not true size controls.
- Valuation fields remain diagnostic-only until official publication timing and revision behavior are verified.

## Next Step

The next research step should not be traditional ML. It should be one of:

1. Build a portfolio-level constrained selector for the frontier protocol, with soft or hard limits on `log_amount_mean_20d_z`, `neg_volatility_20d_z`, `momentum_20d_z`, `turn_xsec_z`, and industry.
2. Evaluate a softer exposure penalty path rather than full residualization.
3. Start external PIT cap/float-cap source evaluation so true size neutrality can replace proxy-only gates.

Until that gate passes, frontier remains `candidate-frontier/backtest_only`.
