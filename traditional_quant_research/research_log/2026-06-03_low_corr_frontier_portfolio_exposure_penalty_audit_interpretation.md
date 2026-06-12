# Low-Corr Frontier Portfolio Exposure Penalty Audit Interpretation

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_low_corr_frontier_portfolio_exposure_penalty_audit_interpretation.md`.


## Summary

This audit adds a portfolio-level exposure penalty to the current frontier protocol. Unlike signal residualization, the penalty acts during Top-N selection: each candidate addition is scored by raw signal minus a penalty for the selected basket's mean exposure across selected style/metrics columns.

Run:

- `run_id`: `low_corr_frontier_neutralization_audit_20260603_081810`
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- Years: `2024,2025,2026`
- Protocol: `horizon=20`, `monthly`, `top_n=200`, `buffer=3.0`, `fee_bps=30`, execution constraints enabled.
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_ic_weighted_score`, `multifactor_low_corr_rank_score`.
- Hard neutralizers for comparison: `log_amount_mean_20d_z`, `momentum_20d_z`, `neg_volatility_20d_z`, `turn_xsec_z`.
- Portfolio exposure penalty columns: `log_amount_mean_20d_z`, `neg_volatility_20d_z`, `momentum_20d_z`, `turn_xsec_z`.
- Portfolio exposure penalty strength: `0.25`.
- Artifact directory: `traditional_quant_research/output/experiments/low_corr_frontier_neutralization_audit/low_corr_frontier_neutralization_audit_20260603_081810`.

Decision:

- No formal strategy candidate is promoted.
- Candidate count remains `0`.
- Portfolio-level exposure penalty support is now usable and tested.
- `strength=0.25` is a useful diagnostic setting, but it is not yet a promotion gate.

## Aggregate Results

At `30 bps`, the exposure-penalized original frontier signals remain strongest:

| signal | variant | mean annualized return | min annualized return | worst max drawdown | mean turnover | total periods |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling IC | original + penalty | `0.353015` | `0.071950` | `-0.092856` | `1.014405` | `16` |
| IC-weighted | original + penalty | `0.277791` | `0.104008` | `-0.085112` | `1.002708` | `18` |
| low-corr | original + penalty | `0.260064` | `0.072976` | `-0.114797` | `1.128810` | `17` |
| IC-weighted | residualized + penalty | `0.179253` | `0.147061` | `-0.155010` | `1.218750` | `17` |
| rolling IC | residualized + penalty | `0.163008` | `0.100373` | `-0.158382` | `1.306488` | `17` |
| low-corr | residualized + penalty | `0.141877` | `0.061917` | `-0.167253` | `1.321310` | `17` |

Compared with the no-penalty run `low_corr_frontier_neutralization_audit_20260603_073116`, the original frontier changes are small:

- Rolling IC mean annualized return: `0.350177` to `0.353015`.
- IC-weighted mean annualized return: `0.280980` to `0.277791`.
- Low-corr mean annualized return: `0.262745` to `0.260064`.

This is encouraging: a small portfolio-level penalty did not break the original frontier. But it also did not solve the main exposure issue.

## Exposure Read

For original non-residualized signals, `strength=0.25` only modestly reduced monthly mean absolute active exposures:

- Rolling IC `log_amount_mean_20d_z`: `1.001891` to `0.998860`.
- Rolling IC `neg_volatility_20d_z`: `0.800373` to `0.796384`.
- Rolling IC `turn_xsec_z`: `0.469745` to `0.467357`.
- IC-weighted `log_amount_mean_20d_z`: `0.986651` to `0.980786`.
- Low-corr `log_amount_mean_20d_z`: `1.019779` to `1.015795`.

For residualized variants, the same penalty produced more visible exposure reduction:

- Rolling IC residualized `log_amount_mean_20d_z`: `0.352700` to `0.281595`.
- Rolling IC residualized `neg_volatility_20d_z`: `0.570413` to `0.489594`.
- Rolling IC residualized `turn_xsec_z`: `0.523184` to `0.466347`.
- IC-weighted residualized `turn_xsec_z`: `0.518614` to `0.432088`.
- Low-corr residualized `log_amount_mean_20d_z`: `0.323317` to `0.250015`.

Interpretation: the portfolio penalty works mechanically, but `0.25` is too weak to materially reduce original signal basket exposures. It is more effective after hard residualization, where the remaining exposure is already lower.

## Research Judgment

This is better than hard full residualization as an infrastructure step because it operates at portfolio construction time and does not destroy the original signal directly. However, this single `0.25` setting does not pass a promotion gate:

- The strongest result is still the original rolling IC frontier.
- Original basket exposure remains high.
- Residualized variants still rank below original variants.
- The experiment is still based on only `16-18` non-overlapping monthly periods, with 2026 contributing only `2` completed periods.

The next useful experiment is a strength grid, not a model upgrade:

- Test `strength=0.0,0.25,0.5,1.0,2.0`.
- Evaluate return, drawdown, turnover, and basket active exposure together.
- Prefer a setting only if it reduces exposure materially without losing the original frontier's annual stability.
- Eventually combine this with industry group cap and external PIT cap/float-cap controls.

Until a grid shows a stable return/exposure tradeoff, this remains `candidate-frontier/backtest_only`.
