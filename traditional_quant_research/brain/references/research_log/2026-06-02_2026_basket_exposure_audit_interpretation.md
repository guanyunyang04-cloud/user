# 2026-06-02 2026 Basket Exposure Audit Interpretation

## Question

After the 2026 monthly/quarterly horizon audit found March and May as the main weak periods, what factor exposures did the actually selected baskets carry?

## Evidence

- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`.
- History warm-up: `2025-01-01`.
- Evaluation window: `2026-01-01` to `2026-06-01`.
- Experiment output: `traditional_quant_research/output/experiments/multifactor_baseline/multifactor_baseline_20260602_130700`.
- Auto research log: `traditional_quant_research/brain/references/research_log/2026-06-02_multifactor_2026_basket_exposure_audit.md`.
- New artifact: `basket_factor_exposure.csv`.

The exposure table groups the actual horizon-aligned Top-N selected codes by exit month/quarter and compares selected-basket factor means with same-date tradeable-universe means. Because these factors are daily cross-sectional z-scores, universe means are approximately `0`; `active_exposure` is therefore the key reading.

## Rolling IC Monthly Exposure

For `multifactor_rolling_ic_weighted_score`, weekly Top-100, buffer `1.0`, 0 bps:

| Period | Mean Net Return | log_amount | momentum | ma20_gap | neg_vol | neg_amp | reversal |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-01 | 0.011740 | -0.9988 | -0.7062 | -0.7795 | 1.0616 | 1.1267 | 0.5822 |
| 2026-02 | 0.003505 | -1.0056 | -0.7014 | -0.6948 | 0.8130 | 0.9066 | 0.5412 |
| 2026-03 | -0.009546 | -1.1916 | -0.6739 | -0.6235 | 0.9062 | 0.9563 | 0.4378 |
| 2026-04 | 0.014858 | -0.9310 | -0.6587 | -0.7276 | 0.8473 | 0.8591 | 0.6785 |
| 2026-05 | -0.013817 | -1.0565 | -0.7422 | -0.7838 | 0.6618 | 0.9868 | 0.5707 |

## Interpretation

The selected baskets consistently tilt toward:

- Smaller/lower-turnover names: `log_amount_mean_20d_z` is around `-1`.
- Weak momentum and negative MA gap after direction handling.
- Defensive low-volatility and low-amplitude names.
- Short-term reversal winners.

March and May do not show a simple one-factor failure, but they do reveal two stress patterns:

- March: liquidity tilt worsened to `-1.1916` and reversal exposure fell to `0.4378`, reducing the short-term reversal edge while keeping a strong small/liquidity-constrained tilt.
- May: momentum and MA-gap weakness deepened, and amplitude defensiveness stayed high while volatility defensiveness fell, suggesting the defensive basket no longer protected returns.

`multifactor_low_corr_rank_score` has a similar exposure profile and also fails in March/May, so the problem is not only rolling-weight estimation. It is likely a regime/exposure issue in the current traditional factor family.

## Decision

Current status remains `candidate_input_watchlist`.

Strategy candidate count remains `0`.

The next gate should be execution constraints, not traditional ML:

- Add limit-up buy and limit-down sell approximations.
- Add suspension holding behavior for positions that cannot exit.
- Re-run March/May slices after execution constraints.
- If constrained results remain weak, add exposure caps or a regime filter before considering model complexity.
