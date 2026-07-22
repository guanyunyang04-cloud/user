# Low-Corr Frontier Neutralization Audit Interpretation

## Context

- Run: `low_corr_frontier_neutralization_audit_20260602_183104`
- Snapshot: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- Protocol: `horizon=20 / monthly / top_n=200 / buffer=3.0 / execution_constraints=True / 30 bps`
- Evaluation years: `2024, 2025, 2026`; final end date `2026-06-01`
- Signals: `multifactor_rolling_ic_weighted_score`, `multifactor_ic_weighted_score`, `multifactor_low_corr_rank_score`
- Neutralizer: `log_amount_mean_20d_z`

This is a proxy-neutralization audit, not a full industry/size neutralization. The project does not yet have a true industry classification or market-cap field in the v2 panel, so `log_amount_mean_20d_z` is used as the current size/liquidity proxy.

## Key Results

At `30 bps`, before and after same-date proxy neutralization:

| signal variant | mean annualized | min annualized | positive years | delta vs original | worst max drawdown | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rolling_ic` original | `0.350177` | `0.063879` | `3/3` | `0.000000` | `-0.093640` | `16` |
| `ic_weighted` original | `0.280980` | `0.113754` | `3/3` | `0.000000` | `-0.085134` | `18` |
| `ic_weighted_proxy_neutral` | `0.262882` | `0.058037` | `3/3` | `-0.018098` | `-0.082963` | `17` |
| `low_corr` original | `0.262745` | `0.076594` | `3/3` | `0.000000` | `-0.114807` | `17` |
| `low_corr_proxy_neutral` | `0.219815` | `0.081606` | `3/3` | `-0.042930` | `-0.110949` | `18` |
| `rolling_ic_proxy_neutral` | `0.207259` | `0.020117` | `3/3` | `-0.142918` | `-0.117474` | `18` |

The direct signal-neutralizer correlation check confirms the residualization worked:

- Original signal daily Pearson correlation to `log_amount_mean_20d_z` is strongly negative:
  - 2024: about `-0.49` to `-0.66`
  - 2025: about `-0.43` to `-0.58`
  - 2026: about `-0.64` to `-0.68`
- Proxy-neutral signal correlation to `log_amount_mean_20d_z` is effectively zero, around `1e-15`.

## Interpretation

The proxy-neutralization gate does not reject the frontier set, because all three neutralized variants remain positive in all three evaluation years.

It materially changes the ranking and confidence:

- `rolling_ic` remains the strongest original signal, but loses about `-0.142918` mean annualized return after neutralizing the size/liquidity proxy. This is a large dependency warning.
- `ic_weighted` is the most robust to this gate. Its proxy-neutral version keeps `0.262882` mean annualized return, only about `-0.018098` below the original, and ranks third overall behind the two strongest original variants.
- `low_corr` remains positive after proxy neutralization, but loses about `-0.042930`; it is still useful as a diversified diagnostic, not a leading candidate.

Basket exposure remains nuanced. Although the signal column is neutralized cross-sectionally, the selected Top-N baskets can still show positive or negative active `log_amount_mean_20d_z` because selection is nonlinear and constrained by Top-N/buffer/execution rules. Therefore signal neutralization does not guarantee basket exposure neutrality.

## Decision

- Status: `candidate-frontier/backtest_only`
- Formal strategy candidate count: `0`
- Frontier set remains:
  - `multifactor_rolling_ic_weighted_score`
  - `multifactor_ic_weighted_score`
  - `multifactor_low_corr_rank_score`
- Best robustness signal for next gate: `multifactor_ic_weighted_score_proxy_neutral`

## Next Gate

The next priority should be to introduce true industry and market-cap fields, then rerun this audit with real industry/size neutralization. Until then, proxy-neutral results are informative but insufficient for `out_of_sample_supported` promotion.
