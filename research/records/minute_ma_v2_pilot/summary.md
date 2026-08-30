# Minute-MA rule study pilot results

Scope: eight fixed symbols and twelve representative dates across 2022-2024; 2025 was not read. This is an implementation and hypothesis-screening pilot, not a full-universe performance claim.
Entries use the next available minute open. Minute horizons count from the fill bar (therefore 1m is the fill bar close); percentage costs are applied in the JSON record.

| Strategy | Signals | Entry executable | 60m net mean | 60m net median | 60m mean, top 1% winners removed | T+1 net mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s0_distance_only | 375 | 375 | -0.194% | -0.163% | -0.220% | -0.404% |
| s0_liquidity_matched | 4482 | 4482 | 0.220% | -0.064% | 0.174% | 0.495% |
| s0_random_matched | 2304 | 2304 | 0.030% | -0.134% | -0.010% | 0.547% |
| s0_strong_no_ma | 1164 | 1164 | 0.526% | 0.073% | 0.471% | 1.272% |
| s1_break_reclaim_stable3 | 241 | 241 | 0.013% | -0.089% | -0.028% | -0.251% |
| s1_near_reversal | 331 | 331 | -0.125% | -0.163% | -0.163% | -0.326% |
| s1_touch_immediate | 314 | 314 | -0.113% | -0.191% | -0.147% | -0.391% |
| s1_touch_reclaim | 285 | 285 | -0.050% | -0.100% | -0.088% | -0.317% |
| s2_first_touch_reclaim | 9 | 9 | 0.053% | -0.503% | -0.235% | -0.118% |
| s2_reclaim_bull_stack | 1 | 1 | 0.627% | 0.627% | — | 1.481% |
| s2_reclaim_positive_slope | 112 | 112 | 0.023% | -0.098% | -0.026% | -0.221% |
| s4_reclaim_volume_proxy | 238 | 238 | -0.002% | -0.089% | -0.038% | -0.320% |

## Control coverage

Cross-sectional liquidity control references: 5374; matched: 4482; unmatched: 892. Matching is same date/minute/hour/MA period, different symbol, prior-20-session turnover within a 2x band, nearest five candidates, deterministic seed 7.

| Strategy | Control | Matched | Mean strategy minus control | Positive difference fraction |
| --- | --- | ---: | ---: | ---: |
| s0_distance_only | s0_random_matched (same_stock_random_time) | 375 | -0.125% | 44.533% |
| s0_distance_only | s0_liquidity_matched (cross_sectional_prior_turnover) | 295 | -0.087% | 51.525% |
| s0_strong_no_ma | s0_random_matched (same_stock_random_time) | 1164 | 0.155% | 45.275% |
| s0_strong_no_ma | s0_liquidity_matched (cross_sectional_prior_turnover) | 996 | -0.178% | 44.177% |
| s1_break_reclaim_stable3 | s0_random_matched (same_stock_random_time) | 241 | -0.046% | 39.419% |
| s1_break_reclaim_stable3 | s0_liquidity_matched (cross_sectional_prior_turnover) | 192 | -0.121% | 54.167% |
| s1_near_reversal | s0_random_matched (same_stock_random_time) | 331 | -0.107% | 40.181% |
| s1_near_reversal | s0_liquidity_matched (cross_sectional_prior_turnover) | 261 | -0.106% | 50.575% |
| s1_touch_immediate | s0_random_matched (same_stock_random_time) | 314 | -0.047% | 45.541% |
| s1_touch_immediate | s0_liquidity_matched (cross_sectional_prior_turnover) | 251 | -0.110% | 49.402% |
| s1_touch_reclaim | s0_random_matched (same_stock_random_time) | 285 | -0.053% | 41.404% |
| s1_touch_reclaim | s0_liquidity_matched (cross_sectional_prior_turnover) | 228 | -0.121% | 50.000% |
| s2_first_touch_reclaim | s0_random_matched (same_stock_random_time) | 9 | -0.043% | 33.333% |
| s2_first_touch_reclaim | s0_liquidity_matched (cross_sectional_prior_turnover) | 9 | 0.043% | 33.333% |
| s2_reclaim_bull_stack | s0_random_matched (same_stock_random_time) | 1 | -0.033% | 0.000% |
| s2_reclaim_bull_stack | s0_liquidity_matched (cross_sectional_prior_turnover) | 1 | -1.075% | 0.000% |
| s2_reclaim_positive_slope | s0_random_matched (same_stock_random_time) | 112 | -0.064% | 43.750% |
| s2_reclaim_positive_slope | s0_liquidity_matched (cross_sectional_prior_turnover) | 89 | -0.037% | 50.562% |
| s4_reclaim_volume_proxy | s0_random_matched (same_stock_random_time) | 238 | -0.078% | 39.076% |
| s4_reclaim_volume_proxy | s0_liquidity_matched (cross_sectional_prior_turnover) | 192 | -0.062% | 53.646% |

## Interpretation

The corrected record keeps all attempted rule versions, separates timing and cross-sectional controls, and leaves missing or blocked outcomes missing. The small fixed pool is sufficient for implementation checks only; strategy selection requires the declared point-in-time development universe and an inventory/order-state replay before the single frozen 2025 validation.
