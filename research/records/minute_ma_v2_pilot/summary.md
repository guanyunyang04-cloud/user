# Minute-MA rule study pilot results

Scope: eight fixed symbols and twelve representative dates across 2022-2024; 2025 was not read. Returns are percentage-cost event outcomes from the next-minute open. Blank cells mean no observed events. The table below uses the pooled rows in `event_study_pilot.json`, combining all observed MA periods for each strategy; its medians are therefore true pooled medians. The JSON also retains one row per MA period under `summary_overall`.

| Strategy | Signals | Executable | 60m net mean | 60m net median | 60m mean, top 1% winners removed | T+1 net mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s0_random_matched | 2304 | 2304 | 0.032% | -0.146% | -0.009% | 0.547% |
| s0_strong_no_ma | 2076 | 2076 | 0.196% | -0.049% | 0.146% | 0.831% |
| s0_distance_only | 375 | 375 | -0.164% | -0.125% | -0.193% | -0.404% |
| s1_touch_immediate | 314 | 314 | -0.119% | -0.208% | -0.154% | -0.391% |
| s1_touch_reclaim | 285 | 285 | -0.073% | -0.125% | -0.108% | -0.317% |
| s1_near_reversal | 331 | 331 | -0.138% | -0.168% | -0.175% | -0.326% |
| s1_break_reclaim_stable3 | 241 | 241 | 0.019% | -0.124% | -0.020% | -0.251% |
| s2_reclaim_positive_slope | 112 | 112 | -0.012% | -0.014% | -0.053% | -0.221% |
| s2_reclaim_bull_stack | 1 | 1 | 0.604% | 0.604% |  | 1.481% |
| s2_first_touch_reclaim | 9 | 9 | 0.036% | -0.503% | -0.255% | -0.118% |
| s4_reclaim_volume_proxy | 133 | 133 | -0.116% | -0.125% | -0.148% | -0.422% |

## 60-minute net mean by development year

These are pooled by strategy within each year. They are useful for seeing the strong 2024 contribution in this small sample; they are not an out-of-sample claim.

| Strategy | 2022 (n) | 2023 (n) | 2024 (n) |
| --- | ---: | ---: | ---: |
| s0_random_matched | -0.084% (1344) | -0.395% (384) | 0.587% (576) |
| s0_strong_no_ma | 0.016% (1200) | -0.344% (366) | 1.009% (510) |
| s0_distance_only | -0.136% (215) | -0.402% (109) | 0.232% (51) |
| s1_touch_immediate | -0.073% (188) | -0.385% (84) | 0.204% (42) |
| s1_touch_reclaim | -0.009% (171) | -0.393% (75) | 0.263% (39) |
| s1_near_reversal | -0.107% (195) | -0.369% (91) | 0.199% (45) |
| s1_break_reclaim_stable3 | 0.109% (146) | -0.362% (62) | 0.340% (33) |
| s2_reclaim_positive_slope | -0.001% (65) | -0.352% (31) | 0.608% (16) |
| s2_reclaim_bull_stack | 0.604% (1) | — | — |
| s2_first_touch_reclaim | 0.257% (8) | — | -1.735% (1) |
| s4_reclaim_volume_proxy | -0.139% (82) | -0.256% (29) | 0.151% (22) |

## Paired 60-minute comparison against the random control

| Strategy | Matched | Mean strategy minus control | Positive difference fraction |
| --- | ---: | ---: | ---: |
| s0_strong_no_ma | 2076 | 0.052% | 49.181% |
| s0_distance_only | 375 | -0.105% | 45.867% |
| s1_touch_immediate | 314 | -0.068% | 46.178% |
| s1_touch_reclaim | 285 | -0.091% | 42.456% |
| s1_near_reversal | 331 | -0.130% | 40.483% |
| s1_break_reclaim_stable3 | 241 | -0.048% | 40.664% |
| s2_reclaim_positive_slope | 112 | -0.114% | 41.964% |
| s2_reclaim_bull_stack | 1 | -0.085% | 0.000% |
| s2_first_touch_reclaim | 9 | -0.059% | 33.333% |
| s4_reclaim_volume_proxy | 133 | -0.287% | 35.338% |

Interpretation: this pilot is a code and hypothesis-screening result, not a full-universe profitability claim. It does not select a winner: 2023 is uniformly weak and the small sample's 2024 contribution is visibly large. Same-day MFE/MAE starts at the entry minute and is only marked observed for a complete contiguous day. The posthoc catch-up rows are retained separately and excluded from the executable summary. A broader development run with a declared point-in-time universe, turnover, overlap, capacity and drawdown is still required before any strategy freeze.
