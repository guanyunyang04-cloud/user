# First-Hit Path Audit

For each event at close t, entry is the next observed open (path day 1). Under A-share T+1, first-hit targets and adverse exits are inspected only from path day 2 through day 20; entry-day adversity is not treated as an executable stop. A positive path label requires the upside threshold to be reached before the adverse threshold; this is stricter than looking at an unordered MFE/MAE envelope.

Thresholds: +5% before -3%, and +10% before -5%. No future state is used for event selection. 2026 is excluded.

| event                        |   year |   rows |   hit_rate |   date_equal_hit_rate |   hac_lcb_date_equal |   median_up10_day |
|:-----------------------------|-------:|-------:|-----------:|----------------------:|---------------------:|------------------:|
| participation_shock          |   2023 |  86137 |     0.2571 |                0.2571 |               0.2300 |            6.0000 |
| positive_participation_shock |   2023 |  26127 |     0.2611 |                0.2617 |               0.2399 |            6.0000 |
| range_expansion              |   2023 |  52443 |     0.2489 |                0.2497 |               0.2270 |            7.0000 |
| breakout20                   |   2023 |  31421 |     0.2508 |                0.2598 |               0.2361 |            6.0000 |
| breakout60                   |   2023 |  13989 |     0.2651 |                0.2823 |               0.2596 |            5.0000 |
| breakout_confirmed           |   2023 |  14662 |     0.2576 |                0.2654 |               0.2446 |            5.0000 |
| retest20                     |   2023 |   2653 |     0.1885 |                0.1921 |               0.1519 |           10.0000 |
| rebreakout20                 |   2023 |   1050 |     0.2305 |                0.2737 |               0.2035 |            8.0000 |
| climax_fade                  |   2023 |    233 |     0.2961 |                0.2851 |               0.2124 |            6.0000 |
| participation_shock          |   2024 |  97467 |     0.3534 |                0.3180 |               0.2538 |            5.0000 |
| positive_participation_shock |   2024 |  33797 |     0.3610 |                0.3117 |               0.2576 |            4.0000 |
| range_expansion              |   2024 |  59465 |     0.3436 |                0.3020 |               0.2409 |            5.0000 |
| breakout20                   |   2024 |  38716 |     0.3339 |                0.2962 |               0.2455 |            3.0000 |
| breakout60                   |   2024 |  20536 |     0.2930 |                0.3022 |               0.2589 |            3.0000 |
| breakout_confirmed           |   2024 |  18555 |     0.3330 |                0.2958 |               0.2480 |            3.0000 |
| retest20                     |   2024 |   2320 |     0.2246 |                0.2774 |               0.2028 |            9.0000 |
| rebreakout20                 |   2024 |    795 |     0.2176 |                0.2283 |               0.1687 |            6.0000 |
| climax_fade                  |   2024 |    260 |     0.3385 |                0.3436 |               0.2339 |            7.0000 |
| participation_shock          |   2025 |  91500 |     0.3226 |                0.3254 |               0.2985 |            6.0000 |
| positive_participation_shock |   2025 |  29854 |     0.3199 |                0.3244 |               0.2994 |            6.0000 |
| range_expansion              |   2025 |  56280 |     0.3243 |                0.3179 |               0.2905 |            7.0000 |
| breakout20                   |   2025 |  39965 |     0.3105 |                0.3193 |               0.2957 |            6.0000 |
| breakout60                   |   2025 |  20877 |     0.3328 |                0.3405 |               0.3181 |            4.0000 |
| breakout_confirmed           |   2025 |  17755 |     0.3128 |                0.3199 |               0.2963 |            5.0000 |
| retest20                     |   2025 |   3079 |     0.2283 |                0.2603 |               0.2130 |            8.0000 |
| rebreakout20                 |   2025 |   1182 |     0.2826 |                0.2832 |               0.2254 |            6.0000 |
| climax_fade                  |   2025 |    266 |     0.3158 |                0.2641 |               0.1947 |            5.0000 |

The path probabilities measure opportunity conditional on a filled entry, not guaranteed executable profit: a strategy must still choose an exit before the adverse threshold and pay costs.
