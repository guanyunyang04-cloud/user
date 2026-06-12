# v2.1 Metrics Semantics and Timing Audit

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-03_v2_metrics_semantics_audit.md`.


- run_id: `v2_metrics_semantics_audit_20260603_061922`
- snapshot_id: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- status: `metrics_semantics_timing_audit`
- tradeable_panel_rows: `7031085`
- pctchg_compare_rows: `7027693`
- pctchg_abs_diff_p95: `0.000047`
- pctchg_close_return_aligned: `True`
- valuation_missingness_ready: `True`
- valuation_pit_timing_ready: `False`
- candidate_count: `0`

## pctChg vs close-to-close

| scope     |   compare_rows |   abs_diff_mean |   abs_diff_median |   abs_diff_p95 |   abs_diff_p99 |   abs_diff_max |   within_1bp_rate |   within_5bp_rate |   gt_100bp_rows |
|:----------|---------------:|----------------:|------------------:|---------------:|---------------:|---------------:|------------------:|------------------:|----------------:|
| tradeable |        7027693 |        0.019885 |           1.5e-05 |        4.7e-05 |          5e-05 |         462.92 |          0.996651 |          0.996659 |           14179 |

## pctChg Yearly

|   year |   compare_rows |   abs_diff_mean |   abs_diff_median |   abs_diff_p95 |   abs_diff_p99 |   abs_diff_max |   within_1bp_rate |   within_5bp_rate |   gt_100bp_rows |
|-------:|---------------:|----------------:|------------------:|---------------:|---------------:|---------------:|------------------:|------------------:|----------------:|
|   2016 |         515653 |        0.03833  |           1.1e-05 |        4.7e-05 |        5.3e-05 |        81.0384 |          0.996771 |          0.996798 |             819 |
|   2017 |         584074 |        0.027262 |           2e-06   |        7e-06   |        9e-06   |       224.964  |          0.996562 |          0.996571 |             937 |
|   2018 |         631044 |        0.025457 |           2e-06   |        7e-06   |        9e-06   |       116.284  |          0.996387 |          0.996392 |            1351 |
|   2019 |         667798 |        0.020069 |           4e-06   |        4.4e-05 |        5.1e-05 |        89.1405 |          0.996804 |          0.996807 |            1354 |
|   2020 |         669808 |        0.016366 |           2.4e-05 |        4.8e-05 |        5.2e-05 |        81.3374 |          0.996818 |          0.996826 |            1274 |
|   2021 |         706336 |        0.022715 |           2.4e-05 |        4.8e-05 |        5e-05   |       346.215  |          0.996728 |          0.996732 |            1495 |
|   2022 |         728669 |        0.01913  |           2.4e-05 |        4.8e-05 |        5e-05   |       396.56   |          0.996889 |          0.996896 |            1526 |
|   2023 |         744491 |        0.014155 |           2.4e-05 |        4.8e-05 |        5e-05   |       184.693  |          0.99696  |          0.996963 |            1452 |
|   2024 |         745029 |        0.013004 |           2.4e-05 |        4.8e-05 |        5e-05   |       145.691  |          0.996195 |          0.9962   |            1905 |
|   2025 |         742322 |        0.012641 |           2.4e-05 |        4.8e-05 |        5e-05   |       462.92   |          0.995994 |          0.996008 |            1691 |
|   2026 |         292469 |        0.013798 |           2.4e-05 |        4.8e-05 |        5e-05   |       324.411  |          0.997726 |          0.997733 |             375 |

## Valuation Fields

| field     |    rows |   non_null_rate |   finite_rate |   zero_rate |   negative_rate |   change_rate |   corr_with_close_ret |          p01 |      p50 |       p99 |               min |              max |
|:----------|--------:|----------------:|--------------:|------------:|----------------:|--------------:|----------------------:|-------------:|---------:|----------:|------------------:|-----------------:|
| peTTM     | 7031085 |               1 |             1 |    0        |        0.158472 |      0.967922 |              0.003832 |  -520.498    | 26.0566  |  791.483  | -474317           |      3.05844e+06 |
| pbMRQ     | 7031085 |               1 |             1 |    2.2e-05  |        0.001535 |      0.967856 |              0.206463 |     0.536941 |  2.33862 |   22.9567 |   -1581.99        |  19077.3         |
| psTTM     | 7031085 |               1 |             1 |    0.00014  |        0.000214 |      0.967768 |              0.083813 |     0.164163 |  2.45193 |   35.9816 |    -923.7         | 122165           |
| pcfNcfTTM | 7031085 |               1 |             1 |    0.000443 |        0.448057 |      0.967899 |              0.00012  | -1477.41     |  7.76321 | 1499.05   |      -2.72991e+07 |      3.70757e+07 |

## Interpretation

The audit checks field semantics after the missingness audit. `pctChg` is compared with v2 close-to-close returns to detect source or adjustment differences. Valuation fields are treated as coverage-ready but not PIT-timing-ready: they should be lagged by at least one trading day until publication timing and revision behavior are independently verified. No strategy candidate is promoted by this audit.
