# 2026-06-02 V2 Data Label Audit

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-02_v2_data_label_audit.md`.


- Data Scope: snapshot `baostock_v2_pit_20160101_20260601_stockbasic_fixed`, `2016-01-04` to `2026-06-01`.
- Universe: `7451610` rows, `2526` dates, `3393` securities.
- Tradeable Panel: `7031085` rows, `2526` dates, `3392` securities.
- Bar Quality Flags: `{'volume_null_rows': 10922, 'amount_null_rows': 10922, 'zero_volume_rows': 136416, 'zero_amount_rows': 136418}`.
- Label Convention: Signals at t use the next tradeable open as entry and the horizon-th next tradeable close as exit per code.
- Key Label Flags: `fwd_ret_1d` available `0.999518`, abs>0.2 rows `223`, abs>0.5 rows `12`, `fwd_ret_5d` available `0.997588`, abs>0.2 rows `115523`, abs>0.5 rows `4578`, `fwd_ret_20d` available `0.990360`, abs>0.2 rows `621066`, abs>0.5 rows `57163`
- Assessment: 审计用于识别数据/标签风险，不构成因子有效性结论。
- Next Step: 对极端标签样本做复权/除权核查，并在全周期单因子诊断中按年份引用本审计结果。

## Label Summary

| label       |    rows |   available_rows |   missing_rows |   available_rate |     mean |      std |       min |       p01 |       p50 |      p99 |      max |   abs_gt_0.1_rows |   abs_gt_0.1_rate |   abs_gt_0.2_rows |   abs_gt_0.2_rate |   abs_gt_0.5_rows |   abs_gt_0.5_rate |
|:------------|--------:|-----------------:|---------------:|-----------------:|---------:|---------:|----------:|----------:|----------:|---------:|---------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|
| fwd_ret_1d  | 7031085 |          7027693 |           3392 |         0.999518 | 0.001097 | 0.025726 | -0.808824 | -0.067347 |  0        | 0.088608 | 0.652275 |             45403 |          0.006457 |               223 |          3.2e-05  |                12 |          2e-06    |
| fwd_ret_5d  | 7031085 |          7014125 |          16960 |         0.997588 | 0.002075 | 0.067733 | -0.972656 | -0.162755 | -0.000903 | 0.211297 | 4.47594  |            667219 |          0.094896 |            115523 |          0.01643  |              4578 |          0.000651 |
| fwd_ret_20d | 7031085 |          6963302 |          67783 |         0.99036  | 0.005509 | 0.139199 | -0.982188 | -0.283324 | -0.005429 | 0.431511 | 6.7589   |           2189403 |          0.311389 |            621066 |          0.088331 |             57163 |          0.00813  |

## Bar Quality

|    rows |   dates |   securities |   open_null_rows |   high_null_rows |   low_null_rows |   close_null_rows |   volume_null_rows |   amount_null_rows |   nonpositive_open_rows |   nonpositive_high_rows |   nonpositive_low_rows |   nonpositive_close_rows |   negative_volume_rows |   negative_amount_rows |   zero_volume_rows |   zero_amount_rows |   high_below_low_rows |   high_below_open_or_close_rows |   low_above_open_or_close_rows |
|--------:|--------:|-------------:|-----------------:|-----------------:|----------------:|------------------:|-------------------:|-------------------:|------------------------:|------------------------:|-----------------------:|-------------------------:|-----------------------:|-----------------------:|-------------------:|-------------------:|----------------------:|--------------------------------:|-------------------------------:|
| 7451610 |    2526 |         3393 |                0 |                0 |               0 |                 0 |              10922 |              10922 |                       0 |                       0 |                      0 |                        0 |                      0 |                      0 |             136416 |             136418 |                     0 |                               0 |                              0 |

## Reject Reasons

| reject_reason     |    rows |
|:------------------|--------:|
| ok                | 7031085 |
| st_on_date        |  298861 |
| suspended_on_date |  121664 |

## Universe By Year

|   year |   rows |   dates |   securities |   tradeable_rows |   st_rows |   suspended_rows |   missing_bar_rows |
|-------:|-------:|--------:|-------------:|-----------------:|----------:|-----------------:|-------------------:|
|   2016 | 577686 |     244 |         2467 |           518103 |     15735 |            48732 |                  0 |
|   2017 | 640611 |     244 |         2763 |           584379 |     17295 |            43735 |                  0 |
|   2018 | 680626 |     243 |         2835 |           631122 |     19948 |            32859 |                  0 |
|   2019 | 699798 |     244 |         2910 |           667879 |     29991 |             4365 |                  0 |
|   2020 | 717104 |     243 |         3043 |           669952 |     45573 |             4977 |                  0 |
|   2021 | 750932 |     243 |         3155 |           706460 |     43310 |             3039 |                  0 |
|   2022 | 762466 |     242 |         3207 |           728743 |     32495 |             2383 |                  0 |
|   2023 | 771488 |     242 |         3231 |           744550 |     26246 |             1810 |                  0 |
|   2024 | 771470 |     242 |         3221 |           745054 |     25424 |             2473 |                  0 |
|   2025 | 772799 |     243 |         3213 |           742360 |     29171 |             2078 |                  0 |
|   2026 | 306630 |      96 |         3203 |           292483 |     13673 |              893 |                  0 |

Artifacts: `traditional_quant_research\output\experiments\v2_data_label_audit\v2_data_label_audit_20260602_092601`
