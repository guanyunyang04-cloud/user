# Two-Day KAMA + ATR Strategy Research

- run_id: `two_day_kama_atr_strategy_research_20260607_184940`
- snapshot_id: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`
- candidate_count: `30624`
- confirmed_candidate_count: `2582`
- confirmed_candidate_rate: `0.084313`
- fee_bps: `30.0`
- best_strategy_id: `open_m3_6__all_day3_close__day1_amount_log10__pos1`
- decision: `strategy_research_candidate_diagnostic`
- strategy_candidate_count: `0`

## Strategy Template

- Detect Day1 after close: limit-up, previous close below previous KAMA, and Day1 close above KAMA.
- Try entry on Day2 open only when the open gap passes the selected filter.
- Since A-share T+1 prevents same-day selling, non-confirmed trades are exited at Day3 open in the stricter policies.
- Confirmed trades require Day2 limit-up plus close above ATR upper; confirmed trades are held to Day3 close, with one variant extending near-limit Day3 opens to Day4 close.
- Portfolio scheduler enforces max concurrent holdings of 1 or 2 stocks.

## Ranked Strategies

| strategy_id                                                                   |   trade_count |   confirmed_trade_rate |   mean_net_ret_pct |   median_net_ret_pct |   win_rate |   p10_net_ret_pct |   p90_net_ret_pct |   positive_year_rate |   min_year_mean_net_ret_pct |   worst_trade_pct |
|:------------------------------------------------------------------------------|--------------:|-----------------------:|-------------------:|---------------------:|-----------:|------------------:|------------------:|---------------------:|----------------------------:|------------------:|
| open_m3_6__all_day3_close__day1_amount_log10__pos1                            |          1250 |              0.0528    |         -0.0898904 |            -0.436806 |   0.4616   |          -7.22231 |           7.67128 |             0.545455 |                   -1.2822   |          -42.8    |
| open_0_6__all_day3_close__day1_amount_log10__pos1                             |          1234 |              0.0705024 |         -0.10852   |            -0.434653 |   0.462723 |          -7.49688 |           7.77214 |             0.454545 |                   -0.772275 |          -42.8    |
| fillable_open__all_day3_close__day1_amount_log10__pos2                        |          2498 |              0.0520416 |         -0.181455  |            -0.446429 |   0.465172 |          -7.32523 |           7.55249 |             0.454545 |                   -0.878795 |          -42.8466 |
| open_lt6__all_day3_close__day1_strength_score__pos1                           |          1252 |              0.0646965 |         -0.350498  |            -1.24631  |   0.40016  |          -7.06435 |           7.07867 |             0.454545 |                   -1.06262  |          -30.1314 |
| open_m3_3__all_day3_close__day1_amount_log10__pos1                            |          1244 |              0.0321543 |         -0.285993  |            -0.564733 |   0.458199 |          -7.1028  |           6.82372 |             0.454545 |                   -1.33074  |          -20.9587 |
| open_3_6__all_day3_close__day1_amount_log10__pos1                             |          1043 |              0.152445  |         -0.359907  |            -1.02757  |   0.425695 |          -8.41692 |           9.88499 |             0.454545 |                   -1.84918  |          -42.8    |
| open_0_6__all_day3_close__day1_amount_log10__pos2                             |          2444 |              0.0666939 |         -0.270731  |            -0.661733 |   0.443126 |          -7.3986  |           7.42405 |             0.363636 |                   -1.00796  |          -61.4111 |
| open_lt6__all_day3_close__day1_amount_log10__pos1                             |          1252 |              0.048722  |         -0.219679  |            -0.434653 |   0.460064 |          -7.37167 |           7.37123 |             0.363636 |                   -1.14457  |          -42.8    |
| fillable_open__all_day3_close__day1_amount_log10__pos1                        |          1253 |              0.0582602 |         -0.252025  |            -0.487383 |   0.460495 |          -7.43887 |           7.5474  |             0.363636 |                   -1.17752  |          -42.8    |
| open_lt6__all_day3_close__day1_amount_log10__pos2                             |          2492 |              0.0461477 |         -0.191494  |            -0.454946 |   0.461075 |          -7.26039 |           7.29171 |             0.272727 |                   -0.621747 |          -42.8466 |
| open_m3_6__all_day3_close__day1_amount_log10__pos2                            |          2487 |              0.0518697 |         -0.240303  |            -0.486916 |   0.458384 |          -7.05222 |           7.12921 |             0.272727 |                   -0.727152 |          -61.4111 |
| open_m3_3__all_day3_close__day2_open_gap_pct__pos1                            |          1244 |              0.0618971 |         -0.540762  |            -1.29297  |   0.406752 |          -7.11532 |           6.91608 |             0.272727 |                   -1.14674  |          -21.1461 |
| open_m3_6__all_day3_close__day1_strength_score__pos1                          |          1250 |              0.064     |         -0.424688  |            -1.30006  |   0.3864   |          -6.90689 |           7.07356 |             0.272727 |                   -1.26325  |          -22.413  |
| open_0_6__all_day3_close__day1_strength_score__pos1                           |          1234 |              0.0842788 |         -0.372823  |            -1.23623  |   0.410049 |          -7.13839 |           7.70707 |             0.272727 |                   -1.28667  |          -30.1314 |
| open_m3_3__all_day3_close__day2_open_gap_pct__pos2                            |          2465 |              0.0600406 |         -0.431928  |            -1.26774  |   0.406897 |          -7.05043 |           7.04389 |             0.272727 |                   -1.37261  |          -21.1461 |
| open_3_6__all_day3_close__day1_amount_log10__pos2                             |          1926 |              0.148494  |         -0.59624   |            -1.35823  |   0.411734 |          -8.57071 |           9.40726 |             0.272727 |                   -1.58414  |          -61.4111 |
| open_0_6__confirm_near_extend_nonconfirm_day3_open__day1_strength_score__pos2 |          2433 |              0.079737  |         -0.905414  |            -1.73824  |   0.337032 |          -6.53986 |           5.14558 |             0.272727 |                   -1.6232   |          -62.9667 |
| open_0_6__all_day3_close__day2_open_gap_pct__pos1                             |          1234 |              0.137763  |         -0.774263  |            -1.81922  |   0.388979 |          -8.601   |           9.91374 |             0.272727 |                   -1.87918  |          -61.4111 |
| open_3_6__all_day3_close__day1_strength_score__pos1                           |          1043 |              0.155321  |         -0.50492   |            -1.44865  |   0.405561 |          -8.23242 |           9.50385 |             0.272727 |                   -2.05015  |          -42.8    |
| open_3_6__all_day3_close__day2_open_gap_pct__pos1                             |          1043 |              0.161074  |         -0.744611  |            -1.7375   |   0.390221 |          -8.60812 |           9.80727 |             0.272727 |                   -2.28437  |          -42.8    |

## Best Strategy Yearly

|   year |   trade_count |   confirmed_trade_rate |   mean_net_ret_pct |   median_net_ret_pct |   win_rate |   min_trade_pct |   max_trade_pct |
|-------:|--------------:|-----------------------:|-------------------:|---------------------:|-----------:|----------------:|----------------:|
|   2016 |           116 |             0.0517241  |          0.540656  |            0.216158  |   0.508621 |        -13.8229 |         19.0766 |
|   2017 |           121 |             0.00826446 |         -0.824702  |           -0.934921  |   0.421488 |        -42.8    |         19.4692 |
|   2018 |           120 |             0.0583333  |          0.482196  |           -0.374763  |   0.466667 |        -13.1539 |         21.3978 |
|   2019 |           122 |             0.0655738  |          0.290593  |           -0.476768  |   0.459016 |        -17.1896 |         20.6828 |
|   2020 |           120 |             0.0583333  |          0.190654  |           -0.354634  |   0.458333 |        -10.8263 |         20.3395 |
|   2021 |           122 |             0.0737705  |         -0.482277  |           -0.737742  |   0.434426 |        -14.9154 |         18.6343 |
|   2022 |           121 |             0.0826446  |          0.321792  |           -0.309781  |   0.479339 |        -18.5222 |         22.323  |
|   2023 |           120 |             0.0416667  |         -0.0576131 |            0.460655  |   0.533333 |        -22.199  |         22.0796 |
|   2024 |           119 |             0.0336134  |         -0.426736  |           -0.954206  |   0.436975 |        -20.9587 |         16.9571 |
|   2025 |           122 |             0.0409836  |         -1.2822    |           -1.41833   |   0.401639 |        -20.1582 |         17.1783 |
|   2026 |            47 |             0.0851064  |          0.905112  |            0.0478261 |   0.510638 |        -10.9368 |         15.6176 |

## Best Strategy Trades Sample

| candidate_id       | code      | name_on_date   | entry_date          | exit_date           | entry_filter   | exit_reason   |   confirmed_two_day_breakout |   day2_open_gap_pct |   net_ret_pct | industry                                    |
|:-------------------|:----------|:---------------|:--------------------|:--------------------|:---------------|:--------------|-----------------------------:|--------------------:|--------------:|:--------------------------------------------|
| 20160106_002095.SZ | 002095.SZ | 生意宝         | 2016-01-07 00:00:00 | 2016-01-08 00:00:00 | open_m3_6      | day3_close    |                            0 |           0.0294204 |     -3.53529  | I64互联网和相关服务                         |
| 20160108_601600.SH | 601600.SH | 中国铝业       | 2016-01-11 00:00:00 | 2016-01-12 00:00:00 | open_m3_6      | day3_close    |                            0 |          -0.819672  |     -5.46529  | C32有色金属冶炼和压延加工业                 |
| 20160112_601958.SH | 601958.SH | 金钼股份       | 2016-01-13 00:00:00 | 2016-01-14 00:00:00 | open_m3_6      | day3_close    |                            0 |           1.287     |     -1.69771  | B09有色金属矿采选业                         |
| 20160114_002407.SZ | 002407.SZ | 多氟多         | 2016-01-15 00:00:00 | 2016-01-18 00:00:00 | open_m3_6      | day3_close    |                            0 |          -1.30167   |      9.54058  | C26化学原料和化学制品制造业                 |
| 20160118_002095.SZ | 002095.SZ | 生意宝         | 2016-01-19 00:00:00 | 2016-01-20 00:00:00 | open_m3_6      | day3_close    |                            0 |           4.03066   |     -0.845703 | I64互联网和相关服务                         |
| 20160120_601111.SH | 601111.SH | 中国国航       | 2016-01-21 00:00:00 | 2016-01-22 00:00:00 | open_m3_6      | day3_close    |                            0 |           0.635324  |     -7.11818  | G56航空运输业                               |
| 20160122_600806.SH | 600806.SH | 退市昆机       | 2016-01-25 00:00:00 | 2016-01-26 00:00:00 | open_m3_6      | day3_close    |                            0 |           0         |      2.52528  | C34通用设备制造业                           |
| 20160127_000738.SZ | 000738.SZ | 航发控制       | 2016-01-28 00:00:00 | 2016-01-29 00:00:00 | open_m3_6      | day3_close    |                            0 |          -0.386972  |     -1.07695  | C37铁路、船舶、航空航天和其他运输设备制造业 |
| 20160129_002782.SZ | 002782.SZ | 可立克         | 2016-02-01 00:00:00 | 2016-02-02 00:00:00 | open_m3_6      | day3_close    |                            0 |           3.03327   |      1.09285  | C39计算机、通信和其他电子设备制造业         |
| 20160202_600604.SH | 600604.SH | 市北高新       | 2016-02-03 00:00:00 | 2016-02-04 00:00:00 | open_m3_6      | day3_close    |                            0 |          -1.97926   |      0.917949 | K70房地产业                                 |
| 20160204_600651.SH | 600651.SH | 飞乐音响       | 2016-02-05 00:00:00 | 2016-02-15 00:00:00 | open_m3_6      | day3_close    |                            0 |          -1.82328   |      5.55714  | C38电气机械和器材制造业                     |
| 20160215_000727.SZ | 000727.SZ | 冠捷科技       | 2016-02-16 00:00:00 | 2016-02-17 00:00:00 | open_m3_6      | day3_close    |                            0 |           1.15237   |     -3.8443   | C39计算机、通信和其他电子设备制造业         |
| 20160217_002043.SZ | 002043.SZ | 兔宝宝         | 2016-02-18 00:00:00 | 2016-02-19 00:00:00 | open_m3_6      | day3_close    |                            0 |           5.69106   |     -6.3      | C20木材加工和木、竹、藤、棕、草制品业       |
| 20160219_600680.SH | 600680.SH | *ST上普        | 2016-02-22 00:00:00 | 2016-02-23 00:00:00 | open_m3_6      | day3_close    |                            1 |           1.3676    |     19.0766   | C39计算机、通信和其他电子设备制造业         |
| 20160223_601718.SH | 601718.SH | ST际华         | 2016-02-24 00:00:00 | 2016-02-25 00:00:00 | open_m3_6      | day3_close    |                            0 |           3.07355   |      5.3443   | C18纺织服装、服饰业                         |
| 20160225_600869.SH | 600869.SH | 远东股份       | 2016-02-26 00:00:00 | 2016-02-29 00:00:00 | open_m3_6      | day3_close    |                            0 |          -2.92553   |      2.30274  | C38电气机械和器材制造业                     |
| 20160229_002285.SZ | 002285.SZ | 世联行         | 2016-03-01 00:00:00 | 2016-03-02 00:00:00 | open_m3_6      | day3_close    |                            0 |           2.38854   |     10.8198   | K70房地产业                                 |
| 20160302_001979.SZ | 001979.SZ | 招商蛇口       | 2016-03-03 00:00:00 | 2016-03-04 00:00:00 | open_m3_6      | day3_close    |                            0 |           1.2735    |     -1.97665  | K70房地产业                                 |
| 20160304_000636.SZ | 000636.SZ | 风华高科       | 2016-03-07 00:00:00 | 2016-03-08 00:00:00 | open_m3_6      | day3_close    |                            0 |           0.470588  |     -0.768384 | C39计算机、通信和其他电子设备制造业         |
| 20160308_002741.SZ | 002741.SZ | 光华科技       | 2016-03-09 00:00:00 | 2016-03-10 00:00:00 | open_m3_6      | day3_close    |                            0 |          -2.38977   |     -8.57586  | C26化学原料和化学制品制造业                 |
| 20160311_600237.SH | 600237.SH | 铜峰电子       | 2016-03-14 00:00:00 | 2016-03-15 00:00:00 | open_m3_6      | day3_close    |                            0 |           4.47761   |     -0.3      | C39计算机、通信和其他电子设备制造业         |
| 20160315_600187.SH | 600187.SH | *ST国中        | 2016-03-16 00:00:00 | 2016-03-17 00:00:00 | open_m3_6      | day3_close    |                            0 |           3.9501    |     -2.9      | D46水的生产和供应业                         |
| 20160317_002292.SZ | 002292.SZ | 奥飞娱乐       | 2016-03-18 00:00:00 | 2016-03-21 00:00:00 | open_m3_6      | day3_close    |                            0 |           1.91017   |      3.55005  | C24文教、工美、体育和娱乐用品制造业         |
| 20160321_600565.SH | 600565.SH | ST迪马         | 2016-03-22 00:00:00 | 2016-03-23 00:00:00 | open_m3_6      | day3_close    |                            0 |          -0.503778  |     -3.97089  | K70房地产业                                 |
| 20160323_603128.SH | 603128.SH | 华贸物流       | 2016-03-24 00:00:00 | 2016-03-25 00:00:00 | open_m3_6      | day3_close    |                            0 |           2.29312   |     -0.689864 | G58多式联运和运输代理业                     |
| 20160325_600359.SH | 600359.SH | 新农开发       | 2016-03-28 00:00:00 | 2016-03-29 00:00:00 | open_m3_6      | day3_close    |                            0 |          -0.424628  |      0.126439 | A01农业                                     |
| 20160329_000657.SZ | 000657.SZ | 中钨高新       | 2016-03-30 00:00:00 | 2016-03-31 00:00:00 | open_m3_6      | day3_close    |                            0 |          -0.0613874 |     -0.361425 | C32有色金属冶炼和压延加工业                 |
| 20160331_601018.SH | 601018.SH | 宁波港         | 2016-04-01 00:00:00 | 2016-04-05 00:00:00 | open_m3_6      | day3_close    |                            0 |          -1.01302   |     -2.05439  | G55水上运输业                               |
| 20160405_600570.SH | 600570.SH | 恒生电子       | 2016-04-06 00:00:00 | 2016-04-07 00:00:00 | open_m3_6      | day3_close    |                            0 |           1.96266   |     -8.42207  | I65软件和信息技术服务业                     |
| 20160407_600643.SH | 600643.SH | 爱建集团       | 2016-04-08 00:00:00 | 2016-04-11 00:00:00 | open_m3_6      | day3_close    |                            0 |          -0.0914913 |     -3.41355  | J69其他金融业                               |

## Interpretation Boundary

- This is a strategy-research diagnostic, not an execution recommendation.
- The highest-return profile depends on buying Day2 open before Day2 confirmation is known.
- A real executable version must include order-fill constraints, non-confirmation loss control, position sizing, and live slippage checks.

## Refined Entry Review

The initial coarse grid found no positive-mean strategy after 30 bps cost. A follow-up pass reused the same `day1_candidate_trades.csv` and tested narrower Day2-open filters that use only Day1-close and Day2-open-known fields.

- refined_report: `traditional_quant_research/output/experiments/two_day_kama_atr_strategy_research/two_day_kama_atr_strategy_research_20260607_184940/strategy_recommendation_refined.md`
- best aggressive diagnostic: `gap_le0_second_atr_below_m5__all_day3_close__day1_strength_score__pos2`
- best aggressive metrics: 83 trades, mean net +2.027150%, median +0.127350%, win 53.01%, positive-year rate 63.64%, worst-year mean -5.139923%, worst trade -17.526393%.
- stability-oriented diagnostic: `gap_m5_m3__all_day3_close__day1_amount_log10__pos2`
- stability-oriented metrics: 662 trades, mean net +0.508275%, median +0.146764%, win 51.21%, positive-year rate 72.73%, worst-year mean -0.499762%, worst trade -16.732927%.
- direct confirmed-entry check: buying after Day2 confirmation at Day3 open is negative overall; all confirmed Day3-open entry to 2-day exit averaged -0.584684% net.
- near-limit Day3 opens after confirmation were positive on 2-3 day exits, but this has severe fill and auction feasibility risk.
- conditional carry remains the strongest effect: confirmed subset entered at Day2 open averaged +6.549586% net to Day3 close and +6.924375% net to Day4 close, but this relies on entering before Day2 confirmation is knowable.

Decision: keep all refined rules as `strategy_research_candidate_diagnostic`; do not increment `strategy_candidate_count`.
