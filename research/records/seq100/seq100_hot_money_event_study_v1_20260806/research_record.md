# Hot-Money and Breakout Event Study

This is a pre-specified, descriptive study of observable participation shocks and price-location states. It does not identify proprietary main-player orders from OHLCV.

## Mathematical object

At signal close t, define an observable state X_t from scale-normalized turnover, range, recent resistance and intraday path statistics. The study estimates E[R_{t->t+h}|X_t in A] and P(MFE_h >= a, MAE_h >= -b | X_t in A), with entry at the next observed open. Date-equal means and HAC intervals treat a trading date as the sampling cluster.

## Data and execution

- Signal window: 2012-01-01 to 2025-12-31; outcomes stop at 2025-12-31; 2026 is forbidden.
- Daily candidate rows: 1567669; five-minute candidate rows with at least 30 bars: 1513536
- Round-trip cost assumption: 60.0 bp, subtracted from gross arithmetic return.
- Entry uses the next observed open. D1 terminal return is a shadow diagnostic only because a new long cannot be sold on its entry day under T+1. Legal terminal returns are D3/D5/D10/D20; MFE uses sellable path days D2..DH, while MAE includes entry-day exposure. The table reports the calendar gap to the entry open so suspensions are visible rather than silently treated as normal fills.

## Interpretation guardrails

- A participation shock is an abnormal amount/volume observation relative to the stock's preceding 20 observations. It is a proxy for attention or trading pressure, not proof of institutional or游资 ownership.
- Breakout and retest levels are computed only from prior highs/lows. Future extrema appear only in MFE/MAE labels.
- Positive MFE is not a tradable profit claim: the path may reverse before an executable exit, and high volatility mechanically raises MFE.
- A first-hit continuation label also starts at path day 2. Reaching a target on the entry day is not counted as a legal sale under T+1.

## Main results

| event                        | period                |   rows |   mean_gross_return |   mean_net_return |   positive_rate |   mean_mfe |   mean_mae |   hac_lcb_net_date_mean |
|:-----------------------------|:----------------------|-------:|--------------------:|------------------:|----------------:|-----------:|-----------:|------------------------:|
| participation_shock          | development_2012_2020 | 661552 |              0.0036 |           -0.0024 |          0.4798 |     0.0550 |    -0.0498 |                 -0.0086 |
| participation_shock          | validation_2021_2022  | 184174 |              0.0022 |           -0.0038 |          0.4598 |     0.0601 |    -0.0539 |                 -0.0093 |
| participation_shock          | oos_2023_2025         | 273342 |             -0.0000 |           -0.0060 |          0.4390 |     0.0572 |    -0.0522 |                 -0.0117 |
| positive_participation_shock | development_2012_2020 | 223118 |             -0.0004 |           -0.0064 |          0.4614 |     0.0560 |    -0.0552 |                 -0.0140 |
| positive_participation_shock | validation_2021_2022  |  63673 |             -0.0025 |           -0.0085 |          0.4297 |     0.0621 |    -0.0612 |                 -0.0146 |
| positive_participation_shock | oos_2023_2025         |  89200 |             -0.0048 |           -0.0108 |          0.4161 |     0.0615 |    -0.0613 |                 -0.0176 |
| range_expansion              | development_2012_2020 | 384770 |              0.0002 |           -0.0058 |          0.4820 |     0.0534 |    -0.0520 |                 -0.0112 |
| range_expansion              | validation_2021_2022  | 116727 |              0.0014 |           -0.0046 |          0.4688 |     0.0566 |    -0.0523 |                 -0.0127 |
| range_expansion              | oos_2023_2025         | 167230 |             -0.0007 |           -0.0067 |          0.4489 |     0.0547 |    -0.0510 |                 -0.0144 |
| breakout20                   | development_2012_2020 | 267097 |              0.0096 |            0.0036 |          0.4819 |     0.0665 |    -0.0552 |                  0.0002 |
| breakout20                   | validation_2021_2022  |  71242 |             -0.0016 |           -0.0076 |          0.4290 |     0.0630 |    -0.0619 |                 -0.0136 |
| breakout20                   | oos_2023_2025         | 109315 |             -0.0056 |           -0.0116 |          0.4118 |     0.0586 |    -0.0584 |                 -0.0168 |
| breakout60                   | development_2012_2020 | 141021 |              0.0177 |            0.0117 |          0.4871 |     0.0820 |    -0.0620 |                  0.0114 |
| breakout60                   | validation_2021_2022  |  35111 |             -0.0025 |           -0.0085 |          0.4162 |     0.0757 |    -0.0750 |                 -0.0152 |
| breakout60                   | oos_2023_2025         |  55080 |             -0.0119 |           -0.0179 |          0.3809 |     0.0631 |    -0.0727 |                 -0.0192 |
| breakout_confirmed           | development_2012_2020 | 125376 |             -0.0013 |           -0.0073 |          0.4531 |     0.0569 |    -0.0572 |                 -0.0168 |
| breakout_confirmed           | validation_2021_2022  |  35043 |             -0.0053 |           -0.0113 |          0.4128 |     0.0632 |    -0.0659 |                 -0.0186 |
| breakout_confirmed           | oos_2023_2025         |  50594 |             -0.0120 |           -0.0180 |          0.3831 |     0.0602 |    -0.0681 |                 -0.0225 |
| retest20                     | development_2012_2020 |  13277 |              0.0031 |           -0.0029 |          0.4924 |     0.0491 |    -0.0496 |                 -0.0091 |
| retest20                     | validation_2021_2022  |   3230 |             -0.0031 |           -0.0091 |          0.4372 |     0.0439 |    -0.0494 |                 -0.0179 |
| retest20                     | oos_2023_2025         |   8013 |              0.0001 |           -0.0059 |          0.4680 |     0.0346 |    -0.0332 |                 -0.0091 |
| rebreakout20                 | development_2012_2020 |   5788 |              0.0019 |           -0.0041 |          0.4808 |     0.0567 |    -0.0583 |                 -0.0124 |
| rebreakout20                 | validation_2021_2022  |   1307 |             -0.0037 |           -0.0097 |          0.4178 |     0.0540 |    -0.0600 |                 -0.0173 |
| rebreakout20                 | oos_2023_2025         |   3011 |             -0.0033 |           -0.0093 |          0.4334 |     0.0424 |    -0.0463 |                 -0.0136 |
| climax_fade                  | development_2012_2020 |   1846 |              0.0032 |           -0.0028 |          0.4680 |     0.0628 |    -0.0552 |                 -0.0093 |
| climax_fade                  | validation_2021_2022  |    473 |             -0.0025 |           -0.0085 |          0.4799 |     0.0573 |    -0.0584 |                 -0.0211 |
| climax_fade                  | oos_2023_2025         |    756 |              0.0012 |           -0.0048 |          0.4471 |     0.0629 |    -0.0557 |                 -0.0142 |
| minute_trend_confirmed       | development_2012_2020 |  21283 |              0.0019 |           -0.0041 |          0.5088 |     0.0531 |    -0.0499 |                 -0.0111 |
| minute_trend_confirmed       | validation_2021_2022  |   7370 |              0.0012 |           -0.0048 |          0.4845 |     0.0522 |    -0.0509 |                 -0.0146 |
| minute_trend_confirmed       | oos_2023_2025         |  11665 |              0.0037 |           -0.0023 |          0.4823 |     0.0529 |    -0.0420 |                 -0.0124 |
| minute_closing_pressure      | development_2012_2020 | 194606 |             -0.0045 |           -0.0105 |          0.4592 |     0.0484 |    -0.0549 |                 -0.0148 |
| minute_closing_pressure      | validation_2021_2022  |  32281 |             -0.0016 |           -0.0076 |          0.4525 |     0.0509 |    -0.0524 |                 -0.0150 |
| minute_closing_pressure      | oos_2023_2025         |  45308 |             -0.0010 |           -0.0070 |          0.4380 |     0.0543 |    -0.0515 |                 -0.0142 |
| minute_intraday_fade         | development_2012_2020 | 206552 |              0.0056 |           -0.0004 |          0.5011 |     0.0590 |    -0.0513 |                 -0.0080 |
| minute_intraday_fade         | validation_2021_2022  |  66573 |              0.0049 |           -0.0011 |          0.4779 |     0.0635 |    -0.0540 |                 -0.0081 |
| minute_intraday_fade         | oos_2023_2025         |  96893 |              0.0034 |           -0.0026 |          0.4569 |     0.0610 |    -0.0516 |                 -0.0105 |
| hot_confirmed                | development_2012_2020 |   9640 |              0.0009 |           -0.0051 |          0.4963 |     0.0521 |    -0.0495 |                 -0.0130 |
| hot_confirmed                | validation_2021_2022  |   3012 |             -0.0001 |           -0.0061 |          0.4532 |     0.0508 |    -0.0507 |                 -0.0124 |
| hot_confirmed                | oos_2023_2025         |   4935 |              0.0038 |           -0.0022 |          0.4646 |     0.0586 |    -0.0439 |                 -0.0139 |
| breakout_confirmed_intraday  | development_2012_2020 |   5605 |              0.0004 |           -0.0056 |          0.4912 |     0.0533 |    -0.0513 |                 -0.0144 |
| breakout_confirmed_intraday  | validation_2021_2022  |   1561 |             -0.0016 |           -0.0076 |          0.4343 |     0.0510 |    -0.0536 |                 -0.0148 |
| breakout_confirmed_intraday  | oos_2023_2025         |   2516 |             -0.0008 |           -0.0068 |          0.4396 |     0.0615 |    -0.0479 |                 -0.0175 |
| retest_confirmed             | development_2012_2020 |   3501 |              0.0025 |           -0.0035 |          0.4976 |     0.0504 |    -0.0531 |                 -0.0068 |
| retest_confirmed             | validation_2021_2022  |    567 |             -0.0065 |           -0.0125 |          0.4180 |     0.0424 |    -0.0548 |                 -0.0215 |
| retest_confirmed             | oos_2023_2025         |    997 |             -0.0017 |           -0.0077 |          0.4594 |     0.0369 |    -0.0355 |                 -0.0130 |

## Excess versus same-date universe

| event                        | period                |   horizon |   mean_excess_vs_date_universe |   hac_se_excess_date_mean |   hac_lcb_excess_date_mean |
|:-----------------------------|:----------------------|----------:|-------------------------------:|--------------------------:|---------------------------:|
| participation_shock          | development_2012_2020 |         3 |                        -0.0009 |                    0.0004 |                    -0.0015 |
| participation_shock          | development_2012_2020 |         5 |                        -0.0028 |                    0.0006 |                    -0.0042 |
| participation_shock          | development_2012_2020 |        20 |                        -0.0077 |                    0.0013 |                    -0.0116 |
| participation_shock          | validation_2021_2022  |         3 |                         0.0002 |                    0.0005 |                    -0.0009 |
| participation_shock          | validation_2021_2022  |         5 |                        -0.0006 |                    0.0008 |                    -0.0026 |
| participation_shock          | validation_2021_2022  |        20 |                        -0.0032 |                    0.0023 |                    -0.0095 |
| participation_shock          | oos_2023_2025         |         3 |                        -0.0013 |                    0.0005 |                    -0.0024 |
| participation_shock          | oos_2023_2025         |         5 |                        -0.0031 |                    0.0007 |                    -0.0049 |
| participation_shock          | oos_2023_2025         |        20 |                        -0.0095 |                    0.0018 |                    -0.0134 |
| positive_participation_shock | development_2012_2020 |         3 |                        -0.0034 |                    0.0005 |                    -0.0042 |
| positive_participation_shock | development_2012_2020 |         5 |                        -0.0069 |                    0.0006 |                    -0.0096 |
| positive_participation_shock | development_2012_2020 |        20 |                        -0.0133 |                    0.0013 |                    -0.0203 |
| positive_participation_shock | validation_2021_2022  |         3 |                        -0.0024 |                    0.0007 |                    -0.0046 |
| positive_participation_shock | validation_2021_2022  |         5 |                        -0.0049 |                    0.0011 |                    -0.0082 |
| positive_participation_shock | validation_2021_2022  |        20 |                        -0.0099 |                    0.0030 |                    -0.0200 |
| positive_participation_shock | oos_2023_2025         |         3 |                        -0.0042 |                    0.0007 |                    -0.0065 |
| positive_participation_shock | oos_2023_2025         |         5 |                        -0.0071 |                    0.0011 |                    -0.0112 |
| positive_participation_shock | oos_2023_2025         |        20 |                        -0.0158 |                    0.0024 |                    -0.0246 |
| range_expansion              | development_2012_2020 |         3 |                        -0.0015 |                    0.0004 |                    -0.0028 |
| range_expansion              | development_2012_2020 |         5 |                        -0.0040 |                    0.0005 |                    -0.0069 |
| range_expansion              | development_2012_2020 |        20 |                        -0.0079 |                    0.0010 |                    -0.0145 |
| range_expansion              | validation_2021_2022  |         3 |                        -0.0013 |                    0.0005 |                    -0.0034 |
| range_expansion              | validation_2021_2022  |         5 |                        -0.0027 |                    0.0008 |                    -0.0060 |
| range_expansion              | validation_2021_2022  |        20 |                        -0.0054 |                    0.0020 |                    -0.0139 |
| range_expansion              | oos_2023_2025         |         3 |                        -0.0024 |                    0.0005 |                    -0.0047 |
| range_expansion              | oos_2023_2025         |         5 |                        -0.0041 |                    0.0007 |                    -0.0078 |
| range_expansion              | oos_2023_2025         |        20 |                        -0.0098 |                    0.0016 |                    -0.0171 |
| breakout20                   | development_2012_2020 |         3 |                         0.0018 |                    0.0010 |                     0.0023 |
| breakout20                   | development_2012_2020 |         5 |                         0.0027 |                    0.0017 |                     0.0030 |
| breakout20                   | development_2012_2020 |        20 |                         0.0056 |                    0.0041 |                     0.0027 |
| breakout20                   | validation_2021_2022  |         3 |                        -0.0016 |                    0.0011 |                    -0.0046 |
| breakout20                   | validation_2021_2022  |         5 |                        -0.0035 |                    0.0017 |                    -0.0081 |
| breakout20                   | validation_2021_2022  |        20 |                        -0.0070 |                    0.0042 |                    -0.0223 |
| breakout20                   | oos_2023_2025         |         3 |                        -0.0028 |                    0.0010 |                    -0.0065 |
| breakout20                   | oos_2023_2025         |         5 |                        -0.0049 |                    0.0015 |                    -0.0111 |
| breakout20                   | oos_2023_2025         |        20 |                        -0.0115 |                    0.0033 |                    -0.0250 |
| breakout60                   | development_2012_2020 |         3 |                         0.0060 |                    0.0014 |                     0.0091 |
| breakout60                   | development_2012_2020 |         5 |                         0.0098 |                    0.0025 |                     0.0142 |
| breakout60                   | development_2012_2020 |        20 |                         0.0163 |                    0.0059 |                     0.0192 |
| breakout60                   | validation_2021_2022  |         3 |                        -0.0008 |                    0.0015 |                    -0.0049 |
| breakout60                   | validation_2021_2022  |         5 |                        -0.0032 |                    0.0024 |                    -0.0098 |
| breakout60                   | validation_2021_2022  |        20 |                        -0.0126 |                    0.0061 |                    -0.0335 |
| breakout60                   | oos_2023_2025         |         3 |                        -0.0039 |                    0.0014 |                    -0.0076 |
| breakout60                   | oos_2023_2025         |         5 |                        -0.0073 |                    0.0023 |                    -0.0140 |
| breakout60                   | oos_2023_2025         |        20 |                        -0.0165 |                    0.0047 |                    -0.0320 |
| breakout_confirmed           | development_2012_2020 |         3 |                        -0.0045 |                    0.0006 |                    -0.0062 |
| breakout_confirmed           | development_2012_2020 |         5 |                        -0.0085 |                    0.0009 |                    -0.0124 |
| breakout_confirmed           | development_2012_2020 |        20 |                        -0.0151 |                    0.0021 |                    -0.0264 |
| breakout_confirmed           | validation_2021_2022  |         3 |                        -0.0041 |                    0.0012 |                    -0.0078 |
| breakout_confirmed           | validation_2021_2022  |         5 |                        -0.0072 |                    0.0018 |                    -0.0129 |
| breakout_confirmed           | validation_2021_2022  |        20 |                        -0.0121 |                    0.0047 |                    -0.0298 |
| breakout_confirmed           | oos_2023_2025         |         3 |                        -0.0055 |                    0.0013 |                    -0.0102 |
| breakout_confirmed           | oos_2023_2025         |         5 |                        -0.0089 |                    0.0019 |                    -0.0168 |
| breakout_confirmed           | oos_2023_2025         |        20 |                        -0.0181 |                    0.0036 |                    -0.0332 |
| retest20                     | development_2012_2020 |         3 |                        -0.0013 |                    0.0009 |                    -0.0036 |
| retest20                     | development_2012_2020 |         5 |                        -0.0010 |                    0.0014 |                    -0.0055 |
| retest20                     | development_2012_2020 |        20 |                         0.0005 |                    0.0033 |                    -0.0106 |
| retest20                     | validation_2021_2022  |         3 |                        -0.0048 |                    0.0021 |                    -0.0106 |
| retest20                     | validation_2021_2022  |         5 |                        -0.0056 |                    0.0031 |                    -0.0128 |
| retest20                     | validation_2021_2022  |        20 |                        -0.0103 |                    0.0069 |                    -0.0270 |
| retest20                     | oos_2023_2025         |         3 |                         0.0003 |                    0.0016 |                    -0.0032 |
| retest20                     | oos_2023_2025         |         5 |                         0.0005 |                    0.0024 |                    -0.0056 |
| retest20                     | oos_2023_2025         |        20 |                         0.0004 |                    0.0060 |                    -0.0157 |
| rebreakout20                 | development_2012_2020 |         3 |                        -0.0026 |                    0.0014 |                    -0.0054 |
| rebreakout20                 | development_2012_2020 |         5 |                        -0.0057 |                    0.0018 |                    -0.0075 |
| rebreakout20                 | development_2012_2020 |        20 |                        -0.0097 |                    0.0044 |                    -0.0202 |
| rebreakout20                 | validation_2021_2022  |         3 |                        -0.0034 |                    0.0029 |                    -0.0075 |
| rebreakout20                 | validation_2021_2022  |         5 |                        -0.0050 |                    0.0042 |                    -0.0117 |
| rebreakout20                 | validation_2021_2022  |        20 |                        -0.0127 |                    0.0096 |                    -0.0383 |
| rebreakout20                 | oos_2023_2025         |         3 |                         0.0001 |                    0.0024 |                    -0.0062 |
| rebreakout20                 | oos_2023_2025         |         5 |                        -0.0001 |                    0.0036 |                    -0.0098 |
| rebreakout20                 | oos_2023_2025         |        20 |                        -0.0028 |                    0.0077 |                    -0.0228 |
| climax_fade                  | development_2012_2020 |         3 |                         0.0019 |                    0.0017 |                    -0.0003 |
| climax_fade                  | development_2012_2020 |         5 |                        -0.0021 |                    0.0024 |                    -0.0069 |
| climax_fade                  | development_2012_2020 |        20 |                        -0.0140 |                    0.0038 |                    -0.0174 |
| climax_fade                  | validation_2021_2022  |         3 |                        -0.0004 |                    0.0037 |                    -0.0089 |
| climax_fade                  | validation_2021_2022  |         5 |                        -0.0041 |                    0.0045 |                    -0.0151 |
| climax_fade                  | validation_2021_2022  |        20 |                        -0.0061 |                    0.0076 |                    -0.0242 |
| climax_fade                  | oos_2023_2025         |         3 |                        -0.0005 |                    0.0029 |                    -0.0070 |
| climax_fade                  | oos_2023_2025         |         5 |                        -0.0036 |                    0.0036 |                    -0.0100 |
| climax_fade                  | oos_2023_2025         |        20 |                        -0.0024 |                    0.0067 |                    -0.0150 |
| minute_trend_confirmed       | development_2012_2020 |         3 |                        -0.0005 |                    0.0008 |                    -0.0031 |
| minute_trend_confirmed       | development_2012_2020 |         5 |                        -0.0033 |                    0.0011 |                    -0.0081 |
| minute_trend_confirmed       | development_2012_2020 |        20 |                        -0.0085 |                    0.0025 |                    -0.0190 |
| minute_trend_confirmed       | validation_2021_2022  |         3 |                        -0.0013 |                    0.0012 |                    -0.0063 |
| minute_trend_confirmed       | validation_2021_2022  |         5 |                        -0.0035 |                    0.0018 |                    -0.0096 |
| minute_trend_confirmed       | validation_2021_2022  |        20 |                        -0.0084 |                    0.0040 |                    -0.0202 |
| minute_trend_confirmed       | oos_2023_2025         |         3 |                        -0.0023 |                    0.0013 |                    -0.0048 |
| minute_trend_confirmed       | oos_2023_2025         |         5 |                        -0.0039 |                    0.0018 |                    -0.0077 |
| minute_trend_confirmed       | oos_2023_2025         |        20 |                        -0.0137 |                    0.0041 |                    -0.0191 |
| minute_closing_pressure      | development_2012_2020 |         3 |                        -0.0047 |                    0.0004 |                    -0.0064 |
| minute_closing_pressure      | development_2012_2020 |         5 |                        -0.0078 |                    0.0006 |                    -0.0107 |
| minute_closing_pressure      | development_2012_2020 |        20 |                        -0.0137 |                    0.0012 |                    -0.0208 |
| minute_closing_pressure      | validation_2021_2022  |         3 |                        -0.0040 |                    0.0006 |                    -0.0064 |
| minute_closing_pressure      | validation_2021_2022  |         5 |                        -0.0055 |                    0.0007 |                    -0.0083 |
| minute_closing_pressure      | validation_2021_2022  |        20 |                        -0.0072 |                    0.0019 |                    -0.0134 |
| minute_closing_pressure      | oos_2023_2025         |         3 |                        -0.0034 |                    0.0006 |                    -0.0049 |
| minute_closing_pressure      | oos_2023_2025         |         5 |                        -0.0054 |                    0.0008 |                    -0.0078 |
| minute_closing_pressure      | oos_2023_2025         |        20 |                        -0.0111 |                    0.0019 |                    -0.0148 |
| minute_intraday_fade         | development_2012_2020 |         3 |                         0.0004 |                    0.0004 |                    -0.0007 |
| minute_intraday_fade         | development_2012_2020 |         5 |                        -0.0016 |                    0.0005 |                    -0.0033 |
| minute_intraday_fade         | development_2012_2020 |        20 |                        -0.0083 |                    0.0012 |                    -0.0126 |
| minute_intraday_fade         | validation_2021_2022  |         3 |                         0.0020 |                    0.0006 |                     0.0004 |
| minute_intraday_fade         | validation_2021_2022  |         5 |                         0.0015 |                    0.0010 |                    -0.0016 |
| minute_intraday_fade         | validation_2021_2022  |        20 |                        -0.0024 |                    0.0024 |                    -0.0093 |
| minute_intraday_fade         | oos_2023_2025         |         3 |                         0.0006 |                    0.0006 |                    -0.0014 |
| minute_intraday_fade         | oos_2023_2025         |         5 |                        -0.0012 |                    0.0009 |                    -0.0037 |
| minute_intraday_fade         | oos_2023_2025         |        20 |                        -0.0081 |                    0.0024 |                    -0.0133 |
| hot_confirmed                | development_2012_2020 |         3 |                        -0.0018 |                    0.0009 |                    -0.0040 |
| hot_confirmed                | development_2012_2020 |         5 |                        -0.0057 |                    0.0012 |                    -0.0101 |
| hot_confirmed                | development_2012_2020 |        20 |                        -0.0130 |                    0.0027 |                    -0.0237 |
| hot_confirmed                | validation_2021_2022  |         3 |                        -0.0017 |                    0.0017 |                    -0.0055 |
| hot_confirmed                | validation_2021_2022  |         5 |                        -0.0031 |                    0.0021 |                    -0.0074 |
| hot_confirmed                | validation_2021_2022  |        20 |                        -0.0083 |                    0.0047 |                    -0.0156 |
| hot_confirmed                | oos_2023_2025         |         3 |                        -0.0033 |                    0.0015 |                    -0.0050 |
| hot_confirmed                | oos_2023_2025         |         5 |                        -0.0065 |                    0.0021 |                    -0.0091 |
| hot_confirmed                | oos_2023_2025         |        20 |                        -0.0194 |                    0.0047 |                    -0.0203 |
| breakout_confirmed_intraday  | development_2012_2020 |         3 |                        -0.0026 |                    0.0011 |                    -0.0047 |
| breakout_confirmed_intraday  | development_2012_2020 |         5 |                        -0.0065 |                    0.0016 |                    -0.0103 |
| breakout_confirmed_intraday  | development_2012_2020 |        20 |                        -0.0153 |                    0.0031 |                    -0.0258 |
| breakout_confirmed_intraday  | validation_2021_2022  |         3 |                        -0.0016 |                    0.0027 |                    -0.0077 |
| breakout_confirmed_intraday  | validation_2021_2022  |         5 |                        -0.0025 |                    0.0033 |                    -0.0101 |
| breakout_confirmed_intraday  | validation_2021_2022  |        20 |                        -0.0076 |                    0.0070 |                    -0.0203 |
| breakout_confirmed_intraday  | oos_2023_2025         |         3 |                        -0.0039 |                    0.0021 |                    -0.0075 |
| breakout_confirmed_intraday  | oos_2023_2025         |         5 |                        -0.0079 |                    0.0023 |                    -0.0127 |
| breakout_confirmed_intraday  | oos_2023_2025         |        20 |                        -0.0243 |                    0.0054 |                    -0.0277 |
| retest_confirmed             | development_2012_2020 |         3 |                        -0.0020 |                    0.0012 |                    -0.0037 |
| retest_confirmed             | development_2012_2020 |         5 |                        -0.0009 |                    0.0018 |                    -0.0034 |
| retest_confirmed             | development_2012_2020 |        20 |                        -0.0021 |                    0.0045 |                    -0.0110 |
| retest_confirmed             | validation_2021_2022  |         3 |                        -0.0086 |                    0.0029 |                    -0.0130 |
| retest_confirmed             | validation_2021_2022  |         5 |                        -0.0115 |                    0.0037 |                    -0.0180 |
| retest_confirmed             | validation_2021_2022  |        20 |                        -0.0196 |                    0.0090 |                    -0.0367 |
| retest_confirmed             | oos_2023_2025         |         3 |                        -0.0013 |                    0.0021 |                    -0.0047 |
| retest_confirmed             | oos_2023_2025         |         5 |                        -0.0019 |                    0.0029 |                    -0.0094 |
| retest_confirmed             | oos_2023_2025         |        20 |                        -0.0006 |                    0.0062 |                    -0.0149 |

## breakout_location_bins

| period                | bucket     |   rows |   mean_gross_return_1 |   positive_rate_1 |   mean_mfe_1 |   mean_mae_1 |   mean_gross_return_3 |   positive_rate_3 |   mean_mfe_3 |   mean_mae_3 |   mean_gross_return_5 |   positive_rate_5 |   mean_mfe_5 |   mean_mae_5 |   mean_gross_return_10 |   positive_rate_10 |   mean_mfe_10 |   mean_mae_10 |   mean_gross_return_20 |   positive_rate_20 |   mean_mfe_20 |   mean_mae_20 |
|:----------------------|:-----------|-------:|----------------------:|------------------:|-------------:|-------------:|----------------------:|------------------:|-------------:|-------------:|----------------------:|------------------:|-------------:|-------------:|-----------------------:|-------------------:|--------------:|--------------:|-----------------------:|-------------------:|--------------:|--------------:|
| development_2012_2020 | >1.5pct    | 147070 |                   nan |               nan |          nan |      -0.0271 |                0.0098 |            0.4912 |       0.0527 |      -0.0492 |                0.0152 |            0.4767 |       0.0810 |      -0.0630 |                 0.0270 |             0.4788 |        0.1309 |       -0.0864 |                 0.0273 |             0.4681 |        0.1893 |       -0.1205 |
| development_2012_2020 | 0.5-1.5pct |  76467 |                   nan |               nan |          nan |      -0.0199 |                0.0021 |            0.4765 |       0.0340 |      -0.0364 |                0.0026 |            0.4837 |       0.0500 |      -0.0471 |                 0.0072 |             0.5102 |        0.0797 |       -0.0661 |                 0.0128 |             0.5081 |        0.1239 |       -0.0931 |
| development_2012_2020 | 0-0.5pct   |  43692 |                   nan |               nan |          nan |      -0.0182 |                0.0020 |            0.4841 |       0.0314 |      -0.0333 |                0.0031 |            0.4945 |       0.0464 |      -0.0432 |                 0.0076 |             0.5180 |        0.0744 |       -0.0611 |                 0.0132 |             0.5105 |        0.1167 |       -0.0874 |
| validation_2021_2022  | >1.5pct    |  40715 |                   nan |               nan |          nan |      -0.0330 |               -0.0008 |            0.4393 |       0.0497 |      -0.0578 |               -0.0025 |            0.4169 |       0.0736 |      -0.0730 |                -0.0029 |             0.4087 |        0.1131 |       -0.0974 |                -0.0050 |             0.3971 |        0.1613 |       -0.1266 |
| validation_2021_2022  | 0.5-1.5pct |  19403 |                   nan |               nan |          nan |      -0.0219 |               -0.0006 |            0.4386 |       0.0343 |      -0.0379 |               -0.0008 |            0.4397 |       0.0501 |      -0.0489 |                 0.0010 |             0.4573 |        0.0795 |       -0.0677 |                 0.0055 |             0.4608 |        0.1218 |       -0.0924 |
| validation_2021_2022  | 0-0.5pct   |  11126 |                   nan |               nan |          nan |      -0.0195 |                0.0003 |            0.4523 |       0.0315 |      -0.0340 |                0.0005 |            0.4543 |       0.0466 |      -0.0442 |                 0.0023 |             0.4796 |        0.0742 |       -0.0618 |                 0.0072 |             0.4711 |        0.1158 |       -0.0856 |
| oos_2023_2025         | 0-0.5pct   |  19677 |                   nan |               nan |          nan |      -0.0160 |                0.0041 |            0.4661 |       0.0308 |      -0.0275 |                0.0041 |            0.4646 |       0.0452 |      -0.0354 |                 0.0048 |             0.4736 |        0.0681 |       -0.0497 |                 0.0068 |             0.4607 |        0.1019 |       -0.0714 |
| oos_2023_2025         | 0.5-1.5pct |  31260 |                   nan |               nan |          nan |      -0.0187 |                0.0043 |            0.4592 |       0.0357 |      -0.0318 |                0.0031 |            0.4493 |       0.0516 |      -0.0406 |                 0.0041 |             0.4617 |        0.0762 |       -0.0558 |                 0.0081 |             0.4607 |        0.1112 |       -0.0782 |
| oos_2023_2025         | >1.5pct    |  59165 |                   nan |               nan |          nan |      -0.0352 |               -0.0090 |            0.4023 |       0.0455 |      -0.0605 |               -0.0135 |            0.3689 |       0.0667 |      -0.0752 |                -0.0144 |             0.3659 |        0.0979 |       -0.0949 |                -0.0082 |             0.3839 |        0.1416 |       -0.1191 |

## retest_depth_bins

| period                | bucket      |   rows |   mean_gross_return_1 |   positive_rate_1 |   mean_mfe_1 |   mean_mae_1 |   mean_gross_return_3 |   positive_rate_3 |   mean_mfe_3 |   mean_mae_3 |   mean_gross_return_5 |   positive_rate_5 |   mean_mfe_5 |   mean_mae_5 |   mean_gross_return_10 |   positive_rate_10 |   mean_mfe_10 |   mean_mae_10 |   mean_gross_return_20 |   positive_rate_20 |   mean_mfe_20 |   mean_mae_20 |
|:----------------------|:------------|-------:|----------------------:|------------------:|-------------:|-------------:|----------------------:|------------------:|-------------:|-------------:|----------------------:|------------------:|-------------:|-------------:|-----------------------:|-------------------:|--------------:|--------------:|-----------------------:|-------------------:|--------------:|--------------:|
| development_2012_2020 | >1.5pct     |   8765 |                   nan |               nan |          nan |      -0.0225 |                0.0013 |            0.4703 |       0.0359 |      -0.0432 |                0.0026 |            0.4906 |       0.0543 |      -0.0559 |                 0.0065 |             0.5148 |        0.0879 |       -0.0796 |                 0.0067 |             0.5103 |        0.1372 |       -0.1148 |
| development_2012_2020 | 0.5-1.5pct  |   4047 |                   nan |               nan |          nan |      -0.0157 |                0.0016 |            0.4628 |       0.0232 |      -0.0277 |                0.0040 |            0.4981 |       0.0367 |      -0.0351 |                 0.0069 |             0.5226 |        0.0617 |       -0.0524 |                 0.0156 |             0.5325 |        0.1011 |       -0.0773 |
| development_2012_2020 | above_level |    203 |                   nan |               nan |          nan |      -0.0334 |                0.0088 |            0.4631 |       0.0574 |      -0.0594 |                0.0177 |            0.4828 |       0.0918 |      -0.0727 |                 0.0208 |             0.4975 |        0.1368 |       -0.0943 |                 0.0107 |             0.4680 |        0.1830 |       -0.1378 |
| development_2012_2020 | 0-0.5pct    |    272 |                   nan |               nan |          nan |      -0.0188 |               -0.0030 |            0.5000 |       0.0240 |      -0.0354 |               -0.0059 |            0.4559 |       0.0365 |      -0.0453 |                -0.0099 |             0.4963 |        0.0586 |       -0.0708 |                -0.0031 |             0.5000 |        0.0955 |       -0.0936 |
| validation_2021_2022  | 0.5-1.5pct  |   1055 |                   nan |               nan |          nan |      -0.0142 |               -0.0021 |            0.4351 |       0.0204 |      -0.0262 |               -0.0031 |            0.4218 |       0.0305 |      -0.0345 |                -0.0008 |             0.4673 |        0.0520 |       -0.0494 |                 0.0064 |             0.5166 |        0.0851 |       -0.0667 |
| validation_2021_2022  | >1.5pct     |   2029 |                   nan |               nan |          nan |      -0.0242 |               -0.0033 |            0.4248 |       0.0330 |      -0.0430 |               -0.0033 |            0.4455 |       0.0500 |      -0.0562 |                -0.0063 |             0.4386 |        0.0782 |       -0.0798 |                -0.0028 |             0.4470 |        0.1205 |       -0.1073 |
| validation_2021_2022  | above_level |     76 |                   nan |               nan |          nan |      -0.0432 |               -0.0073 |            0.3947 |       0.0402 |      -0.0673 |               -0.0025 |            0.4474 |       0.0720 |      -0.0844 |                 0.0060 |             0.4474 |        0.1281 |       -0.1124 |                 0.0124 |             0.3947 |        0.1893 |       -0.1431 |
| validation_2021_2022  | 0-0.5pct    |     71 |                   nan |               nan |          nan |      -0.0206 |               -0.0010 |            0.4648 |       0.0264 |      -0.0335 |               -0.0003 |            0.4085 |       0.0403 |      -0.0422 |                 0.0014 |             0.4366 |        0.0648 |       -0.0576 |                 0.0162 |             0.4789 |        0.1133 |       -0.0784 |
| oos_2023_2025         | above_level |    135 |                   nan |               nan |          nan |      -0.0335 |               -0.0064 |            0.4667 |       0.0637 |      -0.0559 |               -0.0087 |            0.4000 |       0.0850 |      -0.0730 |                -0.0038 |             0.3778 |        0.1189 |       -0.0956 |                -0.0021 |             0.4148 |        0.1595 |       -0.1238 |
| oos_2023_2025         | >1.5pct     |   3394 |                   nan |               nan |          nan |      -0.0181 |               -0.0003 |            0.4440 |       0.0281 |      -0.0314 |               -0.0020 |            0.4378 |       0.0404 |      -0.0406 |                 0.0020 |             0.4582 |        0.0655 |       -0.0558 |                -0.0055 |             0.4272 |        0.0974 |       -0.0845 |
| oos_2023_2025         | 0.5-1.5pct  |   4120 |                   nan |               nan |          nan |      -0.0118 |                0.0010 |            0.4743 |       0.0187 |      -0.0203 |                0.0015 |            0.4840 |       0.0281 |      -0.0266 |                 0.0046 |             0.4988 |        0.0461 |       -0.0377 |                -0.0018 |             0.4505 |        0.0712 |       -0.0589 |
| oos_2023_2025         | 0-0.5pct    |    403 |                   nan |               nan |          nan |      -0.0114 |                0.0052 |            0.5434 |       0.0237 |      -0.0192 |                0.0055 |            0.5360 |       0.0348 |      -0.0258 |                 0.0085 |             0.5211 |        0.0576 |       -0.0386 |                 0.0015 |             0.4318 |        0.0849 |       -0.0604 |

## Decision rule for the next phase

The event table can establish a conditional distribution and a risk veto, but it cannot by itself prove a profitable long-only policy. A candidate rule is admissible only if its OOS net-return HAC lower bound is positive, its result survives year-by-year checks, and entry gaps/turnover are executable. Otherwise the event remains a descriptive state for a later hurdle-plus-ranking model.
