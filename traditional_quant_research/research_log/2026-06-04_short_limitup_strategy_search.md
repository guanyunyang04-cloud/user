# Short Limit-Up Strategy Search

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_short_limitup_strategy_search.md`.


## Scope

- Event file: `traditional_quant_research\output\experiments\all_limitup_event_study\all_limitup_event_study_vector_20260604_192738\all_limitup_events_tplus1_vector.csv`
- Events: 115779
- Date range: 2016-01-04 to 2026-05-28
- Execution: signal on T, buy at T+1 open, earliest managed sell window starts on T+2.
- Daily-bar stop/target handling is conservative: if stop and target both occur in a window, stop wins.

## Decision

- Decision: `shortline_backtest_grid_ready`
- Candidate count: 514
- Ranked rows: 900

## Best Ranked Strategy

- Setup: `signal_one_word_t1_limit_up`
- Sell window: T+2 through T+4 close/high/low window
- Stop/target: 5.0% / 30.0%
- Trades: 11842
- Mean return: 15.31%
- Median return: 21.13%
- Win rate: 69.5%
- Payoff: 5.05
- Positive year rate: 100.0%
- Evidence grade: `shortline_backtest_candidate`

## Top Strategies

| setup_name                      |   sell_window |   stop_loss |   target |     n | mean_ret_pct   | median_ret_pct   | win_rate   |   payoff | positive_year_rate   | evidence_grade               |
|:--------------------------------|--------------:|------------:|---------:|------:|:---------------|:-----------------|:-----------|---------:|:---------------------|:-----------------------------|
| signal_one_word_t1_limit_up     |             3 |           5 |       30 | 11842 | 15.31%         | 21.13%           | 69.5%      |     5.05 | 100.0%               | shortline_backtest_candidate |
| signal_one_word_t1_limit_up     |             3 |           7 |       30 | 11842 | 15.17%         | 21.62%           | 71.6%      |     3.72 | 100.0%               | shortline_backtest_candidate |
| signal_one_word_t1_limit_up     |             3 |          10 |       30 | 11842 | 15.02%         | 22.14%           | 73.7%      |     2.86 | 100.0%               | shortline_backtest_candidate |
| third_plus_t1_limit_up          |             3 |           5 |       30 | 10038 | 14.72%         | 18.08%           | 67.4%      |     5.02 | 100.0%               | shortline_backtest_candidate |
| signal_one_word_t1_limit_up     |             5 |           5 |       30 | 11838 | 14.53%         | 24.73%           | 63.2%      |     5.34 | 100.0%               | shortline_backtest_candidate |
| near_limit_open_signal_one_word |             3 |           5 |       30 | 11394 | 14.54%         | 19.23%           | 65.4%      |     5.11 | 100.0%               | shortline_backtest_candidate |
| third_plus_t1_limit_up          |             3 |           7 |       30 | 10038 | 14.60%         | 19.13%           | 69.7%      |     3.68 | 100.0%               | shortline_backtest_candidate |
| signal_one_word_t1_limit_up     |             5 |           7 |       30 | 11838 | 14.36%         | 30.00%           | 65.5%      |     3.9  | 100.0%               | shortline_backtest_candidate |
| third_plus_t1_limit_up          |             3 |          10 |       30 | 10038 | 14.42%         | 19.91%           | 72.0%      |     2.76 | 100.0%               | shortline_backtest_candidate |
| near_limit_open_signal_one_word |             3 |           7 |       30 | 11394 | 14.25%         | 20.01%           | 67.2%      |     3.72 | 100.0%               | shortline_backtest_candidate |
| near_limit_open_signal_one_word |             5 |           5 |       30 | 11391 | 13.87%         | 20.60%           | 59.9%      |     5.42 | 100.0%               | shortline_backtest_candidate |
| third_plus_t1_limit_up          |             5 |           5 |       30 | 10035 | 13.85%         | 19.25%           | 60.5%      |     5.34 | 100.0%               | shortline_backtest_candidate |

## Best Strategy Yearly Detail

|   year |    n | mean_ret_pct   | median_ret_pct   | win_rate   |
|-------:|-----:|:---------------|:-----------------|:-----------|
|   2016 | 2028 | 21.55%         | 30.00%           | 86.3%      |
|   2017 | 2515 | 21.57%         | 30.00%           | 86.0%      |
|   2018 |  813 | 15.83%         | 20.48%           | 73.9%      |
|   2019 |  993 | 13.40%         | 13.56%           | 67.2%      |
|   2020 | 1205 | 13.78%         | 14.16%           | 64.3%      |
|   2021 |  979 | 12.54%         | 9.14%            | 62.0%      |
|   2022 |  898 | 11.17%         | 8.88%            | 58.2%      |
|   2023 |  420 | 8.02%          | -0.50%           | 49.8%      |
|   2024 |  955 | 7.96%          | -5.00%           | 46.6%      |
|   2025 |  780 | 6.88%          | -4.70%           | 45.8%      |
|   2026 |  256 | 7.38%          | 2.04%            | 52.7%      |

## Interpretation

- This search is a short-line backtest layer, not a live trading guarantee.
- One-word and near-limit-open paths are high-convexity but may be unfilled in reality.
- Tradable candidates should be judged separately from path-strength diagnostics.
- The next upgrade should add minute-level open-board/reseal and orderability evidence.