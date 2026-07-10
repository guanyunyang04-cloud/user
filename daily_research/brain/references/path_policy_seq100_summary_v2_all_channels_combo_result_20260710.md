# Path Policy Seq100 All-Channel Combo Result

Date: `2026-07-10`

## Verdict
- Status: `completed / all_channel_summary_v2_combo / no_default_change / no_active_execution_change`.
- Main evaluation policy remains `2024 validation` + `2025 train-through-2024 forward test`.
- Result: none of the five all-channel `summary_v2` combinations replaces the current default research profile `seq100_todayclose_path_only/daily_only_summary_v2_ohlcva_aux_low`.
- Reason: the current default keeps the strongest paired IC anchors: 2024 validation IC `0.1626` and 2025 forward IC `0.1927`. The all-channel anchor `summary_v2_all_channels` remains the strongest narrow forward Top1 branch (`27.30%`), but its forward IC is only `0.1417`.
- Active artifact impact: unchanged. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

## Artifacts
- Focused comparison CSV: `daily_research/output/path_policy/sequence_path_training/summary_v2_all_channels_combo_val2024_forward2025_comparison_20260710.csv`
- Focused comparison JSON: `daily_research/output/path_policy/sequence_path_training/summary_v2_all_channels_combo_val2024_forward2025_comparison_20260710.json`
- Runner script: `daily_research/output/path_policy/sequence_path_training/run_summary_v2_all_channels_combo_20260709.ps1`
- Runner log: `daily_research/output/path_policy/sequence_path_training/summary_v2_all_channels_combo_20260709_run.log`
- Roll-forward view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json`

## Results
All rows use 60d `path_trade_value_v2_60d`. TopK columns are path-value spread.

| profile | 2024 val IC | 2024 val Top1 | 2024 val Top3 | 2024 val Top10 | 2025 fwd IC | 2025 fwd Top1 | 2025 fwd Top3 | 2025 fwd Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_default_ohlcva_aux_low | 0.1626 | 19.31% | 17.02% | 13.66% | 0.1927 | 22.51% | 7.73% | 6.72% |
| summary_v2_all_channels | 0.1609 | 26.27% | 23.06% | 16.15% | 0.1417 | 27.30% | 15.25% | 8.28% |
| summary_v2_price_delta | 0.1555 | 16.25% | 14.27% | 13.47% | 0.1348 | 7.95% | 9.00% | 6.88% |
| summary_v2_ohlcva_aux | 0.1512 | 26.17% | 21.05% | 15.54% | 0.1464 | 14.47% | 10.53% | 7.57% |
| summary_v2_ohlcva_aux_low | 0.1435 | 23.56% | 18.00% | 13.50% | 0.1625 | 7.16% | 5.82% | 7.43% |
| summary_v2_ohlcva_aux_low_price_delta | 0.1391 | 20.84% | 16.81% | 14.15% | 0.1262 | 18.37% | 8.72% | 5.64% |
| summary_v2_ohlcva_path_equal | 0.1408 | 22.88% | 17.08% | 14.36% | 0.1574 | 11.93% | 8.06% | 5.88% |

## Interpretation
- All-channel `summary_v2` keeps useful narrow Top1 behavior, but its 2025 forward IC is too weak for a balanced default.
- Adding `price_delta` to all-channel `summary_v2` weakened both 2024 TopK and 2025 forward IC.
- Adding VA auxiliary losses to all-channel `summary_v2` preserved some 2024 narrow TopK strength at high weight, but did not repair forward IC and hurt 2025 Top1 materially.
- Equal OHLCVA path loss improved forward IC versus the all-channel anchor, but still stayed below the current default and did not preserve all-channel Top1.
- Current default remains `daily_only_summary_v2_ohlcva_aux_low`: daily-only input plus low-weight VA auxiliary has better IC stability under the preferred two-anchor evaluation policy.
