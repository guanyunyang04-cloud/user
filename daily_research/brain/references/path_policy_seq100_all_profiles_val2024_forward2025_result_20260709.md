# Path Policy Seq100 All Profiles Val2024 Forward2025 Result

Date: `2026-07-09`

## Verdict
- Status: `completed / all_profile_roll_forward_comparison / research_default_updated / no_active_execution_change`.
- Main evaluation policy remains `2024 validation` + `2025 train-through-2024 forward test`.
- Selected default research profile: `seq100_todayclose_path_only/daily_only_summary_v2_ohlcva_aux_low`.
- Reason: among aligned 60d path-output profiles, `ohlcva_aux_low` has both the strongest 2024 validation IC (`0.1626`) and the strongest 2025 forward IC (`0.1927`), while also improving forward Top1 over the previous default (`22.51%` vs `18.44%`).
- Not selected: `summary_v2_all_channels` has the strongest forward Top1 (`27.30%`) but weak forward IC (`0.1417`); `ohlcva_path_equal` and `ohlcva_aux_low_price_delta` are narrow TopK-biased; direct-value rankers have different output semantics and weaker forward evidence.
- Active artifact impact: unchanged. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

## Artifacts
- Full comparison CSV: `daily_research/output/path_policy/sequence_path_training/seq100_all_profiles_val2024_forward2025_comparison_20260709.csv`
- Full comparison JSON: `daily_research/output/path_policy/sequence_path_training/seq100_all_profiles_val2024_forward2025_comparison_20260709.json`
- Roll-forward view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json`
- Fast rerun script: `daily_research/output/path_policy/sequence_path_training/run_rollforward_2025_remaining_profiles_fast_20260709.ps1`

## Results
Path-output rows use 60d `path_trade_value_v2_60d`. Direct-value 5d/10d rows use their own horizon value column and are listed as comparison rankers, not direct replacements for OHLC path-output.

| profile | family | 2024 val IC | 2024 val Top1 | 2024 val Top3 | 2024 val Top10 | 2025 fwd IC | 2025 fwd Top1 | 2025 fwd Top3 | 2025 fwd Top10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_all_channels_base | path_output_60d | 0.1558 | 18.43% | 14.94% | 12.69% | 0.1625 | 14.17% | 6.75% | 4.98% |
| daily_only_base | path_output_60d | 0.1438 | 18.87% | 16.65% | 14.93% | 0.1876 | 13.84% | 6.92% | 4.89% |
| no_intraday_summary | path_output_60d | 0.1520 | 19.15% | 19.76% | 16.22% | 0.1588 | 22.06% | 13.92% | 11.00% |
| no_limit_structure | path_output_60d | 0.1432 | 14.81% | 15.08% | 12.55% | 0.1660 | 15.64% | 12.37% | 9.15% |
| summary_v2_all_channels | path_output_60d | 0.1609 | 26.27% | 23.06% | 16.15% | 0.1417 | 27.30% | 15.25% | 8.28% |
| summary_v2_no60 | path_output_60d | 0.1535 | 14.41% | 16.91% | 14.99% | 0.1387 | 23.61% | 9.99% | 6.28% |
| daily_only_summary_v2 | path_output_60d | 0.1555 | 21.98% | 19.64% | 16.03% | 0.1846 | 18.44% | 10.09% | 6.96% |
| price_delta | path_output_60d | 0.1605 | 19.27% | 14.95% | 13.46% | 0.1783 | 21.54% | 8.63% | 5.35% |
| ohlcva_aux_high | path_output_60d | 0.1502 | 21.95% | 19.64% | 14.86% | 0.1336 | 23.12% | 9.21% | 7.95% |
| ohlcva_aux_low | path_output_60d | 0.1626 | 19.31% | 17.02% | 13.66% | 0.1927 | 22.51% | 7.73% | 6.72% |
| ohlcva_aux_low_price_delta | path_output_60d | 0.1525 | 21.99% | 17.79% | 14.98% | 0.1861 | 24.61% | 11.77% | 6.37% |
| ohlcva_path_equal | path_output_60d | 0.1497 | 19.37% | 18.30% | 15.11% | 0.1768 | 25.18% | 11.74% | 7.28% |
| direct_value_5d | direct_value_ranker | 0.0662 | 2.48% | 1.82% | 1.46% | 0.0328 | 1.61% | 1.21% | 0.88% |
| direct_value_10d | direct_value_ranker | 0.0868 | 4.71% | 3.74% | 2.78% | 0.0626 | 3.82% | 2.79% | 2.45% |
| direct_value_60d | direct_value_ranker | 0.1435 | 21.75% | 18.42% | 14.38% | 0.1358 | 4.02% | 6.83% | 6.95% |

## Interpretation
- The old default `daily_only_summary_v2` remains strong, but `ohlcva_aux_low` is better on the two most important IC anchors: 2024 validation and 2025 forward.
- VA helps only as a light auxiliary representation constraint. High VA weights (`0.05/0.02`) are materially worse; equal OHLCVA path-loss is still too TopK-biased for a balanced default.
- `summary_v2_all_channels` is a useful narrow Top1 comparison but its forward IC collapse makes it unsuitable as the default.
- Removing the 60d multi-horizon summary window (`summary_v2_no60`) did not improve the roll-forward case.
- Direct-value 5d/10d/60d rankers should remain comparison branches. They do not output future OHLC paths, and their 2025 forward evidence is weaker than the selected path-output default.
