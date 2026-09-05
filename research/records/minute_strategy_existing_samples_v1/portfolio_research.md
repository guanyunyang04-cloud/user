# Existing minute-strategy portfolio research

This report summarizes completed account replays over already generated samples.
It is descriptive research evidence; it does not freeze a live strategy or treat the best grid cell as validation.
The companion event-layer files in this record cover every canonical outcome row, six MA periods, eight forward horizons, controls, regimes, features, tails, costs and author-hypothesis mappings.

## Coverage and integrity

| Sample | Signal dates | Signal window | Outcome window | Strategy count | Account results | Lowest available RAM |
|---|---:|---|---|---:|---:|---:|
| primary_development_2022_2023_partial | 331 | 2022-01-04 to 2023-05-18 | 2022-01-04 to 2023-06-02 | 23 | 3220 | 1.5347671508789062 GiB |
| supplemental_2023_09_legacy | 20 | 2023-09-01 to 2023-09-28 | 2023-09-01 to 2023-10-13 | 22 | 3080 | 5.029674530029297 GiB |

The primary sample is the current-contract partial development run ending at signal date 2023-05-18; its later calendar end reflects forward outcome availability. The supplemental sample is a separate 20-day September 2023 legacy-VWAP run. They are never pooled as one time series.

## Account-grid result

| Sample | Accounts | Positive | Positive fraction | Fully resolved | Mean return | Median return | Worst drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary_development_2022_2023_partial | 3220 | 310 | 9.63% | 677 | -30.41% | -32.08% | -84.90% |
| supplemental_2023_09_legacy | 3080 | 655 | 21.27% | 2932 | -3.14% | -3.21% | -21.31% |

The full account result table is `account_results.csv`; it contains every strategy, exit policy, ranker and seed combination, including losing and unresolved cases.

The primary window has many holdings that could not be legally closed before the available bars ended or remained locked. Therefore `account_resolved_results.csv` is the cleanly closed subset, while `account_results.csv` keeps marked-to-market unresolved positions. Neither view is silently substituted for the other.

| Sample | Fully resolved accounts | Resolved positive fraction | Resolved mean return | Resolved median return | Resolved worst drawdown |
|---|---:|---:|---:|---:|---:|
| primary_development_2022_2023_partial | 677 | 12.26% | -26.58% | -27.66% | -72.19% |
| supplemental_2023_09_legacy | 2932 | 21.56% | -3.14% | -3.20% | -21.31% |

## Execution and risk diagnostics

The trade and equity summaries expose cost drag, turnover, time in market, daily volatility and unresolved/blocked exits. The portfolio assumptions use 3 bps commission, 0.1 bps transfer fee, 7 bps slippage and the declared stamp-tax schedule; these costs are not omitted from net returns.

## Ranker and exit summaries

| Sample | Ranker | Policy | Accounts | Positive fraction | Mean return | Median return | Mean drawdown |
|---|---|---|---:|---:|---:|---:|---:|
| primary_development_2022_2023_partial | sector_leader | next_open_5d | 115 | 26.09% | -4.10% | -9.61% | -35.69% |
| primary_development_2022_2023_partial | sector_leader | next_open_3d | 115 | 26.09% | -10.45% | -20.01% | -37.89% |
| primary_development_2022_2023_partial | flow_quality | next_open_5d | 115 | 18.26% | -13.67% | -16.66% | -32.44% |
| primary_development_2022_2023_partial | random_hash | next_open_5d | 115 | 20.87% | -13.83% | -16.96% | -32.45% |
| primary_development_2022_2023_partial | flow_quality | next_open_3d | 115 | 20.00% | -16.73% | -18.56% | -33.07% |
| primary_development_2022_2023_partial | random_hash | next_open_3d | 115 | 14.78% | -18.09% | -17.44% | -33.59% |
| primary_development_2022_2023_partial | trend_structure | next_open_3d | 115 | 21.74% | -20.83% | -23.25% | -39.82% |
| primary_development_2022_2023_partial | trend_structure | next_open_5d | 115 | 12.17% | -21.04% | -22.16% | -34.68% |
| primary_development_2022_2023_partial | flow_quality | next_open_2d | 115 | 11.30% | -22.84% | -23.57% | -35.58% |
| primary_development_2022_2023_partial | random_hash | next_open_2d | 115 | 13.91% | -23.25% | -23.60% | -35.75% |
| primary_development_2022_2023_partial | sector_leader | next_open_2d | 115 | 13.04% | -23.60% | -23.83% | -39.24% |
| primary_development_2022_2023_partial | flow_quality | momentum_trail_4pct_5d | 115 | 9.57% | -26.79% | -29.37% | -37.17% |
| primary_development_2022_2023_partial | random_hash | momentum_trail_4pct_5d | 115 | 8.70% | -26.80% | -29.54% | -36.73% |
| primary_development_2022_2023_partial | sector_leader | momentum_trail_4pct_5d | 115 | 13.04% | -33.16% | -32.66% | -43.89% |
| primary_development_2022_2023_partial | flow_quality | protect_3pct_target6pct_3d | 115 | 5.22% | -35.08% | -35.38% | -42.55% |
| primary_development_2022_2023_partial | sector_leader | protect_3pct_target6pct_3d | 115 | 8.70% | -35.89% | -38.12% | -46.51% |
| primary_development_2022_2023_partial | trend_structure | momentum_trail_4pct_5d | 115 | 0.87% | -36.17% | -36.27% | -43.68% |
| primary_development_2022_2023_partial | sector_leader | next_open_1d | 115 | 4.35% | -36.41% | -42.21% | -47.35% |
| primary_development_2022_2023_partial | random_hash | protect_3pct_target6pct_3d | 115 | 5.22% | -36.84% | -38.58% | -43.45% |
| primary_development_2022_2023_partial | trend_structure | next_open_2d | 115 | 2.61% | -38.00% | -38.37% | -44.58% |
| primary_development_2022_2023_partial | random_hash | fast_failure_2pct_target4pct_2d | 115 | 5.22% | -40.72% | -45.27% | -47.78% |
| primary_development_2022_2023_partial | random_hash | next_open_1d | 115 | 1.74% | -41.49% | -45.42% | -47.82% |
| primary_development_2022_2023_partial | flow_quality | next_open_1d | 115 | 1.74% | -41.66% | -45.58% | -47.93% |
| primary_development_2022_2023_partial | flow_quality | fast_failure_2pct_target4pct_2d | 115 | 0.87% | -41.95% | -43.85% | -47.84% |
| primary_development_2022_2023_partial | sector_leader | fast_failure_2pct_target4pct_2d | 115 | 0.00% | -44.40% | -54.14% | -51.89% |
| primary_development_2022_2023_partial | trend_structure | protect_3pct_target6pct_3d | 115 | 0.00% | -46.71% | -48.38% | -51.31% |
| primary_development_2022_2023_partial | trend_structure | next_open_1d | 115 | 3.48% | -47.49% | -49.04% | -53.36% |
| primary_development_2022_2023_partial | trend_structure | fast_failure_2pct_target4pct_2d | 115 | 0.00% | -53.43% | -54.35% | -57.01% |
| supplemental_2023_09_legacy | flow_quality | next_open_3d | 110 | 34.55% | -1.50% | -1.48% | -5.08% |
| supplemental_2023_09_legacy | random_hash | next_open_3d | 110 | 27.27% | -2.01% | -2.06% | -5.35% |
| supplemental_2023_09_legacy | flow_quality | next_open_2d | 110 | 28.18% | -2.19% | -2.26% | -5.84% |
| supplemental_2023_09_legacy | sector_leader | next_open_2d | 110 | 40.91% | -2.20% | -2.77% | -8.59% |
| supplemental_2023_09_legacy | random_hash | next_open_2d | 110 | 27.27% | -2.29% | -2.24% | -5.95% |
| supplemental_2023_09_legacy | trend_structure | protect_3pct_target6pct_3d | 110 | 28.18% | -2.30% | -2.07% | -5.11% |
| supplemental_2023_09_legacy | flow_quality | next_open_5d | 110 | 23.64% | -2.32% | -2.89% | -6.01% |
| supplemental_2023_09_legacy | random_hash | next_open_5d | 110 | 22.73% | -2.42% | -2.82% | -5.98% |
| supplemental_2023_09_legacy | random_hash | momentum_trail_4pct_5d | 110 | 18.18% | -2.42% | -2.50% | -5.46% |
| supplemental_2023_09_legacy | flow_quality | momentum_trail_4pct_5d | 110 | 14.55% | -2.60% | -2.61% | -5.78% |
| supplemental_2023_09_legacy | sector_leader | next_open_1d | 110 | 40.91% | -2.85% | -3.32% | -8.78% |
| supplemental_2023_09_legacy | trend_structure | next_open_5d | 110 | 21.82% | -2.86% | -2.76% | -5.74% |
| supplemental_2023_09_legacy | flow_quality | protect_3pct_target6pct_3d | 110 | 13.64% | -2.92% | -2.98% | -6.00% |
| supplemental_2023_09_legacy | random_hash | protect_3pct_target6pct_3d | 110 | 17.27% | -2.97% | -3.24% | -5.85% |
| supplemental_2023_09_legacy | trend_structure | momentum_trail_4pct_5d | 110 | 17.27% | -3.05% | -3.12% | -5.83% |
| supplemental_2023_09_legacy | flow_quality | fast_failure_2pct_target4pct_2d | 110 | 14.55% | -3.15% | -3.22% | -6.22% |
| supplemental_2023_09_legacy | trend_structure | fast_failure_2pct_target4pct_2d | 110 | 22.73% | -3.35% | -4.21% | -6.13% |
| supplemental_2023_09_legacy | random_hash | fast_failure_2pct_target4pct_2d | 110 | 15.45% | -3.35% | -3.25% | -6.24% |
| supplemental_2023_09_legacy | trend_structure | next_open_2d | 110 | 25.45% | -3.51% | -4.11% | -6.28% |
| supplemental_2023_09_legacy | trend_structure | next_open_3d | 110 | 18.18% | -3.63% | -4.19% | -6.15% |
| supplemental_2023_09_legacy | sector_leader | fast_failure_2pct_target4pct_2d | 110 | 18.18% | -3.71% | -3.88% | -8.47% |
| supplemental_2023_09_legacy | sector_leader | next_open_3d | 110 | 9.09% | -3.79% | -3.16% | -8.76% |
| supplemental_2023_09_legacy | flow_quality | next_open_1d | 110 | 13.64% | -3.95% | -4.31% | -6.82% |
| supplemental_2023_09_legacy | random_hash | next_open_1d | 110 | 10.91% | -3.99% | -4.38% | -6.75% |
| supplemental_2023_09_legacy | sector_leader | next_open_5d | 110 | 4.55% | -4.23% | -4.87% | -9.20% |
| supplemental_2023_09_legacy | sector_leader | protect_3pct_target6pct_3d | 110 | 27.27% | -4.28% | -2.71% | -8.66% |
| supplemental_2023_09_legacy | trend_structure | next_open_1d | 110 | 16.36% | -4.90% | -5.10% | -7.38% |
| supplemental_2023_09_legacy | sector_leader | momentum_trail_4pct_5d | 110 | 22.73% | -5.16% | -6.26% | -9.34% |

## Best cells (descriptive only)

### primary_development_2022_2023_partial

| Ranker | Seed | Strategy | Policy | Net return | Max drawdown | Closed trades | Resolved |
|---|---:|---|---|---:|---:|---:|---|
| flow_quality | 2 | fu_r3_core_pullback | next_open_3d | 66.16% | -21.47% | 445 | no |
| flow_quality | 3 | s2_reclaim_daily_trend | next_open_5d | 61.69% | -20.38% | 251 | no |
| random_hash | 3 | s2_reclaim_daily_trend | next_open_5d | 58.49% | -20.98% | 251 | no |
| sector_leader | 0 | s2_reclaim_daily_trend | next_open_5d | 58.09% | -26.79% | 322 | no |
| sector_leader | 2 | s2_reclaim_daily_trend | next_open_5d | 58.09% | -26.79% | 322 | no |
| sector_leader | 3 | s2_reclaim_daily_trend | next_open_5d | 58.09% | -26.79% | 322 | no |
| sector_leader | 4 | s2_reclaim_daily_trend | next_open_5d | 58.09% | -26.79% | 322 | no |
| sector_leader | 1 | s2_reclaim_daily_trend | next_open_5d | 58.09% | -26.79% | 322 | no |
| sector_leader | 1 | s2_reclaim_daily_trend | next_open_3d | 52.91% | -22.17% | 427 | no |
| sector_leader | 2 | s2_reclaim_daily_trend | next_open_3d | 52.91% | -22.17% | 427 | no |

### supplemental_2023_09_legacy

| Ranker | Seed | Strategy | Policy | Net return | Max drawdown | Closed trades | Resolved |
|---|---:|---|---|---:|---:|---:|---|
| trend_structure | 4 | s0_strong_no_ma | next_open_1d | 44.03% | -4.92% | 80 | no |
| sector_leader | 2 | s2_reclaim_recent_high | next_open_5d | 11.88% | -2.25% | 20 | yes |
| sector_leader | 4 | s2_reclaim_recent_high | next_open_5d | 11.88% | -2.25% | 20 | yes |
| sector_leader | 1 | s2_reclaim_recent_high | next_open_5d | 11.88% | -2.25% | 20 | yes |
| sector_leader | 0 | s2_reclaim_recent_high | next_open_5d | 11.88% | -2.25% | 20 | yes |
| sector_leader | 3 | s2_reclaim_recent_high | next_open_5d | 11.88% | -2.25% | 20 | yes |
| trend_structure | 4 | s0_strong_no_ma | next_open_3d | 9.09% | -3.93% | 35 | yes |
| trend_structure | 1 | s4_reclaim_volume_proxy | protect_3pct_target6pct_3d | 8.92% | -1.54% | 45 | yes |
| random_hash | 2 | s4_auction_confirmed_reclaim | protect_3pct_target6pct_3d | 8.64% | -2.01% | 45 | yes |
| flow_quality | 2 | s4_auction_confirmed_reclaim | protect_3pct_target6pct_3d | 8.24% | -2.21% | 44 | yes |

These cells are not selected as deployable candidates. They are the extrema of a large comparison grid and therefore carry selection bias, especially in the short supplemental window.

## Cross-sample comparison

The shared-key comparison contains 3080 rows for the 22 strategies present in both samples. It is a descriptive stability check, not an out-of-sample test: the primary and supplemental windows use different sample lengths and the supplemental run uses the legacy VWAP revision.

See `account_seed_stability.csv`, `account_trade_summary.csv`, `account_trade_year.csv`, `account_exit_reason.csv`, `account_equity_summary.csv`, and `account_cross_sample.csv` for the complete machine-readable detail. The event-layer and combined conclusions are in `research_summary.md`.

## Boundaries

- No ungenerated 2023-05-19 onward or 2024 raw minute events were reconstructed.
- No 2025 data was read.
- Duplicate April 2022 and probe directories remain audit-only and are excluded from the canonical event sample.
- Returns include the portfolio runner's declared fees, slippage, cash limits, five-position cap, T+1 and legal-exit handling.
