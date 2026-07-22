# Structured 180x35 Training-window Result

- Status: `completed`.
- Objective: maximize double-slippage continuous-account annualized log growth.
- Capital configurations: Top1/1 slot, Top3/3 slots, Top3/6 slots.
- Selected baseline: `expanding`.
- Expanding retained: `true`.
- IC, path error, cohort alpha, and exit diagnostics are explanatory only.

## 2025 Screen

| model | mean annual log growth |
|---|---:|
| window8y | 0.9163 |
| expanding | 0.6323 |
| window3y | 0.5243 |
| window5y | 0.4356 |
| window12y | 0.2033 |

## Three-year Selected Strategies

| model | config | execution | annual log growth | implied CAGR | total return | max drawdown | win rate | trades | mean hold |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| expanding | Top1/1 | fixed_d7 | 1.3753 | 295.61% | 5185.22% | -46.44% | 60.58% | 104 | 7.07 |
| expanding | Top3/3 | fixed_d7 | 0.9656 | 162.63% | 1521.04% | -33.54% | 57.05% | 312 | 7.02 |
| expanding | Top3/6 | fixed_d9 | 0.8153 | 126.00% | 950.87% | -29.49% | 57.42% | 472 | 9.05 |
| window8y | Top1/1 | fixed_d35 | 0.9154 | 149.79% | 1302.68% | -63.44% | 61.90% | 21 | 35.00 |
| window8y | Top3/3 | rolling_path | 0.9232 | 151.72% | 1334.27% | -32.30% | 55.97% | 293 | 7.47 |
| window8y | Top3/6 | fixed_d41 | 0.7045 | 102.28% | 663.18% | -43.84% | 60.19% | 108 | 41.00 |

## Calendar-year Account Returns

| model | config | execution | 2023 | 2024 | 2025 |
|---|---|---|---:|---:|---:|
| expanding | Top1/1 | fixed_d7 | 221.59% | 582.42% | 140.83% |
| expanding | Top3/3 | fixed_d7 | 165.31% | 193.97% | 107.84% |
| expanding | Top3/6 | fixed_d9 | 94.93% | 164.69% | 103.67% |
| window8y | Top1/1 | fixed_d35 | 53.60% | 50.10% | 508.38% |
| window8y | Top3/3 | rolling_path | 124.13% | 192.63% | 118.68% |
| window8y | Top3/6 | fixed_d41 | 47.75% | 37.01% | 277.00% |

The eight-year model won the 2025 screen but lost the symmetric 2023-2025 confirmation. No 3-epoch rerun was required after applying the registered conflict rule correctly.

No QDP, provider, Stage 3, old checkpoint, base pack, or live-state change was made.
