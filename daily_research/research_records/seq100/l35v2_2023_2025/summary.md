# Structured Input Capital Speed: Pre-Stage-3

2023, 2024, and 2025 are symmetric rolling development folds and are selected together. Cross-model ranking/exit hybrids are excluded.

## Main result

- Selected complete model: `lookback180_turnover`.
- Maximum growth: Top1, 1 slot, `fixed_d7`, CAGR 316.0030%, drawdown -46.2568%, 104 trades.
- Practical Top3/3 result: `lookback180_turnover` with `fixed_d7`, CAGR 176.0673%, drawdown -33.2781%, 312 trades.
- The selected model's own exit wins 0 of 22 model/Top-K/slot comparisons; fixed exit wins the rest.

## Stage comparison

| Stage | Models | Selected model | Global policy | CAGR | Trades |
|---:|---|---|---|---:|---:|
| 1 | lookback100_daily vs lookback180_daily | lookback180_daily | Top1 / 1 slot / fixed_d35 | 228.6560% | 21 |
| 2 | lookback180_daily vs lookback180_turnover | lookback180_turnover | Top1 / 1 slot / fixed_d7 | 316.0030% | 104 |

## Stage 2 capacity frontier

| Top-K | Slots | Model | Policy | CAGR | Drawdown | Trades |
|---:|---:|---|---|---:|---:|---:|
| 1 | 1 | lookback180_turnover | fixed_d7 | 316.0030% | -46.2568% | 104 |
| 1 | 3 | lookback180_daily | fixed_d8 | 129.1384% | -38.2964% | 221 |
| 1 | 6 | lookback180_turnover | fixed_d39 | 92.6126% | -40.5173% | 109 |
| 1 | 12 | lookback180_turnover | fixed_d31 | 68.6870% | -18.9833% | 202 |
| 1 | 24 | lookback180_daily | fixed_d60 | 41.1087% | -25.6896% | 198 |
| 1 | 48 | lookback180_daily | fixed_d60 | 19.8719% | -12.9575% | 202 |
| 3 | 3 | lookback180_turnover | fixed_d7 | 176.0673% | -33.2781% | 312 |
| 3 | 6 | lookback180_turnover | fixed_d9 | 134.6064% | -29.0874% | 472 |
| 3 | 12 | lookback180_turnover | fixed_d34 | 74.7757% | -33.0518% | 256 |
| 3 | 24 | lookback180_turnover | fixed_d48 | 57.5487% | -27.1017% | 347 |
| 3 | 48 | lookback180_turnover | fixed_d49 | 32.8819% | -22.4128% | 465 |

## Annual returns

| Strategy | Cost | 2023 | 2024 | 2025 |
|---|---|---:|---:|---:|
| Maximum growth | base | 237.4949% | 615.7780% | 152.9290% |
| Maximum growth | double_slippage | 221.5939% | 582.4186% | 140.8266% |
| Top3/3 | base | 178.5135% | 208.1855% | 118.0893% |
| Top3/3 | double_slippage | 165.3138% | 193.9687% | 107.8418% |

## Interpretation

The turnover-enhanced 180-day model improves the deployable ranking baseline, but its learned exit does not beat its own best fixed exit at any tested Top-K/slot capacity. Stage 3 has not been started.
