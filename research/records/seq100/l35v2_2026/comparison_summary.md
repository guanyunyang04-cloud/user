# Structured 180x35 2026 Fold and Fixed-D7 Accounts

Full-label window: `2026-01-05..2026-03-19` (48 signal dates).
Extended D7 window: `2026-01-05..2026-07-07` (121 signal dates).
Strict D7+20 window: `2026-01-05..2026-06-08` (101 signal dates).

The new checkpoint is an experimental research fold. No deployment state was changed.

## Selected fixed-D7 account results

| Strategy | Window | Cost | Ending equity | Total return | Max drawdown | Trades |
|---|---|---|---:|---:|---:|---:|
| top1_slots1_fixed_d7 | strict_101 | base | 1061280.01 | 6.1280% | -23.6414% | 15 |
| top1_slots1_fixed_d7 | extended_121 | base | 756459.88 | -24.3540% | -44.7645% | 18 |
| top1_slots1_fixed_d7 | strict_101 | double_slippage | 1039310.36 | 3.9310% | -23.9586% | 15 |
| top1_slots1_fixed_d7 | extended_121 | double_slippage | 737721.33 | -26.2279% | -45.2610% | 18 |
| top3_slots3_fixed_d7 | strict_101 | base | 937425.42 | -6.2575% | -22.6878% | 45 |
| top3_slots3_fixed_d7 | extended_121 | base | 814712.22 | -18.5288% | -32.2824% | 54 |
| top3_slots3_fixed_d7 | strict_101 | double_slippage | 917081.88 | -8.2918% | -22.9926% | 45 |
| top3_slots3_fixed_d7 | extended_121 | double_slippage | 793598.38 | -20.6402% | -32.8426% | 54 |

The strict and extended windows are reported separately. This study evaluates the preselected model and does not run a promotion or model-selection gate.
