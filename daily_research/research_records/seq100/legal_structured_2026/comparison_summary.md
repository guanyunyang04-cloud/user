# Seq100 checkpoint vintages on the complete 2026 realized window

Evaluation window: 2026-01-05 through 2026-03-19; 48 signal dates and 143,840 candidates.

| profile | checkpoint | Top1 alpha | Top3 alpha | Top5 alpha | Top10 alpha | Rank IC | exit regret | path MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legal_flat_baseline | 2023 | 8.0602% | 9.1383% | 8.5685% | 6.6064% | 0.126992 | 0.197248 | 0.124878 |
| legal_flat_baseline | 2024 | 13.7164% | 14.3236% | 14.2449% | 11.4913% | 0.146857 | 0.177496 | 0.118477 |
| legal_flat_baseline | 2025 | 5.1743% | 5.4302% | 3.7881% | 2.5912% | 0.117270 | 0.167142 | 0.114896 |
| legal_flat_baseline | 2026 | 10.7730% | 10.1038% | 10.6851% | 11.3150% | 0.125660 | 0.193705 | 0.116224 |
| structured_joint_turnover | 2023 | 16.8103% | 14.0721% | 11.5819% | 8.6196% | 0.124786 | 0.209969 | 0.125605 |
| structured_joint_turnover | 2024 | 1.1256% | 0.2048% | 0.5308% | 0.6031% | 0.117252 | 0.140728 | 0.104789 |
| structured_joint_turnover | 2025 | 8.9782% | 6.7881% | 5.1212% | 4.4218% | 0.115747 | 0.248496 | 0.132381 |
| structured_joint_turnover | 2026 | 5.7241% | 5.7556% | 6.2226% | 5.9745% | 0.124168 | 0.223065 | 0.126905 |

Fairness and coverage checks: `True`.
2026 freshness gate: `False`.
No formal moving-block interval or finite-capital CAGR is reported for this 48-date window.
