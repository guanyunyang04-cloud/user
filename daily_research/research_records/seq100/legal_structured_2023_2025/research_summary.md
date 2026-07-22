# Seq100 Legal Exit and Structured Path Comparison

- Report only; no final fit, QDP update, pack mutation, or deployment change.
- All ranking and planned exits use the legal D2-D60 price-path domain.
- Percentages below are equal-day annual cohort means; the final row for each profile is an equal-year mean.

## Economic results

| profile | year | epoch | Top1 alpha | Top3 alpha | Top5 alpha | Top10 alpha | Top3 absolute | universe | stress alpha | opportunity alpha | opp-exec gap | rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal_flat_baseline | 2023 | 1 | 2.1636% | 2.0008% | 2.2573% | 1.9421% | 0.2381% | -1.7627% | 1.9985% | 6.5190% | 4.5182% | 0.094197 |
| legal_flat_baseline | 2024 | 1 | 3.2281% | 5.8572% | 6.4742% | 6.4100% | 8.7091% | 2.8518% | 5.8519% | 10.4380% | 4.5807% | 0.148966 |
| legal_flat_baseline | 2025 | 1 | 7.8134% | 5.9069% | 5.8814% | 5.2101% | 8.2106% | 2.3037% | 5.8992% | 13.1347% | 7.2278% | 0.163092 |
| legal_flat_baseline | equal-year | 1 | 4.4017% | 4.5883% | 4.8710% | 4.5207% | 5.7193% | 1.1310% | 4.5832% | 10.0305% | 5.4422% | 0.135418 |
| structured_joint_turnover | 2023 | 1 | 0.4825% | 2.2511% | 1.4094% | 1.3874% | -0.0646% | -2.3157% | 2.2479% | 7.6909% | 5.4398% | 0.062171 |
| structured_joint_turnover | 2024 | 1 | 9.7549% | 9.2810% | 8.1035% | 7.1080% | 9.5198% | 0.2387% | 9.2668% | 15.6399% | 6.3588% | 0.114735 |
| structured_joint_turnover | 2025 | 1 | 5.8806% | 5.5839% | 4.4707% | 4.0234% | 12.7830% | 7.1990% | 5.5738% | 10.8658% | 5.2819% | 0.128918 |
| structured_joint_turnover | equal-year | 1 | 5.3727% | 5.7054% | 4.6612% | 4.1729% | 7.4127% | 1.7073% | 5.6962% | 11.3989% | 5.6935% | 0.101942 |

## Exit and path diagnostics

| profile | year | entry fill | score/return coverage | avg actual exit | Top3 deferral | deferred days | unresolved | exit regret | raw oracle regret | utility MAE/corr | close MAE | direction | path corr | Top3 exit entropy/max share | geometry violations | turnover level/delta MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal_flat_baseline | 2023 | 99.5868% | 100.00%/100.00% | D52.35 | 0.4149% | 1.333 | 0.0000% | 0.162145 | 0.122131 | 0.162405/0.037575 | 0.098673 | 44.6333% | -0.119289 | 0.3987/87.05% | 20808368 | n/a |
| legal_flat_baseline | 2024 | 98.8981% | 100.00%/100.00% | D55.54 | 1.9499% | 11.500 | 0.1377% | 0.208291 | 0.172779 | 0.207887/0.010708 | 0.128895 | 49.6146% | 0.012249 | 0.6515/66.80% | 21703989 | n/a |
| legal_flat_baseline | 2025 | 99.4513% | 100.00%/100.00% | D25.97 | 1.2414% | 1.778 | 0.0000% | 0.177319 | 0.175144 | 0.144579/0.041785 | 0.097342 | 56.9195% | 0.012517 | 0.0881/98.49% | 20208947 | n/a |
| structured_joint_turnover | 2023 | 99.8623% | 100.00%/100.00% | D59.01 | 0.8276% | 1.667 | 0.0000% | 0.183277 | 0.127670 | 0.188258/0.034232 | 0.105358 | 46.0646% | -0.176668 | -0.0000/100.00% | 0 | 0.317642/0.167923 |
| structured_joint_turnover | 2024 | 99.4490% | 100.00%/100.00% | D39.99 | 0.5540% | 1.250 | 0.0000% | 0.196025 | 0.199061 | 0.270428/-0.010934 | 0.133514 | 48.8221% | -0.004874 | 1.2487/56.06% | 0 | 0.375389/0.179914 |
| structured_joint_turnover | 2025 | 99.8628% | 100.00%/100.00% | D58.02 | 1.5110% | 1.545 | 0.0000% | 0.167951 | 0.126185 | 0.221626/0.068979 | 0.109487 | 58.3898% | 0.158975 | -0.0000/100.00% | 0 | 0.383711/0.199984 |

## OHLC path MAE by horizon

| profile | year | D1-5 | D6-10 | D11-20 | D21-40 | D41-60 |
|---|---:|---:|---:|---:|---:|---:|
| legal_flat_baseline | 2023 | 0.033695 | 0.055123 | 0.077888 | 0.106355 | 0.140274 |
| legal_flat_baseline | 2024 | 0.045583 | 0.076184 | 0.101813 | 0.133227 | 0.168420 |
| legal_flat_baseline | 2025 | 0.037292 | 0.054870 | 0.076645 | 0.099671 | 0.126175 |
| structured_joint_turnover | 2023 | 0.038499 | 0.052487 | 0.072283 | 0.109015 | 0.152410 |
| structured_joint_turnover | 2024 | 0.052105 | 0.078246 | 0.107259 | 0.141837 | 0.174588 |
| structured_joint_turnover | 2025 | 0.043793 | 0.063128 | 0.085585 | 0.115041 | 0.148216 |

## Locked judgement

Equal-year Top3 alpha delta (structured - legal flat): `1.1170%`.
Structured positive Top3 years: `3/3`; worst-year alpha: `2.2511%`.
Equal-year exit regret lower: `True`; close-path non-worse years: `0/3`; new exit-day collapse: `True`; structured geometry violations: `0`.
Consistent improvement evidence under the pre-registered rule: `False`.

No automatic winner is declared. The structured result is a combined architecture effect and cannot be attributed to one component.
