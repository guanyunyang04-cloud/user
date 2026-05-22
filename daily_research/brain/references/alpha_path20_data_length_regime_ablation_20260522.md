# Alpha Path20 Data Length / Regime Ablation Result - 2026-05-22

## Scope
- Status: `research / shadow-only / no promotion`.
- Active artifact: `daily_research/output/active_execution_strategy.json` unchanged.
- Objective: test whether longer train-year coverage fixes Path20 `decision_utility_v1` early-best / weak test monthly stability.
- Fixed anchor: `liquid500`, `raw_kline_context_sector_v1`, `gru_sequence_static_context`, seed `7`, `decision_utility_v1`, cost `20`, hit threshold `10`, drawdown penalty `0.10`.
- Audit artifact: `daily_research/output/path_policy/studies/path20_data_length_regime_audit_20260522.json`.

## Studies
- Baseline: `path20_decision_output_liquid500_du_cost20_hit10_dd010_20260521_01`
  - Train years: `2019-2022`; validation `2023`; test `2024`.
  - Pool view: reused `policy_pool_view__c11400fa72ad263f3d1eecfa` (`2018-2024` window).
- 8-year raw window ablation: `path20_data_len_liquid500_du_train2018_2022_20260522_01`
  - Prepare window: `2017-01-01 -> 2024-12-31`; train years `2018-2022`; validation `2023`; test `2024`.
  - New rolling pool view: `policy_pool_view__ded4d32f2d835236bad38af9`.
- Long raw window ablation: `path20_data_len_liquid500_du_train2016_2022_20260522_01`
  - Prepare window: `2015-01-01 -> 2024-12-31`; train years `2016-2022`; validation `2023`; test `2024`.
  - New rolling pool view: `policy_pool_view__71f27acacced2dbbd72a547c`.

## Results
| Study | Best epoch | Test decision rank IC | Test decision spread | Test hit lift | Test monthly spread positive rate | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline train2019-2022 | 4 | 0.040376 | 0.009600 | 0.015789 | 50.0% | failed monthly gate |
| ablation train2018-2022 | 1 | 0.023404 | -0.003732 | 0.014269 | 50.0% | failed test spread + monthly gate |
| ablation train2016-2022 | 1 | 0.029257 | -0.001175 | 0.021602 | 58.3% | failed test spread + monthly gate |

Validation decision metrics stayed positive for both ablations, but neither ablation passed the full decision gate. Both longer-window runs selected epoch `1`, so extending the training window did not fix the early-best symptom.

## Interpretation
- Data length / regime coverage remains a credible contributor: the long `2016-2022` run improved monthly spread positive rate from `50.0%` to `58.3%`.
- Data length alone is not sufficient: both longer-window runs flipped aggregate test decision spread negative.
- Current bottleneck is more likely a combination of target / selection calibration and input regime contamination than pure sample count.
- Repeating larger architecture or liquid800 expansion before target / selection review is not recommended.

## Next Step
- Do a score / selection calibration audit on `decision_utility_v1` before more training:
  - compare `pred_decision_score=max utility` against horizon-selected utility, hit-weighted utility, and short-horizon blends;
  - require validation/test spread, hit lift, and monthly stability to improve without retraining first.
- If selection-only fixes test spread and monthly stability, then rerun one long-window GRU with the revised selection profile.
- If selection-only cannot fix it, test input contamination next: `alpha_prior` ablation and sector/static context ablation on liquid500.
