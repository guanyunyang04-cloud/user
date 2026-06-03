# Daily Research V2 Bad Month Attribution - 2026-06-02

## Summary

- Status: `v2_bad_month_attribution / evidence_grade_candidate_diagnostic / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_bad_month_attribution`.
- Attribution run tag: `v2_bad_month_attribution_20260602_01`.
- Source matrix run tag: `v2_candidate_review_matrix_20260602_01`.
- Variant: `h20_mw080_rb5d_all_c3_7_10_regime_off`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_bad_month_attribution_20260602_01`.
- `v2_candidate_review_matrix_20260602_01`.
- `v2_score_backtest_bridge_20260602_01`.

## Implemented Attribution

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_bad_month_attribution --json`
- Added output artifacts under:
  - `daily_research/output/path_policy/studies/v2_bad_month_attribution_20260602_01/`
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_bad_month_attribution.py`.
- Attribution method:
  - previous target weight x close-to-close stock return;
  - approximate diagnostic only, not an exact simulator decomposition.

## Worst Month Evidence

- Automatically selected month: `2024-08`.
- Portfolio month return: `-0.1337481001056304`.
- Benchmark month return: `-0.0393816434459514`.
- Excess month return: `-0.0982351170117052`.
- Month trading cost return: `0.0085626767036635`.
- Approx contribution sum: `-0.1687804919974033`.
- Contribution rows: `673`.
- Contribution symbols: `64`.
- Contribution dates: `22`.

## Top Negative Symbol Contributors

| stock | contribution_sum | rows | avg_weight | avg_return |
| --- | ---: | ---: | ---: | ---: |
| `600676.SH` | `-0.020553402993597102` | `22` | `0.063623656663571` | `-0.015727940937163374` |
| `600841.SH` | `-0.017640027887475696` | `20` | `0.05605840340161462` | `-0.01628152213939113` |
| `600686.SH` | `-0.015037672200891642` | `22` | `0.04171370057663664` | `-0.005300428708604699` |
| `000550.SZ` | `-0.01370677252464183` | `19` | `0.0381364055409247` | `-0.014731106360411025` |
| `600733.SH` | `-0.013684510513160756` | `18` | `0.06498664016442358` | `-0.016974556323011272` |

## Top Negative Dates

| date | contribution_sum | rows | avg_weight | avg_return |
| --- | ---: | ---: | ---: | ---: |
| `2024-08-08` | `-0.04170574122174424` | `32` | `0.03124999999999996` | `-0.02912638438344885` |
| `2024-08-05` | `-0.03775871148151735` | `39` | `0.025641025641025605` | `-0.041348563782958696` |
| `2024-08-16` | `-0.0291129066804467` | `26` | `0.03846153846153842` | `-0.025023344573200584` |
| `2024-08-06` | `-0.02698982676855728` | `39` | `0.025641025641025605` | `-0.012612883606337614` |
| `2024-08-28` | `-0.024644391786923053` | `29` | `0.03448275862068961` | `-0.0163762539265424` |

## Bucket Evidence

- Worst score bucket: `score_q_5`, contribution `-0.06404371314366673`, row count `142`, avg previous weight `0.07100831708977406`.
- Second worst score bucket: `score_q_4`, contribution `-0.053159923657507845`, row count `129`, avg previous weight `0.047269471450315`.
- Worst weight bucket: `weight_q_5`, contribution `-0.08555559029609913`, row count `142`, avg stock return `-0.008921280673429535`.
- Second worst weight bucket: `weight_q_4`, contribution `-0.048206828166552876`, row count `129`, avg stock return `-0.007471409766622307`.

## Interpretation

- The August 2024 bad month is not explained by low-score tail holdings; the largest negative contribution came from high-score and high-weight buckets.
- The `20` holding-count candidate's bad-month blocker is therefore more likely a high-conviction risk-state / regime / crowding / reversal problem than a simple holding-count problem.
- Cost is visible but not the primary explanation for the worst month: August cost return was `0.0085626767036635`, while excess return was `-0.0982351170117052`.
- The next candidate-matrix expansion should prioritize high-confidence exposure controls, max-weight changes, risk/month-state overlays, and bad-month filters over plain holding-count changes.

## Boundaries

- Do not promote to live/default from this attribution.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Do not treat approximate attribution as exact simulator PnL decomposition.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Next Allowed Actions

- Extend candidate matrix with:
  - max weight `0.04/0.06/0.08`;
  - rebalance frequency `3d/5d/10d`;
  - market-regime or soft-state overlays;
  - high-score/high-weight risk throttles;
  - score confidence filters and reversal/volatility guards.
- Add second-order attribution by sector, liquidity bucket, volatility bucket, and horizon bucket once those sidecars are wired into the attribution report.
- Keep current v2 baseline and candidate bridge research-only until candidate matrix gates pass.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_bad_month_attribution.py -q`: `2 passed`.
- `python -m daily_research.path_policy.v2_bad_month_attribution --json`: completed, selected month `2024-08`.
- `git diff -- daily_research/output/active_execution_strategy.json`: must remain empty.
