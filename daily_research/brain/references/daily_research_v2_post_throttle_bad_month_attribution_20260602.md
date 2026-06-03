# Daily Research V2 Post-Throttle Bad-Month Attribution - 2026-06-02

## Summary

- Status: `v2_post_throttle_bad_month_attribution / evidence_grade_candidate_diagnostic / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_bad_month_attribution`.
- Source matrix: `v2_selective_throttle_matrix_narrow_20260602_01`.
- Source variant: `h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_post_throttle_bad_month_attribution_20240602_m202403`.
- `v2_post_throttle_bad_month_attribution_20240602_m202406`.
- `v2_post_throttle_bad_month_attribution_20240602_m202408`.
- `v2_selective_throttle_matrix_narrow_20260602_01`.

## Implementation

- Extended `daily_research.path_policy.v2_bad_month_attribution` with local state buckets:
  - previous-close-derived recent return, default window `5`;
  - previous-close-derived volatility, default window `20`;
  - previous-day reversal.
- Added output CSVs for:
  - recent-return bucket summary;
  - volatility bucket summary;
  - reversal bucket summary;
  - score x volatility bucket summary;
  - score x reversal bucket summary.
- Boundary:
  - attribution remains approximate: previous target weight x close-to-close stock return;
  - it is a research diagnostic, not a promotion artifact;
  - it does not write active execution artifacts.

## Evidence

### 2024-03

- Run tag: `v2_post_throttle_bad_month_attribution_20240602_m202403`.
- Source candidate monthly excess return: `-0.024138`.
- Portfolio monthly return: `-0.013998`.
- Approx contribution rows: `761`; symbols: `91`; dates: `21`.
- Most negative coarse buckets:
  - score `score_q_4`: contribution `-0.006582`;
  - weight `weight_q_4`: contribution `-0.002505`;
  - recent return `recent_return_q_4`: contribution `-0.009340`;
  - volatility `volatility_q_5`: contribution `-0.012822`;
  - reversal `reversal_q_1`: contribution `-0.019500`.
- Most negative interaction buckets:
  - `score_q_5 x volatility_q_1`: `-0.008560`;
  - `score_q_2 x volatility_q_5`: `-0.007845`;
  - `score_q_5 x reversal_q_3`: `-0.008426`;
  - `score_q_5 x reversal_q_1`: `-0.007178`.

### 2024-06

- Run tag: `v2_post_throttle_bad_month_attribution_20240602_m202406`.
- Source candidate monthly excess return: `-0.027893`.
- Portfolio monthly return: `-0.060261`.
- Approx contribution rows: `664`; symbols: inferred from contribution CSV; dates: inferred from contribution CSV.
- Most negative coarse buckets:
  - score `score_q_4`: contribution `-0.019262`;
  - score `score_q_3`: contribution `-0.017864`;
  - weight `weight_q_3`: contribution `-0.031617`;
  - recent return `recent_return_q_5`: contribution `-0.019505`;
  - recent return `recent_return_q_4`: contribution `-0.016583`;
  - volatility `volatility_q_1`: contribution `-0.023899`;
  - reversal `reversal_q_2`: contribution `-0.015167`.
- Most negative interaction buckets:
  - `score_q_3 x volatility_q_5`: `-0.017159`;
  - `score_q_5 x volatility_q_1`: `-0.014353`;
  - `score_q_5 x reversal_q_2`: `-0.014762`.

### 2024-08

- Run tag: `v2_post_throttle_bad_month_attribution_20240602_m202408`.
- Source candidate monthly excess return: `-0.032376`.
- Portfolio monthly return: `-0.070483`.
- Approx contribution rows: `787`; symbols: inferred from contribution CSV; dates: inferred from contribution CSV.
- Most negative coarse buckets:
  - score `score_q_5`: contribution `-0.031110`;
  - weight `weight_q_5`: contribution `-0.037675`;
  - recent return `recent_return_q_1`: contribution `-0.038043`;
  - volatility `volatility_q_5`: contribution `-0.038263`;
  - volatility `volatility_q_4`: contribution `-0.021312`;
  - reversal `reversal_q_5`: contribution `-0.026824`;
  - reversal `reversal_q_4`: contribution `-0.026123`.
- Most negative interaction buckets:
  - `score_q_4 x volatility_q_5`: `-0.026278`;
  - `score_q_5 x volatility_q_4`: `-0.013152`;
  - `score_q_5 x reversal_q_4`: `-0.027145`.

## Interpretation

- The remaining negative months after selective throttle are local state failures, not a general lack of signal.
- `2024-08` is still the clearest high-confidence / high-weight failure: high score and high weight buckets lose heavily, with high-volatility and positive-reversal interaction buckets contributing the largest losses.
- `2024-06` is not fixed by high-volatility-only throttling. The loss sits in mid/high score buckets, mid weights, high recent-return buckets, and a low-volatility bucket that likely acts like a stale calm-state exposure trap.
- `2024-03` is mixed: high-volatility and weak-reversal buckets are negative, but the month also has benchmark / cost / timing effects because close-to-close held-position contribution is not as negative as the portfolio excess return.
- This supports moving from global path-state sizing to local controls:
  - per-symbol volatility bucket caps;
  - score x reversal bucket sizing;
  - score x volatility bucket sizing;
  - high-score high-weight caps when local reversal / volatility flags are adverse;
  - weight redistribution within the candidate list instead of simple gross exposure shrinkage.

## Next Allowed Actions

- Build a narrow local cap matrix on top of the best selective-throttle candidate.
- Candidate local rules should test:
  - reduce `score_q_5 x reversal_q_4/q_5`;
  - reduce `score_q_4/q_5 x volatility_q_4/q_5`;
  - reduce high recent-return buckets in June-like states;
  - keep gross exposure near the source candidate by redistributing released weight to lower-risk selected names.
- Do not promote to live/default from this attribution.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_bad_month_attribution.py -q`: `2 passed`.
- `python -m daily_research.path_policy.v2_bad_month_attribution --run-tag v2_post_throttle_bad_month_attribution_20240602_m202403 --matrix-run-tag v2_selective_throttle_matrix_narrow_20260602_01 --variant-id h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80 --month 2024-03 --json`: completed.
- `python -m daily_research.path_policy.v2_bad_month_attribution --run-tag v2_post_throttle_bad_month_attribution_20240602_m202406 --matrix-run-tag v2_selective_throttle_matrix_narrow_20260602_01 --variant-id h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80 --month 2024-06`: completed.
- `python -m daily_research.path_policy.v2_bad_month_attribution --run-tag v2_post_throttle_bad_month_attribution_20240602_m202408 --matrix-run-tag v2_selective_throttle_matrix_narrow_20260602_01 --variant-id h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80 --month 2024-08`: completed.
