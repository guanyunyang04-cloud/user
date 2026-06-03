# Daily Research V2 Candidate Review Matrix - 2026-06-02

## Summary

- Status: `v2_candidate_review_matrix / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_candidate_review_matrix`.
- Source bridge: `v2_score_backtest_bridge_20260602_01`.
- Matrix run tag: `v2_candidate_review_matrix_20260602_01`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Score panel: `daily_research/output/path_policy/studies/v2_score_backtest_bridge_20260602_01/v2_ensemble_score_panel_test_for_backtest.csv`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Run Tags

- `v2_candidate_review_matrix_20260602_01`.
- `v2_score_backtest_bridge_20260602_01`.

## Implemented Candidate Matrix

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_candidate_review_matrix --json`
  - `python -m daily_research.path_policy.v2_candidate_review_matrix --run-backtests --json`
- Added output artifacts under:
  - `daily_research/output/path_policy/studies/v2_candidate_review_matrix_20260602_01/`
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_candidate_review_matrix.py`.
- Fixed score backtest artifact collection so candidate review can read:
  - `annual_return_cost_drag`;
  - `excess_annual_return_cost_drag`.

## Candidate Gate

- `monthly_positive_ratio >= 0.75`.
- `negative_month_count <= 2`.
- `max_drawdown >= -0.25`.
- `excess_sharpe >= 1.0`.
- no `deep_bad_month` flag.
- `annual_return_cost_drag <= 0.20`.

## Matrix Evidence

- Matrix variants: `3`.
- Completed shared backtests: `3`.
- Promotion-review eligible variants: `0`.
- Tested variants:
  - `h10_mw080_rb5d_all_c3_7_10_regime_off`;
  - `h20_mw080_rb5d_all_c3_7_10_regime_off`;
  - `h30_mw080_rb5d_all_c3_7_10_regime_off`.
- Output report:
  - `daily_research/output/path_policy/studies/v2_candidate_review_matrix_20260602_01/v2_candidate_review_matrix_report.json`
  - `daily_research/output/path_policy/studies/v2_candidate_review_matrix_20260602_01/v2_candidate_review_matrix_summary.csv`
  - `daily_research/output/path_policy/studies/v2_candidate_review_matrix_20260602_01/v2_candidate_review_matrix_report.md`

## Best Completed Variant

- Variant: `h20_mw080_rb5d_all_c3_7_10_regime_off`.
- Annual return: `0.5451464883423418`.
- Excess annual return: `0.267906232776264`.
- Excess Sharpe: `0.9527468587055732`.
- Max drawdown: `-0.26020174053699163`.
- Annual return cost drag: `0.14983058049201214`.
- Positive month ratio: `0.6363636363636364`.
- Negative month count: `4`.
- Issue flags: `deep_bad_month`, `concentrated_positive_months`.

## Interpretation

- The execution-candidate review path is now stronger than a single score bridge: it can compare score-to-weight mappings across explicit candidate parameters and write machine-readable matrix evidence.
- The first narrow matrix shows that simply changing holding count from `10` to `20` to `30` does not repair the execution-candidate blocker.
- The best candidate remains the earlier `20` holding count shape, but it is still not promotion-review eligible because it fails monthly stability, negative month count, drawdown, excess Sharpe, and bad-month checks.
- Cost drag is visible but not the primary blocker in this narrow matrix; all three variants stay below the `0.20` annual return cost-drag gate after the collector fix.
- The current blocker has moved from "bridge not available" to "candidate mapping and risk/month-stability design not good enough."

## Boundaries

- Do not promote to live/default from this matrix.
- Do not write or rebuild `daily_research/output/active_execution_strategy.json`.
- Do not treat the score panel as a production target-weight panel.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Next Allowed Actions

- Expand the matrix along dimensions more likely to affect monthly stability:
  - rebalance frequency `3d/5d/10d`;
  - max weight `0.04/0.06/0.08`;
  - market-regime or soft-state overlays;
  - score thresholding or confidence filtering;
  - bad-month attribution by month, sector, liquidity bucket, and horizon bucket.
- Keep `mh_v2_reset_tradeable_mainboard_anchor_20260601_01` as the pass-grade research baseline.
- Continue feature/model/loss work only against explicit v2 dataset, pool, and run tags.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_candidate_review_matrix.py daily_research/path_policy/tests/test_v2_score_backtest_bridge.py tools/brain/tests/test_evidence_registry.py::BrainEvidenceRegistryTest::test_registry_indexes_daily_research_v2_score_backtest_bridge -q`: `11 passed`.
- `python -m daily_research.path_policy.v2_candidate_review_matrix --run-backtests --holding-counts 10,20,30 --max-weights 0.08 --rebalance-freqs 5d --rebalance-offset-modes all --transaction-cost-bps-values 3 --slippage-bps-values 7 --sell-tax-bps-values 10 --json`: completed, `3` backtests completed, `0` promotion-review eligible variants.
- `git diff -- daily_research/output/active_execution_strategy.json`: must remain empty.
