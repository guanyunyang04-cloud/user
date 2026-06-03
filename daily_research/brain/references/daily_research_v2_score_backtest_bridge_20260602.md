# Daily Research V2 Score Backtest Bridge - 2026-06-02

## Summary

- Status: `daily_research_v2_score_backtest_bridge / evidence_grade_bridge / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_score_backtest_bridge`.
- Bridge run tag: `v2_score_backtest_bridge_20260602_01`.
- Source anchor: `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- Source seed tags:
  - `mh_v2_reset_tradeable_mainboard_seed7_20260601_01`;
  - `mh_v2_reset_tradeable_mainboard_seed11_20260601_01`;
  - `mh_v2_reset_tradeable_mainboard_seed19_20260601_01`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_amount_checked`.
- Score column: `pred_decision_score`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Implemented Bridge

- Added research-only CLI:
  - `python -m daily_research.path_policy.v2_score_backtest_bridge --json`
  - `python -m daily_research.path_policy.v2_score_backtest_bridge --run-backtest --json`
- Added output artifacts under:
  - `daily_research/output/path_policy/studies/v2_score_backtest_bridge_20260602_01/`
- Added shared engine support:
  - `daily_research.baseline.backtest_external_score_panel` now accepts `--data-source lake`.
  - Score-panel mode now honors `--rebalance-offset-mode all`; earlier it silently collapsed to a single offset.
- Added focused tests:
  - `daily_research/path_policy/tests/test_v2_score_backtest_bridge.py`.

## Score Panel Evidence

- Seed score panel:
  - `daily_research/output/path_policy/studies/v2_score_backtest_bridge_20260602_01/v2_seed_score_panel_test.csv`
  - rows `313413`;
  - dates `2024-01-02 -> 2024-11-18`;
  - symbols `1214`.
- Ensemble score panel:
  - `daily_research/output/path_policy/studies/v2_score_backtest_bridge_20260602_01/v2_ensemble_score_panel_test.csv`
  - rows `104471`;
  - date count `211`;
  - symbols `1214`;
  - ensemble policy `mean_by_date_symbol`, require all `3` seeds.
- Backtest input score panel:
  - `daily_research/output/path_policy/studies/v2_score_backtest_bridge_20260602_01/v2_ensemble_score_panel_test_for_backtest.csv`
- Ensemble diagnostics:
  - future rank IC mean `0.11249056158798118`;
  - future top-bottom spread mean `0.03997717769993395`;
  - top20 hit lift mean `0.009483906454258636`;
  - seed score Spearman mean `0.7572135165771278`;
  - seed top20 pairwise overlap mean `0.2706161137440758`.

## Shared Candidate Backtest

- Shared backtest output:
  - `daily_research/output/path_policy/studies/v2_score_backtest_bridge_20260602_01/shared_backtest/v2_score_backtest_bridge_20260602_01_shared_engine/`
- Metrics:
  - annual return `0.5451464883423418`;
  - excess annual return `0.267906232776264`;
  - excess Sharpe `0.9527468587055732`;
  - max drawdown `-0.26020174053699163`;
  - avg turnover `0.24520370970066055`;
  - total trading cost return `0.07673954187526336`;
  - rebalance mode `5d / all offsets / 5 sleeves`;
  - holding count target `20`;
  - max weight `0.08`;
  - costs: transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`;
  - market regime filter `off`.
- Monthly diagnostics:
  - month count `11`;
  - positive month count `7`;
  - negative month count `4`;
  - positive month ratio `0.6363636363636364`;
  - worst month `-0.09823511701170529`;
  - issue flags: `deep_bad_month`, `concentrated_positive_months`.

## Interpretation

- The v2 score-to-candidate bridge is now file-backed and executable through the shared candidate backtest engine.
- This closes the first missing link between evidence-grade v2 model output and execution-candidate review inputs.
- The candidate backtest is promising on excess return and excess Sharpe, but it is not promotion-grade:
  - monthly positive ratio is below the research gate style threshold;
  - negative months are `4`;
  - month diagnostics flag deep bad month and concentrated positive months.
- Therefore this is a research candidate bridge, not a live/default, paper/broker, production root, or active manifest promotion.

## Boundaries

- Do not write or promote `daily_research/output/active_execution_strategy.json`.
- Do not treat the score panel as a production target-weight panel.
- Do not infer execution readiness from this single candidate mapping.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.
- Next execution-facing work must stay research-only until a candidate review matrix passes.

## Next Allowed Actions

- Build a v2 candidate review matrix across:
  - rebalance frequency and offset mode;
  - holding count and max weight;
  - cost/slippage assumptions;
  - optional risk overlays;
  - strict v2 pool versus traditional-PIT comparison pool.
- Add explicit candidate gate criteria for score-to-weight mappings:
  - monthly positive ratio;
  - negative month count;
  - drawdown;
  - turnover/cost drag;
  - concentration and bad-month attribution.
- Continue v2 feature/model improvements against `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- Keep execution frozen until research gate, candidate backtest bridge, and execution-candidate review all close.

## Verification

- `python -m pytest daily_research/path_policy/tests/test_v2_score_backtest_bridge.py -q`: `7 passed`.
- `python -m daily_research.path_policy.v2_score_backtest_bridge --run-backtest --json`: completed, shared backtest returncode `0`.
- `git diff -- daily_research/output/active_execution_strategy.json`: must remain empty.
