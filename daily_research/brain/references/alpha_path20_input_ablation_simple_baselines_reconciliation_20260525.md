# Alpha Path20 Input Ablation / Simple Baselines Reconciliation - 2026-05-25

## Scope
- Status: `completed forecast evidence reconciliation / research / shadow-only / no promotion`.
- Purpose: register three completed 2026-05-22 Path20 output runs that appeared in frontier as `unregistered_latest_run_tags`.
- Mainline boundary: these studies belong to the historical `alpha_path20_neural_policy_v1` / Path20 namespace. They do not override the current `alpha_multi_horizon_utility_policy_v1` naming migration.
- Active artifact boundary: `daily_research/output/active_execution_strategy.json` remains unchanged and is not authorized for change by this reference.

## Evidence Sources
- `path20_gate_b_simple_baselines_liquid500_du_cost20_hit10_dd010_20260522_01`
  - Summary: `daily_research/output/path_policy/studies/path20_gate_b_simple_baselines_liquid500_du_cost20_hit10_dd010_20260522_01/study_summary.json`
- `path20_input_ablation_no_alpha_liquid500_du_cost20_hit10_dd010_20260522_01`
  - Summary: `daily_research/output/path_policy/studies/path20_input_ablation_no_alpha_liquid500_du_cost20_hit10_dd010_20260522_01/study_summary.json`
- `path20_input_ablation_no_sector_static_liquid500_du_cost20_hit10_dd010_20260522_01`
  - Summary: `daily_research/output/path_policy/studies/path20_input_ablation_no_sector_static_liquid500_du_cost20_hit10_dd010_20260522_01/study_summary.json`

## Shared Setup
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Pool view: `policy_pool_view__c11400fa72ad263f3d1eecfa` (`liquid500`).
- Sector / board view: `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`.
- Role years: train `2019-2022`, validation `2023`, test `2024`.
- Sample counts: train `456984`, validation `108920`, test `109972`, total `675876`.
- Label semantics: `next_open_entry_to_future_open`; cumulative horizons `1/3/5/10/20d`.
- Decision utility contract: cost `20 bps`, hit threshold `10 bps`, drawdown penalty `0.10`.

## Results
| Study | Purpose | Feature / model | Test rank IC 20d | Test spread 20d | Test decision rank IC | Test decision spread | Test hit lift | Gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `path20_gate_b_simple_baselines_liquid500_du_cost20_hit10_dd010_20260522_01` | simple baseline sanity check | `raw_kline_context_sector_v1`; selected `linear_last_day`, seed `7` | 0.110627 | 0.030312 | 0.050982 | 0.018597 | 0.006677 | `forecast_test_confirmed` |
| `path20_input_ablation_no_alpha_liquid500_du_cost20_hit10_dd010_20260522_01` | remove alpha prior from context | `raw_kline_context_no_alpha_prior_v1`; `gru_sequence_static_context`, seed `7` | 0.094436 | 0.027071 | 0.066473 | 0.021358 | 0.021311 | `forecast_test_confirmed` |
| `path20_input_ablation_no_sector_static_liquid500_du_cost20_hit10_dd010_20260522_01` | remove industry / board static fields | `raw_kline_context_v1`; `gru_sequence_static_context`, seed `7` | 0.117113 | 0.023779 | 0.023819 | 0.000267 | 0.014139 | `forecast_test_confirmed` |

## Interpretation
- Fact: all three studies completed and passed validation/test decision utility profile gates in their own `study_summary.json` files.
- Fact: no-alpha-prior GRU kept positive test decision rank, spread, and hit lift, so the Path20 signal is not explained only by legacy alpha-prior replay.
- Fact: removing industry / board static fields preserved strong 20d forecast rank and spread, but nearly eliminated aggregate test decision spread.
- Fact: simple baselines remained surprisingly competitive; `linear_last_day` was selected over `dlinear_sequence` by validation and produced positive 20d and decision metrics on test.
- Inference: input/static context and target selection interact materially. The useful next question is not "more capacity first", but whether no-alpha-prior context, sector/static context, and simple baselines expose a more stable decision utility target.

## Boundaries
- `shadow_only=true`.
- `promotion_allowed=false`.
- `active_execution_strategy_expected_diff=none`.
- These are forecast-stage / decision-utility research results only.
- Do not use this reference to authorize liquid800 expansion, allocator/replay, production retrain, trade-plan bridge, paper/live execution, broker action, active manifest mutation, or live/default change.

## Next Allowed Actions
- Keep these three tags registered as completed Path20 research evidence.
- Use them as input-ablation and simple-baseline context for later `alpha_multi_horizon_utility_policy_v1` target/profile comparisons.
- Before any expansion, compare no-alpha-prior, sector/static, and simple-baseline behavior under the current multi-horizon target framing with explicit run tags and active-artifact guard.
