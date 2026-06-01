# Alpha Multi-Horizon Research Rebuild Infra V2 - 2026-06-01

## Verdict

- Status: `new_lineage_research_rebuild_completed / baseline_anchor_failed_continuation_gate / shadow-only / research-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Study family: `new_lineage_research_rebuild`.
- This is a new local research lineage after `daily_research/output/` and `daily_research/cache/` loss. It is not a recovery of old Stage 2.8 / Stage 3G run payloads.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- Boundary: no live/default payload, no paper/broker payload, no active execution strategy rebuild, no promotion, and no production root change.

## Rebuilt Data Lake

- Provider route: BaoStock-first `research_rebuild_minimal_free`.
- Policy input bundle: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Audit artifact: `daily_research/output/research_data_lake/rebuild_policy_input_audit_20260531.json`.
- Audit verdict: `usable_with_warnings`, with no blockers.
- Warning: `feature_panel_mismatch` on optional sidecar-style feature panels such as industry/security status snapshots. This does not affect the raw no-alpha baseline feature profile used here.
- Market coverage: `2017-01-03 -> 2024-12-31`, `1943` trading dates, `5118` market symbols, `9944274` market rows.
- Benchmark: `000300.SH`, `1943` close rows and `1943` open rows.
- Sidecar dataset ids include `data_platform_industry_concept__11f4d7e34893ebf6edf60c3c`, `data_platform_security_status__1e46db76419d92a204332401`, `data_platform_trading_calendar__1e998f0b04c696089fab410d`, and `data_platform_universe_snapshot__b8bc799a035012ffd404aaed`.

## Rebuilt Views

- Rolling pool view: `policy_pool_view__8f9367ce305548e3ecd04feb`.
- Pool semantics: `rolling_liquidity / rolling_liquid500`.
- Pool source binding: `source_market_dataset_id=policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool QA: `status=computed`, `date_bounds=2018-01-02 -> 2024-12-31`, `membership_rows=1699`, `membership_symbols=5118`, `true_cells=849500`, `universe_size=3668`, `member_count_min=500`, `member_count_median=500`, `member_count_max=500`, `zero_member_days=0`, `rebalance_count=81`, `average_rebalance_turnover=0.435509`.
- Sector/board view: `policy_sector_board_view__49ffa8713d1b322a8c19b72b`.
- Sector/board semantics: `latest_static_snapshot / sector_board_baostock_industry_latest_static`.
- Industry source: lake sidecar `data_platform_industry_concept__11f4d7e34893ebf6edf60c3c`.
- Industry coverage: `industry_rows=5118`, `coverage_ratio=1.0`.
- Board coverage: `board_count=0`, `board_membership_rows=0`, `board_coverage.status=empty_explicit`.
- Interpretation: BaoStock industry is usable as sector metadata; board/concept coverage is explicitly empty rather than fabricated.

## Memmap Anchor

- Seed7 manifest: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed7_20260531_01/forecast_dataset_manifest.json`.
- Validator artifact: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed7_20260531_01/rebuild_memmap_validation.json`.
- Validator status: `ok`, blockers `[]`.
- Source binding: `source_market_dataset_id=policy_input_bundle__45e3d8c059ba718426a9f887`, `source_pool_view_id=policy_pool_view__8f9367ce305548e3ecd04feb`, `source_pool_view_kind=rolling_liquidity`, `source_pool_view_name=rolling_liquid500`.
- Scope: `universe_size=3668`, `feature_count=116`, `feature_store_shape=[1699,3668,116]`.
- Sample rows: total `638527`, train `434939`, validation `100006`, test `103582`.
- Seed11 and seed19 reused the seed7 manifest and did not rebuild the full memmap.

## Run Tags

- `mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed7_20260531_01`
- `mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed11_20260531_01`
- `mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed19_20260531_01`
- `mh_rebuild_infra_v2_fullpool_anchor_20260531_01`

## Diagnostics

- Target calibration artifact: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_anchor_20260531_01/rebuild_target_calibration_audit.json`.
- Target calibration result: `gate_a_passed=true`.
- Decision diagnostics artifact: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_anchor_20260531_01/rebuild_decision_score_diagnostics.json`.
- Decision diagnostics hint: `no_selection_only_fix_found; inspect target_definition_or_input_regime_contamination`.
- Aggregate comparison artifact root: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_anchor_20260531_01/`.
- Aggregate research verdict: `needs_input_redesign`.
- Aggregate next-stage file says default next action would be `run_stage2_horizon_grid_calibration`, but this note applies the stricter rebuild continuation gate below before allowing follow-on research.

## Gate Evidence

Test `pred_decision_score` rows:

| Seed | Rank IC | Spread | Hit lift | Monthly positive rate | Negative months | Gate note |
|---:|---:|---:|---:|---:|---:|---|
| 7 | `0.142275` | `0.055043` | `0.022730` | `0.818182` | `2` | pass |
| 11 | `0.051035` | `0.018477` | `-0.000353` | `0.727273` | `3` | fail |
| 19 | `0.079435` | `0.028512` | `0.015674` | `0.818182` | `2` | pass |

Continuation gate requirements were: all three seeds test rank IC, spread, and hit lift positive; mean monthly positive rate `>=0.75`; max negative months `<=2`.

Result:

- Rank IC and spread were positive for all three seeds.
- Seed11 hit lift was slightly negative.
- Mean monthly positive rate was `0.787879`, which passes the mean threshold.
- Max negative months was `3`, which fails the threshold.
- Therefore the new rebuild baseline anchor fails the continuation gate.

## Interpretation

- The research substrate rebuild succeeded: a BaoStock-first policy input bundle, full rolling_liquid500 pool view, sector/board view, full memmap anchor, three seed trainings, calibration, diagnostics, and comparison artifacts now exist locally under `H:\quant_project`.
- The new lineage is not directly comparable as an old payload restore. Old Stage 2.8 / Stage 3G conclusions remain brain-confirmed historical evidence, but their deleted physical run payloads were not reconstructed.
- The failed continuation gate is a model/rebuild-anchor quality result, not a provider or data lake blocker. It says the new anchor should not be used to resume broad Stage 3 expansion without first diagnosing seed11 instability and the input/target regime issue surfaced by diagnostics.
- This result does not authorize active execution changes, promotion, replay, paper trading, live/default changes, or any production artifact update.

## Next Allowed Actions

- Keep active execution unchanged.
- Treat `policy_input_bundle__45e3d8c059ba718426a9f887` and `policy_pool_view__8f9367ce305548e3ecd04feb` as the new local research substrate for diagnostics only.
- Before continuing multi-horizon expansion, inspect why seed11 failed hit lift and monthly stability under the new lineage.
- Prefer narrow diagnostics around target definition, regime split, feature scaling/interaction, and hit-lift-preserving rank regularization.
- Do not run broad Stage 3 architecture/input expansion from this anchor until a follow-up baseline or diagnostic gate passes.
