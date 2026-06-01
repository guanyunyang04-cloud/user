# Alpha Multi-Horizon Rebuild Lineage Difference Audit - 2026-06-01

## Verdict

- Status: `rebuild_lineage_diff_audit_completed / research-only / shadow-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Source artifact: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_anchor_20260531_01/rebuild_lineage_diff_audit.json`.
- Human report: `daily_research/output/path_policy/studies/mh_rebuild_infra_v2_fullpool_anchor_20260531_01/rebuild_lineage_diff_audit.md`.
- Supporting exports: `rebuild_seed_monthly_spread.csv` and `rebuild_feature_columns.csv` under the same anchor root.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` remains unavailable locally and unchanged by this audit.

## Scope

- The audit compares the file-backed new rebuild lineage against historical Stage 2.8 / Stage 3G / short_v5b evidence boundaries.
- It does not train a model, promote a payload, rebuild a production root, write an active manifest, or authorize paper/live/broker work.
- Historical old Stage 2.8 / Stage 3G conclusions remain brain-confirmed text evidence, but their local file-backed payloads are not available for replay.

## Key File-Backed New-Lineage Facts

- New dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- New pool: `policy_pool_view__8f9367ce305548e3ecd04feb`.
- Feature profile: `raw_kline_context_no_alpha_prior_v1`.
- Feature store shape: `[1699,3668,116]`.
- Sample rows: total `638527`, train `434939`, validation `100006`, test `103582`.
- Feature count: `116`.
- Stock count / universe in memmap: `3668`.
- Current provider route: BaoStock-first `research_rebuild_minimal_free`.

## Historical Comparison Boundary

- Old Stage 2.8 dataset: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Old Stage 2.8 recorded full-pool manifest properties: `universe_size=2430`, `train_rows=452034`, `feature_store_shape=[1699,2430,156]`.
- Old Stage 2.8 `target_norm_head_constraint_v1` recorded gate facts: min rank IC `0.083206`, min spread `0.028098`, min hit lift `0.010771`, mean monthly positive rate `0.878788`, max negative months `2`, 30d concentration `0.776934`.
- Local old Stage 2.8 payload root exists: `false`.
- Local old Stage 3G payload root exists: `false`.
- Local short_v5b production root exists: `false`.
- Local active execution manifest exists: `false`.

## New-Lineage Gate Facts

Test `pred_decision_score` rows:

| Seed | Rank IC | Spread | Hit lift | Monthly positive rate | Negative months |
|---:|---:|---:|---:|---:|---:|
| 7 | `0.142275` | `0.055043` | `0.022730` | `0.818182` | `2` |
| 11 | `0.051035` | `0.018477` | `-0.000353` | `0.727273` | `3` |
| 19 | `0.079435` | `0.028512` | `0.015674` | `0.818182` | `2` |

Aggregate new-lineage test facts:

- Min rank IC: `0.051035`.
- Min spread: `0.018477`.
- Min hit lift: `-0.000353`.
- Max negative months: `3`.
- Seed11 negative test months in the audit monthly spread CSV: `2024-03`, `2024-10`, `2024-11`.

## Root-Cause Ranking

1. `new_lineage_not_payload_restore` - high confidence. The new `source_market_dataset_id` differs from old Stage 2.8, and old payload roots are unavailable.
2. `universe_and_pool_composition_shift` - high confidence. Old universe size was `2430`; new memmap stock count is `3668`.
3. `feature_schema_shift` - high confidence. Old feature store shape `[1699,2430,156]` changed to `[1699,3668,116]`.
4. `seed11_hit_and_month_stability_failure` - high confidence. Seed11 test hit lift is slightly negative and max negative months is `3`.
5. `selection_only_fix_unlikely` - medium-high confidence. Existing decision-score diagnostics say `no_selection_only_fix_found; inspect target_definition_or_input_regime_contamination`.
6. `provider_price_label_regime_shift` - medium confidence. BaoStock-first provider route changed raw market / benchmark / sidecar source semantics; old raw payload is unavailable for direct OHLCV / label diff.

## Interpretation

- The result difference is primarily a substrate mismatch, not a direct same-payload model regression.
- The new rebuild is useful as a diagnostics substrate, but it is not an old Stage 2.8 / Stage 3G replay.
- The new anchor still contains positive rank/spread signal, but it fails continuation because hit lift and month stability weaken in seed11.
- Since the decision-score diagnostic already found no selection-only fix, the next research should focus on data/target/input-regime differences rather than immediately expanding architectures.

## Next Diagnostics

- Compare feature columns and missing old feature groups once old payload or feature manifest is recovered.
- Compare pool membership overlap by date if old Stage 2.8 membership frame is recovered.
- Run common-symbol/date OHLCV and next-open label diff if old `policy_input_bundle__7c8f58d851bce8179e1e9e2d` payload is recovered.
- On the new lineage, isolate seed11 negative months by regime, history bucket, horizon, hit base rate, and liquidity/price buckets.
- Before any execution rebuild, define a same-protocol backtest bridge versus short_v5b with common pool, costs, rebalance policy, benchmark, date splits, and score/target-weight panel semantics.

## Boundary

- Do not unfreeze execution from this audit alone.
- Do not claim new multi-horizon is better or worse than short_v5b until same-protocol candidate backtest exists.
- Do not claim old Stage 2.8 is invalid; its conclusion remains brain-confirmed historical evidence, while file-backed replay is unavailable.
- Do not write or reconstruct `daily_research/output/active_execution_strategy.json` from text evidence.
