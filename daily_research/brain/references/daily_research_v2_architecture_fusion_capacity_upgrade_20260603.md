# Daily Research V2 Architecture Fusion Capacity Upgrade

Date: `2026-06-03`

## Summary

- Status: `architecture_fusion_capacity_upgrade / tier1_three_seed_reviewed / tier2_seed7_diagnostic_complete / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `v2_arch_fusion_scout`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- This is not promotion evidence. Tier 1 has completed three-seed forecast comparison but failed the "not worse than GRU-static" expansion gate. Tier 2 has completed code, task-list, smoke, and one formal seed7 diagnostic only; seed11/19 were intentionally stopped before completion after seed7 underperformed the GRU-static seed7 baseline.

## Implementation

- Added forecast model family `regime_routed_multi_expert_horizon_v1`.
- Registered the new family in `daily_research.path_policy.forecast_training.FORECAST_MODEL_FAMILIES`.
- Kept output profile compatibility with `decision_utility_v1`; for 30d plus horizons `(1,2,3,5,8,10,15,20,30)`, output remains the existing 183-dimensional contract.
- Extended `v2_arch_fusion_scout` so default Tier 1 remains `hybrid_expert_fusion_static_context`, while Tier 2 can be selected with `--model-family regime_routed_multi_expert_horizon_v1`.
- Tier 1 task-list capacity estimate is `4,628,344` parameters; Tier 2 task-list capacity estimate is `11,021,635` parameters, inside the planned `8m_to_15m` band.
- Generated research-only task lists:
  - `daily_research/output/path_policy/studies/v2_arch_fusion_scout_20260603_01/v2_arch_fusion_training_task_list.json`;
  - `daily_research/output/path_policy/studies/v2_arch_fusion_scout_regime_routed_multi_expert_20260603_01/v2_arch_fusion_training_task_list.json`.

## Architecture

Tier 2 `regime_routed_multi_expert_horizon_v1` includes:

- GRU expert for path continuity and short/medium momentum-reversal structure.
- Patch Transformer expert for multiscale temporal events and non-local dependencies.
- DLinear-style expert for trend/seasonal/residual low-frequency structure.
- Stock mixer expert for cross-sectional same-date relative ranking and crowding context.
- Local-state expert for high volatility, runup, reversal, and drawdown state.
- Static/context encoder for symbol, exchange, industry, liquidity bucket, and price bucket.
- Regime router producing 5 expert weights plus diagnostics: `router_weights`, `router_entropy`, `expert_token_diversity`, and `bad_state_intensity`.

The loss contract keeps `horizon_30d_soft_penalty_v1` and adds diagnostic-only auxiliary terms that activate only when router diagnostics exist:

- `router_entropy_floor`;
- `expert_diversity`;
- `bad_state_calibration`.

## Verification

- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_models.py -q`: `10 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_v2_arch_fusion_scout.py -q`: `7 passed`.
- Focused Tier 2 forecast-training checks passed:
  - `test_forecast_model_families_emit_path20_sequence_contract`;
  - `test_make_forecast_model_registers_regime_routed_multi_expert_horizon_v1`.
- Full `daily_research/path_policy/tests/test_forecast_training.py` command exceeded the local timeout because existing micro-training tests are slow; no failing assertion was observed and the residual pytest process was stopped.
- Tier 2 forward checks cover 2D input fallback, 3D sequence input, 4D date-level cross-section input, finite router diagnostics, and router weights summing to 1.
- Tier 2 smoke training completed in a temporary directory with `regime_routed_multi_expert_horizon_v1`, `decision_utility_v1`, `horizon_30d_soft_penalty_v1`, and prediction/checkpoint contract writes.
- Tier 1 formal run completed for seeds `7,11,19`.
- Tier 2 formal diagnostic run completed for seed `7`; seed11/19 were stopped before completion by design after seed7 review.
- `git diff -- daily_research/output/active_execution_strategy.json`: clean.

## Tier 1 Evidence

- Run tag: `v2_arch_fusion_scout_20260603_01`.
- Model family: `hybrid_expert_fusion_static_context`.
- Seed study tags:
  - `mh_v2_arch_fusion_hybrid_expert_seed7_20260603_01`;
  - `mh_v2_arch_fusion_hybrid_expert_seed11_20260603_01`;
  - `mh_v2_arch_fusion_hybrid_expert_seed19_20260603_01`.
- Comparison artifact: `daily_research/output/path_policy/studies/v2_arch_fusion_scout_20260603_01/tier1_hybrid_vs_gru_static_comparison/profile_aggregate.csv`.
- Tier 1 three-seed `pred_decision_score` test aggregate:
  - rank IC mean `0.099503`, min `0.071850`;
  - spread mean `0.042286`, min `0.033544`;
  - hit lift mean `0.009715`, min `0.005282`;
  - monthly positive rate mean `0.848485`;
  - negative month count max `2`;
  - 30d concentration mean `0.017029`.
- GRU-static three-seed baseline on the same comparison:
  - rank IC mean `0.114866`, min `0.103705`;
  - spread mean `0.038425`, min `0.034908`;
  - hit lift mean `0.023851`, min `0.015980`;
  - monthly positive rate mean `0.878788`;
  - negative month count max `2`;
  - 30d concentration mean `0.042104`.
- Decision: Tier 1 remains research-only and does not enter score-backtest bridge. It improves some spread/concentration aspects but is weaker on rank IC, hit lift, and seed stability relative to the current GRU-static horizon soft-penalty branch.

## Tier 2 Evidence

- Run tag: `v2_arch_fusion_scout_regime_routed_multi_expert_20260603_01`.
- Model family: `regime_routed_multi_expert_horizon_v1`.
- Task-list parameter estimate: `11,021,635`.
- Formal completed diagnostic seed: `mh_v2_arch_fusion_regime_routed_multi_expert_seed7_20260603_01`.
- Seed7 status: `forecast_test_confirmed`, stopped by early stopping after `8` epochs; best epoch `2`.
- Resource observation on RTX 2060 6GB: GPU memory was approximately `5.9GB / 6.1GB`, epoch time about `820-987` seconds, making full three-seed confirmation expensive for the current configuration.
- Same-seed `pred_decision_score` test comparison:
  - GRU-static seed7: rank IC `0.103705`, spread `0.038083`, hit lift `0.015980`, monthly positive rate `0.818182`, negative months `2`, worst month `-0.026383`.
  - Tier 1 hybrid seed7: rank IC `0.104936`, spread `0.040553`, hit lift `0.011881`, monthly positive rate `0.909091`, negative months `1`, worst month `-0.015987`.
  - Tier 2 regime-routed seed7: rank IC `0.072883`, spread `0.026168`, hit lift `0.012900`, monthly positive rate `0.909091`, negative months `1`, worst month `-0.037467`.
- Tier 2 diagnostic finding: the router branch strongly suppresses 30d selection concentration (`~0.000029` test 30d concentration), but the main score mapping loses rank/spread versus GRU-static. Some alternative score variants, such as `short_horizon_blend`, improve hit lift but introduce poor monthly quality and deep worst-month behavior, so they are not bridge candidates.
- Decision: stop Tier 2 seed11/19 for this configuration. Continue with router/score-mapping diagnostics or a smaller/calibrated Tier 2 revision before any three-seed confirmation attempt.

## Verdict

- The architecture-capacity objection is now addressable by a staged, auditable fusion chain rather than by further GRU-only loss/throttle tuning.
- Tier 1 higher-capacity fusion did not justify score-backtest bridge expansion.
- Tier 2 organic multi-expert fusion is implemented and trainable, but the first formal seed shows under-transfer in the main score: lower rank IC/spread than GRU-static and worse worst-month despite better monthly positive count.
- Current blocker is `score_mapping_and_router_calibration`, with secondary `resource_cost` for the default 11M-parameter Tier 2 configuration.

## Next Allowed Actions

- Do not bridge Tier 1 or Tier 2 into candidate review from current evidence.
- Diagnose Tier 2 router/score mapping before more seeds: expert weights by month/regime, router entropy, short-horizon collapse, local-state exposure, and why 30d probability was almost eliminated.
- Consider a reduced or calibrated Tier 2 revision before three-seed confirmation: lower hidden dimension or stronger score/hit calibration while keeping all five experts.
- Continue the existing `horizon_30d_soft_penalty_v1` post-throttle repair line only if the user prioritizes execution-candidate transfer over architecture capacity.
- Do not modify active execution strategy, production root, paper/live, broker, or promotion artifacts.
