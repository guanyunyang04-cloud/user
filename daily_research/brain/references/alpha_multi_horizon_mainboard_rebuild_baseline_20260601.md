# Alpha Multi Horizon Mainboard Rebuild Baseline - 2026-06-01

## Facts

- Run tag: `mh_rebuild_mainboard_anchor_20260601_01`.
- Study family: `mainboard_rebuild_baseline`.
- Research program: `alpha_multi_horizon_utility_policy_v1`.
- Dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Corrected pool: `policy_pool_view__74f45f4f83263bccd64a8027`, `rolling_liquid500_mainboard`.
- Wrong-universe diagnostic pool: `policy_pool_view__8f9367ce305548e3ecd04feb`.
- Feature profile: `raw_kline_context_no_alpha_prior_v1`.
- Model family: `gru_sequence_static_context`.
- Loss profile: `target_norm_head_constraint_v1`.
- Horizons: `1,2,3,5,8,10,15,20,30`.
- Seeds completed: `7,11,19`.
- Boundary: research-only / shadow-only; execution remains `frozen_skeleton_only / awaiting_research_rebuild`.

## Mainboard Pool Validation

- Validation artifact: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_anchor_20260601_01/mainboard_pool_validation.json`.
- Status: `ok`.
- Active universe size: `2596`.
- Membership symbols: `3175`.
- Daily member count min / median / max: `500 / 500 / 500`.
- Active excluded prefix counts: `300=0`, `301=0`, `688=0`, `689=0`.
- Blockers: none.

## Memmap Validation

- Validation artifact: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_anchor_20260601_01/mainboard_memmap_validation.json`.
- Seed7 manifest: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_target_norm_head_constraint_raw_seed7_20260601_01/forecast_dataset_manifest.json`.
- Status: `ok`.
- Source market dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Source pool view id: `policy_pool_view__74f45f4f83263bccd64a8027`.
- Feature store shape: `[1699,2596,116]`.
- Feature count before / after cap: `116 / 116`.
- Sample rows: train `449662`, validation `103437`, test `104556`.
- Label semantics: `next_open_entry_to_future_open`.
- Execution mode: `next_open`; `next_open_label_extra_trading_day=1`.
- Stock value excluded prefix counts: `300=0`, `301=0`, `688=0`, `689=0`.

## Three-Seed Corrected Gate

- Summary artifact: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_anchor_20260601_01/mainboard_rebuild_summary.json`.
- Gate status: `near_pass`, not full pass.
- Passed checks:
  - seed count `>=3`;
  - test rank IC min positive;
  - test top-bottom spread min positive;
  - mean monthly positive rate `>=0.75`;
  - max negative months `<=2`;
  - 30d concentration not worse than old Stage 2.8.
- Failed check:
  - test hit lift min positive: failed, min `-0.0071560784524209`.
- Aggregate test row for `pred_decision_score`:
  - rank IC mean / min: `0.0964785043195931 / 0.0840742492770226`;
  - spread mean / min: `0.038824276374481 / 0.0319425596650944`;
  - hit lift mean / min: `0.0055695987400983 / -0.0071560784524209`;
  - monthly positive rate mean / min: `0.8484848484848485 / 0.8181818181818182`;
  - negative month count mean / max: `1.6666666666666667 / 2`;
  - 30d concentration mean: `0.7241828940153283`.

## Feature Schema Diff

- Diff artifact: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_anchor_20260601_01/feature_schema_diff_report.json`.
- Status: `legacy156_not_recovered`.
- Current feature store shape: `[1699,2596,116]`.
- Old expected Stage 2.8 feature store shape: `[1699,2430,156]`.
- Legacy `156` manifest path: none recovered locally.
- Decision: do not fabricate a `legacy156` profile. Keep `feature_schema_shift` as an unresolved root cause.

## Lineage Equivalence Audit

- JSON artifact: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_anchor_20260601_01/mainboard_lineage_equivalence_audit.json`.
- Markdown artifact: `daily_research/output/path_policy/studies/mh_rebuild_mainboard_anchor_20260601_01/mainboard_lineage_equivalence_audit.md`.
- Current corrected mainboard rebuild is a new-lineage mainboard baseline, not an old Stage 2.8 payload replay.
- Historical Stage 2.8 remains brain-confirmed text evidence unless a legacy manifest or payload is recovered.
- The old wrong-universe rebuild stays classified as `wrong_universe_diagnostic`.

## short_v5b Bridge

- Bridge status: `blocked_missing_short_v5b_payload`.
- Expected root: `daily_research/output/short_expert_policy_v5b_execalign_production_default`.
- Bridge allowed now: `false`.
- A future bridge must unify pool, costs, rebalance rule, benchmark, date split, score / target-weight panel semantics, and next-open execution assumption.

## Interpretation

- The universe error has been corrected, and the corrected mainboard-only baseline is materially healthier than the earlier wrong-universe rebuild on rank IC, spread, monthly stability, negative months, and 30d concentration.
- It still cannot be declared better than old `short_v5b` or old Stage 2.8 because the old payload is not file-backed replayable and the old `156` feature schema has not been recovered.
- The next research question is no longer "is the old rebuild valid?" It is: can the new-lineage `[1699,2596,116]` mainboard baseline repair the remaining hit-lift weakness or recover/bridge the old `156` schema under a traceable protocol?

## Boundaries

- Do not write, restore, or promote `daily_research/output/active_execution_strategy.json`.
- Do not use the wrong-universe pool for model victory claims.
- Do not delete execution code; execution stays frozen with skeleton and candidate evaluation wrappers.
- Do not create a `legacy156` feature profile unless old column definitions are recovered from file-backed evidence.
