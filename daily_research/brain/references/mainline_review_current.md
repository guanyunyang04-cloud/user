# daily_research 主线审阅 current

Snapshot date: `2026-05-23`

Scope: this document is the rolling review entry for all attempted project mainlines since project start. It separates facts, inferences, assumptions, current stance, and next allowed actions. It does not replace `daily_research/brain/state_center.md`, and it must not be used as promotion authority. The 2026-05-23 update reviews the newer Path20 / multi-horizon, TDX-free data platform, and execution-reliability evidence after the 2026-05-18 snapshot.

## 0. Operating Rules

### Facts
- Active live/default execution remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- Active execution material truth remains `daily_research/output/active_execution_strategy.json`.
- Current effective live execution profile remains `regoff_k1_20d_ensemble_native_anchor`.
- `continuous_policy` remains `research / shadow_only`.
- `path_policy` remains `research / shadow-only`.
- Current `path_policy` research pointer is `alpha_multi_horizon_utility_policy_v1`; `alpha_path20_neural_policy_v1` is now historical evidence and code/tag namespace, not the current target definition.
- Reusable strict training-safe Gold dataset remains `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Near-term repaired policy input bundle is `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- TDX-free data platform V2 is the current ingestion boundary for new research data: online fetches belong to `daily_research.data_platform` refresh/import, while training/evaluation/diagnostics must read explicit lake dataset ids.
- All project mainlines are switchable current work pointers.
- After an explicit user or governance switch, subsequent work should continue along the newly selected mainline.

### Boundaries
- No live/default change may be inferred from this review.
- No smoke, dry-run, unit test, interrupted run, failed run, or realtime-tail label is completed strategy evidence.
- No loose `latest_*`, `latest`, or `default` pointer is a truth source without freshness and explicit tag checks.
- Any active execution write requires explicit future promotion authority and active artifact guards.
- Research-mainline switching does not automatically change live/default execution, promotion status, or `daily_research/output/active_execution_strategy.json`.
- Prior mainline evidence remains historical or comparison evidence unless the pointer is switched back.

## 0.5 Mainline Map and Classification

### Facts
- Current live/default execution line: `Deep Alpha / short_alpha / short_expert_policy_v5b` plus the execution-alignment production bridge.
- Current `path_policy` research frontier: `alpha_multi_horizon_utility_policy_v1`, with latest completed evidence `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
- Current `continuous_policy` allocation frontier: portfolio-set v5 / DFL-PG / cashflow-decision family through r74, all still `research / shadow-only`.
- Current data infrastructure frontier: TDX-free `data_platform_v2` plus explicit lake dataset ids, not ad hoc online fetches during research.
- Secondary comparison routes remain useful: `alpha_path20_sequence_policy_v1`, r39 allocation objective consolidation, r68 cashflow translation closure, r70 oracle/version-boundary repair, and r73/r74 lake behavior-quality diagnostics.

### Inferences
- The project has two different frontier questions now: path_policy asks "which stocks/horizons rank as profitable opportunities", while continuous_policy asks "how current portfolio capital should move among source, receiver, and cash." Mixing those questions causes false promotion pressure.
- The current live anchor is not the most experimental frontier; it is the most operationally materialized and guarded default.
- Infrastructure mainlines are part of the project mainline history because they decide what evidence is admissible, but they are not model-validity evidence by themselves.

### Assumptions
- "All mainlines" means durable research, execution, data, and governance directions recorded in brain and evidence references, not every individual trial or transient tag.
- This review is allowed to update brain reference state, but it is not authorized to update `daily_research/output/active_execution_strategy.json`.

## 1. Baseline Factor / Rule Trading

### Facts
- Code lives mainly under `daily_research/baseline/`.
- It provides factor calculation, regime/state handling, target-weight construction, backtests, and trade-plan plumbing.
- It still underpins parts of the production trade-plan generation path through `generate_daily_trade_plan.py`.

### Inference
- This was the earliest executable research mainline. It proved the daily research system could produce tradable scores and weights, but it does not solve portfolio cashflow and source/receiver allocation as a learned decision problem.

### Current Stance
- Keep as execution foundation and compatibility layer.
- Do not treat it as the current research frontier.

## 2. Advanced ML / State Profile / Attack-Defense

### Facts
- Code includes `baseline/train_trade_model.py`, `baseline/ml_alpha.py`, `baseline/advanced_ml_runtime.py`, and historical scan/diagnose scripts.
- Historical records show this line introduced model-family comparison, state profiles, walk-forward validation, weak-window diagnosis, and attack-defense thinking.

### Inference
- This line moved the project from fixed factor rules toward learned scoring and robust validation. Many later governance rules come from lessons here: avoid single-window champions, separate formal/recent/live, and keep execution semantics explicit.

### Current Stance
- Keep as historical and diagnostic foundation.
- Do not use old `score + 5d` language as the current live execution semantics without checking the active manifest.

## 3. Deep Alpha / Short Alpha

### Facts
- Code lives mainly under `daily_research/deep_alpha/`.
- Current live anchor descends from the `short_alpha` / `short_expert_policy_v5b` production system.
- Production default currently points to the short expert execution-aligned anchor, not to `continuous_policy`.

### Inference
- This is the current live/default origin line. It connected learned alpha, production full-fit, execution alignment, and daily trade-plan generation.

### Current Stance
- Preserve as live anchor until explicit promotion evidence replaces it.
- Do not restart training or refresh production root during review-only tasks.

## 4. Execution Alignment / Production Default

### Facts
- Key files include `daily_research/execution/run_trade_plan.py`, `daily_research/execution/update_default_candidate_production.py`, `daily_research/execution/app_tasks.py`, and `daily_research/baseline/generate_daily_trade_plan.py`.
- `refresh-production-default` and `activate-single-mapping` are marked danger tasks.
- Active manifest writes require explicit activation and confirmation flags.

### Inference
- This line is not a model research line; it is the safety bridge from research candidate to daily execution.

### Current Stance
- Highest operational caution.
- Any production refresh or active manifest update must be a separate explicit task.

## 5. Continuous Policy r10-r18: Action / Head / Translation Guard

### Facts
- Early `continuous_policy` work introduced action/head semantics and translation guards.

### Inference
- This phase tried to make model output resemble trading actions, but individual action labels did not close portfolio-level cashflow.

### Current Stance
- Keep as historical semantic groundwork.
- Do not return to pure action-head patching as the main route.

## 6. Continuous Policy r19-r30: Receiver / Source / Cash / Listwise / Teacher

### Facts
- This phase expanded receiver/source/cash ranking, listwise objectives, teacher logic, and release/relief mechanics.

### Inference
- It exposed that buy-side and sell-side cannot be learned as independent labels; portfolio funding, cash timing, release pressure, and execution constraints interact.

### Current Stance
- Lessons remain active.
- Avoid adding local penalties that only improve one metric while hiding funding or reversal failure.

## 7. Continuous Policy r31-r39: Unified Allocation Baseline

### Facts
- r39 remains the effective continuous_policy evidence baseline.
- Later state and knowledge centers still refer to r39 allocation objective consolidation as the valid baseline before r40+ research/shadow upgrades.

### Inference
- r31-r39 are the last relatively stable pre-shadow-upgrade reference point for continuous allocation semantics.

### Current Stance
- Treat r39 as baseline, not as live/default.
- Compare later research lines against r39-style behavior blockers before claiming improvement.

## 8. Continuous Policy r40-r48: End-to-End Allocation / Solver / OPE

### Facts
- This phase introduced end-to-end allocation, convex/solver/OPE ideas, and heavier resource experiments.

### Inference
- It shifted the main question away from action imitation toward portfolio allocation as the primary object.

### Current Stance
- Valuable architecture exploration.
- Do not revive heavyweight solver work by default unless it answers a specific blocker better than current lightweight routes.

## 9. Continuous Policy r49-r52: Capital-Flow Closure / Native Allocation Vector

### Facts
- This phase focused on capital-flow closure, true solver entry points, and native allocation vectors.

### Inference
- It clarified that source, receiver, cash, turnover, position cap, and monthly quality must be jointly decided.

### Current Stance
- Keep the capital-flow closure lesson as active design pressure.
- Do not solve failures by only increasing epochs or model width.

## 10. Continuous Policy r53-r55: Cash / Exposure Closure

### Facts
- This phase focused on semantic budget, cash/exposure closure, and cash timing release control.

### Inference
- Cash is not a residual; it is a first-class decision channel. Treating cash as leftover after stock scoring creates unstable behavior.

### Current Stance
- Cash timing remains a blocker class in later r70-r74/v6 work.

## 11. Continuous Policy r56-r61: Release-First / Core-v4

### Facts
- This phase introduced release-first and core-v4 style research branches.
- Later registry separates active profiles from legacy-compatible profiles.

### Inference
- Core-v4 helped establish release-first behavior semantics but is no longer the next-generation main model carrier.

### Current Stance
- Keep as baseline/ablation and historical contract.
- Do not make r61/core-v4 the default new mainline.

## 12. Continuous Policy r62-r64: Data Lake / Strict Gold

### Facts
- r64 produced full-window strict Gold: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Data lake code separates `strict_train` from `realtime_research`.

### Inference
- Data infrastructure is now strong enough that many remaining failures are behavioral and semantic, not missing-data blockers.

### Current Stance
- Use explicit strict dataset ids.
- Never count realtime unobserved labels as completed training evidence.

## 12.5 Data Lake Universal Repair

### Facts
- 2026-05-17 current infrastructure pointer is `data_lake_universal_repair`.
- Repair contract is recorded in `daily_research/brain/references/data_lake_universal_repair_contract_20260517.md`.
- `policy_input_bundle__0f116a9b78c92ff045a6853d` is full-universe-shaped but has missing `000300.SH` benchmark coverage before `2018-05-14`.
- `policy_input_bundle__4db1a32ab6e7d77ac7b8671c` is usable from `2018-05-14`, but is capped at `1200` symbols.
- Data lake policy bundles now support persisted benchmark `open` in `silver_benchmark.parquet`.
- `load_policy_inputs_from_lake(..., require_benchmark_open=True)` blocks silent close-as-open fallback.
- `python -m daily_research.data_lake.policy_input_audit` audits policy input bundle market, benchmark, membership, and feature-panel coverage.
- Cap80 repaired smoke bundle `policy_input_bundle__c4886777fe70bfe6616e1259` passed strict audit with benchmark open required.
- Full repaired bundle `policy_input_bundle__7c8f58d851bce8179e1e9e2d` passed strict audit with benchmark open required: `2010-01-04 -> 2026-05-13`, universe size `3070`, 109 feature panels, benchmark open/close rows `3969/3969`.
- `DEFAULT_POLICY_INPUT_LAKE_DATASET_ID` now points to `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.

### Inference
- The data-base correctness blocker is repaired for fixed-window capped pilots and future memory-safe full-universe work.
- Path20 Stage 1 can resume on capped pilots with the repaired full bundle, but full-universe training still needs memory-safe sequence loading.

### Current Stance
- Current infrastructure repair route, completed through repaired full-bundle audit.
- Do not mutate old bundle contents in place; create and audit a new repaired bundle.
- Use explicit repaired dataset id `policy_input_bundle__7c8f58d851bce8179e1e9e2d` for next Path20 Stage 1 pilots.

## 12.6 TDX-Free Data Platform V2

### Facts
- 2026-05-23 V2 is recorded in `daily_research/brain/references/tdx_free_data_platform_v2_20260523.md`.
- V2 supports domain-based refresh for `market_daily`, `trading_calendar`, `universe_snapshot`, `security_status`, `limit_status`, `industry_concept`, `valuation`, and `money_flow_hotspot`.
- `refresh_daily` now supports `--universe all_a|liquid500|file:<path>|symbols:<csv>`; explicit `--symbols` remains a small-sample/debug path, not the daily-update requirement.
- CSV is an import/patching channel only: it must enter Bronze, normalize through Silver, and register lake datasets before research use.
- First-class providers remain non-TDX: `eastmoney_efinance`, `akshare_eastmoney`, `baostock`, optional Tushare HTTP, and non-default realtime supplementation.
- Verification for V2 recorded `24` data-platform tests, `25` data-lake tests, and focused path/continuous/brain contract tests passing, with active artifact diff empty.

### Inference
- The project has moved from "TDX as an inconvenient dependency" to "TDX-family excluded from the formal research data boundary."
- Future model failures should not be diagnosed by silently falling back to online data inside training/evaluation. They should be diagnosed against explicit lake dataset ids and sidecar metadata.
- Provider health, refresh smoke, and CSV import are data admissibility evidence, not model effectiveness evidence.

### Current Stance
- Treat TDX-free V2 as the current data-ingestion mainline.
- Keep data-platform work separate from live execution and broker automation.
- Require explicit lake dataset ids for PathPolicy / Multi Horizon Utility / continuous_policy research.

## 13. Continuous Policy r65: Portfolio-Set v5

### Facts
- r65 introduced `formal_torch_portfolio_set_v5`.
- `training_contracts.py` marks this backend `promotable=false`.

### Inference
- v5 is an architecture upgrade for portfolio-set reasoning, not a live candidate by default.

### Current Stance
- Keep as shadow research backend.
- Do not infer promotion from GPU/epoch/resume capability.

## 14. Continuous Policy r67-r68: DFL-PG and Cashflow Decision

### Facts
- r67 replaced portfolio-set v5 objective/loss/oracle/profile defaults with paper-driven DFL-PG v1 concepts.
- r68 added `portfolio_cashflow_decision_v1` and connected v5 predictions to simulator/release trace/continuity metrics.

### Inference
- These are important translation-closure advances, but training evidence and behavior acceptance remained insufficient.

### Current Stance
- Keep as mechanism evidence.
- Do not treat as strategy effectiveness evidence.

## 15. Continuous Policy r69-r74: Value Arbitration to Lake Behavior Quality

### Facts
- r69 value arbitration, r71 multi-stage regret, r72 lake evaluator, r73 lake-native utilization, and r74 lake behavior quality are explicit research/shadow lines.
- r72 removed TDX singleton empty as an evaluate/shadow/export blocker by using lake evaluation.
- r73/r74 improved some source/receiver/cashflow plumbing but did not satisfy promotion or behavior acceptance.
- Feature contract degradation and behavior quality remain blockers.

### Inference
- The frontier shifted from "can the pipeline run" to "are source/receiver/cash decisions high quality under stable feature contracts."

### Current Stance
- Continue only with explicit tags and blocker-focused hypotheses.
- Do not add r69/r71/r74 to active default search profiles.

## 16. Continuous Policy Decision-Core v6

### Facts
- `formal_torch_decision_core_v6` is present in `training_contracts.py`.
- It is `promotable=false`.
- `run_continuous_policy_protocol.py` returns promotion gate status `shadow_only` for decision-core v6.
- Latest pointer freshness is risky: latest study tag and latest protocol tag differ.

### Inference
- v6 is a shadow-only unified decision-core research attempt. It may be useful for objective/shape diagnostics, but it is not a promotion route.

### Current Stance
- Review only with explicit protocol/study tags.
- Do not use loose latest v6 summaries as a single truth source.

## 17. Path20 Neural Policy v1

### Facts
- 2026-05-17 user decision switches the current Path20 research mainline pointer to `alpha_path20_neural_policy_v1`.
- Stages `dataset-smoke`, `oracle-smoke`, and `tiny-smoke` are current neural-policy mainline stages.
- Stages `forecast-dataset`, `forecast-train`, and `forecast-walkforward-study` are current neural-policy supervised forecaster Stage 1 stages.
- These stages no longer require `--allow-legacy-neural-policy`.
- Parser default stage is `dataset-smoke`.
- Existing tiny smoke evidence remains `alpha_path20_neural_policy_v1_tiny_smoke_20260516_01`.
- Stage 1 forecast contract is recorded in `daily_research/brain/references/alpha_path20_neural_forecaster_stage1_contract_20260517.md`.
- 2026-05-17 aggressive forecast upgrade expands model families to `linear_last_day`, `mlp_last_day`, `gru_sequence`, and `patch_transformer`.
- `gru_sequence` now uses multi-layer GRU sequence encoding with attention pooling; `patch_transformer` now uses learned positional embeddings, CLS pooling, and multi-scale patches.
- Forecast training now supports device-aware CUDA/CPU selection, AMP, mini-batch loading, early stopping, best checkpoints, learning curves, and multi-seed family summaries.
- 2026-05-17 multi-horizon opportunity upgrade expands Stage 1 cumulative forecast auxiliary targets from `5/10/20d` to `1/3/5/10/20d`.
- Forecast model `aux` output now has 8 dimensions: `cum1/cum3/cum5/cum10/cum20`, downside floor, worst 1d, and upside opportunity.
- Forecast datasets now persist `y_rank_by_horizon` for `1/3/5/10/20d` while keeping `y_rank_20d` for compatibility.
- Forecast training/evaluation now reports multi-horizon `rank_ic`, `top_bottom_spread`, direction accuracy, upside opportunity rank/spread, `validation_multiscale_score`, and `selected_signal_profile`.
- `--forecast-selection-profile` defaults to `multiscale` and can select by `multiscale`, `trend20`, or `short_burst` validation evidence.
- 2026-05-17 input feature profile upgrade adds explicit Stage 1 forecast input contracts:
  - `state_v1` keeps the previous state-feature selection for ablation.
  - `raw_kline_v1` adds raw OHLCV/K-line shape features.
  - `raw_kline_context_v1` is the new default and adds raw K-line, market breadth, benchmark context, and low-cost peer bucket context.
  - `raw_kline_context_no_alpha_prior_v1` removes alpha-prior and old alpha-score dependencies.
- Forecast dataset manifests now record feature profile, feature group counts, feature columns, cap-before/cap-after counts, and raw/market/peer/alpha-prior feature counts.
- Forecast training summaries and best checkpoints record the feature profile metadata used for the run.
- Stage 1 remains a per-stock sequence forecaster, not a cross-sectional attention model; interaction is represented through market/peer/context features.
- 2026-05-17 preflight found `policy_input_bundle__0f116a9b78c92ff045a6853d` blocked for `2018-01-01 -> 2024-12-31` because benchmark coverage starts at `2018-05-14`.
- `policy_input_bundle__4db1a32ab6e7d77ac7b8671c` passed capped preflight from `2018-05-14`, but remains capped at `1200` symbols and is not full-universe evidence.
- Repaired full-universe bundle `policy_input_bundle__7c8f58d851bce8179e1e9e2d` passed strict data-lake audit for `2019-01-01 -> 2024-12-31` with benchmark open required.
- 2026-05-18 capped Stage 1 result is recorded in `daily_research/brain/references/alpha_path20_stage1_formal_cap80_result_20260518.md`.
- Preflight tag `path20_stage1_preflight_cap80_repaired_20260518_01` completed on the repaired bundle with `raw_kline_context_v1`, `role_purge_trading_days=21`, and benchmark open loaded from `silver_benchmark.open`.
- Pilot tag `path20_stage1_pilot_cap80_repaired_20260518_04` completed after fixing an AMP/float16 metric dtype bug in forecast metric evaluation.
- Formal tag `path20_stage1_formal_cap80_repaired_20260518_01` completed with `max_universe_size=80`, `forecast_max_samples_per_role=8000`, four model families, seeds `7,11,19`, CUDA, AMP, best checkpoints, learning curve, and validation/test predictions.
- Formal capped verdict is `forecast_test_confirmed`; selected model is `gru_sequence` seed `19`; selected signal profile is `multiscale`; active execution remained unchanged.
- Selected validation metrics are positive across the five horizons: `rank_ic_1d/3d/5d/10d/20d = 0.054034/0.076536/0.087974/0.088521/0.051951` and `top_bottom_spread_1d/3d/5d/10d/20d = 0.001181/0.002272/0.001612/0.000575/0.003094`.
- Selected validation opportunity metrics are positive: `rank_ic_upside_20d=0.090375`, `top_bottom_spread_upside_20d=0.020895`.
- Test metrics are interpretable only because validation passed; selected test `rank_ic_20d=0.191404` and `top_bottom_spread_20d=0.058073`.
- Patch Transformer family had stronger 20d robustness than the selected GRU family: all three seeds were `multiscale`, family validation `rank_ic_20d_mean=0.077252`, and `top_bottom_spread_20d_mean=0.012649`.
- 2026-05-18 feature ablation result is recorded in `daily_research/brain/references/alpha_path20_stage1_feature_ablation_result_20260518.md`.
- Feature ablation completed four profiles on the repaired bundle with cap80, samples `8000` per role, epochs `60`, seeds `7,11`, CUDA, and AMP:
  - `state_v1`: `forecast_test_confirmed`, selected `patch_transformer` seed `7`, signal profile `trend_20d`, validation `rank_ic_20d=0.052529`, `top_bottom_spread_20d=0.000463`, `rank_ic_upside_20d=-0.035122`.
  - `raw_kline_v1`: `forecast_test_confirmed`, selected `patch_transformer` seed `11`, signal profile `multiscale`, validation `rank_ic_20d=0.099878`, `top_bottom_spread_20d=0.011222`, `rank_ic_upside_20d=-0.006853`.
  - `raw_kline_context_v1`: `forecast_test_confirmed`, selected `gru_sequence` seed `7`, signal profile `multiscale`, validation `rank_ic_20d=0.067361`, `top_bottom_spread_20d=0.004410`, `rank_ic_upside_20d=0.120046`.
  - `raw_kline_context_no_alpha_prior_v1`: `forecast_test_confirmed`, selected `patch_transformer` seed `11`, signal profile `multiscale`, validation `rank_ic_20d=0.086905`, `top_bottom_spread_20d=0.015142`, `rank_ic_upside_20d=-0.050637`.
- 2026-05-18 memory-safe forecast dataset pipe is implemented and recorded in `daily_research/brain/references/path20_memory_safe_forecast_dataset_contract_20260518.md`.
- Stage 1 now supports `--forecast-dataset-mode memmap`, which writes a disk feature store `[date, stock, feature]`, a lightweight sample index, and label memmaps.
- Forecast training now consumes eager or memmap datasets through a unified dataset view and can load lookback windows per batch instead of materializing all `[N,252,F]` samples in RAM.
- Full-universe `--max-universe-size 0` with eager forecast mode is blocked by protocol validation and requires `--forecast-dataset-mode memmap`.
- Memmap mode adds history-valid-ratio filtering and stratified validation/test metrics by train-seen status and history bucket.
- This memory-safe pipe fixes the main RAM feasibility blocker but does not yet implement true cross-stock attention; stock-to-stock relations remain represented through context and peer features.

### Inference
- The current Path20 research question now starts with supervised path forecasting quality before allocator/oracle/replay expansion.
- Neural-policy evidence is now current Path20 mainline evidence, but still shadow-only and not live/default promotion evidence.
- Forecast success must be judged validation-first by signal profile: 20d trend, short burst, multiscale, or failed. Daily MSE alone is insufficient.
- The multi-horizon upgrade is designed to answer whether some stocks have useful 1-5d burst information inside the 20d path, without letting 1d noise dominate selection.
- The feature profile upgrade is designed to test whether raw K-line shape and low-cost interaction context add signal beyond the old state-vector baseline.
- Raw K-line is not treated as automatically sufficient; ablations must compare it against technical/state/context and no-alpha-prior variants.
- True cross-sectional transformer or stock-to-stock attention is deferred until memory-safe day-grouped sequence loading exists.
- The aggressive upgrade increases model capacity and training auditability before long training, but does not itself create strategy evidence or a promotion path.
- Path20 Stage 1 is no longer blocked by the old benchmark coverage gap for capped pilots, provided it uses the repaired full bundle explicitly.
- Capped formal Stage 1 evidence is positive for the prediction task and supports moving to ablation and Stage 2 design, but it remains capped, research-only, and non-promotional.
- The selection rule needs review before larger runs because the chosen GRU seed is valid under the implemented family-level rule, while Patch Transformer looks stronger on 20d family stability.
- Feature ablation supports using raw K-line features: `raw_kline_v1` materially improves validation rank/spread over `state_v1`.
- Feature ablation supports that the signal is not only legacy alpha-prior replay: `raw_kline_context_no_alpha_prior_v1` remains multiscale and positive across validation 1d/3d/5d/10d/20d rank/spread.
- Market/benchmark/peer context appears especially useful for upside opportunity capture because only `raw_kline_context_v1` selected run has strongly positive validation `rank_ic_upside_20d` and upside spread.
- Full-universe Stage 1 training is no longer blocked by the eager-only design, but still requires a real repaired-bundle memmap dataset smoke and training smoke before evidence-grade full-universe runs.

### Current Stance
- Historical Path20 neural-policy research route from 2026-05-17 to the 2026-05-23 naming migration.
- Allow new neural-policy mainline diagnostics with explicit tags and fixed dataset ids.
- Treat Stage 1 forecast evidence as prediction-task evidence only; it does not prove a portfolio strategy until Stage 2 allocator/replay evidence exists.
- Do not touch active execution or infer live/default promotion from capped forecast evidence.
- Keep smoke defaults small; require explicit long-run arguments for evidence-grade training such as larger `--forecast-epochs` and multi-seed settings.
- Use completed feature ablation as the input evidence baseline; do not treat any single profile as final without considering raw K-line, context, and no-alpha-prior tradeoffs.
- Next priority is selection-rule review and Stage 2 allocator/oracle/replay design, with separate handling for multiscale trend/path and upside opportunity capture.
- Do not run full-universe training until repaired-bundle memmap dataset and training smokes pass.

## 17.5 Multi-Horizon Utility Policy v1

### Facts
- 2026-05-23 user/governance decision renamed the current path_policy research pointer to `alpha_multi_horizon_utility_policy_v1`.
- Chinese name: `多 Horizon 交易效用排序主线`.
- The latest completed evidence is `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
- The run used the repaired policy input bundle `policy_input_bundle__7c8f58d851bce8179e1e9e2d`, liquid500 pool `policy_pool_view__c11400fa72ad263f3d1eecfa`, GRU static-context family, seed `7`, train `2019-2022`, validation `2023`, test `2024`, `decision_utility_v1`, and horizon grid `1,2,3,5,8,10,15,20,30`.
- Verdict was `forecast_test_confirmed`: `trade_utility_score` validation rank IC `0.113993`, validation spread `0.028361`, validation hit lift `0.017385`, validation monthly positive rate `81.8%`; test rank IC `0.099703`, test spread `0.040183`, test hit lift `0.022097`, test monthly positive rate `100.0%`.
- Horizon-specific test signal strengthened toward longer horizons: `5d` rank IC `0.030911`, `10d` `0.045247`, `20d` `0.064482`, and `30d` `0.087132`.
- Predicted best horizon collapsed heavily toward `30d` on test, while future best horizon remained distributed.
- Active artifact impact remained empty; no allocator, replay, live/default, liquid800, or multi-seed authority was granted.

### Inference
- The new mainline is no longer fixed 20-day path prediction. It is a multi-horizon trade-utility ranking problem with a strong current 20-30 day tilt.
- The model appears to have found a robust longer-horizon utility ranking signal, but it has not learned a balanced per-sample horizon chooser.
- The current blocker is calibration: constrain or regularize horizon-score selection without destroying validation/test spread, hit lift, and monthly stability.

### Assumptions
- The `path20_...` tag namespace and `daily_research.path_policy.run_alpha_path20_protocol` module name remain historical/code compatibility names until a separate code-level rename is explicitly planned.
- Single-seed liquid500 evidence is enough to justify calibration research, but not enough for universe expansion, allocator/replay, or promotion discussion.

### Current Stance
- Current path_policy research pointer.
- Next allowed action is a constrained horizon-score / calibration variant with explicit `mh_utility_...` style naming when new artifacts are created.
- Only after calibration preserves the useful ranking signal should seeds `7,11,19` be run.
- Only after multi-seed stability should the team consider simplifying the broad grid into a `15/20/30d` longer-horizon utility family.

## 18. Path20 Sequence Policy v1

### Facts
- `alpha_path20_sequence_policy_v1` was the active path20 main research line after 2026-05-16 route consolidation, but the current pointer switched to neural-policy on 2026-05-17.
- Code lives under `daily_research/path_policy/`.
- Main protocol file is `daily_research/path_policy/run_alpha_path20_protocol.py`.
- Current stages include `rl-dataset-smoke`, `rl-train-smoke`, `rl-replay-smoke`, `rl-multiyear-smoke`, `rl-episode-dataset`, `rl-train-episode`, `rl-replay-episode`, `rl-walkforward-study`, `rl-walkforward-matrix`, and `rl-v5-dt-validation-study`.
- Sequence/RL inputs forbid future/oracle/path-label leakage columns.
- Protocol summaries mark `shadow_only=true`, `promotion_allowed=false`, and `active_execution_strategy_expected_diff=none`.

### Inference
- Path20 sequence policy remains a useful shadow comparison route, especially for dataset -> train -> replay -> report evidence, but it is no longer the current Path20 mainline.

### Current Stance
- Secondary / shadow comparison route.
- Keep v5/Decision Transformer evidence validation-first and shadow-only.
- Do not use oracle inputs as default teacher.

## 19. Current Risk Summary

### P0
- Live/default boundary breach: modifying `active_execution_strategy.json` or interpreting research shadow evidence as production.
- Evidence contamination: writing smoke/dry-run/unit/failed/interrupted/realtime-tail as completed evidence.

### P1
- Loose latest misuse while latest study/protocol tags differ.
- Continuous policy semantic drift in simulator, source/receiver/cash, and feature contracts.
- Path20 / multi-horizon protocol becoming large and monolithic while remaining shadow-only.
- Horizon-selection collapse: treating the current `30d` predicted-horizon dominance as a solved horizon chooser would overstate the evidence.
- Data boundary regression: letting formal research fall back to TDX-family online fetches or loose CSV would undo the TDX-free lake-first contract.

### P2
- Health/check aggregation has shown one non-stable concurrent failure; reruns passed, so track as tooling fragility.
- `state_builder.py` DataFrame fragmentation warnings indicate performance debt.
- Slow PathPolicy forecast dataset tests can create false negatives if short timeouts leave residual pytest processes; use selective verification first and long tests only when risk warrants.

## 19.5 Cross-Mainline Analysis

### Facts
- Baseline/deep-alpha/live execution lines are the only lines currently materialized as active daily execution.
- continuous_policy has the richest portfolio-allocation semantics, but remains shadow-only because source/receiver/cash quality, training evidence, and promotion gates are not closed.
- path_policy / multi-horizon has produced stronger ranking evidence, but it is still forecast/selection evidence before allocator/replay.
- data_lake / data_platform lines decide admissible inputs and reproducibility but do not prove trading effectiveness.

### Inferences
- The most dangerous mistake would be combining the strengths of different lines verbally: using path_policy ranking quality, continuous_policy allocation ambition, and live short-alpha materialization as if one integrated model already exists.
- The more coherent next architecture is a staged bridge: multi-horizon utility ranking can become an opportunity prior, continuous_policy can learn funding and cash movement, and the execution bridge can materialize only after formal gates.
- The current state favors "calibrate before expand" for path_policy and "repair feature/behavior contracts before strict resume" for continuous_policy.

### Assumptions
- A future integration between multi-horizon utility ranking and continuous allocation would be a new explicit research route, not an implicit promotion of either existing line.
- The live anchor should remain unchanged unless a future task explicitly authorizes promotion review and active artifact mutation.

## 20. Update Discipline

- Update this document after any task that reviews, changes, promotes, deprecates, or reroutes a mainline.
- Update `daily_research/brain/references/full_codebase_review_20260515.md` or its successor after any full codebase review.
- Rebuild `daily_research/brain/references/evidence_registry.json` after adding or materially changing reference docs.
- Always run:
  - `git diff -- daily_research/output/active_execution_strategy.json`
  - `git diff --check`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
