# daily_research 主线审阅 current

Snapshot date: `2026-05-17`

Scope: this document is the rolling review entry for all attempted project mainlines since project start. It separates facts, inferences, assumptions, current stance, and next allowed actions. It does not replace `daily_research/brain/state_center.md`, and it must not be used as promotion authority.

## 0. Operating Rules

### Facts
- Active live/default execution remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- Active execution material truth remains `daily_research/output/active_execution_strategy.json`.
- Current effective live execution profile remains `regoff_k1_20d_ensemble_native_anchor`.
- `continuous_policy` remains `research / shadow_only`.
- `path_policy` remains `research / shadow-only`.
- Reusable strict training-safe Gold dataset remains `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Near-term path/sequence policy input bundle remains `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- All project mainlines are switchable current work pointers.
- After an explicit user or governance switch, subsequent work should continue along the newly selected mainline.

### Boundaries
- No live/default change may be inferred from this review.
- No smoke, dry-run, unit test, interrupted run, failed run, or realtime-tail label is completed strategy evidence.
- No loose `latest_*`, `latest`, or `default` pointer is a truth source without freshness and explicit tag checks.
- Any active execution write requires explicit future promotion authority and active artifact guards.
- Research-mainline switching does not automatically change live/default execution, promotion status, or `daily_research/output/active_execution_strategy.json`.
- Prior mainline evidence remains historical or comparison evidence unless the pointer is switched back.

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
- These stages no longer require `--allow-legacy-neural-policy`.
- Parser default stage is `dataset-smoke`.
- Existing tiny smoke evidence remains `alpha_path20_neural_policy_v1_tiny_smoke_20260516_01`.

### Inference
- The current Path20 research question moves to path-label quality, oracle upper-bound, forecaster baseline, and neural target-weight allocation.
- Neural-policy evidence is now current Path20 mainline evidence, but still shadow-only and not live/default promotion evidence.

### Current Stance
- Current Path20 research route.
- Allow new neural-policy mainline diagnostics with explicit tags and fixed dataset ids.
- Do not start long training or touch active execution without a separate explicit task.

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
- Path20 protocol becoming large and monolithic while remaining shadow-only.

### P2
- Health/check aggregation has shown one non-stable concurrent failure; reruns passed, so track as tooling fragility.
- `state_builder.py` DataFrame fragmentation warnings indicate performance debt.

## 20. Update Discipline

- Update this document after any task that reviews, changes, promotes, deprecates, or reroutes a mainline.
- Update `daily_research/brain/references/full_codebase_review_20260515.md` or its successor after any full codebase review.
- Rebuild `daily_research/brain/references/evidence_registry.json` after adding or materially changing reference docs.
- Always run:
  - `git diff -- daily_research/output/active_execution_strategy.json`
  - `git diff --check`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`
