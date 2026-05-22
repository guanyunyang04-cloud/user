# Daily Research Full Codebase Review 2026-05-22

This note solidifies the 2026-05-22 handoff review of the complete
`daily_research` codebase. It is a reference document for future agents and
maintainers. It is not promotion authority, not live/default authority, and not
completed strategy evidence.

## Scope

- Status: `codebase_review / handoff_reference / no promotion`.
- Workflow: `brain`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Review object: `daily_research` under `H:/new_tdx64/PYPlugins/user`.
- Purpose: preserve the current understanding of mainlines, module boundaries,
  evidence rules, and next safe actions after a full codebase inspection.

## Evidence Used

- `daily_research/brain/state_center.md`.
- `daily_research/brain/knowledge_center.md`.
- `daily_research/brain/operations_center.md`.
- `daily_research/brain/brain_manifest.json`.
- `daily_research/brain/references/mainline_review_current.md`.
- `daily_research/brain/references/full_codebase_review_20260515.md`.
- `daily_research/brain/references/alpha_path20_decision_score_diagnostics_20260522.md`.
- `daily_research/brain/references/evidence_registry.json`.
- `daily_research.tools.brain_workflow capsule/current-frontier/verify-plan`.
- Source scans over `daily_research/{tools,execution,baseline,deep_alpha,continuous_policy,data_lake,path_policy}`.

## Facts

- `daily_research` is attached to the main brain as the
  `production_research_and_execution_cortex`.
- The current active execution material truth remains
  `daily_research/output/active_execution_strategy.json`.
- Current live/default execution remains
  `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- Current effective live execution profile remains
  `regoff_k1_20d_ensemble_native_anchor`.
- `continuous_policy` remains `research / shadow_only`.
- `path_policy` remains `research / shadow-only`.
- Current Path20 research pointer is `alpha_path20_neural_policy_v1`;
  `alpha_path20_sequence_policy_v1` remains a shadow comparison route.
- Reusable strict training-safe Gold dataset remains
  `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
- Near-term repaired policy input bundle remains
  `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- The reviewed worktree was clean before solidifying this document.
- `daily_research/output/active_execution_strategy.json` had no diff before
  this document was added.
- `daily_research` currently contains 325 Python files.
- Module counts observed during this review:
  - `baseline`: 45 files, 45 Python, 3 tests.
  - `continuous_policy`: 74 files, 74 Python, 30 tests.
  - `data_lake`: 15 files, 15 Python, 2 tests.
  - `deep_alpha`: 53 files, 53 Python, 1 test.
  - `execution`: 49 files, 32 Python, 2 tests.
  - `path_policy`: 30 files, 30 Python, 16 tests.
  - `tools`: 76 files, 74 Python, 13 tests.
- Largest Python risk surfaces include:
  - `daily_research/continuous_policy/model_seq_v3.py`.
  - `daily_research/continuous_policy/portfolio_simulator.py`.
  - `daily_research/continuous_policy/pipeline_utils.py`.
  - `daily_research/continuous_policy/run_self_optimizing_study.py`.
  - `daily_research/path_policy/run_alpha_path20_protocol.py`.
  - `daily_research/baseline/generate_daily_trade_plan.py`.
  - `daily_research/execution/update_default_candidate_production.py`.

## Inferences

- The repository is a combined research, validation, execution, data lake, and
  governance system, not a single-model project.
- The highest operational risk is mixing research evidence with live/default or
  promotion evidence.
- The highest research risk is misdiagnosing allocation semantics problems as
  model-capacity problems.
- `brain` and `tools` are part of the control plane. They are not secondary
  documentation.
- `output/cache/archive` contain large historical artifacts; future review work
  should prefer explicit tags, dataset ids, and brain queries over broad
  recursive scans.
- The current continuous-policy blocker is behavior quality and evidence
  sufficiency, not simply lack of data infrastructure.
- The current Path20 blocker is target / selection / regime calibration, not
  simply lack of architecture size.

## Assumptions

- This review is a handoff and planning reference only.
- No training, evaluate, shadow, export, protocol, study, production refresh, or
  active manifest write is part of this solidification.
- This document belongs in `daily_research/brain/references/` and does not
  replace `state_center.md`, `knowledge_center.md`, or `operations_center.md`.

## Mainline Review

- Baseline factor and rule trading was the earliest executable research and
  trade-plan foundation. It remains important plumbing, but it is not the
  current research frontier.
- Advanced ML, state profile, and attack-defense work introduced model-family
  comparison, walk-forward validation, weak-window diagnosis, and the habit of
  separating formal, recent, and live claims.
- `deep_alpha` / `short_alpha` is the origin of the current live/default
  execution anchor. Preserve it until explicit promotion evidence replaces it.
- `execution` is the production boundary. `refresh-production-default` and
  `activate-single-mapping` remain danger tasks.
- `continuous_policy` phase 10-18 improved action/head semantics and translation
  guards, but did not solve portfolio-level cashflow.
- `continuous_policy` phase 19-30 exposed receiver/source/cash interaction and the
  limits of independent buy/sell labels.
- `continuous_policy` phase 31-39 formed the current effective continuous-policy
  evidence baseline through unified allocation semantics.
- `continuous_policy` phase 40-48 explored end-to-end allocation, solver, and OPE
  directions but did not pass stable confirm.
- `continuous_policy` phase 49-52 clarified capital-flow closure and native
  allocation vector requirements while source/exposure/evidence remained
  blockers.
- `continuous_policy` phase 53-55 showed that cash is a first-class decision
  channel, not a residual after stock scoring.
- `continuous_policy` phase 56-61 improved release-first and core-v4 wiring and
  diagnostics, but did not close release/source/receiver behavior.
- Phase 62-64 completed the data lake and strict Gold foundation for reusable
  training-safe evidence.
- Phase 65 introduced portfolio-set v5 as a shadow research backend, not a
  promotable live candidate.
- Phase 67-68 added DFL-PG and `portfolio_cashflow_decision_v1`, closing important
  translation mechanics while leaving behavior quality and evidence unresolved.
- Phase 69-74 advanced value arbitration, version boundary repair, multi-stage
  regret, lake evaluator, and lake-native behavior-quality diagnostics. These
  remain research / shadow evidence.
- Path20 sequence-policy work remains available as shadow comparison.
- Path20 neural-policy work is the current research pointer, but it is still a
  forecast-stage route and not allocator, replay, promotion, or live evidence.
- Brain/tools governance is a continuing mainline: capsule, doc guard, evidence
  registry, frontier scanner, and active artifact guards are required to prevent
  repeated historical errors.

## Module Review

- `daily_research/brain`: project truth source for current state, hard rules,
  operations, governance, and evidence references.
- `daily_research/tools`: guard and handoff layer. Use `brain_workflow`,
  `doc_guard.py`, `brain_integrity_check.py`, `frontier_scanner.py`, and
  `selective_verification.py` before interpreting current state or changing
  high-risk paths.
- `daily_research/data_lake`: DuckDB/Parquet catalog and training/policy input
  truth layer. Preserve strict train versus realtime research separation.
- `daily_research/baseline`: historical factor/ML system and still-important
  trade-plan bridge, especially `generate_daily_trade_plan.py`.
- `daily_research/deep_alpha`: short-alpha and current live-anchor research
  lineage. Do not retrain or refresh production roots during review-only tasks.
- `daily_research/execution`: production app and active-manifest boundary.
  Treat active writes as separate explicit promotion work.
- `daily_research/continuous_policy`: complete shadow research pipeline for
  daily portfolio decisions. Highest-risk surfaces are simulator semantics,
  profile/loss/version boundaries, source/receiver/cash behavior, and evidence
  classification.
- `daily_research/path_policy`: current Path20 research front. It enforces
  explicit lake dataset ids and forecast/RL leakage boundaries; current neural
  research does not authorize allocator or live work.
- `daily_research/output`, `daily_research/cache`, `daily_research/archive`:
  artifact stores. Use explicit tags and registries; do not infer truth from
  loose latest files or timestamps alone.

## Current Frontier Interpretation

- The current Path20 GRU decision signal is not zero: several aggregate scores
  show positive rank IC, spread, and hit lift.
- Selection-only calibration has not solved monthly instability.
- Longer train windows did not fix the current selection problem.
- Stock mixer has stronger ranking/spread stability but failed hit-lift gates.
- No audited Path20 score has passed the full selection calibration gate.
- The current Path20 bottleneck is more likely `hit_label` / utility target
  definition and regime-aware selection than model capacity alone.
- Do not run liquid800, allocator, replay, or live/default from these results.

## Critical Risks

- Treating `latest_*`, `latest`, or `default` pointers as truth without
  freshness, tag, and same-source checks.
- Treating smoke, dry-run, unit tests, failed runs, interrupted runs, or
  realtime-tail labels as completed strategy evidence.
- Treating `formal_torch_portfolio_set_v5` GPU/epoch/resume ability as
  promotability. It remains `promotable=false`.
- Enabling explicit phase 69/71/74 profiles as default search profiles without
  explicit authority.
- Changing simulator, profile registry, execution bridge, or active manifest
  behavior without targeted tests and active artifact guards.
- Solving behavior failures by adding epochs, loss terms, or model width before
  verifying target construction, cashflow semantics, source/receiver coverage,
  and regime stability.

## Next Allowed Actions

- Keep live/default and active artifact frozen unless a future explicit
  promotion task authorizes otherwise.
- For Path20, inspect target calibration first: hit threshold, utility
  definition, and `hit_lift_top20_mean` versus future utility.
- If target calibration remains coherent, run liquid500 input contamination
  ablations such as `no_alpha_prior` and `no_sector_static`.
- For `continuous_policy`, continue around feature contract health,
  source/receiver coverage, cash timing, source quality, receiver-source
  spread, and sufficient training evidence.
- For governance, keep using task capsule, current-frontier, doc guard,
  brain integrity, and active artifact diff checks before any high-risk action.
- For future code changes, expand tests around simulator semantics,
  profile/loss/version boundaries, execution active guards, and evidence
  registry/frontier behavior.

## Boundaries

- This document does not change live/default execution.
- This document does not change `daily_research/output/active_execution_strategy.json`.
- This document does not authorize promotion, confirmatory, strict resume,
  liquid800 expansion, allocator, replay, or production refresh.
- This document does not convert any smoke, dry-run, unit, failed, interrupted,
  realtime-tail, or protocol-only output into completed strategy evidence.
- Future agents must continue to separate facts, inferences, assumptions, and
  explicit evidence tags before acting.

## Verification

Commands run during this solidification workflow:

```powershell
git status --short
git diff -- daily_research/output/active_execution_strategy.json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -X utf8 -m daily_research.tools.brain_workflow capsule --child daily_research --task "详细检阅daily_research完整代码库并深入分析" --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -X utf8 -m daily_research.tools.brain_workflow current-frontier --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -X utf8 -m daily_research.tools.brain_workflow verify-plan --json
```

No full pytest suite, training, protocol, study, production refresh, or active
manifest write was run for this review.
