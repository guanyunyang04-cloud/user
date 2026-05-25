# Uncompleted Plan Integration Closure - 2026-05-25

## Scope
- Status: `workspace brain / daily_research integration closure / implementation verification`.
- User request: combine and complete all unfinished plans.
- Boundary: no broker connection, no live/default/promotion switch, no production retrain, no active execution strategy edit.

## Completed In This Pass
- Workflow selection now accepts task intent and explains matched terms, decision sources, confidence, intent overrides, and ambiguous signals.
- `intent=mutate` plus a plan-title task can select `executing_plan`; plan-only read requests remain `writing_plan`.
- Capsule lite now exposes `self_evolution_hooks` for execution, debugging, maintenance, architecture, and long-task workflows.
- Workflow guide now states the brain/generic skill boundary: generic skills own general execution method; brain workflow supplies project guards, routing, evidence, and writeback boundaries.
- Workspace brain runtime now supports runtime learning review, proposal queue listing, and proposal status marking.
- Workspace-brain skill now includes a Self-Evolution Final Review entry and was synced to the global Codex skill directory.

## Existing Work Confirmed
- Universal Data Platform V3 components are present and tested: `formal_free_v3`, provider capability matrix, provider health, required/optional domain gate, provenance and manifest quality fields.
- Execution scheduler endpoints are present: `/api/data-sources/scheduler` GET/PATCH and post-close scheduler state in the execution app runtime.
- Paper Trading Ledger is present and tested: SQLite ledger, first-run CSV migration, idempotent trade-plan registration, next-open order application, pending missing-open orders, cash flow adjusted TWR, snapshot export.
- React HelpPage is present: `/help` and `/guide` route to the same read-only guide page with runbook, status dictionary, safety boundaries, and FAQ.
- Alpha multi-horizon root-cause audit and seed7 target-function controls already completed and are referenced in daily_research brain references.
- Output/loss auxiliary profile and horizon-grid comparison tooling already exists; no new long training was launched in this closure.

## Verification
- `pytest tools/brain/tests -q`: `124 passed`.
- `pytest daily_research/data_platform/tests -q`: `38 passed`.
- `pytest daily_research/execution/tests/test_execution_console_v2_api.py daily_research/execution/tests/test_paper_trading.py -q`: `57 passed`.
- `pytest daily_research/path_policy/tests/test_target_calibration_audit.py daily_research/path_policy/tests/test_decision_score_diagnostics.py daily_research/path_policy/tests/test_horizon_root_cause_audit.py -q`: `21 passed`.
- `npm test -- --run`: `20 passed`.
- `npm run build`: passed with only the existing Vite chunk-size warning.
- `pytest daily_research/path_policy/tests/test_models.py daily_research/path_policy/tests/test_output_aux_profile_comparison.py -q`: `9 passed`.
- `pytest daily_research/path_policy/tests/test_forecast_training.py -q`: `22 passed in 803.41s`; rerun used PID-bound long-task wait via `tools.brain.long_task_monitor wait-once`.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty.

## Residual Boundaries
- The broad output/aux/grid experiment matrix has tooling but not a completed multi-seed result matrix; later execution must be gated by prior evidence and use long-task monitoring.
- Existing frontend build emits a Vite chunk-size warning; this is not a functional failure.
- Any future promotion, active manifest mutation, production root update, paper-shadow bridge, allocator/replay bridge, or broker action requires separate explicit authorization.
