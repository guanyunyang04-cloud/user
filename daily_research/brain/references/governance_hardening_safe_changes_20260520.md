# Governance Hardening Safe Changes 2026-05-20

This note records safe code-governance changes only. It is not promotion
authority, not live/default authority, and not completed strategy evidence.

## Facts

- Active execution material truth remains `daily_research/output/active_execution_strategy.json`.
- The active manifest was not intentionally modified by this safe-hardening work.
- `refresh-production-default` and `activate-single-mapping` remain danger tasks.
- `continuous_policy` and `path_policy` research outputs remain shadow-only unless a separate explicit promotion process says otherwise.
- The near-term repaired policy input bundle remains `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.

## Inferences

- Centralizing active manifest write confirmation reduces the chance that two execution entrypoints drift in safety behavior.
- Adding shadow-evidence promotional-claim checks reduces the chance that research or smoke evidence is written up as live/default evidence.
- Characterization tests for external target-weight bridging protect the current `research_raw_target_weight` and `follow_research_raw_no_global_cap` semantics.
- Forecast-stage loose-latest rejection tests protect the explicit dataset-id rule for the current Path20 neural research route.

## Assumptions

- This work is limited to safety, tests, and documentation.
- No training, smoke, pilot, long-run, production refresh, active manifest write, or promotion is part of this change.
- This document belongs in `brain/references/` and does not replace `state_center.md`, `knowledge_center.md`, or `operations_center.md`.

## Change Scope

- Added `daily_research/execution/active_manifest_guard.py`.
- Added execution safety tests under `daily_research/execution/tests/`.
- Added target-weight bridge characterization tests under `daily_research/baseline/tests/`.
- Extended `daily_research/tools/brain_rules.py` to reject shadow evidence marked as promotion/live/active.
- Extended Path20 protocol tests to cover loose latest dataset ids for forecast stages.

## Verification

Run the targeted tests:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest `
  daily_research/execution/tests/test_active_manifest_guards.py `
  daily_research/execution/tests/test_app_tasks_safety.py `
  daily_research/baseline/tests/test_external_target_weight_bridge.py `
  daily_research/tools/tests/test_brain_rules.py `
  daily_research/path_policy/tests/test_rl_protocol.py::test_forecast_stage_rejects_loose_latest_dataset_id `
  -q
```

Run the governance guards:

```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow verify-plan --json
```
