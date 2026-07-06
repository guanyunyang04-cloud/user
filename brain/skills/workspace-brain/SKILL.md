---
name: workspace-brain
description: Use when the user mentions brain, 脑区, 项目大脑, 接管, capsule, governance, branch, agent learning, child brains, brain initialization, routing, proposals, or brain audits.
---
# Workspace Brain Runtime

## Multi-Paradigm Brain
This workspace is a personal research brain expressed as natural-language programming. Brain docs use objects for "what exists", procedures for "how an object method runs", and pure functions for "how task/evidence becomes a judgment". Tools are sensors.

- Start from the user's goal, then identify the relevant object(s): project, data asset, artifact, evidence, tool, branch surface, or external state.
- Use pure functions mentally: `select_relevant_objects(task)`, `activate_boundaries(objects, method)`, `classify_evidence(run)`, and `derive_next_action(state, evidence)`; use procedures only when their object and method are relevant.
- Activate only invariants attached to the touched objects and methods. Do not recite unrelated live/default, broker, registry, branch, or PIT boundaries when the task does not touch them.
- Treat rules, tests, guards, proposals, commits, capsule, route, and local skills as object methods or sensors.
- Prefer research progress, useful artifacts, simple current architecture, and git rollback over compatibility ceremony.
- Agent Meta Protocol is an experience-writeback object: use it for concrete learning opportunities, not as a fixed checklist for every task.

## Default Work Style
- Read whichever brain, code, registry, manifest, output, or reference actually helps.
- For repo-tracked mutation in a brain-governed project, use this skill to confirm object ownership, dirty paths, and any activated protected objects.
- QDP owns the shared canonical data substrate / manifest-first data base: `parquet + dataset.json + active.json`, provider ingest, active table checks, rebuildable caches/features, and data cleanup.
- Memmaps and training packs are downstream research artifacts, not the QDP active data base.
- `daily_research` owns research, model, backtest, execution-candidate, and active artifact evidence; it consumes QDP v2 tables or explicit downstream packs.
- For cross-project tasks, identify objects before choosing a brain: QDP active data base objects are owned by `quant_data_platform`; sequence packs, memmaps, normalization, labels, model outputs, losses, evaluations, stock profiles, and backtests are owned by `daily_research`.
- Treat route/capsule as primary-owner sensors, not exclusive project locks. Mixed QDP + research/model tasks should normally use primary `daily_research` with supporting read-only `quant_data_platform`, unless the task actually changes active QDP datasets or quality facts.
- New model-ready training artifacts default to `daily_research/data/research_store/<artifact_id>/`; old `quant_data_platform/data/qdp_v2/research/sequence_pack/` artifacts are compatibility research artifacts, not QDP active data base.
- If a task changes project architecture, CLI, data pointers, data semantics, quality conclusions, or execution boundaries, sync the relevant child brain hot path or state `brain_sync=false` with a reason in the final answer.
- Keep temporary artifacts near the relevant project or task.

## Optional Diagnostics / Sensors
Use these when they reduce uncertainty, not as mandatory first moves:
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py detect --cwd .
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --intent <read|mutate|writeback> --verbosity lite --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <workspace|workspace_governance|child_brain_id> --json
```
`route` is a sensor, not the final thinker; no route result is required. If it conflicts with user intent, file evidence, or stable memory, use agent judgment and say why.

## Protected Object Types
- `workspace_git_surface`: activated by repo-tracked mutation, commit, cleanup, migration, branch, or worktree actions; check branch and dirty paths then.
- `active_data_base_or_unique_data`: activated by QDP active manifest changes, dataset manifest changes, raw parquet cleanup, PIT state changes, or deletion; identify replacement pointers and sample validation.
- `pit_or_label_semantics`: activated by feature/label/data/training/evaluation conclusions; keep PIT/no-leakage and evidence-grade boundaries.
- `active_execution_artifact`: activated by live/default, paper/broker, promotion, active artifact, or trade-plan changes; inspect daily execution boundaries before modifying.
- `secret_or_external_state`: activated by keys, accounts, deploys, remote services, or real broker state; never treat as ordinary text.
- `cross_project_dirty_work`: activated by managing unrelated dirty paths, outputs, processes, ports, GPU jobs, or provider runs; avoid mixing ownership.

## Direct Change And Verification
- Prefer objective-first direct changes over wrappers, fallback modes, compatibility layers, or process ceremony.
- Run the smallest checks that support the claim: `git diff --check`, focused tests, `brain_sync_audit` for brain truth drift, `doc_guard` for brain docs, and `integrity_check` for structure changes.
- `tools.brain.project_commit`, `agent_run`, health, audit, and resource leases are optional helpers; use them when they clarify scope, long-running state, or rollback.

## Init And Register
Initialize/register a new project brain only when a real new project needs durable memory.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py init --cwd <project-root> [--brain-id <project_id>]
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py register --cwd <project-root> --workspace-root . [--brain-id <project_id>]
```

## On-Demand Health
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode full --timeout-sec 60
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-structure-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py multi-paradigm-lint --cwd . --scope attached
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.brain_sync_audit --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Use compact health for quick orientation and full health only for maintenance, diagnosis, or completion confidence.

## Proposal-Only Evolution
Agent Meta Protocol proposals remain useful for unclear or high-risk behavior changes: inspect with `brain_runtime.py list-proposals --cwd .`; proposed implementation still needs explicit user approval, while user-confirmed low-risk object cleanup can be implemented directly.
