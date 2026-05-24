---
name: workspace-brain
description: Use the workspace main brain as the first entrypoint for project takeover, task routing, branch discipline, brain governance, capsules, evidence lookup, guard checks, self-evolution proposals, and child-brain handoff. Trigger when the user mentions brain, 脑区, 项目大脑, 接管, capsule, governance, main-only, branch, 自进化, daily_research, t0_project, or daily_stock_analysis-main.
---

# Workspace Brain Runtime

## First Move

Run detection first when taking over an unknown workspace:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py detect --cwd .
```

For existing brain workspaces, run a lightweight main-brain capsule before changing tracked files, launching studies, claiming evidence, or entering a child brain:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --intent <read|mutate|long_task|writeback> --verbosity lite --json
```

Run compact health during takeover, anomaly triage, or final verification so catalog, guard, skill sync, and frontier warnings are visible without loading deep evidence:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact
```

Use full health only when compact health reports actionable warnings, routing/evidence is disputed, or you are auditing the brain system:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode full
```

If the project has no `brain/brain_manifest.json`, initialize a minimal brain only when mutation is allowed:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py init --cwd . --brain-id <project_id>
```

Use `preflight_blockers`, `mutation_allowed`, `routing`, `workflow_guide`, and `stop_conditions` as the operating contract. If mutation is blocked, stop and correct the blocker before editing tracked files.

## API Fallback

When native skills are unavailable, use the repository brain platform directly:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --verbosity lite --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --verbosity full --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow workflow-guide --workflow <selected_workflow> --json
```

The lightweight fallback returns schema v2 capsule context, the selected workflow, checklist, stop conditions, validation commands, and writeback routes. Use `--verbosity full` only for audit, promotion, active-artifact, training-evidence, or evidence-conflict deep dives.

## Runtime Contract

- Skill = entrypoint and procedure.
- Brain docs = project truth, current state, governance, evidence, and write routes.
- Tools = deterministic checks, capsules, health reports, sync checks, and writeback plans.
- Generic skills remain responsible for TDD, debugging, planning, frontend, security, and verification. Brain safety boundaries override generic skill defaults.

## Operating Rules

- The main brain is the only agent takeover entrypoint.
- Child brains are project fact layers loaded only after main-brain routing.
- Work on `main` unless the user explicitly changes the branch rule.
- If capsule reports `not_on_main_for_mutation`, do not mutate repo-tracked files.
- Do not modify `daily_research/output/active_execution_strategy.json` without explicit future promotion authority.
- Prefer explicit study tags, protocol tags, dataset ids, and reference docs over loose `latest_*` files.
- Treat smoke, dry-run, failed, interrupted, timeout, and diagnostic-only runs as non-completed evidence.
- Separate facts, inferences, assumptions, and action boundaries in substantial reports.

## Self-Evolution

Generate a runtime learning proposal when a repeated failure, rule conflict, timeout misread, branch violation, missing entrypoint, or stale skill is discovered:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py proposal --cwd . --title "<short title>" --trigger "<fact>" --evidence "<path or observation>" --recommendation "<change proposal>" --severity info --owner-brain workspace --writeback-target brain/references/
```

The proposal is advisory. Do not rewrite core brain docs, workflow rules, or this skill without explicit user confirmation.

## Common Commands

Route a task:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<user task>" --json
```

Bootstrap the workspace or a selected child brain:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain workspace --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id> --json
```

Read or rebuild the evidence index:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q "<r-id/tag/dataset/blocker>" --json
```

Plan a routed writeback:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow writeback-plan --source study:<tag> --json
```

Check or install global skills from canonical repo source:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.skill_install --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.skill_install --install
```

Run guards before finalizing:

```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```

## Writeback

Write workspace-level decisions to `brain/`. Write project-specific facts to the routed child brain. Put dated details in the relevant `references/` directory, then rebuild the evidence registry when daily_research evidence changes.
