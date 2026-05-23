---
name: workspace-brain
description: Use the workspace main brain as the first entrypoint for project takeover, task routing, branch discipline, brain governance, capsules, evidence lookup, guard checks, and child-brain handoff. Trigger when the user mentions brain, main brain, child brain, capsule, takeover, governance, main-only, branch, worktree, daily_research, t0_project, or daily_stock_analysis-main.
---

# Workspace Brain

## First Move

Run a main-brain capsule before changing tracked files, launching studies, claiming evidence, or entering a child brain:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --json
```

Use the `routing` block as the only authority for child-brain entry. If routing is `ambiguous`, stop and clarify or collect more context before entering a project body.

## API Fallback

When native skills are unavailable, use the repository brain platform directly:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow workflow-guide --workflow <selected_workflow> --json
```

This fallback returns schema v2 capsule context, the selected workflow, checklist, stop conditions, validation commands, and writeback routes. It is procedural only; do not copy long project histories into this skill.

## Operating Rules

- The main brain is the only agent takeover entrypoint.
- Child brains are project fact layers loaded only after main-brain routing.
- Work on `main` unless the user explicitly changes the branch rule.
- Do not modify `daily_research/output/active_execution_strategy.json` without explicit future promotion authority.
- Prefer explicit study tags, protocol tags, dataset ids, and reference docs over loose `latest_*` files.
- Treat smoke, dry-run, failed, interrupted, timeout, and diagnostic-only runs as non-completed evidence.
- Separate facts, inferences, assumptions, and action boundaries in substantial reports.

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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow evidence-index --rebuild --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q "<r-id/tag/dataset/blocker>" --json
```

Plan a routed writeback:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow writeback-plan --source study:<tag> --json
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
