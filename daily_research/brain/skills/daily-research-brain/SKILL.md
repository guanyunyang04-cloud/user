---
name: daily-research-brain
description: Use the daily_research brain operating system for project handoff, continuous_policy work, r-number research, study/protocol evidence review, reusable training datasets, data lake questions, active artifact guard checks, and brain evidence writeback. Trigger when the user mentions daily_research, continuous_policy, brain/脑区, rXX, study, training, data lake, evidence writeback, active_execution_strategy.json, or asks to continue/implement project work.
---

# Daily Research Brain

## First Move

Run a task capsule before changing tracked files, launching studies, claiming evidence, or writing brain docs:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<user task>" --json
```

Use the capsule as the current project operating context. It provides boot order, state summary, hard rules, forbidden actions, related references, assumptions, and validation commands.

## API fallback

When this skill is unavailable, such as in an API-only session, use the brain-native workflow fallback instead of relying on plugin injection:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<user task>" --workflow auto --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow workflow-guide --workflow <selected_workflow> --json
```

The fallback returns the selected workflow, checklist, stop conditions, validation commands, and writeback routes. This skill remains a thin procedural entrypoint and must not copy external plugin text or long project histories.

## Operating Rules

- Keep project truth in `daily_research/brain`; keep this skill procedural.
- Work on `main` unless the user explicitly changes the branch rule.
- Do not modify `daily_research/output/active_execution_strategy.json` without explicit future promotion authority.
- Prefer explicit study tags, protocol tags, dataset ids, and reference docs over loose `latest_*` files.
- Treat smoke, dry-run, failed, interrupted, timeout, and diagnostic-only runs as non-completed evidence.
- Treat realtime tail labels with unobserved forward outcomes as non-training evidence.
- Separate facts, inferences, assumptions, and action boundaries in substantial reports.

## Common Commands

Preflight a workflow:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow preflight --workflow <workflow> --task "<task>" --json
```

Read or rebuild the evidence index:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow evidence-index --rebuild --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow query --q "<r-id/tag/dataset/blocker>" --json
```

Plan a writeback from verified evidence:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow writeback-plan --source study:<tag> --json
```

Run guards before finalizing:

```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json
```

## Writeback

Write compact current facts to the main centers. Put dated details in `daily_research/brain/references/`, then rebuild `daily_research/brain/references/evidence_registry.json`.

Do not copy long r-number histories into this skill. Query the brain evidence registry instead.
