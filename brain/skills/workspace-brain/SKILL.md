---
name: workspace-brain
description: Use the workspace main brain as the first entrypoint for project takeover, task routing, branch discipline, brain governance, capsules, evidence lookup, guard checks, runtime learning proposals, and child-brain handoff. Trigger when the user mentions brain, 脑区, 项目大脑, 接管, capsule, governance, main-only, branch, runtime learning, daily_research, t0_project, or daily_stock_analysis-main.
---

# Workspace Brain Runtime

## First Move

Run detection first when taking over an unknown workspace:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py detect --cwd .
```

For existing brain workspaces, run a lightweight main-brain capsule to make routing, guards, evidence hints, and skill sync visible before changing tracked files, launching studies, claiming evidence, or entering a child brain:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --intent <read|mutate|writeback> --verbosity lite --json
```

Schema v3 capsules separate the routing layers:

- `routing.target.id` is the bootstrap target (`workspace` or a child brain id).
- `routing.target.kind` is `workspace`, `child`, or `ambiguous`.
- `routing.target.domain` is the workflow domain, such as `workspace_governance`.
- `workspace_governance` is a workspace domain and bootstrap alias, not a child brain id.
- `meta_cognition` is the controlled learning loop. If `meta_cognition.status != clear`, mention the signal in the final answer or next plan and propose the verification/writeback path.

Run compact health during takeover, anomaly triage, or final verification so catalog, guard, skill sync, and frontier warnings are visible without loading deep evidence:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact
```

Use full health only when compact health reports actionable warnings, routing/evidence is disputed, or you are auditing the brain system:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode full
```

For long training, research, build, data refresh, or other long-running jobs, first run the normal capsule with `--intent mutate`, then use the deterministic monitor:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --intent mutate --verbosity lite --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor template --json
```

The required wait window is `Wait-Process -Id <pid> -Timeout 7200`. The `7200` seconds are one observation window, not a business timeout. After each window, report PID status, elapsed time, progress, ETA, log tail, artifact mtime, and the next decision:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor status --pid <pid> --progress <progress.json> --stdout <stdout.log> --stderr <stderr.log> --artifact-dir <artifact_dir> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor wait-once --pid <pid> --timeout 7200 --progress <progress.json> --stdout <stdout.log> --stderr <stderr.log> --artifact-dir <artifact_dir> --json
```

Prefer `long_task_monitor wait-once/status` for long-task polling. `Start-Sleep` remains useful for very short UI pacing outside the monitor loop. Every user update for training polls should include ETA or state why ETA is not yet estimable.

If the project has no `brain/brain_manifest.json`, initialize a minimal brain only when mutation is allowed:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py init --cwd . --brain-id <project_id>
```

Use `preflight_blockers`, `mutation_allowed`, `routing`, guards, risk signals, capability hints, and writeback routes as the operating contract. Resolve or explicitly account for blockers before risky tracked-file edits.

## Runtime Contract

- Skill = entrypoint and procedure.
- Brain docs = project truth, current state, governance, evidence, and write routes.
- Tools = deterministic checks, capsules, health reports, sync checks, frontier scans, evidence queries, writeback plans, and long-task monitors.
- Local skills remain responsible for TDD, debugging, planning, frontend, security, deployment, and verification method. Brain supplies project facts, routing, guards, evidence, writeback routes, and sync checks.

## Operating Rules

- The main brain is the only agent takeover entrypoint.
- Child brains are project fact layers loaded only after main-brain routing.
- Work on `main` unless the user explicitly changes the branch rule.
- If capsule reports `not_on_main_for_mutation`, resolve branch authority or explicitly account for the blocker before repo-tracked mutation.
- Make `daily_research/output/active_execution_strategy.json` diffs explicit when the task asks for active strategy, promotion, or live-policy changes.
- Prefer explicit study tags, protocol tags, dataset ids, and reference docs over loose `latest_*` files.
- Treat smoke, dry-run, failed, interrupted, timeout, and diagnostic-only runs as non-completed evidence.
- Separate facts, inferences, assumptions, and action boundaries in substantial reports.
- For deterministic cleanup with clear benefit, low fact loss, and tests, remove stale paths completely instead of leaving compatibility shells.

## Runtime Reflection Learning

### Meta Cognition

Meta Cognition is the active learning loop: Observe -> Detect -> Classify -> Route -> Propose -> Verify -> Reuse. It is controlled autonomy: capsules and audits may discover, classify, record, and recommend learning, but core brain docs, workflow rules, skills, guards, and tests change only after user-authorized implementation.

Use this when the user says "this should be learned", "why did it not prompt", "from now on", or when evidence quality is polluted by low-budget runs:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py meta-audit --cwd . --mode compact
```

Capsule `meta_cognition.learning_opportunities[]` names the target layer, owner brain, writeback route, confidence, and required verification. Treat `runtime_learning_hooks` as a compatibility field; `meta_cognition` is the primary contract.

### Reflection Review

Before the final answer for implementation, debugging, long-task, or brain-maintenance work, compare planned steps, blockers, workarounds, user nudges, and verification against the final state. Prefer a structured trace when anything was blocked, skipped, manually bypassed, corrected by the user, or verified after failure:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py reflection-template --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py review --trace-json <trace.json> --cwd . --json
```

Freeform review is a low-confidence fallback for legacy observations only:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py review --cwd . --task "<user task>" --observation "<what happened>" --json
```

If review returns learning candidates, ask for confirmation or generate a runtime learning proposal with `status=proposed`; do not silently rewrite core brain docs.

Generate a runtime learning proposal when a repeated failure, rule conflict, timeout misread, branch violation, missing entrypoint, or stale skill is discovered:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py proposal --cwd . --title "<short title>" --trigger "<fact>" --evidence "<path or observation>" --recommendation "<change proposal>" --severity info --owner-brain workspace --writeback-target brain/references/
```

Review queued proposals:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py list-proposals --cwd .
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py mark-proposal --cwd . --proposal-id <id> --status <approved|implemented|verified|rejected|superseded>
```

The proposal is advisory. Core brain docs, workflow rules, and this skill should change only when the user has explicitly asked for that governance or skill update.

## Common Commands

Route a task:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<user task>" --json
```

Bootstrap the workspace or a selected child brain:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain workspace --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain workspace_governance --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace> --json
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
