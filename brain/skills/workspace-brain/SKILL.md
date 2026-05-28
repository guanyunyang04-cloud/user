---
name: workspace-brain
description: Use the workspace main brain as the first entrypoint for project takeover, task routing, branch discipline, brain governance, capsules, evidence lookup, guard checks, agent learning proposals, and child-brain handoff. Trigger when the user mentions brain, 脑区, 项目大脑, 接管, capsule, governance, main-only, branch, agent learning, daily_research, t0_project, or daily_stock_analysis-main.
---

# Workspace Brain Runtime

## First Move
Detect the workspace, then use workflow capsule as the only capsule entrypoint:
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py detect --cwd .
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<user task>" --workflow auto --intent <read|mutate|writeback> --verbosity lite --json
```
For takeover or verification:
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode full
```
Initialize only when no `brain/brain_manifest.json` exists and mutation is allowed:
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py init --cwd . --brain-id <project_id>
```

## Runtime Contract
- Main brain is the only agent takeover entrypoint; child brains load only after routing.
- Skill = entrypoint and procedure. Brain docs = project truth, evidence, and write routes. Tools = deterministic sensors and guards.
- Schema v4 separates `routing.target` from Agent Meta Protocol. `workspace_governance` is a workspace domain and bootstrap alias, not a child brain id.
- Use `preflight_blockers`, `mutation_allowed`, `routing`, guards, risk signals, capability hints, and writeback routes as the operating contract.
- Work on `main` unless the user explicitly changes branch policy.
- Make `daily_research/output/active_execution_strategy.json` diffs explicit when active strategy, promotion, or live policy is in scope.

## Evidence Model
Prefer explicit dataset ids, protocol tags, registry v3 program/family/run fields, and dated references over loose `latest_*` files.
- `research_programs`: stable problem lines.
- `study_families`: stages or method families.
- `run_tags`: physical run instances; do not create new research semantics by expanding run-tag prefixes.
- Smoke, dry-run, failed, interrupted, timeout, and diagnostic-only runs are non-completed evidence.
- Separate facts, inferences, assumptions, and action boundaries in substantial reports.

## Agent Meta Protocol
Agent owns the meta capability. Brain persists, distributes, and verifies it; capsules, audits, guards, and tests are sensors.
Run the agent meta pass at task start, major decision boundaries, and before final answer. If `agent_meta.review.status != clear` or `agent_review.before_final_required=true`, mention signal, target layer, writeback route, and verification path.
Before-final means closure-boundary meta-question discovery, not a fixed checklist: after the object-level answer is ready but before final response, low-noise check whether the task exposed a problem in frame, success criteria, method choice, authority order, evaluation mechanism, or learning salience. If it did, ask the user whether to evolve the agent/brain protocol; do not auto-create or implement learning without approval.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact
```
Proactively surface pending agent learning approvals. If `agent-meta-audit` or `list-proposals` shows any proposal with status `proposed` or `approved`, mention those pending agent learning approvals in the next substantial update or final answer; if none exist, say so when checked.
Brain burden rule: when rules, docs, compatibility shells, or wording tests slow or mislead the agent, treat it as `brain_rule_obstruction` and propose `brain_burden_governance` cleanup. Hard safety rules still win; operating defaults may be compressed with reasons.

## Reflection And Proposals
Run before-final reflection for implementation, debugging, long-task, or brain-maintenance work when anything was blocked, skipped, manually bypassed, corrected by the user, or verified after failure:
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py reflection-template --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py review --trace-json <trace.json> --cwd . --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py review --cwd . --task "<task>" --observation "<what happened>" --json
```
Create an agent learning proposal for repeated failure, rule conflict, timeout misread, branch violation, missing entrypoint, stale skill, missed meta pass, actor-boundary mismatch, or brain-rule obstruction:
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py proposal --cwd . --title "<short title>" --trigger "<fact>" --evidence "<path or observation>" --recommendation "<change proposal>" --severity info --owner-brain workspace --writeback-target brain/references/
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py list-proposals --cwd .
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py mark-proposal --cwd . --proposal-id <id> --status <approved|implemented|verified|rejected|superseded>
```

## Common Commands
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<user task>" --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <brain_id|workspace|workspace_governance> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q "<research_program/study_family/run_tag/dataset/blocker>" --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow writeback-plan --source run:<tag> --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.skill_install --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.skill_install --install
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor template --json
```
Use `tools.brain.long_task_monitor` for long training, research, build, data refresh, or other long-running jobs after a mutate capsule. Use `trace-poll` when a long task spans polling windows or will feed before-final review. Report PID status, elapsed time, ETA or why no ETA exists, log tail, progress, artifact mtime, and next decision.

## Final Guards
```powershell
git diff -- daily_research/output/active_execution_strategy.json
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Write workspace decisions to `brain/`; write routed project facts to the selected child brain; put dated details in `references/`, then rebuild the evidence registry when daily_research evidence changes.
