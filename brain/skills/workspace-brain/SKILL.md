---
name: workspace-brain
description: Use when the user mentions brain, 脑区, 项目大脑, 接管, capsule, governance, branch, agent learning, child brains, brain initialization, routing, proposals, or brain audits.
---
# Workspace Brain Runtime
## Personal Researcher Default
This workspace is a personal research brain, not a team process gate. The agent owns judgment; brain docs and tools are memory and sensors.
- Optimize for research progress, useful artifacts, simple current architecture, and git rollback.
- Treat rules, tests, guards, proposals, commits, capsule, route, and local skills as tools, not rituals.
- Delete or rewrite stale code, tests, wrappers, compatibility paths, and old docs when they no longer serve the current system.
- Keep only hard boundaries: canonical/unique data, PIT/no-leakage, live/default or active artifacts, secrets/external services, and cross-project dirty ownership.
## Default Work Style
- Start from the user's goal, current files, git diff, and the smallest useful evidence.
- For any brain-governed workspace project, load this skill before repo-tracked work. Use it to confirm target ownership, branch, dirty paths, and hard boundaries.
- Read whichever brain, code, registry, manifest, output, or reference actually helps; no route result is required before reading a relevant project.
- QDP owns the shared canonical data substrate: `canonical_data_v1`, registry, policy bundle, memmap, coverage audit, and data cleanup.
- `daily_research` owns research, model, backtest, execution-candidate, and active artifact evidence; it consumes QDP data.
- Keep temporary artifacts near the relevant project or task; avoid cluttering the workspace root.
## Optional Diagnostics
Use these when they reduce uncertainty, not as mandatory first moves.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py detect --cwd .
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --intent <read|mutate|writeback> --verbosity lite --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <workspace|workspace_governance|child_brain_id> --json
```
`route` is a sensor, not the final thinker. If it conflicts with user intent, file evidence, or stable workspace memory, use agent judgment and say why.
## Hard Boundaries
- Check branch and dirty paths before repo-tracked mutation when risk is nontrivial.
- This workspace defaults to `main` only. Do not create, switch to, or continue on non-`main` branches or extra git worktrees unless the user explicitly authorizes that exception for the current task.
- Never silently change active artifacts, live/default, paper/broker behavior, promotion gates, secrets, external service state, unique data, or PIT/no-leakage rules.
- For QDP data rebuilds, registry pointer changes, memmap cleanup, or deletion of old data, first identify replacement pointers and enough sample validation.
- For daily execution changes, inspect `daily_research/output/active_execution_strategy.json` and promotion boundaries before touching behavior.
## Direct Change And Verification
- Prefer objective-first direct changes over wrappers, fallback modes, compatibility layers, or process ceremony.
- Run the smallest checks that support the claim: `git diff --check`, focused tests, `doc_guard` for brain docs, and `integrity_check` for structure changes.
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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Use compact health for quick orientation and full health only for maintenance, diagnosis, or completion confidence.
## Proposal-Only Evolution
Agent Meta Protocol is a proposal queue: low-risk observations may be written as `proposed`, implementation still needs explicit user approval, and approved brain/tooling directions can then be implemented directly; inspect with `brain_runtime.py list-proposals --cwd .`.
