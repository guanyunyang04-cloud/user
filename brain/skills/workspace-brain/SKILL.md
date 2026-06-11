---
name: workspace-brain
description: Use when the user mentions brain, 脑区, 项目大脑, 接管, capsule, governance, branch, agent learning, child brains, brain initialization, routing, proposals, or brain audits.
---

# Workspace Brain Runtime
## First Hop
Use the runtime and workflow capsule as sensors; `brain/brain_manifest.json` is the contract truth.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py detect --cwd .
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --intent <read|mutate|writeback> --verbosity lite --json
```
## Route Or Bootstrap
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <workspace|workspace_governance|child_brain_id> --json
```
`route` is a sensor, not the final thinker. Only read child-brain context after routing returns `status=selected` and `target.kind=child`; if routing returns `needs_agent_decision` or `ambiguous`, inspect user intent, paths, manifest evidence, and candidates before choosing a bootstrap target. Do not read a child brain because of one generic term such as `brain`, `study`, `training`, `数据集`, or `模型`.
## Project Scope
Capsule/bootstrap exposes the selected `project_profile`; use it for every task namespace: read/write scope, verification, commit, process, and temporary outputs.
- Every task first binds to one `project_id`; use `workspace`/`workspace-brain` for shared brain or tooling changes. If routing is ambiguous, decide or ask before mutation.
- Default write/output/test/commit scope is the selected project profile plus explicit user-added paths. Put temporary reports, logs, short-run artifacts, and diagnostics under that project/task path.
- External project dirty/output/process paths may be reported as status summaries; do not inspect their contents, reuse them, or treat them as task evidence without explicit scope expansion or lease.
- Treat other project dirty paths as external parallel work unless the user explicitly expands scope.
- After verified project work, commit with `tools.brain.project_commit` using the selected project id or `workspace-brain`; external dirty paths are reported as `ignored_external_paths`, not staged.
- Before final after verified mutations, run `tools.brain.project_commit --dry-run --expect-paths <intended changed paths>` or actually commit; if intended paths are out of scope, report that blocker instead of silently stopping.
- Short synchronous commands run from the selected project scope. Polling or asynchronous tasks use the selected project namespace and best observable handle: PID, job/run id, logs, progress, artifact mtime, port/API status, or resource state. Use `tools.brain.agent_run` under `<project>/output/agent_runs/<run_id>/` when a process must outlive the immediate shell wait or needs registered PID/log/progress; cross-project process or resource reads require an explicit lease.
- This is an operating contract: commit and process helpers enforce their slices; ordinary shell reads/writes require the agent to honor the selected project namespace.
## Adaptive Polling
Polling cadence is an agent judgment, not a fixed sleep/window rule.
- Shorten intervals during startup, failure triage, or dense signal changes; lengthen them for stable slow progress, expensive checks, or external rate limits.
- Treat timeout or wait-window exhaustion as "observation window ended", not failed evidence; inspect status, logs, progress, artifacts, and resource signals before deciding.
- Continue when signals still advance and there is no clear error, unsafe resource state, failed artifact, or user stop; stop, downgrade, or ask when failure evidence is concrete.
## Personal Researcher Direct Change
Internal research code, brain tooling, and project scripts default to objective-first direct rewrite when that closes the current goal more simply.
- Optimize for current goal closure, system simplicity, verification, and git rollback; not for small diffs or legacy compatibility rituals.
- Compatibility is evidence-gated: keep old entrypoints, wrappers, aliases, and compatibility layers only with real callers, evidence value, or external interface duty.
- Remove stale shells/tests/helpers when they no longer serve the current system.
- Safety boundaries still come from `project_profile`, active/evidence guards, unrecoverable artifacts, secrets, live/default execution, and cross-project namespace rules.

## Init And Register
Initialize a project brain only when mutation is allowed and no local `brain/brain_manifest.json` exists. `--brain-id` is optional; default comes from the project directory name.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py init --cwd <project-root> [--brain-id <project_id>]
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py register --cwd <project-root> --workspace-root . [--brain-id <project_id>]
```
`register` attaches the project to the workspace main manifest and catalog; do not hand-edit child lists unless the runtime cannot run.

## On-Demand Health
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode full --timeout-sec 60
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Compact health is a fast takeover summary only; first hop stays `detect` + lite capsule. `tools.brain.workflow health` defaults to compact checks; use `--mode standard` for lightweight doc scope and `--mode full` or explicit include flags for project consistency / OpenMP strict lanes. Run full health, audits, and project-profile guards only for maintenance, diagnosis, or completion verification. Surface proposed or approved agent-learning items when checked.

## Proposal-Only Evolution
Agent Meta Protocol learning is proposal-only. Low-risk observations may create `proposed` records; protocol, workflow, skill, guard, or behavior changes require explicit user approval before implementation.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py proposal --cwd . --title "<short>" --trigger "<fact>" --evidence "<path or observation>" --recommendation "<proposal>" --severity info --owner-brain workspace --writeback-target brain/references/
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py list-proposals --cwd .
```

## Final Guards
For brain/tooling edits, run relevant tests plus:
```powershell
git diff --check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --files <changed-brain-docs>
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Bare `doc_guard check` is the full/final maintenance guard; ordinary changed-surface brain docs use `--files` or `--scope changed`. `daily_research/tools/project_consistency_check.py` defaults to `--mode research`; execution/full modes are only for execution or complete maintenance.
Write workspace decisions to `brain/`; routed project facts to the selected child brain; long evidence to `references/`.
