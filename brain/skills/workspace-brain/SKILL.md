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
Only read child-brain context after routing returns `status=selected` and `target.kind=child`; if multiple children match, treat `ambiguous` as a blocker.

## Init And Register
Initialize a project brain only when mutation is allowed and no local `brain/brain_manifest.json` exists. `--brain-id` is optional; default comes from the project directory name.
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py init --cwd <project-root> [--brain-id <project_id>]
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py register --cwd <project-root> --workspace-root . [--brain-id <project_id>]
```
`register` attaches the project to the workspace main manifest and catalog; do not hand-edit child lists unless the runtime cannot run.

## Health And Audits
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Surface proposed or approved agent-learning items when checked.

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
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```
Write workspace decisions to `brain/`; routed project facts to the selected child brain; long evidence to `references/`.
