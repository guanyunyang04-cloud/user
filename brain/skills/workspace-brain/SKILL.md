---
name: workspace-brain
description: Use when the user mentions brain, 脑区, takeover, repository governance, project structure, protected data, or complexity reduction.
---
# Workspace Brain

The brain is a small takeover map. Do not turn it into a workflow engine.

## Read only what the task touches

1. Read `brain/README.md` for workspace topology.
2. Read `brain/object_registry.json` when ownership or protected data matters.
3. Read `brain/state.md` and the relevant child `README.md`/`state.md` for current
   facts.
4. Read archived references only when historical evidence is actually needed.

QDP owns canonical datasets. Daily Research owns downstream packs, registered
models, compact research records, and active studies.

## Mutation rules

- Protect QDP data, research-store data, and registered checkpoint bytes.
- Prefer direct changes to the owning module. Do not add wrappers, fallback
  modes, compatibility aliases, or duplicated registries by default.
- A permanent abstraction needs at least two current consumers.
- An experiment begins as one contract under `daily_research/studies/`; generated
  material stays in ignored output.
- When an experiment finishes or is abandoned, retain its compact conclusion,
  key metrics, semantic contract, and selected checkpoint, then remove its
  runner, logs, predictions, partial tasks, and experiment-only tests.
- Brain hot paths contain current facts only. History belongs in references.

## Verification

Use only checks that support the claim:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development verify
git diff --check
```

For QDP data work, also run QDP status and the appropriate quick/full check. For
training or evaluation semantics, run the focused PIT, purge, candidate,
execution, and numerical tests.

The installed copy at `C:/Users/ASUS/.codex/skills/workspace-brain/SKILL.md` is a
deployment copy. The repository file is canonical and both must match.
