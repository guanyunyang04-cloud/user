# Daily Research Brain Operating Protocol

Snapshot date: `2026-05-14`

## Purpose
This protocol turns the `daily_research` brain from a passive memory store into the default operating system for agents working in this project.

The brain remains the source of project facts, evidence, state, and governance. Skills and tools only provide reusable procedures for reading, checking, and writing that truth.

## Entry Sequence
- Start every substantial `daily_research` task with a task capsule:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<task>" --json`
- Read the capsule before changing tracked files, starting training, running studies, or writing conclusions.
- If the task involves a specific study, protocol, dataset, or r-number, use explicit tags or evidence references rather than loose `latest_*` pointers.
- Keep work on `main` unless the user explicitly changes the branch rule.

## Action Discipline
- Before major actions, separate facts, inferences, assumptions, and action boundaries.
- During work, prefer existing brain workflow tools over ad hoc interpretation.
- After major actions, run the relevant guards and decide whether a brain writeback is required.
- If no writeback is required, say why.

## Evidence Rules
- Failed, interrupted, timeout, smoke, dry-run, and diagnostic-only runs are not completed evidence.
- Realtime tail labels with unobserved forward outcomes are never completed training evidence.
- Full Gold training-set claims require a registered Gold data-lake dataset with row counts, date range, and label completeness.
- Active execution changes require explicit future promotion authority; ordinary research work must leave `daily_research/output/active_execution_strategy.json` unchanged.

## Brain And Skill Split
- Brain files store project truth: current state, rules, evidence, design contracts, and historical verdicts.
- Project skills store repeatable procedures: handoff, preflight, evidence lookup, guarded execution, and writeback.
- Skills must point back to the brain and must not copy long r-number histories.

## Writeback Rules
- Main centers stay compact and current.
- Long dated evidence, command transcripts, and detailed r-number status go to `daily_research/brain/references/`.
- Machine-readable evidence indexing belongs in `daily_research/brain/references/evidence_registry.json`.
- `workflow_registry.json` describes workflows; it is not a replacement for brain truth.
- After any full codebase review, update `daily_research/brain/references/full_codebase_review_20260515.md` or its explicit successor.
- After any mainline review, reroute, deprecation, or promotion analysis, update `daily_research/brain/references/mainline_review_current.md`.
- After adding or materially changing reference review documents, rebuild the evidence registry and run the brain/document guards before claiming the writeback is complete.
