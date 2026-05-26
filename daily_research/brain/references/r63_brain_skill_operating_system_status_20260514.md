# r63 Brain-Skill Operating System Status

Date: `2026-05-14`

## Summary
- Fact: r63 adds `daily_research/brain/brain_operating_protocol.md` as a compact operating protocol for task capsules, preflight, evidence rules, and writeback discipline.
- Fact: `brain_workflow.py` now exposes `capsule`, `evidence-index`, and `query` commands in addition to existing handoff/status/preflight/writeback-plan paths.
- Fact: `daily_research/brain/references/evidence_registry.json` is now the machine-readable index for reference evidence. It indexes recent r-number docs, registry v3 program/family/run fields, blockers, next actions, active-artifact impact, and dataset ids.
- Fact: repo-canonical project skill `daily_research/brain/skills/daily-research-brain` now exists and points agents to brain capsule/preflight/query/writeback commands without copying long r-number history.
- Fact: `install_project_skills.py --dry-run` reports the skill sync plan but does not install globally unless `--install` is explicitly used.
- Fact: `brain_rules.py` adds phased guard checks for active artifact diffs, stale loose latest pointers, failed-trial completed-evidence mistakes, full Gold claims without catalog evidence, and realtime-tail training evidence misuse.

## Verification
- Targeted r63 tests passed: `11 passed`.
- `brain_workflow evidence-index --rebuild --json` wrote a registry with `status=ok`, `record_count=13`, no duplicate ids, and no missing paths.
- `brain_workflow query --q r62 --json` returns the r62 data-lake evidence record and dataset ids.
- `install_project_skills.py --dry-run` reports one project skill and does not create the target directory.

## Boundaries
- This is brain/workflow/skill infrastructure only.
- It is not continuous_policy strategy evidence.
- It does not change live/default/promotion status.
- It does not modify `daily_research/output/active_execution_strategy.json`.

## Next
- Continue using task capsules before substantial `daily_research` work.
- Keep main brain centers compact; put detailed r-number evidence in `references/` and rebuild `evidence_registry.json`.
- If desired later, explicitly run `install_project_skills.py --install` to sync the repo-canonical skill into the Codex global skills directory.
