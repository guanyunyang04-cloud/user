# r66 Brain Maintenance Status - 2026-05-14

## Summary
- Status: `brain / workflow maintenance / control-plane compaction`.
- Production anchor: `daily_research/output/active_execution_strategy.json` remains unchanged.
- This is not strategy evidence, training evidence, promotion support, live/default change, or continuous_policy effectiveness evidence.
- This note uses explicit brain-tool checks and file paths only; it does not infer conclusions from loose `latest_*`.

## Implemented Facts
- Compacted `daily_research/brain/state_center.md` from long dated-log shape into current control-plane state.
- Compacted `daily_research/brain/operations_center.md` into current commands, workflow discipline, data lake commands, and continuous_policy execution rules.
- Compacted `daily_research/brain/continuous_policy_design_contract.md` into current design boundaries, active r65 research contract, success criteria, known blockers, and historical index.
- Compacted `daily_research/brain/knowledge_center.md` into stable facts, hard rules, long-term lessons, methodology, and research-line index.
- Preserved detailed history in existing `daily_research/brain/references/` status docs and archives rather than deleting evidence.
- Updated `daily_research.tools.brain_evidence_registry` to use section-aware evidence extraction for verdict, blockers, and next actions.
- Updated evidence registry tags so portfolio-set v5 evidence is not mislabeled as core-v4 simply because it mentions core-v4 as a baseline.
- Updated `daily_research.tools.brain_rules` with control-plane document length warnings.
- Rebuilt `daily_research/brain/references/evidence_registry.json`.

## Line Count Before And After
- `state_center.md`: before `234`, after `66`.
- `knowledge_center.md`: before `158`, after `64`.
- `operations_center.md`: before `175`, after `75`.
- `continuous_policy_design_contract.md`: before `257`, after `76`.
- All four primary control-plane docs are now under their r66 targets.

## Tooling Fixes
- Evidence registry extraction now prefers explicit sections such as Summary, Verdict, Blocker, Behavior Evidence, Safe Screening Evidence, and Next.
- Evidence registry no longer treats early summary bullets as next allowed actions.
- Evidence registry keeps dataset id queries working for r64/r65 strict Gold evidence.
- Brain rules can now warn when control-plane docs exceed configured line targets without hard-failing first-stage maintenance.
- CLI compatibility is preserved for `brain_workflow capsule`, `preflight`, `query`, and `evidence-index`.

## Preserved Evidence
- r61 core-v4 decision wiring: `daily_research/brain/references/r61_release_first_decision_core_v4_status_20260514.md`.
- r62 research data lake: `daily_research/brain/references/r62_research_data_lake_status_20260514.md`.
- r63 brain-skill operating system: `daily_research/brain/references/r63_brain_skill_operating_system_status_20260514.md`.
- r64 full-universe strict Gold: `daily_research/brain/references/r64_full_universe_gold_data_lake_status_20260514.md`.
- r65 portfolio-set v5: `daily_research/brain/references/r65_portfolio_set_v5_status_20260514.md`.
- Historical archives remain available under `daily_research/brain/references/*archive*` and `*_history_raw_*.md`.

## Verification
- Brain tool regression passed:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/tools/tests/test_brain_capsule.py daily_research/tools/tests/test_brain_evidence_registry.py daily_research/tools/tests/test_brain_rules.py daily_research/tools/tests/test_brain_workflow_cli.py daily_research/tools/tests/test_daily_research_brain_skill.py -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow evidence-index --rebuild --json` returned registry `status=ok`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow query --q r65 --json` returns the r65 portfolio-set v5 evidence record.
- `git diff --check` passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check` passed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json` returned `status=ok`.
- `git diff -- daily_research/output/active_execution_strategy.json` produced no output.

## Remaining Issues
- Loose latest artifacts can still be stale-risk when latest study and latest protocol point to different tags; explicit tag queries remain required.
- `evidence_registry.json` is an index, not the evidence source of truth. Status docs and protocol/study artifacts remain canonical.
- Background OS launcher / poller for long studies is still only a proposed workflow, not yet a dedicated implemented CLI.
- Full-window realtime Gold is still pending and must not be counted as training-safe evidence.
- r65 portfolio-set v5 behavior remains source/receiver dead; this maintenance round does not change strategy behavior.

## Boundary
- r66 is brain/workflow maintenance only.
- Do not use r66 as strategy evidence, training evidence, confirmatory evidence, promotion support, live/default change, or active artifact change.
