# 全脑区强重构合同 2026-05-18

## Summary
- Status: `brain_system_refactor / control-plane governance / no strategy change`.
- Goal: make all workspace brains discoverable, auditable, migration-safe, and language-consistent.
- Language policy: `zh_semantic_en_identifiers_v1`，即中文语义 + 英文工程标识。
- Active execution impact: `daily_research/output/active_execution_strategy.json` must remain unchanged.

## Facts
- Canonical root brain: `brain/`.
- Attached child brains: `daily_research/brain/`, `t0_project/brain/`, `daily_stock_analysis-main/brain/`.
- Noncanonical discovered brain-like roots:
  - `daily_research/cache/brain/`: `cache_legacy`，不作为 truth source。
  - `a_stock_daily_selection/brain/`: `missing_manifest`，只进入审计报告。
- Workflow metadata migrated from a large compatibility JSON to `daily_research/brain/workflows/registry.json` plus `playbooks/*.json`.

## Boundaries
- This refactor changes brain governance and tooling only.
- It does not change Path20, continuous_policy, data lake datasets, model training, allocator, replay, live/default, or promotion state.
- `daily_research/brain/references/evidence_registry.json` remains the daily_research evidence index, not a workspace-wide evidence store.

## Next
- Use `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow audit-brain --scope all --json` for future brain structure audits.
- Treat noncanonical brain roots as warnings until explicitly attached through `brain/brain_manifest.json`.
