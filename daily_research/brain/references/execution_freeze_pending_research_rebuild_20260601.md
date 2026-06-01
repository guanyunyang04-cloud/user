# Execution Freeze Pending Research Rebuild - 2026-06-01

## Verdict

- Status: `execution_frozen_skeleton_only / awaiting_research_rebuild`.
- Scope: `daily_research` execution side.
- Active artifact impact: `daily_research/output/active_execution_strategy.json remains unchanged`.
- User decision: execution should enter a frozen, skeleton-retained state because the research-side result differences versus prior evidence are not yet understood.

## Rationale

- The current new lineage research substrate exists and is usable for diagnostics, but `mh_rebuild_infra_v2_fullpool_anchor_20260531_01` failed the continuation gate because seed11 hit lift was slightly negative and max negative months was `3`.
- The current new lineage is not a recovery of old Stage 2.8 / Stage 3G file-backed run payloads.
- Old `short_expert_policy_v5b` remains a historical live/default fact label, but local production payload and `active_execution_strategy.json` are currently unavailable for file-backed verification.
- The project must not treat either old brain-confirmed live facts or new research-only forecast evidence as sufficient execution authority.

## Frozen Execution Semantics

- Keep: execution code skeleton, manual help flow, read-only status pages, data readiness checks, candidate backtest wrappers, candidate trade-plan wrappers for non-production research evaluation, and active artifact guards.
- Freeze: live/default execution, active manifest promotion, production root rebuild, paper/live/broker integration, automated daily execution, and formal trade-plan production use.
- Allowed next work: research diagnostics, score-panel / target-weight-panel export design, same-protocol candidate backtest design, payload inventory, and explicit recovery planning.
- Disallowed next work without explicit user authorization: writing `active_execution_strategy.json`, claiming short_v5b is runnable from missing local payload, promoting multi-horizon research to live/default, or deleting execution code contracts.

## Rebuild Preconditions

- Explain the research-side difference between new lineage results and prior Stage 2.8 / Stage 3G / short_v5b evidence.
- Produce a same-protocol comparison plan for new multi-horizon models versus old short_v5b using common pool, costs, rebalance policy, benchmark, date splits, and candidate score/target-weight panel semantics.
- Confirm whether old production payload should be restored, archived as unavailable, or intentionally superseded.
- Only after those checks should execution be rebuilt around the selected research mainline.
