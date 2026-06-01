# Daily Research V2 Research Reset - 2026-06-01

## Decision

- Status: `daily_research_v2_research_reset_accepted / research-only / execution-frozen`.
- User decision: switch to a new mainline instead of continuing to pursue strict old Stage 2.8 / short_v5b replay.
- New mainline name: `daily_research_v2_research_reset`.
- Data foundation: BaoStock-first data lake with corrected mainboard pool.
- Corrected dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Corrected mainboard pool: `policy_pool_view__74f45f4f83263bccd64a8027`.
- Current anchor baseline: `mh_rebuild_mainboard_anchor_20260601_01`.
- Execution status remains: `frozen_skeleton_only / awaiting_research_rebuild`.

## Evidence Behind The Reset

- TQ vs BaoStock audit shows the core open / next-open label chain is usable:
  - TQ status: `ok`.
  - Best comparison dividend type: `none`.
  - TQ `none` vs BaoStock price p95 relative diff mean: `0.0`.
  - Next-open label status: `label_equivalent`.
  - Max hit label flip rate: `0.0014824531454960379`.
  - Future decision score correlation: `0.999750308636608`.
  - Top20 hit label flip rate: `0.0`.
- Corrected mainboard-only baseline is not a failure state:
  - Gate status: `near_pass`.
  - Test rank IC min: `0.084074`.
  - Test spread min: `0.031943`.
  - Mean monthly positive rate: `0.848485`.
  - Max negative months: `2`.
  - Remaining blocker: hit lift min `-0.007156`.
- Old strict replay remains blocked:
  - Old Stage 2.8 / Stage 3G / short_v5b physical payloads are not locally replayable.
  - Old `156` feature schema has not been recovered.
  - New corrected feature store is `[1699,2596,116]`, not old `[1699,2430,156]`.

## Interpretation

- The main blocker is no longer evidence that the BaoStock-first data lake is unusable.
- The best current explanation for the old/new result gap is input and protocol non-equivalence: feature schema, pool/sample policy, amount unit, missing/fill policy, and unavailable old payloads.
- Continuing to chase exact old replay has diminishing value because the old artifacts are missing and the old result was not necessarily a final optimum.
- Old Stage 2.8 / short_v5b evidence remains useful as historical prior and benchmark context, but it is no longer the active research target.

## New Research Contract

- Start from the new data lake and corrected mainboard universe.
- Build forward with explicit dataset id, pool id, feature profile, label semantics, benchmark, cost, seed, and gate.
- Treat current `116` features as the current baseline input contract; do not fabricate `legacy156`.
- Continue amount-unit and missing/fill audits only insofar as they affect new feature quality.
- Retain strict multi-seed and monthly stability gates:
  - rank IC positive,
  - top-bottom spread positive,
  - hit lift positive,
  - monthly positive rate stable,
  - negative months bounded,
  - concentration controlled,
  - cost and next-open execution assumptions explicit.
- Future execution rebuild must be based on new v2 evidence, not restoration of old `short_v5b`.

## Governance

- Do not promote corrected `near_pass` baseline.
- Do not restore live/default, paper/broker, production root, or active manifest from this decision.
- Do not delete execution code; keep the frozen skeleton and candidate evaluation wrapper.
- If old `156` manifest or short_v5b payload is found later, use it as a side audit or historical bridge, not as a blocker to v2 progress.

## Next Allowed Actions

- Define the first v2 feature-input audit and candidate feature profile roadmap.
- Investigate amount-unit normalization and missing/fill policy only as new-lineage feature quality issues.
- Run a narrow hit-lift repair matrix on the corrected mainboard baseline.
- Design the first v2 model comparison using the corrected dataset/pool and explicit gates.
- Keep execution frozen until v2 has evidence-grade candidate results and a separately approved execution rebuild plan.
