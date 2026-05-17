# alpha_path20 Mainline Switch Status 2026-05-17

## Verdict
- Status: `current path20 research mainline pointer / research / shadow-only / neural-policy route`.
- User decision on 2026-05-17 switches the current Path20 research mainline pointer to `alpha_path20_neural_policy_v1`.
- The Path20 research mainline is intentionally switchable; this record is the current routing state, not a permanent hierarchy.
- `alpha_path20_sequence_policy_v1` remains available as a shadow comparison / secondary research route, but is no longer the current Path20 mainline.
- This is not live/default promotion; `active_execution_strategy.json remains unchanged`.

## Facts
- Protocol stages `dataset-smoke`, `oracle-smoke`, and `tiny-smoke` are now routed to the current neural-policy mainline.
- These stages no longer require `--allow-legacy-neural-policy`.
- The parser default stage is `dataset-smoke`.
- The parser default policy version for neural stages is `alpha_path20_neural_policy_v1`.
- Sequence/RL stages still require `alpha_path20_sequence_policy_v1` and remain shadow-only.
- All Path20 runs still require explicit tags and fixed lake dataset ids; loose `latest/default/latest_*` ids remain forbidden.
- Existing tiny smoke evidence remains `alpha_path20_neural_policy_v1_tiny_smoke_20260516_01` with lake dataset `policy_input_bundle__0f116a9b78c92ff045a6853d`.

## Inferences
- The current research question moves to path-label, oracle upper-bound, forecaster baseline, and neural target-weight allocation quality.
- The first useful next evidence is not live/default promotion; it is a cleaner neural-policy diagnostic loop that explains whether forecast quality, projection dominance, or allocator utility is the limiting factor.
- Sequence/RL evidence should be interpreted as comparison evidence, not as the current Path20 mainline verdict.

## Assumptions
- The user decision is a switch of the research-routing pointer, not permission to start long training or write active execution artifacts.
- `policy_input_bundle__0f116a9b78c92ff045a6853d` remains the near-term fixed market lake input.
- `next_open` remains the default execution alignment unless a future task explicitly changes it.

## Boundaries
- Shadow-only: no live/default promotion.
- Mainline selection is switchable by explicit future user/project decision.
- No writes to `daily_research/output/active_execution_strategy.json`.
- No loose latest references.
- No long training without a separate explicit task.
- Oracle upper-bound remains an upper-bound diagnostic; it is not deployable evidence by itself.
- Smoke/unit/schema success remains plumbing evidence, not strategy success.

## Next Allowed Actions
- Run neural-policy `dataset-smoke` on a fixed lake slice to revalidate path-label schema and current defaults.
- Run `oracle-smoke` only to quantify upper-bound replay and projection/turnover behavior.
- Run `tiny-smoke` only as short diagnostic evidence for forecast quality and allocator utility.
- Diagnose whether the limiting factor is path forecast quality, target-weight projection dominance, or allocator objective shape before expanding budget.
