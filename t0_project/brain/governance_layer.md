# T0 Project 治理对象

## Governed Objects
### object `experiment_boundary`
`scope`: t0 experiment conclusions, RL prototypes, intraday strategy tests.
`invariant`: experiment evidence stays inside t0 unless routed through production relevance review.

### object `real_execution_adapter`
`scope`: TDX terminal, broker interface, live adapter, real order path.
`invariant`: real side effects require explicit task scope and mode classification.

### object `production_handoff`
`scope`: conclusions that may affect `daily_research` production defaults or execution semantics.
`invariant`: handoff goes through main brain and `daily_research`, with original evidence retained in t0.

## Pure Functions
- `select_governed_object(task) -> experiment_boundary|real_execution_adapter|production_handoff`
- `is_real_side_effect(task) -> bool`
- `requires_daily_research_review(result) -> bool`

## Procedures
### procedure `experiment_work`
`input`: t0 task and selected object
`steps`: choose mode；execute experiment or code change；validate locally；write evidence if durable.

### procedure `production_handoff_review`
`input`: t0 result with possible production relevance
`steps`: summarize facts and evidence grade；write t0 reference；route summary to main brain and `daily_research`.
