# T0 Project 知识对象

## Object Classes
### class `intraday_experiment`
`definition`: 盘中策略、执行抽象、monitoring and RL prototype under experiment scope.
`truth_scope`: conclusions hold inside t0 experiment boundary unless promoted through review.

### class `execution_adapter_stage`
`states`: offline、mock、paper、human_confirmed、real_adapter.
`semantics`: each state describes available evidence and side effects, not model quality by itself.

### class `production_relevance`
`trigger`: result changes `daily_research` assumptions, execution semantics, live/default, or broker wiring.
`route`: main brain plus `daily_research`.

## Long-Term Lessons
- Intraday experiments need isolation, or they can pollute production decisions.
- Body entrypoints and brain map must stay aligned for handoff quality.
- Static validation is engineering evidence; real trading claims require adapter and execution evidence.

## Pure Functions
- `requires_production_review(result) -> bool`
- `is_real_adapter_task(task) -> bool`
- `select_t0_validation(changed_paths) -> static|mock|paper|manual_review`
