# T0 Project 身份对象

## object `t0_project`
`type`: intraday_experiment_brain
`definition`: 盘中 T+0、执行抽象与 RL 原型的独立实验分脑。
`not`: `daily_research` 正式生产执行主线。
`principle`: Agent 无状态，项目大脑有状态。
`north_star`: 验证盘中执行抽象是否清晰、可验证、可逐步接入真实接口。
`staging_path`: paper -> human confirmation -> real adapter.
`methods`: `inspect_experiment_state()`；`run_mock_or_paper_test()`；`write_experiment_evidence()`；`request_production_review()`。

## object `execution_experiment_surface`
`type`: protected_experiment_surface
`state`: live remains protective skeleton; paper/mock is the default experimentation mode.
`activation`: broker, live adapter, real order, production handoff, or daily_research implication.
`invariant`: experimental evidence does not become production default without main brain and `daily_research` review.

## Pure Functions
- `is_production_relevant(task) -> bool`
- `select_execution_mode(task) -> offline|mock|paper|human_confirmed|real_adapter`
- `classify_t0_result(run) -> experiment_evidence|diagnostic|production_review_required`

## Routing
- Current experiment state: `state_center.md`
- Stable lessons and object semantics: `knowledge_center.md`
- Body map and commands: `operations_center.md`
- Protected experiment invariants: `governance_layer.md`
