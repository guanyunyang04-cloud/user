# T0 Project 状态程序

## Object Instances
### object `t0_experiment_project`
`type`: intraday_experiment_instance
`state`: workspace 内盘中执行、执行抽象与 RL 实验分脑；不直接替代 `daily_research` 正式生产主线。
`current_focus`: 保持盘中执行与 RL 研究可接管、可验证、可复盘，并保持与 `daily_research` 的边界清晰。

### object `maintenance_20260511`
`type`: validation_evidence
`state`: 只做离线静态验收，未连接真实通达信终端、券商接口或 live broker。
`evidence`: `t0_project/**/*.py` full `py_compile` passed.
`conclusion`: no experiment conclusion promoted to `daily_research`.

### object `execution_mode`
`type`: runtime_boundary
`state`: default `offline/mock/paper`; real broker or live adapter work requires explicit task activation and review.

## Pure Functions
- `select_relevant_objects(task)`: experiment state, body map, execution mode or production review.
- `derive_next_action(state)`: maintain isolation, align body map, and write production-relevant conclusions to main brain plus `daily_research`.
- `classify_validation(run)`: static compile is code health evidence, not real trading evidence.

## Procedures
### procedure `offline_static_acceptance`
`input`: changed t0 code paths
`steps`: inspect experiment boundary；run targeted static or compile checks；write diagnostic evidence if durable.
`side_effects`: t0 evidence only.

### procedure `production_relevance_review`
`input`: result that may affect formal execution
`steps`: summarize experiment evidence；route to main brain and `daily_research`；keep original result in t0 references.
`side_effects`: brain writeback, no direct production mutation.
