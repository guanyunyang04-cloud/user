# T0 Project 过程目录

## Body Map Objects
- `strategy_entry`: `t0_project/integrated_tq_strategy.py`, `t0_project/backtest_integrated_strategy.py`, `t0_project/select_stocks_only.py`
- `monitor_entry`: `t0_project/my_t0_monitor.py`
- `execution_entry`: `t0_project/execution`
- `rl_entry`: `t0_project/rl_agent`
- `gateway_entry`: `t0_project/tqcenter.py`

## Procedure Entries
### procedure `enter_t0_project`
`input`: task
`steps`: select experiment object；identify body entry；choose offline/mock/paper/real-adapter mode；run targeted work；write durable result to brain/reference.

### procedure `changed_surface_validation`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`
`semantics`: run returned blocking commands when mapped; use manual review or focused tests when not mapped.

### procedure `python_static_compile`
`command`: `python -m py_compile <all t0_project/**/*.py>`
`semantics`: shared-entry or maintenance validation; not a real trading validation.

### procedure `real_adapter_review`
`input`: task touching real TDX, broker or live adapter.
`steps`: activate `execution_experiment_surface`；state execution mode；inspect side effects；seek explicit task scope；write evidence.

## Writeback Routes
- Current experiment objects: `state_center.md`
- Stable experiment classes and lessons: `knowledge_center.md`
- Body map and commands: `operations_center.md`
- Object invariants: `governance_layer.md`
- Single-run evidence: `episodic_memory.md` or `references/`
