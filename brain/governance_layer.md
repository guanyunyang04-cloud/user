# 主脑治理对象
快照日期：`2026-07-01`

治理是对象语义的一部分，不是全局流程模板。

## Agent Meta Protocol
主脑和分脑遵循 `agent meta protocol`：任何会改变长期项目事实、执行边界、数据基底语义、active 指针或跨项目路由的工作，都必须选择受影响对象、执行最小验证，并把稳定结果写回对应 hot path；如果不写回，最终回复必须说明 `brain_sync=false` 的原因。

## Governed Object Types
### object `active_data_base_or_unique_data`
`scope`: QDP v2 `active.json`、`dataset.json`、raw parquet、PIT 状态、不可重建研究证据。
`methods`: `rebuild()`；`switch_pointer()`；`cleanup()`；`delete()`；`migrate()`。
`invariant`: destructive or pointer-changing methods carry replacement pointer, dry-run, sample validation or explicit recovery story.

### object `pit_or_label_semantics`
`scope`: PIT/no-leakage、label completeness、future availability、evidence grade。
`methods`: `build_dataset()`；`change_feature_or_label()`；`train()`；`evaluate()`；`write_conclusion()`。
`invariant`: unobserved labels, future information, low-budget experiments and incomplete runs keep their evidence grade visible.

### object `active_execution_artifact`
`scope`: live/default/paper/broker、active artifact、promotion gate、trade plan。
`methods`: `activate()`；`restore()`；`update()`；`generate_trade_plan()`；`wire_broker()`。
`invariant`: execution changes are selected only by execution tasks with explicit authority and child-brain promotion evidence.

### object `secret_or_external_state`
`scope`: secrets、accounts、external service state、remote deployment、real trading service。
`methods`: `create()`；`rotate()`；`write()`；`deploy()`；`connect()`。
`invariant`: secret material stays outside brain docs and ordinary reports.

### object `cross_project_dirty_work`
`scope`: unrelated dirty paths、background processes、ports、GPU jobs、provider tasks、outputs。
`methods`: `inspect()`；`manage()`；`delete()`；`reuse()`；`commit()`；`wait()`。
`invariant`: unrelated objects are summarized unless the user expands scope or grants a lease.

### object `brain_sync_surface`
`scope`: main brain, child brain hot paths, workspace skill, brain workflow tools.
`methods`: `sync()`；`audit()`；`archive_history()`。
`invariant`: stable project fact changes require hot-path writeback or an explicit `brain_sync=false` reason in final.

### object `object_registry`
`scope`: `brain/object_registry.json` plus loader and CLI consumers.
`methods`: `match_task()`；`match_paths()`；`derive_writeback_targets()`；`derive_validation_commands()`。
`invariant`: object ownership, routing terms, path prefixes and closure validation hints should live in the registry first; code may still apply judgment, but should not duplicate the same object table in multiple modules.

## Pure Functions
- `select_governed_objects(task, paths)`: returns only objects touched by the task and method, using `brain/object_registry.json` when machine-readable evidence is available.
- `select_validation(task, changed_paths)`: chooses the smallest validation that can support the conclusion.
- `route_writeback(result)`: maps current state, stable facts, procedure entries, governance, project facts and long evidence to their files.
- `classify_change_risk(object, method)`: returns ordinary, shared, protected, destructive or external-state risk.
- `requires_brain_sync(change)`: true for durable architecture, CLI, active pointer, data quality, research conclusion or execution-boundary change.

## Procedure Entries
### procedure `normal_task`
`input`: user task
`steps`: understand goal；select objects；act；validate enough；write back only when the result changes durable memory.
`side_effects`: selected object surface only.

### procedure `protected_object_change`
`input`: selected protected object, method, evidence, user authority if needed.
`steps`: inspect object state；state replacement/recovery/evidence path；make scoped change；run object-level validation；write durable summary.
`side_effects`: protected object artifacts or brain memory.

### procedure `brain_structure_reduction`
`input`: brain docs, skill, manifest, workflow or registry changes.
`steps`: keep hot-path files as object tables；move long evidence to references；remove duplicated entry rules；run structure audit, sync audit and multi-paradigm lint.
`side_effects`: brain docs, registry, skill files.

## Guard Entrypoints
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow closure-check --task "<task>" --paths <changed_paths> --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.brain_sync_audit --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-structure-audit --cwd . --mode compact`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py multi-paradigm-lint --cwd . --scope attached`
