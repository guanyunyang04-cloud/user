# 主脑治理对象
快照日期：`2026-06-27`

本文件描述主脑如何选择受保护对象、验证方式和写回位置。治理是对象语义的一部分，不是全局流程模板。

## Governed Object Types
### object `canonical_or_unique_data`
`scope`: QDP canonical lake、registry pointer、policy bundle、memmap、唯一研究证据和不可重建资产。
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

### object `Agent Meta Protocol`
`scope`: agent self-correction, learning writeback, authorization state and repeated failure patterns.
`methods`: `inspect_lesson()`；`record_learning()`；`check_authorization()`。
`invariant`: meta learning supports object judgment; it does not replace user intent, file evidence or project-specific brain objects.

## Pure Functions
- `select_governed_objects(task)`: returns only objects touched by the task and method.
- `select_validation(task, changed_paths)`: chooses the smallest validation that can support the conclusion.
- `route_writeback(result)`: maps current state, stable facts, procedure entries, governance, project facts and long evidence to their files.
- `classify_change_risk(object, method)`: returns ordinary, shared, protected, destructive or external-state risk.
- `should_propose(change)`: true for uncertain/high-risk brain evolution; false for confirmed low-risk semantic cleanup.

## Procedure Entries
### procedure `normal_task`
`input`: user task
`steps`: understand goal；select objects；act；validate enough；write back only when the result changes durable memory.
`side_effects`: selected object surface only.

### procedure `protected_object_change`
`input`: selected protected object, method, evidence, user authority if needed.
`steps`: inspect object state；state replacement/recovery/evidence path；make scoped change；run object-level validation；write durable summary.
`side_effects`: protected object artifacts or brain memory.

### procedure `brain_burden_reduction`
`input`: brain docs, skill, manifest, workflow or registry changes.
`steps`: keep hot-path files as object tables；move long evidence to references；remove duplicated entry rules；run burden audit.
`side_effects`: brain docs, registry, skill files.

## Writeback Routes
- Current object instances: `state_center.md`
- Stable object classes and lessons: `knowledge_center.md`
- Object model and module semantics: `brain_architecture.md`
- Procedure entries and validation ways: `operations_center.md`
- Governed object invariants: `governance_layer.md`
- Project facts: corresponding child brain
- Durable evidence: `references/`

## Guard Entrypoints
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`
