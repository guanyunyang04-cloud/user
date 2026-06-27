# T0 项目脑架构

`t0_project/brain/` 继承主脑多范式自然语言程序模型：对象描述实验面和执行面，过程描述验证/运行/上行复核，函数描述任务是否触碰生产边界。

## Object Layer
- `t0_project`: intraday experiment brain.
- `execution_experiment_surface`: paper/mock/live adapter boundary.
- `production_handoff`: results that may affect `daily_research`.

## Procedure Layer
- `offline_static_acceptance`
- `changed_surface_validation`
- `real_adapter_review`
- `production_relevance_review`

## Function Layer
- `select_execution_mode(task)`
- `requires_production_review(result)`
- `classify_t0_result(run)`

## Body Map
- strategies, monitor, execution, RL and gateway entries are declared in `operations_center.md`.
