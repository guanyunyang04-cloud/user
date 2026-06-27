# Traditional Quant Research 脑区架构

`traditional_quant_research/brain/` 继承主脑多范式自然语言程序模型：对象描述研究线和候选池，过程描述实验/验证/写回，函数描述证据等级和候选边界。

## Object Layer
- `traditional_quant_research`: traditional quant lab brain.
- `data_substrate_v2_1`: Baostock-only research data substrate.
- `frontier_personal_candidate_pool`: current personal backtest candidate set.
- `weak_year_problem`: persistent robustness blocker.

## Procedure Layer
- `formal_personal_candidate_review`
- `data_source_upgrade_review`
- `new_experiment`
- `changed_surface_validation`

## Function Layer
- `classify_research_result(run)`
- `requires_true_size_gate(task)`
- `classify_candidate(row)`
- `derive_next_research_action(state)`

## Body Map
Reusable research core, experiments, data assets and tests are in the project body; long logs and data contracts live under `brain/references/`.
