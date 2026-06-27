# Traditional Quant Research 过程目录

## Body Map Objects
- `core_surface`: `traditional_quant_research/`
- `experiment_surface`: `traditional_quant_research/experiments/`
- `data_surface`: `traditional_quant_research/data/`
- `tests_surface`: `traditional_quant_research/tests/`
- `research_logs`: `traditional_quant_research/brain/references/research_log/`
- `data_catalog`: `traditional_quant_research/brain/references/data_catalog.md`

## Procedure Entries
### procedure `enter_traditional_quant`
`input`: task
`steps`: select research object；read state/knowledge/operation and explicit research logs as needed；run scoped code or analysis；write durable conclusion to research log/reference.

### procedure `changed_surface_validation`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json`
`semantics`: run returned blocking commands for the changed surface.

### procedure `ordinary_test_lane`
`command`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research -m "not research and not slow and not data_heavy and not external and not benchmark" -q`
`semantics`: shared core/schema changes; research experiments use explicit nodeid or stage-specific runs.

### procedure `new_experiment`
`input`: hypothesis, evidence grade, data contract and output location.
`steps`: define hypothesis；write/run experiment；store result under `brain/references/research_log/`；update state only with current summary.

### procedure `data_ingest_or_catalog_update`
`input`: new field/source/data cache.
`steps`: update `brain/references/data_catalog.md`；state field semantics, date range, adjustment, survivorship and PIT/source-grade handling；run focused audit.

## Writeback Routes
- Current candidate pool and next pointer: `state_center.md`
- Stable methods and lessons: `knowledge_center.md`
- Commands and procedures: `operations_center.md`
- Evidence gating and protected objects: `governance_layer.md`
- Long research evidence: `brain/references/research_log/`
