# Quant Data Platform 过程目录

## Runtime Objects
### object `qdp_body_map`
`code`: `quant_data_platform/src`
`configs`: `quant_data_platform/configs`
`registry`: `quant_data_platform/registry`
`data`: `quant_data_platform/data`
`references`: `quant_data_platform/brain/references`
`tests`: `quant_data_platform/tests`

### object `qdp_python_env`
`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`
`pythonpath`: `H:/quant_project/quant_data_platform/src;H:/quant_project`

## Procedure Entries
### procedure `inspect_qdp_status`
`command`: `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json`
`output`: current registry, lake, coverage and canonical status.

### procedure `provider_eval_smoke`
`command`: `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli provider-eval --providers current_qdp --symbols 000001.SZ --windows 2024-06-03:2024-06-07 --json`
`output`: provider connectivity / current_qdp smoke evidence.

### procedure `audit_qdp`
`command`: `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli audit --json`

### procedure `build_sharded_memmap_smoke`
`command`: `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli build-sharded-memmap --profile short_horizon_core_v1 --start-year 2022 --end-year 2022 --max-universe-size 10 --max-shards 1 --json`

### procedure `validate_memmap`
`command`: `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli validate-memmap --manifest <sharded_manifest.json> --json`

### procedure `qdp_tests`
`smoke`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest quant_data_platform/tests -m "smoke and not data_heavy and not external and not benchmark" -q`
`ordinary`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest quant_data_platform/tests -m "not data_heavy and not external and not benchmark" -q`
`full`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest quant_data_platform/tests -q`
`note`: `test_sharded_memmap.py` is integration/data_heavy and selected for memmap/shard/schema changes.

### procedure `cleanup_dry_run`
`semantics`: cleanup first creates a plan; deletion follows only after replacement pointer and enough validation.

## Writeback Routes
- Current data pointers and provider results: `state_center.md`
- Stable source semantics and lessons: `knowledge_center.md`
- Commands and process entries: `operations_center.md`
- Protected data invariants: `governance_layer.md`
- Long audits/provider reports: `references/` or `data/audits` / `data/provider_eval`
