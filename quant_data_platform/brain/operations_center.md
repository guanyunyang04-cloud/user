# Quant Data Platform 操作中枢

## 接管入口
- 主脑路由：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --intent read --verbosity lite --json`
- 本分脑 bootstrap：
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain quant_data_platform --json`

## Body Map
- `quant_data_platform/src`：平台代码。
- `quant_data_platform/configs`：canonical/profile 配置。
- `quant_data_platform/registry`：registry、root manifest、memmap 指针。
- `quant_data_platform/data`：大数据资产、tmp、agent_runs、sharded memmap，默认 Git 忽略。
- `quant_data_platform/brain/references`：数据契约、清理策略、memmap 设计的 canonical 正文。
- `quant_data_platform/tests`：平台单元与集成测试。

## 常用验证
- `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json`
- `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli audit --json`
- `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli build-sharded-memmap --profile short_horizon_core_v1 --start-year 2022 --end-year 2022 --max-universe-size 10 --max-shards 1 --json`
- `PYTHONPATH=H:/quant_project/quant_data_platform/src;H:/quant_project C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli validate-memmap --manifest <sharded_manifest.json> --json`
- `git diff --check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check --scope changed`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest quant_data_platform/tests -q`
- 轻量 smoke lane：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest quant_data_platform/tests -m "smoke and not data_heavy and not external and not benchmark" -q`
- 普通开发 lane：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest quant_data_platform/tests -m "not data_heavy and not external and not benchmark" -q`
- `test_sharded_memmap.py` 属于 `integration + data_heavy`，改 memmap/shard/schema 时显式运行；不让它拖慢无关 brain / registry / profile 小改。

## 清理纪律
- `qdp cleanup --dry-run` 只生成计划，不删除。
- 删除旧 parquet、旧 bundle 或旧 `.dat` 前，必须有 canonical bundle、canonical memmap registry、随机一致性验证和替代指针。
