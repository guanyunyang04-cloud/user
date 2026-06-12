# Quant Data Platform

Canonical brain source: `quant_data_platform/brain/identity_layer.md`.

主脑管辖下的共享量化数据平台分脑。本文只保留项目入口；数据契约、memmap 治理和 canonical v1 说明以 `quant_data_platform/brain/` 为准。

## Entry

- 项目身份：`quant_data_platform/brain/identity_layer.md`
- 当前状态：`quant_data_platform/brain/state_center.md`
- 操作规则：`quant_data_platform/brain/operations_center.md`
- 参考文档：`quant_data_platform/brain/references/`

## Body

- `src/`：平台代码。
- `configs/`：canonical 数据域与 profile 配置。
- `registry/`：可审计 manifest、registry 和指针。
- `data/`：大数据资产、tmp、memmap、agent runs，默认不进 Git。
- `tests/`：平台测试。
