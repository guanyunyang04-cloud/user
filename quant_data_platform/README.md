# Quant Data Platform

Canonical brain source: `quant_data_platform/brain/identity_layer.md`.

主脑管辖下的共享量化数据平台分脑。本文只保留项目入口；当前 v2 active、v3 rebuild、安全阻断、数据契约和 provider 对象以 `quant_data_platform/brain/` 为准。

## Entry

- 项目身份：`quant_data_platform/brain/identity_layer.md`
- 当前状态：`quant_data_platform/brain/state_center.md`
- 过程入口：`quant_data_platform/brain/operations_center.md`
- 参考文档：`quant_data_platform/brain/references/`

## Body

- `src/`：平台代码。
- `configs/`：canonical 数据域与 profile 配置。
- `registry/`：可审计 manifest、registry 和指针。
- `data/`：大数据资产、tmp、memmap、agent runs，默认不进 Git。
- `tests/`：平台测试。

## Current Boundary

- `data/qdp_v2` 仍是 active；不要把 v3 代码存在误写成 v3 数据已经可用。
- `data/qdp_v3` 只承载 compatibility、immutable raw、candidate、audit、pin/rollback 与后续重建产物。
- BaoStock 0.9.3 full compatibility gate 当前已通过；任何后续运行只要该证明失效仍会阻断 live ingest。首次 publish 还需要修复 v2 M0 lineage、完成全历史回灌、factor/identity/PIT/5m semantic audit。
- 训练包和 memmap 归 `daily_research`，不再属于 QDP CLI。
